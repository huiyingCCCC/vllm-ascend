# SPDX-License-Identifier: Apache-2.0
"""Mooncake producer for DeepSpec online hidden-state training.

This connector is intentionally store-only: vLLM's extract_hidden_states
speculative method writes the selected states into a cache-only KV group, and
this class publishes one tensor object per request to Mooncake.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)
MANIFEST_VERSION = 1


def _extract(cache: torch.Tensor, block_ids: list[int], block_size: int, tokens: int) -> torch.Tensor:
    ids = torch.tensor(block_ids, dtype=torch.long, device=cache.device)
    offsets = torch.arange(block_size, dtype=torch.long, device=cache.device)
    slots = (ids[:, None] * block_size + offsets[None, :]).reshape(-1)[:tokens]
    return cache[slots // block_size, slots % block_size][:tokens]


def _safe_key(request_id: str) -> str:
    raw = str(request_id)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return "dspark:" + re.sub(r"[^A-Za-z0-9_.:-]", "_", raw)[:160] + ":" + digest


@dataclass
class PendingSave:
    req_id: str
    key: str
    token_ids: torch.Tensor
    block_ids: list[int]


@dataclass
class DeepSpecMooncakeConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[PendingSave] = field(default_factory=list)


class DeepSpecMooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    """Publish extracted hidden states to Mooncake for a DeepSpec trainer."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    @classmethod
    def _find_cache_kv_group_id(cls, config: "KVCacheConfig | None") -> int:
        if config is None:
            return 0
        from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

        groups = config.kv_cache_groups
        ids = [i for i, group in enumerate(groups) if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)]
        if len(ids) == 1:
            return ids[0]
        if not ids and len(groups) == 1:
            return 0
        raise ValueError("DeepSpecMooncakeConnector requires one isolated HiddenStateCacheSpec group")

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._group_id = self._find_cache_kv_group_id(kv_cache_config)
        group = kv_cache_config.kv_cache_groups[self._group_id]
        self._block_size = group.kv_cache_spec.block_size
        extra = self._kv_transfer_config.get_from_extra_config("mooncake", {}) or {}
        self._mooncake_config = dict(extra)
        self._store = None
        self._cache: torch.Tensor | None = None
        self._rank_zero = True
        self._requests: dict[str, str] = {}
        self._pending: dict[str, PendingSave] = {}
        self._published: set[str] = set()

    def _setup_store(self) -> None:
        if self._store is not None or not self._rank_zero:
            return
        from mooncake.store import MooncakeDistributedStore

        cfg = self._mooncake_config
        self._store = MooncakeDistributedStore()
        result = self._store.setup(
            cfg.get("local_hostname"), cfg.get("metadata_server", "P2PHANDSHAKE"),
            int(cfg.get("global_segment_size", 4 * 1024**3)),
            int(cfg.get("local_buffer_size", 2 * 1024**3)), cfg.get("protocol", "tcp"),
            cfg.get("device_name", ""), cfg.get("master_server_address", "127.0.0.1:50051"),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(f"Mooncake setup failed: status={result}")

    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs: Any) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._rank_zero = get_tensor_model_parallel_rank() == 0
        if not self._rank_zero:
            return
        from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer

        layers = get_layers_from_vllm_config(self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches))
        if len(layers) != 1:
            raise ValueError(f"Expected one hidden-state cache layer, got {len(layers)}")
        self._cache = kv_caches[next(iter(layers))]
        if self._cache.shape[1] != self._block_size:
            raise ValueError("Hidden-state cache block size does not match connector metadata")
        self._setup_store()

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        if num_external_tokens:
            raise ValueError("DeepSpecMooncakeConnector is store-only")

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = DeepSpecMooncakeConnectorMetadata(pending_saves=list(self._pending.values()))
        self._pending.clear()
        for req in scheduler_output.scheduled_new_reqs:
            self._requests[req.req_id] = _safe_key(req.req_id)
        return meta

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        key = self._requests.pop(req_id, _safe_key(req_id))
        params = request.kv_transfer_params or {}
        if params.get("include_output_tokens", False):
            ids = list(request.all_token_ids)[:-1]
        else:
            ids = list(request.prompt_token_ids or [])
        self._pending[req_id] = PendingSave(req_id, key, torch.tensor(ids, dtype=torch.long), list(block_ids))
        return True, {"handle": key}

    def request_finished_all_groups(self, request: "Request", block_ids: tuple[list[int], ...]) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[self._group_id])

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self._rank_zero and self._cache is not None:
            metadata = self._get_connector_metadata() if self.has_connector_metadata() else None
            if isinstance(metadata, DeepSpecMooncakeConnectorMetadata):
                self._setup_store()
                for pending in metadata.pending_saves:
                    if pending.req_id in self._published:
                        continue
                    hidden = _extract(self._cache, pending.block_ids, self._block_size, pending.token_ids.numel())
                    hidden = hidden.detach().to("cpu").contiguous()
                    token_ids = pending.token_ids.contiguous()
                    self._store.put_tensor(f"{pending.key}:hidden_states", hidden)
                    self._store.put_tensor(f"{pending.key}:token_ids", token_ids)
                    manifest = {"version": MANIFEST_VERSION, "status": "ok", "tensors": {
                        "hidden_states": {"shape": list(hidden.shape), "dtype": str(hidden.dtype)},
                        "token_ids": {"shape": list(token_ids.shape), "dtype": str(token_ids.dtype)},
                    }}
                    result = self._store.put(f"{pending.key}:meta", json.dumps(manifest).encode())
                    if isinstance(result, int) and result != 0:
                        raise RuntimeError(f"Mooncake manifest publish failed: {result}")
                    self._published.add(pending.req_id)
        done = set(finished_req_ids)
        done.update(self._published)
        for req_id in done:
            self._published.discard(req_id)
        return done or None, None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"
