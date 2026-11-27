# SPDX-License-Identifier: Apache-2.0
"""Mooncake producer for DeepSpec online hidden-state training.

This connector is intentionally store-only: vLLM's extract_hidden_states
speculative method writes the selected states into a cache-only KV group, and
this class publishes one tensor object per request to Mooncake.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from collections import deque
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
    # CacheOnlyAttentionLayer uses [blocks, hidden_layers, block_tokens, hidden].
    ids = torch.tensor(block_ids, dtype=torch.long, device=cache.device)
    offsets = torch.arange(block_size, dtype=torch.long, device=cache.device)
    slots = (ids[:, None] * block_size + offsets[None, :]).reshape(-1)[:tokens]
    return cache[slots // block_size, :, slots % block_size][:tokens]


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

        def is_hidden_state_group(group: Any) -> bool:
            spec = group.kv_cache_spec
            if isinstance(spec, HiddenStateCacheSpec):
                return True
            inner = getattr(spec, "kv_cache_specs", None)
            return bool(inner) and all(
                isinstance(layer_spec, HiddenStateCacheSpec)
                for layer_spec in inner.values()
            )

        ids = [i for i, group in enumerate(groups) if is_hidden_state_group(group)]
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
        self._failed: set[str] = set()
        self._publish_queue: queue.Queue[PendingSave | None] | None = None
        self._publish_waiting: deque[PendingSave] = deque()
        self._publish_threads: list[threading.Thread] = []
        self._publish_lock = threading.Lock()
        self._stopping = False
        self._publish_inflight: set[str] = set()
        self._awaiting_metadata: set[str] = set()
        self._delayed: set[str] = set()
        self._queue_full_count = 0

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
        writer_count = max(1, int(cfg.get("num_writer_threads", 1)))
        # Keep the queue bounded so producer pressure cannot grow without
        # limit.  A shared Store is intentional: each extra Store would also
        # reserve another local_buffer_size region.
        self._publish_queue = queue.Queue(maxsize=writer_count * 2)
        # Include active writer calls in the bound; Queue.maxsize only counts
        # tasks waiting for a worker.
        for index in range(writer_count):
            worker = threading.Thread(
                target=self._publish_worker,
                name=f"deepspec-mooncake-writer-{index}",
                daemon=True,
            )
            worker.start()
            self._publish_threads.append(worker)

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
        if self._cache.shape[2] != self._block_size:
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
        if self._rank_zero:
            self._delayed.add(req_id)
        return True, {"handle": key}

    def request_finished_all_groups(self, request: "Request", block_ids: tuple[list[int], ...]) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[self._group_id])

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self._rank_zero:
            # The scheduler's finished-id notification and connector metadata
            # can arrive on different iterations. Keep blocks delayed until
            # the corresponding PendingSave has reached the writer queue.
            with self._publish_lock:
                self._awaiting_metadata.update(finished_req_ids)
        if self._rank_zero and self._cache is not None:
            metadata = self._get_connector_metadata() if self.has_connector_metadata() else None
            if isinstance(metadata, DeepSpecMooncakeConnectorMetadata):
                self._setup_store()
                for pending in metadata.pending_saves:
                    if pending.req_id in self._published or pending.req_id in self._failed:
                        continue
                    with self._publish_lock:
                        self._awaiting_metadata.discard(pending.req_id)
                        self._delayed.add(pending.req_id)
                        if pending.req_id not in self._publish_inflight:
                            self._publish_inflight.add(pending.req_id)
                            self._publish_waiting.append(pending)

        # Retry submissions left behind by a full worker queue.
        self._enqueue_waiting()

        with self._publish_lock:
            done = set(self._published)
            done.update(self._failed)
            self._published.clear()
            self._failed.clear()
            self._delayed.difference_update(done)
            self._awaiting_metadata.difference_update(done)
        # A request that has finished generation but has not been published
        # must remain delayed. Returning finished_req_ids here would release
        # its blocks before the writer had copied them.
        if not self._rank_zero:
            done.update(finished_req_ids)
        else:
            done.update(req_id for req_id in finished_req_ids
                        if req_id not in self._delayed
                        and req_id not in self._awaiting_metadata)
        return done or None, None

    def has_pending_push_work(self) -> bool:
        with self._publish_lock:
            waiting = bool(self._publish_waiting or self._publish_inflight)
        queued = self._publish_queue is not None and not self._publish_queue.empty()
        if not waiting and not queued:
            self._queue_full_count = 0
        return waiting or queued

    def _enqueue_waiting(self, *, block: bool = False) -> None:
        if self._publish_queue is None:
            return
        while self._publish_waiting:
            try:
                if block:
                    self._publish_queue.put(self._publish_waiting[0])
                else:
                    self._publish_queue.put_nowait(self._publish_waiting[0])
            except queue.Full:
                self._queue_full_count += 1
                if self._queue_full_count == 1 or self._queue_full_count % 100 == 0:
                    logger.warning(
                        "Mooncake publish backpressure: queue_depth=%d "
                        "waiting=%d queue_capacity=%d full_count=%d",
                        self._publish_queue.qsize(), len(self._publish_waiting),
                        self._publish_queue.maxsize, self._queue_full_count,
                    )
                break
            self._publish_waiting.popleft()

    def _publish_worker(self) -> None:
        assert self._publish_queue is not None
        while True:
            pending = self._publish_queue.get()
            if pending is None:
                self._publish_queue.task_done()
                return
            try:
                self._publish_one(pending)
            except Exception as exc:
                logger.exception(
                    "Mooncake publish failed: request_id=%s handle=%s",
                    pending.req_id, pending.key,
                )
                self._handle_publish_failure(pending, exc)
                with self._publish_lock:
                    self._failed.add(pending.req_id)
            else:
                with self._publish_lock:
                    self._published.add(pending.req_id)
            finally:
                with self._publish_lock:
                    self._publish_inflight.discard(pending.req_id)
                self._publish_queue.task_done()

    def _publish_one(self, pending: PendingSave) -> None:
        started = time.perf_counter()
        hidden = _extract(
            self._cache, pending.block_ids, self._block_size, pending.token_ids.numel()
        )
        extract_done = time.perf_counter()
        hidden = hidden.detach().to("cpu").contiguous()
        copy_done = time.perf_counter()
        token_ids = pending.token_ids.contiguous()
        manifest = {"version": MANIFEST_VERSION, "status": "ok", "tensors": {
            "hidden_states": {"shape": list(hidden.shape), "dtype": str(hidden.dtype)},
            "token_ids": {"shape": list(token_ids.shape), "dtype": str(token_ids.dtype)},
        }}
        for name, tensor in (("hidden_states", hidden), ("token_ids", token_ids)):
            result = self._store.put_tensor(f"{pending.key}:{name}", tensor)
            if isinstance(result, int) and result != 0:
                raise RuntimeError(f"Mooncake {name} publish failed: {result}")
        tensors_done = time.perf_counter()
        # Metadata is the ready marker and must be committed last.
        result = self._store.put(
            f"{pending.key}:meta", json.dumps(manifest).encode()
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(f"Mooncake manifest publish failed: {result}")
        finished = time.perf_counter()
        logger.debug(
            "Mooncake publish timing request_id=%s handle=%s queue_depth=%d "
            "extract_ms=%.3f copy_ms=%.3f tensors_ms=%.3f meta_ms=%.3f total_ms=%.3f",
            pending.req_id, pending.key,
            self._publish_queue.qsize() if self._publish_queue is not None else -1,
            (extract_done - started) * 1000,
            (copy_done - extract_done) * 1000,
            (tensors_done - copy_done) * 1000,
            (finished - tensors_done) * 1000,
            (finished - started) * 1000,
        )

    def shutdown(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._publish_queue is not None:
            self._enqueue_waiting(block=True)
            # Drain queued work before stopping workers.
            self._publish_queue.join()
            for _ in self._publish_threads:
                self._publish_queue.put(None)
            for worker in self._publish_threads:
                worker.join(timeout=5.0)
        self._publish_threads.clear()

    def _handle_publish_failure(self, pending: PendingSave, error: Exception) -> None:
        # Both cleanup and failure notification are best-effort: the same
        # disconnected store node may also make these operations fail.
        keys = [f"{pending.key}:{name}" for name in ("hidden_states", "token_ids", "meta")]
        try:
            results = self._store.batch_remove(keys, force=True)
            if isinstance(results, int):
                results = [results]
            if results is not None and (
                len(results) != len(keys) or any(isinstance(status, int) and status != 0 for status in results)
            ):
                logger.warning("Mooncake partial sample cleanup failed: handle=%s results=%s", pending.key, results)
        except Exception:
            logger.exception("Mooncake partial sample cleanup failed: handle=%s", pending.key)
        try:
            manifest = {"version": MANIFEST_VERSION, "status": "error", "error": str(error)}
            result = self._store.put(f"{pending.key}:meta", json.dumps(manifest).encode())
            if isinstance(result, int) and result != 0:
                logger.warning("Mooncake failure notification failed: handle=%s status=%s", pending.key, result)
        except Exception:
            logger.exception("Mooncake failure notification failed: handle=%s", pending.key)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"
