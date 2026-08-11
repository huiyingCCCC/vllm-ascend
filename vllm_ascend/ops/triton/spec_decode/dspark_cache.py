# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def dspark_masked_cache_store_kernel(
    cache_ptr,
    shared_kv_ptr,
    positions_ptr,
    slot_mapping_ptr,
    num_cache_blocks,
    cache_block_size,
    cache_stride_block,
    cache_stride_token,
    cache_stride_dim,
    shared_kv_stride_token,
    shared_kv_stride_dim,
    positions_stride,
    slot_mapping_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dims = tl.arange(0, BLOCK_DIM)
    dim_mask = dims < HEAD_DIM

    position = tl.load(positions_ptr + token_idx * positions_stride)
    # Match the previous PyTorch path, which promoted slots to int64 before
    # deriving paged-cache addresses.
    slot = tl.load(slot_mapping_ptr + token_idx * slot_mapping_stride).to(tl.int64)
    num_cache_slots = num_cache_blocks * cache_block_size
    valid = (position >= 0) & (slot >= 0) & (slot < num_cache_slots)

    # Keep address calculation in bounds even when the masked store is disabled.
    safe_slot = tl.where(slot < 0, 0, slot)
    safe_slot = tl.where(safe_slot >= num_cache_slots, num_cache_slots - 1, safe_slot)
    block_id = safe_slot // cache_block_size
    block_offset = safe_slot % cache_block_size

    values = tl.load(
        shared_kv_ptr + token_idx * shared_kv_stride_token + dims * shared_kv_stride_dim,
        mask=dim_mask,
        other=0.0,
    )
    cache_offsets = block_id * cache_stride_block + block_offset * cache_stride_token + dims * cache_stride_dim
    # The scheduler owns writable paged blocks, so valid slots are unique
    # within one launch and do not require atomic stores.
    tl.store(cache_ptr + cache_offsets, values, mask=valid & dim_mask)


def dspark_masked_cache_store_impl(
    kv_cache: torch.Tensor,
    shared_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if positions.numel() == 0:
        return
    if not HAS_TRITON:
        raise RuntimeError("DSpark masked cache store requires Triton.")
    if kv_cache.device != shared_kv.device or kv_cache.device != positions.device:
        raise ValueError("DSpark cache, shared KV, and positions must be on the same device.")
    if kv_cache.device != slot_mapping.device:
        raise ValueError("DSpark cache and slot mapping must be on the same device.")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"DSpark positions must be int32 or int64, got dtype={positions.dtype}.")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"DSpark slot mapping must be int32 or int64, got dtype={slot_mapping.dtype}.")
    if kv_cache.dtype != shared_kv.dtype:
        raise ValueError(
            "DSpark cache and shared KV must have the same dtype: "
            f"cache={kv_cache.dtype}, shared_kv={shared_kv.dtype}."
        )
    if kv_cache.ndim < 3:
        raise ValueError(f"DSpark paged cache must have at least 3 dimensions, got shape={tuple(kv_cache.shape)}.")
    if kv_cache.shape[0] == 0 or kv_cache.shape[1] == 0:
        raise ValueError(f"DSpark paged cache must not be empty, got shape={tuple(kv_cache.shape)}.")
    if any(size != 1 for size in kv_cache.shape[2:-1]):
        raise ValueError(f"DSpark paged cache requires one KV head, got shape={tuple(kv_cache.shape)}.")
    if not shared_kv.is_contiguous() or not positions.is_contiguous() or not slot_mapping.is_contiguous():
        raise ValueError("DSpark masked cache store requires contiguous shared KV, positions, and slot mapping.")

    head_dim = shared_kv.shape[-1]
    if head_dim == 0:
        raise ValueError("DSpark shared KV head dimension must not be empty.")
    shared_kv_rows = shared_kv.numel() // head_dim
    if positions.numel() != shared_kv_rows or slot_mapping.numel() != shared_kv_rows:
        raise ValueError(
            "DSpark masked cache store input size mismatch: "
            f"shared_kv_rows={shared_kv_rows}, positions={positions.numel()}, "
            f"slot_mapping={slot_mapping.numel()}."
        )
    if kv_cache.shape[-1] < head_dim:
        raise ValueError(
            f"DSpark paged cache head dimension is too small: cache={kv_cache.shape[-1]}, shared_kv={head_dim}."
        )

    shared_kv_2d = shared_kv.view(shared_kv_rows, head_dim)
    positions_1d = positions.view(-1)
    slot_mapping_1d = slot_mapping.view(-1)
    block_dim = triton.next_power_of_2(head_dim)
    dspark_masked_cache_store_kernel[(shared_kv_rows,)](
        kv_cache,
        shared_kv_2d,
        positions_1d,
        slot_mapping_1d,
        kv_cache.shape[0],
        kv_cache.shape[1],
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(-1),
        shared_kv_2d.stride(0),
        shared_kv_2d.stride(1),
        positions_1d.stride(0),
        slot_mapping_1d.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
    )


def dspark_masked_cache_store_fake(
    kv_cache: torch.Tensor,
    shared_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="dspark_masked_cache_store",
    op_func=dspark_masked_cache_store_impl,
    mutates_args=["kv_cache"],
    fake_impl=dspark_masked_cache_store_fake,
    dispatch_key="PrivateUse1",
)


def dspark_masked_cache_store(
    kv_cache: torch.Tensor,
    shared_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store unique valid slots without dynamic-shape indexing."""
    torch.ops.vllm.dspark_masked_cache_store(kv_cache, shared_kv, positions, slot_mapping)


@triton.jit
def dspark_incremental_cache_sync_kernel(
    paged_cache_ptr,
    ring_cache_ptr,
    ring_positions_ptr,
    block_ids_ptr,
    block_offsets_ptr,
    cache_valid_ptr,
    request_slots_ptr,
    expected_positions_ptr,
    paged_cache_stride_block,
    paged_cache_stride_token,
    paged_cache_stride_dim,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    NUM_RING_SLOTS: tl.constexpr,
):
    entry_idx = tl.program_id(0)
    batch_idx = entry_idx // WINDOW_SIZE
    dims = tl.arange(0, BLOCK_DIM)
    dim_mask = dims < HEAD_DIM

    block_id = tl.load(block_ids_ptr + entry_idx).to(tl.int64)
    block_offset = tl.load(block_offsets_ptr + entry_idx).to(tl.int64)
    cache_valid = tl.load(cache_valid_ptr + entry_idx).to(tl.int1)
    request_slot = tl.load(request_slots_ptr + batch_idx).to(tl.int64)
    expected_position = tl.load(expected_positions_ptr + entry_idx)
    ring_index = tl.maximum(expected_position, 0) % WINDOW_SIZE

    slot_valid = (request_slot >= 0) & (request_slot < NUM_RING_SLOTS)
    safe_request_slot = tl.where(request_slot < 0, 0, request_slot)
    safe_request_slot = tl.where(safe_request_slot >= NUM_RING_SLOTS, NUM_RING_SLOTS - 1, safe_request_slot)
    ring_position_offset = safe_request_slot * WINDOW_SIZE + ring_index
    cached_position = tl.load(ring_positions_ptr + ring_position_offset)
    cache_miss = cache_valid & slot_valid & (cached_position != expected_position)

    paged_offsets = (
        block_id * paged_cache_stride_block + block_offset * paged_cache_stride_token + dims * paged_cache_stride_dim
    )
    values = tl.load(paged_cache_ptr + paged_offsets, mask=cache_miss & dim_mask, other=0.0)
    ring_offsets = ring_position_offset * HEAD_DIM + dims
    tl.store(ring_cache_ptr + ring_offsets, values, mask=cache_miss & dim_mask)
    # Publish the tag after the KV stores. The following attention runs on the
    # same stream, so the completed kernel is the cache visibility boundary.
    tl.store(ring_positions_ptr + ring_position_offset, expected_position, mask=cache_miss)


def dspark_incremental_cache_sync_impl(
    paged_cache: torch.Tensor,
    ring_cache: torch.Tensor,
    ring_positions: torch.Tensor,
    block_ids: torch.Tensor,
    block_offsets: torch.Tensor,
    cache_valid: torch.Tensor,
    request_slots: torch.Tensor,
    expected_positions: torch.Tensor,
) -> None:
    if not HAS_TRITON:
        raise RuntimeError("DSpark incremental cache sync requires Triton.")

    if (
        ring_cache.ndim != 3
        or ring_positions.shape != ring_cache.shape[:2]
        or not ring_cache.is_contiguous()
        or not ring_positions.is_contiguous()
    ):
        raise ValueError("Invalid DSpark ring cache layout.")
    if (
        paged_cache.ndim < 3
        or any(size != 1 for size in paged_cache.shape[2:-1])
        or paged_cache.shape[-1] < ring_cache.shape[-1]
    ):
        raise ValueError("Invalid DSpark paged cache layout.")
    index_tensors = (block_offsets, cache_valid, expected_positions)
    if (
        block_ids.ndim != 2
        or any(tensor.shape != block_ids.shape for tensor in index_tensors)
        or request_slots.numel() != block_ids.shape[0]
        or ring_cache.shape[1] != block_ids.shape[1]
        or not all(tensor.is_contiguous() for tensor in (block_ids, *index_tensors, request_slots))
    ):
        raise ValueError("Invalid DSpark incremental cache sync indices.")

    head_dim = ring_cache.shape[-1]
    block_dim = triton.next_power_of_2(head_dim)
    dspark_incremental_cache_sync_kernel[(block_ids.numel(),)](
        paged_cache,
        ring_cache,
        ring_positions,
        block_ids,
        block_offsets,
        cache_valid,
        request_slots,
        expected_positions,
        paged_cache.stride(0),
        paged_cache.stride(1),
        paged_cache.stride(-1),
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
        WINDOW_SIZE=block_ids.shape[1],
        NUM_RING_SLOTS=ring_cache.shape[0],
    )


def dspark_incremental_cache_sync_fake(
    paged_cache: torch.Tensor,
    ring_cache: torch.Tensor,
    ring_positions: torch.Tensor,
    block_ids: torch.Tensor,
    block_offsets: torch.Tensor,
    cache_valid: torch.Tensor,
    request_slots: torch.Tensor,
    expected_positions: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="dspark_incremental_cache_sync",
    op_func=dspark_incremental_cache_sync_impl,
    mutates_args=["ring_cache", "ring_positions"],
    fake_impl=dspark_incremental_cache_sync_fake,
    dispatch_key="PrivateUse1",
)


def dspark_incremental_cache_sync(
    paged_cache: torch.Tensor,
    ring_cache: torch.Tensor,
    ring_positions: torch.Tensor,
    block_ids: torch.Tensor,
    block_offsets: torch.Tensor,
    cache_valid: torch.Tensor,
    request_slots: torch.Tensor,
    expected_positions: torch.Tensor,
) -> None:
    """Restore only missing absolute positions from the transferable cache."""
    torch.ops.vllm.dspark_incremental_cache_sync(
        paged_cache,
        ring_cache,
        ring_positions,
        block_ids,
        block_offsets,
        cache_valid,
        request_slots,
        expected_positions,
    )
