# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.ops.triton.spec_decode.utils import (
    restore_dspark_context_cache,
    store_dspark_context_kv,
)

DEVICE = "npu"
HEAD_DIM = 160
PAGED_BLOCK_SIZE = 4
WINDOW_SIZE = 4


def test_store_dspark_context_kv() -> None:
    shared_kv = torch.arange(2 * HEAD_DIM, dtype=torch.float32, device=DEVICE).view(2, HEAD_DIM)
    positions = torch.tensor([3, 6], dtype=torch.int32, device=DEVICE)
    slot_mapping = torch.tensor([5, 10], dtype=torch.int64, device=DEVICE)
    request_slots = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    paged_cache = torch.zeros((4, PAGED_BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    ring_cache = torch.zeros((2, WINDOW_SIZE, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    cache_positions = torch.full((2, WINDOW_SIZE), -1, dtype=torch.int32, device=DEVICE)

    store_dspark_context_kv(
        shared_kv,
        positions,
        slot_mapping,
        paged_cache,
        ring_cache,
        cache_positions,
        request_slots,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(paged_cache[1, 1, 0], shared_kv[0])
    torch.testing.assert_close(paged_cache[2, 2, 0], shared_kv[1])
    torch.testing.assert_close(ring_cache[1, 3], shared_kv[0])
    torch.testing.assert_close(ring_cache[0, 2], shared_kv[1])
    torch.testing.assert_close(
        cache_positions,
        torch.tensor([[-1, -1, 6, -1], [-1, -1, -1, 3]], dtype=torch.int32, device=DEVICE),
    )


def test_restore_dspark_context_cache() -> None:
    paged_cache = torch.zeros((3, PAGED_BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    for block_id in range(paged_cache.shape[0]):
        for block_offset in range(PAGED_BLOCK_SIZE):
            paged_cache[block_id, block_offset, 0].fill_(block_id * 10 + block_offset)

    block_table = torch.tensor([[2, 0], [1, -1]], dtype=torch.int32, device=DEVICE)
    context_lens = torch.tensor([6, 3], dtype=torch.int32, device=DEVICE)
    request_slots = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    ring_cache = torch.zeros((2, WINDOW_SIZE, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    cache_positions = torch.full((2, WINDOW_SIZE), -1, dtype=torch.int32, device=DEVICE)

    # Position 3 is already current. Its ring value must not be overwritten.
    ring_cache[1, 3].fill_(99)
    cache_positions[1, 3] = 3

    restore_dspark_context_cache(
        paged_cache,
        block_table,
        context_lens,
        request_slots,
        ring_cache,
        cache_positions,
    )
    torch.npu.synchronize()

    expected_ring_cache = torch.zeros_like(ring_cache)
    expected_ring_cache[0, 0].fill_(10)
    expected_ring_cache[0, 1].fill_(11)
    expected_ring_cache[0, 2].fill_(12)
    expected_ring_cache[1, 0].fill_(0)
    expected_ring_cache[1, 1].fill_(1)
    expected_ring_cache[1, 2].fill_(22)
    expected_ring_cache[1, 3].fill_(99)
    torch.testing.assert_close(ring_cache, expected_ring_cache)
    torch.testing.assert_close(
        cache_positions,
        torch.tensor([[0, 1, 2, -1], [4, 5, 2, 3]], dtype=torch.int32, device=DEVICE),
    )
