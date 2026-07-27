# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dspark_inputs_kernel_single_grid,
)


def test_copy_and_expand_dspark_inputs() -> None:
    device = "npu"
    draft_len = 3
    cache_block_size = 4
    batch_size = 2
    max_model_len = 10

    next_token_ids = torch.tensor([101, 202], dtype=torch.int32, device=device)
    target_positions = torch.tensor([0, 1, 2, 3, 5, 6, 7], dtype=torch.int64, device=device)
    context_slot_mapping = torch.tensor([30, 31, 32, 33, 40, 41, 42], dtype=torch.int64, device=device)
    request_slots = torch.tensor([2, 5], dtype=torch.int32, device=device)
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32, device=device)
    num_rejected_tokens = torch.tensor([1, 0], dtype=torch.int32, device=device)

    num_context = target_positions.numel()
    num_query = batch_size * draft_len
    out_input_ids = torch.empty(num_query, dtype=torch.int32, device=device)
    out_context_positions = torch.empty(num_context, dtype=torch.int32, device=device)
    out_context_slot_mapping = torch.empty(num_context, dtype=torch.int32, device=device)
    out_context_request_slots = torch.empty(num_context, dtype=torch.int32, device=device)
    out_query_positions = torch.empty(num_query, dtype=torch.int32, device=device)
    out_query_slot_mapping = torch.empty(num_query, dtype=torch.int32, device=device)
    out_query_request_slots = torch.empty(num_query, dtype=torch.int32, device=device)
    out_context_lens = torch.empty(batch_size, dtype=torch.int32, device=device)
    out_seq_lens = torch.empty(batch_size, dtype=torch.int32, device=device)

    copy_and_expand_dspark_inputs_kernel_single_grid[(1,)](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        context_slot_mapping_ptr=context_slot_mapping,
        request_slots_ptr=request_slots,
        out_input_ids_ptr=out_input_ids,
        out_context_positions_ptr=out_context_positions,
        out_context_slot_mapping_ptr=out_context_slot_mapping,
        out_context_request_slots_ptr=out_context_request_slots,
        out_query_positions_ptr=out_query_positions,
        out_query_slot_mapping_ptr=out_query_slot_mapping,
        out_query_request_slots_ptr=out_query_request_slots,
        out_context_lens_ptr=out_context_lens,
        out_seq_lens_ptr=out_seq_lens,
        block_table_ptr=block_table,
        block_table_stride=block_table.stride(0),
        query_start_loc_ptr=query_start_loc,
        num_rejected_tokens_ptr=num_rejected_tokens,
        parallel_drafting_token_id=99,
        cache_block_size=cache_block_size,
        draft_len=draft_len,
        total_input_tokens=num_context,
        batch_size=batch_size,
        max_model_len=max_model_len,
        HAS_NUM_REJECTED=True,
        HAS_MAX_MODEL_LEN=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        out_context_positions,
        torch.tensor([0, 1, 2, -1, 5, 6, 7], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_context_slot_mapping,
        torch.tensor([30, 31, 32, -1, 40, 41, 42], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_context_request_slots,
        torch.tensor([2, 2, 2, 2, 5, 5, 5], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_input_ids,
        torch.tensor([101, 99, 99, 202, 99, 99], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_query_positions,
        torch.tensor([3, 4, 5, 8, 9, 0], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_query_slot_mapping,
        torch.tensor([43, 44, 45, 88, 89, -1], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_query_request_slots,
        torch.tensor([2, 2, 2, 5, 5, 5], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_context_lens,
        torch.tensor([3, 8], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        out_seq_lens,
        torch.tensor([6, 10], dtype=torch.int32, device=device),
    )
