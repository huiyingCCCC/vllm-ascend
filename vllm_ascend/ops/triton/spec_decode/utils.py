# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)
        tl.store(num_rejected_tokens_gpu_ptr + offsets, num_rejected, mask=mask)


@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    context_slot_mapping_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    seq_lens_ptr,  # [num_reqs]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    for req_idx in range(0, batch_size):
        ctx_start = tl.load(query_start_loc_ptr + req_idx)
        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
        num_ctx = ctx_end - ctx_start

        for j in range(0, num_ctx):
            ctx_pos_idx = ctx_start + j
            pos = tl.load(target_positions_ptr + ctx_pos_idx)
            tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

            slot = tl.load(context_slot_mapping_ptr + ctx_pos_idx)
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

        if HAS_NUM_REJECTED:
            num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
            valid_ctx_end = ctx_end - num_rejected
        else:
            num_rejected = 0
            valid_ctx_end = ctx_end

        seq_len = tl.load(seq_lens_ptr + req_idx)
        effective_seq_len = seq_len - num_rejected
        last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)

        for q_idx in range(0, num_query_per_req):
            query_pos = last_pos + 1 + q_idx
            query_out_idx = req_idx * num_query_per_req + q_idx

            tl.store(out_query_positions_ptr + query_out_idx, query_pos)

            query_cache_pos = effective_seq_len + q_idx
            block_num_q = query_cache_pos // block_size
            block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q).to(tl.int64)
            slot_q = block_id_q * block_size + (query_cache_pos % block_size)
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

            if q_idx == 0:
                bonus_token = tl.load(next_token_ids_ptr + req_idx)
                tl.store(out_input_ids_ptr + query_out_idx, bonus_token)
            else:
                tl.store(out_input_ids_ptr + query_out_idx, parallel_drafting_token_id)

                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)


@triton.jit
def store_dspark_context_kv_kernel(
    shared_kv_ptr,
    positions_ptr,
    slot_mapping_ptr,
    request_slots_ptr,
    paged_cache_ptr,
    ring_cache_ptr,
    cache_positions_ptr,
    shared_kv_stride,
    shared_kv_head_stride,
    paged_cache_block_stride,
    paged_cache_token_stride,
    paged_cache_head_stride,
    ring_cache_request_stride,
    ring_cache_token_stride,
    ring_cache_head_stride,
    cache_positions_request_stride,
    num_tokens,
    head_dim,
    num_paged_blocks,
    paged_block_size,
    num_request_slots,
    window_size,
    STORE_RING: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    head_valid = head_offsets < head_dim
    is_first_head_block = head_block_idx == 0

    position = tl.load(positions_ptr + token_idx)
    slot = tl.load(slot_mapping_ptr + token_idx)
    page_block = slot // paged_block_size
    page_offset = slot % paged_block_size
    page_valid = (
        (token_idx < num_tokens)
        & (position >= 0)
        & (slot >= 0)
        & (page_block < num_paged_blocks)
    )
    shared_kv = tl.load(
        shared_kv_ptr
        + token_idx * shared_kv_stride
        + head_offsets * shared_kv_head_stride,
        mask=page_valid & head_valid,
        other=0.0,
    )
    tl.store(
        paged_cache_ptr
        + page_block * paged_cache_block_stride
        + page_offset * paged_cache_token_stride
        + head_offsets * paged_cache_head_stride,
        shared_kv,
        mask=page_valid & head_valid,
    )

    if STORE_RING:
        request_slot = tl.load(request_slots_ptr + token_idx)
        ring_offset = position % window_size
        ring_valid = page_valid & (request_slot >= 0) & (request_slot < num_request_slots)
        tl.store(
            ring_cache_ptr
            + request_slot * ring_cache_request_stride
            + ring_offset * ring_cache_token_stride
            + head_offsets * ring_cache_head_stride,
            shared_kv,
            mask=ring_valid & head_valid,
        )
        tl.store(
            cache_positions_ptr + request_slot * cache_positions_request_stride + ring_offset,
            position,
            mask=ring_valid & is_first_head_block,
        )


def store_dspark_context_kv(
    shared_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_cache: torch.Tensor,
    ring_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    request_slots: torch.Tensor | None,
) -> None:
    num_tokens = positions.numel()
    if num_tokens == 0:
        return
    paged_tokens = paged_cache.flatten(start_dim=2)
    head_dim = ring_cache.shape[-1]
    block_h = 128
    store_dspark_context_kv_kernel[(num_tokens, triton.cdiv(head_dim, block_h))](
        shared_kv_ptr=shared_kv,
        positions_ptr=positions,
        slot_mapping_ptr=slot_mapping,
        request_slots_ptr=request_slots if request_slots is not None else 0,
        paged_cache_ptr=paged_tokens,
        ring_cache_ptr=ring_cache,
        cache_positions_ptr=cache_positions,
        shared_kv_stride=shared_kv.stride(0),
        shared_kv_head_stride=shared_kv.stride(-1),
        paged_cache_block_stride=paged_tokens.stride(0),
        paged_cache_token_stride=paged_tokens.stride(1),
        paged_cache_head_stride=paged_tokens.stride(2),
        ring_cache_request_stride=ring_cache.stride(0),
        ring_cache_token_stride=ring_cache.stride(1),
        ring_cache_head_stride=ring_cache.stride(2),
        cache_positions_request_stride=cache_positions.stride(0),
        num_tokens=num_tokens,
        head_dim=head_dim,
        num_paged_blocks=paged_tokens.shape[0],
        paged_block_size=paged_tokens.shape[1],
        num_request_slots=ring_cache.shape[0],
        window_size=ring_cache.shape[1],
        STORE_RING=request_slots is not None,
        BLOCK_H=block_h,
    )


@triton.jit
def restore_dspark_context_cache_kernel(
    paged_cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    request_slots_ptr,
    ring_cache_ptr,
    cache_positions_ptr,
    paged_cache_block_stride,
    paged_cache_token_stride,
    paged_cache_head_stride,
    block_table_stride,
    ring_cache_request_stride,
    ring_cache_token_stride,
    ring_cache_head_stride,
    cache_positions_request_stride,
    batch_size,
    block_table_cols,
    head_dim,
    num_paged_blocks,
    paged_block_size,
    num_request_slots,
    window_size,
    BLOCK_H: tl.constexpr,
):
    request_idx = tl.program_id(0)
    window_offset = tl.program_id(1)
    head_block_idx = tl.program_id(2)
    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    head_valid = head_offsets < head_dim
    is_first_head_block = head_block_idx == 0

    context_len = tl.load(context_lens_ptr + request_idx)
    request_slot = tl.load(request_slots_ptr + request_idx)
    position = context_len - window_size + window_offset
    position_valid = (
        (request_idx < batch_size)
        & (position >= 0)
        & (position < context_len)
        & (request_slot >= 0)
        & (request_slot < num_request_slots)
    )
    block_number = position // paged_block_size
    table_valid = position_valid & (block_number >= 0) & (block_number < block_table_cols)
    block_id = tl.load(
        block_table_ptr + request_idx * block_table_stride + block_number,
        mask=table_valid,
        other=-1,
    )
    page_valid = table_valid & (block_id >= 0) & (block_id < num_paged_blocks)
    ring_offset = position % window_size
    stored_position = tl.load(
        cache_positions_ptr + request_slot * cache_positions_request_stride + ring_offset,
        mask=page_valid,
        other=-1,
    )
    restore_required = page_valid & (stored_position != position)
    page_offset = position % paged_block_size
    context_kv = tl.load(
        paged_cache_ptr
        + block_id * paged_cache_block_stride
        + page_offset * paged_cache_token_stride
        + head_offsets * paged_cache_head_stride,
        mask=restore_required & head_valid,
        other=0.0,
    )
    tl.store(
        ring_cache_ptr
        + request_slot * ring_cache_request_stride
        + ring_offset * ring_cache_token_stride
        + head_offsets * ring_cache_head_stride,
        context_kv,
        mask=restore_required & head_valid,
    )
    tl.store(
        cache_positions_ptr + request_slot * cache_positions_request_stride + ring_offset,
        position,
        mask=restore_required & is_first_head_block,
    )


def restore_dspark_context_cache(
    paged_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    request_slots: torch.Tensor,
    ring_cache: torch.Tensor,
    cache_positions: torch.Tensor,
) -> None:
    batch_size = min(block_table.shape[0], context_lens.numel(), request_slots.numel())
    if batch_size == 0:
        return
    paged_tokens = paged_cache.flatten(start_dim=2)
    head_dim = ring_cache.shape[-1]
    block_h = 128
    restore_dspark_context_cache_kernel[
        (batch_size, ring_cache.shape[1], triton.cdiv(head_dim, block_h))
    ](
        paged_cache_ptr=paged_tokens,
        block_table_ptr=block_table,
        context_lens_ptr=context_lens,
        request_slots_ptr=request_slots,
        ring_cache_ptr=ring_cache,
        cache_positions_ptr=cache_positions,
        paged_cache_block_stride=paged_tokens.stride(0),
        paged_cache_token_stride=paged_tokens.stride(1),
        paged_cache_head_stride=paged_tokens.stride(2),
        block_table_stride=block_table.stride(0),
        ring_cache_request_stride=ring_cache.stride(0),
        ring_cache_token_stride=ring_cache.stride(1),
        ring_cache_head_stride=ring_cache.stride(2),
        cache_positions_request_stride=cache_positions.stride(0),
        batch_size=batch_size,
        block_table_cols=block_table.shape[1],
        head_dim=head_dim,
        num_paged_blocks=paged_tokens.shape[0],
        paged_block_size=paged_tokens.shape[1],
        num_request_slots=ring_cache.shape[0],
        window_size=ring_cache.shape[1],
        BLOCK_H=block_h,
    )
