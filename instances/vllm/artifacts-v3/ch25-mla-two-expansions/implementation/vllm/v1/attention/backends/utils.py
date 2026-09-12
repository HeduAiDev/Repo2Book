# SOURCE: vllm/v1/attention/backends/utils.py
# ch25 切面（站 7-8 / m06/m07）：混批分流的两半——
#   split_decodes_and_prefills（L564-L635 逐字：排好序的批上数边界）
#   reorder_batch_to_split_decodes_and_prefills（L665-L742 逐字：互斥四区
#   划分 + 换位）+ get_num_attention_heads_from_layers（builder 消费）。
# SUBTRACTED：mm/fast-prefill/local-attn/split_prefill_chunks/reshape_
#   query_for_spec_decode 族与 KV layout 覆盖机制——ch15/ch18/ch19/ch21 域
#   （reshape_query_for_spec_decode 按 delete[4] 删——spec-decode 归 ch33）。
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata


# SOURCE: vllm/v1/attention/backends/utils.py:L169-L197 get_num_attention_
#   heads_from_layers —— 逐字（metadata builder 的逐组头数对账）
def get_num_attention_heads_from_layers(
    vllm_config: "VllmConfig", layer_names: list[str]
) -> int | None:
    # SOURCE: vllm/v1/attention/backends/utils.py:L169-L197（锚点双置：声明上方注释同文）
    """Per-TP-rank ``num_heads`` shared by the named Attention layers.

    Use in metadata builders whose plan-time allocations depend on the
    head count: the model-wide ``get_num_attention_heads()`` is wrong
    for models with non-uniform per-layer head counts. All layers in
    one attention group must agree on ``num_heads``; this is asserted.
    Returns ``None`` when no matching Attention layer is found.
    """
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import (
        AttentionLayerBase,
    )

    attn_layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
        layer_names,
    )
    if not attn_layers:
        return None
    heads = {layer.impl.num_heads for layer in attn_layers.values()}
    assert len(heads) == 1, (
        f"All layers in one attention group must share num_heads; "
        f"got {heads} for {layer_names}."
    )
    return heads.pop()


# SOURCE: vllm/v1/attention/backends/utils.py:L564-L635 split_decodes_and_
#   prefills —— 逐字（must_keep：边界计数函数）
def split_decodes_and_prefills(
    common_attn_metadata: "CommonAttentionMetadata",
    decode_threshold: int = 1,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    # SOURCE: vllm/v1/attention/backends/utils.py:L564-L635（锚点双置：声明上方注释同文）
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    The batch is expected to be ordered as:
        decode → short_extend → long_extend → prefill

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered prefills
            even if they are <= decode_threshold, in order to ensure uniformity.
        treat_short_extends_as_decodes: If True (default), short extends
            (query_len <= threshold but still prefilling) are counted as
            decodes. If False, they are counted as prefills.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if query_lens[0].item() > decode_threshold:
        # first request is not decode, so no decode requests
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        # check if we are in a padded uniform batch; this is used for full-CGs, some
        # requests may have a query length of 0 but since they are padding its fine
        # to treat them as decodes (ensures num_decodes matches the captured size)
        if treat_short_extends_as_decodes and torch.all(
            (query_lens == query_lens[0]) | (query_lens == 0)
        ):
            return num_reqs, 0, num_tokens, 0  # all decodes
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        assert common_attn_metadata.is_prefilling is not None
        is_prefill |= common_attn_metadata.is_prefilling

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


# SUBTRACTED: split_prefill_chunks（L638-L662）——workspace 分块预划归
#   build_mla_chunked_context_metadata（mla_attention.py 域）


# SOURCE: vllm/v1/attention/backends/utils.py:L665-L742 reorder_batch_to_
#   split_decodes_and_prefills —— 逐字（must_keep：四区重排函数）
def reorder_batch_to_split_decodes_and_prefills(
    input_batch: Any,
    scheduler_output: Any,
    decode_threshold: int = 1,
) -> bool:
    # SOURCE: vllm/v1/attention/backends/utils.py:L665-L742（锚点双置：声明上方注释同文）
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    The batch is reordered into 4 regions:
        decode:        (num_scheduled <= threshold AND is not prefilling)
        short_extend:  (num_scheduled <= threshold AND is chunked prefilling)
        long_extend:   (num_scheduled > threshold AND is chunked prefilling)
        prefill:       (num_computed == 0)   # First chunks

    Returns:
        True if the batch was modified, False otherwise.
    """
    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [
        scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids
    ]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]
    num_prompt_tokens_np = input_batch.num_prompt_tokens[:num_reqs]

    has_context = num_computed_tokens_np > 0
    is_below_threshold = num_scheduled_tokens_np <= decode_threshold
    done_prefilling = num_computed_tokens_np >= num_prompt_tokens_np

    # Mutually exclusive categories (exactly one True per request):
    # 1. No context yet -> prefill
    # 2. Has context, above threshold -> long_extend
    # 3. Has context, below threshold, still prefilling -> short_extend
    # 4. Has context, below threshold, done prefilling -> decode
    is_pure_prefill = ~has_context
    is_long_extend = has_context & ~is_below_threshold
    is_short_extend = has_context & is_below_threshold & ~done_prefilling
    is_decode = has_context & is_below_threshold & done_prefilling

    # Desired order: decode → short_extend → long_extend → prefill
    req_regions = np.zeros(num_reqs, dtype=np.int32)  # 0 = decode by default
    req_regions[is_short_extend] = 1
    req_regions[is_long_extend] = 2
    req_regions[is_pure_prefill] = 3

    num_decodes = int(is_decode.sum())
    num_short_extends = int(is_short_extend.sum())
    num_long_extends = int(is_long_extend.sum())
    num_prefills = int(is_pure_prefill.sum())

    target_regions = np.repeat(
        [0, 1, 2, 3],
        [num_decodes, num_short_extends, num_long_extends, num_prefills],
    ).astype(np.int32)

    needs_swap = req_regions != target_regions

    if not needs_swap.any():
        return False

    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")
    src_indices = orig_indices[sorted_order]

    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}

    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True
