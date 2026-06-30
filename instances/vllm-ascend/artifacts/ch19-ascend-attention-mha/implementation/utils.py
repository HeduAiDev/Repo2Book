# vllm_ascend/attention/utils.py —— subtract-only 精简版（拆批 / paged 门槛 / CP 开关）
#
# 本章用到三件：
#   - split_decodes_and_prefills：在「已重排」的 batch 上找 decode|prefill 分界（纯 Python，可跑）。
#   - using_paged_attention：decode 能否走 _npu_paged_attention 的多重门槛判据。
#   - enable_cp：运行期 CP（上下文并行）分流开关（f7 在 backend.get_impl_cls 处回收）。
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
from vllm.config import VllmConfig, get_current_vllm_config

from vllm_ascend.utils import AscendDeviceType, get_ascend_config, get_ascend_device_type

# SUBTRACTED: torch.nn.functional / kv_transfer / forward_context / CommonAttentionMetadata 等 import
#   （utils.py:L6-L13）—— KV 传输、PCP/DCP 的 long-seq 工具依赖，本章主线不触达。


# SOURCE: vllm_ascend/attention/utils.py:L44-L55
def using_paged_attention(runtime_shape: int, vllm_config: VllmConfig) -> bool:
    # 多重门槛：任一不满足，decode 也回落到 fused-infer 路径。
    if vllm_config.speculative_config is not None:
        return False
    if get_ascend_device_type() == AscendDeviceType.A5:
        return False
    from vllm.config.compilation import CUDAGraphMode

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return False

    return runtime_shape in get_ascend_config().pa_shape_list


@lru_cache(maxsize=1)
def enable_cp():
    # SOURCE: vllm_ascend/attention/utils.py:L58-L61
    # f7 回收点的开关：prefill/decode 任一 context-parallel size > 1 即启用 CP 版 impl/builder。
    prefill_config = get_current_vllm_config().parallel_config
    return prefill_config.prefill_context_parallel_size > 1 or prefill_config.decode_context_parallel_size > 1


# SOURCE: vllm_ascend/attention/utils.py:L147-L148
@dataclass
class AscendCommonAttentionMetadata:
    """Per-batch attention metadata, shared across layers and backends.

    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    """

    # SOURCE: vllm_ascend/attention/utils.py:L147-L191
    # SUBTRACTED: 真身继承 vllm 的 CommonAttentionMetadata 并多带 ~30 个字段（utils.py:L148-L191，
    #   positions/num_computed_tokens_cpu/graph_pad_size/encoder_seq_lens 等）与 unpadded() 方法
    #   —— 它们服务 eagle/encoder/PCP 等旁路；本精简版只保留 build / split_decodes_and_prefills
    #   实际读到的字段，作可跑的输入容器。
    num_reqs: int = 0
    num_actual_tokens: int = 0
    max_query_len: int = 0
    query_start_loc_cpu: torch.Tensor = None
    block_table_tensor: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    _seq_lens_cpu: torch.Tensor = None
    attn_state: Any = None
    causal: bool = True
    kvcomp_metadata: Any = None
    # SUBTRACTED: prefill_context_parallel_metadata（PCP long-seq 元数据，utils.py:L190）——
    #   仅 CP 场景非 None；本章保留「为 None 的主线」，split_decodes_and_prefills 据此走非 PCP 分支。
    prefill_context_parallel_metadata: Any = None


# SOURCE: vllm_ascend/attention/utils.py:L273-L315
def split_decodes_and_prefills(
    common_attn_metadata: AscendCommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Returns (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens).
    """
    # SUBTRACTED: PCP 分支——query_lens_pcp_full / max_query_len_pcp_full（utils.py:L294-L297）
    #   仅 prefill_context_parallel 时非 None；CP 组排布回指 ch08，本章主线 query_lens_pcp_full=None。
    query_lens_pcp_full = None
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = (query_start_loc[1:] - query_start_loc[:-1]) if query_lens_pcp_full is None else query_lens_pcp_full
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)
