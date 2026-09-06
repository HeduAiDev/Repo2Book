# Subtract-only companion for v3 ch21 — vllm/v1/attention/backends/utils.py
# (pin v0.27.1 / 6e448d0ea). 本章消费面：KV cache layout 的全局覆盖机制
# （get_kv_cache_layout/set_kv_cache_layout——_cached_get_attn_backend 选中
# 后端后的 set_kv_cache_layout 副作用落点；FlashAttentionBackend.get_kv_
# cache_stride_order 读它出 NHD/HND 物理序）+ NULL_BLOCK_ID/PAD_SLOT_ID 块界碑。
from __future__ import annotations

from functools import lru_cache
from typing import Literal, get_args

from ...._host_seams import envs, init_logger

logger = init_logger(__name__)


# SOURCE: vllm/v1/attention/backends/utils.py:L42-L43 KVCacheLayoutType（逐字）
KVCacheLayoutType = Literal["NHD", "HND"]

# SOURCE: vllm/v1/attention/backends/utils.py:L43 _KV_CACHE_LAYOUT_OVERRIDE ——
#   （逐字）模块级覆盖位（set_kv_cache_layout 写、get_kv_cache_layout 读）
_KV_CACHE_LAYOUT_OVERRIDE: KVCacheLayoutType | None = None

# SOURCE: vllm/v1/attention/backends/utils.py:L45-L46 两块界碑（逐字）：
#   PAD_SLOT_ID=写腿 PAD 槽位（slot<0 跳过）；NULL_BLOCK_ID=块表 pad 行
#   （Block 0 保留给 padding——_get_block_table 尾行填它）
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0


# SOURCE: vllm/v1/attention/backends/utils.py:L78-L79 is_valid_kv_cache_layout
#   ——（逐字）
def is_valid_kv_cache_layout(value: str) -> bool:  # SOURCE: vllm/v1/attention/backends/utils.py
    return value in get_args(KVCacheLayoutType)


# SOURCE: vllm/v1/attention/backends/utils.py:L82-L109 get_kv_cache_layout ——
#   惰性解析当前布局：覆盖位 > 环境变量 VLLM_KV_CACHE_LAYOUT > connector 偏好
@lru_cache
def get_kv_cache_layout():
    # Format specified by the code.
    global _KV_CACHE_LAYOUT_OVERRIDE

    cache_layout: Literal["NHD", "HND"] | None = None
    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
        logger.debug_once(
            "`_KV_CACHE_LAYOUT_OVERRIDE` variable detected. "
            "Setting KV cache layout to %s.",
            cache_layout,
        )
        return cache_layout

    # Format specified by the user.
    # SOURCE: vllm/v1/attention/backends/utils.py:L98（逐字）
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    # When neither the user nor the override specified a layout, get default
    if cache_layout is None:
        # SUBTRACTED: get_kv_connector_cache_layout()（L101——KV connector
        #   域 delete[8]；其无 connector 的真实回退是 "NHD"，
        #   见 vllm/distributed/kv_transfer/kv_connector/utils.py:L50）
        cache_layout = "NHD"  # HOST SEAM: 无 connector 的默认物理序（真身同值）
    else:
        assert is_valid_kv_cache_layout(cache_layout)
        logger.info_once(
            "`VLLM_KV_CACHE_LAYOUT` environment variable "
            "detected. Setting KV cache layout to %s.",
            cache_layout,
        )
    return cache_layout


# SOURCE: vllm/v1/attention/backends/utils.py:L112-L115 set_kv_cache_layout ——
#   （逐字）_cached_get_attn_backend 的副作用落点：后端要求特定布局则全局
#   覆写 + 清 get_kv_cache_layout 的 lru_cache
def set_kv_cache_layout(cache_layout: KVCacheLayoutType | None):  # SOURCE: vllm/v1/attention/backends/utils.py
    global _KV_CACHE_LAYOUT_OVERRIDE
    _KV_CACHE_LAYOUT_OVERRIDE = cache_layout
    get_kv_cache_layout.cache_clear()


# SUBTRACTED: compute_mm_prefix_range_tensor（L49-L75——mm_prefix 域，消费
#   者 build() 尾段 delete[5] 删）；PerLayerParameters/get_per_layer_
#   parameters/get_num_attention_heads_from_layers/infer_global_hyper-
#   parameters（L118-L222——FlashInfer plan 域）；make_local_attention_
#   virtual_batches（L277-L420——local attention 域）；make_kv_sharing_fast_
#   prefill_common_attn_metadata（L423-L489——FastPrefill 域，delete[8]）；
#   split_decodes_prefills_and_extends / split_decodes_and_prefills /
#   split_prefill_chunks / reorder_batch_to_split_decodes_and_prefills
#   （L492-L742——调度分桶域，ch10/ch18）；reshape_query_for_spec_decode /
#   reshape_attn_output_for_spec_decode（L745-L771——spec 域）；
#   subclass_attention_metadata（L774-L784——subclass_* 工厂 delete[0] 同域）；
#   KVSharingFastPrefillMetadata / create_fast_prefill_custom_backend
#   （L787-L835——FastPrefill 域，delete[8]）；compute_causal_conv1d_metadata
#   （L838-L884——conv1m 域）；get_dcp_local_seq_lens（L887-L924——DCP 域，
#   消费者 build() DCP 分支 delete[5] 删）；mamba_get_block_table_tensor
#   （L927-L965——mamba 域）。
