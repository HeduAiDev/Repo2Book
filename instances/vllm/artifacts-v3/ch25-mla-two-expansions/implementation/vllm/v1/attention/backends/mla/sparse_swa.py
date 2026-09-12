# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py —— ch25 切面
#（m11/m15：DeepseekV4SWACache——SWA 滑窗子缓存以独立 prefix 注册
# static_forward_context、自报 SlidingWindowMLASpec，L56-L107 逐字）。
# SUBTRACTED：_layer_type_for 分派与 DeepseekSparseSWABackend 的
#   builder/metadata 全家（L34-L53、L110-L856）——NSA/DSA 机制归 ch26
#（本章只立『一个注意力层拆多个 KV 子缓存、全部进同一分组原料池』）。
from __future__ import annotations

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)


# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L56 DeepseekV4SWACache
#   —— 逐字（must_keep：SWA 子缓存——一层多子缓存实证）
class DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
    ):
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L56（锚点双置：声明上方注释同文）
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
        # Since both block types share the same physical tensor, they must use the
        # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
        # determines the SWA block size of 64 tokens per block.
        # TODO(yifan): make SWA block size automatically determined and configurable.
        self.block_size = 64
        # uint8: fp8_ds_mla UE8M0 paged layout. bfloat16 / float8_e4m3fn:
        # contiguous full-cache layout.
        assert self.dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L87-L102 —— 逐字
        #   （fp8_ds_mla 的 UE8M0 分页布局需 576B 对齐；连续 bf16/fp8 走
        #   自然元素页）
        # fp8_ds_mla's UE8M0 paged layout needs 576B alignment; contiguous
        # bf16/fp8 cache uses the natural element-size page.
        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
        )

    # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L104 forward（占位空体）
    def forward(self): ...
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L87-L102（锚点双置：声明上方注释同文）
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L87-L102（锚点双置：声明上方注释同文）

    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L106-L107 ——
        #   真身 return DeepseekSparseSWABackend；SUBTRACTED：delete[3]
        #   （sparse SWA 家族归 ch26）
        raise NotImplementedError(
            "DeepseekSparseSWABackend is ch26's domain (sparse MLA family)."
        )
