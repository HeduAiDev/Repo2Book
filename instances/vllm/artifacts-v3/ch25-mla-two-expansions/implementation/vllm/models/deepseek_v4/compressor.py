# SOURCE: vllm/models/deepseek_v4/compressor.py —— ch25 切面（m12：
# DeepseekCompressor——compress→norm→RoPE→store 一条龙的装配面）。
#   CompressorStateCache（L155-L208 减法子集：block_size 布局注释 +
#   get_kv_cache_spec 逐字）+ DeepseekCompressor（L211-L330 减法子集：
#   构造面逐字，forward 的 triton 融合核调用面删——机制归 ch26）。
# SUBTRACTED：CompressorBackend/CompressorMetadata/CompressorMetadataBuilder
#   （L61-L152）——独立 builder 面归 ch14/ch26；save_partial_states 与
#   compress_norm_rope_store 融合核族（L330+）——triton 域。
from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec


# SUBTRACTED: _prefer_two_stage_compressor / _get_c128_boundary（L42-L58）
#   ——ROCm 两段式压缩分支（delete[1] 同族；ROCm 域）
# SUBTRACTED: CompressorBackend / CompressorMetadata /
#   CompressorMetadataBuilder（L61-L152）——压缩器自己的 metadata builder
#   面（split_decodes_and_prefills 消费）归 ch14/ch26


# SOURCE: vllm/models/deepseek_v4/compressor.py:L155 CompressorStateCache
#   —— 减法子集
class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L163-L189 —— 逐字
        #   （block_size 与 KV 块共享物理张量的布局注释——cache 总量÷4 的
        #   存储账在此定形）
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L191-L203 —— 逐字
        #   （状态子缓存也自报 SlidingWindowMLASpec——同一分组原料池）
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    # SOURCE: vllm/models/deepseek_v4/compressor.py:L205 forward（占位空体）
    def forward(self): ...
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L191-L203（锚点双置：声明上方注释同文）
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L191-L203（锚点双置：声明上方注释同文）

    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L207-L208 —— 真身
        #   return CompressorBackend；SUBTRACTED：builder 面（ch14/ch26）
        raise NotImplementedError("CompressorBackend is out of ch25 scope")

    # SUBTRACTED: update_block_table / run（L209-L330 域）——压缩器写腿的
    #   triton 融合核与块表维护归 ch26


# SOURCE: vllm/models/deepseek_v4/compressor.py:L211 DeepseekCompressor
#   —— 减法子集（must_keep 相关：m12——cache 总量÷4 的执行者，装配面逐字）
class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    # SOURCE: vllm/models/deepseek_v4/compressor.py:L222-L330 __init__
    #   —— 减法子集（eager_scratch_pool 参数按 delete[7] 删；
    #   use_fp4_cache 位保留默认 False）
    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L222-L330（锚点双置：声明上方注释同文）
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        # SUBTRACTED: eager_scratch_pool 位——delete[7]
        # SUBTRACTED: fused_wkv_wgate / norm / wape 投影装配段与
        #   state-cache 构造（compressor.py:L240-L330）——forward 的融合核
        #   链（compress→norm→RoPE→store）归 ch26；本章消费其装配契约面
        #   （compress_ratio/head_dim/prefix——spec 与组化的原料）

    # SUBTRACTED: forward（compress→norm→RoPE→store 的 triton 融合核调用
    #   面，L330+）——ch26（NSA/DSA 机制）
