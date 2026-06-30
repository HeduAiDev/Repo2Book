# vllm_ascend/attention/abstract.py —— subtract-only 精简版（DSA 自有的注意力抽象基类）
#
# 本章用途：佐证「DSA 不走 vLLM 的 MLAAttentionImpl，自起一套」。AscendDSAImpl 继承的就是
# 这里的 DSAAttentionImpl —— 它是 vLLM AttentionImpl[T] 的子类，但签名/职责与 MLA 那条线完全
# 不同（带 compressor / indexer 双子模块的低秩稀疏注意力）。SFA 那条线则直接继承 vLLM 的
# MLAAttentionImpl（见 sfa_v1.py），二者形成「复用 vs 自起一套」的对照。
#
# 本文件几乎逐字保留（抽象基类本就极薄），仅注释说明用途。
from abc import abstractmethod
from typing import Generic, TypeVar

import torch
from vllm.v1.attention.backend import AttentionImpl, AttentionLayer


# SOURCE: vllm_ascend/attention/abstract.py:L11
class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


# SOURCE: vllm_ascend/attention/abstract.py:L18
class DSAAttentionImpl(AttentionImpl[T], Generic[T]):
    @abstractmethod
    # SOURCE: vllm_ascend/attention/abstract.py:L19
    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    # SOURCE: vllm_ascend/attention/abstract.py:L38
    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_cq: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
