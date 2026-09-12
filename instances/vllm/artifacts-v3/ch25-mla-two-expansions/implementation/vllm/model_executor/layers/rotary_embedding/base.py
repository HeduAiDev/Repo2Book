# SOURCE: vllm/model_executor/layers/rotary_embedding/base.py
# ch25 消费面（m02）：RotaryEmbeddingBase（inv_freq/cos_sin_cache 构造）+
# RotaryEmbedding（forward_static/forward_native 静态应用）——数学逐字
#（ch23 同款切面：@CustomOp.register 与平台 kernel 派发位删除——host 直调
# native；真实 CustomOp 面归 ch19/ch23）。
from __future__ import annotations

import torch

from .common import ApplyRotaryEmb


# SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L14-L15
#   RotaryEmbeddingBase —— 减法子集（真实继承 CustomOp；本切面为纯
#   nn.Module 承载同一构造与应用数学）
class RotaryEmbeddingBase(torch.nn.Module):
    """Original rotary positional embedding."""

    # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L21-L105
    #   __init__ —— 减法子集（属性面 + cos_sin_cache 构造逐字；
    #   flashinfer/aiter/bf16 平台位删除）
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L21-L105
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        # SUBTRACTED: use_flashinfer/use_aiter 平台位（base.py:L41-L63）——
        #   kernel 派发域

        if init_cache:
            cache = self._compute_cos_sin_cache()
            cache = cache.to(dtype)
            self.cos_sin_cache: torch.Tensor
            self.register_buffer("cos_sin_cache", cache, persistent=False)

        self.apply_rotary_emb = ApplyRotaryEmb(
            is_neox_style=self.is_neox_style,
        )

    # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L107-L118
    #   _compute_inv_freq（逐字）
    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L107-L118（锚点双置：声明上方注释同文）
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L120-L128
    #   _compute_cos_sin_cache（逐字——cos|sin 拼接缓存）
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L120-L128（锚点双置：声明上方注释同文）
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L110-L133
    #   _match_cos_sin_cache_dtype —— 减法子集
    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L110-L133（锚点双置：声明上方注释同文）
        cos_sin_cache = self.cos_sin_cache
        if (
            cos_sin_cache.device == query.device
            and self.cos_sin_cache.dtype == query.dtype
        ):
            return cos_sin_cache
        return cos_sin_cache.to(query.device, dtype=query.dtype)


# SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L139 RotaryEmbedding
class RotaryEmbedding(RotaryEmbeddingBase):
    # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L141-L160
    #   __init__（逐字——直通基类缓存构造）
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L141-L160
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            init_cache=init_cache,
        )

    # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L178-L216
    #   forward_static（逐字——按 positions 行选 cos_sin、rotary_dim 分段、
    #   key 可 None）
    @staticmethod
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L178-L216
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        query_rot = ApplyRotaryEmb.forward_static(
            query_rot,
            cos,
            sin,
            is_neox_style,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, head_size)
            key_rot = key[..., :rotary_dim]
            key_pass = key[..., rotary_dim:]
            key_rot = ApplyRotaryEmb.forward_static(
                key_rot,
                cos,
                sin,
                is_neox_style,
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L218-L232
    #   forward_native（逐字）
    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation of forward()."""
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:L218-L232
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )

    def __call__(self, positions, query, key=None):
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py 派发位
        #   —— HOST SEAM：host 直调 forward_native（forward_cuda 派发删除）
        return self.forward_native(positions, query, key)

    # SUBTRACTED: forward_cuda/forward_hip（base.py:L234-L~280）——CUDA/AITER
    #   kernel 派发；host 直调 native
