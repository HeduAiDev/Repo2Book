"""ch24 精简版 — 注意力后端抽象（只做减法）。

对应 vllm/v1/attention/backend.py。本模块保留四件抽象核心：
  - AttentionBackend：所有后端的抽象基类（六个抽象 staticmethod 身份证 + 一组
    supports_*/is_* 能力探针 + 聚合器 validate_configuration + get_kv_cache_shape/
    get_kv_cache_stride_order/get_kv_cache_block_dim 的 KV cache 约定）。
  - CommonAttentionMetadata：跨层跨后端共享的 per-batch 元数据（核心 8 字段，f14 回收的
    block_table_tensor / slot_mapping 在此作为接口字段被接住）。
  - AttentionMetadataBuilder：把 CommonAttentionMetadata 翻译成后端专属 metadata 的抽象
    builder，build() 是唯一中心入口。
  - AttentionImpl：后端具体计算的抽象，forward 算注意力、do_kv_cache_update 写 KV。

控制流、命名、签名与真实 vLLM 一致，删除项均标 # SUBTRACTED。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, Generic, TypeVar

import torch


# SOURCE: vllm/v1/attention/backend.py (AttentionType — 本章只用 DECODER/ENCODER 几个常量)
class AttentionType:
    # SUBTRACTED: 真实 AttentionType 定义在 vllm/attention 顶层，含更多类型字符串；本章
    # selector/validate 只需 DECODER 作默认，其余作能力探针对照，删之不影响主链。
    DECODER = "decoder"
    ENCODER = "encoder"
    ENCODER_ONLY = "encoder_only"
    ENCODER_DECODER = "encoder_decoder"


# SOURCE: vllm/v1/attention/backend.py:L55 (AttentionBackend)
class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # Does attention's forward() include kv cache update?
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:  # SOURCE: vllm/v1/attention/backend.py:L68 (get_supported_kernel_block_sizes)
        # SUBTRACTED: 真实返回 [MultipleOf(1)]（MultipleOf 是个轻量包装类，表达「块大小须为
        # base 的倍数」）。本章用具体整数列表表达同一约定，supports_block_size 据此判定。
        return [1]

    @staticmethod
    @abstractmethod
    def get_name() -> str:  # SOURCE: vllm/v1/attention/backend.py:L73 (get_name)
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:  # SOURCE: vllm/v1/attention/backend.py:L79 (get_impl_cls)
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:  # SOURCE: vllm/v1/attention/backend.py:L85 (get_builder_cls)
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(  # SOURCE: vllm/v1/attention/backend.py:L91 (get_kv_cache_shape)
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L98 (get_kv_cache_block_dim)
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        """Discover which tensor dim is the block index, since different
        backends lay out dims differently."""
        _S = 1234567
        shape = cls.get_kv_cache_shape(
            _S,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return shape.index(_S)

    @staticmethod
    # SOURCE: vllm/v1/attention/backend.py:L119 (get_kv_cache_stride_order)
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """
        Get the physical (memory layout) ordering of the kv cache dimensions.

        If this function is unimplemented / raises NotImplementedError,
        the physical layout of the KV cache will match the logical shape.

        Returns:
            A tuple of ints which is a permutation of range(len(shape)).
        """
        raise NotImplementedError

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L155 (get_supported_head_sizes)
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    # ---- 能力探针：每个返回 bool，validate_configuration 据此聚合不合法原因 ----
    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L158 (supports_head_size)
    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L162 (supports_dtype)
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in cls.supported_dtypes

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L165 (supports_kv_cache_dtype)
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return (not cls.supported_kv_cache_dtypes) or (
            kv_cache_dtype in cls.supported_kv_cache_dtypes
        )

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L173 (supports_block_size)
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        supported_kernel_block_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_kernel_block_sizes:
            return True
        for supported_size in supported_kernel_block_sizes:
            # With hybrid_blocks feature, the framework-level block size
            # only needs to be a multiple of the kernel's requirement.
            if block_size % supported_size == 0:
                return True
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L204 (is_mla)
    def is_mla(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L208 (supports_sink)
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L212 (supports_alibi_sqrt)
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L216 (supports_mm_prefix)
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L220 (is_sparse)
    def is_sparse(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L224 (supports_per_head_quant_scales)
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L228 (supports_non_causal)
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L240 (supports_batch_invariance)
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L244 (supports_attn_type)
    def supports_attn_type(cls, attn_type: str) -> bool:
        """By default, only supports decoder attention."""
        return attn_type == AttentionType.DECODER

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L253 (supports_compute_capability)
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        return True

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L257 (supports_combination)
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        return None

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L271 (validate_configuration)
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: "DeviceCapability",
        attn_type: str,
        use_non_causal: bool = False,
        use_batch_invariant: bool = False,
    ) -> list[str]:
        invalid_reasons = []
        if not cls.supports_head_size(head_size):
            invalid_reasons.append("head_size not supported")
        if not cls.supports_dtype(dtype):
            invalid_reasons.append("dtype not supported")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            invalid_reasons.append("kv_cache_dtype not supported")
        if not cls.supports_block_size(block_size):
            invalid_reasons.append("block_size not supported")
        if use_mm_prefix and not cls.supports_mm_prefix():
            invalid_reasons.append(
                "partial multimodal token full attention not supported"
            )
        if use_mla != cls.is_mla():
            if use_mla:
                invalid_reasons.append("MLA not supported")
            else:
                invalid_reasons.append("non-MLA not supported")
        if has_sink and not cls.supports_sink():
            invalid_reasons.append("attention sinks not supported")
        if use_sparse != cls.is_sparse():
            if use_sparse:
                invalid_reasons.append("sparse not supported")
            else:
                invalid_reasons.append("non-sparse not supported")
        if use_per_head_quant_scales and not cls.supports_per_head_quant_scales():
            invalid_reasons.append("per-head quant scales not supported")
        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append("compute capability not supported")
        if not cls.supports_attn_type(attn_type):
            invalid_reasons.append(f"attention type {attn_type} not supported")
        if use_non_causal and not cls.supports_non_causal():
            invalid_reasons.append("non-causal attention not supported")
        if use_batch_invariant and not cls.supports_batch_invariance():
            invalid_reasons.append("batch invariance not supported")
        combination_reason = cls.supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
        )
        if combination_reason is not None:
            invalid_reasons.append(combination_reason)
        return invalid_reasons

    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L336 (get_required_kv_cache_layout)
    def get_required_kv_cache_layout(cls) -> str | None:
        return None


# SOURCE: vllm/v1/attention/backend.py:L344 (AttentionMetadata)
class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


@dataclass
# SOURCE: vllm/v1/attention/backend.py:L352 (CommonAttentionMetadata)
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""
    max_query_len: int
    """Longest query in batch"""
    max_seq_len: int
    """Longest context length (may be an upper bound)"""

    # f14 回收：这两个字段从 model_runner 那头组装、在此被接住，backend 这头消费。
    block_table_tensor: torch.Tensor   # 请求 → 逻辑块号列表（PagedAttention 读用）
    slot_mapping: torch.Tensor         # token → 物理槽位（PagedAttention 写用）

    causal: bool = True

    # SUBTRACTED: 真实 CommonAttentionMetadata 还有约 12 个可选字段（logits_indices_padded/
    # encoder_seq_lens/dcp_local_seq_lens/positions/is_prefilling/seq_lens_cpu_upper_bound
    # + 已弃用的 _seq_lens_cpu/_num_computed_tokens_cpu）以及 naive_query_lens/
    # compute_num_computed_tokens/seq_lens_cpu/num_computed_tokens_cpu/unpadded 等方法
    # （backend.py:L383-L493）。它们服务 cross-attention / 解码上下文并行(DCP) / 向后兼容等
    # 旁支；保留核心 8 字段（含 block_table_tensor/slot_mapping/causal）即可讲清接口契约，
    # 删旁支不影响主流程数值。
    def batch_size(self) -> int:  # SOURCE: vllm/v1/attention/backend.py:L419 (CommonAttentionMetadata.batch_size)
        return self.seq_lens.shape[0]

    def replace(self, **kwargs) -> "CommonAttentionMetadata":  # SOURCE: vllm/v1/attention/backend.py:L427 (CommonAttentionMetadata.replace)
        return replace(self, **kwargs)


M = TypeVar("M")


# SOURCE: vllm/v1/attention/backend.py:L495 (AttentionCGSupport)
class AttentionCGSupport(Enum):
    """Constants for the cudagraph support of the attention backend."""

    ALWAYS = 3
    UNIFORM_BATCH = 2
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    NEVER = 0


# SOURCE: vllm/v1/attention/backend.py:L516 (AttentionMetadataBuilder)
class AttentionMetadataBuilder(ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold: int | None = None
    # Does this backend/builder support updating the block table in existing metadata
    supports_update_block_table: bool = False

    @abstractmethod
    def __init__(  # SOURCE: vllm/v1/attention/backend.py:L528 (AttentionMetadataBuilder.__init__)
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    # SUBTRACTED: get_cudagraph_support / _init_reorder_batch_threshold（backend.py:L540-L580）
    # 是 CUDA graph 捕获与 batch 重排阈值的支持级别声明；本章主线（Common→专属 metadata 翻译）
    # 不进 cudagraph 路径，删之不破坏 build() 主链。

    @abstractmethod
    # SOURCE: vllm/v1/attention/backend.py:L582 (build)
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.

        Args:
            common_prefix_len: The length of the common prefix of the batch.
            common_attn_metadata: The common attention metadata.
            fast_build: The meta-data will prioritize speed of building over
                then speed at execution.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/attention/backend.py:L601 (update_block_table)
    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        """
        Update the block table for the attention metadata.
        Faster when theres multiple kv-cache groups that create virtually the
        same metadata but just with different block tables.
        """
        raise NotImplementedError

    # SUBTRACTED: build_for_cudagraph_capture / build_for_drafting / use_cascade_attention
    # （backend.py:L617-L663）是 cudagraph 捕获、投机解码起草、cascade 判定的默认实现，都委托给
    # build()；本章只讲 build() 这个中心入口，删默认壳不影响翻译主链。


# SOURCE: vllm/v1/attention/backend.py:L685 (AttentionImplBase)
class AttentionImplBase(ABC, Generic[T]):
    """Base class for attention implementations.

    Contains common attributes shared by both standard AttentionImpl and
    MLAAttentionImpl. Does not define a forward method.
    """

    num_heads: int
    head_size: int
    scale: float

    # SUBTRACTED: 真实 AttentionImplBase 含 can_return_lse_for_decode/supports_pcp/
    # dcp_world_size 等并行相关属性与 __new__ 里初始化 DCP/PCP 通信组的逻辑
    # （backend.py:L694-L760）；本章 host 非分布式，dcp_world_size 恒为 1，删并行初始化不影响
    # 单卡主路径。这里直接给一个最小 __init__ 占位（真实里 __new__ 设这些 group 大小）。
    def __init__(self):  # SOURCE: vllm/v1/attention/backend.py:L731 (AttentionImplBase.__new__/init)
        self.dcp_world_size = 1
        self.dcp_rank = 0


# SOURCE: vllm/v1/attention/backend.py:L763 (AttentionImpl)
class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """Standard attention implementation with forward method."""

    kv_cache_dtype: str

    @abstractmethod
    def __init__(  # SOURCE: vllm/v1/attention/backend.py:L774 (AttentionImpl.__init__)
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    # SOURCE: vllm/v1/attention/backend.py:L788 (forward)
    def forward(
        self,
        layer: "AttentionLayer",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    # SOURCE: vllm/v1/attention/backend.py:L910 (do_kv_cache_update — 抽象声明，FA 等覆盖)
    def do_kv_cache_update(
        self,
        layer: "AttentionLayer",
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # SUBTRACTED: 真实里 AttentionImpl.do_kv_cache_update 由各后端覆盖；标准后端把它声明在
        # impl 上、由 FlashAttentionImpl 给出具体实现（见 flash_attn.py）。这里留抽象壳。
        raise NotImplementedError

    # SUBTRACTED: fused_output_quant_supported / fused_rope_kvcache_supported /
    # do_rope_and_kv_cache_update（backend.py:L805-L840）是输出量化融合、RoPE+KVcache 融合的
    # 可选钩子；本章非量化、非融合主路径不调用，删之不影响 forward 主链。


# SUBTRACTED: MLAAttentionImpl / SparseMLAAttentionImpl 两个 MLA 变体类整体
# （backend.py:L843-L1010）。MLA（多头潜在注意力）是独立专题，本章主线是标准（非 MLA）后端 +
# FlashAttention；删去不影响 AttentionBackend/Common/Builder/Impl/FlashAttention 主链。

# SUBTRACTED: subclass_attention_backend / subclass_attention_backend_with_overrides
# 两个工厂函数（backend.py:L1013-L1034）——给特定后端动态造子类的高级机制，与本章主线无关。
