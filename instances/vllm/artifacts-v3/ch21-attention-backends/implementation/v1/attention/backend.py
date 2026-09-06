# Subtract-only companion for v3 ch21 — vllm/v1/attention/backend.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:` with delete[n]
# 编号), plus 章范围外域段以 SUBTRACTED+归属注记收窄（impl-notes §范围裁剪）。
#
# 本章主角文件：注意力后端的协议层——AttentionBackend ABC（四件套身份证 +
# 能力探针 + validate_configuration 聚合器）、CommonAttentionMetadata（跨后端
# 共享元数据核心字段面）、AttentionMetadataBuilder（Common→后端专属 metadata
# 的翻译抽象）、AttentionImplBase/AttentionImpl（后端具体计算抽象）。
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import torch

if TYPE_CHECKING:
    from ..._host_seams import VllmConfigSeam as VllmConfig  # 类型面（seam 装配）
    from .backends.utils import KVCacheLayoutType
    from ..kv_cache_interface import KVCacheSpec, KVQuantMode

from ..kv_cache_interface import get_kv_quant_mode
from ..._host_seams import DeviceCapability  # SOURCE: vllm/platforms/interface.py:L89（HOST SEAM 装载，真身锚见 _host_seams）


# SOURCE: vllm/v1/attention/backend.py:L33-L46 AttentionType ——（逐字）字符串
#   枚举（torch.compile 兼容）
class AttentionType(str, Enum):  # SOURCE: vllm/v1/attention/backend.py
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""


# SOURCE: vllm/v1/attention/backend.py:L49-L53 MultipleOf ——（逐字）kernel 块
#   尺寸约束记法（int=固定值 / MultipleOf=倍数）
class MultipleOf:
    base: int

    def __init__(self, base: int):  # SOURCE: vllm/v1/attention/backend.py:L52 MultipleOf.__init__
        self.base = base


# SOURCE: vllm/v1/attention/backend.py:L56-L401 AttentionBackend —— 后端协议
#   全景（本章抽象核心）：四个抽象 staticmethod 身份证 + 布局声明 +
#   能力探针 + validate_configuration 聚合器
class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    # SOURCE: vllm/v1/attention/backend.py:L59-L64 能力面 ClassVar（逐字）
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # SOURCE: vllm/v1/attention/backend.py:L66-L67 —— KV 写是否含在 forward()
    #   里（FlashAttentionBackend=False → 独立写腿算子先写）
    # Does attention's forward() include kv cache update?
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:  # SOURCE: vllm/v1/attention/backend.py:L69-L71
        return [MultipleOf(1)]

    @staticmethod
    @abstractmethod
    def get_name() -> str:  # SOURCE: vllm/v1/attention/backend.py:L73-L76
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImplBase"]:  # SOURCE: vllm/v1/attention/backend.py:L78-L81
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # SOURCE: vllm/v1/attention/backend.py:L83-L86
        # -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(  # SOURCE: vllm/v1/attention/backend.py:L88-L97
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    def get_kv_cache_block_dim(  # SOURCE: vllm/v1/attention/backend.py:L99-L117
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
    def get_kv_cache_stride_order(  # SOURCE: vllm/v1/attention/backend.py:L119-L148
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """
        Get the physical (memory layout) ordering of the kv cache dimensions.
        Standard attention backends pack K and V into the content dim, giving
        the logical shape [num_blocks, num_heads, block_size, 2 * head_size].
        e.g. if get_kv_cache_stride_order returns (0, 2, 1, 3) then the physical
        ordering of dimensions is
        [num_blocks, block_size, num_heads, 2 * head_size].

        If this function is unimplemented / raises NotImplementedError,
        the physical layout of the KV cache will match the logical shape.

        Args:
            include_num_layers_dimension: if True, includes an additional
                num_layers dimension, which is assumed to be prepended
                to the logical KV cache shape.
                With the above example, a return value (1, 0, 3, 2, 4)
                corresponds to
                [num_blocks, num_layers, block_size, num_heads, 2 * head_size].

                If an additional dimension is NOT included in the returned
                tuple, the physical layout will not include a layers dimension.

        Returns:
            A tuple of ints which is a permutation of range(len(shape)).
        """
        raise NotImplementedError

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:  # SOURCE: vllm/v1/attention/backend.py:L150-L152
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:  # SOURCE: vllm/v1/attention/backend.py:L154-L156
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L158-L161
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L163-L165
        return dtype in cls.supported_dtypes

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L167-L173
        if kv_cache_dtype is None:
            return True
        return (not cls.supported_kv_cache_dtypes) or (
            kv_cache_dtype in cls.supported_kv_cache_dtypes
        )

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L175-L192
        if block_size is None:
            return True

        supported_kernel_block_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_kernel_block_sizes:
            return True

        for supported_size in supported_kernel_block_sizes:
            if isinstance(supported_size, MultipleOf):
                supported_size = supported_size.base
            # With hybrid_blocks feature, the framework-level block size
            # only needs to be a multiple of the kernel's requirement,
            # even if the kernel requires a fixed block_size.
            if block_size % supported_size == 0:
                return True
        return False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:  # SOURCE: vllm/v1/attention/backend.py:L194-L203
        supported_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_sizes:
            return default_block_size

        if cls.supports_block_size(default_block_size):
            return default_block_size

        return min(s.base if isinstance(s, MultipleOf) else s for s in supported_sizes)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L205-L235
        """Whether the backend reads KV pages by the runtime block stride.

        True when ``num_blocks`` is the outermost physical dimension of the KV
        cache, so the backend tolerates a non-contiguous block dim. This gates
        page size padding and cross-layer uniform KV layout.

        Returns:
            True if the backend's physical KV layout is num-blocks-first. False
            otherwise, including when the backend does not define a layered
            stride order.
        """
        try:
            kv_cache_stride_order = cls.get_kv_cache_stride_order(
                include_num_layers_dimension=False
            )
            layered_kv_cache_stride_order = cls.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
        except (AttributeError, NotImplementedError):
            return False

        # Check that attention backend includes a layers dimension.
        if len(layered_kv_cache_stride_order) != len(kv_cache_stride_order) + 1:
            return False

        # stride_order[0] == 0 means num_layers stays first in physical
        # layout (identity permutation), so indexing by block stride is
        # not supported.
        return layered_kv_cache_stride_order[0] != 0

    @classmethod
    def is_mla(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L237-L239
        return False

    @classmethod
    def supports_sink(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L241-L243
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L245-L247
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L249-L251
        return False

    @classmethod
    def is_sparse(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L253-L255
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L257-L259
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L261-L263
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L265-L274
        """Check if backend supports non-causal (bidirectional) attention
        for decoder models.

        Unlike ENCODER_ONLY attention type which implies a different
        execution model, this refers to non-causal attention within the
        standard paged-KV-cache decoder path.
        """
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L276-L278
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L280-L282
        return True

    @classmethod
    def supports_pcp(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L284-L289
        try:
            return cls.get_impl_cls().supports_pcp
        except NotImplementedError:
            return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L291-L298
        """Check if backend supports a given attention type.

        By default, only supports decoder attention.
        Backends should override this to support other attention types.
        """
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:  # SOURCE: vllm/v1/attention/backend.py:L300-L302
        return True

    @classmethod
    def supports_combination(  # SOURCE: vllm/v1/attention/backend.py:L304-L317
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        return None

    @classmethod
    def validate_configuration(  # SOURCE: vllm/v1/attention/backend.py:L319-L393
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
        has_sliding_window: bool = False,
        use_non_causal: bool = False,
        use_batch_invariant: bool = False,
        use_kv_connector: bool = False,
        use_pcp: bool = False,
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
        if has_sliding_window and not cls.supports_sliding_window():
            invalid_reasons.append("sliding window not supported")
        if use_non_causal and not cls.supports_non_causal():
            invalid_reasons.append("non-causal attention not supported")
        if use_batch_invariant and not cls.supports_batch_invariance():
            invalid_reasons.append("batch invariance not supported")
        if use_kv_connector and not cls.supports_kv_connector():
            invalid_reasons.append("KV connector not supported")
        if use_pcp and not cls.supports_pcp():
            invalid_reasons.append("PCP not supported")
        combination_reason = cls.supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
        if combination_reason is not None:
            invalid_reasons.append(combination_reason)
        return invalid_reasons

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":  # SOURCE: vllm/v1/attention/backend.py:L395-L397
        return None

    @classmethod
    def is_ssm(cls) -> bool:  # SOURCE: vllm/v1/attention/backend.py:L399-L401
        return False


# SOURCE: vllm/v1/attention/backend.py:L404-L405 AttentionMetadata ——（逐字）
#   后端私有 metadata 的公共标记基类
class AttentionMetadata:  # SOURCE: vllm/v1/attention/backend.py
    pass


# SOURCE: vllm/v1/attention/backend.py:L408 T
T = TypeVar("T", bound=AttentionMetadata)


# SOURCE: vllm/v1/attention/backend.py:L411-L440 CommonAttentionMetadata ——
#   跨后端共享元数据（核心 10 字段面）
@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    # SOURCE: vllm/v1/attention/backend.py:L420-L422（逐字）
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    # SOURCE: vllm/v1/attention/backend.py:L424-L435（逐字）
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: int
    """Number of requests"""
    # TODO(lucas): rename to num_tokens since it may be padded and this is misleading
    num_actual_tokens: int
    """Total number of tokens in batch"""
    max_query_len: int
    """Longest query in batch"""
    max_seq_len: int
    """Longest context length (may be an upper bound)"""

    # SOURCE: vllm/v1/attention/backend.py:L437-L440 读腿表 + 写腿索引 + 因果位
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool | torch.Tensor = True

    # SUBTRACTED: 长尾可选字段与方法（L442-L600）——delete[1]：logits_
    #   indices_padded/encoder_seq_lens(_cpu)/dcp_local_seq_lens(_cpu)/
    #   positions/is_prefilling/seq_lens_cpu_upper_bound/mm_req_doc_ranges/
    #   rswa_prefix_lens/replayssm_decode_base_cpu/deprecated _seq_lens_cpu 族
    #   + batch_size/naive_query_lens/replace/compute_num_computed_tokens/
    #   token_to_req_indices/unpadded 方法——旁支字段服务于 mm/DCP/R-SWA/
    #   mamba/spec 等扩展态（gpu_model_runner.py L2429-L2449 的 cm_base
    #   kwargs 行连带删，防 TypeError）。


# SOURCE: vllm/v1/attention/backend.py:L603 M
M = TypeVar("M")


# SOURCE: vllm/v1/attention/backend.py:L606-L620 AttentionCGSupport ——（逐字）
#   CG 四档（最弱链降级的值域）
class AttentionCGSupport(Enum):  # SOURCE: vllm/v1/attention/backend.py
    """Constants for the cudagraph support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """Cudagraph supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO cudagraph support"""


# SOURCE: vllm/v1/attention/backend.py:L623-L772 AttentionMetadataBuilder ——
#   Common→后端专属 metadata 的翻译抽象（build 是唯一中心入口）
class AttentionMetadataBuilder(ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    # Do not access directly. Call get_cudagraph_support() instead.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int | None = None
    # Does this backend/builder support updating the block table in existing
    # metadata
    supports_update_block_table: bool = False
    # Whether the builder constructor requires the block-table width.
    requires_block_table_width: ClassVar[bool] = False

    @abstractmethod
    def __init__(  # SOURCE: vllm/v1/attention/backend.py:L637-L648
        self,
        kv_cache_spec: "KVCacheSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(  # SOURCE: vllm/v1/attention/backend.py:L650-L657
        cls: type["AttentionMetadataBuilder"],
        vllm_config: "VllmConfig",
        kv_cache_spec: "KVCacheSpec",
    ) -> AttentionCGSupport:
        """Get the cudagraph support level of this builder class."""
        return cls._cudagraph_support

    def _init_reorder_batch_threshold(  # SOURCE: vllm/v1/attention/backend.py:L659-L689
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        self.reorder_batch_threshold = reorder_batch_threshold
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # If the backend supports spec-as-decode kernels, then we can set
            # the reorder_batch_threshold based on the number of speculative
            # tokens from the config.
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                max_num_queries_for_spec = (
                    1
                    + (2 if speculative_config.parallel_drafting else 1)
                    * speculative_config.num_speculative_tokens
                )
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    max_num_queries_for_spec,
                )

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    @abstractmethod
    def build(  # SOURCE: vllm/v1/attention/backend.py:L691-L709
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
                then speed at execution. Can be used for spec-decode where the
                result of a build call may only be used for few layers/iters.
        """
        raise NotImplementedError

    def update_block_table(  # SOURCE: vllm/v1/attention/backend.py:L711-L724
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        """
        Update the block table for the attention metadata.
        Faster when theres multiple kv-cache groups that create virtually the
        same metadata but just with different block tables.

        Only needs to be implemented if supports_update_block_table is True.
        """
        raise NotImplementedError

    def build_for_cudagraph_capture(  # SOURCE: vllm/v1/attention/backend.py:L726-L736
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        Build attention metadata for CUDA graph capture. Uses build by default.
        Subclasses that override this method should call self.build or
        super().build_for_cudagraph_capture.
        """
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def build_for_drafting(  # SOURCE: vllm/v1/attention/backend.py:L738-L758
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        """
        Build attention metadata for draft model. Uses build by default.

        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    def use_cascade_attention(  # SOURCE: vllm/v1/attention/backend.py:L760-L772
        self,
        common_prefix_len: int,
        query_lens: "object",
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        return False


# SUBTRACTED: AttentionLayer Protocol（L775-L793）——delete[2]：仅供类型检查；
#   Impl.forward 的 layer 参数注解以字符串形式保留（懒求值）。


# SOURCE: vllm/v1/attention/backend.py:L796-L802 AttentionImplBase —— impl
#   抽象基类（DCP/PCP/LSE 族字段与 __new__ 分布式初始化 delete[2] 删）
class AttentionImplBase(ABC, Generic[T]):
    """Base class for attention implementations.

    Contains common attributes and initialization logic shared by both
    standard AttentionImpl and MLAAttentionImpl. Does not define a forward
    method - subclasses define their own forward interfaces.
    """

    # Required attributes that all impls should have
    # SOURCE: vllm/v1/attention/backend.py:L812-L815
    num_heads: int
    head_size: int
    scale: float

    # SUBTRACTED: is_sparse/supports_dense_mha_prefill（L804-L810）与
    #   can_return_lse_for_decode/lse_base_on_e/supports_pcp/supports_dcp/
    #   supports_mtp_with_cp_*/need_to_return_lse_for_decode/supports_
    #   quant_query_input/dcp/pcp/total_cp 字段族（L817-L861）+ __new__ 的
    #   分布式初始化（L863-L889）——delete[2]：精简版单进程无 DCP/PCP/LSE；
    #   组探测的退化路径（dcp_world_size=1）由 FlashAttentionImpl.__init__
    #   的 try/except seam 承载（彼处逐字）。

    def process_weights_after_loading(self, act_dtype: torch.dtype):  # SOURCE: vllm/v1/attention/backend.py:L891-L892
        pass


# SOURCE: vllm/v1/attention/backend.py:L895-L934 AttentionImpl —— 标准 impl
#   抽象（forward 签名所在；fused_* 量化融合钩子 delete[2] 删）
class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """Standard attention implementation with forward method."""

    kv_cache_dtype: str

    @property
    def kv_quant_mode(self) -> "KVQuantMode":  # SOURCE: vllm/v1/attention/backend.py:L900-L903
        """Return the KV cache quantization mode for this layer."""
        return get_kv_quant_mode(self.kv_cache_dtype)

    @abstractmethod
    def __init__(  # SOURCE: vllm/v1/attention/backend.py:L905-L919
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
    def forward(  # SOURCE: vllm/v1/attention/backend.py:L921-L934
        self,
        layer: "object",
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

    # SUBTRACTED: fused_output_quant_supported / fused_qk_norm_rope_kvcache_
    #   supported / fused_rope_kvcache_supported / do_qk_norm_rope_kvcache_
    #   update / do_rope_and_kv_cache_update（L936-L1006）——delete[2]：量化
    #   融合 pass 的探针与落点（ch27 线）。


# SUBTRACTED: MLAAttentionImpl 整类与 subclass_attention_backend/
#   subclass_attention_backend_with_overrides 两工厂（L1009-L1122）——
#   delete[0]：MLA 实现族是 ch24/25 专题；subclass_* 是动态造子类的高级
#   机制，删去不影响四件套主链的可运行与可理解性。
