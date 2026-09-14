# SOURCE: vllm/model_executor/layers/attention/mla_attention.py
# ch26 切面（站 6 的插座侧）：MLAAttention 的 sparse 布线面——
#   __init__（L361-L560 减法：use_sparse/indexer/topk_indices_buffer 形参 +
#   extra_impl_args["topk_indices_buffer"] 显式传参兜底【skip 层没 indexer
#   也能读】+ static_forward_context 自注册 + prefill_backend 落空路径）+
#   _canonicalize_sparse_mla_kv_cache_dtype（L331-L344 逐字）+
#   MLACommonPrefillMetadata（L1400-L1436 逐字字段）+ MLACommonBaseImpl
#   （L2244-L2299 逐字——SparseMLACommonImpl 的基类属性面）。
# SUBTRACTED（章界 → ch25）：MQA 吸收腿/forward_mha 上投影/吸收重排/
#   unified_mla 算子族/metadata builder 全链——ch25 主文件；ch26 只取
#   indexer→impl 的布线插座。forward 直调面保留形态。
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import nn

from vllm import envs  # noqa: F401  (Q/K/V range 常数位——章界外消费)
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context  # noqa: F401  (forward 位)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType, MLAAttentionImpl
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec  # noqa: F401  (类型位)

logger = init_logger(__name__)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L331-L344
#   _canonicalize_sparse_mla_kv_cache_dtype —— 逐字
def _canonicalize_sparse_mla_kv_cache_dtype(
    attn_backend,
    kv_cache_dtype: str,
) -> str:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L331-L344
    #   （锚点双置）
    backend_name = attn_backend.get_name()
    if backend_name == "FLASHMLA_SPARSE" and is_quantized_kv_cache(kv_cache_dtype):
        return "fp8_ds_mla"
    if backend_name == "FLASHINFER_MLA_SPARSE_SM120" and kv_cache_dtype in (
        "auto",
        "fp8",
        "fp8_e4m3",
    ):
        return "fp8_ds_mla"
    return kv_cache_dtype


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1400-L1436
#   MLACommonPrefillMetadata —— 逐字字段面
@dataclass
# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1400-L1436（锚点双置）
class MLACommonPrefillMetadata:
    """Prefill Specific Metadata"""

    @dataclass
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1403-L1425（锚点双置）
    class ChunkedContextMetadata:
        # New for MLA (compared to FlashAttention)
        # For handling chunked prefill
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        seq_lens: torch.Tensor
        context_lens: torch.Tensor
        workspace: torch.Tensor
        token_to_seq: torch.Tensor
        chunk_total_token: list[int]
        has_empty_context: list[bool]

        # for mla DCP
        padded_local_chunk_seq_lens: list[list[int]] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        padded_local_cu_seq_lens: torch.Tensor | None = None
        padded_local_token_to_seq: torch.Tensor | None = None
        cu_seq_lens_lst: list[list[int]] | None = None
        chunk_size: int | None = None
        prefill_tokens_with_context: int | None = None

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: ChunkedContextMetadata | None = None
    q_data_type: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    prefill_backend: object | None = None
    query_lens_cpu: torch.Tensor | None = None
    use_dense_mha: bool = False
    topk_mask_workspace: torch.Tensor | None = None


_A = TypeVar("_A")


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2244-L2299
#   MLACommonBaseImpl —— 逐字（属性面；dense-MHA prefill 机器 → ch25）
class MLACommonBaseImpl(MLAAttentionImpl, Generic[_A]):
    """
    Shared MLA base providing dense-MHA prefill (via the selected
    MLAPrefillBackend) for both dense and sparse impls; subclasses add decode
    (``forward_mqa``).
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:
        #   L2244-L2268 —— 逐字
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2270-L2299
    #   _concat_k_nope_k_pe —— 逐字
    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently concatenate k_nope and k_pe tensors along the last dimension.

        This function avoids the performance penalty of torch.cat with expanded
        non-contiguous tensors by pre-allocating the output and using direct copies.
        """
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:
        #   L2270-L2299（锚点双置）
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        k[..., : k_nope.shape[-1]] = k_nope
        # k_pe is expected to be shape (..., 1, pe_dim) as we add a head dim
        # to it when we call the rotary embedding.
        k[..., k_nope.shape[-1] :] = k_pe.squeeze(-2)
        return k

    def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache,
                    attn_metadata, k_scale, output, output_scale=None):
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py
        #   forward_mha —— 章界位（dense-MHA prefill/上投影机器 → ch25）
        raise NotImplementedError(
            "Dense-MHA prefill machinery (forward_mha up-projection family) "
            "is ch25's domain."
        )


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L347-… MLAAttention
#   —— 减法子集（sparse 布线插座面）
class MLAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class

    This class takes query, and compressed key/value tensors as input.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Run the multi query attention (MQA) kernel.
    3. Return the attention output.
    """

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj,
        dcp_q_replicate: bool = False,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        attn_backend=None,
        use_sparse: bool = False,
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        non_causal_multi_token_decode: bool = False,
        **extra_impl_args,
    ):
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L361-…
        #   —— 减法子集（sparse 布线逐字；KV 量化属性/批不变式/prefill 家族
        #   按 host 消费面承载）
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj
        self.dcp_q_replicate = dcp_q_replicate
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix
        self.indexer = indexer
        self.non_causal_multi_token_decode = non_causal_multi_token_decode
        self.num_kv_heads = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False
        self.quant_config = quant_config

        if attn_backend is not None:
            assert attn_backend.is_mla(), (
                f"MLAAttention: attn_backend must be an MLA backend, "
                f"got {attn_backend.get_name()} instead"
            )
            self.attn_backend = attn_backend
        else:
            self.attn_backend = get_attn_backend(
                self.head_size,
                torch.get_default_dtype(),
                kv_cache_dtype,
                use_mla=True,
                use_sparse=use_sparse,
                num_heads=self.num_heads,
            )

        normalized_kv_cache_dtype = _canonicalize_sparse_mla_kv_cache_dtype(
            self.attn_backend, kv_cache_dtype
        )
        if normalized_kv_cache_dtype != kv_cache_dtype:
            if cache_config is not None:
                cache_config.cache_dtype = normalized_kv_cache_dtype
            kv_cache_dtype = normalized_kv_cache_dtype
            logger.info_once(
                "Using %s KV cache format for %s backend.",
                kv_cache_dtype,
                self.attn_backend.get_name(),
            )

        # Initialize KV cache quantization attributes
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales

        # SUBTRACTED（章界 → ch25 delete[2] 同款）：融合输出量化面
        #   （_detect_output_quant_key/quant_key 探测与临时缓冲交换）——
        #   ch25 的 delete[2] 域

        # Sparse MLA reads top-k indices from a shared buffer. Pass it
        # explicitly so backbone "skip" layers (indexer=None) still find it.
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:
        #   L482-L484 —— 逐字
        if use_sparse:
            extra_impl_args["topk_indices_buffer"] = topk_indices_buffer

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=kv_b_proj,
            indexer=indexer,
            **extra_impl_args,
        )
        self.q_pad_num_heads = getattr(self.impl, "q_pad_num_heads", None)
        self.use_direct_call = not current_platform.opaque_attention_op()

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.prefill_backend = None
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:
        #   L519-L555 —— 减法子集（sparse 无 dense-MHA prefill 时落
        #   top-k MQA-only 的真实降级路径）
        if self.impl.is_sparse and not self.impl.supports_dense_mha_prefill:
            logger.warning_once(
                "Sparse MLA impl has no dense-MHA prefill path; using the top-k "
                "MQA path only."
            )
            self.prefill_backend = None
        else:
            try:
                from vllm.v1.attention.backends.mla.prefill import (
                    get_mla_prefill_backend,
                )

                prefill_backend_cls = get_mla_prefill_backend(vllm_config)
            except ValueError:
                if (
                    not self.impl.is_sparse
                    or vllm_config.attention_config.mla_prefill_backend is not None
                ):
                    raise
                logger.warning_once(
                    "No MLA prefill backend supports this model; sparse MLA will "
                    "use the top-k MQA path only (no dense-MHA prefill)."
                )
                self.prefill_backend = None
            else:
                self.prefill_backend = prefill_backend_cls(
                    num_heads=self.num_heads,
                    scale=self.scale,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    vllm_config=vllm_config,
                )

        self.kv_cache = torch.tensor([])

        self.use_sparse = use_sparse

        # SUBTRACTED（章界 → ch25/ch34）：dcp_a2a 探测与 aiter triton
        #   fp8/fp4 bmm 旗标、_DecodeConcatQuantFP8/QuantFP8 输出量化算子
        #   （ch25 delete[1]/[2] 同族）
        self._vllm_config = get_current_vllm_config()
        self._chunked_prefill_workspace_size: int | None = None

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:
        #   L1140-L1152 —— 逐字（主 MLA 的 spec 自报——IndexCache 的『第二
        #   本账』对照面）
        from vllm.v1.kv_cache_interface import (
            MLAAttentionSpec,
            get_kv_quant_mode,
        )
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=self.kv_cache_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            non_causal_multi_token_decode=self.non_causal_multi_token_decode,
        )

    def get_attn_backend(self):
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py
        #   get_attn_backend —— 装配期后端类回查位
        return self.attn_backend

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py forward
        #   —— 章界位（KV cache 写腿 do_kv_cache_update + MHA/MQA 混批切刀 +
        #   吸收腿 → ch25 主文件；本章只立 indexer→impl 的装配布线插座）
        raise NotImplementedError(
            "MLAAttention.forward's mixed-batch dispatch and absorbed-MQA leg "
            "are ch25's domain; ch26 carries the assembly wiring only."
        )
