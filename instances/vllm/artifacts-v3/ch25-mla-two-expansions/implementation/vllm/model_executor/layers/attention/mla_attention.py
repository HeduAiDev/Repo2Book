# SOURCE: vllm/model_executor/layers/attention/mla_attention.py
# —— ch25 主文件 2（站 3/4/5/8/10/11/12/13 全部 + m01-m09/m16）
# 全文减法。SUBTRACTED 挂 dossier.subtraction_plan.delete 批准项编号：
#   delete[0]：DCP/PCP 分布式路径（all-gather/LSE 归并/_context_parallel_
#     compute_prefill_context/reorg_kvcache/pcp 钩子/dcp_a2a/W_UK_T_dcp_qrep/
#     dcp_q_replicate）
#   delete[1]：ROCm aiter 三分支（fp4/fp8 bmm 替换与 mxfp4/fp8 量化+1024 档
#     预编译循环）
#   delete[2]：融合输出量化整族（_detect_output_quant_key/quant_key 分支/
#     _DecodeConcatQuantFP8/QuantFP8/supports_quant_query_input 分支）
#   delete[3]：稀疏 MLA 分支（use_mha/use_masked_mha 覆写+_DSV32 阈值表/
#     topk 字段）
#   delete[4]：spec-decode 专有（QueryLenSupport 的 spec 联动已收在
#     backend.py；build_for_cudagraph_capture）
#   delete[5]：batch-invariant 告警位
#   delete[8]：maybe_transfer_kv_layer 钩子与 non-causal multi-token decode
#   章界：kv_cache_dtype_skip_layers 检查保留（非批准删除项）；
#     reshape_query_for_spec_decode 消费点在 flashmla.py 侧记删。
# 文件头数学文档（L3-L188）**全文保留**——两种展开的数学真相源。
"""
# MLA Common Components

This file implements common components for MLA implementations.

First we define:

Sq      as Q sequence length
Skv     as KV sequence length

MLA has two possible ways of computing, a data-movement friendly approach and a
compute friendly approach. We generally want to use the compute friendly
approach for "prefill" (i.e. the ratio Sq / Skv is relatively large, often near
1) and the data-movement friendly approach for "decode" (i.e. the ratio
Sq / Skv is small, often near 0).

NOTE what we deem small and large is currently determined by if it is labelled
prefill or decode by the scheduler, but this is something we should probably
tune.

Main reference: DeepseekV2 paper, and FlashInfer Implementation
(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

Deepseek's MLA attention works the following way:
* Use a single latent vector to represent the per-token entry of the KV cache.
* For decode (i.e. the memory friendly approach) the attention "simulates" a
multi-head attention, while the compute is similar to multi-query attention.

Below is an example of both paths assuming batch size = 1

## More Extent Definitions:

C           Context length, `Skv - Sq`
H           hidden size
N           number of attention heads
Lq          latent dimension for Q              1536 in DSV3
Lkv         latent dimension for K/V            512 in DSV3
P           nope dimension, no rope.            128 in DSV3
R           rope dimension, goes through rope.  64 in DSV3
V           V head dim.                         128 in DSV3

## Vector/Matrix Definitions

h_t         hidden states (input to attention)  shape [Sq, H]
q_c         latent/compressed Q                 shape [Sq, Lq]
q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
q_pe        uncompressed Q (rope)               shape [Sq, N, R]
kv_c        latent/compressed KV                shape [Skv, Lkv]
k_pe        decoupled k position embeddings     shape [Skv, R]
new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
new_k_pe    new k_pe from current iter          shape [Sq, R]
cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
cache_k_pe  cached k_pe from previous iters     shape [C, R]
W_DQ        project h_t to q_c                  shape [H, Lq]
W_UQ        project q_c to q_nope               shape [Lq, N * P]
W_QR        project q_c to q_pe                 shape [Lq, N * R]
W_DKV       project h_t to kv_c                 shape [H, Lkv]
W_UK        project kv_c to k_nope              shape [Lkv, N, P]
W_KR        project h_t to k_pe                 shape [H, R]
W_UV        project kv_c to v                   shape [Lkv, N, V]
W_O         project v to h_t                    shape [N * V, H]


## Compute Friendly Approach (i.e. "forward_mha"):

q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      sdpa_o shape [Sq, N, V]
sdpa_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
)
return sdpa_o @ W_O

NOTE: in the actual code,
    `kv_b_proj` is [W_UK; W_UV] concatenated per head
    `q_b_proj` is [W_UQ; W_QR] concatenated per head
    `out_proj` is W_O


## Data-Movement Friendly Approach (i.e. "forward_mqa"):

Runtime
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(-1, N, P)
ql_nope  = einsum("snh,lnh->snl", q_nope, W_UK)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)

// MQA with QK headdim = Lkv + R
//           V headdim = Lkv
//      sdpa_o shape [Sq, N, Lkv]
// NOTE: this is less compute-friendly since Lkv > P
//       but is more data-movement friendly since its MQA vs MHA
sdpa_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)

o = einsum("snl,lnv->snv", sdpa_o.reshape(-1, N, Lkv), W_UV)
return o.view(-1, N * V) @ W_O


## Chunked Prefill

For chunked prefill we want to use the compute friendly algorithm. We are
assuming sufficiently large Sq / Skv ratio, in the future may want to switch to
the data-movement friendly approach if the chunk (i.e. `Sq`) is small.

However, the compute-friendly approach can potentially run out of memory if Skv
is large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`

To mitigate this, we chunk the computation of attention with respect to the
current context (i.e. `cache_kv_c` and `cache_k_pe`) so that we can used a
fixed workspace size.

The chunked prefill approach is as follows:

MCC        Max chunk of context to process per iter, computed dynamically,
           used to bound the memory usage

q_c        = h_t @ W_DQ
q_nope     = (q_c @ W_UQ).view(Sq, N, P)
q_pe       = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c   = h_t @ W_DKV
new_k_pe   = RoPE(h_t @ W_KR)
new_k_nope = (new_kv_c @ W_UK.view(Lkv, N * P)).view(Sq, N, P)
new_v      = (new_kv_c @ W_UV.view(Lkv, N * V)).view(Sq, N, V)

// MHA between queries and new KV
//     with QK headdim = P + R
//           V headdim = V
//    curr_o   shape [Sq, N, V]
//    curr_lse shape [N, Sq], this is just order FA returns
curr_o, curr_lse = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([new_k_nope, new_k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    new_v,
    causal=True,
    return_softmax_lse=True
)

// Compute attention with the already existing context
for chunk_idx in range(cdiv(C, MCC)):
    chunk_start  = chunk_idx * MCC
    chunk_end    = min(chunk_start + MCC, C)
    Sc           = chunk_end - chunk_start
    cache_kv_c_chunk   = cache_kv_c[chunk_start:chunk_end]
    cache_k_pe_chunk   = cache_k_pe[chunk_start:chunk_end]
    cache_k_nope_chunk = (cache_kv_c_chunk @ W_UK).view(-1, N, P)
    cache_v_chunk      = (cache_kv_c_chunk @ W_UV).view(-1, N, V)

    chunk_o, chunk_lse = scaled_dot_product_attention(
        torch.cat([q_nope, q_pe], dim=-1),
        torch.cat([cache_k_nope_chunk,
                   cache_k_pe_chunk.unsqueeze(1).expand(-1, N, -1)],
                   dim=-1),
        cache_v_chunk,
        causal=False,
        return_softmax_lse=True
    )

    curr_o, curr_lse = merge_attn_states(
        suffix_output=curr_o,
        suffix_lse=curr_lse,
        prefix_output=chunk_o,
        prefix_lse=chunk_lse,
    )

return curr_o @ W_O
"""

import functools
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Generic, TypeVar, cast

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
# SUBTRACTED: from vllm._aiter_ops import rocm_aiter_ops ——delete[1]（ROCm）
# SUBTRACTED: from vllm.distributed.parallel_state import (get_dcp_group,
#   get_tp_group, is_global_first_rank)——delete[0]（DCP/PCP）
# SUBTRACTED: from vllm.model_executor.layers.attention.kv_transfer_utils
#   import maybe_transfer_kv_layer——delete[8]（KV connector 归 ch16）
# SUBTRACTED: from vllm.model_executor.layers.attention.pcp import ...——
#   delete[0]（PCP 钩子）
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
# SUBTRACTED: from vllm.model_executor.custom_op import CustomOp——
#   delete[2]（_DecodeConcatQuantFP8 随融合量化族删）
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant,
    get_attention_context,
    set_default_quant_scales,
    should_load_quant_weights,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
)
# SUBTRACTED: from vllm.model_executor.layers.quantization.input_quant_fp8
#   import QuantFP8——delete[2]
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
# SUBTRACTED: QuantKey/kFp8* 常量族 import——delete[2]（融合输出量化）
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv, round_down
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.prefill import (
    MLAPrefillBackend,
    get_mla_prefill_backend,
)
from vllm.v1.attention.backends.utils import (
    get_num_attention_heads_from_layers,
    split_decodes_and_prefills,
)
# SUBTRACTED: from vllm.v1.attention.ops.common import cp_lse_ag_out_ar,
#   cp_lse_ag_out_rs / dcp_alltoall ——delete[0]（DCP LSE 归并）
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_merge_attn_states import mask_empty_context
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

_FP8_DTYPE = current_platform.fp8_dtype()

# SUBTRACTED: _detect_output_quant_key（L295-L328）——delete[2]（融合输出
#   量化整族：FP8/FP4 输出打包的探测与临时缓冲交换）
# SUBTRACTED: _canonicalize_sparse_mla_kv_cache_dtype（L331-L344）——
#   delete[3]（FLASHMLA_SPARSE/FLASHINFER_MLA_SPARSE 后端的 fp8_ds_mla
#   规范化——稀疏 MLA 归 ch26）


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L347 MLAAttention
#   —— 减法子集（must_keep：内层插座——分流决策/写腿/spec 自报都在这）
class MLAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class

    This class takes query, and compressed key/value tensors as input.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L361-L589
    #   __init__ —— 减法子集
    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        dcp_q_replicate: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
        attn_backend: type[AttentionBackend] | None = None,
        use_sparse: bool = False,
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        non_causal_multi_token_decode: bool = False,
        **extra_impl_args,
    ):
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L382-L398
        #   属性面 —— 逐字
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
        # SUBTRACTED: self.W_UK_T_dcp_qrep = None（L392）——delete[0]
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix
        self.indexer = indexer
        self.non_causal_multi_token_decode = non_causal_multi_token_decode
        self.num_kv_heads = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L400-L405
        #   cache_config 面 —— 逐字
        if cache_config is not None:
            kv_cache_dtype: CacheDType = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False
        self.quant_config = quant_config

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L408-L419
        #   kv_cache_dtype_skip_layers —— 逐字
        if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
            from vllm.model_executor.models.utils import extract_layer_index

            layer_idx = extract_layer_index(prefix)
            if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
                kv_cache_dtype = "auto"
                calculate_kv_scales = False
            logger.debug(
                "Layer %s: kv_cache_dtype=%s",
                prefix,
                kv_cache_dtype,
            )

        dtype = torch.get_default_dtype()
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L422-L436
        #   后端选择（断言 is_mla 家族身份证）—— 逐字（must_keep）
        if attn_backend is not None:
            assert attn_backend.is_mla(), (
                f"MLAAttention: attn_backend must be an MLA backend, "
                f"got {attn_backend.get_name()} instead"
            )
            self.attn_backend = attn_backend
        else:
            self.attn_backend = get_attn_backend(
                self.head_size,
                dtype,
                kv_cache_dtype,
                use_mla=True,
                use_sparse=use_sparse,
                num_heads=self.num_heads,
            )

        # SUBTRACTED: _canonicalize_sparse_mla_kv_cache_dtype 与
        #   FLASHINFER_MLA_SPARSE 告警（L438-L459）——delete[3]

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L461-L464
        #   KV cache quant 属性 —— 逐字
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        _init_kv_cache_quant(self, quant_config, prefix)

        # SUBTRACTED: prefix caching + batch invariance 告警
        #   （L466-L479）——delete[5]（VLLM_BATCH_INVARIANT 默认 False）

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L481-L508
        #   impl 构造 —— 减法子集（sparse 的 topk_indices_buffer 传递位删）
        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(  # type: ignore[assignment]
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
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L510 —— 逐字
        self.use_direct_call = not current_platform.opaque_attention_op()

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L512-L518
        #   static_forward_context 自注册 —— 逐字（重复 layer_name 即 raise）
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L520-L550
        #   第二套 prefill 后端装配 —— 减法子集（sparse 支删）
        self.prefill_backend: MLAPrefillBackend | None
        # SUBTRACTED: impl.is_sparse 的 no-dense-MHA-prefill 告警支
        #   （L521-L526）——delete[3]
        try:
            prefill_backend_cls = get_mla_prefill_backend(vllm_config)
        except ValueError:
            # SUBTRACTED: sparse 的 top-k-only 回退支（L531-L540）——delete[3]
            raise
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

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L552 —— 逐字
        self.kv_cache = torch.tensor([])

        self.use_sparse = use_sparse

        # SUBTRACTED: self.dcp_a2a 探测（L556-L561）——delete[0]

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L563-L566
        #   q/k/v 量程常数 —— 逐字
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

        # SUBTRACTED: is_aiter_triton_fp8/fp4_bmm_enabled 两旗标
        #   （L568-L575）——delete[1]（ROCm aiter；CUDA 恒 False 的同型路径）

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L577-L579
        #   forward_impl 属性面 —— 逐字
        self._vllm_config = get_current_vllm_config()
        self._chunked_prefill_workspace_size: int | None = None
        # SUBTRACTED: _decode_concat_quant_fp8_op / _quant_fp8_op 构造
        #   （L580-L589）——delete[2]

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L591-L599
    #   chunked_prefill_workspace_size 属性 —— 逐字
    @property
    def chunked_prefill_workspace_size(self) -> int:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L591-L599（锚点双置：声明上方注释同文）
        if self._chunked_prefill_workspace_size is None:
            self._chunked_prefill_workspace_size = (
                MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                    self._vllm_config
                )
            )
        return self._chunked_prefill_workspace_size

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L601-L685
    #   forward —— 减法子集（must_keep：直调路径全程 + 算子路径）
    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L609-L615
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                q,
                kv_c_normed,
                k_pe,
                _encode_layer_name(self.layer_name),
            )

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L617-L665
        #   直调路径 —— 减法子集（pcp gather 钩子删）
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata_raw = forward_context.attn_metadata
            attn_metadata: MLACommonMetadata
            if isinstance(attn_metadata_raw, dict):
                attn_metadata = attn_metadata_raw[self.layer_name]  # type: ignore[assignment]
            elif isinstance(attn_metadata_raw, list):
                # list[dict[str, AttentionMetadata]]: used in speculative decoding
                # where [0] is the base-model (non-speculative) metadata dict.
                attn_metadata = attn_metadata_raw[0][self.layer_name]  # type: ignore[assignment]
            else:
                attn_metadata = attn_metadata_raw
            self_kv_cache = self.kv_cache
            slot_mapping = forward_context.slot_mapping

            assert isinstance(slot_mapping, dict), (
                f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
            )
            layer_slot_mapping = slot_mapping.get(self.layer_name)
            # SUBTRACTED: maybe_gather_mla_latent_cache_inputs 三元组解包
            #   （L636-L646）——delete[0]（PCP；直通即真实 PCP=1 形态）
            self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
                kv_c_normed,
                k_pe,
                self_kv_cache,
                layer_slot_mapping,
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            self.forward_impl(
                q,
                kv_c_normed,
                k_pe,
                self_kv_cache,
                attn_metadata,
                output=output,
                q_dcp_replicated=q_dcp_replicated,
            )
            return output
        else:
            # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L667-L685
            #   算子路径 —— 逐字（torch.compile 图用的依赖序算子面）
            encoded = _encode_layer_name(self.layer_name)
            kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed,
                k_pe,
                encoded,
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            torch.ops.vllm.unified_mla_attention_with_output(
                q,
                kv_c_normed,
                k_pe,
                output,
                encoded,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                q_dcp_replicated=q_dcp_replicated,
            )
            return output

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L687-L992
    #   forward_impl —— 减法子集（must_keep：两腿分岔点）
    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L703
        assert output is not None, "Output tensor must be provided."
        # SUBTRACTED: _detect_output_quant_key 探测与 quant_output 缓冲交换
        #   （L705-L721）——delete[2]

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L722-L741
        #   profile-run 模拟段 —— 减法子集（quant 分支删）
        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        # SUBTRACTED: dcp_world_size 惰性初始化（L743-L744）——delete[0]
        #   （单进程恒 1；impl.__init__ 直定）

        fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L748-L761
        #   num_actual_tokens 裁剪 —— 减法子集（PCP+DCP 量化 raise 删）
        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        if q_dcp_replicated is not None:
            q_dcp_replicated = q_dcp_replicated[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L763-L764
        #   fp8 cache 视图 —— 逐字（fp8_ds_mla 打包布局除外——其 cache 是 uint8）
        if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L766-L772
        #   分流决策 —— 逐字（must_keep：token 维切一刀）
        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens

        # SUBTRACTED: 稀疏 MLA 的 use_mha/use_masked_mha 覆写段
        #   （L774-L795）——delete[3]（is_sparse 恒 False 短路；
        #   _use_masked_mha/_DSV32_MASKED_MHA_THRESHOLDS 归 ch26）
        # SUBTRACTED: mha_use_quant_output 融合量化输出门（L797-L810）
        #   ——delete[2]（ch27）

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L812-L829
        #   MHA 展开腿调用 —— 减法子集（quant 输出选择删；output_scale=None）
        if num_mha_tokens > 0:
            mha_output = output
            mha_output_scale = None

            self.impl.forward_mha(  # type: ignore[attr-defined]
                q[num_mqa_tokens:],
                k_c_normed[num_mqa_tokens:],
                k_pe[num_mqa_tokens:],
                kv_cache,
                attn_metadata,
                self._k_scale,
                output=mha_output[num_mqa_tokens:num_actual_toks],
                output_scale=mha_output_scale,
            )

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L831-L949
        #   MQA 吸收腿全程 —— 减法子集（must_keep：bmm 吸收 + kernel + 上投影）
        if num_mqa_tokens > 0:
            # SUBTRACTED: q_dcp_replicated 的 qrep_decode 选择（L832-L837）
            #   ——delete[0]（DCP q 复制恒不激活，else 支即唯一路径）
            mqa_q = q[:num_mqa_tokens]
            mqa_output_slice = output[:num_mqa_tokens]

            # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L840-L845
            #   split nope/pe + (B,N,P)→(N,B,P) —— 逐字
            mqa_q_nope, mqa_q_pe = mqa_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            # Convert from (B, N, P) to (N, B, P)
            mqa_q_nope = mqa_q_nope.transpose(0, 1)

            # SUBTRACTED: q_pad_num_heads 头数填充（L847-L852）——章界外
            #   （后端头数对齐的 kernel 适配面；host 参考路径 q_pad=None）
            # SUBTRACTED: aiter fp4/fp8 bmm 两支（L854-L873）——delete[1]

            # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L874-L891
            #   torch.bmm 吸收 —— 减法子集（W_UK_T_dcp_qrep 选择删）
            # Pads the head_dim if necessary (for the underlying kernel)
            N, B, P = mqa_q_nope.shape
            W_UK_T = self.W_UK_T
            assert W_UK_T is not None
            _, _, L = W_UK_T.shape

            mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)

            # Convert from (N, B, L) to (B, N, L)
            mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

            # SUBTRACTED: fp8 拼接量化（L893-L898）——delete[2]
            mqa_q = (mqa_ql_nope, mqa_q_pe)
            # concatenate nope + pe -> (B, N, L + P) (fp8 op above may have fused)
            # SUBTRACTED: DCP all-gather 段（L902-L914）——delete[0]

            # call decode attn
            # SUBTRACTED: sparse 断言支（L917-L918）——delete[3]
            attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]

            # SUBTRACTED: DCP LSE 归并三分支 + PCP finalize（L922-L946）
            #   ——delete[0]（单进程 lse 原样返回）

            # v_up projection
            # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L948-L949 —— 逐字
            self._v_up_proj(attn_out, out=mqa_output_slice)

        # SUBTRACTED: 输出量化打包尾段（L951-L988）——delete[2]

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L990-L992
        #   PCP padding 清零与返回 —— 减法子集（use_pcp=False 时的原样返回）
        if self.use_pcp and output_padded.shape[0] > num_actual_toks:
            output_padded[num_actual_toks:].zero_()
        return output_padded

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L994-L1111
    #   process_weights_after_loading —— 减法子集（must_keep：吸收重排现场）
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T

        # SUBTRACTED: dcp_q_replicate 校验块（L1002-L1015）——delete[0]

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1017-L1035
        #   形状断言 + 拆 W_UK/W_UV —— 逐字
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # SUBTRACTED: aiter mxfp4/fp8 量化与 1024 档预编译循环
        #   （L1037-L1091）——delete[1]（真实此处为 if/elif/else 三支，
        #   前两支删后 else 支成为唯一路径）
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1092-L1096
        #   两份 bmm 副本常驻 —— 逐字（must_keep：W_UV(N,L,V)/W_UK_T(N,P,L)）
        # Convert from (L, N, V) to (N, L, V)
        replace_parameter(self, "W_UV", W_UV.transpose(0, 1), prefer_copy=True)
        # Convert from (L, N, P) to (N, P, L)
        replace_parameter(self, "W_UK_T", W_UK.permute(1, 2, 0), prefer_copy=True)
        # SUBTRACTED: W_UK_T_dcp_qrep all-gather（L1097-L1100）——delete[0]

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1102-L1111 —— 逐字
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1113-L1135
    #   calc_kv_scales —— 逐字（KV 尺度的可选计算位）
    def calc_kv_scales(
        self, q: torch.Tensor, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> None:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1113-L1135（锚点双置：声明上方注释同文）
        """Optional scale calculation for MLA inputs.

        Mirrors Attention.calc_kv_scales. Not all MLA backends require this
        """
        # Use safe defaults if ranges are not present
        q_range = getattr(self, "q_range", torch.tensor(1.0))
        k_range = getattr(self, "k_range", torch.tensor(1.0))
        v_range = getattr(self, "v_range", torch.tensor(1.0))

        self._q_scale.copy_(torch.abs(q).max() / q_range)
        # kv_c_normed is the compressed KV representation; use it for k/v
        kv_abs_max = torch.abs(kv_c_normed).max()
        self._k_scale.copy_(kv_abs_max / k_range)
        self._v_scale.copy_(kv_abs_max / v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        self._k_scale_cpu.fill_(self._k_scale_float)
        self._v_scale_cpu.fill_(self._v_scale_float)
        self.calculate_kv_scales = False

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1137-L1138
    #   get_attn_backend —— 逐字
    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1137-L1138（锚点双置：声明上方注释同文）
        return self.attn_backend

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1140-L1152
    #   get_kv_cache_spec —— 逐字（must_keep：spec 自报）
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1140-L1152（锚点双置：声明上方注释同文）
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

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1154-L1176
    #   _v_up_proj —— 减法子集（must_keep：吸收腿输出上投影 bmm 形状账）
    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1154-L1176（锚点双置：声明上方注释同文）
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        out = out.view(-1, self.num_heads, self.v_head_dim)
        # SUBTRACTED: aiter fp4/fp8 两支（L1158-L1173）——delete[1]
        # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
        torch.bmm(x, self.W_UV, out=out.transpose(0, 1))


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1179-L1211
#   unified_mla_kv_cache_update —— 逐字（算子版写腿）
def unified_mla_kv_cache_update(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1179-L1211（锚点双置：声明上方注释同文）
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        layer_name
    )
    if layer_slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            kv_c_normed,
            k_pe,
            kv_cache,
            layer_slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1214-L1221
#   unified_mla_kv_cache_update_fake —— 逐字
def unified_mla_kv_cache_update_fake(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1214-L1221（锚点双置：声明上方注释同文）
    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1224-L1228
#   算子注册 —— 逐字
direct_register_custom_op(
    op_name="unified_mla_kv_cache_update",
    op_func=unified_mla_kv_cache_update,
    fake_impl=unified_mla_kv_cache_update_fake,
)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1233-L1268
#   unified_mla_attention_with_output —— 减法子集（maybe_transfer_kv_layer
#   装饰器按 delete[8] 删；quant 参数族按 delete[2] 删）
@eager_break_during_capture
def unified_mla_attention_with_output(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    q_dcp_replicated: torch.Tensor | None = None,
) -> None:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1233-L1268（锚点双置：声明上方注释同文）
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, layer, kv_cache, _ = get_attention_context(layer_name)
    layer.forward_impl(
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output=output,
        q_dcp_replicated=q_dcp_replicated,
    )


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1271-L1296
#   fake + 注册 —— 逐字（mutates_args=["output"]）
def unified_mla_attention_with_output_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    q_dcp_replicated: torch.Tensor | None = None,
) -> None:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1271-L1296（锚点双置：声明上方注释同文）
    return


direct_register_custom_op(
    op_name="unified_mla_attention_with_output",
    op_func=unified_mla_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_mla_attention_with_output_fake,
)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1299-L1313
#   QueryLenSupport —— 逐字（builder 的 decode 段 query 长度档位声明）
class QueryLenSupport(Enum):
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1299-L1313（锚点双置：声明上方注释同文）
    """Defines the level of query length support for an attention backend's
    decode pipeline.

    - SINGLE_ONLY: Decode pipeline only supports single-token queries
                   (query_len=1)
    - UNIFORM: Decode pipeline supports uniform multi-token queries
               (all requests must have same query_len > 1)
    - VARLEN: Decode pipeline supports variable-length queries
              (mixed query lengths in same batch)
    """

    SINGLE_ONLY = "single_only"
    UNIFORM = "uniform"
    VARLEN = "varlen"


# SUBTRACTED: dynamic_per_batched_tensor_quant（L1316-L1324）——delete[1]
#   （aiter fp8 量化的伴生工具）
# SUBTRACTED: _DecodeConcatQuantFP8（L1327-L1357）——delete[2]


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396
#   MLACommonBackend —— 逐字（must_keep：家族基类——特形 cache shape +
#   is_mla 身份证）
class MLACommonBackend(AttentionBackend):
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396（锚点双置：声明上方注释同文）
    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396（锚点双置：声明上方注释同文）
        return "TRITON_MLA"

    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396（锚点双置：声明上方注释同文）
        return MLACommonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396（锚点双置：声明上方注释同文）
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396（锚点双置：声明上方注释同文）
        if include_num_layers_dimension:
            # Default to identity permutation to signal cross-layer allocation
            # is unsupported. Each MLA backend must opt in to support cross-layer
            # allocation by overriding this method.
            return (0, 1, 2, 3)
        return (0, 1, 2)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396（锚点双置：声明上方注释同文）
        return [320, 576]

    @classmethod
    def is_mla(cls) -> bool:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1360-L1396（锚点双置：声明上方注释同文）
        return True


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1399-L1436
#   MLACommonPrefillMetadata —— 减法子集（DCP 附字段随 delete[0] 删，
#   chunk_size/prefill_tokens_with_context 保留——merge 消费）
@dataclass
class MLACommonPrefillMetadata:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1399-L1436（锚点双置：声明上方注释同文）
    """Prefill Specific Metadata"""

    @dataclass
    class ChunkedContextMetadata:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1399-L1436（锚点双置：声明上方注释同文）
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

        # SUBTRACTED: for mla DCP 的五个附字段（L1418-L1424）——delete[0]
        chunk_size: int | None = None
        prefill_tokens_with_context: int | None = None

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: ChunkedContextMetadata | None = None
    q_data_type: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    prefill_backend: MLAPrefillBackend | None = None
    query_lens_cpu: torch.Tensor | None = None
    # SUBTRACTED: use_dense_mha / topk_mask_workspace（L1435-L1436）
    #   ——delete[3]（稀疏 MLA 归 ch26）


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1439-L1443
#   MLACommonDecodeMetadata —— 逐字（dcp 附字段保留位）
@dataclass
class MLACommonDecodeMetadata:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1439-L1443（锚点双置：声明上方注释同文）
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    dcp_tot_seq_lens: torch.Tensor | None


D = TypeVar("D", bound=MLACommonDecodeMetadata)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1449-L1491
#   MLACommonMetadata —— 逐字（must_keep：三计数 metadata 载体）
@dataclass
class MLACommonMetadata(AttentionMetadata, Generic[D]):
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1449-L1491（锚点双置：声明上方注释同文）
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    causal: bool = True

    # The dimension of the attention heads
    head_dim: int | None = None

    prefill: MLACommonPrefillMetadata | None = None
    decode: D | None = None

    def __post_init__(self):
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1449-L1491（锚点双置：声明上方注释同文）
        if self.head_dim is not None and not MLACommonBackend.supports_head_size(
            self.head_dim
        ):
            raise ValueError(f"Head dimension {self.head_dim} is not supported by MLA.")


M = TypeVar("M", bound=MLACommonMetadata)
A = TypeVar("A", bound=AttentionMetadata)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1498-L1530
#   MLADims + get_mla_dims —— 逐字（must_keep：双记法兼容）
@dataclass
class MLADims:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1498-L1530（锚点双置：声明上方注释同文）
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


def get_mla_dims(model_config: ModelConfig) -> MLADims:
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1498-L1530（锚点双置：声明上方注释同文）
    hf_text_config = model_config.hf_text_config

    # Check if this is a DeepseekV4 config (uses unified head_dim + rope_head_dim)
    if hasattr(hf_text_config, "compress_ratios"):
        # DeepseekV4 style config: unified head_dim with rope_head_dim
        head_dim = hf_text_config.head_dim
        rope_head_dim = hf_text_config.qk_rope_head_dim
        return MLADims(
            q_lora_rank=hf_text_config.q_lora_rank,
            kv_lora_rank=head_dim,
            qk_nope_head_dim=head_dim - rope_head_dim,
            qk_rope_head_dim=rope_head_dim,
            v_head_dim=head_dim,
        )

    # DeepseekV2/V3 style config
    return MLADims(
        q_lora_rank=getattr(hf_text_config, "q_lora_rank", None),
        kv_lora_rank=hf_text_config.kv_lora_rank,
        qk_nope_head_dim=hf_text_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_text_config.qk_rope_head_dim,
        v_head_dim=hf_text_config.v_head_dim,
    )


# SUBTRACTED: _DSV32_MASKED_MHA_THRESHOLDS / _DSV32_SEQ_LEN_BUCKETS /
#   _use_masked_mha（L1533-L1563）——delete[3]（稀疏 MLA 归 ch26）
# SUBTRACTED: backend_supports_prefill_query_quantization（L1566-L1592）
#   ——delete[2] 族（FP8 prefill 查询量化归 ch27；q_data_type 落 model dtype）


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1595-L1773
#   build_mla_chunked_context_metadata —— 减法子集（DCP 交织规划段随
#   delete[0] 删；单进程 chunk 切片账逐字）
def build_mla_chunked_context_metadata(
    *,
    context_lens_cpu: torch.Tensor,
    prefill_query_start_loc_cpu: torch.Tensor,
    num_prefills: int,
    chunked_prefill_workspace: torch.Tensor,
    chunked_prefill_workspace_size: int,
    block_size: int,
    align_chunk_to_block: bool,
    device: torch.device,
    dcp_world_size: int,
    dcp_local_block_size: int,
    dcp_virtual_block_size: int,
) -> "MLACommonPrefillMetadata.ChunkedContextMetadata | None":
    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1595-L1773（锚点双置：声明上方注释同文）
    """Build chunked-context metadata for an MLA prefill.

    Shared by dense and sparse builders. Splits each prefill's context
    into workspace-sized chunks and, under DCP, plans the per-rank interleaved
    local chunks the all-gather reduction consumes.

    Args:
        context_lens_cpu: Per-prefill context length (seq_len - query_len).
        prefill_query_start_loc_cpu: Prefill query cumulative offsets (0-based).
        num_prefills: Number of prefill requests.
        chunked_prefill_workspace: Scratch buffer the context gather writes to.
        chunked_prefill_workspace_size: Row capacity of the workspace.
        block_size: KV cache page size for chunk-start alignment.
        align_chunk_to_block: Round the chunk size down to ``block_size``.
        device: Target device for the returned tensors.
        dcp_world_size: Decode-context-parallel world size (1 if disabled).
        dcp_local_block_size: Per-rank interleave block size for DCP.
        dcp_virtual_block_size: ``dcp_local_block_size * dcp_world_size``.

    Returns:
        The chunked-context metadata, or None when no prefill has any context.
    """
    # NOTE: it is recommended you read the `Chunked Prefill` section in the
    # comment at the top of the file before trying to understand this code.
    max_context_len = context_lens_cpu.max().item()
    if max_context_len <= 0:
        return None
    num_prefills_with_context = int((context_lens_cpu > 0).sum().item())

    # Currently we allocate an equal amount of workspace for each prefill with
    # context; we could probably use a more advanced algorithm here and allocate
    # more workspace to prefills with longer context lengths.
    max_context_chunk = chunked_prefill_workspace_size // num_prefills_with_context
    if align_chunk_to_block:
        # The `gather_and_maybe_dequant_cache` kernel cannot handle chunk
        # starts that are not aligned to block_size, so round down.
        max_context_chunk = round_down(max_context_chunk, block_size)
    assert max_context_chunk > 0

    num_chunks = cdiv(max_context_len, max_context_chunk)
    # e.g. max_context_chunk=256, num_chunks=3, num_prefills=4 ->
    #   [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
    # Note(simon): this is done on CPU because of downstream's use of `to_list`.
    chunk_starts = torch.empty(
        num_chunks, num_prefills, dtype=torch.int32, pin_memory=False
    ).copy_(
        torch.arange(num_chunks, dtype=torch.int32)
        .multiply_(max_context_chunk)
        .unsqueeze(1)
    )
    chunk_ends = torch.min(
        context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
    )
    chunk_seq_lens = chunk_ends - chunk_starts
    chunk_seq_lens.clamp_(min=0)
    has_empty_context = torch.any(chunk_seq_lens == 0, dim=1).tolist()

    cu_seq_lens_cpu = torch.zeros(
        num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=False
    )
    torch.cumsum(chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32)
    chunk_total_token = cu_seq_lens_cpu[:, -1]

    max_tokens_over_chunk = chunk_total_token.max().item()
    token_to_seq_cpu = torch.zeros(
        (num_chunks, max_tokens_over_chunk), dtype=torch.int32, pin_memory=False
    )
    req_indices = torch.arange(num_prefills, dtype=torch.int32)
    for i in range(num_chunks):
        token_to_seq = torch.repeat_interleave(req_indices, chunk_seq_lens[i])
        token_to_seq_cpu[i, : token_to_seq.shape[0]] = token_to_seq

    prefill_tokens_with_context = prefill_query_start_loc_cpu[
        num_prefills_with_context
    ].item()

    metadata_cls = MLACommonPrefillMetadata.ChunkedContextMetadata
    # SUBTRACTED: dcp_world_size > 1 的交织规划支（L1686-L1756）——delete[0]
    chunked_context_metadata = metadata_cls(
        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
        starts=chunk_starts.to(device, non_blocking=True),
        seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
        seq_lens=chunk_seq_lens,
        context_lens=context_lens_cpu.to(device, non_blocking=True),
        token_to_seq=token_to_seq_cpu.to(device, non_blocking=True),
        chunk_total_token=chunk_total_token,
        workspace=chunked_prefill_workspace,
        has_empty_context=has_empty_context,
        prefill_tokens_with_context=prefill_tokens_with_context,
    )

    assert max(chunked_context_metadata.max_seq_lens) <= chunked_prefill_workspace_size
    return chunked_context_metadata


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1776-L1970
#   MLACommonMetadataBuilder.__init__ 族 —— 减法子集
class MLACommonMetadataBuilder(AttentionMetadataBuilder[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    kv_cache_spec: AttentionSpec

    # Defines the level of query length support for this backend.
    # - SINGLE_ONLY: Only single-token queries (no spec decode support)
    # - UNIFORM: Supports uniform multi-token queries (spec decode with uniform lengths)
    # - VARLEN: Supports variable-length queries (spec decode with mixed lengths)
    # If set to UNIFORM or VARLEN, this will increase `reorder_batch_threshold` when
    # speculative decoding is enabled.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.SINGLE_ONLY

    # Whether this builder can flatten a non-causal query block into decode rows.
    supports_non_causal_multi_token_decode: ClassVar[bool] = False

    # The threshold for reordering the batch into decode and prefill requests.
    # If > 1, the batch will be reordered such that requests with
    # query length <= threshold are classified as decode requests.
    # Use `query_len_support` (above) to set this automatically
    # when speculative decoding is enabled.
    reorder_batch_threshold: int = 1

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1802-L1831
    #   determine_chunked_prefill_workspace_size —— 逐字（must_keep：
    #   workspace 定容公式，144MB vs 3GB 注释）
    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1802-L1831（锚点双置：声明上方注释同文）
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        chunked_prefill_workspace_size = min(
            # Try for 8 full length request or at least 4 pages per-request
            max(
                8 * model_config.max_model_len,
                4 * scheduler_config.max_num_seqs * cache_config.block_size,
            ),
            # For long-context models try not to over-allocate limiting
            # kv-cache space, limiting it to 64k tokens,
            # which would result in the workspace being:
            #   2*(576)*(64*1024) = 144mb
            # (assuming 576 MLA head dim, and fp16)
            # which would result in up-projected context being
            #   2*(192*128)*(64*1024) = 3gb
            # (assuming 192 QK head dim, 128 heads, and fp16)
            64 * 1024,
        )

        # Enforce that we enough for at least 1 page per request
        chunked_prefill_workspace_size = max(
            chunked_prefill_workspace_size,
            scheduler_config.max_num_seqs * cache_config.block_size,
        )

        return chunked_prefill_workspace_size

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1833-L1874
    #   determine_prefill_query_data_type —— 减法子集（FP8 prefill 查询
    #   量化支按 delete[2] 删——恒落 model dtype）
    @staticmethod
    def determine_prefill_query_data_type(
        vllm_config: VllmConfig,
        model_dtype: torch.dtype,
    ) -> torch.dtype:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1833-L1874（锚点双置：声明上方注释同文）
        """
        Determine the query data type for prefill queries.
        Return FP8 dtype if cache is FP8 and prefill query quantization
        is enabled, else model dtype.
        """
        return model_dtype

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1876-L1970
    #   __init__ —— 减法子集（DCP 扩容支删；workspace 张量逐字）
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[M] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.metadata_cls = (
            metadata_cls if metadata_cls is not None else MLACommonMetadata
        )
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.device = device
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self.non_causal_multi_token_decode = getattr(
            kv_cache_spec, "non_causal_multi_token_decode", False
        )

        # A draft cache group can have a different head count from the target.
        self.num_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.aot_schedule = current_platform.is_cuda()

        self.kv_cache_spec = kv_cache_spec
        self.q_data_type = self.determine_prefill_query_data_type(
            vllm_config, self.model_config.dtype
        )

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1911-L1915
        #   DCP 未初始化的同型退化 —— 逐字（"DCP might not be initialized in
        #   testing"）
        self.dcp_world_size = 1
        self.dcp_local_block_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

        self.page_size = self.kv_cache_spec.block_size

        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )

        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1926-L1952
        #   workspace 张量 —— 减法子集（DCP 扩容支删）
        use_packed_fp8_cache = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        self.chunked_prefill_workspace = torch.empty(
            (
                self.chunked_prefill_workspace_size,
                self.model_config.get_head_size(),
            ),
            dtype=torch.bfloat16 if use_packed_fp8_cache else self.q_data_type,
            device=device,
        )

        # Metadata builders are created per ubatch when DBO is enabled. MLA
        # prefill backends keep the prepared metadata on the backend object, so
        # each builder needs its own backend instance to avoid cross-ubatch races.
        self._prefill_backend = self.compilation_config.static_forward_context[
            layer_names[0]
        ].prefill_backend.clone()

        supports_spec_decode = self.query_len_support != QueryLenSupport.SINGLE_ONLY
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold, supports_spec_decode, supports_dcp_with_varlen
        )

        if self.query_len_support == QueryLenSupport.SINGLE_ONLY:
            assert self.reorder_batch_threshold == 1, (
                f"reorder_batch_threshold must be 1 when query_len_support is "
                f"SINGLE_ONLY, got {self.reorder_batch_threshold}"
            )

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1972-L1986
    #   _build_decode —— 逐字
    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> MLACommonDecodeMetadata:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L1972-L1986（锚点双置：声明上方注释同文）
        return MLACommonDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )

    # SUBTRACTED: build_for_cudagraph_capture（L1988-L2003）——delete[4]
    #   （full-CG 捕获面；host 无捕获）

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2005-L2159
    #   build —— 减法子集（must_keep：split_decodes_and_prefills 分流 +
    #   chunked context 装配 + decode 元数据）
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2005-L2159（锚点双置：声明上方注释同文）
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        # SUBTRACTED: non_causal_decode 分支（L2028-L2054）——delete[8]
        #   （non-causal multi-token decode 仅特定 draft 组标记）
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
                treat_short_extends_as_decodes=not self.use_pcp,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes  # prefill_start

            # Upper bound is exact for prefill rows (no D2H sync).
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_query_lens_cpu = (
                query_start_loc_cpu[reqs_start + 1 : num_reqs + 1]
                - query_start_loc_cpu[reqs_start:num_reqs]
            )
            context_lens_cpu = (
                seq_lens_cpu[reqs_start:num_reqs] - prefill_query_lens_cpu
            )
            prefill_query_start_loc = (
                query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            )
            prefill_query_start_loc_cpu = (
                query_start_loc_cpu[reqs_start:] - query_start_loc_cpu[reqs_start]
            )

            chunked_context_metadata = build_mla_chunked_context_metadata(
                context_lens_cpu=context_lens_cpu,
                prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
                num_prefills=num_prefills,
                chunked_prefill_workspace=self.chunked_prefill_workspace,
                chunked_prefill_workspace_size=self.chunked_prefill_workspace_size,
                block_size=self.page_size,
                align_chunk_to_block=True,
                device=device,
                dcp_world_size=self.dcp_world_size,
                dcp_local_block_size=self.dcp_local_block_size,
                dcp_virtual_block_size=self.dcp_virtual_block_size,
            )

            prefill_metadata = MLACommonPrefillMetadata(
                block_table=block_table_tensor[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
                output_dtype=self.model_config.dtype,
                q_data_type=self.q_data_type,
                prefill_backend=self._prefill_backend,
            )

            self._prefill_backend.prepare_metadata(prefill_metadata)

        decode_metadata = None
        if num_decodes > 0:
            # SUBTRACTED: DCP 分布段的 seq_lens 换表与 max_seq_len 分摊
            #   （L2118-L2130）——delete[0]
            decode_metadata = self._build_decode(
                block_table_tensor=block_table_tensor[:num_decodes, ...],
                seq_lens_device=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
                query_start_loc_cpu=query_start_loc_cpu[: num_decodes + 1],
                query_start_loc_device=query_start_loc[: num_decodes + 1],
                num_decode_tokens=num_decode_tokens,
                dcp_tot_seq_lens_device=None,
            )

        attn_metadata = self.metadata_cls(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            causal=True,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata  # type: ignore[return-value]


# SUBTRACTED: reorg_kvcache（L2162-L2232）——delete[0]（DCP 本地 gather 后
#   的 TP 布局重排）


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2235 MLACommon
#   BaseImpl —— 减法子集（dense-MHA prefill 共享基）
class MLACommonBaseImpl(MLAAttentionImpl[A], Generic[A]):
    """
    Shared MLA base providing dense-MHA prefill (via the selected
    MLAPrefillBackend) for both dense and sparse impls; subclasses add decode
    (``forward_mqa``).
    """

    _use_flashinfer_concat_mla_k: bool

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2244-L2268
    #   __init__ —— 逐字
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
        kv_b_proj: ColumnParallelLinear,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2244-L2268（锚点双置：声明上方注释同文）
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
    #   _concat_k_nope_k_pe —— 逐字（flashinfer 融合位保形：host 恒 False
    #   走直拷贝 fallback——真实「无 flashinfer」同型路径）
    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2270-L2299（锚点双置：声明上方注释同文）
        """
        Efficiently concatenate k_nope and k_pe tensors along the last dimension.

        This function avoids the performance penalty of torch.cat with expanded
        non-contiguous tensors by pre-allocating the output and using direct copies.

        Args:
            k_nope: Tensor of shape [..., nope_dim]
            k_pe: Tensor to broadcast and concatenate, typically shape [..., 1, pe_dim]
                or [..., pe_dim]

        Returns:
            Tensor of shape [..., nope_dim + pe_dim]
        """
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )

        if self._use_flashinfer_concat_mla_k:
            torch.ops.vllm.flashinfer_concat_mla_k(k, k_nope, k_pe)
        else:
            # Fallback: Direct copies with efficient broadcasting
            k[..., : k_nope.shape[-1]] = k_nope
            k[..., k_nope.shape[-1] :] = k_pe
        return k

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2301-L2422
    #   _compute_prefill_context —— 逐字（must_keep：chunked 分块循环现场）
    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        assert prefill_metadata.chunked_context is not None

        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

        output = None
        merge_output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            # SOURCE: 三路 gather 分派（L2325-L2355）—— 减法子集
            #   （fp8_ds_mla 与 fp8 prefill 两支保留调用面；host 下 bf16/
            #   auto 走 gather_and_maybe_dequant_cache——_custom_ops seam）
            if self.kv_cache_dtype == "fp8_ds_mla":
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace[:toks],
                    block_table=prefill_metadata.block_table,
                    workspace_starts=prefill_metadata.chunked_context.cu_seq_lens[i],
                    batch_size=attn_metadata.num_prefills,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )
            elif not use_fp8_prefill:
                ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                    token_to_seq=prefill_metadata.chunked_context.token_to_seq[i],
                    num_tokens=prefill_metadata.chunked_context.chunk_total_token[i],
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=k_scale,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )
            else:
                # FP8 path: gather cache without dequantization
                ops.cp_gather_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                    batch_size=attn_metadata.num_prefills,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )

            # Extract kv_c_normed from workspace
            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            # When FP8 weights are used without FP8 prefill, kv_b_proj expects
            # model dtype input and will quantize internally.
            # For quantized layers (AWQ/GPTQ) that lack a .weight attribute,
            # use params_dtype which is the expected input dtype.
            _kv_b_proj_w_dtype = (
                self.kv_b_proj.weight.dtype
                if hasattr(self.kv_b_proj, "weight")
                else self.kv_b_proj.params_dtype
            )
            # For NVFP4, weights are packed uint8 — keep input in model dtype
            # since the NVFP4 linear layer quantizes internally.
            if (
                use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype()
            ) and _kv_b_proj_w_dtype != torch.uint8:
                kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)

            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            # To Do: Use epilogue of kv_b_proj to generate fp8 kv_nope.
            if use_fp8_prefill:
                kv_nope = kv_nope.to(prefill_metadata.q_data_type)
                k_pe = k_pe.to(prefill_metadata.q_data_type)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                prefill_metadata.prefill_backend.run_prefill_context_chunk(
                    chunk_idx=i,
                    q=q,
                    k=k,
                    v=v,
                )
            )
            if prefill_metadata.chunked_context.has_empty_context[i]:
                mask_empty_context(
                    attn_softmax_lse,
                    attn_output,
                    prefill_metadata.query_start_loc,
                    prefill_metadata.chunked_context.cu_seq_lens[i],
                )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                if merge_output is None:
                    merge_output = torch.empty_like(output)
                    merge_output_lse = torch.empty_like(output_lse)
                merge_attn_states(
                    output=merge_output,
                    output_lse=merge_output_lse,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output, merge_output = merge_output, output
                output_lse, merge_output_lse = merge_output_lse, output_lse

        return output, output_lse

    # SUBTRACTED: _context_parallel_compute_prefill_context（L2424-L2579）
    #   ——delete[0]（DCP 全收集 + reorg_kvcache 的分块版）

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2581-L2666
    #   forward_mha —— 减法子集（must_keep：MHA 展开腿——上投影 + 新 token
    #   段 + LSE merge；DCP 支删）
    def forward_mha(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        assert attn_metadata.prefill is not None
        assert self.dcp_world_size != -1

        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

        # Convert q to FP8 if FP8 prefill attention is enabled
        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        has_context = prefill_metadata.chunked_context is not None
        assert output_scale is None or not has_context, (
            "Fused FP8 output is only wired for the non-chunked-context path"
        )

        # SOURCE: 上投影 + per-head K/V 拼装（L2608-L2616）—— 逐字
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        if use_fp8_prefill:
            k = k.to(prefill_metadata.q_data_type)
            v = v.to(prefill_metadata.q_data_type)

        # SOURCE: 新 token 段交 prefill 后端（L2618-L2629）—— 减法子集
        #   （quant 的 out= 直写支删——delete[2]）
        output_prefill = prefill_metadata.prefill_backend.run_prefill_new_tokens(
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
            out=None,
            output_scale=output_scale,
        )

        # SOURCE: 历史上下文分块 + merge（L2631-L2660）—— 减法子集（DCP 支删）
        if has_context:
            assert prefill_metadata.chunked_context is not None
            suffix_output, suffix_lse = output_prefill
            context_output, context_lse = self._compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale
            )

            context_output = context_output[..., : self.v_head_dim]
            suffix_output = suffix_output[..., : self.v_head_dim]

            output = output.view(-1, self.num_heads, self.v_head_dim)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
                prefill_tokens_with_context=prefill_metadata.chunked_context.prefill_tokens_with_context,
            )
        elif output_scale is None:
            # With output_scale set, backend already wrote into `output` in place.
            assert isinstance(output_prefill, torch.Tensor)
            output_prefill = output_prefill[..., : self.v_head_dim]
            output_prefill = output_prefill.flatten(start_dim=-2)
            output.copy_(output_prefill)


# SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2669-L2753
#   MLACommonImpl —— 减法子集（decode 后端共享基 + forward_mqa 抽象）
class MLACommonImpl(MLACommonBaseImpl[M], Generic[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # SUBTRACTED: fused_output_quant_supported（L2675-L2681）——delete[2]

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2683-L2743
    #   __init__ —— 减法子集
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        # DSV3.2 MLA Specific Arguments
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2683-L2743（锚点双置：声明上方注释同文）
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported for MLA")

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
        )
        self.q_lora_rank = q_lora_rank
        self.indexer = indexer
        self.q_pad_num_heads = q_pad_num_heads
        # SUBTRACTED: supports_quant_query_input = True（L2726）——delete[2]
        #   （其唯一消费分支 L893-L898 已删）
        # SUBTRACTED: is_aiter_triton_fp8_bmm_enabled（L2727）——delete[1]

        # Use flashinfer's optimized concat_mla_k kernel when available.
        # The kernel is optimized for DeepSeek V3 dimensions:
        # num_heads=128, nope_dim=128, rope_dim=64
        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )

        # SUBTRACTED: self.dcp_world_size: int = -1（L2739）——delete[0]：
        #   真实由 forward_impl 惰性取 get_dcp_group().world_size；单进程恒 1
        self.dcp_world_size: int = 1

        self.cp_kv_cache_interleave_size: int = (
            get_current_vllm_config().parallel_config.cp_kv_cache_interleave_size
        )

    # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2745-L2753
    #   forward_mqa 抽象 —— 逐字（must_keep：MQA 吸收腿入口——后端子类
    #   各自实现 decode kernel）
    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/model_executor/layers/attention/mla_attention.py:L2745-L2753（锚点双置：声明上方注释同文）
        raise NotImplementedError
