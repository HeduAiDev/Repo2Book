# SOURCE: vllm/models/deepseek_v4/attention.py —— ch25 主文件 4
#（m12/m13/m14/m15：DSV4 第三代——逐层 compress_ratio / spec 自报 /
# 输出侧低秩 / indexer 占位）。切面：
#   _resolve_dsv4_kv_cache_dtype（L91-L120 逐字）
#   DeepseekV4Attention ABC（L123-L349 减法子集：dims/projections/
#     compress_ratio 解析/swa 子缓存/compressor/spec 自报）
#   DeepseekV4IndexerCache（L677-L715 逐字——m15 的 spec 生产者）
#   get_kv_cache_spec（L655-L674 逐字——must_keep）
# SUBTRACTED 挂 dossier.subtraction_plan.delete：
#   delete[3]：DeepseekV4Indexer 装配段（L276-L297）与 forward 内 indexer 面
#   delete[7]：aux_stream 三路并行（aux_stream_list/ln_events/
#     attn_gemm_parallel_execute/execute_in_parallel）与 eager_scratch_pool、
#     PREFILL_CHUNK_SIZE 预留路径
# 章界：forward/attention_impl/_fused_qnorm_rope_kv_insert 的融合 CUDA
#   kernel 链与平台子类实现（flashmla/flashinfer 走读）→ ch26（NSA/DSA）
#   与 ch28（capstone）。
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

# SUBTRACTED: _fill_short_context_topk_indices triton kernel（L68-L87）
#   ——delete[3]（indexer 的短上下文 top-k 填充——ch26）


# SOURCE: vllm/models/deepseek_v4/attention.py:L91-L120 _resolve_dsv4_kv_
#   cache_dtype —— 逐字
def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    # SOURCE: vllm/models/deepseek_v4/attention.py:L91-L120（锚点双置：声明上方注释同文）
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    ``fp8_ds_mla`` format is UE8M0 block-scaled fp8 packed as ``uint8`` (the
    canonical ``fp8_ds_mla`` string is written back onto ``cache_config`` so the
    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


# SOURCE: vllm/models/deepseek_v4/attention.py:L123 DeepseekV4Attention
#   —— 减法子集（must_keep：第三代 MLA 层——compress_ratio 分层 + spec 自报）
class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV4FlashMLAAttention`` /
    ``DeepseekV4FlashInferSM120Attention`` /
    ``DeepseekV4FlashInferMLAAttention`` (CUDA) or
    ``DeepseekV4ROCMAiterMLAAttention`` (ROCm) — selected by the platform-specific
    deepseek_v4 model module. The base is never instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8); False = plain
    # bf16 / per-tensor fp8 KV row. Backends can override the instance hook when
    # a single attention class dispatches across arch-specific layouts.
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # SUBTRACTED: PREFILL_CHUNK_SIZE（L145）——delete[7]（预留路径）

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L123（锚点双置：声明上方注释同文）
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L123（锚点双置：声明上方注释同文）
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L123（锚点双置：声明上方注释同文）
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        # SOURCE: vllm/models/deepseek_v4/attention.py:L175-L177 —— 逐字
        return self.use_fp8_ds_mla_layout

    # SOURCE: vllm/models/deepseek_v4/attention.py:L179-L348 __init__
    #   —— 减法子集（aux_stream/eager_scratch 参数与 ln_events 按 delete[7] 删）
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L187-L215 dims 面 —— 逐字
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # SOURCE: vllm/models/deepseek_v4/attention.py:L207-L213 compress_ratio
        #   逐层解析 —— 逐字（must_keep：MTP 层恒 1 的护栏）
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        # SOURCE: vllm/models/deepseek_v4/attention.py:L226-L262 投影装配
        #   —— 逐字（fused_wqa_wkv/wq_b/wo_a(bmm)/wo_b——输出侧低秩 m14）
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        # SOURCE: vllm/models/deepseek_v4/attention.py:L264-L272 —— 逐字
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer
        # SUBTRACTED: eager_scratch_pool 位（L274）——delete[7]

        # SOURCE: vllm/models/deepseek_v4/attention.py:L276-L297 indexer 装配
        #   位 —— SUBTRACTED：delete[3]（compress_ratio==4 层才建
        #   DeepseekV4Indexer、aux stream 重叠——NSA/DSA 机制归 ch26；
        #   本章只登记『谁在建 indexer、它的 cache 也是 spec 生产者』）
        self.indexer = None

        # SUBTRACTED: aux_stream_list / ln_events（L299-L304）——delete[7]
        #   （三路 GEMM 并行；None 时退化串行——注释原话）

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        # SOURCE: vllm/models/deepseek_v4/attention.py:L306-L317 —— 逐字
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        # SOURCE: vllm/models/deepseek_v4/attention.py:L319-L325 SWA 子缓存
        #   —— 逐字（一层注意力拆多个 KV 子缓存的实证）
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with compilation context for metadata lookup.
        # SOURCE: vllm/models/deepseek_v4/attention.py:L327-L333 —— 逐字
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Create the compressor for layers with compress_ratio > 1; after the
        # attention setup above so its KV-cache prefix (self.prefix) is set.
        # SOURCE: vllm/models/deepseek_v4/attention.py:L337-L348 —— 减法子集
        #   （eager_scratch_pool 参数按 delete[7] 删）
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

    # SUBTRACTED: forward（L350-L396）——attn_gemm_parallel_execute 三路
    #   GEMM 并行 + eager break + 融合 qnorm/rope/kv-insert CUDA kernel 链：
    #   delete[7]（aux stream 面）+ 章界（融合 kernel 走读归 ch26/ch28）
    # SUBTRACTED: _fused_wqa_wkv_gemm / attn_gemm_parallel_execute /
    #   attention_impl / _fused_qnorm_rope_kv_insert / _global_topk_output_
    #   buffers / _q_fp8（L398-L643）——同上

    # SOURCE: vllm/models/deepseek_v4/attention.py:L652-L653 get_attn_backend
    #   —— 逐字
    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L652-L653（锚点双置：声明上方注释同文）
        return self.backend_cls

    # SOURCE: vllm/models/deepseek_v4/attention.py:L655-L674 get_kv_cache_spec
    #   —— 逐字（must_keep：compress_ratio>1 报 MLAAttentionSpec 特账；
    #   ≤1 返回 None——SWA 段由子缓存另报）
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L655-L674（锚点双置：声明上方注释同文）
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )


# SOURCE: vllm/models/deepseek_v4/attention.py:L677 DeepseekV4IndexerCache
#   —— 逐字（must_keep 相关：m15——indexer cache 也是 spec 生产者）
class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        # SOURCE: vllm/models/deepseek_v4/attention.py:L677（锚点双置：声明上方注释同文）
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L677（锚点双置：声明上方注释同文）
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    # SOURCE: vllm/models/deepseek_v4/attention.py:L712 forward（占位空体）
    def forward(self): ...
        # SOURCE: vllm/models/deepseek_v4/attention.py:L677（锚点双置：声明上方注释同文）

    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/models/deepseek_v4/attention.py:L714-L715 —— 真身为
        #   return DeepseekV4IndexerBackend（vllm/v1/attention/backends/mla/
        #   indexer.py）；SUBTRACTED：delete[3]（sparse 家族归 ch26——本章
        #   只登记『它的 cache 也是 spec 生产者』，不引入其 backend 类）
        raise NotImplementedError(
            "DeepseekV4IndexerBackend is ch26's domain (sparse MLA family)."
        )

    # SUBTRACTED: DeepseekV4Indexer 装配类（L718-L1020+）——delete[3]
    #   （wq_b/weights_proj/top-k 打分全链归 ch26）
