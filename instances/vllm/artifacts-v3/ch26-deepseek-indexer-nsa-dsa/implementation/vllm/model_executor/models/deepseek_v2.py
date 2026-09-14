# SOURCE: vllm/model_executor/models/deepseek_v2.py —— ch26 主文件（站 1-4）
# 全文减法：DeepseekV32IndexerCache（L616-L642 逐字）+ Indexer（L645-L819
# 减法——独立小头装配逐字【topk_tokens/n_head/head_dim 全来自 config.index_*
# 与主注意力头数无关；wq_b 复制不切 TP；wk_weights_proj 一枪 GEMM 出
# [head_dim, n_head]；132B k_cache；get_max_prefill_buffer_size】、forward 的
# 可读 else 分支 + 共享尾逐字【站 7：切 rope/nope 两段→k_norm→interleave
# RoPE→q FP8 量化→scale 全折 weights→indexer_op】）+ _try_load_fp8_indexer_wk
#（L822-L871 逐字）+ DeepseekV2MLAAttention 装配段（L968-L1186 减法——投影
# 装配逐字 + is_v32 探测 + skip_topk 三旋钮块 L1091-L1141 逐字【arXiv:
# 2603.12201 注释原文 + MTP 层恒建】+ MLAModules/Wrapper 构造逐字）+
# DeepseekV2Model 的 is_v32/buffer 段（L1370-L1408 减法——站 1：全模型共享
# topk_indices_buffer [max_num_batched_tokens, index_topk] int32）+
# load_weights 的 indexer 切面（L1519-L1602 减法——indexer_fused_mapping/
# indexer_present_prefixes/skip 分支/_try_load 调用逐字）。
# SUBTRACTED（章界）：DeepSeekV2FusedQkvAProjLinear 的 min-latency GEMM
#   （L874-L956——ch25 站 2 域，保名位）、DeepseekV2DecoderLayer 的选型三岔
#   与 MoE/MLP（L1197-L1290——ch25 站 1 / ch28 域；本章只承载 MLA 层）、
#   DeepseekV2Model 的 embed/norm/forward/load 主循环 MoE 段（L1391-L1517、
#   L1553-L1609+——ch28 整机域）、DeepseekAttention/DeepseekV2Attention 旧
#   MHA/GQA 层（ch24/ch25 域）。
from collections.abc import Iterable

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention  # noqa: F401
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.utils import extract_layer_index, make_layers
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L616-L642
#   DeepseekV32IndexerCache —— 逐字（IndexCache 本体：spec 自报
#   MLAAttentionSpec(num_kv_heads=1, head_size=132)——『Only has one vector
#   instead of K + V』注释原文）
class DeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig
    ):
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L617-L629（锚点双置）
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L631-L637 —— 逐字
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )  # Only has one vector instead of K + V

    # SOURCE: deepseek_v2.py:L728-L819 forward —— 减法子集（锚点双置）
    def forward(self): ...

    def get_attn_backend(self) -> AttentionBackend:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L641-L642
        return DeepseekV32IndexerBackend


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L645-L819 Indexer —— 减法
#   子集（__init__ 逐字；forward 保留可读 else 分支 + 共享尾——ROCm inplace
#   / CUDA fused 分支为平台死支，dossier embed_excerpts 的 elide 项）
class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
        is_inplace_rope: bool = False,
    ):
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L646-L726 __init__
        #   —— 逐字（独立小头的字面证据）
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L693-L701 —— 逐字
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

        self.is_inplace_rope = is_inplace_rope
        self.n_head_scale = self.n_head**-0.5
        self.use_fused_indexer_q = (
            current_platform.is_cuda()
            and self.quant_block_size == self.head_dim
            and self.head_dim == 128
            and self.rope_dim == 64
            and self.scale_fmt is not None
        )

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L728-L819 forward
        #   —— 减法子集（else 分支 + 共享尾逐字；ROCm inplace（L734-L750）
        #   与 CUDA fused（L751-L779）分支为平台死支省略——三条分支殊途同归，
        #   dossier embed_excerpts elide 项）
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        # SUBTRACTED: ROCm inplace-RoPE 分支（L734-L750）——ROCm 专属快速路
        #   （is_rocm host 恒 False 死支；数学与 else 分支同构）
        # SUBTRACTED: CUDA fused 分支（L751-L779——fused_indexer_q_rope_quant
        #   一核做 RoPE+量化+scale 折叠）——is_cuda host 恒 False 死支；
        #   数学由 else 分支承载
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L780-L804 else 分支
        #   —— 逐字
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during
        # compilation so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, rope_dim] (MQA).
        k = torch.cat([k_pe, k_nope], dim=-1)

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L806-L819 共享尾
        #   —— 逐字（we only quant q here since k quant is fused with cache
        #   insertion；q_scale·softmax_scale·n_head_scale 全折进 weights）
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head)

        weights = weights * q_scale * self.softmax_scale * self.n_head_scale

        return self.indexer_op(hidden_states, q_fp8, k, weights)


# SOURCE: vllm/model_executor/layers/quantization/utils/fp8_utils.py
#   per_token_group_quant_fp8 的 import 位（真实 import 自
#   vllm.model_executor.layers.quantization.utils.fp8_utils——HOST SEAM 镜像）
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L822-L871
#   _try_load_fp8_indexer_wk —— 逐字（FP8 wk 权重+BF16 weights_proj 的融合
#   加载——『单独训练、独立 checkpoint 形态』的工程痕迹）
def _try_load_fp8_indexer_wk(
    name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
):
    """
    We fuse the WK and weights_proj projections, but in some checkpoints WK is stored
    in FP8 with a separate weight_scale_inv, while weights_proj is stored in BF16.
    Upcasting to BF16 during loading enables the fusion. This function loads the FP8 WK
    weights and scale, and when both are available, dequantizes to BF16 and stores into
    the fused wk_weights_proj.weight parameter.
    """
    # SOURCE: vllm/model_executor/models/deepseek_v2.py:L822-L871（锚点双置）
    if "indexer.wk." not in name or "wk_weights" in name:
        return False  # Weight is not an isolated WK weight for the indexer, ignore.
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale" in name
    if not is_weight and not is_scale:
        return False  # WK is not in FP8 format, ignore.
    # Buffer this tensor (weight or scale) until both have arrived.
    layer_prefix = name.rsplit(".wk.", 1)[0]  # e.g. "model.layers.0.self_attn.indexer"
    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    if any(
        name.startswith(missing_layer_name)
        for missing_layer_name in pp_missing_layer_names
    ):
        return True
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    if "weight" not in entry or "scale" not in entry:
        return True  # still waiting for the other param

    # We have both weight and scale: dequantize FP8 to BF16.
    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    if scale_inv.ndim == 1:
        # Per-channel scale: one scale per row of [out, in]
        group_shape = GroupShape(1, weight_fp8.shape[1])
    else:
        block_size = weight_fp8.shape[1] // scale_inv.shape[1]
        group_shape = GroupShape(block_size, block_size)
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=group_shape,
        out_dtype=torch.bfloat16,
    )

    # Load the dequantized weight into shard 0 of the fused buffer.
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True


# SUBTRACTED: _min_latency_fused_qkv_a_proj_impl/_min_latency_fused_qkv_a_proj_
#   fake/DeepSeekV2FusedQkvAProjLinear 的 min-latency GEMM 位（L874-L956）——
#   ch25 站 2 域（投影几何已由 MergedColumnParallelLinear 承载）


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L959-L1186
#   DeepseekV2MLAAttention —— 减法子集（投影装配 + V3.2 indexer 段逐字）
class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

        For more info see MLACommonImpl in:
        vllm/v1/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        reduce_results: bool = True,
        non_causal_multi_token_decode: bool = False,
    ) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L968-L1066 投影
        #   装配 —— 逐字（qrep 探测单进程恒 False 死支保留）
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Use input_size for projection input dimensions if provided,
        # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA)
        proj_input_size = input_size if input_size is not None else self.hidden_size

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1026-L1033 qrep
        #   探测 —— 逐字（VLLM_DCP_Q_REPLICATE 面；单进程恒 False）
        qrep_enabled = (
            _envs.VLLM_DCP_Q_REPLICATE
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.prefill_context_parallel_size <= 1
        )
        q_proj_cls = (
            _DCPGroupColumnParallelLinear if qrep_enabled else ColumnParallelLinear
        )
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = q_proj_cls(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = q_proj_cls(
                proj_input_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1075-L1080 主
        #   RoPE —— 逐字（is_neox_style=False）
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        # SUBTRACTED: deepseek_yarn 的 mscale 段（L1082-L1089）——rope_type
        #   恒 default（yarn 数学 → ch24 primer 域；get_rope 对非 default
        #   抛 NotImplementedError 同型）

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1091-L1141
        #   IndexCache config + indexer 装配 —— 逐字
        self.is_v32 = hasattr(config, "index_topk")

        # IndexCache config
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        _skip_topk = False
        is_mtp_layer = False
        if self.is_v32:
            _index_topk_freq = getattr(config, "index_topk_freq", 1)
            _index_topk_pattern = getattr(config, "index_topk_pattern", None)
            _index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
            layer_id = extract_layer_index(prefix)

            if _index_topk_pattern is None:
                _skip_topk = (
                    max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq
                    != 0
                )
            elif 0 <= layer_id < len(_index_topk_pattern):
                _skip_topk = _index_topk_pattern[layer_id] == "S"

            # The skip pattern only governs backbone layers. MTP/nextn
            # layers (layer_id >= num_hidden_layers) always build a full
            # indexer: they compute indices at draft step 0 and toggle
            # at runtime via set_skip_topk
            # (index_share_for_mtp_iteration).
            _num_hidden_layers = getattr(config, "num_hidden_layers", None)
            is_mtp_layer = (
                _num_hidden_layers is not None and layer_id >= _num_hidden_layers
            )

        if self.is_v32 and (not _skip_topk or is_mtp_layer):
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
                is_inplace_rope=self.indexer_rope_emb.enabled(),
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1143-L1186
        #   MLAModules + Wrapper 构造 —— 逐字
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            # MTP layers must never start with skip_topk=True: their indexer
            # computes indices at draft step 0, and the runtime toggle
            # (set_skip_topk, index_share_for_mtp_iteration) only exists in
            # the V1 proposer. A frozen True would leave the draft reading a
            # never-written topk buffer.
            skip_topk=_skip_topk and not is_mtp_layer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
            # Do not skip scoring for MTP layers: their top-k buffer may be
            # reused by later draft iterations through index sharing.
            allow_short_prefill_indexer_scoring_skip=not is_mtp_layer,
        )
        self.topk_indices_buffer = topk_indices_buffer

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1188-L1194 —— 逐字
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


# SUBTRACTED: DeepseekV2DecoderLayer（L1197-L1290 选型三岔 + MoE/MLP）——
#   ch25 站 1 / ch28 域；本章以 _DecoderLayerSlice 承载 self_attn 装配位
class _DecoderLayerSlice(nn.Module):
    """DeepseekV2DecoderLayer 的 ch26 切面：只装 self_attn（MoE/MLP → ch28）。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1198-L1267 —— 减法
        #   子集（use_mla 三岔收敛到 MLA 支：DSA 章域；MoE → ch28）
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        assert vllm_config.model_config and qk_rope_head_dim, (
            "ch26 facet carries the MLA branch of the selection fork "
            "(use_mla 三岔全貌见 ch25 站 1)"
        )
        self.use_mla = True
        self.self_attn = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )
        # SUBTRACTED: is_moe_layer/MLP/MoE 装配（L1239-L1290）——ch28 域


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L1370-L1408 DeepseekV2Model
#   —— 减法子集（站 1：is_v32 探测 + 全模型共享 topk_indices_buffer 分配；
#   embed/norm/aux/PP → ch28/ch17 域）
class DeepseekV2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1370-L1408（锚点
        #   双置）
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None
        self.topk_indices_buffer = topk_indices_buffer

        # SUBTRACTED: embed_tokens/norm/make_empty_intermediate_tensors/
        #   aux_hidden_state_layers（L1391-L1416）——ch28 整机域
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: _DecoderLayerSlice(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

    # SUBTRACTED: forward（L1433-L1517）——ch28 整机域（llama_4_scaling 与
    #   residual 主循环）；本章只立装配与 buffer 共享

    # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1519-… load_weights
    #   —— 减法子集（indexer 权重加载切面逐字；MoE/experts 段 → ch28）
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1519-L1602 —— 减法
        #   子集
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        mla_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        # Fused indexer wk + weights_proj (shard 0 = wk, shard 1 = weights_proj)
        _pending_wk_fp8 = getattr(self, "_pending_indexer_wk_fp8", None)
        if _pending_wk_fp8 is None:
            self._pending_indexer_wk_fp8 = _pending_wk_fp8 = {}

        indexer_fused_mapping = [
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        stacked_params_mapping.extend(indexer_fused_mapping)

        stacked_params_mapping.extend(mla_params_mapping)

        # SUBTRACTED: expert_params_mapping（L1553-L1567）——ch28 MoE 域

        pp_missing_layer_names: list[str] = []  # HOST SEAM：单进程无 PP 缺层
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # With index_topk_freq>1 only some layers build an indexer, yet the
        # checkpoint ships indexer weights for all of them; track the built ones.
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1572-L1576 —— 逐字
        indexer_present_prefixes = {
            n.rsplit(".indexer.", 1)[0] for n in params_dict if ".indexer." in n
        }
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if ".indexer." in name and (
                name.rsplit(".indexer.", 1)[0] not in indexer_present_prefixes
            ):
                continue  # this layer has no indexer; drop its checkpoint weights
            # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1585-L1588
            #   —— 逐字

            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # SUBTRACTED: 专家参数映射与 weight_loader 全链（L1604-…）
                #   ——ch23/ch28 域；本章 indexer 切面到 _try_load 为止
                break

        return loaded_params


# envs 的 VLLM_DCP_Q_REPLICATE 消费位（真实 import vllm.envs）
import vllm.envs as _envs  # noqa: E402

# HOST SEAM：DCPGroupColumnParallelLinear 名字位（qrep_enabled 单进程恒
#   False，仅类型标注消费——真实类归 ch34 分布式域）
_DCPGroupColumnParallelLinear = ColumnParallelLinear
