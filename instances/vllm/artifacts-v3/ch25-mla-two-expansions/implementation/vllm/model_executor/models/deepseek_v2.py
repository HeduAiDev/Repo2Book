# SOURCE: vllm/model_executor/models/deepseek_v2.py —— ch25 主文件 3
#（站 1/2：选型三岔 + 投影装配）。切面：
#   DeepSeekV2FusedQkvAProjLinear（L912-L956 减法——min-latency GEMM 位删）
#   DeepseekV2MLAAttention（L959-L1194 减法）
#   DeepseekV2DecoderLayer（L1197-L1267 减法：选型三岔 L1226-L1238 逐字 +
#     self_attn 装配；MoE/MLP/序列并行段 → ch28；forward → ch28 capstone）
# SUBTRACTED 挂 dossier.subtraction_plan.delete：
#   delete[0]：qrep（DCP q 复制）分支（L1026-L1033）
#   delete[3]：V3.2 indexer 装配段（L1091-L1141）与 is_v32 探测
# 章节 elide：yarn mscale 面（L1068-L1089 的 deepseek_yarn 支——章界外）
from __future__ import annotations

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform  # noqa: F401  (yarn/平台位消费点已删)

# SUBTRACTED: import envs / DCPGroupColumnParallelLinear（L25-L60 域）——
#   delete[0]（qrep 分支）；import Indexer 等 sparse 面——delete[3]（ch26）


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L912 DeepSeekV2FusedQkv
#   AProjLinear —— 减法子集（min_latency_fused_qkv_a_proj GEMM 位删——
#   PDL kernel 域；标准 MergedColumnParallelLinear.forward 即其 fallback 支）
class DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: list[int],
        quant_config=None,
        prefix: str = "",
    ):
        # SOURCE: vllm/model_executor/layers/... deepseek_v2.py:L913-L927
        #   —— 逐字（disable_tp=True：q 潜/KV 潜/rope 融合下投影不分片）
        super().__init__(
            input_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            disable_tp=True,
            prefix=prefix,
        )
        # SUBTRACTED: _use_min_latency_gemm 探测（L929-L941）——PDL 融合
        #   GEMM kernel 面（shape 2112×7168 的 DSV3 特化；host 走标准
        #   forward = 真实 fallback 支）

    # SUBTRACTED: forward 的 min-latency GEMM 分派（L943-L956）——标准
    #   MergedColumnParallelLinear.forward（super().forward(input_) 的 else
    #   支）即唯一路径


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L959 DeepseekV2MLAAttention
#   —— 减法子集（must_keep：模型侧 MLA 层装配者）
class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

        For more info see MLACommonImpl in:
        vllm/v1/attention/backends/mla/utils.py
    """

    # SOURCE: vllm/model_executor/models/deepseek_v2.py:L968-L987 __init__
    #   签名 —— 逐字
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        reduce_results: bool = True,
        non_causal_multi_token_decode: bool = False,
    ) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L988-L1004 属性面
        #   —— 逐字
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

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1010-L1024 投影
        #   装配双支 —— 逐字（must_keep：fused_qkv_a_proj / kv_a_proj_with_mqa）
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
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

        # SUBTRACTED: qrep（DCP q 复制）分支（L1026-L1033）——delete[0]
        #   （q_proj_cls = DCPGroupColumnParallelLinear if qrep_enabled ...）
        q_proj_cls = ColumnParallelLinear
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1034-L1050 q 低秩
        #   链双支 —— 逐字（must_keep：q_a_layernorm / q_b_proj）
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
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1051-L1066
        #   kv_a_layernorm + kv_b_proj + o_proj —— 逐字（must_keep）
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

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1068-L1073 rope
        #   type 规范 + L1075-L1080 get_rope —— 减法子集（deepseek_yarn 的
        #   rope_parameters 改写与 mscale 支删——章界外；default 支逐字）
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        # SUBTRACTED: yarn mscale 缩放面（L1082-L1089）——deepseek_yarn 支
        #   章界外（本切面 rope_parameters 恒 default）
        # SUBTRACTED: is_v32 探测与 V3.2 indexer 装配段（L1091-L1141）
        #   ——delete[3]（Indexer 机制归 ch26；is_sparse/is_v32 恒 False 的
        #   装配形态：indexer=None、topk 不建）
        self.indexer_rope_emb = None
        self.indexer = None

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1143-L1161
        #   MLAModules 打包 —— 逐字（must_keep：装配契约）
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
            is_sparse=False,
            topk_indices_buffer=topk_indices_buffer,
        )

        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1163-L1186 Wrapper
        #   交接 —— 减法子集（skip_topk 与 MTP 注释面随 delete[3] 删）
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
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )

    # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1188-L1194 forward
    #   透传 —— 逐字
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1188-L1194（锚点双置：声明上方注释同文）
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


# SUBTRACTED: DeepseekAttention / DeepseekV2Attention 类本体（L511-L956 域）
#   ——老 Deepseek MHA 与非 MLA 分流的普通注意力（非本章主题）；三岔的
#   名字面以下方标记类承载（章界外域段的既有惯例：名字保真、body 收窄）


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L459 DeepseekAttention 名字位
class DeepseekAttention(nn.Module):
    """老 Deepseek（MHA）注意力——类本体归章界外（非 MLA 域）。"""

    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L459（锚点双置：声明上方注释同文）
        # SUBTRACTED: MHA 装配 body——非本章域（选型三岔只消费类名）
        super().__init__()


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L628 DeepseekV2Attention 名字位
class DeepseekV2Attention(nn.Module):
    """非 MLA 分流的普通注意力（use_mla=False 时）——类本体归章界外。"""

    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L628（锚点双置：声明上方注释同文）
        # SUBTRACTED: GQA 装配 body——非本章域（选型三岔只消费类名）
        super().__init__()


# SOURCE: vllm/model_executor/models/deepseek_v2.py:L1197 DeepseekV2DecoderLayer
#   —— 减法子集（must_keep 相关：选型三岔 + self_attn 装配）
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        # verify MLA attention specific fields
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1223-L1238 选型
        #   三岔 —— 逐字（must_keep：use_mha/use_mla/普通三分）
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        self.use_mha = use_mha

        if use_mha:
            attn_cls = DeepseekAttention
        elif model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        # SUBTRACTED: 序列并行 MoE 段（L1244-L1250）——ch28 域
        self.use_sequence_parallel_moe = False
        # SOURCE: vllm/model_executor/models/deepseek_v2.py:L1251-L1267
        #   self_attn 装配 —— 逐字
        self.self_attn = attn_cls(
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
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
        )

        # SUBTRACTED: MoE/MLP 装配段（L1269-L1285）——ch28 capstone 域
        # SUBTRACTED: 前后 RMSNorm 与 routed_scaling（L1286-L1290）——ch28 域

    # SUBTRACTED: forward（L1292-L1330）——残差穿针与 MoE 路由归 ch28
