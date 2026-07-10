"""ch25 companion — DeepSeek-V4 模型定义（只做减法）。

对应 vllm/model_executor/models/deepseek_v4.py。本章把 V4 当作对 Llama 基线(ch22)的一叠 delta：
  - 注意力：DeepseekV4Attention（MLA 低秩 fused_wqa_wkv/wq_b/wo_a/wo_b + q_norm/kv_norm + attn_sink）
            对标 LlamaAttention 的全量 qkv_proj/o_proj。
  - FFN：DeepseekV4MoE（gate 路由 top-k + shared_experts + 双后端 MegaMoE/FusedMoE）对标 LlamaMLP。
  - 残差：DeepseekV4DecoderLayer 用 hc_pre/hc_post 超连接 + hc_mult 多流，取代 Llama 的 add-norm。
  - 收尾：hc_head 把多流压回单流；DeepseekV4Model 末尾暂存 _mtp_hidden_buffer 供 MTP draft。
  - 量化：DeepseekV4FP8Config 惰性解析 expert_dtype(fp4/fp8)，回收 ch22 的 f16 量化 delta。

删除项见各处 # SUBTRACTED（稀疏 indexer/compressor、DeepGEMM/MegaMoE 内核内部、装载长尾、
PP/配置分发样板等——均按 dossier.subtraction_plan.delete 批准）。GPU-only 自定义算子保留调用
边界（torch.ops.vllm.*），其内核实现下放对应专章（ch24 注意力内核 / ch26 FusedMoE）。
"""
from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._runtime import (
    AutoWeightsLoader,
    ColumnParallelLinear,
    Fp8Config,
    FusedMoE,
    GateLinear,
    IntermediateTensors,
    LogitsProcessor,
    MergedColumnParallelLinear,
    Mxfp4MoEMethod,
    ParallelLMHead,
    QuantizationConfig,
    QuantizationMethods,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    SiluAndMulWithClamp,
    UnquantizedFusedMoEMethod,
    VllmConfig,
    VocabParallelEmbedding,
    WeightsMapper,
    current_platform,
    default_weight_loader,
    extract_layer_index,
    fused_topk_bias,
    get_current_vllm_config,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_layer_skipped,
    make_layers,
    maybe_prefix,
    set_weight_attrs,
    support_torch_compile,
)
from .deepseek_v4_attention import (
    DeepseekV4MultiHeadLatentAttentionWrapper,
)

# SUBTRACTED: DeepseekV4Indexer / DeepseekV4MLAModules 的导入——稀疏注意力索引器实现下放 ch24；
#             本章只在 DeepseekV4Attention 里保留「compress_ratio==4 时建 indexer」的分支注释。

_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")


# SOURCE: vllm/model_executor/models/deepseek_v4.py:73 (class DeepseekV4MLP)
class DeepseekV4MLP(nn.Module):
    """SwiGLU MLP，与 LlamaMLP 同构；作 DeepseekV4MoE 的 shared_experts 复用——MoE 里那条
    每 token 必走的 dense 路径。区别只在多了可选的 swiglu_limit clamp。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:74 (DeepseekV4MLP.__init__)
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:117 (DeepseekV4MLP.forward)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# SOURCE: vllm/model_executor/models/deepseek_v4.py:124 (class DeepseekV4FP8Config)
class DeepseekV4FP8Config(Fp8Config):
    """V4 量化配置：linear/attention 恒 FP8 block；MoE 专家按 expert_dtype 惰性解析为 fp4(MXFP4)
    或 fp8(block FP8)。回收 ch22 的 f16「量化压缩」delta。

    expert_dtype 必须惰性解析：本 config 在 VllmConfig 装配期构造，那时 current_vllm_config 尚未
    设置，急切读 hf_config 会永远看到默认 "fp4" 而误路由 Flash-Base 检查点。
    """

    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:144 (DeepseekV4FP8Config.__init__)
        super().__init__(*args, **kwargs)
        self._resolved_expert_dtype: str | None = None

    @property
    def expert_dtype(self) -> str:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:150 (DeepseekV4FP8Config.expert_dtype)
        if self._resolved_expert_dtype is None:
            try:
                hf_config = get_current_vllm_config().model_config.hf_config
            except Exception:
                # vllm_config 尚未设置：推迟决定，等后续调用落在 set_current_vllm_config 内。
                return "fp4"
            expert_dtype = getattr(hf_config, "expert_dtype", "fp4")
            if expert_dtype not in _DEEPSEEK_V4_EXPERT_DTYPES:
                raise ValueError(
                    f"Unsupported DeepSeek V4 expert_dtype={expert_dtype!r}; "
                    f"expected one of {_DEEPSEEK_V4_EXPERT_DTYPES}."
                )
            self._resolved_expert_dtype = expert_dtype
        return self._resolved_expert_dtype

    @property
    def is_scale_e8m0(self) -> bool:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:173 (DeepseekV4FP8Config.is_scale_e8m0)
        # FP4 检查点把 FP8 linear scale 存成 e8m0fnu；FP8 专家检查点(Flash-Base)存 float32。
        return self.expert_dtype == "fp4"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:179 (DeepseekV4FP8Config.get_name)
        return "deepseek_v4_fp8"

    def get_quant_method(self, layer, prefix):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:197 (DeepseekV4FP8Config.get_quant_method)
        # SUBTRACTED: override_quantization_method/is_mxfp4_quant 等配置分发样板（dossier 批准删）；
        #             保留 expert_dtype → Mxfp4MoEMethod / Fp8MoEMethod 的核心分支表达「量化 delta」。
        if isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.expert_dtype == "fp4":
                return Mxfp4MoEMethod(layer.moe_config)
            # expert_dtype == "fp8"：落到 Fp8Config 返回带 block float32 scale 的 Fp8MoEMethod。
        return super().get_quant_method(layer, prefix)


# SOURCE: vllm/model_executor/models/deepseek_v4.py:376 (make_deepseek_v4_expert_params_mapping)
def make_deepseek_v4_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    # 专家权重名映射表（mega 路径用）：把 checkpoint 的 experts.{id}.w{1,2,3} 映射到
    # 精简后的 w13_/w2_ fused 参数，带 expert_id + shard_id 供 weight_loader 多副本装入。
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", "w1"),
            ("w2", "w2"),
            ("w3", "w3"),
        ]
    ]


# SOURCE: vllm/model_executor/models/deepseek_v4.py:395 (class DeepseekV4MegaMoEExperts)
class DeepseekV4MegaMoEExperts(nn.Module):
    """MegaMoE 专家后端：把整批专家计算塞进单个 DeepGEMM 自定义算子（对称缓冲 + FP4/FP8 权重 +
    SM100）。这是「V4 的 MoE 不是 for 循环跑每个专家，而是一 kernel 跑全部专家」的真身。

    本章只保留 forward 到 torch.ops.vllm.deepseek_v4_mega_moe_experts 的算子边界，以及 w13/w2
    量化权重的形状定义；DeepGEMM scale 布局变换/symm buffer/Triton staging 下放 ch26。"""

    _symm_buffer_cache: dict = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:398 (DeepseekV4MegaMoEExperts.__init__)
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # FP4/FP8 量化专家权重：w13(gate+up fuse) 与 w2(down)，权重以 uint8 原始字节存放，
        # 配 block scale（也是 uint8，e8m0）。这是「专家 MXFP4/FP8」量化 delta 的落点。
        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts, 2 * intermediate_size, hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts, 2 * intermediate_size, hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"
        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts, hidden_size, intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts, hidden_size, intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        self._transformed_l1_weights = None
        self._transformed_l2_weights = None

    def _map_global_expert_id(self, expert_id: int) -> int:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:478 (DeepseekV4MegaMoEExperts._map_global_expert_id)
        if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
            return -1
        return expert_id - self.experts_start_idx

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                      return_success: bool = False):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:483 (DeepseekV4MegaMoEExperts.weight_loader)
        local_expert_id = self._map_global_expert_id(expert_id)
        if local_expert_id == -1:
            return False if return_success else None
        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            if "w13_" not in weight_name:
                return False if return_success else None
            shard_offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, shard_offset, self.intermediate_size)
        elif shard_id == "w2":
            if "w2_" not in weight_name:
                return False if return_success else None
        else:
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        if expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                f"vs checkpoint {tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight)
        return True if return_success else None

    def finalize_weights(self) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:537 (DeepseekV4MegaMoEExperts.finalize_weights)
        # SUBTRACTED: DeepGEMM transform_sf_into_required_layout / transform_weights_for_mega_moe
        #             的 scale 布局变换与权重交织（dossier 批准删，下放 ch26）。本章保留方法存在性
        #             与「装载后把 loader 侧参数转成 kernel 消费的视图」这一职责描述。
        if self._transformed_l1_weights is not None:
            return
        raise NotImplementedError(
            "DeepGEMM MegaMoE 权重布局变换下放 ch26；本章仅保留 finalize_weights 边界。"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:602 (DeepseekV4MegaMoEExperts.forward)
        # 单 kernel 跑全部专家：所有路由、分发、量化 GEMM 都在这一个自定义算子里。
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        torch.ops.vllm.deepseek_v4_mega_moe_experts(
            hidden_states, topk_weights, topk_ids, y,
            self.prefix, activation_clamp, fast_math,
        )
        return y


# SOURCE: vllm/model_executor/models/deepseek_v4.py:710 (class DeepseekV4MoE)
class DeepseekV4MoE(nn.Module):
    """MoE 层 delta 焦点：gate(GateLinear) 路由 top-k routed 专家 + 始终走的 shared_experts
    (每 token 必走的 dense 残留) + 双后端（use_mega_moe ? MegaMoE 单算子 : TP FusedMoE）。
    对照 LlamaMLP 的单 dense SwiGLU 即见「稀疏路由 + 共享 dense」混合。"""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:711 (DeepseekV4MoE.__init__)
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        if vllm_config.parallel_config.enable_expert_parallel:
            self.use_mega_moe = (
                vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
            )
        else:
            self.use_mega_moe = False

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        # SUBTRACTED: use_mega_moe 与 scoring_func/expert_dtype 的兼容性校验（NotImplementedError
        #             早退分支）——dossier 批准删的配置样板，不在前向数据流上。

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32

        if is_hash_moe:
            # hash MoE：用 input_ids 查 tid2eid 表直接定专家（V4 特有的一类层），不走打分。
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0, config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=self.use_mega_moe,
                prefix=f"{prefix}.shared_experts",
            )

        if self.use_mega_moe:
            self._init_mega_moe_experts(vllm_config, config, prefix)
        else:
            self._init_fused_moe_experts(config, quant_config, prefix)

    def _init_mega_moe_experts(self, vllm_config: VllmConfig, config, prefix: str) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:803 (DeepseekV4MoE._init_mega_moe_experts)
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.world_size
        self.ep_rank = self.ep_group.rank_in_group
        assert config.n_routed_experts % self.ep_size == 0
        self.n_local_experts = config.n_routed_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.experts = DeepseekV4MegaMoEExperts(
            vllm_config,
            num_experts=config.n_routed_experts,
            num_local_experts=self.n_local_experts,
            experts_start_idx=self.experts_start_idx,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

    def _init_fused_moe_experts(self, config, quant_config, prefix: str) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:829 (DeepseekV4MoE._init_fused_moe_experts)
        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0
        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:860 (DeepseekV4MoE.forward)
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        if not self.use_mega_moe:
            return self._forward_fused_moe(hidden_states, input_ids)

        # mega 路径：gate → fused_topk_bias 路由 → MegaMoE 单算子 → 再加上 shared_experts。
        # 这是「top-k 路由专家 + 共享 dense 相加」数据流最清楚的体现。
        org_shape = hidden_states.shape
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        activation_clamp = (
            float(self.swiglu_limit) if self.swiglu_limit is not None else None
        )
        final_hidden_states = self.experts(
            hidden_states, topk_weights, topk_ids, activation_clamp=activation_clamp,
        )
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states += shared_output
        return final_hidden_states.view(org_shape)

    def _forward_fused_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:901 (DeepseekV4MoE._forward_fused_moe)
        # TP 路径：FusedMoE 内部已聚合 shared_experts（与 mega 路径在外相加的位置不同）。
        # FusedMoE 内部细节交 ch26。
        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits,
                input_ids=input_ids,
            )
        return final_hidden_states.view(org_shape)

    def finalize_mega_moe_weights(self) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:922 (DeepseekV4MoE.finalize_mega_moe_weights)
        if self.use_mega_moe:
            self.experts.finalize_weights()


# SOURCE: vllm/model_executor/models/deepseek_v4.py:927 (class DeepseekV4Attention)
class DeepseekV4Attention(nn.Module):
    """MLA 权重定义层 delta：用低秩 fused_wqa_wkv（q_lora_rank 压缩 q + 压缩 kv）取代 Llama 的
    全量 qkv_proj；q 经 q_norm 后 wq_b 升回 full Q；kv 经 kv_norm；输出也走低秩 wo_a/wo_b
    (o_lora_rank/o_groups)，区别于标准 MLA 直接 o_proj。attn_sink 与解耦 RoPE 是 V4 特征。
    实际执行包进 DeepseekV4MultiHeadLatentAttentionWrapper。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:928 (DeepseekV4Attention.__init__)
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings

        # attn_sink：padded 到 FlashMLA 要求的 ≥64 头，初值 -inf（无 sink 效应）。装载只填前
        # n_local_heads 槽（按 TP head 区间切，是 load_weights 三类特例之一）。
        padded_heads = max(self.n_local_heads, 64)
        self.attn_sink = nn.Parameter(
            torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        # MLA 低秩压缩入口：fused_wqa_wkv 把 hidden 压成 [q_lora_rank, head_dim]——不是全量 QKV。
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)  # 对应标准 MLA q_a_layernorm
        self.wq_b = ColumnParallelLinear(  # q 潜变量升回 full Q
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)  # 对应标准 MLA kv_a_layernorm
        self.wo_a = ColumnParallelLinear(  # 输出低秩第一段（V4 特征：连 o_proj 都低秩）
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(  # 输出低秩第二段，回 hidden_size
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5

        # SUBTRACTED: rope_parameters 的 deepseek_yarn/llama_scaling 标志拼装、mscale 关闭等
        #             yarn 配置细节（dossier 批准删，下放 ch24）。保留 get_rope 调用边界。
        from ._runtime import get_rope
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            is_neox_style=False,
        )

        # SUBTRACTED: compress_ratio==4 时建 DeepseekV4Indexer（稀疏 SWA 索引器）的分支——
        #             稀疏注意力实现下放 ch24（dossier 批准删）。dense 路径 indexer 恒为 None。
        self.indexer = None

        # 包成 V4 专属 MLA 执行 wrapper：持有上面所有投影/归一/sink + rope + indexer。
        self.mla_attn = DeepseekV4MultiHeadLatentAttentionWrapper(
            hidden_size=self.hidden_size,
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.softmax_scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            v_head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            fused_wqa_wkv=self.fused_wqa_wkv,
            q_norm=self.q_norm,
            wq_b=self.wq_b,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            rotary_emb=self.rotary_emb,
            indexer=self.indexer,
            aux_stream_list=aux_stream_list,
            prefix=prefix,
        )

    def forward(self, positions, hidden_states, llama_4_scaling):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1086 (DeepseekV4Attention.forward)
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


# SOURCE: vllm/model_executor/models/deepseek_v4.py:1095 (class DeepseekV4DecoderLayer)
class DeepseekV4DecoderLayer(nn.Module):
    """单层 delta-over-Llama 焦点：attn(MLA) + ffn(MoE)，但残差用 hc_pre/hc_post 超连接
    (torch.ops.vllm.mhc_pre/mhc_post，hc_mult 多流) 取代 Llama 的融合 add-norm。
    持有 hc_{attn,ffn}_{fn,base,scale} 学习参数。被主模型与 MTP 复用。"""

    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1096 (DeepseekV4DecoderLayer.__init__)
        super().__init__()
        # SUBTRACTED: 真实在此 lazy import vllm.model_executor.layers.mhc 注册 mhc_pre/mhc_post
        #             tilelang 内核（GPU-only）；本章只保留 hc_pre/hc_post 对算子的调用边界。

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4Attention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        # 超连接的学习式门控参数：每条残差流的混合系数(fn)、偏置(base)、尺度(scale)，attn/ffn 各一套。
        self.hc_attn_fn = nn.Parameter(torch.empty((mix_hc, hc_dim), dtype=torch.float32), requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty((mix_hc, hc_dim), dtype=torch.float32), requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1172 (DeepseekV4DecoderLayer.hc_pre)
        # 超连接前处理：取代 Llama 的 input/post_attention_layernorm——在 hc_mult 条残差流上做
        # Sinkhorn 归一的学习式混合，产出本分支输入 + 写回时要用的 post/res 混合系数。
        post_mix, res_mix, layer_input = torch.ops.vllm.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )
        return layer_input, post_mix, res_mix

    def hc_post(self, x, residual, post, comb):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1192 (DeepseekV4DecoderLayer.hc_post)
        # 超连接后处理：把分支输出 x 按 post/comb 系数写回 hc_mult 条残差流。
        return torch.ops.vllm.mhc_post(x, residual, post, comb)

    def forward(self, x, positions, input_ids):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1201 (DeepseekV4DecoderLayer.forward)
        # 对照 LlamaDecoderLayer.forward 的 add-norm：这里 hc_pre/hc_post 包住 attn 和 ffn。
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


# SOURCE: vllm/model_executor/models/deepseek_v4.py:1255 (class DeepseekV4Model)
@support_torch_compile
class DeepseekV4Model(nn.Module):
    """主干 delta：embed → unsqueeze.repeat(hc_mult) 展开成 hc_mult 条残差流逐层穿过 →
    暂存 pre-hc_head 残差到 _mtp_hidden_buffer 供 MTP → hc_head 压回单流 → norm。
    对照 LlamaModel 的单残差流。持有 3 条 aux_stream 给 MLA 多 stream GEMM。"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1257 (DeepseekV4Model.__init__)
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        if vllm_config.parallel_config.enable_expert_parallel:
            self.use_mega_moe = (
                vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
            )
        else:
            self.use_mega_moe = False
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # 三条 aux stream：对应 MLA attn_gemm_parallel_execute 里三个非默认流的轻输入 GEMM
        # (compressor kv_score / indexer.weights_proj / indexer.compressor kv_score)；
        # fused_wqa_wkv 留在默认流。
        # SUBTRACTED: torch.cuda.Stream() 实例化（GPU-only）；精简版置 None 占位，保留语义注释。
        aux_stream_list = None

        self.device = current_platform.device_type
        # SUBTRACTED: topk_indices_buffer（Indexer 稀疏注意力共享缓冲，GPU-only）实例化下放 ch24。
        self.topk_indices_buffer = None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config, prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)

        # hc_head 的学习式门控参数（把 hc_mult 流压回单流时用）。
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)

        # Pre-hc_head 残差流缓冲，供 MTP draft。稳定地址（cudagraph pool 之外），让 forward
        # 里的 copy_ 跨捕获形状都刷新正确。这是 target↔draft 之间传 pre-hc_head 残差的桥。
        self._mtp_hidden_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.hc_dim,
            dtype=vllm_config.model_config.dtype,
            device=self.device,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1360 (DeepseekV4Model.embed_input_ids)
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1383 (DeepseekV4Model.forward)
        # SUBTRACTED: PP/intermediate_tensors/inputs_embeds 分支（V4 无 PPMissingLayer，恒不触发）。
        hidden_states = self.embed_input_ids(input_ids)
        # delta：把单残差流展开成 hc_mult 条平行流逐层穿过。
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, positions, input_ids)

        # 暂存 pre-hc_head 残差给 MTP draft（captured copy_）。
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # hc_head 把 hc_mult 条流压回单流，再过最终 norm。
        hidden_states = hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale,
            self.hc_head_base, self.rms_norm_eps, self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1434 (DeepseekV4Model.load_weights)
        # SUBTRACTED: stacked_params_mapping 全量条目仅保留代表性几条（compressor 条目随稀疏分支删）；
        #             本方法保留三类特例装载：①expert e8m0fnu→uint8 view ②expert_mapping 多副本
        #             ③attn_sink 按 TP head 切。其余 regex/长尾按 dossier 批准删。
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # attn_sink 的 TP head 区间。
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # e8m0fnu scale 必须以 uint8 原始字节 view 装入：copy_ 的数值转换
                    # (如 2^-7 → 0) 会毁掉指数字节。这是「量化 delta」的关键装载特例。
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param, loaded_weight, name_mapped,
                            shard_id=shard_id, expert_id=expert_id, return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    # attn_sink 按本 rank 的 head 区间切，只填前 n 个槽（其余保持 -inf）。
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1533 (DeepseekV4Model.get_expert_mapping)
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        return FusedMoE.make_expert_params_mapping(
            self, ckpt_gate_proj_name="w1", ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3", num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1547 (DeepseekV4Model.finalize_mega_moe_weights)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()


# SOURCE: vllm/model_executor/models/deepseek_v4.py:1552 (hc_head)
@torch.compile(backend=current_platform.simple_compile_backend)
def hc_head(hidden_states, hc_fn, hc_scale, hc_base, rms_norm_eps, hc_eps):
    # SOURCE: vllm/model_executor/models/deepseek_v4.py (hc_head)
    """把 hc_mult 条残差流经 RMSNorm + sigmoid 门控加权求和压回单流。主模型末尾与 MTP
    compute_logits 都用——「混合残差」delta 的收尾真身。这是纯 PyTorch（非自定义算子），
    可逐行读懂。"""
    x = hidden_states
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


# SOURCE: vllm/model_executor/models/deepseek_v4.py:1582 (_make_deepseek_v4_weights_mapper)
def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    # SUBTRACTED: 完整 regex 改名映射（fp4/fp8 两套 scale 名称 remap 长尾）按 dossier 批准删，
    #             下放装载专题。本章只保留它存在、按 expert_dtype 分发这一事实。
    return WeightsMapper()


# SOURCE: vllm/model_executor/models/deepseek_v4.py:1507 (class DeepseekV4ForCausalLM)
class DeepseekV4ForCausalLM(nn.Module):
    """顶层入口（对照 LlamaForCausalLM）：持有 DeepseekV4Model + lm_head + logits_processor。
    forward 转发；compute_logits 出主模型 logits；get_mtp_target_hidden_states 暴露
    _mtp_hidden_buffer 给 MTP draft；load_weights 经 AutoWeightsLoader(skip mtp.) 装载并
    finalize MegaMoE 权重。"""

    model_cls = DeepseekV4Model
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1627 (DeepseekV4ForCausalLM.__init__)
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1652 (DeepseekV4ForCausalLM.embed_input_ids)
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1655 (DeepseekV4ForCausalLM.compute_logits)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1662 (DeepseekV4ForCausalLM.forward)
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1674 (DeepseekV4ForCausalLM.get_mtp_target_hidden_states)
        # 暴露 pre-hc_head 残差缓冲 (max_num_batched_tokens, hc_mult*hidden) 给 MTP draft；
        # forward() 填充，每个 target step 后有效。ch28 投机解码据此取目标隐状态。
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1680 (DeepseekV4ForCausalLM.load_weights)
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # SOURCE: vllm/model_executor/models/deepseek_v4.py:1686 (DeepseekV4ForCausalLM.get_expert_mapping)
        return self.model.get_expert_mapping()
