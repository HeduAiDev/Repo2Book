"""ch25 companion — runtime stubs（让 DeepSeek-V4 三个模型文件「能 import、能实例化、能被
结构性巡查」的最小脚手架）。

与 ch22 的 _runtime.py 同性质：本文件里的每个符号都对应真实 vLLM 的同名实体（标 # SOURCE），
把真实实现里依赖 CUDA / DeepGEMM / tilelang / FP8 自定义算子 / 多进程分布式的部分替换为
import-time 可用的 PyTorch-native 占位（标 # SUBTRACTED）。

为什么 ch25 的 companion 不像 ch22 那样在 CPU 上「数值对照」跑起来？因为 DeepSeek-V4 的前向
本质依赖 GPU-only 的自定义算子（torch.ops.vllm.deepseek_v4_attention / deepseek_v4_fp8_einsum
/ deepseek_v4_mega_moe_experts / mhc_pre / mhc_post），以及 SM100 + DeepGEMM + tilelang。
本章的目标（见 dossier）是让读者看懂「delta-over-Llama 的骨架与算子边界」，不是在 host 复现数值。
因此精简版忠实保留控制流、模块结构、算子调用点，而把这些 GPU-only 算子替换成 import-time 占位：
结构可被 inspect / 实例化、可被测试断言，真正的数值路径仍指向真实 vLLM 的算子（标注清楚）。

正文不应喧宾夺主地讲本文件——它只是脚手架。
"""
from __future__ import annotations

import types
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Tensor / expert parallel context
# 真实 vLLM 的 TP/EP world size/rank 来自多进程分布式组（vllm/distributed）。精简版用模块级
# 变量模拟单进程视角，使权重切分/专家划分逻辑能被测到。
# ---------------------------------------------------------------------------
_TP_WORLD_SIZE = 1
_TP_RANK = 0


# SOURCE: vllm/distributed/parallel_state.py:get_tensor_model_parallel_world_size
def get_tensor_model_parallel_world_size() -> int:
    return _TP_WORLD_SIZE


# SOURCE: vllm/distributed/parallel_state.py:get_tensor_model_parallel_rank
def get_tensor_model_parallel_rank() -> int:
    return _TP_RANK


# SOURCE: vllm/distributed/parallel_state.py:get_ep_group
def get_ep_group():
    # SUBTRACTED: 真实返回多进程 expert-parallel GroupCoordinator；精简版给单卡 EP（world=1, rank=0）。
    return types.SimpleNamespace(
        world_size=1, rank_in_group=0, device_group=None
    )


# ---------------------------------------------------------------------------
# Norm / activation
# ---------------------------------------------------------------------------
# SOURCE: vllm/model_executor/layers/layernorm.py (class RMSNorm)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        # SOURCE: vllm/model_executor/layers/layernorm.py (RMSNorm.__init__)
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        # SOURCE: vllm/model_executor/layers/layernorm.py (RMSNorm.forward_native)
        # SUBTRACTED: 真实 RMSNorm 走 fused CUDA custom op（带可选 residual add）；
        #             精简版用 PyTorch-native 等价（forward_native 路径），语义一致。
        if residual is not None:
            x = x + residual
            residual = x
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        x = (x.to(dt)) * self.weight
        return x if residual is None else (x, residual)


# SOURCE: vllm/model_executor/layers/activation.py (class SiluAndMul)
class SiluAndMul(nn.Module):
    def forward(self, x):
        # SOURCE: vllm/model_executor/layers/activation.py (forward)
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


# SOURCE: vllm/model_executor/layers/activation.py (class SiluAndMulWithClamp)
class SiluAndMulWithClamp(nn.Module):
    def __init__(self, limit: float) -> None:
        # SOURCE: vllm/model_executor/layers/activation.py (__init__)
        super().__init__()
        self.limit = limit

    def forward(self, x):
        # SUBTRACTED: 真实带 swiglu_limit clamp 的 fused kernel；精简版 native clamp + SiLU·mul。
        # SOURCE: vllm/model_executor/layers/activation.py (forward)
        d = x.shape[-1] // 2
        gate = x[..., :d].clamp(max=self.limit)
        up = x[..., d:].clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Linear family（保留 fuse 语义/形状；底层用 nn.Linear 占位，权重切分逻辑省略）
# ---------------------------------------------------------------------------
def _default_weight_loader(param, loaded_weight) -> None:
    # SOURCE: vllm/model_executor/models/deepseek_v4.py (_default_weight_loader)
    param.data.copy_(loaded_weight)


default_weight_loader = _default_weight_loader


class _LinearBase(nn.Module):
    # SOURCE: vllm/model_executor/models/deepseek_v4.py (_LinearBase)
    """精简版线性层基类：保留 (in,out,bias,quant_config,prefix) 构造签名与 (out, bias) 返回约定。

    # SUBTRACTED: 真实 vLLM linear 家族含 TP 切分、量化 method、weight_loader（带 shard_id）等；
    #             精简版用单进程 nn.Linear 占位，仅暴露 weight/forward 形状契约供结构巡查与测试。
    """

    def __init__(self, in_features, out_features, bias=False, return_bias=True, **kw):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py (__init__)
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._return_bias = return_bias
        self.weight.weight_loader = _default_weight_loader

    def forward(self, x):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py (forward)
        out = F.linear(x, self.weight, self.bias)
        return (out, None) if self._return_bias else out


# SOURCE: vllm/model_executor/layers/linear.py (class ColumnParallelLinear)
class ColumnParallelLinear(_LinearBase):
    def __init__(self, input_size, output_size, bias=False, quant_config=None,
                 return_bias=True, prefix="", **kw):
        # SOURCE: vllm/model_executor/layers/linear.py (__init__)
        super().__init__(input_size, output_size, bias=bias, return_bias=return_bias)


# SOURCE: vllm/model_executor/layers/linear.py (class RowParallelLinear)
class RowParallelLinear(_LinearBase):
    def __init__(self, input_size, output_size, bias=False, quant_config=None,
                 reduce_results=True, return_bias=True, disable_tp=False, prefix="", **kw):
        # SOURCE: vllm/model_executor/layers/linear.py (__init__)
        super().__init__(input_size, output_size, bias=bias, return_bias=return_bias)


# SOURCE: vllm/model_executor/layers/linear.py (class MergedColumnParallelLinear)
class MergedColumnParallelLinear(_LinearBase):
    def __init__(self, input_size, output_sizes, bias=False, quant_config=None,
                 disable_tp=False, prefix="", **kw):
        # output_sizes 是被 fuse 的多段输出维（如 [q_lora_rank, head_dim]）。
        # SOURCE: vllm/model_executor/layers/linear.py (__init__)
        super().__init__(input_size, sum(output_sizes), bias=bias, return_bias=True)


# SOURCE: vllm/model_executor/layers/linear.py (class ReplicatedLinear)
class ReplicatedLinear(_LinearBase):
    def __init__(self, input_size, output_size, bias=False, quant_config=None,
                 return_bias=True, prefix="", **kw):
        # SOURCE: vllm/model_executor/layers/linear.py (__init__)
        super().__init__(input_size, output_size, bias=bias, return_bias=return_bias)


# ---------------------------------------------------------------------------
# Embedding / LM head / logits
# ---------------------------------------------------------------------------
# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (class VocabParallelEmbedding)
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, quant_config=None, prefix="", **kw):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (__init__)
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))

    def forward(self, input_ids):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (forward)
        return F.embedding(input_ids, self.weight)


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (class ParallelLMHead)
class ParallelLMHead(VocabParallelEmbedding):
    pass


# SOURCE: vllm/model_executor/layers/logits_processor.py (class LogitsProcessor)
class LogitsProcessor(nn.Module):
    def __init__(self, vocab_size, scale: float = 1.0) -> None:
        # SOURCE: vllm/model_executor/layers/logits_processor.py (__init__)
        super().__init__()
        self.scale = scale

    def forward(self, lm_head, hidden_states):
        # SOURCE: vllm/model_executor/layers/logits_processor.py (forward)
        return F.linear(hidden_states, lm_head.weight) * self.scale


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
# SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py (get_rope)
def get_rope(head_size, max_position, rope_parameters=None, is_neox_style=True, **kw):
    # SUBTRACTED: 真实 get_rope 按 rope_type（deepseek_yarn 等）返回带 cos_sin_cache 的内核 module；
    #             精简版给一个仅持有 cos_sin_cache 占位的 stub（V4 输出端 inverse-RoPE 要读它）。
    rope = nn.Module()
    rope.cos_sin_cache = torch.zeros(max_position, head_size)
    return rope


# ---------------------------------------------------------------------------
# Quant config 基类（DeepseekV4FP8Config 继承它）
# ---------------------------------------------------------------------------
# SOURCE: vllm/model_executor/layers/quantization/fp8.py (class Fp8Config)
class Fp8Config:
    # SUBTRACTED: 真实 Fp8Config 含 weight_block_size/ignored_layers/get_quant_method 全套调度；
    #             精简版只保留 DeepseekV4FP8Config 直接引用的字段，使惰性 expert_dtype 解析逻辑可读。
    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/layers/quantization/fp8.py (__init__)
        self.ignored_layers = []
        self.packed_modules_mapping = {}

    def get_quant_method(self, layer, prefix):
        # SOURCE: vllm/model_executor/layers/quantization/fp8.py (get_quant_method)
        return None


# SOURCE: vllm/model_executor/layers/quantization/__init__.py (QuantizationConfig / QuantizationMethods 类型)
QuantizationConfig = object
QuantizationMethods = str


# SOURCE: vllm/model_executor/layers/quantization/mxfp4.py (Mxfp4MoEMethod)
class Mxfp4MoEMethod:
    def __init__(self, moe_config=None):
        # SOURCE: vllm/model_executor/layers/quantization/mxfp4.py (__init__)
        self.moe_config = moe_config


# SOURCE: vllm/model_executor/layers/quantization/fp8.py (UnquantizedFusedMoEMethod 经 fused_moe.layer)
class UnquantizedFusedMoEMethod:
    def __init__(self, moe_config=None):
        # SOURCE: vllm/model_executor/layers/quantization/fp8.py (__init__)
        self.moe_config = moe_config


# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py (is_layer_skipped)
def is_layer_skipped(prefix, ignored_layers, fused_mapping):
    return prefix in (ignored_layers or [])


# ---------------------------------------------------------------------------
# FusedMoE / GateLinear / 路由算子（保留构造签名与算子边界）
# ---------------------------------------------------------------------------
# SOURCE: vllm/model_executor/layers/fused_moe (class GateLinear)
class GateLinear(nn.Module):
    def __init__(self, hidden_size, n_routed_experts, out_dtype=torch.float32,
                 bias=False, prefix="", **kw):
        # SOURCE: vllm/model_executor/layers/fused_moe (__init__)
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_routed_experts, hidden_size))
        self.out_dtype = out_dtype
        # 由 DeepseekV4MoE 外部赋值：噪声校正偏置与 hash 路由表。
        self.e_score_correction_bias = None
        self.tid2eid = None

    def forward(self, x):
        # SOURCE: vllm/model_executor/layers/fused_moe (forward)
        return F.linear(x.to(self.weight.dtype), self.weight).to(self.out_dtype), None


# SOURCE: vllm/model_executor/layers/fused_moe (class FusedMoE)
class FusedMoE(nn.Module):
    """TP-后端专家容器（精简版只保留 DeepseekV4MoE 引用到的构造与调用边界）。"""

    def __init__(self, *, shared_experts=None, gate=None, num_experts=0, top_k=0,
                 hidden_size=0, intermediate_size=0, renormalize=True, quant_config=None,
                 prefix="", scoring_func="softmax", routed_scaling_factor=1.0,
                 e_score_correction_bias=None, hash_indices_table=None, swiglu_limit=None,
                 router_logits_dtype=torch.float32, **kw):
        # SOURCE: vllm/model_executor/layers/fused_moe (__init__)
        super().__init__()
        self.shared_experts = shared_experts
        # 真实 FusedMoE 在内部已聚合 shared_experts；is_internal_router 决定 gate 是否内置。
        self.is_internal_router = False
        self.moe_config = None

    # SUBTRACTED: forward 内部的 dispatch/combine/量化 GEMM 是 ch26（FusedMoE/EP）主题，
    #             本章只保留 DeepseekV4MoE._forward_fused_moe 调用它的边界，不展开。
    def forward(self, hidden_states, router_logits=None, input_ids=None, **kw):
        # SOURCE: vllm/model_executor/layers/fused_moe (forward)
        raise NotImplementedError(
            "FusedMoE forward 内部细节下放 ch26；本章仅保留 DeepseekV4MoE 调用边界。"
        )

    @staticmethod
    def make_expert_params_mapping(model, ckpt_gate_proj_name, ckpt_down_proj_name,
                                   ckpt_up_proj_name, num_experts):
        # SUBTRACTED: 完整专家权重名映射在 fused_moe；精简版返回空表占位（装载长尾已按计划删）。
        # SOURCE: vllm/model_executor/layers/fused_moe (make_expert_params_mapping)
        return []


# SOURCE: vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (fused_topk_bias)
def fused_topk_bias(hidden_states, gating_output, scoring_func, e_score_correction_bias,
                    topk, renormalize, indices_type, input_tokens, hash_indices_table,
                    routed_scaling_factor):
    # SUBTRACTED: 真实 fused_topk_bias 是 sqrtsoftplus 打分 + e_score_correction_bias +
    #             top-k + renormalize 的 Triton 算子；精简版给 native 占位以暴露路由算子边界，
    #             保留输入/输出契约（topk_weights, topk_ids）。
    if scoring_func == "sqrtsoftplus":
        scores = torch.sqrt(F.softplus(gating_output))
    else:
        scores = torch.softmax(gating_output, dim=-1)
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(indices_type)


# ---------------------------------------------------------------------------
# 模型构建 / 权重装载 / 杂项 utils
# ---------------------------------------------------------------------------
# SOURCE: vllm/model_executor/models/utils.py (make_layers)
def make_layers(num_hidden_layers, layer_fn, prefix=""):
    layers = nn.ModuleList(
        layer_fn(prefix=f"{prefix}.{i}") for i in range(num_hidden_layers)
    )
    return 0, num_hidden_layers, layers


# SOURCE: vllm/model_executor/models/utils.py (extract_layer_index)
def extract_layer_index(prefix: str) -> int:
    nums = [int(p) for p in prefix.split(".") if p.isdigit()]
    return nums[-1] if nums else 0


# SOURCE: vllm/model_executor/models/utils.py (maybe_prefix)
def maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


# SOURCE: vllm/model_executor/models/utils.py (class AutoWeightsLoader)
class AutoWeightsLoader:
    # SUBTRACTED: 真实 AutoWeightsLoader 递归 dispatch 到子模块 load_weights、按 mapper 改名、
    #             skip_substrs/skip_prefixes 过滤；精简版只保留 DeepseekV4ForCausalLM 调用边界。
    def __init__(self, module, skip_prefixes=None, skip_substrs=None):
        # SOURCE: vllm/model_executor/models/utils.py (__init__)
        self.module = module

    def load_weights(self, weights, mapper=None):
        # SOURCE: vllm/model_executor/models/utils.py (load_weights)
        return set()


# SOURCE: vllm/model_executor/models/utils.py (class WeightsMapper)
class WeightsMapper:
    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/models/utils.py (__init__)
        pass


# SOURCE: vllm/model_executor/utils.py (set_weight_attrs)
def set_weight_attrs(weight, attrs):
    for k, v in (attrs or {}).items():
        setattr(weight, k, v)


# SOURCE: vllm/sequence.py (IntermediateTensors)
class IntermediateTensors:  # PP 间传递的容器；V4 单 PP-stage 不走，仅类型占位。
    pass


# SOURCE: vllm/platforms (current_platform)
class _CurrentPlatform:
    device_type = "cpu"
    simple_compile_backend = "eager"

    @staticmethod
    def get_device_capability():
        # SOURCE: vllm/platforms (get_device_capability)
        return types.SimpleNamespace(major=10, minor=0)


current_platform = _CurrentPlatform()


# SOURCE: vllm/compilation/decorators.py (support_torch_compile)
def support_torch_compile(cls=None, **kw):
    # SUBTRACTED: torch.compile 包裹是 custom-ops 章主题；精简版做 no-op 透传，保留装饰位置可见。
    if cls is None:
        return lambda c: c
    return cls


# SOURCE: vllm/config (VllmConfig 类型占位)
VllmConfig = object


def get_current_vllm_config():
    # SUBTRACTED: 真实从线程局部 set_current_vllm_config 取；精简版返回 None 触发 expert_dtype 默认分支。
    # SOURCE: vllm/model_executor/models/deepseek_v4.py (get_current_vllm_config)
    raise RuntimeError("no current vllm_config (companion stub)")
