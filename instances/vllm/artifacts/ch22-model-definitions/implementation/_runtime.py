"""ch22 companion — runtime stubs (TP context, norm/act/rope/embedding, Attention,
weight-loading infra).

这是只做减法精简版赖以「在 CPU 上跑起来」的最小运行时。本文件里的每个类/函数都对应
真实 vLLM 的同名实体（标 # SOURCE），并把真实实现里依赖 CUDA / 自定义算子 / 多进程分布式
的部分替换为 PyTorch-native 等价物（标 # SUBTRACTED）。替换后控制流、形状、数值语义与真实
vLLM 的 forward_native 路径一致，因此可在 host 上数值对照。

注意：本文件不属于 ch22 正文重点（正文聚焦 llama.py / linear.py），它只是让模型定义
与线性层能脱离 CUDA 跑起来的脚手架。正文不应喧宾夺主地讲它。
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------
# Tensor-parallel context
#
# 真实 vLLM 的 TP world size / rank 来自多进程分布式组（vllm/distributed）。精简版用一对
# 可设置的模块级变量模拟「当前 rank 处在 tp_size 张卡里的第 tp_rank 张」，使权重切分逻辑能
# 在单进程里被测到（测试通过 set_tp(size, rank) 切换视角）。
# ----------------------------------------------------------------------------
_TP_WORLD_SIZE = 1
_TP_RANK = 0


# SOURCE: vllm/distributed/parallel_state.py:get_tensor_model_parallel_world_size
def get_tensor_model_parallel_world_size() -> int:
    return _TP_WORLD_SIZE


# SOURCE: vllm/distributed/parallel_state.py:get_tensor_model_parallel_rank
def get_tensor_model_parallel_rank() -> int:
    return _TP_RANK


def set_tp(world_size: int, rank: int = 0) -> None:
    # SOURCE: vllm/distributed/parallel_state.py (init_distributed_environment — 精简替身)
    # SUBTRACTED: 真实 world size/rank 由 init_distributed_environment 建进程组得到；
    # 精简版用模块变量在单进程内切换 TP 视角（vllm/distributed/parallel_state.py）。
    global _TP_WORLD_SIZE, _TP_RANK
    assert 0 <= rank < world_size
    _TP_WORLD_SIZE, _TP_RANK = world_size, rank


# SOURCE: vllm/distributed/communication_op.py:tensor_model_parallel_all_reduce
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    # SUBTRACTED: 真实实现走 NCCL all_reduce 汇总各 rank 部分和；单进程精简版只有一份数据，
    # 直接返回输入即为全和（语义等价于 world_size==1 的 all_reduce）。
    return input_


# SOURCE: vllm/distributed/communication_op.py:tensor_model_parallel_all_gather
def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # SUBTRACTED: 真实走 NCCL all_gather 拼接各 rank 列切片；单进程下本 rank 即全部。
    return input_


# SOURCE: vllm/distributed/utils.py:divide
def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


# SOURCE: vllm/model_executor/utils.py:set_weight_attrs
def set_weight_attrs(weight: torch.Tensor, weight_attrs: dict | None) -> None:
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


# ----------------------------------------------------------------------------
# Norm / activation / rope / embedding —— 模型定义所需的非线性层
# 真实实现是 CustomOp（CUDA kernel + forward_native 回退）；精简版直接用 forward_native 的
# PyTorch-native 实现，数值与真实 forward_native 路径一致。
# ----------------------------------------------------------------------------


# SOURCE: vllm/model_executor/layers/layernorm.py:103 (class RMSNorm)
class RMSNorm(nn.Module):
    # SUBTRACTED: 真实 RMSNorm 是 CustomOp，forward 经平台分派到 fused CUDA kernel；
    # 精简版只保留其 forward_static 的 native 数学（含 (hidden, residual) 双参 fuse）。
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        # SOURCE: vllm/model_executor/layers/layernorm.py:112 (RMSNorm.__init__)
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        orig_dtype: torch.dtype,
        weight: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        # SOURCE: vllm/model_executor/layers/layernorm.py:188 (RMSNorm.forward_static)
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual
            residual = x.to(orig_dtype)
        if x.shape[-1] != hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}"
            )
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x.to(orig_dtype)
        x = x * weight
        if residual is None:
            return x
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # SOURCE: vllm/model_executor/layers/layernorm.py:233 (RMSNorm.forward_native)
        return self.forward_static(
            x, self.variance_epsilon, self.hidden_size, x.dtype, self.weight.data, residual
        )


# SOURCE: vllm/model_executor/layers/activation.py:118 (class SiluAndMul)
class SiluAndMul(nn.Module):
    # SUBTRACTED: 真实是 CustomOp（CUDA silu_and_mul kernel）；精简版用 forward_native。
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/activation.py:137 (SiluAndMul.forward_native)
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


# SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:118 (class RotaryEmbedding)
class RotaryEmbedding(nn.Module):
    # SUBTRACTED: 真实 RotaryEmbedding 是 CustomOp，forward 走 CUDA in-place kernel；
    # 精简版只留 forward_native 的 neox-style 旋转数学 + cos/sin cache。
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool = True,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:118 (RotaryEmbedding.__init__)
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.base = base
        self.is_neox_style = is_neox_style
        self.register_buffer("cos_sin_cache", self._compute_cos_sin_cache(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py (RotaryEmbedding._compute_inv_freq)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py (RotaryEmbedding._compute_cos_sin_cache)
        inv_freq = self._compute_inv_freq()
        t = torch.arange(self.max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return torch.cat((cos, sin), dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py:182 (forward_native)
        # neox-style: 把每个 head 的前 rotary_dim 维做旋转，剩余维原样透传。
        def _apply(x: torch.Tensor) -> torch.Tensor:
            # SOURCE: vllm/model_executor/layers/rotary_embedding/base.py (RotaryEmbedding.forward_static)
            orig_shape = x.shape
            x = x.view(*orig_shape[:-1], -1, self.head_size)
            xr = x[..., : self.rotary_dim]
            xp = x[..., self.rotary_dim :]
            cos_sin = self.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            cos = cos.repeat(1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 2).unsqueeze(-2)
            x1, x2 = xr.chunk(2, dim=-1)
            rotate = torch.cat((-x2, x1), dim=-1)
            xr = xr * cos + rotate * sin
            return torch.cat((xr, xp), dim=-1).reshape(orig_shape)

        return _apply(query), _apply(key)


# SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py:33 (get_rope)
def get_rope(
    head_size: int,
    max_position: int,
    rope_parameters: dict | None = None,
    is_neox_style: bool = True,
) -> RotaryEmbedding:
    # SUBTRACTED: 真实 get_rope 缓存 + 按 rope_parameters["rope_type"] 分派 linear/dynamic/
    # llama3/yarn 等多种 scaling 实现；精简版只建标准 RotaryEmbedding（Llama 基线）。
    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000.0)
    rotary_dim = head_size
    return RotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style)


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:VocabParallelEmbedding
class VocabParallelEmbedding(nn.Module):
    # SUBTRACTED: 真实沿 vocab 维按 tp 切分 + masked all_reduce 聚合；精简版（演示 TP=1 主线）
    # 持全量 embedding 表，weight_loader 直装。词表切分细节属 embedding 专题，非本章重点。
    def __init__(self, num_embeddings: int, embedding_dim: int, quant_config=None) -> None:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (VocabParallelEmbedding.__init__)
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        set_weight_attrs(self.weight, {"weight_loader": default_weight_loader})

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (VocabParallelEmbedding.forward)
        return F.embedding(input_, self.weight)


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:ParallelLMHead
class ParallelLMHead(VocabParallelEmbedding):
    # SUBTRACTED: 真实 ParallelLMHead 继承 VocabParallelEmbedding 并管理可选 bias / tie；
    # 精简版只保留作为 [vocab, hidden] 权重供 logits 投影。
    def __init__(self, num_embeddings: int, embedding_dim: int, quant_config=None, prefix: str = "") -> None:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (ParallelLMHead.__init__)
        super().__init__(num_embeddings, embedding_dim, quant_config)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> "ParallelLMHead":
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py (ParallelLMHead.tie_weights)
        self.weight = embed_tokens.weight
        return self


# SOURCE: vllm/model_executor/layers/logits_processor.py:LogitsProcessor
class LogitsProcessor(nn.Module):
    # SUBTRACTED: 真实 LogitsProcessor 做 gather/all_gather + soft-cap + scale；
    # 精简版只保留 hidden @ lm_head.weight.T * scale 的核心投影。
    def __init__(self, vocab_size: int, scale: float = 1.0) -> None:
        # SOURCE: vllm/model_executor/layers/logits_processor.py (LogitsProcessor.__init__)
        super().__init__()
        self.vocab_size = vocab_size
        self.scale = scale

    def forward(self, lm_head: ParallelLMHead, hidden_states: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/logits_processor.py (LogitsProcessor._get_logits)
        logits = F.linear(hidden_states, lm_head.weight)
        if self.scale != 1.0:
            logits = logits * self.scale
        return logits


# ----------------------------------------------------------------------------
# Attention 统一封装入口
# ----------------------------------------------------------------------------

# 模拟 vllm_config.compilation_config.static_forward_context：prefix -> Attention 层。
STATIC_FORWARD_CONTEXT: dict[str, "Attention"] = {}


# SOURCE: vllm/model_executor/layers/attention/attention.py:177 (class Attention)
class Attention(nn.Module):
    """统一注意力封装：吸收 backend 选择与 KV cache，模型定义只给 heads/scale。

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        cache_config=None,
        quant_config=None,
        per_layer_sliding_window=None,
        prefix: str = "",
        attn_type: str = "decoder",
        **extra_impl_args,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/attention/attention.py:189 (Attention.__init__)
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix
        self.attn_type = attn_type

        # SOURCE: vllm/model_executor/layers/attention/attention.py:368
        # 把自身按 layer_name=prefix 注册进 static_forward_context —— 运行期据此取回
        # 本层 kv_cache / attn_metadata。这是 (vllm_config, prefix) 契约里 prefix 的运行期用途。
        if prefix in STATIC_FORWARD_CONTEXT:
            raise ValueError(f"Duplicate layer name: {prefix}")
        STATIC_FORWARD_CONTEXT[prefix] = self

        # SOURCE: vllm/model_executor/layers/attention/attention.py:384
        # 占位 kv cache，真实由 bind_kv_cache 替换为分页缓存张量。
        self.kv_cache = torch.tensor([])

    def process_weights_after_loading(self, dtype: torch.dtype) -> None:
        # SOURCE: vllm/model_executor/layers/attention/attention.py (Attention.process_weights_after_loading)
        # SUBTRACTED: 真实在此初始化/校正 kv-cache 量化 scale；BF16 Llama 无 kv scale，空实现。
        return

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/attention/attention.py:409 (Attention.forward)
        num_tokens = query.shape[0]
        hidden_size = self.num_heads * self.head_size_v
        # reshape 成 [tokens, heads, head_dim]（与真实 forward 一致）
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size_v)
        # SUBTRACTED: 真实 forward 经 use_direct_call 走 unified_kv_cache_update +
        # unified_attention_with_output 两个自定义算子（经 forward_context 取 attn_metadata /
        # kv_cache，再 self.impl.forward 调具体 backend）。精简版无分页 KV cache / metadata，
        # 用 eager full causal SDPA 等价复现「decoder-only 多/分组-查询注意力」的可观察输出，
        # 让模型定义能在 host 上跑通并数值对照。
        output = self._eager_sdpa(query, key, value)
        return output.reshape(num_tokens, hidden_size)

    def _eager_sdpa(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/attention/backends (AttentionImpl.forward — eager 等价替身)
        # SUBTRACTED: 代替 backend impl.forward 的 eager 等价实现（含 GQA 的 KV 头复制）。
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
        q = query.transpose(0, 1)  # [heads, tokens, head_dim]
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        t = q.shape[1]
        mask = torch.full((t, t), float("-inf"), device=q.device).triu(1)
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [heads, tokens, head_dim]
        return out.transpose(0, 1)


# ----------------------------------------------------------------------------
# 权重装载基础设施
# ----------------------------------------------------------------------------


# SOURCE: vllm/model_executor/model_loader/weight_utils.py:1361 (default_weight_loader)
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.copy_(loaded_weight.view(param.shape))
    else:
        assert param.size() == loaded_weight.size(), (
            f"Attempted to load weight ({loaded_weight.size()}) "
            f"into parameter ({param.size()})"
        )
        param.data.copy_(loaded_weight)


# SOURCE: vllm/model_executor/models/utils.py:704 (maybe_prefix)
def maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


# SOURCE: vllm/model_executor/models/utils.py:741 (extract_layer_index)
def extract_layer_index(layer_name: str) -> int:
    # SUBTRACTED: 真实支持 num_attn_module>1 的双整数情形；精简版只留单层号主路径。
    int_vals: list[int] = []
    for subname in layer_name.split("."):
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert len(int_vals) == 1, f"layer name {layer_name} should only contain one integer"
    return int_vals[0]


# SOURCE: vllm/model_executor/models/utils.py:620 (make_layers)
def make_layers(num_hidden_layers: int, layer_fn, prefix: str):
    # SUBTRACTED: 真实按 PP 把 [start, end) 之外填 PPMissingLayer()，并经 get_offloader 包裹；
    # 精简版单 PP-stage：start=0, end=num_hidden_layers，全建实层。
    start_layer, end_layer = 0, num_hidden_layers
    modules = nn.ModuleList(
        [layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)]
    )
    return start_layer, end_layer, modules


# SOURCE: vllm/model_executor/models/utils.py:117 (class AutoWeightsLoader)
class AutoWeightsLoader:
    """递归权重分发器：遇到自带 load_weights 的子模块就委派，否则按参数名直装。"""

    def __init__(self, module: nn.Module, *, skip_prefixes: list[str] | None = None) -> None:
        # SOURCE: vllm/model_executor/models/utils.py:142 (AutoWeightsLoader.__init__)
        # SUBTRACTED: 真实还带 skip_substrs / ignore_unexpected_* 与 rotary 未用权重过滤；
        # 精简版只留 skip_prefixes（tie 时跳 lm_head.）这一主用途。
        self.module = module
        self.skip_prefixes = skip_prefixes or []

    def _can_skip(self, qualname: str) -> bool:
        # SOURCE: vllm/model_executor/models/utils.py:189 (AutoWeightsLoader._can_skip)
        return any(qualname.startswith(p) for p in self.skip_prefixes)

    def _load_module(
        self, base_prefix: str, module: nn.Module, weights: list[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        # SOURCE: vllm/model_executor/models/utils.py:261 (AutoWeightsLoader._load_module)
        # 子模块若自带 load_weights（如 LlamaModel），委派给它一次消费完。
        if module is not self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                for x in module_load_weights(weights):
                    yield self._qual(base_prefix, x)
                return
        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))
        # 按第一段前缀分组
        groups: dict[str, list[tuple[str, torch.Tensor]]] = {}
        order: list[str] = []
        for name, w in weights:
            head, _, rest = name.partition(".")
            if head not in groups:
                groups[head] = []
                order.append(head)
            groups[head].append((rest, w))
        for head in order:
            prefix = self._qual(base_prefix, head)
            if head in child_modules:
                if self._can_skip(prefix + "."):
                    continue
                yield from self._load_module(prefix, child_modules[head], groups[head])
            elif head in child_params:
                if self._can_skip(prefix):
                    continue
                param = child_params[head]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                for rest, w in groups[head]:
                    assert rest == "", f"unexpected nested weight {prefix}.{rest}"
                    weight_loader(param, w)
                    yield prefix
            else:
                raise ValueError(f"No module or parameter named {prefix!r}")

    @staticmethod
    def _qual(prefix: str, rest: str) -> str:
        # SOURCE: vllm/model_executor/models/utils.py:181 (AutoWeightsLoader._get_qualname)
        if prefix == "":
            return rest
        if rest == "":
            return prefix
        return f"{prefix}.{rest}"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/utils.py:342 (AutoWeightsLoader.load_weights)
        weights = [(n, w) for n, w in weights if not self._can_skip(n)]
        return set(self._load_module("", self.module, weights))


# ----------------------------------------------------------------------------
# 三段式权重装载编排所需的最小 VllmConfig
# ----------------------------------------------------------------------------


# SOURCE: vllm/config/model.py:ModelConfig (精简：只留 hf_config + dtype)
class _ModelConfig:
    def __init__(self, hf_config, dtype: torch.dtype = torch.float32) -> None:
        # SOURCE: vllm/config/model.py:ModelConfig.__init__ (精简替身)
        self.hf_config = hf_config
        self.dtype = dtype


# SOURCE: vllm/config/__init__.py:VllmConfig (精简持有 model_config/quant_config/cache_config)
class VllmConfig:
    # SUBTRACTED: 真实 VllmConfig 聚合 ~20 个子配置（parallel/scheduler/compilation/...）；
    # 精简版只留模型构造从中自取的三项：model_config / quant_config / cache_config。
    def __init__(self, hf_config, dtype: torch.dtype = torch.float32) -> None:
        # SOURCE: vllm/config/__init__.py:VllmConfig.__post_init__ (精简替身)
        self.model_config = _ModelConfig(hf_config, dtype)
        self.quant_config = None
        self.cache_config = None


# SOURCE: vllm/model_executor/model_loader/utils.py:40 (initialize_model)
def initialize_model(vllm_config: VllmConfig, *, prefix: str = "", model_class=None) -> nn.Module:
    """三段式第一段：校验 (vllm_config, prefix) 签名并实例化空壳模型。"""
    import inspect

    assert model_class is not None, "精简版需显式传入 model_class（真实由 registry 解析）"
    # SOURCE: vllm/model_executor/model_loader/utils.py:56
    signatures = inspect.signature(model_class.__init__)
    all_params = [p.name for p in signatures.parameters.values()]
    assert "vllm_config" in all_params and "prefix" in all_params, (
        "vLLM model class should accept `vllm_config` and `prefix` as input arguments."
    )
    # SUBTRACTED: 真实在 set_current_vllm_config 上下文里实例化（供 @support_torch_compile /
    # Attention 取 current config）；精简版无 torch.compile / 全局 config，直接构造。
    return model_class(vllm_config=vllm_config, prefix=prefix)


# SOURCE: vllm/model_executor/model_loader/utils.py:99 (process_weights_after_loading)
def process_weights_after_loading(model: nn.Module, vllm_config: VllmConfig) -> None:
    """三段式第三段：量化层 kernel 重排 + 对 Attention 层做后处理。"""
    # SUBTRACTED: 真实先对所有带 quant_method 的层调 process_weights_after_loading（kernel
    # 重排/量化）；精简版 UnquantizedLinearMethod 无需重排，跳过这步。
    for _, module in model.named_modules():
        if isinstance(module, Attention) and hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading(vllm_config.model_config.dtype)


def load_model(model_class, vllm_config: VllmConfig, weights, prefix: str = "") -> nn.Module:
    """三段式装载编排：initialize_model → load_weights → process_weights_after_loading → eval。

    对应 vllm/model_executor/model_loader/base_loader.py:43 (BaseModelLoader.load_model) 与
    vllm/model_executor/model_loader/default_loader.py:376 (DefaultModelLoader.load_weights)。
    """
    # SOURCE: vllm/model_executor/model_loader/base_loader.py:43 (BaseModelLoader.load_model)
    model = initialize_model(vllm_config, prefix=prefix, model_class=model_class)
    # SOURCE: vllm/model_executor/model_loader/default_loader.py:376 (DefaultModelLoader.load_weights)
    model.load_weights(weights)
    process_weights_after_loading(model, vllm_config)
    return model.eval()
