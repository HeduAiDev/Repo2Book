# SOURCE: vllm/model_executor/models/llama.py
# ch23 主角文件（m1/m2/m4/m6 全章脊柱）：Llama 五类拼装——LlamaMLP /
# LlamaAttention（TP 切头数学持有者）/ LlamaDecoderLayer（积木本体）/
# LlamaModel（骨架：embed+make_layers+norm+PP 分段）/ LlamaForCausalLM
# （契约入口：forward 只出 hidden_states、compute_logits 独立、load_weights
# 两级分发）。552 行可整读的最简参考模型。
# SUBTRACTED：dossier.subtraction_plan.delete[0]（EAGLE3 aux hidden states 调用
#   线——bases 标记类保留声明）、delete[1]（layer_types/sliding_window 分支与
#   per_layer_sliding_window 传参）、delete[2]（attention_bias/qkv_bias 兼容
#   变体）、delete[3]（ENCODER_ONLY 分流、attn_cls 三元、EncoderOnlyAttention
#   import、LlamaBidirectional* 适配器两类）。
from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
)
# SUBTRACTED: from vllm.model_executor.layers.attention import
#   EncoderOnlyAttention（llama.py:L40）——delete[3]
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

# SUBTRACTED: from .adapters import as_embedding_model, as_seq_cls_model
#   （llama.py:L58）——delete[3]，适配器面只作叙事（m15）
from .interfaces import (
    EagleModelMixin,
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


# SOURCE: vllm/model_executor/models/llama.py:L79 LlamaMLP
class LlamaMLP(nn.Module):
    # SOURCE: vllm/model_executor/models/llama.py:L80-L113 __init__
    #   MergedColumnParallelLinear(gate_up) + RowParallelLinear(down) + silu 断言
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:L80-L113 __init__
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    # SOURCE: vllm/model_executor/models/llama.py:L115-L119 forward
    #   列并行升维→门控→行并行降维归约的三行
    def forward(self, x):
        # SOURCE: vllm/model_executor/models/llama.py:L115-L119 forward
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


# SOURCE: vllm/model_executor/models/llama.py:L122 LlamaAttention
class LlamaAttention(nn.Module):
    # SOURCE: vllm/model_executor/models/llama.py:L123-L231 __init__+forward
    #   —— 减法子集（delete[1]：layer_types/sliding_window 分支 L182-L201 删除）
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        # SOURCE: vllm/model_executor/models/llama.py:L140-L153 TP 切头数学
        #   （逐字——total%tp 断言、KV 头 partition/replicate 二分支、
        #   num_kv_heads=max(1,·)）
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # SOURCE: vllm/model_executor/models/llama.py:L155-L160 尺寸推导
        #   （逐字——head_dim 回退 hidden//heads、q/kv_size、scaling）
        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # SOURCE: vllm/model_executor/models/llama.py:L162-L170 QKVParallelLinear
        #   构造（逐字——三段融合）
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # SOURCE: vllm/model_executor/models/llama.py:L172-L178 RowParallelLinear
        #   o_proj（逐字）
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # SOURCE: vllm/model_executor/models/llama.py:L180 rotary 工厂调用位
        self._init_rotary_emb(config, quant_config=quant_config)

        # SUBTRACTED: layer_types/sliding_window 分支与 Eagle3 draft 层号修正
        #   （llama.py:L182-L201）——delete[1]：无 layer_types 配置的 Llama
        #   直通 sliding_window=None；滑窗语义归 ch21/ch25

        # SUBTRACTED: attn_cls 三元（EncoderOnlyAttention 分流，llama.py:
        #   L203-L207）——delete[3]：生成式主线 attn_type 恒 DECODER，固定
        #   Attention 插座
        attn_cls = Attention

        # SOURCE: vllm/model_executor/models/llama.py:L209-L219 Attention 插座
        #   构造（减法子集：delete[1] 的 per_layer_sliding_window 传参删除）
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    # SOURCE: vllm/model_executor/models/llama.py:L221-L231 forward
    #   五行：qkv_proj→split→rotary_emb→attn→o_proj；签名无 metadata/kv_cache
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/llama.py:L221-L231 forward
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    # SOURCE: vllm/model_executor/models/llama.py:L233-L245 _init_rotary_emb
    #   （逐字——get_rope 黑盒工厂调用）
    def _init_rotary_emb(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:L233-L245 _init_rotary_emb
        is_neox_style = True

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


# SOURCE: vllm/model_executor/models/llama.py:L248 LlamaDecoderLayer
class LlamaDecoderLayer(nn.Module):
    # SOURCE: vllm/model_executor/models/llama.py:L249-L308 __init__
    #   子集（delete[2]：attention_bias/qkv_bias getattr 变体与 bias_o_proj
    #   传递删除——标准 Llama 三者恒 False；delete[3]：is_causal ENCODER_ONLY
    #   分流删除——固定 DECODER）
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        attn_layer_type: type[nn.Module] = LlamaAttention,
    ) -> None:
        super().__init__()

        # SOURCE: vllm/model_executor/models/llama.py:L258-L260 vllm_config 解包
        #   （逐字——模型文件里唯一碰 vllm_config 的地方）
        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        # SOURCE: vllm/model_executor/models/llama.py:L262-L263 hidden_size 与
        #   max_position（逐字）
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # SUBTRACTED: Smaug attention_bias / internlm3 qkv_bias 兼容变体
        #   （llama.py:L264-L271）——delete[2]：固定 False 数值等价
        attention_bias = False
        bias_o_proj = False

        # SUBTRACTED: is_causal=False 的 ENCODER_ONLY 分流（llama.py:L273-L280）
        #   ——delete[3]：embedding 模型用，→ m15 叙事面
        attn_type = AttentionType.DECODER

        # SOURCE: vllm/model_executor/models/llama.py:L282-L296 self_attn 装配
        #   （逐字——切头参数从 config 传给积木）
        self.self_attn = attn_layer_type(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        # SOURCE: vllm/model_executor/models/llama.py:L297-L304 mlp 装配（逐字）
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        # SOURCE: vllm/model_executor/models/llama.py:L305-L308 双 RMSNorm（逐字）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    # SOURCE: vllm/model_executor/models/llama.py:L310-L327 forward
    #   residual 穿针：首层 None 特判、融合 add-norm 返回二元组、层间一等公民
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # SOURCE: vllm/model_executor/models/llama.py:L310-L327 forward
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    # SOURCE: vllm/model_executor/models/llama.py:L329-L331 get_quant_config
    #   （逐字——子类覆写位）
    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Get quantization config for this layer. Override in subclasses."""
        # SOURCE: vllm/model_executor/models/llama.py:L329-L331 get_quant_config
        return vllm_config.quant_config


# SOURCE: vllm/model_executor/models/llama.py:L334-L343 @support_torch_compile
#   装饰器（逐字——m11 挂钩位之一：piecewise 编译的模型侧声明）
@support_torch_compile(
    # TODO[#32068]: Investigate recompilation
    # mark_unbacked_dims={"input_ids": 0},
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
# SOURCE: vllm/model_executor/models/llama.py:L344 LlamaModel
class LlamaModel(nn.Module, EagleModelMixin):
    # SOURCE: vllm/model_executor/models/llama.py:L345-L354 hf_to_vllm_mapper
    #   （逐字——m6 的名字改写规则：q/k/v→qkv_proj+shard_id、gate/up→gate_up+0/1）
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    # SOURCE: vllm/model_executor/models/llama.py:L356-L395 __init__
    #   embed 首 rank 才建/其余 PPMissingLayer、make_layers 堆 N 层、norm 末
    #   rank 才建、make_empty_intermediate_tensors_factory）
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        # SOURCE: vllm/model_executor/models/llama.py:L356-L395 __init__
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    # SOURCE: vllm/model_executor/models/llama.py:L397-L398 embed_input_ids（逐字）
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # SOURCE: vllm/model_executor/models/llama.py:L400-L439 forward —— 减法子集
    #   （delete[0]：aux_hidden_states 采集线 L419 与 L426-L428 的
    #   _maybe_add_hidden_state 调用、L437-L438 的 aux 返回分支删除——EAGLE3
    #   特征采集归 ch32/33；PP 分段与逐层 islice 主干逐字）
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # SUBTRACTED: aux_hidden_states 初始化与层间采集
        #   （llama.py:L419、L426-L428）——delete[0]
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions, hidden_states, residual, **extra_layer_kwargs
            )
            # SUBTRACTED: _maybe_add_hidden_state 层间采集（llama.py:L426-L428）
            #   ——delete[0]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # SOURCE: vllm/model_executor/models/llama.py:L435 末 rank norm 收尾
        #   （逐字——融合 add-norm 收二元组）
        hidden_states, _ = self.norm(hidden_states, residual)

        # SUBTRACTED: aux_hidden_states 返回分支（llama.py:L437-L438）——delete[0]
        return hidden_states

    # SOURCE: vllm/model_executor/models/llama.py:L441-L443 load_weights
    #   子级：AutoWeightsLoader + hf_to_vllm_mapper
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/llama.py:L441-L443 load_weights
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


# SOURCE: vllm/model_executor/models/llama.py:L446-L454 LlamaForCausalLM bases
#   （逐字——接口标记族：hasattr/isinstance 探测位；Eagle 系归 ch32/33）
class LlamaForCausalLM(
    LocalArgmaxMixin,
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
    SupportsQuant,
):
    # SOURCE: vllm/model_executor/models/llama.py:L455-L464 类属性
    #   packed_modules_mapping/embedding_modules 融合对应关系声明
    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper
    # LoRA specific attributes
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # SOURCE: vllm/model_executor/models/llama.py:L466-L503 __init__
    #   _init_model 建骨架 → 末 PP rank ParallelLMHead（tie 则 tie_weights）+
    #   LogitsProcessor → make_empty_intermediate_tensors 别名）
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        # SOURCE: vllm/model_executor/models/llama.py:L466-L503 __init__
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    # SOURCE: vllm/model_executor/models/llama.py:L505-L511 _init_model（逐字）
    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    # SOURCE: vllm/model_executor/models/llama.py:L513-L514 embed_input_ids（逐字）
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    # SOURCE: vllm/model_executor/models/llama.py:L516-L526 forward
    #   契约方法一：透传给 self.model，返回 hidden_states（lm_head 不在
    #   forward 里））
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # SOURCE: vllm/model_executor/models/llama.py:L516-L526 forward
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    # SOURCE: vllm/model_executor/models/llama.py:L528-L533 compute_logits
    #   契约方法二：经 logits_processor 物化
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # SOURCE: vllm/model_executor/models/llama.py:L528-L533 compute_logits
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    # SOURCE: vllm/model_executor/models/llama.py:L535-L540 load_weights
    #   顶级的 skip_prefixes：tie 时跳 lm_head.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/llama.py:L535-L540 load_weights
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


# SUBTRACTED: LlamaBidirectionalForSequenceClassification /
#   LlamaBidirectionalModel 两适配器类（llama.py:L543-L552）——delete[3]：
#   as_embedding_model/as_seq_cls_model 的包装面，→ m15 叙事
