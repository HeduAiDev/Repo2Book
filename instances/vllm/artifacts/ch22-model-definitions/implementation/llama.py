"""ch22 companion — Llama 模型定义（只做减法）。

对应 vllm/model_executor/models/llama.py。确立 v1 模型契约：
  - (vllm_config, prefix) 统一构造；类属性 packed_modules_mapping 声明 fuse 来源。
  - 四级嵌套 LlamaForCausalLM → LlamaModel → LlamaDecoderLayer → {LlamaAttention, LlamaMLP}。
  - load_weights 用 stacked_params_mapping 把 checkpoint 独立 q/k/v、gate/up 重命名 + 带
    shard_id 装入 fused 参数；其余走 default_weight_loader。

删除项见各处 # SUBTRACTED（PP / Eagle / LoRA注入 / GGUF / 量化 / sliding-window 等分支）。
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from ._runtime import (
    Attention,
    AutoWeightsLoader,
    LogitsProcessor,
    ParallelLMHead,
    RMSNorm,
    SiluAndMul,
    VocabParallelEmbedding,
    default_weight_loader,
    extract_layer_index,
    get_rope,
    get_tensor_model_parallel_world_size,
    make_layers,
    maybe_prefix,
)
from .linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)


# SOURCE: vllm/model_executor/models/llama.py:81 (class LlamaMLP)
class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config=None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:82 (LlamaMLP.__init__)
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # SOURCE: vllm/model_executor/models/llama.py:117 (LlamaMLP.forward)
        x, _ = self.gate_up_proj(x)  # 列并行升维到 2*intermediate
        x = self.act_fn(x)  # SiLU 门控：silu(gate) * up
        x, _ = self.down_proj(x)  # 行并行降维 + all_reduce 归约
        return x


# SOURCE: vllm/model_executor/models/llama.py:124 (class LlamaAttention)
class LlamaAttention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config=None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config=None,
        prefix: str = "",
        attn_type: str = "decoder",
    ) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:125 (LlamaAttention.__init__)
        super().__init__()
        layer_idx = extract_layer_index(prefix)  # noqa: F841 (真实用于 sliding-window 判定)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # KV 头数 >= TP：按 TP 切分（要求整除）。
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # KV 头数 < TP：跨多张卡复制 KV 头（要求 tp 是其整数倍）。
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config, quant_config=quant_config)

        # SUBTRACTED: layer_types/sliding_attention 判定（Eagle3 target_layer_count 调整、
        # per-layer sliding window）—— Llama 标准配置无 sliding window。
        # SUBTRACTED: EncoderOnlyAttention 分支（attn_type==ENCODER_ONLY，嵌入模型才用）。
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=None,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/llama.py:223 (LlamaAttention.forward)
        qkv, _ = self.qkv_proj(hidden_states)  # 列并行，产本 rank 的 q/k/v
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)  # 统一封装：KV cache + backend attention
        output, _ = self.o_proj(attn_output)  # 行并行 + all_reduce
        return output

    def _init_rotary_emb(self, config, quant_config) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:235 (LlamaAttention._init_rotary_emb)
        # SUBTRACTED: GGUF llama 的 is_neox_style=False 特例；精简版恒 neox-style。
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=True,
        )


# SOURCE: vllm/model_executor/models/llama.py:253 (class LlamaDecoderLayer)
class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix: str = "",
        config=None,
        attn_layer_type: type[nn.Module] = LlamaAttention,
    ) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:254 (LlamaDecoderLayer.__init__)
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # 支持带 attention bias 的模型（Smaug/internlm）。
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        bias_o_proj = attention_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # Llama 是 decoder-only，恒用 causal attention。
        # SUBTRACTED: is_causal=False → ENCODER_ONLY 的双向注意力分支（嵌入模型用）。
        attn_type = "decoder"

        self.self_attn = attn_layer_type(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm/model_executor/models/llama.py:316 (LlamaDecoderLayer.forward)
        # pre-norm + 显式 residual 穿针：residual 沿层链显式传递，而非隐式 x += 。
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def get_quant_config(self, vllm_config):
        # SOURCE: vllm/model_executor/models/llama.py:335 (LlamaDecoderLayer.get_quant_config)
        return vllm_config.quant_config


# SOURCE: vllm/model_executor/models/llama.py:350 (class LlamaModel)
# SUBTRACTED: @support_torch_compile 装饰器（torch.compile 是 custom-ops 章主题）；
#             EagleModelMixin 基类（投机解码 draft 特性）。
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:351 (LlamaModel.__init__)
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        # SUBTRACTED: PP 非首段填 PPMissingLayer() 的 else 分支；单 PP-stage 恒建 embedding。
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # SUBTRACTED: PP 非末段填 PPMissingLayer() 的 else 分支；单 PP-stage 恒建 norm。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # SUBTRACTED: make_empty_intermediate_tensors（PP 间传 hidden/residual 用，单卡可略）。

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/llama.py:392 (LlamaModel.embed_input_ids)
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/llama.py:395 (LlamaModel.forward)
        # SUBTRACTED: PP 非首段从 intermediate_tensors 取 hidden/residual 的分支；
        #             Eagle aux_hidden_states 收集；PP 非末段提前 return IntermediateTensors。
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/llama.py:436 (LlamaModel.load_weights)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # SUBTRACTED: rotary_emb.cos_cached/sin_cached skip（ColossalAI checkpoint）、
            #             kv-cache 量化 scale 重映射、FP8 scale/zero_point 名字修正等特例分支。
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # 命中 stacked 映射：把 .q_proj 等重写成 .qkv_proj，带 shard_id 走 fused 装载。
                name = name.replace(weight_name, param_name)
                # SUBTRACTED: GPTQ 额外 bias 跳过；PP missing parameter 跳过。
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 未命中 stacked 映射：embed/norm/lm_head 等非 fused 权重走 default_weight_loader。
                # SUBTRACTED: GPTQ 额外 bias 跳过；PP missing parameter 跳过。
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# SOURCE: vllm/model_executor/models/llama.py:501 (class LlamaForCausalLM)
# SUBTRACTED: SupportsLoRA/SupportsPP/SupportsEagle/SupportsEagle3 基类（LoRA/PP/投机解码特性）。
class LlamaForCausalLM(nn.Module):
    # 声明 fuse 来源：装载时把 checkpoint 的 q/k/v→qkv_proj、gate/up→gate_up_proj。
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes（仅作类属性保留其值）。
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(
        self,
        *,
        vllm_config,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ) -> None:
        # SOURCE: vllm/model_executor/models/llama.py:515 (LlamaForCausalLM.__init__)
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        # SUBTRACTED: PP 非末段建 PPMissingLayer() 的 else 分支；单 PP-stage 恒走 is_last_rank。
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)

    def _init_model(self, vllm_config, prefix: str = "", layer_type=LlamaDecoderLayer):
        # SOURCE: vllm/model_executor/models/llama.py:554 (LlamaForCausalLM._init_model)
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/llama.py:562 (LlamaForCausalLM.embed_input_ids)
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/llama.py:565 (LlamaForCausalLM.forward)
        model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # SOURCE: vllm/model_executor/models/llama.py:577 (LlamaForCausalLM.compute_logits)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SOURCE: vllm/model_executor/models/llama.py:584 (LlamaForCausalLM.load_weights)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
