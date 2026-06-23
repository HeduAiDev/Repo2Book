"""ch25 companion — DeepSeek-V4 MTP draft 模型（只做减法）。

对应 vllm/model_executor/models/deepseek_v4_mtp.py。MTP（多 token 预测）是 V4 对 Llama「单
lm_head」解码头的 delta：作投机解码(ch28) draft——融合 (a) 下一 token 的 embedding(enorm 归一)
与 (b) target 模型暂存的 pre-hc_head 残差(hnorm 归一)，经 e_proj/h_proj 融合后跑一个完整的
DeepseekV4DecoderLayer，输出又一个 pre-hc_head 残差以便多步串联。这是「多 token 预测」与
「混合残差(融合两路信号)」的真身。

  - DeepSeekV4MultiTokenPredictorLayer：单 MTP draft 层。
  - DeepSeekV4MultiTokenPredictor / DeepSeekV4MTP：多 MTP 层 ModuleDict，按 spec_step 选层；
    compute_logits 里补 hc_head 再过 shared_head。

复用主模型的 DeepseekV4DecoderLayer 与 hc_head（同一份骨架），印证「draft 与 target 同构」。
装载长尾（_rewrite_spec_layer_name/逐层缺失校验）按 dossier 批准删；投机解码协议在 ch28。
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from ._runtime import (
    IntermediateTensors,
    LogitsProcessor,
    ParallelLMHead,
    QuantizationConfig,
    RMSNorm,
    ReplicatedLinear,
    VllmConfig,
    VocabParallelEmbedding,
    current_platform,
    maybe_prefix,
    support_torch_compile,
)
from .deepseek_v4 import DeepseekV4DecoderLayer, hc_head


# SOURCE: vllm/model_executor/models/deepseek_mtp.py:43 (class SharedHead)
class SharedHead(nn.Module):
    """draft 的最终 norm + LM head（与 deepseek_mtp 共用）。compute_logits 里用它出 draft logits。"""

    def __init__(self, config, prefix: str, quant_config: QuantizationConfig | None = None) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_mtp.py:43 (__init__)
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_mtp.py:43 (forward)
        return self.norm(hidden_states)


# SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:61 (class DeepSeekV4MultiTokenPredictorLayer)
class DeepSeekV4MultiTokenPredictorLayer(nn.Module):
    """MTP draft 单层：enorm/hnorm + e_proj/h_proj 融合 token embedding 与 target 的 pre-hc_head
    残差，跑一个 DeepseekV4DecoderLayer，输出新 pre-hc_head 残差（可多步串联）。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        aux_stream_list: list | None = None,
    ) -> None:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:62 (DeepSeekV4MultiTokenPredictorLayer.__init__)
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 归一 token embedding
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 归一 target 残差

        # V4 把 e_/h_ proj 分开（带 fp8 linear quant），不像 V3 用 fused eh_proj。
        self.e_proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=False,
            return_bias=False, quant_config=quant_config,
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=False,
            return_bias=False, quant_config=quant_config,
        )

        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        # MTP 自己的 hc_head 参数（compute_logits 里补 hc_head 用）。
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)

        self.shared_head = SharedHead(config=config, prefix=prefix, quant_config=quant_config)
        # 复用主模型同款解码层（draft 与 target 同构）。
        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config, prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:122 (DeepSeekV4MultiTokenPredictorLayer.forward)
        assert inputs_embeds is not None
        # position 0 处不需要 MTP，掩为 0。
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)

        # target 暂存的 pre-hc_head 残差是 flat (T, hc_mult*D)；reshape 回 (T, hc_mult, D) 训练态布局。
        previous_hidden_states = previous_hidden_states.view(
            -1, self.hc_mult, self.config.hidden_size
        )
        previous_hidden_states = self.hnorm(previous_hidden_states)
        # 混合残差：h_proj(target 残差) + e_proj(下一 token embedding) 融合两路信号。
        hidden_states = self.h_proj(previous_hidden_states) + self.e_proj(
            inputs_embeds
        ).unsqueeze(-2)
        hidden_states = self.mtp_block(positions=positions, x=hidden_states, input_ids=None)
        # 返回 flat pre-hc_head 残差，供 num_speculative_tokens>1 时作下一 spec step 的输入；
        # hc_head 推迟到 compute_logits。
        return hidden_states.flatten(1)


# SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:153 (class DeepSeekV4MultiTokenPredictor)
class DeepSeekV4MultiTokenPredictor(nn.Module):
    """多 MTP 层容器：ModuleDict 按层 idx 持有若干 draft 层；forward 按 spec_step 选层；
    compute_logits 补 hc_head 再过 shared_head 出 draft logits。"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:154 (DeepSeekV4MultiTokenPredictor.__init__)
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.device = current_platform.device_type

        # SUBTRACTED: topk_indices_buffer / aux_stream_list 实例化（GPU-only，稀疏注意力/多流）
        #            下放 ch24；精简版置 None 占位。
        self.topk_indices_buffer = None
        aux_stream_list = None

        # 用 ModuleDict 以精确按 checkpoint 层 idx 映射。
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekV4MultiTokenPredictorLayer(
                    vllm_config, self.topk_indices_buffer,
                    f"{prefix}.layers.{idx}", aux_stream_list=aux_stream_list,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:195 (embed_input_ids)
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, spec_step_idx: int = 0) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:198 (DeepSeekV4MultiTokenPredictor.forward)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids, positions, previous_hidden_states, inputs_embeds, current_step_idx,
        )

    def compute_logits(self, hidden_states, spec_step_idx: int = 0) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:217 (DeepSeekV4MultiTokenPredictor.compute_logits)
        # MTP forward 返回 pre-hc_head 残差 (T, hc_mult*D)；这里补 hc_head 压回单流，再过 shared_head。
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        hidden_states = hidden_states.view(-1, mtp_layer.hc_mult, mtp_layer.config.hidden_size)
        hidden_states = hc_head(
            hidden_states, mtp_layer.hc_head_fn, mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base, mtp_layer.rms_norm_eps, mtp_layer.hc_eps,
        )
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


# SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:243 (class DeepSeekV4MTP)
@support_torch_compile
class DeepSeekV4MTP(nn.Module):
    """MTP draft 顶层（与投机解码 ch28 的接口边界）：持有 DeepSeekV4MultiTokenPredictor；
    forward 转发；compute_logits 出 draft logits。被投机解码消费。"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:245 (DeepSeekV4MTP.__init__)
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepSeekV4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:253 (DeepSeekV4MTP.embed_input_ids)
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, hidden_states,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds=None, spec_step_idx: int = 0) -> torch.Tensor:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:256 (DeepSeekV4MTP.forward)
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(self, hidden_states, spec_step_idx: int = 0) -> torch.Tensor | None:
        # SOURCE: vllm/model_executor/models/deepseek_v4_mtp.py:270 (DeepSeekV4MTP.compute_logits)
        return self.model.compute_logits(hidden_states, spec_step_idx)

    # SUBTRACTED: load_weights（_rewrite_spec_layer_name / 逐层缺失校验 / 450+ 行 remap 长尾）
    #             按 dossier 批准删，下放装载专题；ch28 投机解码协议另章。
