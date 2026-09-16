# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/renderers/inputs/preprocess.py —— HOST SEAM（最小承载）：
# 本章消费面只有 BaseServing 的 _extract_prompt_components/
# _extract_prompt_len（取 prompt 文本/token ids/长度，供 get_max_tokens
# 的 input_length 与 parser 预判的 prompt_token_ids）。PromptComponents 与
# 两个 extract 函数逐字；decode 侧的 parse_model_prompt/prompt_to_seq
# （ch6 渲染四步流水内部）删。
from typing import TYPE_CHECKING, NamedTuple

from vllm.inputs import EngineInput, PromptType
from vllm.utils import length_from_prompt_token_ids_or_embeds

if TYPE_CHECKING:
    from vllm.config import ModelConfig


# SOURCE: vllm/renderers/inputs/preprocess.py:L242-L246 —— PromptComponents 逐字
class PromptComponents(NamedTuple):
    text: str | None = None
    token_ids: list[int] | None = None
    embeds: "torch.Tensor | None" = None


# SOURCE: vllm/renderers/inputs/preprocess.py:L249-L252 —— HOST SEAM：
# extract_target_prompt 退化位（真实按 enc_dec/dec_only 分型拆壳；本章
# 精简面只消费已渲染的 TokensInput/str 单壳）
def extract_target_prompt(model_config: "ModelConfig", prompt: object):
    if isinstance(prompt, str):
        return {"prompt": prompt}
    return prompt


# SOURCE: vllm/renderers/inputs/preprocess.py:L256-L267 —— extract_prompt_components 逐字
def extract_prompt_components(
    model_config: "ModelConfig",
    prompt: PromptType | EngineInput,
) -> PromptComponents:
    target_prompt = extract_target_prompt(model_config, prompt)

    return PromptComponents(
        text=target_prompt.get("prompt"),
        token_ids=target_prompt.get("prompt_token_ids"),
        embeds=target_prompt.get("prompt_embeds"),
    )


# SOURCE: vllm/renderers/inputs/preprocess.py:L269-L277 —— extract_prompt_len 逐字
def extract_prompt_len(
    model_config: "ModelConfig",
    prompt: PromptType | EngineInput,
):
    target_prompt = extract_target_prompt(model_config, prompt)

    return length_from_prompt_token_ids_or_embeds(
        target_prompt.get("prompt_token_ids"),
        target_prompt.get("prompt_embeds"),
    )


# SUBTRACTED: vllm/renderers/inputs/preprocess.py 其余（prompt_to_seq /
# parse_model_prompt / parse_enc_dec_prompt / 多模态数据拆装）——ch6 渲染
# 四步流水内部，本章 preprocess_chat 黑盒回指。
