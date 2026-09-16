# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/utils/__init__.py —— HOST SEAM（最小承载）：真实是数千行
# 工具大杂烩；本章消费面只有 random_uuid（request_id/default id 生成，
# api_router→BaseServing._base_request_id→chat_utils 全链消费）与
# length_from_prompt_token_ids_or_embeds（RequestState 构造消费）。
import uuid

MASK_64_BITS = (1 << 64) - 1


# SOURCE: vllm/utils/__init__.py:L11-L12 —— random_uuid 逐字
def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


# SOURCE: vllm/utils/__init__.py:L15-L31 —— length_from_prompt_token_ids_or_embeds
# 逐字（torch.Tensor 注解退化为 object——host 精简环境 embeds 路径不触达，
# 纯 token ids 语义不变）
def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | object | None,
    prompt_embeds: object | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


# SUBTRACTED: vllm/utils/__init__.py 其余（ThreadPool/SharedMemory/
# FlexibleDictCorrection…）——归各自域。
