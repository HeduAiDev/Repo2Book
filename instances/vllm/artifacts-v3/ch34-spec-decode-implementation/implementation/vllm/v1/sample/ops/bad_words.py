# SOURCE: vllm/v1/sample/ops/bad_words.py
# ch34 HOST SEAM（沿 ch30 镜像 + 本章消费面回补）：bad words 前缀匹配屏蔽
# （纯 python 逐行循环——『门控逻辑留在 python』的活例）。本章消费面 =
# apply_bad_words_with_drafts（RejectionSampler.apply_logits_processors 的
# spec 特化：草稿当前缀历史、按 num_draft_tokens 把禁词掩到多个 draft 行）
# 与 apply_bad_words（组合持有的普通 Sampler 的 bonus 位采样）——整文件
# 逐字镜像。
import torch

_SMALLEST_LOGIT = float("-inf")


# SOURCE: vllm/v1/sample/ops/bad_words.py:L9-L27 _apply_bad_words_single_batch —— 逐字
def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if actual_prefix == expected_prefix:
            # Assign to slice to avoid cpu->gpu sync.
            logits[last_token_id : last_token_id + 1] = _SMALLEST_LOGIT


# SOURCE: vllm/v1/sample/ops/bad_words.py:L30-L36 apply_bad_words —— 逐字
def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    for i, bad_words_ids in bad_words_token_ids.items():
        _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])


# SOURCE: vllm/v1/sample/ops/bad_words.py:L39-L58 apply_bad_words_with_drafts
#   —— 逐字（spec 特化：按 num_draft_tokens 行距遍历，past_tokens_ids 是
#   _combine_outputs_with_spec_tokens 造出的逐位前缀行）
def apply_bad_words_with_drafts(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
    num_draft_tokens: list[int],
) -> None:
    # SOURCE: vllm/v1/sample/ops/bad_words.py:L39-L58 apply_bad_words_with_drafts（draft 行位图版）
    start_idx = 0
    remaining = len(bad_words_token_ids)
    for i, n in enumerate(num_draft_tokens):
        if (bad_words_ids := bad_words_token_ids.get(i)) is not None:
            for draft_idx in range(start_idx, start_idx + n):
                _apply_bad_words_single_batch(
                    logits[draft_idx],
                    bad_words_ids,
                    past_tokens_ids[draft_idx],
                )
            remaining -= 1
            if not remaining:
                break
        start_idx += n

