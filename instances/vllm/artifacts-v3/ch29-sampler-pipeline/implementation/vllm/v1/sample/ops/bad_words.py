# SOURCE: vllm/v1/sample/ops/bad_words.py
# ch29 step4：bad words 前缀匹配屏蔽（纯 python 逐行循环——『门控逻辑
# 留在 python』的活例，WC1 痛点③）。
# SUBTRACTED：delete[2] apply_bad_words_with_drafts（L39-L58）——spec
#   decode 的 draft 行位图版（归 ch32/33）；非 spec 路径无消费者、行为不变。
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


# SUBTRACTED: vllm/v1/sample/ops/bad_words.py:L39-L58 apply_bad_words_with_drafts
#   —— delete[2]（spec decode：按 num_draft_tokens 把每个请求的禁词掩到
#   多个 draft 行；RejectionSampler 专用，归 ch32/33）。
