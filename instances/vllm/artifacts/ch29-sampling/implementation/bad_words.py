# 只做减法的忠实精简版 —— 镜像 vllm/v1/sample/ops/bad_words.py（pin f3fef123）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: 模块顶部 SPDX 版权头（vllm/v1/sample/ops/bad_words.py:L1-L2）—— 仅许可证注释，不影响行为。
import torch

# SOURCE: vllm/v1/sample/ops/bad_words.py:L6
_SMALLEST_LOGIT = float("-inf")


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    # SOURCE: vllm/v1/sample/ops/bad_words.py:L9-L26
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if actual_prefix == expected_prefix:
            logits[last_token_id] = _SMALLEST_LOGIT


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    past_tokens_ids: list[list[int]],
) -> None:
    # SOURCE: vllm/v1/sample/ops/bad_words.py:L29-L35
    for i, bad_words_ids in bad_words_token_ids.items():
        _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])


# SUBTRACTED: apply_bad_words_with_drafts —— 投机解码（spec-decode）专用变体，
# 按 num_draft_tokens 把每个请求的多行 draft logits 逐行屏蔽（原 bad_words.py:L38-57）。
# subtraction_plan.delete 批准；非 spec 路径不触发，删除不影响标准采样行为。
