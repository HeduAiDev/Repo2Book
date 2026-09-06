# ch29 主电池三：惩罚三件套真算式 + bad_words 前缀匹配 + 白名单掩码。
# 基准（v0.27.1 现核行号）：
#   - vllm/model_executor/layers/utils.py:L34-L89（scatter_add_ 计数 + OpenAI 两式）
#   - vllm/v1/sample/ops/penalties.py:L10-L56（张量化 + -1 占位替换）
#   - vllm/v1/sample/ops/bad_words.py:L9-L36（前缀匹配、slice 赋值）
#   - vllm/v1/sample/sampler.py:L419-L436（Sampler.apply_penalties 入口）
# OpenAI 定义出处：https://platform.openai.com/docs/api-reference/parameter-details
from __future__ import annotations

import math

import pytest
import torch

from vllm.model_executor.layers.utils import (
    apply_penalties,
    get_token_bin_counts_and_mask,
)
from vllm.v1.sample.ops.bad_words import (
    _apply_bad_words_single_batch,
    apply_bad_words,
)
from vllm.v1.sample.ops.penalties import _convert_to_tensors, apply_all_penalties
from vllm.v1.sample.sampler import Sampler


# ── get_token_bin_counts_and_mask（scatter_add_ 计数）───────────────


def test_token_bin_counts_and_mask():
    """(出现次数, 出现 mask)：vocab_size+1 列做 pad 位，计数后裁掉 pad 列。"""
    tokens = torch.tensor([[1, 1, 3], [0, 2, 5]])  # vocab=5，5 是 pad 值
    counts, mask = get_token_bin_counts_and_mask(tokens, vocab_size=5, num_seqs=2)
    assert counts.shape == (2, 5)  # pad 列已裁
    assert counts[0].tolist() == [0, 2, 0, 1, 0]
    assert counts[1].tolist() == [1, 0, 1, 0, 0]  # pad 位 5 不入账
    assert mask[0].tolist() == [False, True, False, True, False]


# ── apply_penalties（真算式）─────────────────────────────────────────


def test_frequency_penalty_openai_definition():
    """frequency：logit -= freq × output_tokens_tensor 出现次数（L87——
    prompt 张量只喂 repetition 的 mask，不进 frequency 计数）。"""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    out = apply_penalties(
        logits,
        prompt_tokens_tensor=torch.tensor([[1]]),
        output_tokens_tensor=torch.tensor([[1, 1, 2]]),
        presence_penalties=torch.tensor([0.0]),
        frequency_penalties=torch.tensor([0.5]),
        repetition_penalties=torch.tensor([1.0]),
    )
    # token1 在 output 出现 2 次：1.0 - 0.5*2 = 0.0；token2 出现 1 次：-0.5
    assert math.isclose(out[0, 1].item(), 0.0, rel_tol=1e-6)
    assert math.isclose(out[0, 2].item(), -0.5, rel_tol=1e-6)
    assert math.isclose(out[0, 0].item(), 2.0, rel_tol=1e-6)  # 未出现不动


def test_presence_penalty_openai_definition():
    """presence：logit -= presence × 是否出现（0/1，不看次数）。"""
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    out = apply_penalties(
        logits,
        prompt_tokens_tensor=torch.tensor([[3, 3, 3]]),  # 3=vocab_size pad
        output_tokens_tensor=torch.tensor([[1, 1, 2]]),
        presence_penalties=torch.tensor([0.25]),
        frequency_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([1.0]),
    )
    # token1/2 出现过：各 -0.25；token0 未出现不动
    assert math.isclose(out[0, 1].item(), 0.75, rel_tol=1e-6)
    assert math.isclose(out[0, 2].item(), -0.25, rel_tol=1e-6)
    assert math.isclose(out[0, 0].item(), 2.0, rel_tol=1e-6)


def test_repetition_penalty_divide_positive_multiply_negative():
    """repetition（自定义 op 的 torch 参考算式，_custom_ops.py:L309-L323）：
    已出现 token 的正 logit 除以 r、负 logit 乘以 r；未出现位不动。"""
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_penalties(
        logits,
        prompt_tokens_tensor=torch.tensor([[3, 3, 3]]),
        output_tokens_tensor=torch.tensor([[0, 1]]),
        presence_penalties=torch.tensor([0.0]),
        frequency_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([2.0]),
    )
    assert math.isclose(out[0, 0].item(), 1.0, rel_tol=1e-6)   # 2.0/2
    assert math.isclose(out[0, 1].item(), -4.0, rel_tol=1e-6)   # -2.0*2
    assert math.isclose(out[0, 2].item(), 1.0, rel_tol=1e-6)    # 未出现


def test_repetition_identity_at_one():
    """r=1.0 是 no-op。"""
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_penalties(
        logits,
        prompt_tokens_tensor=torch.tensor([[3, 3, 3]]),
        output_tokens_tensor=torch.tensor([[0, 1]]),
        presence_penalties=torch.tensor([0.0]),
        frequency_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([1.0]),
    )
    assert torch.allclose(out, torch.tensor([[2.0, -2.0, 1.0]]))


def test_penalties_combined_worked_example():
    """m6 worked example：5-token 玩具词表三式各走一遍——repetition 先动
    （prompt∪output 出现位），frequency/presence 再吃 output 计数。"""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    out = apply_penalties(
        logits,
        prompt_tokens_tensor=torch.tensor([[0]]),
        output_tokens_tensor=torch.tensor([[1, 1, 2]]),
        presence_penalties=torch.tensor([0.25]),
        frequency_penalties=torch.tensor([0.5]),
        repetition_penalties=torch.tensor([1.5]),
    )
    # token1：rep 正除 1.0/1.5 → freq 0.5*2 → presence 0.25（output 出现）
    expected1 = 1.0 / 1.5 - 0.5 * 2 - 0.25
    # token0：只在 prompt 出现 → 只吃 rep（freq/presence 不看 prompt）
    expected0 = 2.0 / 1.5
    assert math.isclose(out[0, 1].item(), expected1, rel_tol=1e-6)
    assert math.isclose(out[0, 0].item(), expected0, rel_tol=1e-6)
    # token3/4 未出现：原样
    assert math.isclose(out[0, 3].item(), -1.0, rel_tol=1e-6)
    assert math.isclose(out[0, 4].item(), -2.0, rel_tol=1e-6)


# ── apply_all_penalties（张量化 + -1 占位替换）──────────────────────


def test_apply_all_penalties_replaces_minus_one_placeholders():
    """异步调度下未惩罚行是 -1 占位（penalties.py:L24-L29）：先替换成
    vocab_size（非法 id 换合法 pad 位）再 scatter——占位不产生任何计数。"""
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    out = apply_all_penalties(
        logits,
        prompt_token_ids=torch.tensor([[3, 3, 3]]),  # 3=vocab_size pad
        presence_penalties=torch.tensor([0.0]),
        frequency_penalties=torch.tensor([1.0]),
        repetition_penalties=torch.tensor([1.0]),
        output_token_ids=[[1, -1, -1]],  # 上拍未定 → -1 占位
    )
    assert math.isclose(out[0, 1].item(), 0.0, rel_tol=1e-6)  # 1.0 - 1.0*1
    assert math.isclose(out[0, 0].item(), 2.0, rel_tol=1e-6)  # 占位不入账


def test_convert_to_tensors_pads_ragged_lists():
    """ragged 历史补 pad=vocab_size（penalties.py:L41-L56）。"""
    t = _convert_to_tensors([[1, 2], [3]], vocab_size=5, device=torch.device("cpu"))
    assert t.shape == (2, 2)
    assert t.tolist() == [[1, 2], [3, 5]]


def test_sampler_apply_penalties_no_penalties_fast_path():
    """no_penalties=True → 原张量直接返回（sampler.py:L425-L426）。"""
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    md = SamplingMetadata(
        temperature=torch.ones(1), all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.zeros(1),
        output_token_ids=[[]],
        allowed_token_ids_mask=None, bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    logits = torch.tensor([[1.0, 2.0]])
    out = Sampler.apply_penalties(logits, md, output_token_ids=[[]])
    assert out is logits


# ── bad_words（前缀匹配）─────────────────────────────────────────────


def test_bad_words_single_token_always_blocked():
    """单词短语（len=1，prefix 空）→ 恒屏蔽末 token（=该词唯一 token）。
    注意入参契约：_apply_bad_words_single_batch 吃单请求的 1-D 行视图
    （apply_bad_words 传 logits[i]）。"""
    logits = torch.tensor([[5.0, 1.0, 2.0]])
    _apply_bad_words_single_batch(logits[0], [[1]], past_tokens_ids=[0, 0])
    assert logits[0, 1].item() == float("-inf")
    assert logits[0, 0].item() == 5.0  # 其余不动（slice 赋值只写末 token）


def test_bad_words_prefix_must_match_recent_suffix():
    """多词短语：仅当 (len-1) 前缀 == 最近输出同长后缀 时屏蔽末 token。"""
    base = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    # 短语 [1, 3]：past 末尾是 1 → 屏蔽 3
    logits = base.clone()
    _apply_bad_words_single_batch(logits[0], [[1, 3]], past_tokens_ids=[0, 1])
    assert logits[0, 3].item() == float("-inf")
    # past 末尾是 0 → 不屏蔽
    logits2 = base.clone()
    _apply_bad_words_single_batch(logits2[0], [[1, 3]], past_tokens_ids=[1, 0])
    assert logits2[0, 3].item() == 2.0


def test_bad_words_too_long_phrase_skipped():
    """len(bad) > len(past)+1 → continue（凑不出补全上下文）。"""
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    _apply_bad_words_single_batch(logits[0], [[0, 1, 2]], past_tokens_ids=[0])
    assert logits[0, 2].item() == 1.0  # 未被屏蔽


def test_bad_words_exact_length_boundary():
    """len(bad) == len(past)+1（恰好整段历史做前缀）。"""
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    _apply_bad_words_single_batch(logits[0], [[0, 2]], past_tokens_ids=[0])
    assert logits[0, 2].item() == float("-inf")  # 前缀 [0] 匹配


def test_apply_bad_words_dispatches_per_request():
    """dict[req_index, ...] 逐请求分派（L30-36）。"""
    logits = torch.tensor([
        [5.0, 1.0, 2.0],
        [1.0, 5.0, 2.0],
    ])
    apply_bad_words(logits, {0: [[1]]}, past_tokens_ids=[[0], [0]])
    assert logits[0, 1].item() == float("-inf")
    assert logits[1, 1].item() == 5.0  # 请求 1 无禁词


def test_sampler_step4_bad_words_end_to_end():
    """step4 端到端：被禁短语的末 token 永不出门。"""
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    logits = torch.tensor([[1.0, 6.0, 2.0]])
    md = SamplingMetadata(
        temperature=torch.zeros(1), all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={0: [[1]]},  # 禁 token 1
        logitsprocs=LogitsProcessors(),
    )
    out = Sampler()(logits, md)
    assert out.sampled_token_ids.flatten().tolist() == [2]


# ── step3 白名单（经 Sampler 端到端）────────────────────────────────


def test_sampler_step3_allowed_mask_end_to_end():
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    logits = torch.tensor([[5.0, 1.0, 4.0]])
    mask = torch.zeros(1, 3, dtype=torch.bool)
    mask[0, [0, 1]] = True  # 只放行 token 2
    md = SamplingMetadata(
        temperature=torch.zeros(1), all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=[[]],
        allowed_token_ids_mask=mask,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    out = Sampler()(logits, md)
    assert out.sampled_token_ids.flatten().tolist() == [2]
