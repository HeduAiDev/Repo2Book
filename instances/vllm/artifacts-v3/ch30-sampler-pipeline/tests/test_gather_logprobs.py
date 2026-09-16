# ch29 主电池五：step8 gather 三件套（topk + 被采样 logprob + 不排序 rank）
# 与 batched_count_greater_than、SamplingMetadata 冻结快照契约。
# 基准（v0.27.1 现核行号）：
#   - vllm/v1/sample/sampler.py:L304-L356（compute_logprobs + gather_logprobs）
#   - vllm/v1/sample/ops/logprobs.py:L10-L27（rank=(x>=v).sum(-1) 不排序；
#     torch.compile + mark_unbacked 防 1→2 重编译）
#   - vllm/v1/sample/metadata.py:L14-L55（冻结快照 dataclass）
# 支路全景（D2H/切行/装配）归 ch8，本章只补「rank 不排序」与「raw 视角」。
from __future__ import annotations

import dataclasses

import math

import pytest
import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.sampler import Sampler


# ── compute_logprobs / batched_count_greater_than ────────────────────


def test_compute_logprobs_is_log_softmax_fp32():
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    out = Sampler.compute_logprobs(logits)
    expected = torch.log_softmax(
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), dim=-1)
    assert out.dtype == torch.float32
    assert torch.allclose(out, expected)


def test_batched_count_greater_than_counts_geq_including_self():
    """rank = #{x >= v}（含自身）；O(V) 计数不排序。"""
    x = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    values = torch.tensor([[2.0], [5.0]])
    out = batched_count_greater_than(x, values)
    assert out.tolist() == [2, 3]


def test_batched_count_greater_than_rank_with_ties():
    """并列 logprob → 同 rank（计数把并列都算上）。"""
    x = torch.tensor([[0.5, 0.3, 0.3, 0.1]])
    values = torch.tensor([[0.3]])
    assert batched_count_greater_than(x, values).tolist() == [3]


# ── gather_logprobs（列布局 [被采样 | topk]）────────────────────────


def test_gather_logprobs_3token_worked_example():
    """m16 worked example：logprobs=[0.5,0.3,0.2]，被采样 token 2（0.2）。
    第 0 列恒被采样 token；topk 按值降序；rank=#{>=自身}（含并列与自身）。"""
    logprobs = torch.tensor([[0.5, 0.3, 0.2]])
    lt = Sampler.gather_logprobs(logprobs, num_logprobs=2,
                                 token_ids=torch.tensor([2], dtype=torch.int64))
    assert lt.logprob_token_ids.tolist() == [[2, 0, 1]]  # int32、采样位第 0 列
    assert math.isclose(lt.logprobs[0, 0].item(), 0.2, rel_tol=1e-6)
    assert math.isclose(lt.logprobs[0, 1].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(lt.logprobs[0, 2].item(), 0.3, rel_tol=1e-6)
    assert lt.selected_token_ranks.tolist() == [3]  # 0.5、0.3、0.2 >= 0.2
    assert lt.logprob_token_ids.dtype == torch.int32


def test_gather_logprobs_requires_int64_token_ids():
    """assert token_ids.dtype == torch.int64（L332）。"""
    with pytest.raises(AssertionError):
        Sampler.gather_logprobs(torch.tensor([[0.5, 0.3]]), 1,
                                token_ids=torch.tensor([0], dtype=torch.int32))


def test_gather_logprobs_batch_layout():
    logprobs = torch.tensor([[0.1, 0.9, 0.0], [0.0, 0.2, 0.8]])
    lt = Sampler.gather_logprobs(logprobs, num_logprobs=1,
                                 token_ids=torch.tensor([0, 2]))
    assert lt.logprobs.shape == (2, 2)
    assert lt.logprob_token_ids[:, 0].tolist() == [0, 2]  # 被采样恒第 0 列
    assert lt.logprob_token_ids[:, 1].tolist() == [1, 2]  # 各行 top-1
    assert lt.selected_token_ranks.tolist() == [2, 1]  # 行1 采样位=top-1 → rank 1


def test_gather_logprobs_sampled_inside_topk_no_dedup_here():
    """被采样 token 在 topk 内也不在此去重——+1/+0 的合并归
    LogprobsProcessor 输出处理（docstring L52-57，ch8 已铺）。"""
    logprobs = torch.tensor([[0.9, 0.1, 0.0]])
    lt = Sampler.gather_logprobs(logprobs, num_logprobs=1,
                                 token_ids=torch.tensor([0]))
    assert lt.logprob_token_ids.tolist() == [[0, 0]]  # 两列都是 token 0


# ── SamplingMetadata 冻结快照契约 ───────────────────────────────────


def test_metadata_fields_match_v0271_minus_subtractions():
    """快照字段 = v0.27.1 的 19 字段减去 spec_token_ids 与
    thinking_budget_state_holder（subtraction_plan delete[2]/[1]）。"""
    names = {f.name for f in dataclasses.fields(SamplingMetadata)}
    expected = {
        "temperature", "all_greedy", "all_random", "top_p", "top_k",
        "generators", "max_num_logprobs", "no_penalties", "prompt_token_ids",
        "frequency_penalties", "presence_penalties", "repetition_penalties",
        "output_token_ids", "allowed_token_ids_mask", "bad_words_token_ids",
        "logitsprocs", "logprob_token_ids",
    }
    assert names == expected
    assert "spec_token_ids" not in names
    assert "thinking_budget_state_holder" not in names


def test_metadata_logprob_token_ids_defaults_none():
    md = SamplingMetadata(
        temperature=None, all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=[[]], allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=None,
    )
    assert md.logprob_token_ids is None


def test_metadata_output_token_ids_shared_reference():
    """WC4 不变量：output_token_ids 是共享 list 引用——处理器总见最新列表
    （interface.py:L44-48 的注释语义；惩罚/bad_words 吃『到目前为止的历史』）。"""
    history = [1, 2]
    md = SamplingMetadata(
        temperature=None, all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=history, allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=None,
    )
    assert md.output_token_ids is history
    history.append(3)  # 采样后追加新 token——快照视角随之更新
    assert md.output_token_ids[-1] == 3
