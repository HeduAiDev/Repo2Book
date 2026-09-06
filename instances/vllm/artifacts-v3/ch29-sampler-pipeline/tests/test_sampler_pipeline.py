# ch29 主电池一：Sampler.forward 9 步管线的可观察行为（真实 v0.27.1 语义）。
# 基准：vllm/v1/sample/sampler.py:L20-L59 docstring 契约 + L72-L149 forward。
# 全部 host 可跑（CPU 张量、批 < 8 走 pytorch sort 支路；flashinfer 裁决由
# conftest 的 VLLM_USE_FLASHINFER_SAMPLER=0 走真实禁用支路）。
from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS, Sampler


def make_metadata(batch: int, **over) -> SamplingMetadata:
    """按 gpu_input_batch._make_sampling_metadata 产出的冻结快照形状直建
    （快照生产者归 ch18；本章精简版以构造好的元数据为输入——测试自建 state）。"""
    fields = dict(
        temperature=torch.full((batch,), 1.0),
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch),
        presence_penalties=torch.zeros(batch),
        repetition_penalties=torch.ones(batch),
        output_token_ids=[[] for _ in range(batch)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    fields.update(over)
    return SamplingMetadata(**fields)


def ensure_native_binding(sampler: Sampler) -> None:
    """HOST SEAM：减法后 self.forward 仅 is_cuda() 下绑定（delete[4] 计划原文）。
    非 CUDA 环境下测试复现构造期 native 绑定支路（topk_topp_sampler.py
    L100-L102 的 else 位——真实控制流的一部分）。"""
    t = sampler.topk_topp_sampler
    if "forward" not in t.__dict__:
        t.forward = t.forward_native


# ── step7a：greedy 快路径 ─────────────────────────────────────────────


def test_all_greedy_early_return():
    """all_greedy 直接 argmax 早退（L261-271）：整条跳过温度/argmax 不变列/
    top-k/top-p；temperature 可为 None（快照生产者在 all_greedy 时给 None）。"""
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0, -1.0], [-3.0, 0.5, 4.0, 1.0, 2.0]])
    md = make_metadata(2, all_greedy=True, temperature=None)
    sampler = Sampler()
    out = sampler(logits, md)
    assert out.sampled_token_ids.dtype == torch.int32
    assert out.sampled_token_ids.shape == (2, 1)
    assert out.sampled_token_ids.flatten().tolist() == [1, 2]
    assert out.logprobs_tensors is None  # max_num_logprobs=None 默认路径


def test_all_greedy_skips_temperature_logit_mutation():
    """greedy 早退发生在 apply_temperature 之前 → 输入张量不被温度改写。"""
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    before = logits.clone()
    md = make_metadata(1, all_greedy=True, temperature=None)
    Sampler()(logits, md)
    assert torch.equal(logits, before)


def test_sampling_eps_is_1e5():
    """正文反复引用的 greedy 判定阈值（sampler.py:L17）。"""
    assert _SAMPLING_EPS == 1e-5


# ── step1-2：raw 留底 + fp32 ──────────────────────────────────────────


def test_raw_logprobs_computed_before_any_processor():
    """WC2/m4：raw 留底在一切变换之前——logit_bias 能改 argmax（非不变列、
    step5 生效），但 top-k logprobs 仍是原始 logits 的 log_softmax。"""
    logits = torch.tensor([[1.0, 2.0, 3.0, 2.5, 0.5]])
    # logit_bias：请求 0 给 token 3 加 +10 → argmax 2 → 3
    lb = LogitBiasLogitsProcessor(None, torch.device("cpu"), False)
    lb.biases = {0: {3: 10.0}}
    lb.bias_tensor = torch.tensor([10.0], dtype=torch.float32)
    lb.logits_slice = (torch.tensor([0], dtype=torch.int32),
                       torch.tensor([3], dtype=torch.int32))
    md = make_metadata(1, max_num_logprobs=2,
                       logitsprocs=LogitsProcessors([lb]))
    out = Sampler()(logits, md)
    assert out.sampled_token_ids.flatten().tolist() == [3]  # bias 改变了 argmax

    # 但 logprobs 是 raw 视角：log_softmax(原始 logits)
    raw = torch.log_softmax(torch.tensor([1.0, 2.0, 3.0, 2.5, 0.5]), dim=-1)
    lt = out.logprobs_tensors
    # 第 0 列恒为被采样 token（3）的 raw logprob
    assert math.isclose(lt.logprobs[0, 0].item(), raw[3].item(), rel_tol=1e-6)
    # top-2 按原始分布排：token 2（3.0）与 token 3（2.5）
    assert lt.logprob_token_ids[0, 1].item() == 2
    assert lt.logprob_token_ids[0, 2].item() == 3
    assert math.isclose(lt.logprobs[0, 1].item(), raw[2].item(), rel_tol=1e-6)


def test_raw_logits_mode_clones_logits():
    """logprobs_mode='raw_logits'：留底的是 float32 logits 克隆/转型（L89-93）。"""
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    md = make_metadata(1, all_greedy=True, temperature=None, max_num_logprobs=1)
    sampler = Sampler(logprobs_mode="raw_logits")
    out = sampler(logits, md)
    lt = out.logprobs_tensors
    # num_logprobs==1 → 走 gather：被采样 token(1) 第 0 列 + top-1
    assert lt.logprobs.shape == (1, 2)
    assert math.isclose(lt.logprobs[0, 0].item(), 5.0, rel_tol=1e-6)


def test_raw_logprobs_survive_inplace_mutation():
    """m15：9 步全程原地改写 logits；raw 留底先抽好（L84-93），后续 in-place
    不污染它。用惩罚（step6 会 in-place 改 logits）验证。"""
    logits = torch.tensor([[1.0, 2.0, 3.0, 2.5, 0.5]])
    raw_expected = torch.log_softmax(logits.clone(), dim=-1)
    md = make_metadata(
        1,
        max_num_logprobs=2,
        no_penalties=False,
        prompt_token_ids=torch.tensor([[5, 5, 5]]),  # pad=vocab_size
        frequency_penalties=torch.tensor([2.0]),
        output_token_ids=[[2, 2]],
        generators={0: torch.Generator().manual_seed(2024)},  # 混批随机行定种子
    )
    out = Sampler()(logits, md)
    lt = out.logprobs_tensors
    # token 2 出现 2 次：3.0 - 2.0*2 = -1.0 → 处理后分布里 token 3（2.5）领涨；
    # 无论掷骰选中谁，留底 logprobs 都必须是 raw 视角（下两条断言核语义）
    assert out.sampled_token_ids.flatten().tolist() == [3]
    # 但留底的 logprobs 是惩罚之前抽好的 raw 值（未被 in-place 污染）
    assert math.isclose(lt.logprobs[0, 0].item(), raw_expected[0, 3].item(),
                        rel_tol=1e-6)
    assert math.isclose(lt.logprobs[0, 1].item(), raw_expected[0, 2].item(),
                        rel_tol=1e-6)


def test_inplace_ownership_fp32_input_mutated_caller_tensor():
    """m15：fp32 输入时 logits.to(float32) 返回自身 → 后续 in-place 操作
    直接改写调用方张量（想保 raw 的调用方必须自己 clone——RejectionSampler
    的实证注释 rejection_sampler.py:L152-160 同款语义）。"""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    md = make_metadata(
        1, no_penalties=False,
        prompt_token_ids=torch.tensor([[3, 3, 3]]),  # 3=vocab_size pad
        frequency_penalties=torch.tensor([1.0]),
        output_token_ids=[[1]],
    )
    Sampler()(logits, md)
    # token 1 出现 1 次：logit 2.0 - 1.0*1 = 1.0
    assert math.isclose(logits[0, 1].item(), 1.0, rel_tol=1e-6)


def test_fp16_input_promoted_not_mutated():
    """step2 fp32 化对非 fp32 输入会新建张量 → 调用方 fp16 张量不被改写。"""
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    before = logits.clone()
    md = make_metadata(
        1, no_penalties=False,
        prompt_token_ids=torch.tensor([[3, 3, 3]]),
        frequency_penalties=torch.tensor([1.0]),
        output_token_ids=[[1]],
    )
    Sampler()(logits, md)
    assert torch.equal(logits, before)
    assert logits.dtype == torch.float16


# ── step3-4：allowed 白名单 / bad_words ───────────────────────────────


def test_allowed_token_ids_whitelist():
    """step3：白名单外的位 masked_fill_(-inf)（L391-392）。掩码极性见
    gpu_input_batch.py:L282-L283 注释原话：「if the corresponding token
    allowed, the value is False. Since we use masked_fill_ to set -inf.」"""
    logits = torch.tensor([[5.0, 1.0, 4.0, 0.0, 2.0]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    mask[0, [1, 2]] = False  # 只放行 1/2（False=允许位）
    md = make_metadata(1, all_greedy=True, temperature=None,
                       allowed_token_ids_mask=mask)
    out = Sampler()(logits, md)
    assert out.sampled_token_ids.flatten().tolist() == [2]  # 5.0 被砍 → 第二名 4.0


# ── step7：混批 torch.where 合并 ─────────────────────────────────────


def test_mixed_batch_where_merge_greedy_rows_exact():
    """m3：混批两路先各算一遍，再按逐请求 temp<eps 逐行选（L296-301）。
    temp=0 行必须精确等于 greedy argmax；temp>0 行落在支撑集内。"""
    torch.manual_seed(0)
    logits = torch.tensor([
        [0.1, 6.0, 1.0, 0.5, 0.2],   # row0 greedy（temp=0）
        [2.0, 1.9, 1.8, 0.0, 0.0],   # row1 random
        [0.0, 0.1, 0.2, 3.0, 0.3],   # row2 greedy（temp=0）
    ])
    md = make_metadata(
        3,
        temperature=torch.tensor([0.0, 1.0, 0.0]),
        all_greedy=False,
        all_random=False,
    )
    sampler = Sampler()
    ensure_native_binding(sampler)
    out = sampler(logits, md)
    tokens = out.sampled_token_ids.flatten().tolist()
    assert tokens[0] == 1  # greedy 行精确
    assert tokens[2] == 3
    assert tokens[1] in range(5)  # random 行合法


def test_mixed_batch_where_merge_selects_by_temperature_deterministically():
    """m3 定向哨兵：torch.where(temp<eps, greedy, random) 的逐行选择语义用
    哨兵随机结果钉死——greedy 行必须取 greedy_sampled、random 行必须取随机
    路的返回值。背景：上方 end-to-end 测试的 greedy 行 logits 尖峰分布下
    固定种子随机路也掷出 argmax（mutation 实锤：把 `<` 反转成 `>` 该测试
    仍绿——「greedy 行精确」断言与随机路结果巧合一致，掩盖了选择逻辑反转）。
    哨兵 token 4 ≠ 任何 greedy 位（1/3），反转即被看见。"""
    logits = torch.tensor([
        [0.1, 6.0, 1.0, 0.5, 0.2],   # row0 greedy（temp=0 → argmax=1）
        [2.0, 1.9, 1.8, 0.0, 0.0],   # row1 random（temp=1）
        [0.0, 0.1, 0.2, 3.0, 0.3],   # row2 greedy（temp=0 → argmax=3）
    ])
    md = make_metadata(
        3,
        temperature=torch.tensor([0.0, 1.0, 0.0]),
        all_greedy=False,
        all_random=False,
    )
    sampler = Sampler()
    sentinel = torch.tensor([4, 4, 4], dtype=torch.int64)  # 随机路的可区分替身

    class _StubTopKTopP(torch.nn.Module):
        """替身 topk_topp_sampler：返回可区分的哨兵随机结果——只隔离
        掷骰随机性，sample() 的 greedy 计算/温度/where 合并全走真控制流。
        须为 nn.Module 子类：Sampler.topk_topp_sampler 是 nn.Module 注册的
        子模块槽，torch 不允许把普通对象赋进该槽（TypeError）。"""

        def forward(self, logits, generators, k, p):
            return sentinel.clone(), None

    sampler.topk_topp_sampler = _StubTopKTopP()
    out = sampler(logits, md)
    # temp<eps 行取 greedy（1/3）；temp>=eps 行取随机路哨兵（4）
    assert out.sampled_token_ids.flatten().tolist() == [1, 4, 3]


def test_mixed_batch_greedy_row_tracks_processed_logits():
    """greedy 行的 argmax 取自 step3-6 处理后的 logits（min_tokens 封 EOS
    能改变 greedy 结果——非不变列在 greedy 前生效的证据）。"""
    logits = torch.tensor([[1.0, 2.0, 6.0, 0.0]])  # argmax=2(EOS)
    mt = MinTokensLogitsProcessor(None, torch.device("cpu"), False)
    mt.min_toks = {0: (3, [], {2})}
    mt.logits_slice = (torch.tensor([0], dtype=torch.int32),
                       torch.tensor([2], dtype=torch.int32))
    md = make_metadata(
        1, all_greedy=True, temperature=None,
        logitsprocs=LogitsProcessors([mt]),
    )
    out = Sampler()(logits, md)
    assert out.sampled_token_ids.flatten().tolist() == [1]  # EOS 被封 → 第二名


def test_all_random_path_runs_temperature_and_minp():
    """all_random：跳过 greedy 分支（greedy_sampled=None），温度 + argmax
    不变列（min_p）在随机路径生效。"""
    torch.manual_seed(1)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1]])
    mp = MinPLogitsProcessor(type("C", (), {"scheduler_config":
              type("S", (), {"max_num_seqs": 8})})(), torch.device("cpu"), False)
    mp.min_p_cpu[0] = 0.5
    mp.min_p_count = 1
    mp.min_p = mp.min_p_device[:1].unsqueeze(1).clone()
    md = make_metadata(
        1, all_random=True, all_greedy=False,
        temperature=torch.tensor([1.0]),
        logitsprocs=LogitsProcessors([mp]),
    )
    sampler = Sampler()
    ensure_native_binding(sampler)
    out = sampler(logits, md)
    # min_p=0.5：阈值 0.5*max_prob；softmax([3,2,1,.5,.1]) max≈0.665
    # 阈值≈0.33 → 只留 token 0（0.665）与 token 1（0.245<0.33 被砍）
    probs = torch.softmax(torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1]), dim=-1)
    assert probs[1].item() < 0.5 * probs[0].item()  # token1 确在阈值之下
    assert out.sampled_token_ids.flatten().tolist() == [0]


def test_all_greedy_processed_logprobs_mode():
    """logprobs_mode='processed_logprobs'：all_greedy 快路径返回处理后
    logits 的 log_softmax（L267-270）。"""
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    md = make_metadata(1, all_greedy=True, temperature=None, max_num_logprobs=2)
    sampler = Sampler(logprobs_mode="processed_logprobs")
    out = sampler(logits, md)
    # processed（此处无处理器，= 原 logits）log_softmax
    expected = torch.log_softmax(torch.tensor([1.0, 3.0, 2.0]), dim=-1)
    lt = out.logprobs_tensors
    # argmax=token1 既是被采样位（col0）也是 top-1（col1）——并列不去重，
    # +1/+0 合并归 LogprobsProcessor（ch8）
    assert math.isclose(lt.logprobs[0, 0].item(), expected[1].item(), rel_tol=1e-6)
    assert math.isclose(lt.logprobs[0, 1].item(), expected[1].item(), rel_tol=1e-6)
    assert math.isclose(lt.logprobs[0, 2].item(), expected[2].item(), rel_tol=1e-6)


# ── step8-9：gather 三分支与出件 ─────────────────────────────────────


def test_num_logprobs_minus_one_returns_full_unsorted():
    """num_logprobs==-1：整张未排序未排名 raw_logprobs（L122-126）——
    RejectionSampler bonus 位的契约。"""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    md = make_metadata(1, max_num_logprobs=-1)
    out = Sampler()(logits, md)
    lt = out.logprobs_tensors
    expected = torch.log_softmax(torch.tensor([1.0, 2.0, 3.0]), dim=-1)
    assert lt.logprob_token_ids.numel() == 0
    assert lt.selected_token_ranks.numel() == 0
    assert torch.allclose(lt.logprobs, expected.unsqueeze(0))


def test_apply_temperature_div_by_zero_guard():
    """温度除零保护（L228-237）：非 all_random 时 temp<eps 替 1.0。"""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    temp = torch.tensor([0.0])
    out = Sampler.apply_temperature(logits.clone(), temp, all_random=False)
    assert torch.isfinite(out).all()
    assert torch.equal(out, logits)  # 除以 1.0 → 不变
    out2 = Sampler.apply_temperature(logits.clone(), torch.tensor([2.0]), True)
    assert torch.allclose(out2, logits / 2.0)


def test_forward_returns_sampler_output_gpu_tensors_contract():
    """出件契约（L139-148）：int32 [num_reqs,1]；logprobs_tensors 与 logits
    同 device。"""
    logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 2.0]])
    md = make_metadata(2, all_greedy=True, temperature=None, max_num_logprobs=1)
    out = Sampler()(logits, md)
    assert out.sampled_token_ids.shape == (2, 1)
    assert out.sampled_token_ids.dtype == torch.int32
    assert out.logprobs_tensors.logprobs.device == logits.device
    assert out.logprobs_tensors.logprob_token_ids.dtype == torch.int32
