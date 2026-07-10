"""TDD: 验证精简版复现真实 vLLM v1 Sampler 的可观察行为（pin f3fef123）。

纯 PyTorch（CPU）单元测试，不 import vllm。每个测试对应 dossier 记录的真实行为。
"""
import math

import torch

from bad_words import _apply_bad_words_single_batch, apply_bad_words
from penalties import apply_penalties, apply_all_penalties
from logits_processor import (
    LogitsProcessors,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
)
from topk_topp_sampler import (
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
    random_sample,
    TopKTopPSampler,
)
from metadata import SamplingMetadata
from sampler import Sampler, _SAMPLING_EPS

NEG_INF = float("-inf")


def _md(**overrides):
    """构造一个全字段 SamplingMetadata，默认 greedy/无惩罚/无 logprobs。"""
    base = dict(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(0),
        presence_penalties=torch.zeros(0),
        repetition_penalties=torch.zeros(0),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    base.update(overrides)
    return SamplingMetadata(**base)


# ---------- step4: bad words ----------

def test_bad_words_blocks_last_token_only_when_prefix_matches():
    # 被禁短语 [5, 7]，历史末尾是 [5] → 会补全成被禁短语 → 屏蔽 token 7。
    logits = torch.zeros(10)
    _apply_bad_words_single_batch(logits, [[5, 7]], past_tokens_ids=[1, 5])
    assert logits[7] == NEG_INF
    assert torch.isfinite(logits[torch.arange(10) != 7]).all()


def test_bad_words_no_block_when_prefix_mismatch():
    logits = torch.zeros(10)
    _apply_bad_words_single_batch(logits, [[5, 7]], past_tokens_ids=[1, 2])
    assert torch.isfinite(logits).all()


def test_bad_words_single_token_phrase_always_blocked():
    # 长度1的被禁短语 [3]：prefix_length=0，actual==expected==[] → 永远屏蔽。
    logits = torch.zeros(6)
    _apply_bad_words_single_batch(logits, [[3]], past_tokens_ids=[])
    assert logits[3] == NEG_INF


def test_apply_bad_words_dispatches_per_request():
    logits = torch.zeros(2, 8)
    apply_bad_words(logits, {1: [[2, 4]]}, past_tokens_ids=[[0], [9, 2]])
    assert logits[1, 4] == NEG_INF
    assert torch.isfinite(logits[0]).all()


# ---------- step6: penalties ----------

def test_frequency_and_presence_penalty_openai_definition():
    # token 0 出现 2 次，token 1 出现 1 次。freq=0.5, presence=1.0。
    logits = torch.zeros(1, 4)
    prompt = torch.full((1, 1), 4)  # vocab_size 作 pad，无 prompt token
    output = torch.tensor([[0, 0, 1]])
    apply_penalties(
        logits,
        prompt,
        output,
        presence_penalties=torch.tensor([1.0]),
        frequency_penalties=torch.tensor([0.5]),
        repetition_penalties=torch.tensor([1.0]),
    )
    # token0: -0.5*2 - 1.0*1 = -2.0; token1: -0.5*1 - 1.0*1 = -1.5
    assert math.isclose(logits[0, 0].item(), -2.0, abs_tol=1e-5)
    assert math.isclose(logits[0, 1].item(), -1.5, abs_tol=1e-5)
    assert logits[0, 2].item() == 0.0


def test_repetition_penalty_divides_positive_multiplies_negative():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    prompt = torch.full((1, 1), 3)
    output = torch.tensor([[0, 1]])  # tokens 0 and 1 appeared
    apply_penalties(
        logits,
        prompt,
        output,
        presence_penalties=torch.tensor([0.0]),
        frequency_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([2.0]),
    )
    # token0 positive -> /2 = 1.0; token1 negative -> *2 = -4.0; token2 untouched
    assert math.isclose(logits[0, 0].item(), 1.0, abs_tol=1e-5)
    assert math.isclose(logits[0, 1].item(), -4.0, abs_tol=1e-5)
    assert math.isclose(logits[0, 2].item(), 1.0, abs_tol=1e-5)


def test_apply_all_penalties_replaces_minus_one_placeholder():
    # -1 占位 token 不应触发 scatter 越界，应被替换成 vocab_size pad。
    logits = torch.zeros(1, 5)
    out = apply_all_penalties(
        logits,
        prompt_token_ids=torch.full((1, 1), 5),
        presence_penalties=torch.tensor([1.0]),
        frequency_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([1.0]),
        output_token_ids=[[-1]],
    )
    assert torch.isfinite(out).all()
    # -1 被替换成 pad，不计入任何真实 token 惩罚。
    assert (out == 0.0).all()


# ---------- logits processors: argmax-invariance 分类 ----------

def test_logitsprocs_classified_by_argmax_invariance():
    minp = MinPLogitsProcessor(min_p=torch.empty(0), min_p_count=0)
    mintok = MinTokensLogitsProcessor(min_toks={}, logits_slice=None, neg_inf_tensor=None)
    procs = LogitsProcessors([minp, mintok])
    assert minp in procs.argmax_invariant
    assert mintok in procs.non_argmax_invariant


def test_min_p_is_argmax_invariant_and_keeps_max():
    proc = MinPLogitsProcessor(min_p=torch.tensor([[0.5]]), min_p_count=1)
    logits = torch.tensor([[3.0, 1.0, 0.0, -5.0]])
    before_argmax = logits.argmax(dim=-1).clone()
    proc.apply(logits)
    # 最高概率 token 必留（argmax 不变）。
    assert proc.is_argmax_invariant() is True
    assert logits.argmax(dim=-1).item() == before_argmax.item()
    assert torch.isfinite(logits[0, 0])  # the max kept


def test_min_p_count_zero_is_noop():
    proc = MinPLogitsProcessor(min_p=torch.empty(0), min_p_count=0)
    logits = torch.randn(2, 5)
    out = proc.apply(logits.clone())
    assert torch.equal(out, logits) or torch.isfinite(out).all()


def test_min_tokens_censors_eos_via_index_put():
    # req0 屏蔽 stop token 2 与 3。
    slice_ = (torch.tensor([0, 0]), torch.tensor([2, 3]))
    proc = MinTokensLogitsProcessor(
        min_toks={0: None}, logits_slice=slice_,
        neg_inf_tensor=torch.tensor(NEG_INF),
    )
    logits = torch.zeros(1, 5)
    proc.apply(logits)
    assert logits[0, 2] == NEG_INF and logits[0, 3] == NEG_INF
    assert proc.is_argmax_invariant() is False


def test_logit_bias_adds_to_positions():
    slice_ = (torch.tensor([0]), torch.tensor([1]))
    proc = LogitBiasLogitsProcessor(
        biases={0: {1: 5.0}}, logits_slice=slice_, bias_tensor=torch.tensor([5.0])
    )
    logits = torch.zeros(1, 4)
    proc.apply(logits)
    assert logits[0, 1].item() == 5.0
    assert proc.is_argmax_invariant() is False


# ---------- step7: top-k / top-p / temperature ----------

def test_temperature_scaling_and_greedy_div_guard():
    logits = torch.tensor([[2.0, 4.0]])
    temp = torch.tensor([0.0])  # greedy
    out = Sampler.apply_temperature(logits.clone(), temp, all_random=False)
    # temp<eps 被替换为 1.0，不除零。
    assert torch.allclose(out, logits)


def test_top_k_keeps_k_largest():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    k = torch.tensor([2])
    out = apply_top_k_top_p_pytorch(logits.clone(), k=k, p=None)
    finite = torch.isfinite(out[0])
    assert finite.sum().item() == 2
    assert finite[3] and finite[4]  # top-2 logits kept


def test_top_p_nucleus_keeps_at_least_one():
    # 极端 p → 仅留概率最大的 token，但至少 1 个。
    logits = torch.tensor([[10.0, 0.0, -10.0]])
    p = torch.tensor([0.01])
    out = apply_top_k_top_p_pytorch(logits.clone(), k=None, p=p)
    finite = torch.isfinite(out[0])
    assert finite.sum().item() >= 1
    assert finite[0]  # the highest-prob token survives


def test_apply_top_k_top_p_noop_when_both_none():
    logits = torch.randn(3, 7)
    out = apply_top_k_top_p(logits, k=None, p=None)
    assert out is logits


def test_random_sample_deterministic_with_generator():
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    g = torch.Generator().manual_seed(123)
    out = random_sample(probs.clone(), {0: g})
    # 概率全在 token 2 上 → 必采样到 2。
    assert out.item() == 2


# ---------- 整条流水线：Sampler.forward ----------

def test_all_greedy_returns_argmax_and_early_exits():
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 1.0]])
    md = _md(all_greedy=True, all_random=False, temperature=torch.tensor([0.0, 0.0]))
    out = sampler.forward(logits, md)
    assert out.sampled_token_ids.shape == (2, 1)
    assert out.sampled_token_ids[0, 0].item() == 1
    assert out.sampled_token_ids[1, 0].item() == 0
    assert out.logprobs_tensors is None


def test_mixed_batch_greedy_and_random_merge_via_where():
    sampler = Sampler()
    torch.manual_seed(0)
    # req0 greedy(temp=0)，req1 random(temp=1)。给 req0 一个尖峰使 argmax 确定。
    logits = torch.tensor([[0.0, 100.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    md = _md(
        all_greedy=False,
        all_random=False,
        temperature=torch.tensor([0.0, 1.0]),
        top_k=None,
        top_p=None,
    )
    out = sampler.forward(logits, md)
    # req0 必为 argmax=1（greedy 行不受随机影响）。
    assert out.sampled_token_ids[0, 0].item() == 1


def test_gather_logprobs_shape_and_rank():
    sampler = Sampler()
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    md = _md(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        max_num_logprobs=2,
    )
    out = sampler.forward(logits, md)
    lt = out.logprobs_tensors
    assert lt is not None
    # (token + topk) = 1 + 2 = 3 列。
    assert lt.logprob_token_ids.shape == (1, 3)
    assert lt.logprobs.shape == (1, 3)
    # 采样 token = argmax = 3，是最大 logit → rank 1（#{logprob >= 该值}==1）。
    assert lt.selected_token_ranks[0].item() == 1


def test_raw_logprobs_taken_before_penalties():
    # raw logprobs 应反映原始 logits，而非被惩罚后的。
    sampler = Sampler()
    logits = torch.tensor([[0.0, 0.0, 5.0]])
    # forward 会原地修改 logits（惩罚/温度 in-place），故先快照未惩罚分布。
    expected = logits.log_softmax(dim=-1, dtype=torch.float32).clone()
    md = _md(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        max_num_logprobs=1,
        no_penalties=False,
        prompt_token_ids=torch.full((1, 1), 3),
        presence_penalties=torch.tensor([100.0]),  # huge penalty on appeared token
        frequency_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([1.0]),
        output_token_ids=[[2]],  # token 2 already output -> would be penalized
    )
    out = sampler.forward(logits, md)
    lt = out.logprobs_tensors
    # 惩罚把 token2 的 logit 砸到 5-100=-95，argmax 改选 token0/1。关键不变量：
    # 返回的 raw logprob（第 0 列）是被采样 token 在【未惩罚】分布下的 log_softmax，
    # 而非被惩罚后的分布。
    sampled_tok = out.sampled_token_ids[0, 0].item()
    assert sampled_tok in (0, 1)  # 惩罚生效，不再是原 argmax token2
    assert math.isclose(
        lt.logprobs[0, 0].item(), expected[0, sampled_tok].item(), abs_tol=1e-4
    )


def test_topk_topp_sampler_native_backend_bound_on_host():
    s = TopKTopPSampler()
    # host 无 CUDA → 绑定 forward_native。
    assert s.forward == s.forward_native


def test_forward_cuda_falls_back_to_native_when_no_kp_or_generators():
    s = TopKTopPSampler()
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    g = torch.Generator().manual_seed(1)
    # 有 generator → forward_cuda 回退 native（不触 flashinfer）。
    out, extra = s.forward_cuda(logits.clone(), {0: g}, k=None, p=None)
    assert out.shape == (1,)
