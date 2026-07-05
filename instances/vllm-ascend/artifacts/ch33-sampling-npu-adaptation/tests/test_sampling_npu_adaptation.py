"""ch28 —— 采样的 NPU 对位（Gumbel 异步指数随机 + 薄壳子类化 + Triton 优雅回退）。

测的是精简版**复现真仓可观察行为**，不是自洽：算法宝石 random_sample 的 Gumbel-max 与
torch.multinomial 同分布是纯数学，host CPU torch 即可经验验证。
"""

import types

import torch


# ---------------------------------------------------------------------------
# (1) 算法宝石：random_sample 的 Gumbel-max（probs.div_(q).argmax, q~Exp(1)）与多项式采样同分布
# ---------------------------------------------------------------------------
def test_random_sample_matches_categorical_distribution(env):
    torch.manual_seed(0)
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    B = 40000
    probs = p.unsqueeze(0).repeat(B, 1).contiguous()

    # 无种子的 common case：generators 为空 → 整张 q 走一次 exponential_。
    idx = env.sampler.random_sample(probs, {})

    assert idx.shape == (B,)
    freq = torch.bincount(idx, minlength=4).float() / B
    # 经验频率应逼近类别概率 p（Gumbel-max 与 Categorical(p) 同分布）。
    assert torch.allclose(freq, p, atol=0.02), f"freq={freq.tolist()} vs p={p.tolist()}"


def test_random_sample_seeded_is_reproducible(env):
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    def run():
        g0 = torch.Generator().manual_seed(123)
        g1 = torch.Generator().manual_seed(456)
        return env.sampler.random_sample(p.clone(), {0: g0, 1: g1})

    a = run()
    b = run()
    # 每行都有种子（len(generators)==B），逐行 exponential_(generator) → 完全可复现。
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# (2) HAS_TRITON 优雅回退：penalties 走昇腾 Triton 内核 / 不可用回退基类原版
# ---------------------------------------------------------------------------
def test_apply_penalties_falls_back_to_base_without_triton(env):
    env.set_triton(False)
    logits = torch.randn(2, 5)
    sm = types.SimpleNamespace(no_penalties=False)

    out = env.sampler.AscendSampler.apply_penalties(logits, sm, [[1], [2]])

    assert "BASE_Sampler.apply_penalties" in env.rec.calls
    assert "apply_penalties_triton" not in env.rec.calls
    assert out is logits


def test_apply_penalties_uses_triton_kernel_when_available(env):
    env.set_triton(True)
    logits = torch.randn(2, 6)
    sm = types.SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=torch.tensor([[1, 2], [3, 4]]),
        presence_penalties=torch.zeros(2),
        frequency_penalties=torch.zeros(2),
        repetition_penalties=torch.ones(2),
    )

    env.sampler.AscendSampler.apply_penalties(logits, sm, [[1, 2], [3]])

    assert "apply_penalties_triton" in env.rec.calls
    assert "BASE_Sampler.apply_penalties" not in env.rec.calls


def test_apply_penalties_no_penalties_shortcircuits(env):
    env.set_triton(True)
    logits = torch.randn(2, 6)
    sm = types.SimpleNamespace(no_penalties=True)
    out = env.sampler.AscendSampler.apply_penalties(logits, sm, [[1], [2]])
    assert out is logits
    assert "apply_penalties_triton" not in env.rec.calls


# ---------------------------------------------------------------------------
# (3) top-k/top-p pytorch 截断（apply_top_k_top_p 的唯一 host 实现）
# ---------------------------------------------------------------------------
def test_apply_top_k_top_p_pytorch_topk_masks_tail(env):
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    k = torch.tensor([2])  # 仅保留 top-2
    out = env.sampler._apply_top_k_top_p_pytorch(logits.clone(), k, None)
    finite = torch.isfinite(out[0])
    # 最大的两个（值 4、3，索引 3、2）保留，其余置 -inf。
    assert finite.tolist() == [False, False, True, True]


def test_apply_top_k_top_p_noop_when_k_and_p_none(env):
    logits = torch.randn(2, 5)
    out = env.sampler.apply_top_k_top_p(logits, None, None)
    assert out is logits  # 无截断时原样返回


# ---------------------------------------------------------------------------
# (4) AscendTopKTopPSampler.forward_native：BATCH_INVARIANT 回退基类 / 默认走 random_sample
# ---------------------------------------------------------------------------
def test_forward_native_batch_invariant_falls_back_to_base(env):
    env.set_batch_invariant(True)
    s = env.sampler.AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
    out = s.forward_native(torch.randn(2, 4), {}, None, None)
    assert out == ("BASE_native", None)
    assert "BASE_TopKTopP.forward_native" in env.rec.calls


def test_forward_native_default_path_samples_in_range(env):
    env.set_batch_invariant(False)
    env.set_triton(False)
    s = env.sampler.AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
    logits = torch.randn(3, 5)
    tokens, logits_to_return = s.forward_native(logits, {}, None, None)
    assert tokens.shape == (3,)
    assert int(tokens.min()) >= 0 and int(tokens.max()) < 5
    assert logits_to_return is None  # raw_logprobs 模式不回传 processed logits


def test_greedy_sample_is_argmax(env):
    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.0, 1.0]])
    out = env.sampler.AscendSampler.greedy_sample(logits)
    assert out.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# (5) 投机解码拒绝采样：greedy 接受检验（host pytorch 回退）
# ---------------------------------------------------------------------------
def test_rejection_greedy_all_accept_appends_bonus(env):
    PH = env.PLACEHOLDER_TOKEN_ID
    output = torch.full((1, 3), PH, dtype=torch.int32)
    draft = torch.tensor([5, 6], dtype=torch.int32)
    target_argmax = torch.tensor([5, 6])  # 全部命中
    bonus = torch.tensor([[7]], dtype=torch.int32)
    env.rejection.rejection_greedy_sample_pytorch(
        output, torch.tensor([2]), draft, target_argmax, bonus, [2], 2, None
    )
    assert output[0].tolist() == [5, 6, 7]


def test_rejection_greedy_mismatch_truncates_no_bonus(env):
    PH = env.PLACEHOLDER_TOKEN_ID
    output = torch.full((1, 3), PH, dtype=torch.int32)
    draft = torch.tensor([5, 6], dtype=torch.int32)
    target_argmax = torch.tensor([5, 9])  # 第二个 token mismatch
    bonus = torch.tensor([[7]], dtype=torch.int32)
    env.rejection.rejection_greedy_sample_pytorch(
        output, torch.tensor([2]), draft, target_argmax, bonus, [2], 2, None
    )
    # pos0 接受 target 5；pos1 用 target_argmax 的 9 改写；无 bonus。
    assert output[0].tolist() == [5, 9, PH]


# ---------------------------------------------------------------------------
# (6) 投机解码拒绝采样：random 接受判据 target/draft >= u（host pytorch 回退）
# ---------------------------------------------------------------------------
def _random_sample_args(env, uniform_val):
    PH = env.PLACEHOLDER_TOKEN_ID
    output = torch.full((1, 2), PH, dtype=torch.int32)
    cu = torch.tensor([1])
    draft = torch.tensor([1], dtype=torch.int64)
    draft_probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # draft 选 token1，概率 1
    target_probs = torch.tensor([[0.1, 0.6, 0.2, 0.1]])  # target 对 token1 概率 0.6
    bonus = torch.tensor([[7]], dtype=torch.int32)
    recovered = torch.tensor([3], dtype=torch.int64)
    uniform = torch.tensor([uniform_val])
    is_greedy = torch.tensor([False])
    return output, cu, draft, draft_probs, target_probs, bonus, recovered, uniform, is_greedy


def test_rejection_random_accept_then_bonus(env):
    args = _random_sample_args(env, uniform_val=0.3)  # 0.6/1.0=0.6 >= 0.3 → 接受
    output = args[0]
    env.rejection.rejection_random_sample_pytorch(*args, 1, 4, IS_NGRAM=False)
    # 接受 draft token1，全接受补 bonus 7。
    assert output[0].tolist() == [1, 7]


def test_rejection_random_reject_takes_recovered(env):
    args = _random_sample_args(env, uniform_val=0.9)  # 0.6 >= 0.9 → 拒绝
    output = args[0]
    env.rejection.rejection_random_sample_pytorch(*args, 1, 4, IS_NGRAM=False)
    # 被拒 → 取 recovered token3；无 bonus。
    assert output[0, 0].item() == 3
    assert output[0, 1].item() == env.PLACEHOLDER_TOKEN_ID


# ---------------------------------------------------------------------------
# (7) 残差重采：sample_recovered_tokens_pytorch 的 max(0, target-draft)/q argmax
# ---------------------------------------------------------------------------
def test_sample_recovered_tokens_residual_argmax(env):
    out = torch.empty(1, dtype=torch.int64)
    cu = torch.tensor([1])
    draft = torch.tensor([1], dtype=torch.int64)
    draft_probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    target_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    q = torch.ones(1, 4)  # 归一化因子全 1 → 直接对残差取 argmax
    env.rejection.sample_recovered_tokens_pytorch(out, cu, draft, draft_probs, target_probs, q, 4, IS_NGRAM=False)
    # residual = max(0, [0.1, 0.2-1, 0.3, 0.4]) = [0.1, 0, 0.3, 0.4] → argmax = 3
    assert out[0].item() == 3


# ---------------------------------------------------------------------------
# (8) rejection_sample 端到端：HAS_TRITON 关 → greedy 路径 pytorch 回退
# ---------------------------------------------------------------------------
def _greedy_metadata(env):
    return types.SimpleNamespace(all_greedy=True, all_random=False, generators={}, temperature=None)


def test_rejection_sample_greedy_end_to_end_no_triton(env):
    env.set_triton(False)
    draft = torch.tensor([5, 6], dtype=torch.int32)
    # target_logits：行0 argmax=5、行1 argmax=6 → 全命中。
    target_logits = torch.full((2, 8), -10.0)
    target_logits[0, 5] = 10.0
    target_logits[1, 6] = 10.0
    target_logits = target_logits.contiguous()
    bonus = torch.tensor([[7]], dtype=torch.int32)

    out = env.rejection.rejection_sample(
        draft, [2], 2, torch.tensor([2]), None, target_logits, bonus, _greedy_metadata(env)
    )
    assert out[0].tolist() == [5, 6, 7]
    # 走的是 host pytorch 回退，不碰 Triton kernel。
    assert "rejection_greedy_sample_with_triton" not in env.rec.calls


def test_rejection_sample_greedy_uses_triton_when_available(env):
    env.set_triton(True)
    draft = torch.tensor([5, 6], dtype=torch.int32)
    target_logits = torch.full((2, 8), -10.0)
    target_logits[0, 5] = 10.0
    target_logits[1, 6] = 10.0
    target_logits = target_logits.contiguous()
    bonus = torch.tensor([[7]], dtype=torch.int32)

    env.rejection.rejection_sample(
        draft, [2], 2, torch.tensor([2]), None, target_logits, bonus, _greedy_metadata(env)
    )
    # HAS_TRITON 可用 → 走昇腾 Triton greedy kernel（记录替身）。
    assert "rejection_greedy_sample_with_triton" in env.rec.calls
