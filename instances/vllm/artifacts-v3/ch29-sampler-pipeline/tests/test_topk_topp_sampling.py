# ch29 主电池四：top-k/top-p 截断的 pytorch sort 主实现 + 二级分流 +
# Gumbel/exp 掷骰 + 构造期后端绑定与回退。
# 基准（v0.27.1 现核行号）：
#   - vllm/v1/sample/ops/topk_topp_sampler.py:L349-L408（分流 + sort 实现）
#   - vllm/v1/sample/ops/topk_topp_sampler.py:L434-L512（指数噪声三函数 +
#     flashinfer 拒绝采样）
#   - vllm/v1/sample/ops/topk_topp_sampler.py:L77-L182（构造期绑定 + 双后端）
# multinomial 同步之死：L455-L458 docstring 原话。
from __future__ import annotations

import math

import pytest
import torch

import vllm.v1.sample.ops.topk_topp_sampler as tts
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
    empty_exponential_noise_like,
    flashinfer_sample,
    flashinfer_sampler_supported,
    random_sample,
    sample_with_exponential_noise,
)


def _masked_set(row: torch.Tensor) -> set[int]:
    return {i for i in range(row.numel()) if row[i].item() == float("-inf")}


# ── top-k（升序 sort + 第 (V-k) 位阈值 + 严格 <）─────────────────────


def test_top_k_threshold_and_ties():
    """m9：升序 sort 后取第 (V-k) 位值做阈值、严格 < 才 mask——与第 k 名
    并列的 token 都保留（`logits_sort.size(1) - k` 的索引算术）。"""
    logits = torch.tensor([[3.0, 1.0, 2.0, 2.0, 0.5]])
    k = torch.tensor([3], dtype=torch.int32)
    out = apply_top_k_top_p_pytorch(logits, k, None)
    # 升序 [0.5,1.0,2.0,2.0,3.0]；第 (5-3)=2 位 = 2.0；严格 <2.0 的被砍
    assert _masked_set(out[0]) == {1, 4}
    assert out[0, 2].item() == 2.0  # 并列保留
    assert out[0, 3].item() == 2.0
    assert out[0, 0].item() == 3.0


def test_top_k_scatter_back_original_positions():
    """mask 后 scatter 回原位（L407-408）——按原下标核对。"""
    logits = torch.tensor([[0.1, 3.0, 0.2, 2.0, 0.3]])
    out = apply_top_k_top_p_pytorch(logits, torch.tensor([2], dtype=torch.int32), None)
    assert _masked_set(out[0]) == {0, 2, 4}  # 只留 top-2：tokens 1、3


def test_top_k_full_vocab_no_mask():
    k = torch.tensor([5], dtype=torch.int32)
    logits = torch.tensor([[3.0, 1.0, 2.0, 2.0, 0.5]])
    out = apply_top_k_top_p_pytorch(logits, k, None)
    assert _masked_set(out[0]) == set()  # 阈值=最小值，无人 < 它


# ── top-p（nucleus：softmax cumsum ≤ 1-p 反向 mask）─────────────────


def test_top_p_nucleus_smallest_cover():
    """m9：累积概率质量刚超过 p 的最小集合；最末位恒保（at least one）。"""
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]]))
    p = torch.tensor([0.5])
    out = apply_top_k_top_p_pytorch(logits, None, p)
    # 升序 cumsum [0.1,0.3,0.6,1.0]；mask cumsum<=0.5 → 砍 0.1、0.2
    assert _masked_set(out[0]) == {3, 2}
    assert out[0, 0].item() == pytest.approx(math.log(0.4))
    assert out[0, 1].item() == pytest.approx(math.log(0.3))


def test_top_p_at_least_one_kept():
    """top_p_mask[:, -1] = False：极小 p 也至少留 1 个（升序最末=最大位）。"""
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    p = torch.tensor([1e-9])
    out = apply_top_k_top_p_pytorch(logits, None, p)
    assert _masked_set(out[0]) == {0, 2}  # 只留最大位


def test_top_k_and_top_p_applied_sequentially():
    """k 先按 logit 值截、p 再对幸存 k 位生效（docstring 语义）。"""
    logits = torch.log(torch.tensor([[0.45, 0.25, 0.2, 0.1]]))
    out = apply_top_k_top_p_pytorch(
        logits, torch.tensor([3], dtype=torch.int32), torch.tensor([0.6]))
    # top-3 留 0.45/0.25/0.2；对这三者升序 cumsum [0.2,0.45,0.9]，
    # mask cumsum<=0.4 → 砍 0.2 → 最终留 {0.45, 0.25}
    assert _masked_set(out[0]) == {2, 3}


def test_top_p_batch_per_request():
    p = torch.tensor([0.5, 0.95])
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1],
                                     [0.4, 0.3, 0.2, 0.1]]))
    out = apply_top_k_top_p_pytorch(logits, None, p)
    # row0 p=0.5：升序 cumsum [0.1,0.3,0.6,1.0]，mask cumsum<=0.5 → 砍 0.1、0.2
    assert _masked_set(out[0]) == {3, 2}
    # row1 p=0.95：mask cumsum<=0.05，而最小 cumsum 0.1>0.05 → 一个不砍
    assert _masked_set(out[1]) == set()


def test_top_p_boundary_cumsum_strictly_greater():
    """mask 条件是 probs_sum <= 1-p：cumsum 恰等于 1-p 的位也被砍
    （累积质量刚超过 p 的『最小』集合）。"""
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    p = torch.tensor([0.5])
    out = apply_top_k_top_p_pytorch(logits, None, p)
    # 升序 [0.25,0.25,0.5] cumsum [0.25,0.5,1.0]；<=0.5 → 砍前两位
    assert _masked_set(out[0]) == {1, 2}


# ── apply_top_k_top_p 二级分流（m12）────────────────────────────────


def test_routing_no_filters_returns_same_object():
    logits = torch.randn(16, 5)
    assert apply_top_k_top_p(logits, None, None) is logits


def test_routing_small_batch_uses_pytorch_sort():
    """批 < 8 恒走 pytorch sort（L363-364）——即使 HAS_TRITON。"""
    logits = torch.randn(6, 5)
    k = torch.full((6,), 2, dtype=torch.int32)
    with pytest.MonkeyPatch.context() as mp:
        recorded = {}
        real = tts.apply_top_k_top_p_pytorch

        def spy(lobelog, *a, **kw):
            recorded["pytorch"] = True
            return real(lobelog, *a, **kw)

        mp.setattr(tts, "apply_top_k_top_p_pytorch", spy)
        mp.setattr(tts, "HAS_TRITON", True)  # 即使有 Triton
        out = apply_top_k_top_p(logits, k, None)
    assert recorded.get("pytorch")
    assert all(len(_masked_set(r)) == 3 for r in out)


def test_routing_large_batch_with_triton():
    """批 >= 8 且 HAS_TRITON → pivot 截断核（L360-361）。host 上不执行
    真核——以 spy 验证分派决策本身。"""
    logits = torch.randn(16, 5)
    k = torch.full((16,), 2, dtype=torch.int32)
    with pytest.MonkeyPatch.context() as mp:
        recorded = {}
        mp.setattr(
            tts, "apply_top_k_top_p_triton",
            lambda *a, **kw: recorded.setdefault("triton", True) or a[0])
        mp.setattr(tts, "HAS_TRITON", True)
        apply_top_k_top_p(logits, k, None)
    assert recorded.get("triton")

    # HAS_TRITON=False → 同样的批退回 pytorch sort
    with pytest.MonkeyPatch.context() as mp:
        recorded2 = {}
        mp.setattr(
            tts, "apply_top_k_top_p_triton",
            lambda *a, **kw: recorded2.setdefault("triton", True) or a[0])
        mp.setattr(tts, "HAS_TRITON", False)
        real = tts.apply_top_k_top_p_pytorch

        def spy2(x, *a, **kw):
            recorded2["pytorch"] = True
            return real(x, *a, **kw)

        mp.setattr(tts, "apply_top_k_top_p_pytorch", spy2)
        apply_top_k_top_p(logits, k, None)
    assert recorded2.get("pytorch") and "triton" not in recorded2


# ── Gumbel/exp 掷骰（multinomial 同步之死）─────────────────────────


def test_sample_with_exponential_noise_deterministic_math():
    """m10：argmax(p/q) 手算对表。p=[0.6,0.3,0.1]。"""
    probs = torch.tensor([[0.6, 0.3, 0.1]])
    q = torch.tensor([[0.5, 0.9, 0.2]])
    # p/q = [1.2, 0.333, 0.5] → argmax 0
    assert sample_with_exponential_noise(probs, q).tolist() == [0]
    probs2 = torch.tensor([[0.6, 0.3, 0.1]])
    q2 = torch.tensor([[5.0, 5.0, 0.01]])
    # p/q = [0.12, 0.06, 10.0] → argmax 2（小概率 token 也能赢——按概率抽样的本质）
    assert sample_with_exponential_noise(probs2, q2).tolist() == [2]


def test_sample_with_exponential_noise_fp64_avoids_mutating_probs():
    """fp64 分支（q.reciprocal_() 再 mul_）：避免把 probs 除成 0；
    同 dtype 分支是 probs.div_(q)（原地）。"""
    probs = torch.tensor([[0.6, 0.3, 0.1]])
    q = torch.tensor([[0.9, 0.5, 2.0]], dtype=torch.float64)
    before = probs.clone()
    tok = sample_with_exponential_noise(probs, q)
    assert torch.equal(probs, before)  # fp64 路不改 probs
    # p/q 手算：[0.667, 0.6, 0.05] → argmax 0
    assert tok.tolist() == [0]

    probs32 = torch.tensor([[0.6, 0.3, 0.1]])
    q32 = torch.tensor([[0.9, 0.5, 2.0]])
    sample_with_exponential_noise(probs32, q32)
    assert not torch.equal(probs32, torch.tensor([[0.6, 0.3, 0.1]]))  # 原地 div


def test_empty_exponential_noise_dtype():
    probs = torch.zeros(2, 3, dtype=torch.float32)
    assert empty_exponential_noise_like(probs, False).dtype == torch.float32
    assert empty_exponential_noise_like(probs, True).dtype == torch.float64


def test_random_sample_matches_categorical_distribution():
    """m10 统计等价：argmax(p/q) 与按 p 抽样同分布（Gumbel-max 定理）。
    注意 sample_with_exponential_noise 是 probs.div_(q) 原地——真实调用方
    （forward_native）每步喂新 softmax，测试同构。"""
    torch.manual_seed(42)
    n = 4000
    counts = [0, 0, 0]
    for _ in range(n):
        probs = torch.tensor([[0.6, 0.3, 0.1]])
        counts[random_sample(probs, {}, False).item()] += 1
    for i, expected in enumerate([0.6, 0.3, 0.1]):
        assert abs(counts[i] / n - expected) < 0.05, (i, counts)


def test_random_sample_generator_reproducible():
    """有 per-request seed：逐请求 generator 覆写（L467-471，慢路自认 TODO）。
    同 seed 两次调用结果一致。"""
    def draw():
        probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        gen = torch.Generator().manual_seed(1234)
        return random_sample(probs, {0: gen}, False).tolist()

    assert draw() == draw()


def test_random_sample_docstring_contract():
    """random_sample 可直接调用且返回 int64 一维（argmax.view(-1)）。"""
    torch.manual_seed(0)
    probs = torch.tensor([[0.99, 0.005, 0.005]])
    out = random_sample(probs, {}, False)
    assert out.dtype == torch.int64
    assert out.shape == (1,)


# ── forward_native / forward_cuda（构造期绑定与回退）────────────────


def test_forward_native_returns_token_and_mode_views():
    """forward_native：截断→softmax→random_sample；processed 两态返回
    截断后的 logits/logprobs 视角（L143-148）。"""
    torch.manual_seed(7)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    k = torch.tensor([2, 2], dtype=torch.int32)
    sampler = TopKTopPSampler("raw_logprobs", False)
    toks, extra = sampler.forward_native(logits, {}, k, None)
    assert toks.shape == (2,)
    assert extra is None
    # raw 模式不给 processed 视角
    s2 = TopKTopPSampler("processed_logits", False)
    _, extra2 = s2.forward_native(torch.tensor([[3.0, 2.0, 1.0]]), {},
                                  torch.tensor([2], dtype=torch.int32), None)
    assert extra2 is not None
    assert _masked_set(extra2[0]) == {2}  # top-2 之外的位 -inf
    s3 = TopKTopPSampler("processed_logprobs", False)
    _, extra3 = s3.forward_native(torch.tensor([[3.0, 2.0, 1.0]]), {},
                                  torch.tensor([2], dtype=torch.int32), None)
    # log_softmax 保持排序：logprob(3.0) > logprob(2.0) > logprob(-inf 位)
    assert extra3[0, 0].item() > extra3[0, 1].item() > extra3[0, 2].item()


def test_forward_cuda_fallbacks_on_host():
    """forward_cuda 的三个回退（L166-175）：无 k/p、有 per-request
    generator、use_fp64_gumbel——都退 forward_native（host CPU 可验证）。"""
    torch.manual_seed(3)
    sampler = TopKTopPSampler("raw_logprobs", False)
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    # (a) k、p 全 None → 无事可做 → native
    toks, extra = sampler.forward_cuda(logits, {}, None, None)
    assert toks.shape == (1,)
    assert extra is None
    # (b) 有 generator → FlashInfer 0.2.3+ 不支持 → native
    gen = torch.Generator().manual_seed(9)
    toks2, _ = sampler.forward_cuda(logits, {0: gen}, None, None)
    assert toks2.shape == (1,)
    # (c) fp64 gumbel → native
    s64 = TopKTopPSampler("raw_logprobs", True)
    toks3, _ = s64.forward_cuda(logits, {}, None, None)
    assert toks3.shape == (1,)


def test_processed_modes_constant():
    """PROCESSED_LOGPROBS_MODES = 两态（config/model.py:L102-L105）——
    FlashInfer 强制 native 的判据。"""
    assert set(PROCESSED_LOGPROBS_MODES) == {"processed_logits",
                                             "processed_logprobs"}


def test_construction_binds_only_on_cuda():
    """m11 + delete[4]：减法后 self.forward 仅 is_cuda() 下绑定。
    host（CUDA 可见）+ 显式禁用 flashinfer → 绑 forward_native（真实
    L100-L102 的 else 位）；无 CUDA → 不绑定（测试用 seam 复现绑定支路）。"""
    if torch.cuda.is_available():
        s = TopKTopPSampler("raw_logprobs", False)
        assert s.forward.__name__ == "forward_native"
        # processed 两态强制 native（L96-99）
        sp = TopKTopPSampler("processed_logits", False)
        assert sp.forward.__name__ == "forward_native"
    else:
        s = TopKTopPSampler("raw_logprobs", False)
        assert "forward" not in s.__dict__  # 非 CUDA 不绑定（减法后控制流）


def test_construction_binds_forward_cuda_when_enabled(monkeypatch):
    """m11 正支路：env 默认开（conftest 的 0 被 delenv 撤掉）+ 能力裁决
    通过 → 绑 forward_cuda。构造期只做能力裁决、不 import flashinfer
    本体（import 延迟到 flashinfer_sample 内）；本机能力不在 SM80-SM121
    支持域时按真实回退语义绑 native。"""
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    from vllm.platforms import DeviceCapability, current_platform

    s = TopKTopPSampler("raw_logprobs", False)
    cap = current_platform.get_device_capability()
    in_support = (
        cap is not None
        and DeviceCapability(8, 0) <= cap <= DeviceCapability(12, 1)
    )
    if in_support:
        assert s.forward.__name__ == "forward_cuda"
    else:
        assert s.forward.__name__ == "forward_native"


def test_flashinfer_sampler_supported_disabled_by_env():
    """显式禁用支路（L45-50）：env=0 → False（info_once 后返回）。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 才能走到 env 裁决（L43-44 非 CUDA 早退）")
    import os

    old = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    try:
        assert flashinfer_sampler_supported() is False
    finally:
        if old is None:
            os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        else:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = old


def test_flashinfer_sampler_supported_non_cuda_false():
    """非 CUDA 平台恒 False（L43-44）——monkeypatch 平台 seam 验证分支。"""

    class _NonCuda:
        @classmethod
        def is_cuda(cls):
            return False

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts, "current_platform", _NonCuda)
        assert flashinfer_sampler_supported() is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="容器/CUDA 专属")
def test_flashinfer_sample_three_api_branches():
    """flashinfer_sample 三条 API 分发（L475-512）——真 GPU 容器跑：
    k-only / p-only / both 各出合法 token 且落在截断支撑集内。
    注：deterministic=True 不保证两次调用逐位相同（拒绝采样仍消费
    RNG）——契约是 docstring 自述的 statistically equivalent。"""
    pytest.importorskip("flashinfer")
    torch.manual_seed(11)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]],
                          device="cuda")
    k = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    p = torch.tensor([0.9, 0.9], device="cuda")

    tok_k = flashinfer_sample(logits, k, None)
    # top-k=2 支撑集：row0 ∈ {0,1}、row1 ∈ {2,3}
    assert tok_k[0].item() in (0, 1)
    assert tok_k[1].item() in (2, 3)
    tok_p = flashinfer_sample(logits, None, p)
    assert all(0 <= t <= 3 for t in tok_p.cpu().tolist())
    tok_kp = flashinfer_sample(logits, k, p)
    assert tok_kp[0].item() in (0, 1)
    assert tok_kp[1].item() in (2, 3)
