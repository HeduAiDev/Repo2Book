# ch29 主电池二：argmax 不变性二分（WC2/m2）——LogitsProcessors 构造期
# 两列分类 + 三个内置处理器的 apply/is_argmax_invariant 可观察行为。
# 基准（v0.27.1 现核行号）：
#   - vllm/v1/sample/logits_processor/state.py:L148-L166（两列分类容器）
#   - vllm/v1/sample/logits_processor/builtin.py（MinP/LogitBias/MinTokens）
#   - vllm/v1/sample/logits_processor/__init__.py:L185-L218（build_logitsprocs）
# 持久批 update_state 面已按 subtraction_plan delete[5] 删除（归 ch18）——
# 测试按计划 why_safe 自建 state（直接写处理器内部张量）。
from __future__ import annotations

import pytest
import torch

from vllm.v1.sample.logits_processor import (
    LogitsProcessors,
    build_logitsprocs,
)
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class _SeamVllmConfig:
    """HOST SEAM：build_logitsprocs/MinP 构造器消费的字段面
    （vllm_config.speculative_config / scheduler_config.max_num_seqs）。"""

    def __init__(self, max_num_seqs: int = 8, speculative: object = None):
        self.scheduler_config = type("S", (), {"max_num_seqs": max_num_seqs})()
        self.speculative_config = speculative


CPU = torch.device("cpu")


def make_min_p(min_p_value: float) -> MinPLogitsProcessor:
    proc = MinPLogitsProcessor(_SeamVllmConfig(), CPU, False)
    proc.min_p_cpu[0] = min_p_value
    proc.min_p_count = 1
    # 复现 update_state 的张量刷新（builtin.py:L96-L100 的等价终态）
    proc.min_p = proc.min_p_device[:1].unsqueeze(1).clone()
    return proc


# ── 两列分类容器 ──────────────────────────────────────────────────────


def test_two_column_classification_by_is_argmax_invariant():
    """m2：构造期按 is_argmax_invariant() 分两列——结构基础。"""
    mp = make_min_p(0.1)
    lb = LogitBiasLogitsProcessor(None, CPU, False)
    mt = MinTokensLogitsProcessor(None, CPU, False)
    procs = LogitsProcessors([mt, lb, mp])
    assert procs.argmax_invariant == [mp]
    assert procs.non_argmax_invariant == [mt, lb]  # append 顺序保持


def test_all_iterator_chains_both_columns():
    """state.py:L162-L165：all 属性串联两列。"""
    mp = make_min_p(0.1)
    mt = MinTokensLogitsProcessor(None, CPU, False)
    procs = LogitsProcessors([mt, mp])
    assert list(procs.all) == [mp, mt]


def test_is_argmax_invariant_declared_values():
    """三件套的声明（可检测锚点）：MinP=True；MinTokens/LogitBias=False。"""
    assert make_min_p(0.1).is_argmax_invariant() is True
    assert MinTokensLogitsProcessor(None, CPU, False).is_argmax_invariant() is False
    assert LogitBiasLogitsProcessor(None, CPU, False).is_argmax_invariant() is False


def test_interface_is_abstract():
    """LogitsProcessor 是 ABC：apply/is_argmax_invariant 抽象不可直接实例化。"""
    with pytest.raises(TypeError):
        LogitsProcessor(None, CPU, False)  # type: ignore[abstract]


# ── MinP（argmax 不变列唯一内置代表）────────────────────────────────


def test_min_p_threshold_cut_keeps_argmax():
    """m8：阈值 = min_p × max_prob；低于阈值的位砍成 -inf；最高位恒留
    （max_prob ≥ min_p×max_prob 恒成立）→ argmax 不变的数值验证。"""
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    argmax_before = logits.argmax().item()
    proc = make_min_p(0.3)
    out = proc.apply(logits)
    probs = torch.softmax(torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]), dim=-1)
    threshold = 0.3 * probs.max().item()
    expected_masked = {i for i in range(5) if probs[i].item() < threshold}
    masked = {i for i in range(5) if out[0, i].item() == float("-inf")}
    assert masked == expected_masked
    assert argmax_before not in masked  # 最高位必留
    assert out.argmax().item() == argmax_before  # argmax 不变


def test_min_p_zero_count_early_return():
    """min_p_count=0（批内无 min_p 请求）→ 原样返回（builtin.py:L103-104）。"""
    logits = torch.tensor([[4.0, 3.0, 2.0]])
    proc = MinPLogitsProcessor(_SeamVllmConfig(), CPU, False)
    out = proc.apply(logits)
    assert out is logits
    assert torch.equal(out, torch.tensor([[4.0, 3.0, 2.0]]))


def test_get_min_p_by_index_reads_cpu_state():
    proc = MinPLogitsProcessor(_SeamVllmConfig(), CPU, False)
    proc.min_p_cpu[3] = 0.123
    assert proc.get_min_p_by_index(3) == pytest.approx(0.123)


# ── LogitBias（非不变列：稀疏坐标 +=）───────────────────────────────


def test_logit_bias_sparse_add_flips_argmax():
    """L159-162：logits[(req,tok)] += bias——稀疏坐标加，不是全词表加。"""
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
    proc = LogitBiasLogitsProcessor(None, CPU, False)
    proc.biases = {0: {3: 5.0}}
    proc.bias_tensor = torch.tensor([5.0])
    proc.logits_slice = (torch.tensor([0]), torch.tensor([3]))
    out = proc.apply(logits)
    assert out[0, 3].item() == pytest.approx(5.0)  # 0 + 5
    assert out[0, 0].item() == pytest.approx(1.0)  # 其他位不动
    assert out.argmax().item() == 3  # argmax 被 bias 改变 → 非不变列


def test_logit_bias_empty_noop():
    proc = LogitBiasLogitsProcessor(None, CPU, False)
    logits = torch.tensor([[1.0, 2.0]])
    assert torch.equal(proc.apply(logits), logits)


# ── MinTokens（非不变列：封 stop/EOS）───────────────────────────────


def test_min_tokens_masks_stop_tokens():
    """L229-233：未达 min_tokens 的请求把 stop/EOS 的 logit index_put_ -inf。"""
    logits = torch.tensor([[1.0, 2.0, 6.0, 0.5]])  # 6.0 是 EOS
    proc = MinTokensLogitsProcessor(None, CPU, False)
    proc.min_toks = {0: (5, [], {2})}
    proc.logits_slice = (torch.tensor([0]), torch.tensor([2]))
    out = proc.apply(logits)
    assert out[0, 2].item() == float("-inf")
    assert out[0, 1].item() == pytest.approx(2.0)
    assert out.argmax().item() == 1  # argmax 被 censor 改变 → 非不变列


def test_min_tokens_empty_noop():
    proc = MinTokensLogitsProcessor(None, CPU, False)
    logits = torch.tensor([[1.0, 2.0]])
    assert torch.equal(proc.apply(logits), logits)


# ── build_logitsprocs 构造入口 ───────────────────────────────────────


def test_build_logitsprocs_builtin_trio_order():
    """BUILTIN_LOGITS_PROCESSORS = [MinTokens, LogitBias, MinP]
    （__init__.py:L50-L54）→ 两列分类后的落位。"""
    procs = build_logitsprocs(_SeamVllmConfig(), CPU, False, False)
    assert [type(p) for p in procs.non_argmax_invariant] == [
        MinTokensLogitsProcessor, LogitBiasLogitsProcessor]
    assert [type(p) for p in procs.argmax_invariant] == [MinPLogitsProcessor]


def test_build_logitsprocs_pooling_rejects_custom():
    """is_pooling_model + custom → ValueError（STR_POOLING_REJECTS_LOGITSPROCS）。"""
    from vllm.v1.sample.logits_processor import (
        STR_POOLING_REJECTS_LOGITSPROCS,
    )
    with pytest.raises(ValueError, match="Pooling models do not support"):
        build_logitsprocs(_SeamVllmConfig(), CPU, False, True,
                          custom_logitsprocs=["mod:Custom"])
    # 无 custom 的 pooling → 空容器
    procs = build_logitsprocs(_SeamVllmConfig(), CPU, False, True)
    assert procs.argmax_invariant == [] and procs.non_argmax_invariant == []


def test_build_logitsprocs_spec_decode_min_tokens_only():
    """speculative_config 非 None → 只装 MinTokens（min_p/logit_bias 与
    spec 互斥的警告位，__init__.py:L202-210——前指 ch32/33）。"""
    procs = build_logitsprocs(
        _SeamVllmConfig(speculative=object()), CPU, False, False)
    assert [type(p) for p in procs.non_argmax_invariant] == [
        MinTokensLogitsProcessor]
    assert procs.argmax_invariant == []


def test_build_logitsprocs_spec_decode_rejects_custom():
    from vllm.v1.sample.logits_processor import (
        STR_SPEC_DEC_REJECTS_LOGITSPROCS,
    )
    with pytest.raises(ValueError, match="speculative decoding"):
        build_logitsprocs(_SeamVllmConfig(speculative=object()), CPU, False,
                          False, custom_logitsprocs=["mod:Custom"])


def test_min_p_constructor_allocates_max_num_seqs_slots():
    """构造器按 scheduler_config.max_num_seqs 预分配 CPU 槽位（L24-L46）。"""
    proc = MinPLogitsProcessor(_SeamVllmConfig(max_num_seqs=16), CPU, False)
    assert proc.min_p_cpu_tensor.shape == (16,)
    assert proc.min_p_count == 0
    # CPU 设备下不双缓冲（min_p_device 即 cpu tensor，L35-43）
    assert proc.min_p_device is proc.min_p_cpu_tensor
