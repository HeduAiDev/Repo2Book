# ch31 主电池四：apply_grammar_bitmask（V1 正典落地，本章 payoff）
# 基准：vllm/v1/structured_output/utils.py:L86-L175。
# 覆盖：行序重排算术（m10，dossier worked example）→ pinned sorted_bitmask →
# H2D 链 + xgr.apply_token_bitmask_inplace（m11）→ skip_out_indices 快路径 →
# bit=0 → -inf 位语义（m12）。
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import FakeGrammarOutput, FakeSchedulerOutput, cdiv


V = 64


class FakeInputBatch:
    def __init__(self, req_ids):
        self.req_ids = list(req_ids)


class FakeXgr:
    """xgr.apply_token_bitmask_inplace 的 torch 参考实现。

    位语义与索引语义均按 xgrammar 0.2.6 的 apply_token_bitmask_inplace_
    kernel_indices_torch 逐字对齐：bitmask 与 logits 同形（行位 = logits 行位），
    indices 选择要应用的行——`bitmask[indices]` 的位=0 → 该行 -inf。
    （V2 Triton kernel 的 (bit&1)==0 → store(-inf) 是同一语义的紧凑行版。）"""

    def __init__(self):
        self.calls = []

    def apply_token_bitmask_inplace(self, logits, bitmask, indices=None):
        self.calls.append((logits, bitmask, indices))
        if indices is None:
            indices = torch.arange(logits.shape[0], device=logits.device)
        elif torch.is_tensor(indices):
            indices = indices.long()
        bits = bitmask[indices].to(torch.int64)
        unpacked = ((bits[:, :, None] >> torch.arange(32, device=bits.device)) & 1) == 0
        # unpacked: [len(indices), ceil(V/32), 32] → 展平截到词表长
        mask = unpacked.reshape(bits.shape[0], -1)[:, : logits.shape[-1]]
        logits[indices] = torch.where(
            mask, torch.tensor(float("-inf"), dtype=logits.dtype), logits[indices]
        )


def unique_rows(n, cols):
    """每行一个可区分位模式：第 i 行只允许 token i*3。"""
    rows = np.zeros((n, cols), dtype=np.uint32)
    for i in range(n):
        tok = (i * 3) % V
        rows[i, tok // 32] |= np.uint32(1 << (tok % 32))
    return rows.view(np.int32)


@pytest.fixture
def patched_xgr(monkeypatch):
    import vllm.v1.structured_output.utils as u

    fake = FakeXgr()
    monkeypatch.setattr(u, "xgr", fake)
    return fake


class TestReorderArithmetic:
    """m10：掩码行序=调度序、不保证=worker 批序——重排的行算术
    （dossier worked example：reqA 无草稿、reqB 带 3 草稿、批序 [B, A]）。"""

    def test_worked_example_out_indices(self, patched_xgr):
        # 调度序（掩码行序）[A, B]：A 占 1 行、B 占 4 行（3 草稿+bonus）= 5 行
        go = FakeGrammarOutput(["A", "B"], unique_rows(5, V // 32))
        so = FakeSchedulerOutput(
            scheduled_spec_decode_tokens={"B": [7, 8, 9]}
        )
        # worker 批序 [B, A]（≠调度序）
        ib = FakeInputBatch(["B", "A"])
        logits = torch.zeros((5, V), device="meta", dtype=torch.float32)

        from vllm.v1.structured_output.utils import apply_grammar_bitmask

        apply_grammar_bitmask(so, go, ib, logits)

        got_logits, got_bitmask, indices = patched_xgr.calls[0]
        assert tuple(got_logits.shape) == (5, V)
        assert tuple(got_bitmask.shape) == (5, V // 32)
        # 行数对得上 logits（5 行全被语法请求占满）→ 快路径：indices=None 免传
        assert indices is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="值校验需真设备")
    def test_worked_example_reorder_values(self, patched_xgr):
        # 重排正确性：worker 批序 [B, A] → B 的 4 行（掩码行 1..4）进 logits 行
        # 0..3（批序 0 + offset 0），A 的 1 行（掩码行 0）进 logits 行 4（1+3）。
        go = FakeGrammarOutput(["A", "B"], unique_rows(5, V // 32))
        so = FakeSchedulerOutput(
            scheduled_spec_decode_tokens={"B": [7, 8, 9]}
        )
        ib = FakeInputBatch(["B", "A"])
        logits = torch.zeros((5, V), device="cuda", dtype=torch.float32)

        from vllm.v1.structured_output.utils import apply_grammar_bitmask

        apply_grammar_bitmask(so, go, ib, logits)
        got_bitmask = patched_xgr.calls[0][1]
        mask = unique_rows(5, V // 32)  # 掩码行 i 允许 token i*3
        expected = np.stack([mask[1], mask[2], mask[3], mask[4], mask[0]])
        assert np.array_equal(got_bitmask.cpu().numpy(), expected)
        assert got_bitmask.is_cuda

    def test_partial_coverage_passes_indices(self, patched_xgr):
        # 批里混一个非语法请求 → 掩码行数 < logits 行数 → 传 indices
        go = FakeGrammarOutput(["A"], unique_rows(1, V // 32))
        so = FakeSchedulerOutput()
        ib = FakeInputBatch(["B", "A"])  # B 不受约束
        logits = torch.zeros((2, V), device="meta", dtype=torch.float32)

        from vllm.v1.structured_output.utils import apply_grammar_bitmask

        apply_grammar_bitmask(so, go, ib, logits)
        _, got_bitmask, indices = patched_xgr.calls[0]
        assert tuple(got_bitmask.shape) == (2, V // 32)  # sorted 与 logits 同形
        assert indices is not None
        assert indices.dtype == torch.int32
        assert tuple(indices.shape) == (1,)  # 只掩 A 所在的 1 个 logits 行

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="值校验需真设备")
    def test_non_grammar_rows_full_allow(self, patched_xgr):
        # 非语法行保持 sorted_bitmask 初始 -1（全允许）
        go = FakeGrammarOutput(["A"], unique_rows(1, V // 32))
        so = FakeSchedulerOutput()
        ib = FakeInputBatch(["B", "A"])
        logits = torch.zeros((2, V), device="cuda", dtype=torch.float32)

        from vllm.v1.structured_output.utils import apply_grammar_bitmask

        apply_grammar_bitmask(so, go, ib, logits)
        got = patched_xgr.calls[0][1].cpu().numpy()
        assert np.array_equal(got[0], np.full(V // 32, -1, dtype=np.int32))


class TestGpuApplySemantics:
    """m11/m12：H2D 链 + bit=0 → -inf（真 CUDA 或容器路径）。"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_bit_zero_becomes_neg_inf_on_cuda(self):
        pytest.importorskip("xgrammar", reason="真 xgr 落地 kernel（容器内）")
        import xgrammar as xgr

        go = FakeGrammarOutput(["A"], unique_rows(1, V // 32))
        so = FakeSchedulerOutput()
        ib = FakeInputBatch(["A"])
        logits = torch.zeros((1, V), device="cuda", dtype=torch.float32)

        from vllm.v1.structured_output.utils import apply_grammar_bitmask

        apply_grammar_bitmask(so, go, ib, logits)
        allowed = {(0 * 3) % V}
        for tok in range(V):
            if tok in allowed:
                assert logits[0, tok].item() == 0.0
            else:
                assert logits[0, tok].item() == float("-inf")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_reference_semantics_with_fake_kernel(self, patched_xgr):
        # 无真实 kernel 的路径：FakeXgr 参考实现（xgr 0.2.6 torch kernel 同款
        # bitmask[indices] 语义）验证两侧张量搬运与行配对
        go = FakeGrammarOutput(["A", "B"], unique_rows(5, V // 32))
        so = FakeSchedulerOutput(scheduled_spec_decode_tokens={"B": [7, 8, 9]})
        ib = FakeInputBatch(["B", "A"])
        logits = torch.zeros((5, V), device="cuda", dtype=torch.float32)

        from vllm.v1.structured_output.utils import apply_grammar_bitmask

        apply_grammar_bitmask(so, go, ib, logits)
        assert logits.is_cuda
        # B 的 4 行（掩码行 1..4）重排进 logits 行 0..3，A 的行（掩码行 0）进行 4：
        # sorted 行 i 允许的恰是掩码行 (i+1)%5 允许的 token ((i+1)*3)%V
        for row in range(5):
            src_row = (row + 1) % 5  # sorted[row] = mask[(row+1)%5]
            allowed = {(src_row * 3) % V}
            for tok in range(V):
                if tok in allowed:
                    assert logits[row, tok].item() == 0.0
                else:
                    assert logits[row, tok].item() == float("-inf")


class TestNdarrayInput:
    """m09 出发端的呼应：收到的必须是 ndarray（序列化效率注释的物证）。"""

    def test_receives_ndarray_bitmask(self, patched_xgr):
        go = FakeGrammarOutput(["A"], unique_rows(1, V // 32))
        assert isinstance(go.grammar_bitmask, np.ndarray)
        assert go.grammar_bitmask.dtype == np.int32
        assert patched_xgr.calls == []  # （构造即断言型：调用前不误触）
