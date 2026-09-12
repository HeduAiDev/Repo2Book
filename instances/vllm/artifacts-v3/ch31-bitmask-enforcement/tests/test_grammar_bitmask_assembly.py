# ch31 主电池一：StructuredOutputManager.grammar_bitmask 批装配
# （vllm/v1/structured_output/__init__.py:L212-L359）的真实可观察行为。
# 覆盖：预算分配（m04）/并行分支（m06）/串行 spec 窗口（m07）/整行 -1 语义
# （m08）/裁剪与 .numpy() 出发（m09）/跨步复用残留清理。
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import FakeGrammar, FakeReasoner, FakeRequest, make_manager


V = 64  # 词表替身：2 列 int32


def bits_to_row(allowed: set[int]) -> np.ndarray:
    row = np.zeros(V // 32, dtype=np.uint32)
    for tok in allowed:
        row[tok // 32] |= np.uint32(1 << (tok % 32))
    return row.view(np.int32)


class TestBudget:
    """m04：max_num_seqs*(1+num_spec) 行预算，首次惰性分配、跨步复用。"""

    def test_budget_rows_max_seqs_times_one_plus_spec(self):
        mgr = make_manager(V, max_num_seqs=7, num_speculative_tokens=3)
        g = FakeGrammar(V, allowed={1})
        req = FakeRequest("r1", grammar=g)
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        assert mgr._grammar_bitmask.shape[0] == 7 * (1 + 3)
        assert mgr.backend.alloc_calls == [7 * (1 + 3)]

    def test_buffer_reused_across_steps_single_allocation(self):
        mgr = make_manager(V, max_num_seqs=8)
        g = FakeGrammar(V, allowed={1})
        req = FakeRequest("r1", grammar=g)
        mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        assert mgr.backend.alloc_calls == [8]  # 只分配一次

    def test_empty_request_list_fast_return_none(self):
        mgr = make_manager(V)
        assert mgr.grammar_bitmask({}, [], {}) is None


class TestSerialFill:
    """m07 串行分支（小批量/带 spec 时的正路）。"""

    def test_non_spec_request_gets_exactly_bonus_row(self):
        # 无 spec：每请求恰 1 行（bonus/非投机行）
        mgr = make_manager(V)
        g = FakeGrammar(V, allowed={5, 7})
        req = FakeRequest("r1", grammar=g)
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int32
        assert out.shape == (1, V // 32)
        assert np.array_equal(out[0], bits_to_row({5, 7}))
        # fill 进的是跨步复用整块缓冲的第 0 行（缓冲行数 = max_num_seqs，非裁剪后）
        assert [i for _, i in g.fill_calls] == [0]

    def test_spec_window_rows_and_probe_rollback(self):
        # 3 草稿 → 4 行（3 spec 位 + 1 bonus）；试探推进 3 次后整体 rollback(3)
        mgr = make_manager(V, num_speculative_tokens=3)
        g = FakeGrammar(V, allowed=set(range(V)))
        req = FakeRequest("r1", grammar=g)
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [10, 11, 12]})
        assert out.shape[0] == 4
        assert [t for _, t in g.accepts] == [[10], [11], [12]]
        assert g.rollbacks == [3]

    def test_rows_in_request_order(self):
        mgr = make_manager(V)
        g1 = FakeGrammar(V, allowed={1})
        g2 = FakeGrammar(V, allowed={2})
        r1 = FakeRequest("r1", grammar=g1)
        r2 = FakeRequest("r2", grammar=g2)
        out = mgr.grammar_bitmask(
            {"r1": r1, "r2": r2}, ["r1", "r2"], {}
        )
        assert np.array_equal(out[0], bits_to_row({1}))
        assert np.array_equal(out[1], bits_to_row({2}))

    def test_sentinel_minus_one_fills_old_flag_stops_advance_frees_rest(self):
        # -1 哨兵（草稿被 validate 过滤后的补齐位）：当行按旧标志填、
        # 当行即停推进、自次行起整行放行（L298 填行在 L300-L302 翻转之前）
        mgr = make_manager(V, num_speculative_tokens=3)
        g = FakeGrammar(V, allowed={3})
        req = FakeRequest("r1", grammar=g)
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [10, -1, 12]})
        # 行 0：t=10 → 受约束（grammar 写行）
        assert np.array_equal(out[0], bits_to_row({3}))
        # 行 1：t=-1 → 填行发生在翻转之前，仍按旧标志（受约束）
        assert np.array_equal(out[1], bits_to_row({3}))
        # 行 2：t=12 → 翻转已发生 → 整行 -1 放行
        assert np.array_equal(out[2], np.full(V // 32, -1, dtype=np.int32))
        # bonus 行：should_fill_bitmask 仍 True（无 reasoner）→ 受约束
        assert np.array_equal(out[3], bits_to_row({3}))
        # 推进只发生在 t=10；t=12 因 advance_grammar=False 未推进
        assert [t for _, t in g.accepts] == [[10]]

    def test_rejected_draft_raises_assertion_error(self):
        # 非 post_reasoning_end 场景，accept 被拒 = 调度与语法状态失配 → AssertionError
        mgr = make_manager(V, num_speculative_tokens=1)
        g = FakeGrammar(V, allowed={1}, reject_tokens={10})
        req = FakeRequest("r1", grammar=g)
        with pytest.raises(AssertionError):
            mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [10]})

    def test_diffusion_no_bonus_row_when_drafts_present(self):
        # Diffusion LLM 不采 bonus token：有 spec 行时跳过 bonus 行
        mgr = make_manager(V, num_speculative_tokens=2, is_diffusion=True)
        g = FakeGrammar(V, allowed={1})
        req = FakeRequest("r1", grammar=g)
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [10, 11]})
        assert out.shape[0] == 2  # 仅 2 个 spec 行，无 bonus 行

    def test_trim_to_cumulative_index_active_prefix(self):
        # 出发前裁到 cumulative_index 行：只传活跃前缀（m09）
        mgr = make_manager(V, max_num_seqs=8)
        g1 = FakeGrammar(V, allowed={1})
        g2 = FakeGrammar(V, allowed={2})
        r1 = FakeRequest("r1", grammar=g1)
        r2 = FakeRequest("r2", grammar=g2)
        out = mgr.grammar_bitmask({"r1": r1, "r2": r2}, ["r1", "r2"], {})
        assert out.shape[0] == 2  # 缓冲 8 行，只传 2 行
        # .numpy()：ndarray（序列化效率）而非 tensor
        assert isinstance(out, np.ndarray)


class TestFillRowSemantics:
    """m08：单行语义——受约束行写合法表；不受约束/终态行整行 -1（全允许）。"""

    def test_unconstrained_request_full_row_allow_all(self):
        mgr = make_manager(V)
        g = FakeGrammar(V, allowed={5})  # grammar 说只允许 5
        req = FakeRequest("r1", grammar=g)
        mgr.reasoner_cls = FakeReasoner  # 注入 reasoner（装配面已删——同款注入）
        mgr.tokenizer = None
        req.structured_output_request.reasoning_ended = False  # 思考未结束 → 不填
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        assert np.array_equal(out[0], np.full(V // 32, -1, dtype=np.int32))
        assert g.fill_calls == []  # grammar 未被调

    def test_terminated_grammar_full_row_allow_all(self):
        mgr = make_manager(V)
        g = FakeGrammar(V, allowed={5}, terminated=True)
        req = FakeRequest("r1", grammar=g)
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        assert np.array_equal(out[0], np.full(V // 32, -1, dtype=np.int32))

    def test_reused_buffer_residue_cleared(self):
        # 跨步复用缓冲：上拍受约束（写位表），本拍不受约束 → 该行必须被
        # 显式 fill(-1) 重置——残留位会误杀本拍合法 token（正确性必需）
        mgr = make_manager(V)
        g = FakeGrammar(V, allowed={5})
        req = FakeRequest("r1", grammar=g)
        out1 = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        assert np.array_equal(out1[0], bits_to_row({5}))
        # 本拍思考未结束（should_fill=False——注入 reasoner 后判 False）
        mgr.reasoner_cls = FakeReasoner
        mgr.tokenizer = None
        req.structured_output_request.reasoning_ended = False
        out2 = mgr.grammar_bitmask({"r1": req}, ["r1"], {})
        assert np.array_equal(out2[0], np.full(V // 32, -1, dtype=np.int32))


class TestParallelFill:
    """m06：>128 且无 spec 才并行；16 个一任务；行间独写。"""

    def test_parallel_threshold_and_chunking(self):
        mgr = make_manager(V, max_num_seqs=256)
        assert mgr.fill_bitmask_parallel_threshold == 128
        assert mgr.fill_bitmask_parallel_batch_size == 16

        requests, grammars = {}, []
        for i in range(129):
            g = FakeGrammar(V, allowed={i % V})
            grammars.append(g)
            requests[f"r{i}"] = FakeRequest(f"r{i}", grammar=g)

        submissions = []

        def fake_submit(batch):
            submissions.append(list(batch))
            mgr._fill_bitmasks(batch)  # 同步执行（真实由线程池异步跑同一函数）
            from concurrent.futures import Future
            fut = Future()
            fut.set_result(None)
            return fut

        mgr._async_submit_fill_bitmask = fake_submit
        out = mgr.grammar_bitmask(requests, list(requests.keys()), {})

        assert len(submissions) == 9  # 129 = 8*16 + 1 → 8 满批 + 1 尾批
        assert all(len(b) == 16 for b in submissions[:8])
        assert len(submissions[-1]) == 1
        assert out.shape[0] == 129  # 每请求恰 1 行
        # 每请求的行号与其在列表中的序号一致
        for i, g in enumerate(grammars):
            assert g.fill_calls[0][1] == i

    def test_serial_when_at_threshold(self):
        # 恰 128（不 > 128）→ 串行
        mgr = make_manager(V, max_num_seqs=256)
        requests = {
            f"r{i}": FakeRequest(f"r{i}", grammar=FakeGrammar(V, allowed={i % V}))
            for i in range(128)
        }
        called = []

        def fake_submit(batch):
            called.append(len(batch))
            raise AssertionError("serial path must not touch executor")

        mgr._async_submit_fill_bitmask = fake_submit
        out = mgr.grammar_bitmask(requests, list(requests.keys()), {})
        assert out.shape[0] == 128

    def test_serial_when_spec_tokens(self):
        # 带 spec → 串行（跨行有顺序依赖：同 grammar 推进/回滚）
        mgr = make_manager(V, max_num_seqs=256, num_speculative_tokens=1)
        requests = {
            f"r{i}": FakeRequest(f"r{i}", grammar=FakeGrammar(V, allowed={i % V}))
            for i in range(200)
        }

        def fake_submit(batch):
            raise AssertionError("spec path must not use executor")

        mgr._async_submit_fill_bitmask = fake_submit
        out = mgr.grammar_bitmask(
            requests, list(requests.keys()),
            {rid: [7] for rid in requests},
        )
        assert out.shape[0] == 200 * 2  # 每请求 1 spec 行 + 1 bonus 行

    def test_no_executor_when_max_seqs_le_threshold(self):
        # 结构性前提：max_num_seqs ≤ 128 不建池——小部署并行分支是死代码
        mgr = make_manager(V, max_num_seqs=128)
        assert not hasattr(mgr, "executor_for_fillmask")
        mgr2 = make_manager(V, max_num_seqs=129)
        assert hasattr(mgr2, "executor_for_fillmask")


class TestRealThreadPoolPath:
    """真线程池冒烟：129 请求走真 executor_for_fillmask，行间独写并行安全。"""

    def test_real_pool_fills_all_rows(self):
        mgr = make_manager(V, max_num_seqs=256)
        requests = {
            f"r{i}": FakeRequest(f"r{i}", grammar=FakeGrammar(V, allowed={i % V}))
            for i in range(150)
        }
        out = mgr.grammar_bitmask(requests, list(requests.keys()), {})
        assert out.shape[0] == 150
        for i in range(150):
            assert np.array_equal(out[i], bits_to_row({i % V}))
