# ch31 主电池二：思考门控三件套（m14）+ 窗口内思考结束检测（m15）
# （vllm/v1/structured_output/__init__.py:L285-L323/L361-L486，
#  #42452/#43388/#44006 修复行为的守护测试）。
from __future__ import annotations

import numpy as np
import torch

from conftest import FakeGrammar, FakeReasoner, FakeRequest, make_manager


V = 64


def bits_to_row(allowed):
    row = np.zeros(V // 32, dtype=np.uint32)
    for tok in allowed:
        row[tok // 32] |= np.uint32(1 << (tok % 32))
    return row.view(np.int32)


def attach_reasoner(mgr, req, **behavior):
    """注入请求级 reasoner（真实装配在 __init__ L81-L93，编译侧已删——
    精简版按 dossier『直接注入已构造对象』的同一口径注入）。"""
    mgr.reasoner_cls = FakeReasoner
    mgr.tokenizer = None
    reasoner = FakeReasoner(**behavior)
    req.structured_output_request.reasoner = reasoner
    return reasoner


class TestShouldFillBitmask:
    """门控之一：这一步填不填掩码（L361-L379）。"""

    def test_no_reasoner_always_fill(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        assert mgr.should_fill_bitmask(req) is True

    def test_reasoning_not_ended_no_fill(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        r = attach_reasoner(mgr, req)
        r._prompt_end = False
        req.structured_output_request.reasoning_ended = False
        assert mgr.should_fill_bitmask(req) is False

    def test_prompt_check_cached_on_first_call(self):
        # reasoning_ended 未定时先对 prompt 判一次并缓存进请求（只判一次）
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                          prompt_token_ids=[1, 2, 3])
        r = attach_reasoner(mgr, req)
        r._prompt_end = False
        req.structured_output_request.reasoning_ended = None
        assert mgr.should_fill_bitmask(req) is False
        assert mgr.should_fill_bitmask(req) is False
        assert r.prompt_end_calls == [[1, 2, 3]]  # 只调一次，缓存生效

    def test_reasoning_ended_persisted_true(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        attach_reasoner(mgr, req)
        req.structured_output_request.reasoning_ended = True
        assert mgr.should_fill_bitmask(req) is True

    def test_enable_in_reasoning_forces_fill(self):
        # 把约束也加到思考段的开关（覆盖默认判据）
        mgr = make_manager(V, enable_in_reasoning=True)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        attach_reasoner(mgr, req)
        req.structured_output_request.reasoning_ended = False
        assert mgr.should_fill_bitmask(req) is True


class TestShouldAdvance:
    """门控之二：这一步推不推进 FSM（L381-L439，v0.27 new_token_ids 精确窗口）。"""

    def test_not_structured_output_never_advance(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", use_structured_output=False)
        assert mgr.should_advance(req) is False

    def test_no_reasoner_always_advance(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        assert mgr.should_advance(req) is True

    def test_reasoning_not_ended_no_advance(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        attach_reasoner(mgr, req)
        req.structured_output_request.reasoning_ended = False
        assert mgr.should_advance(req) is False

    def test_reasoning_ended_advances(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        attach_reasoner(mgr, req)
        req.structured_output_request.reasoning_ended = True
        assert mgr.should_advance(req) is True

    def test_new_token_ids_window_detects_end_and_records_boundary(self):
        # 本步刚追加的 token 就是精确 delta 窗口；检出结束→置位+记边界+放行推进
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                          all_token_ids=list(range(10)))  # 本步追加 [7,8,9]
        r = attach_reasoner(mgr, req)
        r._streaming = lambda all_len, delta: 9 in delta
        req.structured_output_request.reasoning_ended = False

        assert mgr.should_advance(req, new_token_ids=[7, 8, 9]) is True
        assert req.structured_output_request.reasoning_ended is True
        assert req.structured_output_request.reasoning_end_token_index == 9
        # streaming 首调用收到的 all_token_ids 恰为完整历史、delta 恰为本步窗口；
        # 随后 _find_reasoning_end_index 逐 token 定位边界（idx 7→9 各一次）
        assert r.streaming_calls[0] == (10, [7, 8, 9])
        assert [d for _, d in r.streaming_calls[1:]] == [[7], [8], [9]]

    def test_placeholder_derived_window_fallback(self):
        # new_token_ids=None → 从 num_computed_tokens - num_output_placeholders 推窗口
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                          all_token_ids=list(range(10)),
                          num_computed_tokens=10, num_output_placeholders=3)
        r = attach_reasoner(mgr, req)
        r._streaming = lambda all_len, delta: 9 in delta
        req.structured_output_request.reasoning_ended = False

        assert mgr.should_advance(req) is True  # delta = islice(all, 7, None) = [7,8,9]
        assert req.structured_output_request.reasoning_ended is True

    def test_rejected_drafts_do_not_break_new_token_ids_window(self):
        # #43388：async+spec 草稿被拒后 placeholders 仍 > 0——显式 new_token_ids
        # 窗口不再依赖 placeholders 推导
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                          all_token_ids=list(range(10)),
                          num_output_placeholders=2)  # 卡住的占位数
        r = attach_reasoner(mgr, req)
        r._streaming = lambda all_len, delta: 9 in delta
        req.structured_output_request.reasoning_ended = False
        assert mgr.should_advance(req, new_token_ids=[8, 9]) is True

    def test_enable_in_reasoning_advances_in_reasoning(self):
        mgr = make_manager(V, enable_in_reasoning=True)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        attach_reasoner(mgr, req)
        req.structured_output_request.reasoning_ended = False
        assert mgr.should_advance(req) is True


class TestFindReasoningEndIndex:
    """边界定位（L441-L460）：逐 token 定位；多 token 标记保守回退末位。"""

    def test_single_token_marker_located(self):
        mgr = make_manager(V)
        r = FakeReasoner()
        r._streaming = lambda all_len, delta: delta == [7]
        toks = list(range(10))
        idx = mgr._find_reasoning_end_index(r, toks, start=3)
        assert idx == 7

    def test_multi_token_marker_falls_back_to_last(self):
        # 多 token 标记只在整段 delta 上才认出 → 保守把整步当思考内容
        mgr = make_manager(V)
        r = FakeReasoner()  # 逐 token 永不触发
        toks = list(range(10))
        assert mgr._find_reasoning_end_index(r, toks, start=3) == 9


class TestTrimReasoningForAdvance:
    """#44006 修复物：裁掉混块里的思考 token 再喂 accept_tokens（L462-L486）。"""

    def test_no_structured_request_unchanged(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", use_structured_output=False)
        assert mgr.trim_reasoning_for_advance(req, [1, 2]) == [1, 2]

    def test_no_boundary_unchanged(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}))
        assert mgr.trim_reasoning_for_advance(req, [1, 2]) == [1, 2]

    def test_mixed_block_trimmed(self):
        # 一步内『思考 token + 结束标记 + 语法 token』混块：
        # all len 10、本步 4 个（首 idx 6）、边界 idx 8 → 裁掉前 3 个
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                          all_token_ids=list(range(10)))
        req.structured_output_request.reasoning_end_token_index = 8
        assert mgr.trim_reasoning_for_advance(req, [6, 7, 8, 9]) == [9]

    def test_step_fully_after_boundary_unchanged(self):
        mgr = make_manager(V)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                          all_token_ids=list(range(10)))
        req.structured_output_request.reasoning_end_token_index = 5  # 边界在窗口前
        assert mgr.trim_reasoning_for_advance(req, [6, 7, 8, 9]) == [6, 7, 8, 9]


class TestMidWindowReasoningEnd:
    """m15：掩码装配窗口内的思考结束检测（L285-L323 + L333-L348）。"""

    def test_flip_mid_window_constrains_rest_and_tolerates_rejection(self):
        mgr = make_manager(V, num_speculative_tokens=4)
        # 草稿窗口 [10, 20, 30, 40]，fake reasoner 在 20 处检出思考结束
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={3}),
                          all_token_ids=[1, 2], prompt_token_ids=[1, 2])
        r = attach_reasoner(mgr, req)
        r._prompt_end = False
        req.structured_output_request.reasoning_ended = False  # 起步不受约束
        g = req.structured_output_request.grammar
        g.reject_tokens = {30}  # 翻转后首个草稿被语法拒绝——须容忍（草稿先于掩码存在）

        drafts = [10, 20, 30, 40]
        r._streaming = lambda all_len, delta: delta == [20]

        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": drafts})

        full = np.full(V // 32, -1, dtype=np.int32)
        # 行 0（t=10）、行 1（t=20 翻转发生在填行之后）→ 整行放行
        assert np.array_equal(out[0], full)
        assert np.array_equal(out[1], full)
        # 行 2（t=30）、行 3（t=40）→ 翻转已生效 → 受约束
        assert np.array_equal(out[2], bits_to_row({3}))
        assert np.array_equal(out[3], bits_to_row({3}))
        # bonus 行：bonus_apply = should_fill(False) or apply(True) → True（双触发）
        assert np.array_equal(out[4], bits_to_row({3}))

        # 推进账：结束标记 20 不推进（它是思考内容）；30 被拒但容忍；40 接受
        assert [t for _, t in g.accepts] == [[30], [40]]
        assert g.rollbacks == [1]  # 只有 40 试探成功 → rollback(1)

        # 检测用的假想序列 = history + 全部草稿；逐位前缀试
        hist_len = 2
        assert [ln for ln, _ in r.streaming_calls] == [
            hist_len + 1,  # i=0: simulated = history + [10]
            hist_len + 2,  # i=1: simulated = history + [10, 20]
        ]

        # reasoning_ended 未持久化（要等 should_advance）→ 再问 should_fill 仍 False
        assert mgr.should_fill_bitmask(req) is False

    def test_no_detection_when_already_constrained(self):
        # apply=True 时 detect_reasoning_end 三条件不成立 → 不做窗口内检测
        mgr = make_manager(V, num_speculative_tokens=2)
        req = FakeRequest("r1", grammar=FakeGrammar(V, allowed={3}))
        r = attach_reasoner(mgr, req)
        req.structured_output_request.reasoning_ended = True  # 已结束 → 受约束
        out = mgr.grammar_bitmask({"r1": req}, ["r1"], {"r1": [10, 20]})
        assert np.array_equal(out[0], bits_to_row({3}))
        assert r.streaming_calls == []  # 未进入检测分支
