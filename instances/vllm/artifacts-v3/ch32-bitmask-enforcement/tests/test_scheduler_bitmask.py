# ch31 主电池三：调度侧——门控置位（m05）/行序账本（m03）/草稿语法过滤
# （m17）/update_from_output 真推进（m14 的 scheduler 面）。
# 基准：vllm/v1/core/sched/scheduler.py:L1317-L1343 / L1646-L1668 / L1746-L1791 /
# L1817-L1843 / L2147-L2166 / L2168-L2203。
from __future__ import annotations

import numpy as np
import torch

from conftest import (
    FakeGrammar,
    FakeRequest,
    FakeSchedulerOutput,
    make_manager,
    make_vllm_config,
)


V = 64


def make_scheduler(max_num_seqs=8, num_speculative_tokens=0, manager=None):
    from vllm.v1.core.sched.scheduler import Scheduler

    cfg = make_vllm_config(
        max_num_seqs=max_num_seqs, num_speculative_tokens=num_speculative_tokens
    )
    cfg.parallel_config = type("P", (), {"pipeline_parallel_size": 1})()
    if manager is None:
        manager = make_manager(V, max_num_seqs=max_num_seqs,
                               num_speculative_tokens=num_speculative_tokens)
    return Scheduler(cfg, structured_output_manager=manager)


class TestUpdateAfterScheduleGate:
    """m05：has_structured_output_requests 由『use_structured_output 且非
    prefill chunk』逐请求 or 起来（L1338-L1340）。"""

    def test_or_accumulation(self):
        sched = make_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=5)
        r2 = FakeRequest("r2", use_structured_output=False, num_tokens=5)
        sched.requests["r1"] = r1
        sched.requests["r2"] = r2
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 5, "r2": 5})
        sched._update_after_schedule(so)
        assert so.has_structured_output_requests is True

    def test_all_unstructured_stays_false(self):
        sched = make_scheduler()
        r1 = FakeRequest("r1", use_structured_output=False, num_tokens=5)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 5})
        sched._update_after_schedule(so)
        assert so.has_structured_output_requests is False

    def test_prefill_chunk_excluded(self):
        # 本步不产 logits 的请求不占掩码行：chunked prefill 中段（本步没跑完
        # prompt）is_prefill_chunk 重算为 True → 不置位（行数对账的前提）
        sched = make_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=10,
                         num_computed_tokens=4)  # 跑了 4/10 → 仍是 chunk
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 4})
        sched._update_after_schedule(so)
        assert r1.num_computed_tokens == 8
        assert r1.is_prefill_chunk is True
        assert so.has_structured_output_requests is False

    def test_completed_prefill_counts(self):
        # 最后一段 prefill 跑完（computed >= num_tokens + placeholders）→ 非chunk
        sched = make_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=10,
                         num_computed_tokens=6)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 4})
        sched._update_after_schedule(so)
        assert r1.is_prefill_chunk is False
        assert so.has_structured_output_requests is True


class TestGetGrammarBitmask:
    """m03 前半：门控快返 + 行序账本（L1646-L1668）。"""

    def test_gate_off_returns_none(self):
        sched = make_scheduler()
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 1},
                                 has_structured_output_requests=False)
        assert sched.get_grammar_bitmask(so) is None

    def test_ids_follow_num_scheduled_tokens_order(self):
        # 掩码行序 = num_scheduled_tokens 的迭代顺序（dict 插入序）
        sched = make_scheduler()
        for rid in ["r2", "r1", "r3"]:
            sched.requests[rid] = FakeRequest(
                rid, grammar=FakeGrammar(V, allowed={1}), num_tokens=1
            )
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r2": 1, "r1": 1, "r3": 1},
            has_structured_output_requests=True,
        )
        out = sched.get_grammar_bitmask(so)
        assert out is not None
        assert out.structured_output_request_ids == ["r2", "r1", "r3"]
        assert isinstance(out.grammar_bitmask, np.ndarray)
        assert out.grammar_bitmask.shape[0] == 3

    def test_prefill_chunk_request_not_collected(self):
        sched = make_scheduler()
        r1 = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}), num_tokens=1)
        r2 = FakeRequest("r2", grammar=FakeGrammar(V, allowed={1}), num_tokens=10,
                         num_computed_tokens=4)  # chunk 中段
        r2.is_prefill_chunk = True
        sched.requests["r1"] = r1
        sched.requests["r2"] = r2
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1, "r2": 4},
            has_structured_output_requests=True,
        )
        out = sched.get_grammar_bitmask(so)
        assert out.structured_output_request_ids == ["r1"]

    def test_structured_only_collected(self):
        sched = make_scheduler()
        sched.requests["r1"] = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                                           num_tokens=1)
        sched.requests["r2"] = FakeRequest("r2", use_structured_output=False,
                                           num_tokens=1)
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1, "r2": 1},
            has_structured_output_requests=True,
        )
        out = sched.get_grammar_bitmask(so)
        assert out.structured_output_request_ids == ["r1"]

    def test_all_filtered_returns_none(self):
        sched = make_scheduler()
        r1 = FakeRequest("r1", use_structured_output=False, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1}, has_structured_output_requests=True
        )
        assert sched.get_grammar_bitmask(so) is None


class TestUpdateDraftTokenIdsInOutput:
    """m17：草稿先过语法 validate 过滤 + -1 补齐 + num_invalid 记账
    （L2168-L2203）。"""

    def test_validate_filter_pad_and_accounting(self):
        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1})
        r1 = FakeRequest("r1", grammar=g, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            scheduled_spec_decode_tokens={"r1": [0, 0, 0, 0]},  # 4 个占位
        )
        from vllm.v1.outputs import DraftTokenIds

        # 草稿 [9, 8, 7, 6]：validate 只认前 2 个合法前缀
        g._validate_keep = 2
        sched.update_draft_token_ids_in_output(
            DraftTokenIds(["r1"], [[9, 8, 7, 6]]), so
        )
        assert so.scheduled_spec_decode_tokens["r1"] == [9, 8, -1, -1]
        assert so.num_invalid_spec_tokens == {"r1": 2}

    def test_trim_to_scheduled_count(self):
        # 草稿比本拍已排数长（chunked prefill 场景）→ 截断到已排数
        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1})
        r1 = FakeRequest("r1", grammar=g, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            scheduled_spec_decode_tokens={"r1": [0, 0]},  # 本拍只排 2 个
        )
        from vllm.v1.outputs import DraftTokenIds

        sched.update_draft_token_ids_in_output(
            DraftTokenIds(["r1"], [[9, 8, 7]]), so
        )
        assert so.scheduled_spec_decode_tokens["r1"] == [9, 8]

    def test_no_advance_no_validate(self):
        # should_advance 不通过（use_structured_output=False 旁路）→ 不做语法
        # 过滤，草稿原样入账、不记 invalid
        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1})
        r1 = FakeRequest("r1", grammar=g, num_tokens=1,
                         use_structured_output=False)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            scheduled_spec_decode_tokens={"r1": [0, 0]},
        )
        from vllm.v1.outputs import DraftTokenIds

        sched.update_draft_token_ids_in_output(
            DraftTokenIds(["r1"], [[9, 8]]), so
        )
        assert g.validates == []  # 未过滤
        assert so.scheduled_spec_decode_tokens["r1"] == [9, 8]
        assert so.num_invalid_spec_tokens == {}

    def test_finished_request_skipped(self):
        sched = make_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, is_finished=True)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            scheduled_spec_decode_tokens={"r1": [0, 0]},
        )
        from vllm.v1.outputs import DraftTokenIds

        sched.update_draft_token_ids_in_output(
            DraftTokenIds(["r1"], [[9, 8]]), so
        )
        assert so.scheduled_spec_decode_tokens["r1"] == [0, 0]  # 未被替换
        assert so.num_invalid_spec_tokens == {}

    def test_request_without_placeholders_skipped(self):
        sched = make_scheduler()
        r1 = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}), num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            scheduled_spec_decode_tokens={},  # 无占位条目
        )
        from vllm.v1.outputs import DraftTokenIds

        sched.update_draft_token_ids_in_output(
            DraftTokenIds(["r1"], [[9, 8]]), so
        )
        assert so.num_invalid_spec_tokens == {}


class TestUpdateFromOutputTrueAdvance:
    """真推进唯一写者（L1817-L1843）：should_advance → trim → accept_tokens；
    拒绝即 FINISHED_ERROR。"""

    def _mro(self):
        from vllm.v1.outputs import ModelRunnerOutput

        return ModelRunnerOutput(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            sampled_token_ids=[[7, 8, 9]],
        )

    def test_advance_with_accepted_tokens(self):
        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1})
        r1 = FakeRequest("r1", grammar=g, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
        sched.update_from_output(so, self._mro())
        assert [t for _, t in g.accepts] == [[7, 8, 9]]
        assert r1.is_finished() is False

    def test_mixed_block_trimmed_before_accept(self):
        # #44006：思考+语法混块 → 先 trim 再喂 accept_tokens
        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1})
        r1 = FakeRequest("r1", grammar=g, num_tokens=1,
                         all_token_ids=list(range(10)))
        r1.structured_output_request.reasoning_end_token_index = 8
        # 让 should_advance 放行：无 reasoner → True
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 4})
        sched.update_from_output(so, self._mro())
        assert [t for _, t in g.accepts] == [[9]]  # [7,8,8?] 裁掉思考后只剩 9

    def test_rejection_terminates_request(self):
        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1}, reject_tokens={7})
        r1 = FakeRequest("r1", grammar=g, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
        sched.update_from_output(so, self._mro())
        assert r1.is_finished() is True
        assert r1.status.name == "FINISHED_ERROR"
        assert r1.resumable is False

    def test_no_new_tokens_no_advance(self):
        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1})
        r1 = FakeRequest("r1", grammar=g, num_tokens=1)
        sched.requests["r1"] = r1
        from vllm.v1.outputs import ModelRunnerOutput

        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
        sched.update_from_output(
            so, ModelRunnerOutput(req_ids=["r1"], req_id_to_index={"r1": 0},
                                  sampled_token_ids=[])
        )
        assert g.accepts == []


class TestUpdateDraftTokenIdsSync:
    """同步变体（L2147-L2166，post_step 消费）：should_advance 才 validate。"""

    def test_sync_validate(self):
        from vllm.v1.outputs import DraftTokenIds

        sched = make_scheduler()
        g = FakeGrammar(V, allowed={1})
        g._validate_keep = 1
        r1 = FakeRequest("r1", grammar=g, num_tokens=1)
        sched.requests["r1"] = r1
        sched.update_draft_token_ids(DraftTokenIds(["r1"], [[5, 6]]))
        assert r1.spec_token_ids == [5]

    def test_prefill_chunk_drafts_dropped(self):
        from vllm.v1.outputs import DraftTokenIds

        sched = make_scheduler()
        r1 = FakeRequest("r1", grammar=FakeGrammar(V, allowed={1}),
                         num_tokens=10, num_computed_tokens=2,
                         spec_token_ids=[3, 4])
        r1.is_prefill_chunk = True
        sched.requests["r1"] = r1
        sched.update_draft_token_ids(DraftTokenIds(["r1"], [[5, 6]]))
        assert r1.spec_token_ids == []
