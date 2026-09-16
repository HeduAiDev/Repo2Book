# ch31 主电池六：异步调度下的延后采样链（m16）+ AsyncScheduler 置位
# （vllm/v1/core/sched/async_scheduler.py:L13-L49 / engine/core.py:L625-L739）。
# 因果骨架：pending 置位 → 挂起 deferred → take_draft_token_ids →
# update_draft_token_ids_in_output（草稿先过语法）→ get_grammar_bitmask →
# sample_tokens。
from __future__ import annotations

from collections import deque
from concurrent.futures import Future

import numpy as np
import torch

from conftest import (
    FakeGrammarOutput,
    FakeRequest,
    FakeSchedulerOutput,
    make_manager,
    make_vllm_config,
)


V = 64


def make_async_scheduler(num_sampled_tokens_per_step=1):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    cfg = make_vllm_config(max_num_seqs=8, num_speculative_tokens=0)
    cfg.parallel_config = type("P", (), {"pipeline_parallel_size": 1})()
    mgr = make_manager(V)
    sched = AsyncScheduler.__new__(AsyncScheduler)
    # reduced __init__ 消费面（真实装配归 ch10-ch12）
    sched.requests = {}
    sched.structured_output_manager = mgr
    sched.num_spec_tokens = 0
    sched.num_sampled_tokens_per_step = num_sampled_tokens_per_step
    sched.parallel_config = cfg.parallel_config
    sched.use_v2_model_runner = False
    sched._inflight_prefills = set()
    sched._spec_token_placeholders = [-1] * sched.num_spec_tokens
    sched.pp_size = cfg.parallel_config.pipeline_parallel_size
    return sched


class TestAsyncSchedulerPending:
    """pending_structured_output_tokens 置位 + 占位记账（async_scheduler.py:L19-L44）。"""

    def test_pending_set_when_placeholders_outstanding(self):
        sched = make_async_scheduler()
        # 异步重叠拍：computed 已领先（上拍已排但 token 未回来）——
        # base 的 is_prefill_chunk 重算（computed < tokens+placeholders）不成立，
        # 才进 async 循环置 pending
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=1,
                         num_computed_tokens=2, num_output_placeholders=2)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 1})
        sched._update_after_schedule(so)
        assert r1.is_prefill_chunk is False
        assert so.pending_structured_output_tokens is True

    def test_pending_not_set_without_placeholders(self):
        sched = make_async_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 1})
        sched._update_after_schedule(so)
        assert so.pending_structured_output_tokens is False

    def test_placeholder_increments_by_bonus_plus_spec(self):
        # 每拍占位 += num_sampled_tokens_per_step + 本拍 spec 数（diffusion 无 AR
        # bonus → num_sampled_tokens_per_step == 0）
        sched = make_async_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            scheduled_spec_decode_tokens={"r1": [7, 8]},
        )
        sched._update_after_schedule(so)
        assert r1.num_output_placeholders == 1 + 2

    def test_spec_token_ids_replaced_with_minus_one_placeholders(self):
        # 真草稿在 worker 进程内原地替换——调度侧先摆 -1 占位数组
        sched = make_async_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 1},
                                 num_spec_tokens_to_schedule=3)
        sched._update_after_schedule(so)
        assert r1.spec_token_ids == [-1, -1, -1]

    def test_base_gate_still_applies(self):
        # super()._update_after_schedule 的置位门控在覆写里依旧生效
        sched = make_async_scheduler()
        r1 = FakeRequest("r1", use_structured_output=True, num_tokens=1)
        sched.requests["r1"] = r1
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 1})
        sched._update_after_schedule(so)
        assert so.has_structured_output_requests is True


class TestStepWithBatchQueue:
    """core.py step_with_batch_queue：延后分流 + deferred 兑现链。"""

    def _make_core(self, pending, seed_queue=False):
        from vllm.v1.engine.core import EngineCore

        core = EngineCore.__new__(EngineCore)
        core.batch_queue = deque()
        core.batch_queue_size = 4
        core.check_for_draft_tokens = True
        core.is_pooling_model = False
        core.scheduler = DeferredSpyScheduler(pending, log=[])
        core.model_executor = DeferredSpyExecutor(core.scheduler.log)
        if seed_queue:
            # 真实流程：上一拍入队的 (future, scheduler_output, exec_future)，
            # 本拍 pop 出来先收上批输出，再兑现 deferred 链
            prev = FakeSchedulerOutput(num_scheduled_tokens={"r0": 2})
            fut = Future()
            fut.set_result(("model_output_prev", None))
            exec_prev = Future()
            core.batch_queue.append((fut, prev, exec_prev))
        return core

    def test_immediate_path_when_not_pending(self):
        # pending 为假 → 与 step() 同序：立即算掩码 + sample_tokens(non_block)
        core = self._make_core(pending=False)
        core.step_with_batch_queue()
        assert core.scheduler.log == [
            "dispatch", "bitmask", "sample_tokens", "update_from_output",
        ]

    def test_deferred_path_causal_order(self):
        # pending 为真 → 本拍挂起；先收上批输出，再走兑现链：take_draft →
        # update_draft → bitmask → sample_tokens（草稿必须先回调度器，表才算
        # 得出来）
        core = self._make_core(pending=True, seed_queue=True)
        core.step_with_batch_queue()
        assert core.scheduler.log == [
            "dispatch",              # ② 本拍发车（挂起为 deferred）
            "update_from_output",    # 先收上一批输出（pop 出的上拍条目）
            "take_draft",            # 兑现链第一步：草稿 D2H 回调度器
            "update_draft",          # 草稿先过语法 validate 过滤
            "bitmask",               # 然后才轮到掩码
            "sample_tokens",         # 最后补采样
        ]
        assert core.model_executor.grammar_output_passed is not None
        # deferred 条目重新入队
        assert len(core.batch_queue) == 1

    def test_no_draft_tokens_skips_update(self):
        # take_draft_token_ids 返回 None（非 spec 部署）→ 跳过草稿过滤直奔掩码
        core = self._make_core(pending=True, seed_queue=True)
        core.model_executor.draft = None
        core.step_with_batch_queue()
        assert core.scheduler.log == [
            "dispatch", "update_from_output",
            "take_draft", "bitmask", "sample_tokens",
        ]


class DeferredSpyScheduler:
    def __init__(self, pending, log):
        self.log = log
        self.pending = pending
        self.updated = []

    def has_requests(self):
        return True

    def schedule(self, throttle=False):
        return FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 3},
            has_structured_output_requests=True,
            pending_structured_output_tokens=self.pending,
            total_num_scheduled_tokens=3,
        )

    def get_grammar_bitmask(self, scheduler_output):
        self.log.append("bitmask")
        assert "dispatch" in self.log
        return FakeBitmask()

    def update_from_output(self, scheduler_output, model_output):
        self.log.append("update_from_output")
        self.updated.append((scheduler_output, model_output))
        return {}

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        self.log.append("update_draft")
        # 因序断言：草稿过滤必须发生在掩码计算之前
        assert "bitmask" not in self.log


class FakeBitmask:
    structured_output_request_ids = ["r1"]
    grammar_bitmask = np.ones((1, 2), np.int32)


class DeferredSpyExecutor:
    def __init__(self, log):
        self.log = log
        self.grammar_output_passed = None
        self.draft = ("draft_ids",)

    def execute_model(self, scheduler_output, non_block=False):
        assert non_block is True
        self.log.append("dispatch")
        fut = Future()
        fut.set_result(None)
        return fut

    def sample_tokens(self, grammar_output, non_block=False):
        self.grammar_output_passed = grammar_output
        self.log.append("sample_tokens")
        fut = Future()
        fut.set_result(("model_output", None))
        return fut

    def take_draft_token_ids(self):
        self.log.append("take_draft")
        return self.draft
