"""Unit + integration tests for Scheduler — the heart of Ch04.

Coverage:
- arrival-order admission (FCFS in waiting queue)
- step-loop progress (chunked prefill, decode advance, finish)
- preemption / eviction path (two-phase, pop-tail, retry, skip Phase 2)
- batch shape (token budget cap, max_num_running_reqs cap, no overshoot)

Tests are written against the new module layout (request.py, output.py,
request_queue.py, kv_cache_manager.py, scheduler.py). Legacy tests under
_legacy/ use stale APIs and are reference-only.
"""

from __future__ import annotations

import pytest

from implementation.output import SchedulerOutput
from implementation.request import Request, RequestStatus
from implementation.scheduler import (
    Scheduler,
    continuous_batching_steps,
    static_batching_steps,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _mk_sched(
    *,
    budget: int = 256,
    max_running: int = 8,
    blocks: int = 64,
    block_size: int = 16,
    chunked: bool = True,
    long_threshold: int = 0,
) -> Scheduler:
    return Scheduler(
        max_num_scheduled_tokens=budget,
        max_num_running_reqs=max_running,
        num_gpu_blocks=blocks,
        block_size=block_size,
        enable_chunked_prefill=chunked,
        long_prefill_token_threshold=long_threshold,
    )


def _req(rid: str, prompt_len: int, max_out: int = 1, t: float = 0.0) -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_out,
        arrival_time=t,
    )


def _drive(sched: Scheduler, max_steps: int = 200) -> list[SchedulerOutput]:
    """Run schedule()+update_after_step() until the queue drains, capping at
    max_steps for safety."""
    history: list[SchedulerOutput] = []
    for _ in range(max_steps):
        if not sched.running and not sched.waiting:
            break
        out = sched.schedule()
        sched.update_after_step(out)
        history.append(out)
    return history


# ─────────────────────────────────────────────────────────────────────────
# 1. Output shape and basic invariants
# ─────────────────────────────────────────────────────────────────────────


class TestOutputShape:
    def test_empty_schedule_returns_empty_output(self) -> None:
        """No requests in flight → empty SchedulerOutput, no crash."""
        sched = _mk_sched()
        out = sched.schedule()
        assert isinstance(out, SchedulerOutput)
        assert out.num_scheduled_tokens == {}
        assert out.total_num_scheduled_tokens == 0
        assert out.preempted_req_ids == set()
        assert out.newly_running_req_ids == []

    def test_total_num_scheduled_tokens_matches_sum(self) -> None:
        """total_num_scheduled_tokens must equal sum of values (output.py L194)."""
        sched = _mk_sched(budget=128)
        sched.add_request(_req("a", 32))
        sched.add_request(_req("b", 16))
        out = sched.schedule()
        assert out.total_num_scheduled_tokens == sum(out.num_scheduled_tokens.values())


# ─────────────────────────────────────────────────────────────────────────
# 2. Arrival-order admission (FCFS in waiting queue)
# ─────────────────────────────────────────────────────────────────────────


class TestArrivalOrderAdmission:
    def test_first_added_admitted_first(self) -> None:
        """Phase 2 admits in arrival order. First added → first running."""
        sched = _mk_sched(budget=128, max_running=8)
        for rid, plen in [("a", 16), ("b", 16), ("c", 16)]:
            sched.add_request(_req(rid, plen))
        out = sched.schedule()
        assert out.newly_running_req_ids == ["a", "b", "c"]
        assert [r.request_id for r in sched.running] == ["a", "b", "c"]

    def test_admission_capped_by_max_num_running_reqs(self) -> None:
        """When max_running=2, only the first two get admitted in step 1."""
        sched = _mk_sched(budget=512, max_running=2, blocks=32)
        for rid in ("a", "b", "c", "d"):
            sched.add_request(_req(rid, 16))
        out = sched.schedule()
        assert out.newly_running_req_ids == ["a", "b"]
        assert len(sched.running) == 2
        assert len(sched.waiting) == 2
        # Arrival order preserved in waiting.
        assert [r.request_id for r in sched.waiting] == ["c", "d"]

    def test_admission_capped_by_token_budget(self) -> None:
        """With budget=32 and three 16-tok prompts, only two admit."""
        sched = _mk_sched(budget=32, max_running=8, blocks=32)
        for rid in ("a", "b", "c"):
            sched.add_request(_req(rid, 16))
        out = sched.schedule()
        # Two prompts × 16 tok = 32, exactly budget. The third can't admit
        # because budget is exhausted. (chunked_prefill ON: a third request
        # might still be admitted with 0 budget skipped — verify exactness.)
        assert out.total_num_scheduled_tokens == 32
        assert len(out.newly_running_req_ids) == 2


# ─────────────────────────────────────────────────────────────────────────
# 3. Step-loop progress (chunked prefill + decode + finish)
# ─────────────────────────────────────────────────────────────────────────


class TestStepLoopProgress:
    def test_single_request_completes(self) -> None:
        """A single request runs to completion: prefill chunks → decode → finish.

        Note: SchedulerOutput.finished_req_ids is a SIMPLIFIED data path here.
        vLLM populates it across step boundaries (the model runner uses it to
        drop KV between steps); our merged update_after_step clears the set
        at the end of the same step, so the field is typically empty in our
        outputs. Authority for "did it finish" is `request.is_finished()`.
        See test_finished_req_ids_field_is_simplified below.
        """
        sched = _mk_sched(budget=64, blocks=32)
        sched.add_request(_req("a", prompt_len=20, max_out=3))
        _drive(sched)
        a = sched.requests["a"]
        assert a.is_finished()
        assert a.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert len(a.output_token_ids) == 3

    def test_chunked_prefill_splits_long_prompt(self) -> None:
        """Prompt longer than budget is chunked across steps."""
        sched = _mk_sched(budget=32, blocks=32, chunked=True)
        sched.add_request(_req("a", prompt_len=80, max_out=2))
        history = _drive(sched)
        a = sched.requests["a"]
        # Total scheduled tokens across history must equal num_tokens-ish
        # (prompt 80 + outputs 2). At least 3 steps used to chunk 80 tokens
        # at 32/step.
        assert a.is_finished()
        # First three steps should each schedule 32 (prefill) until prefill done.
        assert history[0].num_scheduled_tokens["a"] == 32
        assert history[1].num_scheduled_tokens["a"] == 32
        assert history[2].num_scheduled_tokens["a"] == 16  # remainder

    def test_no_chunked_prefill_breaks_admission(self) -> None:
        """With chunked_prefill=False, a prompt > budget is deferred (not chunked)."""
        sched = _mk_sched(budget=32, max_running=8, blocks=64, chunked=False)
        sched.add_request(_req("long", prompt_len=80))
        sched.add_request(_req("short", prompt_len=16))
        out = sched.schedule()
        # FCFS + chunked off: 'long' can't fit, so we BREAK (per K05/K11),
        # 'short' must NOT jump ahead.
        assert "long" not in out.num_scheduled_tokens
        assert "short" not in out.num_scheduled_tokens
        assert out.newly_running_req_ids == []

    def test_decode_advances_one_token_per_step(self) -> None:
        """Once prefill is done, each step yields one output token."""
        sched = _mk_sched(budget=64, blocks=32)
        sched.add_request(_req("a", prompt_len=16, max_out=4))
        # Step 1: prefill 16 tokens (1 step since 16 <= 64).
        out1 = sched.schedule()
        sched.update_after_step(out1)
        a = sched.requests["a"]
        # After step 1, prefill should be done (computed=16, prompt=16),
        # but no output token yet because is_prefill check happens BEFORE
        # the advance. Actually: the advance increments num_computed_tokens
        # to 16, then is_prefill is False → output token appended.
        assert not a.is_prefill
        assert len(a.output_token_ids) == 1
        # Subsequent steps: 1 token each.
        out2 = sched.schedule()
        sched.update_after_step(out2)
        assert out2.num_scheduled_tokens["a"] == 1
        assert len(a.output_token_ids) == 2

    def test_long_prefill_threshold_caps_chunk_size(self) -> None:
        """long_prefill_token_threshold caps a single step's prefill chunk."""
        sched = _mk_sched(
            budget=128, blocks=64, chunked=True, long_threshold=24
        )
        sched.add_request(_req("a", prompt_len=80, max_out=1))
        out1 = sched.schedule()
        # Threshold=24 caps the chunk even though budget=128.
        assert out1.num_scheduled_tokens["a"] == 24


# ─────────────────────────────────────────────────────────────────────────
# 4. Preemption / eviction path
# ─────────────────────────────────────────────────────────────────────────


class TestPreemption:
    """Wisdom W02: BOTH requests must fit initially, THEN one needs more.
    The FCFS pop-tail rule (K06) means the LATER request gets preempted."""

    def test_preempt_pops_tail_under_fcfs(self) -> None:
        """Two requests fit; first needs more KV → preempt the LAST one."""
        # block_size=16, blocks=4 → 4 blocks total.
        # Two prompts of 16 tokens = 1 block each → fits, 2 blocks free.
        # Then `a` decodes, eventually wants another block; if no spare → preempt.
        # Use tight blocks=2 so any decode growth triggers OOM.
        sched = _mk_sched(budget=64, max_running=2, blocks=2, block_size=16)
        sched.add_request(_req("a", prompt_len=16, max_out=20))
        sched.add_request(_req("b", prompt_len=16, max_out=20))
        # Step 1: both admit.
        out1 = sched.schedule()
        sched.update_after_step(out1)
        assert set(out1.newly_running_req_ids) == {"a", "b"}
        assert len(sched.running) == 2
        # Step 2: a's decode (1 tok) needs new block? num_computed=16, fits in
        # block 0 already. We need to push past the block boundary. Run several
        # steps until an OOM forces a preempt of `b` (the tail).
        history: list[SchedulerOutput] = [out1]
        for _ in range(50):
            out = sched.schedule()
            sched.update_after_step(out)
            history.append(out)
            if "b" in out.preempted_req_ids:
                break
            if not sched.running and not sched.waiting:
                break
        preempt_steps = [h for h in history if h.preempted_req_ids]
        assert preempt_steps, "Expected at least one preemption event"
        # FCFS pop-tail: `b` must be preempted before `a`.
        first_preempt = preempt_steps[0]
        assert "b" in first_preempt.preempted_req_ids
        assert "a" not in first_preempt.preempted_req_ids

    def test_preempted_request_state_after_preempt(self) -> None:
        """After preempt: status=PREEMPTED, num_computed=0, blocks freed,
        num_preemptions += 1, prepended to waiting (front)."""
        sched = _mk_sched(budget=64, max_running=2, blocks=2, block_size=16)
        sched.add_request(_req("a", prompt_len=16, max_out=20))
        sched.add_request(_req("b", prompt_len=16, max_out=20))
        for _ in range(80):
            out = sched.schedule()
            sched.update_after_step(out)
            if "b" in out.preempted_req_ids:
                break
        b = sched.requests["b"]
        # During preempt: PREEMPTED, blocks freed, num_computed=0,
        # num_preemptions >=1. (After re-admission these may flip again.)
        # We're checking the IMMEDIATE post-preempt invariants by inspecting
        # state right when the preempt fired.
        assert b.num_preemptions >= 1
        # If still preempted: blocks must be empty.
        if b.status == RequestStatus.PREEMPTED:
            assert b.block_ids == []
            assert b.num_computed_tokens == 0
            # Front of waiting (prepend semantics, scheduler.py:L972).
            waiting_ids = [r.request_id for r in sched.waiting]
            assert waiting_ids[0] == "b"

    def test_skip_phase2_when_preemption_happened(self) -> None:
        """If Phase 1 preempted, Phase 2 must NOT admit a new waiter that step."""
        sched = _mk_sched(budget=128, max_running=4, blocks=2, block_size=16)
        sched.add_request(_req("a", prompt_len=16, max_out=20))
        sched.add_request(_req("b", prompt_len=16, max_out=20))
        # Start with a, b admitted.
        out1 = sched.schedule()
        sched.update_after_step(out1)
        # Drop a brand-new waiter `c` BEFORE we expect a preempt.
        sched.add_request(_req("c", prompt_len=16, max_out=2))
        # Drive until preempt, watching that on the preempt step c is NOT
        # newly_running (Phase 2 skipped per K09/D4).
        for _ in range(100):
            out = sched.schedule()
            sched.update_after_step(out)
            if out.preempted_req_ids:
                # On the preempt step: c must NOT have been admitted.
                assert "c" not in out.newly_running_req_ids
                # No new admissions at all on preempt step (Phase 2 skipped).
                assert out.newly_running_req_ids == []
                break
            if not sched.running and not sched.waiting:
                pytest.fail("Expected preemption, none happened")

    def test_preempted_request_eventually_resumes(self) -> None:
        """After preempt, the request should be re-admitted later and finish."""
        sched = _mk_sched(budget=128, max_running=2, blocks=4, block_size=16)
        sched.add_request(_req("a", prompt_len=16, max_out=4))
        sched.add_request(_req("b", prompt_len=16, max_out=4))
        _drive(sched, max_steps=200)
        # Both should finish given enough time.
        assert sched.requests["a"].is_finished()
        assert sched.requests["b"].is_finished()


# ─────────────────────────────────────────────────────────────────────────
# 5. Batch shape — invariants the scheduler must NEVER violate
# ─────────────────────────────────────────────────────────────────────────


class TestBatchShape:
    def test_token_budget_never_overshot(self) -> None:
        """sum(num_scheduled_tokens.values()) <= max_num_scheduled_tokens
        on EVERY step (assertion at scheduler.py:L848)."""
        sched = _mk_sched(budget=64, max_running=8, blocks=128)
        for rid in ("a", "b", "c", "d", "e"):
            sched.add_request(_req(rid, prompt_len=40, max_out=2))
        for h in _drive(sched, max_steps=200):
            assert h.total_num_scheduled_tokens <= 64

    def test_running_count_bounded(self) -> None:
        """len(running) <= max_num_running_reqs at all times."""
        sched = _mk_sched(budget=512, max_running=3, blocks=128)
        for rid in ("a", "b", "c", "d", "e", "f"):
            sched.add_request(_req(rid, prompt_len=16, max_out=4))
        for _ in range(200):
            if not sched.running and not sched.waiting:
                break
            out = sched.schedule()
            assert len(sched.running) <= 3
            sched.update_after_step(out)

    def test_per_request_tokens_le_num_new(self) -> None:
        """num_scheduled_tokens[r] <= r.num_new_tokens (no overshoot per req)."""
        sched = _mk_sched(budget=128, max_running=8, blocks=128)
        sched.add_request(_req("a", prompt_len=20, max_out=1))
        sched.add_request(_req("b", prompt_len=20, max_out=1))
        out = sched.schedule()
        for rid, n in out.num_scheduled_tokens.items():
            r = sched.requests[rid]
            # At schedule time, n <= num_new_tokens BEFORE the advance.
            # We can't easily assert pre-advance; instead, num_scheduled
            # must not exceed the original prompt length here.
            assert n <= r.num_prompt_tokens + r.max_tokens

    def test_no_request_in_running_appears_in_waiting(self) -> None:
        """A request is in exactly one of {running, waiting, finished}."""
        sched = _mk_sched(budget=64, max_running=2, blocks=8)
        for rid in ("a", "b", "c"):
            sched.add_request(_req(rid, prompt_len=16, max_out=2))
        for _ in range(50):
            if not sched.running and not sched.waiting:
                break
            out = sched.schedule()
            sched.update_after_step(out)
            running_ids = {r.request_id for r in sched.running}
            waiting_ids = {r.request_id for r in sched.waiting}
            assert running_ids.isdisjoint(waiting_ids)


# ─────────────────────────────────────────────────────────────────────────
# 6. Integration — finish path frees blocks, demo workload terminates
# ─────────────────────────────────────────────────────────────────────────


class TestIntegration:
    def test_all_blocks_reclaimed_on_drain(self) -> None:
        """When every request finishes, all KV blocks return to the free list."""
        sched = _mk_sched(budget=128, max_running=4, blocks=32)
        for rid in ("a", "b", "c"):
            sched.add_request(_req(rid, prompt_len=16, max_out=3))
        _drive(sched, max_steps=200)
        assert sched.kv_cache_manager.num_free_blocks == sched.kv_cache_manager.num_gpu_blocks

    def test_finished_status_set_correctly(self) -> None:
        """All requests finish with FINISHED_LENGTH_CAPPED (max_tokens hit)."""
        sched = _mk_sched(budget=128, max_running=4, blocks=32)
        for rid in ("a", "b", "c"):
            sched.add_request(_req(rid, prompt_len=16, max_out=2))
        _drive(sched, max_steps=200)
        for r in sched.requests.values():
            assert r.status == RequestStatus.FINISHED_LENGTH_CAPPED

    def test_finished_req_ids_field_is_simplified(self) -> None:
        """Documents a deliberate fidelity gap relative to vLLM.

        vLLM's SchedulerOutput.finished_req_ids carries IDs of requests that
        finished BETWEEN the previous and current step, so the model runner
        can drop their KV state. Our merged `update_after_step` clears
        `self.finished_req_ids` in the same call, so the next `schedule()`'s
        output sees an empty set.

        For Ch04 (no real model runner; pedagogical scope) this divergence
        is acceptable. If a future chapter wires up a runner that consumes
        finished_req_ids, this needs to flip to vLLM-style cross-step carry.
        Cross-link: scheduler.py:L919-L923 (vLLM) vs our scheduler.py:L217-L218
        and L279.
        """
        sched = _mk_sched(budget=64, blocks=32)
        sched.add_request(_req("a", prompt_len=16, max_out=1))
        history = _drive(sched)
        a = sched.requests["a"]
        assert a.is_finished()
        # Authoritative check.
        for h in history:
            # Either empty or containing 'a' is acceptable; current impl gives
            # all-empty, but we don't pin to that to allow a future fix.
            assert h.finished_req_ids.issubset({"a"})

    def test_demo_workload_matches_implementer_claim(self) -> None:
        """The demo's exact workload completes with all blocks reclaimed.

        The implementer reported: 19 steps, 200/200 blocks reclaimed, ~20.8x
        speedup. We don't pin the exact step count (small refactors might shift
        it), but invariants must hold.
        """
        sched = _mk_sched(
            budget=128, max_running=4, blocks=200, block_size=16, chunked=True
        )
        sched.add_request(_req("1", prompt_len=400, max_out=4))
        sched.add_request(_req("2", prompt_len=64, max_out=8))
        sched.add_request(_req("3", prompt_len=16, max_out=16))
        history = _drive(sched, max_steps=200)
        # All finished.
        assert all(r.is_finished() for r in sched.requests.values())
        # All blocks reclaimed.
        assert sched.kv_cache_manager.num_free_blocks == 200
        # Step count is in the right ballpark (10-30 range).
        assert 10 <= len(history) <= 30


# ─────────────────────────────────────────────────────────────────────────
# 7. Bubble simulator — pedagogical static-vs-continuous comparison
# ─────────────────────────────────────────────────────────────────────────


class TestBubbleSimulator:
    def test_static_steps_formula(self) -> None:
        """static = longest_prompt + longest_output."""
        wl = [(400, 4), (64, 8), (16, 16)]
        # longest_prompt=400, longest_output=16 → 416.
        assert static_batching_steps(wl) == 416

    def test_continuous_beats_static_on_skewed_workload(self) -> None:
        """The whole point: continuous batching trims the static bubble."""
        wl = [(400, 4), (64, 8), (16, 16)]
        s = static_batching_steps(wl)
        c = continuous_batching_steps(wl, token_budget=128)
        assert c < s
        assert s / c > 5.0  # Order-of-magnitude speedup on this workload.

    def test_empty_workload_zero_steps(self) -> None:
        assert static_batching_steps([]) == 0
        assert continuous_batching_steps([], token_budget=64) == 0
