"""ch12 测试：复现真实 vLLM step_with_batch_queue 的可观察行为。

测的不是精简版自洽，而是它复现 dossier 记录的真实 vLLM 行为：
  - max_concurrent_batches → batch_queue_size → step_fn 绑定的链路（三处 executor）；
  - 「填管道优先于取结果」：能调度且队未满且队尾未完成 → return (None, True)；
  - appendleft 进 / pop 出的 FIFO 语义；
  - 队满 / 无更多请求 / 队尾已完成 三个触发切换到取结果模式；
  - deferred sampling：pending_structured_output_tokens 时推迟采样，下半段补 bitmask 重入队；
  - deferred + spec decode 先 take_draft_token_ids；
  - has_work() 因队列非空而保活。

纯单元测试，不 import vllm → host python3 -m pytest 可跑。
"""

from concurrent.futures import Future

import pytest

from implementation.engine_core_batch_queue import (
    SchedulerOutput,
    FakeExecutor,
    FakeScheduler,
    _MinimalEngineCore,
)
from implementation.executor_max_concurrent_batches import (
    AbstractExecutor,
    MultiprocExecutor,
    UniProcExecutor,
    _ParallelConfig,
    _SchedulerConfig,
)


def make_core(max_concurrent_batches=2, async_scheduling=False, use_spec_decode=False):
    execu = FakeExecutor(max_concurrent_batches=max_concurrent_batches)
    sched = FakeScheduler()
    core = _MinimalEngineCore(
        model_executor=execu,
        scheduler=sched,
        async_scheduling=async_scheduling,
        use_spec_decode=use_spec_decode,
    )
    return core, execu, sched


# ---------------------------------------------------------------------------
# max_concurrent_batches → batch_queue_size → step_fn 绑定
# ---------------------------------------------------------------------------


def test_abstract_executor_default_one():
    # 基类默认 1（vllm/v1/executor/abstract.py:L256-L258）
    assert AbstractExecutor().max_concurrent_batches == 1


def test_multiproc_pp_size():
    # PP>1 → pp_size（multiproc_executor.py:L474-L478）
    e = MultiprocExecutor(_ParallelConfig(pipeline_parallel_size=4), _SchedulerConfig())
    assert e.max_concurrent_batches == 4


def test_multiproc_single_card_async_is_two():
    # pp<=1 且 async_scheduling → 2
    e = MultiprocExecutor(
        _ParallelConfig(pipeline_parallel_size=1),
        _SchedulerConfig(async_scheduling=True),
    )
    assert e.max_concurrent_batches == 2


def test_multiproc_single_card_no_async_is_one():
    e = MultiprocExecutor(
        _ParallelConfig(pipeline_parallel_size=1),
        _SchedulerConfig(async_scheduling=False),
    )
    assert e.max_concurrent_batches == 1


def test_uniproc_async_is_two_else_one():
    # uniproc_executor.py:L63-L65
    assert UniProcExecutor(_SchedulerConfig(async_scheduling=True)).max_concurrent_batches == 2
    assert UniProcExecutor(_SchedulerConfig(async_scheduling=False)).max_concurrent_batches == 1


def test_step_fn_binds_to_plain_step_when_size_one():
    # batch_queue_size==1 → batch_queue is None → step_fn 绑 step（core.py:L213-L214）
    core, _, _ = make_core(max_concurrent_batches=1)
    assert core.batch_queue is None
    assert core.step_fn == core.step


def test_step_fn_binds_to_batch_queue_when_size_gt_one():
    core, _, _ = make_core(max_concurrent_batches=2)
    assert core.batch_queue is not None
    assert core.batch_queue.maxlen == 2
    assert core.step_fn == core.step_with_batch_queue


# ---------------------------------------------------------------------------
# 填管道优先于取结果 + FIFO（appendleft / pop）
# ---------------------------------------------------------------------------


class _NeverDoneFuture(Future):
    """模拟『最旧批仍在飞』：done() 恒 False，但 result() 仍可取（替身里立即 set）。"""

    def done(self):  # type: ignore[override]
        return False


class PendingExecutor(FakeExecutor):
    """让最早入队那批的 future 看起来还没 done()，以触发『填管道』判定。"""

    def sample_tokens(self, grammar_output, non_block=False):
        f = _NeverDoneFuture()
        from implementation.engine_core_batch_queue import ModelRunnerOutput

        f.set_result(ModelRunnerOutput())
        return f


def test_fill_pipeline_priority_returns_none_true():
    # 队未满 + 真调度了 token + 队尾未 done → 直接 return (None, True)，不阻塞取结果
    sched = FakeScheduler()
    execu = PendingExecutor(max_concurrent_batches=3)
    core = _MinimalEngineCore(model_executor=execu, scheduler=sched)
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=5, batch_id=1))

    out, executed = core.step_with_batch_queue()
    assert out is None
    assert executed is True
    # 新批已 appendleft 入队，深度变 1
    assert len(core.batch_queue) == 1


def test_appendleft_then_pop_is_fifo():
    # 先调度的批先取结果：批1 先入，批2 后入；队满后 pop 应先拿批1。
    sched = FakeScheduler()
    execu = PendingExecutor(max_concurrent_batches=2)
    core = _MinimalEngineCore(model_executor=execu, scheduler=sched)
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=5, batch_id=1))
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=5, batch_id=2))

    # 第一次：批1 入队，队未满(1<2)且队尾未 done → (None, True)
    out1, ex1 = core.step_with_batch_queue()
    assert out1 is None and ex1 is True
    # 第二次：批2 入队，此时队满(2==2)，不满足 len<size → 落到 pop 取结果
    out2, ex2 = core.step_with_batch_queue()
    assert out2 is not None
    # FIFO：pop 出的应是最早入队的批1
    assert sched.update_from_output_calls[0][0].batch_id == 1


def test_no_more_requests_falls_to_pop():
    # 队未满但 scheduler 没有更多请求 → 不能只填，必须 pop 取结果
    core, execu, sched = make_core(max_concurrent_batches=3)
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=5, batch_id=1))
    # 第一次：批1 入队；队尾 future 已 done()（默认 FakeExecutor 立即完成）→ 不满足
    # 『not done()』→ 直接落到 pop
    out, executed = core.step_with_batch_queue()
    assert out is not None
    assert sched.update_from_output_calls[0][0].batch_id == 1


def test_queue_tail_done_falls_to_pop():
    # 队尾最旧批已完成（done()==True）→ 即便队未满也切到取结果模式
    core, execu, sched = make_core(max_concurrent_batches=3)
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=5, batch_id=7))
    out, _ = core.step_with_batch_queue()
    # 默认 FakeExecutor 的 future 立即 done → 直接 pop
    assert out is not None


# ---------------------------------------------------------------------------
# 取结果 / 异常路径
# ---------------------------------------------------------------------------


def test_empty_everything_returns_none_false():
    # scheduler 无请求且队空 → (None, False)
    core, _, _ = make_core(max_concurrent_batches=2)
    out, executed = core.step_with_batch_queue()
    assert out is None
    assert executed is False


def test_sample_none_raises_underlying_exec_error():
    # 采样 future 返回 None 表示底层 execute_model 失败 → exec_model_fut.result() 抛真异常
    sched = FakeScheduler()

    class FailingExecutor(FakeExecutor):
        def execute_model(self, scheduler_output, non_block=False):
            f = Future()
            f.set_exception(RuntimeError("execute_model boom"))
            return f

        def sample_tokens(self, grammar_output, non_block=False):
            f = Future()
            f.set_result(None)  # 信号：前序 execute_model 失败
            return f

    execu = FailingExecutor(max_concurrent_batches=2)
    core = _MinimalEngineCore(model_executor=execu, scheduler=sched)
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=5, batch_id=1))
    # 队未满，但队尾 future（None 的 sample future）.done() 为 True → 落到 pop
    with pytest.raises(RuntimeError, match="execute_model boom"):
        core.step_with_batch_queue()


def test_no_tokens_scheduled_no_sampling():
    # total_num_scheduled_tokens==0 → model_executed False → 走 not model_executed 支，
    # future = exec_future（不调 sample_tokens / get_grammar_bitmask）
    core, execu, sched = make_core(max_concurrent_batches=2)
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=0, batch_id=1))
    out, executed = core.step_with_batch_queue()
    assert executed is False
    assert sched.get_grammar_bitmask_calls == []


# ---------------------------------------------------------------------------
# deferred sampling（结构化输出）
# ---------------------------------------------------------------------------


def test_immediate_sampling_when_no_pending_tokens():
    # 非 pending → 上半段立即 get_grammar_bitmask + sample_tokens
    core, execu, sched = make_core(max_concurrent_batches=3)
    sched.queue(
        SchedulerOutput(
            total_num_scheduled_tokens=5,
            pending_structured_output_tokens=False,
            batch_id=1,
        )
    )
    core.step_with_batch_queue()
    # 立即采样路径：上半段调用过一次 get_grammar_bitmask
    assert len(sched.get_grammar_bitmask_calls) == 1


def _seed_prior_inflight_batch(core, batch_id):
    """构造『上一步已调度、结果待取』的在飞批，直接 appendleft 进队列。

    这复现真实 vLLM 的不变量：deferred 批之所以叫 deferred，是『推迟到处理完上一步
    输出之后再采样』——所以队列里必然已有一个先前在飞的批供下半段 pop。这里手动 seed
    该前序状态，再调度 deferred 批，聚焦观察下半段的 deferred 兑现逻辑。"""
    from implementation.engine_core_batch_queue import ModelRunnerOutput

    prior_sched = SchedulerOutput(total_num_scheduled_tokens=5, batch_id=batch_id)
    sample_fut = Future()
    sample_fut.set_result(ModelRunnerOutput(batch_id=batch_id))
    exec_fut = Future()
    exec_fut.set_result(ModelRunnerOutput(batch_id=batch_id))
    core.batch_queue.appendleft((sample_fut, prior_sched, exec_fut))


def test_deferred_sampling_postpones_then_recovers():
    # pending_structured_output_tokens=True → 上半段不采样不入队（deferred_scheduler_output 置位），
    # 落到下半段：先 pop 取上一批（在飞）结果，再补 get_grammar_bitmask + sample_tokens + appendleft 重入队。
    sched = FakeScheduler()
    execu = FakeExecutor(max_concurrent_batches=3)
    core = _MinimalEngineCore(model_executor=execu, scheduler=sched)

    # seed 一个先前在飞的批（队列非空是 deferred 路径的前提）
    _seed_prior_inflight_batch(core, batch_id=1)

    # 调度一个 deferred 批
    sched.queue(
        SchedulerOutput(
            total_num_scheduled_tokens=5,
            has_structured_output_requests=True,
            pending_structured_output_tokens=True,
            batch_id=2,
        )
    )

    before = len(sched.get_grammar_bitmask_calls)
    out, executed = core.step_with_batch_queue()
    # 下半段先 pop 出前序批1 并 update_from_output
    assert sched.update_from_output_calls[0][0].batch_id == 1
    # deferred 批不在上半段采样，但下半段（pop 批之后）会补一次 get_grammar_bitmask
    assert len(sched.get_grammar_bitmask_calls) > before
    # deferred 批被重新 appendleft 入队，等待后续 pop
    assert any(t[1].batch_id == 2 for t in core.batch_queue)


def test_deferred_with_spec_decode_takes_draft_tokens():
    # deferred + use_spec_decode → 下半段先 take_draft_token_ids + update_draft_token_ids_in_output
    sched = FakeScheduler()
    execu = FakeExecutor(max_concurrent_batches=3)
    execu.draft_token_ids = [[1, 2, -1]]  # 非 None，满足 assert
    core = _MinimalEngineCore(
        model_executor=execu, scheduler=sched, use_spec_decode=True
    )
    # seed 前序在飞批供下半段 pop
    _seed_prior_inflight_batch(core, batch_id=1)

    # 调度 deferred 批
    sched.queue(
        SchedulerOutput(
            total_num_scheduled_tokens=5,
            has_structured_output_requests=True,
            pending_structured_output_tokens=True,
            batch_id=9,
        )
    )
    core.step_with_batch_queue()

    # deferred + spec decode：下半段先 take_draft_token_ids 再 update_draft_token_ids_in_output
    assert len(sched.update_draft_calls) == 1
    assert sched.update_draft_calls[0][0] == [[1, 2, -1]]


# ---------------------------------------------------------------------------
# has_work 保活
# ---------------------------------------------------------------------------


def test_has_work_true_when_queue_nonempty():
    core, execu, sched = make_core(max_concurrent_batches=3)
    execu_pending = PendingExecutor(max_concurrent_batches=3)
    core = _MinimalEngineCore(model_executor=execu_pending, scheduler=sched)
    sched.queue(SchedulerOutput(total_num_scheduled_tokens=5, batch_id=1))
    core.step_with_batch_queue()  # 入队 (None, True)
    # 现在 scheduler 没有更多请求，但队列非空 → has_work 仍 True
    assert sched.has_requests() is False
    assert bool(core.batch_queue) is True
    assert core.has_work() is True


def test_has_work_false_when_idle():
    core, _, _ = make_core(max_concurrent_batches=2)
    assert core.has_work() is False
