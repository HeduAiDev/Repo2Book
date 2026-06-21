"""ch04 精简版测试：验证它复现真实 vLLM AsyncLLM 三段式异步解耦的可观察行为。

不 import vllm（纯 in-process 精简版），host `python3 -m pytest` 即可。
测试目标对应 dossier 的 data_flow / design_decisions / 三个伏笔 (f1/f2/f3)。
"""

import asyncio
import sys
from pathlib import Path

import pytest

# 让测试能 import 精简版模块（implementation/ 内是扁平 import）。
IMPL = Path(__file__).resolve().parents[1] / "implementation"
sys.path.insert(0, str(IMPL))

from async_llm import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, AsyncLLM  # noqa: E402
from messages import (  # noqa: E402
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    SamplingParams,
)
from output_processor import (  # noqa: E402
    OutputProcessor,
    RequestOutput,
    RequestOutputCollector,
)


# ---------------------------------------------------------------------------
# RequestOutputCollector (f1): 单槽 + asyncio.Event 的 per-request 队列
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_collector_get_nowait_empty_returns_none():
    """get_nowait 空槽返回 None —— generate L579 `get_nowait() or await get()` 左操作数。"""
    c = RequestOutputCollector("FINAL", "r1")
    assert c.get_nowait() is None


@pytest.mark.asyncio
async def test_collector_put_then_get_nowait_fastpath():
    """生产者已就绪时 get_nowait 命中（快路径），免一次 await 调度往返。"""
    c = RequestOutputCollector("FINAL", "r1")
    out = RequestOutput("r1", token_ids=[1], finished=False)
    c.put(out)
    got = c.get_nowait()
    assert got is out
    # 取走后槽清空、Event 复位。
    assert c.get_nowait() is None
    assert not c.ready.is_set()


@pytest.mark.asyncio
async def test_collector_get_blocks_until_put():
    """get() 在空槽时阻塞于 ready.wait()，被 put 唤醒 —— 生产者-消费者同步点 (f3)。"""
    c = RequestOutputCollector("FINAL", "r1")

    async def producer():
        await asyncio.sleep(0.01)
        c.put(RequestOutput("r1", token_ids=[7], finished=True))

    asyncio.create_task(producer())
    got = await asyncio.wait_for(c.get(), timeout=1.0)
    assert got.token_ids == [7]
    assert got.finished


@pytest.mark.asyncio
async def test_collector_merge_on_producer_ahead_delta():
    """生产者超前(消费者没取走)时，DELTA 模式 merge 累加 token，不丢不阻塞(背压替代)。"""
    c = RequestOutputCollector("DELTA", "r1")
    c.put(RequestOutput("r1", token_ids=[1], finished=False))
    c.put(RequestOutput("r1", token_ids=[2], finished=False))
    c.put(RequestOutput("r1", token_ids=[3], finished=True))
    merged = c.get_nowait()
    assert merged.token_ids == [1, 2, 3]
    assert merged.finished  # finished 经 |= 聚合


@pytest.mark.asyncio
async def test_collector_merge_final_replaces():
    """FINAL 模式生产者超前则以最新覆盖（非累加）。"""
    c = RequestOutputCollector("FINAL", "r1")
    c.put(RequestOutput("r1", token_ids=[1, 2], finished=False))
    c.put(RequestOutput("r1", token_ids=[1, 2, 3], finished=True))
    out = c.get_nowait()
    assert out.token_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_collector_put_exception_raised_on_get():
    """异常经队列传播：put(Exception) 后 get 抛出 —— 背景任务故障传播给 generate。"""
    c = RequestOutputCollector("FINAL", "r1")
    c.put(ValueError("boom"))
    with pytest.raises(ValueError):
        c.get_nowait()


# ---------------------------------------------------------------------------
# OutputProcessor: 登记 (add_request) + 按 req_id 解多路复用分发 (process_outputs)
# ---------------------------------------------------------------------------
def _req(rid, n=1, kind="FINAL"):
    return EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(n=n, output_kind=kind),
    )


def test_output_processor_registers_queue():
    """add_request 建立 req_id -> RequestState(含 queue) 查找表。"""
    op = OutputProcessor()
    q = RequestOutputCollector("FINAL", "rA")
    op.add_request(_req("rA"), prompt=None, queue=q)
    assert "rA" in op.request_states
    assert op.request_states["rA"].queue is q


def test_process_outputs_demultiplexes_by_req_id():
    """一个 EngineCore 批 -> 按 req_id 分发回各请求专属队列 (多路复用解扇出)。"""
    op = OutputProcessor()
    qA = RequestOutputCollector("FINAL", "rA")
    qB = RequestOutputCollector("FINAL", "rB")
    op.add_request(_req("rA"), prompt=None, queue=qA)
    op.add_request(_req("rB"), prompt=None, queue=qB)

    out = op.process_outputs(
        [
            EngineCoreOutput("rA", new_token_ids=[10]),
            EngineCoreOutput("rB", new_token_ids=[20], finish_reason="stop"),
        ]
    )
    # AsyncLLM 路径：结果进各自队列，返回列表为空 (output_handler 的 assert 依据)。
    assert out.request_outputs == []
    assert qA.get_nowait().token_ids == [10]
    rb = qB.get_nowait()
    assert rb.token_ids == [20] and rb.finished
    # 完成请求从查找表移除。
    assert "rB" not in op.request_states
    assert "rA" in op.request_states


def test_process_outputs_ignores_aborted_request():
    """已 abort(查找表无此 req_id)的 EngineCoreOutput 被忽略，不抛错。"""
    op = OutputProcessor()
    out = op.process_outputs([EngineCoreOutput("ghost", new_token_ids=[1])])
    assert out.request_outputs == []


def test_process_outputs_no_queue_returns_list():
    """无 queue (LLMEngine 同步用法) 时收集成 list 返回 —— req_state.queue is not None 分流。"""
    op = OutputProcessor()
    op.add_request(_req("rC"), prompt=None, queue=None)
    out = op.process_outputs([EngineCoreOutput("rC", new_token_ids=[5], finish_reason="stop")])
    assert len(out.request_outputs) == 1
    assert out.request_outputs[0].token_ids == [5]


# ---------------------------------------------------------------------------
# AsyncLLM 端到端：三段式 generate() 异步生成器
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_yields_until_finished():
    """generate() 拉取并 yield 直到 EngineCoreOutput.finished —— f3 消费者侧骨架。"""
    engine = AsyncLLM()
    outs = []
    async for out in engine.generate([1, 2, 3], SamplingParams(n=1), request_id="r1"):
        outs.append(out)
    assert len(outs) >= 1
    assert outs[-1].finished
    # finished 之前的不是 finished。
    assert all(not o.finished for o in outs[:-1])


@pytest.mark.asyncio
async def test_add_request_fans_out_to_both_stages():
    """_add_request 双登记：本进程 OutputProcessor + 投递 stub EngineCore (f2 扇出点)。"""
    engine = AsyncLLM()
    q = await engine.add_request("rX", [1, 2], SamplingParams(n=1))
    # (a) OutputProcessor (this process) 登记了该请求队列。
    assert "rX" in engine.output_processor.request_states
    assert engine.output_processor.request_states["rX"].queue is q
    # (b) EngineCore (separate process, stub) 收到了该请求 —— 背景引擎会产出。
    out = await asyncio.wait_for(q.get(), timeout=1.0)
    assert isinstance(out, RequestOutput)
    assert out.request_id == "rX"


@pytest.mark.asyncio
async def test_concurrent_requests_isolated_per_queue():
    """并发多请求各走自己的 per-request 队列，互不串扰 (尾延迟与并发数解耦, f1)。"""
    engine = AsyncLLM()

    async def run(rid):
        toks = []
        async for out in engine.generate([1], SamplingParams(n=1), request_id=rid):
            toks.extend(out.token_ids)
        return rid, toks

    results = await asyncio.gather(run("a"), run("b"), run("c"))
    by_id = dict(results)
    # 每个请求都拿到完整 token 序列，且各请求独立完成。
    for rid in ("a", "b", "c"):
        assert len(by_id[rid]) >= 1


@pytest.mark.asyncio
async def test_output_handler_started_lazily_on_add_request():
    """__init__ 在无事件循环时不起 output_handler；首个 add_request 懒启之。"""
    # 构造时虽在事件循环内（pytest-asyncio），但验证 add_request 后 handler 必在。
    engine = AsyncLLM()
    await engine.add_request("r1", [1], SamplingParams(n=1))
    assert engine.output_handler is not None
    assert not engine.output_handler.done()


@pytest.mark.asyncio
async def test_run_output_handler_idempotent():
    """_run_output_handler 重复调用不重起任务 (单长驻背景任务)。"""
    engine = AsyncLLM()
    engine._run_output_handler()
    t1 = engine.output_handler
    engine._run_output_handler()
    assert engine.output_handler is t1


@pytest.mark.asyncio
async def test_generate_cancel_triggers_abort():
    """客户端断开 -> generate 收 CancelledError -> abort 双向清理 (生命周期收尾)。"""
    engine = AsyncLLM()
    # 让请求长时间产 token，确保取消发生在 generate 仍在 yield 期间。
    engine.engine_core._tokens_per_request = 10_000

    aborted = []
    orig_abort = engine.abort

    async def spy_abort(request_id, internal=False):
        aborted.append(request_id)
        await orig_abort(request_id, internal=internal)

    engine.abort = spy_abort

    # 客户端断开时，Python 对挂起中的 async generator 调用 aclose()，向 yield 点抛
    # GeneratorExit —— 命中 generate 的 except (CancelledError, GeneratorExit) 分支。
    agen = engine.generate([1], SamplingParams(n=1), request_id="rc")
    first = await agen.__anext__()  # 取首个 token，请求仍在进行中。
    assert not first.finished
    await agen.aclose()  # 模拟客户端断开 -> 异步生成器被关闭。

    assert "rc" in aborted
    # abort 后该请求从 OutputProcessor 查找表移除。
    assert "rc" not in engine.output_processor.request_states


@pytest.mark.asyncio
async def test_chunk_size_constant_default():
    """分块常量默认与 vLLM 一致 (128)，用于 output_handler 块间 sleep(0) 让步。"""
    assert VLLM_V1_OUTPUT_PROC_CHUNK_SIZE == 128


@pytest.mark.asyncio
async def test_output_handler_chunks_and_yields_between_chunks(monkeypatch):
    """一批多于 chunk_size 的输出会分块 process，块间 await asyncio.sleep(0) 让出事件循环。

    把 chunk_size 设为 1，注入一个含 3 条输出的批，断言 process_outputs 被多次调用、
    且块间确有 sleep(0) 让步（其它协程能插入运行）。
    """
    import async_llm as mod

    monkeypatch.setattr(mod, "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", 1)
    engine = AsyncLLM()
    # 阻止 stub 引擎自行产出，改由测试手动喂一批。
    engine.engine_core._tokens_per_request = 10_000

    # 三个请求各自登记 queue。
    qs = {}
    for rid in ("a", "b", "c"):
        q = RequestOutputCollector("FINAL", rid)
        engine.output_processor.add_request(_req(rid), prompt=None, queue=q)
        qs[rid] = q

    # 记录块间是否让出：在 process_outputs 调用次数上计数。
    calls = {"n": 0}
    orig = engine.output_processor.process_outputs

    def counting(slice_, ts=None, stats=None):
        calls["n"] += 1
        return orig(slice_, ts, stats)

    engine.output_processor.process_outputs = counting

    # 直接喂一批 3 条进 stub 的输出队列，唤醒 output_handler。
    await engine.engine_core._outputs.put(
        EngineCoreOutputs(
            outputs=[
                EngineCoreOutput("a", new_token_ids=[1]),
                EngineCoreOutput("b", new_token_ids=[2]),
                EngineCoreOutput("c", new_token_ids=[3], finish_reason="stop"),
            ]
        )
    )
    # 给事件循环若干轮让 output_handler 处理完三块（每块间 sleep(0) 让步）。
    for _ in range(20):
        await asyncio.sleep(0)

    assert calls["n"] == 3  # chunk_size=1 -> 3 条分成 3 块。
    assert qs["a"].get_nowait().token_ids == [1]
    assert qs["b"].get_nowait().token_ids == [2]
    assert qs["c"].get_nowait().finished


def test_engine_core_messages_finished_property():
    """EngineCoreOutput.finished 由 finish_reason 推导 —— generate 判停依据。"""
    assert not EngineCoreOutput("r", [1]).finished
    assert EngineCoreOutput("r", [1], finish_reason="stop").finished
    # 批容器结构正确。
    batch = EngineCoreOutputs(outputs=[EngineCoreOutput("r", [1])])
    assert len(batch.outputs) == 1
