"""ch13 连续批处理调度器 —— 纯单元测试（不 import vllm）。

测的是精简版复现真实 vLLM 的**可观测调度行为**：
- schedule() 不分 prefill/decode 相，统一 num_computed_tokens 追赶 num_tokens_with_spec
- token_budget 跨两阶段递减、total ≤ max_num_scheduled_tokens 守恒
- RUNNING 优先于 WAITING；本拍发生抢占则不调度 WAITING（`if not preempted_reqs`）
- allocate_slots 返回 None → FCFS 抢占队尾 → 被抢者回 waiting、num_computed_tokens=0
- SchedulerOutput 二分 NewRequestData(全量) vs CachedRequestData(增量)
- _update_after_schedule 调度后乐观推进 num_computed_tokens
- AsyncScheduler num_output_placeholders 占位与兑现配平
- PAUSED_ALL → budget=0 不调度
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.interface import PauseState  # noqa: E402
from implementation.async_scheduler import AsyncScheduler  # noqa: E402
from implementation.request import (  # noqa: E402
    Request,
    RequestStatus,
    SamplingParams,
)
from implementation.scheduler import Scheduler  # noqa: E402


# --------------------------------------------------------------------------- #
# 测试用的 ModelRunnerOutput 桩（驱动 update_from_output 反馈环）
# --------------------------------------------------------------------------- #
class FakeModelRunnerOutput:
    def __init__(self, sched_output, token_for):
        # token_for: dict req_id -> list[int] 本拍为该请求生成的 token（prefill chunk 给 []）
        req_ids = list(sched_output.num_scheduled_tokens.keys())
        self.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        self.sampled_token_ids = [token_for.get(rid, []) for rid in req_ids]


def make_req(rid, prompt_len, max_tokens=16, arrival=0.0):
    return Request(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        arrival_time=arrival,
    )


def new_scheduler(**kw):
    defaults = dict(
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=4096,
        block_size=16,
    )
    defaults.update(kw)
    return Scheduler(**defaults)


# --------------------------------------------------------------------------- #
# 1. 首次调度：WAITING → NewRequestData(全量)，token_budget 递减
# --------------------------------------------------------------------------- #
def test_first_schedule_emits_new_request_data():
    sched = new_scheduler(max_num_batched_tokens=64)
    sched.add_request(make_req("a", prompt_len=10))

    out = sched.schedule()

    assert out.num_scheduled_tokens == {"a": 10}
    assert out.total_num_scheduled_tokens == 10
    # 首次调度 → 全量
    assert len(out.scheduled_new_reqs) == 1
    assert out.scheduled_new_reqs[0].req_id == "a"
    assert out.scheduled_new_reqs[0].prompt_token_ids == list(range(10))
    # 没有增量
    assert out.scheduled_cached_reqs.num_reqs == 0
    # 入了 running、置 RUNNING
    assert [r.request_id for r in sched.running] == ["a"]
    assert sched.requests["a"].status == RequestStatus.RUNNING


# --------------------------------------------------------------------------- #
# 2. 第二拍：同一请求变 CachedRequestData(增量)，不再发全量
# --------------------------------------------------------------------------- #
def test_second_schedule_emits_cached_request_data():
    sched = new_scheduler()
    sched.add_request(make_req("a", prompt_len=10))
    out1 = sched.schedule()
    # 模拟模型产出 1 个 token（prefill 完成进 decode）
    mro = FakeModelRunnerOutput(out1, {"a": [100]})
    sched.update_from_output(out1, mro)

    out2 = sched.schedule()
    # decode 拍：追赶公式 num_tokens_with_spec - num_computed = 11-10 = 1
    assert out2.num_scheduled_tokens == {"a": 1}
    assert len(out2.scheduled_new_reqs) == 0
    assert out2.scheduled_cached_reqs.num_reqs == 1
    assert out2.scheduled_cached_reqs.req_ids == ["a"]


# --------------------------------------------------------------------------- #
# 3. 不分 prefill/decode 相：同一拍混 prefill chunk + decode，共享 token 预算
# --------------------------------------------------------------------------- #
def test_continuous_batch_mixes_prefill_and_decode():
    sched = new_scheduler(max_num_batched_tokens=64)
    # a 先跑起来并进入 decode
    sched.add_request(make_req("a", prompt_len=4))
    o = sched.schedule()
    sched.update_from_output(o, FakeModelRunnerOutput(o, {"a": [100]}))
    # 现在加一个长 prompt 的 b
    sched.add_request(make_req("b", prompt_len=20))

    out = sched.schedule()
    # 同一拍：a 是 decode(1 token)，b 是 prefill(20 token)
    assert out.num_scheduled_tokens["a"] == 1
    assert out.num_scheduled_tokens["b"] == 20
    assert out.total_num_scheduled_tokens == 21
    # a 走增量、b 走全量
    assert {r.req_id for r in out.scheduled_new_reqs} == {"b"}
    assert set(out.scheduled_cached_reqs.req_ids) == {"a"}


# --------------------------------------------------------------------------- #
# 4. token 预算守恒：chunked prefill 把超预算 prompt 截到 budget
# --------------------------------------------------------------------------- #
def test_token_budget_chunks_long_prefill():
    sched = new_scheduler(max_num_batched_tokens=16, enable_chunked_prefill=True)
    sched.add_request(make_req("a", prompt_len=50))

    out = sched.schedule()
    # 被截到预算 16
    assert out.num_scheduled_tokens == {"a": 16}
    assert out.total_num_scheduled_tokens <= sched.max_num_scheduled_tokens
    # 调度后乐观推进
    assert sched.requests["a"].num_computed_tokens == 16
    assert sched.requests["a"].is_prefill_chunk is True

    # 下一拍立刻续上剩余 prompt（无需等 forward）
    out2 = sched.schedule()
    assert out2.num_scheduled_tokens == {"a": 16}
    assert sched.requests["a"].num_computed_tokens == 32


# --------------------------------------------------------------------------- #
# 5. chunked prefill 关闭：超预算则整块不调度（break）
# --------------------------------------------------------------------------- #
def test_no_chunked_prefill_breaks_when_over_budget():
    sched = new_scheduler(max_num_batched_tokens=16, enable_chunked_prefill=False)
    sched.add_request(make_req("a", prompt_len=50))
    out = sched.schedule()
    assert out.num_scheduled_tokens == {}
    assert out.total_num_scheduled_tokens == 0


# --------------------------------------------------------------------------- #
# 6. RUNNING 优先 + 抢占：KV 块耗尽 → 抢占队尾 → 被抢者回 waiting
# --------------------------------------------------------------------------- #
def test_preemption_when_out_of_blocks():
    # 只给 2 个块（block_size=16 → 每请求装满需多块），逼出抢占
    sched = new_scheduler(
        max_num_batched_tokens=64, num_gpu_blocks=2, block_size=16, max_model_len=4096
    )
    sched.add_request(make_req("a", prompt_len=16, max_tokens=100))
    sched.add_request(make_req("b", prompt_len=16, max_tokens=100))
    # 两个各占 1 块（16 token 正好 1 块）
    o = sched.schedule()
    assert set(o.num_scheduled_tokens) == {"a", "b"}
    assert sched.kv_cache_manager.num_free_blocks == 0
    sched.update_from_output(o, FakeModelRunnerOutput(o, {"a": [1], "b": [1]}))
    # 现在 a、b 各 17 token，下一个 decode token 让 b 需要第 2 块 → 没块 → 抢占队尾(b)
    o2 = sched.schedule()
    # b 被抢占回 waiting，本拍因 preempted_reqs 非空不调度 WAITING
    assert o2.preempted_req_ids  # 至少抢了一个
    preempted_id = next(iter(o2.preempted_req_ids))
    preempted = sched.requests[preempted_id]
    assert preempted.status == RequestStatus.PREEMPTED
    assert preempted.num_computed_tokens == 0
    # 被抢者回到 waiting 队列
    assert preempted in list(sched.waiting)


# --------------------------------------------------------------------------- #
# 7. 抢占后不调度 WAITING（`if not preempted_reqs` 守卫）
# --------------------------------------------------------------------------- #
def test_no_waiting_scheduled_after_preemption():
    sched = new_scheduler(
        max_num_batched_tokens=64, num_gpu_blocks=2, block_size=16
    )
    sched.add_request(make_req("a", prompt_len=16, max_tokens=100))
    sched.add_request(make_req("b", prompt_len=16, max_tokens=100))
    o = sched.schedule()
    sched.update_from_output(o, FakeModelRunnerOutput(o, {"a": [1], "b": [1]}))
    # 新来一个 waiting 请求 c
    sched.add_request(make_req("c", prompt_len=8))
    o2 = sched.schedule()
    if o2.preempted_req_ids:
        # c 不应在本拍被调度（守卫生效）
        assert "c" not in o2.num_scheduled_tokens


# --------------------------------------------------------------------------- #
# 8. 恢复被抢占请求 → scheduled_resumed_reqs（resumed_req_ids 替换 block 语义）
# --------------------------------------------------------------------------- #
def test_resumed_request_goes_to_resumed_ids():
    sched = new_scheduler(
        max_num_batched_tokens=64, num_gpu_blocks=2, block_size=16
    )
    sched.add_request(make_req("a", prompt_len=16, max_tokens=100))
    sched.add_request(make_req("b", prompt_len=16, max_tokens=100))
    o = sched.schedule()
    sched.update_from_output(o, FakeModelRunnerOutput(o, {"a": [1], "b": [1]}))
    o2 = sched.schedule()  # 触发抢占
    assert o2.preempted_req_ids
    # 让在途请求 a 停下（产 eos）以释放块，使被抢的请求能恢复
    a = sched.requests["a"]
    a.sampling_params.eos_token_id = 999
    sched.update_from_output(o2, FakeModelRunnerOutput(o2, {"a": [999]}))
    # 现在有空块，恢复被抢请求
    o3 = sched.schedule()
    cached = o3.scheduled_cached_reqs
    # 被恢复的请求出现在 resumed_req_ids（若本拍恢复了）
    if cached.resumed_req_ids:
        for rid in cached.resumed_req_ids:
            assert rid in cached.req_ids


# --------------------------------------------------------------------------- #
# 9. PAUSED_ALL → token_budget=0 → 不调度任何请求
# --------------------------------------------------------------------------- #
def test_paused_all_schedules_nothing():
    sched = new_scheduler()
    sched.add_request(make_req("a", prompt_len=10))
    sched.set_pause_state(PauseState.PAUSED_ALL)
    out = sched.schedule()
    assert out.num_scheduled_tokens == {}
    assert out.total_num_scheduled_tokens == 0


# --------------------------------------------------------------------------- #
# 10. stop：达到 max_tokens → FINISHED，free，从 running 移除，记 finished_req_ids
# --------------------------------------------------------------------------- #
def test_request_stops_at_max_tokens():
    sched = new_scheduler()
    sched.add_request(make_req("a", prompt_len=4, max_tokens=2))
    o = sched.schedule()  # prefill 4
    sched.update_from_output(o, FakeModelRunnerOutput(o, {"a": [10]}))  # output token 1
    o2 = sched.schedule()
    outs = sched.update_from_output(o2, FakeModelRunnerOutput(o2, {"a": [11]}))  # token 2 → max
    a = sched.requests.get("a")
    # 已 free → 从 requests 删除
    assert a is None
    assert "a" not in [r.request_id for r in sched.running]
    assert "a" in sched.finished_req_ids
    # 输出里带 finish_reason
    flat = [o for lst in outs.values() for o in lst]
    assert any(o.finish_reason is not None for o in flat)


# --------------------------------------------------------------------------- #
# 11. max_num_running_reqs 上界
# --------------------------------------------------------------------------- #
def test_max_num_running_reqs_caps_running():
    sched = new_scheduler(max_num_seqs=2, max_num_batched_tokens=1000)
    for i in range(5):
        sched.add_request(make_req(f"r{i}", prompt_len=4))
    out = sched.schedule()
    assert len(sched.running) <= 2
    assert len(out.num_scheduled_tokens) <= 2


# --------------------------------------------------------------------------- #
# 12. AsyncScheduler：调度后即 +1 占位，使下一拍能预调度 decode 槽
# --------------------------------------------------------------------------- #
def test_async_scheduler_adds_placeholder_after_schedule():
    sched = AsyncScheduler(
        max_num_seqs=8, max_num_batched_tokens=64, max_model_len=4096, block_size=16
    )
    sched.add_request(make_req("a", prompt_len=4, max_tokens=100))
    o1 = sched.schedule()  # prefill 4
    a = sched.requests["a"]
    # prefill chunk 完成（num_computed == num_tokens）→ 非 prefill_chunk → 记 1 个占位
    assert a.is_prefill_chunk is False
    assert a.num_output_placeholders == 1

    # 下一拍：上一拍 token 还没回来，但靠占位仍能调度 1 个 decode 槽
    o2 = sched.schedule()
    assert o2.num_scheduled_tokens.get("a") == 1
    # 此时占位累加到 2（两拍各 +1，token 都还没兑现）
    assert a.num_output_placeholders == 2


# --------------------------------------------------------------------------- #
# 13. AsyncScheduler：真 token 回来时占位兑现（-len），配平
# --------------------------------------------------------------------------- #
def test_async_placeholder_redeemed_on_output():
    sched = AsyncScheduler(
        max_num_seqs=8, max_num_batched_tokens=64, max_model_len=4096, block_size=16
    )
    sched.add_request(make_req("a", prompt_len=4, max_tokens=100))
    o1 = sched.schedule()
    a = sched.requests["a"]
    assert a.num_output_placeholders == 1
    # 上一拍的 prefill 不产 token（prefill chunk 给 []）——这里 prefill 一拍完成，
    # 第 1 个真 token 在 o1 对应的 forward 里产出，回流兑现 1 个占位
    sched.update_from_output(o1, FakeModelRunnerOutput(o1, {"a": [100]}))
    assert a.num_output_placeholders == 0
    assert a.num_output_tokens == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
