"""TDD tests for ch14 — preemption loop + request lifecycle回流.

These tests pin the *observable behavior of real vLLM v1*（按 dossier 记录），
不是测精简版自洽。精简版纯单元（不 import vllm）→ host python3 -m pytest。

覆盖：
  - RUNNING 阶段 allocate_slots 失败 → FCFS LIFO 抢占（self.running.pop() 抢末尾）
  - _preempt_request 副作用：free KV / status→PREEMPTED / num_computed_tokens=0 /
    清 spec / num_preemptions++ / waiting 回队头（prepend）
  - 抢占终止：preempted_req == request 则 break（把自己都抢了仍分不到）
  - WAITING 守卫：本拍发生过抢占（preempted_reqs 非空）就完全跳过 WAITING
  - 双队列防队头阻塞：阻塞态进 skipped_waiting，遍历跳过不卡后续
  - 抢占请求回流落点：status==PREEMPTED → scheduled_resumed_reqs
  - check_stop：EOS / stop_token_ids / length / repetition 的 token 级停止
  - _update_request_with_output：逐 token append + check_stop + 截断
  - update_from_output：spec 回退、停止分流、批量摘除
  - _handle_stopped_request / _free_request / _free_blocks 完成态闭环
  - AsyncScheduler num_output_placeholders 回扣 + discard_latest_async_tokens
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation"))

from async_scheduler import AsyncScheduler  # noqa: E402
from request import Request, RequestStatus  # noqa: E402
from request_queue import FCFSRequestQueue, SchedulingPolicy  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from utils import check_stop, remove_all  # noqa: E402


# ---------- helpers ----------

def make_request(req_id, prompt_len=4, max_tokens=100, min_tokens=0,
                 eos_token_id=None, stop_token_ids=None):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        eos_token_id=eos_token_id,
        stop_token_ids=stop_token_ids or [],
    )


def make_scheduler(block_capacity=8, max_num_running_reqs=8, cls=Scheduler):
    return cls(
        block_capacity=block_capacity,
        max_num_running_reqs=max_num_running_reqs,
        max_model_len=1024,
    )


# ---------- RequestStatus state machine ----------

def test_status_is_finished_uses_greater_than_preempted():
    # is_finished = status > PREEMPTED（IntEnum 顺序排布的核心设计）
    assert not RequestStatus.is_finished(RequestStatus.WAITING)
    assert not RequestStatus.is_finished(RequestStatus.RUNNING)
    assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_REPETITION)


def test_finished_reason_map():
    from request import FinishReason
    assert RequestStatus.get_finished_reason(
        RequestStatus.FINISHED_STOPPED) == FinishReason.STOP
    assert RequestStatus.get_finished_reason(
        RequestStatus.FINISHED_LENGTH_CAPPED) == FinishReason.LENGTH
    assert RequestStatus.get_finished_reason(
        RequestStatus.FINISHED_REPETITION) == FinishReason.REPETITION
    assert RequestStatus.get_finished_reason(RequestStatus.RUNNING) is None


# ---------- check_stop (token-level stop) ----------

def test_check_stop_eos():
    r = make_request("a", eos_token_id=42)
    r.append_output_token_ids(42)
    assert check_stop(r, max_model_len=1024) is True
    assert r.status == RequestStatus.FINISHED_STOPPED


def test_check_stop_stop_token_id_sets_stop_reason():
    r = make_request("a", stop_token_ids=[7])
    r.append_output_token_ids(7)
    assert check_stop(r, max_model_len=1024) is True
    assert r.status == RequestStatus.FINISHED_STOPPED
    assert r.stop_reason == 7


def test_check_stop_min_tokens_gate():
    # min_tokens 未达：即便命中 EOS 也不停
    r = make_request("a", min_tokens=3, eos_token_id=42)
    r.append_output_token_ids(42)
    assert check_stop(r, max_model_len=1024) is False
    assert r.status == RequestStatus.WAITING  # 未改


def test_check_stop_length_capped_by_max_tokens():
    r = make_request("a", max_tokens=2)
    r.append_output_token_ids(100)
    assert check_stop(r, max_model_len=1024) is False
    r.append_output_token_ids(101)
    assert check_stop(r, max_model_len=1024) is True
    assert r.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_check_stop_eos_takes_priority_over_length():
    # 同一 token 既是 EOS 又达 length 上限：EOS（STOPPED）优先
    r = make_request("a", max_tokens=1, eos_token_id=42)
    r.append_output_token_ids(42)
    assert check_stop(r, max_model_len=1024) is True
    assert r.status == RequestStatus.FINISHED_STOPPED  # 不是 LENGTH_CAPPED


# ---------- _update_request_with_output: trim after stop ----------

def test_update_request_with_output_trims_after_stop():
    sched = make_scheduler()
    r = make_request("a", eos_token_id=99)
    new_ids, stopped = sched._update_request_with_output(r, [1, 2, 99, 3, 4])
    assert stopped is True
    # 停止 token 之后的不再返回
    assert new_ids == [1, 2, 99]


def test_update_request_with_output_no_stop_returns_all():
    sched = make_scheduler()
    r = make_request("a", max_tokens=100)
    new_ids, stopped = sched._update_request_with_output(r, [1, 2, 3])
    assert stopped is False
    assert new_ids == [1, 2, 3]


# ---------- remove_all ----------

def test_remove_all_single_item_inplace():
    lst = [1, 2, 3]
    out = remove_all(lst, {2})
    assert out == [1, 3]
    assert out is lst  # 单元素快路径原地


def test_remove_all_multi_item_new_list():
    lst = [1, 2, 3, 4]
    out = remove_all(lst, {2, 4})
    assert out == [1, 3]


# ---------- FCFS queue prepend semantics ----------

def test_fcfs_prepend_goes_to_front():
    q = FCFSRequestQueue()
    a, b, c = make_request("a"), make_request("b"), make_request("c")
    q.add_request(a)
    q.add_request(b)
    q.prepend_request(c)  # 抢占回队头
    assert q.peek_request() is c
    assert list(q) == [c, a, b]


# ---------- preemption loop (LIFO) ----------

def test_preemption_fcfs_lifo_pops_running_tail():
    # block_capacity=2：两个请求各占 1 块就满了；第三个进来需抢占。
    sched = make_scheduler(block_capacity=2, max_num_running_reqs=8)
    r0, r1 = make_request("r0"), make_request("r1")
    # 手动放入 running 并占块（模拟已调度）
    for r in (r0, r1):
        r.status = RequestStatus.RUNNING
        sched.running.append(r)
        sched.requests[r.request_id] = r
        sched.kv_cache_manager.allocate_slots(r, 1)

    # 现在为一个新请求 r2 在 running 里腾位：直接驱动抢占循环
    r2 = make_request("r2")
    r2.status = RequestStatus.RUNNING
    sched.requests[r2.request_id] = r2
    preempted = sched._run_preemption_loop_for(r2, num_new_tokens=1)

    # LIFO：抢的是 running 末尾 r1
    assert r1 in preempted
    assert r1.status == RequestStatus.PREEMPTED
    # 被抢者回 waiting 队头
    assert sched.waiting.peek_request() is r1
    # 抢占副作用
    assert r1.num_computed_tokens == 0
    assert r1.num_preemptions == 1


def test_preempt_request_side_effects():
    sched = make_scheduler()
    r = make_request("r")
    r.status = RequestStatus.RUNNING
    r.num_computed_tokens = 5
    r.spec_token_ids = [1, 2]
    sched.kv_cache_manager.allocate_slots(r, 1)
    used_before = sched.kv_cache_manager.num_free_blocks

    sched._preempt_request(r, timestamp=0.0)

    assert r.status == RequestStatus.PREEMPTED
    assert r.num_computed_tokens == 0
    assert r.spec_token_ids == []
    assert r.num_preemptions == 1
    assert sched.waiting.peek_request() is r
    # KV 被释放（free block 数回升）
    assert sched.kv_cache_manager.num_free_blocks > used_before


def test_preemption_terminates_when_self_is_only_victim():
    # 容量 1，running 里只有目标请求自己 → 抢到自己就 break
    sched = make_scheduler(block_capacity=1)
    r = make_request("r")
    r.status = RequestStatus.RUNNING
    sched.running.append(r)
    sched.requests[r.request_id] = r
    # r 自己占满唯一的块
    sched.kv_cache_manager.allocate_slots(r, 1)
    # 再为 r 申请更多（已无空块）→ 抢占循环抢到自己后 break，无法调度
    new_blocks, preempted = sched._allocate_with_preemption(r, num_new_tokens=1)
    assert new_blocks is None
    assert r in preempted  # 把自己也抢了


# ---------- schedule(): WAITING guard skipped after preemption ----------

def test_waiting_skipped_when_preemption_happened():
    sched = make_scheduler(block_capacity=1)
    # running 里一个请求，占满块
    r0 = make_request("r0")
    r0.status = RequestStatus.RUNNING
    sched.running.append(r0)
    sched.requests["r0"] = r0
    sched.kv_cache_manager.allocate_slots(r0, 1)
    r0.num_computed_tokens = 1

    # waiting 里有新请求
    rw = make_request("rw")
    sched.add_request(rw)

    out = sched.schedule()
    # r0 分不到新块 → 抢占自己 → 本拍发生抢占 → WAITING 完全跳过
    assert out.preempted_req_ids  # 发生过抢占
    assert "rw" not in out.scheduled_new_reqs  # 新请求未被调度


# ---------- dual queue: anti head-of-line blocking ----------

def test_blocked_request_skipped_does_not_starve_others():
    sched = make_scheduler(block_capacity=8)
    blocked = make_request("blocked")
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS  # 阻塞态
    ready = make_request("ready")
    # 阻塞态进 skipped_waiting，可调度态进 waiting
    sched._enqueue_waiting_request(blocked)
    sched._enqueue_waiting_request(ready)
    assert sched.skipped_waiting.peek_request() is blocked
    assert sched.waiting.peek_request() is ready

    out = sched.schedule()
    # ready 被调度（未被 blocked 卡住队头）
    assert "ready" in out.scheduled_new_reqs
    # blocked 仍在 skipped（被跳过，回 skipped_waiting）
    assert any(r.request_id == "blocked" for r in sched.skipped_waiting)


def test_enqueue_routes_by_blocked_status():
    sched = make_scheduler()
    normal = make_request("n")
    blocked = make_request("b")
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    sched._enqueue_waiting_request(normal)
    sched._enqueue_waiting_request(blocked)
    assert normal in list(sched.waiting)
    assert blocked in list(sched.skipped_waiting)


def test_select_waiting_queue_fcfs_prefers_skipped():
    sched = make_scheduler()
    sched.waiting.add_request(make_request("w"))
    sched.skipped_waiting.add_request(make_request("s"))
    q = sched._select_waiting_queue_for_scheduling()
    assert q is sched.skipped_waiting


# ---------- preempted request resumed落点 ----------

def test_preempted_request_resumed_lands_in_resumed_reqs():
    sched = make_scheduler(block_capacity=8)
    r = make_request("r")
    r.status = RequestStatus.PREEMPTED  # 被抢占回流后
    r.num_computed_tokens = 0
    sched.waiting.prepend_request(r)
    out = sched.schedule()
    # PREEMPTED → scheduled_resumed_reqs（不是 new）
    assert "r" in out.scheduled_resumed_reqs
    assert "r" not in out.scheduled_new_reqs


def test_waiting_request_lands_in_new_reqs():
    sched = make_scheduler(block_capacity=8)
    r = make_request("r")  # status WAITING
    sched.add_request(r)
    out = sched.schedule()
    assert "r" in out.scheduled_new_reqs
    assert "r" not in out.scheduled_resumed_reqs


# ---------- update_from_output: stop分流 + 摘除 + free ----------

def test_update_from_output_running_request_stops_and_frees():
    sched = make_scheduler(block_capacity=8)
    r = make_request("r", eos_token_id=99, max_tokens=100)
    sched.add_request(r)
    sched.schedule()  # 拉进 running
    assert r in sched.running

    # 模型吐出 EOS → 停止 → 真完成 → free
    sched.update_from_output({"r": [5, 99, 6]})
    assert r.status == RequestStatus.FINISHED_STOPPED
    assert r not in sched.running                 # 从 running 摘除
    assert "r" in sched.finished_req_ids          # 登记完成
    assert "r" not in sched.requests              # 从 requests 字典删除
    # 停止 token 之后的被截断
    assert r.output_token_ids == [5, 99]


def test_update_from_output_skips_already_finished():
    sched = make_scheduler()
    r = make_request("r")
    r.status = RequestStatus.FINISHED_STOPPED
    sched.requests["r"] = r
    # 不应抛错；已 finished 跳过
    sched.update_from_output({"r": [1, 2]})
    assert r.output_token_ids == []  # 未追加


def test_update_from_output_spec_rejection_rolls_back():
    sched = make_scheduler(block_capacity=8)
    r = make_request("r", max_tokens=100)
    r.status = RequestStatus.RUNNING
    r.num_computed_tokens = 10
    sched.requests["r"] = r
    sched.running.append(r)
    # 草稿 3 个 spec token，仅接受 1 个（generated 长度 = accepted+1 = 2）
    sched.update_from_output(
        {"r": [1, 2]},
        scheduled_spec_decode_tokens={"r": [10, 11, 12]},
    )
    # num_rejected = 3 - (2-1) = 2 → num_computed_tokens 回扣 2
    assert r.num_computed_tokens == 10 - 2


def test_handle_stopped_request_returns_true_non_resumable():
    sched = make_scheduler()
    r = make_request("r")
    r.status = RequestStatus.FINISHED_STOPPED
    assert sched._handle_stopped_request(r) is True


def test_free_request_registers_and_frees():
    sched = make_scheduler(block_capacity=8)
    r = make_request("r")
    r.status = RequestStatus.FINISHED_STOPPED
    sched.requests["r"] = r
    sched.kv_cache_manager.allocate_slots(r, 1)
    free_before = sched.kv_cache_manager.num_free_blocks
    sched._free_request(r)
    assert "r" in sched.finished_req_ids
    assert "r" not in sched.requests
    assert sched.kv_cache_manager.num_free_blocks > free_before


# ---------- AsyncScheduler placeholder回扣 ----------

def test_async_placeholders_decrement_by_actual_tokens():
    sched = make_scheduler(block_capacity=8, cls=AsyncScheduler)
    r = make_request("r", max_tokens=100)
    r.status = RequestStatus.RUNNING
    r.num_output_placeholders = 3
    new_ids, stopped = sched._update_request_with_output(r, [1, 2])
    assert new_ids == [1, 2]
    # 占位随实际 token 回扣
    assert r.num_output_placeholders == 3 - 2


def test_async_discard_latest_async_tokens():
    sched = make_scheduler(block_capacity=8, cls=AsyncScheduler)
    r = make_request("r")
    r.discard_latest_async_tokens = True
    new_ids, stopped = sched._update_request_with_output(r, [1, 2, 3])
    # 强制抢占下的在途 token 被丢弃
    assert new_ids == []
    assert stopped is False
    assert r.discard_latest_async_tokens is False
