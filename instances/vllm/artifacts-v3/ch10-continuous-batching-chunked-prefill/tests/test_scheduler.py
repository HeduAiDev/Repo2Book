"""ch10 连续批处理与 chunked prefill —— 纯单元测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测调度行为**
（锚点 = vllm/v1/core/sched/scheduler.py 行号，基线 v0.27.1）：
- 迭代级调度契约：schedule() 产出 {req_id: num_tokens}（interface.py:L53-L67）
- 无 prefill/decode 相位：num_new_tokens = num_tokens_with_spec
  + num_output_placeholders − num_computed_tokens 追赶公式（scheduler.py:L516-L532）
- decode 稳态每拍恰 1 token（追赶公式的特例二）
- 单一 token 预算跨 RUNNING/WAITING 两阶段分账 + 守恒断言（L459/L523/L637/L913/L1073/L1108-L1113）
- RUNNING 先于 WAITING（L483）；本拍抢占过则整拍不收新（L684 守卫）
- chunked prefill 三闸：long_prefill_token_threshold / enable_chunked_prefill 开关
  整拍 break / min(token_budget)（L899-L913）
- allocate_slots None=不够：RUNNING 侧抢队尾重试（L576-L629）、
  WAITING 侧直接 break 绝不抢占（L987-L994）+ full_sequence_must_fit 全序列准入门（L982）
- 出队入 running：WAITING→RUNNING / PREEMPTED→resumed 分流（L1055-L1075）
- SchedulerOutput 二分 NewRequestData(全量)/CachedRequestData(增量)
  + prev_step_scheduled_req_ids 差量判定（L1131-L1163 / L1410-L1467）
- _update_after_schedule 乐观推进 + is_prefill_chunk 标记（L1317-L1343）
- PAUSED_ALL → token_budget=0 短路（L460-L462）
- 预算默认值：config 2048/128 基线 + arg_utils 按硬件仲裁表（config/scheduler.py:L42-L44,
  arg_utils.py:L2541-L2563）

⑤拍（update_from_output）不在本章精简范围——测试用 request.append_output_token_ids
手工模拟 worker 采样回填（真实 Request 的方法，vllm/v1/request.py:L249）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.arg_utils import GiB_bytes, EngineArgs, UsageContext  # noqa: E402
from implementation.interface import PauseState  # noqa: E402
from implementation.request import Request, RequestStatus, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #
def make_request(req_id: str, prompt_len: int, max_tokens: int = 64) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def make_scheduler(
    max_num_batched_tokens: int = 2048,
    max_num_seqs: int = 128,
    max_model_len: int = 8192,
    num_gpu_blocks: int = 1 << 30,
    block_size: int = 16,
    long_prefill_token_threshold: int = 0,
    enable_chunked_prefill: bool = True,
    max_num_scheduled_tokens: int | None = None,
    scheduler_reserve_full_isl: bool = True,
) -> Scheduler:
    config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        scheduler_reserve_full_isl=scheduler_reserve_full_isl,
    )
    return Scheduler(
        config,
        max_model_len=max_model_len,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
    )


# --------------------------------------------------------------------------- #
# 追赶公式（m2/m16）：decode 稳态每拍 1 token；公式字段定义
# --------------------------------------------------------------------------- #
class TestCatchUpFormula:
    def test_request_ledger_fields(self):
        """num_tokens = len(_all_token_ids)；num_tokens_with_spec = +spec（request.py:L271-L277）"""
        req = make_request("r", 10)
        assert req.num_tokens == 10
        assert req.num_tokens_with_spec == 10
        req.append_output_token_ids(7)
        req.spec_token_ids.extend([1, 2])
        assert req.num_tokens == 11
        assert req.num_tokens_with_spec == 13

    def test_decode_steady_state_one_token_per_step(self):
        """完成 prefill 的请求在后续每拍恰领 1 token（追赶公式特例二）"""
        sched = make_scheduler()
        req = make_request("r", 16)
        sched.add_request(req)
        out1 = sched.schedule()  # 首拍 prefill 全量 16
        assert out1.num_scheduled_tokens == {"r": 16}
        for step in range(3):
            req.append_output_token_ids(step)  # 模拟 ⑤ 拍回填采样 token
            out = sched.schedule()
            assert out.num_scheduled_tokens == {"r": 1}
            assert req.num_computed_tokens == 17 + step

    def test_prefill_chunk_request_moves_through_running_phase(self):
        """未完 chunk 的请求下一拍从 RUNNING 阶段续切（m16 稳态闭环）"""
        # max_model_len 需大于 prompt：追赶公式受 max_model_len - computed - 1
        # 保险钳制（L527-L532，给采样 token 留位），prompt==max_model_len 的
        # 请求在真实 vLLM 里会在入口被拒（prompt too long）。
        sched = make_scheduler(max_num_batched_tokens=2048, max_model_len=16384)
        req = make_request("r", 8192)
        sched.add_request(req)
        chunks = []
        for _ in range(4):
            out = sched.schedule()
            chunks.append(out.num_scheduled_tokens["r"])
            assert req.num_in_flight_tokens == sum(chunks)  # 乐观推进的在途计数
        assert chunks == [2048, 2048, 2048, 2048]
        assert req.num_computed_tokens == 8192
        assert not req.is_prefill_chunk
        assert req not in sched._inflight_prefills


# --------------------------------------------------------------------------- #
# 单一 token 预算：跨两阶段分账 + 守恒（m3）
# --------------------------------------------------------------------------- #
class TestTokenBudget:
    def test_running_first_waiting_shares_remainder(self):
        """RUNNING 先吃预算：2 个 decode 各 1，新 8192 prompt 只领余量 2046"""
        sched = make_scheduler(max_num_batched_tokens=2048, max_model_len=16384)
        d1, d2 = make_request("d1", 16), make_request("d2", 16)
        sched.add_request(d1)
        sched.add_request(d2)
        sched.schedule()  # 首拍双双 prefill 完
        d1.append_output_token_ids(1)
        d2.append_output_token_ids(2)
        big = make_request("big", 8192)
        sched.add_request(big)
        out = sched.schedule()
        assert out.num_scheduled_tokens == {"d1": 1, "d2": 1, "big": 2046}
        # 守恒：Σ = max_num_scheduled_tokens（L1109-L1110 断言的行为面）
        assert out.total_num_scheduled_tokens == 2048
        assert out.total_num_scheduled_tokens == sum(
            out.num_scheduled_tokens.values()
        )

    def test_max_num_scheduled_tokens_fallback_to_batched(self):
        """max_num_scheduled_tokens 缺省回落 max_num_batched_tokens（L110-L114 / m19）"""
        sched = make_scheduler(max_num_batched_tokens=4096)
        assert sched.max_num_scheduled_tokens == 4096
        sched2 = make_scheduler(
            max_num_batched_tokens=4096, max_num_scheduled_tokens=1024
        )
        assert sched2.max_num_scheduled_tokens == 1024
        req = make_request("r", 8192)
        sched2.add_request(req)
        out = sched2.schedule()
        assert out.num_scheduled_tokens == {"r": 1024}  # 预算以 1024 为准

    def test_paused_all_short_circuits_budget(self):
        """PAUSED_ALL → token_budget=0 一拍全停（L460-L462 / m18）"""
        sched = make_scheduler()
        sched.add_request(make_request("r", 16))
        sched._pause_state = PauseState.PAUSED_ALL
        out = sched.schedule()
        assert out.total_num_scheduled_tokens == 0
        assert out.num_scheduled_tokens == {}
        assert len(sched.waiting) == 1  # 请求原封不动


# --------------------------------------------------------------------------- #
# chunked prefill 三闸（m10）
# --------------------------------------------------------------------------- #
class TestChunkedPrefill:
    def test_long_prefill_threshold_clamps_chunks(self):
        """threshold 钳制对 WAITING 首拍与 RUNNING 续拍都生效（L521-L522/L899-L901）"""
        sched = make_scheduler(
            max_num_batched_tokens=2048, long_prefill_token_threshold=1024
        )
        req = make_request("r", 4096)
        sched.add_request(req)
        out1 = sched.schedule()
        assert out1.num_scheduled_tokens == {"r": 1024}  # 首拍被 threshold 钳
        assert req.is_prefill_chunk
        out2 = sched.schedule()  # 续拍走 RUNNING 追赶公式，同样被钳
        assert out2.num_scheduled_tokens == {"r": 1024}
        assert req.num_computed_tokens == 2048

    def test_chunked_disabled_breaks_whole_step(self):
        """enable_chunked_prefill=False 且超预算 → 整拍 break 不收新（L905-L911）"""
        sched = make_scheduler(
            max_num_batched_tokens=2048, enable_chunked_prefill=False
        )
        req = make_request("r", 8192)
        sched.add_request(req)
        out = sched.schedule()
        assert out.num_scheduled_tokens == {}
        assert out.total_num_scheduled_tokens == 0
        assert len(sched.waiting) == 1  # 请求仍在 waiting 等大盘

    def test_full_sequence_must_fit_admission_gate(self):
        """WC4：整序列准入门——首 chunk 装得下≠整条装得下（L982 + kv_cache_manager.py:L472-L488）"""
        # 池只有 2 块（32 token），预算 32，prompt 1000：
        # 开门（默认 True）：按整序列 63 块 vs 空闲 2 块 → 拒之门外
        sched = make_scheduler(
            max_num_batched_tokens=32, num_gpu_blocks=2, max_model_len=4096
        )
        assert sched.scheduler_reserve_full_isl
        req = make_request("r", 1000)
        sched.add_request(req)
        out = sched.schedule()
        assert out.num_scheduled_tokens == {}
        assert len(sched.waiting) == 1

        # 关门：只检查第一个 chunk（32 token = 2 块 ≤ 空闲 2 块）→ 放行
        sched2 = make_scheduler(
            max_num_batched_tokens=32,
            num_gpu_blocks=2,
            max_model_len=4096,
            scheduler_reserve_full_isl=False,
        )
        req2 = make_request("r", 1000)
        sched2.add_request(req2)
        out2 = sched2.schedule()
        assert out2.num_scheduled_tokens == {"r": 32}
        assert req2.is_prefill_chunk  # 剩 968 token 留给后续拍


# --------------------------------------------------------------------------- #
# 前缀命中折算（m9，黑盒 get_computed_blocks）
# --------------------------------------------------------------------------- #
class TestPrefixCacheDiscount:
    def test_hit_tokens_counted_as_computed(self):
        """get_computed_blocks 命中数直接当已算：被减数变小（L744-L766）"""
        sched = make_scheduler()
        req = make_request("r", 256)
        sched.add_request(req)
        # 黑盒桩：命中 64 token（真实语义 = 前缀缓存命中块数，链式哈希归 ch15）
        sched.kv_cache_manager.get_computed_blocks = (
            lambda request: (sched.kv_cache_manager.empty_kv_cache_blocks, 64, 0)
        )
        out = sched.schedule()
        assert out.num_scheduled_tokens == {"r": 256 - 64}
        # 命中数先记入账本（L1075: computed=64），随后乐观推进 += 192
        # （L1330）——schedule() 返回时账面 64 命中 + 192 已排 = 256 全量
        assert req.num_computed_tokens == 256


# --------------------------------------------------------------------------- #
# RUNNING 先行 + 抢占守卫（m5/m7/m8）
# --------------------------------------------------------------------------- #
class TestRunningFirstAndPreemption:
    def _prefill_two_decodes(self, num_gpu_blocks: int) -> Scheduler:
        """两请求首拍 prefill 完、各回填 1 token，进入 decode 稳态"""
        sched = make_scheduler(num_gpu_blocks=num_gpu_blocks)
        r1, r2 = make_request("r1", 16), make_request("r2", 16)
        sched.add_request(r1)
        sched.add_request(r2)
        sched.schedule()
        r1.append_output_token_ids(1)
        r2.append_output_token_ids(2)
        return sched

    def test_kv_exhaustion_preempts_running_tail(self):
        """块不够 → FCFS 抢 running 队尾：被抢者 computed 归零、回 waiting 队头（L576-L629）"""
        # 池 3 块：r1/r2 各 1 块，decode 各需 +1 块 → r2(队尾) 被抢给 r1 让路
        sched = self._prefill_two_decodes(num_gpu_blocks=3)
        r1, r2 = sched.requests["r1"], sched.requests["r2"]
        new = make_request("new", 16)
        sched.add_request(new)
        out = sched.schedule()
        # r1 正常 decode 1 token；r2 被抢占自己 break 出 RUNNING
        assert out.num_scheduled_tokens == {"r1": 1}
        assert r2.status == RequestStatus.PREEMPTED
        assert r2.num_computed_tokens == 0  # recompute-only：账本清零
        assert r2.num_preemptions == 1
        assert sched.waiting.peek_request() is r2  # 回 waiting 队头（prepend）
        assert out.preempted_req_ids == {"r2"}
        # 本拍抢占过 → 整拍不收新（L684 守卫）
        assert "new" not in out.num_scheduled_tokens
        assert new.status == RequestStatus.WAITING

    def test_preempted_request_resumes_as_resumed_req(self):
        """恢复请求走 scheduled_resumed_reqs 分流（L1062-L1063），且上拍没调度过 → 补全量 token"""
        sched = self._prefill_two_decodes(num_gpu_blocks=3)
        r1, r2 = sched.requests["r1"], sched.requests["r2"]
        out2 = sched.schedule()  # r1 decode 占掉最后空闲块；r2 被抢占
        assert out2.preempted_req_ids == {"r2"}
        # 模拟 ⑤拍生命周期收尾（ch9/ch11 范围，真实 finish_requests 走
        # running 移除 + _free_request_blocks → kv_cache_manager.free）
        sched.running.remove(r1)
        sched.kv_cache_manager.free(r1)
        out3 = sched.schedule()
        # r2 从 waiting 队头恢复：status PREEMPTED → resumed 分流
        assert "r2" in out3.num_scheduled_tokens
        assert r2.status == RequestStatus.RUNNING
        cached = out3.scheduled_cached_reqs
        assert cached.resumed_req_ids == {"r2"}
        # r1 上一拍调度过（增量即可）；r2 上拍被抢没调度过 → 补传全量 all_token_ids
        # （recompute-only 抢占不清输出：恢复时重放 prompt + 已生成 token，
        #  正是 WAITING 公式用 num_tokens 而非 num_prompt_tokens 的原因）
        assert "r1" not in cached.all_token_ids
        assert cached.all_token_ids["r2"] == list(range(16)) + [2]

    def test_waiting_admission_never_preempts(self):
        """WAITING 侧 allocate_slots None → break，在场请求绝不被抢占（L987-L994）"""
        sched = make_scheduler(num_gpu_blocks=2)
        r1 = make_request("r1", 16)
        sched.add_request(r1)
        sched.schedule()  # r1 占 1 块
        r1.append_output_token_ids(1)
        victim = make_request("victim", 32)  # 需要 2 块，空闲只剩 1
        sched.add_request(victim)
        out = sched.schedule()
        assert out.num_scheduled_tokens == {"r1": 1}  # r1 正常 decode
        assert victim.status == RequestStatus.WAITING  # 新请求原地等
        assert r1.status == RequestStatus.RUNNING  # 在场请求毫发无损
        assert out.preempted_req_ids == set()

    def test_max_num_running_reqs_caps_waiting_admission(self):
        """len(running) ≥ max_num_seqs → WAITING 阶段 break（L690-L692）"""
        sched = make_scheduler(max_num_seqs=1)
        r1 = make_request("r1", 16)
        sched.add_request(r1)
        sched.schedule()
        r1.append_output_token_ids(1)
        r2 = make_request("r2", 16)
        sched.add_request(r2)
        out = sched.schedule()
        assert out.num_scheduled_tokens == {"r1": 1}
        assert r2.status == RequestStatus.WAITING


# --------------------------------------------------------------------------- #
# num_new_tokens==0 continue-not-break（m6）
# --------------------------------------------------------------------------- #
class TestZeroContinue:
    def test_stuck_request_does_not_block_lower_priority(self):
        """已追平的请求 num_new=0 → continue 让后面的请求照常领（L557-L573）"""
        sched = make_scheduler()
        stuck = make_request("stuck", 16)
        ready = make_request("ready", 16)
        sched.add_request(stuck)
        sched.add_request(ready)
        sched.schedule()  # 双双 prefill 完
        ready.append_output_token_ids(9)  # 只有 ready 回填了新 token
        # stuck 追平（computed == num_tokens）→ 本拍 0 新 token
        out = sched.schedule()
        assert out.num_scheduled_tokens == {"ready": 1}
        assert stuck in sched.running  # 没被移除，也没被调度
        assert stuck.num_computed_tokens == 16


# --------------------------------------------------------------------------- #
# SchedulerOutput 二分：new 全量 / cached 增量（m14 / WC5）
# --------------------------------------------------------------------------- #
class TestOutputDichotomy:
    def test_first_schedule_emits_full_new_request_data(self):
        """首次调度 → NewRequestData 全量（prompt + block_ids + computed）"""
        sched = make_scheduler()
        req = make_request("r", 16)
        sched.add_request(req)
        out = sched.schedule()
        assert len(out.scheduled_new_reqs) == 1
        data = out.scheduled_new_reqs[0]
        assert data.req_id == "r"
        assert data.prompt_token_ids == list(range(16))
        assert data.num_computed_tokens == 0
        assert data.block_ids[0]  # 拿到了 KV 块

    def test_second_step_emits_cached_diff_only(self):
        """老请求 → CachedRequestData 增量；上拍调度过 → 不补 all_token_ids"""
        sched = make_scheduler()
        req = make_request("r", 16)
        sched.add_request(req)
        sched.schedule()
        req.append_output_token_ids(5)
        out = sched.schedule()
        assert out.scheduled_new_reqs == []
        cached = out.scheduled_cached_reqs
        assert cached.req_ids == ["r"]
        assert cached.all_token_ids == {}  # 上拍调度过，只发 diff
        assert cached.num_computed_tokens == [16]  # 乐观推进后的账面值
        assert len(cached.new_block_ids) == 1


# --------------------------------------------------------------------------- #
# 乐观推进（m15）
# --------------------------------------------------------------------------- #
class TestOptimisticAdvance:
    def test_update_after_schedule_advances_before_gpu_runs(self):
        """调度后立即 += n（GPU 还没算）+ is_prefill_chunk 标记（L1317-L1343）"""
        sched = make_scheduler()
        req = make_request("r", 4096)
        sched.add_request(req)
        sched.schedule()  # 首拍 2048
        assert req.num_computed_tokens == 2048
        assert req.num_in_flight_tokens == 2048
        assert req.is_prefill_chunk
        assert req in sched._inflight_prefills
        sched.schedule()  # 次拍续 2048
        assert req.num_computed_tokens == 4096
        assert not req.is_prefill_chunk
        assert req not in sched._inflight_prefills


# --------------------------------------------------------------------------- #
# add_request / 队列（m17）
# --------------------------------------------------------------------------- #
class TestRequestQueue:
    def test_add_request_registers_and_enqueues_in_fcfs_order(self):
        """requests 登记 + waiting 队尾入队（L2213-L2235 主路径）"""
        from implementation.request_queue import (
            FCFSRequestQueue,
            SchedulingPolicy,
            create_request_queue,
        )

        sched = make_scheduler()
        r1, r2 = make_request("r1", 4), make_request("r2", 4)
        sched.add_request(r1)
        sched.add_request(r2)
        assert sched.requests == {"r1": r1, "r2": r2}
        assert sched.waiting.pop_request() is r1
        assert sched.waiting.pop_request() is r2
        assert r1.status == RequestStatus.WAITING

        # FCFSRequestQueue 四操作语义（request_queue.py:L75-L93）
        q = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(q, FCFSRequestQueue)
        q.add_request(r1)
        q.add_request(r2)
        assert q.peek_request() is r1
        q.prepend_request(make_request("r0", 4))  # 抢占回队头靠它（ch11 钩子）
        assert q.pop_request().request_id == "r0"
        assert q.pop_request().request_id == "r1"

    def test_scheduler_default_policy_is_fcfs(self):
        sched = make_scheduler()
        from implementation.request_queue import SchedulingPolicy

        assert sched.policy == SchedulingPolicy.FCFS


# --------------------------------------------------------------------------- #
# 预算默认值地形（m4）：config 基线 + arg_utils 硬件仲裁表
# --------------------------------------------------------------------------- #
class TestBudgetDefaults:
    def test_config_classvar_baselines(self):
        """DEFAULT_MAX_NUM_BATCHED_TOKENS=2048 / DEFAULT_MAX_NUM_SEQS=128（config L42-L44）"""
        assert SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS == 2048
        assert SchedulerConfig.DEFAULT_MAX_NUM_SEQS == 128
        config = SchedulerConfig()
        assert config.max_num_batched_tokens == 2048
        assert config.max_num_scheduled_tokens is None
        assert config.max_num_seqs == 128
        assert config.long_prefill_token_threshold == 0
        assert config.enable_chunked_prefill is True  # v1 默认开
        assert config.scheduler_reserve_full_isl is True

    def test_get_batch_defaults_big_gpu_not_a100(self):
        """≥70GiB 且非 A100 → LLM 16384 / API server 8192，seqs 1024"""
        tokens, seqs = EngineArgs.get_batch_defaults(
            world_size=1, device_memory=80 * GiB_bytes, device_name="nvidia h100 80gb hbm3"
        )
        assert tokens[UsageContext.LLM_CLASS] == 16384
        assert tokens[UsageContext.OPENAI_API_SERVER] == 8192
        assert seqs[UsageContext.LLM_CLASS] == 1024
        assert seqs[UsageContext.OPENAI_API_SERVER] == 1024

    def test_get_batch_defaults_a100_exception(self):
        """A100 反例：大预算反降吞吐（PR #17885）→ 走 else 档 8192/2048、256"""
        tokens, seqs = EngineArgs.get_batch_defaults(
            world_size=1, device_memory=80 * GiB_bytes, device_name="nvidia a100-sxm4-80gb"
        )
        assert tokens[UsageContext.LLM_CLASS] == 8192
        assert tokens[UsageContext.OPENAI_API_SERVER] == 2048
        assert seqs[UsageContext.LLM_CLASS] == 256

    def test_get_batch_defaults_small_gpu(self):
        """<70GiB → else 档"""
        tokens, _ = EngineArgs.get_batch_defaults(
            world_size=1, device_memory=24 * GiB_bytes, device_name="nvidia rtx 4090"
        )
        assert tokens[UsageContext.LLM_CLASS] == 8192
        assert tokens[UsageContext.OPENAI_API_SERVER] == 2048


# --------------------------------------------------------------------------- #
# 混相批：256×1 vs 1×8192 的算术（theory[2]）
# --------------------------------------------------------------------------- #
class TestMixedBatch:
    def test_decode_batch_and_prefill_chunk_share_one_step(self):
        """decode 们各领 1 token、大 prompt 同拍切 chunk 混进同一批"""
        sched = make_scheduler(max_num_batched_tokens=2048)
        decodes = []
        for i in range(4):
            r = make_request(f"d{i}", 16)
            sched.add_request(r)
            decodes.append(r)
        sched.schedule()
        for i, r in enumerate(decodes):
            r.append_output_token_ids(i)
        big = make_request("big", 8192)
        sched.add_request(big)
        out = sched.schedule()
        assert all(out.num_scheduled_tokens[f"d{i}"] == 1 for i in range(4))
        assert out.num_scheduled_tokens["big"] == 2048 - 4
        assert out.total_num_scheduled_tokens == 2048
        # 混相批 = 4 个 decode 增量 + 1 个 NewRequestData 全量
        assert len(out.scheduled_cached_reqs.req_ids) == 4
        assert len(out.scheduled_new_reqs) == 1
