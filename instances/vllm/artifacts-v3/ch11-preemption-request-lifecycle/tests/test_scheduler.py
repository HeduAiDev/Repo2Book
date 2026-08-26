"""ch11 抢占与请求的一生 —— 纯单元测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测行为**
（锚点 = vllm/v1/... 行号，基线 v0.27.1）：
- RequestStatus 单 IntEnum：WAITING→三阻塞子态→RUNNING↔PREEMPTED→FINISHED_*，
  is_finished = status > PREEMPTED 一次整数比较（request.py:L348-L375）
- RUNNING 侧 allocate_slots None → while True 抢 FCFS 队尾重试（scheduler.py:L575-L629）；
  preempted_req==request 仍 None → 整拍放弃（L622-L629）
- _preempt_request 六件事（L1274-L1315）：free 块（哈希保留）/ PREEMPTED /
  num_computed_tokens=0 / 清 spec_token_ids / stale 标记（assign 不累加）/
  num_preemptions+1 → waiting.prepend_request 回队头
- 守卫：本拍抢占过 → 整拍不收新（L683-L684 `not preempted_reqs`）
- WAITING 侧 None → break 绝不抢占（L987-L994）
- 前缀重命中：free 不清哈希 → get_computed_blocks 重命中自己的前缀（L744-L766 +
  block_pool.py:L719-L742 只动 ref_cnt/自由队列）；max_cache_hit_length =
  num_tokens-1（全命中也必须重算最后一个 token 才有 logits，
  kv_cache_manager.py:L253-L259）
- 恢复准入带水位：watermark 仅 WAITING/PREEMPTED 且 has_scheduled_reqs 计入
  headroom（kv_cache_manager.py:L463-L470 + L521-L527；config 默认 0.0）
- 回流落位：PREEMPTED→scheduled_resumed_reqs + resumed_req_ids（L1055-L1075）
- 双队列防队头阻塞：阻塞态→skipped_waiting、_try_promote False→跳过（L700-L711）、
  stale 在途推迟恢复（L713-L722）、step_skipped_waiting 步末重排（L1099-L1101）
- update_from_output 热循环（L1670-L1951）：req_id_to_index 定位采样行、扣
  num_in_flight_tokens、stale 锁步 drain、abort 期完成 continue、drop-mode 丢弃、
  finish_reason 先抓再 handle、status_before_stop 分流、remove_all 批量摘除
- check_stop 五连判顺序（utils.py:L94-L130）：min_tokens→EOS→stop_token_ids→
  长度封顶→重复检测；停止 token 之后截断（scheduler.py:L2108-L2109）
- _handle_stopped_request：非流式请求恒 True（流式会话已删，L2076-L2092）
- _free_request/_free_blocks 终点：finished_req_ids 登记 + del requests（L2300-L2338）
- finish_requests 外部 abort：FINISHED_ABORTED、三队列摘除、幂等（L2237-L2298）
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.engine import FinishReason  # noqa: E402
from implementation.kv_cache_manager import (  # noqa: E402
    KVCacheManager,
    get_request_block_hasher,
)
from implementation.output import ModelRunnerOutput  # noqa: E402
from implementation.request import (  # noqa: E402
    RepetitionDetectionParams,
    Request,
    RequestStatus,
    SamplingParams,
)
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.utils import check_stop, remove_all  # noqa: E402


# --------------------------------------------------------------------------- #
# 构造辅助：镜像真实装配（EngineCore 用 get_request_block_hasher 造 Request，
# core.py:L220-L227/L983）
# --------------------------------------------------------------------------- #
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
    watermark: float = 0.0,
) -> Scheduler:
    config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        scheduler_reserve_full_isl=scheduler_reserve_full_isl,
        watermark=watermark,
    )
    return Scheduler(
        config,
        max_model_len=max_model_len,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
    )


def make_request(
    sched: Scheduler,
    req_id: str,
    prompt_len: int,
    max_tokens: int = 64,
    eos_token_id: int | None = None,
    stop_token_ids: list[int] | None = None,
    min_tokens: int = 0,
    repetition_detection: RepetitionDetectionParams | None = None,
) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            repetition_detection=repetition_detection,
        ),
        block_hasher=get_request_block_hasher(sched.kv_cache_manager.block_size),
    )


def step(
    sched: Scheduler, tokens_by_req: dict[str, list[int]]
) -> tuple[object, list]:
    """走完一拍：schedule() → 假想 forward 采样 → update_from_output()。"""
    out = sched.schedule()
    req_ids = list(out.num_scheduled_tokens)
    mro = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[tokens_by_req.get(rid, []) for rid in req_ids],
    )
    outputs = sched.update_from_output(out, mro)
    flat = [o for eco in outputs.values() for o in eco.outputs]
    return out, flat


# --------------------------------------------------------------------------- #
# 状态机账本（m10/WC2）：RequestStatus 单 IntEnum，全章地图
# --------------------------------------------------------------------------- #
class TestStateMachine:
    def test_status_ordering_one_integer_comparison(self):
        """WAITING→三阻塞子态→RUNNING↔PREEMPTED→FINISHED_*；>PREEMPTED 即完成"""
        s = RequestStatus
        assert s.WAITING < s.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        assert s.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR < s.WAITING_FOR_REMOTE_KVS
        assert s.WAITING_FOR_REMOTE_KVS < s.WAITING_FOR_STREAMING_REQ
        assert s.WAITING_FOR_STREAMING_REQ < s.RUNNING < s.PREEMPTED
        assert s.PREEMPTED < s.FINISHED_STOPPED
        # is_finished = status > PREEMPTED：非终态全 False，终态全 True
        for st in (
            s.WAITING,
            s.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            s.WAITING_FOR_REMOTE_KVS,
            s.WAITING_FOR_STREAMING_REQ,
            s.RUNNING,
            s.PREEMPTED,
        ):
            assert not RequestStatus.is_finished(st)
        for st in (
            s.FINISHED_STOPPED,
            s.FINISHED_LENGTH_CAPPED,
            s.FINISHED_ABORTED,
            s.FINISHED_IGNORED,
            s.FINISHED_ERROR,
            s.FINISHED_REPETITION,
        ):
            assert RequestStatus.is_finished(st)

    def test_finished_reason_map(self):
        """终态→对外 FinishReason；WAITING_FOR_STREAMING_REQ→STOP 特殊映射"""
        s = RequestStatus
        assert s.get_finished_reason(s.FINISHED_STOPPED) == FinishReason.STOP
        assert s.get_finished_reason(s.FINISHED_LENGTH_CAPPED) == FinishReason.LENGTH
        assert s.get_finished_reason(s.FINISHED_ABORTED) == FinishReason.ABORT
        assert s.get_finished_reason(s.FINISHED_ERROR) == FinishReason.ERROR
        assert (
            s.get_finished_reason(s.FINISHED_REPETITION) == FinishReason.REPETITION
        )
        assert (
            s.get_finished_reason(s.WAITING_FOR_STREAMING_REQ) == FinishReason.STOP
        )
        assert s.get_finished_reason(s.RUNNING) is None


# --------------------------------------------------------------------------- #
# 抢占环 + _preempt_request 六件事 + 守卫（m1/m2/m3/m5）
# --------------------------------------------------------------------------- #
class TestPreemptionRing:
    def _two_decodes(self, num_gpu_blocks: int) -> tuple[Scheduler, Request, Request]:
        """r1/r2 各 prefill 16 token（1 块）、各回填 1 token，进入 decode 稳态"""
        sched = make_scheduler(num_gpu_blocks=num_gpu_blocks)
        r1 = make_request(sched, "r1", 16)
        r2 = make_request(sched, "r2", 16)
        sched.add_request(r1)
        sched.add_request(r2)
        step(sched, {"r1": [1], "r2": [2]})  # 首拍 prefill 产出首 token
        return sched, r1, r2

    def test_kv_exhaustion_preempts_fcfs_tail_six_things(self):
        """块不够 → 抢 running 队尾：六件事逐项核（L575-L629 / L1274-L1315）"""
        # 池 2 块：r1/r2 prefill 各持 1、空闲 0。decode 第 17 token 都需第 2 块
        # → r1 先到先得失败 → 抢队尾 r2 给 r1 让路（r2 非『自我抢占』）
        sched, r1, r2 = self._two_decodes(num_gpu_blocks=2)
        r2.spec_token_ids = [9, 9]  # 预置 spec，验证『清 spec』这件事
        new = make_request(sched, "new", 16)
        sched.add_request(new)
        out, _ = step(sched, {"r1": [3]})  # r1 正常 decode；r2 被抢
        assert out.num_scheduled_tokens == {"r1": 1}
        # 六件事（m3）：
        assert r2.status == RequestStatus.PREEMPTED  # ①PREEMPTED
        assert r2.num_computed_tokens == 0  # ②账本清零（recompute-only）
        assert r2.spec_token_ids == []  # ③清 spec_token_ids
        assert r2.num_preemptions == 1  # ④被抢计数
        assert r2.num_stale_output_tokens == r2.num_in_flight_tokens == 0  # ⑤stale
        assert sched.waiting.peek_request() is r2  # ⑥回 waiting 队头（prepend）
        # free 块：r2 的 1 块已归还池（抢占后 r1 又领走 → 空闲 0）
        assert sched.kv_cache_manager.num_free_blocks == 0
        assert out.preempted_req_ids == {"r2"}
        # 守卫（m5）：本拍抢占过 → 整拍不收新
        assert "new" not in out.num_scheduled_tokens
        assert new.status == RequestStatus.WAITING

    def test_preempt_self_gives_up_whole_step(self):
        """把自己都抢了仍分不到 → break 放弃；本拍 preempted_reqs 非空（L622-L629）"""
        sched = make_scheduler(num_gpu_blocks=1)
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill 占满唯一块
        out, outputs = step(sched, {})  # decode 需第 2 块 → 无处可抢
        assert out.num_scheduled_tokens == {}
        assert out.total_num_scheduled_tokens == 0
        assert r1.status == RequestStatus.PREEMPTED
        assert r1.num_preemptions == 1
        assert out.preempted_req_ids == {"r1"}
        assert outputs == []  # 没有请求被调度 → 没有输出

    def test_waiting_admission_never_preempts(self):
        """WAITING 侧 allocate_slots None → break，在场请求绝不被抢（L987-L994）"""
        sched = make_scheduler(num_gpu_blocks=2)
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # r1 prefill 占 1 块
        victim = make_request(sched, "victim", 32)  # 整序列需 2 块，只剩 1
        sched.add_request(victim)
        # r1 在 2 块（32 槽）内继续 decode；第 33 token 需第 3 块 → 池尽，
        # RUNNING 侧自我抢占。victim 全程毫发无损：WAITING 侧 None 只 break
        for i in range(20):
            if r1.status == RequestStatus.PREEMPTED:
                break
            step(sched, {"r1": [2 + i]})
        assert r1.status == RequestStatus.PREEMPTED
        assert r1.num_preemptions == 1
        assert victim.status == RequestStatus.WAITING
        assert victim.num_preemptions == 0


# --------------------------------------------------------------------------- #
# 前缀重命中（m7/F2）：free 不清哈希 → 恢复=重载+补算
# --------------------------------------------------------------------------- #
class TestPrefixRehit:
    def test_preempted_request_rehits_own_prefix(self):
        """被抢者恢复：get_computed_blocks 重命中自己的满块——『重算』只补尾段"""
        # 池 4 块、prompt 64（恰 4 块）：decode 第 65 token 需第 5 块 → 自我被抢
        sched = make_scheduler(num_gpu_blocks=4)
        r1 = make_request(sched, "r1", 64)
        sched.add_request(r1)
        step(sched, {"r1": [7]})  # prefill 64 → 4 块全占，首 token 已出
        out_pre, _ = step(sched, {})  # decode 需第 5 块 → 抢自己
        assert r1.status == RequestStatus.PREEMPTED
        assert out_pre.preempted_req_ids == {"r1"}
        # 下一拍：恢复准入。块哈希 free 后仍在表 → 命中 4 块=64 token，
        # 但 max_cache_hit_length = num_tokens-1 = 64（L253-L259 全命中也重算
        # 最后一个 token）→ 命中 64，num_new = 65-64 = 1：只补 1 token
        out, _ = step(sched, {"r1": [8]})
        assert out.num_scheduled_tokens == {"r1": 1}
        assert r1.status == RequestStatus.RUNNING
        # 恢复分流（m9）：PREEMPTED → scheduled_resumed_reqs → resumed_req_ids
        assert out.scheduled_cached_reqs.resumed_req_ids == {"r1"}
        # 账本：准入记命中数 64，乐观推进 +1 → 65（= num_tokens，追平）
        assert r1.num_computed_tokens == 65
        # 对照：无前缀缓存的世界里这是 65 token 的全量重算

    def test_full_prompt_hit_still_recomputes_last_block(self):
        """新请求撞满前缀：cap=num_tokens-1 → 命中按块对齐向下取（L253-L259）"""
        # r1 64 token 跑完并 free（哈希留表）；r2 同 prompt 64：
        # cap=63 → 命中 3 块=48 token，只排 64-48=16
        sched = make_scheduler(num_gpu_blocks=8)
        r1 = make_request(sched, "r1", 64, max_tokens=1)
        sched.add_request(r1)
        out1, outs1 = step(sched, {"r1": [7]})  # 首拍出 1 token 即长度封顶
        assert r1.is_finished()
        assert r1.get_finished_reason() == FinishReason.LENGTH
        r2 = make_request(sched, "r2", 64)
        sched.add_request(r2)
        out2, _ = step(sched, {"r2": [5]})
        assert out2.num_scheduled_tokens == {"r2": 16}
        assert r2.num_computed_tokens == 64  # 48 命中 + 16 新算
        assert out2.scheduled_new_reqs[0].req_id == "r2"  # 首调度走 new 全量


# --------------------------------------------------------------------------- #
# 水位 watermark（m8/WC3）：仅 WAITING/PREEMPTED 准入且 has_scheduled_reqs 计入
# --------------------------------------------------------------------------- #
class TestWatermark:
    def test_watermark_blocks_arithmetic(self):
        """watermark_blocks = int(watermark × num_blocks)（kv_cache_manager L168-L171）"""
        m = KVCacheManager(num_gpu_blocks=10, block_size=16, max_model_len=8192,
                           watermark=0.5)
        assert m.watermark_blocks == 5
        m2 = KVCacheManager(num_gpu_blocks=10, block_size=16, max_model_len=8192,
                            watermark=0.3)
        assert m2.watermark_blocks == int(0.3 * 10)
        assert SchedulerConfig().watermark == 0.0  # 默认关

    def _decode_steady(self, num_gpu_blocks: int, watermark: float):
        """r1 prompt 128（8 块）prefill 完 + 1 decode（第 9 块），留 1 空闲"""
        sched = make_scheduler(num_gpu_blocks=num_gpu_blocks, watermark=watermark)
        r1 = make_request(sched, "r1", 128)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill 8 块（首拍 running 空 → 水位不适用）
        step(sched, {"r1": [2]})  # decode 领第 9 块
        assert sched.kv_cache_manager.num_free_blocks == 1
        return sched, r1

    def test_watermark_gates_waiting_admission_only(self):
        """headroom 计入 required_blocks：空闲 1 < 需 1+5 → 拒之门外（L463-L488）"""
        sched, r1 = self._decode_steady(num_gpu_blocks=10, watermark=0.5)
        small = make_request(sched, "small", 16)  # 整序列恰 1 块
        sched.add_request(small)
        out, _ = step(sched, {"r1": [3]})
        assert out.num_scheduled_tokens == {"r1": 1}  # r1 照常 decode
        assert small.status == RequestStatus.WAITING  # 水位拒收
        # 关掉水位：required 1 ≤ 1 → 放行
        sched2, _ = self._decode_steady(num_gpu_blocks=10, watermark=0.0)
        small2 = make_request(sched2, "small", 16)
        sched2.add_request(small2)
        out2, _ = step(sched2, {"r1": [4]})
        assert "small" in out2.num_scheduled_tokens

    def test_watermark_not_applied_when_no_scheduled_reqs(self):
        """running 空（首拍准入）不吃水位——否则引擎永远起步不了（L466-L470）"""
        sched = make_scheduler(num_gpu_blocks=10, watermark=0.5)
        # r1 整序列 8 块：若水位被错误计入，8+5=13 > 10 → None；实际应放行
        r1 = make_request(sched, "r1", 128)
        sched.add_request(r1)
        out, _ = step(sched, {"r1": [1]})
        assert out.num_scheduled_tokens == {"r1": 128}
        assert r1.status == RequestStatus.RUNNING

    def test_watermark_not_applied_to_running_growth(self):
        """RUNNING 侧 decode 增长分配不吃水位（否则正常 decode 也被压）"""
        sched = make_scheduler(num_gpu_blocks=10, watermark=0.5)
        r1 = make_request(sched, "r1", 128)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # 8 块，free 2
        # decode 需第 9 块：required 1+0（RUNNING 不在 WAITING/PREEMPTED）≤ 2 ✓
        out, _ = step(sched, {"r1": [2]})
        assert out.num_scheduled_tokens == {"r1": 1}


# --------------------------------------------------------------------------- #
# stale 在途输出协议（m4）
# --------------------------------------------------------------------------- #
class TestStaleProtocol:
    def _scheduled_then_preempt(self, sched, drop: bool):
        """模拟 async 重叠：本拍已调度（in_flight=1）→ 输出回来前被抢（stale=1）"""
        out = sched.schedule()  # r1 decode 领 1 token
        r1 = sched.requests["r1"]
        assert r1.num_in_flight_tokens == 1  # _update_after_schedule 乐观 +1
        sched.running.remove(r1)  # _preempt_request 要求已出 running
        sched._preempt_request(r1, time.monotonic(), drop_stale_output=drop)
        return out, r1

    def test_preempt_assigns_stale_from_in_flight(self):
        """num_stale_output_tokens ← num_in_flight_tokens（assign 不累加，L1307）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        out = sched.schedule()
        r1.num_in_flight_tokens = 2  # 模拟 async/PP 在途（同步版此处恒 0）
        sched.running.remove(r1)
        sched._preempt_request(r1, time.monotonic())
        assert r1.num_stale_output_tokens == 2
        assert r1.num_output_placeholders == 0  # 占位计数同步清零

    def test_stale_delivery_drains_in_lockstep(self):
        """stale 输出仍要送达 + 每拍按调度数锁步 drain（L1737-L1743）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill
        out, r1 = self._scheduled_then_preempt(sched, drop=False)
        assert r1.num_stale_output_tokens == 1
        mro = ModelRunnerOutput(
            req_ids=["r1"], req_id_to_index={"r1": 0}, sampled_token_ids=[[42]]
        )
        outputs = sched.update_from_output(out, mro)
        flat = [o for eco in outputs.values() for o in eco.outputs]
        # 送达：stale 的 42 仍出现在输出里（不丢弃）
        assert flat and flat[0].new_token_ids == [42]
        # 锁步 drain + 扣在途：两者都按本拍调度数冲销
        assert r1.num_stale_output_tokens == 0
        assert r1.num_in_flight_tokens == 0
        # token 也记进请求账本（recompute-only 不清输出）
        assert r1.output_token_ids[-1] == 42

    def test_drop_mode_discards_stale_output(self):
        """drop_stale_output=True → 整段丢弃不外送（L1757-L1759）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        out, r1 = self._scheduled_then_preempt(sched, drop=True)
        mro = ModelRunnerOutput(
            req_ids=["r1"], req_id_to_index={"r1": 0}, sampled_token_ids=[[42]]
        )
        outputs = sched.update_from_output(out, mro)
        flat = [o for eco in outputs.values() for o in eco.outputs]
        assert flat == []  # 整段丢弃
        assert 42 not in r1.output_token_ids  # 也不入账

    def test_stale_defers_resume_one_step(self):
        """stale 未排空前恢复被推迟（L713-L722：现在恢复会重采输出稍后要送的位置）"""
        sched = make_scheduler(num_gpu_blocks=4)
        r1 = make_request(sched, "r1", 64)
        sched.add_request(r1)
        step(sched, {"r1": [7]})
        step(sched, {})  # r1 被抢（自我），回 waiting 队头
        assert r1.status == RequestStatus.PREEMPTED
        r1.num_stale_output_tokens = 1  # 模拟 async 下未排干的 stale 份额
        r2 = make_request(sched, "r2", 16)
        sched.add_request(r2)
        out, _ = step(sched, {"r2": [1]})
        # r1 在 waiting 队头但被推迟；r2 照常准入
        assert out.num_scheduled_tokens == {"r2": 16}
        assert r1.status == RequestStatus.PREEMPTED
        assert r1 in sched.skipped_waiting  # 推迟落位：跳过收集队列
        # stale 排干后（async 下由 update_from_output 每拍冲销）下一拍恢复
        r1.num_stale_output_tokens = 0
        out2, _ = step(sched, {"r2": [2]})
        assert "r1" in out2.num_scheduled_tokens
        assert out2.scheduled_cached_reqs.resumed_req_ids == {"r1"}

    def test_sync_preemption_stale_self_neutralizes(self):
        """同步版自中和：抢占发生在上拍输出已回账后 → stale 恒 0（theory[4]）"""
        sched = make_scheduler(num_gpu_blocks=1)
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill 完，in_flight 已被 update 归零
        assert r1.num_in_flight_tokens == 0
        step(sched, {})  # decode 触发抢占
        assert r1.num_preemptions == 1
        assert r1.num_stale_output_tokens == 0  # 协议只在 async/PP 下咬合


# --------------------------------------------------------------------------- #
# 双队列防队头阻塞（m6）
# --------------------------------------------------------------------------- #
class TestDualQueue:
    def test_blocked_statuses_routed_to_skipped(self):
        """_is_blocked_waiting_status 三阻塞态 + 路由（L2050-L2062）"""
        sched = make_scheduler()
        blocked = make_request(sched, "b", 16)
        blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        plain = make_request(sched, "p", 16)
        plain.status = RequestStatus.PREEMPTED
        sched._enqueue_waiting_request(blocked)
        sched._enqueue_waiting_request(plain)
        assert blocked in sched.skipped_waiting
        assert plain in sched.waiting

    def test_blocked_head_does_not_starve_ready_tail(self):
        """阻塞请求卡队头不饿死后面的可调度者（单队列会全体饿死）"""
        sched = make_scheduler()
        b = make_request(sched, "b", 16)
        b.status = RequestStatus.WAITING_FOR_STREAMING_REQ  # 手动置阻塞态
        ready = make_request(sched, "ready", 16)
        sched._enqueue_waiting_request(b)  # → skipped_waiting
        sched.add_request(ready)  # → waiting
        out, _ = step(sched, {"ready": [1]})
        assert out.num_scheduled_tokens == {"ready": 16}
        # b 没被丢：跳过收集 → 步末 prepend 回 skipped 队头等下拍重试
        assert b in sched.skipped_waiting
        assert b.status == RequestStatus.WAITING_FOR_STREAMING_REQ

    def test_step_skipped_requeued_ahead_of_older_skipped(self):
        """步末重排：本拍跳过者整体 prepend 回 skipped（L1099-L1101）"""
        sched = make_scheduler()
        older = make_request(sched, "older", 16)
        older.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        sched._enqueue_waiting_request(older)  # 老的阻塞者先进 skipped
        # 构造第二个跳过者：stale 在途的 PREEMPTED 请求
        newer = make_request(sched, "newer", 16)
        newer.status = RequestStatus.PREEMPTED
        newer.num_stale_output_tokens = 1
        sched._enqueue_waiting_request(newer)  # PREEMPTED 非 blocked → waiting
        ready = make_request(sched, "ready", 16)
        sched.add_request(ready)
        out, _ = step(sched, {"ready": [1]})
        assert out.num_scheduled_tokens == {"ready": 16}
        # 两队都有人：FCFS skipped 优先 → older 先被 peek 并跳过，然后轮到
        # waiting 队头的 newer（stale>0）也被跳过，最后 ready 准入。
        # 步末 step_skipped_waiting=[newer, older]（prepend 收集，后跳者在前）
        # 经 prepend_requests 的 extendleft 反转整体插回 skipped 队头 →
        # 重试序 = 本拍跳过序（older 先重试）
        assert list(sched.skipped_waiting) == [older, newer]


# --------------------------------------------------------------------------- #
# ⑤ 拍热循环：判停、分流、终点（m11-m16）
# --------------------------------------------------------------------------- #
class TestUpdateFromOutput:
    def test_decode_step_emits_token_and_settles_ledger(self):
        """定位采样行→扣在途→逐 token→外送（L1728-L1764 / L2094-L2111）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill
        out, flat = step(sched, {"r1": [42]})
        assert out.num_scheduled_tokens == {"r1": 1}
        assert len(flat) == 1
        assert flat[0].request_id == "r1"
        assert flat[0].new_token_ids == [42]
        assert flat[0].finish_reason is None  # 未停
        assert r1.num_in_flight_tokens == 0  # 已回账
        # 乐观推进：本拍调度的 1 token 已入账（16 prefill + 1）；
        # 本拍采样出的 42 要到下拍才算 computed
        assert r1.num_computed_tokens == 17

    def test_mid_prefill_chunk_emits_nothing(self):
        """未完 chunk 的 prefill：model runner 回空行 → 不外送（L2098-L2100 注）"""
        sched = make_scheduler(max_num_batched_tokens=64)
        r1 = make_request(sched, "r1", 128)
        sched.add_request(r1)
        out1, flat1 = step(sched, {})  # 首拍 chunk 64，无采样行
        assert out1.num_scheduled_tokens == {"r1": 64}
        assert flat1 == []
        assert r1.is_prefill_chunk

    def test_eos_finishes_and_frees_end_to_end(self):
        """EOS→FINISHED_STOPPED→free→下拍 finished_req_ids 通告 worker（m16）"""
        sched = make_scheduler(num_gpu_blocks=4)
        r1 = make_request(sched, "r1", 16, eos_token_id=99)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill 首 token=1，未停
        out, flat = step(sched, {"r1": [99]})
        assert r1.status == RequestStatus.FINISHED_STOPPED
        assert flat[0].finish_reason == FinishReason.STOP
        # 终点：requests 除名 + running 摘除 + 块归还
        assert r1.request_id not in sched.requests
        assert r1 not in sched.running
        assert sched.kv_cache_manager.num_free_blocks == 4
        # 下拍 SchedulerOutput.finished_req_ids 通知 worker 清缓存
        out2, _ = step(sched, {})
        assert out2.finished_req_ids == {"r1"}

    def test_stop_token_sets_stop_reason(self):
        """stop_token_ids 命中 → FINISHED_STOPPED + stop_reason=token id（L108-L111）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16, stop_token_ids=[7])
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        _, flat = step(sched, {"r1": [7]})
        assert r1.status == RequestStatus.FINISHED_STOPPED
        assert r1.stop_reason == 7

    def test_tokens_after_stop_are_trimmed(self):
        """命中即截断：停止 token 之后的不再外送（L2108-L2109）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16, eos_token_id=6)
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        _, flat = step(sched, {"r1": [5, 6, 7]})  # 6 命中 EOS，7 被截
        assert flat[0].new_token_ids == [5, 6]
        assert r1.output_token_ids[-2:] == [5, 6]  # 7 不入账

    def test_min_tokens_gates_eos(self):
        """五连判顺序即优先级：min_tokens 门槛先于 EOS（L100-L101）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16, eos_token_id=5, min_tokens=3)
        sched.add_request(r1)
        # 第 1 输出 token（prefill 采样）=1：1<3 被门槛拦下（纵使下一拍是 EOS）
        step(sched, {"r1": [1]})
        assert not r1.is_finished()
        _, _ = step(sched, {"r1": [5]})  # 第 2 输出 token=5=EOS：2<3 仍拦下
        assert not r1.is_finished()
        _, _ = step(sched, {"r1": [5]})  # 第 3 输出 token：3≥3 → 放行 EOS
        assert r1.status == RequestStatus.FINISHED_STOPPED

    def test_length_capped_by_max_tokens(self):
        """output ≥ max_tokens → FINISHED_LENGTH_CAPPED（L112-L117）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16, max_tokens=2)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # 第 1 个输出 token（prefill 采样）
        _, flat = step(sched, {"r1": [2]})  # 第 2 个 → 封顶
        assert r1.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert r1.get_finished_reason() == FinishReason.LENGTH

    def test_repetition_detected(self):
        """重复检测：N-gram 模式重复 → FINISHED_REPETITION（L119-L128）"""
        det = RepetitionDetectionParams(
            max_pattern_size=2, min_pattern_size=1, min_count=3
        )
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16, repetition_detection=det)
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        _, flat = step(sched, {"r1": [2]})
        # 输出 [1,2,1,2,1,2]：继续喂到模式成立
        for _ in range(2):
            if r1.is_finished():
                break
            step(sched, {"r1": [1]})
            if r1.is_finished():
                break
            step(sched, {"r1": [2]})
        assert r1.status == RequestStatus.FINISHED_REPETITION
        assert r1.stop_reason == "repetition_detected"


class TestCheckStopDirect:
    """check_stop 五连判的直接单元核（顺序即优先级）"""

    def _req(self, output, **kw):
        sched = make_scheduler()
        req = make_request(sched, "r", 4, **kw)
        for t in output:
            req.append_output_token_ids(t)
        return req

    def test_max_model_len_cap(self):
        req = self._req([1, 2, 3, 4], max_tokens=64)
        # num_tokens=8 ≥ max_model_len=8 → LENGTH_CAPPED
        assert check_stop(req, 8)
        assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED

    def test_repetition_pattern_direct(self):
        det = RepetitionDetectionParams(
            max_pattern_size=2, min_pattern_size=1, min_count=3
        )
        req = self._req([1, 2, 1, 2, 1, 2], repetition_detection=det)
        assert check_stop(req, 8192)
        assert req.status == RequestStatus.FINISHED_REPETITION

    def test_no_stop_returns_false(self):
        req = self._req([5], max_tokens=64)
        assert not check_stop(req, 8192)
        assert req.status == RequestStatus.RUNNING or req.status == RequestStatus.WAITING


class TestStoppedPreemptedRarePath:
    def test_stops_on_preempted_step_goes_stopped_preempted(self):
        """被抢当拍完成的罕见路径：从 waiting 摘除（L1895-L1907/L1946-L1952）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16, eos_token_id=42)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill
        # async 重叠：本拍 r1 已调度，输出回来前被抢
        out = sched.schedule()
        r1 = sched.requests["r1"]
        sched.running.remove(r1)
        sched._preempt_request(r1, time.monotonic())
        assert r1.status == RequestStatus.PREEMPTED
        assert r1 in sched.waiting
        mro = ModelRunnerOutput(
            req_ids=["r1"], req_id_to_index={"r1": 0}, sampled_token_ids=[[42]]
        )
        outputs = sched.update_from_output(out, mro)
        flat = [o for eco in outputs.values() for o in eco.outputs]
        # 完成于 PREEMPTED 态：输出仍带 finish_reason（先抓再 handle）
        assert flat[0].finish_reason == FinishReason.STOP
        assert r1.status == RequestStatus.FINISHED_STOPPED
        # stopped_preempted_reqs：从 waiting/skipped 双队列摘除 + 终点 free
        assert r1 not in sched.waiting
        assert r1 not in sched.skipped_waiting
        assert r1.request_id not in sched.requests


# --------------------------------------------------------------------------- #
# 外部死法 finish_requests（m17）：abort 双投递的引擎侧落点
# --------------------------------------------------------------------------- #
class TestFinishRequests:
    def test_abort_running_request_full_path(self):
        """断连 abort → FINISHED_ABORTED、running 摘除、块归还、幂等"""
        sched = make_scheduler(num_gpu_blocks=4)
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        assert r1 in sched.running
        aborted = sched.finish_requests("r1", RequestStatus.FINISHED_ABORTED)
        assert aborted == [r1]
        assert r1.status == RequestStatus.FINISHED_ABORTED
        assert r1.get_finished_reason() == FinishReason.ABORT
        assert r1 not in sched.running
        assert r1.request_id not in sched.requests
        assert sched.kv_cache_manager.num_free_blocks == 4
        assert sched.finished_req_ids == {"r1"}
        # 幂等：再 abort 同一 id → no-op
        assert sched.finish_requests("r1", RequestStatus.FINISHED_ABORTED) == []
        # 未知 id 同样 no-op
        assert sched.finish_requests("ghost", RequestStatus.FINISHED_ABORTED) == []

    def test_abort_waiting_request(self):
        """WAITING 态摘除走 waiting/skipped 双队列（L2281-L2283）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        assert r1 in sched.waiting
        aborted = sched.finish_requests(["r1"], RequestStatus.FINISHED_ABORTED)
        assert aborted == [r1]
        assert r1 not in sched.waiting
        assert r1.request_id not in sched.requests

    def test_abort_preempted_request(self):
        """被抢者（在 waiting 里）abort：同样走 waiting 摘除"""
        sched = make_scheduler(num_gpu_blocks=1)
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        step(sched, {})  # 触发自我抢占 → PREEMPTED 回 waiting
        assert r1.status == RequestStatus.PREEMPTED
        aborted = sched.finish_requests(["r1"], RequestStatus.FINISHED_ABORTED)
        assert aborted == [r1]
        assert r1 not in sched.waiting

    def test_abort_during_execution_is_idempotent_in_update(self):
        """执行期 abort：update_from_output 对已完成请求 continue 不炸（L1747-L1755）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})  # prefill
        out = sched.schedule()  # r1 decode 已调度（in_flight=1）
        # 客户端断连：输出回来前 finish（ch9 abort 双投递的第二落点）
        sched.finish_requests("r1", RequestStatus.FINISHED_ABORTED)
        mro = ModelRunnerOutput(
            req_ids=["r1"], req_id_to_index={"r1": 0}, sampled_token_ids=[[42]]
        )
        outputs = sched.update_from_output(out, mro)
        flat = [o for eco in outputs.values() for o in eco.outputs]
        assert flat == []  # 已完成的请求不产输出，也不报错


# --------------------------------------------------------------------------- #
# remove_all 快路径（m20）与其他收尾
# --------------------------------------------------------------------------- #
class TestRemoveAll:
    def test_single_item_fast_path_in_place(self):
        """单元素走 in-place remove 快路径（utils.py:L84-L89）"""
        lst = ["a", "b", "c"]
        ret = remove_all(lst, {"b"})
        assert ret is lst  # 原地改
        assert lst == ["a", "c"]

    def test_multi_items_returns_new_list(self):
        lst = ["a", "b", "c", "d"]
        ret = remove_all(lst, {"a", "c"})
        assert ret == ["b", "d"]
        assert lst == ["a", "b", "c", "d"]  # 原 list 不动

    def test_empty_set_noop(self):
        lst = ["a"]
        assert remove_all(lst, set()) is lst


class TestHandleStopped:
    def test_non_resumable_always_true(self):
        """流式会话已删：_handle_stopped_request 恒 True（真完成）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        step(sched, {"r1": [1]})
        assert sched._handle_stopped_request(r1) is True


class TestLedgerCounters:
    def test_request_lifecycle_counters_initialized(self):
        """四个生命周期计数器构造即置零（request.py:L150-L162）"""
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        assert r1.num_output_placeholders == 0
        assert r1.num_stale_output_tokens == 0
        assert r1.drop_stale_output is False
        assert r1.num_in_flight_tokens == 0
        assert r1.num_preemptions == 0
        assert r1.num_computed_tokens == 0

    def test_append_output_token_ids_extends_block_hashes(self):
        """append 连带增量块哈希——前缀恢复的伏线在每次输出时都在续（L249-L260）"""
        sched = make_scheduler(block_size=4)
        r1 = make_request(sched, "r1", 6)  # 6 token：1 满块 + 2 余
        assert len(r1.block_hashes) == 1  # 构造时已算 1 个满块
        r1.append_output_token_ids(7)  # 第 7 token：仍未满第 2 块
        assert len(r1.block_hashes) == 1
        r1.append_output_token_ids(8)  # 第 8 token：第 2 块满
        assert len(r1.block_hashes) == 2

    def test_get_request_counts(self):
        sched = make_scheduler()
        r1 = make_request(sched, "r1", 16)
        sched.add_request(r1)
        assert sched.get_request_counts() == (0, 1)
        step(sched, {"r1": [1]})
        assert sched.get_request_counts() == (1, 0)
