"""ch12 异步调度 —— 单元+契约测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测行为**
（锚点 = vllm/... 行号，基线 v0.27.1 现核，非 v2 资产的 v0.21.0 旧行号）：
- 默认仲裁：async_scheduling=None → True（pooling 模型/非 EAGLE 系 spec 方法/
  disable_padded_drafter_batch/executor 不支持/ROCm DBO 五类才降 False）
  （vllm/config/vllm.py:L1095-L1143）；显式 True 撞不兼容直接 raise（L1064-L1094）
- 深度仲裁：max_concurrent_batches——async+V1+单 PP→2 / V2 runner→pp_size+1 /
  纯 PP→pp_size / 无 async 无 PP→1（vllm/config/vllm.py:L539-L550）
- 换型：get_scheduler_cls async → AsyncScheduler（vllm/config/scheduler.py:L170-L178）
- 装配：batch_queue_size>1 才建 deque(maxlen)；step_fn 静态绑定只看队列建没建
  （core.py:L206-L212 / L231-L233）
- step_with_batch_queue 两态循环（core.py:L625-L739）：上半段盲调度→
  execute_model(non_block)→立即采样或 deferred→appendleft 三元组→
  『填管道优先』return (None, model_executed)；下半段 pop 最老批→
  future.result（None ⇒ exec_future 重抛真异常）→aborts→update_from_output→
  deferred 补采重新入队（复用同一 exec_future）
- post_step 三条件短路：async 下 draft token 在 worker 进程更新（core.py:L616-L623）
- has_work：bool(batch_queue) 在飞批保活（core.py:L1365-L1371）
- InprocClient：离线门面直调 step_fn、outputs None 兜底（core_client.py:L306-L322）
- 占位账本（async_scheduler.py:L19-L49 / L51-L70）：非 prefill-chunk 请求
  num_output_placeholders += num_sampled_tokens_per_step + spec 数、
  spec_token_ids 换 -1 占位列表、pending_structured_output_tokens 置位；
  真 token 到达扣减（stale 不扣，防 underflow）+ cache_blocks(computed−ph) 转正
- 追赶公式占位项：num_new_tokens = num_tokens_with_spec + ph − computed
  （scheduler.py:L516-L520）；early-stop 剪枝 computed+2−ph ≥ prompt+max_tokens
  跳过多余一步（scheduler.py:L488-L502）
- 抢占 async 账单：num_stale_output_tokens = num_in_flight_tokens（赋值不累加）、
  占位清零（scheduler.py:L1296-L1308）；热循环锁步 drain + stale 照送不动计数器
  （scheduler.py:L1736-L1743）；spec 拒绝回扣 computed/ph（L1769-L1784，stale 不回扣）
- worker 影子①②：采样 token 留 GPU（prev_sampled_token_ids 缓存张量 +
  prev_req_id_to_index 槽位表）、token_ids_cpu 行只写 -1、num_tokens_no_spec
  照常推进（gpu_model_runner.py:L3797-L3842；gpu_input_batch.py:L309-L311）
- GPU 回填：_prepare_input_ids 正常拍直拷 / async 拍 common-case 单 slice 直拷 /
  变过按 index scatter（gpu_model_runner.py:L1784-L1891；_compute_prev_positions
  L1769-L1782）；乐观 seq_lens + discard_request_mask（L2081-L2105）
- 异步输出对：AsyncGPUModelRunnerOutput 构造即发起拷贝+event.record 不等待、
  get_output event.synchronize 才放行（gpu_model_runner.py:L259-L350）；
  AsyncOutputFuture result() 才 get_output 惰性收割（uniproc_executor.py:L26-L42）
- 同步禁区 tripwire：enable_gpu_sync_check 门未开 wrapper 直通、非 CUDA 平台
  no-op（gpu_worker.py:L846-L848 + utils/gpu_sync_debug.py:L26-L33/L158-L165）
"""
import os
import sys
import threading
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.async_scheduler import AsyncScheduler  # noqa: E402
from implementation.core_client import InprocClient  # noqa: E402
from implementation.core import EngineCore  # noqa: E402
from implementation.gpu_model_runner import (  # noqa: E402
    AsyncGPUModelRunnerOutput,
    GPUModelRunner,
)
from implementation.outputs import (  # noqa: E402
    EMPTY_MODEL_RUNNER_OUTPUT,
    ModelRunnerOutput,
)
from implementation.request import Request, RequestStatus, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.uniproc_executor import (  # noqa: E402
    AsyncOutputFuture,
    UniProcExecutor,
)
from implementation.vllm_config import VllmConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# 构造辅助：真实装配的最小镜像
# --------------------------------------------------------------------------- #


def make_vllm_config(
    *,
    async_scheduling=None,
    pp_size=1,
    runner_type="generate",
    spec_method=None,
    disable_padded_drafter_batch=False,
    executor_backend="uniproc",
    use_v2_model_runner=False,
    max_model_len=64,
    arbitrate=True,
):
    """配置栈：SchedulerConfig + Parallel/Model/Speculative + VllmConfig。

    arbiter 在 VllmConfig.__post_init__ 里就地跑（vllm/config/vllm.py:L972 起的
    async 段即 L1064-L1143），与真实装配一致。
    """
    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(async_scheduling=async_scheduling),
        pp_size=pp_size,
        runner_type=runner_type,
        spec_method=spec_method,
        disable_padded_drafter_batch=disable_padded_drafter_batch,
        executor_backend=executor_backend,
        use_v2_model_runner=use_v2_model_runner,
        max_model_len=max_model_len,
    )
    if arbitrate:
        cfg.check_and_set_default_async_scheduling()
    return cfg


def make_request(
    request_id="req-0",
    prompt=(1, 2),
    max_tokens=8,
    structured=False,
):
    params = SamplingParams(max_tokens=max_tokens)
    req = Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        sampling_params=params,
        structured_output_request=object() if structured else None,
    )
    return req


def one_hot_row(token_id: int, vocab: int = 16) -> list[float]:
    """脚本化 logits 行：argmax == token_id（greedy 采样可预测）。"""
    row = [0.0] * vocab
    row[token_id] = 1.0
    return row


class ScriptedExecutor:
    """Executor 替身：与 UniProcExecutor 同接口（execute_model/sample_tokens/
    take_draft_token_ids 都收 non_block），但 forward 结果由脚本给定、
    copy 完成时机由测试显式驱动（release()）。仅用于 B 段循环形状测试；
    D/E 段走真实 UniProcExecutor + worker 链。"""

    def __init__(self, sampled_rows=None):
        self.sampled_rows = list(sampled_rows or [])
        self.execute_calls = 0
        self.sample_calls = 0
        self.take_draft_calls = 0
        self.fail_execute_with = None

    def execute_model(self, scheduler_output, non_block=False):
        self.execute_calls += 1
        # concurrent.futures.Future 与 EngineCore 用法兼容（result/done）
        from concurrent.futures import Future

        fut = Future()
        if self.fail_execute_with is not None:
            fut.set_exception(self.fail_execute_with)
        else:
            # 真实 uniproc：空批返回 EMPTY（ch9 两段式契约的 0-token 早退支）
            fut.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
        return fut

    def sample_tokens(self, grammar_output, non_block=False):
        from concurrent.futures import Future

        self.sample_calls += 1
        fut = Future()
        if self.sampled_rows:
            row = self.sampled_rows.pop(0)
            fut.set_result(
                ModelRunnerOutput(
                    req_ids=["req-0"],
                    req_id_to_index={"req-0": 0},
                    sampled_token_ids=[list(row)],
                )
            )
        else:
            fut.set_result(None)  # 模拟 sample_tokens 拿不到输出的失败路径
        return fut

    def take_draft_token_ids(self):
        self.take_draft_calls += 1
        return None


def make_engine(cfg, executor=None):
    return EngineCore(cfg, model_executor=executor)


# =========================================================================== #
# A 配置三件套：仲裁 / 深度 / 换型（m1 / m2 / m3）
# =========================================================================== #


def test_arbitration_none_defaults_to_true():
    # vllm/config/vllm.py:L1095-L1143：默认 None → True（『默认心跳』出处）
    cfg = make_vllm_config(async_scheduling=None)
    assert cfg.scheduler_config.async_scheduling is True


def test_arbitration_pooling_model_disables():
    # vllm.py:L1097-L1106：pooling 模型异步反而拖慢 → 默认关
    cfg = make_vllm_config(runner_type="pooling")
    assert cfg.scheduler_config.async_scheduling is False


def test_arbitration_non_eagle_spec_method_disables():
    # vllm.py:L1107-L1118：medusa 不在 EAGLE 系/NgramGPU/dspark 之列
    cfg = make_vllm_config(spec_method="medusa")
    assert cfg.scheduler_config.async_scheduling is False


def test_arbitration_eagle_spec_method_stays_async():
    # vllm.py:L1107-L1118 反例：eagle 在 EAGLE 系 → 落到 else → True
    cfg = make_vllm_config(spec_method="eagle")
    assert cfg.scheduler_config.async_scheduling is True


def test_arbitration_disable_padded_drafter_batch_disables():
    # vllm.py:L1119-L1127
    cfg = make_vllm_config(
        spec_method="eagle", disable_padded_drafter_batch=True
    )
    assert cfg.scheduler_config.async_scheduling is False


def test_arbitration_unsupported_executor_disables():
    # vllm.py:L1128-L1134：ray 后端 supports_async_scheduling() 未覆写 → False
    cfg = make_vllm_config(executor_backend="ray")
    assert cfg.scheduler_config.async_scheduling is False


def test_explicit_true_hard_fails_on_unsupported_executor():
    # vllm.py:L1064-L1094：显式 True 撞不兼容直接 raise（不做静默降级）
    cfg = make_vllm_config(
        async_scheduling=True, executor_backend="ray", arbitrate=False
    )
    with pytest.raises(ValueError, match="does not support async scheduling"):
        cfg.check_and_set_default_async_scheduling()


def test_max_concurrent_batches_matrix():
    # vllm/config/vllm.py:L539-L550 全矩阵
    # async + V1 + 单 PP → 2（重叠双缓冲）
    cfg = make_vllm_config(async_scheduling=True, arbitrate=False)
    assert cfg.max_concurrent_batches == 2
    # async + V2 runner → pp_size + 1（消末段气泡）
    cfg = make_vllm_config(
        async_scheduling=True, use_v2_model_runner=True, pp_size=4, arbitrate=False
    )
    assert cfg.max_concurrent_batches == 5
    # V1 + async + PP>1 → 落到 return pp_size（V1 不完全支持 async+PP）
    cfg = make_vllm_config(async_scheduling=True, pp_size=4, arbitrate=False)
    assert cfg.max_concurrent_batches == 4
    # 无 async 纯 PP → pp_size（填流水线）
    cfg = make_vllm_config(async_scheduling=False, pp_size=4, arbitrate=False)
    assert cfg.max_concurrent_batches == 4
    # 无 async 无 PP → 1（不建队列）
    cfg = make_vllm_config(async_scheduling=False, pp_size=1, arbitrate=False)
    assert cfg.max_concurrent_batches == 1


def test_get_scheduler_cls_swaps_scheduler_class():
    # vllm/config/scheduler.py:L170-L178：async 不只换 step，连调度器类一起换
    assert (
        SchedulerConfig(async_scheduling=True).get_scheduler_cls() is AsyncScheduler
    )
    assert SchedulerConfig(async_scheduling=False).get_scheduler_cls() is Scheduler
    # 仲裁后的默认（None→True）也换型
    cfg = make_vllm_config()
    assert cfg.scheduler_config.get_scheduler_cls() is AsyncScheduler


def test_engine_assembly_queue_and_step_fn_binding():
    # core.py:L206-L212：>1 才建 deque(maxlen)；L231-L233：step_fn 静态绑定
    # 只看队列建没建，不看 async_scheduling 字样
    cfg = make_vllm_config()  # 默认仲裁 → async → 深度 2
    engine = make_engine(cfg, executor=ScriptedExecutor())
    assert engine.batch_queue_size == 2
    assert engine.batch_queue is not None
    assert engine.batch_queue.maxlen == 2
    assert engine.step_fn == engine.step_with_batch_queue
    assert engine.async_scheduling is True

    # 关掉 async：深度 1 → 不建队列 → step_fn 是同步 step（三级间接的落点）
    cfg_sync = make_vllm_config(async_scheduling=False)
    engine_sync = make_engine(cfg_sync, executor=ScriptedExecutor())
    assert engine_sync.batch_queue is None
    assert engine_sync.step_fn == engine_sync.step


# =========================================================================== #
# B step_with_batch_queue 两态循环（m4 / m5 / m21 / m17 / m20）
# =========================================================================== #


def _populated_async_engine(sampled_rows, max_tokens=8):
    """默认仲裁配置 + AsyncScheduler + 脚本 executor + 一个 2-token 请求。"""
    cfg = make_vllm_config()
    executor = ScriptedExecutor(sampled_rows=sampled_rows)
    engine = make_engine(cfg, executor=executor)
    req = make_request(max_tokens=max_tokens)
    engine.scheduler.add_request(req)
    return engine, executor, req


def test_fill_pipeline_priority_returns_none_until_full():
    # core.py:L679-L687：队未满且还有活 → return (None, model_executed)，
    # 不等结果——上半段『填管道优先于取模型输出』
    engine, executor, req = _populated_async_engine([[7]])
    outputs, executed = engine.step_with_batch_queue()
    assert outputs is None
    assert executed is True
    assert len(engine.batch_queue) == 1  # 批 A 已在飞


def test_queue_full_pops_oldest_and_returns_outputs():
    # core.py:L695-L714：第二次调用先照常调度批 B，队满 [B, A] → pop 最老批 A
    # → future.result → update_from_output → 返回真输出（deque appendleft/pop = FIFO）
    engine, executor, req = _populated_async_engine([[7], [8]])
    engine.step_with_batch_queue()  # 批 A 入队
    outputs, executed = engine.step_with_batch_queue()  # 调度 B + 收 A
    assert executed is True
    assert outputs is not None
    # 批 A 是 prefill（2 token）→ update_from_output 产出 req-0 的首 token 7
    eco = outputs[0].outputs[0]
    assert eco.request_id == "req-0"
    assert eco.new_token_ids == [7]
    assert len(engine.batch_queue) == 1  # 还剩批 B 在飞


def test_none_from_sample_reraises_execute_exception():
    # core.py:L701-L706：sample future 出 None ⇒ 原 execute_model 失败——
    # exec_model_fut.result() 重抛真异常，而不是吞成 "unexpected error"
    engine, executor, req = _populated_async_engine([])  # sample → None
    boom = RuntimeError("real worker failure")
    executor.fail_execute_with = boom
    engine.step_with_batch_queue()  # 批 A 入队（sample future=None 已定）
    with pytest.raises(RuntimeError, match="real worker failure"):
        engine.step_with_batch_queue()


def test_has_work_kept_alive_by_inflight_batch():
    # core.py:L1365-L1371：scheduler 无请求但队列非空 → has_work() True
    # （max_tokens=2：拍3 剪枝出空批 C，拍4 pop 完队列才清空）
    engine, executor, req = _populated_async_engine([[7], [8]], max_tokens=2)
    engine.step_with_batch_queue()  # A 入队
    engine.step_with_batch_queue()  # B 入队 + pop A（t7 到账）
    engine.step_with_batch_queue()  # C 空批入队 + pop B（t8 → 长度封顶完成）
    # 请求已终，但队列还挂着空批 C —— has_work 靠 bool(batch_queue) 保活
    assert len(engine.batch_queue) == 1
    assert engine.has_work() is True
    # 忙循环继续转：在飞批收完（含 finished_req_ids 清账拍的空批）才归 False
    while engine.batch_queue:
        engine.step_with_batch_queue()
    assert engine.has_work() is False


def test_post_step_async_short_circuits_draft_update():
    # core.py:L616-L623：三条件短路——async 下 take_draft_token_ids 不回传主进程
    cfg = make_vllm_config(spec_method="eagle")  # async + check_for_draft_tokens
    engine = make_engine(cfg, executor=ScriptedExecutor())
    assert engine.check_for_draft_tokens is True
    engine.post_step(model_executed=True)
    assert engine.model_executor.take_draft_calls == 0
    # 同步版：check_for_draft_tokens and not async and model_executed → 取
    cfg_sync = make_vllm_config(async_scheduling=False, spec_method="eagle")
    engine_sync = make_engine(cfg_sync, executor=ScriptedExecutor())
    engine_sync.post_step(model_executed=True)
    assert engine_sync.model_executor.take_draft_calls == 1
    engine_sync.post_step(model_executed=False)  # model_executed=False → 不取
    assert engine_sync.model_executor.take_draft_calls == 1


def test_inproc_client_handles_none_outputs():
    # core_client.py:L319-L322：离线门面直调 step_fn；outputs 为 None 兜底
    cfg = make_vllm_config()
    client = InprocClient(cfg, model_executor=ScriptedExecutor())
    req = make_request(max_tokens=8)
    client.engine_core.scheduler.add_request(req)
    first = client.get_output()  # 上半段 return (None, True) → 兜底空包
    assert list(first.outputs) == []
    assert client.engine_core.step_fn == client.engine_core.step_with_batch_queue


# =========================================================================== #
# C 占位账本（m6 / m7 / m8 / m9 / m15 / m16）
# =========================================================================== #


def _async_scheduler(max_model_len=64, num_gpu_blocks=64):
    cfg = make_vllm_config()
    sched = cfg.scheduler_config.get_scheduler_cls()(
        vllm_config=cfg, log_stats=False, num_gpu_blocks=num_gpu_blocks
    )
    assert isinstance(sched, AsyncScheduler)
    return sched


def test_placeholder_plus_one_on_decode_schedule():
    # async_scheduler.py:L38-L44：非 prefill-chunk 请求 ph += 1（无 spec）
    sched = _async_scheduler()
    req = make_request()
    sched.add_request(req)
    out = sched.schedule()
    assert req.num_computed_tokens == 2  # prefill 全量（追赶=2+0-0）
    assert req.is_prefill_chunk is False  # 全量 prefill 不是 chunk
    assert req.num_output_placeholders == 1  # 占位 +1
    # spec_token_ids 换成 -1 占位列表（num_spec_tokens_to_schedule=0 → 空）
    assert req.spec_token_ids == []


def test_prefill_chunk_gets_no_placeholder():
    # async_scheduler.py:L28-L29：is_prefill_chunk → continue（不占位）
    sched = _async_scheduler()
    long_req = make_request(prompt=tuple(range(6)), max_tokens=8)
    sched.scheduler_config.max_num_scheduled_tokens = 4  # 6-token prompt 分两块
    sched.max_num_scheduled_tokens = 4
    sched.add_request(long_req)
    sched.schedule()
    assert long_req.is_prefill_chunk is True
    assert long_req.num_output_placeholders == 0
    # 第二拍把余下 2 token 排完 → 仍非 chunk 全量收尾后才占位
    sched.schedule()
    assert long_req.num_output_placeholders == 1


def test_catchup_formula_blind_schedules_next_position():
    # scheduler.py:L516-L520：输出未回也能算出『本拍排 1 个位置』
    # —— num_new_tokens = num_tokens_with_spec + ph - computed
    sched = _async_scheduler()
    req = make_request()
    sched.add_request(req)
    sched.schedule()  # 拍 1：prefill，ph=1
    out2 = sched.schedule()  # 拍 2：盲调度（此刻 t 还没回来）
    assert out2.num_scheduled_tokens[req.request_id] == 1
    assert req.num_output_placeholders == 2
    assert req.num_in_flight_tokens == 3  # 2 + 1（乐观计入在飞步）


def test_early_stop_pruning_skips_extra_step():
    # scheduler.py:L488-L502：computed+2−ph ≥ prompt+max_tokens → 跳过
    # worked example 拍 3：computed=3, ph=1 → 3+2-1=4 ≥ 2+2=4
    # （ph=1 的前提：拍 2 的批 A 已 pop 交货——engine 两态循环同拍完成）
    sched = _async_scheduler()
    req = make_request(prompt=(1, 2), max_tokens=2)
    sched.add_request(req)
    sched.schedule()  # 拍 1：prefill 2 token；ph=1, computed=2
    sched.schedule()  # 拍 2：排 1 个位置；ph=2, computed=3
    # 拍 2 下半段（pop 批 A）：t1=7 到账 → ph 2-1=1、tws=3
    mro = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[7]],
    )
    sched.update_from_output(sched.last_output, mro)
    out3 = sched.schedule()  # 拍 3：确信上拍已达 max_tokens → 剪枝
    assert req.request_id not in out3.num_scheduled_tokens
    assert out3.total_num_scheduled_tokens == 0


def test_pending_structured_output_flag_only_when_placeholder():
    # async_scheduler.py:L31-L33：use_structured_output 且 ph>0 才置 pending
    sched = _async_scheduler()
    req = make_request(structured=True)
    sched.add_request(req)
    out1 = sched.schedule()  # prefill：ph 仍是 0 → pending 不置
    assert out1.pending_structured_output_tokens is False
    assert out1.has_structured_output_requests is True
    out2 = sched.schedule()  # decode：ph=1>0 → pending 置位
    assert out2.pending_structured_output_tokens is True


def test_delivery_decrements_placeholders_and_converts_blocks():
    # async_scheduler.py:L51-L70：真 token 到达 ph -= len(new_token_ids)
    # （assert ≥0）；cache_blocks(computed − ph) 把乐观块转正式
    sched = _async_scheduler()
    req = make_request()
    sched.add_request(req)
    sched.schedule()
    sched.schedule()  # ph=2, computed=3
    mro = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[7]],
    )
    outs = sched.update_from_output(sched.last_output, mro)
    assert req.output_token_ids == [7]
    assert req.num_output_placeholders == 1  # 2-1
    # cache_blocks 参数 = computed(3) − ph(1) = 2 = 真实已算（不变式化身）
    assert sched.kv_cache_manager.cache_blocks_calls[-1] == (
        req.request_id,
        2,
    )


def test_stale_delivery_does_not_decrement_placeholders():
    # async_scheduler.py:L59-L63：抢占时占位已清零，stale 送达再扣就 underflow
    sched = _async_scheduler()
    req = make_request()
    sched.add_request(req)
    sched.schedule()  # ph=1
    # 手工制造 stale 送达（真实由抢占+恢复产生）：token 照收，占位不动
    new_tokens, stopped = sched._update_request_with_output(
        req, [7], is_stale=True
    )
    assert req.output_token_ids == [7]  # token 照收
    assert req.num_output_placeholders == 1  # 不扣（扣了就 underflow）


def test_preempt_zeros_placeholders_and_marks_stale():
    # scheduler.py:L1296-L1308：num_stale_output_tokens = num_in_flight_tokens
    # （赋值不累加）；num_output_placeholders = 0
    sched = _async_scheduler()
    req = make_request()
    sched.add_request(req)
    sched.schedule()
    sched.schedule()
    assert req.num_output_placeholders == 2
    assert req.num_in_flight_tokens == 3
    sched._preempt_request(req, time.monotonic())
    assert req.num_output_placeholders == 0
    assert req.num_stale_output_tokens == 3  # = 在飞数，赋值不累加
    assert req.num_preemptions == 1
    assert req.status == RequestStatus.PREEMPTED


def test_stale_lockstep_drain_delivers_without_touching_counters():
    # scheduler.py:L1736-L1743 + async_scheduler.py stale 路径：
    # 恢复后的 stale 输出照送（保 spec acceptance），但锁步扣减 stale 计数、
    # 不回扣已清零的占位
    sched = _async_scheduler()
    req = make_request()
    sched.add_request(req)
    sched.schedule()
    sched.schedule()
    sched._preempt_request(req, time.monotonic())  # stale=3, ph=0
    # 恢复：重新 prefill 全量 2 token
    out = sched.schedule()
    assert req.num_output_placeholders == 1
    # 送达一拍 1 token（stale 在途的输出照送）
    mro = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[7]],
    )
    outs = sched.update_from_output(sched.last_output, mro)
    eco = outs[0].outputs[0]
    assert eco.new_token_ids == [7]  # 照送
    # stale 计数锁步扣减（3-2=1：恢复拍只排了 2 token）
    assert req.num_stale_output_tokens == 1
    # 非 stale 时扣占位；本拍请求是 RUNNING 且非 stale → ph 2-1=1？
    # 注意：恢复拍的 delivery 非 stale（stale 的是旧在飞输出，尚未到账）
    assert req.num_output_placeholders == 1


def test_spec_rejection_rolls_back_computed_and_placeholders():
    # scheduler.py:L1769-L1784：num_rejected 同步回退 computed 与 ph；
    # stale 不回扣（stale 的拒绝数先于抢占回滚、不得应用）
    sched = _async_scheduler()
    req = make_request()
    sched.add_request(req)
    sched.schedule()  # 拍 1：prefill，computed=2、ph=1
    so = sched.last_output
    # 人灌 spec 状态：本批带 3 个草稿 token（spec 深水归 ch33，此处只对账）
    so.scheduled_spec_decode_tokens[req.request_id] = [11, 12, 13]
    req.spec_token_ids = [11, 12, 13]
    # 采样结果：只有 bonus token（num_sampled=1），0 草稿被接受 → 拒绝 3
    mro = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[7]],
    )
    computed_before = req.num_computed_tokens
    # 占位语义对齐 _update_after_schedule：1 个 bonus + 3 个 spec = 4
    req.num_output_placeholders = 4
    sched.update_from_output(so, mro)
    num_rejected = 3
    assert req.num_computed_tokens == computed_before - num_rejected
    # 回扣 3（spec 全拒）+ delivery [7] 扣 1（bonus）→ 4-3-1=0
    assert req.num_output_placeholders == 0


# =========================================================================== #
# D worker 影子状态（m10 / m11 / m12 / m13 / m18 / m19）
# =========================================================================== #


def _runner(max_num_reqs=8, max_model_len=64, vocab=16, async_on=True):
    cfg = make_vllm_config(async_scheduling=async_on if async_on else False)
    cfg.scheduler_config.async_scheduling = async_on
    return GPUModelRunner(
        cfg, max_num_reqs=max_num_reqs, max_model_len=max_model_len, vocab_size=vocab
    )


def _fill_batch(runner, reqs_prompts: dict[str, list[int]]):
    """把请求按序落进持久批（_update_states 的 seam 面）。"""
    for rid, prompt in reqs_prompts.items():
        runner.input_batch.add_request(rid, prompt)
        runner.requests[rid] = type(
            "ReqState", (), {"num_tokens": len(prompt), "all_token_ids": list(prompt)}
        )()


def test_bookkeeping_caches_gpu_tensor_and_writes_minus_one():
    # gpu_model_runner.py:L3797-L3842：async 分支——sampled_token_ids 整张量
    # 缓存进 prev_sampled_token_ids（不 _to_list）；token_ids_cpu 行只写 -1；
    # is_token_ids=True、num_tokens_no_spec 照常推进
    runner = _runner()
    _fill_batch(runner, {"req-0": [1, 2, 3]})
    runner.input_batch.num_tokens_no_spec[0] = 3
    sampled = torch.tensor([[9]], dtype=torch.int64)

    class _SO:  # SamplerOutput 站位
        sampled_token_ids = sampled
        logprobs_tensors = None

    runner._bookkeeping_sync(
        scheduler_output=None,
        sampler_output=_SO(),
        logits=None,
        hidden_states=None,
        num_scheduled_tokens=1,
    )

    assert runner.input_batch.prev_sampled_token_ids is sampled  # 同一张量=留 GPU
    assert runner.input_batch.prev_req_id_to_index == {"req-0": 0}
    assert runner.input_batch.token_ids_cpu[0, 3] == -1  # 占位 -1
    assert runner.input_batch.is_token_ids[0, 3]
    assert runner.input_batch.num_tokens_no_spec[0] == 4  # 账本照走


def test_prev_req_id_to_index_excludes_discarded_rows():
    # gpu_model_runner.py:L3809-L3813：discard 行不进槽位表
    runner = _runner()
    _fill_batch(runner, {"req-0": [1], "req-1": [2], "req-2": [3]})
    for i in range(3):
        runner.input_batch.num_tokens_no_spec[i] = 1
    sampled = torch.tensor([[7], [8], [9]], dtype=torch.int64)
    runner.input_batch.prev_sampled_token_ids = None

    class _SO:
        sampled_token_ids = sampled
        logprobs_tensors = None

    # req-1 被标丢弃（discard_request_mask=True → invalid 行）
    runner.discard_request_mask.np[1] = True
    runner._bookkeeping_sync(
        scheduler_output=None,
        sampler_output=_SO(),
        logits=None,
        hidden_states=None,
        num_scheduled_tokens=3,
    )
    assert runner.input_batch.prev_req_id_to_index == {"req-0": 0, "req-2": 2}
    # invalid 行不写占位；有效行写 -1
    assert runner.input_batch.token_ids_cpu[1, 1] == 0  # 未动
    assert runner.input_batch.token_ids_cpu[0, 1] == -1


def _prime_prev(runner, tokens: list[int]):
    """伪造上拍影子：prev_sampled_token_ids=Nx1 GPU 张量 + 槽位表。"""
    runner.input_batch.prev_sampled_token_ids = torch.tensor(
        [[t] for t in tokens], dtype=torch.int32
    )
    runner.input_batch.prev_req_id_to_index = {
        rid: i for i, rid in enumerate(runner.input_batch.req_ids)
    }


def test_prepare_input_ids_normal_case_copies_cpu_buffer():
    # gpu_model_runner.py:L1801-L1807：prev 为 None（正常拍）→ 整段 copy_to_gpu
    runner = _runner()
    _fill_batch(runner, {"req-0": [1, 2]})
    runner.input_ids.cpu[:2] = [1, 2]  # CPU 侧已备好的调度窗口

    class _SO:
        scheduled_spec_decode_tokens = {}

    runner._prepare_input_ids(_SO(), num_reqs=1, total_num_scheduled_tokens=2,
                              cu_num_tokens=np.array([2], dtype=np.int64))
    assert runner.input_ids.gpu[:2].tolist() == [1, 2]


def test_prepare_input_ids_common_case_single_slice():
    # gpu_model_runner.py:L1868-L1877：批次未变未重排 → 单 slice 直拷
    # （真实调用序：_compute_prev_positions 在 L2095 先于 _prepare_input_ids）
    runner = _runner()
    _fill_batch(runner, {"req-0": [1, 2], "req-1": [3, 4]})
    _prime_prev(runner, [7, 8])  # 上拍采出 7/8
    runner._compute_prev_positions(2)
    # 本拍各排 1 个 decode token → cu 末端=[1,2]，flattened_index=0/1 == prev 0/1
    runner._prepare_input_ids(
        type("SO", (), {"scheduled_spec_decode_tokens": {}})(),
        num_reqs=2, total_num_scheduled_tokens=2,
        cu_num_tokens=np.array([1, 2], dtype=np.int64),
    )
    assert runner.input_ids.gpu[:2].tolist() == [7, 8]


def test_prepare_input_ids_scatters_when_reordered():
    # gpu_model_runner.py:L1878-L1891：批次变过/重排 → 按 index scatter 回填
    runner = _runner()
    _fill_batch(runner, {"req-0": [1, 2], "req-1": [3, 4], "req-2": [5, 6]})
    _prime_prev(runner, [7, 8, 9])
    # 重排：req-2 提到最前 + 新请求 req-x 垫中间（经持久批的加删 seam 落位）
    for rid in ("req-2", "req-x", "req-0", "req-1"):
        runner.input_batch.remove_request(rid)
    for rid, prompt in (
        ("req-2", [5, 6]),
        ("req-x", [11, 12]),
        ("req-0", [1, 2]),
    ):
        runner.input_batch.add_request(rid, prompt)
    runner.input_batch.prev_req_id_to_index = {"req-0": 0, "req-1": 1, "req-2": 2}
    runner._compute_prev_positions(3)
    # input_ids.cpu 已含 CPU 侧可见部分（新请求/预填窗口），common 之外先落底
    runner.input_ids.cpu[:3] = [100, 101, 102]
    runner.input_ids.gpu[:3] = torch.tensor([100, 101, 102], dtype=torch.int32)
    runner._prepare_input_ids(
        type("SO", (), {"scheduled_spec_decode_tokens": {}})(),
        num_reqs=3, total_num_scheduled_tokens=3,
        cu_num_tokens=np.array([1, 2, 3], dtype=np.int64),
    )
    got = runner.input_ids.gpu[:3].tolist()
    # 位置 0（req-2 的采样 token）从 prev 行 2 回填；req-x 无 prev 保留 CPU 值；
    # 位置 2（req-0 的采样 token）从 prev 行 0 回填
    assert got == [9, 101, 7]


def test_compute_prev_positions_maps_new_requests_to_minus_one():
    # gpu_model_runner.py:L1769-L1782：cur → prev 映射，-1=新请求
    runner = _runner()
    _fill_batch(runner, {"req-0": [1], "req-1": [2]})
    runner.input_batch.prev_req_id_to_index = {"req-1": 0}
    runner._compute_prev_positions(2)
    assert runner.prev_positions.np[:2].tolist() == [-1, 0]


def test_optimistic_seq_lens_and_discard_mask():
    # gpu_model_runner.py:L2081-L2105：乐观 seq_lens = computed + scheduled
    # （假定 draft 全接受）；discard = optimistic < num_tokens
    runner = _runner()
    _fill_batch(runner, {"req-0": [1, 2], "req-1": [3, 4, 5, 6]})
    runner.input_batch.num_computed_tokens_cpu_tensor[:2] = torch.tensor([2, 2])
    num_scheduled = np.array([1, 1], dtype=np.int32)
    runner._compute_optimistic_seq_lens(num_scheduled)
    assert runner.optimistic_seq_lens_cpu[:2].tolist() == [3, 3]
    runner._compute_discard_request_mask(2)
    # req-0: 3 >= 2 → 不丢弃；req-1: 3 < 4 → 丢弃（还没到采样位置）
    assert runner.discard_request_mask.np[:2].tolist() == [False, True]


def test_async_output_copy_event_gates_get_output():
    # gpu_model_runner.py:L286-L306/L308-L314：构造即发起拷贝+record（不等待）；
    # get_output 阻塞至事件完成才放行
    sampled = torch.tensor([[5], [6]], dtype=torch.int64)
    out = ModelRunnerOutput(
        req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}, sampled_token_ids=None
    )
    async_out = AsyncGPUModelRunnerOutput(
        model_runner_output=out,
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        invalid_req_indices=[1],
        async_output_copy_stream=None,  # host seam
        vocab_size=16,
    )
    # 拷贝已发起（host seam：buffer 就位）但事件未完成 → get_output 必须阻塞
    assert async_out.sampled_token_ids_cpu.shape == (2, 1)
    done = threading.Event()
    holder = {}

    def _get():
        holder["out"] = async_out.get_output()

    t = threading.Thread(target=_get, daemon=True)
    t.start()
    t.join(timeout=0.3)
    assert t.is_alive(), "get_output 在事件未完成前不得返回"
    async_out.async_copy_ready_event.set()  # 模拟 D2H DMA 完成
    t.join(timeout=2.0)
    assert not t.is_alive()
    result = holder["out"]
    # invalid 行清空、有效行落地（L319-L322 语义）
    assert result.sampled_token_ids == [[5], []]


def test_async_output_future_harvests_lazily_once():
    # uniproc_executor.py:L26-L42：result() 才 get_output；二次 result 不再收割
    calls = []

    class _Lazy(AsyncGPUModelRunnerOutput):  # 计数收割次数
        def get_output(self):
            calls.append(1)
            return EMPTY_MODEL_RUNNER_OUTPUT

    fut = AsyncOutputFuture(_Lazy(
        model_runner_output=EMPTY_MODEL_RUNNER_OUTPUT,
        sampled_token_ids=torch.zeros(1, 1, dtype=torch.int64),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=None,
        vocab_size=16,
    ), single_value=True)
    with pytest.raises(RuntimeError, match="timeout not implemented"):
        fut.result(timeout=1)
    assert calls == []
    assert fut.result() is EMPTY_MODEL_RUNNER_OUTPUT
    assert fut.result() is EMPTY_MODEL_RUNNER_OUTPUT
    assert len(calls) == 1


def test_gpu_sync_check_gate_off_until_enabled(monkeypatch):
    # gpu_worker.py:L846-L848 + utils/gpu_sync_debug.py:L26-L33：门未开 wrapper
    # 直通；enable 只在 VLLM_GPU_SYNC_CHECK 设置时生效；非 CUDA 平台 no-op
    from implementation import gpu_sync_debug

    monkeypatch.delenv("VLLM_GPU_SYNC_CHECK", raising=False)
    marker = []
    calls = []

    @gpu_sync_debug.with_gpu_sync_check
    def guarded():
        calls.append(1)
        return "ok"

    gpu_sync_debug._sync_check_enabled = True
    assert guarded() == "ok"  # 非 CUDA 平台：真实 else 分支即恒等装饰（L158-L165）
    assert calls == [1]
    marker  # noqa: B018


def test_sample_tokens_resets_prev_cache_then_recaches():
    # gpu_model_runner.py:L4609 + L3797-L3813：sample_tokens 开头清 prev 缓存、
    # bookkeeping 随后缓存新采样——上一拍的影子不会跨拍泄漏
    from implementation.output import SchedulerOutput

    runner = _runner()
    _fill_batch(runner, {"req-0": [1, 2]})
    _prime_prev(runner, [7])  # 上一拍的残留
    runner.enqueue_logits([{"req-0": one_hot_row(9)}])
    so = SchedulerOutput.make_empty()
    so.num_scheduled_tokens = {"req-0": 1}
    so.total_num_scheduled_tokens = 1
    runner.execute_model(so)  # 两段式：返回 None 暂存
    result = runner.sample_tokens(None)
    assert isinstance(result, AsyncGPUModelRunnerOutput)
    assert result.sampled_token_ids_cpu.tolist() == [[9]]
    assert runner.input_batch.prev_sampled_token_ids.tolist() == [[9]]  # 新缓存


# =========================================================================== #
# E 端到端：一轮重叠心跳（theory worked example：prompt=2、max_tokens=2）
# =========================================================================== #


class RealChain:
    """真实链装配：EngineCore → UniProcExecutor → worker → GPUModelRunner seam。

    forward 由脚本 logits 驱动（ch17 边界）；D2H 完成由 release() 显式驱动。
    """

    def __init__(self, *, prompt=(1, 2), max_tokens=2, structured=False):
        self.cfg = make_vllm_config()
        self.engine = EngineCore(self.cfg)
        req = make_request(prompt=prompt, max_tokens=max_tokens, structured=structured)
        self.req = req
        self.engine.scheduler.add_request(req)
        runner = self.engine.model_executor.driver_worker.model_runner
        self.runner = runner

    def script(self, tokens: list[int]):
        self.runner.enqueue_logits([{self.req.request_id: one_hot_row(t)} for t in tokens])

    def release(self):
        self.runner.release_async_copies()

    def step(self):
        outputs, executed = self.engine.step_fn()
        self.engine.post_step(model_executed=executed)
        return outputs, executed


def test_overlapped_heartbeat_four_beats():
    # 一轮重叠心跳全链：占位账本消长 + GPU 回填 + early-stop + 收尾
    # （theory[2] worked example：prompt=2、max_tokens=2、无 spec）
    chain = RealChain(max_tokens=2)
    req = chain.req
    engine = chain.engine
    chain.script([7, 9])  # 两拍采样脚本：t1=7, t2=9
    sched = engine.scheduler

    # 拍 1：盲调度 prefill；填管道优先 → (None, True)；A 在飞
    out1, ex1 = chain.step()
    assert out1 is None and ex1 is True
    assert len(engine.batch_queue) == 1
    assert req.num_output_placeholders == 1
    assert req.num_computed_tokens == 2
    assert req.num_computed_tokens - req.num_output_placeholders == 1  # 不变式

    # 拍 2：调度批 B（追赶公式 2+1-2=1——盲排下一个位置）；
    # 批 A 的 D2H 尚未完成 → future.result 等待前先放行 DMA
    chain.release()
    out2, ex2 = chain.step()
    assert ex2 is True
    # 账本：拍2 加 1（→2）、pop 批 A 交货 t7 又扣 1（→1）——同拍完成
    assert req.num_output_placeholders == 1
    assert req.num_computed_tokens == 3
    # 拍 2 内 pop 批 A：首 token 7 到账
    eco2 = out2[0].outputs[0]
    assert eco2.new_token_ids == [7]
    assert req.output_token_ids == [7]
    # 不变式：computed − ph == 真实已算（此刻 = 前 2 个 prompt 位，t7 位仍在飞）
    assert req.num_computed_tokens - req.num_output_placeholders == 2
    # worker 影子：t1 从未落 CPU（token_ids_cpu 行是 -1）
    row = chain.runner.input_batch.token_ids_cpu[0]
    assert row[2] == -1
    # 批 B 的输入由 GPU 回填：input_ids.gpu 首位 == 7（真 token 全程没经过 CPU）
    assert chain.runner.input_ids.gpu[0].item() == 7

    # 拍 3：early-stop 剪枝（computed 3+2−ph 1=4 ≥ 2+2=4 → 不排多余一步）；
    # 上半段 total=0（不采样），pop 批 B：t2=9 到账 → 长度封顶完成
    chain.release()
    out3, ex3 = chain.step()
    assert ex3 is False  # 本拍没调度 token
    eco3 = out3[0].outputs[0]
    assert eco3.new_token_ids == [9]
    assert req.output_token_ids == [7, 9]
    assert req.is_finished()
    assert req.get_finished_reason().name == "LENGTH"
    # 不变式收束：pop 批 B 交货 t9 后 ph 1→0；computed(3) − ph(0) == 3
    # = 有确认 KV 的位置数（t0,t1 两 prompt 位 + t7 的 decode 位——t9 是
    # 位置 2 的采样输出，max_tokens 已到、位置 3 不再前向）
    assert req.num_computed_tokens - req.num_output_placeholders == 3

    # 拍 4+：请求已终、队列还挂着在飞空批 → has_work 保活，收完才归 False
    chain.release()
    while engine.has_work():
        chain.step()
    assert len(engine.batch_queue) == 0
    assert engine.has_work() is False


def test_deferred_sampling_reschedules_after_update():
    # m14：structured + 占位在途 → 上半段不采样；下半段 pop+update_from_output
    # 之后补 bitmask+sample_tokens，批重新 appendleft 入队（复用同一 exec_future）
    chain = RealChain(max_tokens=8, structured=True)
    req = chain.req
    engine = chain.engine
    chain.script([7, 8, 9, 10])
    order = []

    _upd = engine.scheduler.update_from_output
    engine.scheduler.update_from_output = lambda so, mo: (
        order.append("update"),
        _upd(so, mo),
    )[1]
    _spl = engine.model_executor.sample_tokens
    engine.model_executor.sample_tokens = lambda g, non_block=False: (
        order.append("sample"),
        _spl(g, non_block),
    )[1]

    # 拍 1：prefill，ph=0 → 不 pending → 立即采样
    out1, _ = chain.step()
    assert order == ["sample"]
    order.clear()

    # 拍 2：decode，ph=1>0 → pending 置位 → 上半段不采样（deferred）；
    # 下半段 pop 批 A、update_from_output 之后才补采 → 顺序必须是 update→sample
    chain.release()
    out2, _ = chain.step()
    assert order == ["update", "sample"]
    # deferred 批重新入队待后续轮次 pop
    assert len(engine.batch_queue) == 1
    # 采出的 token 走到了 deferred 批的 future 里（下一拍 pop 到账）
    chain.release()
    out3, _ = chain.step()
    ids = [t for o in out3.values() for e in o.outputs for t in e.new_token_ids]
    assert 8 in ids  # 拍 2 deferred 采出的 token 最终照常到账
