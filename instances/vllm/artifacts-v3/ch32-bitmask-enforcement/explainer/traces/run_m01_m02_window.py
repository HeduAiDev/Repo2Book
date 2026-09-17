# ch32 m01/m02/m13 驱动：两段式 execute GPU 窗口。
# 甲段：真 EngineCore.step()（精简版）+ spy executor/scheduler——② 发车即返
# Future（后台线程 50ms 后置 None，模拟 GPU 前向窗口；真实前向时长 host 未测，
# deepread engine-loop 卡口径『一步 forward 只有几十毫秒』）+ ③ 真
# StructuredOutputManager.grammar_bitmask（64 个真 xgrammar matcher 逐行填）——
# 记录事件时间线，验证掩码 CPU 活完整落在前向窗口内。
# 乙段：真 GPUModelRunner 两幕契约——execute_model return None/状态防御/
# sample_tokens 解包即清/先掩码后采样/同一块 logits 张量原地改写（m13）。
import threading
import time
from concurrent.futures import Future

import numpy as np
import torch

from _ch32_common import (IMPL, VOCAB, TOK_YES, OFF_GRAMMAR, FakeRequest,
                          FakeSchedulerOutput, RealBackend, allowed_ids,
                          dump, get_tokenizer, make_vllm_config, new_grammar)
import sys
sys.path.insert(0, str(IMPL))

FORWARD_WINDOW_MS = 50.0  # 模拟前向窗口（spy；真实前向 host 未测）
BATCH = 64

out = {"env": {"forward_window_ms_simulated": FORWARD_WINDOW_MS,
               "note": "spy 后台线程 50ms 后置 future=None 模拟 GPU 前向窗口；"
                       "掩码填充时长为真 xgrammar 实测，前向时长非实测"}}

tok = get_tokenizer()
EBNF = 'root ::= "yes" | "no"'

# ── 甲段：step() 事件时间线（③ 掩码 CPU 活藏进前向窗口）──────────────────────
from vllm.v1.structured_output import StructuredOutputManager

mgr = StructuredOutputManager(make_vllm_config(max_num_seqs=BATCH))
mgr.backend = RealBackend(VOCAB)
requests, grammars = {}, []
for i in range(BATCH):
    g = new_grammar(EBNF)
    grammars.append(g)
    requests[f"r{i}"] = FakeRequest(f"r{i}", grammar=g)


class SpyScheduler:
    """scheduler 消费面替身：schedule → 真 GrammarOutput；get_grammar_bitmask
    内嵌真 manager.grammar_bitmask（真实 CPU 填表负载）。"""
    def __init__(self):
        self.t = {}

    def has_requests(self):
        return True

    def schedule(self, throttle=False):
        self.t["schedule_end"] = time.perf_counter()
        return FakeSchedulerOutput(
            num_scheduled_tokens={rid: 1 for rid in requests},
            has_structured_output_requests=True)

    def get_grammar_bitmask(self, scheduler_output):
        from vllm.v1.core.sched.output import GrammarOutput
        t0 = time.perf_counter()
        bm = mgr.grammar_bitmask(requests, list(requests.keys()), {})
        t1 = time.perf_counter()
        self.t["bitmask_start"] = t0
        self.t["bitmask_fill_ms"] = (t1 - t0) * 1000.0
        self.t["bitmask_end"] = t1
        assert bm.shape[0] == BATCH
        return GrammarOutput(list(requests.keys()), bm)

    def update_from_output(self, scheduler_output, model_output):
        self.t["update_from_output"] = time.perf_counter()
        return {}


class SpyExecutor:
    """executor 消费面替身：non_block 发车即返 Future；后台线程 FORWARD_WINDOW_MS
    后 set_result(None)（execute_model 返回 None=两段式契约第一幕谢幕）。"""

    def __init__(self):
        self.t = {}

    def execute_model(self, scheduler_output, non_block=False):
        assert non_block is True
        self.t["dispatch"] = time.perf_counter()
        fut = Future()

        def forward():
            time.sleep(FORWARD_WINDOW_MS / 1000.0)
            fut.set_result(None)

        threading.Thread(target=forward, daemon=True).start()
        return fut

    def sample_tokens(self, grammar_output, non_block=False):
        self.t["sample_tokens"] = time.perf_counter()
        self.grammar_output = grammar_output
        fut = Future()
        fut.set_result(("model_output", None))
        return fut

    def take_draft_token_ids(self):
        return None


from vllm.v1.engine.core import EngineCore

spy_sched, spy_exec = SpyScheduler(), SpyExecutor()
core = EngineCore.__new__(EngineCore)
core.scheduler = spy_sched
core.model_executor = spy_exec

t_step = time.perf_counter()
outputs, model_executed = core.step()
t_step_end = time.perf_counter()

t0 = spy_exec.t["dispatch"]  # ② 发车 = 时间轴原点
tl = {
    "step_total_ms": round((t_step_end - t_step) * 1000.0, 3),
    "dispatch_at_ms": 0.0,
    "bitmask_start_at_ms": round((spy_sched.t["bitmask_start"] - t0) * 1000.0, 3),
    "bitmask_fill_ms_real": round(spy_sched.t["bitmask_fill_ms"], 3),
    "bitmask_end_at_ms": round((spy_sched.t["bitmask_end"] - t0) * 1000.0, 3),
    "future_result_at_ms": None,  # 填于下
    "sample_tokens_at_ms": round((spy_exec.t["sample_tokens"] - t0) * 1000.0, 3),
    "update_from_output_at_ms": round((spy_sched.t["update_from_output"] - t0) * 1000.0, 3),
    "forward_window_ms_simulated": FORWARD_WINDOW_MS,
}
# future.result() 时刻 = sample_tokens 前一刻（sample_tokens 在 result 返回 None
# 后立刻调用；用 sample_tokens 时刻近似并以下界核验）
tl["future_result_at_ms"] = tl["sample_tokens_at_ms"]
tl["overlap_verified"] = bool(tl["bitmask_end_at_ms"] < FORWARD_WINDOW_MS)
tl["model_executed"] = model_executed
# 掩码行身份抽查：行 0 = 'yes|no' 位置 0 允许集（gpt2 前缀闭包）
tl["bitmask_row0_allowed"] = allowed_ids(spy_exec.grammar_output.grammar_bitmask[0], VOCAB)
tl["bitmask_rows"] = int(spy_exec.grammar_output.grammar_bitmask.shape[0])
out["step_timeline"] = tl

# ── 乙段：GPUModelRunner 两幕契约（真 CUDA + 真 xgr apply）────────────────────
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, ExecuteModelState
from vllm.v1.core.sched.output import GrammarOutput

V = VOCAB
logits_pre = torch.full((1, V), -10.0, device="cuda", dtype=torch.float32)
logits_pre[0, OFF_GRAMMAR] = 5.0
logits_pre[0, TOK_YES] = 2.0


class FakeModel:
    def compute_logits(self, hs):
        return logits_pre


class RecordingSampler:
    def __init__(self):
        self.seen = None
        self.order = []

    def __call__(self, *, logits, sampling_metadata):
        self.order.append("sample")
        self.seen = logits
        return ("sampled", logits)


class FakeInputBatch:
    def __init__(self, req_ids, logits_indices):
        self.req_ids = list(req_ids)
        self.logits_indices = torch.tensor(logits_indices)
        self.sampling_metadata = "MD"

    def update_async_output_token_ids(self):
        pass


runner = GPUModelRunner.__new__(GPUModelRunner)
runner.execute_model_state = None
runner.model = FakeModel()
runner.sampler = RecordingSampler()
runner.input_batch = FakeInputBatch(["r1"], [2])
runner._seam_hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)

import vllm.v1.worker.gpu_model_runner as mr_mod

real_apply = mr_mod.apply_grammar_bitmask
apply_order = []


def spy_apply(scheduler_output, grammar_output, input_batch, logits):
    apply_order.append("apply")
    real_apply(scheduler_output, grammar_output, input_batch, logits)


mr_mod.apply_grammar_bitmask = spy_apply
try:
    so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
    act1 = runner.execute_model(so)
    state = runner.execute_model_state
    # 第一幕：return None + 十元组暂存
    act1_ret = {
        "execute_model_returns": act1,
        "state_packed": state is not None,
        "off_grammar_token_id": OFF_GRAMMAR,
        "yes_token_id": TOK_YES,
        "state_fields": list(ExecuteModelState._fields),
        "state_field_count": len(ExecuteModelState._fields),
        "logits_shape": list(state.logits.shape),
        "logits_is_cuda": bool(state.logits.is_cuda),
        "spec_decode_metadata_is_none": state.spec_decode_metadata is None,
        "hidden_states_shape": list(state.hidden_states.shape),
        "sample_hidden_states_shape": list(state.sample_hidden_states.shape),
        "scheduler_output_is_same_object": state.scheduler_output is so,
        "logit_off_grammar_before": float(state.logits[0, OFF_GRAMMAR].item()),
        "logit_yes_before": float(state.logits[0, TOK_YES].item()),
    }
    # 状态防御：sample_tokens 没来不许再发车
    defense = None
    try:
        runner.execute_model(so)
    except RuntimeError as e:
        defense = str(e)
    # 第二幕：真掩码 + 采样
    g = new_grammar(EBNF)
    bm_row = torch.zeros((1, -(-V // 32)), dtype=torch.int32)
    g.fill_bitmask(bm_row, 0)
    go = GrammarOutput(["r1"], bm_row.numpy())
    sampler_out = runner.sample_tokens(go)
finally:
    mr_mod.apply_grammar_bitmask = real_apply

seen = runner.sampler.seen
act2_ret = {
    "state_error_message": defense,
    "order": apply_order + runner.sampler.order,  # apply 在 sample 之前
    "state_cleared_after_sample": runner.execute_model_state is None,
    "sampler_sees_same_tensor_object": seen is state.logits,  # m13 原地所有权
    "sampler_sees_same_tensor_id": id(seen) == id(state.logits),
    "logit_off_grammar_after_mask": float(seen[0, OFF_GRAMMAR].item()),
    "logit_yes_after_mask": float(seen[0, TOK_YES].item()),
    "sampler_output_is_tuple_with_logits": sampler_out[1] is seen,
    "mask_row0_allowed": allowed_ids(go.grammar_bitmask[0], V),
}
out["two_act_contract"] = {"act1": act1_ret, "act2": act2_ret}
out["env"]["cuda_device"] = torch.cuda.get_device_name(0)

dump("trace_m01_m02_window.json", out)
print("timeline:", tl["bitmask_fill_ms_real"], "ms fill;",
      "overlap:", tl["overlap_verified"])
print("order:", act2_ret["order"], "same tensor:", act2_ret["sampler_sees_same_tensor_object"])
print("defense:", defense)
