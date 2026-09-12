# ch30 m11/m12/m13 驱动脚本：异步编译门三幕。
#   m11 门控时间线：受控慢编译（Event 压住工作线程）下，逐拍 schedule() 观察
#       「plain 请求照常入批 / 语法请求留侧队」，就绪后当拍晋升入批。
#   m12 三态探测：受控 Future 验证 _check_grammar_completion 的
#       result(timeout=0.0001) 三态（pending→None / 成品→原地替换 / Exception→存异常）。
#   m13 失败隔离：坏 structural_tag 从引擎侧注入（前端本会拦），异常经 Future
#       传回 → 晋升检查记账 grammar_compile_error_reqs → 同拍 update_from_output
#       尾部 FINISHED_ERROR 只杀单请求、good 旁人无感、回执空 token。
# 环境：host Miniconda CPython 3.11.11 + xgrammar 0.2.6 + gpt2。
import json
import pathlib
import sys
import threading
import time
from concurrent.futures import Future

IMPL = pathlib.Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

from transformers import AutoTokenizer

from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         StructuredOutputsConfig, VllmConfig)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.request import StructuredOutputRequest

TOKENIZER_NAME = "gpt2"
VOCAB = 50257
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def make_vllm_config():
    return VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER_NAME, vocab_size=VOCAB),
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        scheduler_config=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=8192),
        structured_outputs_config=StructuredOutputsConfig(backend="auto"),
        speculative_config=None,
    )


def make_request(req_id, so=None, frontend=True):
    sp = SamplingParams(structured_outputs=so)
    if so is not None:
        if frontend:
            cfg = make_vllm_config()
            sp._validate_structured_outputs(
                cfg.model_config, cfg.structured_outputs_config, tok)
        else:
            so._backend = "xgrammar"  # 引擎侧注入：绕过前端（模拟前端漏网）
    return Request(request_id=req_id, prompt_token_ids=[1, 2, 3],
                   sampling_params=sp, pooling_params=None)


class ModelRunnerOutputStub:
    def __init__(self, req_id_to_index, sampled_token_ids):
        self.req_id_to_index = req_id_to_index
        self.sampled_token_ids = sampled_token_ids


out = {"env": {"tokenizer": TOKENIZER_NAME, "vocab_size": VOCAB}}

# ══ m12：三态探测（受控 Future，确定性）════════════════════════
three = []
req = StructuredOutputRequest(params=StructuredOutputsParams(regex="a"))
f = Future()
req.grammar = f
three.append({
    "state": "Future 未完成（编译线程在编）",
    "grammar_property": str(req.grammar),
    "is_grammar_ready": req.is_grammar_ready,
    "timeout_used_s": 0.0001,  # request.py:L53 原文 result(timeout=0.0001)
    "note": "超时抛 TimeoutError → return False（忙等变体，每次成本钳在百微秒级）",
})
f2 = Future()
f2.set_result("PRODUCT")
req2 = StructuredOutputRequest(params=StructuredOutputsParams(regex="a"))
req2.grammar = f2
three.append({
    "state": "Future 恰好完成",
    "grammar_property": str(req2.grammar),
    "is_grammar_ready": req2.is_grammar_ready,
    "replaced_in_place": not isinstance(req2._grammar, Future),
    "note": "result() 直接拿到成品，_grammar 原地替换——就绪单调（不回退）",
})
f3 = Future()
f3.set_exception(ValueError("bad schema"))
req3 = StructuredOutputRequest(params=StructuredOutputsParams(regex="a"))
req3.grammar = f3
three.append({
    "state": "Future 装着异常（编译失败）",
    "grammar_property_type": type(req3.grammar).__name__,
    "grammar_property_str": str(req3.grammar),
    "is_grammar_ready": req3.is_grammar_ready,
    "note": "except Exception as e: self._grammar = e（request.py:L57-L58）——异常不抛穿，存进 _grammar",
})
out["three_state_polling"] = three

# ══ m11：门控时间线（受控慢编译）══════════════════════════════
manager = StructuredOutputManager(make_vllm_config())
sched = Scheduler(make_vllm_config(), kv_cache_config=None,
                  structured_output_manager=manager, block_size=16)

gr = make_request("gr-1", so=StructuredOutputsParams(grammar='root ::= "yes" | "no"'))
plain = make_request("plain-1")

hold = threading.Event()   # 压住工作线程：编译开始但不放行
started = threading.Event()
real_create = manager._create_grammar


def slow_create(request):
    started.set()
    hold.wait(timeout=30)
    return real_create(request)


manager._create_grammar = slow_create
manager.grammar_init(gr)          # IO 线程角色：提交线程池
sched.add_request(gr)             # 站 5：阻塞态 → skipped_waiting 侧队
sched.add_request(plain)
started.wait(timeout=10)          # 确认工作线程已进入编译

timeline = []
# 拍 1：编译被压住——plain 进批，gr-1 留侧队
out1 = sched.schedule()
timeline.append({
    "step": 1, "phase": "编译中（工作线程被压住）",
    "scheduled": sorted(out1.num_scheduled_tokens.keys()),
    "gr1_status": gr.status.name,
    "gr1_in_skipped_waiting": gr in sched.skipped_waiting,
    "plain1_status": plain.status.name,
    "note": "schedule() WAITING 相位窥队头→晋级检查 grammar=None→False→prepend 回侧队",
})
hold.set()  # 放行编译
deadline = time.monotonic() + 10
while gr.structured_output_request.grammar is None and time.monotonic() < deadline:
    time.sleep(0.005)
# 拍 2：就绪 → 晋升 → 当拍入批
out2 = sched.schedule()
timeline.append({
    "step": 2, "phase": "编译完成（Future→成品已替换）",
    "scheduled": sorted(out2.num_scheduled_tokens.keys()),
    "gr1_status": gr.status.name,
    "gr1_in_skipped_waiting": gr in sched.skipped_waiting,
    "has_structured_output_requests": out2.has_structured_output_requests,
    "grammar_type": type(gr.structured_output_request.grammar).__name__,
    "note": "晋级 status=WAITING→当拍 WAITING 相位收进批→RUNNING",
})
out["gate_timeline"] = timeline

# ══ m13：编译失败只杀单请求 ═══════════════════════════════════
manager2 = StructuredOutputManager(make_vllm_config())
sched2 = Scheduler(make_vllm_config(), kv_cache_config=None,
                   structured_output_manager=manager2, block_size=16)
bad = make_request("bad-1", so=StructuredOutputsParams(structural_tag="{not json"),
                   frontend=False)   # 前端本会拦的坏规格，从引擎侧注入
good = make_request("good-1", so=StructuredOutputsParams(regex="[0-9]+"))
manager2.grammar_init(bad)
manager2.grammar_init(good)
sched2.add_request(bad)
sched2.add_request(good)

deadline = time.monotonic() + 10
while not isinstance(bad.structured_output_request.grammar, Exception) \
        and time.monotonic() < deadline:
    time.sleep(0.005)

fail_rows = []
fail_rows.append({
    "phase": "编译线程抛异常",
    "detail": "compile_structural_tag('{not json') 在 xgrammar 库内解析失败 → raise → Future 装异常",
    "grammar_state": type(bad.structured_output_request.grammar).__name__,
    "grammar_str": str(bad.structured_output_request.grammar)[:60],
})
promoted = sched2._try_promote_blocked_waiting_request(bad)
fail_rows.append({
    "phase": "晋升检查（_try_promote_blocked_waiting_request）",
    "detail": "isinstance(grammar, Exception) → 记账 grammar_compile_error_reqs，return False",
    "promote_returns": promoted,
    "bad1_in_error_reqs": "bad-1" in sched2.grammar_compile_error_reqs,
    "bad1_status": bad.status.name,
})
outputs = sched2.update_from_output(
    SchedulerOutput(num_scheduled_tokens={}),
    ModelRunnerOutputStub({}, []))
flat = [o for lst in outputs.values() for o in lst]
fail_rows.append({
    "phase": "同拍 update_from_output 尾部收账",
    "detail": "finish_requests(FINISHED_ERROR) 只杀 bad-1；good 旁人无感",
    "bad1_status": bad.status.name,
    "bad1_finished_reason": str(bad.get_finished_reason()),
    "good1_status": good.status.name,
    "good1_alive_in_requests": "good-1" in sched2.requests,
    "error_reqs_cleared": len(sched2.grammar_compile_error_reqs) == 0,
    "receipt_request_id": flat[0].request_id if flat else None,
    "receipt_new_token_ids": flat[0].new_token_ids if flat else None,
})
out["failure_isolation"] = fail_rows

# ══ 表格行建议 ═════════════════════════════════════════════
out["table_rows_m11"] = [
    ["拍 1", "schedule() 窥侧队队头 gr-1", "grammar 探测（100µs 超时）→ None",
     "编译中：未就绪", "plain-1 进批；gr-1 prepend 回侧队，状态仍 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"],
    ["拍 2", "schedule() 再窥 gr-1", "grammar → XgrammarGrammar（Future 已原地替换）",
     "就绪：晋级", "gr-1 status=WAITING→当拍入批→RUNNING；has_structured_output_requests=True"],
]
out["table_rows_m12"] = [
    ["探测 1", "_grammar=Future（未完成）", "result(timeout=0.0001) 抛 TimeoutError",
     "未就绪", "grammar property 返回 None；is_grammar_ready=False"],
    ["探测 2", "_grammar=Future（恰好完成）", "result() 拿到成品",
     "就绪", "_grammar 原地替换为成品；is_grammar_ready=True；重复读幂等"],
    ["探测 3", "_grammar=Future（装异常）", "except Exception as e: self._grammar = e",
     "失败态", "grammar 返回 ValueError；is_grammar_ready=True（探测完成≠成功）"],
]
out["table_rows_m13"] = [
    ["编译线程", "_create_grammar → compile_grammar(STRUCTURAL_TAG, '{not json')",
     "xgrammar 库内解析失败抛 RuntimeError", "失败", "raise → Future 装异常（__init__.py:L177-L192）"],
    ["调度线程·晋升检查", "_try_promote_blocked_waiting_request(bad-1)",
     "isinstance(grammar, Exception)=True", "记账", "return False；grammar_compile_error_reqs+={'bad-1'}；不晋升"],
    ["同拍·收账", "update_from_output 尾部 finish_requests(FINISHED_ERROR)",
     "error_req_ids={'bad-1'}", "只杀单请求",
     "bad-1=FINISHED_ERROR 出列；good-1 无感存活；回执 new_token_ids=[]"],
]

p = pathlib.Path(__file__).with_name("trace_m11_gate.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("WROTE", p)
print(json.dumps(out["three_state_polling"], ensure_ascii=False, indent=1))
print(json.dumps(out["gate_timeline"], ensure_ascii=False, indent=1))
print(json.dumps(out["failure_isolation"], ensure_ascii=False, indent=1))
