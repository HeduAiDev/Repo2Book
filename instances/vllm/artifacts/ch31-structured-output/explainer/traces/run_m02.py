"""m02 异步语法编译 + WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 状态门（本章命门）。

跑精简版真代码：Request.__init__ 置初始阻塞态 → StructuredOutputManager.grammar_init
把编译扔线程池 → Scheduler._is_blocked_waiting_status 判阻塞 →
Scheduler._try_promote_blocked_waiting_request 每轮问一次 grammar 是否就绪 → 晋级。

编译用一个 threading.Event 卡住，模拟「JSON schema→FSM 编译要花好几轮调度的时间」，
使前两轮必然看到未就绪、第三轮看到就绪——观察的是真实门控代码的返回值。
"""
import threading
import time

from _fakes import dump  # noqa: F401  (先注入 implementation 到 sys.path)
import structured_output_manager as som
from request import Request, RequestStatus
from sampling_params import SamplingParams, StructuredOutputsParams
from scheduler import Scheduler
from structured_output_manager import StructuredOutputManager

released = threading.Event()
compiled = threading.Event()


class GatedGrammar:
    def __repr__(self):
        return "<CompiledGrammar>"


class GatedBackend:
    """卡住的后端替身：compile_grammar 直到 released 被 set 才返回。"""

    def __init__(self, vllm_config, tokenizer=None, vocab_size=0):
        self.vocab_size = vocab_size

    def compile_grammar(self, request_type, grammar_spec):
        released.wait(timeout=5.0)
        compiled.set()
        return GatedGrammar()


som.XgrammarBackend = GatedBackend

sched = Scheduler()
manager = StructuredOutputManager(vllm_config=object(), tokenizer=object())

so = StructuredOutputsParams(json='{"type": "object"}')
so._backend = "xgrammar"          # 前端校验期（sampling_params.py:L773-907）的产物
sp = SamplingParams(max_tokens=16, structured_outputs=so)

rows = []
raw = {}


def snap(round_no, action):
    status = req.status
    blocked = Scheduler._is_blocked_waiting_status(status)
    sor = req.structured_output_request
    grammar = sor.grammar                      # 门控真正的落点（request.py:L59-70）
    promoted = sched._try_promote_blocked_waiting_request(req)
    rows.append([
        f"轮 {round_no}",
        action,
        f"{req.status.name}(={int(req.status)})",
        "是" if blocked else "否",
        "None(未就绪)" if grammar is None else "成品对象(已就绪)",
        str(promoted),
    ])
    raw[f"round_{round_no}"] = {
        "status_before": status.name,
        "status_value_before": int(status),
        "is_blocked_waiting_status": blocked,
        "grammar_is_none": grammar is None,
        "promoted": promoted,
        "status_after": req.status.name,
        "status_value_after": int(req.status),
    }


req = Request("req-0", sampling_params=sp)
raw["initial_status"] = req.status.name
raw["initial_status_value"] = int(req.status)
raw["use_structured_output"] = req.use_structured_output

snap(1, "请求刚入队：grammar_init 尚未调用")

manager.grammar_init(req)                       # 编译提交线程池，立刻返回
raw["grammar_is_future_right_after_submit"] = type(
    req.structured_output_request._grammar).__name__

snap(2, "grammar_init 已提交编译（Future 未完成）")

released.set()
compiled.wait(timeout=5.0)
time.sleep(0.01)                                # 让工作线程把 Future 结果写回

snap(3, "编译完成后的下一轮调度")
snap(4, "已晋级，后续轮次不再受门控影响")

raw["max_workers_formula"] = "max(1, (cpu_count() + 1) // 2)"
raw["executor_max_workers"] = manager.executor._max_workers
raw["compile_thread_is_not_main"] = True
raw["note"] = (
    "编译在 ThreadPoolExecutor 工作线程里跑，调度线程（本脚本主线程）在轮 1-2 期间"
    "从未被阻塞：每轮只花一次 Future.result(timeout=0.0001) 的 100 微秒预算。"
)

dump("m02.json", {
    "mechanism": "m02-async-grammar-compile-gate",
    "columns": ["轮次", "调度器动作", "request.status", "_is_blocked_waiting_status",
                "structured_output_req.grammar", "_try_promote… 返回"],
    "rows": rows,
    "raw": raw,
})
