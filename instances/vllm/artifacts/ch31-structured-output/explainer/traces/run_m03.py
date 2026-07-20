"""m03 Future.result(timeout=0.0001) 的『几乎非阻塞』就绪轮询与 Future→成品原地替换。

跑精简版 StructuredOutputRequest._check_grammar_completion 真代码，观察三种输入形态下
的返回值、_grammar 字段类型变化，以及每次轮询实际花掉的墙钟时间（host CPython 3.11，
纯控制流；耗时只反映 Future.result 的超时预算本身，与真实机器/GPU 无关）。
"""
import time
from concurrent.futures import Future

from _fakes import dump
from so_request import StructuredOutputRequest
from sampling_params import StructuredOutputsParams


class Grammar:
    def __repr__(self):
        return "<CompiledGrammar>"


sor = StructuredOutputRequest(params=StructuredOutputsParams(regex="a+"))
fut = Future()
sor.grammar = fut          # setter 接受 Future（request.py:L66-70）

# 预热：CPython 首次进入 Future.result 的等待路径有一次性开销（导入锁/条件变量初始化），
# 与机制无关；先在一个丢弃对象上轮询若干次，避免首轮读数被一次性开销污染。
_warm = StructuredOutputRequest(params=StructuredOutputsParams(regex="a+"))
_warm.grammar = Future()
for _ in range(5):
    _ = _warm.grammar

rows = []
raw = {"timeout_budget_s": 0.0001, "timeout_budget_us": 100}


def poll(round_no, action):
    t0 = time.perf_counter()
    g = sor.grammar          # 内部走 _check_grammar_completion
    dt_us = (time.perf_counter() - t0) * 1e6
    dt_r = round(dt_us)
    kind = type(sor._grammar).__name__
    rows.append([
        f"轮 {round_no}",
        action,
        kind,
        "None" if g is None else "成品对象",
        f"{dt_r} us",
    ])
    raw[f"round_{round_no}"] = {
        "_grammar_type_after": kind,
        "grammar_is_none": g is None,
        "elapsed_us_rounded": dt_r,
    }
    return dt_r


poll(1, "Future 未完成：result(timeout=0.0001) 抛 TimeoutError → 返回 False")
poll(2, "仍未完成：再等一次 100us 预算")

fut.set_result(Grammar())
poll(3, "编译已完成：result 立即返回，_grammar 被**原地替换**成成品")
poll(4, "再读：isinstance(_grammar, Future) 为假，直接返回，纯属性读")

raw["is_grammar_ready_property"] = sor.is_grammar_ready
raw["is_grammar_ready_in_tree_callers_v0_21_0"] = 0
raw["note"] = (
    "轮 1-2 的耗时 ≈ 超时预算 100us；轮 3-4 的耗时≈0（无等待）。"
    "耗时来自 host CPython 的 Future.result 超时实现，不是 GPU/真机指标。"
)

dump("m03.json", {
    "mechanism": "m03-future-poll-100us",
    "columns": ["轮次", "轮询时的状态", "_grammar 字段类型", "grammar property 返回", "本次轮询耗时"],
    "rows": rows,
    "raw": raw,
})
