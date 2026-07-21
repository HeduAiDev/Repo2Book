#!/usr/bin/env python3
"""M7 驱动脚本：真跑 pin 里的核间同步参数契约（新代 create_sync_block + 旧代 aux_ops）。

取证方式（宿主无昇腾 NPU / 未编译 ascend_ir pybind）：
  * 被测函数体 **逐字取自 pin**——用 ast 按行号切出源码后 exec，不改一个字符：
      - third_party/ascend/language/cann/extension/core.py: builtin / create_sync_block
        / sync_block_set / sync_block_wait / sync_block_all（新代）
      - third_party/ascend/language/cann/extension/aux_ops.py: sync_block_set /
        sync_block_wait（旧代，带 DeprecationWarning）
      - third_party/ascend/language/cann/extension/_utils.py: custom_op
  * 只替换它们的外部依赖：`PIPE` 枚举（成员名按 core.py:L111-L119 从源码解析，
    值用占位整数——真值来自未编译的 ascend_ir pybind）、`semantic`（记录型替身）、
    `_constexpr_to_value`（复刻 triton 语义：constexpr → .value）、builder（记录型替身）。
  * 因此表里的「通过/报错、报什么错、默认 pipe 配成什么、旧代把哪些实参丢了」
    都是 pin 真实控制流的产物；不涉及任何真机数值。

用法：python3 run_m7_sync_contract.py   → 打印并写出 m7_sync_contract.json
"""
import ast
import enum
import json
import os
import warnings
from pathlib import Path

PIN = "2badfc89e70a9b7a5e88463a116c2feddce4b101"
_here = Path(__file__).resolve()
_cand = _here.parents[4] / "source" if len(_here.parents) > 4 else None
SRC = Path(os.environ.get(
    "R2B_SRC",
    str(_cand if (_cand and _cand.exists())
        else "/mnt/e/Laboratory/Repo2Book/instances/triton-ascend/source")))
EXT = SRC / "third_party/ascend/language/cann/extension"


# ----------------------------------------------------------------- 逐字切源码
def grab(path, names):
    """按 ast 行号从 pin 文件里切出这些顶层 def/class 的源码：从 `def`/`class` 行起逐字
    照抄，函数体一字未动；装饰器行（@builtin / @_tensor_member_fn）不在切片内——
    @builtin 只做「_builder 是否传了」的存在性检查（core.py:L70-L86），与本表的契约无关。"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out, spans = {}, {}
    for n in ast.parse(text).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            start = n.lineno  # def 行（装饰器行不含在内）
            src = "\n".join(lines[start - 1:n.end_lineno])
            out[n.name] = src
            spans[n.name] = f"L{start}-L{n.end_lineno}"
    missing = set(names) - set(out)
    assert not missing, f"未在 {path} 找到: {missing}"
    return out, spans


CORE_SRC, CORE_SPAN = grab(EXT / "core.py",
                           {"create_sync_block", "sync_block_set", "sync_block_wait",
                            "sync_block_all", "builtin", "PIPE"})
AUX_SRC, AUX_SPAN = grab(EXT / "aux_ops.py", {"sync_block_set", "sync_block_wait"})
UTIL_SRC, UTIL_SPAN = grab(EXT / "_utils.py", {"custom_op"})

# PIPE 成员名：从 pin 的 class PIPE 源码里解析（core.py:L111-L119）
PIPE_NAMES = [t.targets[0].id for t in ast.parse(CORE_SRC["PIPE"]).body[0].body
              if isinstance(t, ast.Assign)]
PIPE = enum.Enum("PIPE", {n: i for i, n in enumerate(PIPE_NAMES)})


# ----------------------------------------------------------------- 替身依赖
CALLS = []


class constexpr:  # 极简替身，够 _constexpr_to_value 用
    def __init__(self, value):
        self.value = value


def _constexpr_to_value(v):
    return v.value if isinstance(v, constexpr) else v


class MockSemantic:
    def _rec(self, fn, sender, receiver, event_id, sp, rp):
        CALLS.append({"layer": "semantic(new)", "fn": fn, "sender": sender,
                      "receiver": receiver,
                      "event_id": (f"constexpr({event_id.value})"
                                   if isinstance(event_id, constexpr) else event_id),
                      "sender_pipe": sp.name, "receiver_pipe": rp.name})
        return f"{fn}({sender}->{receiver}, id={event_id}, {sp.name}/{rp.name})"

    def create_sync_block_set(self, sender, receiver, event_id, sp, rp, _builder):
        return self._rec("create_sync_block_set", sender, receiver, event_id, sp, rp)

    def create_sync_block_wait(self, sender, receiver, event_id, sp, rp, _builder):
        return self._rec("create_sync_block_wait", sender, receiver, event_id, sp, rp)


class MockBuilder:
    def create_custom_op_for_inter_core_sync(self, op_name, mode_or_sender, id):
        CALLS.append({"layer": "builder(old)", "fn": "create_custom_op_for_inter_core_sync",
                      "op_name": op_name, "mode_or_sender": mode_or_sender, "id": id})
        return None

    def sync_block_all(self, mode, event_id):
        CALLS.append({"layer": "builder(new)", "fn": "sync_block_all",
                      "mode": mode, "event_id": event_id})


def _load():
    """exec 逐字源码，装配两代实现。"""
    from functools import wraps
    from typing import TypeVar
    g = {"enum": enum, "wraps": wraps, "T": TypeVar("T"),
         "PIPE": PIPE, "semantic": MockSemantic(),
         "_constexpr_to_value": _constexpr_to_value,
         "TRITON_BUILTIN": "__triton_builtin__", "ASCEND_BUILTIN": "__ascend_builtin__"}
    exec(compile(CORE_SRC["builtin"], "core.py", "exec"), g)
    for n in ("create_sync_block", "sync_block_set", "sync_block_wait", "sync_block_all"):
        exec(compile(CORE_SRC[n], "core.py", "exec"), g)
    new = {"create_sync_block": g["create_sync_block"],
           "sync_block_set": g["sync_block_set"],
           "sync_block_wait": g["sync_block_wait"],
           "sync_block_all": g["sync_block_all"]}

    _ir = type("ir", (), {"builder": object})
    gu = {"ir": _ir, "tl": type("tl", (), {})}
    exec(compile(UTIL_SRC["custom_op"], "_utils.py", "exec"), gu)
    ga = {"ir": _ir, "tensor": object, "builtin": g["builtin"], "_constexpr_to_value": _constexpr_to_value,
          "custom_op": gu["custom_op"], "core": type("m", (), {"constexpr": constexpr})}
    for n in ("sync_block_set", "sync_block_wait"):
        exec(compile(AUX_SRC[n], "aux_ops.py", "exec"), ga)
    old = {"sync_block_set": ga["sync_block_set"], "sync_block_wait": ga["sync_block_wait"]}
    return new, old


def run_case(label, gen, fn, kwargs):
    before = len(CALLS)
    def _j(v):
        if isinstance(v, PIPE):
            return v.name
        if isinstance(v, constexpr):
            return f"constexpr({v.value})"
        return v

    rec = {"case": label, "generation": gen, "call": fn.__name__,
           "args": {k: _j(v) for k, v in kwargs.items() if k != "_builder"}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn(**kwargs)
            rec["outcome"] = "OK"
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 —— 就是要抓契约违例
            rec["outcome"] = type(e).__name__
            rec["error"] = str(e)
        rec["warnings"] = [f"{x.category.__name__}: {x.message}" for x in w]
    rec["emitted"] = CALLS[before:]
    return rec


def main():
    new, old = _load()
    b = MockBuilder()
    cases = []

    # ---- 新代 al.sync_block_set / wait（core.py）
    cases.append(run_case("cube→vector, event_id=0, 不给 pipe", "new",
                          new["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=0, _builder=b)))
    cases.append(run_case("vector→cube, event_id=15, 不给 pipe", "new",
                          new["sync_block_wait"],
                          dict(sender="vector", receiver="cube", event_id=15, _builder=b)))
    cases.append(run_case("cube→cube（同核）", "new", new["sync_block_set"],
                          dict(sender="cube", receiver="cube", event_id=1, _builder=b)))
    cases.append(run_case("cube→vector, event_id=16（越界）", "new", new["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=16, _builder=b)))
    cases.append(run_case("cube→vector, event_id=-1（负）", "new", new["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=-1, _builder=b)))
    cases.append(run_case("sender='aicore'（非法核名）", "new", new["sync_block_set"],
                          dict(sender="aicore", receiver="vector", event_id=2, _builder=b)))
    cases.append(run_case("只给 sender_pipe，receiver_pipe 留空", "new", new["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=3,
                               sender_pipe=PIPE.PIPE_V, _builder=b)))
    cases.append(run_case("两侧 pipe 都显式给（PIPE_V / PIPE_MTE1）", "new",
                          new["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=4,
                               sender_pipe=PIPE.PIPE_V, receiver_pipe=PIPE.PIPE_MTE1,
                               _builder=b)))
    cases.append(run_case("event_id 传 constexpr(5)（不是 int）", "new",
                          new["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=constexpr(5),
                               _builder=b)))
    cases.append(run_case("sync_block_all('all_sub_vector', 7)（新代独有模式）", "new",
                          new["sync_block_all"],
                          dict(mode="all_sub_vector", event_id=7, _builder=b)))

    cases.append(run_case("event_id 传 constexpr(99)（越界但包在 constexpr 里）", "new",
                          new["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=constexpr(99),
                               _builder=b)))

    # ---- 旧代 aux_ops.sync_block_set / wait
    cases.append(run_case("旧代 cube→vector, event_id=3", "old", old["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=3, _builder=b)))
    cases.append(run_case("旧代 vector→cube, event_id=3", "old", old["sync_block_wait"],
                          dict(sender="vector", receiver="cube", event_id=3, _builder=b)))
    cases.append(run_case("旧代 event_id 传 constexpr(99)", "old", old["sync_block_set"],
                          dict(sender="cube", receiver="vector", event_id=constexpr(99),
                               _builder=b)))
    cases.append(run_case("旧代 cube→cube（同核）", "old", old["sync_block_set"],
                          dict(sender="cube", receiver="cube", event_id=3, _builder=b)))

    out = {
        "pin": PIN,
        "mechanism": "M7 (+M6/M8/M10/M11 旁证)",
        "harness": "pin 的 create_sync_block / 两代 sync_block_* / custom_op 逐字 exec；PIPE 成员名从 core.py 解析，semantic 与 builder 为记录型替身",
        "sources": {
            "core.py::create_sync_block": CORE_SPAN["create_sync_block"],
            "core.py::sync_block_set": CORE_SPAN["sync_block_set"],
            "core.py::sync_block_wait": CORE_SPAN["sync_block_wait"],
            "core.py::sync_block_all": CORE_SPAN["sync_block_all"],
            "core.py::class PIPE": CORE_SPAN["PIPE"],
            "aux_ops.py::sync_block_set": AUX_SPAN["sync_block_set"],
            "aux_ops.py::sync_block_wait": AUX_SPAN["sync_block_wait"],
            "_utils.py::custom_op": UTIL_SPAN["custom_op"],
        },
        "pipe_members_python": PIPE_NAMES,
        "pipe_count_python": len(PIPE_NAMES),
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_ok": sum(1 for c in cases if c["outcome"] == "OK"),
            "n_assertion": sum(1 for c in cases if c["outcome"] == "AssertionError"),
            "n_valueerror": sum(1 for c in cases if c["outcome"] == "ValueError"),
            "n_typeerror": sum(1 for c in cases if c["outcome"] == "TypeError"),
            "n_deprecation_warned": sum(1 for c in cases if c["warnings"]),
            "event_id_range": [0, 15],
        },
    }
    Path(__file__).with_name("m7_sync_contract.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in cases:
        print(f"[{c['generation']}] {c['case']:44s} -> {c['outcome']:15s} "
              f"{(c['error'] or '')[:70]}")
        for e in c["emitted"]:
            print("      emit:", e)
        for w in c["warnings"]:
            print("      warn:", w[:90])
    print("SUMMARY", out["summary"])
    print("PIPE(python) =", PIPE_NAMES)


if __name__ == "__main__":
    main()
