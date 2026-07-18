"""ch04 explainer 驱动脚本 —— m2「visit_Call 第四岔」worked example 取真实 trace。

复用 tests/conftest.py 的 env 夹具（FakeBuilder/FakeAscendBuilder 站在 host 无法
拥有的 C++ 绑定 ir.builder/ascendnpu_ir_builder 位置上，只记录『调用被路由到哪个
对象』——本章要看的正是路由，不是 MLIR 语义）。

我们把三种被调对象喂给真实、逐字未改的 CodeGenerator.visit_Call：
  ① al.sub_vec_id  —— @al.builtin 装饰（第三方扩展，双标记）
  ② tl_load        —— @tl.builtin 装饰（基座，单标记 __triton_builtin__）
  ③ plain_python   —— 无任何标记的普通可调用对象
逐调用记录:两个标记的 getattr 结果、入口门 language.core.is_builtin、选路谓词
extension.is_builtin、最终选中的 _builder、以及该 builder 是否真的收到了 emit 调用
+ 插入点/loc 是否被搬运（m6）。原始输出写 dispatch_trace.json，trace_source=run。
"""
import json
import sys
import types
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"
sys.path.insert(0, str(TESTS_DIR))

import conftest  # noqa: E402


def _identity_visit(gen):
    def visit(node):
        if isinstance(node, types.SimpleNamespace) and hasattr(node, "arg"):
            return (node.arg, node.value)
        return node
    gen.visit = visit


def _call_node(fn, args=(), kwargs=None):
    kwargs = kwargs or {}
    return types.SimpleNamespace(
        func=fn,
        args=list(args),
        keywords=[types.SimpleNamespace(arg=k, value=v) for k, v in kwargs.items()],
    )


def main():
    # 手动驱动 pytest 夹具（它是个 generator function，取 __wrapped__ 拿原函数）。
    fixture_gen = conftest.env.__wrapped__()
    mods = next(fixture_gen)
    try:
        gen = conftest.make_generator(mods, options=types.SimpleNamespace(arch="ascend910b"))
        _identity_visit(gen)

        TRITON_BUILTIN = "__triton_builtin__"
        ASCEND_BUILTIN = "__ascend_builtin__"

        # ① al.* 内建：用扩展侧 @builtin 装饰（third_party/.../extension/core.py）。
        al_sub_vec_id = mods.ext_core.sub_vec_id  # 已在源码里 @builtin 装饰

        # ② tl.* 内建：用基座 @builtin 装饰（triton/language/core.py）。
        @mods.core.builtin
        def tl_load(_builder=None):
            # 模拟一个基座算子在主 builder 上 emit：真实里是 create_load 等；
            # 这里调 create_module 只为在 main builder.calls 留痕（FakeBuilder 无 create_load）。
            _builder.get_insertion_point()  # 触碰一次主 builder，证明确实拿到它
            return "loaded"

        # ③ 普通 Python 可调用对象，无任何 builtin 标记。
        def plain_python(x):
            return x * 2

        cases = [
            ("al.sub_vec_id", al_sub_vec_id, (), {}),
            ("tl_load", tl_load, (), {}),
            ("plain_python", plain_python, (7,), {}),
        ]

        records = []
        for label, fn, args, kws in cases:
            # 调用前把两个 builder 的调用流清空，便于隔离本次调用的落点。
            gen.builder.calls.clear()
            gen.ascend_builder.calls.clear()
            # 给主 builder 一个可辨识的插入点，观察它是否被搬到所选 builder（m6）。
            gen.builder._ip = f"main-ip-before-{label}"

            has_triton = getattr(fn, TRITON_BUILTIN, False)
            has_ascend = getattr(fn, ASCEND_BUILTIN, False)
            entry_gate = mods.core.is_builtin(fn)            # 入口门读 __triton_builtin__
            selects_ascend = mods.extension.is_builtin(fn)   # 选路读 __ascend_builtin__

            node = _call_node(fn, args=args, kws=kws) if False else _call_node(fn, args=args, kwargs=kws)
            result = gen.visit_Call(node)

            # 判定实际落点:哪个 builder 收到了 emit/触碰调用。
            ascend_touched = len(gen.ascend_builder.calls) > 0
            main_touched = any(c[0] != "get_loc" for c in gen.builder.calls) or len(gen.builder.calls) > 0
            if ascend_touched:
                routed_to = "ascend_builder"
            elif has_triton:
                routed_to = "builder"
            else:
                routed_to = "(fell through — no builder)"

            # m6:所选 builder 是否被搬入主 builder 的插入点。
            selected_calls = gen.ascend_builder.calls if selects_ascend else gen.builder.calls
            ip_synced_to_selected = any(
                c[0] == "restore_insertion_point" and c[1] == f"main-ip-before-{label}"
                for c in selected_calls
            )

            records.append({
                "case": label,
                "has___triton_builtin__": bool(has_triton),
                "has___ascend_builtin__": bool(has_ascend),
                "entry_gate_language_core_is_builtin": bool(entry_gate),
                "selector_extension_is_builtin": bool(selects_ascend),
                "routed_to": routed_to,
                "result": result,
                "ascend_builder_calls": [c[0] for c in gen.ascend_builder.calls],
                "main_builder_calls": [c[0] for c in gen.builder.calls],
                "ip_synced_from_main_to_selected_builder": bool(ip_synced_to_selected),
            })

        out = {
            "trace_source": "run",
            "note": "FakeBuilder/FakeAscendBuilder 站位真实 C++ 绑定;记录路由落点+插入点搬运,不模拟 MLIR 语义(需真机)",
            "generator_options": {"arch": "ascend910b"},
            "builder_identity": {
                "self.builder": type(gen.builder).__name__,
                "self.ascend_builder": type(gen.ascend_builder).__name__,
                "same_context": gen.builder.context == gen.ascend_builder.context,
                "context_value": gen.builder.context,
                "is_second_distinct_builder": gen.builder is not gen.ascend_builder,
            },
            "cases": records,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        out_path = Path(__file__).resolve().parent / "dispatch_trace.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        try:
            next(fixture_gen)
        except StopIteration:
            pass


if __name__ == "__main__":
    main()
