"""m1 —— register_custom_op 注册流程的逐轮 trace。

对照 third_party/ascend/language/cann/extension/custom_op.py:L324-345：
装饰的必须是**类**；未设 name 用类名兜底；名字不得与注册表已有项撞车；
core/pipe/mode 三字段必须存在且分别是 CORE/PIPE/MODE 枚举实例；
最后 inspect.signature(op) 抽 __init__ 形参存到 op.signature，把类写进
_custom_op_registry[op.name]。

每轮记录：注册表条目数(前→后)、断言判定、抽到的 signature 形参。
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import build_env, dump  # noqa: E402


def main():
    env, cleanup = build_env(simt_enabled=False)
    try:
        C = env.ext_core
        co = env.custom_op
        reg = co._custom_op_registry
        rounds = []

        def attempt(label, make, note):
            before = len(reg)
            before_names = sorted(reg)
            try:
                obj = make()
                ok = True
                err = ""
                name = getattr(obj, "name", None)
                params = list(getattr(obj, "signature").parameters) if hasattr(obj, "signature") else []
            except AssertionError as e:
                ok = False
                err = str(e)
                name = None
                params = []
            after = len(reg)
            rounds.append({
                "round": len(rounds) + 1,
                "label": label,
                "note": note,
                "registry_size_before": before,
                "registry_size_after": after,
                "registry_names_before": before_names,
                "registry_names_after": sorted(reg),
                "accepted": ok,
                "assert_message": err,
                "op_name": name,
                "signature_params": params,
                "signature_param_count": len(params),
            })

        # 轮 1：import 期已经发生过的一次注册(builtin_custom_ops.py 的
        # `@register_custom_op class _index_select`)——这就是注册表里唯一的初始条目。
        rounds.append({
            "round": 1,
            "label": "import 期：@register_custom_op class _index_select",
            "note": "builtin_custom_ops.py 顶层装饰器在 import 时就跑完了注册",
            "registry_size_before": 0,
            "registry_size_after": len(reg),
            "registry_names_before": [],
            "registry_names_after": sorted(reg),
            "accepted": True,
            "assert_message": "",
            "op_name": env.builtin_custom_ops._index_select.name,
            "signature_params": list(env.builtin_custom_ops._index_select.signature.parameters),
            "signature_param_count": len(env.builtin_custom_ops._index_select.signature.parameters),
        })

        # 轮 2：合法的自定义算子类——四要素齐全，顺利入表。
        def r2():
            @co.register_custom_op
            class Scale:
                name = "scale"
                core = C.CORE.VECTOR
                pipe = C.PIPE.PIPE_V
                mode = C.MODE.SIMD
                symbol = "scale_sym"

                def __init__(self, x, alpha, out=None):
                    pass

            return Scale

        attempt("注册合法类 Scale(name/core/pipe/mode 齐)", r2, "四要素齐全 → 入表")

        # 轮 3：不设 name 字段 —— 用类名兜底。
        def r3():
            @co.register_custom_op
            class Relu6:
                core = C.CORE.VECTOR
                pipe = C.PIPE.PIPE_V
                mode = C.MODE.SIMT

                def __init__(self, x, out=None):
                    pass

            return Relu6

        attempt("注册未设 name 的类 Relu6", r3, "if not hasattr(op,'name') → setattr(op,'name',op.__name__)")

        # 轮 4：重名 —— 注册表唯一性断言拦下。
        def r4():
            @co.register_custom_op
            class Scale2:
                name = "scale"
                core = C.CORE.CUBE
                pipe = C.PIPE.PIPE_M
                mode = C.MODE.SIMD

                def __init__(self, x, out=None):
                    pass

            return Scale2

        attempt("重名注册(name='scale' 已被占)", r4, "assert op.name not in _custom_op_registry")

        # 轮 5：缺 mode 字段。
        def r5():
            @co.register_custom_op
            class NoMode:
                name = "no_mode"
                core = C.CORE.VECTOR
                pipe = C.PIPE.PIPE_V

                def __init__(self, x, out=None):
                    pass

            return NoMode

        attempt("缺 mode 字段", r5, "assert hasattr(op,'mode')")

        # 轮 6：mode 字段类型不对(字符串而非 MODE 枚举)。
        def r6():
            @co.register_custom_op
            class BadMode:
                name = "bad_mode"
                core = C.CORE.VECTOR
                pipe = C.PIPE.PIPE_V
                mode = "SIMT"

                def __init__(self, x, out=None):
                    pass

            return BadMode

        attempt("mode 传裸字符串 'SIMT'", r6, "assert isinstance(op.mode, core.MODE)")

        # 轮 7：装饰的是函数不是类。
        def r7():
            @co.register_custom_op
            def not_a_class(x):
                pass

            return not_a_class

        attempt("装饰函数而非类", r7, "assert inspect.isclass(op) —— 类装饰器，不是函数装饰器")

        trace = {
            "mechanism": "m1",
            "source": "third_party/ascend/language/cann/extension/custom_op.py:L324-345",
            "assert_count_in_pin": 8,
            "assert_list_in_pin": [
                "inspect.isclass(op)",
                "op.name not in _custom_op_registry",
                "hasattr(op,'core')", "hasattr(op,'pipe')", "hasattr(op,'mode')",
                "isinstance(op.core, CORE)", "isinstance(op.pipe, PIPE)", "isinstance(op.mode, MODE)",
            ],
            "final_registry": sorted(reg),
            "final_registry_size": len(reg),
            "rounds": rounds,
        }
        dump(trace, "m1.json")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
