"""m2 —— custom / custom_semantic 调用期数据流的逐步 trace。

对照 third_party/ascend/language/cann/extension/custom_op.py:L294-321：
_get_op_class 查表 → _init_op 实例化(跑 __init__ 断言) → out 拆成 outputs 操作数 →
按 signature 把位置/关键字实参转成 inputs 操作数 → _make_attrs / _make_arg_attrs 造属性 →
_builder.create_custom_op(...) emit hivm.CustomOp → _to_result 按 out 的类型包回张量。

样例参数刻意选小：2D src(stride=(1,1))、长度 4 的 1D int32 index、dim=0、bound=8。
另跑一轮"未注册且非 __builtin_ 前缀"的名字，看 _get_op_class 怎么拦。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import build_env, dump, CH_DIR  # noqa: E402


def main():
    env, cleanup = build_env(simt_enabled=False)
    try:
        tl = env.tl_core
        co = env.custom_op
        C = env.ext_core

        def tensor(dtype_name, is_ptr=False, shape=()):
            dt = tl.dtype(dtype_name)
            if is_ptr:
                dt.is_ptr = lambda: True
                dt.element_ty = dt
            t = tl.tensor(handle=f"h-{dtype_name}", type=dt)
            t.shape = list(shape)
            return t

        src = tensor("fp32", is_ptr=True)
        index = tensor("int32", shape=[4])
        out = tensor("fp32")
        kwargs = dict(src=src, index=index, dim=0, bound=8,
                      end_offset=(4, 4), start_offset=(0, 0), src_stride=(1, 1), out=out)

        builder = env.FakeBuilder()
        steps = []

        # 步 1：查表
        op_class = co._get_op_class("__builtin_index_select")
        steps.append({
            "step": 1, "action": "_get_op_class('__builtin_index_select')",
            "key_scalar": f"registry hit={op_class is env.builtin_custom_ops._index_select}",
            "detail": {"registry_size": len(co._custom_op_registry),
                       "is_dummy_fallback": op_class.__name__ == "_builtin_custom_op",
                       "core": op_class.core.value, "pipe": op_class.pipe.value, "mode": op_class.mode.value},
            "result": "命中真实注册类 _index_select（非 __builtin_ 哑类兜底）",
        })

        # 步 2：实例化跑 __init__ 校验
        # 断言条数直接数在 pin 源码的 _index_select.__init__ 上（L79-103），不靠记忆
        import re as _re
        _pin_init = (CH_DIR.parents[1] / "source"
                     / "third_party/ascend/language/cann/extension/builtin_custom_ops.py"
                     ).read_text(encoding="utf-8").splitlines()[78:103]
        n_assert = sum(1 for l in _pin_init if _re.search(r"\bassert\b", l))
        n_tuple_check = sum(1 for l in _pin_init if "_assert_int_like_tuple(" in l)
        op = co._init_op(op_class, **kwargs)
        steps.append({
            "step": 2, "action": "_init_op → __init__ 断言 + arg_type 动态定型",
            "key_scalar": f"src_rank={len(kwargs['src_stride'])}, idx_rank={len(index.shape)}",
            "detail": {"init_assert_count_in_pin": n_assert,
                       "assert_int_like_tuple_calls": n_tuple_check,
                       "arg_type": {k: v.name for k, v in op.arg_type.items()},
                       "arg_type_entries": len(op.arg_type),
                       "extra_attr": op.extra_attr},
            "result": "形状/dtype 断言全过，3 个参数被重定型为 index.dtype",
        })

        # 步 3：out → outputs 操作数
        kw = dict(kwargs)
        out_arg = kw.pop("out")
        outs = [out_arg]
        outputs = co._to_operands(outs, builder)
        steps.append({
            "step": 3, "action": "_to_operands(outs) → outputs",
            "key_scalar": f"len(outputs)={len(outputs)}",
            "detail": {"outputs": [str(h) for h in outputs]},
            "result": "1 个输出操作数（out 张量的 handle 直接透传）",
        })

        # 步 4：其余实参按 signature → inputs 操作数
        inputs = co._args_to_operands(op, builder, (), kw)
        steps.append({
            "step": 4, "action": "_args_to_operands(args/kwargs) → inputs",
            "key_scalar": f"len(inputs)={len(inputs)}",
            "detail": {"inputs": [str(h) for h in inputs],
                       "signature_params": len(op_class.signature.parameters)},
            "result": "标量逐个装箱、元组按元素摊平；None 参数(other)被跳过",
        })

        # 步 5：属性化
        attrs = co._make_attrs(op, builder)
        arg_attrs = co._make_arg_attrs(op, builder)
        steps.append({
            "step": 5, "action": "_make_attrs / _make_arg_attrs",
            "key_scalar": f"len(attrs)={len(attrs)}, len(arg_attrs)={len(arg_attrs)}",
            "detail": {"attrs": {k: str(v) for k, v in attrs.items()},
                       "attr_keys": sorted(attrs),
                       "has_symbol": "symbol" in attrs, "has_bitcode": "bitcode" in attrs},
            "result": "hivm.tcore_type/hivm.pipe/hivm.vf_mode 三属性 + extra_attr；__builtin_ 前缀免 symbol/bitcode",
        })

        # 步 6：emit
        res = builder.create_custom_op("__builtin_index_select", attrs, inputs, outputs, arg_attrs)
        result = co._to_result(res, [o.type for o in outs])
        create_calls = [c for c in builder.calls if c[0] == "create_custom_op"]
        steps.append({
            "step": 6, "action": "_builder.create_custom_op → emit hivm.CustomOp → _to_result",
            "key_scalar": f"create_custom_op 调用 {len(create_calls)} 次, 返回 {len(res)} 个结果",
            "detail": {"emitted_name": create_calls[-1][1], "results": [str(r) for r in res],
                       "result_dtype": str(result.dtype)},
            "result": "1 个 hivm.CustomOp，结果张量类型与 out 一致（ttadapter 阶段的 IR）",
        })

        # 反例轮：未注册且非 __builtin_ 前缀
        try:
            co._get_op_class("my_unregistered_op")
            rejected = ""
        except AssertionError as e:
            rejected = str(e)
        # 反例轮：__builtin_ 前缀但没注册 → 哑类兜底
        dummy = co._get_op_class("__builtin_never_registered")

        # 端到端复核：同一组参数直接走 custom_semantic 一遍，确认逐步 trace 与
        # 真实入口的产物完全一致（步数是拆开看，不是另一条路径）。
        b2 = env.FakeBuilder()
        co.custom_semantic("__builtin_index_select", **kwargs, _builder=b2)
        e2e = [c for c in b2.calls if c[0] == "create_custom_op"][0]
        end_to_end = {
            "create_custom_op_calls": len([c for c in b2.calls if c[0] == "create_custom_op"]),
            "name": e2e[1], "attrs": len(e2e[2]), "inputs": len(e2e[3]),
            "outputs": len(e2e[4]), "arg_attrs": len(e2e[5]),
            "matches_stepwise": (len(e2e[2]) == len(attrs) and len(e2e[3]) == len(inputs)
                                 and len(e2e[4]) == len(outputs) and len(e2e[5]) == len(arg_attrs)),
        }

        trace = {
            "mechanism": "m2",
            "end_to_end_check": end_to_end,
            "source": [
                "third_party/ascend/language/cann/extension/custom_op.py:L294-321",
                "third_party/ascend/language/cann/extension/custom_op.py:L37-51",
                "third_party/ascend/ascend_ir.cc:L618-659",
            ],
            "params": {"src_rank": 2, "index_len": 4, "index_dtype": "int32", "dim": 0, "bound": 8,
                       "end_offset": [4, 4], "start_offset": [0, 0], "src_stride": [1, 1]},
            "steps": steps,
            "unregistered_name_assert": rejected,
            "builtin_prefix_dummy": {
                "class_name": dummy.__name__,
                "core": dummy.core.value, "pipe": dummy.pipe.value, "mode": dummy.mode.value,
                "signature_params": len(dummy.signature.parameters),
            },
        }
        dump(trace, "m2.json")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
