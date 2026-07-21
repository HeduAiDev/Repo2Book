"""m7 —— 自定义算子参数的动态定型：self.arg_type / type-hint / al.int64。

对照：
  third_party/ascend/language/cann/extension/custom_op.py:L133-152(_args_to_operands)
    —— 逐个形参取 `ty = op.arg_type.get(param.name, param.annotation)`，再 _to_value；
  third_party/ascend/language/cann/extension/builtin_custom_ops.py:L99-103
    —— __init__ 里 `self.arg_type['end_offset'] = index.dtype` 这三行；
  third_party/ascend/language/cann/extension/core.py:L93-101
    —— al.int64(x)：int 子类，实例带 `.type = tl.int64`，_to_value 的
       `ty = getattr(value, 'type', ty)` 会优先采信它。

本脚本给 _to_value 挂一层**只读记录**的包装（不改精简版代码），逐形参录下
「ty 从哪来 / 最终 ty 是什么」。两轮只改一件事：index 张量的 dtype（int32 → int64），
看被 arg_type 重定型的三个参数是否跟着变。
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import build_env, dump  # noqa: E402


def main():
    env, cleanup = build_env(simt_enabled=False)
    try:
        tl = env.tl_core
        co = env.custom_op
        C = env.ext_core
        Op = env.builtin_custom_ops._index_select

        def tensor(dtype_name, is_ptr=False, shape=()):
            dt = tl.dtype(dtype_name)
            if is_ptr:
                dt.is_ptr = lambda: True
                dt.element_ty = dt
            t = tl.tensor(handle=f"h-{dtype_name}", type=dt)
            t.shape = list(shape)
            return t

        recorded = []
        orig_to_value = co._to_value

        def recording_to_value(value, builder, ty=None):
            out = orig_to_value(value, builder, ty)
            recorded.append({
                "value": repr(value),
                "requested_ty": None if ty is None else str(ty),
                "handle": str(out),
                "handle_py_type": type(out).__name__,
            })
            return out

        rounds = []
        for idx_dtype in ("int32", "int64"):
            src = tensor("fp32", is_ptr=True)
            index = tensor(idx_dtype, shape=[4])
            out_t = tensor("fp32")
            kwargs = dict(src=src, index=index, dim=0, bound=C.int64(8),
                          end_offset=(4, 4), start_offset=(0, 0), src_stride=(1, 1))

            op = co._init_op(Op, out=out_t, **kwargs)

            per_param = []
            bind = Op.signature.bind(out=out_t, **kwargs)
            for param in Op.signature.parameters.values():
                v = bind.arguments.get(param.name)
                if v is None:
                    per_param.append({"param": param.name, "passed": None,
                                      "ty_source": "跳过（值为 None）", "resolved_ty": None})
                    continue
                if param.name in op.arg_type:
                    src_of_ty, ty = "self.arg_type（__init__ 按 index.dtype 动态定型）", op.arg_type[param.name]
                elif param.annotation is not inspect.Parameter.empty:
                    src_of_ty, ty = "签名 type-hint", param.annotation
                else:
                    src_of_ty, ty = "无（_to_value 回落到值自身的 .type 或 Python 类型默认值）", None
                per_param.append({
                    "param": param.name,
                    "passed": repr(v),
                    "value_dot_type": str(getattr(v, "type", None)),
                    "ty_source": src_of_ty,
                    "resolved_ty": None if ty is None else str(ty),
                })

            recorded.clear()
            b = env.FakeBuilder()
            co._to_value = recording_to_value
            try:
                inputs = co._args_to_operands(op, b, (), dict(kwargs))
            finally:
                co._to_value = orig_to_value

            rounds.append({
                "round": len(rounds) + 1,
                "index_dtype": idx_dtype,
                "arg_type_after_init": {k: v.name for k, v in op.arg_type.items()},
                "extra_attr": op.extra_attr,
                "per_param": per_param,
                "operands_emitted": len(inputs),
                "to_value_records": list(recorded),
            })

        trace = {
            "mechanism": "m7",
            "source": ["third_party/ascend/language/cann/extension/custom_op.py:L133-152",
                       "third_party/ascend/language/cann/extension/builtin_custom_ops.py:L99-103",
                       "third_party/ascend/language/cann/extension/core.py:L93-101"],
            "caveat": ("精简版 _to_value 按 subtraction_plan 删掉了 12 条逐位宽精确分支"
                       "（pin custom_op.py:L78-108 的 get_int64/get_int16/... ），只留 int→get_int32、"
                       "float→get_fp32 两条默认路径。所以本 trace 观测的是**ty 是怎么被解析出来的**"
                       "（arg_type / type-hint / 值自身的 .type），不是最终 handle 的位宽——"
                       "位宽分派那一步在真实源码里才有。"),
            "rounds": rounds,
        }
        dump(trace, "m7.json")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
