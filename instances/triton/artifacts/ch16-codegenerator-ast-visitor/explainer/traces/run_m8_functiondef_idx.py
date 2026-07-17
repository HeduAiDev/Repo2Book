#!/usr/bin/env python3
"""ch16 m8 — 忠实重放 visit_FunctionDef 的参数下降循环(code_generator.py:L414-L433)。

真·CodeGenerator 离不开 ir.builder/MLIR 栈,无法 headless 独跑;此脚本逐字复刻
L414-L433 的两条 index 逻辑(i=Python 参数序,idx=真进 IR 的参数位),对一个具体
kernel 签名跑出 idx 轨迹 + set_arg_attr 落点。divisibility 值=16 取自
python/triton/backends/compiler.py:L77(self.property_values["tt.divisibility"] = 16)。
"""
import json

# ---- 具体 kernel(读者能心算) ---------------------------------------------
#   @triton.jit
#   def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
arg_names = ["x_ptr", "y_ptr", "out_ptr", "n_elements", "BLOCK_SIZE"]

# specialization 拆出来的两张表(见 ast_to_ttir L1277-L1291 + AttrsDescriptor):
#   constants: 被标 constexpr 的参数序号 -> 编译期值。BLOCK_SIZE 在序号 4,取 1024。
constants = {4: 1024}
#   attributes: launch 期算出的参数属性。三个指针 16 对齐 -> tt.divisibility=16;
#   n_elements=1000 不被 16 整除 -> 无属性(compiler.py:L86-L90 的 is_divisible_by_16 判据)。
DIVISIBILITY_VALUE = 16  # backends/compiler.py:L77
attributes = {
    0: [("tt.divisibility", DIVISIBILITY_VALUE)],
    1: [("tt.divisibility", DIVISIBILITY_VALUE)],
    2: [("tt.divisibility", DIVISIBILITY_VALUE)],
}

# ---- 逐字复刻 L414-L433 -----------------------------------------------------
def _is_constexpr(x):
    return True  # 占位:constants 里的值都已是编译期 Python 值

trace = []
arg_values = []          # 供函数体求值的实参(constexpr 塞 Python 值/非 constexpr 塞 SSA 句柄)
set_arg_attr_calls = []  # 记录每次 fn.set_arg_attr(idx, name, value)
idx = 0
for i in range(len(arg_names)):
    row = {"i": i, "name": arg_names[i], "idx_before": idx}
    if i in constants:                       # L415-L421
        row["branch"] = "constexpr"
        row["in_constants"] = True
        row["set_arg_attr"] = None
        arg_values.append(("constexpr", constants[i]))
        row["arg_value"] = f"constexpr({constants[i]})"
        row["idx_after"] = idx               # continue -> idx 不加
        row["ir_param_pos"] = None
        trace.append(row)
        continue
    else:
        row["branch"] = "tensor"
        row["in_constants"] = False
        if i in attributes:                  # L422-L424
            for name, value in attributes[i]:
                set_arg_attr_calls.append({"idx": idx, "name": name, "value": value})
            row["set_arg_attr"] = [f"set_arg_attr({idx}, '{n}', {v})" for n, v in attributes[i]]
        else:
            row["set_arg_attr"] = None
        arg_values.append(("tensor", f"fn.args({idx})"))   # L430
        row["arg_value"] = f"tensor(fn.args({idx}))"
        row["ir_param_pos"] = idx
        idx += 1                              # L431 -> 只有非 constexpr 才 ++
        row["idx_after"] = idx
        trace.append(row)

out = {
    "kernel": "add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr)",
    "python_param_count": len(arg_names),
    "ir_param_count": idx,                    # 最终 idx = 真进 IR 的参数个数
    "constexpr_folded": [arg_names[i] for i in constants],
    "divisibility_args": [c["idx"] for c in set_arg_attr_calls],
    "set_arg_attr_calls": set_arg_attr_calls,
    "trace": trace,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
