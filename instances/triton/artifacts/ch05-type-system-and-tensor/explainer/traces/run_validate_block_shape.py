"""ch05 mech m04-validate-block-shape — 2 的幂 + numel<=2^20 的把关（pin 3.2.0 实证）。

pin 配方（INSTANCE.md）：_utils.validate_block_shape 与 pin 源码逐字节一致，纯 Python 无需 GPU。
直接喂各种 shape 观测 numel / 抛错，再用 block_type 构造复现同一把关（构造即调 validate）。
运行：v32/bin/python run_validate_block_shape.py > validate_block_shape.txt
"""
from triton.language._utils import validate_block_shape, TRITON_MAX_TENSOR_NUMEL, is_power_of_two
from triton.language.core import block_type, float16

print("TRITON_MAX_TENSOR_NUMEL =", TRITON_MAX_TENSOR_NUMEL, "= 2^20 =", 2 ** 20)
print("is_power_of_two(1) =", is_power_of_two(1))
print("is_power_of_two(1024) =", is_power_of_two(1024))
print("is_power_of_two(1000) =", is_power_of_two(1000))
print()

cases = [
    [16, 16],       # ok: numel 256
    [1024, 1024],   # ok: numel 1048576 == 2^20 exactly (boundary, allowed)
    [2048, 1024],   # fail: numel 2097152 > 2^20
    [1000, 16],     # fail: 1000 not power of 2
]
print("== validate_block_shape(shape) -> numel or ValueError ==")
for shape in cases:
    try:
        numel = validate_block_shape(shape)
        print(f"shape={shape}  numel={numel}  OK  (2^20={2**20})")
    except ValueError as e:
        print(f"shape={shape}  ValueError: {e}")
print()

print("== block_type(...) construction runs the same gate ==")
for shape in [[128, 64], [2048, 1024]]:
    try:
        bt = block_type(float16, shape)
        print(f"block_type(fp16, {shape})  numel={bt.numel}  name={bt.name}")
    except ValueError as e:
        print(f"block_type(fp16, {shape})  ValueError: {e}")
