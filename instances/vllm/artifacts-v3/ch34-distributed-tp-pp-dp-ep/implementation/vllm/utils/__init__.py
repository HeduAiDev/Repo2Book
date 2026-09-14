# SOURCE: vllm/utils/__init__.py —— 精简版只携带本章触碰的两个名字：
# divide（RowParallelLinear 的整除断言）与 is_moe_layer（device communicator 的
# MoE 模块判定）。真实文件是大杂烩 re-export 层（ch01 域）。


# SOURCE: vllm/utils/__init__.py divide — 整除校验（真实实现）
def divide(a, b):
    """Calculate a divided by b, asserting divisibility."""
    assert b != 0, "divisor must be non-zero"
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


# SOURCE: vllm/utils/__init__.py is_moe_layer —— HOST SEAM：按模块属性判定 MoE 层
# （真实实现 isinstance 于 FusedMoE 族；ch26 域。精简版只消费布尔结果）。
def is_moe_layer(module) -> bool:  # HOST SEAM
    return bool(getattr(module, "is_moe", False))
