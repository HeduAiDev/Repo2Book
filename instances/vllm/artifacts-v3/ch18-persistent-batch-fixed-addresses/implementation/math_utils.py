# SOURCE: vllm/utils/math_utils.py
# 本章消费面：cdiv——InputBatch 装配期 placeholder_max_num_blocks 的上取整
# （gpu_model_runner.py:L697-L698）与 block_table get_block_table_width 的
# token_alignment 对齐（block_table.py:L39）。
# SUBTRACTED: 其余数学工具（ch13/ch14 域）。


# SOURCE: vllm/utils/math_utils.py cdiv
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    # SOURCE: vllm/utils/math_utils.py cdiv 函数体
    return -(a // -b)
