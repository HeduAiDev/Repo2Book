# SOURCE: vllm/utils/math_utils.py
# 本章消费面：cdiv（块数上取整——ext_comp 段分配与坏块截断的算术底座）。
# SUBTRACTED: 其余（取 min/max 桥接等）——ch13/14 各章切面。
from typing import Union


# SOURCE: vllm/utils/math_utils.py cdiv
def cdiv(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Ceiling division."""
    # SOURCE: vllm/utils/math_utils.py（(a + b - 1) // b）
    return (a + b - 1) // b
