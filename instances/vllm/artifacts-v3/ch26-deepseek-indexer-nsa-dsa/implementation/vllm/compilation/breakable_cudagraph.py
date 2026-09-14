# SOURCE: vllm/compilation/breakable_cudagraph.py
# HOST SEAM（ch25 同款）：eager_break_during_capture 的直通装饰器——真实实现
# 在 CUDA graph 捕获期把被饰函数切到 eager 执行（ch19 域）；host 无捕获，
# 直通即真实非捕获形态。BreakableCUDAGraphCapture.is_active() 恒 False 是
# maybe_execute_in_parallel 的多流禁用判定消费位。
from __future__ import annotations

from functools import wraps


# SOURCE: vllm/compilation/breakable_cudagraph.py BreakableCUDAGraphCapture
#   —— HOST SEAM：非捕获态
class BreakableCUDAGraphCapture:
    # SOURCE: vllm/compilation/breakable_cudagraph.py is_active —— HOST SEAM
    @staticmethod
    def is_active() -> bool:
        # SOURCE: vllm/compilation/breakable_cudagraph.py is_active
        return False


# SOURCE: vllm/compilation/breakable_cudagraph.py eager_break_during_capture
#   —— HOST SEAM：直通装饰器
def eager_break_during_capture(fn):
    # SOURCE: vllm/compilation/breakable_cudagraph.py —— HOST SEAM 直通
    @wraps(fn)
    # SOURCE: vllm/compilation/breakable_cudagraph.py —— HOST SEAM 直通（锚点双置）
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
