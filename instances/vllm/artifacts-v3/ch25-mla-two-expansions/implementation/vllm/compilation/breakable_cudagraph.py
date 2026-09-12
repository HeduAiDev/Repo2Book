# SOURCE: vllm/compilation/breakable_cudagraph.py
# HOST SEAM：eager_break_during_capture 装饰器的 host 退化——真实在
# breakable cudagraph 捕获期把被饰函数切换到 eager 执行（ch19 域）；
# host 无捕获，恒为直通装饰器。unified_mla_attention_with_output 的
# 装饰位（mla_attention.py:L1231）保形。
from __future__ import annotations


# SOURCE: vllm/compilation/breakable_cudagraph.py eager_break_during_capture
#   —— HOST SEAM：无捕获直通
def eager_break_during_capture(fn):
    # SOURCE: vllm/compilation/breakable_cudagraph.py —— HOST SEAM 直通位
    return fn
