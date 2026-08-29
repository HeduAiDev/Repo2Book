# Subtract-only companion for v3 ch19 — vllm/utils/gc_utils.py
# (pin v0.27.1 / 6e448d0ea). Startup-tail GC freeze faces consumed by
# Worker.compile_or_warm_up_model (m18).
from __future__ import annotations

import gc


# SUBTRACTED: vllm/utils/gc_utils.py GCDebugConfig/GCDebugger 观测族与
#   maybe_attach_gc_debug_callback 的回调装配（L12-L95、L111-L127——观测，
#   VLLM_GC_DEBUG 默认关闭路径；freeze_gc_heap 主体保留）。
# SOURCE: vllm/utils/gc_utils.py:L96-L108 freeze_gc_heap — 启动收尾：冻结
# GC 堆，服务期 GC 不再扫描静态对象
def freeze_gc_heap() -> None:  # SOURCE: vllm/utils/gc_utils.py:L96-L108
    """
    Freeze all objects tracked by the garbage collector. It should be invoked
    after server init / warmup, to reduce GC overhead from static objects
    during serving time.
    """
    # Ensure all static objects are pushed down to the oldest generation for
    # freeze
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()


# SOURCE: vllm/utils/gc_utils.py:L111-L127 maybe_attach_gc_debug_callback —
# HOST SEAM no-op（VLLM_GC_DEBUG 默认 ""；观测域）
def maybe_attach_gc_debug_callback() -> None:  # SOURCE: vllm/utils/gc_utils.py:L111-L127
    # SUBTRACTED: VLLM_GC_DEBUG 启用时的 GCDebugger 回调装配（观测域，
    #   默认关闭路径为直接返回）
    return None
