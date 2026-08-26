# Subtract-only companion for v3 ch17 — vllm/utils/gc_utils.py (pin v0.27.1
# / 6e448d0ea)。compile_or_warm_up_model 尾部的 freeze_gc_heap（m9 启动期
# 前移收口：静态对象全堆冻结，推理期 GC 不再扫描）与 GC debug 钩。
#
# freeze_gc_heap 逐字（ch09 同款）；GCDebugConfig/GCDebugger 的完整面按需裁除。

from __future__ import annotations

import gc

from .._host_seams import envs


# SOURCE: vllm/utils/gc_utils.py:L96-L108 freeze_gc_heap — 逐字
def freeze_gc_heap() -> None:
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
    # Freeze all GC tracked objects
    gc.freeze()


# SOURCE: vllm/utils/gc_utils.py:L111-L123 maybe_attach_gc_debug_callback —
# 宿主替身：VLLM_GC_DEBUG 未配置时不挂任何回调（真实分支语义）
# SOURCE: (见 impl-notes.md §Source Map——utils/gc_utils.py)
def maybe_attach_gc_debug_callback() -> None:  # HOST SEAM（GCDebugConfig 面裁除）
    if not envs.VLLM_GC_DEBUG:
        return
    # SUBTRACTED: GCDebugConfig/GCDebugger 与 gc.callbacks 挂接（gc_utils.py:
    #   L115-L123——GC 调试器深水面）。
