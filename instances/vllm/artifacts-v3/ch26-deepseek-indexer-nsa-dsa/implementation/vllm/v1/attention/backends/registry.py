# SOURCE: vllm/v1/attention/backends/registry.py
# HOST SEAM：decode 家族注册表子集（本章引用成员 FLASHMLA_SPARSE ——selector
# 显式支经真实注册流解析）。完整注册表与优先级面归 ch21/ch25。
from __future__ import annotations

import enum


# SOURCE: vllm/v1/attention/backends/registry.py AttentionBackendEnum —— HOST
#   SEAM 子集（本章成员位）
class AttentionBackendEnum(enum.Enum):
    FLASHMLA_SPARSE = "FLASHMLA_SPARSE"

    def get_class(self) -> type:
        # SOURCE: vllm/v1/attention/backends/registry.py get_class —— HOST
        #   SEAM（显式模块解析）
        from importlib import import_module

        table = {
            "FLASHMLA_SPARSE": (
                "vllm.v1.attention.backends.mla.flashmla_sparse",
                "FlashMLASparseBackend",
            ),
        }
        module_name, cls_name = table[self.value]
        return getattr(import_module(module_name), cls_name)
