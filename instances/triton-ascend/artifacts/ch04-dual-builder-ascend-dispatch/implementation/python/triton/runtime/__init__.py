# SOURCE: python/triton/runtime/__init__.py:L4 `from .jit import JITFunction, ...`
# SUBTRACTED: 真实还重导出 KernelInterface/MockTensor/TensorWrapper/reinterpret 等，
# 本章只需要 JITFunction 这一个名字（visit_Call 的 isinstance 分支判据）。
from .jit import JITFunction

__all__ = ["JITFunction"]
