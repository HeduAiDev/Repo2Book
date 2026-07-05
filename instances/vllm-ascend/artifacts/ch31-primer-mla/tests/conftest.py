"""ch31 测试脚手架：implementation/ 是一份纯 NumPy 的论文参考实现（无 vLLM/CANN 依赖），
直接把 implementation/ 目录加进 sys.path 供各测试模块 import。
"""
import sys
from pathlib import Path

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))
