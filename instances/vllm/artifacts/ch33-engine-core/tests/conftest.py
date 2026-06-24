import os
import sys

# 让测试能 import implementation/ 下的纯精简版模块（不 import vllm）。
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "implementation"),
)
