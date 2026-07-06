import os
import sys

# 让测试能 import 精简版的 implementation 包（不 import vllm）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
