import os
import sys

# 让 `from implementation import ...` 在本章目录下可解析。
_CHAPTER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHAPTER_ROOT not in sys.path:
    sys.path.insert(0, _CHAPTER_ROOT)
