"""把精简版 implementation/ 目录加进 sys.path，让测试可 import 纯 Python 部分。"""
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))
