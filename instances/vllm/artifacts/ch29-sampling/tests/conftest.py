# 让 tests/ 能 import 精简版模块（它们彼此用裸名 import，如 `from metadata import ...`）。
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
