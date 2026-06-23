import sys
from pathlib import Path

# 让测试能 import 精简版模块（implementation/ 内是扁平 import）。
IMPL = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
