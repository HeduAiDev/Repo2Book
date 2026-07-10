import sys
from pathlib import Path

# 让精简版的扁平 import（import llm / import core_client ...）可用。
IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))
