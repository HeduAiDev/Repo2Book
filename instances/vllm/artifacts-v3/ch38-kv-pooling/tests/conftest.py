# ch38 测试 conftest：把精简版根加进 sys.path，让 `import vllm...` 落到
# implementation/vllm 包（与目标代码仓同树同名同结构）。
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_IMPL = _HERE.parents[0] / "implementation"

if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))
