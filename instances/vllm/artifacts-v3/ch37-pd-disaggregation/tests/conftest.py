# ch37 测试 conftest：把精简版的两个根加进 sys.path。
#   implementation/                              → `import vllm...`（引擎与 connector）
#   implementation/tests/v1/kv_connector/...     → `import toy_proxy_server`（disaggregator）
# 真实仓库里后者是 tests/ 下的集成测试工具，路径同构保留。
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_IMPL = _HERE.parents[0] / "implementation"
_PROXY = _IMPL / "tests" / "v1" / "kv_connector" / "nixl_integration"

for _p in (_IMPL, _PROXY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
