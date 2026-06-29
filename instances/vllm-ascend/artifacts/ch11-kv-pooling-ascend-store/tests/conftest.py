"""把精简版 implementation/ 放上 sys.path，让纯控制流测试能 import。

host 无 NPU/CANN/mooncake：测试只导入 config_data / kv_transfer / pool_scheduler /
backend.backend（纯 Python 控制流），绝不导入 backend.mooncake_backend（依赖
torch_npu / mooncake，host 不可发车）。后端契约用纯内存 FakeBackend 替身验证。
"""
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))
