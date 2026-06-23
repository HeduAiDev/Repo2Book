"""把 ch24 精简版 implementation/ 目录加入 sys.path，使 backend/registry/selector/
platform_cuda/flash_attn/attention_layer 可顶层 import（与 registry 中 FLASH_ATTN 类路径
'flash_attn.FlashAttentionBackend' 的懒加载解析一致）。"""

import sys
from pathlib import Path

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))

# 测试目录也加入 path，使 register_backend 覆盖测试里 _OverrideBackend 的 __module__
# (test_ch24_attention) 可被 resolve_obj_by_qualname 重新 import 解析。
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
