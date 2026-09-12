# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/serve/utils/fingerprint.py —— HOST SEAM（最小
# 承载）：本章消费面只有 GenerateBaseServing 构造期的 get_system_fingerprint
# （system_fingerprint 盖终止块，m5）与 init_generate_state 的
# set_default_fingerprint_mode。真实实现聚合版本/配置指纹串。
from typing import Any

# SOURCE: vllm/entrypoints/serve/utils/fingerprint.py —— HOST SEAM：模块级
# 默认模式位（真实为 _default_mode 全局）
_default_mode = "full"


# SOURCE: vllm/entrypoints/serve/utils/fingerprint.py:L31-L40 —— HOST SEAM：
# set_default_fingerprint_mode（记录模式；构造期早于 serving 对象）
def set_default_fingerprint_mode(mode: str = "full", value: Any = None) -> None:
    global _default_mode
    _default_mode = mode


# SOURCE: vllm/entrypoints/serve/utils/fingerprint.py:L42-L46 —— HOST SEAM：
# get_system_fingerprint 退化（真实经缓存聚合 vllm 版本+配置指纹；精简
# 环境返回 None——GenerateBaseServing 的 try/except 本就容许 None）
def get_system_fingerprint(vllm_config: Any) -> str | None:
    return None


# SUBTRACTED: vllm/entrypoints/serve/utils/fingerprint.py:L48-L84
# build_system_fingerprint 指纹聚合实现——部署指纹域，本章按 m5 只消费
# 「盖终止块」语义。
