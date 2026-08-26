# Subtract-only companion for v3 ch17 — vllm/utils/system_utils.py (pin
# v0.27.1 / 6e448d0ea)。本章消费的小件：update_environment_variables（顺序
# 约束第一步）。
#
# SUBTRACTED: system_utils.py 的其余部分（decorate_logs / set_process_title /
#   物理卡号映射 / bind_for_port 等——进程装饰与平台面已随删除项 4 裁除）。
# get_mp_context 与 _maybe_force_spawn 的宿主替身在 executor/multiproc_executor.py
# 文末（消费方就地标注）。

from __future__ import annotations

import os

from .._host_seams import init_logger

logger = init_logger(__name__)


# SOURCE: vllm/utils/system_utils.py:L34-L44 update_environment_variables — 逐字
def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v
