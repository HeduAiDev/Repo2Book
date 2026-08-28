# SOURCE: vllm/envs.py（本章切面：只取 retention 旋钮一个环境位）
# VLLM_PREFIX_CACHE_RETENTION_INTERVAL——稀疏驻留粒度（m17）：未设=稠密 / 0=
# 只留 replay 边界 / 正数=每段一条；coordinator 构造时读入并经
# _validate_prefix_cache_retention_interval 校验（kv_cache_coordinator.py:L30-L57）。
# SUBTRACTED: envs.py 其余全部环境位（ch03/ch05 各章切面）。
import os


# SOURCE: vllm/envs.py:L302 VLLM_PREFIX_CACHE_RETENTION_INTERVAL
# SUBTRACTED: @dataclass envs 装配面（ch03）——本章以模块级直读等价复现
#   L1120-L1124 的 lambda 语义（在 os.environ 里则 int()，否则 None）。
VLLM_PREFIX_CACHE_RETENTION_INTERVAL: int | None = (
    int(os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
    if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in os.environ
    else None
)
