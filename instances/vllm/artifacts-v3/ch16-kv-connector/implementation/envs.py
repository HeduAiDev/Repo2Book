# SOURCE: vllm/envs.py
# 本章消费面：VLLM_PREFIX_CACHE_RETENTION_INTERVAL（retention 旋钮——
# coordinator 构造期读入；默认 None=dense，本章不驱动稀疏驻留）。
# SUBTRACTED: 其余环境位（ch13/14 各章切面）。
import os

# SOURCE: vllm/envs.py VLLM_PREFIX_CACHE_RETENTION_INTERVAL
VLLM_PREFIX_CACHE_RETENTION_INTERVAL: int | None = (
    int(os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
    if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in os.environ
    else None
)
