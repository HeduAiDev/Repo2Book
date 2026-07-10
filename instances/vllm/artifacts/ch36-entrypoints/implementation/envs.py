"""精简版 envs —— 仅本章 entrypoints 主线读到的环境开关。

与真实 vllm/envs.py 同名同语义；真实文件是惰性 __getattr__ 工厂，这里直接给值。
# SUBTRACTED: vllm/envs.py 的惰性 __getattr__ 工厂与数百个其他开关，只留本章用到的 4 个。
"""

import os

# SOURCE: vllm/envs.py:VLLM_KEEP_ALIVE_ON_ENGINE_DEATH —— watchdog 是否在引擎死后仍保活
VLLM_KEEP_ALIVE_ON_ENGINE_DEATH = bool(int(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "0")))

# SOURCE: vllm/envs.py:VLLM_LOG_STATS_INTERVAL —— lifespan 后台 do_log_stats 周期(秒)
VLLM_LOG_STATS_INTERVAL = float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10.0"))

# SOURCE: vllm/envs.py:VLLM_HTTP_TIMEOUT_KEEP_ALIVE
VLLM_HTTP_TIMEOUT_KEEP_ALIVE = int(os.getenv("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5"))

# SOURCE: vllm/envs.py:VLLM_API_KEY
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
