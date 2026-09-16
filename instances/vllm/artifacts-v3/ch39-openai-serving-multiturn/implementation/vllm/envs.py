# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/envs.py —— HOST SEAM：真实文件是自动生成的大环境变量表
# （~千行）。这里只承载本章精简版代码实际消费的变量，语义与真实默认值
# 一致：环境变量优先，缺省回落到 v0.27.1 的默认值。
from __future__ import annotations

import os


# SOURCE: vllm/envs.py —— HOST SEAM：属性化环境变量读取（真实为
# `environment_variables` dataclass 实例；本章仅按同名属性取值）。
class _Envs:
    # SUBTRACTED: 其余 ~150 个 VLLM_* 变量（vllm/envs.py 全表）——本章
    # 服务面代码不消费，host 精简环境不逐个复刻。

    # SOURCE: vllm/envs.py —— VLLM_API_KEY 默认 None
    @property
    def VLLM_API_KEY(self) -> str | None:
        return os.getenv("VLLM_API_KEY")

    # SOURCE: vllm/envs.py —— VLLM_SERVER_DEV_MODE 默认 False
    @property
    def VLLM_SERVER_DEV_MODE(self) -> int:
        return int(bool(os.getenv("VLLM_SERVER_DEV_MODE", "0") not in ("", "0")))

    # SOURCE: vllm/envs.py —— VLLM_DEBUG_LOG_API_SERVER_RESPONSE 默认 0
    @property
    def VLLM_DEBUG_LOG_API_SERVER_RESPONSE(self) -> int:
        return int(bool(os.getenv("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "0")))

    # SOURCE: vllm/envs.py —— VLLM_KEEP_ALIVE_ON_ENGINE_DEATH 默认 0
    @property
    def VLLM_KEEP_ALIVE_ON_ENGINE_DEATH(self) -> int:
        return int(bool(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "0")))

    # SOURCE: vllm/envs.py —— VLLM_ALLOW_RUNTIME_LORA_UPDATING 默认 False
    @property
    def VLLM_ALLOW_RUNTIME_LORA_UPDATING(self) -> bool:
        return os.getenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0") == "1"

    # SOURCE: vllm/envs.py —— VLLM_SKIP_MODEL_NAME_VALIDATION 默认 False
    @property
    def VLLM_SKIP_MODEL_NAME_VALIDATION(self) -> bool:
        return os.getenv("VLLM_SKIP_MODEL_NAME_VALIDATION", "0") == "1"

    # SOURCE: vllm/envs.py —— VLLM_ENFORCE_STRICT_TOOL_CALLING 默认 0
    @property
    def VLLM_ENFORCE_STRICT_TOOL_CALLING(self) -> int:
        return int(bool(os.getenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "0")))

    # SOURCE: vllm/envs.py —— VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS
    # 默认 False（gpt-oss 系统指令进 developer 消息而非 system 消息）
    @property
    def VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS(self) -> bool:
        return os.getenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "0") == "1"

    # SOURCE: vllm/envs.py —— VLLM_SYSTEM_START_DATE 默认 None
    # （None 时 harmony preamble 用 datetime.now() 当前日期——非确定性来源）
    @property
    def VLLM_SYSTEM_START_DATE(self) -> str | None:
        return os.getenv("VLLM_SYSTEM_START_DATE")

    # SOURCE: vllm/envs.py —— VLLM_HTTP_TIMEOUT_KEEP_ALIVE 默认 5
    @property
    def VLLM_HTTP_TIMEOUT_KEEP_ALIVE(self) -> int:
        return int(os.getenv("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5"))

    # SOURCE: vllm/envs.py —— VLLM_DISABLE_LOG_LOGO 默认 False
    @property
    def VLLM_DISABLE_LOG_LOGO(self) -> bool:
        return os.getenv("VLLM_DISABLE_LOG_LOGO", "0") == "1"

    # SOURCE: vllm/envs.py —— VLLM_WORKER_MULTIPROC_METHOD 默认 None
    @property
    def VLLM_WORKER_MULTIPROC_METHOD(self) -> str | None:
        return os.getenv("VLLM_WORKER_MULTIPROC_METHOD")

    # SOURCE: vllm/envs.py —— VLLM_V1_OUTPUT_PROC_CHUNK_SIZE 默认 64
    @property
    def VLLM_V1_OUTPUT_PROC_CHUNK_SIZE(self) -> int:
        return int(os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "64"))

    # SOURCE: vllm/envs.py —— VLLM_MAX_N_SEQUENCES 默认 16384
    # （SamplingParams._verify_args 的 n 上限）
    @property
    def VLLM_MAX_N_SEQUENCES(self) -> int:
        return int(os.getenv("VLLM_MAX_N_SEQUENCES", "16384"))

    # SOURCE: vllm/envs.py —— VLLM_LOG_STATS_INTERVAL 默认 5
    # （lifespan 的 do_log_stats 后台任务节拍）
    @property
    def VLLM_LOG_STATS_INTERVAL(self) -> int:
        return int(os.getenv("VLLM_LOG_STATS_INTERVAL", "5"))

    # SOURCE: vllm/envs.py —— VLLM_DISABLE_REQUEST_ID_RANDOMIZATION 默认 0
    # （assign_request_id 的 8 位随机后缀开关）
    @property
    def VLLM_DISABLE_REQUEST_ID_RANDOMIZATION(self) -> int:
        return int(bool(os.getenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0")))


envs = _Envs()


# SOURCE: vllm/envs.py —— HOST SEAM：模块级 __getattr__ 委托位（真实
# envs.py 的访问惯例是 `import vllm.envs as envs; envs.VLLM_X`——属性
# 挂在模块上；这里委托到 _Envs 实例，消费面两种写法等价）
def __getattr__(name: str):
    return getattr(envs, name)
