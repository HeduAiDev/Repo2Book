# SOURCE: vllm/engine/arg_utils.py
# 预算默认值的**硬件与使用场景仲裁**（m4）：config ClassVar 的 2048 只是测试便利
# 基线，真实部署在 EngineArgs.create_engine_config 里按显存 + UsageContext 覆盖。
# A100 反例注释（PR #17885：大预算反而降吞吐）是『同一 vLLM、不同卡不同默认』的
# 源码证据。只保留 get_batch_defaults 的 GPU 决策表；删除项全部 dossier.delete 批准
# 或属邻章（TPU/CPU 平台表、平台探测装配）。
from __future__ import annotations

from enum import Enum


# SOURCE: vllm/utils/mem_constants.py:L18 GiB_bytes
GiB_bytes = 1 << 30


# SOURCE: vllm/usage/usage_lib.py:L112 UsageContext
class UsageContext(str, Enum):
    # SUBTRACTED: UNKNOWN_CONTEXT / API_SERVER / OPENAI_BATCH_RUNNER /
    #   ENGINE_CONTEXT 成员（L113/L115/L117/L118，使用统计上报用的枚举面）——
    #   预算仲裁表只用到 LLM_CLASS 与 OPENAI_API_SERVER 两档。
    LLM_CLASS = "LLM_CLASS"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"


# SOURCE: vllm/engine/arg_utils.py EngineArgs
class EngineArgs:
    # SUBTRACTED: EngineArgs 其余上百个字段与 create_engine_config 全量装配
    #   （vllm/engine/arg_utils.py 全文，ch03 话头）——本章只留预算默认值仲裁
    #   这一个方法；max_num_batched_tokens/max_num_seqs 字段本身留在调用方
    #   （create_engine_config）按返回表覆写。

    # SOURCE: vllm/engine/arg_utils.py:L2514-L2516 get_batch_defaults
    @classmethod
    def get_batch_defaults(
        cls,
        world_size: int,
        device_memory: int = 0,
        device_name: str = "",
    ) -> tuple[dict[UsageContext | None, int], dict[UsageContext | None, int]]:
        # SOURCE: vllm/engine/arg_utils.py:L2521-L2522
        default_max_num_batched_tokens: dict[UsageContext | None, int]
        default_max_num_seqs: dict[UsageContext | None, int]

        # SUBTRACTED: current_platform 的设备名/显存探测（L2528-L2540——
        #   探测失败回退 device_memory=0/device_name="" 的行为，这里直接以
        #   参数默认值承载；world_size 仅 CPU/DP 档使用，保留参数签名）。

        # NOTE(Kuntai): Setting large `max_num_batched_tokens` for A100 reduces
        # throughput, see PR #17885 for more details.
        # So here we do an extra device name check to prevent such regression.
        # SOURCE: vllm/engine/arg_utils.py:L2541-L2563 GPU 决策表
        if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
            # For GPUs like H100 and MI300x, use larger default values.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 1024,
                UsageContext.OPENAI_API_SERVER: 1024,
            }
        else:
            # TODO(woosuk): Tune the default values for other hardware.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256,
                UsageContext.OPENAI_API_SERVER: 256,
            }
        return default_max_num_batched_tokens, default_max_num_seqs
        # SUBTRACTED: TPU 分档表（L2565-L2583）与 CPU 分档表（L2585-L2594，
        #   world_size 线性放大）——非 GPU 平台的等价变体，各归其平台章。
