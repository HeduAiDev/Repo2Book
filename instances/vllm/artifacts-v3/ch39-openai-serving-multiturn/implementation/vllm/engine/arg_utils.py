# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/engine/arg_utils.py —— HOST SEAM（最小承载）：本章消费面
# 只有 build_async_engine_client 的 AsyncEngineArgs.from_cli_argss(args)
# 与 _api_process_count/_api_process_rank 两个多 API server 携带位
# （ch34 已立的 client_count/client_index）；真实是数千行 EngineArgs
# 配置族（ch3 域）。
from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/engine/arg_utils.py —— HOST SEAM：EngineArgs 最小位
@dataclass
class EngineArgs:
    # SOURCE: vllm/engine/arg_utils.py —— model 位
    model: str = "test-model"

    # SOURCE: vllm/engine/arg_utils.py —— enable_log_requests 位
    enable_log_requests: bool = False
    # SOURCE: vllm/engine/arg_utils.py —— aggregate_engine_logging 位
    aggregate_engine_logging: bool = False
    # SOURCE: vllm/engine/arg_utils.py —— disable_log_stats 位
    disable_log_stats: bool = False

    # SUBTRACTED: vllm/engine/arg_utils.py 其余 ~150 个引擎配置字段
    # （ch3 域：EngineArgs→VllmConfig 全链）。

    # SOURCE: vllm/engine/arg_utils.py:L1666-L1675 —— from_cli_args 逐字
    # （dataclasses.fields 按字段名从 Namespace 拷贝）
    @classmethod
    def from_cli_args(cls, args: Any):
        import dataclasses

        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Set the attributes from the parsed arguments.
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args


# SOURCE: vllm/engine/arg_utils.py —— HOST SEAM：AsyncEngineArgs 最小位
@dataclass
class AsyncEngineArgs(EngineArgs):
    # SOURCE: vllm/engine/arg_utils.py —— _api_process_count（多 API server
    # 携带位，ch34）
    _api_process_count: int = field(default=1)
    # SOURCE: vllm/engine/arg_utils.py —— _api_process_rank（同上）
    _api_process_rank: int = field(default=0)
