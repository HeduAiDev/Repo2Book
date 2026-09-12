# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/engine/core_client.py —— HOST SEAM（最小承载）：真实
# 1871 行是 ZMQ/进程间 EngineCore 客户端全族（ch5 域）；本章消费面只有
# AsyncLLM 的三跳：add_request_async / abort_requests_async /
# get_output_async（output_handler 已删）。退化位为可注入的记录型客户端。
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/core_client.py —— HOST SEAM：EngineCoreStatus
# 位（真实跨进程引擎状态镜像：engine_dead 旗标）
class EngineCoreStatus:
    # SOURCE: vllm/v1/engine/core_client.py —— engine_dead 位
    def __init__(self, engine_dead: bool = False):
        self.engine_dead = engine_dead


# SOURCE: vllm/v1/engine/core_client.py —— HOST SEAM：EngineCoreClient
# 记录型退化位（真实 InprocClient/ZmqClient 族按拓扑分发，ch5）
class EngineCoreClient:
    # SOURCE: vllm/v1/engine/core_client.py —— 构造位
    def __init__(self):
        self.added_requests: list[Any] = []
        self.aborted_requests: list[str] = []
        self.resources = EngineCoreStatus()

    # SOURCE: vllm/v1/engine/core_client.py —— add_request_async 位
    # （真实：EngineCoreRequest 经 IPC 通道发给 EngineCore 进程）
    async def add_request_async(self, request: Any) -> None:
        self.added_requests.append(request)

    # SOURCE: vllm/v1/engine/core_client.py —— abort_requests_async 位
    # （真实：ABORT 消息跨进程让引擎停算释放 KV——abort 两跳的第二跳）
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.aborted_requests.extend(request_ids)

    # SUBTRACTED: vllm/v1/engine/core_client.py 其余 ~1800 行
    # （get_output_async/ shuts down/健康检查/DP 拓扑族）——ch5 ZMQ 域；
    # 本章 output_handler 后台循环已删，不消费输出拉取。
