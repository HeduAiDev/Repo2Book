# SOURCE: vllm/v1/core/sched/interface.py
# SchedulerInterface —— 调度器接口面（EngineCore 消费的方法签名）。
# PauseState 等暂停体系归 ch11；本章只保接口方法。
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import EngineCoreOutputs
    from .output import SchedulerOutput
    from .outputs import ModelRunnerOutput
    from .request import Request, RequestStatus


# SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface
class SchedulerInterface(ABC):
    # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.add_request
    @abstractmethod
    def add_request(self, request: "Request") -> None:
        # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.add_request
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.schedule
    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.schedule
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.update_from_output
    @abstractmethod
    def update_from_output(
        self, scheduler_output: "SchedulerOutput", model_output: "ModelRunnerOutput"
    ) -> "dict[int, EngineCoreOutputs]":
        # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.update_from_output
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.has_requests
    @abstractmethod
    def has_requests(self) -> bool:
        # SOURCE: vllm/v1/core/sched/interface.py SchedulerInterface.has_requests
        raise NotImplementedError

    # SUBTRACTED: pause/connector/streaming 面——各归邻章。
