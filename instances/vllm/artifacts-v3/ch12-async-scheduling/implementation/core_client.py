# SOURCE: vllm/v1/engine/core_client.py
# InprocClient —— 离线 LLM() 门面（m20 的对照组）：无忙循环，get_output 直调
# engine_core.step_fn() 并 post_step；默认配置下 step_fn 同样是重叠版（outputs
# 为 None 时兜底空包）。
from __future__ import annotations

from .core import EngineCore
from .engine import EngineCoreOutputs
from .logger import init_logger

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/core_client.py:L306 InprocClient
class InprocClient:
    """InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # SOURCE: vllm/v1/engine/core_client.py:L316-L317 __init__
    def __init__(self, *args, model_executor=None, **kwargs):
        # SOURCE: vllm/v1/engine/core_client.py:L317
        self.engine_core = EngineCore(*args, model_executor=model_executor, **kwargs)

    # SOURCE: vllm/v1/engine/core_client.py:L319-L322 get_output（逐字）
    def get_output(self) -> EngineCoreOutputs:
        # SOURCE: vllm/v1/engine/core_client.py:L320-L322
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    # SUBTRACTED: get_status/add_request 的 EngineCoreRequest 预处理面
    #   （L324-L336——Part II 的请求装配）；其余 client 族（ZMQ 消息面归 ch5）。
