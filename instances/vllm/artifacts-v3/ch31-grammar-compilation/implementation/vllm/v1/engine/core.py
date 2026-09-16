# SOURCE: vllm/v1/engine/core.py
# 只做减法的忠实精简版：本章消费面 = preprocess_add_request（站 3 入口、
# grammar_init 的唯一调用点、线程安全注释所在）。真实 EngineCore 是引擎
# 忙循环主体（ch9 域，五千行）——精简版只载其与语法门直接相关的骨架面。
# SUBTRACTED: SPDX 版权头；EngineCore.__init__ 的数百行装配（executor/
#   scheduler/KV/模型加载——ch9/ch17 域）。
from vllm.config import VllmConfig
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


# SOURCE: vllm/v1/engine/core.py EngineCore（骨架面）
class EngineCore:
    # SOURCE: vllm/v1/engine/core.py:L71-L95 装配面 —— HOST SEAM（语法门切面）
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        # SUBTRACTED: mm_receiver_cache 的真实构造（mm 域）——L977-L982 的
        #   消费分支保留原样，字段置 None 即该分支永不触发。
        self.mm_receiver_cache = None
        # SUBTRACTED: request_block_hasher 的真实构造（prefix cache 域，
        #   ch15）——from_engine_core_request 按参数直传，置 None 语义同源。
        self.request_block_hasher = None
        self.structured_output_manager = StructuredOutputManager(vllm_config)
        # SUBTRACTED: 其余全部装配面（executor/loop/outputs 队列——ch9 域）。

    # SOURCE: vllm/v1/engine/core.py:L969-L991 preprocess_add_request —— 逐字
    #   （调用点 = 输入 IO 线程 core.py:L1718；docstring 原话 'allow request
    #   initialization running in parallel with Model forward'）
    # SOURCE: vllm/v1/engine/core.py:L969-L991
    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave
