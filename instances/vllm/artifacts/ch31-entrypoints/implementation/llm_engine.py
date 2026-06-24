"""LLMEngine —— v1 同步引擎（精简版，只做减法）。

与真实 vllm/v1/engine/llm_engine.py 同名、同结构、同控制流：

  * from_engine_args —— 【关键澄清锚点】envs.VLLM_ENABLE_V1_MULTIPROCESSING(默认True)
                        → 把 enable_multiprocessing 强翻为 True → multiprocess_mode=True。
  * __init__         —— 建 InputProcessor/OutputProcessor + EngineCoreClient.make_client(
                        multiprocess_mode, asyncio_mode=False)。离线默认 → SyncMPClient。
  * add_request      —— input_processor.process_inputs → output_processor.add_request +
                        engine_core.add_request；n>1 走 ParentRequest 扇出。
  * step             —— 同步单拍：get_output()→process_outputs→abort 停止串→（记 stats）。
  * has_unfinished_requests / get_num_unfinished_requests —— _run_engine while 循环的边界。

把所有 # SUBTRACTED 分支删回去 ≈ 真实 LLMEngine 的同步主干。
"""

from __future__ import annotations

from copy import copy

import envs
from core_client import EngineCoreClient
from messages import (
    PoolingRequestOutput,
    RequestOutput,
    SamplingParams,
)
from processors_stub import InputProcessor, OutputProcessor, ParentRequest


# SOURCE: vllm/v1/engine/llm_engine.py:L47 (class LLMEngine)
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    # SOURCE: vllm/v1/engine/llm_engine.py:L50 (LLMEngine.__init__)
    def __init__(
        self,
        vllm_config=None,
        executor_class=None,
        log_stats: bool = False,
        usage_context=None,
        stat_loggers=None,
        multiprocess_mode: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = getattr(vllm_config, "model_config", None)
        self.log_stats = log_stats
        # SUBTRACTED: tracing 初始化、external_launcher_dp / dp_group / should_execute_dummy_batch
        #   等数据并行状态（llm_engine.py:L62-L88）—— DP 是 ch04/DP 侧主题；单卡离线场景
        #   dp_group=None、should_execute_dummy_batch 恒 False，删后非 DP 主控制流完全等价。
        self.dp_group = None

        # SUBTRACTED: self.renderer = renderer_from_config(...)（tokenizer 装配，ch05）。
        #   原 vllm/v1/engine/llm_engine.py:L90。stub 的 input_processor 不需要它。
        self.renderer = None

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer=None)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            tokenizer=None,
            log_stats=self.log_stats,
            stream_interval=1,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        # SOURCE: vllm/v1/engine/llm_engine.py:L104 (EngineCoreClient.make_client)
        # 【硬分叉点】asyncio_mode=False：同步引擎绝不进 asyncio 事件循环（ch04 是 True）。
        # multiprocess_mode 由上游 from_engine_args 传入，离线默认 True → SyncMPClient。
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        # SUBTRACTED: logger_manager 初始化、model_executor(v0 兼容)、external_launcher_dp 复用、
        #   reset_mm_cache()（llm_engine.py:L112-L132）—— 可观测性/v0 兼容/DP 旁路。
        self.multiprocess_mode = multiprocess_mode

    @classmethod
    def from_engine_args(
        cls,
        engine_args=None,
        usage_context=None,
        stat_loggers=None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        # SOURCE: vllm/v1/engine/llm_engine.py:L151 (LLMEngine.from_engine_args)
        """Creates an LLM engine from the engine arguments."""

        # SUBTRACTED: vllm_config = engine_args.create_engine_config(usage_context)；
        #   executor_class = Executor.get_class(vllm_config)（配置归一/执行器选型，ch03）。
        #   原 vllm/v1/engine/llm_engine.py:L162-L163。stub 直接透传 engine_args 作 config。
        vllm_config = engine_args
        executor_class = None

        # 【关键澄清】形参默认 enable_multiprocessing=False 是【误导性】的——真实默认路径靠
        # envs.VLLM_ENABLE_V1_MULTIPROCESSING（默认 True）把它强翻为 True，于是离线走 SyncMPClient。
        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not getattr(engine_args, "disable_log_stats", True),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    # SOURCE: vllm/v1/engine/llm_engine.py:L179 (get_num_unfinished_requests)
    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    # SOURCE: vllm/v1/engine/llm_engine.py:L182 (has_unfinished_requests)
    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        # SUBTRACTED: dp_group is None 时还要 or self.engine_core.dp_engines_running()，
        #   以及 dp_group 非空走 has_unfinished_requests_dp（DP 跨 rank 聚合 + dummy batch）。
        #   原 llm_engine.py:L184-L194。单卡离线 dp_group=None 且无 DP engines → 直接返回。
        return has_unfinished

    def get_supported_tasks(self):
        # SOURCE: vllm/v1/engine/llm_engine.py:L196 (get_supported_tasks) — stub
        return ("generate", "encode", "embed")

    # SOURCE: vllm/v1/engine/llm_engine.py:L203 (abort_request)
    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""
        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    # SOURCE: vllm/v1/engine/llm_engine.py:L209 (add_request)
    def add_request(
        self,
        request_id: str,
        prompt,
        params,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> str:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # SUBTRACTED: isinstance(prompt, EngineCoreRequest) 弃用分支（v0.18 移除，warning_once）。
        #   原 vllm/v1/engine/llm_engine.py:L226-L239。本章只走 else 的正常渲染路径。
        # Process raw inputs into the request.
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            supported_tasks=self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
        )
        # SUBTRACTED: prompt_text, _, _ = extract_prompt_components(model_config, prompt)
        #   （供 OutputProcessor 拼最终文本，ch08）。原 llm_engine.py:L252。
        prompt_text = None

        self.input_processor.assign_request_id(request)

        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return req_id

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request)
        for idx in range(n):
            request_id, child_params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.params = child_params

            # Make a new RequestState and queue.
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

        return req_id

    # SOURCE: vllm/v1/engine/llm_engine.py:L287 (step)
    def step(self) -> list:
        # SUBTRACTED: should_execute_dummy_batch 分支（DP 时空跑一拍对齐）；非 DP 恒 False。
        #   原 llm_engine.py:L288-L291。

        # 1) Get EngineCoreOutput from the EngineCore.
        # SUBTRACTED: record_function_or_nullcontext(...) torch profiler 包裹（4 处 with）。
        #   原 llm_engine.py:L294/L298/L308/L312。可观测性旁路，不影响 4 步主干。
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        # SUBTRACTED: iteration_stats = IterationStats() if log_stats else None（stats 旁路）。
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs,
            engine_core_timestamp=None,
            iteration_stats=None,
        )
        self.output_processor.update_scheduler_stats(None)

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        # SUBTRACTED: logger_manager.record(...) + do_log_stats_with_interval()（stat logging 旁路）。
        #   原 llm_engine.py:L312-L323。

        return processed_outputs.request_outputs
