# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/engine/protocol.py —— 忠实承载（消费面）：EngineClient 是
# handler 依赖的抽象协议（must_keep；key_classes：AsyncLLM 是全仓唯一实现
# ——OpenAI 层只面向此抽象编程，ch4 站 1 已立）。generate/abort/
# check_health/pause_resume/shutdown 方法面 + renderer/input_processor/
# model_config 暴露逐字；weight-transfer（RL 训练面）与 fault-tolerance
# 尾段（容错面）删——均非本章服务面主线，且无消费点。
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import ModelConfig, VllmConfig
from vllm.inputs import EngineInput, PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask

if TYPE_CHECKING:
    from vllm.v1.engine import PauseMode
    from vllm.v1.engine.input_processor import InputProcessor

# SUBTRACTED: vllm/engine/protocol.py:L10-L13 weight_transfer 导入与
# L23 fault_tolerance 导入——RL 训练面/容错面非本章主线（对应尾段方法
# 一并删，见文件尾 SUBTRACTED 注记）。


# SOURCE: vllm/engine/protocol.py:L29-L38 —— StreamingInput 逐字
@dataclass
class StreamingInput:
    """Input data for a streaming generation request.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
    """

    prompt: EngineInput
    sampling_params: SamplingParams | None = None


# SOURCE: vllm/engine/protocol.py:L41-L281 —— EngineClient 逐字
class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    vllm_config: VllmConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    input_processor: "InputProcessor"

    # SOURCE: vllm/engine/protocol.py:L49-L51
    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    # SOURCE: vllm/engine/protocol.py:L53-L55
    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    # SOURCE: vllm/engine/protocol.py:L57-L59 —— errored（WC3 渲染前
    # 预检与 watchdog 的判定源）
    @property
    @abstractmethod
    def errored(self) -> bool: ...

    # SOURCE: vllm/engine/protocol.py:L61-L63 —— dead_error（render_chat_request
    # 抛出的引擎死讯）
    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    # SOURCE: vllm/engine/protocol.py:L65-L85 —— generate（站 7 发车点：
    # 只建协程不驱动；站 10 断连 abort 的载体）
    @abstractmethod
    def generate(
        self,
        prompt: EngineInput
        | PromptType
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    # SOURCE: vllm/engine/protocol.py:L87-L100
    @abstractmethod
    def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    # SOURCE: vllm/engine/protocol.py:L102-L110
    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    # SOURCE: vllm/engine/protocol.py:L112-L124
    @abstractmethod
    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Notify the engine that a KV-transfer request was rejected before
        engine admission, so connector-side cleanup can run (e.g. free
        prefill blocks pinned on the P node).
        """
        ...

    # SOURCE: vllm/engine/protocol.py:L126-L127
    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    # SOURCE: vllm/engine/protocol.py:L129-L130
    @abstractmethod
    async def do_log_stats(self) -> None: ...

    # SOURCE: vllm/engine/protocol.py:L132-L135
    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    # SOURCE: vllm/engine/protocol.py:L137-L139
    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    # SOURCE: vllm/engine/protocol.py:L141-L143
    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine"""
        ...

    # SOURCE: vllm/engine/protocol.py:L145-L147
    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    # SOURCE: vllm/engine/protocol.py:L149-L151
    @abstractmethod
    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache"""
        ...

    # SOURCE: vllm/engine/protocol.py:L153-L162
    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache"""
        ...

    # SOURCE: vllm/engine/protocol.py:L164-L167
    @abstractmethod
    async def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None:
        """Sleep the engine"""
        ...

    # SOURCE: vllm/engine/protocol.py:L169-L172
    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake the engine"""
        ...

    # SOURCE: vllm/engine/protocol.py:L174-L177
    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    # SOURCE: vllm/engine/protocol.py:L179-L182
    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    # SOURCE: vllm/engine/protocol.py:L184-L205
    @abstractmethod
    async def pause_generation(
        self,
        *,
        mode: "PauseMode" = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation/encoding requests.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  and return partial results with "abort" reason (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED. Use ``mode="wait"`` instead.
            clear_cache: DEPRECATED. Whether to clear KV and prefix caches
                after draining.
        """
        ...

    # SOURCE: vllm/engine/protocol.py:L207-L210
    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        ...

    # SOURCE: vllm/engine/protocol.py:L212-L215
    @abstractmethod
    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        ...

    # SOURCE: vllm/engine/protocol.py:L217-L220
    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine with optional timeout."""
        ...

    # SOURCE: vllm/engine/protocol.py:L222-L226
    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None:
        """Scale the engine"""
        raise NotImplementedError

    # SOURCE: vllm/engine/protocol.py:L228-L236
    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    # SOURCE: vllm/engine/protocol.py:L248-L250
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks"""
        raise NotImplementedError

# SUBTRACTED: vllm/engine/protocol.py:L238-L246 handle_fault/get_status
# （fault-tolerance 面）与 L252-L281 init_weight_transfer_engine/
# start_weight_update/update_weights 等 weight-transfer 尾段（RL 训练面）
# ——非本章服务面主线（m18 家族亦不展开），且依赖独立模块树。
