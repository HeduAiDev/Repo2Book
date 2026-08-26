# Subtract-only companion for v3 ch17 — vllm/v1/executor/abstract.py (pin
# v0.27.1 / 6e448d0ea). Same names, same structure, same control flow; only
# dossier-approved deletions, each marked `# SUBTRACTED:`.
#
# Deletions here (dossier subtraction_plan.delete):
#   #1  ray 分支（RayExecutorV2 / RayDistributedExecutor 及 VLLM_USE_RAY_V2
#       判断，abstract.py:L60-L68）与 external_launcher 分支（L77-L80）与文末
#       兼容导入块（L371-L380）；
#   #3  init_kv_output_aggregator（L280-L284，KVOutputAggregator 属 PD 解耦）；
#   #4  @instrument 观测装饰（可观测性装饰项）。
# vllm-internal imports are package-relative (ch10 v3 convention); external
# deps outside this chapter mirror in .._host_seams.

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import Literal, TypeVar, overload

from .._host_seams import (
    GrammarOutput,
    KVConnectorHandshakeMetadata,
    KVCacheConfig,
    KVCacheSpec,
    LoRARequest,
    ReconfigureDistributedRequest,
    SchedulerOutput,
    SupportedTask,
    init_logger,
)
from ..utils.import_utils import resolve_obj_by_qualname
from ..worker.worker_base import CompilationTimes, WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


# SOURCE: vllm/v1/executor/abstract.py:L37-L42 Executor — 执行臂第一层
class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """

    # SOURCE: vllm/v1/executor/abstract.py:L44 uses_ray
    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    # SOURCE: vllm/v1/executor/abstract.py:L45 supports_pp
    supports_pp: bool = False  # whether the executor supports PP

    # SOURCE: vllm/v1/executor/abstract.py:L47-L92 get_class — 工厂分发
    @staticmethod
    def get_class(vllm_config) -> type["Executor"]:
        # SOURCE: vllm/v1/executor/abstract.py:L49-L51
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        # SOURCE: vllm/v1/executor/abstract.py:L53-L59 type 分支
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        # SUBTRACTED: ray 分支（abstract.py:L60-L68——RayExecutorV2 /
        #   RayDistributedExecutor 按 VLLM_USE_RAY_V2_EXECUTOR_BACKEND 二选一）
        #   ——分布式编排归 ch34；删除后 "ray" 落到自定义 qualname 分支。
        # SOURCE: vllm/v1/executor/abstract.py:L69-L72 mp 分支
        elif distributed_executor_backend == "mp":
            from implementation.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        # SOURCE: vllm/v1/executor/abstract.py:L73-L76 uni 分支
        elif distributed_executor_backend == "uni":
            from implementation.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        # SUBTRACTED: external_launcher 分支（abstract.py:L77-L80——torchrun
        #   离线特例，ExecutorWithExternalLauncher 全类随之删除）。
        # SOURCE: vllm/v1/executor/abstract.py:L81-L87 自定义 qualname 分支
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        # SOURCE: vllm/v1/executor/abstract.py:L88-L91
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    # SUBTRACTED: @instrument(span_name="Executor init") 观测装饰（可观测性
    #   装饰项——与 ch09 同款处理；行为不变）。
    # SOURCE: vllm/v1/executor/abstract.py:L95-L112 Executor.__init__
    def __init__(
        self,
        vllm_config,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self.kv_output_aggregator = None
    @abstractmethod
    # SOURCE: vllm/v1/executor/abstract.py:L114-L116 _init_executor 抽象钩子
    def _init_executor(self) -> None:
        raise NotImplementedError

    # SOURCE: vllm/v1/executor/abstract.py:L118-L120 initialize_from_config
    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Initialize the KV caches on the underlying workers."""
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))

    # SOURCE: vllm/v1/executor/abstract.py:L122-L137 compile_or_warm_up_model
    def compile_or_warm_up_model(self) -> None:
        """Compile/warm up the model and capture cudagraphs on workers."""
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        # Propagate compilation time from workers back to the main process.
        # With TP>1, compilation happens in worker processes, so the main
        # process config is never updated. Use max across workers since they
        # compile in parallel.
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )

    # SOURCE: vllm/v1/executor/abstract.py:L139-L144 register_failure_callback
    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    # SOURCE: vllm/v1/executor/abstract.py:L146-L147 determine_available_memory
    def determine_available_memory(self) -> list[int]:  # in bytes
        return self.collective_rpc("determine_available_memory")

    # SOURCE: vllm/v1/executor/abstract.py:L149-L150 get_kv_cache_specs
    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")
    @overload
    # SOURCE: vllm/v1/executor/abstract.py:L152-L185 collective_rpc 契约 docstring
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        pass
    @overload
    # SOURCE: vllm/v1/executor/abstract.py:L187-L196 collective_rpc non_block 重载
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        pass
    @abstractmethod
    # SOURCE: vllm/v1/executor/abstract.py:L198-L202 collective_rpc 抽象实现
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        raise NotImplementedError

    # SOURCE: vllm/v1/executor/abstract.py:L204-L207 get_kv_connector_handshake_metadata
    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[tuple[int, int], KVConnectorHandshakeMetadata]]:
        return self.collective_rpc("get_kv_connector_handshake_metadata")
    @overload
    # SOURCE: vllm/v1/executor/abstract.py:L209-L213 execute_model 重载 (blocking)
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> "ModelRunnerOutput | None":
        pass
    @overload
    # SOURCE: vllm/v1/executor/abstract.py:L215-L219 execute_model 重载 (non_block)
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future["ModelRunnerOutput | None"]:
        pass

    # SOURCE: vllm/v1/executor/abstract.py:L221-L227 execute_model 薄封装
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]
    @overload
    # SOURCE: vllm/v1/executor/abstract.py:L229-L233 sample_tokens 重载 (blocking)
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> "ModelRunnerOutput":
        pass
    @overload
    # SOURCE: vllm/v1/executor/abstract.py:L235-L239 sample_tokens 重载 (non_block)
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future["ModelRunnerOutput"]:
        pass

    # SOURCE: vllm/v1/executor/abstract.py:L241-L247 sample_tokens 薄封装
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> "ModelRunnerOutput | Future[ModelRunnerOutput]":
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    # SOURCE: vllm/v1/executor/abstract.py:L249-L250 execute_dummy_batch
    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    # SOURCE: vllm/v1/executor/abstract.py:L252-L254 take_draft_token_ids
    def take_draft_token_ids(self) -> "DraftTokenIds | None":
        output: list["DraftTokenIds"] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    # SOURCE: vllm/v1/executor/abstract.py:L256-L257 profile
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    # SOURCE: vllm/v1/executor/abstract.py:L259-L268 save_sharded_state
    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )
    @abstractmethod
    # SOURCE: vllm/v1/executor/abstract.py:L270-L274 check_health
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    # SOURCE: vllm/v1/executor/abstract.py:L276-L278 shutdown
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.collective_rpc("shutdown")

    # SUBTRACTED: init_kv_output_aggregator（abstract.py:L280-L284）——
    #   KVOutputAggregator 属 PD 解耦/KV 连接器聚合（ch16/ch36，删除项 3）；
    #   kv_output_aggregator 字段保留恒 None，mp 侧聚合分支同步删除。
    @cached_property  # Avoid unnecessary RPC calls
    # SOURCE: vllm/v1/executor/abstract.py:L286-L290 supported_tasks
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    # SOURCE: vllm/v1/executor/abstract.py:L292-L294 add_lora
    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    # SOURCE: vllm/v1/executor/abstract.py:L296-L298 remove_lora
    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    # SOURCE: vllm/v1/executor/abstract.py:L300-L302 pin_lora
    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    # SOURCE: vllm/v1/executor/abstract.py:L304-L308 list_loras
    def list_loras(self) -> set[int]:
        sets: list[set[int]] = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    # SOURCE: vllm/v1/executor/abstract.py:L310-L312 reset_mm_cache
    def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache in each worker."""
        self.collective_rpc("reset_mm_cache")

    # SOURCE: vllm/v1/executor/abstract.py:L314-L316 reset_encoder_cache
    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        self.collective_rpc("reset_encoder_cache")

    # SOURCE: vllm/v1/executor/abstract.py:L318-L329 sleep
    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    # SOURCE: vllm/v1/executor/abstract.py:L331-L356 wake_up
    def wake_up(self, tags: list[str] | None = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    # SOURCE: vllm/v1/executor/abstract.py:L358-L361 reinitialize_distributed
    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        raise NotImplementedError
    @classmethod
    # SOURCE: vllm/v1/executor/abstract.py:L363-L368 supports_async_scheduling
    def supports_async_scheduling(cls) -> bool:
        """
        Whether the executor supports async scheduling.
        """
        return False


# SUBTRACTED: 文末兼容导入块（abstract.py:L371-L380——为 external_launcher /
#   旧引用做 backwards-compat 的 UniProcExecutor / ExecutorWithExternalLauncher
#   再导出）——随删除项 1 一并裁除；本精简包内以包内相对导入直达。
