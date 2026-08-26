# Subtract-only companion for v3 ch17 — vllm/v1/worker/worker_base.py (pin
# v0.27.1 / 6e448d0ea). Same names, same structure, same control flow; only
# dossier-approved deletions, each marked `# SUBTRACTED:`.
#
# Deletions here (dossier subtraction_plan.delete):
#   #4  @instrument 观测装饰（init_worker 上）；
#   #8  worker_extension_cls 动态注入体（L261-L287）、mm_receiver_cache 的
#       构建分支（L309-L315——只留 shared_worker_lock 缺失时的 warning 早退）。

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import torch
import torch.nn as nn

from .._host_seams import (
    VllmConfig,
    init_logger,
    ir,
    load_general_plugins,
    set_current_vllm_config,
)
from ..utils.import_utils import resolve_obj_by_qualname
from ..utils.system_utils import update_environment_variables

if TYPE_CHECKING:
    from .._host_seams import GrammarOutput, SchedulerOutput
    from .._host_seams import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


# SOURCE: vllm/v1/worker/worker_base.py:L34-L36 CompilationTimes
class CompilationTimes(NamedTuple):
    language_model: float
    encoder: float


# SOURCE: vllm/v1/worker/worker_base.py:L39-L43 WorkerBase — 执行臂第二层契约
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    # SOURCE: vllm/v1/worker/worker_base.py:L45-L96 WorkerBase.__init__
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
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
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from .._host_seams import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

        # IR op priority and torch-wrap state are constant for the worker's
        # lifetime.
        vllm_config.kernel_config.ir_op_priority.set_default()
        ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )

    # SOURCE: vllm/v1/worker/worker_base.py:L98-L100 get_kv_cache_spec
    def get_kv_cache_spec(self) -> dict:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L102-L108 compile_or_warm_up_model
    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Prepare model for execution through compilation/warmup.

        Returns:
            Compilation times (language_model, encoder) in seconds.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L110-L112 check_health
    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    # SOURCE: vllm/v1/worker/worker_base.py:L114-L118 init_device
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L120-L123 reset_mm_cache
    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    # SOURCE: vllm/v1/worker/worker_base.py:L125-L126 get_model
    def get_model(self) -> nn.Module:
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L128-L130 apply_model
    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    # SOURCE: vllm/v1/worker/worker_base.py:L132-L136 get_model_inspection
    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    # SOURCE: vllm/v1/worker/worker_base.py:L138-L140 load_model
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L142-L151 execute_model 两段式契约
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L153-L157 sample_tokens 契约
    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L159-L163 get_cache_block_size_bytes
    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L165-L166 add_lora
    def add_lora(self, lora_request) -> bool:
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L168-L169 remove_lora
    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L171-L172 pin_lora
    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L174-L175 list_loras
    def list_loras(self) -> set[int]:
        raise NotImplementedError
    @property
    # SOURCE: vllm/v1/worker/worker_base.py:L177-L180 vocab_size
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    # SOURCE: vllm/v1/worker/worker_base.py:L182-L184 shutdown
    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


# SOURCE: vllm/v1/worker/worker_base.py:L187-L194 WorkerWrapperBase — 延迟初始化
class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    # SOURCE: vllm/v1/worker/worker_base.py:L196-L216 WorkerWrapperBase.__init__
    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank: int = rpc_rank
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    # SOURCE: vllm/v1/worker/worker_base.py:L218-L220 WorkerWrapperBase.shutdown
    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    # SOURCE: vllm/v1/worker/worker_base.py:L222-L227 update_environment_variables
    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    # SUBTRACTED: @instrument(span_name="Worker init") 观测装饰（删除项 4）。
    # SOURCE: vllm/v1/worker/worker_base.py:L229-L319 init_worker — 延迟初始化点
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        vllm_config.enable_trace_function_call_for_thread()

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        # SUBTRACTED: worker_extension_cls 动态注入（worker_base.py:L261-L287——
        #   把扩展类塞进 worker_class.__bases__ 以扩展 collective_rpc 可调面，
        #   OOT/RL 插件扩展点，删除项 8）。

        # SUBTRACTED: assigned_physical_gpu_ids 透传（L289-L293——逻辑到物理
        #   卡号映射，删除项 4 平台适配轴）。
        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        # SUBTRACTED: mm_receiver_cache 构建分支（worker_base.py:L309-L315——
        #   MULTIMODAL_REGISTRY.worker_receiver_cache_from_config；删除项 8，
        #   _apply_mm_cache 在 mm_receiver_cache=None 时为 no-op，控制流不变
        #   ——构建位退化为 None 赋值（多模态 shm 缓存不入本档）。
        self.mm_receiver_cache = None

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)

    # SOURCE: vllm/v1/worker/worker_base.py:L321-L325 initialize_from_config
    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    # SOURCE: vllm/v1/worker/worker_base.py:L327-L331 init_device
    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    # SOURCE: vllm/v1/worker/worker_base.py:L333-L334 __getattr__ 透传
    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    # SOURCE: vllm/v1/worker/worker_base.py:L336-L344 _apply_mm_cache
    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    # SOURCE: vllm/v1/worker/worker_base.py:L346-L351 WorkerWrapperBase.execute_model
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    # SOURCE: vllm/v1/worker/worker_base.py:L353-L358 reset_mm_cache
    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
