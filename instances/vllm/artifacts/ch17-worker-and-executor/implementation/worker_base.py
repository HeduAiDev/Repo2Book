# 只做减法的忠实精简版 —— 镜像 vllm/v1/worker/worker_base.py（pin f3fef123）。
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# 本章主线：WorkerWrapperBase 的延迟初始化（先记 rpc_rank，待 init_worker 时按
# parallel_config.worker_cls 字符串 qualname 解析真实 Worker 类再实例化）+ __getattr__ 透传，
# 这是 collective_rpc 能用 getattr(self.worker, method) 命中具体方法的根基。
#
# SUBTRACTED: 模块顶部 SPDX 版权头（vllm/v1/worker/worker_base.py:L1-L2）—— 仅许可证注释，不影响行为。
# SUBTRACTED: torch / torch.nn / VllmConfig / set_current_vllm_config / LoRARequest /
#   MULTIMODAL_REGISTRY / instrument / update_environment_variables / KVCacheSpec 等真实 import
#   （vllm/v1/worker/worker_base.py:L4-L17）—— 它们牵入 CUDA/torch/vllm 配置体系；本精简版用
#   serial_utils.resolve_obj_by_qualname（真实同名工具的最小镜像）即可演示『字符串类名→类对象』解析。

from collections.abc import Callable
from typing import Any, TypeVar

from serial_utils import resolve_obj_by_qualname

_R = TypeVar("_R")


# SOURCE: vllm/v1/worker/worker_base.py:L38-L177
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    # SOURCE: vllm/v1/worker/worker_base.py:L44-L88
    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """Initialize common worker components."""
        self.vllm_config = vllm_config
        # SUBTRACTED: 把 model_config/cache_config/lora_config/load_config/parallel_config/
        #   scheduler_config/device_config/speculative_config/observability_config/
        #   kv_transfer_config/compilation_config 逐个从 vllm_config 摊平到 self 的样板赋值
        #   （vllm/v1/worker/worker_base.py:L64-L74）—— 纯字段搬运，与本章控制流无关。
        self.parallel_config = vllm_config.parallel_config
        # SUBTRACTED: current_platform 绑定（vllm/v1/worker/worker_base.py:L76-L78）—— 平台/CUDA 探测。
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        # SUBTRACTED: self.device: torch.device | None；self.model_runner: nn.Module | None
        #   （vllm/v1/worker/worker_base.py:L87-L88）—— torch 类型，留作锚点。
        self.device = None
        self.model_runner = None

    # SUBTRACTED: get_kv_cache_spec / compile_or_warm_up_model / reset_mm_cache / get_model /
    #   apply_model / get_model_inspection / load_model / sample_tokens /
    #   get_cache_block_size_bytes / add_lora / remove_lora / pin_lora / list_loras /
    #   vocab_size 等纯虚接口方法（vllm/v1/worker/worker_base.py:L90-L172）—— 它们是各硬件后端
    #   要实现的契约清单；本章只需保留 init_device / execute_model / check_health / load_model /
    #   shutdown 这条生命周期主线上的纯虚声明示范。

    # SOURCE: vllm/v1/worker/worker_base.py:L102-L104
    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    # SOURCE: vllm/v1/worker/worker_base.py:L106-L110
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L130-L132
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L134-L143
    def execute_model(self, scheduler_output):
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L174-L176
    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


# SOURCE: vllm/v1/worker/worker_base.py:L179-L345
class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    # SOURCE: vllm/v1/worker/worker_base.py:L188-L208
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
        """
        self.rpc_rank: int = rpc_rank
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config = None

    # SOURCE: vllm/v1/worker/worker_base.py:L210-L212
    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    # SOURCE: vllm/v1/worker/worker_base.py:L222-L305
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        # SUBTRACTED: vllm_config.enable_trace_function_call_for_thread() 与
        #   load_general_plugins()（vllm/v1/worker/worker_base.py:L235-L239）—— 调试 trace / 插件
        #   加载副作用，与『按字符串解析 Worker 类』的控制流正交。

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

        # SUBTRACTED: worker_extension_cls 动态注入（把扩展类塞进 worker_class.__bases__ 以扩展
        #   collective_rpc 可调方法，vllm/v1/worker/worker_base.py:L253-L279）—— RLHF/外部插件扩展点，
        #   默认 worker_extension_cls 为空时整段不执行，删去不影响默认控制流。
        # SUBTRACTED: shared_worker_lock / mm_receiver_cache(shm 多模态缓存) 装配
        #   （vllm/v1/worker/worker_base.py:L281-L301）—— 多模态主题，本章 worker 无多模态接收缓存。

        kwargs.pop("shared_worker_lock", None)
        # SUBTRACTED: with set_current_vllm_config(self.vllm_config):  上下文管理器
        #   （vllm/v1/worker/worker_base.py:L303）—— 把当前 config 设进线程局部，便于 worker 构造期
        #   读取全局 config；本精简版 worker 直接从 kwargs 拿 vllm_config，无需该上下文。
        self.worker = worker_class(**kwargs)

    # SUBTRACTED: initialize_from_config（vllm/v1/worker/worker_base.py:L307-L311）—— 同模式透传
    #   KV cache 配置；KV 内存属后续章节，本章不展开。

    # SOURCE: vllm/v1/worker/worker_base.py:L313-L317
    def init_device(self):
        assert self.vllm_config is not None
        # SUBTRACTED: with set_current_vllm_config(self.vllm_config):（vllm/v1/worker/worker_base.py:L315）
        #   —— 同上，config 上下文管理器。
        self.worker.init_device()

    # SOURCE: vllm/v1/worker/worker_base.py:L319-L320
    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    # SUBTRACTED: _apply_mm_cache（vllm/v1/worker/worker_base.py:L322-L330）—— 多模态特征缓存改写，
    #   本章 worker 无多模态。

    # SOURCE: vllm/v1/worker/worker_base.py:L332-L337
    def execute_model(self, scheduler_output):
        # SUBTRACTED: self._apply_mm_cache(scheduler_output)（vllm/v1/worker/worker_base.py:L335）—— 同上。
        return self.worker.execute_model(scheduler_output)

    # SUBTRACTED: reset_mm_cache（vllm/v1/worker/worker_base.py:L339-L344）—— 多模态缓存清理。
