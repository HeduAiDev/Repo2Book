# 只做减法的忠实精简版 —— 镜像 vllm/v1/executor/abstract.py（pin f3fef123）。
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# 本章主线：
#   Executor.get_class()  —— 工厂三态分发（type / 'mp' / 'uni' / 自定义 qualname）；
#   Executor.__init__ → _init_executor() 抽象钩子（子类拉起 worker）；
#   collective_rpc 抽象契约 + execute_model/sample_tokens 等薄封装（所有指令统一走 collective_rpc）。
#
# SUBTRACTED: 模块顶部 SPDX 版权头（vllm/v1/executor/abstract.py:L1-L2）。
# SUBTRACTED: time / cached_property / KVOutputAggregator / KVConnector* / LoRARequest /
#   SupportedTask / instrument / VllmConfig / 各 v1 类型的真实 import
#   （vllm/v1/executor/abstract.py:L3-L28）—— 牵入完整 vllm 配置/KV/LoRA 体系；本章只示范分发与
#   collective_rpc 薄封装的控制流，相关方法体里这些类型仅作占位。

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING

from serial_utils import resolve_obj_by_qualname

FailureCallback = Callable[[], None]


# SOURCE: vllm/v1/executor/abstract.py:L37-L372
class Executor(ABC):
    """Abstract base class for vLLM executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    supports_pp: bool = False  # whether the executor supports PP

    @staticmethod
    # SOURCE: vllm/v1/executor/abstract.py:L47-L92
    def get_class(vllm_config) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        # SUBTRACTED: elif distributed_executor_backend == "ray": 两条 ray 实现分支
        #   （RayExecutorV2 / RayDistributedExecutor，受 VLLM_USE_RAY_V2_EXECUTOR_BACKEND 控制，
        #   vllm/v1/executor/abstract.py:L60-L68）—— ray 是另一套编排，超出本章 mp/uni 范围；
        #   工厂保留下面 'mp'/'uni'/自定义三条分支已足以示范字符串后端名→Executor 子类的映射。
        elif distributed_executor_backend == "mp":
            from multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            from uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        # SUBTRACTED: elif distributed_executor_backend == "external_launcher":
        #   ExecutorWithExternalLauncher（vllm/v1/executor/abstract.py:L77-L80）—— torchrun 多引擎
        #   离线推理特例，本章不展开。
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    # SUBTRACTED: @instrument(span_name="Executor init") 装饰器（vllm/v1/executor/abstract.py:L94）
    #   —— tracing 埋点，与控制流无关。
    # SOURCE: vllm/v1/executor/abstract.py:L94-L112
    def __init__(
        self,
        vllm_config,
    ) -> None:
        self.vllm_config = vllm_config
        # SUBTRACTED: 把 model_config/cache_config/lora_config/load_config/parallel_config/
        #   scheduler_config/device_config/speculative_config/observability_config 逐个摊平到 self
        #   的样板赋值（vllm/v1/executor/abstract.py:L100-L108）—— 纯字段搬运。本章子类用到的
        #   parallel_config / scheduler_config 在此显式保留。
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        # SUBTRACTED: self.kv_output_aggregator: KVOutputAggregator | None = None
        #   （vllm/v1/executor/abstract.py:L112）—— PD 解耦/KV 连接器聚合，属 ch15/16 主题。

    @abstractmethod
    # SOURCE: vllm/v1/executor/abstract.py:L114-L116
    def _init_executor(self) -> None:
        raise NotImplementedError

    # SUBTRACTED: initialize_from_config（vllm/v1/executor/abstract.py:L118-L137）—— KV cache 初始化 +
    #   编译预热的 collective_rpc 封装；KV/编译属后续章节。

    # SOURCE: vllm/v1/executor/abstract.py:L139-L144
    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    # SOURCE: vllm/v1/executor/abstract.py:L146-L147
    def determine_available_memory(self) -> list[int]:  # in bytes
        return self.collective_rpc("determine_available_memory")

    # SUBTRACTED: get_kv_cache_specs / get_kv_connector_handshake_metadata（同模式 collective_rpc
    #   封装，vllm/v1/executor/abstract.py:L149-L150,L204-L207）—— KV 主题，本章保留
    #   determine_available_memory 一例示范『一切引擎指令都是 collective_rpc 薄封装』即可。
    # SUBTRACTED: collective_rpc 的两个 @overload 类型签名（non_block Literal[False]/[True] →
    #   list/Future，vllm/v1/executor/abstract.py:L152-L196）—— 纯类型层精确化，运行期无效果；
    #   返回类型随 non_block 变化一句话即可，不必抄两段重复签名。

    @abstractmethod
    # SOURCE: vllm/v1/executor/abstract.py:L198-L202
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        """Execute an RPC call on all workers.

        method: worker 方法名 str，或被序列化后发给所有 worker 执行的 callable
                （callable 额外接收 worker 自身作为首个 self 参数）。
        timeout: 秒；超时抛 TimeoutError；None 表示无限等。
        args/kwargs: 透传给 worker 方法。
        non_block: True 时返回 Future（列表），不阻塞。
        返回: 每个 worker 结果组成的 list。
        建议仅用于传控制消息，数据平面另设通道。
        """
        raise NotImplementedError

    # SUBTRACTED: execute_model / sample_tokens 各自的两个 @overload 签名
    #   （vllm/v1/executor/abstract.py:L209-L219,L229-L239）—— 同上，纯类型层。
    # SOURCE: vllm/v1/executor/abstract.py:L221-L227
    def execute_model(self, scheduler_output, non_block: bool = False):
        output = self.collective_rpc(
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    # SOURCE: vllm/v1/executor/abstract.py:L241-L247
    def sample_tokens(self, grammar_output, non_block: bool = False):
        output = self.collective_rpc(
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    # SUBTRACTED: execute_dummy_batch / take_draft_token_ids / max_concurrent_batches / profile /
    #   save_sharded_state / init_kv_output_aggregator / supported_tasks / add_lora / remove_lora /
    #   pin_lora / list_loras / reset_mm_cache / reset_encoder_cache / sleep / wake_up /
    #   reinitialize_distributed / supports_async_scheduling
    #   （vllm/v1/executor/abstract.py:L249-L372）—— 全是同模式的 collective_rpc 薄封装或特性开关；
    #   正文已用 execute_model/sample_tokens/determine_available_memory/shutdown 示范该模式，
    #   不必逐个抄。check_health / shutdown 这两个生命周期相关的保留如下。

    @abstractmethod
    # SOURCE: vllm/v1/executor/abstract.py:L274-L278
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    # SOURCE: vllm/v1/executor/abstract.py:L280-L282
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.collective_rpc("shutdown")
