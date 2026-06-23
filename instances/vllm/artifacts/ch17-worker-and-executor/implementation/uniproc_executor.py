# 只做减法的忠实精简版 —— 镜像 vllm/v1/executor/uniproc_executor.py（pin f3fef123）。
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# 本章作用：最简对照。uni 后端没有子进程，driver_worker 就在本进程，collective_rpc 退化为
# 同进程 run_method 直调——读者先理解『一个 worker 时控制平面是什么』，再看 mp 如何复杂化。
#
# SUBTRACTED: 模块顶部 SPDX 版权头（vllm/v1/executor/uniproc_executor.py:L1-L2）。
# SUBTRACTED: os / torch / torch.distributed / current_platform / get_distributed_init_method /
#   get_ip / get_open_port / AsyncModelRunnerOutput 等真实 import
#   （vllm/v1/executor/uniproc_executor.py:L3-L21 部分）—— 牵入 CUDA/分布式/网络；本精简版
#   _distributed_args 返回占位三元组即可，无需真实端口/IP。

from collections.abc import Callable
from concurrent.futures import Future
from multiprocessing import Lock
from typing import Any

from abstract import Executor
from serial_utils import run_method
from worker_base import WorkerWrapperBase


# SOURCE: vllm/v1/executor/uniproc_executor.py:L26-L141
class UniProcExecutor(Executor):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L27-L53
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        # SUBTRACTED: async_output_thread / max_concurrent_batches>1 的 ThreadPoolExecutor 装配
        #   （vllm/v1/executor/uniproc_executor.py:L40-L44）—— 异步调度输出线程，与『uni 是 mp 的
        #   最简退化』无关，默认同步调度路径不创建该线程。

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        # SUBTRACTED: VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 分支（弹性专家并行扩容时走 elastic_ep_execute，
        #   vllm/v1/executor/uniproc_executor.py:L49-L52）—— EP 特性开关，默认走 load_model。
        self.driver_worker.load_model()
        # SUBTRACTED: current_platform.update_block_size_for_backend(self.vllm_config)
        #   （vllm/v1/executor/uniproc_executor.py:L53）—— 平台相关 block size 调整。

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L55-L61
    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        # SUBTRACTED: 真实实现用 get_distributed_init_method(get_ip(), get_open_port()) 取真实端口，
        #   并按 device_config 解析 local_rank（vllm/v1/executor/uniproc_executor.py:L57-L61）——
        #   牵入网络/设备；单 worker 演示用占位 ("local", 0, 0) 即可。
        return "local", 0, 0

    # SUBTRACTED: max_concurrent_batches（vllm/v1/executor/uniproc_executor.py:L63-L65）—— 异步调度
    #   并发批数，本章不展开异步调度流水。

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L67-L100
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        # SUBTRACTED: non_block 分支里 AsyncModelRunnerOutput.get_output() 的异步线程池处理
        #   （vllm/v1/executor/uniproc_executor.py:L83-L94）—— 异步调度输出搬运；这里保留
        #   『同步算出结果 → 包进一个已完成 Future』这条核心退化路径。
        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    # SUBTRACTED: uni 版 execute_model / sample_tokens / take_draft_token_ids 的 single_value 重写
    #   （vllm/v1/executor/uniproc_executor.py:L102-L128）—— 与基类同语义，仅多传 single_value=True；
    #   本章用基类 execute_model（走 collective_rpc 默认 list 返回）即可对照 mp。

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L130-L133
    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L135-L137
    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()

# SUBTRACTED: class ExecutorWithExternalLauncher(UniProcExecutor)（vllm/v1/executor/uniproc_executor.py
#   :L144-L190）—— torchrun 兼容启动器的多引擎 TP 离线推理特例，本章不展开。
