# SOURCE: vllm/v1/worker/gpu_worker.py
# ch34 切面（站 5/m9/m10/m12）：AsyncIntermediateTensors（懒同步包装）+ Worker 的
# PP 接力段（execute_model：收割上拍 isend → irecv → 转调 → isend）+
# init_worker_distributed_environment（worker 入全局 world）。
# 删除（dossier 删除项 8）：execute_model 的 enable_sp 预备块（L1035-L1062）与
# use_v2_model_runner pooling 分支（L1082-L1087）。Worker 的其余生产特性面
# （sleep/权重迁移/profiler/弹性 EP 执行器）归 ch17 域收窄——行内标注。

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from types import NoneType

import torch

from vllm.distributed import Handle
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput, AsyncModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput

# SUBTRACTED: 其余 import（L23-L86：ec/kv transfer/weight transfer/sleep/
#   startup_plan/warmup/elastic_ep）——各归其域（ch16/ch17/ch39）。

logger = init_logger(__name__)


# SOURCE: vllm/v1/worker/gpu_worker.py:L96-L125 AsyncIntermediateTensors — 逐字
#   （__getattribute__ 钩子：谁先碰 .tensors 谁负责 wait_for_comm）
class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""
    # SOURCE: vllm/v1/worker/gpu_worker.py:L96-L125（锚点双置）

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        # SOURCE: vllm/v1/worker/gpu_worker.py:L99-L108（锚点双置）
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def wait_for_comm(self) -> None:
        # SOURCE: vllm/v1/worker/gpu_worker.py:L110-L119（锚点双置）
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        # SOURCE: vllm/v1/worker/gpu_worker.py:L121-L125（锚点双置）
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


# SOURCE: vllm/v1/worker/gpu_worker.py:L128-L1389 Worker —— 协作件装配子集
#   （真实基类 WorkerBase 归 ch17 域，本类以无基类承载；sleep/权重迁移/profiler/
#   弹性 EP 执行器/init_device 的 GPU 识别段按 ch17 域收窄——行内标注）。
#   _pp_send_work 的初始化位 L175-L176 逐字保留。
class Worker:
    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # SOURCE: vllm/v1/worker/gpu_worker.py:L129-L179（锚点双置）
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        # SUBTRACTED: 生产特性面（L145-L179：精度设置/弹性 EP 执行器/FT 哨兵/
        #   sleep 缓冲/权重迁移引擎/profiler/use_v2_model_runner）——ch17 域。

        # pending non-blocking PP send work from the previous iteration
        self._pp_send_work: list[Handle] = []

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1019-L1107 execute_model —— 逐字
    #   minus 删除项 8（enable_sp 预备块 L1035-L1062 与 pooling 分支
    #   L1082-L1087）及 @with_gpu_sync_check 装饰器（L1018，ch17 域的异步 CUDA
    #   错误检查面）；PP 接力三段（收割→irecv→转调→isend）即 ch17 站 9 的展开
    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ensure any previous non-blocking PP sends are complete
        # SOURCE: vllm/v1/worker/gpu_worker.py:L1019-L1107（锚点双置）
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        all_gather_tensors = {}
        parallel_config = self.vllm_config.parallel_config

        # SUBTRACTED: enable_sp 序列并行预备块（L1026-L1033 的 num_scheduled_
        #   tokens/compilation_config 预备行 + L1035-L1062 的 SP 判定块）——删除
        #   项 8（SP 属进阶组合，m23 一句带过）。

        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        # SUBTRACTED: annotate_profile 观测上下文（L1078）——ch17 域。
        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors
        )
        if isinstance(
            output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
        ):
            return output
        # SUBTRACTED: use_v2_model_runner 的 pooling 分支（L1082-L1087）——删除项 8。

        assert isinstance(output, IntermediateTensors)
        # SUBTRACTED: parallel_config 的二次绑定行（L1094，顶部绑定已承载同值）。
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # launch non-blocking send of intermediate tensors
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None


# SOURCE: vllm/v1/worker/gpu_worker.py:L1347-L1389 init_worker_distributed_
#   environment —— 逐字 minus batch_invariance/eplb env 覆写（ch19/EPLB 域）；
#   ensure_model_parallel_initialized 的调用位即站 5 的『worker 入全局 world』出口
def init_worker_distributed_environment(
    vllm_config,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    # SOURCE: vllm/v1/worker/gpu_worker.py:L1347-L1389（锚点双置）
    parallel_config = vllm_config.parallel_config
    # SUBTRACTED: init_batch_invariance / override_envs_for_eplb（L1356-L1362）
    #   ——ch19/EPLB 域。
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"

    timeout = None
    if parallel_config.distributed_timeout_seconds is not None:
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        backend,
        timeout,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # SUBTRACTED: ensure_ec_transfer_initialized（L1387-L1389）——ch16 域。
