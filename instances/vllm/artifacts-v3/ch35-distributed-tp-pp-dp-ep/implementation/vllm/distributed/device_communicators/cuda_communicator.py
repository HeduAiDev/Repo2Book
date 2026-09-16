# SOURCE: vllm/distributed/device_communicators/cuda_communicator.py
# ch34 切面（m6）：CudaCommunicator.all_reduce 七级回退链 + all_gatherv/reduce_
# scatterv（EP 的 AgRs 积木）+ dispatch/combine 转发到 all2all_manager。
# 删除（dossier 删除项 2）：_log_all_reduce_backend_selection（L206-L273，日志拼装）、
# mori/deepep_v2/nixl/flashinfer 的 all2all import 分支（L146-L201 只留 AgRs 与
# 『其余后端』注释）、custom_all_gather/custom_reduce_scatter（L343-L356）、
# symm-mem 的 _get_symm_scratch/_reduce_scatter_symm_mem/_all_gather_symm_mem/
# _all_gather_batched_symm_mem 家族（L442-L706，注为省略）、checkpoint_prepare/
# restore（L588-L602，flashinfer workspace 域）。全部保留行对 pin 现核。

from __future__ import annotations

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.all_reduce_utils import (
    should_nccl_symm_mem_ag_rs,
    should_nccl_symm_mem_allreduce,
)
from vllm.distributed.device_communicators.pynccl import register_nccl_symmetric_ops
from vllm.distributed.device_communicators.pynccl_allocator import (
    is_symmetric_memory_enabled,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

# SUBTRACTED: from vllm.distributed.device_communicators.all_reduce_utils import
#   NCCL_SYMM_MEM_ALL_REDUCE_CONFIG（唯一消费者 _log_all_reduce_backend_selection
#   已按删除项 2 裁除）。
# SUBTRACTED: from vllm._aiter_ops import rocm_aiter_ops（L9）——ROCm aiter 使能面，
#   其唯一消费（use_aiter_allreduce 判定）由 seam env 表达；宿主非 ROCm。
# SUBTRACTED: from ..utils import StatelessProcessGroup（L22）——类型面（弹性 EP
#   的 tcp_store_group 参数）；seam 下无 stateless 构造路径。
# SUBTRACTED: from .aiter_custom_all_reduce import AiterCustomAllreduce（L23）——
#   顶层 import 在真实文件仅为类型注解服务；构造点在 __init__ 内（保留）。

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


# SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L29 CudaCommunicator
class CudaCommunicator(DeviceCommunicatorBase):
    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L30-L207
    #   __init__ —— 逐字 minus 删除项（all2all 后端家族分支 L146-L201 收窄为 AgRs）
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group=None,
        use_all2all: bool = False,
    ):
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L30-L207（锚点双置）
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all=use_all2all,
        )
        if "tp" not in unique_name:
            # custom allreduce or torch symm mem can be used only by tp
            use_custom_allreduce = False
            use_torch_symm_mem = False
            use_flashinfer_allreduce = False
            use_aiter_allreduce = False
        else:
            from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
            use_torch_symm_mem = envs.VLLM_ALLREDUCE_USE_SYMM_MEM
            use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER
            # SUBTRACTED: use_aiter_allreduce 的 rocm_aiter_ops 判定（L62-L64）
            #   ——ROCm 域；宿主/精简版恒 False。
            use_aiter_allreduce = False

        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem = use_torch_symm_mem
        self.use_flashinfer_allreduce = use_flashinfer_allreduce
        self.use_aiter_allreduce = use_aiter_allreduce

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )
        from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group if tcp_store_group is None else tcp_store_group,
                device=self.device,
            )
            if is_symmetric_memory_enabled():
                register_nccl_symmetric_ops(self.pynccl_comm)

        self.ca_comm: CustomAllreduce | None = None
        self.qr_comm: QuickAllReduce | None = None
        self.symm_mem_comm: SymmMemCommunicator | None = None
        self.fi_ar_comm: FlashInferAllReduce | None = None
        self.aiter_ar_comm = None

        if use_torch_symm_mem and current_platform.is_cuda():
            self.symm_mem_comm = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        if self.use_flashinfer_allreduce and self.world_size > 1:
            self.fi_ar_comm = FlashInferAllReduce(
                group=self.cpu_group,
                device=self.device,
            )

        # SUBTRACTED: use_aiter_allreduce 的 AiterCustomAllreduce 构造（L111-L115）
        #   ——ROCm 域（aiter_ar_comm 恒 None）。

        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                symm_mem_enabled=(
                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled
                ),
            )

        if use_custom_allreduce and self.world_size > 1 and current_platform.is_rocm():
            # Initialize a custom quick all-reduce implementation for AMD.
            # Quick reduce is designed as a complement to custom allreduce
            # (vLLM's or AITER's), so it is initialized for either backend.
            # Based on quickreduce (https://github.com/mk1-project/quickreduce).
            # On ROCm, 'use_custom_allreduce==True' means it must currently be
            # an MI300 series.
            self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)

        # SUBTRACTED: _log_all_reduce_backend_selection()（L136-L137）——日志拼装
        #   （删除项 2）。

        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            # SUBTRACTED: 其余 all2all 后端分支（L146-L201：deepep_high_throughput /
            #   deepep_low_latency / mori_* / deepep_v2 / nixl_ep / flashinfer_*
            #   双侧）——删除项 2。真 A2A 后端实现同一 dispatch/combine 接口，
            #   正文以『替换实现』一句带过；Unknown 后端的 ValueError 防线保留：
            else:
                raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

            logger.info_once(
                "Using %s all2all manager.",
                self.all2all_manager.__class__.__name__,
            )

    # SUBTRACTED: _log_all_reduce_backend_selection（L209-L273）——删除项 2。

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L275-L341
    #   all_reduce —— 七级回退链逐字
    def all_reduce(self, input_):
        # since currently we perform copy input -> symm_input -> out-of-place AR
        # return symm_output, we don't need to check if input is symmetric
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L275-L341（锚点双置）
        if self.pynccl_comm is not None and should_nccl_symm_mem_allreduce(
            self.pynccl_comm.world_size, input_
        ):
            out = torch.ops.vllm.all_reduce_symmetric_with_copy(input_)
            if out is not None:
                return out
        # always try quick reduce first, then flashinfer, then the AITER or vLLM
        # custom allreduce, and then pynccl. (quick reduce just for ROCM MI3*)
        qr_comm = self.qr_comm
        if (
            qr_comm is not None
            and not qr_comm.disabled
            and qr_comm.should_quick_allreduce(input_)
        ):
            out = qr_comm.quick_all_reduce(input_)
            assert out is not None
            return out
        fi_ar_comm = self.fi_ar_comm
        if (
            fi_ar_comm is not None
            and not fi_ar_comm.disabled
            and fi_ar_comm.should_use_fi_ar(input_)
        ):
            out = fi_ar_comm.all_reduce(input_)
            assert out is not None
            return out
        aiter_ar_comm = self.aiter_ar_comm
        if (
            aiter_ar_comm is not None
            and not aiter_ar_comm.disabled
            and aiter_ar_comm.should_custom_ar(input_)
        ):
            out = aiter_ar_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_)
        ):
            out = ca_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        symm_mem_comm = self.symm_mem_comm
        if symm_mem_comm is not None and symm_mem_comm.should_use_symm_mem(input_):
            out = symm_mem_comm.all_reduce(input_)
            assert out is not None
            return out
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
            return out
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
        return out

    # SUBTRACTED: custom_all_gather / custom_reduce_scatter（L343-L356）——删除项 2
    #   （CustomAllreduce 的可选 gather/scatter 加速）。

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L355-L389
    #   all_gather —— 逐字 minus symm-mem 分支的方法体（删除项 2 注为省略：
    #   dim==0 ∧ symm-mem 使能 → _all_gather_symm_mem 家族已裁，host 上谓词恒
    #   False 不可达）
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Route uniform dim-0 all-gathers through NVLS symmetric memory when
        # enabled (mirrors reduce_scatter); otherwise fall back to the
        # PyNccl/base-class all-gather. Sequence parallelism's
        # gather-before-GEMM uses dim=0 with tp-aligned (uniform) shards.
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L355-L389（锚点双置）
        if dim < 0:
            dim += input_.dim()
        # SUBTRACTED: if dim == 0 and should_nccl_symm_mem_ag_rs():
        #     return self._all_gather_symm_mem(input_.contiguous())（L366-L367）
        #   ——symm-mem 路径注为省略（删除项 2）；谓词 seam 恒 False。

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            return super().all_gather(input_, dim)

        # On ROCm, the base-class all_gather (all_gather_into_tensor) is faster
        # than the manual pynccl + torch.empty + movedim + reshape path below,
        # which adds a per-call output allocation and (for dim != 0) an extra
        # copy on every step. This is on the hot path for TP forward passes, so
        # keep ROCm on the base-class collective to avoid a decode regression.
        if current_platform.is_rocm():
            return super().all_gather(input_, dim)

        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        pynccl_comm.all_gather(output_tensor, input_.contiguous())
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L391-L416
    #   reduce_scatter —— 逐字 minus symm-mem 分支的方法体（同上注为省略）
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L391-L416（锚点双置）
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        # SUBTRACTED: if should_nccl_symm_mem_ag_rs():
        #     output = self._reduce_scatter_symm_mem(input_tensor)（L410-L411）
        #   ——symm-mem 路径注为省略（删除项 2）；谓词 seam 恒 False。
        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L418-L454
    #   reduce_scatterv —— 逐字 minus symm-mem 判定行（use_symm_mem 恒 False）
    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L418-L457（锚点双置）
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size, f"{len(sizes)} == {world_size}"
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        # SUBTRACTED: use_symm_mem 分支（L441-L443：
        #   use_symm_mem = sizes is None and should_nccl_symm_mem_ag_rs() →
        #   _reduce_scatter_symm_mem）——删除项 2 注为省略。
        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            pynccl_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
        else:
            pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    # SUBTRACTED: _get_symm_scratch / _reduce_scatter_symm_mem（L456-L517）——
    #   NCCL symmetric-memory scratch 家族（删除项 2 注为省略）。

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L560-L570
    #   broadcast —— 逐字
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L560-L570（锚点双置）
        if self.world_size == 1:
            return tensor

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.broadcast(tensor, src)
            return tensor
        else:
            raise ValueError("No PyNCCL communicator found")

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L572-L586
    #   destroy —— 逐字
    def destroy(self):
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L572-L586（锚点双置）
        if self.pynccl_comm is not None:
            self.pynccl_comm.destroy()
            self.pynccl_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.aiter_ar_comm is not None:
            self.aiter_ar_comm.close()
            self.aiter_ar_comm = None
        if self.fi_ar_comm is not None:
            self.fi_ar_comm.destroy()
            self.fi_ar_comm = None
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None  # type: ignore[assignment]

    # SUBTRACTED: checkpoint_prepare / checkpoint_restore（L584-L602）——flashinfer
    #   workspace 域（随删除项 2 的后端分支裁除）。

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L604-L668
    #   all_gatherv —— 逐字 minus symm-mem 分支（同上注为省略）
    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L604-L658（锚点双置）
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None and not pynccl_comm.disabled

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        # SUBTRACTED: if sizes is None and should_nccl_symm_mem_ag_rs():
        #     return self._all_gather_symm_mem / _all_gather_batched_symm_mem
        #   （L626-L630）——symm-mem 路径注为省略（删除项 2）。

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L629-L647（锚点双置）
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )
            if sizes is not None:
                pynccl_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pynccl_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        pynccl_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pynccl_comm.group_end()

        return output_list

    # SUBTRACTED: _all_gather_symm_mem / _all_gather_batched_symm_mem（L670-L706）
    #   ——删除项 2 注为省略。

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L710-L769
    #   dispatch_router_logits / dispatch / combine —— 逐字（转发到 all2all_manager）
    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L710-L731（锚点双置）

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L733-L755 dispatch
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    # SOURCE: vllm/distributed/device_communicators/cuda_communicator.py:L757-L769 combine
    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )

    # SUBTRACTED: batch_isend_irecv（L771-L777）——NCCL 组批量 P2P 面（真 A2A 后端
    #   的 DeepEP 族消费；本章消费面之外，pynccl seam 的 send/recv/
    #   batch_isend_irecv 一并按 ch26 域裁除——基类版本保留 NotImplementedError
    #   兜底位）。
