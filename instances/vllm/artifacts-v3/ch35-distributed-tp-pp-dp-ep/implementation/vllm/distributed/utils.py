# SOURCE: vllm/distributed/utils.py
# ch34 切面：无状态进程组构造族（stateless_init_torch_distributed_process_group /
# init_gloo_process_group / stateless_destroy…）+ 双群组 timeout 谓词 + TCPStore
# 客户端缓存 + split_tensor_along_last_dim（TP 切输入的工具函数）。
# SUBTRACTED：StatelessProcessGroup 大类（~L300-L520，基于 TCPStore 的对象/Tensor
# 通道——弹性 EP 域，删除项 1 同族）、libnode/自定义轮询/pickle monkeypatch 面。
# StatelessProcessGroup 以 HOST SEAM 类型位保留（cuda_communicator 的注解面）。

from __future__ import annotations

import functools
from collections.abc import Sequence
from datetime import timedelta

import torch
from torch.distributed import ProcessGroup, TCPStore, Store
from torch.distributed.distributed_c10d import (
    PrefixStore,
    _unregister_process_group,
)
from torch.distributed.rendezvous import rendezvous

from vllm.logger import init_logger
from vllm.utils import divide
from vllm.utils.network_utils import get_tcp_uri
from vllm.utils.system_utils import suppress_stdout

logger = init_logger(__name__)

# SUBTRACTED: os/pickle/sys/time/uuid/deque/dataclasses import 与 sched_yield 轮询
#   常量（L9-L42）——被裁面的依赖。


# SOURCE: vllm/distributed/utils.py:L99-L124 split_tensor_along_last_dim —— 逐字
def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


# SOURCE: vllm/distributed/utils.py:L199-L513 StatelessProcessGroup —— HOST SEAM
#   真实大类实现基于 TCPStore 的 rank 间对象/Tensor 通道（弹性 EP 域，删除项 1
#   同族裁除）；本章消费面只剩 cuda_communicator 的注解引用。
class StatelessProcessGroup:  # HOST SEAM
    # SOURCE: vllm/distributed/utils.py:L199-L513（锚点双置）
    pass


# SOURCE: vllm/distributed/utils.py:L517-L523 get_cached_tcp_store_client —— 逐字
@functools.lru_cache(maxsize=1)
def get_cached_tcp_store_client(host: str, port: int) -> TCPStore:
    """Return a cached TCPStore client.

    Cached so that every call with the same ``(host, port)`` reuses the
    same connection. A new ``(host, port)`` evicts the old entry.
    """
    # SOURCE: vllm/distributed/utils.py:L517-L523（锚点双置）
    return TCPStore(host, port, is_master=False, wait_for_workers=False)


# SOURCE: vllm/distributed/utils.py:L526-L533 get_cpu_distributed_timeout_or_none
#   —— 逐字
def get_cpu_distributed_timeout_or_none() -> timedelta | None:
    # SOURCE: vllm/distributed/utils.py:L526-L533（锚点双置）
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    timeout_seconds = vllm_config.parallel_config.cpu_distributed_timeout_seconds
    return timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None


# SOURCE: vllm/distributed/utils.py:L536-L543 get_distributed_timeout_or_none
#   —— 逐字
def get_distributed_timeout_or_none() -> timedelta | None:
    # SOURCE: vllm/distributed/utils.py:L536-L543（锚点双置）
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    timeout_seconds = vllm_config.parallel_config.distributed_timeout_seconds
    return timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None


# SOURCE: vllm/distributed/utils.py:L546-L573 init_gloo_process_group —— 逐字
def init_gloo_process_group(
    prefix_store: PrefixStore,
    group_rank: int,
    group_size: int,
    timeout: timedelta,
) -> ProcessGroup:
    """
    Stateless init ProcessGroup with gloo backend compatible with
    different torch versions.
    """
    with suppress_stdout():
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
        from torch.distributed.distributed_c10d import ProcessGroupGloo

        backend_class = ProcessGroupGloo(
            prefix_store, group_rank, group_size, timeout=timeout
        )
        backend_type = ProcessGroup.BackendType.GLOO
        device = torch.device("cpu")
        pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()

        pg._register_backend(device, backend_type, backend_class)
    return pg


# SOURCE: vllm/distributed/utils.py:L576-L684 stateless_init_torch_distributed_
#   process_group —— 逐字（docstring 全文保留）
def stateless_init_torch_distributed_process_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    backend: str,
    group_name: str | None = None,
    return_store: bool = False,
    listen_socket=None,
) -> ProcessGroup | tuple[ProcessGroup, Store]:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.

    This function is useful when we are not sure about the total number of
    processes in the process group. For example, we may have process
    1, 2, ..., 8 who want to communicate, and process 9 might be the same
    process as process 1, or it might be a different process; process 10
    might be the same process as process 5, or it might be a different
    process. In this case, how can we reliably form a communication channel
    within process 9 and 10, without affecting the communication channel
    within process 1, 2, ..., 8?

    One possible solution is to figure out if process 9 and 10 are the same
    as process 1 and 5 beforehand, and then form a communication channel
    based on the information, adjusting the ranks and world_size etc. However,
    figuring out the information is not always easy, and it will interfere
    with the main communication channel.

    Our solution is to always form a communication channel with process 1, 2,
    ..., 8, and then use this function to form another communication channel
    with process 9 and 10. This way, regardless of whether process 9 and 10
    are the same as process 1 and 5, the main communication channel is
    always formed with process 1, 2, ..., 8, and the additional communication
    channel is always formed with process 9 and 10.

    When *listen_socket* is provided, the rendezvous step
    is skipped and a ``TCPStore`` server is created directly using the
    pre-bound socket.  This is useful for eliminating TOCTOU races
    between port allocation and binding.
    """
    # SOURCE: vllm/distributed/utils.py:L576-L684（锚点双置）
    init_method = get_tcp_uri(host, port)
    backend = torch.distributed.Backend(backend)  # it is basically string
    timeout = torch.distributed.distributed_c10d._get_default_timeout(backend)
    if backend == "gloo":
        gloo_timeout = get_cpu_distributed_timeout_or_none()
        if gloo_timeout is not None:
            timeout = gloo_timeout
    else:
        device_timeout = get_distributed_timeout_or_none()
        if device_timeout is not None:
            timeout = device_timeout

    # SUBTRACTED: listen_socket 的 TCPStore 直建分支（L642-L652：
    #   create_tcp_store(host, port, listen_socket=…, is_master=True, …)）——
    #   _coord_store_port 路径（弹性 EP 面）不携带；本章消费面走 rendezvous。
    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout)
    )
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    if backend == "gloo":
        pg = init_gloo_process_group(
            prefix_store=prefix_store,
            group_rank=group_rank,
            group_size=group_size,
            timeout=timeout,
        )
        # SUBTRACTED: 非 gloo 的设备后端臂（else 分支 L662-L673，
        #   current_platform.stateless_init_device_torch_dist_pg）——本章引擎级
        #   dp_group 恒 gloo（'use gloo since the engine process might not
        #   have cuda device'），整臂裁除。

    if group_name is not None:
        from torch._C._distributed_c10d import _register_process_group

        pg._set_group_name(group_name)
        _register_process_group(group_name, pg)

    if return_store:
        return pg, store
    else:
        return pg


# SOURCE: vllm/distributed/utils.py:L687-L693 stateless_destroy_torch_distributed_
#   process_group —— 逐字
def stateless_destroy_torch_distributed_process_group(pg: ProcessGroup) -> None:
    """
    Destroy ProcessGroup returned by
        stateless_init_torch_distributed_process_group().
    """
    # SOURCE: vllm/distributed/utils.py:L687-L693（锚点双置）
    pg.shutdown()
    _unregister_process_group(pg.group_name)
