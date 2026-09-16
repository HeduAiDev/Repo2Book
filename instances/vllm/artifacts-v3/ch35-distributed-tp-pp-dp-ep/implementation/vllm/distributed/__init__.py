# SOURCE: vllm/distributed/__init__.py —— re-export 子集（真实文件 re-export
# parallel_state/utils/communication_op 的公开面）。本章消费面（all2all.py 与
# gpu_worker 等的 import 路径）：Handle 与 get_*_group 族。

from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    Handle,
    GroupCoordinator,
    destroy_distributed_environment,
    destroy_model_parallel,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    init_model_parallel_group,
    in_the_same_node_as,
    set_custom_all_reduce,
)

__all__ = [
    "Handle",
    "GroupCoordinator",
    "destroy_distributed_environment",
    "destroy_model_parallel",
    "get_dp_group",
    "get_ep_group",
    "get_pcp_group",
    "get_pp_group",
    "get_tp_group",
    "get_world_group",
    "init_distributed_environment",
    "init_model_parallel_group",
    "in_the_same_node_as",
    "set_custom_all_reduce",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
]
# SUBTRACTED: 其余 re-export（stateless/coordinator 序列化面）——弹性 EP 域。
