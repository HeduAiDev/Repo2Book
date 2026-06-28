# 基座 vLLM 分布式接缝（subtract-only 桩）—— 昇腾「加法式复用」的对象。
#
# 本文件不是昇腾代码：它把昇腾 import 的那几个基座符号
#   (GroupCoordinator / init_model_parallel_group / get_world_group / get_tp_group)
# 以及基座 initialize_model_parallel 里 CP 组（PCP/DCP）的排布代数，逐字保留下来，
# 只把「真实 hccl/gloo 进程组创建 + 集合通信」这一段 SUBTRACTED（host 无 NPU/CANN）。
# 控制流与算出的 group_ranks 与真实源码一致——companion 只验排布，不发起通信。
import torch

_WORLD = None
_TP = None
_DCP = None
_PCP = None


class GroupCoordinator:
    # SOURCE: vllm/distributed/parallel_state.py:L290-L317
    """PyTorch ProcessGroup wrapper for a group of processes.

    昇腾各并行组都是它的一个实例（复用、不修改）。消费方只取
    .world_size / .rank_in_group / .device_group + all_gather/all_to_all 等原语。
    """

    def __init__(
        self,
        group_ranks,
        local_rank,
        torch_distributed_backend=None,
        use_device_communicator=True,
        use_message_queue_broadcaster=False,
        group_name=None,
    ):
        # SOURCE: vllm/distributed/parallel_state.py:L319-L360
        # SUBTRACTED: __init__ 主体 —— torch.distributed.new_group 建 device/cpu group、
        #   device_communicator 装配、_register_group。host 无 hccl，真实建组/通信不可跑。
        #   这里只记录 group_ranks 并按「本进程全局 rank == local_rank」选出所属子组，
        #   逐字保留真实源码的 rank_in_group / world_size 推导（ranks.index / len）。
        #   原 vllm/distributed/parallel_state.py:L319-L420 的进程组装配全部省略。
        self.group_ranks = [list(g) for g in group_ranks]
        self.local_rank = local_rank
        self.group_name = group_name
        self.rank = local_rank  # global rank（单进程 companion 下等于 local_rank）
        self.ranks = next(
            (g for g in self.group_ranks if self.rank in g), self.group_ranks[0]
        )
        self.world_size = len(self.ranks)
        self.rank_in_group = (
            self.ranks.index(self.rank) if self.rank in self.ranks else 0
        )
        self.device_group = None  # SUBTRACTED: 真实 ProcessGroup

    def destroy(self):
        # SOURCE: vllm/distributed/parallel_state.py:L_destroy
        # SUBTRACTED: 真实 destroy_process_group。companion 无真实组，置空即可。
        self.group_ranks = []


def init_model_parallel_group(
    group_ranks,
    local_rank,
    backend,
    use_message_queue_broadcaster=False,
    group_name=None,
    use_device_communicator=True,
):
    # SOURCE: vllm/distributed/parallel_state.py:L1159-L1174
    # 基座工厂：吃 group_ranks(list[list[int]]) → 造一个 GroupCoordinator。
    # 昇腾每建一个昇腾专属组都调它——这是「加法式复用」的接缝。
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
    )


def get_world_group():
    # SOURCE: vllm/distributed/parallel_state.py:L_get_world_group
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def get_tp_group():
    # SOURCE: vllm/distributed/parallel_state.py:L_get_tp_group
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_pcp_group():
    # SOURCE: vllm/distributed/parallel_state.py:L_get_pcp_group
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP


def get_dcp_group():
    # SOURCE: vllm/distributed/parallel_state.py:L_get_dcp_group
    assert _DCP is not None, "decode context parallel group is not initialized"
    return _DCP


def init_world_group(world_size, local_rank=0, backend="hccl"):
    # SOURCE: vllm/distributed/parallel_state.py:L_init_world_group
    # SUBTRACTED: 真实 init_distributed_environment(hccl) 建 _WORLD（new_group 全 rank）。
    #   companion 仅记录 world_size/local_rank，供昇腾 init 取 backend/world_size 用。
    global _WORLD
    _WORLD = GroupCoordinator([list(range(world_size))], local_rank, backend)
    return _WORLD


def initialize_model_parallel(
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    decode_context_model_parallel_size,
    tensor_model_parallel_size,
    world_size,
    local_rank=0,
    backend="hccl",
):
    # SOURCE: vllm/distributed/parallel_state.py:L1494-L1633
    # 基座在 init_ascend_model_parallel 之前建好 TP/PP/DP/PCP/DCP（worker 时序证据）。
    # 昇腾能在其上做加法的前提 = 这张 all_ranks 5D 网格与昇腾那张【逐维一致】。
    # SUBTRACTED: _PP/_DP/enable_elastic_ep 分支 + use_message_queue_broadcaster 细节
    #   + 各 assert/注释 —— 本章 CP 归口只需 TP/DCP/PCP 三组的排布。原:L1494-L1568,L1635+
    # the layout order is: ExternalDP x DP x PP x PCP x TP
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )

    # Build the tensor model-parallel groups.
    global _TP
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _TP = init_model_parallel_group(
        group_ranks, local_rank, backend, group_name="tp"
    )

    # Build the DCP model-parallel groups. dcp 不增 world，复用 TP 的 GPU，
    # 把一个 TP 组切成 tp_size//dcp_size 个 DCP 子组（dcp_size ≤ tp_size）。
    global _DCP
    group_ranks = all_ranks.reshape(
        -1, decode_context_model_parallel_size
    ).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DCP = init_model_parallel_group(
        group_ranks, local_rank, backend, group_name="dcp"
    )

    # Build the PCP model-parallel groups. transpose(3,4) 把 pcp 维换到末尾再切，
    # 故 PCP 组沿 tp 维以步长 tp 跳取。
    global _PCP
    group_ranks = (
        all_ranks.transpose(3, 4)
        .reshape(-1, prefill_context_model_parallel_size)
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    _PCP = init_model_parallel_group(
        group_ranks, local_rank, backend, group_name="pcp"
    )


def reset_base_groups():
    # SOURCE: vllm/distributed/parallel_state.py:L_destroy_model_parallel
    # 测试夹具：清空基座/昇腾共享的进程级全局，保证用例间幂等。
    global _WORLD, _TP, _DCP, _PCP
    _WORLD = _TP = _DCP = _PCP = None
