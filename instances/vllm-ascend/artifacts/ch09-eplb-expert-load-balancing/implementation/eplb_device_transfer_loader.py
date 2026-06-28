# vllm_ascend/eplb/core/eplb_device_transfer_loader.py —— subtract-only companion（ch09 主线③）
#
# D2DExpertWeightLoader：异步 P2P 权重搬运器，WAITING→READY→TRANSFERRING 三态机。
#   generate_expert_d2d_transfer_task 攒 dist.P2POp（isend/irecv）→ asyn_expert_weight_transfer
#   batch_isend_irecv 异步发车 → update_expert_map_and_weight 等 req.wait 收车 + copy_ 落地。
# 仅借用 vLLM 通信原语（torch.distributed.P2POp/batch_isend_irecv）+ ch08 的 get_dynamic_eplb_group。
# host 无分布式后端：真实 P2P 由 eplb_runtime_stub 的 record-only dist 接住，只验三态机控制流。
# 源码顶部 TODO：待 vLLM issue 22246 合入即删除本套实现。
from enum import Enum

from eplb_runtime_stub import dist, get_dynamic_eplb_group, logger, record_function_or_nullcontext

# SUBTRACTED: import torch.distributed as dist —— host 无后端，改用 record-only 替身。
#   原 eplb_device_transfer_loader.py:L19
# SUBTRACTED: from vllm.logger import logger / from vllm.v1.utils import record_function_or_nullcontext
#   / from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group —— 经桩接住。
#   原 eplb_device_transfer_loader.py:L20-L23


# SOURCE: vllm_ascend/eplb/core/eplb_device_transfer_loader.py:L26-L29
class ExpertWeightUpdateState(Enum):
    WAITING = 0  # waiting for updated expert_map by EplbWorker
    READY = 1  # ready for d2d expert weights updating
    TRANSFERRING = 2  # d2d finished and waiting for updating expert_map into model


# SOURCE: vllm_ascend/eplb/core/eplb_device_transfer_loader.py:L32-L130
class D2DExpertWeightLoader:
    def __init__(self):  # SOURCE: eplb_device_transfer_loader.py:L33-L41
        self.comm_op_list = None
        self.updated_expert_map = None
        self.updated_log2phy_map = None
        self.layer_id = -1  # layer id to be updated
        self.state = ExpertWeightUpdateState.WAITING
        self.recv_expert_list = []
        self.num_layers = 0
        self.comm_group = get_dynamic_eplb_group()

    def set_adator(self, eplb_adaptor):  # SOURCE: eplb_device_transfer_loader.py:L43-L44
        self.eplb_adaptor = eplb_adaptor

    def generate_expert_d2d_transfer_task(self, expert_send_info, expert_recv_info, updated_expert_map, layer_id):
        # SOURCE: eplb_device_transfer_loader.py:L46-L75
        # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.warning_once(
                "[eplb/d2d_loader] Current D2D weight update is on-going, cannot accept new update task"
            )
            return

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id][global_expert_id_to_send].item()
            for src_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][local_expert_id]:
                self.comm_op_list.append(
                    dist.P2POp(dist.isend, src_tensor, dst_rank, group=self.comm_group.device_group)
                )

        for buffer_tensor_id, recv_info in enumerate(expert_recv_info):
            recv_rank, global_expert_id_to_recv = recv_info
            for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[buffer_tensor_id]:
                self.comm_op_list.append(
                    dist.P2POp(dist.irecv, buffer_tensor, recv_rank, group=self.comm_group.device_group)
                )
            local_expert_to_replace = self.updated_expert_map[global_expert_id_to_recv].item()
            self.recv_expert_list.append((local_expert_to_replace, buffer_tensor_id))

        self.state = ExpertWeightUpdateState.READY

    def set_log2phy_map(self, log2phy_map):  # SOURCE: eplb_device_transfer_loader.py:L77-L78
        self.updated_log2phy_map = log2phy_map

    def asyn_expert_weight_transfer(self, reqs):
        # SOURCE: eplb_device_transfer_loader.py:L80-L90
        # Only when send/recv tasks are parsed into self.comm_op_list, d2d send/recv tasks can be launched
        if self.state != ExpertWeightUpdateState.READY:
            return

        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            ret_list = dist.batch_isend_irecv(self.comm_op_list)
            reqs.extend(ret_list)

        self.state = ExpertWeightUpdateState.TRANSFERRING

    def update_expert_map_and_weight(self, reqs):
        # SOURCE: eplb_device_transfer_loader.py:L92-L130
        # Only after send/recv tasks have been launched, expert_map and weight can be updated
        if self.state != ExpertWeightUpdateState.TRANSFERRING:
            return

        # Waiting for send/recv tasks finish
        if reqs:
            with record_function_or_nullcontext("EPLB weight D2D wait"):
                for req in reqs:
                    req.wait()

        if self.comm_op_list is not None:
            self.comm_op_list = None

        # update expert_map
        self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)

        # update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)

        # update expert weight
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(self.layer_id, local_expert_to_replace, buffer_tensor_id)

        logger.debug(
            "[eplb/d2d_loader] Layer %s D2D transfer completed, updated_experts=%s",
            self.layer_id,
            len(self.recv_expert_list),
        )

        if self.layer_id == self.num_layers - 1:
            logger.info("[eplb/d2d_loader] Full expert weight update cycle completed, total_layers=%s", self.num_layers)

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING
