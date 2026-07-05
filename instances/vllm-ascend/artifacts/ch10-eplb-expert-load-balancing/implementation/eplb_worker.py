# vllm_ascend/eplb/core/eplb_worker.py —— subtract-only companion（ch09 主线② 子进程 + 规划）
#
# EplbProcess：子进程容器，两条跨进程队列解耦计算与规划——
#   planner_q（唤醒信号，无界）/ block_update_q（规划结果，maxsize=1 背压）。
#   _launch_process 用 daemon Process 起子进程，worker_process 为子进程主循环：
#   阻塞 planner_q.get() → worker.do_update() 跑规划 → block_update_q.put(结果)。
# EplbWorker：子进程内真正干活者——读 shared_dict → policy.rebalance_experts 算新放置 →
#   compose_expert_update_info_greedy 用 expert_map 的 -1 差集算每层 send/recv → pack 跨进程。
# 源码顶部无 vllm 对位文件（vLLM 尚未合入此特性）。
from multiprocessing import Process, Queue
from typing import Any

import numpy as np
import torch

from eplb_runtime_stub import dist, generate_log2phy_map, logger

from policy_factory import PolicyFactory

# SUBTRACTED: import torch.distributed as dist —— host 无后端，仅用 dist.get_rank()，经桩接住。
#   原 eplb_worker.py:L22
# SUBTRACTED: from vllm.logger import logger / from vllm_ascend.eplb.core.eplb_utils import
#   generate_log2phy_map —— 经 eplb_runtime_stub 接住。原 eplb_worker.py:L23,L25


# SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L29-L322
class EplbWorker:
    def __init__(self, shared_dict, policy_type, enable_d2d: bool = True):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L30-L37
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type)
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d
        self.rank_id = dist.get_rank()
        self.multi_stage = policy_type == 3

    def do_update(self):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L39-L107
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D
        # Get initial expert_map
        torch.set_num_threads(1)
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            if self.old_expert_maps is not None:
                self.num_local_experts = self.old_expert_maps.max() + 1
            else:
                raise ValueError("Failed to get expert_maps from shared_dict.")

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            logger.debug("[eplb/worker] No moe_load data available yet, skipping this cycle")
            return

        # Get the updated expert table based on the workload information
        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)
        _, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)

        # SUBTRACTED: rank0 的 _calculate_hotness/_compute_imbalance/latest_expert_hotness 块（原:L66-L93）
        #   —— 仅算热度不均衡指标喂监控（注释明标 ms-service-metric begin/end），不参与 new_placement
        #   计算与 send/recv 生成，删后规划结果不变（dossier delete 批准）。multi_stage 的
        #   load_info.sum(0) 旁支随之一并删除。

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        self.check_expert_placement(old_placement, new_placement)
        new_expert_maps = self.local2global(new_placement)
        self.update_expert_map(new_expert_maps)

        update_info = self.compose_expert_update_info_greedy(new_expert_maps, self.old_expert_maps)
        self.old_expert_maps = new_expert_maps
        logger.debug("[eplb/worker] EPLB Process compute complete")

        packed_update_info = self.pack_update_info(update_info)

        return packed_update_info

    def check_expert_placement(self, old_placement, new_placement):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L109-L145
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]

        for layer_id in range(num_layers):
            # check if any logical expert is not placed on any rank
            if torch.unique(new_placement[layer_id]).numel() < torch.unique(old_placement[layer_id]).numel():
                logger.error("[eplb/worker] There exists expert not placed on any rank in layer %s", layer_id)
                new_placement[layer_id] = old_placement[layer_id]
                continue

            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]

                # check if same logical experts are placed on the same NPU
                if new_placement_check.numel() != torch.unique(new_placement_check).numel():
                    logger.error(
                        "[eplb/worker] Replicated experts are placed on the same NPU; "
                        "expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                # check if there is any experts movement inside one NPU
                expert_not_move = torch.isin(new_placement_check, old_placement_check)
                if not torch.equal(new_placement_check[expert_not_move], old_placement_check[expert_not_move]):
                    logger.error(
                        "[eplb/worker] Expert movement inside NPU detected; "
                        "expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L148-L202
        num_layers = current_expert_maps.shape[0]
        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                yield (
                    expert_send_info_this_layer,
                    expert_recv_info_this_layer,
                    updated_expert_maps_this_layer,
                    layer_id,
                )

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where(
                (current_expert_maps_this_layer == -1) & (updated_expert_maps_this_layer != -1)
            )

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where(
                (current_expert_maps_this_layer != -1) & (updated_expert_maps_this_layer == -1)
            )

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)[0]
                else:
                    candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

                # TODO: improve selection criterion of NPU sending expert_id,
                # considering intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                updated_expert_maps_this_layer,
                layer_id,
            )

    def calculate_rebalance_experts(self, load_info, old_placement):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L204-L212
        """
        Compute `new_map` by calling the `rebalance_experts` method of the policy instance.
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(old_placement, load_info)
        return changed, priority, new_map

    def get_init_expert_maps(self):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L214-L218
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L220-L225
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):  # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L227-L228
        self.shared_dict["expert_maps"] = expert_maps

    def global2local(self, placement: torch.Tensor, E_local: int) -> tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L230-L243
        L, G, _ = placement.shape
        device = placement.device

        pt_local = torch.full((L, G, E_local), fill_value=-1, dtype=torch.long, device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L245-L263
        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global), fill_value=-1, dtype=torch.long, device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def pack_update_info(self, update_info_generator):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L265-L288
        """
        Pack a list of update info tuples for efficient IPC.
        """
        send_all = []
        recv_all = []
        maps = []
        log2phy_all = []
        layer_ids = []

        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:
            send_info_this_rank = send_info.get(self.rank_id, [])
            recv_info_this_rank = recv_info.get(self.rank_id, [])
            send_all.append(send_info_this_rank)
            recv_all.append(recv_info_this_rank)

            maps.append(new_expert_map[self.rank_id].numpy().tolist())

            log2phy_map = generate_log2phy_map(new_expert_map, self.rank_id)
            log2phy_all.append(log2phy_map.numpy().tolist())

            layer_ids.append(layer_id)

        return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))

    # SUBTRACTED: @staticmethod _compute_imbalance(原:L290-L309) 与 _calculate_hotness(原:L311-L322)
    #   —— 仅供 do_update 的 rank0 监控块算热度/不均衡指标（已删），无其它调用方（dossier delete 批准）。


# SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L325-L388
class EplbProcess:
    def __init__(self, shared_dict, policy_type: int = 0, enable_d2d: bool = True):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L326-L340
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q: Queue[Any] = Queue()
        self.block_update_q: Queue[Any] = Queue(maxsize=1)

        # Create EplbWorker instance
        self.worker = EplbWorker(self.shared_dict, self.policy_type, self.enable_d2d)

    def worker_process(self, planner_q, block_update_q):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L342-L378
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up,
        call do_update, then notify main process update is complete.
        """
        # SUBTRACTED: ms_service_metric 的 try/except 初始化块（原:L347-L354）—— 纯指标采集
        #   （华为内部 metric adapter），失败也只 warning，与在线热迁移主线零耦合（delete 批准）。
        # SUBTRACTED: if self.policy_type == 3: from ...policy_flashlb import warm_up; warm_up()
        #   （原:L356-L359）—— FlashLB 专属预热，非主线 DefaultEplb 路径（delete 批准）。
        while True:
            try:
                planner_q.get()

                packed_update_info = self.worker.do_update()

                while True:
                    if not block_update_q.empty():
                        continue
                    block_update_q.put(packed_update_info)
                    break

            except Exception as e:
                logger.warning(
                    "[eplb/worker] Subprocess crashed, EPLB optimization will stop. error=%s",
                    e,
                    exc_info=True,
                )
                break

    def _launch_process(self):
        # SOURCE: vllm_ascend/eplb/core/eplb_worker.py:L380-L387
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(target=self.worker_process, args=(self.planner_q, self.block_update_q), daemon=True)

        proc.start()
        return proc
