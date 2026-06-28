# vllm_ascend/eplb/eplb_updator.py —— subtract-only companion（ch09 主线① 节拍状态机）
#
# EplbUpdator 跑在推理主进程侧，由 model_runner 每个 step 调一次 forward_before/forward_end。
# 靠 cur_iterations 单调计数 + 三个间隔常量（expert_heat_collection_interval 默认400 /
# algorithm_execution_interval 默认30 / num_moe_layers）算出「这一拍该 gather 负载 / 唤醒规划
# / 取规划 / 搬第几层权重」，满一整轮自动归零。三个判定函数把节拍全表达成对 interval 的算术比较。
# 仅借用 vLLM 通信原语：compute_and_set_moe_load 走 ch08 _DYNAMIC_EPLB 组的 all_gather（f6 回收）。
# 源码顶部 TODO：待 vLLM issue 22246 合入即删除本 updator。
import numpy
import torch

from eplb_runtime_stub import dist, envs, get_dynamic_eplb_group, logger, record_function_or_nullcontext

from eplb_device_transfer_loader import D2DExpertWeightLoader
from eplb_worker import EplbProcess
from vllm_adaptor import VllmEplbAdaptor

# SUBTRACTED: import torch.distributed as dist / import vllm.envs as envs / from vllm.logger import
#   logger / from vllm.v1.utils import record_function_or_nullcontext / from
#   vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group —— host 无后端，经
#   eplb_runtime_stub 接住。原 eplb_updator.py:L20-L25


# SOURCE: vllm_ascend/eplb/eplb_updator.py:L31-L184
class EplbUpdator:
    def __init__(self, eplb_config, loader: D2DExpertWeightLoader, eplb_process: EplbProcess, process):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L32-L39
        self.eplb_config = eplb_config
        self.multi_stage = eplb_config.eplb_policy_type == 3
        self.init_eplb(self.eplb_config.expert_map_path, process)
        self.eplb_loader = loader
        self.eplb_process = eplb_process
        self.shared_dict = self.eplb_process.shared_dict
        self.comm_group = get_dynamic_eplb_group()

    def set_adaptor(self, adaptor: VllmEplbAdaptor):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L41-L47
        self.adaptor = adaptor
        self.num_moe_layers = self.adaptor.num_moe_layers
        local_load = self.adaptor.get_rank_expert_workload()
        self.world_size = dist.get_world_size()
        self.device = local_load.device
        self.eplb_loader.num_layers = self.adaptor.num_dense_layers + self.adaptor.num_moe_layers

    def init_eplb(self, expert_map_path, process):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L49-L75
        self.rank_id = dist.get_rank()
        self.num_expert_load_gather = 10
        self.periodic_load_gather = True
        self.expert_heat_collection_interval: torch.int64 = self.eplb_config.expert_heat_collection_interval
        self.expert_map_path = expert_map_path
        self.expert_map_record_path = self.eplb_config.expert_map_record_path

        try:
            if not envs.VLLM_ALLOW_EXPERT_LOAD_COLLECTING:
                self.num_expert_load_gather = self.expert_heat_collection_interval
                self.periodic_load_gather = False
        except Exception:
            logger.debug("[eplb/updator] VLLM_ALLOW_EXPERT_LOAD_COLLECTING unavailable in current vllm version.")
            self.num_expert_load_gather = self.expert_heat_collection_interval
            self.periodic_load_gather = False

        self.reqs = []
        self.update_info_all = []

        self.cur_iterations: torch.int64 = 0

        self.algorithm_execution_interval: torch.int64 = self.eplb_config.algorithm_execution_interval

        self.process = process

        logger.info("[eplb/updator] Launched EPLB subprocess, pid=%s", self.process.pid)

    def update_iteration(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L77-L87
        self.cur_iterations += 1
        if self.cur_iterations == (
            self.expert_heat_collection_interval + self.algorithm_execution_interval + self.num_moe_layers
        ):
            logger.debug("[eplb/updator] Full EPLB cycle completed, clearing moe loads and resetting iteration counter")
            # SUBTRACTED: if self.expert_map_record_path is not None: _export_tensor_to_file(...)（原:L83-L84）
            #   —— 落盘记录 expert 映射是离线分析用途，非在线热迁移主线（主线 record_path=None）（delete 批准）。
            self.adaptor.model.clear_all_moe_loads()
            self.cur_iterations = 0

    def get_update_info_flag(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L89-L90
        return self.cur_iterations == (self.expert_heat_collection_interval + self.algorithm_execution_interval - 1)

    def wakeup_eplb_worker_flag(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L92-L93
        return self.cur_iterations == (self.expert_heat_collection_interval - 1)

    def update_expert_weight_flag(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L95-L99
        weight_update_counter = self.cur_iterations - (
            self.expert_heat_collection_interval + self.algorithm_execution_interval
        )
        return weight_update_counter >= 0 and weight_update_counter < self.num_moe_layers

    def wakeup_eplb_worker(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L101-L102
        self.eplb_process.planner_q.put(1)

    def forward_before(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L104-L125
        # Batch after eplb process being triggered, get update info provided by eplb process
        if self.get_update_info_flag():
            self.update_info_all = self.eplb_process.block_update_q.get()
        if self.update_expert_weight_flag():
            with record_function_or_nullcontext("EPLB generate p2p task"):
                (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = (
                    self.update_info_all.pop(0)
                )
                log2phy_map_this_rank = torch.from_numpy(numpy.array(log2phy_map))
                self.eplb_loader.set_log2phy_map(log2phy_map_this_rank)
                updated_expert_map_this_rank = torch.from_numpy(numpy.array(updated_expert_map))
                self.eplb_loader.generate_expert_d2d_transfer_task(
                    expert_send_info,
                    expert_recv_info,
                    updated_expert_map_this_rank,
                    layer_id + self.adaptor.num_dense_layers,
                )

                # set asynchronous stream for d2d expert weight update
                self.reqs = []
                self.eplb_loader.asyn_expert_weight_transfer(self.reqs)

    def forward_end(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L127-L136
        if self.wakeup_eplb_worker_flag():
            with record_function_or_nullcontext("EPLB gather moe load"):
                self.compute_and_set_moe_load()
                self.wakeup_eplb_worker()

        # SUBTRACTED: 原守卫为 `... and self.expert_map_record_path is None`（原:L133）—— 落盘模式
        #   下不搬权重；主线 record_path=None 时该守卫恒真，删去后行为不变（delete 批准）。
        if self.update_expert_weight_flag():
            self.eplb_loader.update_expert_map_and_weight(self.reqs)

        self.update_iteration()

    def compute_and_set_moe_load(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L138-L148
        local_load = self.adaptor.get_rank_expert_workload().unsqueeze(1)
        moe_load = self.comm_group.all_gather(local_load, dim=1).cpu()

        # SUBTRACTED: if self.multi_stage: moe_load = moe_load.permute(2, 0, 1, 3)（原:L142-L143）
        #   —— 仅 policy_type==3(FlashLB) 才重排，主线 DefaultEplb 不走（delete 批准）。

        self.shared_dict["moe_load"] = moe_load
        logger.debug("[eplb/updator] Updated shared_dict['moe_load'] shape=%s", moe_load.shape)

        return moe_load

    def warm_up_eplb(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L150-L174
        logger.info("[eplb/updator] Starting EPLB warm-up, rank=%s, world_size=%s", self.rank_id, self.world_size)
        self.shared_dict["expert_maps"] = self.adaptor.get_global_expert_map()
        self.compute_and_set_moe_load()
        # SUBTRACTED: dummy P2P 预热环（原:L155-L173 的 src_tensor/comm_op_list/batch_isend_irecv/wait）
        #   —— 仅用 1 元素张量预热通信链路，不搬真实权重；删通信细节，保留 shared_dict 初始化与
        #   首次 compute_and_set_moe_load（delete 批准）。

    def shutdown(self):
        # SOURCE: vllm_ascend/eplb/eplb_updator.py:L176-L183
        """
        Clean up the EPLB process.
        """
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
            logger.info("[eplb/updator] EPLB subprocess terminated")
