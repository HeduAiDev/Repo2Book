# vllm_ascend/eplb/adaptor/vllm_adaptor.py —— subtract-only companion（ch09 取数源）
#
# VllmEplbAdaptor：模型适配层。把 MoE 各层的 expert 权重张量 / expert_map / log2phy_map /
# 预分配 buffer 登记成 per-layer 句柄表——供 D2DExpertWeightLoader 搬运、供 EplbUpdator 采集
# moe_load。屏蔽不同量化方案的权重命名差异（本精简版只留非量化 ['w13_weight','w2_weight'] 代表）。
# 依赖真实模型对象与 NPU，host 不实例化，仅作可读控制流呈现（loader/updator 的取数契约）。
# 源码顶部 TODO：待 vLLM issue 22246 合入即删除本 adaptor。
import json
from typing import Any

import torch

from eplb_runtime_stub import dist, logger

# SUBTRACTED: import torch.distributed as dist / from vllm.logger import logger —— 经桩接住。
#   原 vllm_adaptor.py:L22-L23
# SUBTRACTED: from vllm_ascend.ascend_config import get_ascend_config /
#   from vllm_ascend.quantization.methods.base import QuantType —— 仅量化分支（已删）用。
#   原 vllm_adaptor.py:L25-L26


# SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L29-L169
class VllmEplbAdaptor:
    def __init__(self, model, **args):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L30-L57
        super().__init__(**args)
        if hasattr(model, "language_model"):
            self.model = model.language_model
            self.config = model.config.text_config
        else:
            self.model = model
            self.config = model.config
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_dense_layers = getattr(self.config, "first_k_dense_replace", 0)
        self.num_moe_layers = self.config.num_hidden_layers - self.num_dense_layers

        self.expert_map_per_layer_cpu = dict()  # copy of expert map on CPU to avoid device synchronize frequently

        self.num_local_experts = self.model.model.layers[-1].mlp.experts.local_num_experts
        self.expert_param_per_layer = dict()
        self.init_expert_param_per_layer()

        num_buffer_tensor = self.num_local_experts
        self.buffer_tensor_list: list[list[Any]] = [[] for _ in range(num_buffer_tensor)]
        self.init_buffer_tensor(num_buffer_tensor)

        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = self.model.get_log2phy_map(
                self.num_dense_layers + layer_idx
            )

    def init_buffer_tensor(self, num_buffer_tensor):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L59-L65
        for buffer_id in range(num_buffer_tensor):
            for name in self.expert_weight_names:
                complete_name = "model.layers." + str(self.num_dense_layers) + ".mlp.experts." + name
                expert_tensor = self.param_dict[complete_name][0]
                buffer_tensor = torch.empty_like(expert_tensor)
                self.buffer_tensor_list[buffer_id].append(buffer_tensor)

    def init_expert_param_per_layer(self):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L67-L118
        self.param_dict = dict()
        # SUBTRACTED: W8A8/W4A8/MXFP4/MXFP8 量化分支（原:L69-L102 的 if quant_config is not None: ...）
        #   —— 量化方案只改 expert_weight_names 的张量名清单，不改搬运控制流（仍是 per-layer per-expert
        #   张量列表）；保留非量化 ['w13_weight','w2_weight'] 代表即可讲清「每 expert 是多块张量」（delete 批准）。
        self.expert_weight_names = ["w13_weight", "w2_weight"]

        for layer_idx in range(self.num_dense_layers, self.config.num_hidden_layers):
            self.expert_param_per_layer[layer_idx] = list()
            for name in self.expert_weight_names:
                param_key = f"model.layers.{layer_idx}.mlp.experts.{name}"
                param_value = getattr(self.model.model.layers[layer_idx].mlp.experts, name)
                self.param_dict[param_key] = param_value
            for local_expert_id in range(self.num_local_experts):
                per_expert_param = list()
                for name in self.expert_weight_names:
                    per_expert_param.append(
                        self.param_dict["model.layers." + str(layer_idx) + ".mlp.experts." + name][local_expert_id]
                    )
                self.expert_param_per_layer[layer_idx].append(per_expert_param)

    def get_rank_expert_workload(self) -> torch.Tensor:
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L120-L122
        self.moe_load = self.model.get_all_moe_loads()
        return self.moe_load

    def _export_tensor_to_file(self, expert_maps, expert_map_record_path: str):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L124-L146
        if self.rank_id == 0:
            num_local_experts = expert_maps.max() + 1

            expert_maps_list = expert_maps.tolist()
            record: dict[str, Any] = {"moe_layer_count": len(expert_maps_list), "layer_list": []}

            for layer_idx, layer_data in enumerate(expert_maps_list):
                layer_record: dict[str, Any] = {
                    "layer_id": layer_idx,
                    "device_count": len(layer_data),
                    "device_list": [],
                }

                for device_idx, experts in enumerate(layer_data):
                    placement = [experts.index(i) for i in range(num_local_experts)]
                    device_record = {"device_id": device_idx, "device_expert": placement}
                    layer_record["device_list"].append(device_record)

                record["layer_list"].append(layer_record)

            with open(expert_map_record_path, "w") as f:
                json.dump(record, f, indent=4)

    def do_update_expert_map(self, layer_id, updated_expert_map):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L148-L149
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L151-L156
        for expert_tensor, buffer_tensor in zip(
            self.expert_param_per_layer[layer_id][local_expert_to_replace], self.buffer_tensor_list[buffer_tensor_id]
        ):
            expert_tensor.copy_(buffer_tensor)
            logger.debug("Expert tensor shape is :%s", expert_tensor.shape)

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L158-L160
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def get_global_expert_map(self):
        # SOURCE: vllm_ascend/eplb/adaptor/vllm_adaptor.py:L162-L169
        all_layer_global_expert_map = []
        for layer_id in range(self.num_moe_layers):
            map_cpu = self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_id] = map_cpu[self.rank_id]

        return torch.stack(all_layer_global_expert_map)
