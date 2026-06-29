# vllm_ascend/_310p/sharded_state_loader_310p.py —— subtract-only 精简版（ch17 主线之一：权重加载适配）
#
# 直接两层继承 vLLM 基类、跳过昇腾中间层：ShardedStateLoader310(ShardedStateLoader)，
# ShardedStateLoader 直接 from vllm.model_executor.model_loader。两处特化：
#   (1) save_model 永远单 part（part_idx=0），去掉基类按 max_size 切多 part 的逻辑，
#       简化 310P/CANN 的加载契约；
#   (2) 额外 generate_quant_description 产出 parameters_type_map.json——逐参数 dtype/
#       量化类型描述，是 310P 量化加载格式所需。
#
# save_model 依赖 safetensors/分布式 rank，但 generate_quant_description 的逐参数
# dtype 分类 + JSON 落盘是纯 Python（用假 state_dict + 临时目录即可跑，见 ../tests）。
import json
import os
from pathlib import Path

import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader import ShardedStateLoader


# SOURCE: vllm_ascend/_310p/sharded_state_loader_310p.py:L27
class ShardedStateLoader310(ShardedStateLoader):
    # SOURCE: vllm_ascend/_310p/sharded_state_loader_310p.py:L28-L29
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        # SOURCE: vllm_ascend/_310p/sharded_state_loader_310p.py:L31-L49
        from safetensors.torch import save_file
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        # 永远单 part：无视 max_size 的多 part 切分（对照基类 L178-L214 的 part_idx 递增）。
        part_idx = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())

        filename = ShardedStateLoader.DEFAULT_PATTERN.format(rank=rank, part=part_idx)
        save_file(
            state_dict,
            os.path.join(path, filename),
        )

    @staticmethod
    def generate_quant_description(
        model: torch.nn.Module,
        path: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        """Generate a mapping of parameter names to their corresponding quantization types."""
        # SOURCE: vllm_ascend/_310p/sharded_state_loader_310p.py:L51-L80
        quant_description = {}
        if quant_config is None:
            quantize_type = "FLOAT"
        else:
            try:
                quantize_type = quant_config.quant_description.get("model_quant_type", "FLOAT")
            except AttributeError:
                quantize_type = "FLOAT"
        quant_description["model_quant_type"] = quantize_type
        quant_description["version"] = "1.0.0"
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        for name, tensor in state_dict.items():
            if name.endswith(".weight") or name.endswith(".bias"):
                if tensor.dtype in [torch.int8, torch.int32, torch.int64]:
                    quant_description[name] = quantize_type
                else:
                    quant_description[name] = "FLOAT"
            else:
                quant_description[name] = "FLOAT"

        json_path = Path(path) / "parameters_type_map.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(quant_description, f, indent=2)
