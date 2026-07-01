# ch30 vLLM 侧扩展点(2)：LoRA 候选层元组 + from_layer —— subtract-only 精简版
#
# 真实源码 vllm/lora/utils.py：_all_lora_classes 是 LoRA 层候选元组，from_layer 顺序遍历它、
# 第一个 can_replace_layer 为 True 的类胜出。这正是昇腾「全局类替换 trick」的作用对象——
# refresh_all_lora_classes() 把 4 个 Ascend 类追加到本元组尾部，昇腾类便进了候选池。
#
# 按 subtraction_plan 思路（embed_excerpt 已 elide）删去：_all_lora_classes 中间 12 个 vLLM
# 内置类（只留首尾代表，含 QKVParallelLinearWithLoRA —— 昇腾薄壳继承的基类）。from_layer
# 控制流逐字保留。

from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    QKVParallelLinearWithLoRA,
    VocabParallelEmbeddingWithLoRA,
)
from vllm.lora.layers.base import BaseLayerWithLoRA


# Order matters here: more specific wrappers must be checked before generic
# merged/column-parallel wrappers in from_layer().
# SOURCE: vllm/lora/utils.py:L78-L95
# SUBTRACTED: utils.py:L82-L93 中间 12 个 vLLM 内置 LoRA 层类（ColumnParallelLinearWithLoRA /
#   MergedQKVParallelLinearWithLoRA / LogitsProcessorWithLoRA / FusedMoEWithLoRA / ...）。
#   只留 VocabParallelEmbeddingWithLoRA + QKVParallelLinearWithLoRA 两个代表（后者是昇腾
#   AscendQKVParallelLinearWithLoRA 的基类）。
_all_lora_classes: tuple[type[BaseLayerWithLoRA], ...] = (
    VocabParallelEmbeddingWithLoRA,
    QKVParallelLinearWithLoRA,
)


# SOURCE: vllm/lora/utils.py:L106-L124
def from_layer(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list,
    model_config: PretrainedConfig | None = None,
) -> nn.Module:
    for lora_cls in _all_lora_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if lora_cls.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance_layer = lora_cls(layer)
            instance_layer.create_lora_weights(max_loras, lora_config, model_config)
            return instance_layer
    return layer
