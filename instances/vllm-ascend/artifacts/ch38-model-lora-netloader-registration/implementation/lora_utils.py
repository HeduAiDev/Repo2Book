# ch30 变体(2) LoRA 全局类替换 trick —— subtract-only 精简版
#
# 真实源码 vllm_ascend/lora/utils.py（整文件 82 行）。零删除：4 个 Ascend*LinearWithLoRA
# 薄壳 + refresh_all_lora_classes 全是「全局类替换 trick」的本体，逐字保留。
#
# 注：dossier 的 embed_excerpt 仅内嵌 1 个类作代表（叙事用），但实现侧 4 个类都不在
# subtraction_plan.delete 里，故全部保留（薄壳继承范式 + can_replace_layer 改匹配目标）。

import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace

from vllm_ascend.ops.linear import (
    AscendQKVParallelLinear,
)


# SOURCE: vllm_ascend/lora/utils.py:L18-L28
class AscendQKVParallelLinearWithLoRA(QKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # SOURCE: vllm_ascend/lora/utils.py:L19-L28
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1


# SOURCE: vllm_ascend/lora/utils.py:L31-L41
class AscendMergedQKVParallelLinearWithLoRA(MergedQKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # SOURCE: vllm_ascend/lora/utils.py:L31-L41
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


# SOURCE: vllm_ascend/lora/utils.py:L44-L54
class AscendMergedQKVParallelLinearWithShardedLoRA(MergedQKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # SOURCE: vllm_ascend/lora/utils.py:L44-L54
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


# SOURCE: vllm_ascend/lora/utils.py:L57-L67
class AscendQKVParallelLinearWithShardedLoRA(QKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # SOURCE: vllm_ascend/lora/utils.py:L57-L67
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1


# SOURCE: vllm_ascend/lora/utils.py:L70-L82
def refresh_all_lora_classes():
    ascend_classes = (
        AscendQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithShardedLoRA,
        AscendQKVParallelLinearWithShardedLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Append the Ascend classes in a deterministic order.
    vllm.lora.utils._all_lora_classes = (
        *vllm.lora.utils._all_lora_classes,
        *ascend_classes,
    )
