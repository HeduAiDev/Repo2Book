#!/usr/bin/env python3
"""ch30 机制 lora-class-replacement 教学轨迹驱动。

忠实复刻 vllm_ascend/lora/utils.py:L70-L82 的「全局类替换 trick」控制流 +
vllm/lora/utils.py:L106-L124 from_layer 的顺序遍历匹配逻辑。

真源码 import vllm / vllm_ascend, host 无法装 NPU 栈, 故用纯 Python stub
复刻两处逻辑(元组 splat 追加 + 顺序 can_replace_layer 扫描), 控制流一字对齐:
  - refresh 前 _all_lora_classes 有 16 个 vLLM 内置类, 无一认 AscendQKVParallelLinear;
  - refresh 后 splat 追加 4 个 Ascend 类 -> 20 项, 第 17 项 (1-based) 严格类型相等命中。
纯控制流, host python3 直接跑。
"""
import json


# ---- stub 源层类型(对位 vllm_ascend/ops/linear.AscendQKVParallelLinear) ----
class QKVParallelLinear:            # vLLM 原版 QKV 线性层
    pass


class AscendQKVParallelLinear:      # 昇腾改过的 QKV 线性层(ch23 线性层主线产物)
    pass


# ---- vLLM 内置 LoRA 层: can_replace_layer 只认 QKVParallelLinear 一族, ----
# ---- 认不出 AscendQKVParallelLinear。此处用统一 stub 代表 16 个内置类。 ----
def _builtin_can_replace(source_layer, packed_len):
    # 内置类严格类型相等只认 vLLM 原生层, 对昇腾变种一律 False
    return type(source_layer) is QKVParallelLinear and packed_len == 1


class BuiltinLoRA:
    def __init__(self, name):
        self.__name__ = name

    def can_replace_layer(self, source_layer, packed_len):
        return _builtin_can_replace(source_layer, packed_len)


# ---- 4 个 Ascend*LinearWithLoRA 薄壳: 只重写 can_replace_layer, ----
# ---- 严格类型相等 type(source) is AscendQKVParallelLinear + len 判据。 ----
class AscendQKVParallelLinearWithLoRA:
    __name__ = "AscendQKVParallelLinearWithLoRA"

    def can_replace_layer(self, source_layer, packed_len):
        # SOURCE: vllm_ascend/lora/utils.py:L28
        return type(source_layer) is AscendQKVParallelLinear and packed_len == 1


class AscendMergedQKVParallelLinearWithLoRA:
    __name__ = "AscendMergedQKVParallelLinearWithLoRA"

    def can_replace_layer(self, source_layer, packed_len):
        return type(source_layer) is AscendQKVParallelLinear and packed_len == 3


class AscendMergedQKVParallelLinearWithShardedLoRA:
    __name__ = "AscendMergedQKVParallelLinearWithShardedLoRA"

    def can_replace_layer(self, source_layer, packed_len):
        return type(source_layer) is AscendQKVParallelLinear and packed_len == 3


class AscendQKVParallelLinearWithShardedLoRA:
    __name__ = "AscendQKVParallelLinearWithShardedLoRA"

    def can_replace_layer(self, source_layer, packed_len):
        return type(source_layer) is AscendQKVParallelLinear and packed_len == 1


# vLLM 原始全局元组: 16 个内置 LoRA 层类(名字不影响判据, 统一 stub)
_all_lora_classes = tuple(BuiltinLoRA(f"builtin_{i}") for i in range(16))


def refresh_all_lora_classes(classes):
    """SOURCE: vllm_ascend/lora/utils.py:L70-L82 —— splat 追加到尾部。"""
    ascend_classes = (
        AscendQKVParallelLinearWithLoRA(),
        AscendMergedQKVParallelLinearWithLoRA(),
        AscendMergedQKVParallelLinearWithShardedLoRA(),
        AscendQKVParallelLinearWithShardedLoRA(),
    )
    return (*classes, *ascend_classes)


def from_layer(classes, source_layer, packed_len):
    """SOURCE: vllm/lora/utils.py:L106-L124 —— 顺序遍历, 第一个 True 胜出。"""
    for idx, lora_cls in enumerate(classes, start=1):   # 1-based 位置
        if lora_cls.can_replace_layer(source_layer, packed_len):
            return idx, type(lora_cls).__name__
    return None, "返回原层(LoRA失效)"


def scan_report(classes, source_layer, packed_len):
    """记录前多少项判 False, 命中在第几项。"""
    first_false_run = 0
    hit_index, hit_name = from_layer(classes, source_layer, packed_len)
    for idx, lora_cls in enumerate(classes, start=1):
        if lora_cls.can_replace_layer(source_layer, packed_len):
            break
        first_false_run += 1
    return {
        "tuple_len": len(classes),
        "leading_false_count": first_false_run,
        "hit_index_1based": hit_index,
        "hit_class": hit_name,
        "lora_active": hit_index is not None,
    }


if __name__ == "__main__":
    # 源层 = 一个昇腾 QKV 线性层, 非合并(packed_modules_list 长度 1)
    src = AscendQKVParallelLinear()
    packed_len = 1

    before = scan_report(_all_lora_classes, src, packed_len)
    after_classes = refresh_all_lora_classes(_all_lora_classes)
    after = scan_report(after_classes, src, packed_len)

    trace = {
        "source_layer": "AscendQKVParallelLinear",
        "packed_modules_list_len": packed_len,
        "builtin_count": 16,
        "ascend_appended": 4,
        "before_refresh": before,
        "after_refresh": after,
    }
    print(json.dumps(trace, ensure_ascii=False, indent=2))
