# vllm_ascend/device/device_op.py —— subtract-only 精简版（KV 写回的算子适配层）
#
# 本章只用到一件事：把 reshape_and_cache 落到 torch_npu._npu_reshape_and_cache。
# DeviceOperator 是「按设备型号选适配器」后得到的别名；标准 NPU 主线落在 BaseDeviceAdaptor。
#
# host 无 CANN/torch_npu：测试在 sys.modules 桩一个「记录调用」的 torch_npu 替身，
# 以验证「reshape_and_cache 确把 key/value/slot_indices 透传给 _npu_reshape_and_cache」这条控制流，
# 算子本身不真算（昇腾才有内核）。
import torch_npu

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# SUBTRACTED: mxfp_compat / triton fla kernels / QuantType 等数十行 import（device_op.py:L18-L39）
#   —— 量化/triton 算子适配，非本章 KV 写回主线。


# SOURCE: vllm_ascend/device/device_op.py:L42
class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        # SOURCE: vllm_ascend/device/device_op.py:L43-L47
        # 真身：torch_npu 把 [num_tokens, num_kv_heads, head_size] 的新 K/V 按 slot_indices
        # 散写进分页 cache（形状 (2, num_blocks, block_size, num_kv_heads, head_size)）。
        torch_npu._npu_reshape_and_cache(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache, slot_indices=slot_mapping
        )

    # SUBTRACTED: npu_moe_init_routing / 量化 / allgather 等数十个设备算子方法（device_op.py:L49+）
    #   —— MoE/通信/量化适配，非本章注意力 KV 写回主线。


# SUBTRACTED: A5DeviceAdaptor(BaseDeviceAdaptor)（device_op.py:L785+）—— A5 芯片特化覆写，
#   标准主线落到 BaseDeviceAdaptor 这条（见 get_device_adaptor）。


# SOURCE: vllm_ascend/device/device_op.py:L1663-L1667
def get_device_adaptor() -> type["BaseDeviceAdaptor"]:
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        # SUBTRACTED: return A5DeviceAdaptor（device_op.py:L1666）—— A5 特化已减，主线返回基类。
        pass
    return BaseDeviceAdaptor


# SOURCE: vllm_ascend/device/device_op.py:L1670
# 模块加载期按设备型号定下别名；reshape_and_cache 经它落到 torch_npu 算子。
DeviceOperator: type["BaseDeviceAdaptor"] = get_device_adaptor()
