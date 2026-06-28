"""Subtract-only companion — 设备分代（横切线索）+ 平台量化常量。

规范源码：vllm_ascend/utils.py

设备分代是横切关注点：A2/A3/A5 是训推一体卡的几代，_310P 是纯推理卡（能力受限，
自带 _310p/ 子包另写实现）。is_310p() 是全代码库最常见的‘设备分代’分流入口。

设备分代 = 构建期烙印 + 运行期复核：
  _init_ascend_device_type 从打包时写入的 vllm_ascend._build_info 读出本包面向哪代（缺省 A2）；
  check_ascend_device_type 再用 torch_npu 真问硬件 soc_version 复核、不符即 assert 报错。
"""
from enum import Enum

# SOURCE: vllm_ascend/utils.py:L48-L50 —— 平台 supported_quantization 引用的方法名常量
ASCEND_QUANTIZATION_METHOD = "ascend"
COMPRESSED_TENSORS_METHOD = "compressed-tensors"
FP8_METHOD = "fp8"


# SOURCE: vllm_ascend/utils.py:L768-L772
class AscendDeviceType(Enum):
    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


# SOURCE: vllm_ascend/utils.py:L775 —— 模块级分代缓存（体现‘只检测一次’）
_ascend_device_type = None


# SOURCE: vllm_ascend/utils.py:L778-L787
def _init_ascend_device_type():
    global _ascend_device_type
    from vllm_ascend import _build_info  # type: ignore

    device_type = getattr(_build_info, "__device_type__", None)
    if device_type is None:
        soc_version = getattr(_build_info, "__soc_version__", "ASCEND910B1").upper()
        device_type = "_310P" if "310P" in soc_version else "A2"
    _ascend_device_type = AscendDeviceType[device_type]


# SOURCE: vllm_ascend/utils.py:L790-L811
def check_ascend_device_type(soc_version=None):
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()

    # SUBTRACTED: 原为 `soc_version = torch_npu.npu.get_soc_version()` —— 真问硬件；
    #   host 无 torch_npu/CANN，精简版把它降为入参以便单测‘运行期复核’的区间映射表。
    #   原 vllm_ascend/utils.py:L794
    if 220 <= soc_version <= 225:
        cur_device_type = AscendDeviceType.A2
    elif 250 <= soc_version <= 255:
        cur_device_type = AscendDeviceType.A3
    elif 200 <= soc_version <= 205:
        cur_device_type = AscendDeviceType._310P
    elif soc_version == 260:
        cur_device_type = AscendDeviceType.A5
    else:
        raise RuntimeError(f"Can not support soc_version: {soc_version}.")

    assert _ascend_device_type == cur_device_type, (
        f"Current device type: {cur_device_type} does not match the installed version's device type: "
        f"{_ascend_device_type}, please check your installation package."
    )


# SOURCE: vllm_ascend/utils.py:L814-L818
def get_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()
    return _ascend_device_type


# SOURCE: vllm_ascend/utils.py:L122-L123
def is_310p():
    return get_ascend_device_type() == AscendDeviceType._310P
