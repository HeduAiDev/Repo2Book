# vllm_ascend/utils.py —— subtract-only 精简版（ch17 全栈分流总开关）
#
# 本章主线第 0 步：310P 的全栈特化由一个布尔 is_310p() 在运行期分流。它读的是
# 构建期烧进 _build_info 的 SOC 版本字符串（"Ascend310P3"），把 "310P" 子串映射到
# AscendDeviceType._310P 枚举并缓存。platform / attention / distributed / ops 各处只
# 用这一个布尔决定是否走 310 路径——"差异收进 _310p/，外部只留一个布尔分流"。
#
# 这里只保留与 310P 分流相关的常量与枚举/初始化函数；utils.py 其余数百个
# 与本章正交的辅助（stream 管理、custom op、量化、对齐工具等）按减法折叠。
from enum import Enum

# SOURCE: vllm_ascend/utils.py:L51
SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]

# SOURCE: vllm_ascend/utils.py:L54-L55
ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

# SUBTRACTED: utils.py:L47-L79 其余模块级常量/缓存（COMPILATION_PASS_KEY、各 stream
#   句柄、_CUSTOM_OP_ENABLED 等）—— 与 310P 分流主题正交。


# SOURCE: vllm_ascend/utils.py:L768-L772
class AscendDeviceType(Enum):
    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


# SOURCE: vllm_ascend/utils.py:L775
_ascend_device_type = None


# SOURCE: vllm_ascend/utils.py:L778-L786
def _init_ascend_device_type():
    global _ascend_device_type
    from vllm_ascend import _build_info  # type: ignore

    device_type = getattr(_build_info, "__device_type__", None)
    if device_type is None:
        soc_version = getattr(_build_info, "__soc_version__", "ASCEND910B1").upper()
        # "310P 子串 → _310P 枚举"的真正分流点：构建期烧进 _build_info，首次访问初始化缓存。
        device_type = "_310P" if "310P" in soc_version else "A2"
    _ascend_device_type = AscendDeviceType[device_type]


# SUBTRACTED: check_ascend_device_type()（utils.py:L789-L809）—— 运行期用 torch_npu
#   的 soc_version 数值区间二次校验缓存与硬件一致；host 无 torch_npu 不可跑，且与
#   "字符串→枚举分流"主线正交。


# SOURCE: vllm_ascend/utils.py:L812-L816
def get_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()
    return _ascend_device_type


# SOURCE: vllm_ascend/utils.py:L122-L123
def is_310p():
    return get_ascend_device_type() == AscendDeviceType._310P
