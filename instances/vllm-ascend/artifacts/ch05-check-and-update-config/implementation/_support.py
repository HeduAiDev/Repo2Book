# 章 ch05 精简版 —— 外部依赖的「只做减法」替身（让纯 Python 控制流能在 host 上跑）
#
# 本文件不属于本章被解读的主线代码，只是把真实源码 import 的若干外部符号
# （vLLM 日志器、两个编译枚举、注意力后端枚举、昇腾设备探测助手）替换成最小可运行替身，
# 以便 _fix_incompatible_config / check_and_update_config / 三级取值 能在无 NPU/CANN 的 host 上
# 按**与真源码同一控制流**跑通。每个替身都标注其真实源码出处与被删内容。
#
# SUBTRACTED: 真源码用 `from vllm.logger import logger`（vLLM 自带带 info_once 去重的 logger）。
#             这里换成 stdlib logging，并补一个 info_once 别名——仅日志副作用，不影响任何控制流分支。
import logging

logger = logging.getLogger("vllm_ascend")
if not hasattr(logger, "info_once"):
    logger.info_once = logger.info  # vLLM logger 的去重日志，控制流无关，退化为普通 info


class CompilationMode:
    # SOURCE: vllm/config/compilation.py:L37 (class CompilationMode(enum.IntEnum))
    # SUBTRACTED: 原枚举含 STOCK_TORCH_COMPILE=2 等成员；本章 check_and_update_config 只比较
    #             NONE / VLLM_COMPILE，故只保留这两个成员（值与基座一致）。
    NONE = 0
    VLLM_COMPILE = 3


class CUDAGraphMode:
    # SOURCE: vllm/config/compilation.py:L53 (class CUDAGraphMode(enum.Enum))
    # SUBTRACTED: 原枚举含 PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE 及
    #             requires_piecewise_compilation()/has_full_cudagraphs() 方法；本章保留的
    #             cudagraph 改写片段只用到 NONE（其余分支属删减计划批准的周边）。
    NONE = 0


class AttentionBackendEnum:
    # SOURCE: vllm/v1/attention/backends/registry.py:L34 (class AttentionBackendEnum(Enum))
    # SUBTRACTED: 原枚举登记所有后端；_fix_incompatible_config 段8 只比较 FLASH_ATTN
    #             （训推一致时保留它、其余 backend reset 为 None），故只留 FLASH_ATTN。
    FLASH_ATTN = "FLASH_ATTN"


class AscendDeviceType:
    # SOURCE: vllm_ascend/utils.py:L768 (class AscendDeviceType(Enum))
    # SUBTRACTED: 原枚举含 A2 / _310P 等并带 npu-smi 探测；host 无 NPU，固定为非 310P。
    A2 = 1
    _310P = 2


def is_310p():
    # SOURCE: vllm_ascend/utils.py:L122
    # SUBTRACTED: 真实现 `return get_ascend_device_type() == AscendDeviceType._310P`，
    #             经 npu-smi 查 SOC 版本；host 无 NPU，恒定返回 False（走默认 NPUWorker 分支）。
    return False


def get_ascend_device_type():
    # SOURCE: vllm_ascend/utils.py:L812
    # SUBTRACTED: 真实现探测真实 SOC；host 上恒定返回 A2（非 310P）。
    return AscendDeviceType.A2


def refresh_block_size(vllm_config):
    # SOURCE: vllm_ascend/utils.py:L1246
    # SUBTRACTED: 真实现按昇腾约束重算 cache_config.block_size；与本章「配置改写总闸」主线
    #             无关，退化为 no-op（worker_cls 段尾调用它，仅保留调用点以忠实控制流）。
    return None
