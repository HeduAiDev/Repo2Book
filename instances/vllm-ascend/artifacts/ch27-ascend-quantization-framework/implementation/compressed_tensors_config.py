#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/compressed_tensors_config.py

入口2 的「先删后替换」手法：vLLM 内置已占用 'compressed-tensors' 方法名，昇腾先从
QUANTIZATION_METHODS 移除原版，再用同名 @register_quantization_config 注册自己的 Config 顶替。
本章只聚焦这一注册手法；AscendCompressedTensorsConfig 把 LLM-Compressor 格式适配到昇腾 scheme
的具体逻辑（target_scheme_map 解析 / get_quant_method 等）属于本章范围之外。
"""

from typing import Any

# SUBTRACTED: torch / logger / cast / QuantizeMethodBase 与 compressed_tensors 适配相关 import
#   （find_matched_target / QuantizationArgs 等，原 L20-L38）——仅被下方已减去的 Config 方法体使用。
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD


# Remove the original compressed_tensors method to replace with our implementation
# SOURCE: vllm_ascend/quantization/compressed_tensors_config.py:L42
def _remove_quantization_method():
    if COMPRESSED_TENSORS_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(COMPRESSED_TENSORS_METHOD)


_remove_quantization_method()


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
# SOURCE: vllm_ascend/quantization/compressed_tensors_config.py:L52
class AscendCompressedTensorsConfig(QuantizationConfig):
    """Config class for LLM-Compressor (compressed_tensors) quantization on Ascend.

    This class adapts the compressed_tensors format to work with Ascend's
    quantization implementations.
    """

    # SOURCE: vllm_ascend/quantization/compressed_tensors_config.py:L60
    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.quant_description = config

    # SOURCE: vllm_ascend/quantization/compressed_tensors_config.py:L73
    def get_name(self) -> str:
        return "compressed-tensors"

    @classmethod
    # SOURCE: vllm_ascend/quantization/compressed_tensors_config.py:L85
    def get_config_filenames(cls) -> list[str]:
        return []

    # SUBTRACTED: 本 Config 的其余方法——get_supported_act_dtypes / from_config /
    #   _quantization_scheme_map_from_config（解析 config_groups 的 target_scheme_map）/
    #   get_quant_method（按 LinearBase/FusedMoE 分发到昇腾 scheme）等（原 L76-L429）。
    #   入口2 在本章只演示「先删后替换」的注册手法；适配 LLM-Compressor 格式的细节超出本章范围。
