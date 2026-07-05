#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/fp8_config.py

入口3：fp8 同样「先删 vLLM 原版 fp8 / deepseek_v4_fp8 再替换」，并复用同一个 Config 给
deepseek_v4_fp8 别名注册。三入口注册手法完全一致——继承 QuantizationConfig、实现钩子、启动期注册。
"""

from typing import Any, Optional, cast

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class


# SOURCE: vllm_ascend/quantization/fp8_config.py:L18
def remove_quantization_method():
    if FP8_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(FP8_METHOD)
    if "deepseek_v4_fp8" in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove("deepseek_v4_fp8")


remove_quantization_method()


# SOURCE: vllm_ascend/quantization/fp8_config.py:L28
def create_scheme_for_layer(
    quant_description: dict[str, Any],
    prefix: str,
    layer_type: str,
    packed_modules_mapping: dict[str, Any] | None = None,
):
    """Create a quantization scheme instance for a layer."""
    logger.info_once("Using the vLLM Ascend fp8 Quantization now!")
    quant_type = "FP8"

    # Use registry to get scheme class
    scheme_cls = get_scheme_class(quant_type, layer_type)
    if scheme_cls is not None:
        return scheme_cls(quant_description)

    raise NotImplementedError(f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}.")


@register_quantization_config(FP8_METHOD)
# SOURCE: vllm_ascend/quantization/fp8_config.py:L56
class AscendFp8Config(QuantizationConfig):
    # SOURCE: vllm_ascend/quantization/fp8_config.py:L58
    def __init__(
        self,
        ignore: list[str],
        quant_format: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        self.quant_description = config if config is not None else {}

    # SOURCE: vllm_ascend/quantization/fp8_config.py:L69
    def __repr__(self) -> str:
        return "Fp8Config:\n" + super().__repr__()

    @classmethod
    # SOURCE: vllm_ascend/quantization/fp8_config.py:L72
    def get_name(cls) -> str:
        return FP8_METHOD

    @classmethod
    # SOURCE: vllm_ascend/quantization/fp8_config.py:L76
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float8_e4m3fn, torch.float16, torch.bfloat16]

    @classmethod
    # SOURCE: vllm_ascend/quantization/fp8_config.py:L80
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    @classmethod
    # SOURCE: vllm_ascend/quantization/fp8_config.py:L84
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    # SOURCE: vllm_ascend/quantization/fp8_config.py:L88
    def from_config(cls, config: dict[str, Any]) -> "AscendFp8Config":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))

        return cls(
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    # SOURCE: vllm_ascend/quantization/fp8_config.py:L99
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = FP8_METHOD

            scheme = create_scheme_for_layer(self.quant_description, prefix, "ds_linear", self.packed_modules_mapping)
            quant_method = AscendLinearMethod(scheme)
            return quant_method
        if isinstance(layer, FusedMoE):
            layer.ascend_quant_method = FP8_METHOD
            scheme = create_scheme_for_layer(self.quant_description, prefix, "w4a8_moe", self.packed_modules_mapping)
            quant_method = AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None


# SOURCE: vllm_ascend/quantization/fp8_config.py:L124
# deepseek_v4_fp8 is handled identically to fp8 on Ascend — reuse the same config.
register_quantization_config("deepseek_v4_fp8")(AscendFp8Config)
