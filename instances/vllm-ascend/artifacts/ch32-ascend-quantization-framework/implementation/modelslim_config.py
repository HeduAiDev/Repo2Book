#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/modelslim_config.py

入口1：@register_quantization_config('ascend') 把 AscendModelSlimConfig 注进 vLLM 量化注册表。
- get_quant_method(layer, prefix)：按 isinstance(layer) 四岔分发到 scheme + wrapper（总枢纽）。
- get_linear_quant_type / get_quant_type_for_layer / create_scheme_for_layer：从 ModelSlim 生成的
  quant_description（quant_model_description.json）逐层查出 quant_type，再经 registry 取 scheme。
"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Optional

import regex as re
import torch
from transformers import PretrainedConfig
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
from vllm.model_executor.models.utils import WeightsMapper

from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD

from .methods import get_scheme_class

# The config filename that ModelSlim generates after quantizing a model.
MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"

# key: model_type / value: dict of fused module name -> list of original module names
# SUBTRACTED: 二十余个模型的 packed_modules_model_mapping 条目（原 modelslim_config.py:L66-L259）——
#   纯静态同质表；保留 qwen3_moe + deepseek_v3 两个代表即可讲清「融合模块名->原始模块名」用途。
packed_modules_model_mapping: dict[str, dict[str, list[str]]] = {
    "qwen3_moe": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    },
    "deepseek_v3": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    },
}

# SUBTRACTED: QUANT_MODEL_PREFIX_MAPPINGS / QUANT_MODEL_SUBSTR_MAPPINGS（原 L262-L281）——
#   checkpoint 命名规整到 vLLM 命名的字符串映射，与本章「注册+分发+适配」主线无关。


# SOURCE: vllm_ascend/quantization/modelslim_config.py:L284
def get_packed_modules_mapping(model_type: str) -> dict[str, list[str]]:
    """Get packed modules mapping for a model type."""
    return packed_modules_model_mapping.get(model_type, {})


# SOURCE: vllm_ascend/quantization/modelslim_config.py:L297
def get_linear_quant_type(
    quant_description: dict[str, Any], prefix: str, packed_modules_mapping: dict[str, Any]
) -> str | None:
    """Determine the quantization type for a linear layer."""
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name) for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        for shard_prefix in shard_prefixes:
            shard_quant_type = quant_description[shard_prefix + ".weight"]

            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                err_msg = (
                    f"Not all shards of {prefix} are quantized with same quant type. "
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard "
                    f"uses {quant_type}. Please check quantization config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
    else:
        quant_type = quant_description[prefix + ".weight"]
    return quant_type


# SOURCE: vllm_ascend/quantization/modelslim_config.py:L334
def get_quant_type_for_layer(
    quant_description: dict[str, Any],
    prefix: str,
    layer_type: str,
    packed_modules_mapping: dict[str, Any] | None = None,
) -> str | None:
    """Determine the quantization type for a layer."""
    if packed_modules_mapping is None:
        packed_modules_mapping = dict()
    # Attention
    if layer_type == "attention":
        layer_indexer_quant_type = quant_description.get(f"{prefix}.indexer.quant_type")
        if layer_indexer_quant_type is not None:
            return layer_indexer_quant_type
        if "fa_quant_type" in quant_description:
            return quant_description["fa_quant_type"]
        if "indexer_quant_type" in quant_description:
            return quant_description["indexer_quant_type"]
    # Linear / MoE
    return get_linear_quant_type(quant_description, prefix, packed_modules_mapping)


# SOURCE: vllm_ascend/quantization/modelslim_config.py:L366
def create_scheme_for_layer(
    quant_description: dict[str, Any],
    prefix: str,
    layer_type: str,
    packed_modules_mapping: dict[str, Any] | None = None,
):
    """Create a quantization scheme instance for a layer."""
    logger.info_once("Using the vLLM Ascend modelslim Quantization now!")
    quant_type = get_quant_type_for_layer(quant_description, prefix, layer_type, packed_modules_mapping)

    if quant_type is None:
        err_msg = f"Could not determine quantization type for layer {prefix} (layer_type={layer_type})."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Use registry to get scheme class
    scheme_cls = get_scheme_class(quant_type, layer_type)
    if scheme_cls is not None:
        return scheme_cls()

    err_msg = f"Currently, vLLM Ascend doesn't support quant_type={quant_type} for layer_type={layer_type}."
    logger.error(err_msg)
    raise NotImplementedError(err_msg)


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
# SOURCE: vllm_ascend/quantization/modelslim_config.py:L401
class AscendModelSlimConfig(QuantizationConfig):
    """Config class for Ascend ModelSlim quantization.

    This class is a general class that parses quantization configs
    that are supported on Ascend hardware, specifically for models
    quantized using the ModelSlim tool.
    """

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L410
    def __init__(self, quant_config: dict[str, Any] | None = None):
        super().__init__()
        self.quant_description = quant_config if quant_config is not None else {}
        self._apply_extra_quant_adaptations()
        self.model_type: str | None = None
        self.hf_to_vllm_mapper: WeightsMapper | None = None
        self._mapper_applied = False
        self._add_kvcache_quant_metadata()

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L419
    def __repr__(self) -> str:
        return "AscendModelSlimConfig:\n" + super().__repr__()

    @classmethod
    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L422
    def get_name(cls) -> str:
        return ASCEND_QUANTIZATION_METHOD

    @classmethod
    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L426
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L430
    def get_min_capability(cls) -> int:
        logger.error("Ascend hardware does not support 'get_min_capability' feature.")
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    @classmethod
    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L435
    def get_config_filenames(cls) -> list[str]:
        # Return empty list so that vllm's get_quant_config() skips the
        # file-based lookup; the config file is loaded in maybe_update_config().
        return []

    @classmethod
    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L444
    def from_config(cls, config: dict[str, Any]) -> "AscendModelSlimConfig":
        return cls(config)

    @classmethod
    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L448
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config: Any = None) -> str | None:
        if hf_quant_cfg is not None:
            quant_method = hf_quant_cfg.get("quant_method", None)
            if not quant_method and torch.npu.is_available():
                return ASCEND_QUANTIZATION_METHOD
        return None

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L456
    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """Apply the vLLM model-specific mapper to this quantization config."""
        if self._mapper_applied and self.hf_to_vllm_mapper is hf_to_vllm_mapper:
            return

        self.hf_to_vllm_mapper = hf_to_vllm_mapper
        self._mapper_applied = True

        if self.quant_description:
            self.quant_description = hf_to_vllm_mapper.apply_dict(self.quant_description)
            self._add_kvcache_quant_metadata()
            logger.info("Applied hf_to_vllm_mapper to quant_description keys")

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L479
    def get_cache_scale(self, name: str) -> str | None:
        """Map checkpoint C8 KV scale/offset names to vLLM parameter names."""
        if self.quant_description.get("kv_cache_type") != "C8":
            return None
        _C8_SCALE_MAPPING = {
            "k_proj.kv_cache_scale": "attn.k_cache_scale",
            "k_proj.kv_cache_offset": "attn.k_cache_offset",
            "v_proj.kv_cache_scale": "attn.v_cache_scale",
            "v_proj.kv_cache_offset": "attn.v_cache_offset",
        }
        for src_suffix, dst_suffix in _C8_SCALE_MAPPING.items():
            if name.endswith(src_suffix):
                return name[: -len(src_suffix)] + dst_suffix
        return None

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L494
    def quant_prefix_mapper(self, model_type: str, prefix: str) -> str:
        self.model_type = model_type
        # SUBTRACTED: lm_head 重命名 + QUANT_MODEL_PREFIX/SUBSTR_MAPPINGS 的模型专属前缀/子串改写
        #   （把 checkpoint 命名规整到 vLLM 命名，原 L498-L509）——与主线无关，原样返回 prefix。
        return prefix

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L512
    def get_quant_method(self, layer: torch.nn.Module, prefix: str, tid2eid=None) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendEmbeddingMethod,
            AscendFusedMoEMethod,
            AscendKVCacheMethod,
            AscendLinearMethod,
        )

        vllm_config = get_current_vllm_config()
        model_type = vllm_config.model_config.hf_config.model_type

        # SUBTRACTED: minimax/minimax_m2、bailing_hybrid 的 prefix.replace 命名适配（原 L523-L537）——
        #   模型专属命名规整，不改变下方四岔分发骨架。
        if model_type in packed_modules_model_mapping:
            self.packed_modules_mapping = packed_modules_model_mapping[model_type]
        prefix = self.quant_prefix_mapper(model_type, prefix)
        # SUBTRACTED: 每个分支的 logger.debug 选路日志（原 L547/L550/L556/.../L580）。

        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
                # Delayed import to avoid circular import
                from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

                return AscendUnquantizedLinearMethod()
            scheme = create_scheme_for_layer(self.quant_description, prefix, "linear", self.packed_modules_mapping)
            return AscendLinearMethod(scheme)
        elif isinstance(layer, AttentionLayerBase) and (
            self.is_fa_quant_layer(prefix) or self.is_indexer_quant_layer(prefix)
        ):
            scheme = create_scheme_for_layer(self.quant_description, prefix, "attention", self.packed_modules_mapping)
            return AscendKVCacheMethod(scheme)
        # SUBTRACTED: AttentionLayerBase + is_c8_quant_layer 的 C8 专属分支——返回
        #   AscendKVCacheMethod(AscendC8KVCacheAttentionMethod(...))（原 L558-L562）；
        #   保留上面 fa/indexer 那一支即足以展示「attention 量化也走同一适配器范式」。
        elif isinstance(layer, FusedMoE):
            if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
                # Delayed import to avoid circular import
                from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            scheme = create_scheme_for_layer(self.quant_description, prefix, "moe", self.packed_modules_mapping)
            return AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid)
        elif isinstance(layer, VocabParallelEmbedding):
            if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
                return UnquantizedEmbeddingMethod()
            scheme = create_scheme_for_layer(self.quant_description, prefix, "linear", self.packed_modules_mapping)
            return AscendEmbeddingMethod(scheme)
        return None

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L583
    def is_layer_skipped_ascend(self, prefix: str, fused_mapping: Mapping[str, list[str]] = MappingProxyType({})):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name) for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = any(
                key.startswith(prefix) and key.endswith(".weight") and value == "FLOAT"
                for key, value in self.quant_description.items()
            )

        assert is_skipped is not None
        return is_skipped

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L612
    def is_fa_quant_layer(self, prefix):
        if self.enable_fa_quant:
            layer_id_str = "".join(re.findall(r"\.(\d+)\.", prefix))
            if layer_id_str.isdigit() and int(layer_id_str) in self.kvcache_quant_layers:
                return True
        return False

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L630
    def is_indexer_quant_layer(self, prefix):
        if self.enable_indexer_quant:
            layer_id_str = "".join(re.findall(r"\.(\d+)\.", prefix))
            if layer_id_str.isdigit() and int(layer_id_str) in self.indexer_quant_layers:
                return True
        return False

    # SUBTRACTED: is_c8_quant_layer / enabling_fa_quant / get_kv_quant_dtype / get_kv_quant_split_factor
    #   （C8/fa KV cache 量化的判定与 dtype/split 计算，原 L619-L660）——KV cache 量化算法细节，
    #   是本章可选深水区；上面 fa/indexer 判定与 AscendKVCacheMethod 分发入口已足够说明范式。

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L662
    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        """Load the ModelSlim quantization config from model directory."""
        from vllm_ascend.quantization.utils import get_model_file

        # If quant_description is already populated (e.g. from from_config()),
        # there is nothing to do.
        if self.quant_description:
            return

        # Try to get the config file (local or remote)
        config_path = get_model_file(model_name, MODELSLIM_CONFIG_FILENAME, revision=revision)

        if config_path is not None:
            with open(config_path) as f:
                self.quant_description = json.load(f)
            self._apply_extra_quant_adaptations()
            self._add_kvcache_quant_metadata()
            return

        # SUBTRACTED: 诊断用 glob 扫 *.json + 长篇用户友好 ValueError 提示文案（原 L704-L760）——
        #   纯错误信息字符串拼接，不影响逻辑；保留「找不到则报错」语义。
        raise ValueError(
            f"ModelSlim quantization config '{MODELSLIM_CONFIG_FILENAME}' not found for model '{model_name}'."
        )

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L762
    def _apply_extra_quant_adaptations(self) -> None:
        """Apply extra adaptations to the quant_description dict.

        This handles known key transformations such as shared_head and
        weight_packed mappings.
        """
        # SUBTRACTED: hc_head_fn 模型专属的批量 key 改写（补 model. 前缀 / attn->self_attn / ffn->mlp /
        #   w1/w2/w3->gate/down/up_proj / head->lm_head / embed->embed_tokens，原 L768-L809）。
        extra_quant_dict = {}
        for k in self.quant_description:
            if "shared_head" in k:
                new_k = k.replace(".shared_head.", ".")
                extra_quant_dict[new_k] = self.quant_description[k]
            if "weight_packed" in k:
                new_k = k.replace("weight_packed", "weight")
                extra_quant_dict[new_k] = self.quant_description[k]
        self.quant_description.update(extra_quant_dict)

    # SOURCE: vllm_ascend/quantization/modelslim_config.py:L821
    def _add_kvcache_quant_metadata(self):
        fa_quant_type = self.quant_description.get("fa_quant_type", "")
        self.enable_fa_quant = fa_quant_type != ""
        self.kvcache_quant_layers = []
        indexer_quant_type = self.quant_description.get("indexer_quant_type", "")
        self.enable_indexer_quant = indexer_quant_type != ""
        self.indexer_quant_layers = []
        kv_quant_type = self.quant_description.get("kv_cache_type", "")
        self.enable_c8_quant = kv_quant_type == "C8"
        self.c8_quant_layers = []
        if self.enable_fa_quant or self.enable_indexer_quant or self.enable_c8_quant:
            for key in self.quant_description:
                _id = "".join(re.findall(r"\.(\d+)\.", key))
                if "fa_k.scale" in key:
                    self.kvcache_quant_layers.append(int(_id))
                if "indexer.quant_type" in key:
                    self.indexer_quant_layers.append(int(_id))
                if "k_proj.kv_cache_scale" in key:
                    self.c8_quant_layers.append(int(_id))
