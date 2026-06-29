# vllm_ascend/utils.py —— subtract-only 精简版（ch15 配角：DP 同步是否可跳过 + soc 判定 + 谓词）
#
# 只保留本章用到的：AscendDeviceType 枚举、几个「按 config 判定」的纯 Python 谓词
# （is_moe_model / is_drafter_moe_model / enable_sp / flashcomm2_enable / has_layer_idx /
# get_ascend_device_type / speculative_enable_dispatch_gmm_combine_decode /
# is_hierarchical_communication_enabled），以及 must_keep 的 should_skip_allreduce_across_dp_group。
# 其余数百个工具函数与本章正交，整体折叠。
import os
from enum import Enum
from typing import Any

import torch
from packaging.version import InvalidVersion, Version  # noqa: F401

# SOURCE: vllm_ascend/utils.py（节选）
from vllm.config import VllmConfig
from vllm_ascend.ascend_config import get_ascend_config

_IS_MOE_MODEL = None
_IS_DRAFTER_MOE_MODEL = None
_ENABLE_SP = None
_HAS_LAYER_IDX = None
_ascend_device_type = None


# SOURCE: vllm_ascend/utils.py:L558
def vllm_version_is(target_vllm_version: str):
    import vllm

    vllm_version = vllm.__version__
    # SUBTRACTED: envs_ascend.VLLM_VERSION 覆盖与 InvalidVersion 报错（utils.py:L559-L573）——
    #   环境变量覆盖与异常文案，删后版本判定主路径不变。
    return Version(vllm_version) == Version(target_vllm_version)


# SOURCE: vllm_ascend/utils.py:L818
def lmhead_tp_enable() -> bool:
    return get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size > 0


# SOURCE: vllm_ascend/utils.py:L768
class AscendDeviceType(Enum):
    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


# SOURCE: vllm_ascend/utils.py:L812
def get_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()
    return _ascend_device_type


# SOURCE: vllm_ascend/utils.py:L778
def _init_ascend_device_type():
    global _ascend_device_type
    from vllm_ascend import _build_info  # type: ignore

    device_type = getattr(_build_info, "__device_type__", None)
    if device_type is None:
        soc_version = getattr(_build_info, "__soc_version__", "ASCEND910B1").upper()
        device_type = "_310P" if "310P" in soc_version else "A2"
    _ascend_device_type = AscendDeviceType[device_type]


# SOURCE: vllm_ascend/utils.py:L847
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except AssertionError:
            vllm_config = None

    additional_config = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
    refresh = additional_config.get("refresh", False) if additional_config else False

    if _ENABLE_SP is None or refresh:
        if additional_config is not None and "enable_flashcomm1" in additional_config:
            _ENABLE_SP = bool(additional_config["enable_flashcomm1"])
        else:
            _ENABLE_SP = get_ascend_config().enable_flashcomm1
        # SUBTRACTED: enable_flashcomm1 取不到时回落 envs_ascend 与 shared_expert_dp 联动分支
        #   （vllm_ascend/utils.py:L870-L878）—— 环境变量回落，删后主路径判定不变。
    return bool(_ENABLE_SP)


# SOURCE: vllm_ascend/utils.py:L1165
def flashcomm2_enable() -> bool:
    config_val = get_ascend_config().enable_flashcomm2_parallel_size
    return config_val > 0


# SOURCE: vllm_ascend/utils.py:L1155
def has_layer_idx(model_instance: "torch.nn.Module") -> bool:
    if model_instance is None:
        return False

    global _HAS_LAYER_IDX
    if _HAS_LAYER_IDX is None:
        _HAS_LAYER_IDX = hasattr(model_instance, "model") and hasattr(model_instance.model, "start_layer")
    return _HAS_LAYER_IDX


# SOURCE: vllm_ascend/utils.py:L881
def is_moe_model(vllm_config: VllmConfig):
    """Checks if the model is a MoE model by config"""
    global _IS_MOE_MODEL
    if _IS_MOE_MODEL is None:
        model_configs = vllm_config.model_config.hf_text_config.to_dict()
        _IS_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_MOE_MODEL


# SOURCE: vllm_ascend/utils.py:L890
def is_drafter_moe_model(vllm_config: VllmConfig):
    """Checks if the drafter model is a MoE model by config"""
    global _IS_DRAFTER_MOE_MODEL
    if _IS_DRAFTER_MOE_MODEL is None:
        model_configs = vllm_config.speculative_config.draft_model_config.hf_text_config.to_dict()
        _IS_DRAFTER_MOE_MODEL = _is_contain_expert(model_configs)
        if not model_configs or not model_configs.get("architectures"):
            return _IS_DRAFTER_MOE_MODEL
        if "Eagle3DeepseekV2ForCausalLM" in model_configs["architectures"]:
            _IS_DRAFTER_MOE_MODEL = False
    return _IS_DRAFTER_MOE_MODEL


# SOURCE: vllm_ascend/utils.py:L903
def speculative_enable_dispatch_gmm_combine_decode(vllm_config: VllmConfig) -> bool:
    """When draft contains MOE Arch and non-w8a8, disable dispatch_gmm_combine_decode."""
    if vllm_config.speculative_config is None:
        return True
    speculative_method = getattr(vllm_config.speculative_config, "method", None)
    if speculative_method in [None, "ngram", "suffix"]:
        return True
    if speculative_method in ["eagle", "eagle3"]:
        if is_drafter_moe_model(vllm_config):
            draft_model_config = vllm_config.speculative_config.draft_model_config
            hf_text_config = draft_model_config.hf_text_config
            quant_type = getattr(hf_text_config, "moe_quantize", None)
            if quant_type is None:
                quant_type = getattr(hf_text_config, "quantize", None)
            return quant_type == "w8a8_dynamic"
        else:
            return True
    if speculative_method == "mtp":
        mtp_quant_type = getattr(vllm_config.model_config.hf_text_config, "mtp_quantize", None)
        return mtp_quant_type == "w8a8_dynamic"
    return False


# SOURCE: vllm_ascend/utils.py:L926
def _is_contain_expert(config: Any):
    if isinstance(config, dict):
        for k, v in config.items():
            if "expert" in str(k):
                return True
            if _is_contain_expert(v):
                return True
    return False


# SOURCE: vllm_ascend/utils.py:L1077
def is_hierarchical_communication_enabled():
    return (
        os.getenv("HCCL_INTRA_ROCE_ENABLE", "") == "0" and os.getenv("HCCL_INTRA_PCIE_ENABLE", "") == "1"
    ) or get_ascend_config().enable_mc2_hierarchy_comm


# SOURCE: vllm_ascend/utils.py:L1083
def should_skip_allreduce_across_dp_group(vllm_config, is_draft_model: bool = False) -> bool:
    """Decide whether to skip the all-reduce across the DP group.

    Skipping is applicable for all dense models and for moe models only on ranks
    that act as KV consumers. We skip the DP all-reduce when either:
    - Both the prefill and decode communication methods are MC2 (or FUSED_MC2), or
    - Decode requires MC2 and ascend_config.recompute_scheduler_enable is True.

    Skipping means each rank may have a different number of tokens, so MC2 needs
    a non-zero global_bs and must NOT receive mc2_mask.
    """
    if is_hierarchical_communication_enabled():
        return False

    # For dense models, since we don't actually need dp communication, we simply skip it.
    # This usually happens when main model is moe while eagle draft model is dense.
    is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
    if not is_context_moe_model:
        return True

    # Only applicable to MoE models on KV consumer ranks.
    is_kv_consumer = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer
    if not is_kv_consumer:
        return False

    from vllm_ascend.ascend_forward_context import select_moe_comm_method
    from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType

    def needs_mc2(n: int) -> bool:  # SOURCE: vllm_ascend/utils.py:L1113
        return select_moe_comm_method(n, vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}

    compilation_config = vllm_config.compilation_config
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    uniform_decode_query_len = 1 if not speculative_config else 1 + speculative_config.num_speculative_tokens
    decode_max_num_seqs = getattr(scheduler_config, "decode_max_num_seqs", 0)
    max_num_reqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)

    # Determine whether decode must use MC2. Use max cudagraph capture size
    # if available, otherwise use the maximal uniform decode token count.
    if compilation_config.cudagraph_capture_sizes:
        potential_max_tokens = max(
            compilation_config.max_cudagraph_capture_size,
            min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len,
            ),
        )
        # SUBTRACTED: potential_max_tokens 与 max_cudagraph_capture_size 不一致时的 warning_once
        #   日志（vllm_ascend/utils.py:L1135-L1144）—— 仅告警，删后判定逻辑不变。
    else:
        potential_max_tokens = min(max_num_reqs * uniform_decode_query_len, 512)

    decode_must_use_mc2 = needs_mc2(potential_max_tokens)
    # For prefill, use the scheduler's max_num_batched_tokens for a single batch.
    prefill_must_use_mc2 = needs_mc2(scheduler_config.max_num_batched_tokens)
    # Skip all-reduce if decode requires MC2 and either prefill also
    # requires MC2 or recompute-based scheduler is enabled.
    return decode_must_use_mc2 and (prefill_must_use_mc2 or get_ascend_config().recompute_scheduler_enable)
