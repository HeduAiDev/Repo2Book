# SOURCE: vllm/config/vllm.py
# v3 ch31 脊柱切面：VllmConfig 的两个 property——num_speculative_tokens
# （L564-L575，掩码行数预算的 spec 维来源）与 use_v2_model_runner
# （L577-L658，V1/V2 两条落地路径的真实选择器，m18 诚实性落点），
# 连同三个判据 helper 逐字承载。装配面（真实 VllmConfig 是 ~700 行 dataclass
# 组装，ch03 全文已立）按本章消费字段以 HOST SEAM 裁剪：
# scheduler_config.max_num_seqs / model_config（is_diffusion·skip_tokenizer_init
# ·architectures·is_moe·runner_type）/ structured_outputs_config.
# enable_in_reasoning / parallel_config.prefill_context_parallel_size /
# speculative_config（method·num_speculative_tokens）/ diffusion_config.
# canvas_length。
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

if TYPE_CHECKING:
    from vllm.config import (
        DiffusionConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        StructuredOutputsConfig,
    )

logger = init_logger(__name__)

# SOURCE: vllm/config/vllm.py:L60-L68 DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES —— 逐字
DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "GraniteMoeForCausalLM",
        "InklingForCausalLM",
        "InklingForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        "LongcatFlashNgramForCausalLM",
        "Qwen2MoeForCausalLM",
    }
)

# SOURCE: vllm/config/vllm.py:L72-L79 ROCM_EXCLUDED_V2_MODEL_RUNNER_ARCHITECTURES —— 逐字
# Architectures that default to V1 on ROCm: the V2 runner faults during the
# profile run. VLLM_USE_V2_MODEL_RUNNER=1 still forces V2.
# TODO: fix V2 enablement
ROCM_EXCLUDED_V2_MODEL_RUNNER_ARCHITECTURES = frozenset(
    {
        "KimiK3ForConditionalGeneration",
    }
)


# SOURCE: vllm/config/vllm.py:L90-L100 default_v2_model_runner_architectures —— 逐字
@lru_cache
# SOURCE: vllm/config/vllm.py:L90-L100 default_v2_model_runner_architectures —— 逐字
def default_v2_model_runner_architectures() -> frozenset[str]:
    """Architectures defaulting to the V2 model runner on this platform."""
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        return (
            DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
            - ROCM_EXCLUDED_V2_MODEL_RUNNER_ARCHITECTURES
        )
    return DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES


# SOURCE: vllm/config/vllm.py VllmConfig —— 本章切面（HOST SEAM 装配）
class VllmConfig:
    """VllmConfig 的 ch31 切面：两个 property 逐字 + 消费字段承载。

    真实 VllmConfig（config/vllm.py，ch03 全文已立）是装配一切子配置的
    dataclass；本章只消费六个子配置的字段面（见文件头注）。
    """

    # SOURCE: vllm/config/vllm.py VllmConfig 装配面 —— HOST SEAM 消费字段切面（真实 dataclass 归 ch03）
    def __init__(
        self,
        scheduler_config: "SchedulerConfig | None" = None,
        model_config: "ModelConfig | None" = None,
        speculative_config: "SpeculativeConfig | None" = None,
        diffusion_config: "DiffusionConfig | None" = None,
        parallel_config: "ParallelConfig | None" = None,
        structured_outputs_config: "StructuredOutputsConfig | None" = None,
    ):
        from vllm.config import (
            DiffusionConfig,
            ModelConfig,
            ParallelConfig,
            SchedulerConfig,
            SpeculativeConfig,
            StructuredOutputsConfig,
            _as_config,
        )

        # HOST SEAM：dict → 子配置（测试按消费字段直建）
        self.scheduler_config = _as_config(SchedulerConfig, scheduler_config) or SchedulerConfig()
        self.model_config = _as_config(ModelConfig, model_config) or ModelConfig()
        self.speculative_config = _as_config(SpeculativeConfig, speculative_config)
        self.diffusion_config = _as_config(DiffusionConfig, diffusion_config)
        self.parallel_config = _as_config(ParallelConfig, parallel_config) or ParallelConfig()
        self.structured_outputs_config = (
            _as_config(StructuredOutputsConfig, structured_outputs_config)
            or StructuredOutputsConfig()
        )

    # SOURCE: vllm/config/vllm.py:L563-L575 num_speculative_tokens —— 逐字
    @property
    # SOURCE: vllm/config/vllm.py:L563-L575 num_speculative_tokens —— 逐字
    def num_speculative_tokens(self) -> int:
        if (
            self.speculative_config is not None
            and self.speculative_config.num_speculative_tokens is not None
        ):
            return self.speculative_config.num_speculative_tokens
        if (
            self.diffusion_config is not None
            and self.diffusion_config.canvas_length is not None
        ):
            return self.diffusion_config.canvas_length
        return 0

    # SOURCE: vllm/config/vllm.py:L577-L623 use_v2_model_runner —— 逐字
    @property
    # SOURCE: vllm/config/vllm.py:L577-L623 use_v2_model_runner —— 逐字
    def use_v2_model_runner(self) -> bool:
        use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        if use_v2_model_runner is not None:
            return use_v2_model_runner

        # PCP runtime support is implemented only by the V2 model runner.
        if self.parallel_config.prefill_context_parallel_size > 1:
            return True

        # DSpark is implemented only by the V2 GPU model runner, and DeepSeek-V4
        # is not otherwise a default-V2 architecture, so force V2 for it. If V2
        # is unsupported for the rest of the config, _validate_v2_model_runner
        # raises rather than silently falling back to V1 (which can't run dspark).
        if (
            self.speculative_config is not None
            and self.speculative_config.method == "dspark"
        ):
            return True

        # Mixed sliding/full DFlash drafts need multiple KV groups (V2 only);
        # force V2 as for dspark, since a hybrid target otherwise defaults to V1.
        if self._dflash_needs_multi_kv_group():
            return True

        if self.model_config is not None and self.model_config.is_diffusion:
            return True

        if not self._is_default_v2_model_runner_model():
            return False

        if not HAS_TRITON:
            logger.warning_once(
                "Model Runner V2 requires Triton; using the V1 model runner instead."
            )
            return False

        unsupported = self._get_v2_model_runner_unsupported_features()
        if unsupported:
            logger.warning_once(
                "Model Runner V2 does not yet support %s; using the V1 model "
                "runner instead.",
                ", ".join(unsupported),
            )
            return False

        return True

    # SOURCE: vllm/config/vllm.py:L625-L635 _dflash_needs_multi_kv_group —— 逐字
    def _dflash_needs_multi_kv_group(self) -> bool:
        """Whether a DFlash draft mixes sliding-window and full attention."""
        spec = self.speculative_config
        if spec is None or spec.method != "dflash":
            return False
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is None:
            return False
        layer_types = getattr(draft_config.hf_config, "layer_types", None) or []
        num_sliding = sum(lt == "sliding_attention" for lt in layer_types)
        return 0 < num_sliding < len(layer_types)

    # SOURCE: vllm/config/vllm.py:L637-L658 _is_default_v2_model_runner_model —— 逐字
    def _is_default_v2_model_runner_model(self) -> bool:
        model_config = self.model_config
        if model_config is None:
            return False

        if model_config.runner_type != "generate":
            return False

        architectures = getattr(model_config, "architectures", [])
        default_architectures = default_v2_model_runner_architectures()
        is_default_v2_architecture = any(
            arch in default_architectures for arch in architectures
        )

        if getattr(model_config, "is_hybrid", False) and (
            not is_default_v2_architecture
        ):
            return False

        if getattr(model_config, "is_attention_free", False):
            return False
        return is_default_v2_architecture or not model_config.is_moe

    # SOURCE: vllm/config/vllm.py:L2166 起 _get_v2_model_runner_unsupported_features
    #   —— 消费切片承载：PCP-without-MLA 与 spec 方法两判据逐字；
    #   stock torch.compile / SP+TP / external_launcher PP / parallel_drafting
    #   等分支删（compilation_config/pass_config 的深层装配面，本章不展开）。
    # SOURCE: vllm/config/vllm.py:L2166 起 _get_v2_model_runner_unsupported_features —— 消费切片（PCP/spec 两判据逐字，深层判据删）
    def _get_v2_model_runner_unsupported_features(self) -> list[str]:
        """Collect features not yet supported by the V2 model runner."""
        unsupported: list[str] = []
        model_config = self.model_config
        speculative_config = self.speculative_config

        if self.parallel_config.prefill_context_parallel_size > 1 and not (
            model_config is not None and getattr(model_config, "use_mla", False)
        ):
            unsupported.append("prefill context parallelism")

        if speculative_config is not None:
            # TODO: ngram / ngram_gpu are not supported by the v2 model runner yet
            if speculative_config.method in ("ngram", "ngram_gpu"):
                unsupported.append("ngram/ngram_gpu speculative decoding")
            elif speculative_config.method not in (
                "eagle",
                "eagle3",
                "mtp",
                "dflash",
                "dspark",
            ):
                unsupported.append(f"speculative method '{speculative_config.method}'")

        return unsupported
