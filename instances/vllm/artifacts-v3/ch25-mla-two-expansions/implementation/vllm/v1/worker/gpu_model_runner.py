# SOURCE: vllm/v1/worker/gpu_model_runner.py —— ch25 切面（站 7/站 6 的
# runner 三薄层；ch17/ch21/ch23 已立 runner 主体，本章只取 MLA 相关三方法）：
#   _may_reorder_batch（L1115-L1138 逐字——must_keep 相关）
#   calculate_reorder_batch_threshold（L7220-L7238 逐字——取各组最小）
#   get_kv_cache_spec（L7800-L7837 减法：kv_sharing 分派 → ch23 域）
# SUBTRACTED：runner 其余全量（模型装载/执行/图捕获/调度位）——ch17/ch19/
#   ch21/ch23 域。
from __future__ import annotations

from dataclasses import replace
from functools import reduce
from typing import TYPE_CHECKING, Any, cast

from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import (
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# has_ec_transfer 位（真实 get_kv_cache_spec 首行的 KV connector 预生产检查）
# —— SUBTRACTED：ch16 域（delete[8] 同族；本章 kv_transfer_config 恒 None）


# SOURCE: vllm/v1/worker/gpu_model_runner.py GPUModelRunner —— 减法薄层
#   （三个真实方法 + 其消费的实例位；方法主体逐字）
class GPUModelRunnerSlice:
    """ch25 切面：MLA 相关三方法挂在 runner 承载类上（真实方法主体逐字）。"""

    # ── 站 7：每拍批重排 ───────────────────────────────────────────────

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1115-L1138 _may_reorder_batch
    #   —— 逐字（docstring 原话点名 MLA：compute-bound vs memory-bound 分离）
    def _may_reorder_batch(self, scheduler_output) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1115-L1138（锚点双置：声明上方注释同文）
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7220-L7238 calculate_
    #   reorder_batch_threshold —— 逐字（must_keep 相关：取各组 builder
    #   自报的最小值；_attn_group_iterator 的组迭代面归 ch21/ch17——
    #   本章以实例位承载）
    def calculate_reorder_batch_threshold(self) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7220-L7238（锚点双置：声明上方注释同文）
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        min_none_high = (
            lambda a, b: a if b is None else b if a is None else min(a, b)
        )

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    # ── 站 6：spec 收集（分组原料）────────────────────────────────────

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7800-L7837 get_kv_cache_spec
    #   —— 减法子集（has_ec_transfer 早退与 kv_sharing 分派 → ch16/ch23；
    #   主体逐字）
    def get_kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7800-L7837（锚点双置：声明上方注释同文）
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            # SUBTRACTED: kv_sharing_target_layer_name 的跨层共享分派
            #   （L7814-L7825）——ch23 域（shared_kv_cache_layers 面）
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                if isinstance(spec, AttentionSpec):
                    backend = attn_module.get_attn_backend()
                    # indexes_kv_by_block_stride() -> get_kv_cache_stride_order()
                    # -> get_kv_cache_layout() needs the current vLLM config.
                    with set_current_vllm_config(self.vllm_config):
                        indexes = backend.indexes_kv_by_block_stride()
                    spec = replace(spec, indexes_kv_by_block_stride=indexes)
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec
