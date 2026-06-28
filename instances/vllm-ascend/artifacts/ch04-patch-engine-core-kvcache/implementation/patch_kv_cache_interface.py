# 案例2 — MLAAttentionSpec 子类化扩展 DSA / Sparse-C8（3 元组 KV → 4 元组）
# 技法①整类替换：子类整体重写 page_size_bytes，末尾三处模块属性重绑（含技法⑤from-import 别名）。
# subtract-only：与 vllm_ascend/patch/platform/patch_kv_cache_interface.py 同名/同结构/同控制流。
#
# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L1-L18
from dataclasses import dataclass, field

import torch
import vllm.model_executor.layers.attention.mla_attention
import vllm.v1.kv_cache_interface
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L21-L22
def _get_c8_k_cache_dtype() -> torch.dtype:
    return torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else torch.int8


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L25-L26
def _get_c8_k_scale_cache_dtype() -> torch.dtype:
    return torch.float32 if get_ascend_device_type() == AscendDeviceType.A5 else torch.float16


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L29-L195
@dataclass(frozen=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLAAttentionSpec extended to support DSA models, with optional Sparse C8 support.

    When Sparse C8 is enabled, the KV cache tuple changes from
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: bfloat16)
    to
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: int8, kv_cache[3]: float16).

    The semantic meaning of each KV cache entry is as follows:
    1. kv_cache[0] stores kv_lora.
    2. kv_cache[1] stores k_rope.
    3. kv_cache[2] stores the key tensor from the indexer module.
    4. kv_cache[3] stores the key scale tensor from the indexer module,
       and exists only when Sparse C8 is enabled.

    The main changes are as follows:
    1. The key tensor from the indexer module stored in kv_cache[2] is
       converted from bf16 to int8 to reduce memory usage. It is then
       processed with int8 precision in Lightning_indexer computation
       to improve computational efficiency.
    2. The quantization scale of the key tensor in the indexer module
       must also be stored for the Lightning_indexer_quant operator,
       and is therefore saved in kv_cache[3].
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    sparse_head_dim: tuple[int, ...] | None = None
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_cache_dtype)
    c8_k_scale_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_scale_cache_dtype)

    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L62-L89
        if self.cache_sparse_c8:
            assert self.sparse_head_dim is not None
            assert len(self.sparse_head_dim) == 3
            num_heads_per_page = self.block_size * self.num_kv_heads

            kv_lora_rank, qk_rope_head_dim, index_head_dim = self.sparse_head_dim

            # A5: kv_lora and k_rope are merged into a single CKV tensor (fp8).
            # A3: separate kv_lora + k_rope (bf16).
            if qk_rope_head_dim == 0:
                kv_dtype = self.c8_k_cache_dtype  # A5 CKV: float8_e4m3fn
                kv_dim = kv_lora_rank
            else:
                kv_dtype = self.dtype  # A3 kv_lora + k_rope: bfloat16
                kv_dim = kv_lora_rank + qk_rope_head_dim

            kv_bytes = num_heads_per_page * kv_dim * get_dtype_size(kv_dtype)
            qli_bytes = num_heads_per_page * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            qli_scale_bytes = num_heads_per_page * 1 * get_dtype_size(self.c8_k_scale_cache_dtype)
            return kv_bytes + qli_bytes + qli_scale_bytes

        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    # SUBTRACTED: sparse_kv_cache_ratio 属性及其内嵌 get_sparse_head_dim_virtual
    # (patch_kv_cache_interface.py:L91-L160)——各 kv_cache[i] 字节占比的虚拟维度换算（A5/A3 分流）。
    # page_size_bytes 已完整表达 Sparse-C8 显存模型（3→4 元组 KV），删除占比换算不影响案例2 主旨。

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L162-L185
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same quantization method."
        )
        cache_sparse_c8_set = set(spec.cache_sparse_c8 for spec in specs)
        assert len(cache_sparse_c8_set) == 1, (
            "All attention layers in the same KV cache group must use the same sparse C8 setting."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            scale_dim=specs[0].scale_dim,
            scale_dtype=specs[0].scale_dtype,
            sparse_head_dim=specs[0].sparse_head_dim,
            dtype=specs[0].dtype,
            cache_dtype_str=cache_dtype_str_set.pop(),
            cache_sparse_c8=specs[0].cache_sparse_c8,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L187-L195
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L198-L211
def _init_mla_cache_fields(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    """Shared MLA cache init logic for quantiztion format across different models."""
    FP8_DTYPE = "fp8_ds_mla"
    MODEL_VERSIONS = ["v32", "deepseek_v4"]
    if spec.cache_dtype_str != FP8_DTYPE:
        return
    assert spec.model_version in MODEL_VERSIONS, "Invalid model version."
    assert (spec.model_version == "v32" and spec.compress_ratio == 1) or (
        spec.model_version == "deepseek_v4" and spec.compress_ratio in [0, 4, 128]
    ), "Invalid compress ratio."
    if spec.compress_ratio > 1:
        assert spec.block_size % spec.compress_ratio == 0, (
            f"Block size {spec.block_size} must be divisible by compress ratio."
        )


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L214-L261
@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    @property
    def storage_block_size(self) -> int:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L224-L226
        return self.block_size

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L228-L230
        return self.storage_block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L232-L261
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_interface.py:L264-L266
# 技法①整类替换的三处重绑点：前两处在 kv_cache_interface 模块，第三处是 mla_attention 模块里
# 已 from-import 的 MLAAttentionSpec 别名（技法⑤from-import 缓存陷阱：建 spec 的调用方在
# mla_attention 持有自己的引用，只改 kv_cache_interface.MLAAttentionSpec 会漏网）。
vllm.v1.kv_cache_interface.MLAAttentionSpec = AscendMLAAttentionSpec
vllm.v1.kv_cache_interface.SlidingWindowMLASpec = AscendSlidingWindowMLASpec
vllm.model_executor.layers.attention.mla_attention.MLAAttentionSpec = AscendMLAAttentionSpec
