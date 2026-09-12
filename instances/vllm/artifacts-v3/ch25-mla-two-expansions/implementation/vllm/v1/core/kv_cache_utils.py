# SOURCE: vllm/v1/core/kv_cache_utils.py —— ch25 主文件 5（站 6 / m11：
# hybrid 组化四级分流）。切面 = get_kv_cache_groups 及其直接依赖族（逐条
# 行号见各 # SOURCE 锚）。SUBTRACTED：块哈希/前缀缓存/FreeKVCacheBlock
# Queue/get_kv_cache_configs 分配器面（L57-L880、L1283-L1445、L1855+）——
# ch13/ch14/ch15 域；unify_kv_cache_spec_page_size 的 MambaSpec 支——ch14。
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L882-L910 create_kv_cache_group_specs
#   —— 逐字
def create_kv_cache_group_specs(
    kv_cache_spec: "dict[str, KVCacheSpec]", grouped_layer_names: list[list[str]]
) -> "list[KVCacheGroupSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L882-L910（锚点双置：声明上方注释同文）
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


# SOURCE: vllm/v1/core/kv_cache_utils.py:L912-L936 is_kv_cache_spec_uniform
#   —— 逐字
def is_kv_cache_spec_uniform(kv_cache_spec: "dict[str, KVCacheSpec]") -> bool:
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L912-L936（锚点双置：声明上方注释同文）
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1013-L1021 get_uniform_page_size
#   —— 逐字
def get_uniform_page_size(kv_cache_specs) -> int:
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1013-L1021（锚点双置：声明上方注释同文）
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1022-L1038 _get_kv_cache_groups_
#   uniform_spec —— 逐字
def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: "dict[str, KVCacheSpec]",
) -> "list[KVCacheGroupSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1022-L1038（锚点双置：声明上方注释同文）
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1039-L1055 _get_kv_cache_groups_
#   uniform_type —— 逐字
def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> "list[KVCacheGroupSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1039-L1055（锚点双置：声明上方注释同文）
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1056-L1069 is_kv_cache_page_size_
#   uniform —— 逐字
def is_kv_cache_page_size_uniform(kv_cache_spec: "dict[str, KVCacheSpec]") -> bool:
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1056-L1069（锚点双置：声明上方注释同文）
    """
    Whether all layers in the given KVCacheSpec have the same page size.
    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same page size, False otherwise.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1070-L1134 unify_kv_cache_spec_
#   page_size —— 减法子集（MambaSpec 支按章界删——ch14）
def unify_kv_cache_spec_page_size(
    kv_cache_spec: "dict[str, KVCacheSpec]",
) -> "dict[str, KVCacheSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1070-L1134（锚点双置：声明上方注释同文）
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Two
    cases cannot be unified by block size alone and pad their physical page to
    the maximum instead: Mamba layers, whose page size comes from state shapes
    and is independent of block size; and attention layers whose page does not
    evenly divide the maximum and whose backend opts in via
    ``AttentionSpec.indexes_kv_by_block_stride`` (the padded page is read through
    a strided view, which not every backend handles). Raise NotImplementedError
    if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        # SUBTRACTED: MambaSpec 的页填充支（kv_cache_utils.py:L1083-L1096）
        #   ——ch14（Mamba 域；本章 spec 谱系无 Mamba）
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif (
                isinstance(layer_spec, AttentionSpec)
                and layer_spec.indexes_kv_by_block_stride
            ):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for attention layers whose backend indexes KV "
                    "pages by the block stride (indexes_kv_by_block_stride is "
                    "True)."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1135-L1137 is_kv_cache_type_
#   attention_free —— 逐字
def is_kv_cache_type_attention_free(kv_cache_spec: "dict[str, KVCacheSpec]") -> bool:
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1135-L1137（锚点双置：声明上方注释同文）
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1140-L1283 _get_kv_cache_groups_
#   uniform_page_size —— 逐字（注释长文全保留——hybrid 组化的假设清单）
def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: "dict[str, KVCacheSpec]",
) -> "list[KVCacheGroupSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1140-L1283（锚点双置：声明上方注释同文）
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window attention
    layers. There are 3 layers in the pattern (1 * full, 2 * sw), so there
    are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. However,
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Attempt to further merge same-type layers based on whether their KV
    # cache specs can be merged, to minimize the group count. This benefits
    # situations where specs share a block layout and differ only in a
    # property it can reconcile (e.g. full attention layers differing only in
    # sliding window / attention chunk size).
    layer_buckets: list[list[str]] = []
    spec_buckets: list[list] = []
    for layer_spec, layer_names in same_type_layers.items():
        for names, specs in zip(layer_buckets, spec_buckets):
            try:
                # A raise means that the specs are incompatible.
                type(specs[0]).merge([*specs, layer_spec])
            except (AssertionError, ValueError):
                continue
            names.extend(layer_names)
            specs.append(layer_spec)
            break
        else:
            layer_buckets.append(list(layer_names))
            spec_buckets.append([layer_spec])

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    grouped_layers = []
    for layers in layer_buckets:
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because the
        # 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1446-L1531 _promote_local_kv_cache_
#   specs —— 逐字
def _promote_local_kv_cache_specs(
    kv_cache_spec: "dict[str, KVCacheSpec]",
) -> "dict[str, KVCacheSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1446-L1531（锚点双置：声明上方注释同文）
    """Use full-attention allocation for local-attention cache specs.

    The returned specs affect KV cache management only. Attention modules keep
    their original sliding-window or chunked-local compute behavior.
    """
    promoted_specs = kv_cache_spec.copy()

    if is_kv_cache_spec_uniform(
        promoted_specs
    ) or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs):
        return promoted_specs

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in promoted_specs.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in promoted_specs.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in promoted_specs.values()
    )
    full_block_sizes = {
        spec.block_size
        for spec in promoted_specs.values()
        if isinstance(spec, FullAttentionSpec)
    }
    full_attention_block_size = (
        next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    )

    def promoted_page_size_padded(spec: AttentionSpec, block_size: int) -> int | None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L1446-L1531（锚点双置：声明上方注释同文）
        if spec.page_size_padded is None:
            return None
        unpadded_page_size = (
            spec.unpadded_page_size_bytes * block_size // spec.block_size
        )
        return max(spec.page_size_padded, unpadded_page_size)

    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowMLASpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                    cache_dtype_str=spec.cache_dtype_str,
                    alignment=spec.alignment,
                    compress_ratio=spec.compress_ratio,
                    model_version=spec.model_version,
                )
            elif isinstance(spec, SlidingWindowSpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    head_size_v=spec.head_size_v,
                    dtype=spec.dtype,
                    kv_quant_mode=spec.kv_quant_mode,
                    sliding_window=spec.sliding_window,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                )

    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")

    return promoted_specs


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1534-L1566 _try_get_full_allocation_
#   fallback_groups —— 逐字
def _try_get_full_allocation_fallback_groups(
    kv_cache_spec: "dict[str, KVCacheSpec]",
) -> "list[KVCacheGroupSpec] | None":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1534-L1566（锚点双置：声明上方注释同文）
    """Try a supported full-allocation fallback for local-attention layers."""
    if any(isinstance(spec, HiddenStateCacheSpec) for spec in kv_cache_spec.values()):
        return None
    if any(
        isinstance(spec, (SlidingWindowMLASpec, ChunkedLocalAttentionSpec))
        for spec in kv_cache_spec.values()
    ):
        return None

    has_mla = any(isinstance(spec, MLAAttentionSpec) for spec in kv_cache_spec.values())
    has_regular_swa = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    if not (has_mla and has_regular_swa):
        return None

    try:
        promoted_specs = _promote_local_kv_cache_specs(kv_cache_spec)
    except ValueError:
        return None
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(promoted_specs)
    if uniform_spec is None:
        return None
    logger.warning(
        "KV cache page sizes cannot be unified; treating sliding-window "
        "layers as full attention for cache allocation. Sliding-window "
        "attention compute is unchanged."
    )
    return _get_kv_cache_groups_uniform_type(uniform_spec)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1568-L1591 unify_hybrid_kv_cache_
#   specs —— 逐字（disable 路径的原话 warning——章首 why_chains 引用的退路）
def unify_hybrid_kv_cache_specs(kv_cache_spec: "dict[str, KVCacheSpec]"):
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1568-L1591（锚点双置：声明上方注释同文）
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )
    kv_cache_spec.update(_promote_local_kv_cache_specs(kv_cache_spec))


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1593-L1633 group_and_unify_kv_cache_
#   specs —— 逐字（DSV4 案：四级分流的第三级）
def group_and_unify_kv_cache_specs(
    kv_cache_spec: "dict[str, KVCacheSpec]",
) -> "list[UniformTypeKVCacheSpecs] | None":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1593-L1633（锚点双置：声明上方注释同文）
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    ):
        return None

    # SlidingWindowMLASpec models with uniform page sizes don't need tuple packing.
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return None

    mla_specs: "dict[str, KVCacheSpec]" = {}
    grouped_swa_mla_specs: dict = defaultdict(dict)
    # NOTE: Here we group SWA layers by (block_size, sliding_window), which separates
    # SWA layers, C4I+C4A layers, and C128A layers into three different groups. It can
    # be fragile with only block_size and sliding_window as keys, but fine for now.
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            mla_specs[name] = spec

    assert len(mla_specs) > 0
    mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
    assert mla_uniform_spec is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [mla_uniform_spec, *swa_uniform_specs]


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1635-L1669 _approximate_gcd —— 逐字
def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1670-L1756 _get_kv_cache_groups_
#   uniform_groups —— 逐字（DSV4 案：layer tuple 对齐的组切分）
def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> "list[KVCacheGroupSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1670-L1756（锚点双置：声明上方注释同文）
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [
        g_spec.get_num_layer_tuples() for g_spec in grouped_specs
    ]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(
        num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
    )
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [
        round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group
    ]

    swa_mla_specs = grouped_specs[1:]
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in swa_mla_specs
        for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their
    # numbers of layer tuples. The packed block planner overlays groups, so
    # their page sizes do not need to match.
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        layers_per_size: dict = defaultdict(list)

        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            layers_per_size[layer_spec.page_size_bytes].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [
                name for layer_tuple in group_layer_tuples for name in layer_tuple
            ]
            group_layer_specs = {
                name: sm_spec.kv_cache_specs[name] for name in group_layer_names
            }
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, *swa_mla_groups]


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1757-L1780 _annotate_eagle_groups_
#   deepseek_v4 —— 减法子集（spec 面收窄：本章无 speculative_config）
def _annotate_eagle_groups_deepseek_v4(
    vllm_config,
    kv_cache_spec: "dict[str, KVCacheSpec]",
    kv_cache_groups: "list[KVCacheGroupSpec]",
) -> None:
    # SOURCE: spec_config 探测（L1758-L1760）——本章 speculative_config 恒
    #   None → 早退（spec-decode 归 ch33）
    return None


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1781-L1852 get_kv_cache_groups
#   —— 逐字（must_keep：hybrid 组化四级分流入口）
def get_kv_cache_groups(
    vllm_config, kv_cache_spec: "dict[str, KVCacheSpec]"
) -> "list[KVCacheGroupSpec]":
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1781-L1852（锚点双置：声明上方注释同文）
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)
    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):
        # DeepseekV4 case: All layers need the same number of token slots,
        # yet some layers are full attention while others are sliding window
        # attention in different sizes. Need to group layers into multiple
        # UniformTypeKVCacheSpecs.
        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)
        return kv_cache_groups

    # Pull HiddenStateCacheSpec layers out before the general multi-group
    # path so they don't affect page-size unification or grouping.
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    filtered_spec = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, HiddenStateCacheSpec)
    }

    # Prefer preserving each layer's cache semantics. If physical pages cannot
    # be unified, try a supported allocation-only fallback before failing.
    try:
        filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    except NotImplementedError:
        fallback_groups = _try_get_full_allocation_fallback_groups(kv_cache_spec)
        if fallback_groups is None:
            raise
        return fallback_groups
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            new_bs = max(common_page // per_token, 1)
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))

    return groups
