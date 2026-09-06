# Subtract-only companion for v3 ch21 — vllm/v1/worker/gpu_model_runner.py
# (pin v0.27.1 / 6e448d0ea). 本章切面（站 6-12）：KV 初始化与每拍 metadata
# 的后端线——initialize_attn_backend 归组（AttentionGroupKey 等价类）、
# _check_and_update_cudagraph_mode 最弱链降级、prepare/initialize 元数据
# builder、initialize_kv_cache[_tensors]（裸显存 as_strided 定形 + bind）、
# _build_attention_metadata（cm_base 组装→逐组换表→build 翻译→layer_name
# 铺设）、_get_slot_mappings、execute_model 的 set_forward_context 段。
# __init__ 是 ENGINE SEAM 切面构造（真实 L456-L760 的本章消费字段直供，
# ch17/ch22 同款）；采样/logits/调度/编译域以章界注记收窄（impl-notes）。
from __future__ import annotations

import copy
from collections import defaultdict
from math import prod
from typing import Any, NamedTuple, TypeAlias

import torch

from ..._host_seams import (
    CUDAGraphMode,
    InputBatchSeam,
    _CpuGpuSeam,
    _CudagraphDispatcherSeam,
    get_layers_from_vllm_config,
    init_logger,
    record_function_or_nullcontext,
)
from ...forward_context import set_forward_context
from ...utils.torch_utils import get_dtype_size
from ..attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    AttentionMetadata,
    CommonAttentionMetadata,
)
from ..kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVQuantMode,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from .utils import AttentionGroup, bind_kv_cache, prepare_kernel_block_sizes

logger = init_logger(__name__)

# SUBTRACTED: gpu_model_runner.py 的 import 面其余（L1-L460——采样/观测/
#   spec-decode/connector/mamba/Lora 等域）与 __init__ 主体（L456-L760）——
#   ENGINE SEAM 切面构造承载本章消费字段（见下）。

# SOURCE: vllm/v1/worker/gpu_model_runner.py:L255 PerLayerAttnMetadata ——（逐字）
AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


# ── gpu/attn_utils.py 的定形实现（站 7 消费点） ────────────────────────────


# SOURCE: vllm/v1/worker/gpu/attn_utils.py:L211-L264 _reshape_attention_kv_
#   cache ——（逐字）逻辑形×stride_order 置换 → 物理连续视图 → permute 回
#   逻辑形（packing/padded-page 两支随 ch14/ch27 域收窄，无 padding 支保留）
def _reshape_attention_kv_cache(  # SOURCE: vllm/v1/worker/gpu/attn_utils.py
    kv_raw_tensor: torch.Tensor,
    kv_cache_spec: AttentionSpec,
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride_order: tuple[int, ...],
    num_blocks: int,
    packing: tuple[int, int] | None,
) -> torch.Tensor:
    permuted_kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]
    dtype = kv_cache_spec.dtype

    if packing is not None:
        offset, block_stride = packing
        assert inv_order[0] == 0
        page_bytes = prod(kv_cache_shape[1:]) * get_dtype_size(dtype)
        kv_cache = (
            kv_raw_tensor.view(-1, block_stride)[:, offset : offset + page_bytes]
            .view(dtype)
            .view(permuted_kv_cache_shape)
        )
    elif kv_cache_spec.page_size_padded is not None:
        # Use a strided view to skip the padding between physical pages.
        #
        # Only num-blocks-first layouts are supported (the block dimension is
        # dim 0 of the unpermuted shape). kv-first layouts such as ROCm's
        # ``(2, num_blocks, ...)`` are intentionally not supported here. For a
        # num-blocks-first layout the only stride that must change is the block
        # stride: every other (contiguous) stride already steps within the
        # unpadded region of a page, so no further adjustment is needed.
        assert kv_cache_shape[0] == num_blocks, (
            "Padded KV pages require a num-blocks-first KV cache layout (got "
            f"shape {kv_cache_shape} with num_blocks={num_blocks}); "
            "kv-first layouts are not supported."
        )
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        page_stride = kv_cache_spec.page_size_bytes // dtype_size

        num_blocks_dim = inv_order[0]
        strides = list(torch.empty(permuted_kv_cache_shape, device="meta").stride())
        strides[num_blocks_dim] = page_stride

        kv_cache = torch.as_strided(
            kv_raw_tensor.view(dtype),
            size=permuted_kv_cache_shape,
            stride=tuple(strides),
        )
    else:
        # No padding — safe to use a contiguous view.
        kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)

    return kv_cache.permute(*inv_order)


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L456 GPUModelRunner —— ENGINE
#   SEAM 切面构造：真实 __init__（L456-L760）的选后端/metadata/前向上下文
#   消费字段直供（模型/采样/cudagraph 捕获装配面归 ch17/ch18/ch19 域，
#   测试在此注入）；站方法（下列各 SOURCE 段）为真实代码只删不增。
class GPUModelRunner:
    def __init__(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456-L760（ENGINE SEAM 切面）
        self,
        vllm_config,
        device: torch.device,
    ):
        # 本章消费的 config 面（真身 ch03/ch17 域装配）
        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config  # 恒 None（seam）

        # 站 6-7 状态面（真实字段）
        self.attn_groups: list[list[AttentionGroup]] = []
        self.kv_cache_config: KVCacheConfig | None = None
        self._kernel_block_sizes: list[int] | None = None
        self.kv_caches: list[torch.Tensor] = []
        self.shared_kv_cache_layers: dict[str, str] = {}  # delete[8] 同域恒空
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()
        self.runner_only_attn_layers: set[str] = set()
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.uniform_decode_query_len = None  # ch19 降级链输入面（seam 直供）

        # ch19 域接口侧（降级链 + keys 预生成——HOST SEAM 观测位）
        self.cudagraph_dispatcher = _CudagraphDispatcherSeam()
        self.kv_cache_config = None

        # 每拍 metadata 的持久缓冲面（ch18/ch22 域装配；本章消费字段直供，
        # 测试按拍覆写）
        self.input_batch = InputBatchSeam(self.max_num_reqs)
        self.query_start_loc = _CpuGpuSeam(
            torch.zeros(self.max_num_reqs + 1, dtype=torch.int32)
        )
        self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32)
        self.optimistic_seq_lens_cpu = torch.zeros(self.max_num_reqs, dtype=torch.int32)
        self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64)

        # 两段式契约状态位（真实 L464 附近的 execute_model_state 协议面）
        self.execute_model_state = None
        # 最弱链观测位（_check_and_update_cudagraph_mode 记录）
        self.seam_min_cg_support = None
        # execute_model 切面的观测/直供位（ch18/ch19 域产出的消费面）
        self.seam_num_reqs = 1
        self.seam_num_tokens = 1
        self.seam_max_query_len = 1
        self.seam_cudagraph_mode = CUDAGraphMode.NONE
        self.seam_batch_desc = None
        self.seam_model_output: dict[str, Any] = {}
        self.seam_seen_metadata: dict[str, Any] = {}
        self.seam_seen_slot_mapping: dict[str, Any] = {}

        # SUBTRACTED: 真实 __init__ 的模型装配/采样缓冲/cudagraph runner/
        #   调度参数/lora/connector 装配（L456-L760 主体）——ch17/ch18/ch19 域。

    # ── 站 6：归组与最弱链 ────────────────────────────────────────────────

    def initialize_attn_backend(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7020-L7125（delete[8] 删 FastPrefill/CP 检查）
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7030-L7045
            """Deduplication key for attention groups within a KV cache group.

            Splits on per-rank ``num_heads_q`` in addition to backend + spec
            so layers with different Q-head counts (e.g. a spec-decode draft
            with fewer attention heads than its target) get separate metadata
            builders. The builders' scratch (e.g. ``softmax_segm_*`` in
            ``triton_attn``, ``num_qo_heads`` in FlashInfer) is sized by
            ``num_heads_q`` and assumes uniformity within the group; see
            ``get_num_attention_heads_from_layers`` in
            ``vllm/v1/attention/backends/utils.py``.
            """

            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec
            num_heads_q: int

        def get_attn_backends_for_group(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7047-L7089
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            from ...model_executor.layers.attention_layer_base import (
                AttentionLayerBase,
            )

            layer_type = AttentionLayerBase
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                # SUBTRACTED: FastPrefill 包装分支（L7064-L7068）——delete[8]：
                #   kv_sharing_fast_prefill 是正交特性（create_fast_prefill_
                #   custom_backend 属 ch22 快预填域）。

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                # Non-Attention layer types (e.g. Mamba1, ShortConv) do not
                # expose ``num_heads``; fall back to 0 so they cluster as
                # before. Such layers never coexist with Attention in a
                # single KV cache group (different KVCacheSpec), so the
                # fallback can never spuriously merge them with attention
                # layers.
                num_heads_q = getattr(layers[layer_name], "num_heads", 0)
                key = (full_cls_name, layer_kv_cache_spec, num_heads_q)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec, num_heads_q
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7091-L7105
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for key, layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    key.attn_backend,
                    layer_names,
                    key.kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(
            attention_backend_list,
            kv_cache_config.kv_cache_groups,
            is_profiling=is_profiling,
        )

        # SUBTRACTED: check_attention_cp_compatibility（L7121-L7122）——
        #   delete[8]：CP 兼容检查是正交特性（分布式 Part）。

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7127-L7148
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

        # SUBTRACTED: drafter attention backend 初始化（L7150-L7159）——
        #   spec-decode 域（ch12/ch33）。

    def _check_and_update_cudagraph_mode(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7161-L7202
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
        is_profiling: bool = False,
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_attn_backend = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_attn_backend = attn_backend.__name__
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            min_cg_support,
            min_cg_attn_backend,
            self.uniform_decode_query_len,
            use_v2_model_runner=False,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
            is_profiling=is_profiling,
        )
        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # SUBTRACTED: drafter 的 cudagraph dispatcher 初始化（L7204-L7218）
        #   ——spec-decode 域（ch12/ch33）。

        # ENGINE SEAM 观测位：最弱链值记录（测试断言能力传染的量化）
        self.seam_min_cg_support = min_cg_support

    def calculate_reorder_batch_threshold(self) -> None:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7220-L7238
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        from functools import reduce

        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)  # noqa: E731

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

    # ── 站 7：KV 定形与 bind ──────────────────────────────────────────────

    def _attn_group_iterator(self):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7355-L7356（逐字）
        import itertools

        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7358-L7362（逐字）
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _allocate_kv_cache_tensors(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7312-L7353（逐字）
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        packed_backing: torch.Tensor | None = None
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            if kv_cache_tensor.block_stride > 0:
                # Allocate once; all packed tensors alias the same backing.
                if packed_backing is None:
                    packed_backing = torch.zeros(
                        kv_cache_tensor.size,
                        dtype=torch.int8,
                        device=self.device,
                    )
                tensor = packed_backing
            else:
                tensor = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=self.device
                )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7364-L7479（逐字）
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False

        # Map layer names to (offset, block_stride) within the packed
        # backing tensor so we can create strided views per layer.
        layer_packing: dict[str, tuple[int, int]] = {}
        for kv_tensor in self.kv_cache_config.kv_cache_tensors:
            if kv_tensor.block_stride > 0:
                for ln in kv_tensor.shared_by:
                    layer_packing[ln] = (kv_tensor.offset, kv_tensor.block_stride)
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                packing = layer_packing.get(layer_name)
                if packing is not None:
                    _, blk_stride = packing
                    num_blocks = raw_tensor.numel() // blk_stride
                else:
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    # For MLA with compression, storage_block_size != block_size
                    if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
                        shape_block_size = kv_cache_spec.storage_block_size
                    else:
                        shape_block_size = kernel_block_size

                    # Skipped layers (--kv-cache-dtype-skip-layers) need
                    # the unquantized shape.
                    layer_cache_dtype_str = (
                        "auto"
                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
                        else getattr(
                            kv_cache_spec,
                            "cache_dtype_str",
                            None,
                        )
                        or self.cache_config.cache_dtype
                    )
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        shape_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=layer_cache_dtype_str,
                    )
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    kv_caches[layer_name] = _reshape_attention_kv_cache(
                        raw_tensor,
                        kv_cache_spec,
                        kv_cache_shape,
                        kv_cache_stride_order,
                        kernel_num_blocks,
                        packing,
                    )

                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    page_size_bytes = kv_cache_spec.page_size_bytes
                    # Hold a single contiguous [num_blocks, 1, 1, page_size_bytes]
                    # int8 page view per layer; the layer's bind_kv_cache unpacks
                    # each block's bytes into its conv/ssm state views. Keeping
                    # one tensor per layer lets the KV connector register it
                    # without special-casing Mamba.
                    kv_caches[layer_name] = raw_tensor[
                        : num_blocks * page_size_bytes
                    ].view(num_blocks, 1, 1, page_size_bytes)
                else:
                    raise NotImplementedError

        # Reconcile divergent KV layouts to blocks-first. Triggered by hybrid
        # attention/mamba models, and by encoder-decoder models whose shared
        # decoder/cross-attention allocation mixes K/V-first and blocks-first
        # backends (see _has_mixed_attention_kv_layout).
        if has_attn and (
            has_mamba or self._has_mixed_attention_kv_layout(kernel_block_sizes)
        ):
            self._update_hybrid_attention_mamba_layout(kv_caches, kernel_block_sizes)

        return kv_caches

    def _has_mixed_attention_kv_layout(self, kernel_block_sizes: list[int]) -> bool:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7481-L7505（逐字）
        """Whether attention groups disagree on the physical KV cache layout.

        Encoder-decoder models (e.g. Whisper) share one raw KV allocation
        between a decoder self-attention layer (K/V-first ROCM_ATTN, block dim
        1) and a cross-attention layer (blocks-first, block dim 0). Mixed block
        dims mean a block ID maps to different bytes per layer, so the shared
        buffer must be normalized to a single (blocks-first) layout.
        """
        block_dims: set[int] = set()
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue
            block_dims.add(
                group.backend.get_kv_cache_block_dim(
                    kernel_block_sizes[group.kv_cache_group_id],
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=self.cache_config.cache_dtype,
                )
            )
        return len(block_dims) > 1

    def _update_hybrid_attention_mamba_layout(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7507-L7539（逐字）
        self, kv_caches: dict[str, torch.Tensor], kernel_block_sizes: list[int]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_block_sizes[group.kv_cache_group_id],
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype_str=self.cache_config.cache_dtype,
            )
            # block_dim: 0 means (num_blocks, 2, ...); 1 means (2, num_blocks, ...).
            if block_dim == 0:
                continue
            assert block_dim == 1
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                hidden_size = kv_cache.shape[2:].numel()
                kv_cache.as_strided_(
                    size=kv_cache.shape,
                    stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                )

    def initialize_kv_cache_tensors(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7541-L7594
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer.
        """

        # SUBTRACTED: uniform KV cache 支（L7556-L7569）——kv-connector 混布
        #   分配域（ch16；use_uniform_kv_cache 无 kv transfer group 恒 False，
        #   走 general 路径——本章站 7 主线）。
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(
            kv_cache_raw_tensors, kernel_block_sizes
        )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def initialize_kv_cache(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7624-L7681（delete[8] 删 mamba/connector/kv_sharing 尾部）
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        from copy import deepcopy

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        # SUBTRACTED: _mamba_bufs 置位 / may_add_encoder_only_layers /
        #   maybe_add_kv_sharing_layers_to_kv_cache_groups（L7637-L7639）——
        #   delete[8]：mamba 缓冲与 encoder-only/kv_sharing 组装配是正交特性。
        self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
        # SUBTRACTED: initialize_mamba_ssu_backend（L7641-L7643）——delete[8]。
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        self._kernel_block_sizes = kernel_block_sizes

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # SUBTRACTED: may_reinitialize_input_batch（L7657-L7658）——ch22 域
        #   （InputBatch 块表重建全文；本章 input_batch 由 seam 装配位承载）。
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )
        # ENGINE SEAM 观测位：定形产物（真实侧由后续消费者取用）
        self.seam_kv_caches = kv_caches

        # SUBTRACTED: spec-decode drafter 校验与 KV transfer 注册尾部
        #   （L7663-L7681）——delete[8]：connector 挂接是正交特性。

    # ── 站 8-10：每拍 metadata ────────────────────────────────────────────

    def _get_slot_mappings(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4082-L4154（逐字）
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: Any | None = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """
        Build slot mappings in both formats needed by the system.

        Args:
            num_tokens_padded: Total number of tokens (padded)
            num_reqs_padded: Total number of requests (padded)
            num_tokens_unpadded: Actual number of tokens (unpadded)
            ubatch_slices: Optional ubatch slicing info for DBO

        Returns:
            A tuple of:
            - slot_mappings_by_gid: dict[int, torch.Tensor] for attention metadata
            - slot_mappings_by_layer: dict[str, torch.Tensor] or list for ForwardContext
        """
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        def _get_slot_mapping(kv_cache_gid: int):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4113-L4132
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4125-L4126 ——
                #   块表算出的槽位表（ch22 域产物；seam 的 input_batch 以
                #   裸张量承载 slot_mapping.gpu 切片面）
                blk_table = self.input_batch.block_tables[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        if ubatch_slices is not None:
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        return slot_mappings_by_gid, slot_mappings_by_layer

    def _build_attention_metadata(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2284-L2618（delete[7] 删扩展态分支）
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: Any | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        Returns:
            tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: PerLayerAttnMetadata = {}
        # SUBTRACTED: ubatch 的 list 形态（L2312-L2313——delete[7]：DBO 不进
        #   主线，attn_metadata 恒 dict）。

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_block_table(kv_cache_gid: int):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2325-L2341
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_tables[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # Fill unused block table entries with NULL_BLOCK_ID (null block)
            # for CUDAGraph padding. Block 0 is reserved for padding.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2338-L2340 ——尾行
            #   填 NULL_BLOCK_ID（块表 pad 值；Block 0 保留给 padding）
            from ..attention.backends.utils import NULL_BLOCK_ID

            blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
            return blk_table_tensor

        assert slot_mappings is not None
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        # SUBTRACTED: routed_experts 快照（L2347-L2358）——delete[7]。
        # SUBTRACTED: num_computed/num_prompt/seq_lens_cpu/is_prefilling 装填
        #   与 async-spec/upper-bound 面（L2360-L2379）——delete[7]：其消费
        #   字段随 delete[1] 的 Common 字段删。
        # SUBTRACTED: mm_prefix 双向区段 / R-SWA 前缀长 / replayssm 基址
        #   （L2382-L2427）——delete[7]：mm/R-SWA/replayssm 扩展态。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2430-L2449 cm_base 组装
        #   ——所有后端所有层共享的字段每拍只算这一次；block_table/slot_
        #   mapping 先填组 0（组 0 之外在逐组循环里换）。（delete[1] 的
        #   kwargs 行连带删——被删字段不得再传，防 TypeError）
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
        )

        # SUBTRACTED: dcp_local_seq_lens 装填（L2451-L2464）与 kv_sharing_
        #   fast_prefill 的 logits 面（L2466-L2470）——delete[7]。

        # Cache attention metadata builds across hybrid KV-cache groups
        # The only thing that changes between different hybrid KV-cache groups when the
        # same metadata builder and KVCacheSpec is the same is the block table, so we
        # can cache the attention metadata builds and just update the block table using
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2481-L2556
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            # SUBTRACTED: spec-decode 的 mamba/GDN/linear extra args（
            #   L2500-L2524）——delete[7]：spec-decode 扩展态。

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                )
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            # SUBTRACTED: ubatch 的 dict 选择（L2548-L2553）——delete[7]：
            #   DBO 不进主线，直接铺进 attn_metadata。

            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        # SUBTRACTED: spec_decode_common_attn_metadata 初始化（L2560）——
        #   delete[7]。
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy.copy(cm_base)  # shallow copy

            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            # SUBTRACTED: _get_encoder_seq_lens（L2566-L2571）——delete[7]：
            #   encoder-decoder 组的特例（enc-dec 域）。
            if kv_cache_gid > 0:
                cm.block_table_tensor = _get_block_table(kv_cache_gid)
                cm.slot_mapping = slot_mappings[kv_cache_gid]

            # SUBTRACTED: spec-decode drafter 的 per-group 捕获（L2576-L2598）
            #   ——delete[7]。

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    # SUBTRACTED: split_attn_metadata（L2601-L2603）——delete[7]：
                    #   DBO 不进主线（ubatch_slices 恒 None）。
                    raise NotImplementedError("ubatching → ch12 DBO 域")
                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        # SUBTRACTED: spec_decode_common_attn_metadata 的 unpadded 尾部
        #   （L2608-L2616）——delete[7]。

        return attn_metadata, None

    # ── 站 8/10-11：execute_model 的 metadata/前向上下文段 ────────────────

    def execute_model(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4166-L4535（本章切面）
        self,
        scheduler_output: Any,
        intermediate_tensors: Any | None = None,
    ) -> Any:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4171-L4175 两段式契约
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # SUBTRACTED: ngram-GPU replace 复制（L4180-L4195）与 KV connector
        #   preemption（L4197-L4200）——ch18 域 / delete[8]。
        # SUBTRACTED: _update_states/_prepare_inputs 前处理主体（L4203-L4248）
        #   ——ch18/ch22 域全文（持久批状态与 positions 装填）；本章的批
        #   形状以 seam 直供位承载（seam_num_reqs/seam_num_tokens/
        #   seam_max_query_len——ch18 域产出的消费面）。
        num_reqs = self.seam_num_reqs
        num_tokens_unpadded = self.seam_num_tokens
        max_num_scheduled_tokens = self.seam_max_query_len

        # SUBTRACTED: cascade attention prefix 预计算（L4255-L4263）——
        #   ch20 cascade 域（本章 common_prefix_len=0 恒不级联）。
        cascade_attn_prefix_lens = None

        # SUBTRACTED: _determine_batch_execution_and_padding（L4265-L4278）
        #   ——ch19 域（BatchDescriptor 查表/padding 裁决）；ch19 域产出以
        #   观测位直供（seam_cudagraph_mode/seam_batch_desc）。
        cudagraph_mode = self.seam_cudagraph_mode
        batch_desc = self.seam_batch_desc
        num_tokens_across_dp = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4289-L4292 ——padded
        #   口径（无 batch_desc 时退 unpadded）
        num_tokens_padded = (
            batch_desc.num_tokens if batch_desc is not None else num_tokens_unpadded
        )
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc is not None and batch_desc.num_reqs is not None else num_reqs
        )
        # SUBTRACTED: maybe_create_ubatch_slices（L4293-L4299）——DBO 域。
        ubatch_slices_padded = None
        logits_indices = None
        use_spec_decode = False
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens if (
            scheduler_output is not None and hasattr(scheduler_output, "num_scheduled_tokens")
        ) else {}

        # SUBTRACTED: mamba align 预处理（L4320-L4362）——mamba 域。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4307-L4318 ——
        #   has_separate_kv_update 裁决（存在后端 KV 写独立成 op →
        #   slot_mappings 必须用 padded 维度匹配 key/value 张量）
        # True if any attention backend handles KV cache update separately
        # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
        # slot_mappings must use padded dimensions to match the key/value tensors.
        has_separate_kv_update = not all(
            all(
                g.backend.forward_includes_kv_cache_update
                for g in self.attn_groups[id]
            )
            for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
            if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
        )
        pad_attn = cudagraph_mode == CUDAGraphMode.FULL

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4367-L4376 ——组版+层版
        #   两种口径的槽位表
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens_padded
            if pad_attn or has_separate_kv_update
            else num_tokens_unpadded,
            num_reqs_padded=(
                num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
            ),
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4378-L4392 ——每拍
        #   metadata 组装+翻译+铺设
        attn_metadata, spec_decode_common_attn_metadata = (
            self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=None,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                slot_mappings=slot_mappings_by_group,
            )
        )

        # SUBTRACTED: _preprocess（L4394-L4403）——ch18/ch22 域（positions/
        #   input_ids 装填）；seam 前向直供张量。

        # SUBTRACTED: calculate_kv_scales 的 cudagraph 降档（L4405-L4411）
        #   ——delete[9] 同域；encoder eager 首拍（L4413-L4418）——enc-dec 域；
        #   eplb 元数据（L4425-L4431）——ch33 域。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4432-L4443 ——
        #   set_forward_context 设模块级全局 _forward_context（attn_metadata
        #   + slot_mapping 两通道按 layer_name 供层取数）
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=False,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
        ):
            model_output = self._model_forward(
                num_tokens=num_tokens_padded,
            )

        # SUBTRACTED: logits/采样/PP 尾段（L4458-L4535）——ch18 域（两段式
        #   契约的后半 sample_tokens 归彼）。
        return model_output

    def _model_forward(  # SOURCE: vllm/v1/worker/gpu_model_runner.py ——ENGINE SEAM
        #   （真实 _model_forward 经 compiled wrapper 跑整个模型，ch17/ch19
        #   域；本章以逐 Attention 层直调承载：每层 forward 内部先写腿后读腿
        #   ——attention.py 真身的两算子调用序），观测位记录前向内按
        #   layer_name 取到的 metadata/slot_mapping。
        self,
        num_tokens: int,
    ) -> dict[str, Any]:
        from ...forward_context import get_forward_context

        fc = get_forward_context()
        out: dict[str, Any] = {}
        for layer_name, layer in self.compilation_config.static_forward_context.items():
            if not hasattr(layer, "impl"):
                continue
            n_heads, head_size = layer.num_heads, layer.head_size
            n_kv = layer.num_kv_heads
            q = torch.zeros(num_tokens, n_heads * head_size, dtype=self.model_config.dtype)
            k = torch.zeros(num_tokens, n_kv * head_size, dtype=self.model_config.dtype)
            v = torch.zeros_like(k)
            out[layer_name] = layer.forward(q, k, v)
        # 观测位（ENGINE SEAM）：前向内取到的两通道
        self.seam_seen_metadata = dict(fc.attn_metadata) if isinstance(fc.attn_metadata, dict) else {}
        self.seam_seen_slot_mapping = dict(fc.slot_mapping) if isinstance(fc.slot_mapping, dict) else {}
        return out

    # SUBTRACTED: gpu_model_runner.py 其余（采样/logits_processor/lora/
    #   profile_report/_preprocess/_update_states/cudagraph 捕获族等）——
    #   ch17/ch18/ch19/ch22 各域全文。


