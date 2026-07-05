# vllm_ascend/worker/model_runner_v1.py —— subtract-only 精简版（ch16 KV 显存几何）
#
# 本章主线：NPUModelRunner 覆写基座 gpu_model_runner.py 的 KV cache 物化路径，改「内存几何」——
#   initialize_kv_cache_tensors  三步骨架：allocate → reshape → bind（按模型特化）
#     ├─ _allocate_kv_cache_tensors   昇腾内存几何核心：int8 裸分配 + 按 split_factor 拆 K/V + 2MB 对齐
#     │    ├─ _align_up / _align_memory          对齐原语（纯算术，可在 host 跑）
#     │    ├─ _allocate_int8_cache_tensor        「KV 一律 int8 裸分配」统一物化点（含 kv_transfer 对齐分支）
#     │    └─ _allocate_sparse_c8_indexer_tensors dsa_k / dsa_k_scale 共享一块对齐 int8 的两个视图
#     ├─ _reshape_kv_cache_tensors    .view(dtype).view(shape) 把裸字节还原成 KV；MLA 拆 nope(k)/rope(v)
#     │    └─ _adjust_kv_layout       as_strided 按 page_size_bytes 跨步重排 NPU 物理布局
#     └─ bind                          deepseek_v4 自定层序 / longcat num_attn_module=2 / 普通 bind_kv_cache
#   辅线：may_reinitialize_input_batch（多 group/异构 block_size 重建 NPUInputBatch + kernel_block_sizes）
#         get_kv_cache_spec（MLA → AscendMLAAttentionSpec，回指 ch04）
#
# host 无 NPU/CANN：真实 torch_npu 显存分配/物理布局不真跑；但对齐算术、calc_split_factor、
# as_strided 维度推导、bind 三分支派发都是纯 Python/CPU torch，可在 host 验证与真实仓一致的控制流。
import logging
import math
from collections import defaultdict
from copy import deepcopy

import torch

# SOURCE: vllm_ascend/worker/model_runner_v1.py:L39-L162（节选本章用到的导入）
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.cp_utils import get_total_cp_world_size
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.utils import (
    calc_split_factor,
    kv_cache_spec_uses_sparse_c8,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

# SUBTRACTED: model_runner_v1.py:L20-L162 其余数百行导入（attention 后端实体 / spec_decode /
#   mamba_utils / EC·KV connector / pcp_manager / AscendDeviceType·get_ascend_device_type 等）。
#   本精简版按 subtraction_plan 删去 A5 设备分叉，故无需 get_ascend_device_type；这些导入与
#   「KV 张量分配/重排/绑定」主线正交。

logger = logging.getLogger(__name__)


# SOURCE: vllm_ascend/worker/model_runner_v1.py:L255
class NPUModelRunner(GPUModelRunner):
    # 仅呈现本章 KV 几何相关方法；__init__ 大段字段初始化（self.use_sparse / use_compress /
    # c8_*_cache_dtype / sparse_head_dim / enable_hamming_sparse / kernel_block_sizes 等，
    # model_runner_v1.py:L255-L600）折叠——测试按需注入这些属性。

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3700
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, is_profiling: bool = False) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_bufs = None
        self._mamba_copy_bufs = None
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any(
            [isinstance(attn_group[0].kv_cache_spec, MambaSpec) for attn_group in self.attn_groups]
        )

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # SUBTRACTED: 投机解码 drafter.initialize_attn_backend（L3729-L3735）、
        #   has_kv_transfer_group() register_kv_caches（L3737-L3738）、
        #   enable_return_routed_experts → init_routed_experts_capturer（L3740-L3741）
        #   及 _bind_routed_experts_capturer 方法（L3743-L3756）——KV 几何主线之外的旁路注册，
        #   删去不影响 KV 张量如何物化/绑定的控制流（原 model_runner_v1.py:L3729-L3756）。

    def _align_memory(self, tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3758
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3764
    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        if self.model_config.hf_text_config.model_type == "deepseek_v4":
            from vllm_ascend.utils import extract_dsv4_layer_index

            assert len(self.kv_caches) == 0
            for layer_name in sorted(
                    kv_caches,
                    key=lambda name: (extract_dsv4_layer_index(
                        self.model_config.hf_text_config, name), name)):
                self.kv_caches.append(kv_caches[layer_name])
            for layer_name, kv_cache in kv_caches.items():
                self.compilation_config.static_forward_context[
                    layer_name].kv_cache = [kv_cache]
        else:
            from vllm.v1.worker.utils import bind_kv_cache

            num_attn_module = 2 if self.model_config.hf_text_config.model_type == "longcat_flash" else 1
            bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches, num_attn_module)

        if self.enable_hamming_sparse is True:
            from vllm_ascend.worker.kvcomp_utils import init_and_bind_hashk_cache
            init_and_bind_hashk_cache(
                kv_caches=kv_caches,
                num_attn_module=num_attn_module,
                vllm_config=self.vllm_config,
                device=self.device,
                compilation_config=self.compilation_config,
                kvcomp_meta_data=self.kvcomp_meta_data
            )

        return kv_caches

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3815
    def _get_layer_kv_cache_specs(self, kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
        layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
        for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
            group_spec = group_kv_cache_spec.kv_cache_spec
            for layer_name in group_kv_cache_spec.layer_names:
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
                else:
                    layer_kv_cache_spec[layer_name] = group_spec
        return layer_kv_cache_spec

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3826
    def _get_attention_kv_cache_dims(self, layer_name: str, kv_cache_spec: AttentionSpec) -> tuple[int, int]:
        if isinstance(kv_cache_spec, MLAAttentionSpec):
            attn_layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
                [layer_name],
            )
            attn_layer = attn_layers[layer_name]
            if isinstance(attn_layer, MLAAttention):
                # DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim
                return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim
            # CacheOnlyAttentionLayer uses MLAAttentionSpec but isn't MLAAttention
            if isinstance(attn_layer, CacheOnlyAttentionLayer):
                return kv_cache_spec.head_size, kv_cache_spec.head_size
            raise TypeError(
                f"Expected MLAAttention layer for {layer_name}, got {type(attn_layer).__name__}."
            )

        head_size_v = kv_cache_spec.head_size_v if hasattr(kv_cache_spec, "head_size_v") else kv_cache_spec.head_size
        return kv_cache_spec.head_size, head_size_v

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3847
        return (value + alignment - 1) // alignment * alignment

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3851
    def _allocate_int8_cache_tensor(
        self,
        numel: int,
        alignment: int,
    ) -> torch.Tensor:
        """Allocate an int8 raw cache tensor.

        When KV transfer is enabled, the returned tensor's data_ptr is aligned
        to `alignment`. This keeps the original Mooncake/ADXL alignment behavior.
        """
        if numel <= 0:
            raise ValueError(f"Invalid cache tensor size: {numel}")

        if self.vllm_config.kv_transfer_config is None:
            return torch.zeros(numel, dtype=torch.int8, device=self.device)

        raw_tensor = torch.zeros(
            numel + alignment,
            dtype=torch.int8,
            device=self.device,
        )
        return self._align_memory(raw_tensor, alignment)[:numel]

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3874
    def _allocate_sparse_c8_indexer_tensors(
        self,
        dsa_k_tensor_size: int,
        dsa_k_scale_tensor_size: int,
        alignment: int,
        scale_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate dsa_k and dsa_k_scale from one aligned int8 raw allocation.

        Both returned tensors are logical views into the same underlying storage.
        This reduces HCCL/Mooncake registration count because register_buffer
        can merge these two views into one registered memory range.
        """
        if dsa_k_tensor_size <= 0:
            raise ValueError(
                f"Invalid dsa_k_tensor_size: {dsa_k_tensor_size}"
            )
        if dsa_k_scale_tensor_size <= 0:
            raise ValueError(
                f"Invalid dsa_k_scale_tensor_size: {dsa_k_scale_tensor_size}"
            )

        scale_dtype_size = torch.empty((), dtype=scale_dtype).element_size()

        # Ensure the scale view starts at an address aligned for scale_dtype.
        scale_offset = self._align_up(dsa_k_tensor_size, scale_dtype_size)
        total_raw_size = scale_offset + dsa_k_scale_tensor_size

        sparse_c8_raw_tensor = self._allocate_int8_cache_tensor(
            total_raw_size,
            alignment,
        )

        dsa_k_tensor = sparse_c8_raw_tensor[:dsa_k_tensor_size]
        dsa_k_scale_tensor = sparse_c8_raw_tensor[
            scale_offset:scale_offset + dsa_k_scale_tensor_size
        ]

        assert dsa_k_tensor.is_contiguous()
        assert dsa_k_scale_tensor.is_contiguous()
        assert dsa_k_scale_tensor.data_ptr() % scale_dtype_size == 0
        assert dsa_k_scale_tensor.numel() % scale_dtype_size == 0

        return dsa_k_tensor, dsa_k_scale_tensor

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L3929
    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cache and v cache, and the addr of both are aligned by 2M
        """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, torch.Tensor | torch.Tensor | None | None] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        # If some tensors are shared by linear layers and attention layers,
        # the same tensor format must be maintained even if some layers
        # have only linear or attention layers, for example, the mtp layer.
        self.hybrid_with_attn_and_mamba = False
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            use_mamba, use_attn = False, False
            for layer_name in kv_cache_tensor.shared_by:
                if isinstance(layer_kv_cache_spec[layer_name], MambaSpec):
                    use_mamba = True
                if isinstance(layer_kv_cache_spec[layer_name], AttentionSpec):
                    use_attn = True
            self.hybrid_with_attn_and_mamba = self.hybrid_with_attn_and_mamba or (use_mamba and use_attn)
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                # SUBTRACTED: 单张量分支（mamba/linear_attn/hybrid attn-mamba 或 cache_only_layers，
                #   model_runner_v1.py:L3965-L3980）与 use_compress 单张量分支（L3981-L3992）的分配体。
                #   两者都是「kv_transfer? _align_memory 对齐 : torch.zeros(int8)」的重复（与下方标准
                #   attn 分支同构、且不拆 K/V），保留标准 attn 这一份足以示范内存几何，按
                #   subtraction_plan.delete 折叠其分配体。must_keep 的 int8 裸分配在下方仍可见。
                if "attn" in layer_name and layer_name not in kv_cache_raw_tensors and not use_mamba:
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    current_kv_cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(current_kv_cache_spec, AttentionSpec)

                    if self.use_sparse:
                        # for deepseek v3.2, we split the kv cache according to the corresponding ratio
                        kv_cache_spec = layer_kv_cache_spec[layer_name]
                        current_sparse_c8 = kv_cache_spec_uses_sparse_c8(kv_cache_spec)
                        sparse_kv_cache_ratio = kv_cache_spec.sparse_kv_cache_ratio
                        # SUBTRACTED: A5 代际 sparse-c8 的 ratio 排布分叉
                        #   (current_sparse_c8 and get_ascend_device_type()==A5: ckv/qli/qli_scale)，
                        #   保留 A3/通用一支（k/v/dsa_k[/scale]），控制流同构（L4008-L4015）。
                        # A3 sparse C8: (k_ratio, v_ratio, qli_ratio, qli_scale_ratio)
                        k_tensor_split_factor = sparse_kv_cache_ratio[0]
                        v_tensor_split_factor = sparse_kv_cache_ratio[1]
                        dsa_k_tensor_split_factor = sparse_kv_cache_ratio[2]
                        dsa_k_scale_tensor_split_factor = sparse_kv_cache_ratio[3] if current_sparse_c8 else None
                    else:
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        assert k_dim > 0 and v_dim > 0
                        kv_head_dim_list = [
                            k_dim,
                            v_dim,
                        ]
                        if enable_fa_quant(self.vllm_config):
                            k_tensor_split_factor, v_tensor_split_factor = (
                                self.vllm_config.quant_config.get_kv_quant_split_factor(layer_name, kv_head_dim_list)
                            )
                        else:
                            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor(kv_head_dim_list)

                    k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
                    if v_tensor_split_factor is not None:
                        v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)
                    else:
                        v_tensor_size = None
                    dsa_k_tensor_size = None
                    dsa_k_scale_tensor_size = None
                    #### for deepseek sparse attention
                    if self.use_sparse:
                        dsa_k_tensor_size = int(kv_cache_tensor.size // dsa_k_tensor_split_factor)
                    if self.use_sparse and current_sparse_c8:
                        dsa_k_scale_tensor_size = int(kv_cache_tensor.size // dsa_k_scale_tensor_split_factor)

                    # Allocate raw int8 tensors. Even bf16/fp16 KV cache entries
                    # are allocated as int8 raw bytes first and then viewed as
                    # the target dtype in _reshape_kv_cache_tensors.
                    dsa_k_tensor = None
                    dsa_k_scale_tensor = None
                    v_tensor = None
                    k_tensor = self._allocate_int8_cache_tensor(
                        k_tensor_size,
                        alignment,
                    )
                    if v_tensor_size is not None:
                        v_tensor = self._allocate_int8_cache_tensor(
                            v_tensor_size,
                            alignment,
                        )

                    if self.use_sparse:
                        assert dsa_k_tensor_size is not None

                        if current_sparse_c8:
                            assert dsa_k_scale_tensor_size is not None

                            (
                                dsa_k_tensor,
                                dsa_k_scale_tensor,
                            ) = self._allocate_sparse_c8_indexer_tensors(
                                dsa_k_tensor_size=dsa_k_tensor_size,
                                dsa_k_scale_tensor_size=dsa_k_scale_tensor_size,
                                alignment=alignment,
                                scale_dtype=current_kv_cache_spec.scale_dtype,
                            )
                        else:
                            dsa_k_tensor = self._allocate_int8_cache_tensor(
                                dsa_k_tensor_size,
                                alignment,
                            )

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the attn kvcache for all shared layers
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            if self.use_sparse:
                                if current_sparse_c8:
                                    # SUBTRACTED: A5 三元组装配 (k_tensor, dsa_k, dsa_k_scale)
                                    #   （get_ascend_device_type()==A5，L4089-L4092），保留 A3 四元组。
                                    kv_cache_raw_tensors[layer_name_inner] = (
                                        k_tensor, v_tensor, dsa_k_tensor, dsa_k_scale_tensor
                                    )
                                else:
                                    kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor, dsa_k_tensor)
                            else:
                                kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor)
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4111
    def _adjust_kv_layout(
        self,
        raw_tensor: torch.Tensor,
        kv_cache_shape_list: list[int],
        kv_cache_dtype_list: list[int],
        page_size_bytes: int,
        overlap_full_kv_cache: bool = False,
    ):
        reshaped_kv_tensors = []
        base_storage_offset_bytes = raw_tensor.storage_offset()
        storage_offset_bytes = base_storage_offset_bytes
        for idx, (shape, dtype) in enumerate(zip(kv_cache_shape_list, kv_cache_dtype_list)):
            if overlap_full_kv_cache and idx == 2:
                storage_offset_bytes = base_storage_offset_bytes
            dtype_size = get_dtype_size(dtype)
            num_element_per_page = (
                page_size_bytes // dtype_size
            )

            stride = torch.empty(shape).stride()
            target_stride = (num_element_per_page, *stride[1:])
            assert storage_offset_bytes % dtype_size == 0
            tensor = torch.as_strided(
                raw_tensor.view(dtype),
                size=shape,
                stride=target_stride,
                storage_offset=storage_offset_bytes // dtype_size,
            )
            reshaped_kv_tensors.append(tensor)
            storage_offset_bytes += stride[0] * dtype_size
        return reshaped_kv_tensors

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4144
    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        for group in self._kv_cache_spec_attn_group_iterator():
            attn_backend = group.backend
            current_kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                current_kv_cache_spec = layer_kv_cache_spec[layer_name]

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if self.use_compress and isinstance(current_kv_cache_spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                    kv_tensor = kv_cache_raw_tensors[layer_name]
                    sum_page_size_bytes = kv_tensor.numel()
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
                    assert num_blocks == kv_cache_config.num_blocks, \
                        f"num_blocks: {num_blocks} should be equal to " \
                        f"kv_cache_config.num_blocks: {kv_cache_config.num_blocks}"
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, current_kv_cache_spec.block_size,
                        current_kv_cache_spec.num_kv_heads,
                        current_kv_cache_spec.head_size)
                    kv_cache_shape_list = [kv_cache_shape]
                    kv_cache_dtype_list = [current_kv_cache_spec.dtype]
                    overlap_full_kv_cache = False

                    if hasattr(current_kv_cache_spec, "scale_dim") and current_kv_cache_spec.scale_dim != 0:
                        indexer_k_shape = kv_cache_shape
                        indexer_scale_shape = self.attn_backend.get_kv_cache_shape(
                                                num_blocks, current_kv_cache_spec.block_size,
                                                current_kv_cache_spec.num_kv_heads,
                                                current_kv_cache_spec.scale_dim
                                                )
                        # SUBTRACTED: A5 代际 indexer 的三段视图 (k/scale/full) + overlap_full_kv_cache
                        #   分支（get_ascend_device_type()==A5，L4195-L4210），保留 A3 的两段视图。
                        kv_cache_shape_list = [indexer_k_shape, indexer_scale_shape]
                        kv_cache_dtype_list = [
                            current_kv_cache_spec.dtype, current_kv_cache_spec.scale_dtype
                        ]
                        overlap_full_kv_cache = False

                    kv_cache = self._adjust_kv_layout(kv_tensor,
                                           kv_cache_shape_list,
                                           kv_cache_dtype_list,
                                           current_kv_cache_spec.page_size_bytes,
                                           overlap_full_kv_cache=overlap_full_kv_cache,
                                           )

                    kv_caches[layer_name] = kv_cache
                elif isinstance(current_kv_cache_spec, AttentionSpec):
                    # cache_only_layers (extract_hidden_states) are allocated
                    # as a single tensor by the branch at the top of
                    # _allocate_kv_cache_tensors; route them to the dedicated
                    # elif branch below before the sparse branch tries to
                    # unpack them as a (k, v, dsa_k[, scale]) tuple.
                    if self.use_sparse and "cache_only_layers" not in layer_name:
                        current_sparse_c8 = kv_cache_spec_uses_sparse_c8(current_kv_cache_spec)
                        if current_sparse_c8:
                            # SUBTRACTED: A5 的 (k, dsa_k, dsa_k_scale) 三元组解包（L4235-L4244），
                            #   保留 A3 的 (k, v, dsa_k, dsa_k_scale) 四元组。
                            raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor, raw_dsa_k_scale_tensor = (
                                kv_cache_raw_tensors[layer_name]  # type: ignore
                            )
                            assert raw_dsa_k_tensor is not None
                            assert raw_dsa_k_scale_tensor is not None
                            sum_page_size_bytes = (
                                raw_k_tensor.numel()
                                + raw_v_tensor.numel()
                                + raw_dsa_k_tensor.numel()
                                + raw_dsa_k_scale_tensor.numel()
                            )
                        else:
                            raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                                layer_name]
                            assert raw_dsa_k_tensor is not None
                            sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    elif self.use_hybrid_blocks and self.hybrid_with_attn_and_mamba:
                        # Currently, we ensure that the same kvcache format is used even if there
                        # is no shared layer, such as the full attention mtp layer of qwen3.5, etc.
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name], kv_cache_raw_tensors[layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel()
                    elif "cache_only_layers" in layer_name:
                        # Single tensor for extract_hidden_states (no K/V split)
                        raw_tensor = kv_cache_raw_tensors[layer_name]
                        assert raw_tensor is not None
                        assert raw_tensor.numel() % current_kv_cache_spec.page_size_bytes == 0
                        num_blocks = raw_tensor.numel() // current_kv_cache_spec.page_size_bytes
                        assert num_blocks >= kv_cache_config.num_blocks
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                        k_cache = raw_tensor.view(current_kv_cache_spec.dtype).view(kv_cache_shape)
                        kv_caches[layer_name] = k_cache
                        continue  # Skip the rest of the AttentionSpec handling
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name
                        ]
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate. Since different GPUs may have
                    # different number of layers and memory capacities,
                    # `kv_cache_config.num_blocks` is set to the min. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    # SUBTRACTED: hybrid attn+mamba 的 conv_block_padding 切分（use_hybrid_blocks 且
                    #   get_supported_kernel_block_sizes，L4301-L4331）——把 KV padding 块对齐到 mamba
                    #   page_size 是另一专题（与 ch04 mamba block 相关），本章聚焦 attn/MLA KV 几何，
                    #   保留下方常规 get_kv_cache_shape 一支（else，L4332-L4338）即可。
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks,
                        current_kv_cache_spec.block_size,
                        current_kv_cache_spec.num_kv_heads,
                        current_kv_cache_spec.head_size,
                    )
                    if not isinstance(current_kv_cache_spec, MLAAttentionSpec):
                        k_shape = kv_cache_shape[1:]
                        if hasattr(current_kv_cache_spec, "head_size_v"):
                            v_shape = (*kv_cache_shape[1:-1], current_kv_cache_spec.head_size_v)
                        else:
                            v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        k_shape = (
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            k_dim,
                        )
                        # SUBTRACTED: A5 sparse-c8 的 ckv k_shape 重算（kv_lora + k_rope*2 + 4*4，
                        #   get_ascend_device_type()==A5，L4355-L4367），保留 A3 的 (…, k_dim)。
                        v_shape = (
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            v_dim,
                        )
                    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype
                    if enable_fa_quant(self.vllm_config):
                        k_cache_dtype, v_cache_dtype = self.vllm_config.quant_config.get_kv_quant_dtype(
                            layer_name, current_kv_cache_spec.dtype, self.model_config
                        )
                    # SUBTRACTED: A5 sparse-c8 ckv 改 float8_e4m3fn（self.c8_k_cache_dtype，L4381-L4382）。
                    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
                    # SUBTRACTED: A5 sparse-c8 时 v_cache=None（L4385-L4386），保留 A3 的 v_cache view。
                    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)

                    if self.use_sparse:
                        dsa_k_cache_shape = (
                            num_blocks,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            self.model_config.hf_text_config.index_head_dim,
                        )
                        if current_sparse_c8:
                            # dsa_k
                            dsa_k_cache = raw_dsa_k_tensor.view(self.c8_k_cache_dtype).view(dsa_k_cache_shape)
                            # dsa_k_scale
                            dsa_k_scale_cache_shape = (
                                num_blocks,
                                current_kv_cache_spec.block_size,
                                current_kv_cache_spec.num_kv_heads,
                                1,
                            )
                            assert raw_dsa_k_scale_tensor is not None
                            dsa_k_scale_cache = (
                                raw_dsa_k_scale_tensor
                                .view(self.c8_k_scale_cache_dtype)
                                .view(dsa_k_scale_cache_shape)
                            )
                            # SUBTRACTED: A5 的 (k, dsa_k, dsa_k_scale) 三元组装配（L4413-L4414），
                            #   保留 A3 的 (k, v, dsa_k, dsa_k_scale) 四元组。
                            kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)
                        else:
                            # dsa_k
                            dsa_k_cache = raw_dsa_k_tensor.view(current_kv_cache_spec.dtype).view(dsa_k_cache_shape)
                            kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(current_kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel() % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // current_kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    # NOTE(zxr): in order to keep all tensor contiguous, we align ssm and kv block
                    # with same page size, so have to add extra padding block for kv. Mamba 的
                    # conv/ssm 跨步切分细节（与 ch04 mamba block 相关）非本章主线，仅保留循环骨架。
                    for shape, dtype in zip(current_kv_cache_spec.shapes, current_kv_cache_spec.dtypes):
                        target_shape = (num_blocks, *shape)
                        target_idx += math.prod(target_shape) * get_dtype_size(dtype)
                        tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4464
    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        self.kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            # SUBTRACTED: pcp_size>1 时 pcp_manager.initialize_slot_mapping()（L4485-L4486）——
            #   context-parallel 是正交特性，删去不影响 kernel_block_sizes 的装配主控制流。
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from the backend.
                attn_groups = self.attn_groups[kv_cache_group_id]
                backends = [attn_group.backend for attn_group in attn_groups]
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
                selected_kernel_size = select_common_block_size(
                    kv_manager_block_size, backends
                )
                self.kernel_block_sizes.append([selected_kernel_size])
            else:
                # This is likely Mamba or other non-attention cache, no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block.
                self.kernel_block_sizes.append([0])

        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            max_num_blocks_per_req = cdiv(max_model_len, block_sizes[i] * get_total_cp_world_size())
            # SUBTRACTED: MambaSpec 的 mamba_blocks_per_req 容量上调（enable_prefix_caching /
            #   num_speculative_blocks，L4519-L4524）——InputBatch 内部容量细节，非重建判定主线。
            max_num_blocks.append(max_num_blocks_per_req)

        if (block_sizes != [self.cache_config.block_size]
                or self.kernel_block_sizes != [[self.cache_config.block_size]]
                or len(kv_cache_config.kv_cache_groups) > 1):
            # SUBTRACTED: CPU offload assert（offload_config.uva.cpu_offload_gb==0，L4530-L4534）——
            #   权重 offload 是正交特性。
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config
                    else 0
                ),
                kernel_block_sizes=self.kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                kv_cache_groups=kv_cache_config.kv_cache_groups,
            )

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4657
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        """
        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}

        kv_cache_spec: dict[str, list[KVCacheSpec]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        # NOTE: Must process Attention/MLAAttention before MambaBase to maintain
        # ordering expected by graph parameter update logic in attention backends.
        mamba_layers: dict[str, MambaBase] = {}
        attn_layer_names = set()
        for layer_name, attn_module in attn_layers.items():
            # SUBTRACTED: kv_sharing 跳过分支（Attention 且 kv_sharing_target_layer_name，
            #   L4676-L4686）、use_compress 直取 spec 分支（L4687-L4690）、普通 Attention 分支
            #   （L4691-L4694）——本章对 spec 只需展示 MLA→AscendMLAAttentionSpec 这一回指 ch04 的落点。
            if isinstance(attn_module, MLAAttention):
                if self.use_sparse:
                    # `MLAAttentionSpec` is temporarily patched to `AscendMLAAttentionSpec`.
                    # Re-importing it at runtime will therefore resolve to the patched class.
                    # Rename it here to make this behavior explicit.
                    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=self.block_size,
                        num_kv_heads=1,
                        head_size=sum(self.sparse_head_dim),
                        sparse_head_dim=self.sparse_head_dim,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                        cache_sparse_c8=self.ascend_config.is_sparse_c8_layer(layer_name),
                    )
                elif spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                    if getattr(attn_module.impl, "fa_quant_layer", False):
                        head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                        dtype, cache_dtype_str = attn_module.impl.dtype, None
                    else:
                        head_size, dtype, cache_dtype_str = spec.head_size, spec.dtype, spec.cache_dtype_str
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        cache_dtype_str=cache_dtype_str,
                    )
                    attn_layer_names.add(layer_name)

            elif isinstance(attn_module, MambaBase):
                mamba_layers[layer_name] = attn_module
            # SUBTRACTED: CacheOnlyAttentionLayer 分支（extract_hidden_states，L4732-L4753）——
            #   draft model 的 spec 重建，非本章 MLA 主线。

        # SUBTRACTED: mamba page_size 对齐尾段（把 attn_page_size 抬到 mamba_page_size_padded，
        #   L4755-L4764）——与 _allocate 顶部 mamba 单张量分支呼应，本章不展开 mamba。

        return kv_cache_spec
