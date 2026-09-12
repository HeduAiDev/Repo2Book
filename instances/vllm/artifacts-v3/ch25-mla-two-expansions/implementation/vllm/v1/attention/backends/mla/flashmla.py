# SOURCE: vllm/v1/attention/backends/mla/flashmla.py
# ch25 切面（站 7/13 / m07 / must_keep flash_mla_with_kvcache）：
# FlashMLA 后端四件套全文减法——
#   FlashMLABackend（家族身份证 + stride_order + kernel block 64）
#   FlashMLADecodeMetadata / FlashMLAMetadata（sched meta 载体）
#   FlashMLAMetadataBuilder（reorder_batch_threshold=128 'process small
#     prefills with decode pathway' + _build_decode 的 get_mla_metadata 计划）
#   FlashMLAImpl.forward_mqa（MQA kernel 真实调用面——bf16 kernel 逐字；
#     fp8 双 kernel 分派按 delete[6] 删）
# SUBTRACTED：@cache 的 get_mla_metadata 缓存位无关紧要面、fp8 dense 元数据
#   族（L184-L206）——delete[6]；VLLM_BATCH_INVARIANT 手工 tile_scheduler
#   段（L287-L315）——delete[5]；reshape_query_for_spec_decode /
#   reshape_attn_output_for_spec_decode（L284/L344）——delete[4]（spec
#   decode 归 ch33：uniform 段 view 等价）。
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType  # noqa: F401  (类型位)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_dense_supported,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L47-L105 FlashMLABackend
#   —— 减法子集（supports_combination 探针面归 ch21）
class FlashMLABackend(MLACommonBackend):
    # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L48 supported_dtypes
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L49-L55
    #   supported_kv_cache_dtypes
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list["int | MultipleOf"]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L57-L59 —— 逐字
        return [64]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L61-L67 —— 逐字
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L69-L71 —— 逐字
        return "FLASHMLA"

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L73-L75 —— 逐字
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L77-L79 —— 逐字
        return FlashMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L81-L83 —— 逐字
        return capability.major in [9, 10]

    # SUBTRACTED: supports_combination（L85-L105）——探针面归 ch21


# SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L108-L110
#   FlashMLADecodeMetadata —— 逐字
@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L108-L110（锚点双置：声明上方注释同文）
    scheduler_metadata: FlashMLASchedMeta


# SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L113-L115
#   FlashMLAMetadata —— 逐字
@dataclass
class FlashMLAMetadata(MLACommonMetadata[FlashMLADecodeMetadata]):
    # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L113-L115（锚点双置：声明上方注释同文）
    pass


# SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L118-L213
#   FlashMLAMetadataBuilder —— 减法子集
class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):
    # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L119-L122 —— 逐字
    #   （must_keep：后端自报阈值 128——'process small prefills with decode
    #   pathway'；FA-MLA 同款=512）
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    reorder_batch_threshold: int = 128  # process small prefills with decode pathway
    # ^ TODO(matt): tune this

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L124-L133 —— 逐字
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, FlashMLAMetadata
        )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )

        self.cg_buf_tile_scheduler_metadata = None
        self.cg_buf_num_splits = None
        self.is_fp8_kvcache = is_quantized_kv_cache(
            vllm_config.cache_config.cache_dtype
        )

        num_sms = num_compute_units(self.device.index)

        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L147-L159 —— 逐字
        #   （host：has_full_cudagraphs() 恒 False → 不分配 CG 缓冲）
        if vllm_config.compilation_config.has_full_cudagraphs():
            self.cg_buf_tile_scheduler_metadata = torch.zeros(
                # Upper bound on size (<= #SMs, TileSchedulerMetaDataSize)
                # TileSchedulerMetaDataSize = 8
                (num_sms, 8),
                device=self.device,
                dtype=torch.int32,
            )
            self.cg_buf_num_splits = torch.empty(
                (vllm_config.scheduler_config.max_num_seqs + 1),
                device=self.device,
                dtype=torch.int32,
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashMLADecodeMetadata:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L161-L213 ——
        #   减法子集（fp8 dense 元数据族 L184-L206 按 delete[6] 删）
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # we use the max but all should be the same due to uniform length requirement
        max_query_len = query_lens_cpu.max().item()
        num_q_heads = self.num_q_heads
        if self.dcp_world_size > 1:
            num_q_heads *= self.dcp_world_size
        num_q_tokens_per_head_k = max_query_len * num_q_heads // 1
        scheduler_metadata, _ = get_mla_metadata(
            seq_lens_device,
            num_q_tokens_per_head_k,
            1,  # MQA for the decode path
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        # SUBTRACTED: get_mla_metadata_dense_fp8 与 CG 缓冲拷贝段
        #   （L184-L206）——delete[6]

        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            scheduler_metadata=scheduler_metadata,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


# SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L216-L264 FlashMLAImpl
#   —— 减法子集（can_return_lse_for_decode=True 的单进程形态——delete[6]）
class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L219-L246 —— 逐字
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        is_supported, reason = is_flashmla_dense_supported()
        assert is_supported, reason

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashMLAImpl"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla.py:L266-L346 ——
        #   减法子集（must_keep：MQA kernel 真实调用面）
        # TODO: (zyongye) decode function for mla here
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        # mypy assertion: q is now always a tensor
        assert isinstance(q, torch.Tensor)

        num_decodes = attn_metadata.num_decodes
        # SUBTRACTED: reshape_query_for_spec_decode(q, num_decodes)
        #   （L284）——delete[4]；uniform decode 段的等价 view：
        q = q.view(num_decodes, -1, *q.shape[1:])

        scheduler_metadata = attn_metadata.decode.scheduler_metadata
        # SUBTRACTED: VLLM_BATCH_INVARIANT 手工 tile_scheduler 段
        #   （L287-L315）——delete[5]
        # SUBTRACTED: flash_mla_with_kvcache_fp8 支（L317-L330）——delete[6]
        #   （fp8/非 fp8 双 kernel 分派保留 bf16 一支）
        o, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=scheduler_metadata,
            softmax_scale=self.scale,
            causal=True,
            is_fp8_kvcache=False,
        )

        # SUBTRACTED: reshape_attn_output_for_spec_decode(o)（L344）——
        #   delete[4]（等价 flatten）
        o = o.reshape(-1, *o.shape[2:])

        return o, lse
