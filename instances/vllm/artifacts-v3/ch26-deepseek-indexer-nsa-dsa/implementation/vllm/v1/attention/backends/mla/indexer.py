# SOURCE: vllm/v1/attention/backends/mla/indexer.py —— ch26 主文件（站 5/m04）
# 全文减法：split_indexer_prefill_chunks（L77-L123 逐字）+ 双后端族
# （DeepseekV32IndexerBackend L126-L167 / DeepseekV4IndexerBackend L170-L177）+
# metadata 族 + get_max_prefill_buffer_size（L442-L452 逐字——魔数 40 的
# workspace 账注释原文）+ DeepseekV32IndexerMetadataBuilder（L455-L978 减法）+
# build_prefill_chunk_metadata（L981-L1088 减法）。
# SUBTRACTED（挂 dossier.subtraction_plan.delete[0] 批准项——indexer.py 侧
# 只删五小块，dcp_world_size=1 时单卡语义逐字不变）：
#   delete[0]①：_dcp_localize_decode_seq_lens 方法（L596-L614）
#   delete[0]②：build 里 build_prefill_chunk_metadata 调用的 dcp 形参透传
#     （L855-L857，函数形参自带默认 0/1/1）
#   delete[0]③：global_seq_lens_for_decode 守卫块（L883-L885）连同 None-直通
#     下游（L893-L901 的 _prepare_global_decode_seq_lens 调用 + L963 的
#     global_seq_lens= 传参 + dataclass 字段 L420 + 方法本体 L736-L762）
#   delete[0]④：_dcp_localize 调用块（L922-L928，含其上方 DCP 局部化长注释）
#   delete[0]⑤：build_prefill_chunk_metadata 的 DCP 段（L1013-L1027，守卫
#     dcp_world_size>1 单卡恒假）
#   ⚠ L886-L921（seq_lens/block_table 切片、_prepare_decode_tensors、
#     seq_lens_is_buffer_view）是 decode 主路径——绝不删。
# HOST SEAM：_prepare_uniform_decode_kernel / BuildPrefillChunkMetadataKernel
# 为 @triton.jit/VllmJitKernel——host 以 TritonKernelShim 下标/直接调用垫片
# 承载精确数学（调用点逐字；kernel 体见真实源锚）。
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import TritonKernelShim
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L43-L74
#   _prepare_uniform_decode_kernel —— HOST SEAM 精确数学：
#   idx = req_id*max_decode_len + local；per-token 上下文长 =
#   seq_len - max_decode_len + local + 1；块表行复制；decode_lens 全置 1。
def _prepare_uniform_decode_math(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE=1024,
):
    # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L43-L74 —— HOST SEAM
    max_d = int(max_decode_len)
    total = seq_lens_ptr.shape[0] * max_d
    for idx in range(total):
        req_id, local_idx = idx // max_d, idx % max_d
        seq_len = int(seq_lens_ptr[req_id].item())
        decode_seq_lens_ptr[idx] = seq_len - max_d + local_idx + 1
        expanded_block_table_ptr[idx] = block_table_ptr[req_id]
    if total < decode_lens_ptr.shape[0]:
        decode_lens_ptr[total:] = 0
    decode_lens_ptr[:total] = 1


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L43 _prepare_uniform_
#   decode_kernel —— HOST SEAM 垫片（kernel[grid](...) 调用面逐字）
_prepare_uniform_decode_kernel = TritonKernelShim(_prepare_uniform_decode_math)


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L77-L123
#   split_indexer_prefill_chunks —— 逐字（双预算切块：N 约束 + logits 约束）
def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L77-L123（锚点双置）
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else max(1, chunk_m)
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L126-L167
#   DeepseekV32IndexerBackend —— 逐字（V3.2 indexer cache 后端：块 64、
#   头维 {32,64,128}、identity stride order = 不兼容跨层布局信号）
class DeepseekV32IndexerBackend(AttentionBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L127-L129
        return True

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L130-L132
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L134-L137
        return [1, 64] if current_platform.is_rocm() else [64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L139-L141
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L142-L145
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L147-L156 —— 逐字
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L158-L167 —— 逐字
        if include_num_layers_dimension:
            # DeepseekV32Indexer kernels do not support cross-layer
            # KV cache layout. Identity permutation keeps num_layers
            # first, signaling incompatibility.
            return (0, 1, 2, 3)
        return (0, 1, 2)


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L170-L177
#   DeepseekV4IndexerBackend —— 逐字（V4 子类块 256——两代共享后端族证据）
class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L171-L173
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L175-L177
        return [256]


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L180-L196
#   DeepseekV32IndexerPrefillChunkMetadata —— 逐字
@dataclass
# SOURCE: indexer.py:L180-L196（锚点双置）
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    # Under DCP (dcp_world_size > 1) these hold this rank's local row bounds;
    # otherwise they hold the global bounds.
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False
    local_cu_seq_lens: torch.Tensor | None = None
    local_total_seq_lens: int = 0
    max_local_total_seq_lens: int = 0


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L205-L398
#   BuildPrefillChunkMetadataKernel —— HOST SEAM 精确数学（DCP_WORLD=1 形态：
#   cu_ks[out] = row_start（本请求 K 行基址）；cu_ke[out] = row_start +
#   global_ctx // COMPRESS_RATIO（逐 token 因果长）；token_to_seq = 请求号；
#   row_start 别名 cu_compressed_seq_lens。kernel 的 DCP 分支
#   （DCP_WORLD>1 的 interleave 均分）随 delete[0]⑤ 的守卫恒假，不进镜像）
def _build_prefill_chunk_metadata_math(
    query_start_loc,
    uncompressed_seq_lens,
    cu_compressed_seq_lens,
    row_start_cu_compressed_seq_lens,
    token_to_seq,
    cu_compressed_seq_len_ks,
    cu_compressed_seq_len_ke,
    query_slice_start,
    query_slice_stop,
    DCP_RANK,
    DCP_WORLD,
    DCP_INTERLEAVE,
    num_reqs=None,
    COMPRESS_RATIO=1,
    BLOCK_SIZE=1024,
):
    # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L242-L296 —— HOST SEAM
    for batch_idx in range(num_reqs):
        query_start = int(query_start_loc[batch_idx].item())
        query_end = int(query_start_loc[batch_idx + 1].item())
        query_len = query_end - query_start

        seq_start = int(cu_compressed_seq_lens[batch_idx].item())
        seq_end = int(cu_compressed_seq_lens[batch_idx + 1].item())
        compressed_seq_len = seq_end - seq_start

        row_start = int(row_start_cu_compressed_seq_lens[batch_idx].item())
        uncompressed_seq_len = int(uncompressed_seq_lens[batch_idx].item())
        start_pos = uncompressed_seq_len - query_len

        for offset in range(query_len):
            abs_pos = query_start + offset
            if not (query_slice_start <= abs_pos < query_slice_stop):
                continue
            out_pos = abs_pos - query_slice_start
            # cu_seq_len_ks: row start in the gathered K buffer.
            cu_compressed_seq_len_ks[out_pos] = row_start
            # cu_seq_len_ke: row start + per-token context length.
            global_ctx = start_pos + 1 + offset
            len_per_token = global_ctx // COMPRESS_RATIO
            cu_compressed_seq_len_ke[out_pos] = row_start + len_per_token

        # Compute token_to_seq
        for offset in range(compressed_seq_len):
            token_to_seq[seq_start + offset] = batch_idx


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L401 _BUILD_PREFILL_CHUNK_
#   METADATA_KERNEL —— HOST SEAM 垫片（直接调用面逐字）
_BUILD_PREFILL_CHUNK_METADATA_KERNEL = _build_prefill_chunk_metadata_math


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L404-L406
#   DeepseekV32IndexerPrefillMetadata —— 逐字
@dataclass
# SOURCE: indexer.py:L404-L406（锚点双置）
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L409-L420
#   DeepSeekV32IndexerDecodeMetadata —— 减法（delete[0]③：global_seq_lens
#   字段删——其唯一写者/读者同批删除）
@dataclass
# SOURCE: indexer.py:L409-L420（锚点双置）
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_fp4_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    # SUBTRACTED: global_seq_lens 字段（indexer.py:L420，默认 None）——
    #   delete[0]③（DCP 深讲归 ch34）


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L423-L439
#   DeepseekV32IndexerMetadata —— 逐字
@dataclass
# SOURCE: indexer.py:L423-L439（锚点双置）
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L442-L452
#   get_max_prefill_buffer_size —— 逐字（魔数 40 的 workspace 账注释原文）
def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L442-L452（锚点双置）
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L455-L978
#   DeepseekV32IndexerMetadataBuilder —— 减法子集（delete[0]①-④；
#   __init__ 的 DCP 标量位/interleave 守卫/fp4 断言/压缩坐标换算全逐字保留）
class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    # The indexer opts out of the shared reorder-threshold vote (see __init__),
    # so this is None; its own split uses self.decode_threshold.
    reorder_batch_threshold: int | None = None
    requires_block_table_width = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L461-L467 —— 逐字
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, block_table_width: int, **kwargs) -> None:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L469-… __init__
        #   —— 逐字
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        parallel_config = self.vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.pcp_world_size = parallel_config.prefill_context_parallel_size
        self.use_pcp = self.pcp_world_size > 1
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        # The DCP sparse-indexer code is parameterized by interleave size, but
        # interleave > 1 is not yet validated end-to-end (gsm8k parity fails),
        # so fail closed here rather than silently produce wrong output.
        if self.dcp_world_size > 1 and self.cp_kv_cache_interleave_size > 1:
            raise NotImplementedError(
                "DCP sparse indexer currently supports only "
                f"cp_kv_cache_interleave_size=1 (got "
                f"{self.cp_kv_cache_interleave_size})."
            )
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = (
            self.vllm_config.attention_config.use_fp4_indexer_cache
        )

        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )

        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        self.reorder_batch_threshold = None
        # NOTE: SM100 datacenter GPUs support any next_n natively via the
        # multi-atom paged MQA logits kernels (FP8 and FP4 indexer
        # caches). Outside the SM100 family the FP8
        # paged MQA logits kernel only supports next_n in (1, 2)
        # (deepgemm smxx_fp8_fp4_paged_mqa_logits.hpp:233), so flatten there.
        self.use_flattening = not current_platform.is_device_capability_family(
            100
        ) and next_n not in (1, 2)
        logger.info_once(
            "DSA indexer decode path: use_flattening=%s "
            "(next_n=%d, use_fp4_indexer_cache=%s)",
            self.use_flattening,
            next_n,
            self.use_fp4_indexer_cache,
        )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # Shared workspace for decode seq_lens. Native MTP views this as
        # (B, max_decode_len) at runtime, keeping context_lens contiguous even
        # when max_decode_len is smaller than next_n.
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # SUBTRACTED: global_decode_seq_lens_buffer（indexer.py:L545-L549）——
        #   delete[0]③（唯一消费者 _prepare_global_decode_seq_lens 已删）
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.expanded_block_table_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens, block_table_width),
            dtype=torch.int32,
            device=self.device,
        )

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio
        if self.dcp_world_size > 1 and self.compress_ratio > 1:
            raise NotImplementedError(
                "DCP is not supported with sparse indexer KV compression "
                f"(compress_ratio={self.compress_ratio})."
            )

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (DeepseekV4)
            # Compressed slot mapping output buffer
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

    # SUBTRACTED: _dcp_localize_decode_seq_lens 方法（indexer.py:L596-L614）
    #   ——delete[0]①（单卡不被调用；DCP 深讲归 ch34）

    # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L616-L734
    #   _prepare_decode_tensors —— 逐字（decode 展开：flatten 两路 + native 2D）
    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Expand seq_lens/block_table/decode_lens for the decode kernels.

        Flatten path (not use_native, max_decode_len > 1):
          Each multi-token decode request is expanded into individual
          single-token entries so the kernel always sees next_n=1.

        Native path (use_native or max_decode_len == 1):
          Plain decode or spec-decode with 2D per-token context lengths.

        Returns (seq_lens, block_table, decode_lens, batch_size, requires_padding).
        seq_lens is 1D (batch_size,) for flatten/plain, 2D (B, max_decode_len)
        for native MTP.
        """
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L616-L734（锚点双置）
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            if min_decode_len == max_decode_len:
                # Uniform decode lengths.
                num_decode_tokens = num_decodes * max_decode_len
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                expanded_offsets = torch.repeat_interleave(
                    seq_lens - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.decode_seq_lens_buffer[:actual_expanded] = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len): token j attends to
                # L - max_decode_len + j + 1 KV tokens.
                seq_lens_buffer = self.decode_seq_lens_buffer[
                    : num_decodes * max_decode_len
                ].view(num_decodes, max_decode_len)
                seq_lens_buffer[:] = (
                    seq_lens.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                )
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    # SUBTRACTED: _prepare_global_decode_seq_lens 方法（indexer.py:L736-L762）
    #   ——delete[0]③（其 None-直通 + 调用点同批删除；DCP 归 ch34）

    # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L764-L978 build —— 减法
    #   子集（delete[0]②③④；decode 主路径 L886-L964 与压缩坐标换算逐字）
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L764-…（锚点双置）
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                require_uniform=not self.use_flattening,
                treat_short_extends_as_decodes=not self.use_pcp,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        if self.compress_ratio > 1:
            padded_num_tokens = num_tokens
            if self.pcp_world_size > 1:
                padded_num_tokens = slot_mapping.shape[0] // self.pcp_world_size
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
            if self.pcp_world_size > 1:
                compressed_slot_mapping = get_pcp_group().all_gather(
                    self.compressed_slot_mapping_buffer[:padded_num_tokens],
                    dim=0,
                )
            compressed_seq_lens = seq_lens // self.compress_ratio

        prefill_metadata = None
        if num_prefills > 0:
            # This CPU value is an upper bound for async-spec extend rows.  It
            # is safe for chunking/allocation because CUDA metadata below is
            # built from exact device seq_lens and gather ignores the tail.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below).
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    self.compress_ratio,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                    # SUBTRACTED: dcp_rank/dcp_world_size/cp_kv_cache_
                    #   interleave_size 透传（indexer.py:L855-L857）——
                    #   delete[0]②（形参默认 0/1/1，单卡语义不变）
                )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            # SUBTRACTED: global_seq_lens_for_decode 守卫块（indexer.py:
            #   L883-L885）与其上 DCP 局部化时序长注释（L875-L882）——
            #   delete[0]③（dcp_local_seq_lens 单卡恒 None）

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len <= next_n

            # SUBTRACTED: _prepare_global_decode_seq_lens 调用（indexer.py:
            #   L893-L901）——delete[0]③

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )

            seq_lens_is_buffer_view = (use_native and next_n > 1) or (
                not use_native and max_decode_len > 1
            )

            # SUBTRACTED: _dcp_localize 调用块（indexer.py:L922-L928）——
            #   delete[0]④（对已展开的逐 token 全局界做本 rank 局部化；
            #   单卡无 DCP）

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                if seq_lens_is_buffer_view:
                    seq_lens //= self.compress_ratio
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
            # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
            # kernels see the same (B, next_n) layout as the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            # DeepGEMM is required for the paged MQA logits on CUDA devices
            if current_platform.is_cuda() and has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
                # SUBTRACTED: global_seq_lens= 传参（indexer.py:L963）——
                #   delete[0]③
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata


# SOURCE: vllm/v1/attention/backends/mla/indexer.py:L981-L1088
#   build_prefill_chunk_metadata —— 减法子集（delete[0]⑤；因果界核调用面
#   逐字）
def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    # SOURCE: vllm/v1/attention/backends/mla/indexer.py:L981-…（锚点双置）
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    local_cu_seq_lens = cu_seq_lens
    local_total_seq_lens = total_seq_lens
    max_local_total_seq_lens = total_seq_lens
    # SUBTRACTED: DCP 段（indexer.py:L1013-L1027——dcp_world_size>1 时算本
    #   rank 局部行界）——delete[0]⑤（守卫单卡恒假）

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    # Under DCP the kernel writes this rank's local row bounds into
    # cu_seq_len_ks/ke; otherwise local_cu_seq_lens aliases cu_seq_lens.
    _BUILD_PREFILL_CHUNK_METADATA_KERNEL(
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        local_cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        num_reqs=num_reqs,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
        local_cu_seq_lens=local_cu_seq_lens,
        local_total_seq_lens=local_total_seq_lens,
        max_local_total_seq_lens=max_local_total_seq_lens,
    )
