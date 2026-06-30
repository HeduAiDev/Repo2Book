# vllm_ascend/attention/attention_v1.py —— subtract-only 精简版（标准 MHA 的 NPU 内核与状态机）
#
# 本章主角：一台五态机（AscendAttentionState）决定标准注意力后端走哪条 torch_npu 算子路径——
# 把基座 vLLM FlashAttention 的 CUDA 内核逐一换成 NPU 算子：
#   纯 decode → _npu_paged_attention（forward_paged_attention）
#   prefill/混批 → npu_fused_infer_attention_score（forward_fused_infer_attention）
#   写 KV → _npu_reshape_and_cache（reshape_and_cache → DeviceOperator）
#   workspace 预取 → *_get_workspace（full_graph_pa，NPU 算子的特有节拍）
#
# host 无 CANN/torch_npu：测试在 sys.modules 桩一个「记录调用」的 torch_npu 替身。下列纯 Python
# 控制流可在 host 验证与真仓一致：五态机分流 / split_decodes_and_prefills 拆批 /
# slot_mapping·block_table 装配 / forward_impl 按状态选 paged vs fused。真实算子不真算（昇腾才有内核）。
from dataclasses import dataclass
from enum import Enum

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import get_graph_params, update_graph_params_workspaces
from vllm_ascend.device.device_op import DeviceOperator

# SUBTRACTED: 大量 import（attention_v1.py:L25,L39-L40,L44-L66）—— TP rank 工具 / SchedulerOutput /
#   CrossAttentionSpec / context_parallel.common_cp（PCP/DCP 元数据）/ kvcomp_attn（hamming 稀疏）/
#   flashcomm2 / weak_ref_tensors / KVCompMetaData / draft-graph params 等，均服务已减的旁路特性。

# default max value of sliding window size
SWA_INT_MAX = 2147483647


# SOURCE: vllm_ascend/attention/attention_v1.py:L73-L74
@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L77-L82
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L84-L90
        # f7 收口：运行期 enable_cp() 为真 → 切到 CP 版 impl（CP 组排布回指 ch08）。
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L92-L98
        # f7 收口：同上，enable_cp() 为真 → CP 版 builder。
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L100-L108
        # 首维 2 = key/value 合存一张；slot_mapping 寻址、_npu_reshape_and_cache 都据此布局。
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    # SUBTRACTED: swap_blocks / copy_blocks（attention_v1.py:L110-L136）—— KV 块搬运/拷贝
    #   （preemption/CoW 辅助，纯索引赋值），非注意力前向主线（ch18 已点其非 v1 契约）。

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L138-L140
        return [128]


# SOURCE: vllm_ascend/attention/attention_v1.py:L143-L148
class AscendAttentionState(Enum):
    # 五态机本体：当前 batch 形态落在哪个状态，forward_impl / _get_fia_params 据此分流。
    PrefillNoCache = 0   # 纯新 prefill，无历史 KV（block_table=None，KV 即本步算的）
    PrefillCacheHit = 1  # 带前缀缓存的 prefill（读 paged cache）
    DecodeOnly = 2       # 纯单 token 解码（paged 路径唯一触发态）
    ChunkedPrefill = 3   # 分块 prefill 与 decode 混批（AscendMetadata 默认态）
    SpecDecoding = 4     # 投机解码（每 req 多 token，decode_threshold>1）


# SOURCE: vllm_ascend/attention/attention_v1.py:L151-L152
@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # SOURCE: vllm_ascend/attention/attention_v1.py:L151-L210
    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # SUBTRACTED: num_actual_tokens_pcp_padded（attention_v1.py:L166）—— PCP padding 计数，CP 旁路。
    # Number of tokens excluding padding.
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence (computed tokens + new tokens).
    # TODO(Angazenn): seq_lens 家族字段冗余，待 vLLM-Ascend attention schema 统一后简化。
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # SUBTRACTED: prefill(pcp) / decode_meta(dcp) 字段（attention_v1.py:L199-L202）—— CP 元数据，
    #   运行期 enable_cp() 为真才填；本章主线为 None（f7 在 backend 处收口，CP 回指 ch08）。

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # SUBTRACTED: reshape_cache_event（attention_v1.py:L207-L208）—— disaggregated PD 的 KV 传输同步事件。

    kvcomp_metadata: object | None = None


# SOURCE: vllm_ascend/attention/attention_v1.py:L213
class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.
    """

    # 是否重排 batch：把 query_len <= 该阈值的请求拉到批前段（decode 段）。
    reorder_batch_threshold: int = 1

    # SOURCE: vllm_ascend/attention/attention_v1.py:L226-L257
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        # decode_threshold = 1 + 投机 token 数（≤16，受 npu_fused_infer_attention_score TND 布局限制）。
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L259-L267
        return AttentionCGSupport.ALWAYS

    # SOURCE: vllm_ascend/attention/attention_v1.py:L269-L270
    def reorder_batch(self, input_batch, scheduler_output) -> bool:
        return False

    # SOURCE: vllm_ascend/attention/attention_v1.py:L272-L332
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        # 拆批：在已重排 batch 上找 decode|prefill 分界。
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available) over seq_lens_cpu (None in async spec decode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        # slot_mapping[i] = 第 i 个 token 要写进的全局物理槽号（回指 ch16/ch17）。
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # SUBTRACTED: CrossAttentionSpec / parallel_drafting 对 seq_lens·slot_mapping 的覆盖分支
        #   （attention_v1.py:L297-L304）—— encoder-decoder / 并行草稿特例，非标准 MHA 主线。
        attn_state = common_attn_metadata.attn_state

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            kvcomp_metadata=common_attn_metadata.kvcomp_metadata,
        )
        return attn_metadata

    # SOURCE: vllm_ascend/attention/attention_v1.py:L334-L354
    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        # 图捕获只对 DecodeOnly / ChunkedPrefill / SpecDecoding 造 dummy metadata。
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


# SOURCE: vllm_ascend/attention/attention_v1.py:L357
class AscendAttentionBackendImpl(AttentionImpl):
    # SOURCE: vllm_ascend/attention/attention_v1.py:L358-L398
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
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        # SUBTRACTED: is_kv_producer（PD KV 传输，attention_v1.py:L390-L392）/ enable_c8_quant
        #   （INT8 量化开关，L393-L395）/ layerIndex·enable_hamming_sparse（hamming 稀疏，L397-L398）。
        self.sinks = sinks

    # SUBTRACTED: update_graph_params（attention_v1.py:L400-L673）+ full_graph_fia / full_graph_fia_v2
    #   的图录制体（L680-L920）—— ACL 图捕获/replay 的参数更新与录制路径（仅 _EXTRA_CTX.capturing /
    #   draft model 才进）；eager 主前向不经过。下面只保留 full_graph_pa 的「workspace 预取节拍」范例。

    # SOURCE: vllm_ascend/attention/attention_v1.py:L922-L945
    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # workspace 预取——NPU 算子相对 CUDA flash 内核的特有节拍：
            # 先用 *_get_workspace 量出算子所需的外部显存并缓存，再把真正算子录进 ACL 图。
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)
            # SUBTRACTED: ExternalEvent / graph_task_group_begin..end 把 _npu_paged_attention 录进
            #   ACL 图（attention_v1.py:L947-L983）—— 图捕获机制本身，本章只取 workspace 预取这一节拍。
            return output

    # SOURCE: vllm_ascend/attention/attention_v1.py:L985-L1043
    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # 状态机的另一半：按五态各自整理 key/value/block_size/block_table/actual_seq_lengths_kv。
        # PrefillNoCache doesn't need key_cache, but other modes do.
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            if self.key_cache is None and kv_cache is not None:
                # SUBTRACTED: kv_cache 的 isinstance 形状判断懒初始化（attention_v1.py:L991-L997，
                #   与 forward 顶部同套路）；这里直接取 [0]/[1]。
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.key_cache is None:
                raise RuntimeError(
                    f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # 纯新 prefill：KV 就是本步算的，无 block_table。
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            # 前缀命中：从 paged cache 读历史 KV（view 成 (num_block, block_size, -1)）。
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(num_block, block_size, -1)  # type: ignore
            value = self.value_cache.view(num_block, block_size, -1)  # type: ignore
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(num_block, block_size, -1)  # type: ignore
            value = self.value_cache.view(num_block, block_size, -1)  # type: ignore
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill (与 DecodeOnly 分支体一致)。
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(num_block, block_size, -1)  # type: ignore
            value = self.value_cache.view(num_block, block_size, -1)  # type: ignore
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    # SOURCE: vllm_ascend/attention/attention_v1.py:L1045-L1164
    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # SUBTRACTED: _EXTRA_CTX.capturing → full_graph_fia / full_graph_fia_v2（attention_v1.py:L1057-L1065）
        #   —— 图捕获路径（含 workspace 预取），见 full_graph_pa 范例；eager 主前向不走。
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        # SUBTRACTED: enable_hamming_sparse 的 reshape_and_cache_kvcomp / get_kvcomp_decode_params
        #   分支（attention_v1.py:L1066,L1070-L1075）—— hamming 稀疏注意力默认关。
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # SUBTRACTED: self.sinks 分支——npu_fused_infer_attention_score_v2 + learnable_sink
        #   （attention_v1.py:L1085-L1110）—— attention-sink（如 gpt-oss）专用；标准 MHA sinks=None，
        #   走下面 else 的 npu_fused_infer_attention_score 三选一（sparse_mode 0/4/3）。
        if not attn_metadata.causal:
            # sparse_mode=0：不加 mask（非因果，如双向/encoder）。
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=0,
            )
        elif self.sliding_window is not None:
            # sparse_mode=4：滑动窗口（配 pre_tokens=window）。
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                pre_tokens=self.sliding_window,
                next_tokens=0,
                sparse_mode=4,
            )
        else:
            # sparse_mode=3：因果下三角。
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )

        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    # SOURCE: vllm_ascend/attention/attention_v1.py:L1166-L1185
    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        # decode 主路径：分页注意力，吃 block_table + context_lens（=seq_lens）。
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    # SUBTRACTED: _forward_encoder_attention（attention_v1.py:L1187-L1205）—— 非自回归 encoder/pooling
    #   模型分支（npu_fusion_attention），非标准 MHA 自回归解码主线。

    # SOURCE: vllm_ascend/attention/attention_v1.py:L1207-L1227
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
        )

    # SOURCE: vllm_ascend/attention/attention_v1.py:L1229-L1256
    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        # 写 KV：把本步新算的 K/V 按 slot_mapping 写回分页 cache（torch_npu._npu_reshape_and_cache）。
        if len(kv_cache) > 1:
            # SUBTRACTED: is_kv_producer 的 reshape_cache_event 建/录（attention_v1.py:L1239-L1240,
            #   L1254-L1255）—— disaggregated PD 的 KV 传输同步事件。
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
        return query, key, value, output

    # SOURCE: vllm_ascend/attention/attention_v1.py:L1258-L1277
    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        # 状态分流核心：唯有「DecodeOnly + using_paged_attention 多门槛 + 无滑窗」走 paged；
        # 其余一律走 fused-infer（含未命中门槛回落的 decode）。
        num_tokens = query.shape[0]
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return output

    # SOURCE: vllm_ascend/attention/attention_v1.py:L1279-L1343
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        # SUBTRACTED: enable_hamming_sparse 取 layerIndex（attention_v1.py:L1303-L1304，hamming 专用）。
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # SUBTRACTED: pooling model 分支——model_runner_type=="pooling" and not causal →
        #   _forward_encoder_attention（attention_v1.py:L1333-L1337）—— 非自回归模型分支。
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


# SUBTRACTED: AscendC8AttentionBackendImpl（attention_v1.py:L1346-L1783）—— INT8 KV 量化(C8/QuaRot)
#   子类，经 kv_c8.py 的 class surgery 才激活；量化 K/V 写 NZ 5D paged cache(npu_scatter_pa_kv_cache)、
#   decode 走 FIA V1 BNSD + perchannel antiquant、prefill gather+dequant。作减法候选/选讲——标准 MHA
#   主路径不走它，正文点名其存在与定位即可，整个子类（含 _quantize_kv_to_int8 / _forward_c8_decode /
#   _forward_c8_chunked_prefill 等私有方法）从精简版删除而不破坏 AscendAttentionBackendImpl 主控制流。
