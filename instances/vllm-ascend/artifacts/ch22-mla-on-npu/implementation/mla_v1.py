# vllm_ascend/attention/mla_v1.py —— subtract-only 精简版（MLA 在 NPU 上：权重吸收 + prefill/decode 拆分）
#
# 本章主角：AscendMLAImpl —— ch18 路由选出的 AscendMLABackend 的 impl_cls，把通用 MLA
# （vLLM 的 MLACommonImpl/MLACommonMetadataBuilder）的两条路径逐一换成最密集的 torch_npu 融合算子：
#   权重吸收 absorb  → process_weights_after_loading：kv_b_proj.weight 经 npu_format_cast(FRACTAL_ND)
#                      拆成 W_UK/W_UV、permute 成 W_UK_T；_q_proj_and_k_up_proj 运行期 torch.bmm 把
#                      q_nope「吸收」进 latent 空间（decode 期省掉对缓存 KV 的显式解压）。
#   写 KV cache      → exec_kv_decode/exec_kv_prefill：npu_kv_rmsnorm_rope_cache 一把做
#                      RMSNorm + RoPE + 写分页 KV cache。
#   decode 注意力    → _forward_decode：npu_fused_infer_attention_score_v2（K=V=缓存隐向量，MQA 吸收）→ _v_up_proj。
#   prefill 注意力   → _forward_prefill：npu_fused_infer_attention_score(TND) + _compute_prefill_context
#                      （分块算历史 context + npu_attention_update 在线 softmax 合并 LSE）。
#   三段 metadata    → AscendMLAMetadataBuilder.build：split_decodes_and_prefills 切 decode/prefill，
#                      派生 build_prefill_metadata / build_decode_metadata / build_chunked_metadata。
#   前向派发         → forward → _mla_preprocess 按 has_decode/has_prefill 双路分流。
#
# host 无 CANN/torch_npu：测试在 sys.modules 桩一个「记录调用」的 torch_npu 替身。可在 host 验证、与真仓
# 一致的纯 Python·形状级控制流：absorb 的 split/permute/bmm 形状代数 / 三段 metadata 装配 /
# prefill chunked-context 的 LSE 合并循环 / forward 按 decode-prefill 派发。真实
# npu_kv_rmsnorm_rope_cache / npu_fused_infer_attention_score_v2 / npu_format_cast 等算子不真跑（昇腾才有内核）。
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    MLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    ascend_chunked_prefill_workspace_size,
    enable_cp,
    split_decodes_and_prefills,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    maybe_trans_nz,
)

# SUBTRACTED: 大量 import（mla_v1.py:L4,L6,L10,L19,L26,L31-L36,L38-L51,L53-L55,L58-L64,L66-L67）——
#   numpy / F（仅 head_padding>0 的 F.pad）/ logger / PAD_SLOT_ID（MTP padding）/
#   context_parallel.common_cp（CP 元数据）/ enabling_mlapo·trans_rope_weight·connector 钩子 /
#   acl_graph 的 draft/graph params（图捕获）/ layer_shard_linear（layer-sharding）/
#   quantization.methods（W8A8/MXFP8/fa_quant）/ get_weight_prefetch_method·weak_ref_tensors /
#   NPUInputBatch·SchedulerOutput——均服务已减的量化/CP/图捕获/MTP/分布式旁路特性。

BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1
# SUBTRACTED: MAX_O_PROJ_PREFETCH_SIZE / MLAPO_MAX_SUPPORTED_TOKENS（mla_v1.py:L70,L74）——
#   o_proj 权重预取上限 / mlapo 融合算子 token 上限，随预取与 mlapo 旁路一并删去。


# SOURCE: vllm_ascend/attention/mla_v1.py:L77-L115
class AscendMLABackend(AttentionBackend):
    # ch18 路由选出的 MLA 后端壳：把 get_impl_cls/get_builder_cls 落到本章的 Impl/Builder。
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm_ascend/attention/mla_v1.py:L80-L85
        # HACK(Ronald1995): vllm `initialize_kv_cache` 对 attention name 做断言，这里设回 FLASH_ATTN 规避。
        return "ASCEND_MLA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        # SOURCE: vllm_ascend/attention/mla_v1.py:L87-L93
        # f-收口：运行期 enable_cp() 为真 → 切到 CP 版 builder（CP 旁支回指 ch08）。
        if enable_cp():
            from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaCPMetadataBuilder

            return AscendMlaCPMetadataBuilder
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        # SOURCE: vllm_ascend/attention/mla_v1.py:L95-L103
        # MLA 的 KV cache 形状：(num_blocks, block_size, num_kv_heads, head_size)。
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_impl_cls() -> type["MLAAttentionImpl"]:
        # SOURCE: vllm_ascend/attention/mla_v1.py:L105-L111
        # 同上 f-收口：enable_cp() 为真 → CP 版 impl；主线返回本章 AscendMLAImpl。
        if enable_cp():
            from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaCPImpl

            return AscendMlaCPImpl
        return AscendMLAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # SOURCE: vllm_ascend/attention/mla_v1.py:L113-L115
        return [128]


@dataclass
class ChunkedContextMetadata:
    # SOURCE: vllm_ascend/attention/mla_v1.py:L118-L133
    """Metadata for chunked context handling in MLA attention.

    Manages sequence boundaries and workspace for chunked prefill processing.
    """

    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    workspace: torch.Tensor
    chunk_seq_lens: torch.Tensor
    chunk_seq_lens_npu: torch.Tensor
    chunk_actual_seq_lengths_kv_list: list[list[int]]


@dataclass
class AscendMLAPrefillMetadata:
    # SOURCE: vllm_ascend/attention/mla_v1.py:L136-L153
    """Prefill Specific Metadata for Ascend"""

    attn_mask: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: list[int]
    context_lens: torch.Tensor
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_lens: int
    chunked_context: ChunkedContextMetadata | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    # SUBTRACTED: pcp_metadata（mla_v1.py:L152）—— CP 旁支元数据。
    actual_seq_lengths_q: list[int] | None = None


@dataclass
class AscendMLADecodeMetadata:
    # SOURCE: vllm_ascend/attention/mla_v1.py:L156-L172
    """Decode-specific metadata for Ascend MLA attention."""

    # MLA 的 RoPE 在注意力后端内施加，故 decode 段随身带 input_positions 与 cos/sin。
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_lens: int
    seq_lens_list: list[int]
    actual_seq_lengths_q: list[int] | None = None
    attn_mask: torch.Tensor | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    # SUBTRACTED: cp_seq_len / dcp_mtp_attn_mask（mla_v1.py:L171-L172）—— CP / MTP 旁支字段。


@dataclass
class AscendMLAMetadata:
    # SOURCE: vllm_ascend/attention/mla_v1.py:L175-L225
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # context_len / query_len / seq_len 的定义：
    # |---------- context_len ----------|-- query_len(newTokens) --|
    # |-------------------- seq_len ----------------------|

    num_actual_tokens_pcp_padded: int
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    block_tables: torch.Tensor

    # New for MLA (compared to FlashAttention): 用于 prefill/decode 拆分。
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    num_input_tokens: int = 0  # Number of tokens including padding.

    query_lens: list[int] | None = None
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # 默认 chunked prefill（未传 attn_state 时）。
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: AscendMLADecodeMetadata | None = None
    prefill: AscendMLAPrefillMetadata | None = None
    # SUBTRACTED: reshape_cache_event（mla_v1.py:L216）—— disaggregated PD KV 传输同步事件。

    def __post_init__(self):
        # SOURCE: vllm_ascend/attention/mla_v1.py:L218-L225
        pass


M = TypeVar("M", bound=AscendMLAMetadata)


# SOURCE: vllm_ascend/attention/mla_v1.py:L231-L291
class AscendMLAMetadataBuilder(MLACommonMetadataBuilder[AscendMLAMetadata]):
    """继承 vLLM 的 MLACommonMetadataBuilder[AscendMLAMetadata]，三段 metadata 总装配。"""

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendMLAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        # SOURCE: vllm_ascend/attention/mla_v1.py:L237-L291
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendMLAMetadata,
            supports_dcp_with_varlen,
        )

        scheduler_config = vllm_config.scheduler_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

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
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.chunk_seq_lens: torch.Tensor = None
        self.cu_seq_lens_cpu: torch.Tensor = None
        self.num_chunks: torch.Tensor = None
        self.max_context_chunk = 0
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.context_lens_cpu: torch.Tensor = None
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.graph_pad_size = 0
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        # SOURCE: vllm_ascend/attention/mla_v1.py:L293-L295
        return ascend_chunked_prefill_workspace_size(vllm_config)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendMLAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # SOURCE: vllm_ascend/attention/mla_v1.py:L297-L305
        return AttentionCGSupport.UNIFORM_BATCH

    # SUBTRACTED: reorder_batch（mla_v1.py:L307-L352）—— 把 query_len<=decode_threshold 的请求换到批前段
    #   （decode 段在前）的 batch 重排，依赖 NPUInputBatch/SchedulerOutput 运行期对象，非形状级主线。
    # SUBTRACTED: pad_actual_seq_len_q_mtp_enable_pad / pad_actual_seq_len_q_mtp_disable_pad
    #   （mla_v1.py:L354-L419）—— Torchair/ACL-fullgraph + MTP 投机解码的等长 padding，划入减法。

    # SOURCE: vllm_ascend/attention/mla_v1.py:L421-L425
    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    # SOURCE: vllm_ascend/attention/mla_v1.py:L427-L487
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMLAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        # 三段拆分的源头：在已重排 batch 上找 decode|prefill 分界。
        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        )
        self.set_num_actual_tokens(common_attn_metadata)
        assert self.num_decodes + self.num_prefills == num_reqs
        assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens

        self.slot_mapping = common_attn_metadata.slot_mapping[: self.num_actual_tokens]

        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.query_lens = query_seq_lens_cpu[:num_reqs]
        # Prefer _seq_lens_cpu (always available) over seq_lens_cpu (None in async spec decode).
        if common_attn_metadata._seq_lens_cpu is not None:
            self.seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            self.seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:block_table_size]

        # num_prefills>0 才建 prefill 段、num_decodes>0 才建 decode 段——双路装配骨架。
        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(common_prefix_len, common_attn_metadata)

        decode_metadata = None
        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(common_prefix_len, common_attn_metadata)
        return self.metadata_cls(  # type: ignore
            num_actual_tokens_pcp_padded=self.num_actual_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens.tolist(),
            slot_mapping=self.slot_mapping,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=self.attn_mask_builder.get_splitfuse_attn_mask(),
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=self.block_table,
            seq_lens=self.seq_lens,
            seq_lens_cpu=self.seq_lens,
        )

    # SOURCE: vllm_ascend/attention/mla_v1.py:L489-L531
    def build_chunked_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        if not self.chunked_prefill_enabled:
            return None
        num_reqs = common_attn_metadata.num_reqs

        num_computed_tokens_cpu = self.seq_lens - self.query_lens
        reqs_start = self.num_decodes  # prefill_start

        self.context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
        max_context_len_cpu = self.context_lens_cpu.max().item()
        if not max_context_len_cpu > 0:
            return None
        num_prefills_with_context_cpu = (self.context_lens_cpu > 0).sum().item()
        # 每块 workspace 有界：max_context_chunk = workspace_size // 带历史的prefill数 → round_down 到 block_size。
        self.max_context_chunk = self.chunked_prefill_workspace_size // num_prefills_with_context_cpu
        self.max_context_chunk = round_down(self.max_context_chunk, self.block_size)

        assert self.max_context_chunk > 0
        self.num_chunks = cdiv(max_context_len_cpu, self.max_context_chunk)
        chunk_starts = (
            torch.arange(self.num_chunks, dtype=torch.int32).unsqueeze(1).expand(-1, self.num_prefills)
            * self.max_context_chunk
        )
        chunk_ends = torch.min(self.context_lens_cpu.unsqueeze(0), chunk_starts + self.max_context_chunk)
        self.chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
        self.cu_seq_lens_cpu = torch.zeros(self.num_chunks, self.num_prefills + 1, dtype=torch.int32, pin_memory=True)
        torch.cumsum(self.chunk_seq_lens, dim=1, out=self.cu_seq_lens_cpu[:, 1:], dtype=torch.int32)
        chunk_actual_seq_lengths_kv_list = [
            torch.cumsum(self.chunk_seq_lens[i], dim=0).tolist() for i in range(self.num_chunks)
        ]
        return ChunkedContextMetadata(
            cu_seq_lens=self.cu_seq_lens_cpu.pin_memory().to(self.device, non_blocking=True),
            starts=chunk_starts.pin_memory().to(self.device, non_blocking=True),
            seq_tot=self.chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=self.chunk_seq_lens.max(dim=1).values.tolist(),
            chunk_seq_lens=self.chunk_seq_lens,
            chunk_seq_lens_npu=self.chunk_seq_lens.npu(),
            workspace=self.chunked_prefill_workspace,
            chunk_actual_seq_lengths_kv_list=chunk_actual_seq_lengths_kv_list,
        )

    # SOURCE: vllm_ascend/attention/mla_v1.py:L533-L543
    def get_block_table_size(self, common_attn_metadata: AscendCommonAttentionMetadata, build_metadata_step: int):
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            # SUBTRACTED: graph_pad_size>num_reqs 的 fullgraph block_table 扩张分支（mla_v1.py:L537-L541）——
            #   ACL-fullgraph + MTP 等长 padding，非主线。
            return common_attn_metadata.num_reqs
        return self.num_decodes

    # SOURCE: vllm_ascend/attention/mla_v1.py:L545-L580
    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLAPrefillMetadata:
        query_start_loc = common_attn_metadata.query_start_loc

        input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()

        chunked_context_metadata = self.build_chunked_metadata(common_prefix_len, common_attn_metadata)
        # 已重排 batch：把 prefill 段从 decode 段之后切出来（reqs_start=num_decodes / tokens_start=num_decode_tokens）。
        reqs_start = self.num_decodes  # prefill_start
        tokens_start = self.num_decode_tokens
        max_query_len = self.query_lens[reqs_start:].max().item()
        max_seq_lens = self.seq_lens[reqs_start:].max().item()
        prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]

        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_mla(prefill_input_positions)
        prefill_query_lens = self.query_lens[reqs_start:].to(torch.int32)
        # actual_seq_lengths_q = prefill_query_lens 的累积和（TND 变长右边界，回指 ch19）。
        actual_seq_lengths_q = torch.cumsum(prefill_query_lens, dim=0).tolist()
        return AscendMLAPrefillMetadata(
            attn_mask=self.attn_mask_builder.get_splitfuse_attn_mask(),
            query_lens=prefill_query_lens,
            seq_lens=self.seq_lens,
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=self.block_table[reqs_start:, ...],
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            query_start_loc=prefill_query_start_loc,
            chunked_context=chunked_context_metadata,
            sin=sin,
            cos=cos,
            actual_seq_lengths_q=actual_seq_lengths_q,
        )

    # SOURCE: vllm_ascend/attention/mla_v1.py:L582-L662
    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLADecodeMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()

        # Notice that num_decodes != num_decode_tokens in SpecDecoding Scenario
        actual_seq_lengths_q = query_start_loc_cpu[1 : self.num_decodes + 1].tolist()
        max_seq_lens = self.seq_lens[: self.num_decodes].max().item()
        self.seq_lens = self.seq_lens[: self.num_decodes]
        input_positions = input_positions[: self.num_decode_tokens]

        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_DECODE)
        self.block_table = self.block_table[:block_table_size]

        seq_lens_list = self.seq_lens.tolist()

        # SUBTRACTED: graph_pad_size>self.num_decodes / >num_reqs 的整段 MTP-spec-decode batch padding
        #   （mla_v1.py:L601-L647：pad_actual_seq_len_q_* / slot/block_table/position 补零）——
        #   ACL-fullgraph 等长 padding 与投机解码相关，非投机非 fullgraph 主路不触发。

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=True)
        decode_metadata = AscendMLADecodeMetadata(
            input_positions=input_positions,
            block_table=self.block_table,
            seq_lens=self.seq_lens,
            seq_lens_list=seq_lens_list,
            max_seq_lens=max_seq_lens,
            attn_mask=self.attn_mask_builder.get_splitfuse_attn_mask(),
            actual_seq_lengths_q=actual_seq_lengths_q,
            sin=sin[: self.num_decode_tokens, ...],
            cos=cos[: self.num_decode_tokens, ...],
            # SUBTRACTED: cp_seq_len（mla_v1.py:L660）—— 恒为 None 的 CP 旁支字段。
        )
        return decode_metadata

    # SOURCE: vllm_ascend/attention/mla_v1.py:L664-L680
    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        # 图捕获只对 DecodeOnly / SpecDecoding 造 dummy metadata。
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


# SOURCE: vllm_ascend/attention/mla_v1.py:L683-L689
class DecodeMLAPreprocessResult(NamedTuple):
    ql_nope: torch.Tensor | None = None  # 已吸收进 latent 的 q
    q_pe: torch.Tensor | None = None
    k_nope: torch.Tensor | None = None  # 缓存的隐向量 kv_c（latent）
    k_pe: torch.Tensor | None = None
    decode_q_wo_k_up: torch.Tensor | None = None
    dequant_scale_q_nope: torch.Tensor | None = None


# SOURCE: vllm_ascend/attention/mla_v1.py:L692-L697
class PrefillMLAPreprocessResult(NamedTuple):
    q_nope: torch.Tensor | None = None
    q_pe: torch.Tensor | None = None
    k_nope: torch.Tensor | None = None  # 多一个：显式 kv_b_proj 解压出的 k_nope
    k_pe: torch.Tensor | None = None
    value: torch.Tensor | None = None  # 多一个：解压出的 value


# SOURCE: vllm_ascend/attention/mla_v1.py:L700-L704
class AscendMLAImpl(MLAAttentionImpl):
    """MLA 计算核心：权重吸收 + prefill(MHA)/decode(MQA 吸收) 双路。

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # SOURCE: vllm_ascend/attention/mla_v1.py:L706-L783
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
        **kwargs,
    ):
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args —— 低秩压缩相关维度与投影层（由 MLA layer 经 kwargs 注入）。
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        ascend_config = get_ascend_config()
        self.enable_kv_nz = ascend_config.enable_kv_nz

        # SUBTRACTED: enable_shared_expert_dp / ring_mla_mask_size / speculative_config / enable_mlapo /
        #   is_kv_producer·is_kv_both / layer_name·fa_quant_layer·dtype / layer_sharding_kwargs +
        #   register_all_layers_to_shard_weight_series（mla_v1.py:L746,L749-L777）—— 共享专家 DP /
        #   ring-mla / mlapo / fa-quant / KV-transfer / layer-sharding 旁路特性的初始化，均随对应分支删去。

        # For models whose num_heads is not a power of 2, ascend ops require padding heads to next power of 2.
        self.num_heads_padded = 1 << (self.num_heads - 1).bit_length()
        self.head_padding = self.num_heads_padded - self.num_heads

    # SUBTRACTED: update_graph_params（mla_v1.py:L785-L898）—— ACL 图捕获/replay 时逐层更新
    #   npu_fused_infer_attention_score_v2.out 入参的录制路径，仅图捕获/draft-model 进，eager 主前向不经过。

    # SOURCE: vllm_ascend/attention/mla_v1.py:L900-L907
    def _v_up_proj(self, x):
        # decode 注意力输出从 latent 投回 V，闭合吸收对称性。
        # Convert from (N, B, L)/(N, B, 1, L) to (N, B, L)
        x = x.view(self.num_heads, -1, self.kv_lora_rank)
        # Multiply (N, B, L) x (N, L, V) -> (B, N, V)
        x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_y=(1, 0, 2))
        # Convert from (B, N, V) to (B, N * V)
        x = x.reshape(-1, self.num_heads * self.v_head_dim)
        return x

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        # SOURCE: vllm_ascend/attention/mla_v1.py:L909-L922
        # 运行期吸收：把 q_nope 经 W_UK_T 投到 latent 空间（核心招式）。
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    # SOURCE: vllm_ascend/attention/mla_v1.py:L924-L992
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # 权重吸收的准备：把 kv_b_proj.weight 拆成 W_UK/W_UV，并 permute 成运行期 bmm 的右乘 W_UK_T。
        # NOTE: We currently do not support quant kv_b_proj.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        # 先 cast 到 FRACTAL_ND(=2)，让后续 .T/view/split 这些 PyTorch 张量操作可用（区别于喂 cube 的 NZ=29）。
        kv_b_proj_weight = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # SUBTRACTED: else 分支（已有 W_UV 时 copy_ 回原地址，mla_v1.py:L955-L957）—— graph+RL 重捕获场景，
        #   精简版只走首次 if 分支建张量。
        if not hasattr(self, "W_UV"):
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

        # SUBTRACTED: enable_mlapo / fa_quant_layer 两条量化融合旁支（_process_weights_for_fused_mlapo /
        #   _process_weights_for_fused_mlapo_a5 / _process_weights_for_fused_fa_quant，mla_v1.py:L962-L985,
        #   L994-L1115）整体划入减法 + layer_sharding 的 post_process（L990-L992）；精简版只走 else 主线：
        #   W_UK_T 经 maybe_trans_nz 转 FRACTAL_NZ(29) 喂昇腾 cube。
        self.W_UK_T = maybe_trans_nz(self.W_UK_T)

    # SUBTRACTED: _process_weights_for_fused_fa_quant / _process_weights_for_fused_mlapo[_a5]
    #   （mla_v1.py:L994-L1115）—— W8A8/MXFP8/FA-quant/mlapo 的融合权重处理，与 absorb 主控制流正交。

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1117-L1124
    def get_context_seq_len_npu(self, index: int, attn_metadata: AscendMLAMetadata):
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.chunk_seq_lens_npu is not None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        assert 0 <= index < iters
        return prefill_metadata.chunked_context.chunk_seq_lens_npu[index]

    # SUBTRACTED: _reorg_kvcache（mla_v1.py:L1126-L1134）—— CP 下从 cache 读出的 kv 重排，非 CP 时恒等
    #   返回 (kv_c_normed, k_pe)，故精简版在 _compute_prefill_context 中直接用读出值。

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1136-L1241
    def _compute_prefill_context(
        self,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        rope_dim,
        attn_metadata,
        prefix_output,
        prefix_lse,
    ):
        # chunked-context：分块从分页 cache 读历史 kv_c → kv_b_proj 解压 → 逐块算注意力 →
        # 与新 token 段 prefix 的 (out, lse) 一起喂 npu_attention_update 做在线 softmax 合并。
        assert len(kv_c_and_k_pe_cache) > 1
        prefill_metadata = attn_metadata.prefill
        if prefill_metadata is None or prefill_metadata.chunked_context is None:
            return prefix_output, prefix_lse

        iters = len(prefill_metadata.chunked_context.seq_tot)
        cache_kv_c = kv_c_and_k_pe_cache[0]
        cache_k_pe = kv_c_and_k_pe_cache[1]
        num_heads = cache_k_pe.size(2)
        latent_kv_dim = kv_c_and_k_pe_cache[0].size(-1)

        actual_seq_lengths_q = prefill_metadata.actual_seq_lengths_q

        if iters == 0:
            return prefix_output, prefix_lse

        num_tokens = q_nope.size(0)
        D = self.v_head_dim
        H = self.num_heads

        if prefix_lse.dim() == 2:
            prefix_lse = prefix_lse.transpose(0, 1).unsqueeze(-1)
        prefix_output = prefix_output.to(torch.float32)
        prefix_lse = prefix_lse.to(torch.float32)
        # out_list/lse_list 首项即新 token 段的 prefix（FIA 在 _forward_prefill 已算出）。
        out_list = [prefix_output.reshape(num_tokens * H, D)]
        lse_list = [prefix_lse.reshape(num_tokens * H)]

        common_kwargs = {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_heads,
            "input_layout": "TND",
            "atten_mask": None,
            "sparse_mode": 0,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "softmax_lse_flag": True,
            "actual_seq_lengths": actual_seq_lengths_q,
        }

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            context_seq_len_npu = self.get_context_seq_len_npu(i, attn_metadata)
            kv_c_normed = torch.empty(toks, num_heads, latent_kv_dim, dtype=cache_kv_c.dtype, device=cache_kv_c.device)
            k_pe = torch.empty(toks, num_heads, rope_dim, dtype=q_pe.dtype, device=q_pe.device)

            # 从分页 cache 读第 i 块历史 kv_c / k_pe。
            DeviceOperator.kv_cache_load(
                cache_kv_c,
                cache_k_pe,
                prefill_metadata.block_table,
                context_seq_len_npu,
                prefill_metadata.chunked_context.starts[i],
                key=kv_c_normed,
                value=k_pe,
            )
            # SUBTRACTED: _reorg_kvcache（CP 重排，恒等）/ fa_quant+A5 的 kv_c_normed 反量化
            #   （mla_v1.py:L1204-L1215）—— 非 CP、非量化主线直接用读出的 kv_c_normed。
            kv_c_normed = kv_c_normed.squeeze()
            # 显式 kv_b_proj 解压历史 kv_c → k_nope / v（与 prefill 同走 MHA 风格）。
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))

            actual_seq_lengths_kv = prefill_metadata.chunked_context.chunk_actual_seq_lengths_kv_list[i]
            common_kwargs["actual_seq_lengths_kv"] = actual_seq_lengths_kv

            # SUBTRACTED: head_padding>0 时 cat(k_nope,k_pe)（mla_v1.py:L1223-L1224）—— head_padding=0 主线
            #   用 query_rope/key_rope 解耦 RoPE。
            common_kwargs["query_rope"] = q_pe
            common_kwargs["key_rope"] = k_pe
            query = q_nope
            key = k_nope

            chunk_out, chunk_lse = torch_npu.npu_fused_infer_attention_score(query, key, v, **common_kwargs)

            if chunk_lse.dim() == 2:
                chunk_lse = chunk_lse.transpose(0, 1).unsqueeze(-1)
            chunk_out = chunk_out.to(torch.float32)
            chunk_lse = chunk_lse.to(torch.float32)
            out_list.append(chunk_out.reshape(num_tokens * H, D))
            lse_list.append(chunk_lse.reshape(num_tokens * H))

        # 在线 softmax 合并（等价基座的 merge_attn_states，回指 ch19 的 LSE/fused-infer 思路）。
        output_final, _ = torch_npu.npu_attention_update(tuple(lse_list), tuple(out_list), 0)
        return output_final.view(num_tokens, H, D), None

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1243-L1310
    def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        # prefill 主路：先对当前新 token 算注意力（npu_fused_infer_attention_score, TND），
        # 再 _compute_prefill_context 合并历史 chunked context。
        assert attn_metadata.prefill is not None
        assert len(kv_c_and_k_pe_cache) > 1
        num_tokens = q_nope.size(0)
        prefill_meta = attn_metadata.prefill

        actual_seq_lengths_q = prefill_meta.actual_seq_lengths_q
        actual_seq_lengths_kv = actual_seq_lengths_q.copy()

        # SUBTRACTED: dtype!=bf16 时 to(bfloat16)/末尾转回（mla_v1.py:L1261-L1269,L1306-L1308）——
        #   FIA TND 仅支持 bf16 的兼容转换；精简版输入即 bf16。

        attn_output = torch.empty(num_tokens, self.num_heads, self.v_head_dim, dtype=q_nope.dtype, device=q_nope.device)
        attn_lse = torch.empty(self.num_heads, num_tokens, dtype=torch.float32, device=q_nope.device)

        common_kwargs = {
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_heads,
            "input_layout": "TND",
            "atten_mask": prefill_meta.attn_mask,
            "sparse_mode": 3,  # 因果下三角
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "block_table": None,
            "block_size": 0,
            "softmax_lse_flag": True,  # 额外吐 LSE 供 chunked-context 合并
            "actual_seq_lengths": actual_seq_lengths_q,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
        }

        # SUBTRACTED: head_padding>0 时 cat(q_nope,q_pe)/cat(k_nope,k_pe)（mla_v1.py:L1290-L1292）——
        #   head_padding=0 主线用 query_rope/key_rope 解耦 RoPE。
        common_kwargs["query_rope"] = q_pe
        common_kwargs["key_rope"] = k_pe
        query, key = q_nope, k_nope

        attn_output, attn_lse = torch_npu.npu_fused_infer_attention_score(query, key, value, **common_kwargs)

        attn_output, attn_lse = self._compute_prefill_context(
            q_nope, q_pe, kv_c_and_k_pe_cache, self.qk_rope_head_dim, attn_metadata, attn_output, attn_lse
        )

        attn_output = attn_output.reshape([num_tokens, self.num_heads * self.v_head_dim])
        return attn_output

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1312-L1342
    def exec_kv_decode(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
    ):
        # 融合算子 npu_kv_rmsnorm_rope_cache 一把做：kv_a_layernorm 的 RMSNorm + k_pe 的 RoPE + 写分页 KV cache。
        # decode 取算子前两个返回值（写进 cache 的隐向量 k_nope=kv_c, k_pe）。
        assert self.kv_a_layernorm is not None
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA_NZ" if self.enable_kv_nz else "PA"
        c_kv_scale = None
        # SUBTRACTED: A5+fa_quant 的 c_kv_scale=fak_descale_reciprocal（mla_v1.py:L1328-L1329）—— 量化旁支。
        k_pe, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,  # type: ignore[union-attr]
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            c_kv_scale=c_kv_scale,
            epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
            cache_mode=cache_mode,
        )
        return k_pe, k_nope

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1344-L1375
    def exec_kv_prefill(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
    ):
        # 与 decode 同算子，差别：is_output_kv=True 且取第 3/4 个返回值（未量化输出 KV，供 prefill 显式解压）。
        assert self.kv_a_layernorm is not None
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"
        c_kv_scale = None
        # SUBTRACTED: A5+fa_quant 的 c_kv_scale=fak_descale_reciprocal（mla_v1.py:L1360-L1361）—— 量化旁支。
        _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,  # type: ignore[union-attr]
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            c_kv_scale=c_kv_scale,
            epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
            cache_mode=cache_mode,
            is_output_kv=True,
        )
        return k_pe, k_nope

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1377-L1387
    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1389-L1593
    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
        dequant_scale_q_nope=None,
    ) -> torch.Tensor:
        # decode 主路（MQA 吸收）：q 已吸收进 latent，对缓存隐向量 k_nope 做 MQA（K=V=k_nope）。
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = q_nope.size(0)
        actual_seq_lengths = None

        # SUBTRACTED: fa_quant / enable_kv_nz 的 k_nope/k_pe NZ 排布 view（mla_v1.py:L1408-L1423）——
        #   标准 PA 主线把 k_nope/k_pe view 成分页布局 [num_blocks, num_kv_heads, block_size, L/R]。
        k_nope = k_nope.view(-1, self.num_kv_heads, block_size, self.kv_lora_rank)
        k_pe = k_pe.view(-1, self.num_kv_heads, block_size, self.qk_rope_head_dim)

        # SUBTRACTED: SpecDecoding TND_NTD 与 fa_quant/A5/enable_kv_nz 的多 input_layout 分支
        #   （mla_v1.py:L1428-L1497）—— 精简版固定 head_padding=0 + 标准 bf16 PA，走 else: BNSD_NBSD 主线。
        # The output layout is set to NBSD to eliminate the need for a transpose operation after attention.
        # Input shape: [num_tokens, num_heads, seq_len, dim]
        input_layout = "BNSD_NBSD"
        q_nope = q_nope.view(num_tokens, self.num_heads, 1, -1).contiguous()
        q_pe = q_pe.view(num_tokens, self.num_heads, 1, -1)
        sparse_mode = 0
        attn_mask = None

        common_kwargs = {
            "query_rope": q_pe,
            "key_rope": k_pe,
            "num_query_heads": self.num_heads_padded,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": input_layout,
            "atten_mask": attn_mask,
            "sparse_mode": sparse_mode,
            "softmax_scale": self.scale,
            "block_table": decode_meta.block_table,
            "block_size": block_size,
            "actual_seq_qlen": actual_seq_lengths,
            "actual_seq_kvlen": decode_meta.seq_lens_list,
        }
        # SUBTRACTED: fa_quant 的 extra_args 量化入参 + _EXTRA_CTX.capturing 的图捕获录制
        #   （get_max_workspace 预取 + graph_task_group + weak_ref_tensors，mla_v1.py:L1517-L1587）——
        #   eager 主线直接调用 FIA v2；注意第 2/3 入参同为 k_nope（缓存隐向量），即对 latent 做 MQA。
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(q_nope, k_nope, k_nope, **common_kwargs)

        # SUBTRACTED: head_padding>0 时 attn_output[:num_heads] 截断（mla_v1.py:L1591-L1592）—— head_padding=0。
        return self._v_up_proj(attn_output)

    # SUBTRACTED: reorg_decode_q（mla_v1.py:L1595-L1596）—— 恒等返回 (decode_q_nope, decode_q_pe)，未被主线调用。

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1598-L1618
    def mla_preprocess_prefill(self, q_c, kv_no_split, kv_cache, attn_metadata):
        # prefill 走 MHA 风格：不吸收——q_proj 出满维 q_nope/q_pe，exec_kv_prefill 额外返回 k_c_normed，
        # 显式经 kv_b_proj 解压成 k_nope/value。
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        prefill_kv_no_split = kv_no_split[num_decode_tokens:num_actual_tokens]
        prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
        prefill_q = self.q_proj(prefill_q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
        prefill_q_pe = prefill_q[..., self.qk_nope_head_dim :]
        prefill_q_nope = prefill_q[..., : self.qk_nope_head_dim]
        cos = attn_metadata.prefill.cos
        sin = attn_metadata.prefill.sin
        prefill_slots = attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens]
        prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
        prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(prefill_kv_no_split, cos, sin, kv_cache, prefill_slots)
        prefill_k_nope, prefill_value = (
            self.kv_b_proj(prefill_k_c_normed)[0]
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        )
        prefill_k_pe = prefill_k_pe.view(prefill_q_c.shape[0], self.num_kv_heads, -1)
        prefill_k_pe = prefill_k_pe.expand((*prefill_k_nope.shape[:-1], -1))
        return PrefillMLAPreprocessResult(prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe, prefill_value)

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1620-L1638
    def mla_preprocess_decode(self, q_c, kv_no_split, kv_cache, attn_metadata):
        # decode 走吸收路径：_q_proj_and_k_up_proj 出已吸收的 ql_nope → rope_single 给 q_pe 加 RoPE →
        # exec_kv_decode 写 KV cache 并返回缓存的隐向量 k_nope。
        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_q_c = q_c[:num_decode_tokens]
        cos = attn_metadata.decode.cos
        sin = attn_metadata.decode.sin
        decode_ql_nope, decode_q_pe = self._q_proj_and_k_up_proj(decode_q_c)
        decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
        dequant_scale_q_nope = None
        # SUBTRACTED: fa_quant + A5 动态量化（npu_dynamic_quant，mla_v1.py:L1628-L1632）—— 量化旁支。
        decode_slots = attn_metadata.slot_mapping[:num_decode_tokens:1]
        decode_kv_no_split = kv_no_split[:num_decode_tokens]
        decode_k_pe, decode_k_nope = self.exec_kv_decode(decode_kv_no_split, cos, sin, kv_cache, decode_slots)
        return DecodeMLAPreprocessResult(
            decode_ql_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope=dequant_scale_q_nope
        )

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1640-L1691
    def _mla_preprocess(self, layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv):
        # MLA Preprocess:
        # 1. fused_qkv_a_proj + q_a_layernorm 一把出 q_c 和 kv_no_split。
        # 2. 按 has_decode/has_prefill 双路分流写 KV cache 并各自取出 q/k。
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        if self.fused_qkv_a_proj is not None:
            # SUBTRACTED: weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream
            #   （mla_v1.py:L1653-L1656）—— 权重预取旁路优化。
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)  # type: ignore[misc]
            # allgather need contiguous data
            kv_no_split = kv_no_split.contiguous()
        else:
            q_c = hidden_states
            kv_no_split = self.kv_a_proj_with_mqa(hidden_states)[0]  # type: ignore[misc]

        # SUBTRACTED: maybe_all_gather_and_maybe_unpad(Flash-Comm V1) / layer_sharding reach_layer /
        #   wait_for/maybe_save_kv_layer_to_connector / reshape_cache_event（mla_v1.py:L1669-L1683,L1689-L1690）
        #   —— 张量并行/连接器/KV-transfer 旁路；单实例非分离推理下 need_gather_q_kv=False、connector=None。

        decode_preprocess_res = None
        prefill_preprocess_res = None
        # Preprocess for decode tokens
        if has_decode:
            decode_preprocess_res = self.mla_preprocess_decode(q_c, kv_no_split, kv_cache, attn_metadata)
        # Preprocess for prefill tokens
        if has_prefill:
            prefill_preprocess_res = self.mla_preprocess_prefill(q_c, kv_no_split, kv_cache, attn_metadata)
        return decode_preprocess_res, prefill_preprocess_res

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1693-L1694
    def get_num_actual_tokens(self, attn_metadata: M):
        return attn_metadata.num_actual_tokens

    # SUBTRACTED: forward_mha / forward_mqa（mla_v1.py:L1696-L1716）—— 本版二者仅 raise NotImplementedError
    #   （注释明说 'Use forward() instead'）；真正的 decode/prefill 分流发生在 forward() 内部。
    #   'MQA 吸收 / MHA' 是 vLLM 基类 docstring 给的两种数学等价写法概念，不是本版的实际方法名。

    # SOURCE: vllm_ascend/attention/mla_v1.py:L1718-L1804
    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 总入口与真实分流处：_mla_preprocess 按 decode/prefill 出两份预处理结果，分别走
        # _forward_decode（MQA 吸收）/_forward_prefill（MHA），写进 o_proj_input 不同切片，再 o_proj。
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            # SUBTRACTED: layer_sharding reach_layer 循环（mla_v1.py:L1730-L1732）—— layer-sharding 旁路。
            return output.fill_(0)

        num_actual_tokens = self.get_num_actual_tokens(attn_metadata)
        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        num_decode_tokens = attn_metadata.num_decode_tokens
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        o_proj_input_shape = (_EXTRA_CTX.num_tokens, self.num_heads * self.v_head_dim)
        o_proj_input = torch.zeros(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)

        # SUBTRACTED: (fa_quant or enable_mlapo) 且纯 decode 时走 DeviceOperator.mla_preprocess_only_decode
        #   的快路（mla_v1.py:L1749-L1757）—— 精简版只留 else: _mla_preprocess 主线。
        # MLA Preprocess
        decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
            layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv
        )
        if decode_preprocess_res is not None:
            # MLA Preprocess for decoding
            output_decode = self._forward_decode(
                decode_preprocess_res.ql_nope,
                decode_preprocess_res.q_pe,
                decode_preprocess_res.k_nope,
                decode_preprocess_res.k_pe,
                kv_cache[0].shape[1],
                attn_metadata,
                decode_preprocess_res.dequant_scale_q_nope,
            )

            o_proj_input[:num_decode_tokens] = output_decode

        if prefill_preprocess_res is not None:
            output_prefill = self._forward_prefill(
                prefill_preprocess_res.q_nope,
                prefill_preprocess_res.q_pe,
                prefill_preprocess_res.k_nope,
                prefill_preprocess_res.k_pe,
                prefill_preprocess_res.value,
                kv_cache,
                attn_metadata,
            )

            o_proj_input[num_decode_tokens:num_actual_tokens] = output_prefill
        # SUBTRACTED: o_proj 前 weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream
        #   预取 + 末尾 is_kv_producer 的 maybe_save_kv_layer_to_connector（mla_v1.py:L1792-L1798,L1802-L1803）
        #   —— 旁路优化/连接器。
        # O proj
        output[...] = self.o_proj(o_proj_input, is_prefill=prefill_preprocess_res is not None)[0]

        del o_proj_input
        return output_padded
