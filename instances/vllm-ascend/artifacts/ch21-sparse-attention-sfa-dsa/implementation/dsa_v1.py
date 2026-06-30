# vllm_ascend/attention/dsa_v1.py —— subtract-only 精简版（DSA：DeepSeek Sparse Attention + Lightning Indexer）
#
# 本章主角之二：AscendDSAImpl —— ch18 路由 (use_mla,use_sparse,use_compress)=(T,F,T) 选出的
# AscendDSABackend 的 impl。与 SFA「直接继承 vLLM MLA」不同，DSA 自起一套抽象：
#   - AscendDSAImpl 继承昇腾自有的 DSAAttentionImpl（abstract.py），不走 vLLM MLAAttentionImpl；
#   - AscendDSAMetadataBuilder 继承 vLLM 的 AttentionMetadataBuilder（非 MLA builder），DSA 自带一套；
#   - 但内部仍内联 MLA 式低秩 prolog（wq_a/wq_b/wkv/q_norm/kv_norm）—— 仍「建在 MLA 之上」。
#
# Lightning Indexer（章节核心）：每个 query 只挑出 index_topk=512 个最相关 KV，再只对这 512 个算注意力。
#   - 元数据期：build_prefill_metadata / build_decode_metadata 各自调
#       npu_quant_lightning_indexer_metadata(sparse_count=index_topk=512, sparse_mode=3) 预建 qli_metadata；
#   - 前向期：indexer_select_qli → _indexer_qli → npu_quant_lightning_indexer(sparse_count=index_topk)
#       出 compress_topk_idxs(512)；再 npu_sparse_attn_sharedkv(cmp_sparse_indices=compress_topk_idxs)
#       只对 top-512 算注意力 → wo_a/wo_b 输出投影。
#
# vLLM 主干无对位后端：DSA 是插件「增量扩展」出的算法。
#
# host 无 CANN/torch_npu：真实 npu_quant_lightning_indexer_metadata / npu_quant_lightning_indexer /
# npu_sparse_attn_sharedkv / compressor 等私有算子由测试「记录调用」替身承接，只验元数据装配与
# top-512 选择/稀疏注意力的入参（sparse_count=512, sparse_mode=3, cmp_sparse_indices=…），不真算。
#
# SUBTRACTED: 本文件原 2897 行。删除 subtraction_plan 批准的：CP（dsa_cp* / DSACPMetadataBuilder）、
#   多流重叠（multistream_dsv4_dsa_overlap / cv_indexer_select_qli / _mla_prolog_multistream /
#   prefill_comm_compute_overlap）、drafting/投机解码（build_for_drafting / build_*_for_drafting）、
#   compress_ratio 的 <=1 与 128 旁支（保 ==4 主稀疏路）、W8A8 动态量化分支（_is_w8a8_dynamic 各处）、
#   A5 代际 o_proj（npu_dynamic_mx_quant / npu_transpose_quant_batchmatmul）、IndexCache 复用、
#   build 的 ratio→metadata 缓存字典 ping-pong（直接重算）、元数据缓存的压缩位置闭包。
import math
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import torch
import torch.nn.functional as F  # noqa: F401  (保留：原文件 prefill/decode 压缩位置 pad 用)
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# SUBTRACTED: import CVLinearWrapper / AscendUnquantizedLinearMethod / w8a8_dynamic / triton rms_norm /
#   attention_calculation_stream / npu_stream_switch / olora_tp_enable / NPUInputBatch —— 服务已减的
#   CV 拆分/量化/多流/CP。原 dsa_v1.py:L22-L33,L40-L43。

BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1


# SOURCE: vllm_ascend/attention/dsa_v1.py:L180
class AscendDSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L183
    def get_name() -> str:
        # HACK(Ronald1995): 同 SFA，改名绕过 model-runner 名断言（回指 ch18）。
        return "ASCEND_DSA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L190
    def get_builder_cls():
        # SUBTRACTED: enable_dsa_cp() → AscendDSACPMetadataBuilder 分支（CP）。原 dsa_v1.py:L192-L197。
        return AscendDSAMetadataBuilder

    @staticmethod
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L200
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L208
    def get_impl_cls() -> type["DSAAttentionImpl"]:
        # SUBTRACTED: enable_dsa_cp() → AscendDSACPImpl 分支（CP）。原 dsa_v1.py:L210-L215。
        return AscendDSAImpl

    @staticmethod
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L218
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8, 16, 32, 64, 128]


@dataclass
# SOURCE: vllm_ascend/attention/dsa_v1.py:L223
class AscendDSAPrefillMetadata:
    """Prefill Specific Metadata：携带 sas_metadata（稀疏注意力元数据）+ qli_metadata（Lightning Indexer 元数据）。"""

    attn_mask: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    max_seq_lens: int

    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    start_pos: torch.Tensor | None = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_c4_cmp_seqlen_list: torch.Tensor = None
    cu_c128_cmp_seqlen_list: torch.Tensor = None


@dataclass
# SOURCE: vllm_ascend/attention/dsa_v1.py:L249
class AscendDSADecodeMetadata:
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_kv: int
    max_seqlen_q: int
    seq_lens_list: list[int]
    max_seq_lens: int
    slot_mapping: torch.Tensor

    query_start_loc: torch.tensor = None
    query_start_loc_cpu: torch.tensor = None
    attn_mask: torch.Tensor | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    cp_seq_len: torch.Tensor = None
    batch_seq_mask: torch.Tensor = None
    start_pos: torch.Tensor = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None


@dataclass
# SOURCE: vllm_ascend/attention/dsa_v1.py:L276
class AscendDSAMetadata:
    """沿用 ch20 MLA 的 decode/prefill 拆分范式，但 DSA 自带（非 MLA builder 产出）。"""

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    num_input_tokens: int = 0
    query_lens: list[int] | None = None
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: AscendDSADecodeMetadata | None = None
    prefill: AscendDSAPrefillMetadata | None = None
    hadamard: torch.Tensor | None = None  # dsv4 indexer 的哈达玛旋转矩阵
    start_pos: torch.Tensor | None = None


DSAMetadataList: TypeAlias = list[AscendDSAMetadata]


# SOURCE: vllm_ascend/attention/dsa_v1.py:L333
def _require_prefill_metadata(metadata: AscendDSAMetadata) -> AscendDSAPrefillMetadata:
    assert metadata.prefill is not None
    return metadata.prefill


# SOURCE: vllm_ascend/attention/dsa_v1.py:L338
def _require_decode_metadata(metadata: AscendDSAMetadata) -> AscendDSADecodeMetadata:
    assert metadata.decode is not None
    return metadata.decode


# SOURCE: vllm_ascend/attention/dsa_v1.py:L343
class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    """DSA 自带 builder：继承 vLLM 的 AttentionMetadataBuilder（非 MLA builder）。
    build → split_decodes_and_prefills 拆 decode/prefill → 各自 build_prefill_metadata / build_decode_metadata，
    每条路都预建 qli_metadata（Lightning Indexer 元数据）+ sas_metadata（稀疏注意力元数据）。"""

    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    hadamard = None
    block_size: int | None = 128

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L359
    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = metadata_cls if metadata_cls is not None else AscendDSAMetadata
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size
        self.decode_threshold = 1
        self.reorder_batch_threshold = self.decode_threshold
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        self.start_pos_prefill = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device
        )
        self.start_pos_decode = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device
        )
        self.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
        # SUBTRACTED: speculative_config / spec_slot_mapping / attn_mask_builder /
        #   scipy.linalg.hadamard 构建（deepseek_v4 索引旋转矩阵，host 无 scipy）/ cos_cache 等。
        #   原 dsa_v1.py:L373-L441。

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L500
    def set_num_actual_tokens(self, common_attn_metadata: AscendCommonAttentionMetadata):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L506
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc

        # SUBTRACTED: common_ratio_to_sas_metadata 缓存命中分支（避免每步重算）；精简版每次直接重算。
        #   原 dsa_v1.py:L516-L564 的 dict ping-pong。
        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        )
        self.set_num_actual_tokens(common_attn_metadata)
        assert self.num_decodes + self.num_prefills == num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        if self.num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            cos, sin = get_cos_and_sin_dsa(input_positions, True)
        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.query_lens = query_seq_lens_cpu[:num_reqs]

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.slot_mapping[:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)
        self.block_table = common_attn_metadata.block_table_tensor[: self.get_block_table_size(common_attn_metadata)]

        # decode/prefill 各自装配（含各自的 qli_metadata + sas_metadata）
        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(common_prefix_len, common_attn_metadata)

        decode_metadata = None
        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(common_prefix_len, common_attn_metadata, None)

        return self.metadata_cls(
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=None,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=AscendDSAMetadataBuilder.hadamard,
        )

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L604
    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendDSAPrefillMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        reqs_start = self.num_decodes  # prefill 请求起始位置
        tokens_start = self.num_decode_tokens

        input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()
        max_query_len = self.query_lens[reqs_start:].max().item()
        _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seq_lens = _seq_lens_cpu[reqs_start:].max().item()
        prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)
        prefill_seq_lens = self.seq_lens[reqs_start:]
        num_prefill = prefill_seq_lens.shape[0]
        compress_cos, compress_sin = cos, sin  # SUBTRACTED: 压缩位置 pad 的 _get_padded_compressed_position
        #   闭包与 c{ratio}_cos 缓存。原 dsa_v1.py:L658-L717。

        prefill_slot_mapping = self.slot_mapping[: self.num_prefill_tokens]
        self.start_pos_prefill.fill_(0)
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        self.start_pos_prefill[:num_prefill] = self.seq_lens[reqs_start:] - seq_lens_q

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        # (a) 稀疏注意力元数据 sas_metadata：经门面拿 npu_sparse_attn_sharedkv_metadata（保 compress_ratio==4 主路）
        # SUBTRACTED: compress_ratio<=1 / ==128 两档 sas 分支（模型变体配置）。原 dsa_v1.py:L738-L761,L791-L816。
        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=prefill_query_start_loc,
            cu_seqlens_ori_kv=prefill_query_start_loc,
            cu_seqlens_cmp_kv=None,
            seqused_q=self.seqused_q,
            seqused_kv=self.seq_lens[reqs_start:],
            max_seqlen_q=seq_lens_q.max(),
            max_seqlen_kv=self.seq_lens[reqs_start:].max(),
            batch_size=len(self.seq_lens[reqs_start:]),
            cmp_topk=index_topk,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=True,
        )

        # (b) Lightning Indexer 元数据 qli_metadata：sparse_count=index_topk=512、sparse_mode=3（章节核心）
        qli_metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=prefill_query_start_loc[1:].clone(),
            actual_seq_lengths_key=self.seq_lens[reqs_start:].clone(),
            num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
            num_heads_k=1,
            head_dim=self.model_config.hf_config.index_head_dim,  # 128
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=len(self.seq_lens[reqs_start:]),
            max_seqlen_q=seq_lens_q.max().item(),
            max_seqlen_k=self.seq_lens[reqs_start:].max().item(),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.model_config.hf_config.index_topk,  # 512
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(self.seqused_q.device),
        )

        return AscendDSAPrefillMetadata(
            attn_mask=None,
            query_lens=self.query_lens[reqs_start:].to(torch.int32),
            seq_lens=self.seq_lens[reqs_start:],
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=self.block_table[reqs_start:, ...],
            slot_mapping=prefill_slot_mapping,
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            query_start_loc=prefill_query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=self.start_pos_prefill[:num_prefill],
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
        )

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L862
    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs_actual: int | None,
    ) -> AscendDSADecodeMetadata:
        query_start_loc = common_attn_metadata.query_start_loc[: self.num_decodes + 1]
        input_positions = common_attn_metadata.positions[: self.num_decode_tokens].long()
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: self.num_decodes + 1]
        _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seq_lens = _seq_lens_cpu[: self.num_decodes].max().item()
        seq_lens_list = _seq_lens_cpu[: self.num_decodes].tolist()
        max_seqlen_kv = torch.max(_seq_lens_cpu[: self.num_decodes]).item()
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        self.start_pos_decode[: self.num_decodes] = self.seq_lens[: self.num_decodes] - seq_lens_q
        slot_mapping = self.slot_mapping[: self.num_decode_tokens]
        compress_cos, compress_sin = cos, sin  # SUBTRACTED: 压缩位置 pad 闭包 + c{ratio} 缓存。原 dsa_v1.py:L922-L1076。

        # decode 路同样预建 qli_metadata：sparse_count=index_topk=512、sparse_mode=3（与 prefill 对称）
        self.decode_qli_metadata[:1024] = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=query_start_loc[1:].clone(),
            actual_seq_lengths_key=self.seq_lens[: self.num_decodes].clone(),
            num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
            num_heads_k=1,
            head_dim=self.model_config.hf_config.index_head_dim,  # 128
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=len(self.seq_lens[: self.num_decodes]),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.model_config.hf_config.index_topk,  # 512
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(self.seqused_q.device),
        )

        return AscendDSADecodeMetadata(
            input_positions=input_positions,
            block_table=self.block_table[: self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_DECODE)],
            slot_mapping=slot_mapping,
            seq_lens=self.seq_lens[: self.num_decodes],
            seq_lens_list=seq_lens_list,
            max_seq_lens=max_seq_lens,
            max_seqlen_kv=max_seqlen_kv,
            max_seqlen_q=max_seqlen_q,
            attn_mask=None,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            sin=sin[: self.num_decode_tokens, ...],
            cos=cos[: self.num_decode_tokens, ...],
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=self.start_pos_decode[: self.num_decodes],
            sas_metadata=self.decode_sas_metadata,
            qli_metadata=self.decode_qli_metadata,
        )

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L1346
    def get_block_table_size(self, common_attn_metadata: AscendCommonAttentionMetadata, build_metadata_step: int = 0):
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            return common_attn_metadata.num_reqs
        return self.num_decodes

    # SUBTRACTED: build_for_drafting / build_prefill_metadata_for_drafting / build_decode_metadata_for_drafting /
    #   build_for_graph_capture / reorder_batch —— 投机解码与图捕获专用。原 dsa_v1.py:L443-L499,L1124-L1377。


# SOURCE: vllm_ascend/attention/dsa_v1.py:L1378
class AscendDSAImpl(DSAAttentionImpl):
    """DSA 主 impl：继承昇腾自有 DSAAttentionImpl（非 vLLM MLAAttentionImpl）。
    forward 拆 [decode|prefill] → 各走 Lightning Indexer 选 top-512 + npu_sparse_attn_sharedkv 稀疏注意力。"""

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L1384
    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        # MLA 式低秩 prolog 权重（内联，仍「建在 MLA 之上」）
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs["q_norm_without_weight"]
        self.kv_norm = kwargs["kv_norm"]
        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")
        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]
        self.eps = kwargs["eps"]
        self.attn_sink = kwargs["attn_sink"]
        self.vllm_config = get_current_vllm_config()
        # SUBTRACTED: CVLinearWrapper(cv_wq_a/cv_wkv/cv_wq_b) 拆 V/C（多流）/ multistream_* / IndexCache /
        #   compressor.* 全套参数解包。原 dsa_v1.py:L1422-L1485。

        # indexer 子模块参数：独立小投影 + compressor，index_topk 钉死 top-k 预算
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk = self.indexer.index_topk  # 512：DSA 每 query 的 top-k 预算

        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

    # ===== forward 主脊：拆 [decode|prefill] → 各自走稀疏路 → 合并过 o_proj(wo_a/wo_b) =====
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L1574
    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataList,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        output_padded = output
        has_prefill = attn_metadata[0].num_prefills > 0
        has_decode = attn_metadata[0].num_decodes > 0
        decode_tokens = attn_metadata[0].num_decode_tokens
        actual_tokens = attn_metadata[0].num_actual_tokens

        # SUBTRACTED: need_prefill_gather（prefill_comm_compute_overlap 延迟 all-gather）。原 dsa_v1.py:L1595-L1608。
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, need_gather_q_kv)
        prefill_hidden_states = hidden_states[decode_tokens:actual_tokens]
        decode_hidden_states = hidden_states[:decode_tokens]

        forward_context = get_forward_context()
        o_proj_input_shape = (forward_context.num_tokens, self.n_local_heads, self.head_dim)
        o_proj_input = torch.empty(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        assert kv_cache is not None, "kv_cache tensor tuple must be provided."
        if has_prefill:
            output_prefill = self._forward_prefill(layer_name, prefill_hidden_states, kv_cache, attn_metadata, False)
            o_proj_input[decode_tokens:actual_tokens] = output_prefill
        if has_decode:
            output_decode = self._forward_decode(layer_name, decode_hidden_states, kv_cache, attn_metadata)
            o_proj_input[:decode_tokens] = output_decode

        cos = attn_metadata[0].cos[layer_name]
        sin = attn_metadata[0].sin[layer_name]
        num_tokens = o_proj_input.shape[0]

        # 逆 RoPE（解开 prolog 时加的旋转）
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o_proj 输出投影 wo_a → wo_b
        # SUBTRACTED: A5 的 npu_dynamic_mx_quant + npu_transpose_quant_batchmatmul 旁支 / olora_tp 分支。
        #   原 dsa_v1.py:L1652-L1687。保非 A5 主路。
        o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, -1)
        o_proj_input = torch_npu.npu_transpose_batchmatmul(
            o_proj_input,
            self.wo_a.weight,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=1,
        )
        o_proj_input = o_proj_input.reshape(num_tokens, -1)
        output[...] = self.wo_b(o_proj_input)
        return output_padded

    # ===== prefill 稀疏路（compress_ratio==4 主路）=====
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L1866
    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        need_prefill_gather: bool = False,
    ):
        (compress_kv_cache, swa_kv_cache, state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
        )
        # SUBTRACTED: compress_ratio<=1 / ==128 的 attn_metadata 解包旁支；保 ==4（5 元组）主路。
        #   原 dsa_v1.py:L1885-L1892。
        (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = (
            attn_metadata
        )
        compress_common_attn_metadata = compressor_attn_metadata

        common_prefill_metadata = _require_prefill_metadata(compress_common_attn_metadata)
        swa_prefill_metadata = _require_prefill_metadata(swa_metadata)
        cos = common_prefill_metadata.cos[layer_name]
        sin = common_prefill_metadata.sin[layer_name]
        actual_seq_lengths_query = common_prefill_metadata.query_start_loc
        actual_seq_lengths_key = common_prefill_metadata.seq_lens

        # ---- MLA 式低秩 prolog（内联）：q = wq_b(q_norm(wq_a(h)))，kv = kv_norm(wkv(h))，各自加 RoPE ----
        # SUBTRACTED: multistream / need_prefill_gather / W8A8 动态量化(share_hs_quant) 三条 prolog 旁路；
        #   保 native bf16 主路。原 dsa_v1.py:L1901-L1949,L1960-L1968。
        q_a = self.wq_a(hidden_states)
        qr = self.q_norm(q_a)
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        qr_pertoken_scale = None
        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[self.nope_head_dim, self.head_dim]
        )
        kv = self.wkv(hidden_states)
        kv = self.kv_norm(kv)
        assert self.rope_head_dim is not None
        kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[self.nope_head_dim, self.head_dim]
        )
        # 写滑窗 KV cache
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_prefill_metadata.slot_mapping)

        compress_cos = common_prefill_metadata.compress_cos[layer_name]
        compress_sin = common_prefill_metadata.compress_sin[layer_name]

        attn_op = DeviceOperator.get_dsa_sparse_attn_op()  # npu_sparse_attn_sharedkv
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
        DeviceOperator.add_dsa_sparse_attn_extra_kwargs(extra_attn_kwargs, cu_seqlens_ori_kv=actual_seq_lengths_query)

        compressor_prefill_metadata = _require_prefill_metadata(compressor_attn_metadata)
        # 阶段一：Lightning Indexer 选 top-512
        # SUBTRACTED: skip_topk(IndexCache) / multistream(cv_indexer_select_qli) 分支。原 dsa_v1.py:L2023-L2039。
        compress_topk_idxs = self.indexer_select_qli(
            x=hidden_states,
            qr=qr,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            compressed_cos=compress_cos,
            compressed_sin=compress_sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            with_prefill=True,
            qr_pertoken_scale=qr_pertoken_scale,
        )

        # 压缩 KV 入 cache（compressor 算子）
        coff = 2 if self.compressor_overlap else 1
        compressed_kv = torch.ops._C_ascend.compressor(
            hidden_states,
            self.compressor_wkv.weight,
            self.compressor_wgate.weight,
            state_cache.squeeze(-2),
            self.compressor_ape,
            self.compressor_norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table=_require_prefill_metadata(compressor_kv_state_metadata).block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=common_prefill_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        if compressed_kv.numel() == 0:
            compressed_kv = None
        DeviceOperator.dsa_kv_compress_scatter(
            compress_kv_cache, compressed_kv, compressor_prefill_metadata.slot_mapping
        )

        # 阶段二：只对 top-512 个 KV 算稀疏注意力（cmp_sparse_indices=compress_topk_idxs）
        DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
            extra_attn_kwargs, cu_seqlens_cmp_kv=common_prefill_metadata.cu_c4_cmp_seqlen_list
        )
        attn_output = attn_op(
            q,
            ori_kv=swa_kv_cache,
            cmp_kv=compress_kv_cache,
            cmp_sparse_indices=compress_topk_idxs,
            ori_block_table=swa_prefill_metadata.block_table,
            cmp_block_table=compressor_prefill_metadata.block_table,
            cu_seqlens_q=actual_seq_lengths_query,
            seqused_kv=actual_seq_lengths_key,
            sinks=self.attn_sink,
            metadata=common_prefill_metadata.sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            **extra_attn_kwargs,
        )[0]
        return attn_output

    # ===== decode 稀疏路（compress_ratio==4 主路，与 prefill 对称）=====
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L2186
    def _forward_decode(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
    ):
        assert attn_metadata[0].decode is not None
        (compress_kv_cache, swa_kv_cache, state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
        )
        # SUBTRACTED: compress_ratio<=1 / ==128 解包旁支；保 ==4 主路。原 dsa_v1.py:L2206-L2213。
        (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = (
            attn_metadata
        )
        common_decode_metadata = _require_decode_metadata(compressor_attn_metadata)
        swa_decode_metadata = _require_decode_metadata(swa_metadata)
        cos = common_decode_metadata.cos[layer_name]
        sin = common_decode_metadata.sin[layer_name]
        actual_seq_lengths_query = common_decode_metadata.query_start_loc
        actual_seq_lengths_key = common_decode_metadata.seq_lens

        # MLA 式低秩 prolog（native 主路）
        # SUBTRACTED: multistream / W8A8 动态量化分支。原 dsa_v1.py:L2221-L2275,L2287-L2323。
        q_a = self.wq_a(hidden_states)
        qr = q = self.q_norm(q_a)
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        qr_pertoken_scale = None
        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[self.nope_head_dim, self.head_dim]
        )
        kv = self.wkv(hidden_states)
        kv = self.kv_norm(kv)
        assert self.rope_head_dim is not None
        kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[self.nope_head_dim, self.head_dim]
        )
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_decode_metadata.slot_mapping)

        compressor_decode_metadata = _require_decode_metadata(compressor_attn_metadata)
        compressor_state_decode_metadata = _require_decode_metadata(compressor_kv_state_metadata)
        compress_cos = common_decode_metadata.compress_cos[layer_name]
        compress_sin = common_decode_metadata.compress_sin[layer_name]

        # 阶段一：Lightning Indexer 选 top-512
        compress_topk_idxs = self.indexer_select_qli(
            x=hidden_states,
            qr=qr,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            compressed_cos=compress_cos,
            compressed_sin=compress_sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            with_prefill=False,
            qr_pertoken_scale=qr_pertoken_scale,
        )

        coff = 2 if self.compressor_overlap else 1
        compressed_kv = torch.ops._C_ascend.compressor(
            hidden_states,
            self.compressor_wkv.weight,
            self.compressor_wgate.weight,
            state_cache.squeeze(-2),
            self.compressor_ape,
            self.compressor_norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table=compressor_state_decode_metadata.block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=common_decode_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        if compressed_kv.numel() == 0:
            compressed_kv = None
        DeviceOperator.dsa_kv_compress_scatter(
            compress_kv_cache, compressed_kv, compressor_decode_metadata.slot_mapping
        )

        # 阶段二：只对 top-512 算稀疏注意力
        attn_op = DeviceOperator.get_dsa_sparse_attn_op()
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
        attn_output = attn_op(
            q,
            ori_kv=swa_kv_cache,
            cmp_kv=compress_kv_cache,
            cmp_sparse_indices=compress_topk_idxs,
            ori_block_table=swa_decode_metadata.block_table,
            cmp_block_table=compressor_decode_metadata.block_table,
            cu_seqlens_q=actual_seq_lengths_query,
            seqused_kv=actual_seq_lengths_key,
            sinks=self.attn_sink,
            metadata=compressor_decode_metadata.sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            **extra_attn_kwargs,
        )[0]
        return attn_output

    # ===== Lightning Indexer 链路：投影 q/压缩 KV → 量化写 indexer cache → 出 top-512 索引 =====
    # SOURCE: vllm_ascend/attention/dsa_v1.py:L2509
    def _indexer_qkv_prepare(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata

        # SUBTRACTED: W8A8 动态量化(npu_quant_matmul) 分支；保 native inderxer_wq_b 投影。原 dsa_v1.py:L2534-L2547。
        q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)  # [T, N, D]
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.indexcom_head_dim - self.rope_head_dim, self.indexcom_head_dim],
        )
        q = rotate_activation(q, indexer_kv_scale_metadata.hadamard)
        coff = 2 if self.compressor_overlap else 1

        if with_prefill:
            kv_block_table = _require_prefill_metadata(indexer_kv_state_metadata).block_table
            start_pos = _require_prefill_metadata(indexer_kv_scale_metadata).start_pos
        else:
            kv_block_table = _require_decode_metadata(indexer_kv_state_metadata).block_table
            start_pos = _require_decode_metadata(indexer_kv_scale_metadata).start_pos

        # indexer 自己的 compressor：把 KV 压成更短代表(cmp_ratio)再打分，进一步压索引开销
        kv = torch.ops._C_ascend.compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            indexer_state_cache.squeeze(-2),
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            state_block_table=kv_block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        if kv.numel() == 0:
            kv = None
        elif self.indexcom_rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        return (q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache,
                indexer_kv_state_metadata, indexer_kv_scale_metadata, with_prefill)

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L2641
    def _indexer_quant_scatter(
        self, q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, indexer_kv_scale_metadata, with_prefill
    ):
        slot_mapping = (
            indexer_kv_scale_metadata.prefill.slot_mapping
            if with_prefill
            else indexer_kv_scale_metadata.decode.slot_mapping
        )
        return DeviceOperator.indexer_quant_scatter(
            q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, slot_mapping
        )

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L2660
    def _indexer_qli(
        self, q, weights, q_scale, indexer_k_cache, indexer_scale_cache, indexer_kv_scale_metadata, with_prefill
    ):
        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            qlens = indexer_kv_scale_metadata.prefill.query_start_loc[1:]
            kvlens = indexer_kv_scale_metadata.prefill.seq_lens
            block_table = indexer_kv_scale_metadata.prefill.block_table
            qli_metadata = indexer_kv_scale_metadata.prefill.qli_metadata
        else:
            assert indexer_kv_scale_metadata.decode is not None
            qlens = indexer_kv_scale_metadata.decode.query_start_loc[1:]
            kvlens = indexer_kv_scale_metadata.decode.seq_lens
            block_table = indexer_kv_scale_metadata.decode.block_table
            qli_metadata = indexer_kv_scale_metadata.decode.qli_metadata

        # Lightning Indexer 真身：出 top-index_topk(=512) 个最相关 KV 的索引 topk_idxs
        topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(
            query=q,
            key=indexer_k_cache,
            weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
            key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_key=kvlens,
            block_table=block_table,
            metadata=qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,  # 512
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False,
        )
        return topk_idxs

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L2610
    def _indexer_qli_finish(
        self, q, kv, weights, indexer_k_cache, indexer_scale_cache, indexer_full_cache,
        indexer_kv_state_metadata, indexer_kv_scale_metadata, with_prefill,
    ):
        q, q_scale, kv, kv_scale = self._indexer_quant_scatter(
            q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, indexer_kv_scale_metadata, with_prefill
        )
        return self._indexer_qli(
            q, weights, q_scale, indexer_k_cache, indexer_scale_cache, indexer_kv_scale_metadata, with_prefill
        )

    # SOURCE: vllm_ascend/attention/dsa_v1.py:L2706
    def indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor | None = None,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        q, kv, ik, isc, ifc, indexer_kv_state_meta, isc_meta, wp = self._indexer_qkv_prepare(
            x, qr, kv_cache, attn_metadata, cos, sin, compressed_cos, compressed_sin,
            actual_seq_lengths_query, with_prefill, qr_pertoken_scale,
        )
        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexer_heads**-0.5)
        return self._indexer_qli_finish(q, kv, weights, ik, isc, ifc, indexer_kv_state_meta, isc_meta, wp)

    # SUBTRACTED: cv_indexer_select_qli（多流版）/ _mla_prolog_multistream / _mla_prolog_prefill_overlap /
    #   dsa_warmup_with_multistream / _get_indexcache_topk_indices / _update_indexcache_topk_indices /
    #   _is_w8a8_dynamic —— 多流重叠/IndexCache/量化辅助。原 dsa_v1.py:L1487-L1573,L1691-L1865,L2739+。


# SOURCE: vllm_ascend/attention/dsa_v1.py:L61
def hadamard_transform_ref(x: torch.Tensor, hadamard: torch.Tensor, scale: float = 1.0):
    # 哈达玛旋转参考实现（dsv4 indexer 用）：把激活旋到便于稀疏打分的基。
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


# SOURCE: vllm_ascend/attention/dsa_v1.py:L78
def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, hadamard=hadamard, scale=hidden_size**-0.5)
