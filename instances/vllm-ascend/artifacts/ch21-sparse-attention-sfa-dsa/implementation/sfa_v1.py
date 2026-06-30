# vllm_ascend/attention/sfa_v1.py —— subtract-only 精简版（SFA：在 MLA 之上叠「稀疏选择」）
#
# 本章主角之一：AscendSFAImpl —— ch18 路由 (use_mla,use_sparse,use_compress)=(T,T,F) 选出的
# AscendSFABackend 的 impl。SFA 是 MLA 的「稀疏增量」：
#   - AscendSFAMetadataBuilder 继承 ch20 的 MLACommonMetadataBuilder —— 元数据装配整套复用 MLA；
#   - AscendSFAImpl 继承 vLLM 的 MLAAttentionImpl —— 低秩 KV 压缩 / 权重吸收 / prefill-decode 范式整套复用；
#   - 在 MLA 之上，forward 只多做两件事（两段式稀疏）：
#       阶段一 indexer_select_pre/post_process → DeviceOperator.indexer_select_post_process
#               → npu_lightning_indexer(sparse_count=2048, sparse_mode=3) 选出每个 query 最相关的 top-k 个 KV；
#       阶段二 _execute_sparse_flash_attention_process → npu_sparse_flash_attention(sparse_indices=topk_indices)
#               只对这 top-k 个 KV 算全精度注意力，把 O(L) 降到 O(top-k)。
#   - assert self.indexer is not None —— 稀疏选择的「索引器」是 SFA 的必备件。
#
# vLLM 主干无对位后端：这是插件在 MLA 基础上「增量扩展」出的算法，不是替换某个内核。
#
# host 无 CANN/torch_npu：真实 npu_lightning_indexer / npu_sparse_flash_attention / npu_rotary_mul 等
# 算子由测试的「记录调用」替身承接，只验两段式控制流（选 top-k → 只对 top-k 算）与入参，不真算。
#
# SUBTRACTED: 本文件原 1376 行。删除 subtraction_plan 批准的：CP/上下文并行（enable_dsa_cp* /
#   DSACPContext / all_gather_async / _handle_o_proj_weight_switch_and_forward / sfa_cp 子类）、
#   MLAPO 融合与 A5 变体（enable_mlapo / _sfa_preprocess_with_mlapo / _process_weights_for_fused_mlapo*）、
#   use_sparse_c8_indexer（INT8 量化索引：q/k_hadamard / npu_dynamic_quant / npu_lightning_indexer_quant /
#   kv_cache 3↔4 元组切换 / store_kv_block）、KV connector/PD 钩子、IndexCache 复用（skip_topk /
#   topk_indices_buffer / _get_indexcache_topk_indices / _update_indexcache_topk_indices）。
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    MLAAttentionImpl,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.device.device_op import DeviceOperator

# SUBTRACTED: import scipy / HAS_TRITON / triton rope / context_parallel.* / layer_shard_linear /
#   quantization.methods / all_gather_async / connector 钩子 / NPUInputBatch —— 均服务已减的
#   量化/CP/triton/MLAPO/PD 旁路。原 sfa_v1.py:L4,L14,L23-L62。
HAS_TRITON = False  # SUBTRACTED: host 无 triton；RoPE 走 npu_rotary_mul 主路（保留等价分支）

# token count limits within bmm_transpose operator
# SOURCE: vllm_ascend/attention/sfa_v1.py:L68
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024


# SOURCE: vllm_ascend/attention/sfa_v1.py:L71
class AscendSFABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L74
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.（回指 ch18：故意改名绕过 model-runner 名断言）
        return "ASCEND_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L81
    def get_builder_cls():
        # SUBTRACTED: enable_cp() → AscendSFACPMetadataBuilder 分支（CP）。原 sfa_v1.py:L83-L86。
        return AscendSFAMetadataBuilder

    @staticmethod
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L89
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L99
    def get_impl_cls() -> type["AscendSFAImpl"]:
        # SUBTRACTED: enable_cp() → AscendSFACPImpl 分支（CP）。原 sfa_v1.py:L101-L104。
        return AscendSFAImpl

    @staticmethod
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L107
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


@dataclass
# SOURCE: vllm_ascend/attention/sfa_v1.py:L124
class AscendSFAMetadata:
    """Metadata for MLACommon（继承 MLA 元数据范式：cos/sin/block_table/slot_mapping/seq_lens）。"""

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_input_tokens: int = 0  # Number of tokens including padding.
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    block_size: int = 0
    # SUBTRACTED: dsa_cp_context / sfa_cp_metadata / reshape_cache_event / group_* (CP/connector/c8 字段)。
    #   原 sfa_v1.py:L155-L164。


M = TypeVar("M", bound=AscendSFAMetadata)


# SOURCE: vllm_ascend/attention/sfa_v1.py:L170
class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata]):
    """建在 MLA 之上的字面证据 (1)：直接继承 ch20 的 MLACommonMetadataBuilder，元数据装配整套复用。"""

    # SOURCE: vllm_ascend/attention/sfa_v1.py:L176
    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendSFAMetadata,
            supports_dcp_with_varlen,
        )
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size
        # SUBTRACTED: speculative_config / dsa_cp / c8_reshape_optim 等 build 期分支与 build() 本体——
        #   元数据装配复用 MLA 基类，本章只需可见「继承关系」。原 sfa_v1.py:L197-L387。


# SOURCE: vllm_ascend/attention/sfa_v1.py:L390
class AscendSFAImpl(MLAAttentionImpl):
    """建在 MLA 之上的字面证据 (2)：直接继承 vLLM 的 MLAAttentionImpl，低秩压缩/权重吸收整套复用；
    forward 在 MLA 之上叠加两段式稀疏选择。"""

    o_proj_full_pool: torch.Tensor | None = None
    q_hadamard: torch.Tensor | None = None
    k_hadamard: torch.Tensor | None = None

    # SOURCE: vllm_ascend/attention/sfa_v1.py:L403
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
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args（低秩压缩相关，回指 ch20）
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.q_b_proj = kwargs["q_b_proj"]

        # 稀疏选择的「索引器」是 SFA 的必备件（无它就退化成普通 MLA）。
        assert self.indexer is not None, "Indexer is required for DSA."

        self.local_num_heads = self.num_heads
        self.vllm_config = get_current_vllm_config()
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )

        # indexer param：独立的小投影（n_head=64, head_dim=128）+ RoPE，算一个轻量「相关性代理分数」。
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.wq_b = self.indexer.wq_b
        self.wk_weights_proj = self.indexer.wk_weights_proj
        self.k_norm = self.indexer.k_norm
        self.is_rope_neox_style = True
        self.use_torch_npu_lightning_indexer = False
        self.use_sparse_c8_indexer = False
        # SUBTRACTED: enable_mlapo / enable_dsa_cp* / enable_sp / use_index_cache / skip_topk /
        #   topk_indices_buffer / use_sparse_c8_indexer 的 c8 cache dtype 初始化 / o_proj_tp 全套——
        #   服务已减的 MLAPO/CP/IndexCache/C8。原 sfa_v1.py:L443-L560。

    # ---- 以下 rope_single / exec_kv / _q_proj_and_k_up_proj / _v_up_proj / _get_full_kv 是 MLA 式
    #      低秩 prolog 助手（回指 ch20，SFA 在其基础上叠稀疏）；权重吸收 W_UK_T/W_UV 来自 ch20 的
    #      process_weights_after_loading（本章已减）。----

    # SOURCE: vllm_ascend/attention/sfa_v1.py:L763
    def rope_single(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    # SOURCE: vllm_ascend/attention/sfa_v1.py:L853
    def _get_full_kv(self, k, attn_metadata):
        return k

    # SOURCE: vllm_ascend/attention/sfa_v1.py:L856
    def exec_kv(self, kv_no_split, cos, sin, kv_cache, slots, attn_metadata):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]：一把做 RMSNorm + RoPE + 写分页 KV cache（回指 ch20）
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"
        # SUBTRACTED: enable_dsa_cp 的 is_output_kv=True 分支（CP）。原 sfa_v1.py:L872-L885。
        torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
        )
        return None, None

    # SOURCE: vllm_ascend/attention/sfa_v1.py:L901
    def _q_proj_and_k_up_proj(self, x):
        # MLA 权重吸收：把 q_nope「吸收」进 latent 空间（回指 ch20）。
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.local_num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )
        q_nope = q_nope.transpose(0, 1)  # (B, N, P) -> (N, B, P)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)  # (N, B, P) x (N, P, L) -> (N, B, L)
        return ql_nope.transpose(0, 1), q_pe  # (N, B, L) -> (B, N, L)

    # SOURCE: vllm_ascend/attention/sfa_v1.py:L915
    def _v_up_proj(self, x):
        # SUBTRACTED: batch_matmul_transpose 与纯 torch.bmm 两条等价旁支（不同算子可用性）；保
        #   npu_transpose_batchmatmul 主路。原 sfa_v1.py:L917-L925,L933-L939。
        x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
        x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
        x = x.reshape(-1, self.local_num_heads * self.v_head_dim)
        return x

    # ===== 阶段一·key 侧：把 hidden_states 投影成 indexer 的 k_li 并加 RoPE =====
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L961
    def indexer_select_pre_process(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        kw, _ = self.wk_weights_proj(x)
        k_li = kw[:, : self.head_dim]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        # SUBTRACTED: HAS_TRITON 的 rope_forward_triton_siso 等价分支（host 无 triton）；保 npu_rotary_mul。
        #   原 sfa_v1.py:L972-L977。
        k_li_pe, k_li_nope = torch.split(
            k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        k_li_pe = k_li_pe.unsqueeze(2)
        k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
        k_li_pe = k_li_pe.squeeze(2)
        k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        # SUBTRACTED: use_sparse_c8_indexer 的 k_hadamard + npu_dynamic_quant（INT8 量化索引）。
        #   原 sfa_v1.py:L992-L998。
        k_li_scale = None
        return k_li, k_li_scale

    # ===== 阶段一·query 侧：投影成 q_li、加 RoPE，交 DeviceOperator 跑 lightning indexer 出 top-k =====
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L1002
    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ):
        kw, _ = self.wk_weights_proj(x)
        weights = kw[:, self.head_dim :]
        # SUBTRACTED: isinstance(q_c, tuple) 的 MLAPO-C8 预量化分支（npu_quant_matmul 直吃 fp8）；
        #   保 native 的 wq_b 投影。原 sfa_v1.py:L1015-L1033。
        q_li, _ = self.wq_b(q_c)
        q_li = q_li.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]

        # SUBTRACTED: HAS_TRITON 等价分支。原 sfa_v1.py:L1037-L1040。
        q_li_pe, q_li_nope = torch.split(
            q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )
        q_li_pe = q_li_pe.unsqueeze(2)
        q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
        q_li_pe = q_li_pe.squeeze(2)
        q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)  # [b*s,64,128]

        q_li_scale = None
        q_li_shape_ori = None
        # SUBTRACTED: use_sparse_c8_indexer 的 q_hadamard + npu_dynamic_quant。原 sfa_v1.py:L1053-L1057。

        # 门面派发：default 分支走 npu_lightning_indexer(sparse_count=2048, sparse_mode=3) 出 topk_indices。
        return DeviceOperator.indexer_select_post_process(
            self,
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            kv_cache,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            self.use_sparse_c8_indexer,
            self.use_torch_npu_lightning_indexer,
        )

    # ===== 阶段二：只对 top-k 个 KV 算全精度注意力 =====
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L1093
    def _execute_sparse_flash_attention_process(
        self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
    ):
        return DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

    # ===== forward 主脊：MLA 式低秩 prolog → 选 top-k → 只对 top-k 算 → _v_up_proj → o_proj =====
    # SOURCE: vllm_ascend/attention/sfa_v1.py:L1107
    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens
        num_input_tokens = attn_metadata.num_input_tokens
        output_padded = output

        # ---- native（非 MLAPO）主路：MLA 式 q/kv 低秩 prolog（回指 ch20）----
        # SUBTRACTED: enable_mlapo / enable_dsa_cp / enable_sp 三条 prolog 旁路与 o_proj_tp 全 gather。
        #   原 sfa_v1.py:L1142-L1167,L1175-L1178,L1191-L1275,L1358-L1370。
        assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
        q_c = self.q_a_layernorm(q_c)

        # 阶段一·key 侧：建 indexer 的 k_li
        k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)

        # 写 MLA 低秩 KV cache（kv_cache[0]/[1]）
        self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata)

        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        q_pe = self.rope_single(q_pe, cos, sin)
        k_li = self._get_full_kv(k_li, attn_metadata)

        if kv_cache is not None:
            # 把 indexer 的 k_li 写进 indexer KV cache（kv_cache[2]），供 lightning indexer 打分。
            # SUBTRACTED: is_kv_producer 的 reshape_cache_event / c8_enable_reshape_optim store_kv_block /
            #   use_sparse_c8_indexer 的 scale cache 写入。原 sfa_v1.py:L1277-L1326。
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, k_li.shape[-1]),
                slot_mapping.view(-1, 1),
                k_li.view(-1, k_li.shape[-1]),
            )

        # 阶段一·query 侧 + 选 top-k（sparse_count=2048）
        # SUBTRACTED: skip_topk / use_index_cache 的 IndexCache 复用分支（跨层共享 top-k）。
        #   原 sfa_v1.py:L1329-L1343。
        topk_indices = self.indexer_select_post_process(
            x=hidden_states,
            q_c=q_c,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
        )

        # 阶段二：只对 top-k 个 KV 算全精度稀疏 flash 注意力
        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
        )

        attn_output = self._v_up_proj(attn_output)
        output[...] = self.o_proj(attn_output)[0]
        return output_padded
