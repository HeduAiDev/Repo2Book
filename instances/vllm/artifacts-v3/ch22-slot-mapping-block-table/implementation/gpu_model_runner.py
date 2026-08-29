# SOURCE: vllm/v1/worker/gpu_model_runner.py
# ch22 切面：GPUModelRunner 的**块表线**——_update_states 的块号差量调和落
# 行（站1-2）、_prepare_inputs 的 commit 先行 + GPU 端 positions/seq_lens +
# compute_slot_mapping 派发（站3-5）、_get_slot_mappings 双口径装配（站8-9）、
# _build_attention_metadata 的块表尾行 NULL_BLOCK_ID + CommonAttentionMetadata
# 收束（站10）、execute_model 的 has_separate_kv_update 裁决 + set_forward_
# context 逐层铺设（站11）、_preprocess 的 positions 尾清零（四件套之一）、
# may_reinitialize_input_batch 的模式装配（m10）。
# 模型执行全景归 ch17；持久批次/采样归 ch18；编译与 CUDA graph 本体归 ch19；
# 注意力数学归 ch20；后端选择/逐组混布/builder 归 ch21。
# SUBTRACTED：dossier.subtraction_plan.delete[0..14] 批准项（逐条就地标注
# 「delete[N]」）；章界外域段以「SUBTRACTED + 归属章」注记（切面惯例）。
# HOST SEAM：device=CPU 时块表/kernel 全走 block_table.py 的 CPU 镜像；
# _model_forward 以脚本化单 Attention 层前向承载（ch17 模型域，ch18 同款
# seam——两算子序列 unified_kv_cache_update → unified_attention_with_output
# 是真实 Attention.forward 的调用序）。
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np
import torch

from ._host_seams import (
    CUDAGraphMode,
    EMPTY_MODEL_RUNNER_OUTPUT,
    InputBatchSeam,
    init_logger,
    record_function_or_nullcontext,
)
from .attention import unified_attention_with_output, unified_kv_cache_update
from .backend import CommonAttentionMetadata
from .backends_utils import NULL_BLOCK_ID
from .block_table import SlotMappingMode
from .flash_attn import FlashAttentionMetadata
from .forward_context import get_forward_context, set_forward_context
from .kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)
from .torch_utils import PIN_MEMORY
from .v1_utils import CpuGpuBuffer

logger = init_logger(__name__)

# SUBTRACTED: vllm 顶层 import 归一（replace/envs/mamba_utils/copy 等——各
#   归属域见就地注释）；NULL_BLOCK_ID 从 backends_utils 折入（下方）。


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L437 ExecuteModelState ——
#   两段式契约的暂存协议本体（slot_mappings 随行带过——m12）
# SOURCE: vllm/v1/worker/gpu_model_runner.py:L438 ExecuteModelState
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: Any
    logits: torch.Tensor | None
    spec_decode_metadata: Any | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor | None
    sample_hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: Any | None
    cudagraph_stats: Any | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


# SUBTRACTED: 三支 mixin 基类（LoRA/KVConnector/ECConnector——ch33/ch16 域）。
# SOURCE: vllm/v1/worker/gpu_model_runner.py:L453 GPUModelRunner —— 本章舞台类
#   （块表线切面；真实类为千行执行器，ch17/ch18 全文）
class GPUModelRunner:
    # HOST SEAM 装配位：真实 __init__（L456-L760）由 vllm_config 装配模型/
    # 采样/cudagraph/spec 面并派生 attn_groups——本章切面经装配参数直供
    # （ch13 同款切面构造）；持久缓冲块逐字镜像 L763-L845 的块表线子集。
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456 __init__ 签名（切面化）
    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any,
        attn_groups: list[list],
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.device = device
        self.kv_cache_config = kv_cache_config
        self.attn_groups = attn_groups
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_encoder_len = 0  # HOST SEAM：encoder 域装配位（delete[7]——恒 0）
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.num_spec_tokens = 0

        # 切面配置位（各域默认关——真实守卫行为分支无涉）
        self.uses_mrope = False  # delete[0]
        self.uses_xdrope_dim = 0  # delete[0]
        self.enable_prompt_embeds = False  # delete[1]
        self.supports_mm_inputs = False  # mm 域
        self.is_mm_prefix_lm = False  # mm 域
        self.use_async_scheduling = False  # delete[2] 域（ch12）
        self.use_async_spec_decode = False  # delete[2]（ch12/ch33）
        self.calculate_kv_scales = False
        self.eplb_state = None
        self.use_aux_hidden_state_outputs = False
        self.broadcast_pp_output = False  # delete[10]
        self.execute_model_state: ExecuteModelState | None = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L763-L845 持久缓冲块
        #   （"Persistent buffers for CUDA graphs."——块表线子集逐字镜像；
        #   prev_positions/num_accepted_tokens 等影子缓冲随 delete[2] 删）
        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.optimistic_seq_lens_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.num_scheduled_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        # SUBTRACTED: prev_num_draft_tokens/prev_positions/discard_request_
        #   mask/num_decode_draft_tokens/num_accepted_tokens/inputs_embeds/
        #   is_token_ids/mrope/xdrope 缓冲（L780-L782、L785-L833——delete[0][1][2]）。

        # OPTIMIZATION: Cache the arange tensors rather than creating them
        # every step. Keep it in int64 to avoid overflow with long context.
        # - arange_np: immutable [0, 1, 2, ...] used as source for batched computation
        # - query_pos: CpuGpuBuffer for the computed batched arange result
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L838-L845
        arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
        self.arange_np = np.arange(arange_size, dtype=np.int64)
        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)

        # InputBatch 装配（delete[14]：块表线字段面；构造镜像 gpu_input_batch
        #   的真实装配序）+ may_reinitialize 的初始化占位账
        self.requests: dict[str, Any] = {}
        self._init_block_sizes = [
            g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
        ]
        self._init_kernel_block_sizes = list(self._init_block_sizes)
        self._init_max_num_blocks = [
            g.kv_cache_spec.max_num_blocks_per_req(vllm_config, self.max_model_len)
            for g in kv_cache_config.kv_cache_groups
        ]
        self._init_slot_mapping_modes = [
            SlotMappingMode.TOKEN_TO_KV_SLOT
            if get_kv_cache_spec_kind(g.kv_cache_spec) != KVCacheSpecKind.MAMBA
            else SlotMappingMode.NONE
            for g in kv_cache_config.kv_cache_groups
        ]
        self.input_batch = InputBatchSeam(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=device,
            block_sizes=list(self._init_block_sizes),
            kernel_block_sizes=list(self._init_kernel_block_sizes),
            max_num_blocks_per_req=list(self._init_max_num_blocks),
            num_spec_tokens=self.num_spec_tokens,
            cp_kv_cache_interleave_size=(
                vllm_config.parallel_config.cp_kv_cache_interleave_size
            ),
            slot_mapping_modes=list(self._init_slot_mapping_modes),
        )

        # HOST SEAM 观测位（ch19 域的 padded 口径装配——真实 _determine_batch_
        #   execution_and_padding 的产出；测试按需置 FULL/BatchDescriptor）
        self.seam_cudagraph_mode = CUDAGraphMode.NONE
        self.seam_batch_desc = None
        # HOST SEAM 观测位（ch17 模型域前向 seam 的记录器）
        self.seam_model_output: dict[str, torch.Tensor] = {}
        self.seam_seen_slot_mapping = None
        self.seam_seen_layer_name = None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1046 _make_buffer
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1048-L1054
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=PIN_MEMORY,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1074 _init_model_kwargs
    #   （HOST SEAM：模型 kwargs 装配面 → ch17；恒空 dict）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1074 _init_model_kwargs（HOST SEAM：ch17 模型 kwargs 面）
    def _init_model_kwargs(self) -> dict:
        return {}

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1192 _update_states —— 块表
    #   线切面（站1：块号差量调和 → append_row 落 CPU 行）
    def _update_states(self, scheduler_output: Any) -> Callable | None:
        """Update the cached states and the persistent batch with the scheduler
        output（块表线切面——ch18 讲过整段调和，本章只跟块号这一条线）.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.
        """
        # SUBTRACTED: finished/unscheduled 移除、新块 GPU 置零、CoW 拷贝、
        #   encoder 释放、新请求建 CachedRequestState、streaming 更新
        #   （L1202-L1330——ch18 域全文/ch13 第 8 站；ngram 镜像账 delete[11]；
        #   mrope/xdrope 初始化 delete[0]）。
        # SUBTRACTED: PP is_last_rank（L1333——delete[10]：其唯一消费支
        #   L1408-L1439/L1476-L1494 token 回填/写回亦删，单 rank 恒 last）。
        req_data = scheduler_output.scheduled_cached_reqs
        # SUBTRACTED: ngram-GPU 的 original_num_spec_per_req 记账与
        #   update_scheduler_for_invalid_drafts（L1337-L1351——delete[11]）；
        #   async spec 的 prev_num_draft_tokens 清零（L1352-L1353——delete[2]）。
        # SUBTRACTED: async spec 乐观纠偏全家（L1363-L1403——delete[2]：
        #   prev_num_draft_len 记账/乐观 extend/deferred corrections——
        #   非 async spec 配置下不进）。

        for i, req_id in enumerate(req_data.req_ids):
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1356-L1361 解包
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            # SUBTRACTED: num_output_tokens/new_token_ids 解包（L1360——其
            #   消费支 PP/async 均已删：L1408-L1439——delete[10]/delete[2]）。
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # SUBTRACTED: async spec 乐观纠偏（L1363-L1403——delete[2]）。

            # Update the cached states.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1405-L1406
            req_state.num_computed_tokens = num_computed_tokens

            # SUBTRACTED: PP 非末 rank token 回填与 async 纠偏对齐
            #   （L1408-L1439——delete[10]/delete[2]）。

            # Update the block IDs.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1441-L1452 —— 差量
            #   extend / 恢复整表替换（块号过线的第一跳）
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1454-L1457 判别位
                #   （SUBTRACTED：async 恢复 L1459-L1463——delete[2]；
                #   reqs_to_add.append + ngram 追踪 L1465-L1468——落位面 →
                #   ch18 add_request/add_row；本章 add_row 由块表单测直验）。
                continue

            # Update the persistent batch.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1471-L1474 —— 块号
            #   落 CPU 页表行（站1 终点：block_table.append_row）
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # SUBTRACTED: PP token 写回段（L1476-L1494——delete[10]）。

        # SUBTRACTED: reqs_to_add 落位/condense 压实/重排钩子/refresh_metadata
        #   （L1499 起——ch18 域全文）；deferred spec corrections 的返回值
        #   （delete[2]——本章恒返回 None）。
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1743 _get_cumsum_and_arange
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        arange_out: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        """Get the cumulative sum and batched arange of the given array.
        E.g., [2, 5, 3] -> [2, 7, 10], arange written to
        arange_out[:10] as [0, 1, 0, 1, 2, 3, 4, 0, 1, 2].
        Equivalent to but faster than:
        np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1756-L1757
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1759
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1761-L1765
        np.subtract(
            self.arange_np[:total_num_tokens],
            cumsums_offsets,
            out=arange_out[:total_num_tokens],
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1767
        return cu_num_tokens

    # SUBTRACTED: _compute_prev_positions / _prepare_input_ids（L1769-L1913
    #   ——delete[2] 前者；后者固定地址前缀上载与 async scatter → ch12/ch18）。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1960 _prepare_inputs ——
    #   块表线切面（站3-5：commit 先行 → GPU positions/seq_lens → 派发）
    def _prepare_inputs(
        self,
        scheduler_output: Any,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        Any | None,
    ]:
        """
        Returns:
            tuple[logits_indices, spec_decode_metadata]
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1972-L1975
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1977-L1979 —— 先行拷贝
        #   三行（站3：H2D 与 CPU 活重叠）
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1981-L1983
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1985-L1989
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )

        # SUBTRACTED: CPU 侧 positions_np 与 token 收集（L1991-L1995、
        #   L2007-L2070——token_indices/index_select 扁平收集 → ch18 第 6 站；
        #   prompt_embeds 收集 → delete[1]）；mrope/xdrope 计算（L1997-L2005
        #   ——delete[0]——GPU 端 positions 不经此路，见 L2188-L2191）。

        # Prepare the attention metadata.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2073-L2078 —— query_
        #   start_loc 非递减 pad（四件套之一：'kernels like FlashAttention
        #   requires that'）
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        # Compute optimistic seq_lens (assumes all draft tokens from previous
        # iteration accepted). Store in optimistic_seq_lens_cpu for use by
        # _build_attention_metadata (max_seq_len) and discard_request_mask.
        # seq_lens (GPU) will be computed later using the same optimistic values.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2081-L2090 —— 无条件装配
        #   半边（_build_attention_metadata 的 max_seq_len/seq_lens_cpu 消费
        #   它；无 spec 时 optimistic == 精确值）保留；spec 纠偏半边见下删。
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # SUBTRACTED: async spec 乐观纠偏全家（L2092-L2141——delete[2]：
        #   prev_positions 装配/discard_request_mask/num_accepted_tokens 事件
        #   同步；mamba 预处理 L2143-L2150——delete[3]；num_computed_tokens 的
        #   GPU 纠偏支 L2152-L2173——delete[2]，else 直拷支去缩进为无条件↓）。

        # Update num_computed_tokens on GPU. ... Otherwise, just copy from CPU.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2174-L2178（else 支原样，
        #   delete[2] 批准后去缩进）
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2180-L2195 —— GPU 端
        #   组装（站4：换算的输入已是 GPU 张量、全程不落 CPU）
        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2197-L2201 —— 派发
        #   compute_slot_mapping（站5：吃 query_start_loc.gpu 与 GPU positions）
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # SUBTRACTED: _prepare_input_ids 前缀上载（L2204-L2209——ch12/ch18）；
        #   mrope/xdrope GPU 拷贝与 async 漂移修正（L2211-L2230——delete[0][2]）。

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2232-L2241 非 spec 支
        #   （common case 逐字）
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
        # SUBTRACTED: spec decode 支（L2242-L2267——num_draft_tokens 收集/
        #   _calc_spec_decode_metadata → ch12/ch33）与 LoRA 热切换
        #   （L2269-L2277——ch33 域）。

        return (
            logits_indices,
            spec_decode_metadata,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2284 _build_attention_metadata
    #   —— 站10：块表尾行 NULL_BLOCK_ID + CommonAttentionMetadata 收束
    def _build_attention_metadata(
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
    ) -> tuple[dict, CommonAttentionMetadata | None]:
        """
        Returns:
            tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2303-L2309
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: dict = {}
        # SUBTRACTED: ubatch 的 list 形态（L2311-L2313——delete[5]）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2315-L2321 max_seq_len
        if for_cudagraph_capture:
            # For some attention backends (e.g., FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2325-L2341 _get_block_table
        #   闭包 —— 行侧装配（尾行填 NULL_BLOCK_ID）
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2325 _get_block_table 闭包
        def _get_block_table(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            # SUBTRACTED: EncoderOnlyAttentionSpec 零表支（L2328-L2333——
            #   delete[7]：encoder-only 模型无 paged KV，本章组面不进）。
            blk_table = self.input_batch.block_table[kv_cache_gid]
            blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # Fill unused block table entries with NULL_BLOCK_ID (null block)
            # for CUDAGraph padding. Block 0 is reserved for padding.
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
            return blk_table_tensor

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2343-L2345
        assert slot_mappings is not None
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        # SUBTRACTED: routed_experts 私有快照（L2347-L2358——delete[6]：共享
        #   buffer 会被下拍覆写、async D2H 在途——MoE 专家路由捕获特性默认关）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2360-L2375 CPU 列镜像 +
        #   is_prefilling（尾行清 False 防 condense 残留误判）
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        seq_lens_cpu_upper_bound = seq_lens_cpu

        # is_prefilling: True if request is still in prefill phase.
        # Used by mamba backends to distinguish actual decodes from
        # short extends.
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
        # Zero out padded rows so stale data from condense() doesn't
        # misclassify padding as prefill in CUDA graph mode.
        is_prefilling[num_reqs:] = False

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2377-L2380（async 模式
        #   CPU 镜像让位 GPU 真相——本章 use_async_spec_decode 恒 False，
        #   判别位保留）
        if self.use_async_spec_decode:
            # GPU tensors are authoritative in async mode.
            seq_lens_cpu = None
            num_computed_tokens_cpu = None

        # SUBTRACTED: mm_prefix bidirectional ranges 块（L2382-L2415——多模态
        #   域，is_mm_prefix_lm=False 恒不进）；R-SWA prefix 块（L2417-L2422
        #   ——ch21 域，rswa_window=None 恒不进）；replayssm 块（L2424-L2428
        #   ——delete[14] 域，use_replayssm=False 恒不进）。
        req_doc_ranges = None
        rswa_prefix_lens = None
        replayssm_decode_base_cpu = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2430-L2449 cm_base 收束
        #   —— 读腿表（block_table_tensor）与写腿索引（slot_mapping）连同
        #   query_start_loc/seq_lens/num_reqs/positions 一起过桥
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            replayssm_decode_base_cpu=replayssm_decode_base_cpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            is_prefilling=is_prefilling,
            positions=self.positions[:num_tokens_padded],
            mm_req_doc_ranges=req_doc_ranges,
            rswa_prefix_lens=rswa_prefix_lens,
        )

        # SUBTRACTED: dcp_local_seq_lens 块（L2451-L2464——delete[13]：dcp
        #   多 rank 部署 → 分布式 Part，单卡不进）；kv_sharing_fast_prefill
        #   块（L2466-L2470——delete[9]）。

        # SUBTRACTED: 逐组 builder.build 循环与 hybrid metadata 缓存复用
        #   （L2472-L2600——后端 metadata 构建域 → ch21；delete[8] 批准的
        #   cached_attn_metadata/update_block_table 命中支在其内）。以
        #   ENGINE SEAM 承载 builder 的非级联 pass-through 半边：common →
        #   FlashAttentionMetadata 字段直拷（真身 flash_attn.py:L458-L476
        #   解包 + L672-L697 组装，单卡/非级联/非量化字段面）。
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            blk_table_tensor_i = (
                block_table_gid_0 if kv_cache_gid == 0 else _get_block_table(kv_cache_gid)
            )
            attn_metadata_i = FlashAttentionMetadata(
                num_actual_tokens=cm_base.num_actual_tokens,
                max_query_len=cm_base.max_query_len,
                query_start_loc=cm_base.query_start_loc,
                max_seq_len=cm_base.max_seq_len,
                seq_lens=cm_base.seq_lens,
                block_table=blk_table_tensor_i,
                slot_mapping=slot_mappings[kv_cache_gid],
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
                causal=cm_base.causal,
            )
            for layer_name in kv_cache_group.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata, spec_decode_common_attn_metadata

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3545 _preprocess ——
    #   四件套之 positions 尾清零的所在（L3663-L3664）
    def _preprocess(
        self,
        scheduler_output: Any,
        num_input_tokens: int,  # Padded
        intermediate_tensors: Any | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        Any | None,
        dict[str, Any],
        Any | None,
    ]:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3558
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # SUBTRACTED: is_first_rank/is_encoder_decoder（L3559-L3560——PP/
        #   encoder 域 delete[10]/delete[7]，仅被已删支消费）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3562-L3564 spec 占位符
        #   截断（判别位保留；speculative_config 恒 None 不触发）
        if self.speculative_config is not None:
            self.input_ids.gpu[:num_input_tokens].clamp_(min=0)

        # SUBTRACTED: mm encoder/embed 收集与 prompt_embeds 支（L3566-L3647
        #   ——多模态域 + delete[1]）。
        ec_connector_output = None

        # For text-only models, we use token ids as input.
        # While it is possible to use embeddings as input just like the
        # multimodal models, it is not desirable for performance since
        # then the embedding layer is not included in the CUDA graph.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3648-L3655（else 支
        #   原样——上方两支删后为唯一路径）
        input_ids = self.input_ids.gpu[:num_input_tokens]
        inputs_embeds = None
        model_kwargs = self._init_model_kwargs()

        # SUBTRACTED: mrope/xdrope 两支（L3657-L3660——delete[0]）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3662-L3664 —— 四件套
        #   之四：positions 尾部清零（padded 区不残留上一拍真位置）
        positions = self.positions[:num_input_tokens]
        if num_input_tokens > num_scheduled_tokens:
            self.positions[num_scheduled_tokens:num_input_tokens].zero_()

        # SUBTRACTED: PP intermediate_tensors 收集与 encoder-decoder 支
        #   （L3666-L3679——delete[10]/encoder 域）。

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4082 _get_slot_mappings ——
    #   slot 双口径装配（站9：padded 前缀切片 + 尾段 fill_(-1) + by-layer dict）
    def _get_slot_mappings(
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
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4106-L4111
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4113-L4132 _get_slot_mapping
        #   闭包（尾段 -1 的所在）
        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            # SUBTRACTED: EncoderOnlyAttentionSpec 零向量支（L4115-L4123
            #   ——delete[7]）。
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            blk_table = self.input_batch.block_table[kv_cache_gid]
            slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4128-L4130
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4134-L4137
        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4139-L4143 —— 同组各层
        #   铺成 dict[layer_name]（供 ForwardContext 逐层取用）
        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        # SUBTRACTED: ubatch（DBO）切片分支（L4145-L4152——delete[5]）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4154
        return slot_mappings_by_gid, slot_mappings_by_layer

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4165 execute_model —— 块表
    #   线切面（站8/11：双口径裁决 + 逐层铺设 + 前向 seam）
    @torch.inference_mode()
    def execute_model(
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

        # SUBTRACTED: routed_experts 缓冲清理（L4177-L4178——delete[6]）；
        #   ngram-GPU 的 scheduler_output replace 复制（L4180-L4195——
        #   delete[11]：可变性裁决协议，ch18 域全文已立）；KV connector
        #   handle_preemptions（L4197-L4200——delete[9]）。

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("gpu_model_runner: preprocess"):
            # SUBTRACTED: synchronize_input_prep 防踩上下文（ch18 域 pinned
            #   buffer 协议——L4205-L4206 的第二项）。
            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(scheduler_output)

            # SUBTRACTED: EC connector 消费者早退（L4210-L4216——delete[9]）。

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4218-L4233 空拍早退
            #   （SUBTRACTED：external_launcher+DP dummy-run 边角 L4219-L4230
            #   ——delete[12]；kv_connector_no_forward L4231-L4234——delete[9]）
            if not num_scheduled_tokens:
                # Return empty ModelRunnerOutput if no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            # SUBTRACTED: kv_sharing_fast_prefill 断言（L4236-L4241——
            #   delete[9]）。

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4243-L4248
            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4250-L4253
            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            # SUBTRACTED: cascade attention prefix 预计算（L4255-L4263——
            #   delete[4]：cascade_attn_enabled=False 时不执行）。
            cascade_attn_prefix_lens = None

            # SUBTRACTED: _determine_batch_execution_and_padding（L4265-L4278
            #   ——ch19 CUDA graph 域：BatchDescriptor 查表命中/实际 batch
            #   padding 到捕获形状）。HOST SEAM 承载：ch19 域产出以观测位
            #   直供（seam_cudagraph_mode/seam_batch_desc；num_tokens_across_dp
            #   /cudagraph_stats 恒 None）。
            cudagraph_mode = self.seam_cudagraph_mode
            batch_desc = self.seam_batch_desc
            should_ubatch = False  # delete[5]
            num_tokens_across_dp = None
            cudagraph_stats = None

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4289-L4292
            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            # SUBTRACTED: maybe_create_ubatch_slices（L4293-L4299——delete[5]；
            #   slices 恒 None）。
            ubatch_slices = None
            ubatch_slices_padded = None

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4307-L4318 ——
            #   padded/unpadded 裁决（m7 本体：存在后端 KV 写独立成 op →
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

            # SUBTRACTED: mamba align 预处理块（L4320-L4362——delete[3]：
            #   preprocess_mamba/mamba_bufs/deferred corrections 应用位）。

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4364-L4365
            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4367-L4376 —— 裁决
            #   落地（三元选择 padded/unpadded 交给 _get_slot_mappings）
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

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4378-L4392
            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4394-L4403
            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4405-L4411
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4413-L4418
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4420-L4424
        defer_kv_connector_finalize = self.speculative_config is not None
        # Update the EPLB meta.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4425-L4431（eplb_state
        #   恒 None 不进；判别位保留）
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(
                self.model_config,
                num_tokens_unpadded,
                ubatch_slices_padded,
            )
        # SUBTRACTED: maybe_get_kv_connector_output 上下文（L4445-L4448——
        #   delete[9]：产出变量 kv_connector_output 以 None 占位）。
        kv_connector_output = None
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4432-L4444 —— 逐层铺设
        #   （站11：slot_mapping by-layer dict 进 ForwardContext）
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
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
        ):
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4450-L4456 前向调用
            #   位（ENGINE SEAM：_model_forward 是 ch17 域的编译包装器——
            #   本章以脚本化单 Attention 层前向承载，见方法体注）
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4459-L4465
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            # SUBTRACTED: logits 计算与 PP 广播段（L4467-L4514——采样域
            #   ch18：hidden_states[logits_indices]/compute_logits/pooling/
            #   broadcast_tensor_dict；delete[10] 的 PP 两臂在其内）——本章
            #   logits 恒 None（两段式契约的后半归 ch18 sample_tokens）。
            logits = None
            sample_hidden_states = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4516-L4527 ExecuteModelState
        #   打包（slot_mappings 随行带过——m12）
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4530-L4535
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        return None

    # ENGINE SEAM（ch17 模型域，ch18 同款承载）：真实 _model_forward 走编译
    #   包装器调模型 forward；本章以脚本化单 Attention 层前向承载——每层先
    #   unified_kv_cache_update（写腿，attention.py 真身）再 unified_
    #   attention_with_output（读腿，吃同一 dummy 数据依赖）——真实
    #   Attention.forward 的两算子调用序（attention.py:L775-L846）。
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4450 调用位
    def _model_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None,
        inputs_embeds: torch.Tensor | None,
        **model_kwargs,
    ):
        num_tokens = positions.shape[0]
        for layer_name, layer in (
            self.vllm_config.compilation_config.static_forward_context.items()
        ):
            impl = layer.impl
            q = torch.zeros(
                num_tokens, impl.num_heads, impl.head_size,
                dtype=torch.float32, device=self.device)
            k = torch.zeros(
                num_tokens, impl.num_kv_heads, impl.head_size,
                dtype=torch.float32, device=self.device)
            v = torch.zeros_like(k)
            out = torch.zeros(
                num_tokens, impl.num_heads * impl.head_size, device=self.device)
            dummy = unified_kv_cache_update(k, v, layer_name)
            unified_attention_with_output(
                q, k, v, out, layer_name, kv_cache_dummy_dep=dummy)
            self.seam_model_output[layer_name] = out
            # HOST SEAM 观测位：前向内按 layer_name 取到的 slot_mapping
            fc = get_forward_context()
            self.seam_seen_slot_mapping = fc.slot_mapping.get(layer_name)
            self.seam_seen_layer_name = layer_name
        return None  # hidden_states 占位（postprocess/logits 域 → ch18）

    # SUBTRACTED: sample_tokens / _bookkeeping_sync / _dummy_run 等（L4552 起
    #   ——两段式契约后半与异步输出包裹 → ch12/ch18 域）。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7240 may_reinitialize_input_
    #   batch —— 模式装配（m10：块尺寸/模式与初始化占位不符时连同块表重建）
    def may_reinitialize_input_batch(
        self, kv_cache_config: Any, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        what it was originally created with. This happens when the final
        block size (determined after model loading) differs from the
        placeholder used during __init__, or when there are multiple
        KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7254-L7272
        block_sizes = []
        max_num_blocks = []
        slot_mapping_modes = []
        # HOST SEAM：max_encoder_len 装配位（encoder 域 delete[7]——本章恒 0）
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            kv_cache_spec_kind = get_kv_cache_spec_kind(kv_cache_spec)
            # SUBTRACTED: ENCODER_ONLY 组 continue（L7261-L7262——delete[7]：
            #   encoder-only 模型无 paged KV，本章组面不进）。
            block_size = kv_cache_spec.block_size
            block_sizes.append(block_size)
            if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:
                slot_mapping_modes.append(SlotMappingMode.NONE)
            else:
                slot_mapping_modes.append(SlotMappingMode.TOKEN_TO_KV_SLOT)
            max_num_blocks_per_req = kv_cache_spec.max_num_blocks_per_req(
                self.vllm_config, max_model_len
            )
            max_num_blocks.append(max_num_blocks_per_req)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7274-L7301 —— 不符即
        #   连块表重建 InputBatch（delete[14]：采样参数列等不进本章承载）
        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
            or max_num_blocks != self._init_max_num_blocks
            or slot_mapping_modes != self._init_slot_mapping_modes
        ):
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
            self._init_max_num_blocks = max_num_blocks
            self._init_slot_mapping_modes = slot_mapping_modes
            self.input_batch = InputBatchSeam(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                num_spec_tokens=self.num_spec_tokens,
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=(
                    self.input_batch.logitsprocs_need_output_token_ids
                ),
                is_pooling_model=False,
                cp_kv_cache_interleave_size=(
                    self.parallel_config.cp_kv_cache_interleave_size
                ),
                reasoning_config=self.vllm_config.reasoning_config,
                use_replayssm=self.cache_config.use_replayssm,
                slot_mapping_modes=slot_mapping_modes,
            )
