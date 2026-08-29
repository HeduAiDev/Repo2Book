# Subtract-only companion for v3 ch19 — vllm/v1/worker/gpu_model_runner.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`), plus
# 本章切面之外的域段以「SUBTRACTED + 归属章」注记收窄（impl-notes §范围裁剪）。
#
# 本章切面（dossier.code_spine / stations）——执行臂中层的『执行形态』维度：
#   * padding 四件套（m13）：query_start_loc 非递减尾（_prepare_inputs
#     L2072-L2078）/ block_table 尾行 NULL_BLOCK_ID（_build_attention_metadata
#     的 _get_block_table 闭包 L2325-L2341）/ slot_mapping 尾部 -1
#     （_get_slot_mappings L4082-L4154 全文）/ positions 尾部清零
#     （_preprocess L3662-L3664）；
#   * 一拍裁决（station 10/m12 消费侧）：_pad_for_sequence_parallelism +
#     _is_uniform_decode + _determine_batch_execution_and_padding（L3911-L4044
#     全文）+ execute_model 的两个 pinned 段（裁决调用点 L4245-L4292、
#     set_forward_context 注入段 L4432-L4456）+ _model_forward；
#   * 捕获编排（station 9/m15）：capture_model（L6814-L6918，delete[6] 的
#     profiler/encoder/lock_workspace 三段删）+ _warmup_and_capture
#     （L6920-L6966）+ _capture_cudagraphs（L6968-L7018，tqdm/DBO 支删）+
#     _freeze_gc；
#   * 全图 wrapper 挂载（m14 尾）：load_model 尾段（L5435-L5479，breakable/
#     ubatching 分支删）；
#   * 最弱链降级（station 5 上游/m16）：_check_and_update_cudagraph_mode
#     （L7161-L7202，drafter 尾段删）。
#
# SUBTRACTED（章界外域）：模型加载/采样/输出装配（ch17）、持久批次与输入
#   收集主体（ch18——_prepare_inputs/_preprocess 只留 padding 段）、attention
#   metadata 装配（ch20/ch21——_build_attention_metadata 只留 block_table 段）、
#   KV cache 初始化（ch13-16）、LoRA/KV-connector mixin 基类、profiler 观测族。
from __future__ import annotations

import gc
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..._host_seams import (
    NULL_BLOCK_ID,
    EncoderOnlyAttentionSpec,
    coordinate_batch_across_dp,
    envs,
    graph_capture,
    init_logger,
    record_function_or_nullcontext,
    round_up,
)
from ...compilation.counter import compilation_counter
from ...compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from ...compilation.monitor import set_cudagraph_capturing_enabled
from ...config import CUDAGraphMode, CompilationMode
from ...forward_context import BatchDescriptor, set_forward_context
from ..attention.backend import AttentionBackend, AttentionCGSupport
from ..cudagraph_dispatcher import CudagraphDispatcher

# HOST 注记: 章界外类型以 Any 注解面承载（真源类型在 ch10/ch12/ch17/ch33
#   域；`from __future__ import annotations` 下注解永不求值，逐字保留真实名）。
if TYPE_CHECKING:
    SchedulerOutput = Any
    IntermediateTensors = Any
    ModelRunnerOutput = Any
    AsyncModelRunnerOutput = Any
    KVCacheGroupSpec = Any
    KVCacheConfig = Any
    PerLayerAttnMetadata = Any
    CommonAttentionMetadata = Any
    UBatchSlices = Any

logger = init_logger(__name__)


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L453-L458 GPUModelRunner 类头
#   （三个 mixin 基类 LoRA/KV-connector/EC-connector 随其域 SUBTRACTED——
#   ch16/LoRA 域；maybe_remove_all_loras 等基类方法只保留调用点）
class GPUModelRunner:
    def __init__(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456-L478（配置绑定头 L461-L477）
        self,
        vllm_config: Any,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device = device
        # SUBTRACTED: __init__ 主体（L479-L859——持久缓冲块（ch18 m05）、
        #   input_batch、sampler、profiler 装配（ch17/ch20））。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L860-L863 —— uniform
        #   decode 查询长（1+num_spec_tokens）与运行期查表器的构造点
        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)
        # SUBTRACTED: __init__ 尾段（L865-L1100——mm_budget/ubatch/workspace
        #   等装配，多模态/DBO/workspace 域）。

    # SUBTRACTED: _update_states/_prepare_input_ids 等持久批次族（ch18）、
    #   sample_tokens/_bookkeeping_sync（ch17/ch18）、_execute_mm_encoder
    #   （多模态域）。

    def _prepare_inputs(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1960-L2282（切面：L2072-L2078 padding 段）
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, Any]:
        """
        Returns:
            tuple[logits_indices, spec_decode_metadata]
        """
        # SUBTRACTED: 输入收集主体（L1974-L2071——ch18 持久批次域：np.repeat/
        #   cumsum/arange/index_select 扁平收集、embeds/mrope/xdrope 特例、
        #   乐观 seq_lens、GPU positions/seq_lens 上载）。
        # Prepare the attention metadata.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2073-L2078 —— padding
        #   四件套之一：query_start_loc 活跃前缀写 CU 偏移、尾部填非递减
        #   （kernels like FlashAttention requires that）
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        # SUBTRACTED: 乐观 seq_lens/GPU 上载尾段（L2079-L2282——ch12/ch18 域）。

    def _build_attention_metadata(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2284-L2440（切面：L2325-L2341 block_table 段）
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: "UBatchSlices | None" = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple["PerLayerAttnMetadata", "CommonAttentionMetadata | None"]:
        """
        Returns:
            tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # SUBTRACTED: metadata 装配主体（L2303-L2324——ch20/ch21 后端域：
        #   attn_metadata dict 构造、max_seq_len、for_cudagraph_capture 支）。
        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2325-L2341 _get_block_table
        #   闭包 —— padding 四件套之二：block_table 尾行填 NULL_BLOCK_ID
        #   （Block 0 is reserved for padding）
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
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # Fill unused block table entries with NULL_BLOCK_ID (null block)
            # for CUDAGraph padding. Block 0 is reserved for padding.
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
            return blk_table_tensor

        # SUBTRACTED: 尾段（L2343-L2440——metadata builder 装配与 per-layer
        #   消费，ch20/ch21 域；_get_block_table 的调用点在其中）。

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3526-L3532
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _preprocess(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3545-L3760（切面：L3662-L3664 positions 段）
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: "IntermediateTensors | None" = None,
    ) -> None:
        # SUBTRACTED: 输入装配主体（L3546-L3661——ch18/ch20 域：input_ids/
        #   inputs_embeds 选择、mrope/xdrope 特例）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3662-L3664 —— padding
        #   四件套之四：positions 尾部清零（pad 行的 position 归零）
        positions = self.positions[:num_input_tokens]
        if num_input_tokens > num_scheduled_tokens:
            self.positions[num_scheduled_tokens:num_input_tokens].zero_()
        # SUBTRACTED: 尾段（L3666-L3760——encoder 输入/中间张量同步，PP 域）。

    def _model_forward(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3879-L3909
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3903-L3909
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @staticmethod
    def _is_uniform_decode(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3911-L3930
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    def _determine_batch_execution_and_padding(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3932-L4044
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Compute LoRA state for cudagraph dispatch
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3977-L3989 dispatch_cudagraph
        #   闭包 —— 判责核心：cascade/encoder 输入禁 FULL 经 invalid_modes 下传
        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3977-L3989
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4003-L4027 —— DP 各 rank
        #   对 padded 尺寸达成一致后 re-dispatch（cudagraph key 不一致会让
        #   集合通信挂死；全貌归 ch34）
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4029-L4036 —— num_paddings
        #   观测（bs=9 pad 到 16 白算 7 行的账目出口）
        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _get_slot_mappings(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4082-L4154
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
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

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4113-L4132 _get_slot_mapping
        #   闭包 —— padding 四件套之三：slot_mapping 尾部填 -1（KV 写 kernel
        #   跳过 pad 槽；position→物理 slot 的 Triton 数学归 ch22）
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
                blk_table = self.input_batch.block_table[kv_cache_gid]
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

    @torch.inference_mode()
    def execute_model(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4165-L4278（切面：裁决调用点段）
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: "IntermediateTensors | None" = None,
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None":
        # SUBTRACTED: 入口族（L4171-L4244——两段式状态断言、ngram 浅拷贝、
        #   KV connector 预处理、_update_states、空批早退：ch12/ch16/ch18 域）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4245-L4247 —— 每拍 token
        #   数组与最大调度长（乐观值，ch12）
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

        # SUBTRACTED: _prepare_inputs 调用与 cascade 前缀长计算（L4250-L4263
        #   ——ch18/ch20 域；cascade_attn_prefix_lens 由 ch20 的 cascade 注意力
        #   装配产出，此处消费其 None/非 None 语义）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4265-L4278 —— 一拍①裁决
        #   调用点：_determine_batch_execution_and_padding 决定 mode 与 padded
        #   BatchDescriptor
        (
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=cascade_attn_prefix_lens is not None,
            num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4280-L4292 —— padded
        #   维度落账（下游全部按 padded 形状取缓冲）
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        # SUBTRACTED: ubatch 切片（L4293-L4299——DBO 扩展态）、attention
        #   metadata 装配与 input 上载（L4301-L4419——ch20/ch21/ch18 域；
        #   attn_metadata/slot_mappings/positions 等由此段产出）。

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4420-L4431 前置注记与
        #   EPLB 前向准备（eplb_state 域随 ch17 删；注记原文保留——ch18 固定
        #   地址与本章回放的接缝）
        # SUBTRACTED: defer_kv_connector_finalize/eplb_state.prepare_forward
        #   （L4424-L4431——ch16/ch34 域）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4432-L4456 —— 一拍④注入：
        #   set_forward_context 把 mode+descriptor 经 thread-local 传给图内的
        #   wrapper（判责在 dispatcher、wrapper 盲信），内层 _model_forward
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
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        # SUBTRACTED: 后处理与返回族（L4458-L4535——hidden_states 解包、PP
        #   中间张量返回、两段式暂存打包：ch17/ch18 域）。

    # SUBTRACTED: sample_tokens/_bookkeeping_sync（L4552-L4840——ch17/ch18
    #   两段式契约后半）。

    def load_model(self, load_dummy_weights: bool = False) -> None:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5303-L5479（切面：尾部包装段）
        """
        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        # SUBTRACTED: 模型加载主体（L5306-L5434——权重加载/量化/初始化，
        #   ch17 域）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5435-L5445 STOCK_TORCH_
        #   COMPILE 支：model.compile 直接编译
        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            from ..._host_seams import _apply_constrain_to_fx_strides_patch

            _apply_constrain_to_fx_strides_patch()
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWrapper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # SUBTRACTED: BreakableCUDAGraphWrapper 分支（L5452-L5462——第三代
        #   方案实验态默认关，m17 路线图脚注）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5463-L5469 —— FULL 图挂
        #   在模型外（PIECEWISE 图挂在编译器内的每片上，两层各认领自己的 mode）
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        # SUBTRACTED: UBatchWrapper 分支（L5470-L5478——DBO 扩展态）与
        #   get_offloader().post_init()（L5480——offloader 域，随 delete[5]）。

    # SUBTRACTED: 其余加载后处理（L5482-L5828——eagle3aux/pooler 等，域外）。

    def _dummy_run(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5830-L5845 —— 签名与模式语义文档
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        randomize_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                  the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                  needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
            profile_seq_lens: If provided, use this value for seq_lens instead
                of max_query_len. Used to profile attention workspace that
                scales with context length.
        """
        # SUBTRACTED: dummy 批次装配与执行本体（L5873-L6098——ch17/ch18 域：
        #   假 SchedulerOutput 构造 → _update_states/_prepare_inputs/execute_
        #   model 的 dummy 驱动）。伴读版只钉 warmup/capture 编排的调用点
        #   （_warmup_and_capture 与 compile_or_warm_up_model）；直接调用
        #   即为越界。
        raise NotImplementedError(
            "dummy batch assembly & forward is ch17/ch18 domain; the ch19 "
            "companion pins the warmup/capture orchestration call sites only"
        )

    # SUBTRACTED: profile_run/determine_available_memory（L6433 起——ch14
    #   显存账本域）。

    @staticmethod
    @contextmanager
    def _freeze_gc():  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6536-L6548
        gc.collect()
        should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
        if should_freeze:
            gc.freeze()
        try:
            yield
        finally:
            if should_freeze:
                gc.unfreeze()
                gc.collect()

    # SUBTRACTED: _maybe_init_encoder_cudagraph_manager（L6638 起——多模态
    #   encoder 图域，delete[6]）。

    def capture_model(self) -> int:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6814-L6918
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        # SUBTRACTED: encoder cudagraph manager 初始化（L6822-L6823——
        #   delete[6]：多模态扩展态）。

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6829-L6832
        set_cudagraph_capturing_enabled(True)

        # SUBTRACTED: torch profiler 装配（L6834-L6869——delete[6]：观测域）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6871-L6893 —— 捕获窗口：
        #   冻 GC + graph_capture 上下文内，按降序形状逐组 _capture_cudagraphs，
        #   图池显存差 = 前后 mem_get_info 差值
        with self._freeze_gc(), graph_capture(device=self.device):
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

            for (
                runtime_mode,
                batch_descs,
            ) in self.cudagraph_dispatcher.get_capture_descs():
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                )
                torch.accelerator.synchronize()

            # SUBTRACTED: encoder cudagraph 捕获段（L6887-L6890——delete[6]）。

            torch.accelerator.synchronize()
            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6895-L6900
        set_cudagraph_capturing_enabled(False)

        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

        # SUBTRACTED: lock_workspace（L6905-L6907——delete[6]：workspace 池
        #   （kernel_warmup 引入）的锁）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6909-L6918
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def _warmup_and_capture(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6920-L6966
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
        profiler: AbstractContextManager[Any] | None = None,
    ) -> None:
        if profiler is None:
            profiler = nullcontext()
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6929-L6951 —— 先热身：
        #   num_warmups 次 eager dummy run（mode=NONE，让 JIT/内存分配稳定）
        for _ in range(num_warmups):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=force_attention,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )
        if num_warmups > 0:
            # Warmups may use auxiliary streams. Ensure all of their work has
            # completed before beginning CUDA graph capture.
            torch.accelerator.synchronize()
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6952-L6966 —— 再进捕获
        #   窗口：is_graph_capturing=True 再跑一次（wrapper 内首遇即捕）
        with (
            profiler,
            torch.profiler.record_function(
                f"capture_{desc.num_tokens}_{cudagraph_runtime_mode.name}"
            ),
        ):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                is_graph_capturing=True,
                profile_seq_lens=profile_seq_lens,
            )

    def _capture_cudagraphs(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6968-L7018
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
        profiler: AbstractContextManager[Any] | None = None,
    ) -> None:
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        uniform_decode = batch_descriptors[0].uniform

        # SUBTRACTED: tqdm 进度条（L6984-L6993——观测域；is_global_first_rank
        #   守卫随进度条删）。

        # We skip EPLB here since we don't want to record dummy metrics
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6995-L7017（DBO
        #   allow_microbatching 阈值裁决随扩展态删——恒 False 直捕）
        for batch_desc in batch_descriptors:
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=False,
                profiler=profiler,
            )
            torch.accelerator.synchronize()
        self.maybe_remove_all_loras(self.lora_config)

    # SUBTRACTED: initialize_attn_backend（L7020 起——ch21 后端装配域）。

    def _check_and_update_cudagraph_mode(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7161-L7202
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list["KVCacheGroupSpec"],
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

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7174-L7186 —— 最弱链：
        #   遍历全部注意力组取最小 CG 支持（一个后端拖累全模型档位）
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
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7187-L7202 —— 降级链
        #   （resolve_cudagraph_mode_and_sizes）+ keys 初始化触发
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
        # SUBTRACTED: drafter 的 dispatcher 初始化尾段（L7204-L7216——
        #   spec decode 域，ch33）。
