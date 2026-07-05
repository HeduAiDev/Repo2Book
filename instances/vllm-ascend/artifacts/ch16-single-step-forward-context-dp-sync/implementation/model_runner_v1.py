# vllm_ascend/worker/model_runner_v1.py —— subtract-only 精简版（ch15 主角骨架）
#
# 本章主线：NPUModelRunner.execute_model 把一拍前向串起来——
#   _prepare_inputs（attn 元数据前的输入整形）
#   → _determine_batch_execution_and_padding（内含 DP 跨卡同步 _sync_metadata_across_dp）
#   → _build_attention_metadata（建好交给前向，后端实体留 ch18）
#   → set_ascend_forward_context（包基座再注入昇腾上下文）包住 _model_forward
#   → sample_hidden_states / compute_logits → 存 ExecuteModelState
#   → sample_tokens 里 _sample 派发 sampler / rejection_sampler。
#
# DP 同步打包（_sync_metadata_across_dp / _post_process_cudagraph_mode）是纯 Python，可在
# host 跑（真实 dist.all_reduce 由测试以两卡求和模拟）。真实前向算子/采样器不真跑。
import time
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

# SOURCE: vllm_ascend/worker/model_runner_v1.py:L20-L155（节选）
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import (
    enable_sp,
    lmhead_tp_enable,
    should_skip_allreduce_across_dp_group,
    vllm_version_is,
)

# SUBTRACTED: get_global_experts_capturer（vllm 0.21.0 时从
#   vllm.model_executor.layers.fused_moe.routed_experts_capturer 条件导入，
#   model_runner_v1.py:L85-90）、EMPTY_MODEL_RUNNER_OUTPUT（vllm.v1.outputs）、
#   ExecuteModelState（NamedTuple，model_runner_v1.py:L237）等 execute_model 主干引用的符号
#   —— 测试以桩注入；与本章派发骨架/forward-context/DP 同步主线正交。

# SUBTRACTED: 数百行其它 import（attention 后端实体 / spec_decode proposer / mamba_utils /
#   quantization / EC·KV connector / pcp_manager 等，model_runner_v1.py:L20-L155）——
#   与本章「单拍前向派发骨架 + forward context 注入 + DP 同步」正交，按 subtraction_plan 折叠。


# SOURCE: vllm_ascend/worker/model_runner_v1.py:L4867
def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[1, :].min().item())


# SOURCE: vllm_ascend/worker/model_runner_v1.py:L255
class NPUModelRunner(GPUModelRunner):
    # 仅呈现本章主线方法；__init__ 与设备猴补/图捕获接缝见 ch14，字段初始化大段折叠。

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L1904
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: "IntermediateTensors | None" = None,
    ):
        if self.vllm_config.model_config.enable_return_routed_experts:
            # routed-experts capturer：把上一拍挂起的拷贝收尾 / 清缓冲（按版本派发）。
            if vllm_version_is("0.21.0"):
                capturer = get_global_experts_capturer()
                if capturer is not None:
                    capturer.finalize_pending_copy()
            elif self.routed_experts_initialized:
                self.routed_experts_capturer.clear_buffer()

        if self.ascend_config.profiling_chunk_config.need_timing:
            # profiling：记一拍前向的起始时刻（校准完成由 scheduler 跨进程置位）。
            if getattr(scheduler_output, "disable_profiling_timing", False):
                self.ascend_config.profiling_chunk_config.need_timing = False
            else:
                self._sync_device()
                self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        # SUBTRACTED: ngram_gpu 的 scheduler_output.replace 分支（model_runner_v1.py:L1933-1945）
        #   —— 投机解码 ngram 专属优化，不影响主干前向数据流（subtraction_plan.delete）。

        self._start_dump_data()
        # async 调度下 spec 信息缺失时 deepcopy scheduler_output，避免污染 engine-core 进程。
        if self.use_async_scheduling and self.num_spec_tokens and self._draft_token_ids is None:
            # SUBTRACTED: 同处 PCP(Parallel Context Processing)+MM 的 deepcopy OR-分支
            #   （model_runner_v1.py:L1955-1962）—— PCP 为可选特性，本章以 pcp_size==1 直路讲解。
            scheduler_output = deepcopy(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                deferred_state_corrections_fn = self._update_states(scheduler_output)

                # SUBTRACTED: has_ec_transfer/EC connector 生产者早返回 +
                #   kv_sharing_fast_prefill assert（model_runner_v1.py:L1993-2023）——
                #   encoder-cache 传输与特性断言为横切（subtraction_plan.delete）。

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # corner case：external launcher + DP，num_scheduled_tokens 可能为 0，
                        # 先 dummy_run 让 coordinate_batch_across_dp 被调到，避免各卡失步。
                        self._dummy_run(1)
                    if not has_kv_transfer_group():
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                # 1) 输入整形：positions / token_indices / input_ids gather + logits_indices。
                (
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                    num_scheduled_tokens_compressed_list,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
                # SUBTRACTED: pcp_size>1 时改用 pcp_manager.total_num_sampled_tokens_pcp +
                #   cascade attention prefix lens 预计算（model_runner_v1.py:L2042-L2053）——
                #   PCP / cascade 为可选特性（subtraction_plan.delete）。

                # 2) 批执行模式 + padding（内部做 DP 跨卡同步）。
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
                    use_cascade_attn=False,
                    force_eager=self.model_config.enforce_eager,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                num_tokens_padded = batch_desc.num_tokens

                # SUBTRACTED: mamba_cache_mode=='align' 预处理块、use_compress(DSA) 块
                #   （model_runner_v1.py:L2095-2148）—— Mamba/稀疏注意力专属预处理
                #   （subtraction_plan.delete）。

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0

                # 3) 建注意力元数据（后端实体 AscendAttentionBackend/MLA 留 ch18）。
                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    max_query_len=max_num_scheduled_tokens,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                )

            # 4) 预处理：得 input_ids / inputs_embeds / positions / intermediate_tensors。
            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded,
                intermediate_tensors,
            )
            update_cos_sin(positions)

        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # 5) Run forward pass —— set_ascend_forward_context 包住一次前向（本章关键接缝）。
        clear_kv_metadata = self.speculative_config is None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
                skip_compiled=has_encoder_input,
                has_sinks=self._has_sinks,
                input_ids=input_ids,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **({"defer_finalize": not clear_kv_metadata}),
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
        with record_function_or_nullcontext("post process"):
            aux_hidden_states = None
            # SUBTRACTED: use_aux_hidden_state_outputs 拆包 + pcp_manager.get_restore_hidden_states
            #   + broadcast_pp_output 罕见分支（model_runner_v1.py:L2272-2312）—— PCP/PP 横切。
            if not get_pp_group().is_last_rank:
                # PP 非末 rank：返回中间张量。
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                self.kv_connector_output = kv_connector_output
                return hidden_states
            if self.is_pooling_model:
                output = self._pool(hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output)
                output.kv_connector_output = kv_connector_output
                return output

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)

            # 存一拍执行态，execute_model 返回 None；采样在 sample_tokens 里做。
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
                batch_desc,
            )
            self.kv_connector_output = kv_connector_output

        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()
        return None

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L748
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ):
        """
        :return: tuple[logits_indices, spec_decode_metadata, total_num_scheduled_tokens, ...]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first so it overlaps CPU ops.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # SUBTRACTED: 输入整形主体（model_runner_v1.py:L760-L1272）—— req_indices=np.repeat、
        #   positions_np = num_computed_tokens + query_pos、token_indices = positions +
        #   req*max_len、torch.index_select 取 input_ids、mrope / PCP / use_compress 分支。
        #   这些纯索引算术与本章主线（产出 input_ids/positions/logits_indices 三件套喂给后续
        #   attn 元数据与前向）正交，正文讲「输入整形」概念即可（subtraction_plan：_prepare_inputs 主体）。
        #   该段会就地填好 self.query_start_loc 等缓冲，下面 logits_indices 据此切出。
        num_scheduled_tokens_compressed_list = None

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # 无 spec：每请求最后一个 token 位 = query_start_loc-1。
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
            # SUBTRACTED: use_cp(PCP) 时改走 pcp_manager.get_logits_indices（L1280-1282）。
            logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            # SUBTRACTED: 有 spec 分支（model_runner_v1.py:L1288-L1325）—— logits_indices 来自
            #   self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens, ...)，其入参由上面
            #   被折叠的整形主体产出；投机解码细节非本章主线，整段一并折叠，dense/MoE 标准前向取上一分支。
            raise NotImplementedError("spec-decode logits_indices path is out of scope for ch15.")
        self.logits_indices = logits_indices

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
            num_scheduled_tokens_compressed_list,
        )

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L2846
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ):
        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        is_all_decode = np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
        uniform_decode = (
            (
                (is_all_decode if self.speculative_config else True)
                and (max_num_scheduled_tokens == self.uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
        has_encoder_output = self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        # ruff: noqa: E731
        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):  # SOURCE: vllm_ascend/worker/model_runner_v1.py:L2884
            if force_eager:
                return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                valid_modes=valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
                num_active_loras=num_active_loras,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(num_tokens_padded, use_cascade_attn or has_encoder_output)
        num_tokens_padded = batch_descriptor.num_tokens

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            _, num_tokens_across_dp, synced_cudagraph_mode = self._sync_metadata_across_dp(
                num_tokens=num_tokens_padded,
                cudagraph_mode=cudagraph_mode,
                allow_dp_padding=(cudagraph_mode != CUDAGraphMode.NONE) or enable_sp(self.vllm_config),
            )

            # Extract DP padding if there is any
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so every rank agrees on the graph mode.
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={synced_cudagraph_mode},
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded
        cudagraph_stats = None
        # SUBTRACTED: observability_config.cudagraph_metrics 时填 CUDAGraphStat
        #   （model_runner_v1.py:L2925-2932）—— 指标统计，删后控制流不变。

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L627
    def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> "tuple[int, torch.Tensor | None, CUDAGraphMode]":
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        if self.dp_size == 1:
            return num_tokens, None, cudagraph_mode

        if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, cudagraph_mode

        # On certain devices, CPU-side all_reduce may return dirty data.
        # When dp_allreduce_on_npu is True, route DP metadata
        # synchronization through the NPU device group to avoid data corruption.
        device_str, group = (
            ("npu", get_dp_group().device_group)
            if self.ascend_config.dp_allreduce_on_npu
            else ("cpu", get_dp_group().cpu_group)
        )
        packed_tensor = torch.zeros(2, self.dp_size, device=device_str, dtype=torch.int32)
        packed_tensor[0][self.dp_rank] = num_tokens
        packed_tensor[1][self.dp_rank] = cudagraph_mode.value
        dist.all_reduce(packed_tensor, group=group)
        if device_str == "npu":
            packed_tensor = packed_tensor.cpu()

        # Unpack the results
        num_tokens_across_dp = packed_tensor[0, :]
        max_tokens_across_dp = int(num_tokens_across_dp.max().item())
        synced_cudagraph_mode = CUDAGraphMode(_post_process_cudagraph_mode(packed_tensor))

        # Create a tensor for num_tokens_after_padding
        if allow_dp_padding or is_draft_model:
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32
            )
        else:
            num_tokens_after_padding = num_tokens_across_dp.cpu()

        return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L2942
    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: Any = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list | None = None,
        num_scheduled_tokens_compressed_list: list | None = None,
    ):
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        attn_metadata: dict = {}
        spec_decode_common_attn_metadata = None
        # SUBTRACTED: 逐 kv_cache_group / 逐后端的 metadata builder 主体
        #   （model_runner_v1.py:L2970-L3083+）—— 取 max_seq_len、PCP block_table、
        #   build CommonAttentionMetadata、调各后端 AttentionBackend.build()、cascade/spec
        #   分支。注意力后端实体 AscendAttentionBackend/MLA 留 ch18，本章只需「建好 attn_metadata
        #   交给前向」这一接口语义，这里返回填好的 per-layer 字典（subtraction_plan.delete）。
        return attn_metadata, spec_decode_common_attn_metadata

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L2756
    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        assert self.model is not None
        forward_context = get_forward_context()
        assert forward_context is not None

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
            **model_kwargs,
        }
        run_model = partial(self.model, **model_inputs)

        if self.enable_enpu:
            # The soft segmentation scenario requires event.record first, then event.wait
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded, positions)
            hidden_states = run_model()
        else:
            hidden_states = run_model()
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded, positions)

        # flashcomm v1(SP)：对 hidden_states all_gather 回收（SP 切分的逆操作）。
        if forward_context.flash_comm_v1_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        return hidden_states

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L2553
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        self.input_batch.update_async_output_token_ids()
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[: self.input_batch.num_reqs]
            if self.input_batch.sampling_metadata.top_k is not None and get_ascend_config().enable_reduce_sample:
                max_topk = self.input_batch.top_k_cpu[self.input_batch.top_k_cpu < logits.shape[1]].max()
                self.sampler.prepare_sampling(max_topk)
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[: len(spec_decode_metadata.logits_indices)]
        if self.input_batch.sampling_metadata.top_k is not None and get_ascend_config().enable_reduce_sample:
            max_topk = self.input_batch.top_k_cpu[self.input_batch.top_k_cpu < logits.shape[1]].max()
            self.rejection_sampler.prepare_sampling(max_topk)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output
