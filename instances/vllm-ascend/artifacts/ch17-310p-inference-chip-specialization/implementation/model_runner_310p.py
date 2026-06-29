# vllm_ascend/_310p/model_runner_310p.py —— subtract-only 精简版（ch17 主线之一：ModelRunner 特化）
#
# NPUModelRunner310(NPUModelRunner) 再继承一层（vLLM GPUModelRunner → 昇腾 NPUModelRunner
# +猴补 ch14 → 310），只覆写设备相关路径。本章保留 4 个教学落点：
#   (1) __init__：super 之后换 NPUInputBatch310、把 _acl_format 钉死 FRACTAL_NZ、换 310 sampler；
#   (2) _prepare_inputs：整个 decode 元数据走 CPU NumPy + copy_to_gpu，绕开 Triton slot-mapping；
#   (3) _update_states：CPU 替换 Triton 的连锁后果——在 condense 步手动 stream.synchronize() 补流序；
#   (4) initialize_kv_cache_tensors / _allocate_kv_cache_tensors：拒绝 MLA/Sparse/KV-transfer +
#       FRACTAL_NZ 分配 + block_size*head_size<=128*128 约束。
#
# 全部触 torch_npu/runner 运行时，host 不真跑；与 ACL graph 捕获 / spec-decode / CP/PCP /
# 多模态 / 异步调度正交的大段覆写按 subtraction_plan.delete 折叠。
from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend._310p.block_table import MultiGroupBlockTable as MultiGroupBlockTable310
from vllm_ascend._310p.kv_block_zeroer import AscendKVBlockZeroer310
from vllm_ascend._310p.npu_input_batch import NPUInputBatch310 as NPUInputBatch
from vllm_ascend._310p.sample.rejection_sampler import AscendRejectionSampler310
from vllm_ascend._310p.sample.sampler import AscendSampler310
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

# SUBTRACTED: 仅被已删方法使用的 import（math / get_dtype_size / MambaSpec / contextmanager /
#   nullcontext / nn / CUDAGraphMode / get_pp_group / cdiv / get_total_cp_world_size /
#   prepare_mrope_cos_sin_slices_from_runner / update_num_computed_tokens_for_batch_change /
#   lmhead_tp_enable 等）随对应方法/分支一并折叠。

# SOURCE: vllm_ascend/_310p/model_runner_310p.py:L56-L57
_NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN = 1
_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128


# SOURCE: vllm_ascend/_310p/model_runner_310p.py:L60
class NPUModelRunner310(NPUModelRunner):
    """
    310P model runner with a distinct ACL graph capture/replay contract from 910B:

    - Capture: ACLGraphWrapper records the full forward inside ``torch.npu.graph``.
      310P attention calls NPU ops directly (paged / splitfuse), without mainline
      ``full_graph_fia`` / ``full_graph_pa`` graph_task registration.
    - Replay: refresh shared runner buffers (block_table, seq_lens, query_start_loc,
      slot_mapping via CPU prepare + copy_to_gpu) so tensor addresses stay stable,
      then ``aclgraph.replay()``.
    """

    # Inherited from parent runner; annotated here to satisfy strict type checks.
    uniform_decode_query_len: int
    _mtp_spec_dummy_capture: bool = False

    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L76-L104
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super 之后立刻用 _310p 的 NPUInputBatch310（此处别名为 NPUInputBatch）覆盖 input_batch。
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=self.input_batch.logitsprocs,
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens if self.vllm_config.speculative_config else 0
            ),
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )
        # 把权重/KV 布局格式钉死为 FRACTAL_NZ（310P 始终用 NZ）。
        self._acl_format = ACL_FORMAT_FRACTAL_NZ
        logger.info_once("Weight layout uses FRACTAL_NZ.")
        self.sampler = AscendSampler310()
        if getattr(self, "rejection_sampler", None) is not None:
            self.rejection_sampler = AscendRejectionSampler310(self.sampler)
        if self.speculative_config is not None and self.speculative_config.method == "ngram":
            # 310P ngram requires decode-only graph shapes to be built with q_len=1.
            # Keep dispatcher's internal query_len in sync to avoid key-init assert.
            self.cudagraph_dispatcher.uniform_decode_query_len = _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN
            logger.info_once("Ngram speculative decoding uses uniform_decode_query_len=1 for graph capture.")

    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L106-L117
    def _update_states(self, scheduler_output: SchedulerOutput):
        deferred = super()._update_states(scheduler_output)
        if scheduler_output.finished_req_ids:
            # condense() rewrites block_table.np (move_row). Drain the previous
            # step's ACL graph replay on the NPU stream before the condensed
            # CPU layout is uploaded and read as attn_metadata.block_tables.
            # Main-line Ascend relies on the end-of-_prepare_inputs Triton
            # slot-mapping kernel (reads block_table.gpu) for stream ordering;
            # 310P uses CPU NumPy for slot_mapping and needs this barrier on
            # layout-change steps only.
            torch.npu.current_stream().synchronize()
        return deferred

    # SUBTRACTED: temporary_modify_uniform_decode_query_len / _determine_batch_execution_and_padding /
    #   _build_attention_metadata / _pad_query_start_loc_for_fia（L119-L228）—— ACL graph 捕获与
    #   spec-decode 的图形状细节覆写，与本章四大主线关系弱，按 subtraction_plan.delete 折叠。

    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L229-L239
    def _build_attn_state(self, num_reqs, num_scheduled_tokens, num_valid_tokens):
        attn_state = super()._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)
        if (
            self.speculative_config is not None
            and self.speculative_config.method == "mtp"
            and not np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] == 0)
            and np.all(num_scheduled_tokens == self.uniform_decode_query_len)
        ):
            attn_state = AscendAttentionState.SpecDecoding
            self.attn_state = attn_state
        return attn_state

    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L241-L579
    def _prepare_inputs(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int, list[np.ndarray[Any, Any]] | None]:
        """
        310P cannot use the Triton slot-mapping kernel or the generic NPU Add
        kernels used by the base runner for decode metadata. Keep those pieces
        on CPU and upload the prepared tensors.
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        self.input_batch.block_table.commit_block_table(num_reqs)

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # SUBTRACTED: spec-decode 时按 scheduled_spec_decode_tokens 修正 num_valid_tokens
        #   (L260-L270) —— 单卡非 spec 主路径 num_valid_tokens == num_scheduled_tokens。
        num_valid_tokens = num_scheduled_tokens
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
        self.with_prefill = with_prefill

        # —— 核心：positions 在 CPU 用 np.add 算，再走 NumPy slot_mapping（替换基类 Triton kernel）——
        cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
        positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            self.query_pos.np[: cu_num_tokens[-1]],
            out=positions_np,
        )
        block_table = cast(MultiGroupBlockTable310, self.input_batch.block_table)
        block_table.compute_slot_mapping(
            req_indices,
            positions_np[:total_num_scheduled_tokens],
        )

        # SUBTRACTED: CP/PCP 支路（init_batch_info / generate_pcp_mtp_input / update_tokens_for_pcp，
        #   L289-L325）—— CP（context parallel）正交，单卡 dcp=pcp=1。
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # input_ids 从 token_ids_cpu 按 (positions, req) 索引 gather 到 host 再上传。
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        # SUBTRACTED: enable_prompt_embeds / req_prompt_embeds 多模态嵌入填充支路（L337-L381）。

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()
        # SUBTRACTED: _has_gdn（gated-delta-net）query_start_loc（L387-L391）—— 线性注意力另类 KV。

        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        self._compute_prev_positions(num_reqs)
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
        # SUBTRACTED: uses_mrope / uses_xdrope 多模态位置编码（L406-L417）、discard_request_indices
        #   丢弃请求统计（L419-L436）—— 与"CPU 替换 Triton"主线正交。

        # SUBTRACTED: async-scheduling 的 num_accepted_tokens / prev_positions /
        #   update_num_computed_tokens_for_batch_change 支路（L438-L472, L490-L511）。
        #   非异步默认：num_computed_tokens 直接拷 CPU 值。
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        self.positions[:total_num_scheduled_tokens].copy_(
            self._positions_cpu_buf[:total_num_scheduled_tokens],
            non_blocking=True,
        )
        self.seq_lens[:num_reqs].copy_(
            self.optimistic_seq_lens_cpu[:num_reqs],
            non_blocking=True,
        )
        self.seq_lens[num_reqs:].fill_(0)

        # 单卡非 spec-decode 主路径：logits_indices = 每 req 末 token。
        # SUBTRACTED: use_spec_decode 分支（_calc_spec_decode_metadata，L513-L551）、lmhead_tp/lora
        #   padding（L555-L560）、pcp cache_local_schedule_layout（L562-L572）。
        spec_decode_metadata = None
        logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        self.logits_indices = logits_indices

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
            None,  # num_scheduled_tokens_compressed_list (not used in 310P)
        )

    # SUBTRACTED: _dummy_run / _model_forward / _check_and_update_cudagraph_mode（L581-L658）——
    #   ACL graph 捕获/重放与图模式判定细节，属图模式章节。

    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L659-L668
    def _init_kv_zero_meta(self) -> None:
        """310P uses torch zeroing because Triton is not available."""
        # 装配 310 版 KV zeroer（去 Triton，切片 .zero_()）。
        self._kv_block_zeroer = AscendKVBlockZeroer310(self.device, self.pin_memory)
        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=(self.compilation_config.static_forward_context),
        )

    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L670-L701
    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Override the base class method.
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # 310P 能力边界写成早失败：MLA / Deepseek-Sparse / KV-transfer 直接 raise。
        # 310P limitation: KV transfer is not supported
        if self.vllm_config.kv_transfer_config is not None:
            logger.error("KV cache transfer is not supported.")
            raise ValueError("KV cache transfer is not supported for 310P.")
        if self.use_sparse:
            logger.error("Deepseek Sparse Attention is not supported.")
            raise ValueError("Deepseek Sparse Attention is not supported for 310P.")
        if self.model_config.use_mla:
            logger.error("MLAAttention is not supported.")
            raise ValueError("MLAAttention is not supported for 310P.")
        # Initialize the memory buffer for KV cache
        kv_caches = self._allocate_kv_cache_tensors(kv_cache_config)
        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L703-L791
    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache size. The buffer needs to be reshaped to the desired shape before being used by
        the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer.
        """
        # init kv cache tensors
        kv_cache: dict[str, list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]] = {}
        # get kv cache spec for each layer
        layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
        for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
            for layer_name in group_kv_cache_spec.layer_names:
                layer_kv_cache_spec[layer_name] = group_kv_cache_spec.kv_cache_spec
        # Allocate kv cache buffers according to the kv_cache_config and kv_cache_spec
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if layer_name in self.runner_only_attn_layers:
                    continue
                # SUBTRACTED: linear_attn(Mamba) 分支（L727-L745）—— Mamba/linear attention 是
                #   另一类 KV，与 310P 的 attention KV(FRACTAL_NZ + 128*128 约束)主线正交。
                if "attn" in layer_name and layer_name not in kv_cache:
                    kv_cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(kv_cache_spec, AttentionSpec)
                    assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks
                    # Page attention operation on 310P limits block_size * head_size <= 128 * 128
                    supported_sizes = [
                        support_size
                        for support_size in self.attn_backend.get_supported_kernel_block_sizes()
                        if support_size * kv_cache_spec.head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
                    ]
                    if supported_sizes:
                        block_size = supported_sizes[0]
                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                        )
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size
                        )
                    k_shape = kv_cache_shape[1:]
                    v_shape = k_shape
                    dtype = kv_cache_spec.dtype
                    # 用 FRACTAL_NZ 专用格式分配 (k_cache, v_cache)（而非普通 torch.empty）。
                    k_cache = torch_npu.empty_with_format(
                        size=k_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=v_shape, dtype=dtype, device=self.device, acl_format=self._acl_format
                    )
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache[layer_name_inner] = (k_cache, v_cache)
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache.keys()), "Some layers are not correctly initialized"
        return kv_cache

    # Override this function because of tensor.copy_(other) accuracy issue.
    # TODO: This override will be removed after tensor.copy_(other) accuracy issue is resolved.
    # SOURCE: vllm_ascend/_310p/model_runner_310p.py:L795-L814
    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return
        # SUBTRACTED: async-scheduling 分支（L816-L886）—— prev_sampled_token_ids 从上一步
        #   缓存的 token 在 NPU 上 scatter 回 input_ids 的索引拼装，与异步调度正交；本章只需
        #   "正常调度：input_ids.copy_to_gpu" 主路径。

    # SUBTRACTED: may_reinitialize_input_batch（L888+）—— KV-cache group 变更时重建 input_batch
    #   的善后，与本章主线正交。
