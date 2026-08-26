# SOURCE: vllm/v1/worker/gpu_model_runner.py
# GPUModelRunner —— 本章 worker 侧切面（执行臂的影子状态）：
#   * AsyncGPUModelRunnerOutput（L259-L350）：构造即在专用 copy stream 发起
#     non_blocking D2H + event.record（不等待）；get_output 才 event.synchronize
#     ——EngineCore 的阻塞点缩成一次拷贝事件（m12）；
#   * _prepare_input_ids（L1784-L1891）：正常拍直拷 / async 拍 common-case 单
#     slice 直拷 / 变过按 index scatter——真 token 全程不落 CPU（m11）；
#   * _compute_prev_positions（L1769-L1782）：cur → prev 槽位映射（-1=新请求）；
#   * 乐观 seq_lens + discard_request_mask（L2081-L2105，m18）；
#   * _bookkeeping_sync（L3723-L3862）：async 分支留 GPU + token_ids_cpu 写 -1
#     占位（m10）；sync 分支 _to_list 对照保留；
#   * sample_tokens 的 prev 缓存清点（L4604-L4609）+ AsyncGPUModelRunnerOutput
#     包裹位（L4782-L4840）；
#   * synchronize_input_prep（L3864-L3877）：pinned buffer 防踩协议。
# 前向本体/compute_logits/cudagraph 是 ENGINE SEAM（ch17：测试脚本化 logits 行，
# 不在环内伪造 forward）。设备在 HOST 侧是 CPU 张量（无 CUDA），stream/event 以
# HOST SEAM 承载同一契约面。
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
import torch

from .gpu_input_batch import CpuGpuBuffer, InputBatch, PIN_MEMORY
from .outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from .output import GrammarOutput, SchedulerOutput


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L276 torch.cuda.Event(blocking=True)
# —— HOST SEAM：CPU host 无 CUDA 事件；threading.Event 站同一契约位（record=入队
# 未完成、synchronize=阻塞至完成、set=HOST 侧模拟 DMA 完成）。
class HostEvent:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L274-L276 — HOST SEAM
    def __init__(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L274-L276 — HOST SEAM
        self._ev = threading.Event()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L306 event.record — HOST SEAM
    def record(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L306 — HOST SEAM（拷贝已
        # 入队，事件未完成——完成的时刻由拷贝流决定）
        pass

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L314 event.synchronize — HOST SEAM
    def synchronize(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L314 — HOST SEAM（阻塞至完成）
        self._ev.wait()

    # HOST SEAM test hook：模拟 D2H DMA 完成（真实由 copy stream 硬件推进）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L306 event.record — HOST SEAM
    def set(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L306 — HOST SEAM
        self._ev.set()

    # HOST SEAM：事件是否已完成
    # SOURCE: vllm/v1/worker/gpu_model_runner.py event 查询面 — HOST SEAM
    def is_set(self) -> bool:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py event 查询面 — HOST SEAM
        return self._ev.is_set()


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L735-L743 async_output_copy_stream
# —— HOST SEAM：CPU host 无 CUDA stream；以轻量对象承载 wait_stream/上下文面。
class HostCopyStream:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L740 — HOST SEAM
    def __init__(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L740 — HOST SEAM
        pass

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L289 wait_stream — HOST SEAM
    def wait_stream(self, other):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L289 — HOST SEAM
        pass


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L288 with torch.cuda.stream(...) —
# HOST SEAM：上下文管理器站 stream 上下文位
@contextmanager
def _host_stream_ctx(_stream):
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L288 — HOST SEAM
    yield


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L287 torch.cuda.current_stream()
# —— HOST SEAM
def _current_stream():
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L287 — HOST SEAM
    return HostCopyStream()


# Wrapper for ModelRunnerOutput to support overlapped execution.
# SOURCE: vllm/v1/worker/gpu_model_runner.py:L259 AsyncGPUModelRunnerOutput（m12）
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L260-L270 __init__ 签名
    # SUBTRACTED: routed_experts/check_ep_fault 参数与快照面（L268-L269——MoE
    #   诊断，dossier.delete 第 7 条批准）。
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: Any | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: Any | None,
        vocab_size: int = 0,
    ):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L271-L272
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L274-L276
        self.async_copy_ready_event = HostEvent()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L278-L284
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        # SUBTRACTED: _routed_experts/_has_fault（L283-L284——第 7 条）。

        # Initiate the copy on a separate stream, but do not synchronize it.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L286-L306 构造即发起拷贝
        default_stream = _current_stream()
        if async_output_copy_stream is None:
            async_output_copy_stream = HostCopyStream()
        with _host_stream_ctx(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            # HOST SEAM（CPU）：.to("cpu") 对 CPU 张量返回自身——拷贝语义由
            # 快照承载（真实：copy stream 上的 non_blocking D2H）。
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            ).clone()
            # SUBTRACTED: logprobs_tensors/routed_experts/ep-fault 的随行拷贝
            #   （L293-L305——第 6/7 条批准）。
            self.async_copy_ready_event.record()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L308 get_output（m12 契约）
    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L313-L314 event.synchronize
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L316-L318
        del self._logprobs_tensors
        del self._sampled_token_ids
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L319-L332（spec 输出解析的
        # else 分支随第 6 条删——本章 max_gen_len 恒 1）
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L334-L335
        output.sampled_token_ids = valid_sampled_token_ids
        # SUBTRACTED: logprobs_lists/routed_experts/ep-fault 回填与抛错
        #   （L336-L348——第 6/7 条）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L350
        return output


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L437-L450 ExecuteModelState（两段式
# 契约的暂存态——ch9 立过；深水字段随 ch17 删，保结构位）
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L441-L450
    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: Any | None
    spec_decode_common_attn_metadata: Any | None
    hidden_states: Any | None
    sample_hidden_states: Any | None
    aux_hidden_states: Any | None
    ec_connector_output: Any | None
    cudagraph_stats: Any | None
    slot_mappings: Any | None


# SOURCE: vllm/v1/sample/sampler.py:L239-L241 greedy_sample（逐字——temperature=0
# 分支；完整采样栈归 ch08）
class Sampler:
    # SOURCE: vllm/v1/sample/sampler.py:L239-L241 greedy_sample (逐字)
    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/sampler.py:L239-L241
        return torch.argmax(logits, dim=-1, keepdim=True)


# SOURCE: vllm/v1/worker/gpu_output.py SamplerOutput（消费面：sampled_token_ids
# 张量 + logprobs_tensors 位）
class SamplerOutput(NamedTuple):
    # SOURCE: vllm/v1/worker/gpu_output.py SamplerOutput 字段面
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Any | None


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L453 GPUModelRunner —— 本章切面
class GPUModelRunner:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456 __init__（装配切面）
    def __init__(
        self,
        vllm_config: Any,
        max_num_reqs: int = 256,
        max_model_len: int = 4096,
        vocab_size: int = 16,
        max_num_tokens: int = 4096,
    ):
        # SUBTRACTED: 模型/采样器/attention 后端/cudagraph/connector/mamba
        #   装配（L456-L734——ch17 执行域）。
        self.vllm_config = vllm_config
        self.use_async_scheduling = (
            vllm_config.scheduler_config.async_scheduling
            if vllm_config is not None
            else False
        )
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs
        self.device = torch.device("cpu")  # HOST SEAM：CPU 张量代 GPU
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L764 input_ids 持久缓冲
        self.input_ids = CpuGpuBuffer(max_num_tokens, dtype=torch.int32)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L774-L776 optimistic_seq_lens
        # _cpu（m18：假定 draft 全接受的乐观长度）
        self.optimistic_seq_lens_cpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        # SUBTRACTED: num_computed_tokens GPU 张量（L777-L779——乐观值经 CPU
        #   镜像承载）。
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        # Maps current batch position -> previous batch position (-1 for new reqs)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L784-L785 prev_positions
        self.prev_positions = CpuGpuBuffer(max_num_reqs, dtype=torch.int64)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L802-L804 discard_request_mask
        self.discard_request_mask = CpuGpuBuffer(max_num_reqs, dtype=torch.bool)
        # SUBTRACTED: query_start_loc/seq_lens/positions/mrope/encoder 等注意力
        #   元数据缓冲（L765-L833——ch17/ch22）。
        # spec 面：恒无 spec（dossier.delete 第 6 条——深水归 ch33）。
        self.prev_num_spec_tokens = 0

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L735 async_output_copy_stream
        self.async_output_copy_stream: Any | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L738 prepare_inputs_event
        self.prepare_inputs_event: Any | None = None
        if self.use_async_scheduling:
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L740
            self.async_output_copy_stream = HostCopyStream()  # HOST SEAM
            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
            # under TP contention that spin can balloon and make the rank a straggler.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L741-L743
            self.prepare_inputs_event = HostEvent()  # HOST SEAM

        # SOURCE: vllm/v1/worker/gpu_model_runner.py execute_model_state 暂存位
        # （两段式契约——ch9 立过）
        self.execute_model_state: ExecuteModelState | None = None
        # SOURCE: vllm/v1/worker/gpu_model_runner.py 持久 InputBatch（ch18 边界）
        self.input_batch = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            vocab_size=vocab_size,
        )
        # SOURCE: vllm/v1/worker/gpu_model_runner.py requests 簿记（L2097 消费位）
        self.requests: dict[str, Any] = {}

        # ENGINE SEAM（ch17 边界）：脚本化 forward——每步一个 {req_id: logits 行}
        # 字典；真实前向在 GPU 上算，HOST 侧由测试经 enqueue_logits 脚本化。
        self._scripted_logits: deque = deque()
        self._script_cond = threading.Condition()
        self._pending_async: list[AsyncGPUModelRunnerOutput] = []
        self.trace: list[tuple[int, str]] = []  # ENGINE SEAM observation

    # ------------------------------------------------------------------ #
    # ENGINE SEAM test hooks（ch17 边界）
    # ------------------------------------------------------------------ #
    # ENGINE SEAM test hook：脚本化 logits 行注入
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 模型前向 — ENGINE SEAM 注入位
    def enqueue_logits(self, steps: list) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 — ENGINE SEAM
        with self._script_cond:
            for step in steps:
                self._scripted_logits.append(
                    {rid: list(row) for rid, row in dict(step).items()}
                )
            self._script_cond.notify_all()

    # ENGINE SEAM test hook：完成 D2H（HOST 侧模拟 DMA 落地——真实由 copy stream
    # 硬件推进，EngineCore 的 future.result() 等的就是这个事件）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L306 event.record — ENGINE SEAM 完成位
    def release_async_copies(self) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L306 — ENGINE SEAM
        for out in self._pending_async:
            out.async_copy_ready_event.set()
        self._pending_async.clear()

    # ------------------------------------------------------------------ #
    # 乐观纠错群（m18）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2081-L2090 optimistic_seq_lens
    # （ENGINE SEAM：从 _prepare_inputs 内联块抽出以便单测，控制流逐字）
    def _compute_optimistic_seq_lens(self, num_scheduled_tokens: np.ndarray) -> None:
        # Compute optimistic seq_lens (assumes all draft tokens from previous
        # iteration accepted). Store in optimistic_seq_lens_cpu for use by
        # _build_attention_metadata (max_seq_len) and discard_request_mask.
        # seq_lens (GPU) will be computed later using the same optimistic values.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2085-L2090
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[: self.input_batch.num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[: self.input_batch.num_reqs],
        )
        self.optimistic_seq_lens_cpu[self.input_batch.num_reqs :].fill_(0)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2100-L2105 discard_request_mask
    # （ENGINE SEAM：同上抽出，控制流逐字）
    def _compute_discard_request_mask(self, num_reqs: int) -> None:
        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2102-L2105
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        self.discard_request_mask.np[:num_reqs] = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)

    # ------------------------------------------------------------------ #
    # 下一拍 GPU 回填（m11）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1769 _compute_prev_positions
    def _compute_prev_positions(self, num_reqs: int) -> None:
        """Build prev_positions mapping: current pos -> previous pos (-1 if new).

        Populates self.prev_positions.np[:num_reqs] with the mapping.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1774-L1782
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        prev_positions = self.prev_positions.np[:num_reqs]

        if not prev_req_id_to_index:
            prev_positions.fill(-1)
            return

        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            prev_positions[i] = prev_req_id_to_index.get(req_id, -1)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1784 _prepare_input_ids（m11）
    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids.

        Uses self.prev_positions[:num_reqs] which maps current pos -> prev pos
        (-1 for new requests).
        """

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1801-L1807 正常拍
        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            # SUBTRACTED: enable_prompt_embeds 的 inputs_embeds/is_token_ids
            #   补拷（L1804-L1806——多模态，dossier.delete 第 7 条批准）。
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1809-L1820 索引细算
        # （spec draft 索引的深挖归 ch33——draft_len 本章恒 0，形状只剩
        # 『采样 token 一列』）
        prev_positions = self.prev_positions.np[:num_reqs]
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        prev_indices: list[int] = []
        common_indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1822-L1849 逐请求映射
        for cur_index in range(num_reqs):
            prev_index = prev_positions[cur_index]
            if prev_index < 0:
                continue
            prev_indices.append(prev_index)
            req_id = self.input_batch.req_ids[cur_index]
            # We need to compute the flattened input_ids index of the
            # last token in each common request.
            draft_len = len(scheduled_spec_tokens.get(req_id, ()))
            total_num_spec_tokens += draft_len
            flattened_index = cu_num_tokens[cur_index].item() - 1
            # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
            # sample_flattened_indices = [0, 2, 5]
            # spec_flattened_indices = [1,   3, 4,    6, 7]
            sample_flattened_indices.append(flattened_index - draft_len)
            spec_flattened_indices.extend(
                range(flattened_index - draft_len + 1, flattened_index + 1)
            )
            start = prev_index * self.prev_num_spec_tokens
            # prev_draft_token_indices is used to find which draft_tokens_id
            # should be copied to input_ids
            # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
            # flatten draft_tokens_id [1,2,3,4,5,6]
            # draft_len of each request [1, 2, 1]
            # then prev_draft_token_indices is [0,   2, 3,   4]
            prev_draft_token_indices.extend(range(start, start + draft_len))
            common_indices_match &= prev_index == flattened_index
            max_flattened_index = max(max_flattened_index, flattened_index)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1851-L1867 三岔口
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        # SUBTRACTED: enable_prompt_embeds 的 is_token_ids 补拷（L1853-L1857
        #   ——第 7 条）。
        if num_common_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # we need to copy the input_ids_cpu to the GPU first.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1858-L1863
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1864-L1867
            return
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1868-L1877 common-case
        # 优化：批次未变未重排 → 单 slice 直拷
        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1878-L1891 scatter 兜底
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # SUBTRACTED: spec draft 的二段 scatter（L1893-L1913——dossier.delete
        #   第 6 条批准：_draft_token_ids 恒 None、spec_flattened_indices 恒空，
        #   深挖归 ch33）。

    # ------------------------------------------------------------------ #
    # 两段式契约（ch9 立过；本章保 async 接缝）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4165 execute_model
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Any | None = None,
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | Any | None":
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4171-L4175 State error 防御
        # （逐字——ch9 立过）
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # SUBTRACTED: routed capturer/ngram copy/kv preempt（L4177-L4200——
        #   第 6/7 条与 connector）。
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4207-L4208 持久批同步
        # （_update_states——ch17 GPU 面；ENGINE SEAM 镜像：新请求落批/完成清退）
        self._update_states(scheduler_output)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4218-L4233 空批早退
        if not num_scheduled_tokens:
            # Return empty ModelRunnerOutput if no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        # ENGINE SEAM（ch17 边界）：输入准备面（真实 _prepare_inputs 的
        # gather/attn metadata 在 ch17/ch18；本章保 _prepare_input_ids 与乐观
        # 纠错两块真码），随后脚本化前向。
        logits = self._seam_prepare_and_forward(scheduler_output)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4516-L4527 暂存（逐字；
        # 深水字段恒 None——前向本体是 seam）
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            None,  # spec_decode_metadata（第 6 条）
            None,  # spec_decode_common_attn_metadata（第 6 条）
            None,  # hidden_states（ch17）
            None,  # sample_hidden_states（ch17）
            None,  # aux_hidden_states（ch17）
            None,  # ec_connector_output（connector）
            None,  # cudagraph_stats（观测）
            None,  # slot_mappings（ch22）
        )

        # SUBTRACTED: deferred state corrections（L4530-L4533——spec 纠偏回调，
        #   第 6 条批准，深挖归 ch33）。

        return None

    # ENGINE SEAM（ch17/18 边界）：输入准备（保两块真码）+ 脚本化前向。
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1960 _prepare_inputs — SEAM 切面
    def _seam_prepare_and_forward(self, scheduler_output: "SchedulerOutput") -> torch.Tensor:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1960 _prepare_inputs — SEAM
        num_reqs = self.input_batch.num_reqs
        total = scheduler_output.total_num_scheduled_tokens
        # cu_num_tokens（各请求调度窗口的末端累计——真实经 arange 累减计算，
        # ch18；SEAM：直接 cumsum）
        sched_list = [
            scheduler_output.num_scheduled_tokens.get(r, 0)
            for r in self.input_batch.req_ids
        ]
        cu_num_tokens = np.cumsum(sched_list).astype(np.int64)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2072-L2079 调度窗口 gather
        # （真实经 req_indices/positions 收集——ch17/18；SEAM：按行窗口拷贝，
        # 语义=async 下窗口内 token_ids_cpu 的 -1 占位照进 input_ids.cpu）
        offset = 0
        for i, rid in enumerate(self.input_batch.req_ids):
            n = scheduler_output.num_scheduled_tokens.get(rid, 0)
            if n == 0:
                continue
            start = self.input_batch.num_computed_tokens.get(rid, 0)
            end = min(start + n, self.input_batch.max_model_len)
            span = self.input_batch.token_ids_cpu[i, start:end]
            self.input_ids.cpu[offset : offset + n] = span[:n]
            offset += n

        # 乐观 seq_lens + prev_positions + discard mask（真码）
        num_scheduled_np = np.array(sched_list, dtype=np.int32)
        self._compute_optimistic_seq_lens(num_scheduled_np)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2092-L2095
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)
        self._compute_discard_request_mask(num_reqs)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2085-L2090 的算术镜像：
        # 乐观值推进 computed（真实由 GPU 侧 num_computed_tokens 张量承载，
        # HOST 侧写回 InputBatch 记账字典）
        for i, rid in enumerate(self.input_batch.req_ids):
            self.input_batch.num_computed_tokens[rid] = int(
                self.optimistic_seq_lens_cpu[i].item()
            )
            self.input_batch.num_computed_tokens_cpu_tensor[i] = int(
                self.optimistic_seq_lens_cpu[i].item()
            )

        # _prepare_input_ids（真码——GPU 回填主战场）
        self._prepare_input_ids(scheduler_output, num_reqs, total, cu_num_tokens)

        # ENGINE SEAM（ch17）：脚本化前向站位（每请求一行 logits）
        return self._seam_logits()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1192 _update_states — ENGINE
    # SEAM（ch17/18 边界：真实是块表/采样元数据/condense 的重批；本章保
    # 新请求落批/完成清退/恢复刷新三动作）
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1192 — ENGINE SEAM
        for rid in scheduler_output.finished_req_ids:
            if rid in self.input_batch.req_ids:
                self.input_batch.remove_request(rid)
                self.requests.pop(rid, None)
        for new_req in scheduler_output.scheduled_new_reqs:
            self.input_batch.add_request(
                new_req.req_id, list(new_req.prompt_token_ids)
            )
            self.requests[new_req.req_id] = type(
                "CachedRequestState",
                (),
                {
                    "num_tokens": len(new_req.prompt_token_ids),
                    "all_token_ids": list(new_req.prompt_token_ids),
                },
            )()
        # SOURCE: vllm/v1/worker/gpu_model_runner.py _update_states 的
        # CachedRequestData 落位面（真实把调度器侧 computed 写进持久批张量）
        cached = scheduler_output.scheduled_cached_reqs
        for idx, rid in enumerate(cached.req_ids):
            if rid in self.input_batch.num_computed_tokens:
                self.input_batch.num_computed_tokens[rid] = cached.num_computed_tokens[
                    idx
                ]
                row = self.input_batch.req_ids.index(rid)
                self.input_batch.num_computed_tokens_cpu_tensor[row] = (
                    cached.num_computed_tokens[idx]
                )
        # SUBTRACTED: resumed 行刷新/condense/块表同步——ch18 内景；本章测试
        #   无跨 worker 恢复场景。

    # ENGINE SEAM（ch17）：脚本化前向——request-keyed logits 行；有界等待
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 模型前向 — SEAM 站位
    def _seam_logits(self) -> torch.Tensor:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 — SEAM 站位
        step = self._pop_scripted_rows()
        rows = []
        for rid in self.input_batch.req_ids:
            if rid not in step:
                raise RuntimeError(
                    f"no scripted logits row for request {rid!r} "
                    "(ch17 boundary: script each request's row)"
                )
            rows.append(step[rid])
        time.sleep(0.001)  # ENGINE SEAM：一步前向的节拍
        return torch.tensor(rows, dtype=torch.float32)

    # ENGINE SEAM（ch17）：有界等待脚本行
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 前向的 GPU 等待位 — SEAM
    def _pop_scripted_rows(self) -> dict:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 — SEAM
        with self._script_cond:
            deadline = time.monotonic() + 5.0
            while not self._scripted_logits:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "scripted forward ran dry (ch17 boundary: the real "
                        "engine waits on the GPU here)"
                    )
                self._script_cond.wait(remaining)
            return self._scripted_logits.popleft()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4553 sample_tokens（本章切面）
    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | Any":
        # SUBTRACTED: execute_model_state 为 None 的 PP/kv-conn 特例与 PP 广播
        #   （L4556-L4564/L4594-L4602——PP 面，dossier.delete 第 5 条批准：
        #   _pp_broadcast/_pp_receive_prev_sampled_token_ids 随删）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4566-L4580 解包暂存态
        (
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
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4582-L4586 掩码应用调用位
        # （掩码内核归 ch30——SEAM 全 1 位掩码数值不变）
        if grammar_output is not None:
            self._apply_grammar_bitmask_seam(grammar_output, logits)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4588-L4589 采样调用位
        sampler_output = self._sample(logits, spec_decode_metadata)

        # SUBTRACTED: _update_states_after_model_execute（L4591-L4593——GPU 侧
        #   num_computed 纠偏与 spec 回调，第 6 条批准，深挖归 ch33/ch17）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4604-L4609 prev 缓存清点
        # （spec 的 _draft_* 清理随第 6 条删；保 prev_sampled_token_ids=None——
        # 上一拍影子不跨拍泄漏，_bookkeeping_sync 随后缓存新采样）
        self.input_batch.prev_sampled_token_ids = None

        # SUBTRACTED: spec drafter 决策与 propose_draft_token_ids 闭包（L4611-
        #   L4751——第 6 条批准：无 spec 配置 draft 路径不执行）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4716-L4731 bookkeeping 调用位
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            None,  # hidden_states（ch17 前向深水）
            scheduler_output.total_num_scheduled_tokens,
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4766-L4780 ModelRunnerOutput
        # 组装（精简字段）
        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4782-L4792 同步版直返
        if not self.use_async_scheduling:
            return output

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4794-L4829 异步包裹
        # （SUBTRACTED: routed_experts 快照 clone 面（L4797-L4818——第 7 条；
        #   注释原话 'Without clones, the copy stream would read torn data' 归
        #   MoE 章））
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4830-L4838 set_async_
        # sampled_token_ids（penalties 消费的上拍快照）
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        # ENGINE SEAM（HOST）：挂起待完成的 D2H（真实由 copy stream 硬件推进；
        # 测试经 release_async_copies 模拟 DMA 完成）
        self._pending_async.append(async_output)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4840
        return async_output

    # SOURCE: vllm/v1/structured_output/utils.py:L86-L175 apply_grammar_bitmask
    # —— ENGINE SEAM（ch30 边界）：掩码内核与重排归 ch30；本章位掩码恒全 1
    # （=无约束），数值不变——只保调用位与行序语义。
    def _apply_grammar_bitmask_seam(self, grammar_output, logits: torch.Tensor) -> None:
        # SOURCE: vllm/v1/structured_output/utils.py:L86-L175 — ENGINE SEAM
        self.trace.append((time.perf_counter_ns(), "apply_bitmask"))

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3692 _sample 调用位（采样栈
    # 归 ch08；本章测试全 greedy）
    def _sample(
        self, logits: torch.Tensor, spec_decode_metadata: Any | None = None
    ) -> SamplerOutput:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3692-L3721（rejection
        #   sampler 分支随 spec 删——greedy 是 temperature=0 的真实分支）
        self.trace.append((time.perf_counter_ns(), "greedy_sample"))
        sampled = Sampler.greedy_sample(logits)
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)

    # ------------------------------------------------------------------ #
    # 写回持久批次：async 分支留 GPU + -1 占位（m10 绝对核心）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3723 _bookkeeping_sync
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        num_scheduled_tokens: int,
    ) -> tuple[
        dict[str, int],
        Any | None,
        list[list[int]],
        dict[str, Any] | None,
        list[str],
        dict[str, int],
        list[int],
    ]:
        # SUBTRACTED: VLLM_COMPUTE_NANS_IN_LOGITS（L3739-L3741——观测）与
        #   generators 回退（L3747-L3750——seed 面板）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3743-L3746 discard 行定位
        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3752-L3755
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = {
            rid: i for i, rid in enumerate(req_ids_output_copy)
        }

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3757-L3759
        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3762-L3796 同步分支对照
        # （routed D2H 快照随第 7 条删）
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            # SUBTRACTED: spec 的 RejectionSampler.parse_output（L3789-L3796
            #   ——第 6 条批准，归 ch33）。
        else:
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3797-L3800 invalid 行
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3802-L3808 影子①
            # （worker 侧真 token 的 GPU 住所）
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3809-L3813 上拍槽位表
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        # SUBTRACTED: use_pp 的回传分支（dossier.delete 第 5 条批准——PP 面）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3820-L3846 写回循环
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3822-L3825 async 分支：
            # token_ids_cpu 行只写占位 -1
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3832-L3838
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3840-L3842 账本照走
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3844-L3846 CachedRequest
            # State 的输出面（ENGINE SEAM：worker 侧行长同步；调度器侧
            # update_from_output 才是 output_token_ids 的权威增长点）
            req_id = req_ids[req_idx]
            req_state = self.requests.get(req_id)
            if req_state is not None:
                req_state.num_tokens = end_idx
                req_state.all_token_ids = (
                    req_state.all_token_ids[:end_idx]
                    if len(req_state.all_token_ids) > end_idx
                    else req_state.all_token_ids
                    + [-1] * (end_idx - len(req_state.all_token_ids))
                )

        # SUBTRACTED: prompt logprobs 取用（L3848-L3852——ch8）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3854-L3862 返回七元组
        return (
            {},  # num_nans_in_logits（观测面已删）
            logprobs_lists,
            valid_sampled_token_ids,
            None,  # prompt_logprobs_dict（ch8）
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7839 _to_list（同步分支的
    # D2H——真实经 event.synchronize 等 copy stream；HOST SEAM：CPU 直接 tolist）
    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7839-L7850 — HOST SEAM
        return sampled_token_ids.tolist()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3864 synchronize_input_prep
    # （pinned buffer 防踩：等上拍 prepare_inputs 的 CPU 张量用完）
    @contextmanager
    def synchronize_input_prep(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3866-L3868
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3870-L3877
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    # SUBTRACTED: _pp_broadcast_prev_sampled_token_ids/_pp_receive_prev_
    #   sampled_token_ids_to_input_batch（L4842-L4880s——PP 面，第 5 条）、
    #   propose_draft_token_ids/_copy_draft_token_ids_to_cpu/update_num_
    #   computed_tokens_for_batch_change（第 6 条——spec 深水归 ch33）、
    #   routed/mamba/cascade/mm 面（第 7 条）。
