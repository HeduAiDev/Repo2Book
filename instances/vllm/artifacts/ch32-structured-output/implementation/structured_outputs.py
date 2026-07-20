# SOURCE: vllm/v1/worker/gpu/structured_outputs.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。这是本章 m11「两条并存路径」里
# opt-in 的那一条——只有 VLLM_USE_V2_MODEL_RUNNER=1 时 GPUModelRunnerV2 才会构造
# StructuredOutputsWorker、调用这里的 apply_grammar_bitmask，走到自写的 Triton
# kernel；默认部署走的是 utils.py 里那条 xgrammar 库函数路径（见 m11）。与真实
# vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: SPDX 版权头。
# SUBTRACTED: `from vllm.triton_utils import tl, triton` 换成直接 `import triton` /
# `import triton.language as tl`（vLLM 的 triton_utils 只是一层惰性导入包装，效果
# 等价——同一惯例见 ch18 block_table.py）。
# SUBTRACTED: `from vllm.utils.math_utils import cdiv` 换成本文件内的等价实现
# （math_utils.cdiv 本体就是一行整数上取整，没有额外逻辑）。
import numpy as np
import torch

from buffer_utils import async_copy_to_gpu
from input_batch import InputBatch

try:
    import triton
    import triton.language as tl
except ImportError:  # host 未装 triton 时，本文件仍可被 import（不含 GPU 测试）
    triton = None
    tl = None


def cdiv(a: int, b: int) -> int:
    # SOURCE: vllm/utils/math_utils.py cdiv（原样等价：整数上取整除法）
    return -(-a // b)


class StructuredOutputsWorker:
    def __init__(self, max_num_logits: int, vocab_size: int, device: torch.device):
        # SOURCE: vllm/v1/worker/gpu/structured_outputs.py:L12-21
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.cuda.Stream()

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: "list[str]",
        grammar_bitmask: np.ndarray,
    ) -> None:
        # SOURCE: vllm/v1/worker/gpu/structured_outputs.py:L23-80
        if not grammar_req_ids:
            return

        # Asynchronously copy the bitmask to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Construct bitmask -> logits mapping
        mapping: list[int] = []
        req_ids = input_batch.req_ids
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        # Asynchronously copy the mapping to GPU.
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=True
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)


if triton is not None:
    # Adapted from
    # https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
    @triton.jit
    def _apply_grammar_bitmask_kernel(
        logits_ptr,
        logits_stride,
        logits_indices_ptr,
        bitmask_ptr,
        bitmask_stride,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        # SOURCE: vllm/v1/worker/gpu/structured_outputs.py:L86-115
        bitmask_idx = tl.program_id(0)
        logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

        # Load the bitmask.
        block_id = tl.program_id(1)
        bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_bitmask = tl.load(
            bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
            mask=bitmask_offset < bitmask_stride,
        )
        # Unpack the bitmask.
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        # Apply the bitmask to the logits.
        block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            logits_ptr + logits_idx * logits_stride + block_offset,
            -float("inf"),
            mask=bitmask & (block_offset < vocab_size),
        )
