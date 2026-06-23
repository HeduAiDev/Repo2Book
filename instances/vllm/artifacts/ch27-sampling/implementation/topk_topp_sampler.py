# 只做减法的忠实精简版 —— 镜像 vllm/v1/sample/ops/topk_topp_sampler.py（pin f3fef123）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: SPDX 版权头 + `from vllm import envs` / rocm_aiter_ops / FlashInferBackend /
# init_logger 等平台依赖 import —— 它们只服务被减掉的 flashinfer/aiter/logger 分支。
import torch
import torch.nn as nn


# SUBTRACTED: `from vllm.triton_utils import HAS_TRITON` 与 triton kernel 的条件 import。
# 真实代码在 HAS_TRITON 时从 topk_topp_triton 导入 apply_top_k_top_p_triton；
# 本精简版把 920 行 Triton 内核减成一个 wrapper 桩（见文件末尾），HAS_TRITON 置 False
# 使 apply_top_k_top_p 走可运行的 pytorch sort 主路径（与真实 batch<8 路径同构）。
HAS_TRITON = False


class TopKTopPSampler(nn.Module):
    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L22-104
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self, logprobs_mode: str = "raw_logprobs") -> None:
        # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L30-104
        super().__init__()
        self.logprobs_mode = logprobs_mode
        # flashinfer optimization does not apply if intermediate
        # logprobs/logits after top_k/top_p need to be returned
        #
        # SUBTRACTED: CUDA+flashinfer / CPU-arch / ROCm-aiter 三大分支的具体绑定与
        # logger.info/warning_once 措辞（原 L33-102）。真实代码按 device/platform/
        # logprobs_mode 把 self.forward 绑成 forward_cuda/forward_native/forward_cpu/
        # forward_hip 之一；多级回退（flashinfer 硬件不支持→静默 native、CPU 特例、
        # ROCm aiter）保证启动不崩。host 无 CUDA → 落到最后的 else 分支：forward_native。
        self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L106-125
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L127-152
        """More optimized implementation for top-k and top-p sampling."""
        # Fall back to the PyTorch-native path when FlashInfer has nothing
        # to do (no top-k / top-p filter) or when per-request generators
        # are present (unsupported by FlashInfer 0.2.3+).
        if (k is None and p is None) or generators:
            # SUBTRACTED: logger.debug_once 关于 generator 回退的提示（L139-144）。
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "FlashInfer does not support returning logits/logprobs"
        )
        # flashinfer sampling functions expect contiguous logits.
        return flashinfer_sample(logits.contiguous(), k, p, generators), None

    # SUBTRACTED: forward_cpu / forward_hip / aiter_sample（topk_topp_sampler.py:L154-244）
    # —— CPU 专路（exponential_ 逐 generator + compiled_random_sample）与 ROCm/aiter 后端。
    # 它们与 CUDA 主路径算法同构；保留 forward_native/forward_cuda 两个代表即可讲清多后端分发。


# SUBTRACTED: compiled_random_sample（topk_topp_sampler.py:L249-254）—— forward_cpu 专用的
# torch.compile 包装，针对 pytorch#151218 的 workaround。


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L257-267
    if p is None and k is None:
        return logits

    if HAS_TRITON and logits.shape[0] >= 8:
        # SOURCE: 二级分流——大 batch 且有 Triton 时走 Qrita pivot-truncation 内核（不排序整 vocab）。
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L270-311
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # SUBTRACTED: apply_top_k_only 免排序 top-k-only 快路（L289 + L314-334）——
            # 仅 allow_cpu_sync 且 p is None 时走，含 GPU->CPU sync 的特化优化；
            # 主路径 sort 已覆盖 top-k 语义。这里直接落到下方 sort 实现。
            pass

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L337-358
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L361-403
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.
    """
    import flashinfer

    # SUBTRACTED: flashinfer 版本校验（L379-382）—— 仅 ImportError 守护，不改算法。
    assert not (k is None and p is None)
    if k is None:
        # Top-p only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # Top-k only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Both top-k and top-p.
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/topk_topp_triton.py:L965-1051
    """
    Apply combined top-k and top-p masking using Triton.

    Top-k is applied first (by logit value), then top-p is applied
    to the remaining k values (by probability).
    """
    # SUBTRACTED: 整个 wrapper 体（assert fp32/2D、dummy 指针、按 SM 数定 NUM_PROGRAMS、
    # 每设备 buffer 与两张正态分位查找表缓存、_topk_topp_kernel 启动）以及 920 行
    # _topk_topp_kernel 内核本体（topk_topp_triton.py:L70-1051）。该内核是 Park et al.
    # "Qrita" pivot-truncation 的工程实现：用高斯 sigma 截断 + 三分搜索逼近 pivot，把
    # top-k/top-p 复杂度由 O(V log V) 排序降到 ~O(V) 多趟扫描，in-place 写回。本章只论证
    # "batch>=8 时为何另起内核"，不逐行讲三分搜索。其外部契约（in-place 截断 logits）与
    # apply_top_k_top_p_pytorch 等价。
    raise NotImplementedError(
        "Triton Qrita kernel subtracted; HAS_TRITON=False routes to the "
        "pytorch sort path. See vllm/v1/sample/ops/topk_topp_triton.py."
    )
