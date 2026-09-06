# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py
# ch29 主角文件之二：TopKTopPSampler 构造期后端绑定（m11）+ top-k/top-p
# 截断与掷骰分流（m9/m10/m12/m13）。保留 CUDA/native 两代表后端；平台
# 变体与 CUDA 主路径算法同构（native 实现共享）。
# SUBTRACTED：delete[4] 平台后端变体——forward_cpu（L184-L212）、
#   _init_aiter_ops（L214-L228）、forward_hip（L230-L252）、aiter_sample
#   （L254-L290）、forward_xpu（L292-L336）、compiled_random_sample
#   （L339-L346）、_skip_aiter_sampler_on_gfx1250（L21-L25）、__init__ 的
#   CPU/XPU/ROCm/兜底分支（L103-L129，删后 self.forward 仅 is_cuda() 下
#   绑定）、apply_top_k_top_p 的 CPU 分支（L355-L358，删后
#   allow_cpu_sync 恒 False）、apply_top_k_only（L411-L431，CPU 免排序
#   特化成死码一并删）。
import torch
import torch.nn as nn

from vllm import envs
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

logger = init_logger(__name__)

# SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L9
#   `from vllm._aiter_ops import rocm_aiter_ops` 与 L12 的 CpuArchEnum
#   —— delete[4] 连带 import 修剪（ROCm aiter / CPU arch 分支已删，
#   仅余消费位 is_cuda 的 current_platform）。


# SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L21-L25
#   _skip_aiter_sampler_on_gfx1250 —— delete[4]（ROCm gfx1250 探测，
#   仅 __init__ 已删的 aiter 分支消费）。


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L28-L74 flashinfer_sampler_supported —— 逐字（显式开但不可用 → RuntimeError、默认静默回退的裁决语义）
def flashinfer_sampler_supported() -> bool:
    """Decide whether FlashInfer's top-p/top-k sampler can be used.

    Returns False (with appropriate logging) when ``VLLM_USE_FLASHINFER_SAMPLER``
    is 0, when the platform isn't CUDA, when the GPU's compute capability is
    unsupported. Raises ``RuntimeError`` if the user explicitly opted in
    via the env var but FlashInfer is unavailable.

    Assumes flashinfer is installed, as guaranteed by ``requirements/cuda.txt``;
    otherwise importing the FlashInfer backend below raises ``ImportError``.

    Note: callers must additionally ensure ``logprobs_mode`` doesn't require
    post-top-k/top-p logits/logprobs for any request whose logprobs will be
    returned in this step, since FlashInfer doesn't expose those.
    """
    if not current_platform.is_cuda():
        return False
    if not envs.VLLM_USE_FLASHINFER_SAMPLER:
        logger.info_once(
            "FlashInfer top-p/top-k sampling disabled via "
            "VLLM_USE_FLASHINFER_SAMPLER=0."
        )
        return False
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    capability = current_platform.get_device_capability()
    assert capability is not None
    unsupported_reason: str | None = None
    if not FlashInferBackend.supports_compute_capability(capability):
        unsupported_reason = (
            f"unsupported compute capability {capability.as_version_str()}"
        )

    if unsupported_reason is None:
        logger.info_once("Using FlashInfer for top-p & top-k sampling.", scope="global")
        return True
    if envs.is_set("VLLM_USE_FLASHINFER_SAMPLER"):
        raise RuntimeError(
            f"FlashInfer top-p/top-k sampling unavailable: {unsupported_reason}. "
            "Unset VLLM_USE_FLASHINFER_SAMPLER=1."
        )
    logger.warning_once(
        "FlashInfer top-p/top-k sampling unavailable: %s; falling back. "
        "Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.",
        unsupported_reason,
    )
    return False


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L77-L83 TopKTopPSampler —— 逐字
class TopKTopPSampler(nn.Module):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L85-L102 __init__ （is_cuda 绑定位逐字）
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ) -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel
        if current_platform.is_cuda():
            # FlashInfer doesn't expose post-top-k/top-p logits/logprobs,
            # so it can't be used when the configured mode requires them.
            can_use_flashinfer = (
                logprobs_mode not in PROCESSED_LOGPROBS_MODES
                and flashinfer_sampler_supported()
            )
            self.forward = (
                self.forward_cuda if can_use_flashinfer else self.forward_native
            )
        # SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L103-L129
        #   __init__ 的 CPU/XPU/ROCm/兜底四分支 —— delete[4]（平台变体；
        #   删后 self.forward 仅 is_cuda() 下绑定——本类不定义 forward，
        #   非 CUDA 环境调用方按构造期 native 绑定支路（L100-L102 的
        #   else 位）显式取 forward_native；CUDA 主路径行为不变）。

    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L131-L153 forward_native —— 逐字
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        return (
            random_sample(probs, generators, self.use_fp64_gumbel),
            logits_to_return,
        )

    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L155-L182 forward_cuda —— 逐字
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        # Fall back to the PyTorch-native path when FlashInfer has nothing
        # to do (no top-k / top-p filter) or when per-request generators
        # are present (unsupported by FlashInfer 0.2.3+).
        if (k is None and p is None) or generators:
            if generators:
                logger.debug_once(
                    "FlashInfer 0.2.3+ does not support "
                    "per-request generators. Falling back to "
                    "PyTorch-native implementation."
                )
            return self.forward_native(logits, generators, k, p)
        if self.use_fp64_gumbel:
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in PROCESSED_LOGPROBS_MODES, (
            "FlashInfer does not support returning logits/logprobs"
        )
        # flashinfer sampling functions expect contiguous logits.
        # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
        # because of slicing operation in logits_processor.
        return flashinfer_sample(logits.contiguous(), k, p, generators), None

    # SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L184-L212 forward_cpu
    #   —— delete[4]（CPU 平台变体：与 forward_native 同构、多一条
    #   compiled_random_sample 快路；算法共享 native 实现）。
    # SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L214-L228
    #   _init_aiter_ops —— delete[4]（ROCm aiter 惰性导入面）。
    # SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L230-L252 forward_hip
    #   —— delete[4]（ROCm/aiter 路径，结构与 forward_cuda 同构）。
    # SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L254-L290 aiter_sample
    #   —— delete[4]（aiter 三 API 分发）。
    # SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L292-L336 forward_xpu
    #   —— delete[4]（XPU 自定义核路径）。


# SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L339-L346
#   compiled_random_sample —— delete[4]（forward_cpu 专属的 @torch.compile
#   快路，消费方已删）。


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L349-L364 apply_top_k_top_p （二级分流；CPU 分支已按 delete[4] 减去）
def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    # SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L355-L358
    #   `if current_platform.is_cpu(): ...` —— delete[4]（CPU 平台分支：
    #   有 Triton 走同一 triton 核、否则 pytorch+allow_cpu_sync=True；
    #   删后 CPU 张量与 CUDA 张量同走下方两级分流，仅 k-only+CPU 时
    #   少了 apply_top_k_only 免排序特化——掩码语义等价（同为严格 <
    #   阈值），CUDA 路径行为不变）。

    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L367-L408 apply_top_k_top_p_pytorch —— 逐字（教学主实现：升序 sort → 第 (V-k) 位阈值严格 < → softmax cumsum ≤ 1-p → scatter 回原位）
def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

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


# SUBTRACTED: vllm/v1/sample/ops/topk_topp_sampler.py:L411-L431 apply_top_k_only
#   —— delete[4]（CPU allow_cpu_sync 路径的免排序 top-k 特化；CPU 分支
#   删除后 allow_cpu_sync 恒 False、成死码一并删——pytorch sort 路径的
#   严格 < 阈值掩码与它语义等价）。


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L434-L438 empty_exponential_noise_like —— 逐字
def empty_exponential_noise_like(
    probs: torch.Tensor, use_fp64_gumbel: bool
) -> torch.Tensor:
    dtype = torch.float64 if use_fp64_gumbel else probs.dtype
    return torch.empty(probs.shape, dtype=dtype, device=probs.device)


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L441-L447 sample_with_exponential_noise —— 逐字（Gumbel 掷骰本体：probs/q 取 argmax；fp64 时先 reciprocal 再乘避免中途降精度）
def sample_with_exponential_noise(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if q.dtype == probs.dtype:
        scores = probs.div_(q)
    else:
        scores = q.reciprocal_()
        scores.mul_(probs)
    return scores.argmax(dim=-1).view(-1)


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L450-L472 random_sample —— 逐字 （multinomial 同步之死的替代：docstring 原话 "torch.multinomial causes CPU-GPU synchronization"；无 seed 行先批量生成、有 seed 逐请求覆写）
def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L475-L512 flashinfer_sample —— 逐字（拒绝采样免排序；docstring 自述 statistically equivalent）
def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator] = {},  # noqa
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.
    """
    import flashinfer

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


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L515-L519 _to_tensor_scalar_tuple —— 逐字（原 aiter_sample 的 (tensor, 0)/(None, scalar) 归一辅助； 其唯一消费方 aiter_sample 已按 delete[4] 删除，此处按「只删批准项」 原样保留——无行为影响）
def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)
