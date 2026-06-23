# SUBTRACTED: SPDX 版权头（vllm/v1/sample/rejection_sampler.py:L1-L2）。
# SUBTRACTED: 一批 import：LogprobsLists/LogprobsTensors（logprobs 装配，归 ch10/ch27）、
#             MinTokensLogitsProcessor、apply_bad_words_with_drafts、apply_all_penalties
#             —— 仅服务被减掉的 logprobs/penalties/bad_words 旁路；Sampler 与
#             SpeculativeConfig 在精简版里用轻量替身（见下）。
# SUBTRACTED: `from ...spec_decode.utils import unconditional_to_conditional_rates`
#             —— synthetic 模式专用，subtraction_plan.delete 批准删除。
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.nn as nn
from triton import jit
import triton.language as tl

from metadata import SpecDecodeMetadata
from sampling_metadata import SamplingMetadata
from topk_topp import apply_top_k_top_p

PLACEHOLDER_TOKEN_ID: "tl.constexpr" = -1
GREEDY_TEMPERATURE: "tl.constexpr" = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step.
MAX_SPEC_LEN = 128


# SOURCE: vllm/v1/sample/rejection_sampler.py:L37-L389
class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L60-L85
    # SUBTRACTED: logprobs_mode 探测（is_processed/is_logits_logprobs_mode）只服务
    #             logprobs 装配；synthetic_conditional_rates / synthetic_mode 初始化
    #             （spec_config=="synthetic" 分支，L72-L85）—— subtraction_plan.delete
    #             批准。精简版只保留 self.sampler。
    def __init__(self, sampler):
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L60-L85
        super().__init__()
        self.sampler = sampler

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L87-L195
    # SUBTRACTED: logprobs_tensors 计算分支（L181-L190 调 _get_logprobs_tensors）——
    #             归 ch10/ch27 语境；max_num_logprobs=None 时本就跳过。
    # SUBTRACTED: synthetic_mode / synthetic_conditional_rates 透传给 rejection_sample
    #             （L177-L178）—— synthetic 删除项。
    # SUBTRACTED: logprobs_mode_override / is_processed_logprobs_mode 相关的 bonus
    #             采样旁路（L136-L141, L151-L155）—— logprobs 装配；精简版直接采 bonus。
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L87-L195
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch creates a
        # new tensor with separate storage from the original logits tensor. This
        # means any in-place operations on bonus_logits won't affect the
        # original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_token_ids = self.sampler(bonus_logits, sampling_metadata)

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore, it is
        # safe to update `target_logits` in place.
        target_logits = logits[target_logits_indices].to(torch.float32)
        target_logits = target_logits.clone()
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L246-L281
    # SUBTRACTED: logprobs_tensors filter 分支（L270-L274）—— logprobs 装配，归 ch10。
    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
    ) -> list[list[int]]:
        """Parse the output of the rejection sampler.

        output_token_ids: [batch_size, max_spec_len + 1]. 被拒/未用位置为
        PLACEHOLDER_TOKEN_ID(-1)，在此过滤；越界 token (>= vocab_size) 同样过滤。
        """
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L246-L281
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs


# SOURCE: vllm/v1/sample/rejection_sampler.py:L392-L503
# SUBTRACTED: synthetic_mode / synthetic_conditional_rates 形参与透传给两个 kernel
#             （L407-L408, L442, L462-L463, L499）—— synthetic 删除项；精简版固定
#             SYNTHETIC_MODE=False，准则退化为标准 greedy/random。
def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L392-L503
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # Create output buffer. Pre-filled with PLACEHOLDER_TOKEN_ID; second dim is
    # max_spec_len + 1 (the +1 is the bonus slot).
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    # [num_tokens]
    uniform_probs: torch.Tensor | None = None
    if not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Compute probability distribution from target logits.
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    assert uniform_probs is not None
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids


# SOURCE: vllm/v1/sample/rejection_sampler.py:L506-L561 (apply_sampling_constraints)
# SUBTRACTED: 详细 docstring 略；逻辑逐字保留。
def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """对草稿位置的目标 logits 应用温度/top-k/top-p。greedy 时原样返回。"""
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L506-L561
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )
    return apply_top_k_top_p(logits, top_k, top_p)


# SOURCE: vllm/v1/sample/rejection_sampler.py:L564-L601 (expand_batch_to_tokens)
def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] based on cu_num_tokens.

    E.g. x=[a,b,c], cu_num_tokens=[2,5,6] -> [a,a,b,b,b,c].
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


# SOURCE: vllm/v1/sample/rejection_sampler.py:L604-L656
def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """生成 [num_tokens] 的 U[0,1) 随机数，供随机路径接受准则使用。"""
    # NOTE(woosuk): We deliberately use float64 instead of float32 here because
    # when using float32, there's a non-negligible chance that uniform_prob is
    # sampled to be exact 0.0 (pytorch#16706). Using float64 mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


# SOURCE: vllm/v1/sample/rejection_sampler.py:L659-L703
def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    # q ~ Exp(1); inv_q = 1/q. argmax(prob * inv_q) is Gumbel-max sampling from
    # the (unnormalized) residual distribution prob -- no explicit normalization.
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# SOURCE: vllm/v1/sample/rejection_sampler.py:L706-L757
# SUBTRACTED: SYNTHETIC_MODE 分支与 uniform_probs_ptr/synthetic_conditional_rates_ptr
#            形参（L716-L718, L737-L742）—— synthetic 删除项；保留标准 greedy 准则。
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
):
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L706-L757
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            token_id = target_argmax_id
            # greedy: accept iff draft == argmax; otherwise reject & stop. The
            # rejected position still writes target_argmax_id (recovered = greedy
            # target token), matching greedy semantics.
            rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# SOURCE: vllm/v1/sample/rejection_sampler.py:L760-L826
# SUBTRACTED: SYNTHETIC_MODE 分支与 synthetic_conditional_rates_ptr 形参
#            （L774, L776, L793-L795）—— synthetic 删除项；保留标准随机准则。
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L760-L826
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NO_DRAFT_PROBS (ngram): draft_prob = 1, accept criterion degenerates
            # to accepting with probability p_target(x).
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr
                    + (start_idx + pos) * vocab_size
                    + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            # NOTE(woosuk): While the draft probability should never be 0, we
            # check it to avoid NaNs. If it happens to be 0, we reject.
            # accept iff target_prob/draft_prob >= uniform_prob, i.e. accept with
            # probability min(1, p_target(x)/p_draft(x)).
            accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# SOURCE: vllm/v1/sample/rejection_sampler.py:L829-L850 (expand_kernel)
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L829-L850
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


# SOURCE: vllm/v1/sample/rejection_sampler.py:L853-L921
@jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L853-L921
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float("-inf")
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            # ngram residual = target_probs with the draft token masked out.
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            # residual numerator (p_target - p_draft)_+.
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction: Gumbel-max score = prob * inv_q.
        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
