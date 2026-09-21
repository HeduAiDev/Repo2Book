# SOURCE: vllm/v1/sample/rejection_sampler.py —— 本章主角文件（只做减法）
# 投机解码验证期采样器（V1）：v0.27 起组合持有普通 Sampler 实例（不再复制
# 管线）——bonus 位外采可带 top_p/top_k；forward 切 target/bonus logits、
# spec 特化约束、rejection_sample 双 Triton kernel、parse_output 还原变长。
# 减法 = dossier subtraction_plan.delete 三项（详见各 SUBTRACTED 标记）：
#   [0] synthetic 模式全链（测试注入用，生产 spec_config 传 None 不触发）
#   [1] logprobs 装配（_get_logprobs_tensors/forward 尾部/parse_output 过滤分支）
#   [2] thinking budget 支路（apply_logits_processors 的 holder 调用块）
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L16
#   `from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput`
#   的 LogprobsLists/LogprobsTensors 两名——delete[1] logprobs 装配连带 import
#   修剪（唯一消费方 _get_logprobs_tensors/parse_output 的 logprobs 形参已删）。
# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L24
#   `from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates`
#   —— delete[0] synthetic 删除项（唯一下游 L84 已随删）。

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


# SOURCE: vllm/v1/sample/rejection_sampler.py:L38-L59 RejectionSampler 类 docstring
#   —— 逐字（四类 token 术语定义 + arXiv:2211.17192 自引）
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

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L61-L90 __init__ —— v0.27 组合
    #   结构逐字保留（self.sampler / use_fp64_gumbel 透传 / logprobs_mode 探测）
    # SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L77-L90
    #   synthetic_conditional_rates 构造（spec_config.rejection_sample_method
    #   =="synthetic" 才走）与 self.synthetic_mode——delete[0]；合成接受率基准
    #   注入非真实推理路径，spec_config=None（生产默认）时本就不触发。
    #   spec_config/device 形参保留（真实构造签名，gpu_model_runner.py:L656-L658
    #   按位传入）。
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L61-L90 __init__（v0.27 组合结构）
        super().__init__()
        self.sampler = sampler
        self.use_fp64_gumbel = getattr(sampler, "use_fp64_gumbel", False)
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode in PROCESSED_LOGPROBS_MODES
        self.is_logits_logprobs_mode = logprobs_mode in (
            "raw_logits",
            "processed_logits",
        )

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L92-L201 forward —— 主线逐字
    #   （bonus 外采 L133-L147 / target fp32+clone 保 raw L152-L160 / 约束 /
    #   rejection_sample L173-L185）
    # SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L182-L183
    #   rejection_sample 的 synthetic_mode/synthetic_conditional_rates 透传
    #   —— delete[0] synthetic 删除项。
    # SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L187-L196
    #   logprobs_tensors 计算（max_num_logprobs 非 None 时调 _get_logprobs_tensors）
    #   —— delete[1] logprobs 装配（归 ch8/ch30 语境）；默认路径（无请求要
    #   logprobs）本就不进此分支，删后输出 token 逐位不变。
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L92-L201 forward（bonus 外采/target 切片/约束/rejection_sample）
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
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
            use_fp64_gumbel=self.use_fp64_gumbel,
        )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            # SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L187-L196 ——
            #   delete[1]：此处按计划改传 None（保留返回签名）
            logprobs_tensors=None,
        )

    # SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L203-L250
    #   _get_logprobs_tensors 整方法 —— delete[1] logprobs 装配（accepted
    #   token logprobs 的拼装与 gather，归 ch8/ch30 语境）。

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L252-L287 parse_output ——
    #   valid_mask（≠PLACEHOLDER 且 <vocab_size）过滤还原逐字保留
    # SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L277-L280
    #   logprobs_tensors 过滤分支与 logprobs_tensors 形参 —— delete[1]
    #   （连带 import 的 LogprobsLists；返回签名保持二元组、第二元恒 None）。
    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
    ) -> tuple[list[list[int]], None]:
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L252-L287 parse_output（valid_mask 还原变长）
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L289-L346 apply_logits_processors
    #   —— spec 特化逐字保留（_combine 前缀历史 / repeat_indices 展开 /
    #   allowed mask / bad_words_with_drafts / MinTokens.apply_with_spec_decode）
    # SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L339-L345
    #   thinking budget 支路（holder 调用块）—— delete[2]；未设
    #   reasoning_config 时 holder 恒 None、整块跳过（ch30 m7 同款处理）。
    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L289-L346 apply_logits_processors（spec 特化约束）
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )
        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        repeat_indices: torch.Tensor | None = None
        need_repeat_indices = (
            sampling_metadata.allowed_token_ids_mask is not None or has_penalties
        )
        if need_repeat_indices:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )
        return logits

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L348-L374 apply_penalties —— 逐字
    #   （prompt/presence/frequency/repetition 经 repeat_indices 展开到草稿位）
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L348-L374 apply_penalties（repeat_indices 展开）
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    # SOURCE: vllm/v1/sample/rejection_sampler.py:L376-L391 _combine_outputs_with_spec_tokens
    #   —— 逐字（惩罚/bad_words 的『草稿当前缀』历史：每个草稿位一行、
    #   到此位为止的前缀；空草稿请求整段跳过——注意与 Sampler 里的同名方法
    #   语义不同：那边是拼整段 spec、逐请求一行）
    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        # SOURCE: vllm/v1/sample/rejection_sampler.py:L376-L391 _combine_outputs_with_spec_tokens（逐位前缀行）
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


# SOURCE: vllm/v1/sample/rejection_sampler.py:L394-L507 rejection_sample —— 调度
#   核心逐字保留（[B, max_spec_len+1] 预填 -1 / all_greedy 早退 / recovered
#   预采 / random kernel 收尾；is_greedy mask 双 kernel 同批共存）
# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L409-L410
#   synthetic_mode/synthetic_conditional_rates 形参 —— delete[0]。
# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L440-L445 —— synthetic 生成条件
#   与注释里的 synthetic 句（"synthetic mode needs them in the greedy kernel
#   too"），按计划只留 `not sampling_metadata.all_greedy`。
# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L464-L466 —— greedy kernel 调用
#   的 uniform_probs/synthetic_conditional_rates/SYNTHETIC_MODE 三参（uniform
#   在 greedy kernel 内只有 synthetic 分支消费，随删；标准 greedy 准则不需要）。
# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L503-L505 —— random kernel 调用
#   的 synthetic_conditional_rates/SYNTHETIC_MODE 两参。
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
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L394-L507 rejection_sample（调度核心）
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

    # Create output buffer.
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
        use_fp64_gumbel,
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


# SOURCE: vllm/v1/sample/rejection_sampler.py:L510-L565 apply_sampling_constraints
#   —— 逐字（温度/top-k/top-p 从 [batch] 扩到逐草稿位；greedy 的温度 0 替 1
#   防除零；all_greedy 原样返回）
def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L510-L565 apply_sampling_constraints（逐草稿位温度/top-k/top-p）
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
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
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
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

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


# SOURCE: vllm/v1/sample/rejection_sampler.py:L568-L605 expand_batch_to_tokens
#   —— 逐字（[batch]→[num_tokens] 的 kernel 化展开；MAX_NUM_TOKENS=MAX_SPEC_LEN
#   防 kernel 重编译）
def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L568-L605 expand_batch_to_tokens（[batch]→[num_tokens] 展开）
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
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


# SOURCE: vllm/v1/sample/rejection_sampler.py:L608-L660 generate_uniform_probs
#   —— 逐字（float64 uniform：float32 有非平凡概率采到精确 0.0，pytorch#16706；
#   n=0 请求跳过生成保可复现；有 seed 的逐请求 generator 覆写）
def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L608-L660 generate_uniform_probs（float64 uniform）
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens: int
            Total number of tokens.
        num_draft_tokens: List[List[int]]
            Number of draft tokens per request.
        generators: Optional[Dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device: torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand: torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


# SOURCE: vllm/v1/sample/rejection_sampler.py:L663-L710 sample_recovered_tokens
#   —— 逐字（v0.27 布局：每请求只建一行 q~Exp(1)（[batch, vocab]），全部草稿位
#   共享——只有首个拒绝位的 recovered 被消费，边缘分布正确即可）
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
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L663-L710 sample_recovered_tokens（每请求一份 q）
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q_dtype = torch.float64 if use_fp64_gumbel else torch.float32
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=q_dtype,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
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
        USE_FP64_GUMBEL=use_fp64_gumbel,
    )
    return recovered_token_ids


# SOURCE: vllm/v1/sample/rejection_sampler.py:L713-L769 rejection_greedy_sample_kernel
#   —— 标准 greedy 准则逐字保留（draft==target_argmax 接受、拒绝位写 argmax、
#   早停、全收补 bonus）
# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L723-L724 —— 签名里的
#   uniform_probs_ptr（synthetic mode only，注释原话）/synthetic_conditional_rates_ptr
#   两形参与 L748-L754 SYNTHETIC_MODE 分支 —— delete[0] synthetic 删除项；
#   删后本 kernel 只剩标准 greedy 准则（生产路径 SYNTHETIC_MODE=False 时的
#   逐字节等价）。
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
):
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L715-L769 rejection_greedy_sample_kernel（greedy 准则）
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            token_id = target_argmax_id
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


# SOURCE: vllm/v1/sample/rejection_sampler.py:L772-L845 rejection_random_sample_kernel
#   —— 算法心脏逐字保留（接受判据 L829 `draft_prob > 0 and target_prob /
#   draft_prob >= uniform_prob`；NO_DRAFT_PROBS 退化；padded draft(-1) 直接拒；
#   拒绝位 recovered+早停；全收补 bonus）
# SUBTRACTED: vllm/v1/sample/rejection_sampler.py:L786/L788 —— 签名里的
#   synthetic_conditional_rates_ptr 形参与 SYNTHETIC_MODE constexpr，及
#   L812-L814 的 SYNTHETIC_MODE 分支 —— delete[0] synthetic 删除项。
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
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
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L774-L845 rejection_random_sample_kernel（random 准则·算法心脏）
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if draft_token_id < 0:
                # -1 is used for padded draft token ids that should be rejected.
                accepted = False
            else:
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
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
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


# SOURCE: vllm/v1/sample/rejection_sampler.py:L848-L869 expand_kernel —— 逐字
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L850-L869 expand_kernel（展开写位）
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = tl.zeros([], dtype=cu_num_tokens_ptr.dtype.element_ty)
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


# SOURCE: vllm/v1/sample/rejection_sampler.py:L872-L953 sample_recovered_tokens_kernel
#   —— 残差 kernel 逐字保留（prob=max(p_t−p_d,0)、NO_DRAFT_PROBS 屏蔽 draft
#   token 的 p_t、score=prob·inv_q 免归一化 argmax、OOV mask −inf + clamp
#   数值卫生）
@triton.jit
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
    USE_FP64_GUMBEL: tl.constexpr,
):
    # SOURCE: vllm/v1/sample/rejection_sampler.py:L873-L953 sample_recovered_tokens_kernel（残差 Gumbel-max）
    req_idx = tl.program_id(0)
    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    if USE_FP64_GUMBEL:
        max_val = tl.full((), float("-inf"), tl.float64)
    else:
        max_val = tl.full((), float("-inf"), tl.float32)
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
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
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction.
        # Mask out-of-vocabulary entries to -inf so they can never win
        # the argmax — prevents producing recovered_id >= vocab_size
        # when all valid entries in the last tile have zero probability.
        score = prob * inv_q
        score = tl.where(vocab_mask, score, float("-inf"))
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    recovered_id = tl.minimum(recovered_id, vocab_size - 1)
    tl.store(output_token_ids_ptr + token_idx, recovered_id)
