# SOURCE: vllm/v1/sample/sampler.py
# ch29 主角文件：Sampler——9 步采样管线编排者（docstring L20-L59 即目录）。
# forward 做 step1-2/8-9，apply_logits_processors 做 step3-6，sample 做
# step7；全程吃 SamplingMetadata 冻结快照、logits 全程被原位改写（m15）。
# SUBTRACTED：delete[0] logprob_token_ids 旁路——gather_specific_token_logprobs
#   全方法（L151-L225）与 forward 的取值块（L114-L118）/注释（L111-L112）/
#   消费块（L133-L136）。保留 L113（logprob_token_ids_tensors = None）与
#   L120-L121 的 `if num_logprobs is None` 分支头原样——max_num_logprobs 为
#   None 的默认路径（无请求要 logprobs）依赖这个分支给 logprobs_tensors 赋
#   None，升格/删头都会让默认路径落到 gather_logprobs(None,…) 炸掉。
#   logprob_token_ids 恒 None/空时整条旁路不执行，主路径行为不变
#   （generative_scoring 契约面归 ch8 登记）。
# SUBTRACTED：delete[1] thinking budget 调用块（L404-L416）——第 9 类状态
#   （m7 登记轻讲，正文内嵌真源码；未设 reasoning_config 时 holder 恒 None、
#   整块跳过，标准 9 步路径逐字节不变）。
# SUBTRACTED：delete[2] spec decode 组合分支——_combine_outputs_with_spec_tokens
#   （L358-L369）与 apply_logits_processors 的 predict_bonus_token spec 合并
#   （L382-L388）——非 spec 路径 predict_bonus_token=False、不触发（归 ch32/33）。

import torch
import torch.nn as nn

from vllm.config.model import LogprobsMode
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


# SOURCE: vllm/v1/sample/sampler.py:L20-L59 Sampler 类 9 步 docstring —— 逐字
class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    # SOURCE: vllm/v1/sample/sampler.py:L61-L70 Sampler.__init__ —— 逐字
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode, use_fp64_gumbel)
        self.pin_memory = PIN_MEMORY
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel

    # SOURCE: vllm/v1/sample/sampler.py:L72-L149 forward —— 逐字 （logprob_token_ids 旁路三段已按 delete[0] 减去、保留分支头）
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None or sampling_metadata.logprob_token_ids:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # SUBTRACTED: vllm/v1/sample/sampler.py:L111-L112/L114-L118
        #   logprob_token_ids 旁路的取值段（注释两行 + `if
        #   sampling_metadata.logprob_token_ids:` 块——generative_scoring
        #   稀疏 gather 归 ch8 第 1 站登记；恒 None/空时不执行，保留下方
        #   L113 的 None 初值不动）。
        logprob_token_ids_tensors = None

        if num_logprobs is None:
            logprobs_tensors = logprob_token_ids_tensors
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        # SUBTRACTED: vllm/v1/sample/sampler.py:L133-L136
        #   「both num_logprobs and logprob_token_ids → prefer the latter」
        #   消费块 —— delete[0]（logprob_token_ids_tensors 恒 None，此块
        #   恒不触发）。

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    # SUBTRACTED: vllm/v1/sample/sampler.py:L151-L225
    #   gather_specific_token_logprobs 全方法 —— delete[0]（按请求稀疏
    #   token id 表 gather logprobs：padded 矩阵 + valid_mask + 采样位第 0
    #   列；generative_scoring API 专用旁路，归 ch8）。

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/sampler.py:L227-L237 apply_temperature —— 逐字
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/sampler.py:L239-L241 greedy_sample —— 逐字
        return logits.argmax(dim=-1).view(-1)

    # SOURCE: vllm/v1/sample/sampler.py:L243-L302 sample —— 逐字（step7： greedy 早退 / 温度 / argmax 不变列 / top-k-top-p / torch.where 合并）
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.logprob_token_ids
                ):
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/sampler.py:L304-L306 compute_logprobs —— 逐字
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        # SOURCE: vllm/v1/sample/sampler.py:L308-L356 gather_logprobs —— 逐字 （step8：topk + 被采样 logprob + rank 不排序；raw 视角取的是第 1 步留底的、未经惩罚/温度的张量）
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: maximum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        # Avoid 0/1 specialization recompile on the batch dimension
        # of the compiled batched_count_greater_than. mark_unbacked makes
        # the size fully symbolic so dynamo doesn't specialize when
        # batch_size transitions from 1 to >=2.
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    # SUBTRACTED: vllm/v1/sample/sampler.py:L358-L369
    #   _combine_outputs_with_spec_tokens —— delete[2]（把 base outputs 与
    #   spec draft tokens 拼起来供惩罚/bad_words 看「到目前为止的历史」；
    #   仅 predict_bonus_token=True 的 RejectionSampler 路径消费，归 ch32/33）。

    # SOURCE: vllm/v1/sample/sampler.py:L371-L417 apply_logits_processors —— step3-6 逐字（spec 合并与 thinking budget 两段已按 delete[2]/[1] 减去）
    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )
        output_token_ids = sampling_metadata.output_token_ids
        # SUBTRACTED: vllm/v1/sample/sampler.py:L382-L388
        #   `if predict_bonus_token and any_penalties_or_bad_words:` spec 合并
        #   分支 —— delete[2]（_combine_outputs_with_spec_tokens 的唯一消费位；
        #   非 spec 路径 predict_bonus_token=False 恒不触发）。

        # Apply allowed token ids.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply logits processors which can impact greedy sampling.
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        # SUBTRACTED: vllm/v1/sample/sampler.py:L404-L416 thinking budget 调用块
        #   —— delete[1]（holder.update_state + holder.apply_to_logits：预算
        #   耗尽把 think_end token 的 logit 顶格 1e9 强制出门；未设
        #   reasoning_config 时 holder 恒 None、整块跳过——m7 登记轻讲、
        #   正文内嵌真源码 thinking_budget_state.py:L20-L81，不依赖精简版）。
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/sampler.py:L419-L436 apply_penalties —— 逐字
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )
