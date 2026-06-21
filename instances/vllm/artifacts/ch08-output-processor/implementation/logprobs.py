"""LogprobsProcessor: incremental sample/prompt logprobs accumulation.

Subtract-only. The byte-fallback UTF-8 correction machinery
(_correct_decoded_token / _verify_tokens / _get_sampled_context_ids) is
subtracted per the subtraction_plan; the increment + cumulative_logprob
accumulation and the DELTA pop_prompt_logprobs semantics are preserved.

To run without ``import vllm``, the logprob containers are plain lists and the
EngineCoreOutput carries pre-pythonized sample/prompt logprob payloads
(list of (token_id, logprob) pairs) rather than the real torch tensors /
LogprobsLists; the accumulation control flow matches vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass


# SOURCE: vllm/v1/engine/logprobs.py:L29 LogprobsProcessor
# SUBTRACTED: tokenizer-side detokenization of logprob tokens, FlatLogprobs
#   containers, num_logprobs/num_prompt_logprobs rank handling beyond the
#   sampled logprob — byte-fallback + container plumbing (subtraction_plan).
@dataclass
class LogprobsProcessor:
    # Logprobs for this request (None if disabled).
    logprobs: list | None
    prompt_logprobs: list | None
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None

    # SOURCE: vllm/v1/engine/logprobs.py:L42
    @classmethod
    def from_new_request(
        cls,
        tokenizer,
        request,
    ) -> "LogprobsProcessor":
        # SOURCE: vllm/v1/engine/logprobs.py:L42
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.num_logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(None if num_logprobs is None else []),
            prompt_logprobs=(None if num_prompt_logprobs is None else []),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    # SOURCE: vllm/v1/engine/logprobs.py:L69
    def _update_sample_logprobs(self, new_sample_logprobs) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).
        """
        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        # new_sample_logprobs: list of per-position (sampled_logprob, payload)
        for sampled_token_logprob, payload in new_sample_logprobs:
            # Sampler puts the sampled logprob in first.
            self.cumulative_logprob += sampled_token_logprob
            # Update with the Logprob container for this pos.
            self.logprobs.append(payload)

    # SOURCE: vllm/v1/engine/logprobs.py:L121
    def _update_prompt_logprobs(self, new_prompt_logprobs) -> None:
        """Update with prompt logprobs from EngineCore."""
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None
        # Make Logprob for each position.
        for payload in new_prompt_logprobs:
            self.prompt_logprobs.append(payload)

    # SOURCE: vllm/v1/engine/logprobs.py:L189
    def pop_prompt_logprobs(self) -> list | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    # SUBTRACTED: _verify_tokens / _correct_decoded_token / _get_sampled_context_ids
    #   byte-fallback U+FFFD correction (logprobs.py:L208-L346) — orthogonal
    #   detokenization edge case (subtraction_plan).

    # SOURCE: vllm/v1/engine/logprobs.py:L348
    def update_from_output(self, output) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
