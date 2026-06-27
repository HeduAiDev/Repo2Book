"""RequestOutput + RequestOutput.add (DELTA merge), subtract-only.

The ``add`` method is the real engine behind RequestOutputCollector's DELTA
merge: when the producer (output_handler) outruns the consumer (generate()),
multiple deltas land in the same mailbox slot and are merged here.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any

from ._types import CompletionOutput


# SOURCE: vllm/outputs.py:L88 RequestOutput
# SUBTRACTED: metrics / lora_request / encoder_prompt(_token_ids) /
#   num_cached_tokens / kwargs forward-compat + warning — orthogonal carry
#   fields (LoRA/encoder/metrics out of Stage 3 scope, subtraction_plan).
class RequestOutput:
    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: list | None,
        outputs: list[CompletionOutput],
        finished: bool,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> None:
        # SOURCE: vllm/outputs.py:L109
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.kv_transfer_params = kv_transfer_params

    # SOURCE: vllm/outputs.py:L147
    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        """Merge subsequent RequestOutput into this one"""

        self.finished |= next_output.finished
        self.kv_transfer_params = next_output.kv_transfer_params

        for next_completion in next_output.outputs:
            for i, completion in enumerate(self.outputs):
                if completion.index == next_completion.index:
                    if aggregate:
                        # Merge outputs with same index
                        completion.text += next_completion.text
                        if not isinstance(completion.token_ids, MutableSequence):
                            completion.token_ids = list(completion.token_ids)
                        completion.token_ids.extend(next_completion.token_ids)
                        if next_completion.logprobs:
                            assert completion.logprobs is not None
                            completion.logprobs.extend(next_completion.logprobs)
                        completion.cumulative_logprob = (
                            next_completion.cumulative_logprob
                        )
                        completion.finish_reason = next_completion.finish_reason
                        completion.stop_reason = next_completion.stop_reason
                    else:
                        # Replace the output with the new one
                        self.outputs[i] = next_completion
                    break
            else:
                self.outputs.append(next_completion)


# SOURCE: vllm/outputs.py:L196 STREAM_FINISHED sentinel
# A finished sentinel pushed to a collector to unblock generate() with no payload.
STREAM_FINISHED = RequestOutput(
    request_id="",
    prompt=None,
    prompt_token_ids=None,
    prompt_logprobs=None,
    outputs=[],
    finished=True,
)
