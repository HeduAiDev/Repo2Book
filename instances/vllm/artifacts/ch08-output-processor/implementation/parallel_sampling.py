"""ParentRequest: n>1 child -> parent aggregation, subtract-only.

Only the aggregation side (get_outputs) is in this chapter's scope; the request
splitting side (_get_child_sampling_params / get_child_info / observe_*) is
subtracted per the subtraction_plan (covered on the input/ch04 side).
"""

from __future__ import annotations

from typing import cast

from ._types import CompletionOutput, RequestOutputKind


# SOURCE: vllm/v1/engine/parallel_sampling.py:L13 ParentRequest
class ParentRequest:
    """Info, state & processing for parallel sampling request."""

    request_id: str
    external_req_id: str
    sampling_params: object

    # To track the completion of child requests
    child_requests: set[str]

    # To aggregate child completions when not streaming
    output_aggregator: list[CompletionOutput]

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L36
    # SUBTRACTED: max_num_generation_tokens / cached_child_sampling_params init
    #   (request-splitting side, subtraction_plan).
    def __init__(self, request_id: str, external_req_id: str, sampling_params) -> None:
        # SOURCE: vllm/v1/engine/parallel_sampling.py:L36
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.sampling_params = sampling_params

        self.child_requests = set()
        self.output_aggregator = (
            [cast(CompletionOutput, None)] * sampling_params.n
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )

    # SUBTRACTED: _get_child_sampling_params / get_child_info / n property /
    #   observe_num_generation_tokens (parallel_sampling.py:L52-L94, L128+) —
    #   request-splitting side, out of Stage 3 scope (subtraction_plan).

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L100
    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        already_finished_and_returned: bool = False
        if completion_output.finished():
            if child_request_id in self.child_requests:
                self.child_requests.remove(child_request_id)
            else:
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                already_finished_and_returned = True

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # If streaming, just return the current output
            #
            # DO NOT output finished and already returned child request to client again
            outputs = [] if already_finished_and_returned else [completion_output]
        else:
            # If not streaming, aggregate the n final outputs.
            self.output_aggregator[completion_output.index] = completion_output
            outputs = [] if self.child_requests else self.output_aggregator

        finished = not self.child_requests
        return outputs, finished
