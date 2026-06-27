"""Stage-3 output processor single loop — subtract-only port of the
detokenize / stop-string slice of ``vllm/v1/engine/output_processor.py``.

Only the three lines that connect to incremental detokenization are kept:
  1. ``detokenizer.update(new_token_ids, finish_reason == STOP)``  → stop_string
  2. ``if stop_string:`` rewrite finish_reason = STOP / stop_reason
  3. ``reqs_to_abort.append(req_id)`` when EngineCore had not itself finished.

Everything else in process_outputs (stats / logprobs / pooling / streaming-input /
RequestOutput assembly / tracing / parallel sampling) is subtracted per
subtraction_plan.delete — orthogonal subsystems covered by other chapters.
"""

from __future__ import annotations

from ._types import (
    EngineCoreOutput,
    FinishReason,
    OutputProcessorOutput,
)
from .detokenizer import IncrementalDetokenizer


# SOURCE: vllm/v1/engine/output_processor.py RequestState (detokenize-relevant subset)
# SUBTRACTED: logprobs_processor / queue / make_request_output / parent_req /
#   streaming_input / stats fields — only the detokenizer handle the stop loop
#   reads is kept (subtraction_plan.delete).
class RequestState:
    # SOURCE: vllm/v1/engine/output_processor.py RequestState (detokenize-relevant subset)
    def __init__(self, request_id: str, detokenizer: IncrementalDetokenizer):
        # SOURCE: vllm/v1/engine/output_processor.py RequestState.__init__
        self.request_id = request_id
        self.detokenizer = detokenizer


# SOURCE: vllm/v1/engine/output_processor.py OutputProcessor (subset)
class OutputProcessor:
    def __init__(self):
        # SOURCE: vllm/v1/engine/output_processor.py OutputProcessor.__init__
        # SUBTRACTED: external_req_ids / lora_states / parallel-sampling state /
        #   log_stats / tokenizer — orthogonal to the stop-string slice.
        self.request_states: dict[str, RequestState] = {}

    # SOURCE: vllm/v1/engine/output_processor.py:L597
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.
        """
        # SUBTRACTED: request_outputs list assembly (make_request_output) — the
        #   ch09 slice only tracks reqs_to_abort (subtraction_plan.delete).
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # SUBTRACTED: 1) stats; is_prefilling / num_cached_tokens bookkeeping.

            new_token_ids = engine_core_output.new_token_ids
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason

            # SUBTRACTED: `if pooling_output is None:` guard (pooling product line)
            #   and the logprobs_processor.update_from_output call (ch10).
            assert req_state.detokenizer is not None
            # 2) Detokenize the token ids into text and perform stop checks.
            stop_string = req_state.detokenizer.update(
                new_token_ids, finish_reason == FinishReason.STOP
            )
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string

            # SUBTRACTED: 4) make_request_output + queue/return dispatch +
            #   streaming-input branch (subtraction_plan.delete).

            # Free completed requests.
            if finish_reason is not None:
                self._finish_request(req_state)
                if not engine_core_output.finished:
                    # If req not finished in EngineCore, but Detokenizer
                    # detected stop string, abort needed in EngineCore.
                    reqs_to_abort.append(req_id)

                # SUBTRACTED: per-request finished stats + tracing.

        return OutputProcessorOutput(
            reqs_to_abort=reqs_to_abort,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L714
    def _finish_request(self, req_state: RequestState) -> None:
        # SUBTRACTED: external_req_ids cleanup + parent_req aggregation
        #   (parallel sampling / external id mapping — subtraction_plan.delete).
        self.request_states.pop(req_state.request_id, None)
