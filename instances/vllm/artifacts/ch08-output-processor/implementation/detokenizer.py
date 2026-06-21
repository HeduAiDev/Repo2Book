"""IncrementalDetokenizer: incremental detokenization + stop-string detection.

Subtract-only. The abstract ``decode_next`` and the two concrete Fast/Slow
implementations in real vLLM live inside the tokenizer library (tokenizers
DecodeStream / python detokenize_incrementally with prefix/read offsets). Those
internals — including byte-fallback recovery — are subtracted per the
subtraction_plan; the core incremental algorithm (output_text accumulation +
offset bookkeeping + check_stop_strings) is preserved verbatim.

To stay runnable without ``import vllm`` / the tokenizers library, the concrete
SlowIncrementalDetokenizer takes a plain ``decode`` callable (id -> str) instead
of the real tokenizer; the surrounding update/get_next_output_text/stop logic is
identical to vLLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ._types import EngineCoreOutput  # noqa: F401  (parallels real import surface)


# SOURCE: vllm/v1/engine/detokenizer.py:L30
class IncrementalDetokenizer:
    def __init__(self):
        # SOURCE: vllm/v1/engine/detokenizer.py:L31
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/engine/detokenizer.py:L34
        return self.token_ids

    # SOURCE: vllm/v1/engine/detokenizer.py:L38
    def num_output_tokens(self) -> int:
        return len(self.token_ids)

    # SOURCE: vllm/v1/engine/detokenizer.py:L41
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        self.token_ids.extend(new_token_ids)
        return None

    # SOURCE: vllm/v1/engine/detokenizer.py:L45
    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer,
        request,
    ) -> "IncrementalDetokenizer":
        # SOURCE: vllm/v1/engine/detokenizer.py:L48
        # SUBTRACTED: USE_FAST_DETOKENIZER / FastIncrementalDetokenizer选择
        #   (tokenizers DecodeStream path is tokenizer-internal — subtraction_plan).
        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()
        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)


# SOURCE: vllm/v1/engine/detokenizer.py:L68
class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    def __init__(self, request):
        # SOURCE: vllm/v1/engine/detokenizer.py:L69
        super().__init__()

        # Stop strings
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""

    # SOURCE: vllm/v1/engine/detokenizer.py:L95
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1) Detokenize the new token ids incrementally.
        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # Support min_tokens.
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                stop_check_offset = len(self.output_text)

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            self.token_ids.append(skipped_stop_token_id)

        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        # SOURCE: vllm/v1/engine/detokenizer.py:L144
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/detokenizer.py:L148
    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            if not buffer_length:
                return self.output_text
            return self.output_text[:-buffer_length]

        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


# SOURCE: vllm/v1/engine/detokenizer.py:L245 SlowIncrementalDetokenizer
# SUBTRACTED: prefix_offset/read_offset prompt-token bookkeeping +
#   detokenize_incrementally + skip_special_tokens/spaces_between_special_tokens
#   + byte-fallback recovery — all tokenizer-internal (subtraction_plan: only the
#   normal decode path is kept). The injected ``decode`` callable stands in for
#   the real per-token tokenizer decode.
class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: Callable[[int], str], request):
        # SOURCE: vllm/v1/engine/detokenizer.py:L246
        super().__init__(request)
        self.tokenizer = tokenizer

    # SOURCE: vllm/v1/engine/detokenizer.py:L286
    def decode_next(self, next_token_id: int) -> str:
        return self.tokenizer(next_token_id)


# SOURCE: vllm/v1/engine/detokenizer.py:L304
def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.
    """
    if not new_char_count or not stop:
        return None

    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        if include_in_output:
            # Truncate to end of stop string.
            stop_index += stop_string_len
            if stop_index >= len(output_text):
                # No truncation required.
                return stop_str, -1

        # Truncate the output text to either the beginning
        # or end of the stop string.
        return stop_str, stop_index
    return None
