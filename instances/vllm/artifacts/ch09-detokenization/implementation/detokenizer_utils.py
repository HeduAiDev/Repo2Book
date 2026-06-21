"""Slow-path incremental detokenization helpers — subtract-only port of
``vllm/tokenizers/detokenizer_utils.py``.

Reproduced verbatim (names / structure / control flow identical to real vLLM):
  - convert_prompt_ids_to_tokens : initial prefix_offset / read_offset priming.
  - detokenize_incrementally     : the prefix/read double-window algorithm that
                                   defeats the tokenizer space-cleanup pass and
                                   guards the UTF-8 byte-fallback boundary.

Per subtraction_plan only the non-fast-with-added-vocab ``else`` branch
(``_convert_tokens_to_string_with_added_encoders``) and the unused
``convert_ids_list_to_tokens`` helper are removed — both are behaviour-equivalent
detours that never change the offset / UTF-8 logic this chapter studies.
"""

from __future__ import annotations

from ._types import TokenizerLike


# SOURCE: vllm/tokenizers/detokenizer_utils.py:L8
def _replace_none_with_empty(tokens: list[str | None]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""


# SUBTRACTED: _convert_tokens_to_string_with_added_encoders
#   (vllm/tokenizers/detokenizer_utils.py:L14-L51) — the non-fast tokenizer with
#   added-vocab path. It splits added tokens out and re-joins, but produces a
#   string behaviour-equivalent to convert_tokens_to_string for the offset/UTF-8
#   logic studied here. Companion takes the fast / no-added-vocab branch only
#   (subtraction_plan.delete).

# SOURCE: vllm/tokenizers/detokenizer_utils.py:L54
# 5 is an arbitrary value that should work for all
# tokenizers (bigger = more conservative).
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5


# SOURCE: vllm/tokenizers/detokenizer_utils.py:L59
def convert_prompt_ids_to_tokens(
    tokenizer: TokenizerLike,
    prompt_ids: list[int],
    skip_special_tokens: bool = False,
) -> tuple[list[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :],
        skip_special_tokens=skip_special_tokens,
    )
    read_offset = len(new_tokens)
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    return new_tokens, prefix_offset, read_offset


# SUBTRACTED: convert_ids_list_to_tokens
#   (vllm/tokenizers/detokenizer_utils.py:L83-L104) — a per-id decode helper that
#   is never reached from the detokenizer call chain in this chapter
#   (subtraction_plan.delete: "本层级未用").


# SOURCE: vllm/tokenizers/detokenizer_utils.py:L110
# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: TokenizerLike,
    all_input_ids: list[int],
    prev_tokens: list[str] | None,
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> tuple[list[str], str, int, int]:
    # SOURCE: vllm/tokenizers/detokenizer_utils.py:L110
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.
    """
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset, read_offset) = convert_prompt_ids_to_tokens(
            tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens
        )
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if 0 <= new_token_id < len(tokenizer):
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens
        )
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        else:
            # This is required to guard against out-of-vocab prompt token ids
            # (for example when using dummy weights)
            _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    else:
        new_tokens = [""]
    output_tokens = prev_tokens + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    # SUBTRACTED: the `else` branch (non-fast tokenizer with added vocab calling
    #   _convert_tokens_to_string_with_added_encoders) — companion keeps the
    #   fast / no-added-vocab path, which is behaviour-equivalent for offsets and
    #   UTF-8 (subtraction_plan.delete). The `if` guard is preserved verbatim.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset]
        )
        new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        return new_tokens, "", prefix_offset, read_offset

    new_text = new_text[len(prefix_text) :]
    return new_tokens, new_text, read_offset, len(output_tokens)
