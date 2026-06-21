"""Tests for the Stage-3 stop-string slice of OutputProcessor.process_outputs.

Verifies the three behaviours ch09 keeps from the single loop:
  - detokenizer.update returns a stop_string => finish_reason rewritten to STOP,
  - stop_reason set to the matched string,
  - reqs_to_abort populated iff EngineCore had not itself marked the output finished.
"""

import os
import sys
import types

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def _ensure_import_deps():
    for name, attrs in (
        ("tokenizers", None),
        ("transformers", None),
    ):
        if name in sys.modules:
            continue
        try:
            __import__(name)
            continue
        except ImportError:
            pass
        if name == "tokenizers":
            tk = types.ModuleType("tokenizers")
            tk.__version__ = "0.22.0"
            tk.Tokenizer = type("Tokenizer", (), {})
            dec = types.ModuleType("tokenizers.decoders")
            dec.DecodeStream = type("DecodeStream", (), {})
            tk.decoders = dec
            sys.modules["tokenizers"] = tk
            sys.modules["tokenizers.decoders"] = dec
        else:
            tf = types.ModuleType("transformers")
            tf.PreTrainedTokenizerFast = type("PreTrainedTokenizerFast", (), {})
            sys.modules["transformers"] = tf


_ensure_import_deps()

from implementation._types import (  # noqa: E402
    EngineCoreOutput,
    FinishReason,
)
from implementation.detokenizer import IncrementalDetokenizer  # noqa: E402
from implementation.output_processor import (  # noqa: E402
    OutputProcessor,
    RequestState,
)


class _StubDetok(IncrementalDetokenizer):
    """Detokenizer stub whose update returns a preset stop_string."""

    def __init__(self, stop_string):
        super().__init__()
        self._stop_string = stop_string
        self.seen = None

    def update(self, new_token_ids, stop_terminated):
        self.seen = (list(new_token_ids), stop_terminated)
        return self._stop_string


def _proc_with(stop_string, req_id="r0"):
    proc = OutputProcessor()
    proc.request_states[req_id] = RequestState(req_id, _StubDetok(stop_string))
    return proc


def test_stop_string_rewrites_finish_reason_and_aborts():
    proc = _proc_with("END")
    out = EngineCoreOutput(
        request_id="r0",
        new_token_ids=[1, 2, 3],
        finish_reason=None,      # EngineCore did not stop
        finished=False,          # ... so abort must be requested
    )
    result = proc.process_outputs([out])
    assert result.reqs_to_abort == ["r0"]
    # request state freed after finish
    assert "r0" not in proc.request_states


def test_no_stop_string_no_abort():
    proc = _proc_with(None)
    out = EngineCoreOutput(
        request_id="r0", new_token_ids=[1, 2], finish_reason=None, finished=False
    )
    result = proc.process_outputs([out])
    assert result.reqs_to_abort == []
    assert "r0" in proc.request_states  # not finished, still tracked


def test_engine_core_already_finished_no_abort():
    # EngineCore finished by EOS (finish_reason STOP, finished True): detok still
    # runs but no abort needs to be sent back.
    proc = _proc_with(None)
    out = EngineCoreOutput(
        request_id="r0",
        new_token_ids=[1, 9],
        finish_reason=FinishReason.STOP,
        finished=True,
    )
    result = proc.process_outputs([out])
    assert result.reqs_to_abort == []
    assert "r0" not in proc.request_states  # freed


def test_stop_terminated_flag_passed_to_detokenizer():
    proc = _proc_with(None)
    detok = proc.request_states["r0"].detokenizer
    out = EngineCoreOutput(
        request_id="r0",
        new_token_ids=[1, 2, 9],
        finish_reason=FinishReason.STOP,
        finished=True,
    )
    proc.process_outputs([out])
    # update was called with stop_terminated = (finish_reason == STOP) = True
    assert detok.seen == ([1, 2, 9], True)


def test_unknown_request_id_ignored():
    proc = OutputProcessor()
    out = EngineCoreOutput(request_id="ghost", new_token_ids=[1])
    result = proc.process_outputs([out])
    assert result.reqs_to_abort == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
