"""m15 worked-example driver — the OpenAI three-field record
(serving.py:L1140-L1165 _get_top_logprobs + L1167-L1231 _create_chat_logprobs):
token / logprob (clamped at -9999.0) / bytes = list(token.encode('utf-8')).
The corrected-empty token ('' after U+FFFD repair moved the text to a later
position) yields an EMPTY bytes list — bytes always tells the byte truth.
Also: top_logprobs truncation, the missing-step fallback (decode + default
logprob), the logprob_token_ids return_all mode, and return_as_token_id.
"""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import logprobs_lane as lane  # noqa: E402
import tokenizers  # noqa: E402
import tokenizers.decoders  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def R(x, nd=4):
    return round(float(x), nd)


class ByteFallbackFace:
    def __init__(self, tk):
        self._tk = tk
        self.backend_tokenizer = tk

    def decode(self, ids):
        return self._tk.decode([ids] if isinstance(ids, int) else list(ids))


def byte_fallback_tokenizer():
    vocab = {f"<0x{i:02X}>": i for i in range(256)}
    vocab["hello"] = 256
    vocab[" world"] = 257
    tk = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token=None)
    )
    tk.decoder = tokenizers.decoders.ByteFallback()
    return ByteFallbackFace(tk)


def content_row(c):
    """ChatCompletionLogProbsContent -> plain dict."""
    row = {"token": c.token, "logprob": R(c.logprob), "bytes": c.bytes}
    if c.top_logprobs is not None:
        row["top_logprobs"] = [
            {
                "token": t.token,
                "logprob": R(t.logprob),
                "bytes": t.bytes,
            }
            for t in c.top_logprobs
        ]
    return row


def main():
    tk = byte_fallback_tokenizer()
    serving = lane.OpenAIServingChat()

    # The completing position of the 中 example (m8): sampled 173 decoded
    # '中' with an empty-string alternative 228, plus a plain candidate.
    step = {
        173: lane.Logprob(-0.05, 1, "中"),
        228: lane.Logprob(-0.25, 2, ""),
        256: lane.Logprob(-0.9, 3, "hello"),
    }
    out1 = serving._create_chat_logprobs(
        token_ids=[173],
        top_logprobs=[step],
        tokenizer=tk,
        num_output_top_logprobs=1,
    )
    # clamp: a genuinely tiny probability below the OpenAI floor.
    step2 = {65: lane.Logprob(-12345.6, 5, "A")}
    out2 = serving._create_chat_logprobs(
        token_ids=[65],
        top_logprobs=[step2],
        tokenizer=tk,
        num_output_top_logprobs=2,
    )
    # missing step (logprobs=None step, e.g. detokenize-off request): the
    # fallback decodes the token and uses the -9999.0 default logprob.
    out3 = serving._create_chat_logprobs(
        token_ids=[257],
        top_logprobs=[None],
        tokenizer=tk,
        num_output_top_logprobs=1,
    )
    # logprob_token_ids mode: return_all ignores the truncation.
    out4 = serving._create_chat_logprobs(
        token_ids=[173],
        top_logprobs=[step],
        tokenizer=tk,
        num_output_top_logprobs=1,
        logprob_token_ids=[173, 228, 256],
    )
    # return_as_token_id face.
    out5 = serving._create_chat_logprobs(
        token_ids=[173],
        top_logprobs=[step],
        tokenizer=tk,
        num_output_top_logprobs=1,
        return_as_token_id=True,
    )

    trace = {
        "mechanism": "m15",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "vocab_notes": "173=<0xAD> 228=<0xE4> 256=hello 257=' world' 65='A'",
            "step_dict": "173:'中'(-0.05) 228:''(-0.25) 256:'hello'(-0.9)",
            "case2_input_logprob": -12345.6,
        },
        "case1_bytes_truth": content_row(out1.content[0]),
        "case2_clamp": content_row(out2.content[0]),
        "case3_missing_step_fallback": content_row(out3.content[0]),
        "case4_return_all_sparse_mode": content_row(out4.content[0]),
        "case5_return_as_token_id": content_row(out5.content[0]),
        "notes": {
            "clamp": "max(logprob, -9999.0) — OpenAI JSON cannot carry -inf/-12345.6",
            "empty_bytes": "the corrected-empty token text is '' -> bytes=[] (empty list): the char's bytes were attributed to the completing token",
            "zhong_bytes": "'中'.encode('utf-8') == [228, 184, 173]",
        },
    }

    p = Path(__file__).resolve().parent / "m15_openai_bytes.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
