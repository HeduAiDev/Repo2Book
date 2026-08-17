"""Driver for m6 (params clone & completion) — host run against the ch06
subtract-only companion (pin vLLM v0.27.1).

Worked example around InputProcessor.process_inputs L320-L339 with a small
config (max_model_len=32) and a 3-token prompt:
- round 1: max_tokens unset -> clone gets max_tokens = 32 - 3 = 29; the
  CALLER's params object is untouched (None) — the clone isolates the two
  sides (the TODO at L323 pays this clone tax in multiproc mode);
- round 2: generation_config eos_token_id [2, 5] -> _eos_token_id=2,
  _all_stop_token_ids={2,5}, stop_token_ids=[5] on the clone; caller still
  clean;
- round 3: bad_words expansion — with a word tokenizer a leading space
  produces no new token (dedup -> 1 entry); with a prefix-space tokenizer
  (real BPE behavior: " word" != "word") both variants are kept (2 entries);
- round 4: an EXPLICIT max_tokens=5 is left alone (default only fills unset);
- round 5: pooling params clone path (pooling_params cloned, sampling None).
"""
import importlib
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_downlink")
downlink = td.downlink


class PrefixSpaceTokenizer(td.SeamWordTokenizer):
    """Word tokenizer where a leading space DOES produce a distinct first
    token — mirroring real BPE/HF behavior where " word" and "word" tokenize
    differently (the add_prefix_space parameter update_from_tokenizer exists
    for, sampling_params.py:L676-L715)."""

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" ") and text.strip():
            first, *rest = text.split()
            return [self._word_id("▁" + first)] + [
                self._word_id(w) for w in rest
            ]
        return super().encode(text, add_special_tokens)


def process(llm, params, token_ids=(3, 4, 5), supported_tasks=("generate",)):
    return llm.input_processor.process_inputs(
        "req-m6",
        downlink.tokens_input(list(token_ids)),
        params,
        supported_tasks=supported_tasks,
    )


def snap(p):
    return {
        "max_tokens": p.max_tokens,
        "eos_token_id": p._eos_token_id,
        "all_stop_token_ids": sorted(p._all_stop_token_ids),
        "stop_token_ids": list(p.stop_token_ids),
        "bad_words": list(p.bad_words),
        "bad_words_token_ids": (
            [list(x) for x in p._bad_words_token_ids]
            if p._bad_words_token_ids is not None else None
        ),
    }


def main():
    out = {
        "driver": "run_m6_params_completion.py",
        "mechanism": "m6 params.clone() 补全（max_tokens 默认 / eos 注入 / bad_words 展开）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "max_model_len": 32,
        "prompt_token_ids": [3, 4, 5],
        "seq_len": 3,
        "tokenizer_eos_token_id": 2,
    }

    # -- rounds 1+2: defaults fill + eos injection, caller isolated ----------
    cfg = td.mk_config(
        model_config={
            "max_model_len": 32,
            "generation_config_fields": {"eos_token_id": [2, 5]},
        }
    )
    llm, renderer, tok = td.mk_engine(td.SeamWordTokenizer(), config=cfg)

    p1 = downlink.SamplingParams(max_tokens=None)
    req1 = process(llm, p1)
    out["round1_max_tokens_default"] = {
        "action": "process_inputs(params=SamplingParams(max_tokens=None)), prompt 3 token",
        "caller_before": snap(p1),
        "clone_after": snap(req1.sampling_params),
        "max_tokens_default_filled": req1.sampling_params.max_tokens,
        "formula": "max_model_len - seq_len = 32 - 3",
        "formula_result": 32 - 3,
        "caller_object_mutated": p1.max_tokens is not None,
        "clone_is_distinct_object": req1.sampling_params is not p1,
    }

    out["round2_eos_injection"] = {
        "action": "generation_config eos_token_id=[2, 5] + tokenizer eos=2",
        "generation_config_fields": cfg.model_config.generation_config_fields,
        "clone_eos_token_id": req1.sampling_params._eos_token_id,
        "clone_all_stop_token_ids": sorted(req1.sampling_params._all_stop_token_ids),
        "clone_stop_token_ids": req1.sampling_params.stop_token_ids,
        "caller_eos_token_id": p1._eos_token_id,
        "caller_all_stop_token_ids": sorted(p1._all_stop_token_ids),
        "caller_untouched": p1._eos_token_id is None,
    }

    # -- round 3: bad_words expansion, two tokenizer behaviors ----------------
    p2 = downlink.SamplingParams(max_tokens=4, bad_words=["stopword"])
    req2 = process(llm, p2)
    plain = {
        "tokenizer": "词级（前缀空格不产生新 token）",
        "bad_words_token_ids": [list(x) for x in req2.sampling_params._bad_words_token_ids],
        "entries_kept": len(req2.sampling_params._bad_words_token_ids),
        "caller_bad_words_token_ids": p2._bad_words_token_ids,
    }

    cfg_ps = td.mk_config(
        model_config={
            "max_model_len": 32,
            "generation_config_fields": {"eos_token_id": [2, 5]},
        }
    )
    llm_ps, _, _ = td.mk_engine(PrefixSpaceTokenizer(), config=cfg_ps)
    p3 = downlink.SamplingParams(max_tokens=4, bad_words=["stopword"])
    req3 = process(llm_ps, p3)
    prefix = {
        "tokenizer": "前缀空格产生新 token（真实 BPE 行为）",
        "bad_words_token_ids": [list(x) for x in req3.sampling_params._bad_words_token_ids],
        "entries_kept": len(req3.sampling_params._bad_words_token_ids),
        "first_variants_differ": (
            req3.sampling_params._bad_words_token_ids[0]
            != req3.sampling_params._bad_words_token_ids[1]
        ),
    }
    out["round3_bad_words_expansion"] = {
        "action": 'params.bad_words=["stopword"] → update_from_tokenizer 两种变体（带/不带前缀空格）',
        "word_tokenizer": plain,
        "prefix_space_tokenizer": prefix,
    }

    # -- round 4: explicit max_tokens is left alone -----------------------------
    p4 = downlink.SamplingParams(max_tokens=5)
    req4 = process(llm, p4)
    out["round4_explicit_max_tokens_kept"] = {
        "action": "process_inputs(params=SamplingParams(max_tokens=5))",
        "clone_after_max_tokens": req4.sampling_params.max_tokens,
        "explicit_value_preserved": req4.sampling_params.max_tokens == 5,
    }

    # -- round 5: pooling clone path --------------------------------------------
    llm_pool, _, _ = td.mk_engine(
        td.SeamWordTokenizer(), config=cfg, supported_tasks=("embed",)
    )
    p5 = downlink.PoolingParams()
    req5 = process(llm_pool, p5, supported_tasks=("embed",))
    out["round5_pooling_clone"] = {
        "action": "process_inputs(params=PoolingParams(), supported_tasks=(\"embed\",))",
        "pooling_params_cloned": req5.pooling_params is not None,
        "clone_is_distinct_object": req5.pooling_params is not p5,
        "sampling_params_none": req5.sampling_params is None,
    }

    dest = Path(__file__).resolve().parent / "m6_params_completion.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
