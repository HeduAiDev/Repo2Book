"""m06 rollback 是为投机解码留的口子：max_rollback_tokens = num_speculative_tokens。

跑精简版 XgrammarBackend/XgrammarGrammar 真代码：speculative_config.num_speculative_tokens
= 3 → GrammarMatcher(max_rollback_tokens=3)；然后走一遍「推进 → 草稿被拒 → 回退」，
观察 num_processed_tokens 与底层 matcher 已接受序列长度**成对**回退。
"""
from _fakes import FakeXgrModule, dump
import backend_xgrammar as bx
from backend_types import StructuredOutputOptions

fake = FakeXgrModule()
bx.xgr = fake


class Cfg:
    class speculative_config:
        num_speculative_tokens = 3


backend = bx.XgrammarBackend(Cfg(), tokenizer=object(), vocab_size=128)
grammar = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, 'root ::= "ab"')
matcher = grammar.matcher

rows = []
raw = {
    "num_speculative_tokens": backend.num_speculative_tokens,
    "matcher_max_rollback_tokens": matcher.max_rollback_tokens,
}


def snap(action, ret=None):
    rows.append([
        action,
        str(ret),
        str(grammar.num_processed_tokens),
        str(len(matcher.accepted)),
        str(matcher.max_rollback_tokens),
    ])
    raw.setdefault("steps", []).append({
        "action": action,
        "returned": ret,
        "num_processed_tokens": grammar.num_processed_tokens,
        "matcher_accepted_len": len(matcher.accepted),
        "matcher_accepted": list(matcher.accepted),
    })


snap("建好 grammar（尚未喂任何 token）")
r = grammar.accept_tokens("req-0", [11, 12])
snap("accept_tokens([11, 12])：已确认的真 token 推进状态机", r)
r = grammar.accept_tokens("req-0", [21, 22, 23])
snap("accept_tokens([21, 22, 23])：3 个投机草稿 token 也先推进", r)
grammar.rollback(2)
snap("验收拒掉后 2 个草稿 → rollback(2)", None)
grammar.rollback(3)
snap("再 rollback(3)：正好触到 max_rollback_tokens 上限", None)

raw["rollback_calls_on_matcher"] = matcher.rollback_calls
raw["lm_format_enforcer_rejects_spec_decode"] = (
    "backend_lm_format_enforcer.py:L120-129 —— max_rollback_tokens > 0 时直接 "
    "raise ValueError('LM Format Enforcer backend does not support speculative tokens')"
)
raw["invariant_check_num_processed_equals_accepted_len"] = all(
    s["num_processed_tokens"] == s["matcher_accepted_len"] for s in raw["steps"]
)

dump("m06.json", {
    "mechanism": "m06-rollback-for-spec-decode",
    "columns": ["动作", "返回", "num_processed_tokens", "matcher 已接受 token 数",
                "max_rollback_tokens"],
    "rows": rows,
    "raw": raw,
})
