"""m07 accept_tokens（推进）与 validate_tokens（试走再回退）的语义分离。

跑精简版 XgrammarGrammar 真代码 + 调度器两个真实调用点
（Scheduler._validate_spec_tokens_against_grammar → validate_tokens，
  Scheduler._advance_grammar_on_sampled_tokens → accept_tokens）。
matcher 把 token 99 设为「不被语法接受」，用来看出前缀截断与失败短路。
"""
from _fakes import FakeXgrModule, dump
import backend_xgrammar as bx
from backend_types import StructuredOutputOptions
from scheduler import Scheduler

fake = FakeXgrModule()
bx.xgr = fake

backend = bx.XgrammarBackend(object(), tokenizer=object(), vocab_size=128)
grammar = backend.compile_grammar(StructuredOutputOptions.REGEX, "[ab]+")
matcher = grammar.matcher
matcher.reject_token = 99          # 99 是语法不接受的 token


class Req:
    request_id = "req-0"
    use_structured_output = True

    class structured_output_request:
        pass


req = Req()
req.structured_output_request = type("SOR", (), {"grammar": grammar})()

rows = []
raw = {"reject_token": 99, "steps": []}


def snap(caller, action, ret):
    rows.append([
        caller,
        action,
        str(ret),
        str(grammar.num_processed_tokens),
        str(len(matcher.accepted)),
        str(len(matcher.rollback_calls)),
    ])
    raw["steps"].append({
        "caller": caller,
        "action": action,
        "returned": ret,
        "num_processed_tokens": grammar.num_processed_tokens,
        "matcher_accepted_len": len(matcher.accepted),
        "matcher_accepted": list(matcher.accepted),
        "rollback_calls": list(matcher.rollback_calls),
    })


r = Scheduler._validate_spec_tokens_against_grammar(req, [31, 32, 99])
snap("scheduler.py:L1617-1621", "validate_tokens([31, 32, 99]) 试走草稿", r)

r = Scheduler._advance_grammar_on_sampled_tokens(req, [31, 32])
snap("scheduler.py:L1360-1369", "accept_tokens([31, 32]) 真推进（验收通过的 token）",
     "None(未抛错)")

r = Scheduler._validate_spec_tokens_against_grammar(req, [99, 31])
snap("scheduler.py:L1617-1621", "validate_tokens([99, 31]) 首个就被拒", r)

r = grammar.accept_tokens("req-0", [99])
snap("XgrammarGrammar", "accept_tokens([99]) 推进失败（真的走不通）", r)

raw["note"] = (
    "validate_tokens 先真 accept 再 rollback(k)，等价于『不推进』——第 1 行前后 "
    "num_processed_tokens 与 matcher 已接受数都回到 0，且多了一次 rollback 调用；"
    "第 3 行首 token 即被拒，k=0，源码 `if len(accepted_tokens) > 0` 决定不发 rollback。"
)

dump("m07.json", {
    "mechanism": "m07-accept-vs-validate",
    "columns": ["调用点", "动作", "返回", "num_processed_tokens",
                "matcher 已接受 token 数", "累计 rollback 调用次数"],
    "rows": rows,
    "raw": raw,
})
