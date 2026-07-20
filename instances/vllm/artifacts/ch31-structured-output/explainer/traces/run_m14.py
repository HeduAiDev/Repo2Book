"""m14 is_terminated 的语义分歧：xgrammar 缓存标志位 vs guidance 的 EOS + rollback_lag。

跑精简版两个后端的真代码（outlines / lm-format-enforcer 两家按 subtraction_plan 批准项5
未纳入精简版，故不进本表，只在正文以源码行号引述）：
  - XgrammarGrammar：_is_terminated 是**缓存标志位**，只在 accept_tokens 末尾/rollback
    里刷新；一旦为真，后续 accept_tokens 直接短路返回 False。
  - GuidanceGrammar：看 tokens 里有没有 EOS；若此刻 matcher 已 stopped 且之前未终止，
    置 rollback_lag=1，于是随后的 rollback(n) 实际只退 n-1 格。
"""
from _fakes import FakeLLGuidanceModule, FakeXgrModule, dump
import backend_guidance as bg
import backend_xgrammar as bx
from backend_types import StructuredOutputOptions

EOS = 2
rows = []
raw = {"eos_token": EOS, "xgrammar": [], "guidance": []}

# ── xgrammar ────────────────────────────────────────────────────────────────
fake_xgr = FakeXgrModule()
bx.xgr = fake_xgr
xb = bx.XgrammarBackend(object(), tokenizer=object(), vocab_size=128)
xg = xb.compile_grammar(StructuredOutputOptions.REGEX, "[ab]+")
xg.matcher.terminating_token = EOS


def xsnap(action, ret):
    rows.append(["xgrammar", action, str(ret),
                 str(xg.matcher.is_terminated()), str(xg.is_terminated()),
                 str(xg.num_processed_tokens)])
    raw["xgrammar"].append({
        "action": action, "returned": ret,
        "matcher_is_terminated": xg.matcher.is_terminated(),
        "grammar_is_terminated": xg.is_terminated(),
        "num_processed_tokens": xg.num_processed_tokens,
    })


xsnap("accept_tokens([31])", xg.accept_tokens("r", [31]))
xsnap("accept_tokens([2])  # 2 = EOS，底层 matcher 进终态",
      xg.accept_tokens("r", [EOS]))
xsnap("accept_tokens([32]) # 终态短路，连 matcher 都不碰",
      xg.accept_tokens("r", [32]))
xsnap("rollback(1) # 标志位随之刷新", xg.rollback(1))

# ── guidance ────────────────────────────────────────────────────────────────
fake_llg = FakeLLGuidanceModule()
bg.llguidance = fake_llg
gb = bg.GuidanceBackend(object(), tokenizer=["tok"] * 128, vocab_size=128)
gg = gb.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')
gg.ll_tokenizer.eos_token = EOS


def gsnap(action, ret):
    rows.append(["guidance", action, str(ret),
                 str(gg.ll_matcher.is_stopped()), str(gg.is_terminated()),
                 f"rollback_lag={gg.rollback_lag}"])
    raw["guidance"].append({
        "action": action, "returned": ret,
        "ll_matcher_is_stopped": gg.ll_matcher.is_stopped(),
        "grammar_is_terminated": gg.is_terminated(),
        "rollback_lag": gg.rollback_lag,
        "consumed": list(gg.ll_matcher.consumed),
        "rollback_calls": list(gg.ll_matcher.rollback_calls),
    })


gsnap("accept_tokens([41, 42])", gg.accept_tokens("r", [41, 42]))
gg.ll_matcher.stopped = True     # 语法此刻已走到可停位置
gsnap("accept_tokens([2])  # EOS 且 matcher 已 stopped → rollback_lag 置 1",
      gg.accept_tokens("r", [EOS]))
gsnap("rollback(2) # 实际只退 2-1=1 格", gg.rollback(2))

raw["guidance_rollback_calls_on_ll_matcher"] = list(gg.ll_matcher.rollback_calls)
raw["outlines_semantics"] = (
    "backend_outlines.py:L155-160 —— is_terminated 返回**上一次**的 is_finished()，"
    "故意延迟一步，好让 EOS 还能发出去（精简版未含 outlines，见 subtraction_plan 批准项5）"
)
raw["lm_format_enforcer_semantics"] = (
    "backend_lm_format_enforcer.py:L81-88 —— 看 current_tokens_prefix 末位是不是 EOS"
)

dump("m14.json", {
    "mechanism": "m14-terminated-semantics-divergence",
    "columns": ["后端", "动作", "返回", "底层库的终止判定", "is_terminated()", "附加状态"],
    "rows": rows,
    "raw": raw,
})
