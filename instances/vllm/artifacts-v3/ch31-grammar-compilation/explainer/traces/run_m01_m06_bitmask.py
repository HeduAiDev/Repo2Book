# ch30 m1/m6/m8/m9/m21 驱动脚本：choice ["yes","no"] 经真实前端校验→
# xgrammar 编译→GrammarMatcher FSM→fill_bitmask 逐位置合法集→
# apply_token_bitmask_inplace 前后 argmax→accept/validate/rollback 语义。
# 环境：host Miniconda CPython 3.11.11 + xgrammar 0.2.6 + gpt2(50257) HF 本地缓存。
# 取证环境与 pin 的差异（explainer 铁律 exp-0718-1，须显式标注）：
#   - xgrammar==0.2.6 是 vLLM requirements/common.txt 钉版区间
#     (>=0.2.1,<1.0.0) 内的 Windows wheel；该版 GrammarMatcher 的
#     max_rollback_tokens kwarg 已弃用限制语义（构造发 DeprecationWarning、
#     内部恒无限回滚）——vLLM 传参行为与 pin 源码一致（backend_xgrammar.py:
#     L119-L123 原样传 num_speculative_tokens），库侧行为差异在 trace 中记录。
#   - tokenizer=gpt2(vocab 50257)，非 13 万级生产词表；128k 词表的
#     16KB/行账在 dossier theory[1]（源码公式 __init__.py:L232-L234）。
import json
import pathlib
import sys
import warnings

IMPL = pathlib.Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

import torch
import xgrammar as xgr
from transformers import AutoTokenizer

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.request import get_structured_output_key

TOKENIZER_NAME = "gpt2"
VOCAB = 50257
YES, NO, EOS, OFF_GRAMMAR = 8505, 3919, 50256, 4242  # gpt2: "yes","no",<|endoftext|>,词表内语法外

out = {}

tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
import importlib.metadata

out["env"] = {
    "tokenizer": TOKENIZER_NAME,
    "vocab_size": VOCAB,
    "xgrammar_version": importlib.metadata.version("xgrammar"),
    "torch_version": torch.__version__,
    "anchor_choice_ids": {"yes": YES, "no": NO, "eos": EOS, "off_grammar_4242_decoded": tok.decode([OFF_GRAMMAR])},
}

# ── 前端校验（站 1 真实路径：_validate_structured_outputs 会把 choice→EBNF 改写）──
# 这里直接给 EBNF grammar（改写本身在 run_m02_m04_params_key.py 里单测）。
sp = SamplingParams(
    structured_outputs=StructuredOutputsParams(grammar='root ::= "yes" | "no"'))
so = sp.structured_outputs
out["structured_output_key"] = {
    "repr": repr(get_structured_output_key(so)),
}

# ── 编译（站 4 真实路径：XgrammarBackend.compile_grammar 五分派的 GRAMMAR 分支）──
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend


def make_vllm_config(num_speculative_tokens=None):
    from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                             SpeculativeConfig, StructuredOutputsConfig, VllmConfig)
    return VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER_NAME, vocab_size=VOCAB),
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        scheduler_config=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=8192),
        structured_outputs_config=StructuredOutputsConfig(backend="auto"),
        speculative_config=None if num_speculative_tokens is None
        else SpeculativeConfig(num_speculative_tokens=num_speculative_tokens),
    )


backend = XgrammarBackend(make_vllm_config(), tokenizer=tok, vocab_size=VOCAB)
g = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, 'root ::= "yes" | "no"')
out["compile"] = {
    "matcher_type": type(g.matcher).__name__,
    "ctx_type": type(g.ctx).__name__,
    "num_speculative_tokens": backend.num_speculative_tokens,
    "max_rollback_tokens_passed": backend.num_speculative_tokens,  # backend_xgrammar.py:L119-L123
}

# spec 配置下编译 → 捕获 DeprecationWarning（max_rollback_tokens kwarg 已弃用限制语义）
spec_backend = XgrammarBackend(
    make_vllm_config(num_speculative_tokens=1), tokenizer=tok, vocab_size=VOCAB)
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    spec_backend.compile_grammar(
        StructuredOutputOptions.GRAMMAR, 'root ::= "yes" | "no"')
dep = [str(w.message) for w in wlist if issubclass(w.category, DeprecationWarning)]
out["compile"]["spec_deprecation_warning"] = dep[:1]

# ── 掩码分配与行布局（m21）──
bm_full = xgr.allocate_token_bitmask(16, VOCAB)  # 精简真实签名 allocate_token_bitmask(max_num_seqs, vocab)
out["bitmask_layout"] = {
    "rows_allocated": int(bm_full.shape[0]),
    "int32_per_row": int(bm_full.shape[1]),
    "vocab": VOCAB,
    "bytes_per_row": int(bm_full.shape[1] * 4),
    "bytes_total_16_rows": int(bm_full.numel() * 4),
    "ceil_50257_over_32": -(-VOCAB // 32),
}


def allowed_ids(row):
    return [t for t in range(VOCAB) if (int(row[t // 32]) >> (t % 32)) & 1]


def fill_and_decode(grammar, n_rows=1):
    bm = xgr.allocate_token_bitmask(n_rows, VOCAB)
    grammar.fill_bitmask(bm, 0)
    ids = allowed_ids(bm[0])
    return bm, ids


# ── m6：FSM 逐位置（轮次表数据）──
seq = []
# 位置 0
bm0, ids0 = fill_and_decode(g)
seq.append({
    "pos": 0,
    "allowed_count": len(ids0),
    "allowed_ids": ids0,
    "allowed_tokens": [repr(tok.decode([t])) for t in ids0],
    "accept_token": YES,
    "accept_returns": g.accept_tokens("req-1", [YES]),
    "num_processed_tokens_after": g.num_processed_tokens,
    "is_terminated": g.is_terminated(),
})
# 位置 0 反例：拒收不前进（新 matcher 保证从位置 0 开始）
g_bad = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, 'root ::= "yes" | "no"')
reject = {
    "accept_token": OFF_GRAMMAR,
    "decoded": repr(tok.decode([OFF_GRAMMAR])),
    "accept_returns": g_bad.accept_tokens("req-1", [OFF_GRAMMAR]),
    "num_processed_tokens_after": g_bad.num_processed_tokens,
}
# 位置 1：accept YES 之后
bm1, ids1 = fill_and_decode(g)
seq.append({
    "pos": 1,
    "allowed_count": len(ids1),
    "allowed_ids": ids1,
    "allowed_tokens": [repr(tok.decode([t])) for t in ids1],
    "accept_token": EOS,
    "accept_returns": g.accept_tokens("req-1", [EOS]),
    "num_processed_tokens_after": g.num_processed_tokens,
    "is_terminated": g.is_terminated(),
})
out["fsm_positions"] = {"sequence": seq, "reject_at_pos0": reject}

# ── m1：掩码前后 argmax 翻转（apply_token_bitmask_inplace = pin utils.py:L113-L161
#    调的同一个 xgrammar 库函数）──
logits = torch.full((1, VOCAB), -10.0)
logits[0, OFF_GRAMMAR] = 5.0   # 垃圾 token 得分最高
logits[0, YES] = 2.0
logits[0, NO] = 1.0
argmax_before = int(logits[0].argmax())
g_m1 = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, 'root ::= "yes" | "no"')
bm_m1 = xgr.allocate_token_bitmask(1, VOCAB)
g_m1.fill_bitmask(bm_m1, 0)  # 位置 0 的允许集
xgr.apply_token_bitmask_inplace(logits, bm_m1)
argmax_after = int(logits[0].argmax())
out["argmax_flip"] = {
    "logit_off_grammar_4242": 5.0,
    "logit_yes_8505": 2.0,
    "logit_no_3919": 1.0,
    "logit_fill_others": -10.0,
    "argmax_before_id": argmax_before,
    "argmax_before_token": repr(tok.decode([argmax_before])),
    "logit_4242_after_mask": float(logits[0, OFF_GRAMMAR]),  # -inf
    "argmax_after_id": argmax_after,
    "argmax_after_token": repr(tok.decode([argmax_after])),
    "mask_bit_convention": "bit=1 allowed; bit=0 -> -inf (xgrammar 约定，dossier WC1 errata)",
}

# ── m8：validate（试走不推进）vs accept（真推进）──
g_v = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, 'root ::= "yes" | "no"')
v1 = g_v.validate_tokens([YES, NO])
c1 = g_v.num_processed_tokens
v2 = g_v.validate_tokens([YES, NO])  # 幂等
a1 = g_v.accept_tokens("req-1", [YES])
c2 = g_v.num_processed_tokens
v3 = g_v.validate_tokens([NO])  # 位置 1 只允许 EOS
out["validate_vs_accept"] = {
    "validate_1_input": [YES, NO],
    "validate_1_returns": v1,
    "num_processed_after_validate": c1,
    "validate_repeat_returns": v2,
    "accept_yes_returns": a1,
    "num_processed_after_accept": c2,
    "validate_at_pos1_no_returns": v3,
}

# ── m9：rollback 与计数成对（'root ::= "a" "b" "c"'，gpt2 "a"=64 "b"=65 "c"=66）──
g_r = backend.compile_grammar(
    StructuredOutputOptions.GRAMMAR, 'root ::= "a" "b" "c"')
ids_abc = [64, 65, 66]
a_abc = g_r.accept_tokens("req-1", ids_abc)
c_abc = g_r.num_processed_tokens
g_r.rollback(1)
c_after_rb = g_r.num_processed_tokens
bm_rb, ids_rb = fill_and_decode(g_r)  # 回退到位置 2：应只允许 "c"
a_replay = g_r.accept_tokens("req-1", [66])
c_replay = g_r.num_processed_tokens
out["rollback"] = {
    "grammar": 'root ::= "a" "b" "c"',
    "token_ids": {"a": 64, "b": 65, "c": 66},
    "accept_abc_returns": a_abc,
    "num_processed_after_abc": c_abc,
    "rollback_n": 1,
    "num_processed_after_rollback": c_after_rb,
    "allowed_after_rollback_count": len(ids_rb),
    "allowed_after_rollback_tokens": [repr(tok.decode([t])) for t in ids_rb],
    "accept_c_replay_returns": a_replay,
    "num_processed_after_replay": c_replay,
}

# ── 表格行建议（供 explainer.json 直接复制，数字与本 trace 一致）──
out["table_rows_m6"] = [
    ["位置 0", "fill_bitmask", f"允许 {len(ids0)} 个 token：{ids0}",
     "全词表 50257 中仅 choice 前缀合法", f"accept(8505 'yes')={seq[0]['accept_returns']}"],
    ["位置 0 反例", "accept_token(4242)", f"num_processed_tokens={reject['num_processed_tokens_after']}",
     "词表内但语法外", f"accept 返回 {reject['accept_returns']}（拒收不前进）"],
    ["位置 1", "fill_bitmask", f"允许 {len(ids1)} 个 token：{ids1}",
     "串 'yes' 已完整，只差停机", f"accept(50256 EOS)={seq[1]['accept_returns']}"],
    ["终态", "is_terminated", f"num_processed_tokens={seq[1]['num_processed_tokens_after']}",
     "完整串+EOS", f"is_terminated={seq[1]['is_terminated']}"],
]

p = pathlib.Path(__file__).with_name("trace_m01_m06_bitmask.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("WROTE", p)
print(json.dumps({k: out[k] for k in ("structured_output_key", "bitmask_layout", "argmax_flip")},
                 ensure_ascii=False, indent=1))
print("pos0 allowed:", out["fsm_positions"]["sequence"][0]["allowed_count"],
      out["fsm_positions"]["sequence"][0]["allowed_tokens"])
print("pos1 allowed:", out["fsm_positions"]["sequence"][1]["allowed_count"],
      out["fsm_positions"]["sequence"][1]["allowed_tokens"])
print("validate:", out["validate_vs_accept"])
print("rollback:", json.dumps(out["rollback"], ensure_ascii=False))
