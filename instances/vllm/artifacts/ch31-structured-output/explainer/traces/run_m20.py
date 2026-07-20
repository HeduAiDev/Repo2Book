"""m20 校验期不只是选后端，还会**原地改写请求**：choice → EBNF grammar。

跑精简版真代码：SamplingParams._validate_structured_outputs（backend='auto'）→
validate_xgrammar_grammar（backend_xgrammar.py:L286-295 的 choice 分支）→
so_params.choice = None; so_params.grammar = choice_as_grammar(...)。
观察改写前后 params 字段与 structured_output_key 的变化，并确认 xgrammar 的
compile_grammar 对 CHOICE 枚举根本没有分支（走 else → ValueError）。
"""
from _fakes import FakeXgrModule, dump
import backend_xgrammar as bx
from backend_types import StructuredOutputOptions
from sampling_params import (SamplingParams, StructuredOutputsConfig,
                             StructuredOutputsParams)
from so_request import get_structured_output_key

fake = FakeXgrModule()
bx.xgr = fake

so = StructuredOutputsParams(choice=["red", "blue"])
sp = SamplingParams(max_tokens=16, structured_outputs=so)

rows = []
raw = {"steps": []}


def snap(stage):
    opt, spec = get_structured_output_key(so)
    rows.append([
        stage,
        repr(so.choice),
        repr(so.grammar),
        f"{opt.name}(={opt.value})",
        repr(spec),
        repr(so._backend),
    ])
    raw["steps"].append({
        "stage": stage, "choice": so.choice, "grammar": so.grammar,
        "key_option": opt.name, "key_option_value": opt.value,
        "key_spec": spec, "backend": so._backend,
        "backend_was_auto": so._backend_was_auto,
    })


snap("用户提交（前端校验前）")
sp._validate_structured_outputs(StructuredOutputsConfig(backend="auto"),
                                tokenizer=object())
snap("_validate_structured_outputs 之后（引擎侧看到的就是这份）")

# 引擎侧：CHOICE 枚举在 xgrammar 的 compile_grammar 里没有分支
backend = bx.XgrammarBackend(object(), tokenizer=object(), vocab_size=128)
try:
    backend.compile_grammar(StructuredOutputOptions.CHOICE, '["red", "blue"]')
    choice_branch = "存在"
    err = None
except ValueError as e:
    choice_branch = "不存在（落 else → ValueError）"
    err = str(e)

# 改写后的 (GRAMMAR, ebnf) 才是真正被编译的东西
opt, spec = get_structured_output_key(so)
grammar_obj = backend.compile_grammar(opt, spec)
rows.append([
    "引擎侧 compile_grammar 实际分派",
    "—", "—",
    f"{opt.name}(={opt.value})",
    repr(spec),
    f"compiler 分支 = {grammar_obj.ctx.tag}",
])
raw["compile_dispatch"] = {
    "choice_branch_in_compile_grammar": choice_branch,
    "error_message_for_CHOICE": err,
    "dispatched_branch_for_rewritten_key": grammar_obj.ctx.tag,
    "compiler_calls": [(c.tag if hasattr(c, "tag") else c) for c in []],
    "num_branches_in_compile_grammar": 5,
    "num_enum_members": len(StructuredOutputOptions),
}
raw["note"] = (
    "六个入口枚举，引擎侧 compile_grammar 只有 5 个分支——差的那个是 CHOICE，"
    "因为它在校验期就被改写成 GRAMMAR 了。"
)

dump("m20.json", {
    "mechanism": "m20-validation-rewrites-request",
    "columns": ["阶段", "params.choice", "params.grammar", "structured_output_key 枚举",
                "structured_output_key 规格串", "_backend / 分派结果"],
    "rows": rows,
    "raw": raw,
})
