# SOURCE: vllm/v1/structured_output/backend_xgrammar.py
# 只做减法的忠实精简版。ch31 已讲透 XgrammarBackend 的完整编译分派（五个 if/elif 分支）
# 与 XgrammarGrammar 六方法；本章只回指两处与本章 payoff 直接相关的落点：
# (1) allocate_token_bitmask——StructuredOutputManager.grammar_bitmask 第一次调用时
#     用它按行数预算分配缓冲；(2) compile_grammar 构造 xgr.GrammarMatcher 时把
#     max_rollback_tokens 钉死为 num_speculative_tokens——这是「回滚步数为什么有
#     上界」的源头（本章 m05 串行填充的 rollback(state_advancements) 之所以合法，
#     正是因为这个上界够用）。
#
# SUBTRACTED: SPDX 版权头、compile_grammar 里产出 ctx 的五个分支（JSON/JSON_OBJECT/
# REGEX/GRAMMAR/STRUCTURAL_TAG 的 xgr.Grammar.from_* 调用与 Mistral 特判，
# backend_xgrammar.py:L79-114）、validate_xgrammar_grammar /
# has_xgrammar_unsupported_json_features 两个校验函数、XgrammarGrammar 的六方法实现——
# 全部已在 ch31 §31.4-§31.6 逐段讲过，本章不重复实现。compile_grammar 本体因此拆成
# 独立方法 compile_grammar_tail（只保留原函数最后一段，ctx 作为已构造好的入参传入），
# 与 ch31 Source Map 里 `_advance_grammar_on_sampled_tokens` 同一惯例：抽取真实源码里
# 同一函数的一段、改名以便独立测试，控制流逐字不变。
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    try:
        import xgrammar as xgr
    except ImportError:  # host 未装 xgrammar，见 dossier.analyst_notes_on_plan
        xgr = None


@dataclass
class XgrammarGrammar:
    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L148-152（字段声明；
    # 六方法实现属 ch31，此处不重复，只作 compile_grammar_tail 的返回值容器）
    matcher: "object"
    vocab_size: int
    ctx: "object"


@dataclass
class XgrammarBackend:
    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L40-76（精简到本章所需字段）
    #
    # SUBTRACTED: vllm_config/tokenizer/vocab_size 的推导过程（tokenizer_group、
    # encoded_vocab 等，backend_xgrammar.py:L40-76）——ch31 已讲；这里直接接受
    # 已算好的 num_speculative_tokens 与 vocab_size。
    vocab_size: int
    num_speculative_tokens: int = 0

    def compile_grammar_tail(self, ctx: "object") -> XgrammarGrammar:
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L115-122
        # 只保留 compile_grammar 的最后一段——不管前面走的是哪个 if/elif 分支
        # （ch31 范围）产出 ctx，构造 matcher 时都统一把 max_rollback_tokens 设成
        # num_speculative_tokens：一步内最多试探性推进这么多步，语法后端的
        # 回滚缓冲要按这个数预先开好。
        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L124-125
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)
