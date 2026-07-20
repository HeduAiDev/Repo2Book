# SOURCE: vllm/v1/structured_output/utils.py
# 只做减法的忠实精简版——真实文件另有 apply_grammar_bitmask（批量装配，下一章）、
# OutlinesVocabulary/get_outlines_cache*（outlines 后端专用，批准项5整体删除）、
# grammar_is_likely_lark/convert_lark_to_ebnf（Lark 方言转换细节，批准项8删除）。
# 精简版只保留 choice_as_grammar——choice → EBNF 改写链条的落点（与
# backend_xgrammar.validate_xgrammar_grammar 同进退，dossier must_keep）。
#
# SUBTRACTED: SPDX 版权头；`import regex as re`（第三方 regex 包，仅为兼容更复杂的
# 正则语法，这里用到的转义模式标准库 re 完全等价）改用标准库 re。
import re


def choice_as_grammar(choice: list[str]) -> str:
    # SOURCE: vllm/v1/structured_output/utils.py:L451-459
    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string."""
        # Escape double quotes and backslashes
        return re.sub(r'(["\\])', r"\\\1", s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
    return grammar
