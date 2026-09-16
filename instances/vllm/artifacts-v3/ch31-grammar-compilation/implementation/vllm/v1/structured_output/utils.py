# SOURCE: vllm/v1/structured_output/utils.py
# 只做减法的忠实精简版（pin v0.27.1 / 6e448d0ea）。保留四件：ReDoS 超时护栏
# （m19）与校验期改写工具箱（choice→EBNF / lark 判定 / lark→EBNF，m4）。
# SUBTRACTED: SPDX 版权头；`from __future__ import annotations` 之外的其余
#   import（hashlib/importlib.metadata/os/sqlite3/tempfile/torch/cachetools/
#   GrammarOutput·SchedulerOutput/PIN_MEMORY·async_tensor_h2d/xgr·oc 惰性
#   位——只服务已删段）；L45 CACHE=None（outlines 缓存挂载位，delete[4]）。
#   章边界删除（impl-notes 有账）：
#   ① L86-L176 apply_grammar_bitmask——worker 侧掩码落地（重排/pinned H2D/
#     apply_token_bitmask_inplace），与 scheduler.get_grammar_bitmask 同为
#     ch31 交棒件，不进本章精简版（dossier scope_note 边界）。
#   ② L178-L388 outlines 专属全家（OutlinesVocabulary/get_outlines_cache_path/
#     OutlinesDiskCache/get_outlines_cache/_reduced_vocabulary/及其两个模块级
#     正则/get_outlines_vocabulary）——delete[4]（outlines 后端整体删除）。
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import TypeVar

import regex as re

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_T = TypeVar("_T")


# SOURCE: vllm/v1/structured_output/utils.py:L48-L83 compile_regex_with_timeout
#   —— 逐字（ReDoS 超时护栏：xgrammar/outlines/LMFE 三家 regex 编译共用）
# SOURCE: vllm/v1/structured_output/utils.py:L48-L83
def compile_regex_with_timeout(fn: Callable[[str], _T], pattern: str) -> _T:
    """Run a regex compilation callable with a timeout.

    Prevents ReDoS attacks where adversarial regex patterns (e.g. nested
    quantifiers like ``(a+)+b``) cause exponential DFA state-space explosion,
    hanging the inference worker indefinitely.

    Args:
        fn: Single-argument callable that takes the pattern and performs
            the regex compilation.
        pattern: The regex pattern string, passed to *fn* and included in
            timeout error messages.

    Raises:
        ValueError: If compilation exceeds the configured timeout.
    """
    timeout = envs.VLLM_REGEX_COMPILATION_TIMEOUT_S
    if timeout <= 0:
        return fn(pattern)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, pattern)
    try:
        result = future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise ValueError(
            f"Regex compilation timed out after {timeout}s. "
            "The pattern may be too complex or contain constructs that "
            "cause exponential state-space explosion (e.g. nested "
            f"quantifiers). Pattern: {pattern[:200]}"
        ) from None
    else:
        executor.shutdown(wait=False)
        return result


# SOURCE: vllm/v1/structured_output/utils.py:L391-L420 grammar_is_likely_lark
#   —— 逐字（无 ::= 即 Lark）
# SOURCE: vllm/v1/structured_output/utils.py:L391-L420
def grammar_is_likely_lark(grammar_str: str) -> bool:
    """
    Check if grammar appears to use Lark syntax.

    Args:
        grammar_str: Input grammar string

    Returns:
        bool: True if grammar appears to be in Lark format, False otherwise

    Examples:
        >>> grammar_is_likely_lark("rule: 'abc'")
        True
        >>> grammar_is_likely_lark("rule ::= 'abc'")
        False
    """
    if not grammar_str or not isinstance(grammar_str, str):
        return False

    for line in grammar_str.split("\n"):
        # Remove both comment styles
        line = re.sub(r"(#|//).*$", "", line).strip()
        if not line:
            continue

        # Look for EBNF rule definition
        if "::=" in line:
            return False

    return True


# SOURCE: vllm/v1/structured_output/utils.py:L423-L550 convert_lark_to_ebnf
#   —— 逐字
# SOURCE: vllm/v1/structured_output/utils.py:L423-L550
def convert_lark_to_ebnf(grammar_str: str) -> str:
    """
    Convert a Lark grammar string to EBNF format.

    EBNF reference:
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
    Lark grammar reference:
    https://lark-parser.readthedocs.io/en/latest/grammar.html

    Args:
        grammar_str: Input grammar in Lark format

    Returns:
        str: Converted grammar in EBNF format

    Examples:
        >>> print(convert_lark_to_ebnf("rule: 'hello'"))
        root ::= rule
        rule ::= "hello"
    """
    if not isinstance(grammar_str, str):
        raise ValueError(f"Grammar must be a string, got {type(grammar_str)}")
    if not grammar_str.strip():
        raise ValueError("Grammar string cannot be empty")

    defined_rules = set()
    referenced_rules = set()
    output_lines = []

    # SOURCE: vllm/v1/structured_output/utils.py:L452-L454（嵌套）
    def clean_line(line: str) -> str:
        """Remove comments and whitespace from line."""
        return re.sub(r"(#|//).*$", "", line).strip()

    # SOURCE: vllm/v1/structured_output/utils.py:L456-L459（嵌套）
    def check_quotes(text: str, rule_name: str, line_num: int) -> None:
        """Validate quote matching in text."""
        if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
            raise ValueError(f"Mismatched quotes in {rule_name} on line {line_num}")

    # SOURCE: vllm/v1/structured_output/utils.py:L461-L466（嵌套）
    def extract_references(text: str) -> set[str]:
        """Extract rule references from text."""
        # Remove quoted strings and special characters
        text = re.sub(r'"[^"]*"', "", text)
        text = re.sub(r"[+*?()|\[\]{}]", " ", text)
        return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text))

    # First pass: Find root rule and validate rule definitions
    lines = [clean_line(line) for line in grammar_str.split("\n")]
    first_rule = None

    for line_num, line in enumerate(lines, 1):
        if not line or line.startswith("|"):
            continue

        if ":" in line:
            try:
                name = line.split(":", 1)[0].strip().strip("?")
                defined_rules.add(name)
                if first_rule is None:
                    first_rule = name
                if name == "start":
                    first_rule = "start"
            except IndexError as e:
                raise ValueError(
                    f"Invalid rule format on line {line_num}. "
                    "Expected 'rule_name: definition'"
                ) from e

    if not defined_rules:
        raise ValueError("No valid rules found in grammar")

    # Add root rule
    output_lines.append(f"root ::= {first_rule}")

    # Second pass: Process rule definitions and alternatives
    current_rule = None
    current_definition = []

    for line_num, line in enumerate(lines, 1):
        if not line:
            continue

        try:
            if ":" in line and not line.startswith("|"):
                # Save previous rule if exists
                if current_rule:
                    output_lines.append(
                        f"{current_rule} ::= {' | '.join(current_definition)}"
                    )

                # Process new rule
                name, definition = line.split(":", 1)
                current_rule = name.strip().strip("?")

                check_quotes(definition, f"rule '{current_rule}'", line_num)
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)
                referenced_rules.update(extract_references(definition))
                current_definition = [definition.strip()]

            elif line.startswith("|"):
                if not current_rule:
                    raise ValueError(
                        f"Alternative '|' on line {line_num} "
                        "without a preceding rule definition"
                    )

                alt_def = line[1:].strip()
                check_quotes(
                    alt_def, f"alternative for rule '{current_rule}'", line_num
                )
                alt_def = re.sub(r"'([^']*)'", r'"\1"', alt_def)
                referenced_rules.update(extract_references(alt_def))
                current_definition.append(alt_def)

        except ValueError as e:
            raise ValueError(f"Error on line {line_num}: {str(e)}") from e

    # Add final rule if exists
    if current_rule:
        output_lines.append(f"{current_rule} ::= {' | '.join(current_definition)}")

    # Validate all rules are defined
    undefined_rules = referenced_rules - defined_rules - {"root"}
    if undefined_rules:
        raise ValueError(
            f"Referenced rules are not defined: {', '.join(sorted(undefined_rules))}"
        )

    return "\n".join(output_lines)


# SOURCE: vllm/v1/structured_output/utils.py:L553-L561 choice_as_grammar
#   —— 逐字（choice→EBNF 改写链条的产出端）
# SOURCE: vllm/v1/structured_output/utils.py:L553-L561
def choice_as_grammar(choice: list[str]) -> str:
    # SOURCE: vllm/v1/structured_output/utils.py:L554-L557（嵌套）
    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string."""
        # Escape double quotes and backslashes
        return re.sub(r'(["\\])', r"\\\1", s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    grammar = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
    return grammar
