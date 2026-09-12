# SOURCE: vllm/v1/structured_output/backend_xgrammar.py
# 只做减法的忠实精简版（pin v0.27.1 / 6e448d0ea）——默认后端主线。
# 删除项（subtraction_plan.delete）：[1] Mistral/tekken 分词器兼容分支；
# [2] structural_tag 的 deprecated 编译路径；[7] logger 调用/TYPE_CHECKING
# 断言块/# type: ignore/类头 NOTE 的 jump-forward 展望注释。
# SUBTRACTED: SPDX 版权头。
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
# SUBTRACTED: `from vllm.utils.mistral import is_mistral_tokenizer`（delete[1]
#   ——L42-L59 Mistral/tekken 兼容分支的唯一消费者，分支删后成死 import）
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    choice_as_grammar,
    compile_regex_with_timeout,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


# SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L35-L132 XgrammarBackend
@dataclass
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self):
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L37-L40
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )

        # SUBTRACTED: vllm/v1/structured_output/backend_xgrammar.py:L42-L59
        #   Mistral/tekken 分词器的手工 TokenizerInfo 构造（RAW/BYTE_FALLBACK
        #   vocab_type/stop_token_ids/add_prefix_space——delete[1]）。只影响
        #   TokenizerInfo 的构造方式，编译→FSM→门控主控制流不变；非 Mistral
        #   模型即走保留的 from_huggingface 路径。
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L60-L64
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size,
        )
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L65-L70
        #   GrammarCompiler：cache_enabled=True + 字节预算（m10 编译复用的
        #   真实所在——库内 LRU+512MB，v0.21 无此参数）
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L72-L76
        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L78-L126 compile_grammar
    #   —— 编译五分派（JSON/JSON_OBJECT/GRAMMAR/REGEX/STRUCTURAL_TAG；
    #   **无 CHOICE 分支**——校验期已把 choice 原地改写成 EBNF grammar）
    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L78-L126
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            # JSON_OBJECT = compile_json_schema('{"type": "object"}') 的语法糖
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            # ReDoS 超时护栏（m19）
            ctx = compile_regex_with_timeout(
                self.compiler.compile_regex,
                grammar_spec,
            )
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            # SUBTRACTED: vllm/v1/structured_output/backend_xgrammar.py:L97-L109
            #   `s_tag = json.loads(grammar_spec)` + `if "structures" in s_tag:`
            #   的 deprecated 拆解分支（StructuralTagItem 列表 + triggers）
            #   ——delete[2]。精简版只走新路径 compile_structural_tag(str)。
            ctx = self.compiler.compile_structural_tag(grammar_spec)
        else:
            # SUBTRACTED: logger.error("Validation should have already
            #   occurred. Please file an issue.")（L112-L114——delete[7]）
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L128-L129
    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L131-L132
    def destroy(self):
        del self.compiler


# SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L135-L203 XgrammarGrammar
#   —— 六方法参考实现（StructuredOutputGrammar ABC 的 xgrammar 落地）
# SUBTRACTED: 类头 NOTE 注释 L137-L142（"This would be a generic-enough class
#   ... jump-forward decoding"——delete[7]：类头 NOTE 的 jump-forward 展望注释）
@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L152-L171 accept_tokens
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                # SUBTRACTED: logger.error("Failed to advance FSM for request
                #   %s for tokens %s. Please file an issue.", ...)（L162-L167
                #   ——delete[7]；真实源码自认『拒收=引擎 bug』的措辞归正文引用）
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L173-L188 validate_tokens
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L190-L193 rollback
    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L195-L196 fill_bitmask
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L198-L199 is_terminated
    def is_terminated(self) -> bool:
        return self._is_terminated

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L201-L203 reset
    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()


# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
# SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L206-L222 —— 逐字
STRING_SUPPORTED_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


# SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L225-L269
#   has_xgrammar_unsupported_json_features —— 逐字（能力矩阵预检 m3）
# SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L225-L269
def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L228-L267（嵌套递归）
    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True

        # Check for array unsupported keywords
        if obj.get("type") == "array" and any(
            key in obj
            for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # Unsupported keywords for strings
        if (
            obj.get("type") == "string"
            and "format" in obj
            and obj["format"] not in STRING_SUPPORTED_FORMATS
        ):
            return True

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(
            key in obj for key in ("patternProperties", "propertyNames")
        ):
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


# SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L272-L363 validate_xgrammar_
#   grammar —— 校验期试编 + choice→EBNF/lark→EBNF 原地改写的唯一发生地
def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L282-L291 regex 试编
    if so_params.regex:
        try:
            compile_regex_with_timeout(
                xgr.Grammar.from_regex,
                so_params.regex,
            )
        except Exception as err:
            raise ValueError(
                f"Failed to transform regex into a grammar: {err}"
            ) from err

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L293-L303
    #   choice→EBNF 原地改写（choice=None/grammar=EBNF——引擎侧五分派因此
    #   无 CHOICE 分支）
    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(
                f"Failed to transform choices into a grammar: {err}"
            ) from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L305-L325 json 分支
    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError(
                "The provided JSON schema contains features not supported by xgrammar."
            )

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(
                f"Failed to transform json schema into a grammar: {err}"
            ) from err
        return

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L327-L343 grammar 分支
    #   （lark 判定 + 转换 + 试编）
    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            # xgrammar supports EBNF grammars only
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError(
                    "Failed to convert the grammar from Lark to EBNF. "
                ) from e

        # Test parsing EBNF grammar, possibly already converted from Lark
        try:
            # parse the grammar, but we aren't compiling it.
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
        return

    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L345-L363 structural_tag
    if so_params.structural_tag:
        try:
            # SUBTRACTED: L347-L359 `s_tag = json.loads(...)` + deprecated
            #   structures/triggers 拆解（StructuralTagItem 列表构造 +
            #   from_structural_tag(tags, triggers)）——delete[2]。精简版只走
            #   新路径 from_structural_tag(str)。
            xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise ValueError("Invalid structural tag specification.") from e
