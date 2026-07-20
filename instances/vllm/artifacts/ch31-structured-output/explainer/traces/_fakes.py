"""驱动脚本共用的测试替身（与 tests/test_structured_output.py 同一手法）。

被替身的只有**外部库对象**（xgrammar / llguidance），精简版自身的分派逻辑、状态推进、
门控判定一字不改地被真实执行。host 无 CUDA/无 vllm/无 xgrammar/llguidance
（见 dossier.analyst_notes_on_plan：本轮容器不可用），故所有 trace 都是纯控制流观测，
不含任何吞吐/编译耗时类真机数字（m03 的微秒数除外，且已注明是 host CPython 观测）。
"""
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))


class FakeMatcher:
    """xgr.GrammarMatcher 替身：记住已接受 token 序列，可指定某 token 被拒/触发终态。"""

    def __init__(self, ctx, max_rollback_tokens=0):
        self.ctx = ctx
        self.max_rollback_tokens = max_rollback_tokens
        self.accepted = []
        self.terminated_flag = False
        self.reject_token = None
        self.terminating_token = None
        self.rollback_calls = []

    def accept_token(self, token):
        if token == self.reject_token:
            return False
        self.accepted.append(token)
        if token == self.terminating_token:
            self.terminated_flag = True
        return True

    def rollback(self, n):
        self.rollback_calls.append(n)
        if n > 0:
            del self.accepted[len(self.accepted) - n:]
        self.terminated_flag = False

    def is_terminated(self):
        return self.terminated_flag

    def fill_next_token_bitmask(self, bitmask, idx):
        bitmask[idx] = list(self.accepted)

    def reset(self):
        self.accepted = []


class FakeCtx:
    def __init__(self, tag, spec):
        self.tag = tag
        self.spec = spec


class FakeCompiler:
    """xgr.GrammarCompiler 替身：cache_enabled=True 的**库内**缓存行为的最小复刻。

    真实缓存在 xgrammar 库内部（backend_xgrammar.py:L64-69 只是把 cache_enabled 传下去），
    host 无该库，故这里用 dict 复刻其可观察行为。计数字段供 trace 区分
    「vLLM 侧调了几次 compile_grammar」与「库内真正编译了几次」。
    """

    def __init__(self, tokenizer_info=None, max_threads=8, cache_enabled=True):
        self.cache_enabled = cache_enabled
        self._cache = {}
        self.compile_calls = 0
        self.real_compiles = 0
        self.cache_hits = 0

    def _compile(self, tag, spec, **kw):
        self.compile_calls += 1
        key = (tag, spec, tuple(sorted(kw.items())))
        if self.cache_enabled and key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.real_compiles += 1
        ctx = FakeCtx(tag, spec)
        if self.cache_enabled:
            self._cache[key] = ctx
        return ctx

    def compile_json_schema(self, spec, any_whitespace=True):
        return self._compile("json_schema", spec, any_whitespace=any_whitespace)

    def compile_grammar(self, spec):
        return self._compile("grammar", spec)

    def compile_regex(self, spec):
        return self._compile("regex", spec)

    def compile_structural_tag(self, spec):
        return self._compile("structural_tag", spec)


class FakeXgrGrammar:
    @staticmethod
    def from_ebnf(spec):
        if "BAD" in spec:
            raise ValueError("bad ebnf")
        return True

    @staticmethod
    def from_regex(spec):
        return True

    @staticmethod
    def from_json_schema(schema):
        return True

    @staticmethod
    def from_structural_tag(spec):
        return True


class FakeXgrModule:
    """模块级 `xgr` 名字的替身；compilers/matchers 让调用方观察计数。"""

    Grammar = FakeXgrGrammar

    def __init__(self):
        self.compilers = []
        self.matchers = []
        outer = self

        class TokenizerInfo:
            @staticmethod
            def from_huggingface(tokenizer, vocab_size=None):
                return {"vocab_size": vocab_size}

        def GrammarCompiler(tokenizer_info, max_threads=8, cache_enabled=True):
            c = FakeCompiler(tokenizer_info, max_threads, cache_enabled)
            outer.compilers.append(c)
            return c

        def GrammarMatcher(ctx, max_rollback_tokens=0):
            m = FakeMatcher(ctx, max_rollback_tokens=max_rollback_tokens)
            outer.matchers.append(m)
            return m

        self.TokenizerInfo = TokenizerInfo
        self.GrammarCompiler = GrammarCompiler
        self.GrammarMatcher = GrammarMatcher


class FakeLLMatcher:
    def __init__(self, ll_tokenizer, serialized_grammar):
        self.ll_tokenizer = ll_tokenizer
        self.serialized_grammar = serialized_grammar
        self.consumed = []
        self.stopped = False
        self.rollback_calls = []

    def get_error(self):
        return None

    def is_stopped(self):
        return self.stopped

    def consume_tokens(self, tokens):
        self.consumed.extend(tokens)
        return True

    def validate_tokens(self, tokens):
        return len(tokens)

    def rollback(self, n):
        self.rollback_calls.append(n)
        if n > 0:
            del self.consumed[len(self.consumed) - n:]
        self.stopped = False

    def reset(self):
        self.consumed = []


class FakeLLTokenizer:
    def __init__(self, eos_token=2, vocab_size=128):
        self.eos_token = eos_token
        self.vocab_size = vocab_size


class FakeLLGuidanceModule:
    """模块级 `llguidance` 替身；统计 grammar_from_json_schema 调用次数——
    guidance 完全没有编译缓存，该计数等于请求数。"""

    def __init__(self):
        outer = self
        self.schema_compiles = 0
        self.matchers = []

        class hf:
            @staticmethod
            def from_tokenizer(tokenizer, vocab_size):
                return FakeLLTokenizer(vocab_size=vocab_size)

        class LLMatcher(FakeLLMatcher):
            def __new__(cls, ll_tokenizer, serialized_grammar):
                m = FakeLLMatcher(ll_tokenizer, serialized_grammar)
                outer.matchers.append(m)
                return m

            @staticmethod
            def grammar_from_json_schema(spec, defaults=None):
                outer.schema_compiles += 1
                return f"llg_grammar({spec})"

        self.hf = hf
        self.LLMatcher = LLMatcher

    @staticmethod
    def grammar_from(tp, spec):
        return f"llg_{tp}({spec})"


def dump(path, payload):
    import json
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
