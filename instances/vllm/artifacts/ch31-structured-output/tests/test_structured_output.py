"""TDD：验证精简版复现真实 vLLM v1 结构化输出的可观察行为（pin ad7125a4 / v0.21.0）。

纯 Python 单元测试，不 import vllm、不依赖真实 xgrammar/llguidance（本轮无容器，
host 也没有这两个第三方库——见 dossier.analyst_notes_on_plan）。凡是精简版代码要调用
xgrammar/llguidance 的地方，测试里用轻量 Fake 对象替身注入对应模块的 `xgr`/
`llguidance` 名字——这是标准 TDD 测试替身手法，被替身的是外部库对象，vLLM 自身的
控制流代码（分派逻辑、状态推进、掩码调用）一字不改地被真实执行、真实断言。
"""
import json as json_mod
from concurrent.futures import Future

import pytest

import backend_guidance as bg
import backend_xgrammar as bx
from backend_types import StructuredOutputOptions
from request import Request, RequestStatus
from sampling_params import SamplingParams, StructuredOutputsConfig, StructuredOutputsParams
from scheduler import Scheduler
from so_request import StructuredOutputRequest, get_structured_output_key
from structured_output_manager import StructuredOutputManager
from utils import choice_as_grammar


# ───────────────────────── StructuredOutputsParams：六选一互斥 ─────────────────────────


def test_structured_outputs_params_accepts_single_constraint():
    p = StructuredOutputsParams(regex="a+")
    assert p.regex == "a+"
    assert p._backend is None
    assert p._backend_was_auto is False


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(json='{"type": "object"}', regex="a+"),
        dict(choice=["a", "b"], grammar="root ::= \"a\""),
        dict(json_object=True, structural_tag="{}"),
    ],
)
def test_structured_outputs_params_rejects_multiple_constraints(kwargs):
    with pytest.raises(ValueError, match="only use one kind"):
        StructuredOutputsParams(**kwargs)


def test_structured_outputs_params_rejects_zero_constraints():
    with pytest.raises(ValueError, match="must use one kind"):
        StructuredOutputsParams()


def test_all_constraints_none():
    assert StructuredOutputsParams(regex="a+").all_constraints_none() is False


# ───────────────────────── get_structured_output_key：六种归一 ─────────────────────────


@pytest.mark.parametrize(
    "kwargs,expected_option",
    [
        (dict(json='{"type": "object"}'), StructuredOutputOptions.JSON),
        (dict(json_object=True), StructuredOutputOptions.JSON_OBJECT),
        (dict(regex="a+"), StructuredOutputOptions.REGEX),
        (dict(choice=["a", "b"]), StructuredOutputOptions.CHOICE),
        (dict(grammar='root ::= "a"'), StructuredOutputOptions.GRAMMAR),
        (dict(structural_tag="{}"), StructuredOutputOptions.STRUCTURAL_TAG),
    ],
)
def test_get_structured_output_key_option(kwargs, expected_option):
    option, _spec = get_structured_output_key(StructuredOutputsParams(**kwargs))
    assert option is expected_option


def test_get_structured_output_key_normalizes_dict_and_list_to_json_string():
    option, spec = get_structured_output_key(
        StructuredOutputsParams(json={"type": "object"})
    )
    assert option is StructuredOutputOptions.JSON
    assert json_mod.loads(spec) == {"type": "object"}

    option, spec = get_structured_output_key(
        StructuredOutputsParams(choice=["red", "blue"])
    )
    assert option is StructuredOutputOptions.CHOICE
    assert json_mod.loads(spec) == ["red", "blue"]


# ───────────────────────── choice_as_grammar：choice → EBNF ─────────────────────────


def test_choice_as_grammar_escapes_quotes_and_backslashes():
    grammar = choice_as_grammar(["a", 'b"c', "d\\e"])
    assert grammar == 'root ::= "a" | "b\\"c" | "d\\\\e"'


# ───────────────────────── StructuredOutputRequest：Future 门控 ─────────────────────────


def test_from_sampling_params_returns_none_without_constraints():
    assert StructuredOutputRequest.from_sampling_params(None) is None
    assert StructuredOutputRequest.from_sampling_params(SamplingParams()) is None


def test_from_sampling_params_returns_request_with_constraints():
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    sor = StructuredOutputRequest.from_sampling_params(sp)
    assert sor is not None
    assert sor.params is sp.structured_outputs


def test_grammar_property_returns_none_while_future_pending():
    sor = StructuredOutputRequest(params=StructuredOutputsParams(regex="a+"))
    fut = Future()
    sor.grammar = fut
    assert sor.grammar is None
    assert sor.is_grammar_ready is False


def test_grammar_property_resolves_future_and_replaces_it_in_place():
    sor = StructuredOutputRequest(params=StructuredOutputsParams(regex="a+"))
    fut = Future()
    sor.grammar = fut
    fut.set_result("GRAMMAR_OBJ")

    assert sor.grammar == "GRAMMAR_OBJ"
    # 幂等：第二次读不再是 Future（_grammar 已被原地替换成成品）。
    assert sor.grammar == "GRAMMAR_OBJ"
    assert not isinstance(sor._grammar, Future)


def test_grammar_property_accepts_grammar_object_directly_when_sync():
    sor = StructuredOutputRequest(params=StructuredOutputsParams(regex="a+"))
    sor.grammar = "SYNC_GRAMMAR"
    assert sor.grammar == "SYNC_GRAMMAR"


def test_structured_output_key_is_cached_property():
    params = StructuredOutputsParams(json={"type": "object"})
    sor = StructuredOutputRequest(params=params)
    key1 = sor.structured_output_key
    key2 = sor.structured_output_key
    assert key1 == key2
    assert key1[0] is StructuredOutputOptions.JSON


# ───────────────────────── Request：初始阻塞态 ─────────────────────────


def test_request_status_waiting_for_grammar_when_structured_output_present():
    sp = SamplingParams(
        max_tokens=10, structured_outputs=StructuredOutputsParams(regex="a+")
    )
    req = Request("r1", sampling_params=sp)
    assert req.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert req.use_structured_output is True


def test_request_status_plain_waiting_without_structured_output():
    sp = SamplingParams(max_tokens=10)
    req = Request("r2", sampling_params=sp)
    assert req.status == RequestStatus.WAITING
    assert req.use_structured_output is False


# ───────────────────────── xgrammar 后端：五个分支、无 CHOICE ─────────────────────────


class _FakeMatcher:
    def __init__(self, ctx, max_rollback_tokens=0):
        self.ctx = ctx
        self.max_rollback_tokens = max_rollback_tokens
        self.accepted: list[int] = []
        self.terminated_flag = False
        self.reject_token = None  # 设为某 token 值即模拟该 token 被拒

    def accept_token(self, token):
        if token == self.reject_token:
            return False
        self.accepted.append(token)
        return True

    def rollback(self, n):
        del self.accepted[len(self.accepted) - n :]

    def is_terminated(self):
        return self.terminated_flag

    def fill_next_token_bitmask(self, bitmask, idx):
        bitmask[idx] = list(self.accepted)

    def reset(self):
        self.accepted = []


class _FakeGrammarCtx:
    def __init__(self, tag, spec):
        self.tag = tag
        self.spec = spec


class _FakeCompiler:
    def __init__(self):
        self.calls: list[tuple] = []

    def compile_json_schema(self, spec, any_whitespace=True):
        self.calls.append(("json_schema", spec, any_whitespace))
        return _FakeGrammarCtx("json_schema", spec)

    def compile_grammar(self, spec):
        self.calls.append(("grammar", spec))
        return _FakeGrammarCtx("grammar", spec)

    def compile_regex(self, spec):
        self.calls.append(("regex", spec))
        return _FakeGrammarCtx("regex", spec)

    def compile_structural_tag(self, spec):
        self.calls.append(("structural_tag", spec))
        return _FakeGrammarCtx("structural_tag", spec)


class _FakeXgrGrammar:
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


class _FakeXgrModule:
    Grammar = _FakeXgrGrammar

    class GrammarMatcher:
        def __new__(cls, ctx, max_rollback_tokens=0):
            return _FakeMatcher(ctx, max_rollback_tokens=max_rollback_tokens)


@pytest.fixture()
def fake_xgr(monkeypatch):
    fake = _FakeXgrModule()
    monkeypatch.setattr(bx, "xgr", fake)
    return fake


def _make_xgrammar_backend(fake_xgr):
    backend = bx.XgrammarBackend.__new__(bx.XgrammarBackend)
    backend.compiler = _FakeCompiler()
    backend.num_speculative_tokens = 4
    backend.disable_any_whitespace = False
    backend.vocab_size = 1000
    return backend


@pytest.mark.parametrize(
    "option,spec,expected_call",
    [
        (StructuredOutputOptions.JSON, '{"type": "object"}', "json_schema"),
        (StructuredOutputOptions.JSON_OBJECT, "", "json_schema"),
        (StructuredOutputOptions.GRAMMAR, 'root ::= "a"', "grammar"),
        (StructuredOutputOptions.REGEX, "a+", "regex"),
        (StructuredOutputOptions.STRUCTURAL_TAG, '{"triggers": []}', "structural_tag"),
    ],
)
def test_xgrammar_compile_grammar_dispatch(fake_xgr, option, spec, expected_call):
    backend = _make_xgrammar_backend(fake_xgr)
    grammar = backend.compile_grammar(option, spec)
    assert backend.compiler.calls[0][0] == expected_call
    assert grammar.matcher.max_rollback_tokens == 4


def test_xgrammar_compile_grammar_has_no_choice_branch(fake_xgr):
    """m10：CHOICE 走不到 xgrammar 的编译分派——校验期已把它改写成 GRAMMAR。"""
    backend = _make_xgrammar_backend(fake_xgr)
    with pytest.raises(ValueError, match="not of valid supported types"):
        backend.compile_grammar(StructuredOutputOptions.CHOICE, '["a", "b"]')


def test_validate_xgrammar_grammar_rewrites_choice_to_ebnf_grammar(fake_xgr):
    """m20：校验期不只是选后端，还会原地改写请求——choice → EBNF grammar。"""
    sp = SamplingParams(
        structured_outputs=StructuredOutputsParams(choice=["red", "blue"])
    )
    bx.validate_xgrammar_grammar(sp)

    assert sp.structured_outputs.choice is None
    assert sp.structured_outputs.grammar == 'root ::= "red" | "blue"'


def test_validate_xgrammar_grammar_propagates_ebnf_compile_failure(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(grammar="BAD"))
    with pytest.raises(ValueError, match="Invalid grammar specification"):
        bx.validate_xgrammar_grammar(sp)


def test_has_xgrammar_unsupported_json_features():
    assert bx.has_xgrammar_unsupported_json_features(
        {"type": "number", "multipleOf": 2}
    )
    assert bx.has_xgrammar_unsupported_json_features(
        {"type": "array", "uniqueItems": True}
    )
    assert bx.has_xgrammar_unsupported_json_features(
        {"type": "string", "format": "not-a-real-format"}
    )
    assert not bx.has_xgrammar_unsupported_json_features({"type": "object"})


# ───────────────────────── XgrammarGrammar：六方法契约 ─────────────────────────


def _grammar_with_fake_matcher(**kwargs):
    matcher = _FakeMatcher(ctx=None, max_rollback_tokens=kwargs.pop("max_rollback_tokens", 2))
    return bx.XgrammarGrammar(matcher=matcher, ctx=None, vocab_size=100, **kwargs), matcher


def test_xgrammar_grammar_accept_tokens_advances_and_counts():
    grammar, matcher = _grammar_with_fake_matcher()
    ok = grammar.accept_tokens("req-1", [1, 2, 3])
    assert ok is True
    assert grammar.num_processed_tokens == 3
    assert matcher.accepted == [1, 2, 3]


def test_xgrammar_grammar_accept_tokens_fails_and_stops_counting():
    grammar, matcher = _grammar_with_fake_matcher()
    matcher.reject_token = 99
    ok = grammar.accept_tokens("req-1", [1, 99, 3])
    assert ok is False
    assert grammar.num_processed_tokens == 1  # 只有 token 1 被计入


def test_xgrammar_grammar_accept_tokens_short_circuits_when_terminated():
    grammar, matcher = _grammar_with_fake_matcher()
    grammar._is_terminated = True
    assert grammar.accept_tokens("req-1", [1]) is False
    assert matcher.accepted == []  # 根本没调用 matcher


def test_xgrammar_grammar_validate_tokens_does_not_advance_state():
    grammar, matcher = _grammar_with_fake_matcher()
    matcher.reject_token = 30
    accepted = grammar.validate_tokens([10, 20, 30, 40])
    assert accepted == [10, 20]
    # 试走完会 rollback 回去，不留下痕迹。
    assert matcher.accepted == []
    assert grammar.num_processed_tokens == 0  # validate_tokens 完全不碰这个计数


def test_xgrammar_grammar_rollback_reverts_counter_and_terminated_flag():
    grammar, matcher = _grammar_with_fake_matcher()
    grammar.accept_tokens("req-1", [1, 2, 3])
    matcher.terminated_flag = True  # 模拟 rollback 之后重新查询 is_terminated
    grammar.rollback(2)
    assert grammar.num_processed_tokens == 1
    assert matcher.accepted == [1]
    assert grammar._is_terminated is True


def test_xgrammar_grammar_fill_bitmask_delegates_to_matcher():
    grammar, matcher = _grammar_with_fake_matcher()
    grammar.accept_tokens("req-1", [7])
    bitmask = {}
    grammar.fill_bitmask(bitmask, 3)
    assert bitmask[3] == [7]


def test_xgrammar_grammar_reset_clears_counter_and_matcher():
    grammar, matcher = _grammar_with_fake_matcher()
    grammar.accept_tokens("req-1", [1, 2])
    grammar.reset()
    assert grammar.num_processed_tokens == 0
    assert matcher.accepted == []


def test_xgrammar_grammar_max_rollback_tokens_equals_num_speculative_tokens(fake_xgr):
    """rollback 是为投机解码留的口子：max_rollback_tokens = num_speculative_tokens。"""
    backend = _make_xgrammar_backend(fake_xgr)
    backend.num_speculative_tokens = 5
    grammar = backend.compile_grammar(StructuredOutputOptions.REGEX, "a+")
    assert grammar.matcher.max_rollback_tokens == 5


# ───────────────────────── guidance 后端：第二实现，rollback_lag ─────────────────────────


class _FakeLLMatcher:
    def __init__(self):
        self.consumed: list[int] = []
        self.stopped = False
        self.rolled_back_by = None
        self.error = None

    def is_stopped(self):
        return self.stopped

    def consume_tokens(self, tokens):
        self.consumed.extend(tokens)
        return True

    def validate_tokens(self, tokens):
        return len(tokens)

    def rollback(self, n):
        self.rolled_back_by = n

    def get_error(self):
        return self.error


def _guidance_grammar(eos_token=999):
    matcher = _FakeLLMatcher()
    tokenizer = type("T", (), {"eos_token": eos_token})()
    grammar = bg.GuidanceGrammar(ll_matcher=matcher, ll_tokenizer=tokenizer, vocab_size=10)
    return grammar, matcher


def test_guidance_grammar_accept_tokens_normal_path():
    grammar, matcher = _guidance_grammar()
    assert grammar.accept_tokens("req-1", [1, 2, 3]) is True
    assert matcher.consumed == [1, 2, 3]
    assert grammar.terminated is False


def test_guidance_grammar_eos_sets_terminated_and_rollback_lag():
    """guidance 独有：EOS 后回滚要少退一格，xgrammar 没有这个概念。"""
    grammar, matcher = _guidance_grammar(eos_token=999)
    matcher.stopped = True  # 模拟 EOS 后 matcher 已经 stopped
    result = grammar.accept_tokens("req-1", [1, 999])
    assert result is True
    assert grammar.terminated is True
    assert grammar.rollback_lag == 1


def test_guidance_grammar_rollback_subtracts_lag():
    grammar, matcher = _guidance_grammar()
    grammar.rollback_lag = 1
    grammar.terminated = True
    grammar.rollback(3)
    assert matcher.rolled_back_by == 2  # 3 - rollback_lag(1)
    assert grammar.terminated is False
    assert grammar.rollback_lag == 0


def test_has_guidance_unsupported_json_features():
    assert bg.has_guidance_unsupported_json_features({"patternProperties": {}})
    assert not bg.has_guidance_unsupported_json_features({"type": "object"})


# ───────────────────────── auto 阶梯：xgrammar → guidance → (outlines 已删) ─────────────────────────


def test_explicit_xgrammar_backend_selection(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    cfg = StructuredOutputsConfig(backend="xgrammar")
    sp._validate_structured_outputs(cfg, tokenizer="tok")
    assert sp.structured_outputs._backend == "xgrammar"
    assert sp.structured_outputs._backend_was_auto is False


def test_auto_ladder_succeeds_on_xgrammar_first(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    cfg = StructuredOutputsConfig(backend="auto")
    sp._validate_structured_outputs(cfg, tokenizer="tok")
    assert sp.structured_outputs._backend == "xgrammar"
    assert sp.structured_outputs._backend_was_auto is True


def test_auto_ladder_falls_back_to_guidance_when_xgrammar_json_unsupported(fake_xgr):
    # multipleOf 触发 has_xgrammar_unsupported_json_features -> xgrammar 校验失败；
    # guidance 侧同一 schema 没有 patternProperties -> 不 skip_guidance -> 落 guidance。
    schema = {"type": "number", "multipleOf": 2}
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    cfg = StructuredOutputsConfig(backend="auto")
    sp._validate_structured_outputs(cfg, tokenizer="tok")
    assert sp.structured_outputs._backend == "guidance"
    assert sp.structured_outputs._backend_was_auto is True


def test_auto_ladder_never_reaches_outlines_when_guidance_accepts(fake_xgr):
    """m11/m19：auto 阶梯只在 xgrammar→guidance→outlines 之间降级，从不选
    lm-format-enforcer；guidance 能接的 schema 不会走到 outlines 分支（本精简版
    outlines 分支直接 raise，可作为『没走到那里』的反向证据）。"""
    schema = {"type": "number", "multipleOf": 2}
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    cfg = StructuredOutputsConfig(backend="auto")
    sp._validate_structured_outputs(cfg, tokenizer="tok")  # 不应抛异常
    assert sp.structured_outputs._backend == "guidance"


def test_auto_ladder_outlines_branch_is_structurally_present_but_unimplemented(fake_xgr):
    # patternProperties 同时不被 xgrammar(multipleOf 无关但 json 校验会走别的失败路径)
    # 与 guidance 支持 -> skip_guidance=True -> 精简版明确 raise，标出边界而非静默改行为。
    schema = {"type": "number", "multipleOf": 2, "patternProperties": {"^x": {}}}
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    cfg = StructuredOutputsConfig(backend="auto")
    with pytest.raises(NotImplementedError, match="outlines"):
        sp._validate_structured_outputs(cfg, tokenizer="tok")


def test_validate_structured_outputs_requires_tokenizer(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    cfg = StructuredOutputsConfig(backend="auto")
    with pytest.raises(ValueError, match="tokenizer"):
        sp._validate_structured_outputs(cfg, tokenizer=None)


def test_validate_structured_outputs_noop_without_constraints(fake_xgr):
    sp = SamplingParams()
    cfg = StructuredOutputsConfig(backend="auto")
    sp._validate_structured_outputs(cfg, tokenizer="tok")  # no-op, no raise


def test_validate_structured_outputs_rejects_empty_choice_list(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(grammar='root ::= "a"'))
    sp.structured_outputs.choice = []  # 绕开 __post_init__，模拟已构造对象被复用
    cfg = StructuredOutputsConfig(backend="xgrammar")
    with pytest.raises(ValueError, match="cannot be an empty list"):
        sp._validate_structured_outputs(cfg, tokenizer="tok")


def test_validate_structured_outputs_rejects_empty_grammar_string(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    sp.structured_outputs.grammar = "   "
    cfg = StructuredOutputsConfig(backend="xgrammar")
    with pytest.raises(ValueError, match="cannot be an empty string"):
        sp._validate_structured_outputs(cfg, tokenizer="tok")


def test_request_level_backend_selection_rejected_when_mismatched(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    sp.structured_outputs._backend = "guidance"  # 上一次请求/复用留下的痕迹
    cfg = StructuredOutputsConfig(backend="xgrammar")
    with pytest.raises(ValueError, match="Request-level structured output backend"):
        sp._validate_structured_outputs(cfg, tokenizer="tok")


def test_request_level_backend_selection_allowed_when_previously_auto(fake_xgr):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    sp.structured_outputs._backend = "xgrammar"
    sp.structured_outputs._backend_was_auto = True
    cfg = StructuredOutputsConfig(backend="auto")  # 引擎仍是 auto，允许沿用
    sp._validate_structured_outputs(cfg, tokenizer="tok")
    assert sp.structured_outputs._backend == "xgrammar"


# ───────────────────────── StructuredOutputManager：异步编译 + 惰性建后端 ─────────────────────────


class _FakeBackend:
    instances_created = 0

    def __init__(self, vllm_config, tokenizer, vocab_size):
        _FakeBackend.instances_created += 1
        self.vllm_config = vllm_config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.compiled = []

    def compile_grammar(self, request_type, grammar_spec):
        self.compiled.append((request_type, grammar_spec))
        return f"GRAMMAR[{request_type}:{grammar_spec}]"


@pytest.fixture(autouse=True)
def _reset_fake_backend_counter():
    _FakeBackend.instances_created = 0
    yield


def _manager_with_fake_backend(monkeypatch, external_launcher=False):
    monkeypatch.setattr(
        "structured_output_manager.XgrammarBackend", _FakeBackend
    )
    manager = StructuredOutputManager(
        vllm_config=object(), tokenizer=None, external_launcher=external_launcher
    )
    return manager


def _structured_request(regex="a+", backend="xgrammar"):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex=regex))
    sp.structured_outputs._backend = backend
    return Request("r-async", sampling_params=sp)


def test_grammar_init_builds_backend_lazily_once(monkeypatch):
    manager = _manager_with_fake_backend(monkeypatch)
    req1 = _structured_request()
    req2 = _structured_request()

    manager.grammar_init(req1)
    manager.grammar_init(req2)

    assert _FakeBackend.instances_created == 1
    assert manager.backend is not None


def test_grammar_init_submits_to_executor_and_resolves_future(monkeypatch):
    manager = _manager_with_fake_backend(monkeypatch)
    req = _structured_request(regex="a+")

    manager.grammar_init(req)

    # 提交异步：短暂之后应能拿到编译结果（线程池，真实并发）。
    grammar = req.structured_output_request.grammar
    deadline_tries = 0
    while grammar is None and deadline_tries < 1000:
        grammar = req.structured_output_request.grammar
        deadline_tries += 1
    assert grammar == f"GRAMMAR[{StructuredOutputOptions.REGEX}:a+]"


def test_grammar_init_sync_path_when_external_launcher(monkeypatch):
    manager = _manager_with_fake_backend(monkeypatch, external_launcher=True)
    assert manager._use_async_grammar_compilation is False
    req = _structured_request(regex="b+")

    manager.grammar_init(req)

    # 同步路径：grammar_init 返回时已经是成品，不需要等待。
    assert req.structured_output_request._grammar == (
        f"GRAMMAR[{StructuredOutputOptions.REGEX}:b+]"
    )


def test_grammar_init_noop_without_structured_output_request(monkeypatch):
    manager = _manager_with_fake_backend(monkeypatch)
    req = Request("plain", sampling_params=SamplingParams(max_tokens=5))
    manager.grammar_init(req)  # 不应报错，也不应建后端
    assert manager.backend is None


def test_grammar_init_rejects_unsupported_backend(monkeypatch):
    manager = _manager_with_fake_backend(monkeypatch)
    req = _structured_request(backend="totally-unknown")
    with pytest.raises(ValueError, match="Unsupported structured output backend"):
        manager.grammar_init(req)


# ───────────────────────── Scheduler：阻塞态门 + 推进语法状态机 ─────────────────────────


@pytest.mark.parametrize(
    "status,expected",
    [
        (RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, True),
        (RequestStatus.WAITING_FOR_REMOTE_KVS, True),
        (RequestStatus.WAITING_FOR_STREAMING_REQ, True),
        (RequestStatus.WAITING, False),
        (RequestStatus.RUNNING, False),
    ],
)
def test_is_blocked_waiting_status(status, expected):
    assert Scheduler._is_blocked_waiting_status(status) is expected


def test_try_promote_blocked_waiting_request_false_while_grammar_pending():
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    req = Request("r1", sampling_params=sp)
    req.structured_output_request.grammar = Future()  # 未完成

    scheduler = Scheduler()
    promoted = scheduler._try_promote_blocked_waiting_request(req)

    assert promoted is False
    assert req.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


def test_try_promote_blocked_waiting_request_true_once_grammar_ready():
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    req = Request("r1", sampling_params=sp)
    fut = Future()
    req.structured_output_request.grammar = fut
    fut.set_result("READY_GRAMMAR")

    scheduler = Scheduler()
    promoted = scheduler._try_promote_blocked_waiting_request(req)

    assert promoted is True
    assert req.status == RequestStatus.WAITING


def test_try_promote_blocked_waiting_request_ignores_other_statuses():
    sp = SamplingParams(max_tokens=5)
    req = Request("r2", sampling_params=sp)
    assert req.status == RequestStatus.WAITING

    scheduler = Scheduler()
    promoted = scheduler._try_promote_blocked_waiting_request(req)
    assert promoted is False


def test_advance_grammar_on_sampled_tokens_calls_accept_tokens():
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    req = Request("r1", sampling_params=sp)
    grammar, matcher = _grammar_with_fake_matcher()
    req.structured_output_request.grammar = grammar

    Scheduler._advance_grammar_on_sampled_tokens(req, [1, 2, 3])

    assert matcher.accepted == [1, 2, 3]
    assert grammar.num_processed_tokens == 3


def test_advance_grammar_on_sampled_tokens_raises_when_grammar_rejects():
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    req = Request("r1", sampling_params=sp)
    grammar, matcher = _grammar_with_fake_matcher()
    matcher.reject_token = 5
    req.structured_output_request.grammar = grammar

    with pytest.raises(RuntimeError, match="grammar rejected tokens"):
        Scheduler._advance_grammar_on_sampled_tokens(req, [5])


def test_advance_grammar_on_sampled_tokens_noop_for_plain_request():
    req = Request("plain", sampling_params=SamplingParams(max_tokens=5))
    Scheduler._advance_grammar_on_sampled_tokens(req, [1, 2, 3])  # 不应报错


def test_validate_spec_tokens_against_grammar_is_a_try_walk():
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
    req = Request("r1", sampling_params=sp)
    grammar, matcher = _grammar_with_fake_matcher()
    matcher.reject_token = 30
    req.structured_output_request.grammar = grammar

    accepted = Scheduler._validate_spec_tokens_against_grammar(req, [10, 20, 30, 40])

    assert accepted == [10, 20]
    # 试走完全没有推进真实状态——accept 计数器纹丝不动。
    assert grammar.num_processed_tokens == 0
    assert matcher.accepted == []


def test_validate_spec_tokens_against_grammar_passthrough_for_plain_request():
    req = Request("plain", sampling_params=SamplingParams(max_tokens=5))
    result = Scheduler._validate_spec_tokens_against_grammar(req, [1, 2, 3])
    assert result == [1, 2, 3]


def test_collect_structured_output_request_ids_filters_prefill_and_plain():
    class FakeReq:
        def __init__(self, use_so, is_prefill_chunk=False):
            self.use_structured_output = use_so
            self.is_prefill_chunk = is_prefill_chunk

    requests = {
        "a": FakeReq(True),
        "b": FakeReq(False),
        "c": FakeReq(True, is_prefill_chunk=True),
    }
    ids = Scheduler._collect_structured_output_request_ids(["a", "b", "c"], requests)
    assert ids == ["a"]
