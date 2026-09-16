# 站 2-7 + m11/m12/m13/m14/m15/m16：语法对象的一生——进门即阻塞 → IO 线程
# 异步起编 → 100µs 探测 → 晋级/失败收账 → 采样后推进。
# 真实行为基准（v0.27.1 现核）：
#   - request.py:L97/L109-L114：带约束的生成请求出生即
#     WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR（不是 WAITING）
#   - __init__.py:L166-L175：async 分支 executor.submit；external_launcher
#     同步回退且异常也包 Future
#   - request.py:L50-L75：_check_grammar_completion 三态
#     （Future→成品/Exception 原地替换，timeout=0.0001）
#   - scheduler.py:L2050-L2062 阻塞态入 skipped_waiting 侧队；L2678-L2712
#     晋级三分支；L1954-L1968 编译失败同拍收账只杀单请求；L1817-L1843
#     用真采样 token 推进 FSM、拒收=FINISHED_ERROR+resumable=False
import time
from concurrent.futures import Future

import pytest
from conftest import make_vllm_config
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    XgrammarGrammar,
)
from vllm.v1.structured_output.request import StructuredOutputRequest

YES, EOS = 8505, 50256  # gpt2: "yes", <|endoftext|>

_TOK = None


def _tokenizer():
    # 前端校验需要 tokenizer（站 1 真实路径）；HF 层有缓存，重复调用无成本。
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        from conftest import TOKENIZER_NAME
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOK


def so_params(**kw):
    return StructuredOutputsParams(**kw)


def make_request(req_id, prompt=(1, 2, 3), so=None, *, frontend=True):
    """经真实前端校验（站 1：verify→_validate_structured_outputs 设 _backend）
    后构造 Request——引擎侧 grammar_init 读的 params._backend 正是它留下的。"""
    sp = SamplingParams(structured_outputs=so)
    if so is not None:
        if frontend:
            cfg = make_vllm_config()
            sp._validate_structured_outputs(
                cfg.model_config, cfg.structured_outputs_config, _tokenizer())
        else:
            # 引擎侧注入：绕过前端（真实前端会拦下坏语法——编译失败路径
            # 测试需要模拟『过了前端、编译期才暴露』的请求）
            so._backend = "xgrammar"
    return Request(request_id=req_id, prompt_token_ids=list(prompt),
                   sampling_params=sp, pooling_params=None)


@pytest.fixture()
def grammar_request():
    return make_request("gr-1", so=so_params(grammar='root ::= "yes" | "no"'))


@pytest.fixture()
def plain_request():
    return make_request("plain-1")


# ── 站 2：进门即阻塞 ────────────────────────────────────────────
class TestBornBlocked:
    def test_structured_request_born_in_grammar_gate(self, grammar_request):
        assert grammar_request.status == (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR)
        assert grammar_request.use_structured_output is True
        assert isinstance(grammar_request.structured_output_request,
                          StructuredOutputRequest)

    def test_plain_request_born_waiting(self, plain_request):
        assert plain_request.status == RequestStatus.WAITING
        assert plain_request.use_structured_output is False
        assert plain_request.structured_output_request is None

    def test_reasoning_kwargs_wired_through(self):
        sp = SamplingParams(structured_outputs=so_params(regex="a"))
        req = Request(request_id="r", prompt_token_ids=[1],
                      sampling_params=sp, pooling_params=None,
                      reasoning_ended=False,
                      reasoning_parser_kwargs={"foo": 1})
        assert req.structured_output_request.reasoning_ended is False
        assert req.structured_output_request.reasoning_parser_kwargs == {"foo": 1}


# ── 站 3：IO 线程异步起编 ────────────────────────────────────────
class TestGrammarInit:
    def test_no_structured_output_is_noop(self, manager, plain_request):
        manager.grammar_init(plain_request)
        assert plain_request.structured_output_request is None

    def test_async_submit_hangs_future_on_request(self, manager, grammar_request):
        manager.grammar_init(grammar_request)
        g = grammar_request.structured_output_request
        # 挂的是 Future（线程池在编）；小语法编译极快，可能在探测前就绪
        # ——未就绪窗口（grammar property 返回 None）由
        # TestThreeStatePolling 用受控 Future 精确验证。
        assert isinstance(g._grammar, Future) or isinstance(
            g._grammar, XgrammarGrammar)

    def test_lazy_backend_created_once(self, manager, grammar_request, tokenizer):
        assert manager.backend is None
        manager.grammar_init(grammar_request)
        assert isinstance(manager.backend, XgrammarBackend)
        other = make_request("gr-2", so=so_params(regex="a"))
        manager.grammar_init(other)
        # 全引擎唯一后端：第二个请求不再构造
        backend_after = manager.backend
        manager.grammar_init(make_request("gr-3", so=so_params(regex="b")))
        assert manager.backend is backend_after

    def test_executor_is_half_cpu_thread_pool(self, manager):
        import multiprocessing
        assert manager.executor._max_workers == max(
            1, (multiprocessing.cpu_count() + 1) // 2)

    def test_compile_completes_and_replaces_future(self, manager, grammar_request):
        manager.grammar_init(grammar_request)
        g = grammar_request.structured_output_request
        deadline = time.monotonic() + 10
        while g.grammar is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert isinstance(g._grammar, XgrammarGrammar)  # Future→成品原地替换
        assert g.grammar is g._grammar
        assert g.is_grammar_ready is True

    def test_unsupported_backend_name_raises(self, manager, grammar_request):
        grammar_request.sampling_params.structured_outputs._backend = "nope"
        with pytest.raises(ValueError, match="Unsupported structured output"):
            manager.grammar_init(grammar_request)


# ── 站 3（core 侧）：preprocess_add_request ─────────────────────
class TestPreprocessAddRequest:
    def test_io_thread_entry_submits_compilation(self, tokenizer):
        vllm_config = make_vllm_config()
        core = EngineCore(vllm_config)
        so = so_params(regex="[0-9]+")
        cfg = make_vllm_config()
        sp = SamplingParams(structured_outputs=so)
        sp._validate_structured_outputs(
            cfg.model_config, cfg.structured_outputs_config, _tokenizer())
        ecr = EngineCoreRequest(
            request_id="io-1", prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=sp,
            pooling_params=None,
            arrival_time=time.time(), lora_request=None, cache_salt=None,
            data_parallel_rank=None, current_wave=3,
        )
        req, wave = core.preprocess_add_request(ecr)
        assert wave == 3
        assert req.use_structured_output
        assert req.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        assert isinstance(req.structured_output_request._grammar, Future)

    def test_plain_request_no_grammar_init(self):
        core = EngineCore(make_vllm_config())
        ecr = EngineCoreRequest(
            request_id="io-2", prompt_token_ids=[1], mm_features=None,
            sampling_params=SamplingParams(structured_outputs=None),
            pooling_params=None, arrival_time=time.time(),
            lora_request=None, cache_salt=None, data_parallel_rank=None,
        )
        req, _ = core.preprocess_add_request(ecr)
        assert req.status == RequestStatus.WAITING


# ── m12：三态轮询（100µs 非阻塞探测） ───────────────────────────
class TestThreeStatePolling:
    def test_pending_future_times_out_to_false(self):
        req = StructuredOutputRequest(params=so_params(regex="a"))
        f = Future()
        req.grammar = f  # setter 收 Future
        assert req.is_grammar_ready is False
        assert req.grammar is None

    def test_ready_future_replaced_monotonically(self):
        req = StructuredOutputRequest(params=so_params(regex="a"))
        f = Future()
        f.set_result("PRODUCT")
        req.grammar = f
        # 恰好完成的那次探测直接拿到结果（不再返回 None）
        assert req.grammar == "PRODUCT"
        # 就绪单调：_grammar 不再是 Future，重复读幂等
        assert not isinstance(req._grammar, Future)
        assert req.grammar == "PRODUCT"

    def test_exception_future_becomes_exception_state(self):
        # v0.27.1 新形态：异常不抛穿，而是以 Exception 形态存进 _grammar
        req = StructuredOutputRequest(params=so_params(regex="a"))
        f = Future()
        f.set_exception(ValueError("bad schema"))
        req.grammar = f
        g = req.grammar
        assert isinstance(g, ValueError)
        assert req.is_grammar_ready is True  # 异常态也视为『探测完成』


# ── m14：external_launcher 回退同步 ─────────────────────────────
class TestSyncFallback:
    def test_sync_mode_returns_product_not_future(self, manager_factory,
                                                  grammar_request):
        m = manager_factory(distributed_executor_backend="external_launcher")
        assert m._use_async_grammar_compilation is False
        m.grammar_init(grammar_request)
        g = grammar_request.structured_output_request
        assert not isinstance(g._grammar, Future)
        assert isinstance(g._grammar, XgrammarGrammar)

    def test_sync_mode_wraps_failure_in_future(self, manager_factory):
        m = manager_factory(distributed_executor_backend="external_launcher")
        req = make_request("bad-1", so=so_params(structural_tag="{not json"),
                           frontend=False)
        m.grammar_init(req)
        g = req.structured_output_request
        # 同步分支异常也包 Future——两条路径对下游同形
        assert isinstance(g._grammar, Future)
        assert isinstance(g.grammar, Exception)


# ── 站 5/6：侧队与门控晋升（调度器） ────────────────────────────
class TestGateScheduling:
    @pytest.fixture()
    def sched(self, manager):
        return Scheduler(make_vllm_config(),
                         kv_cache_config=None,
                         structured_output_manager=manager,
                         block_size=16)

    @staticmethod
    def _slow_compile(manager, monkeypatch, delay=0.4):
        """让编译确定性地慢——小语法的真实编译 <1ms，观察『未就绪窗口』
        需要把工作线程压住（门控行为本身才是被测对象）。"""
        import threading
        real = manager._create_grammar
        started = threading.Event()

        def slow(request):
            started.set()
            time.sleep(delay)
            return real(request)

        monkeypatch.setattr(manager, "_create_grammar", slow)
        return started

    def _wait_ready(self, req, timeout=10):
        g = req.structured_output_request
        deadline = time.monotonic() + timeout
        while (g is None or g.grammar is None) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert g.grammar is not None

    def test_blocked_status_goes_to_side_queue(self, sched, grammar_request,
                                               plain_request):
        sched.add_request(plain_request)
        sched.add_request(grammar_request)
        # 阻塞态进 skipped_waiting 侧队而非正常 waiting——不挡人
        assert grammar_request in sched.skipped_waiting
        assert grammar_request not in sched.waiting
        assert plain_request in sched.waiting

    def test_schedule_serves_others_while_grammar_compiling(
            self, sched, manager, grammar_request, plain_request, monkeypatch):
        self._slow_compile(manager, monkeypatch)
        manager.grammar_init(grammar_request)
        sched.add_request(grammar_request)
        sched.add_request(plain_request)

        out = sched.schedule()
        # 没编译完的请求不进批；别人的 token 一个不耽误
        assert "plain-1" in out.num_scheduled_tokens
        assert "gr-1" not in out.num_scheduled_tokens
        assert grammar_request.status == (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR)
        # 未就绪请求被 prepend 回侧队，下一拍继续窥
        assert grammar_request in sched.skipped_waiting

    def test_promotion_admits_same_step_when_ready(self, sched, manager,
                                                   grammar_request,
                                                   plain_request, monkeypatch):
        self._slow_compile(manager, monkeypatch)
        manager.grammar_init(grammar_request)
        sched.add_request(grammar_request)
        sched.add_request(plain_request)
        sched.schedule()  # 第一拍：阻塞

        self._wait_ready(grammar_request)
        out = sched.schedule()  # 第二拍：就绪 → 晋升 → 当拍入批
        assert "gr-1" in out.num_scheduled_tokens
        assert grammar_request.status == RequestStatus.RUNNING
        assert out.has_structured_output_requests is True

    def test_try_promote_direct_semantics(self, sched, manager, grammar_request,
                                          monkeypatch):
        self._slow_compile(manager, monkeypatch)
        manager.grammar_init(grammar_request)
        # 未就绪：grammar None → False
        assert sched._try_promote_blocked_waiting_request(grammar_request) is False
        self._wait_ready(grammar_request)
        # 就绪：status → WAITING
        assert sched._try_promote_blocked_waiting_request(grammar_request) is True
        assert grammar_request.status == RequestStatus.WAITING

    def test_try_promote_rejects_plain_blocked_request(self, sched,
                                                       plain_request):
        # 非 WAITING_FOR_* 阻塞态：直接 False（防御分支）
        plain_request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        assert sched._try_promote_blocked_waiting_request(plain_request) is False


# ── m13：编译失败只杀单请求 ─────────────────────────────────────
class TestCompileFailureIsolation:
    @pytest.fixture()
    def sched(self, manager):
        return Scheduler(make_vllm_config(),
                         kv_cache_config=None,
                         structured_output_manager=manager,
                         block_size=16)

    def _make_engine_core_output(self, sched):
        from vllm.v1.core.sched.output import SchedulerOutput
        return SchedulerOutput(num_scheduled_tokens={})

    def test_failure_recorded_and_finished_in_same_step(self, sched, manager):
        # 前端本会拦下的坏 structural_tag，直接从引擎侧注入——
        # compile 的 json.loads 失败经 Future 传回调度器。
        bad = make_request("bad-1", so=so_params(structural_tag="{not json"),
                        frontend=False)
        good = make_request("good-1", so=so_params(regex="[0-9]+"))
        manager.grammar_init(bad)
        manager.grammar_init(good)
        sched.add_request(bad)
        sched.add_request(good)

        deadline = time.monotonic() + 10
        while not isinstance(bad.structured_output_request.grammar,
                             Exception) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert isinstance(bad.structured_output_request.grammar, Exception)

        # 晋升检查识别异常 → 记账，不晋升
        assert sched._try_promote_blocked_waiting_request(bad) is False
        assert "bad-1" in sched.grammar_compile_error_reqs
        assert bad.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

        # 同拍 update_from_output 尾部收账：只杀这一个
        outputs = sched.update_from_output(self._make_engine_core_output(sched),
                                           _ModelRunnerOutput({}, []))
        assert bad.status == RequestStatus.FINISHED_ERROR
        assert bad.get_finished_reason() is not None
        assert "bad-1" not in sched.requests  # 已出列
        assert "bad-1" not in sched.grammar_compile_error_reqs  # 已清账
        assert good.status != RequestStatus.FINISHED_ERROR  # 旁人无感
        # 前端拿到空 token 的错误回执
        flat = [o for lst in outputs.values() for o in lst]
        assert [o.request_id for o in flat] == ["bad-1"]
        assert flat[0].new_token_ids == []

    def test_compile_error_recorded_via_schedule_promotion_too(
            self, sched, manager):
        bad = make_request("bad-2", so=so_params(structural_tag="{nope"),
                        frontend=False)
        manager.grammar_init(bad)
        sched.add_request(bad)
        deadline = time.monotonic() + 10
        while not isinstance(bad.structured_output_request.grammar,
                             Exception) and time.monotonic() < deadline:
            time.sleep(0.01)
        sched.schedule()  # 窥队头触发晋升检查 → 记账
        assert "bad-2" in sched.grammar_compile_error_reqs


class _ModelRunnerOutput:
    """测试替身：update_from_output 只消费这两个字段（真实面 =
    gpu_model_runner 采样回传，ch18 域）。"""

    def __init__(self, req_id_to_index, sampled_token_ids):
        self.req_id_to_index = req_id_to_index
        self.sampled_token_ids = sampled_token_ids


# ── 站 7/m16：采样后推进（FSM 唯一写者） ─────────────────────────
class TestAdvanceAfterSampling:
    @pytest.fixture()
    def sched(self, manager):
        return Scheduler(make_vllm_config(),
                         kv_cache_config=None,
                         structured_output_manager=manager,
                         block_size=16)

    def _promote_and_schedule(self, sched, manager, req):
        manager.grammar_init(req)
        sched.add_request(req)
        deadline = time.monotonic() + 10
        while req.structured_output_request.grammar is None and \
                time.monotonic() < deadline:
            time.sleep(0.01)
        out = sched.schedule()
        assert req.request_id in out.num_scheduled_tokens
        return out

    def test_sampled_tokens_advance_fsm(self, sched, manager, grammar_request):
        out = self._promote_and_schedule(sched, manager, grammar_request)
        grammar = grammar_request.structured_output_request.grammar
        n_before = grammar.num_processed_tokens
        sched.update_from_output(
            out, _ModelRunnerOutput({"gr-1": 0}, [[YES, EOS]]))
        assert grammar_request._output_token_ids == [YES, EOS]
        assert grammar.num_processed_tokens == n_before + 2
        assert grammar.is_terminated() is True

    def test_grammar_rejection_is_engine_bug_finishes_error(self, sched, manager):
        req = make_request("victim", so=so_params(grammar='root ::= "yes"'))
        out = self._promote_and_schedule(sched, manager, req)
        # 掩码之下不该采出非法 token——喂 3919("no") 模拟『链断了』
        sched.update_from_output(
            out, _ModelRunnerOutput({"victim": 0}, [[3919]]))
        assert req.status == RequestStatus.FINISHED_ERROR
        assert req.resumable is False
        assert req not in sched.running

    def test_plain_request_skips_advance(self, sched, manager, plain_request):
        sched.add_request(plain_request)
        out = sched.schedule()
        sched.update_from_output(
            out, _ModelRunnerOutput({"plain-1": 0}, [[42]]))
        assert plain_request.status == RequestStatus.RUNNING
        assert plain_request._output_token_ids == [42]


# ── m15：线程安全不变式的可观察面 ────────────────────────────────
class TestThreadSafetyInvariants:
    def test_grammar_setter_accepts_future_and_product(self):
        req = StructuredOutputRequest(params=so_params(regex="a"))
        f = Future()
        req.grammar = f
        assert req._grammar is f
        req.grammar = "PRODUCT"
        assert req._grammar == "PRODUCT"
