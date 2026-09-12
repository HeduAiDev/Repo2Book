# m17：思考模型联动（推进侧）——should_advance 判『思考段结束后语法才生效』，
# reasoning_end_token_index + trim_reasoning_for_advance 剔除混入的思考 token。
# 真实行为基准（v0.27.1 现核）：
#   - __init__.py:L381-L439 should_advance：reasoner None 快返回 True（非思考
#     模型常量路径）、reasoning_ended 缓存、is_reasoning_end_streaming 探测、
#     new_token_ids 直传的 delta 窗口（#43388）
#   - __init__.py:L442-L460 _find_reasoning_end_index：逐 token 定位边界，
#     无单 token 触发时回退末位（保守全算思考）
#   - __init__.py:L462-L486 trim_reasoning_for_advance：边界前后成片返回
#     后缀（#44006——思考 token 喂进 accept_tokens 会杀死请求）
from conftest import make_vllm_config
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


class FakeReasoner:
    """ReasoningParser 的最小替身（is_reasoning_end_streaming /
    is_reasoning_end 两个判停面）——真实装配链（ReasoningParserManager）
    已按减法计划删除，reasoner_cls 由测试直注（等价于装配产物）。"""

    def __init__(self, tokenizer=None, **kwargs):
        pass

    def is_reasoning_end_streaming(self, all_token_ids, delta_ids):
        # 边界 token（50256 之后的 99）出现即判思考结束
        return 99 in list(delta_ids)


def _manager(reasoner_cls=None, enable_in_reasoning=False):
    m = StructuredOutputManager(
        make_vllm_config(enable_in_reasoning=enable_in_reasoning))
    m.reasoner_cls = reasoner_cls
    return m


def _request(prompt=(1, 2), reasoning_ended=None):
    sp = SamplingParams(
        structured_outputs=StructuredOutputsParams(regex="a"))
    req = Request(request_id="r", prompt_token_ids=list(prompt),
                  sampling_params=sp, pooling_params=None,
                  reasoning_ended=reasoning_ended)
    # 直接挂上已就绪的 grammar 替身（本测试只测门，不测编译）
    req.structured_output_request._grammar = object()
    return req


# ── 常量路径：非思考模型 ────────────────────────────────────────
class TestConstantPath:
    def test_no_structured_output_never_advances(self):
        m = _manager()
        sp = SamplingParams(structured_outputs=None)
        req = Request(request_id="p", prompt_token_ids=[1],
                      sampling_params=sp, pooling_params=None)
        assert m.should_advance(req, new_token_ids=[5]) is False

    def test_reasoner_none_always_advances(self):
        # reasoner None（不配 reasoning_parser 的模型即此）→ 快返回 True
        m = _manager(reasoner_cls=None)
        req = _request()
        assert m.should_advance(req, new_token_ids=[5]) is True

    def test_enable_in_reasoning_overrides(self):
        # 思考内结构化：直接推进
        m = _manager(reasoner_cls=FakeReasoner, enable_in_reasoning=True)
        req = _request()
        assert m.should_advance(req, new_token_ids=[5]) is True


# ── 思考分支：边界探测 ─────────────────────────────────────────
class TestThinkingBranch:
    def test_reasoning_not_ended_blocks_advance(self):
        m = _manager(reasoner_cls=FakeReasoner)
        req = _request()
        assert m.should_advance(req, new_token_ids=[7, 8]) is False
        assert req.structured_output_request.reasoning_ended is not True

    def test_reasoning_ended_cached_short_circuits(self):
        m = _manager(reasoner_cls=FakeReasoner)
        req = _request(reasoning_ended=True)
        assert m.should_advance(req, new_token_ids=[7]) is True

    def test_reasoning_ends_this_step_sets_boundary(self):
        m = _manager(reasoner_cls=FakeReasoner)
        req = _request(prompt=[1, 2])
        req.append_output_token_ids([7, 8, 99])  # 99 = 思考结束边界
        assert m.should_advance(req, new_token_ids=[7, 8, 99]) is True
        assert req.structured_output_request.reasoning_ended is True
        # 边界 = all_token_ids 中最后一个思考 token 的绝对索引
        # all_token_ids = [1, 2, 7, 8, 99] → 99 的索引 4
        assert req.structured_output_request.reasoning_end_token_index == 4

    def test_delta_window_uses_new_token_ids_only(self):
        # #43388：new_token_ids 直传的修复——delta 窗口只看本步 token；
        # 历史里早已存在的 99 不触发。
        m = _manager(reasoner_cls=FakeReasoner)
        req = _request(prompt=[1, 99, 2])  # prompt 里就有 99（不在 delta 内）
        assert m.should_advance(req, new_token_ids=[7]) is False


# ── _find_reasoning_end_index：逐 token 定位 ────────────────────
class TestFindReasoningEndIndex:
    def test_fires_on_single_token(self):
        m = _manager(reasoner_cls=FakeReasoner)
        # start=2：prefix=[1,2]，从 idx 2 起逐 token 探测；99 在 idx 4 触发
        assert m._find_reasoning_end_index(
            FakeReasoner(), [1, 2, 7, 8, 99], 2) == 4

    def test_falls_back_to_last_index_when_no_trigger(self):
        m = _manager(reasoner_cls=FakeReasoner)
        # 全程无 99 → 保守回退末位（整步都算思考）
        assert m._find_reasoning_end_index(
            FakeReasoner(), [1, 2, 7, 8], 2) == 3


# ── trim_reasoning_for_advance：剔除混入的思考 token ────────────
class TestTrimReasoning:
    def test_no_boundary_returns_unchanged(self):
        m = _manager()
        req = _request()
        assert m.trim_reasoning_for_advance(req, [7, 8]) == [7, 8]

    def test_no_structured_request_returns_unchanged(self):
        m = _manager()
        sp = SamplingParams(structured_outputs=None)
        req = Request(request_id="p", prompt_token_ids=[1],
                      sampling_params=sp, pooling_params=None)
        assert m.trim_reasoning_for_advance(req, [7]) == [7]

    def test_trims_mixed_step(self):
        # #44006 场景：一步内混思考 token + 边界 + 语法 token
        # all_token_ids=[1,2,7,8,99,50,51]，边界 idx=4（99），
        # 本步 new=[7,8,99,50,51]（first_idx=2）→ 剔 3 个思考 token
        m = _manager()
        req = _request(prompt=[1, 2])
        req.structured_output_request.reasoning_end_token_index = 4
        req.append_output_token_ids([7, 8, 99, 50, 51])
        assert m.trim_reasoning_for_advance(req, [7, 8, 99, 50, 51]) == [50, 51]

    def test_step_after_boundary_returns_unchanged(self):
        # 边界之后的整步：num_reasoning = 5 - 5 = 0 → 原样返回
        m = _manager()
        req = _request(prompt=[1, 2])
        req.structured_output_request.reasoning_end_token_index = 4
        req.append_output_token_ids([7, 8, 99, 50, 51])
        assert m.trim_reasoning_for_advance(req, [50, 51]) == [50, 51]
