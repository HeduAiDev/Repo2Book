# ch32 explainer 驱动脚本共用件：实现树挂载、请求/配置替身（与 tests/conftest.py
# 同口径）、真 xgrammar 六方法适配器（pin backend_xgrammar.py XgrammarGrammar 的
# 最小复刻——编译侧归 ch31 已删，本章注入成品 grammar）、gpt2 常量。
# 取证环境（explainer 铁律 exp-0718-1，explainer.json trace_environment 同款标注）：
#   host = Miniconda CPython 3.11.11 + torch 2.11.0+cu128（RTX PRO 6000 Blackwell，
#   sm_120）+ xgrammar 0.2.6 + triton 3.7.1 + transformers(gpt2 本地缓存)。
#   gpt2 vocab=50257 非生产 13 万词表；xgrammar 0.2.6 GrammarMatcher 不传
#   max_rollback_tokens（vLLM pin 传 num_speculative_tokens，该 kwarg 在 0.2.6
#   已弃用限制语义——库侧恒无限回滚，见 ch31 trace compile.spec_deprecation_warning）。
import pathlib
import sys
import types

IMPL = pathlib.Path(__file__).resolve().parents[2] / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import numpy as np
import torch

from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

TOKENIZER_NAME = "gpt2"
VOCAB = 50257
# gpt2 锚点 token id（与 ch31 trace_m01_m06_bitmask.json 一致）
TOK_A, TOK_B, TOK_C = 64, 65, 66          # 'a','b','c'
TOK_YES, TOK_NO = 8505, 3919
EOS = 50256
OFF_GRAMMAR = 4242                         # '####'——词表内语法外


def cdiv(a, b):
    return -(-a // b)


# ── 请求/配置替身（tests/conftest.py 同口径）─────────────────────────────────
class FakeStructuredRequest:
    def __init__(self, grammar=None):
        self.grammar = grammar
        self.reasoning_ended = None
        self.reasoning_end_token_index = None
        self.reasoner = None
        self.reasoning_parser_kwargs = None


class FakeRequest:
    def __init__(self, request_id, *, use_structured_output=True, grammar=None,
                 prompt_token_ids=(), all_token_ids=None, num_tokens=0,
                 num_computed_tokens=0, num_output_placeholders=0,
                 spec_token_ids=None, is_finished=False,
                 num_in_flight_tokens=0):
        self.request_id = request_id
        self.use_structured_output = use_structured_output
        self.structured_output_request = (
            FakeStructuredRequest(grammar) if grammar is not None or use_structured_output
            else None
        )
        self.prompt_token_ids = list(prompt_token_ids)
        self.all_token_ids = list(all_token_ids or [])
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_output_placeholders = num_output_placeholders
        self.num_in_flight_tokens = num_in_flight_tokens
        self.spec_token_ids = spec_token_ids
        self.is_prefill_chunk = False
        self.status = None
        self.resumable = True
        self._finished = is_finished

    def is_finished(self):
        if self._finished:
            return True
        if self.status is None:
            return False
        from vllm.v1.request import RequestStatus
        return RequestStatus.is_finished(self.status)


class FakeSchedulerOutput:
    def __init__(self, num_scheduled_tokens=None,
                 scheduled_spec_decode_tokens=None,
                 has_structured_output_requests=False,
                 pending_structured_output_tokens=False,
                 total_num_scheduled_tokens=None,
                 num_spec_tokens_to_schedule=0):
        self.num_scheduled_tokens = dict(num_scheduled_tokens or {})
        self.scheduled_spec_decode_tokens = dict(scheduled_spec_decode_tokens or {})
        self.has_structured_output_requests = has_structured_output_requests
        self.pending_structured_output_tokens = pending_structured_output_tokens
        self.total_num_scheduled_tokens = (
            total_num_scheduled_tokens
            if total_num_scheduled_tokens is not None
            else sum(self.num_scheduled_tokens.values())
        )
        self.num_spec_tokens_to_schedule = num_spec_tokens_to_schedule
        self.num_invalid_spec_tokens = None


def make_vllm_config(max_num_seqs=8, num_speculative_tokens=0, *,
                     is_diffusion=False, enable_in_reasoning=False):
    sc = types.SimpleNamespace(max_num_seqs=max_num_seqs)
    mc = types.SimpleNamespace(skip_tokenizer_init=True, is_diffusion=is_diffusion)
    soc = types.SimpleNamespace(enable_in_reasoning=enable_in_reasoning)
    cfg = types.SimpleNamespace(
        scheduler_config=sc, model_config=mc, structured_outputs_config=soc,
        parallel_config=types.SimpleNamespace(pipeline_parallel_size=1),
    )
    cfg.num_speculative_tokens = num_speculative_tokens
    return cfg


def make_manager(vocab_size=64, max_num_seqs=8, num_speculative_tokens=0, **kw):
    from vllm.v1.structured_output import StructuredOutputManager
    mgr = StructuredOutputManager(
        make_vllm_config(max_num_seqs=max_num_seqs,
                         num_speculative_tokens=num_speculative_tokens, **kw))
    mgr.backend = FakeBackend(vocab_size)
    return mgr


class FakeBackend:
    """allocate_token_bitmask 按真实语义：int32、全 -1（全允许）预分配。"""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.alloc_calls = []

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        import torch as _t
        self.alloc_calls.append(max_num_seqs)
        return _t.full((max_num_seqs, cdiv(self.vocab_size, 32)), -1, dtype=_t.int32)

    def destroy(self):
        pass


# ── 真 xgrammar 六方法适配器（pin backend_xgrammar.py:L136-L196 最小复刻）────
class XgrGrammar(StructuredOutputGrammar):
    """真 GrammarMatcher 的六方法契约承载（编译侧归 ch31——按 dossier 减法
    计划本章删除，grammar 成品注入；本适配器即真实 XgrammarGrammar 的行为：
    accept/validate/rollback/fill/is_terminated 语义与 pin 逐字一致，
    仅去掉 openai_gptoss 侧的遥测）。观测账本记录调用于 trace。"""

    def __init__(self, matcher, vocab_size=VOCAB):
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.num_processed_tokens = 0
        self._is_terminated = False
        self.accepts = []      # [(req_id, [tokens])]
        self.validates = []    # [tokens]
        self.rollbacks = []    # [n]
        self.fill_calls = []   # [(buffer_rows, index)]

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        self.accepts.append((request_id, list(tokens)))
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        self.validates.append(list(tokens))
        accepted = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted.append(token)
            else:
                break
        if accepted:
            self.matcher.rollback(len(accepted))
        return accepted

    def rollback(self, num_tokens: int) -> None:
        self.rollbacks.append(num_tokens)
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.fill_calls.append((tuple(bitmask.shape), idx))
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        raise NotImplementedError


class RealBackend:
    """allocate_token_bitmask 走真 xgrammar（与 pin backend_xgrammar.py:L128-L133
    同一调用：xgr.allocate_token_bitmask(max_num_seqs, vocab_size)）。"""

    def __init__(self, vocab_size=VOCAB):
        self.vocab_size = vocab_size
        self.alloc_calls = []

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        import xgrammar as xgr
        self.alloc_calls.append(max_num_seqs)
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        pass


_TOK = None
_COMPILER = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOK


def compile_grammar(ebnf: str):
    """真 xgrammar 编译（GrammarCompiler 带 LRU 缓存——ch31 m10 已立；
    同一 ebnf 反复编译命中缓存，256 个 matcher 的构造不重复编译）。"""
    global _COMPILER
    import xgrammar as xgr
    if _COMPILER is None:
        _COMPILER = xgr.GrammarCompiler(
            xgr.TokenizerInfo.from_huggingface(get_tokenizer()), max_threads=1)
    return _COMPILER.compile_grammar(ebnf)


def new_matcher(ebnf: str) -> "xgr.GrammarMatcher":
    import xgrammar as xgr
    return xgr.GrammarMatcher(compile_grammar(ebnf))


def new_grammar(ebnf: str) -> XgrGrammar:
    return XgrGrammar(new_matcher(ebnf))


def allowed_ids(row_int32, vocab_size) -> list:
    u = np.asarray(row_int32).astype(np.uint32)
    return [t for t in range(vocab_size) if (int(u[t // 32]) >> (t % 32)) & 1]


def row_full_allow(row_int32) -> bool:
    """整行 -1（32 位全 1=全允许）判定。"""
    return bool(np.all(np.asarray(row_int32) == -1))


def dump(name, out):
    p = pathlib.Path(__file__).with_name(name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        import json
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("WROTE", p)
