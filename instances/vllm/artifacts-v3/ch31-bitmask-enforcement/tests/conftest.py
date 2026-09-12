# ch31《约束解码 II：bitmask 落地》测试配置与共用替身。
# 行为基准 = 真实 vLLM v0.27.1（6e448d0ea，instances/vllm/source 现核行号）：
#   - vllm/v1/structured_output/__init__.py（批装配/思考门控/出发）
#   - vllm/v1/structured_output/utils.py:L86-L175（worker 侧重排+H2D+apply）
#   - vllm/v1/core/sched/scheduler.py（门控置位/行序账本/草稿过滤/真推进）
#   - vllm/v1/worker/gpu_model_runner.py（两段式窗口两幕）
#   - vllm/v1/engine/core.py（step 四段/deferred 兑现链）
#   - vllm/v1/worker/gpu/structured_outputs.py（V2 落地+Triton kernel）
import os
import pathlib
import sys
import types

IMPL_DIR = pathlib.Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))

import numpy as np
import torch

from vllm.v1.structured_output.backend_types import StructuredOutputGrammar


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ─────────────────────────────────────────────────────────────────────────────
# grammar / backend 替身：实现 ch30 已立的六方法契约
# （vllm/v1/structured_output/backend_types.py:L31-L95 StructuredOutputGrammar）。
# fill_bitmask 写真实位语义：allowed 集内 token 位=1（允许）、其余位=0（禁→-inf），
# 与 xgrammar fill_next_token_bitmask 的位语义一致（m08：位语义 1=允许/0=禁）。
# ─────────────────────────────────────────────────────────────────────────────
class FakeGrammar(StructuredOutputGrammar):
    """六方法契约的可编程替身（编译归 ch30——本章直接注入成品 grammar）。
    继承真实 ABC：scheduler.update_from_output L1823 的
    isinstance(grammar, StructuredOutputGrammar) 语义断言要求替身履约。"""

    def __init__(self, vocab_size: int, allowed=None, terminated=False,
                 reject_tokens=(), validate_keep=None):
        self.vocab_size = vocab_size
        self.allowed = set(allowed) if allowed is not None else set(range(vocab_size))
        self.terminated = terminated
        self.reject_tokens = set(reject_tokens)
        # validate_tokens(toks) -> toks[:validate_keep(toks)]；None = 全过
        self._validate_keep = validate_keep
        # 观测账本
        self.fill_calls = []        # [(bitmask_id(buffer 行数), index)]
        self.accepts = []           # [(req_id, [token...])]
        self.rollbacks = []         # [n]
        self.validates = []         # [tokens]

    def fill_bitmask(self, bitmask: torch.Tensor, index: int) -> None:
        # uint32 视图置位（bit 31 在有符号 int32 上 1<<31 溢出——真实 xgrammar
        # 的位语义即无符号置位后按 int32 存储）
        row = np.zeros(bitmask.shape[1], dtype=np.uint32)
        for tok in self.allowed:
            if tok // 32 < row.shape[0]:
                row[tok // 32] |= np.uint32(1 << (tok % 32))
        bitmask[index] = torch.from_numpy(row.view(np.int32))
        self.fill_calls.append((tuple(bitmask.shape), index))

    def is_terminated(self) -> bool:
        return self.terminated

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        self.accepts.append((request_id, list(tokens)))
        return not any(t in self.reject_tokens for t in tokens)

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        self.validates.append(list(tokens))
        if self._validate_keep is None:
            return list(tokens)
        if callable(self._validate_keep):
            n = self._validate_keep(tokens)
        else:
            n = self._validate_keep
        return list(tokens[:n])

    def rollback(self, num_tokens: int) -> None:
        self.rollbacks.append(num_tokens)

    def reset(self):
        self.allowed = set(range(self.vocab_size))
        self.terminated = False


class FakeBackend:
    """allocate_token_bitmask 按真实语义：int32、全 -1（全允许）预分配。"""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.alloc_calls = []

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        self.alloc_calls.append(max_num_seqs)
        return torch.full(
            (max_num_seqs, cdiv(self.vocab_size, 32)), -1, dtype=torch.int32
        )

    def destroy(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# reasoner 替身：is_reasoning_end / is_reasoning_end_streaming 两个观测面
# （vllm/v1/structured_output/__init__.py:L361-L439 的消费契约）。
# ─────────────────────────────────────────────────────────────────────────────
class FakeReasoner:
    def __init__(self, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer
        self.prompt_end_calls = []          # is_reasoning_end 调用账
        self.streaming_calls = []           # (all_token_ids 长度, delta_ids)
        # 编程面：prompt 级结论 / 流式级判定回调
        self._prompt_end = False
        self._streaming = None              # callable(all_len, delta)->bool

    def is_reasoning_end(self, prompt_token_ids) -> bool:
        self.prompt_end_calls.append(list(prompt_token_ids))
        return self._prompt_end

    def is_reasoning_end_streaming(self, all_token_ids, delta_ids) -> bool:
        # 物化一次（真实调用方传 islice 迭代器——重复 list() 会得到空表）
        delta = list(delta_ids)
        all_ids = list(all_token_ids)
        self.streaming_calls.append((len(all_ids), delta))
        if self._streaming is None:
            return False
        return self._streaming(len(all_ids), delta)


# ─────────────────────────────────────────────────────────────────────────────
# request / config / scheduler_output 替身（字段面按真实 Request/
# StructuredOutputRequest/VllmConfig 的消费切片直建；装配归邻章）。
# ─────────────────────────────────────────────────────────────────────────────
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
        # 真实 Request.is_finished 的状态语义：FINISHED_* > PREEMPTED
        # （update_from_output 语法拒绝只置 status=FINISHED_ERROR）
        if self._finished:
            return True
        if self.status is None:
            return False
        from vllm.v1.request import RequestStatus

        return RequestStatus.is_finished(self.status)


def make_vllm_config(max_num_seqs=8, num_speculative_tokens=0, *,
                     skip_tokenizer_init=True, is_diffusion=False,
                     enable_in_reasoning=False):
    """按 StructuredOutputManager 消费面直建的配置替身（真实 VllmConfig 装配归
    ch03；manager 只读 5 个字段：scheduler_config.max_num_seqs /
    num_speculative_tokens / model_config.skip_tokenizer_init /
    model_config.is_diffusion / structured_outputs_config.enable_in_reasoning）。"""
    sc = types.SimpleNamespace(max_num_seqs=max_num_seqs)
    mc = types.SimpleNamespace(skip_tokenizer_init=skip_tokenizer_init,
                               is_diffusion=is_diffusion)
    soc = types.SimpleNamespace(enable_in_reasoning=enable_in_reasoning)
    cfg = types.SimpleNamespace(
        scheduler_config=sc, model_config=mc, structured_outputs_config=soc,
    )
    # num_speculative_tokens 在真实 VllmConfig 是 property（config/vllm.py:L564-L575）
    cfg.num_speculative_tokens = num_speculative_tokens
    return cfg


def make_manager(vocab_size=64, max_num_seqs=8, num_speculative_tokens=0, **kw):
    from vllm.v1.structured_output import StructuredOutputManager

    cfg = make_vllm_config(
        max_num_seqs=max_num_seqs,
        num_speculative_tokens=num_speculative_tokens,
        **kw,
    )
    mgr = StructuredOutputManager(cfg)
    mgr.backend = FakeBackend(vocab_size)
    return mgr


class FakeGrammarOutput:
    """GrammarOutput 替身（真实 dataclass 见 vllm/v1/core/sched/output.py:L286-L291）。"""

    def __init__(self, ids, bitmask):
        self.structured_output_request_ids = ids
        self.grammar_bitmask = bitmask


class FakeSchedulerOutput:
    """SchedulerOutput 消费切片替身（真实 dataclass 见
    vllm/v1/core/sched/output.py:L195-L283）。"""

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
