"""精简版支撑类型：本章只保留 parallel sampling 扇出/归并触及的字段，

其余 vLLM 真实字段一律 SUBTRACTED（见各处注释）。这些不是杜撰的新抽象，
而是真实 vLLM 同名类型的「只删不增」忠实子集，便于读者无需 import vllm/torch/msgspec
即可跑通 n>1 扇出的完整控制流。
"""

from __future__ import annotations

import enum
import uuid
from copy import copy
from dataclasses import dataclass


# SOURCE: vllm/sampling_params.py — RequestOutputKind
class RequestOutputKind(enum.Enum):
    # SUBTRACTED: CUMULATIVE 取值此处用不到，仅保留 DELTA/FINAL_ONLY 两路用于
    #            区分流式逐条转发 vs 非流式聚合（vllm/sampling_params.py）
    DELTA = 1
    FINAL_ONLY = 2


# SOURCE: vllm/sampling_params.py:class SamplingParams
@dataclass
class SamplingParams:
    """采样参数。本章只关心 n / seed / output_kind 三字段——它们决定扇出路数、

    child 种子递进、以及归并是流式还是聚合。
    """

    n: int = 1
    seed: int | None = None
    output_kind: RequestOutputKind = RequestOutputKind.DELTA
    # SUBTRACTED: temperature/top_p/top_k/max_tokens/stop/logprobs 等全部真实采样字段
    #            （vllm/sampling_params.py）——它们随 copy() 一并被复制到 child，
    #            但与扇出/归并控制流无关，故省略以聚焦本章主线。

    def clone(self) -> "SamplingParams":
        # SOURCE: vllm/sampling_params.py — copy(self) 浅拷贝父 params 派生 child
        return copy(self)


# SOURCE: vllm/outputs.py:L22 class CompletionOutput
@dataclass
class CompletionOutput:
    """一路（一个 child）的生成结果。index 即扇出时的 idx，贯穿到 FINAL_ONLY 归位。"""

    index: int
    text: str = ""
    token_ids: list[int] | None = None
    finish_reason: str | None = None
    # SUBTRACTED: cumulative_logprob/logprobs/routed_experts/stop_reason/lora_request
    #            （vllm/outputs.py:L43-L49）——输出细节属 ch08，本章只需 index 与是否 finished。

    # SOURCE: vllm/outputs.py:L50 def finished
    def finished(self) -> bool:
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L80 class EngineCoreRequest
@dataclass
class EngineCoreRequest:
    """跨进程 IPC 载荷。本章只关注 request_id（child 唯一内部 id）、

    external_req_id（n 路共享对外 id）、sampling_params（child 被强制 n=1）三字段。
    """

    request_id: str
    sampling_params: SamplingParams | None = None
    # SOURCE: vllm/v1/engine/__init__.py:L120 external_req_id 默认 None，由 assign_request_id 填入
    external_req_id: str | None = None
    # SUBTRACTED: prompt_token_ids/mm_features/pooling_params/arrival_time/lora_request/
    #            cache_salt/prompt_embeds/client_index/current_wave/priority/trace_headers/
    #            reasoning_* 等真实字段（vllm/v1/engine/__init__.py:L87-L123）——
    #            tokenize/多模态/DP/reasoning 属 ch04/ch05，与扇出无关。

    @property
    def params(self) -> SamplingParams:
        # SOURCE: vllm/v1/engine/__init__.py:L131 @property def params
        # SUBTRACTED: pooling_params 分支（vllm/v1/engine/__init__.py:L134-L138）——
        #            本章只走 sampling 路径。
        assert self.sampling_params is not None
        return self.sampling_params


# SOURCE: vllm/v1/engine/output_processor.py — RequestOutputCollector
class RequestOutputCollector:
    """对外队列：n 路 child 共享同一个 collector，对 generate() 消费者透明。

    精简为一个 list 缓冲，只保留「同一请求一个 queue」这一本章关键不变量。
    """

    # SOURCE: vllm/v1/engine/output_processor.py — RequestOutputCollector.__init__
    def __init__(self, output_kind: RequestOutputKind, request_id: str) -> None:
        self.output_kind = output_kind
        self.request_id = request_id
        self.items: list = []

    # SOURCE: vllm/v1/engine/output_processor.py — RequestOutputCollector.put
    def put(self, item) -> None:
        # SUBTRACTED: 真实实现用 asyncio merge/聚合 DELTA（vllm/v1/engine/output_processor.py）；
        #            这里只追加，足以观察「都进同一 queue」。
        self.items.append(item)


@dataclass
class RequestOutput:
    # SOURCE: vllm/outputs.py — RequestOutput（对外返还的请求级输出）
    """归并后对外返还。本章关键：request_id 是 external_req_id（n 路对外同一个）。"""

    request_id: str
    outputs: list  # list[CompletionOutput]
    finished: bool
    # SUBTRACTED: prompt/prompt_token_ids/metrics/kv_transfer_params 等（vllm/outputs.py）——
    #            非本章主线。


# SOURCE: vllm/utils/__init__.py:L11 def random_uuid
_MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    # SOURCE: vllm/utils/__init__.py:L12 — 16 hex 字符（uuid4.int & 64bit mask）
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"
