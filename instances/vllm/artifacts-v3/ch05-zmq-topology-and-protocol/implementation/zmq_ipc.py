# Subtract-only companion for v3 ch05 «ZMQ 拓扑与消息协议» (Part II: L0 紫色
# ZMQ 边界带放大——前端进程 ⇄ EngineCore 进程之间到底怎么对话).
#
# FAITHFUL SUBSET of the real vLLM cross-process transport at pin v0.27.1
# (6e448d0ea). It keeps vLLM's names, structure and control flow; it only
# DELETES branches approved in the dossier subtraction_plan (plus the
# mechanical deletions listed in impl-notes.md) and marks every deletion with
# `# SUBTRACTED:`. Mapping rule: take the real vLLM source, drop every
# SUBTRACTED branch, and you should get (approximately) this file.
#
# The ZMQ topology and message protocol are kept FULLY REAL: client ROUTER
# (bind) / engine DEALER (connect, identity=engine_index 2-byte little-endian)
# on the input path, engine PUSH (connect) / client PULL (bind) on the output
# path, HWM=0 everywhere (make_zmq_socket), byte-tag first frame
# (EngineCoreRequestType), msgpack multi-frame payloads with zero-copy
# aux_buffer frames, the two-layer startup handshake, the DEALER-must-speak-
# first ready message, and the single-frame ENGINE_CORE_DEAD death sentinel.
# Engines really run in background processes (mp spawn, the real
# launch_core_engines / CoreEngineProcManager path) and tests talk to them
# over the same wire vLLM uses.
#
# Host seams (each marked `HOST SEAM` / `ENGINE SEAM` and documented in
# impl-notes.md): msgspec is backed by a wire-compatible shim
# (_msgspec_seam.py, real msgpack bytes); the engine *interior* (scheduler /
# model executor = the ch09 five-beat busy loop) is a scripted seam reached
# through the real UTILITY thin RPC; the config classes are field seams (the
# full assembly line is the ch03 product); vLLM itself is Linux-only, so
# ipc:// addresses fall back to loopback tcp on win32 hosts.
#
# Runs on a CPU host WITHOUT the vllm package. Every def/class carries a
# `# SOURCE: vllm/...:Lxxx` ref into the pinned tree (line numbers re-verified
# against v0.27.1 on 2026-08-16, not copied from v2's v0.21.0 assets).

from __future__ import annotations

import asyncio
import contextlib
import enum
import ipaddress
import logging
import multiprocessing as mp
import queue
import signal
import socket
import sys
import tempfile
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from inspect import isclass, signature
from logging import DEBUG
from threading import Thread
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import Any, TypeAlias, TypeVar
from uuid import uuid4

import numpy as np
import psutil
import torch
import zmq
import zmq.asyncio

import _msgspec_seam

# The pinned vLLM does `import msgspec` / `from msgspec import msgpack`;
# both names below are the HOST SEAM namespace (see _msgspec_seam.py).
msgspec = _msgspec_seam.seam_msgspec
msgpack_ext = msgspec.msgpack  # bound name avoids clashing with `import msgpack`

_R = TypeVar("_R")  # Return type for collective_rpc

VLLM_VERSION = "0.27.1"  # HOST SEAM of vllm.version.__version__ (pin)


# ============================================================================
# Host seams — stdlib stand-ins so the module runs without the vllm package.
# Each mirrors the real interface subset the kept code touches.
# ============================================================================


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
def init_logger(name: str):
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    seen: set[str] = set()

    # SOURCE: vllm/logger.py once-messaging wrapper (info_once/warning_once)
    class _Once:  # HOST SEAM
        # SOURCE: vllm/logger.py once-messaging wrapper (_Once.__init__ — host seam)
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, msg, *args):  # SOURCE: vllm/logger.py once-wrapper call
            key = (self._fn.__name__, msg)
            if key not in seen:
                seen.add(key)
                self._fn(msg, *args)

    log.info_once = _Once(log.info)
    log.warning_once = _Once(log.warning)
    log.debug_once = _Once(log.debug)
    return log


logger = init_logger(__name__)


# SOURCE: vllm/envs.py envs — environment flag seam (defaults per pin v0.27.1)
class envs:  # HOST SEAM
    # SOURCE: vllm/envs.py:L27 VLLM_ENGINE_READY_TIMEOUT_S
    VLLM_ENGINE_READY_TIMEOUT_S: int = 600
    # SOURCE: vllm/envs.py:L149 VLLM_ENABLE_V1_MULTIPROCESSING
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    # SOURCE: vllm/envs.py:L160 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE (consumed by ch04/ch07)
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    # SOURCE: vllm/envs.py:L208 VLLM_MSGPACK_ZERO_COPY_THRESHOLD
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    # SOURCE: vllm/envs.py:L209 VLLM_ALLOW_INSECURE_SERIALIZATION
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False
    # SOURCE: vllm/envs.py:L228 VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
    VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS: int = 5
    # SOURCE: vllm/envs.py:L67 VLLM_WORKER_MULTIPROC_METHOD
    VLLM_WORKER_MULTIPROC_METHOD: str = "fork"
    # SOURCE: vllm/envs.py:L17 + L702-L704 VLLM_RPC_BASE_PATH — pin 默认
    # tempfile.gettempdir()（TYPE_CHECKING 注解与运行时 lambda 同源；
    # 消费位 vllm/utils/network_utils.py:L142 get_open_zmq_ipc_path）。
    # round-1 曾误写 "/tmp/vllm_rpc"（v0.21.0 时代旧默认）：Linux 真平台上
    # 该目录不存在，第一条 ROUTER bind 即 ZMQError——见 impl-notes 回修记录。
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    # SOURCE: vllm/envs.py VLLM_PORT
    VLLM_PORT: int | None = None


# SOURCE: vllm/envs.py:L27 VLLM_ENGINE_READY_TIMEOUT_S import target
VLLM_ENGINE_READY_TIMEOUT_S = envs.VLLM_ENGINE_READY_TIMEOUT_S


# SOURCE: vllm/utils/torch_utils.py PIN_MEMORY — no CUDA on the host seam
PIN_MEMORY = False  # HOST SEAM: torch.cuda.is_available() is False here


# SOURCE: vllm/exceptions.py VLLMServerError — seam base (Exception替身, ch04 同款)
class VLLMServerError(Exception):  # HOST SEAM
    pass


# SOURCE: vllm/v1/engine/exceptions.py:L12-L21 EngineDeadError (逐字, 基类为 seam)
class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable."""

    # SOURCE: vllm/v1/engine/exceptions.py:L12-L21 EngineDeadError.__init__ (逐字)
    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = (
            "EngineCore encountered an issue. "
            "See stack trace (above) for the root cause."
        )

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context


# SOURCE: vllm/utils/async_utils.py in_loop — seam of the loop-affinity check
def in_loop(loop: asyncio.AbstractEventLoop) -> bool:  # HOST SEAM
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


# SOURCE: vllm/utils/system_utils.py:L168-L181 get_mp_context — HOST SEAM (win32):
# vLLM is Linux-only and defaults to fork; Windows hosts only have spawn.
# SOURCE: vllm/utils/system_utils.py:L168-L181 get_mp_context
def get_mp_context():
    mp_method = "spawn" if sys.platform == "win32" else envs.VLLM_WORKER_MULTIPROC_METHOD
    return mp.get_context(mp_method)


# SOURCE: vllm/v1/engine/coordinator.py stats/kill helpers — seam for the kept
# process-tree kill in the module-level shutdown() helper below.
# SOURCE: vllm/v1/engine/coordinator.py stats/kill helpers
def kill_process_tree(pid: int):  # HOST SEAM
    try:
        psutil.Process(pid).kill()
    except psutil.Error:
        pass


# ============================================================================
# vllm/v1/engine/__init__.py — wire-format structs (byte tags + payloads)
# ============================================================================

# Reserved UTILITY call ids: the DP wave/EEP and fault-tolerance control
# planes fold their notifications into the same utility channel. They are
# only named here; the control planes themselves live in ch34/ch39.
# SOURCE: vllm/v1/engine/__init__.py:L33-L35
EEP_NOTIFICATION_CALL_ID = -1

FT_STATUS_CALL_ID = -2

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
# SOURCE: vllm/v1/engine/__init__.py:L29-L31
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")


# SOURCE: vllm/v1/engine/__init__.py:L43-L65 FinishReason (逐字)
class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g. KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    # SOURCE: vllm/v1/engine/__init__.py:L64-L65 FinishReason.__str__ (逐字)
    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


# SOURCE: vllm/v1/engine/__init__.py:L68-L94 EngineCoreReadyResponse (逐字)
@dataclass
# SOURCE: vllm/v1/engine/__init__.py:L68-L94 EngineCoreReadyResponse (逐字)
class EngineCoreReadyResponse:
    """Sent from EngineCore to each frontend at the end of engine startup.

    Contains post-initialization config that may differ from the original
    values (e.g. max_model_len after KV cache auto-fitting).
    """

    max_model_len: int
    num_gpu_blocks: int
    block_size: int
    dp_stats_address: str | None
    dtype: str
    vllm_version: str
    world_size: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    decode_context_parallel_size: int
    data_parallel_rank: int
    max_num_seqs: int
    max_num_batched_tokens: int
    instance_id: str
    # KV cache capacity (None for encoder-only/attention-free models).
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None
    kv_events_config: KVEventsConfig | None = None  # ch37 邻域 seam


# SOURCE: vllm/v1/engine/__init__.py:L97-L154 EngineCoreRequest (字段逐字;
# msgspec.Struct 为 host seam, 线格式 array_like/omit_defaults 等价)
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    prompt_is_token_ids: list[bool] | None = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    trace_headers: Mapping[str, str] | None = None
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None

    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None

    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    abort_immediately: bool = False

    # SOURCE: vllm/v1/engine/__init__.py:L148-L154 params property (逐字)
    @property
    # SOURCE: vllm/v1/engine/__init__.py:L148-L154 params property (逐字)
    def params(self):
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


# SUBTRACTED: vllm/v1/engine/__init__.py:L157-L181 EngineCoreEvent + events
# payload (tracing 轴, delete 项 6/机械删除——events 字段随 EngineCoreOutput 删)


# SOURCE: vllm/v1/engine/__init__.py:L184-L215 EngineCoreOutput (字段子集:
# SUBTRACTED new_logprobs/new_prompt_logprobs_tensors (ch7 logprob 载荷) /
# events (tracing) / kv_transfer_params+ec_transfer_params (ch36 KV 迁移) /
# trace_headers (ch4 前端透传) / prefill_stats+routed_experts (metrics/MoE 轴))
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    request_id: str
    new_token_ids: list[int]

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    num_nans_in_logits: int = 0

    # SOURCE: vllm/v1/engine/__init__.py:L213-L215 finished property (逐字)
    @property
    # SOURCE: vllm/v1/engine/__init__.py:L213-L215 finished property (逐字)
    def finished(self) -> bool:
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L218-L227 UtilityOutput (逐字)
class UtilityOutput(
    msgspec.Struct,
    array_like=True,
    gc=False,
):
    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    result: UtilityResult | None = None


# SOURCE: vllm/v1/engine/__init__.py:L230-L258 EngineCoreOutputs (SUBTRACTED
# wave_complete/start_wave — DP wave 控制面, delete 项 1)
class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None  # metrics 轴 seam
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # SUBTRACTED: wave_complete/start_wave (DP 控制面, delete 项 1 → ch34)

    # SOURCE: vllm/v1/engine/__init__.py:L256-L258 __post_init__ (逐字)
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


# SOURCE: vllm/v1/engine/__init__.py:L261-L274 EngineCoreRequestType (逐字)
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    # SUBTRACTED: START_DP_WAVE = b"\x02" (DP wave 控制面, delete 项 1 → ch34)
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"


# SUBTRACTED: vllm/v1/engine/__init__.py:L277-L299 ReconfigureDistributedRequest
# / ReconfigureRankType / EngineStatusType (elastic EP + FT 控制面, delete 项 2/3 → ch39)


# ============================================================================
# Config / request seams — the full assembly line is the ch03 product; only
# the fields the kept transport code reads are mirrored here.
# ============================================================================


# SOURCE: vllm/sampling_params.py SamplingParams — 字段 seam (跨线 dataclass 载荷)
@dataclass
# SOURCE: vllm/sampling_params.py SamplingParams
class SamplingParams:  # HOST SEAM
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1


# SOURCE: vllm/pooling_params.py PoolingParams — seam 占位 (ch6 邻域)
@dataclass
# SOURCE: vllm/pooling_params.py PoolingParams
class PoolingParams:  # HOST SEAM
    task: str = "embed"


# SOURCE: vllm/lora/request.py LoRARequest — seam 占位
@dataclass
# SOURCE: vllm/lora/request.py LoRARequest
class LoRARequest:  # HOST SEAM
    lora_name: str | None = None


# SOURCE: vllm/multimodal/inputs.py MultiModalFeatureSpec — seam 占位 (ch6 邻域)
class MultiModalFeatureSpec:  # HOST SEAM
    pass


# SOURCE: vllm/v1/metrics/stats.py SchedulerStats — metrics 轴 seam (消费端已删,
# 字段保留供 EngineCoreOutputs.scheduler_stats 注解)
@dataclass
# SOURCE: vllm/v1/metrics/stats.py SchedulerStats
class SchedulerStats:  # HOST SEAM
    num_running_reqs: int = 0


# SOURCE: vllm/config/kv_events.py KVEventsConfig — seam 占位 (ch37 邻域)
@dataclass
# SOURCE: vllm/config/kv_events.py KVEventsConfig
class KVEventsConfig:  # HOST SEAM
    pass


# SOURCE: vllm/config.py MultimodalConfig.mm_tensor_ipc — 字段 seam (m6 开关)
@dataclass
# SOURCE: vllm/config.py MultimodalConfig.mm_tensor_ipc
class MultimodalConfig:  # HOST SEAM
    mm_tensor_ipc: str = "none"


# SOURCE: vllm/config.py ParallelConfig — 字段 seam (delete 项 1/2 的 DP 旗标
# 只作默认值保留, 其消费分支已删)
@dataclass
# SOURCE: vllm/config.py ParallelConfig
class ParallelConfig:  # HOST SEAM
    data_parallel_size: int = 1
    data_parallel_size_local: int = 1
    data_parallel_rank: int = 0
    data_parallel_rank_local: int | None = None
    data_parallel_index: int = 0
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_master_port: int | None = None
    _data_parallel_master_port_list: list = field(default_factory=list)
    data_parallel_rpc_port: int | None = None
    data_parallel_external_lb: bool = False
    data_parallel_backend: str = "mp"
    local_engines_only: bool = False
    enable_elastic_ep: bool = False  # SUBTRACTED 消费分支 (delete 项 2 → ch39)
    enable_fault_tolerance: bool = False  # SUBTRACTED 消费分支 (delete 项 3 → ch39)
    world_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    decode_context_parallel_size: int = 1


# SOURCE: vllm/config.py CacheConfig — 字段 seam (ready 回填目标)
@dataclass
# SOURCE: vllm/config.py CacheConfig
class CacheConfig:  # HOST SEAM
    num_gpu_blocks: int | None = None
    block_size: int = 16
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None


# SOURCE: vllm/config.py SchedulerConfig — 字段 seam
@dataclass
# SOURCE: vllm/config.py SchedulerConfig
class SchedulerConfig:  # HOST SEAM
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True


# SOURCE: vllm/config.py ModelConfig — 字段 seam
@dataclass
# SOURCE: vllm/config.py ModelConfig
class ModelConfig:  # HOST SEAM
    max_model_len: int = 4096
    dtype: str = "torch.float32"
    runner_type: str = "generate"
    is_moe: bool = False
    multimodal_config: MultimodalConfig | None = None


# SOURCE: vllm/config.py VllmConfig — 字段 seam (装配线是 ch03 章产物)
@dataclass
class VllmConfig:  # HOST SEAM
    model_config: ModelConfig
    cache_config: CacheConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    instance_id: str
    shutdown_timeout: int = 0
    compilation_config: Any | None = None
    kv_transfer_config: Any | None = None

    # SOURCE: vllm/v1/engine/core.py:L1192 vllm_config.__post_init__() (握手后回调)
    def __post_init__(self):
        pass


# ============================================================================
# vllm/v1/request.py — request seam (from_engine_core_request 子集)
# ============================================================================


# SOURCE: vllm/v1/request.py:L348-L364 RequestStatus (子集逐字)
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()


# SOURCE: vllm/v1/request.py:L373-L375 RequestStatus.get_finished_reason + L378-L390
# _FINISHED_REASON_MAP (子集逐字; 映射体 L383-L388)
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}


# SOURCE: vllm/v1/request.py:L60-L82 Request — 字段 seam (ch9 调度器的输入面)
class Request:  # HOST SEAM
    # SOURCE: vllm/v1/request.py:L60-L82 Request.__init__ — 字段 seam (ch9)
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        client_index: int = 0,
        prompt_embeds: torch.Tensor | None = None,
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list | None = None,
        sampling_params: Any | None = None,
        pooling_params: Any | None = None,
        arrival_time: float = 0.0,
        lora_request: Any | None = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        resumable: bool = False,
        external_req_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict | None = None,
        abort_immediately: bool = False,
        structured_output_request: Any | None = None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.client_index = client_index
        self.prompt_embeds = prompt_embeds
        self.prompt_is_token_ids = prompt_is_token_ids
        self.mm_features = mm_features
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.arrival_time = arrival_time
        self.lora_request = lora_request
        self.cache_salt = cache_salt
        self.priority = priority
        self.trace_headers = trace_headers
        self.resumable = resumable
        self.external_req_id = external_req_id
        self.reasoning_ended = reasoning_ended
        self.reasoning_parser_kwargs = reasoning_parser_kwargs
        self.abort_immediately = abort_immediately
        self.structured_output_request = structured_output_request
        self.status: RequestStatus | None = None
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(prompt_token_ids or [])

    # SOURCE: vllm/v1/request.py:L223-L247 from_engine_core_request (子集逐字)
    @classmethod
    # SOURCE: vllm/v1/request.py:L223-L247 from_engine_core_request (子集逐字)
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            resumable=request.resumable,
            external_req_id=request.external_req_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
        )

    # SOURCE: vllm/v1/request.py:L267-L269 use_structured_output property (逐字)
    @property
    # SOURCE: vllm/v1/request.py:L267-L269 use_structured_output property (逐字)
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None


# ============================================================================
# ENGINE SEAM (ch9 boundary) — the scheduler / model-executor interior of the
# engine is the ch09 five-beat busy loop. This seam keeps the surface the
# retained EngineCore / EngineCoreProc code touches and lets tests play the
# scheduler by scripting each step's outputs (delivered over the real UTILITY
# thin RPC). No forward pass is faked: the engine only emits what tests hand
# it, exactly like the ch04 companion's emit_step_outputs contract.
# ============================================================================


# SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager — seam
# (grammar_init runs on the input thread, off the busy loop; body is ch31 邻域)
# SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager
class StructuredOutputManager:  # ENGINE SEAM
    # SOURCE: vllm/v1/structured_output/__init__.py:L114 grammar_init — ENGINE SEAM no-op
    def grammar_init(self, request: Request) -> None:
        return None

    # SOURCE: vllm/v1/structured_output/__init__.py:L488 clear_backend — ENGINE SEAM no-op
    def clear_backend(self) -> None:
        return None


# SOURCE: vllm/v1/core/sched/scheduler.py Scheduler (ch9 章产物) — ENGINE SEAM:
# add_request/has_requests/finish_requests 镜像真实接口; take_scheduled_batch +
# update_from_output 站在 scheduler.schedule + model_executor.execute_model +
# scheduler.update_from_output 的位置 (分桶镜像 scheduler.py:L1924
# outputs[request.client_index].append 与 L2015-L2016 的组装字典)。
class SchedulerSeam:
    """Scripted stand-in for the real Scheduler (ch9 boundary).

    Tests enqueue one "step" at a time (`enqueue_step_outputs`); each step is
    a list of per-request entries {request_id, new_token_ids, finish_reason}.
    `take_scheduled_batch` plays schedule+execute (it waits briefly for a
    script, like the real engine waits on the forward pass);
    `update_from_output` buckets the step's outputs per request.client_index
    and attaches finished-request sets — the m9 routing seam.
    """

    # SOURCE: vllm/v1/core/sched/scheduler.py Scheduler.__init__ (ch9) — ENGINE SEAM
    def __init__(self):
        self.requests: dict[str, Request] = {}
        self._client_index: dict[str, int] = {}
        self._scripted: deque[list[dict]] = deque()
        self._pending_finishes: dict[int, dict[str, FinishReason]] = {}
        self._cond = threading.Condition()

    # SOURCE: vllm/v1/core/sched/scheduler.py add_request (ch9)
    def add_request(self, request: Request) -> None:  # ENGINE SEAM
        with self._cond:
            self.requests[request.request_id] = request
            self._client_index[request.request_id] = request.client_index
            self._cond.notify_all()

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2406-L2419 has_requests (ch9) —
    # real: has_unfinished_requests() or has_finished_requests() (finished
    # requests stay bookkept until update_from_output flushes them, so an
    # abort keeps the engine stepping and rides the next step's outputs);
    # seam mirror: _pending_finishes is the finished-but-unflushed set.
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2406-L2419 has_requests (ch9)
    def has_requests(self) -> bool:  # ENGINE SEAM
        return bool(self.requests) or bool(self._pending_finishes)

    # SOURCE: vllm/v1/core/sched/scheduler.py finish_requests (ch9) — 返回被
    # finish 的 Request 列表 (EngineCore._send_abort_outputs 的输入)
    # SOURCE: vllm/v1/core/sched/scheduler.py finish_requests (ch9)
    def finish_requests(
        self, request_ids: list[str] | None, status: RequestStatus
    ) -> list[Request]:  # ENGINE SEAM
        ids = list(self.requests) if request_ids is None else request_ids
        finished: list[Request] = []
        with self._cond:
            for rid in ids:
                req = self.requests.pop(rid, None)
                if req is None:
                    continue
                req.status = status
                reason = _FINISHED_REASON_MAP[status]
                self._pending_finishes.setdefault(req.client_index, {})[rid] = reason
                finished.append(req)
            self._cond.notify_all()
        return finished

    # SOURCE: vllm/v1/engine/core.py:L595-L604 schedule + execute_model (ch9)
    def take_scheduled_batch(self) -> tuple[bool, list[dict]]:  # ENGINE SEAM
        deadline = time.monotonic() + 0.05
        with self._cond:
            while not self._scripted:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, []
                self._cond.wait(remaining)
            return True, list(self._scripted.popleft())

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1924 outputs[request.client_index].append
    # + L2015-L2016 {client_index: EngineCoreOutputs(outputs=outs)} (分桶镜像)
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1924 outputs[request.client_index].append
    def update_from_output(self, scheduled: list[dict]) -> dict[int, EngineCoreOutputs]:  # ENGINE SEAM
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        finished_ids: dict[int, set[str]] = defaultdict(set)
        with self._cond:
            for entry in scheduled:
                rid = entry["request_id"]
                ci = self._client_index.get(rid, 0)
                reason = entry["finish_reason"]
                outputs[ci].append(
                    EngineCoreOutput(
                        request_id=rid,
                        new_token_ids=entry["new_token_ids"],
                        finish_reason=None if reason is None else FinishReason(reason),
                    )
                )
                if reason is not None:
                    self.requests.pop(rid, None)
                    finished_ids[ci].add(rid)
            # Finishes marked since the last step (aborts from
            # _process_aborts_queue) ride on this step's messages.
            for ci, pending in self._pending_finishes.items():
                for rid, reason in pending.items():
                    if rid in finished_ids.get(ci, ()):
                        continue
                    outputs[ci].append(
                        EngineCoreOutput(request_id=rid, new_token_ids=[], finish_reason=reason)
                    )
                    finished_ids[ci].add(rid)
            self._pending_finishes.clear()
            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step. (mirror of scheduler.py:L2015-L2016)
            engine_core_outputs = {}
            for ci, outs in outputs.items():
                eco = EngineCoreOutputs(outputs=outs)
                if finished_ids.get(ci):
                    eco.finished_requests = finished_ids[ci]
                engine_core_outputs[ci] = eco
            return engine_core_outputs

    # (busy loop 产出注入位: step() 产出 → output_queue) — ENGINE SEAM: 测试注入一步的产出
    # SOURCE: vllm/v1/engine/core.py:L1436-L1442
    def enqueue_step_outputs(self, steps: list[dict]) -> None:  # ENGINE SEAM
        with self._cond:
            self._scripted.append(list(steps))
            self._cond.notify_all()

    # SOURCE: vllm/v1/core/sched/scheduler.py shutdown (ch9)
    def shutdown(self) -> None:  # ENGINE SEAM
        return None


# SOURCE: vllm/v1/executor/mp_unipro... MultiprocExecutor (ch03 工厂①/ch09) —
# ENGINE SEAM: supported_tasks 是 UTILITY RPC get_supported_tasks 的返回值;
# num_gpu_blocks 的确定 (真实: determine_available_memory 剖析) 由 seam 给定值代行;
# fail_executor 走真实 executor_fail_callback 布线 (worker 死亡时的真实机制)。
class UniprocExecutor:  # ENGINE SEAM
    # (类头; 真实无 __init__, worker 生命周期在 _init_executor) — ENGINE SEAM
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L45-L48 UniProcExecutor
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.supported_tasks = ("generate", "pooling")
        self._failure_callback: Callable[[], None] | None = None
        if vllm_config.cache_config.num_gpu_blocks is None:
            vllm_config.cache_config.num_gpu_blocks = 128

    # SOURCE: vllm/v1/engine/core.py:L134-L135 register_failure_callback (逐字位置)
    def register_failure_callback(self, callback: Callable[[], None]) -> None:  # ENGINE SEAM
        self._failure_callback = callback

    # SOURCE: vllm/v1/executor/multiproc_executor worker-death path (ch9)
    def fail_executor(self) -> None:  # ENGINE SEAM test hook
        cb = self._failure_callback
        assert cb is not None, "no failure callback registered"
        cb()

    # SOURCE: vllm/v1/engine/core.py:L754-L755 model_executor.shutdown()
    def shutdown(self) -> None:  # ENGINE SEAM
        return None


# ============================================================================
# vllm/utils/network_utils.py — the single ZMQ socket factory (m1/m7/m14)
# ============================================================================


# SOURCE: vllm/utils/network_utils.py:L27-L31 close_sockets (逐字)
def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


# SOURCE: vllm/utils/network_utils.py:L103-L108 is_valid_ipv6_address (逐字)
def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


# SOURCE: vllm/utils/network_utils.py:L134-L138 get_tcp_uri (逐字)
def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


# SOURCE: vllm/utils/network_utils.py:L141-L142 get_open_zmq_ipc_path (逐字)
def get_open_zmq_ipc_path() -> str:
    # HOST SEAM (win32): zmq has no ipc:// transport on Windows; loopback tcp
    # keeps the same bind-then-connect flow (and still resolves through
    # LAST_ENDPOINT for the tcp:0 placeholder path).
    if sys.platform == "win32":
        return get_tcp_uri("127.0.0.1", get_open_port())
    base_rpc_path = envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_rpc_path}/{uuid4()}"


# SOURCE: vllm/utils/network_utils.py:L146-L148 get_open_zmq_inproc_path (逐字)
def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"


# SOURCE: vllm/utils/network_utils.py:L150-L167 get_open_port — SUBTRACTED:
# VLLM_DP_MASTER_PORT 保留段 (DP 主进程端口预留, delete 项 1)
# SOURCE: vllm/utils/network_utils.py:L150-L167 get_open_port
def get_open_port() -> int:
    return _get_open_port()


# SOURCE: vllm/utils/network_utils.py:L169-L208 _get_open_port (核心分支逐字)
def _get_open_port(
    start_port: int | None = None,
    max_attempts: int | None = None,
) -> int:
    start_port = start_port if start_port is not None else envs.VLLM_PORT
    port = start_port
    if port is not None:
        attempts = 0
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
            attempts += 1
            if max_attempts is not None and attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not find open port after {max_attempts} "
                    f"attempts starting at port {start_port}"
                )
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


# SOURCE: vllm/utils/network_utils.py:L242-L259 split_zmq_path (逐字)
def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    scheme, _, rest = path.partition("://")
    if not scheme or not rest:
        raise ValueError(f"Invalid zmq path: {path}")
    host, _, port = rest.rpartition(":")
    return scheme, host, port


# SUBTRACTED: vllm/utils/network_utils.py:L283 注释行保留于 make_zmq_socket 处
# SOURCE: vllm/utils/network_utils.py:L283-L342 make_zmq_socket — SUBTRACTED:
# router_handover 参数链 (delete 项 2, elastic EP → ch39) 与 XPUB_VERBOSE
# (XPUB 仅 DP coordinator 使用, delete 项 1)
# SOURCE: vllm/utils/network_utils.py:L283-L342 make_zmq_socket
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
) -> zmq.Socket | zmq.asyncio.Socket:
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    # SUBTRACTED: vllm/utils/network_utils.py:L328-L329 XPUB_VERBOSE (coordinator, 项 1)
    # SUBTRACTED: vllm/utils/network_utils.py:L331-L335 IPv6 检测尾段
    # (依赖 urllib3 parse_url 的 split_zmq_path; 本章拓扑不走 IPv6, 见 impl-notes)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


# SOURCE: vllm/utils/network_utils.py:L346-L370 zmq_socket_ctx (逐字, minus router_handover)
@contextlib.contextmanager
# SOURCE: vllm/utils/network_utils.py:L346-L370 zmq_socket_ctx (逐字, minus router_handover)
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    linger: int = 0,
    identity: bytes | None = None,
) -> Generator[zmq.Socket, None, None]:
    """Context manager for a ZMQ socket"""
    ctx = zmq.Context()
    try:
        yield make_zmq_socket(
            ctx,
            path,
            socket_type,
            bind=bind,
            identity=identity,
        )
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)


# ============================================================================
# vllm/v1/engine/utils.py — engine process orchestration + startup handshake
# ============================================================================

# SOURCE: vllm/v1/engine/utils.py:L42 STARTUP_POLL_PERIOD_MS
STARTUP_POLL_PERIOD_MS = 10000


# SOURCE: vllm/v1/engine/utils.py:L45-L49 CoreEngineState (逐字)
class CoreEngineState(enum.Enum):
    NEW = enum.auto()
    CONNECTED = enum.auto()
    READY = enum.auto()


# SOURCE: vllm/v1/engine/utils.py:L52-L58 CoreEngine (逐字)
class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    # SOURCE: vllm/v1/engine/utils.py:L54-L58 CoreEngine.__init__ (逐字)
    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


# SOURCE: vllm/v1/engine/utils.py:L61-L74 EngineZmqAddresses (逐字)
@dataclass
# SOURCE: vllm/v1/engine/utils.py:L61-L74 EngineZmqAddresses (逐字)
class EngineZmqAddresses:
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None


# SOURCE: vllm/v1/engine/utils.py:L77-L85 EngineHandshakeMetadata (逐字)
@dataclass
# SOURCE: vllm/v1/engine/utils.py:L77-L85 EngineHandshakeMetadata (逐字)
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]


# SOURCE: vllm/v1/engine/utils.py:L120-L250 CoreEngineProcManager — SUBTRACTED:
# set_assigned_physical_gpu_ids_for_dp_rank / numa_utils.configure_subprocess
# (平台绑定与 NUMA 亲和, delete 项 6 邻域; 见 impl-notes 机械删除表)
class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    # SOURCE: vllm/v1/engine/utils.py:L126-L153 CoreEngineProcManager.__init__
    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type,
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        context = get_mp_context()
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        # SUBTRACTED: vllm/v1/engine/utils.py:L154 `from vllm.v1.engine.core import
        # EngineCoreProc` (单模块化; 见 impl-notes 机械删除表)
        self.processes: list[BaseProcess] = []
        local_dp_ranks = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            # Start EngineCore in background process.
            local_dp_ranks.append(local_index)
            self.processes.append(
                context.Process(
                    target=EngineCoreProc.run_engine_core,
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )

        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        try:
            for proc in self.processes:
                proc.start()
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.shutdown()

    # SOURCE: vllm/v1/engine/utils.py:L216-L220 shutdown (逐字)
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine core processes with configurable timeout."""
        self.manager_stopped.set()
        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)

    # SOURCE: vllm/v1/engine/utils.py:L222-L239 monitor_engine_liveness (逐字)
    def monitor_engine_liveness(self) -> None:
        """Monitor engine core process liveness."""

        sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
        sentinels = set(sentinel_to_proc.keys())

        while sentinels and not self.manager_stopped.is_set():
            died_sentinels = connection.wait(sentinels, timeout=1)

            for sentinel in died_sentinels:
                proc = sentinel_to_proc.pop(sentinel)
                exitcode = proc.exitcode
                if exitcode != 0 and not self.manager_stopped.is_set():
                    self.failed_proc_name = proc.name
            if died_sentinels:
                break

        self.shutdown()

    # SOURCE: vllm/v1/engine/utils.py:L241-L242 sentinels (逐字)
    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    # SOURCE: vllm/v1/engine/utils.py:L244-L250 finished_procs (逐字)
    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


# SOURCE: vllm/v1/engine/utils.py:L253-L277 SignalCallback (逐字)
class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""

    # SOURCE: vllm/v1/engine/utils.py:L256-L260 SignalCallback.__init__ (逐字)
    def __init__(self, callback: Callable[[], None]):
        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    # SOURCE: vllm/v1/engine/utils.py:L267-L270 _run (逐字)
    def _run(self):
        self._event.wait()
        if not self._stopped:
            self._callback()

    # SOURCE: vllm/v1/engine/utils.py:L272-L273 trigger (逐字)
    def trigger(self):
        self._event.set()

    # SOURCE: vllm/v1/engine/utils.py:L275-L277 stop (逐字)
    def stop(self):
        self._stopped = True
        self._event.set()


# SOURCE: vllm/v1/utils.py:L590-L645 shutdown — HOST SEAM: kill_process_tree
# 走 psutil (真实: vllm.utils 的进程树击杀)
# SOURCE: vllm/v1/utils.py:L590-L645 shutdown
def shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
    """Shutdown processes with timeout.

    Args:
        procs: List of processes to shutdown
        timeout: Maximum time in seconds for graceful shutdown
    """
    if timeout is None:
        # Keep a small grace period for best-effort cleanup paths that do not
        # have a user-configured shutdown timeout.
        timeout = 5.0

    logger.debug(
        "[shutdown] Process manager: start process_count=%d timeout=%ss names=%s",
        len(procs),
        timeout,
        (",").join([proc.name for proc in procs]),
    )

    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            logger.info(
                "[shutdown] Process manager: send sigterm to process %s", proc.name
            )
            proc.terminate()

    # Allow time for remaining procs to terminate.
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    remaining_procs = [
        (proc.pid, proc.name)
        for proc in procs
        if proc.is_alive() and proc.pid is not None
    ]
    if remaining_procs:
        logger.warning(
            "[shutdown] Process manager: force killing remaining processes count=%d",
            len(remaining_procs),
        )
    for pid, proc_name in remaining_procs:
        logger.warning(
            "[shutdown] Process manager: force killing remaining process %s pid %d",
            proc_name,
            pid,
        )
        kill_process_tree(pid)

    logger.debug_once("[shutdown] Process manager: complete")


# SUBTRACTED: vllm/v1/engine/utils.py:L280-L352 set_assigned_physical_gpu_ids...
# / get_physical_gpu_ids_for_local_dp_rank (平台 GPU 绑定, delete 项 6 邻域)
# SUBTRACTED: vllm/v1/engine/utils.py:L355-L364 _apply_dp_identity_suffix (Ray DP)
# SUBTRACTED: vllm/v1/engine/utils.py:L367-L1003 CoreEngineActorManager (Ray 后端, delete 项 1)


# SOURCE: vllm/v1/utils.py:L152-L163 get_engine_client_zmq_addr (逐字)
def get_engine_client_zmq_addr(
    local_only: bool,
    host: str,
    port: int = 0,
) -> str:
    """Return an IPC path (``local_only=True``) or ``tcp://host:port``.

    ``port=0`` lets the kernel assign the port at ``bind()`` time; the
    caller must recover it via ``getsockopt(zmq.LAST_ENDPOINT)``."""
    if local_only:
        return get_open_zmq_ipc_path()
    return get_tcp_uri(host, port)


# SOURCE: vllm/v1/engine/utils.py:L1005-L1050 get_engine_zmq_addresses — SUBTRACTED:
# defer_api_server_ports 参数 (Rust 前端消费, delete 项 8) 与 elastic EP 翻转 (项 2)
def get_engine_zmq_addresses(
    vllm_config: VllmConfig,
    num_api_servers: int = 1,
) -> EngineZmqAddresses:
    """Allocate ZMQ addresses for engine-client communication.

    By default each TCP address is a ``tcp://host:0`` placeholder; the
    consumer (API-server child or single-process ``MPClient``) binds, then
    recovers the kernel-assigned port via ``getsockopt(zmq.LAST_ENDPOINT)``
    and writes it back into ``addresses`` before the engine handshake.

    IPC paths are unaffected."""
    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_size = parallel_config.data_parallel_size
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/features/data_parallel/data_parallel_offline.py.
    offline_mode = local_start_index is not None

    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )

    def _addr() -> str:  # SOURCE: vllm/v1/engine/utils.py:L1042-L1045 (port-0 占位)
        if client_local_only:
            return get_open_zmq_ipc_path()
        return get_tcp_uri(host, 0)

    return EngineZmqAddresses(
        inputs=[_addr() for _ in range(num_api_servers)],
        outputs=[_addr() for _ in range(num_api_servers)],
    )


# SOURCE: vllm/v1/engine/utils.py:L1053-L1203 launch_core_engines — SUBTRACTED:
# DPCoordinator 运行段 (delete 项 1); Ray 后端分支 (项 1); external-LB 双握手
# (client_handshake_address) 链 (项 1); yield 由四元组收窄为三元组
@contextlib.contextmanager
# SOURCE: vllm/v1/engine/utils.py:L1053-L1203 launch_core_engines
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type,
    log_stats: bool,
    addresses: EngineZmqAddresses,
) -> Generator[
    tuple[
        CoreEngineProcManager | None,
        EngineZmqAddresses,
        Queue | None,
    ]
]:
    """Launch engine and DP coordinator processes as needed."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    # Create a single tensor IPC queue for sharing multimodal tensors between
    # API servers and engine core. Returns a single queue since we only support
    # DP=1 for this data flow.
    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = get_mp_context().Queue()

    # SUBTRACTED: vllm/v1/engine/utils.py:L1087-L1110 run_coordinator block (项 1)
    # SUBTRACTED: vllm/v1/engine/utils.py:L1112-L1123 ray backend block (项 1)

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        # SUBTRACTED: vllm/v1/engine/utils.py:L1136-L1145 rank>0 external-LB 分支 (项 1)
        raise AssertionError(
            "SUBTRACTED path: rank>0 external LB (ch34). Unreachable with dp=1."
        )

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # Preserve "port=0 means auto-pick" for the handshake address, which
    # is consumed by engines spawned in this process and so cannot defer
    # port resolution to bind time.
    rpc_port = parallel_config.data_parallel_rpc_port or get_open_port()
    handshake_address = get_engine_client_zmq_addr(handshake_local_only, host, rpc_port)

    # SUBTRACTED: vllm/v1/engine/core_client.py external-LB 本地握手 (delete 项 1)
    local_handshake_address = handshake_address

    if local_engines_only and dp_rank > 0:
        # SUBTRACTED: vllm/v1/engine/utils.py:L1163-L1166 external-LB 本地握手 (项 1)
        raise AssertionError(
            "SUBTRACTED path: local_engines_only rank>0 (ch34). Unreachable with dp=1."
        )

    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        # Start local engines.
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=None,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        yield local_engine_manager, addresses, tensor_queue

        # Now wait for engines to start.
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            local_engine_manager,
        )


# SOURCE: vllm/v1/engine/utils.py:L1206-L1346 wait_for_engine_startup — SUBTRACTED:
# coordinator 进程 sentinel (项 1); remote headless 校验段 (external LB, 项 1);
# MoE DP config-hash 校验段 (项 1)
# SOURCE: vllm/v1/engine/utils.py:L1206-L1346 wait_for_engine_startup
def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    coordinated_dp: bool,
    proc_manager: CoreEngineProcManager | None,
):
    # Wait for engine core process(es) to send ready messages.
    local_count = parallel_config.data_parallel_size_local
    remote_count = len(core_engines) - local_count
    # [local, remote] counts
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    if proc_manager is not None:
        for sentinel in proc_manager.sentinels():
            # HOST SEAM (win32): the real line registers the spawn sentinel
            # with the zmq poller; on win32 the sentinel is a raw pipe HANDLE
            # (an int) that zmq.Poller misreports as immediately readable
            # (POLLERR) while the child is still alive. There we keep it out
            # of the poller and detect exited children via finished_procs()
            # every iteration instead (same observable contract: engine death
            # during startup -> RuntimeError "Engine core initialization
            # failed" listing the failed procs).
            if sys.platform != "win32":
                poller.register(sentinel, zmq.POLLIN)
    while any(conn_pending) or any(start_pending):
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if sys.platform == "win32" and proc_manager is not None:
            finished = proc_manager.finished_procs()
            if finished:
                raise RuntimeError(
                    "Engine core initialization failed. "
                    "See root cause above. "
                    f"Failed core proc(s): {finished}"
                )
        if not events:
            if any(conn_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue
        if len(events) > 1 or events[0][0] != handshake_socket:
            # One of the local core processes exited.
            finished = proc_manager.finished_procs() if proc_manager else {}
            raise RuntimeError(
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"
            )

        # Receive HELLO and READY messages from the input socket.
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        eng_index = int.from_bytes(eng_identity, "little")
        engine = next((e for e in core_engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        msg = msgpack_ext.decode(ready_msg_bytes)
        status, local, headless = msg["status"], msg["local"], msg["headless"]
        if local != engine.local:
            raise RuntimeError(
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # SUBTRACTED: vllm/v1/engine/utils.py:L1277-L1290 remote headless 校验 (项 1)

        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # Send init message with DP config info.
            init_message = msgpack_ext.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={
                        k: getattr(parallel_config, k)
                        for k in (
                            "data_parallel_master_ip",
                            "data_parallel_master_port",
                            "_data_parallel_master_port_list",
                            "data_parallel_size",
                        )
                    }
                    if coordinated_dp
                    else {},
                )
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            conn_pending[0 if local else 1] -= 1
            start_pending[0 if local else 1] += 1
            engine.state = CoreEngineState.CONNECTED
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # SUBTRACTED: vllm/v1/engine/utils.py:L1315-L1330 MoE DP config hash 校验 (项 1)
            start_pending[0 if local else 1] -= 1
            engine.state = CoreEngineState.READY
        else:
            raise RuntimeError(
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        logger.debug(
            "%s from %s core engine process %s.",
            status,
            "local" if local else "remote",
            eng_index,
        )


# ============================================================================
# vllm/v1/serial_utils.py — MsgpackEncoder/MsgpackDecoder (m4/m5/m6 编解码)
# ============================================================================

# SOURCE: vllm/v1/serial_utils.py:L41-L43 CUSTOM_TYPE_* (RAW_VIEW 保留; PICKLE/
# CLOUDPICKLE 常量随 pickle 回退分支删, delete 项 7)
CUSTOM_TYPE_RAW_VIEW = 3

# SOURCE: vllm/v1/serial_utils.py:L54 bytestr (逐字)
bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


# SOURCE: vllm/v1/serial_utils.py:L57-L71 OOBTensorConsumer (逐字)
class OOBTensorConsumer(ABC):
    @abstractmethod
    # SOURCE: vllm/v1/serial_utils.py:L58-L65 OOBTensorConsumer.__call__ (逐字)
    def __call__(self, tensor: torch.Tensor) -> dict | None:
        """
        Called with tensors for the current message.
        Returns None to reject the tensor (falls back to regular serialization),
        otherwise a dict with arbitrary placeholder data to be included
        in the serialized message.
        """
        return None

    @abstractmethod
    # SOURCE: vllm/v1/serial_utils.py:L68-L71 OOBTensorConsumer.new_message (逐字)
    def new_message(self) -> None:
        """Called at the start of each new encoded message."""
        pass


# dtype, shape, metadata -> tensor
# SOURCE: vllm/v1/serial_utils.py:L74-L75 OOBTensorProvider (逐字)
OOBTensorProvider = Callable[[str, tuple[int, ...], dict], torch.Tensor]


# SOURCE: vllm/v1/serial_utils.py:L78-L82 _log_insecure_serialization_warning (逐字)
def _log_insecure_serialization_warning():
    logger.warning_once(
        "Allowing insecure serialization using pickle due to "
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
    )


# SUBTRACTED: vllm/v1/serial_utils.py:L85-L126 _typestr / _encode_type_info_recursive
# / _decode_type_info_recursive (不安全回退的类型账本, delete 项 7)


# SOURCE: vllm/v1/serial_utils.py:L129-L133 UtilityResult (逐字)
class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""

    # SOURCE: vllm/v1/serial_utils.py:L132-L134 UtilityResult.__init__ (逐字)
    def __init__(self, r: Any = None):
        self.result = r


# SOURCE: vllm/v1/serial_utils.py:L136-L310 MsgpackEncoder — SUBTRACTED:
# _encode_mm_items/_encode_mm_item/_encode_mm_field_elem/_encode_nested_tensors/
# _encode_mm_field (多模态工厂, delete 项 7 → ch6); pickle/cloudpickle 回退
# (项 7, VLLM_ALLOW_INSECURE_SERIALIZATION 逃生舱在 enc_hook 的 TypeError 里点名)
class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.

    When a ``oob_tensor_consumer`` is provided, tensors (CUDA and CPU) will be
    offered to it for out-of-band handling.
    """

    # SOURCE: vllm/v1/serial_utils.py:L149-L164 MsgpackEncoder.__init__ (逐字)
    def __init__(
        self,
        size_threshold: int | None = None,
        oob_tensor_consumer: OOBTensorConsumer | None = None,
    ):
        if size_threshold is None:
            size_threshold = envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD
        self.encoder = msgpack_ext.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: list[bytestr] | None = None
        self.size_threshold = size_threshold
        self.oob_tensor_consumer = oob_tensor_consumer
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    # SOURCE: vllm/v1/serial_utils.py:L166-L178 encode (逐字; encoder 为 host seam)
    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    # SOURCE: vllm/v1/serial_utils.py:L180-L189 encode_into (逐字; encoder 为 host seam)
    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    # SOURCE: vllm/v1/serial_utils.py:L191-L235 enc_hook — SUBTRACTED: mm items
    # 分派 (项 7) 与 insecure pickle 分支 (项 7); UtilityResult 保留安全分支
    # SOURCE: vllm/v1/serial_utils.py:L191-L235 enc_hook
    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Fall back to pickle for object or void kind ndarrays.
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        if isinstance(obj, slice):
            # We are assuming only int-based values will be used here.
            return tuple(
                int(v) if v is not None else None
                for v in (obj.start, obj.stop, obj.step)
            )

        if isinstance(obj, UtilityResult):
            result = obj.result
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                return None, result
            # SUBTRACTED: vllm/v1/serial_utils.py:L216-L219 递归类型账本 (项 7)

        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable"
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based serialization."
            )

        # SUBTRACTED: vllm/v1/serial_utils.py:L228-L235 pickle/cloudpickle Ext 分支 (项 7)
        raise AssertionError("unreachable: insecure serialization subtracted")

    # SOURCE: vllm/v1/serial_utils.py:L237-L255 _encode_ndarray (逐字)
    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.aux_buffers is not None
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack_ext.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data

    # SOURCE: vllm/v1/serial_utils.py:L257-L273 _encode_tensor (逐字)
    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | dict | memoryview]:
        oob_consumer = self.oob_tensor_consumer
        # view the tensor as a contiguous 1D array of bytes
        if obj.nbytes < self.size_threshold and obj.is_cpu:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack_ext.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
        elif oob_consumer is not None and (data := oob_consumer(obj)) is not None:
            assert isinstance(data, dict)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            assert self.aux_buffers is not None
            data = len(self.aux_buffers)
            self.aux_buffers.append(tensor_data(obj))
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    # SUBTRACTED: vllm/v1/serial_utils.py:L275-L310 _encode_mm_* (delete 项 7 → ch6)


# SOURCE: vllm/v1/serial_utils.py:L313-L483 MsgpackDecoder — SUBTRACTED: mm decode
# 工厂 (项 7); pickle ext 分支 (项 7); decoder 为 host seam
class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    ``oob_tensor_provider`` must be used when an OOBTensorConsumer is used on the
    encoder side.
    """

    # SOURCE: vllm/v1/serial_utils.py:L323-L338 MsgpackDecoder.__init__ (逐字)
    def __init__(
        self,
        t: Any | None = None,
        share_mem: bool = True,
        oob_tensor_provider: OOBTensorProvider | None = None,
    ):
        self.share_mem = share_mem
        self.pin_tensors = PIN_MEMORY
        args = () if t is None else (t,)
        self.decoder = msgpack_ext.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )
        self.aux_buffers: Sequence[bytestr] = ()
        self.oob_tensor_provider = oob_tensor_provider
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    # SOURCE: vllm/v1/serial_utils.py:L340-L348 decode (逐字)
    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):  # type: ignore
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    # SOURCE: vllm/v1/serial_utils.py:L350-L365 dec_hook — SUBTRACTED: mm 分支 (项 7)
    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if t is slice:
                return slice(*obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
        return obj

    # SOURCE: vllm/v1/serial_utils.py:L367-L379 _decode_utility_result — SUBTRACTED:
    # insecure 递归转换 (项 7), 保留其 TypeError 守卫
    # SOURCE: vllm/v1/serial_utils.py:L367-L379 _decode_utility_result
    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        result_type, result = obj
        if result_type is not None:
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise TypeError(
                    "VLLM_ALLOW_INSECURE_SERIALIZATION must "
                    "be set to use custom utility result types"
                )
        return UtilityResult(result)

    # SUBTRACTED: vllm/v1/serial_utils.py:L381-L387 _convert_result (项 7)

    # SOURCE: vllm/v1/serial_utils.py:L389-L397 _decode_ndarray (逐字)
    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        arr = np.frombuffer(buffer, dtype=dtype)
        if not self.share_mem:
            arr = arr.copy()
        return arr.reshape(shape)

    # SOURCE: vllm/v1/serial_utils.py:L399-L425 _decode_tensor (逐字)
    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        if isinstance(data, dict):
            assert self.oob_tensor_provider, (
                "Received OOB tensor but tensor provider is not set"
            )
            return self.oob_tensor_provider(dtype, shape, data)

        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        buffer = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer.nbytes:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Clone ensures tensor is backed by pytorch-owned memory for safe
        # future async CPU->GPU transfer.
        # Pin larger tensors for more efficient CPU->GPU transfer.
        if not is_aux:
            arr = arr.clone()
        elif not self.share_mem:
            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    # SUBTRACTED: vllm/v1/serial_utils.py:L427-L471 _decode_mm_* (delete 项 7 → ch6)

    # SOURCE: vllm/v1/serial_utils.py:L473-L483 ext_hook — SUBTRACTED: pickle 分支 (项 7)
    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data

        raise NotImplementedError(f"Extension type code {code} is not supported")


# SUBTRACTED: vllm/v1/serial_utils.py:L486-L510 run_method (collective_rpc 面, 项 5/1)
# SUBTRACTED: vllm/v1/serial_utils.py:L513-L628 PydanticMsgspecMixin (API server 面, 项 8)


# SOURCE: vllm/v1/utils.py:L777-L787 tensor_data (逐字)
def tensor_data(tensor: torch.Tensor) -> memoryview:
    """Get the raw data of a tensor as a uint8 memoryview, useful for
    serializing and hashing.

    Args:
        tensor: The input tensor.

    Returns:
        A memoryview of the tensor data as uint8.
    """
    return tensor.flatten().cpu().contiguous().view(torch.uint8).numpy().data


# ============================================================================
# vllm/v1/engine/tensor_ipc.py — OOB shared-memory bypass (m6, kept in full)
# ============================================================================

# SOURCE: vllm/v1/engine/tensor_ipc.py:L1-L11 module docstring (逐字)
"""Tensor IPC transport via torch.multiprocessing.Queue.

This module contains the queue-based transport logic for sharing tensors
between processes (e.g., API server -> engine core). The msgpack layer
emits/consumes lightweight :class:`TensorIpcData` values, while transport
state such as request association, handle generation, queue routing, buffering,
and cleanup lives here.
"""

# SOURCE: vllm/v1/engine/tensor_ipc.py:L27
TensorIpcQueue = Queue


# SOURCE: vllm/v1/engine/tensor_ipc.py:L30-L42 TensorIpcData (逐字)
@dataclass
# SOURCE: vllm/v1/engine/tensor_ipc.py:L30-L42 TensorIpcData (逐字)
class TensorIpcData:
    """
    Data sent via torch.multiprocessing.Queue for zero-copy IPC.

    Contains the tensor_id and the actual tensor. The tensor is
    shared in memory (GPU or CPU) for efficient inter-process communication.
    """

    sender_id: str
    message_id: int
    tensor_id: int
    tensor: torch.Tensor


# SOURCE: vllm/v1/engine/tensor_ipc.py:L45-L105 TensorIpcSender (逐字, minus debug log 块)
class TensorIpcSender(OOBTensorConsumer):
    """Send-side logic for tensor IPC via torch.multiprocessing.Queue.

    Uses a single queue targeting rank 0 (the only rank that consumes
    multimodal tensors during TP>1 / PP>1. Note: DP>1 not supported).
    """

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L52-L60 TensorIpcSender.__init__
    def __init__(self, queue: TensorIpcQueue):
        self.queue = queue
        self._tensor_id_counter = 0
        self._message_counter = 0
        self._sender_id = uuid.uuid4().hex[:8]

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L58-L63 set_target_engine (逐字)
    def set_target_engine(self, target_engine: int) -> None:
        if target_engine != 0:
            raise IndexError(
                "TensorIpcSender only supports a single queue; "
                f"got target engine {target_engine}"
            )

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L65-L67 new_message (逐字)
    def new_message(self) -> None:
        self._message_counter += 1
        self._tensor_id_counter = 0

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L69-L105 __call__ (逐字, minus L90-L96 debug)
    def __call__(self, tensor: torch.Tensor) -> dict[str, Any] | None:
        """Send tensor via queue, return its handle. Returns None if failed."""
        try:
            # Move tensor to shared memory for IPC
            # This is required for proper inter-process communication
            if not tensor.is_shared():
                tensor = tensor.share_memory_()

            metadata = {
                "sender_id": self._sender_id,
                "message_id": self._message_counter,
                "tensor_id": self._tensor_id_counter,
            }

            self._tensor_id_counter += 1

            ipc_data = TensorIpcData(**metadata, tensor=tensor)  # type: ignore[arg-type]

            # Use a timeout to avoid blocking indefinitely
            self.queue.put(ipc_data, timeout=10.0)

            # SUBTRACTED: vllm/v1/engine/tensor_ipc.py:L90-L96 logger.debug 块 (delete 项 6)

            return metadata
        except Exception as e:
            logger.warning(
                "Failed to send tensor via IPC queue: %s. "
                "Falling back to standard serialization.",
                e,
            )
            return None


# SOURCE: vllm/v1/engine/tensor_ipc.py:L108-L111 _Sender (逐字)
@dataclass
# SOURCE: vllm/v1/engine/tensor_ipc.py:L108-L111 _Sender (逐字)
class _Sender:
    current_message_id: int = -1
    tensors: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)


# SOURCE: vllm/v1/engine/tensor_ipc.py:L114-L178 TensorIpcReceiver (逐字, minus debug)
class TensorIpcReceiver:
    """Receive-side logic for tensor IPC via torch.multiprocessing.Queue.

    Wraps the queue receive logic previously embedded in MsgpackDecoder.
    """

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L120-L123 TensorIpcReceiver.__init__
    def __init__(self, queue: TensorIpcQueue):
        self.queue = queue
        self._tensor_buffers = defaultdict[str, _Sender](_Sender)

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L124-L178 __call__ (逐字, minus L156-L163 debug)
    def __call__(
        self, dtype: str, shape: tuple[int, ...], meta: dict[str, Any]
    ) -> torch.Tensor:
        """Retrieve a tensor from torch.multiprocessing.Queue.

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        """

        # Create lookup key from handle
        sender_id: str = meta["sender_id"]
        message_id: int = meta["message_id"]
        tensor_id: int = meta["tensor_id"]

        # Drain all available tensors. We save them regardless if this is
        # the one we're waiting for as they may arrive out of order from
        # multiple producers.
        while True:
            sender = self._tensor_buffers.get(sender_id)
            if sender is not None:
                tensors = sender.tensors
                tensor = tensors.get(message_id, {}).pop(tensor_id, None)
                if tensor is not None:
                    if sender.current_message_id != message_id:
                        while tensors and (mid := next(iter(tensors))) < message_id:
                            if sender.tensors.pop(mid):
                                # HOST SEAM fix: pin L151-L154 passes 1 arg to 2
                                # placeholders (crashes under pytest log capture).
                                logger.warning(
                                    "Discarding stale tensors from message %d of sender %s",
                                    mid,
                                    sender_id,
                                )
                        sender.current_message_id = message_id
                    # SUBTRACTED: vllm/v1/engine/tensor_ipc.py:L156-L163 logger.debug 块 (项 6)
                    return tensor

            ipc_data: TensorIpcData = self.queue.get(timeout=10.0)

            # Store tensor
            sender = self._tensor_buffers[ipc_data.sender_id]
            if sender.current_message_id > ipc_data.message_id:
                logger.warning(
                    "Ignoring stale tensor from sender %s", ipc_data.sender_id
                )
                continue

            sender.tensors.setdefault(ipc_data.message_id, {})[ipc_data.tensor_id] = (
                ipc_data.tensor
            )


# ============================================================================
# vllm/v1/engine/core.py — EngineCore (interior = ch9 seam) + EngineCoreProc
# ============================================================================

# SOURCE: vllm/v1/engine/core.py:L98 HANDSHAKE_TIMEOUT_MINS (逐字)
HANDSHAKE_TIMEOUT_MINS = 5


# SOURCE: vllm/v1/engine/core.py:L103-L104 EngineCore (docstring 逐字)
class EngineCore:
    """Inner loop of vLLM's Engine."""

    # SOURCE: vllm/v1/engine/core.py:L106-L247 __init__ — SUBTRACTED: 插件加载/
    # KV cache 初始化/真实 Scheduler 构造/mm registry/batch queue/GC freeze/
    # envs cache (ch06/ch09 章域 + delete 项 6; scheduler/executor 由 ENGINE SEAM 代行)
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        self.vllm_config = vllm_config
        self.log_stats = log_stats

        # Setup Model. (ENGINE SEAM: executor_class 是 ch03 工厂①的产物)
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        # Setup scheduler. (ENGINE SEAM: SchedulerSeam stands in for the ch09
        # Scheduler built from scheduler_config.get_scheduler_cls())
        self.structured_output_manager = StructuredOutputManager()
        self.scheduler: SchedulerSeam = SchedulerSeam()

        # SOURCE: vllm/v1/engine/core.py:L231-L233 step_fn 装配 (minus batch_queue)
        self.step_fn = self.step

        # SOURCE: vllm/v1/engine/core.py:L236 aborts_queue (逐字)
        self.aborts_queue = queue.Queue[list[str]]()

        # SOURCE: vllm/v1/engine/core.py:L238 _idle_state_callbacks (逐字)
        self._idle_state_callbacks: list[Callable] = []

    # SOURCE: vllm/v1/engine/core.py:L361-L364 get_supported_tasks — SUBTRACTED:
    # _log_pooler_config (pooling 日志, metrics 轴)
    # SOURCE: vllm/v1/engine/core.py:L361-L364 get_supported_tasks
    def get_supported_tasks(self) -> tuple:
        supported_tasks = self.model_executor.supported_tasks
        return supported_tasks

    # SOURCE: vllm/v1/engine/core.py:L439-L483 add_request — SUBTRACTED: pooling
    # task 校验 (ch6) / kv_transfer+ec_transfer 警告 (ch36) / abort_immediately
    # (KV 迁移拒绝, ch36 邻域); 保留 request_id 类型校验与 scheduler.add_request
    # SOURCE: vllm/v1/engine/core.py:L439-L483 add_request
    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        self.scheduler.add_request(request)

    # SOURCE: vllm/v1/engine/core.py:L485-L491 abort_requests (逐字)
    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # SOURCE: vllm/v1/engine/core.py:L584-L614 step — SUBTRACTED: scheduler.schedule /
    # model_executor.execute_model / sample_tokens / stats capture (ch9 五拍);
    # ENGINE SEAM: take_scheduled_batch 站在 schedule+execute 的位置
    # SOURCE: vllm/v1/engine/core.py:L584-L614 step
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        model_executed, scheduled = self.scheduler.take_scheduled_batch()
        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(scheduled)

        return engine_core_outputs, model_executed

    # SOURCE: vllm/v1/engine/core.py:L616-L623 post_step — SUBTRACTED: draft-token
    # 更新 (spec decode, ch15 邻域)
    # SOURCE: vllm/v1/engine/core.py:L616-L623 post_step
    def post_step(self, model_executed: bool) -> None:
        return None

    # SOURCE: vllm/v1/engine/core.py:L741-L749 _process_aborts_queue (逐字)
    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    # SOURCE: vllm/v1/engine/core.py:L751-L767 shutdown — SUBTRACTED: gc.unfreeze /
    # cleanup_dist_env_and_memory (host seam 无分布式环境)
    # SOURCE: vllm/v1/engine/core.py:L751-L767 shutdown
    def shutdown(self):
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()
        logger.debug_once("[shutdown] EngineCore: local resource teardown complete")

    # SUBTRACTED: vllm/v1/engine/core.py:L769-L866 profile/reset_mm_cache/
    # reset_prefix_cache/reset_encoder_cache/_reset_caches/pause_scheduler/
    # resume_scheduler/is_scheduler_paused (具体 utility 方法体, delete 项 4/5)

    # SOURCE: vllm/v1/engine/core.py:L969-L991 preprocess_add_request — SUBTRACTED:
    # mm_receiver_cache 块 (mm 接收缓存, delete 项 7 → ch6); grammar_init 调用保留
    # (输入线程执行、不占忙循环——structured_output_manager 为 seam)
    # SOURCE: vllm/v1/engine/core.py:L969-L991 preprocess_add_request
    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        req = Request.from_engine_core_request(request, None)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    # ── ENGINE SEAM utility hooks (ch9 boundary) ───────────────────────────
    # 测试扮演调度器的注入口: 经真实 UTILITY 薄 RPC 过线 (getattr 反射分派),
    # 引擎只回放测试喂的产出——与 ch04 的 emit_step_outputs 同款契约, 不伪造 forward。

    # SOURCE: vllm/v1/engine/core.py:L1090 (busy loop 产出注入位) — ENGINE SEAM
    def enqueue_step_outputs(self, steps: list) -> None:
        """Script the outputs of one engine step (test instrumentation; the
        real engine produces these in step(), ch9). Each entry:
        {"request_id", "new_token_ids", "finish_reason"}."""
        self.scheduler.enqueue_step_outputs(steps)

    # SOURCE: vllm/v1/engine/core.py (request 簿记) — ENGINE SEAM 观察口
    def get_request_info(self, request_id: str) -> dict:
        req = self.scheduler.requests.get(request_id)
        if req is None:
            raise KeyError(f"unknown request {request_id!r}")
        return {
            "client_index": req.client_index,
            "prompt_len": len(req.prompt_token_ids or ()),
            "has_embeds": req.prompt_embeds is not None,
            "embeds_numel": 0 if req.prompt_embeds is None else req.prompt_embeds.numel(),
        }

    # SOURCE: vllm/v1/engine/core.py (request 簿记) — ENGINE SEAM 观察口
    def get_request_embeds_head(self, request_id: str, n: int) -> list:
        req = self.scheduler.requests.get(request_id)
        if req is None or req.prompt_embeds is None:
            raise KeyError(f"request {request_id!r} has no embeds")
        return req.prompt_embeds.flatten()[:n].tolist()

    # SOURCE: vllm/v1/engine/core.py:L1583-L1585 (utility 失败路径) — ENGINE SEAM 测试钩
    def boom_method(self):
        raise RuntimeError("boom_method exploded (test hook)")

    # SOURCE: vllm/v1/engine/core.py:L1029-L1031 executor_fail_callback 注入位 —
    # ENGINE SEAM 测试钩: the real failure fires *inside* the model executor
    # (ch9); this hook lets a test trigger the same registered callback over
    # the real UTILITY reflective dispatch, so the EXECUTOR_FAILED sentinel,
    # busy-loop raise, and single-frame dead message all take the real path.
    # SOURCE: vllm/v1/engine/core.py:L1029-L1031 executor_fail_callback 注入位
    def fail_executor(self) -> None:
        self.model_executor.fail_executor()

    # SUBTRACTED: vllm/v1/engine/core.py:L868-L966 sleep/wake_up/collective_rpc/
    # set_weight_version (具体 utility 方法体, delete 项 5)
    # SUBTRACTED: vllm/v1/engine/core.py:L993-L999 _eep_* (elastic EP, delete 项 2)


# SOURCE: vllm/v1/engine/core.py:L1002-L1005 EngineShutdownState (逐字)
class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


# SOURCE: vllm/v1/engine/core.py:L1008-L1009 EngineCoreProc (docstring 逐字)
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    # SOURCE: vllm/v1/engine/core.py:L1011 ENGINE_CORE_DEAD (逐字)
    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses

    # SOURCE: vllm/v1/engine/core.py:L1015-L1127 __init__ — SUBTRACTED:
    # has_coordinator/internal_dp_balancing/publish_dp_lb_stats/last_counts
    # (DP stats, delete 项 1); _init_data_parallel 调用 (项 1);
    # enable_fault_tolerance/ft_sentinel (项 3); ready_event 的 coordinator
    # 等待循环 (项 1); client_handshake_address 参数 (external LB 双握手, 项 1)
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type,
        log_stats: bool,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        # SOURCE: vllm/v1/engine/core.py:L1027-L1031 (逐字)
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        # SOURCE: vllm/v1/engine/core.py:L1033-L1036 (逐字)
        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # SOURCE: vllm/v1/engine/core.py:L1038-L1042 (逐字)
        # Receiver for tensor IPC
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")

        # SOURCE: vllm/v1/engine/core.py:L1044-L1050 (逐字)
        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
        ) as addresses:
            # SUBTRACTED: vllm/v1/engine/core.py:L1051-L1068 DP stats 装配 (项 1)
            self.addresses = addresses
            self.process_input_queue_block = True

            # SOURCE: vllm/v1/engine/core.py:L1074-L1080 super().__init__ (minus
            # internal_dp_balancing/include_finished_set, delete 项 1)
            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
            )

            # SUBTRACTED: vllm/v1/engine/core.py:L1082-L1090 fault tolerance (项 3)

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            # SOURCE: vllm/v1/engine/core.py:L1092-L1119 (逐字)
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # SUBTRACTED: vllm/v1/engine/core.py:L1121-L1127 coordinator READY
            # 等待循环 (delete 项 1)

    # SOURCE: vllm/v1/engine/core.py:L1129-L1192 _perform_handshakes — SUBTRACTED:
    # client_handshake_address 双握手分支 (external LB, delete 项 1)
    @contextmanager
    # SOURCE: vllm/v1/engine/core.py:L1129-L1192 _perform_handshakes
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        # SUBTRACTED: vllm/v1/engine/core.py:L1172-L1189 rank0+本地双握手 (项 1)
        with handshake as addresses:
            yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    # SOURCE: vllm/v1/engine/core.py:L1194-L1231 _perform_handshake — SUBTRACTED:
    # DP config-hash 附着 (delete 项 1)
    @contextmanager
    # SOURCE: vllm/v1/engine/core.py:L1194-L1231 _perform_handshake
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
            }
            # SUBTRACTED: vllm/v1/engine/core.py:L1225-L1229 parallel_config_hash (项 1)

            handshake_socket.send(msgpack_ext.encode(ready_msg))

    # SOURCE: vllm/v1/engine/core.py:L1233-L1269 startup_handshake (逐字)
    @staticmethod
    # SOURCE: vllm/v1/engine/core.py:L1233-L1269 startup_handshake (逐字)
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgpack_ext.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgpack_ext.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    # SOURCE: vllm/v1/engine/core.py:L1271-L1360 run_engine_core — SUBTRACTED:
    # maybe_register_config_serialize_by_value / set_process_title /
    # maybe_init_worker_tracer / decorate_logs / numa affinity (delete 项 6);
    # kv_transfer_config dp 改名 (ch36 邻域); DPEngineCoreProc MoE 分支 (项 1)
    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            parallel_config.data_parallel_index = dp_rank
            # Non-MoE DP ranks are completely independent, so treat like DP=1.
            # Note that parallel_config.data_parallel_index will still reflect
            # the original DP rank.
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            # SOURCE: vllm/v1/engine/core.py:L1322-L1327 wakeup_engine (逐字)
            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            # SOURCE: vllm/v1/engine/core.py:L1330-L1340 signal_handler (逐字)
            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                logger.info(
                    "[shutdown] EngineCore: trigger received signal=%s",
                    signal_name,
                )
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.info_once("[shutdown] EngineCore: exiting busy loop")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()

    # SUBTRACTED: vllm/v1/engine/core.py:L1362-L1363 _init_data_parallel (DP 覆写点, 项 1)

    # SOURCE: vllm/v1/engine/core.py:L1365-L1371 has_work — SUBTRACTED:
    # engines_running (DP wave 旗标) / batch_queue (PP 批队列) 两读数 (项 1/ch9)
    # SOURCE: vllm/v1/engine/core.py:L1365-L1371 has_work
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return self.scheduler.has_requests()

    # SOURCE: vllm/v1/engine/core.py:L1373-L1375 is_running (逐字)
    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    # SOURCE: vllm/v1/engine/core.py:L1377-L1389 run_busy_loop — SUBTRACTED:
    # @fault_tolerant_wrapper (项 3); _maybe_publish_request_counts 双调用 (项 1)
    # SOURCE: vllm/v1/engine/core.py:L1377-L1389 run_busy_loop
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

        raise SystemExit

    # SUBTRACTED: vllm/v1/engine/core.py:L1391-L1402 _maybe_publish_request_counts
    # (DP LB stats 发布, delete 项 1)

    # SOURCE: vllm/v1/engine/core.py:L1404-L1433 _process_input_queue (逐字)
    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    # SOURCE: vllm/v1/engine/core.py:L1435-L1452 _process_engine_step — SUBTRACTED:
    # KV-connector 让步 sleep (ch36 邻域)
    # SOURCE: vllm/v1/engine/core.py:L1435-L1452 _process_engine_step
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        return model_executed

    # SOURCE: vllm/v1/engine/core.py:L1454-L1457 _notify_idle_state_callbacks (逐字)
    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    # SOURCE: vllm/v1/engine/core.py:L1459-L1505 _handle_shutdown — SUBTRACTED:
    # drain-timeout 模式与 pause wait/keep 差异 (delete 项 4); 保留最小路径:
    # REQUESTED → abort 在途请求 → 无工作即退出
    # SOURCE: vllm/v1/engine/core.py:L1459-L1505 _handle_shutdown
    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)
            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info(
                "[shutdown] EngineCore: request processing complete; "
                "starting resource teardown"
            )
            return False

        return True

    # SOURCE: vllm/v1/engine/core.py:L1507-L1540 _handle_client_request — SUBTRACTED:
    # _reject_add_in_shutdown/_reject_utility_in_shutdown 调用 (项 4)
    # SOURCE: vllm/v1/engine/core.py:L1507-L1540 _handle_client_request
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    # SUBTRACTED: vllm/v1/engine/core.py:L1542-L1567 _reject_add_in_shutdown /
    # _reject_utility_in_shutdown (delete 项 4)

    # SOURCE: vllm/v1/engine/core.py:L1569-L1586 _invoke_utility_method (逐字)
    @staticmethod
    # SOURCE: vllm/v1/engine/core.py:L1569-L1586 _invoke_utility_method (逐字)
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    # SOURCE: vllm/v1/engine/core.py:L1588-L1603 _convert_msgspec_args (逐字; msgspec 为 seam)
    @staticmethod
    # SOURCE: vllm/v1/engine/core.py:L1588-L1603 _convert_msgspec_args (逐字; msgspec 为 seam)
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    # SOURCE: vllm/v1/engine/core.py:L1605-L1617 _send_engine_dead (逐字)
    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    # SOURCE: vllm/v1/engine/core.py:L1619-L1643 _make_ready_response — SUBTRACTED:
    # dp_stats_address=frontend_stats_publish_address (DP stats, 项 1 → None);
    # kv_events_config=scheduler.get_kv_event_publisher_config() (ch37 邻域)
    # SOURCE: vllm/v1/engine/core.py:L1619-L1643 _make_ready_response
    def _make_ready_response(self) -> EngineCoreReadyResponse:
        parallel_config = self.vllm_config.parallel_config
        scheduler_config = self.vllm_config.scheduler_config
        return EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            block_size=self.vllm_config.cache_config.block_size,
            dp_stats_address=None,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
            vllm_version=VLLM_VERSION,
            world_size=self.vllm_config.parallel_config.world_size,
            data_parallel_size=parallel_config.data_parallel_size,
            kv_cache_size_tokens=self.vllm_config.cache_config.kv_cache_size_tokens,
            kv_cache_max_concurrency=(
                self.vllm_config.cache_config.kv_cache_max_concurrency
            ),
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            data_parallel_rank=self.engine_index,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            instance_id=self.vllm_config.instance_id,
        )

    # SOURCE: vllm/v1/engine/core.py:L1645-L1741 process_input_sockets — SUBTRACTED:
    # coord_socket/XSUB 订阅分支 (delete 项 1); b"READY" coordinator 通知分支 (项 1);
    # FT_UTILITY_METHOD 拦截 (项 3); coordinator READY 等待 (项 1)
    # SOURCE: vllm/v1/engine/core.py:L1645-L1741 process_input_sockets
    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding with optional tensor IPC receiver.
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            # SUBTRACTED: vllm/v1/engine/core.py:L1669-L1698 coordinator XSUB + READY (项 1)

            # Register sockets with poller.
            poller = zmq.Poller()
            ready_response = self._make_ready_response()
            ready_payload = msgpack_ext.encode(ready_response)
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            # SUBTRACTED: vllm/v1/engine/core.py:L1695-L1698 coordinator READY 收 (项 1)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # SUBTRACTED: vllm/v1/engine/core.py:L1706-L1710 b"READY" 忽略分支 (项 1)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    elif request_type == EngineCoreRequestType.UTILITY:
                        request = generic_decoder.decode(data_frames)
                        client_idx, call_id, method, args = request
                        # SUBTRACTED: vllm/v1/engine/core.py:L1725-L1729 FT 拦截 (项 3)
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    # SOURCE: vllm/v1/engine/core.py:L1743-L1810 process_output_sockets — SUBTRACTED:
    # coord_socket 建立与 client_index==-1 哨兵分支 (delete 项 1 → ch34)
    # SOURCE: vllm/v1/engine/core.py:L1743-L1810 process_output_sockets
    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Payload buffers that can't be reused yet because zmq may still be
        # sending them.
        # Buffers of the zero-copy tensor/ndarray frames don't need tracking
        # here: zmq itself holds a reference to each until it's done with it.
        pending = deque[tuple[zmq.MessageTracker, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            # SUBTRACTED: vllm/v1/engine/core.py:L1767-L1775 coord_socket (项 1)
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                # SUBTRACTED: vllm/v1/engine/core.py:L1788-L1793 client_index==-1
                # coordinator 哨兵 (项 1 → ch34)

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reclaimed = pending.pop()[1]
                    if len(reuse_buffers) < max_reuse_bufs:
                        reuse_buffers.append(reclaimed)

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = self._send_msg_tracking_payload(
                    sockets[client_index], buffers
                )
                if not tracker.done:
                    pending.appendleft((tracker, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    # SOURCE: vllm/v1/engine/core.py:L1812-L1827 _send_msg_tracking_payload (逐字)
    @staticmethod
    # SOURCE: vllm/v1/engine/core.py:L1812-L1827 _send_msg_tracking_payload (逐字)
    def _send_msg_tracking_payload(
        socket: zmq.Socket, buffers: Sequence[bytestr]
    ) -> zmq.MessageTracker:
        """Send `buffers` as a zero-copy multipart message, returning a tracker
        for the *first* frame.

        Used instead of `Socket.send_multipart()` because we reuse the buffer
        passed to `MsgpackEncoder.encode_into()`: `send_multipart()` returns a
        tracker for the last frame only.
        """
        more_flag = zmq.SNDMORE if len(buffers) > 1 else 0
        tracker = socket.send(buffers[0], more_flag, copy=False, track=True)
        if more_flag:
            socket.send_multipart(buffers[1:], copy=False)
        return tracker

    # SOURCE: vllm/v1/engine/core.py:L1829-L1836 _handle_request_preproc_error (逐字)
    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    # SUBTRACTED: vllm/v1/engine/core.py:L1838-L1885 pause_scheduler/_pause_complete
    # (delete 项 4)

    # SOURCE: vllm/v1/engine/core.py:L1887-L1895 _send_finish_outputs_to_client (逐字)
    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    # SOURCE: vllm/v1/engine/core.py:L1897-L1900 _send_abort_outputs_to_client (逐字)
    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    # SOURCE: vllm/v1/engine/core.py:L1902-L1905 _send_error_outputs_to_client (逐字)
    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    # SOURCE: vllm/v1/engine/core.py:L1907-L1915 _send_abort_outputs (逐字)
    def _send_abort_outputs(self, aborted_reqs: list[Request]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for request in aborted_reqs:
                by_client[request.client_index].add(request.request_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


# SUBTRACTED: vllm/v1/engine/core.py:L1918-L2488 DPEngineCoreProc /
# DPMoEEngineCoreActor / EngineCoreActor (delete 项 1 → ch34)


# ============================================================================
# vllm/v1/engine/core_client.py — the client half (stations 1, 5, 10)
# ============================================================================

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

# SOURCE: vllm/v1/engine/core_client.py:L75 EngineIdentity (逐字)
EngineIdentity = bytes


# SOURCE: vllm/v1/engine/core_client.py:L78-L139 EngineCoreClient — SUBTRACTED:
# 具体 utility 方法面 (delete 项 5) / elastic EP / fault tolerance / DP status
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    # SOURCE: vllm/v1/engine/core_client.py:L89-L112 make_client (逐字, minus @instrument)
    @staticmethod
    # SOURCE: vllm/v1/engine/core_client.py:L89-L112 make_client (逐字, minus @instrument)
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool,
    ) -> "EngineCoreClient":
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    # SOURCE: vllm/v1/engine/core_client.py:L114-L139 make_async_mp_client —
    # SUBTRACTED: DP 分流 (delete 项 1 → ch34)
    @staticmethod
    # SOURCE: vllm/v1/engine/core_client.py:L114-L139 make_async_mp_client
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        # SUBTRACTED: vllm/v1/engine/core_client.py:L133-L138 DPAsyncMPClient /
        # DPLBAsyncMPClient 分流 (项 1)
        return AsyncMPClient(*client_args)

    # SOURCE: vllm/v1/engine/core_client.py:L141-L142 shutdown (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/engine/core_client.py:L141-L142 shutdown (逐字)
    def shutdown(self, timeout: float | None = None) -> None: ...

    # SOURCE: vllm/v1/engine/core_client.py:L144-L151 get_output/add_request (逐字)
    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L147-L148 get_supported_tasks (abstract)
    def get_supported_tasks(self) -> tuple:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L150-L151 add_request (abstract)
    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/engine/core_client.py:L153-L192 profile/reset_*/sleep/
    # weight-version/lora/collective_rpc (delete 项 5)

    # SOURCE: vllm/v1/engine/core_client.py:L194-L195 abort_requests (逐字)
    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/engine/core_client.py:L197-L232 lora/save/collective
    # (delete 项 5) / dp_engines_running 保留于下方

    # SOURCE: vllm/v1/engine/core_client.py:L223-L226 dp_engines_running (逐字)
    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/engine/core_client.py:L228-L232 commit/prepare_elastic_ep (项 2)

    # SOURCE: vllm/v1/engine/core_client.py:L234-L241 async 面 (逐字)
    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L237-L238 get_supported_tasks_async (abstract)
    async def get_supported_tasks_async(self) -> tuple:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L240-L241 add_request_async (abstract)
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/engine/core_client.py:L243-L295 async utility 族 (项 5)

    # SOURCE: vllm/v1/engine/core_client.py:L268-L269 abort_requests_async (逐字)
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/engine/core_client.py:L297-L303 handle_fault/get_status (项 3)


# SOURCE: vllm/v1/engine/core_client.py:L306-L402 InprocClient — SUBTRACTED:
# 具体 utility 方法代理 (delete 项 5)
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # SOURCE: vllm/v1/engine/core_client.py:L316-L317 __init__ (逐字)
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    # SOURCE: vllm/v1/engine/core_client.py:L319-L322 get_output (逐字)
    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    # SOURCE: vllm/v1/engine/core_client.py:L324-L325 get_supported_tasks (逐字)
    def get_supported_tasks(self) -> tuple:
        return self.engine_core.get_supported_tasks()

    # SOURCE: vllm/v1/engine/core_client.py:L327-L329 add_request (逐字)
    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    # SOURCE: vllm/v1/engine/core_client.py:L331-L333 abort_requests (逐字)
    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    # SOURCE: vllm/v1/engine/core_client.py:L335-L336 shutdown (逐字)
    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    # SUBTRACTED: vllm/v1/engine/core_client.py:L338-L399 utility 代理 (项 5)

    # SOURCE: vllm/v1/engine/core_client.py:L401-L402 dp_engines_running (逐字)
    def dp_engines_running(self) -> bool:
        return False


# SOURCE: vllm/v1/engine/core_client.py:L405-L493 BackgroundResources — SUBTRACTED:
# coordinator / first_req_* / stats_update_* 套接字与任务 (delete 项 1/项 8)
@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: zmq.Context
    # If CoreEngineProcManager, it manages local engines.
    engine_manager: CoreEngineProcManager | None = None
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None
    shutdown_path: str | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    # SOURCE: vllm/v1/engine/core_client.py:L428-L488 __call__ — SUBTRACTED:
    # coordinator.shutdown (项 1); async-case 的 first_req/stats 套接字 (项 1)
    def __call__(self):
        """Clean up background resources."""

        logger.debug_once("[shutdown] MPClient: background resource cleanup start")
        self.engine_dead = True
        if self.engine_manager is not None:
            self.engine_manager.shutdown(
                timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
            )

        if isinstance(self.output_socket, zmq.asyncio.Socket):
            # Async case.
            loop = self.output_queue_task._loop if self.output_queue_task else None

            sockets = (self.output_socket, self.input_socket)

            tasks = (self.output_queue_task,)

            # SOURCE: vllm/v1/engine/core_client.py:L454-L459 close_sockets_and_tasks
            def close_sockets_and_tasks():
                close_sockets(sockets)
                for task in tasks:
                    if task is not None and not task.done():
                        with contextlib.suppress(Exception):
                            task.cancel()

            if loop is not None:
                if in_loop(loop):
                    close_sockets_and_tasks()
                elif not loop.is_closed():
                    loop.call_soon_threadsafe(close_sockets_and_tasks)
            else:
                # Loop has been closed, try to clean up directly.
                del tasks
                del close_sockets_and_tasks
                close_sockets(sockets)
                del self.output_queue_task
        else:
            # Sync case.

            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            close_sockets((self.output_socket, self.input_socket))

            if self.shutdown_path is not None:
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(self.shutdown_path)
                    # Send shutdown signal.
                    shutdown_sender.send(b"")

        logger.debug_once("[shutdown] MPClient: background resource cleanup complete")

    # SOURCE: vllm/v1/engine/core_client.py:L490-L493 validate_alive (逐字)
    def validate_alive(self, frames: Sequence[zmq.Frame]):
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


# SUBTRACTED: vllm/v1/engine/core_client.py:L496-L500 ElasticScalingCache (delete 项 2 → ch39)


# SOURCE: vllm/v1/engine/core_client.py:L503-L514 MPClient (docstring 逐字)
class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    # SOURCE: vllm/v1/engine/core_client.py:L516-L680 __init__ — SUBTRACTED:
    # stats_update_address (DP stats, 项 1); router_handover 参数链 (项 2);
    # coordinator 解包与断言 (项 1); engines_running 保留 (DP 旗标默认 False)
    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
    ):
        self.vllm_config = vllm_config

        # ZMQ setup.
        # SOURCE: vllm/v1/engine/core_client.py:L527-L528 (逐字)
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        # SOURCE: vllm/v1/engine/core_client.py:L530-L534 (逐字)
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            # State used for data parallel.
            # SOURCE: vllm/v1/engine/core_client.py:L538 (逐字)
            self.engines_running = False
            parallel_config = vllm_config.parallel_config

            # SUBTRACTED: vllm/v1/engine/core_client.py:L540-L543 elastic EP
            # handover 注释与旗标 (delete 项 2 → ch39)

            tensor_queue: Queue | None = None
            if client_addresses:
                # Engines are managed externally to this client.
                # SOURCE: vllm/v1/engine/core_client.py:L548-L563 (逐字, minus
                # stats_update_address 与 router_handover)
                input_address = client_addresses["input_address"]
                output_address = client_addresses["output_address"]
                # SUBTRACTED: vllm/v1/engine/core_client.py:L551 stats_update_address (项 1)
                # Tensor queues passed via client_addresses for multi-API-server case
                tensor_queue = client_addresses.get("tensor_queue")
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    input_address,
                    zmq.ROUTER,
                    bind=True,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )

                # Report bound endpoints back so the parent can forward
                # them to engines (mirrors the DPCoordinator pattern).
                # SOURCE: vllm/v1/engine/core_client.py:L565-L585 (逐字)
                actual_address_pipe: connection.Connection | None = client_addresses.get(
                    "actual_address_pipe"
                )
                if actual_address_pipe is not None:
                    try:
                        actual_input = self.input_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_output = self.resources.output_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_address_pipe.send(
                            {
                                "input_address": actual_input,
                                "output_address": actual_output,
                            }
                        )
                    finally:
                        actual_address_pipe.close()
            else:
                # Engines are managed by this client.
                # SOURCE: vllm/v1/engine/core_client.py:L587-L607 (逐字, minus router_handover)
                addresses = get_engine_zmq_addresses(vllm_config)
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    addresses.inputs[0],
                    zmq.ROUTER,
                    bind=True,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, addresses.outputs[0], zmq.PULL
                )

                # Resolve ``tcp://host:0`` placeholders to bound endpoints
                # before engines DEALER-connect. No-op for IPC.
                # SOURCE: vllm/v1/engine/core_client.py:L600-L607 (逐字)
                addresses.inputs[0] = self.input_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                addresses.outputs[0] = self.resources.output_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()

                # SOURCE: vllm/v1/engine/core_client.py:L609-L613 launch (minus
                # coordinator, delete 项 1; yield 三元组)
                with launch_core_engines(
                    vllm_config, executor_class, log_stats, addresses
                ) as (engine_manager, addresses, tensor_queue):
                    self.resources.engine_manager = engine_manager

                # SUBTRACTED: vllm/v1/engine/core_client.py:L615-L619 stats address (项 1)

            # Serialization setup with tensor queues for multimodal tensor IPC.
            # SOURCE: vllm/v1/engine/core_client.py:L621-L629 (逐字)
            tensor_ipc_sender: TensorIpcSender | None = None
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is not None and model_config.multimodal_config is not None:
                mm_tensor_ipc = model_config.multimodal_config.mm_tensor_ipc
                if mm_tensor_ipc == "torch_shm" and tensor_queue is not None:
                    tensor_ipc_sender = TensorIpcSender(tensor_queue)

            # SOURCE: vllm/v1/engine/core_client.py:L629-L630 (逐字)
            self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)

            # SOURCE: vllm/v1/engine/core_client.py:L632-L644 (逐字)
            dp_size = parallel_config.data_parallel_size
            dp_rank = parallel_config.data_parallel_index
            dp_local_size = parallel_config.data_parallel_size_local
            offline_mode = parallel_config.data_parallel_rank_local is not None
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
            self.engine_ranks_managed = (
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
            )
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )

            # ZMQ identity of each engine that this client will talk to.
            # SOURCE: vllm/v1/engine/core_client.py:L646-L649 (逐字)
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]

            # Wait for ready messages from each engine on the input socket.
            # SOURCE: vllm/v1/engine/core_client.py:L651-L669 (逐字)
            identities = set(self.core_engines)
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
                ):
                    raise TimeoutError(
                        f"Timed out waiting for engine core processes to "
                        f"start. This is often caused by slow weight loading "
                        f"for large models. Waited "
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                        f"timeout, set the environment variable: "
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                    )
                identity, payload = sync_input_socket.recv_multipart()
                identities.remove(identity)
                self._apply_ready_response(payload)

            # SOURCE: vllm/v1/engine/core_client.py:L671-L672 (逐字)
            self.core_engine: EngineIdentity = self.core_engines[0]
            self.utility_results: dict[int, AnyFuture] = {}

            # Start monitoring engine core processes for unexpected failures
            # SOURCE: vllm/v1/engine/core_client.py:L674-L675 (逐字)
            self.start_engine_core_monitor()

            success = True
        finally:
            if not success:
                self._finalizer()

    # SOURCE: vllm/v1/engine/core_client.py:L682-L693 shutdown (逐字)
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        if self._finalizer.detach() is not None:
            timeout_str = "default" if timeout is None else f"{timeout}s"
            logger.info("[shutdown] MPClient: start timeout=%s", timeout_str)
            if self.resources.engine_manager is not None:
                logger.info_once("[shutdown] MPClient: stopping engine manager")
                self.resources.engine_manager.shutdown(timeout=timeout)
                logger.info_once("[shutdown] MPClient: engine manager stopped")
            logger.info_once("[shutdown] MPClient: cleaning up background resources")
            self.resources()
            logger.info_once("[shutdown] MPClient: complete")

    # SOURCE: vllm/v1/engine/core_client.py:L695-L699 _format_exception (逐字)
    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    # SOURCE: vllm/v1/engine/core_client.py:L701-L703 ensure_alive (逐字)
    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    # SOURCE: vllm/v1/engine/core_client.py:L705-L706 dp_engines_running (逐字)
    def dp_engines_running(self) -> bool:
        return self.engines_running

    # SOURCE: vllm/v1/engine/core_client.py:L708-L735 start_engine_core_monitor (逐字)
    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        engine_manager = self.resources.engine_manager
        if engine_manager is None:
            # No engine processes to monitor
            return

        self_ref = weakref.ref(self)

        # Monitor engine core process liveness. If any die unexpectedly,
        # marks the engine as dead, and shuts down the client.
        # SOURCE: vllm/v1/engine/core_client.py:L719-L731 monitor_engine_cores (逐字)
        def monitor_engine_cores():
            engine_manager.monitor_engine_liveness()
            _self = self_ref()
            if not _self or not _self._finalizer.alive or _self.resources.engine_dead:
                return
            _self.resources.engine_dead = True
            logger.warning_once(
                "[shutdown] MPClient: engine core exited unexpectedly; starting cleanup"
            )
            _self.shutdown()
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()

    # SOURCE: vllm/v1/engine/core_client.py:L737-L777 _apply_ready_response —
    # SUBTRACTED: dp_stats_address 块 (external DP LB, delete 项 1)
    def _apply_ready_response(self, payload: bytes) -> None:
        """Decode an EngineCoreReadyResponse and sync any post-initialization
        config changes (e.g. auto-fitted max_model_len) back to the frontend."""
        if not payload:
            return
        vllm_config = self.vllm_config
        response = msgpack_ext.decode(payload, type=EngineCoreReadyResponse)
        vllm_config.model_config.max_model_len = min(
            vllm_config.model_config.max_model_len, response.max_model_len
        )

        # Setup KV cache config with initialization state from
        # engine core process. Sum num_gpu_blocks from all engines in DP case.
        # SOURCE: vllm/v1/engine/core_client.py:L750-L752 (逐字)
        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        num_gpu_blocks += response.num_gpu_blocks
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

        # Sync block_size: may be enlarged by _align_hybrid_block_size in the
        # worker for hybrid Mamba models.
        # SOURCE: vllm/v1/engine/core_client.py:L754-L768 (逐字)
        cache_config = vllm_config.cache_config
        cache_config.block_size = response.block_size
        # Keep these as per-engine cache_config_info values; do not sum across DP.
        cache_config.kv_cache_size_tokens = (
            getattr(cache_config, "kv_cache_size_tokens", None)
            if getattr(cache_config, "kv_cache_size_tokens", None) is not None
            else response.kv_cache_size_tokens
        )
        cache_config.kv_cache_max_concurrency = (
            getattr(cache_config, "kv_cache_max_concurrency", None)
            if getattr(cache_config, "kv_cache_max_concurrency", None) is not None
            else response.kv_cache_max_concurrency
        )

        # SUBTRACTED: vllm/v1/engine/core_client.py:L770-L777 dp_stats_address 同步 (项 1)


# SOURCE: vllm/v1/engine/core_client.py:L780-L799 _process_utility_output (逐字)
def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


# SOURCE: vllm/v1/engine/core_client.py:L802-L971 SyncMPClient — SUBTRACTED:
# is_dp 旗标 (项 1); 具体 utility 方法面 (项 5); wave_complete 消费 (项 1)
class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    # SOURCE: vllm/v1/engine/core_client.py:L806-L870 __init__ (逐字, minus @instrument/is_dp)
    def __init__(
        self, vllm_config: VllmConfig, executor_class: type, log_stats: bool
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path

        # SOURCE: vllm/v1/engine/core_client.py:L831-L859 process_outputs_socket (逐字)
        def process_outputs_socket():
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        # SOURCE: vllm/v1/engine/core_client.py:L861-L867 (逐字)
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        # SOURCE: vllm/v1/engine/core_client.py:L869-L870 (逐字)
        self.resources.output_socket = None

    # SOURCE: vllm/v1/engine/core_client.py:L872-L882 get_output — SUBTRACTED:
    # wave_complete 消费 (delete 项 1)
    # SOURCE: vllm/v1/engine/core_client.py:L872-L882 get_output
    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()

        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L884-L891 _send_input (逐字)
    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        self.input_socket.send_multipart(msg, copy=False)

    # SOURCE: vllm/v1/engine/core_client.py:L893-L899 call_utility (逐字)
    def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        return future.result()

    # SOURCE: vllm/v1/engine/core_client.py:L901-L902 get_supported_tasks (逐字)
    def get_supported_tasks(self) -> tuple:
        return self.call_utility("get_supported_tasks")

    # SOURCE: vllm/v1/engine/core_client.py:L904-L907 add_request (逐字, minus is_dp)
    def add_request(self, request: EngineCoreRequest) -> None:
        self._send_input(EngineCoreRequestType.ADD, request)

    # SOURCE: vllm/v1/engine/core_client.py:L909-L911 abort_requests (逐字)
    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # SUBTRACTED: vllm/v1/engine/core_client.py:L913-L971 具体 utility 方法 (项 5)


# SOURCE: vllm/v1/engine/core_client.py:L974-L1246 AsyncMPClient — SUBTRACTED:
# _engine_status/FT 装配 (项 3); EEP 通知回调 (项 2); 具体 utility 方法 (项 5)
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    # SOURCE: vllm/v1/engine/core_client.py:L978-L1014 __init__ (逐字, minus FT)
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type,
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()

        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    # SOURCE: vllm/v1/engine/core_client.py:L1016-L1091 _ensure_output_queue_task —
    # SUBTRACTED: EEP notification_callback_handler (项 2); FT 分支 (项 3)
    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Any] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self)
        output_socket = resources.output_socket
        assert output_socket is not None

        # SUBTRACTED: vllm/v1/engine/core_client.py:L1033-L1035 EEP 回调钩 (项 2)

        # SOURCE: vllm/v1/engine/core_client.py:L1037-L1087 process_outputs_socket —
        # SUBTRACTED: EEP/FT 分支 (项 2/3)
        # SOURCE: vllm/v1/engine/core_client.py:L1037-L1087 process_outputs_socket
        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(
                            outputs.utility_output, utility_results
                        )
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    # SOURCE: vllm/v1/engine/core_client.py:L1093-L1102 get_output_async (逐字)
    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L1104-L1114 _send_input (逐字)
    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Any:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine)

    # SOURCE: vllm/v1/engine/core_client.py:L1116-L1123 _send_input_message (逐字)
    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity
    ) -> Any:
        self.ensure_alive()
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        return self.input_socket.send_multipart((engine,) + message, copy=False)

    # SOURCE: vllm/v1/engine/core_client.py:L1125-L1140 utility RPC (逐字)
    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    # SOURCE: vllm/v1/engine/core_client.py:L1128-L1140 _call_utility_async
    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine)
        self._ensure_output_queue_task()
        return await future

    # SOURCE: vllm/v1/engine/core_client.py:L1142-L1143 get_supported_tasks_async (逐字)
    async def get_supported_tasks_async(self) -> tuple:
        return await self.call_utility_async("get_supported_tasks")

    # SOURCE: vllm/v1/engine/core_client.py:L1145-L1148 add_request_async (逐字)
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1150-L1152 abort_requests_async (逐字)
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # SUBTRACTED: vllm/v1/engine/core_client.py:L1154-L1246 具体 utility 方法与
    # FT/status 面 (delete 项 5/3)


# SUBTRACTED: vllm/v1/engine/core_client.py:L1249-L1696 DPAsyncMPClient /
# DPLBAsyncMPClient (delete 项 1 → ch34)
