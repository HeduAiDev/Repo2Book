# Subtract-only companion for v3 ch09 «EngineCore 的逐拍循环» (Part III:
# 引擎的心跳——调度循环; the L0 loop box blown up beat by beat).
#
# FAITHFUL SUBSET of the real vLLM engine-core loop at pin v0.27.1
# (6e448d0ea). It keeps vLLM's names, structure and control flow; it only
# DELETES branches approved in the dossier subtraction_plan (plus mechanical
# deletions listed in impl-notes.md) and marks every deletion with
# `# SUBTRACTED:`. Mapping rule: take the real vLLM source, drop every
# SUBTRACTED branch, and you should get (approximately) this file.
#
# What this chapter owns, kept FULLY REAL:
#   * EngineCore.step() — the five beats: ① schedule → ② execute_model
#     (non_block=True) → ③ get_grammar_bitmask → ④ future.result() +
#     conditional sample_tokens → aborts → ⑤ update_from_output (F1 payoff);
#   * the busy loop: run_busy_loop / _process_input_queue (idle parks in
#     input_queue.get(block=True)) / _process_engine_step (incl. the 1ms GIL
#     yield) / has_work / _handle_shutdown (abort & drain modes);
#   * the two IO threads + two queue.Queue (process_input_sockets /
#     process_output_sockets, encode_into buffer reuse, first-frame tracker,
#     linger=4000), the two-layer startup handshake (HELLO → addresses →
#     READY, then EngineCoreReadyResponse on every data DEALER), the death
#     sentinel ENGINE_CORE_DEAD, and the signal path (REQUESTED + WAKEUP);
#   * the worker two-phase contract: execute_model returns None and stashes
#     ExecuteModelState; sample_tokens unpacks → apply_grammar_bitmask →
#     greedy sample; misuse raises the "State error" guard verbatim;
#     AsyncOutputFuture waits only the D2H copy event.
#
# Host seams (each marked `HOST SEAM` / `ENGINE SEAM` inline and registered
# in impl-notes.md): msgspec is backed by _msgspec_seam.py (genuine msgpack
# bytes); the deep interiors that belong to other chapters' companions are
# same-signature minimal seams — the scheduler's RUNNING/WAITING loop body
# (ch10/ch11), the model forward itself (ch17; tests script the logits rows
# through the real UTILITY thin RPC — no forward is faked inside the loop),
# the grammar FSM compiler (ch30; the bitmask rows are scripted the same way),
# the full sampling stack (greedy argmax is the temperature=0 branch of the
# real Sampler), and the xgrammar bitmask kernel (a CPU stand-in with the
# kernel's documented set-disallowed-to--inf semantics). vLLM itself is
# Linux-only, so ipc:// addresses fall back to loopback tcp on win32 hosts
# and mp uses spawn there.
#
# Runs on a CPU host WITHOUT the vllm package. Every def/class carries a
# `# SOURCE: vllm/...:Lxxx` ref into the pinned tree (line numbers re-verified
# against v0.27.1 on 2026-08-22, not copied from v2's v0.21.0 assets).

from __future__ import annotations

import contextlib
import enum
import gc
import ipaddress
import queue
import signal
import socket
import sys
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from inspect import isclass, signature
from logging import DEBUG
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from typing import Any, NamedTuple, TypeAlias, TypeVar

import numpy as np
import psutil
import torch
import zmq

import _msgspec_seam

# The pinned vLLM does `import msgspec` / `from msgspec import msgpack`; both
# names below are the HOST SEAM namespace (see _msgspec_seam.py).
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
    log = __import__("logging").getLogger(name)
    if not log.handlers:
        log.addHandler(__import__("logging").NullHandler())
    seen: set[str] = set()

    # SOURCE: vllm/logger.py once-messaging wrapper (info_once/warning_once)
    class _Once:  # HOST SEAM
        # SOURCE: vllm/logger.py once-messaging wrapper (_Once.__init__ — host seam)
        def __init__(self, fn):
            self._fn = fn

        # SOURCE: vllm/logger.py once-wrapper call
        def __call__(self, msg, *args):
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
    # SOURCE: vllm/envs.py:L208 VLLM_MSGPACK_ZERO_COPY_THRESHOLD
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False
    VLLM_PORT: int | None = None
    VLLM_RPC_BASE_PATH: str = "/tmp/vllm"
    VLLM_WORKER_MULTIPROC_METHOD: str = "fork"


# SOURCE: vllm/utils/torch_utils.py PIN_MEMORY — host seam (no CUDA host)
PIN_MEMORY = False  # HOST SEAM


# SOURCE: vllm/utils/system_utils.py:L168-L181 get_mp_context — HOST SEAM (win32):
# the real default is fork/unix; zmq socket handles and torch CUDA context do
# not survive fork, and win32 has no fork at all — spawn keeps the same
# observable contract (EngineCoreProc.run_engine_core in a fresh process).
# SOURCE: vllm/utils/system_utils.py:L168-L181 get_mp_context
def get_mp_context():
    import multiprocessing as mp

    method = "spawn" if sys.platform == "win32" else envs.VLLM_WORKER_MULTIPROC_METHOD
    return mp.get_context(method)


# SOURCE: vllm/utils/__init__.py kill_process_tree — HOST SEAM: psutil stand-in
# (真实: vllm.utils 的进程树击杀)
# SOURCE: vllm/utils/__init__.py kill_process_tree — HOST SEAM: psutil stand-in
def kill_process_tree(pid: int):  # HOST SEAM
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        pass


# SOURCE: vllm/utils/gc_utils.py:L96-L108 freeze_gc_heap (逐字)
def freeze_gc_heap() -> None:
    """
    Freeze all objects tracked by the garbage collector. It should be invoked
    after server init / warmup, to reduce GC overhead from static objects
    during serving time.
    """
    # Ensure all static objects are pushed down to the oldest generation for
    # freeze
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    # Freeze all GC tracked objects
    gc.freeze()


# SOURCE: vllm/envs.py enable_envs_cache — HOST SEAM no-op (env 读取缓存)
def enable_envs_cache() -> None:
    return None


# SOURCE: vllm/distributed cleanup_dist_env_and_memory — HOST SEAM no-op
# (无分布式环境的 CPU host)
# SOURCE: vllm/distributed cleanup_dist_env_and_memory — HOST SEAM no-op
def cleanup_dist_env_and_memory() -> None:
    return None


# SOURCE: vllm/logging_utils/dump_input.py:L56-L64 dump_engine_exception —
# SUBTRACTED: _dump_engine_exception 内部 (config/调度输出的对象转储器, 观测域);
# HOST SEAM 日志替身保住『异常路径上绝不二次抛』的契约
# SOURCE: vllm/logging_utils/dump_input.py:L56-L64 dump_engine_exception
def dump_engine_exception(config, scheduler_output, scheduler_stats):
    # NOTE: ensure we can log extra info without risking raises
    # unexpected errors during logging
    with contextlib.suppress(Exception):
        logger.error(
            "Dumping input data for V1 LLM engine (v%s) with config: %s",
            VLLM_VERSION,
            config,
        )


# SOURCE: vllm/plugins load_general_plugins — HOST SEAM no-op (插件面, 观测域)
def load_general_plugins() -> None:
    return None


# SOURCE: vllm/v1/serial_utils.py:L486-L512 run_method — SUBTRACTED:
# bytes/cloudpickle 反序列化分支 (不安全序列化逃生舱, ch05 域); getattr 反射
# 与直接调用两条路径逐字
# SOURCE: vllm/v1/serial_utils.py:L486-L512 run_method
def run_method(
    obj: Any,
    method: str | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        func = method
    return func(*args, **kwargs)


# ============================================================================
# vllm/utils/network_utils.py — the ZMQ socket factory + address helpers
# ============================================================================


# SOURCE: vllm/utils/network_utils.py:L27-L31 close_sockets (逐字)
def close_sockets(sockets: Sequence[zmq.Socket]):
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


# SOURCE: vllm/utils/network_utils.py:L141-L142 get_open_zmq_ipc_path (逐字) —
# HOST SEAM (win32): zmq has no ipc:// transport on Windows; loopback tcp
# keeps the same bind-then-connect flow (and still resolves through
# LAST_ENDPOINT for the tcp:0 placeholder path).
# SOURCE: vllm/utils/network_utils.py:L141-L142 get_open_zmq_ipc_path
def get_open_zmq_ipc_path() -> str:
    if sys.platform == "win32":
        return get_tcp_uri("127.0.0.1", get_open_port())
    base_rpc_path = envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_rpc_path}/{uuid.uuid4()}"


# SOURCE: vllm/utils/network_utils.py:L150-L208 get_open_port — SUBTRACTED:
# VLLM_DP_MASTER_PORT 保留段 (DP 主进程端口预留, delete 项 1)
# SOURCE: vllm/utils/network_utils.py:L150-L208 get_open_port
def get_open_port() -> int:
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


# SOURCE: vllm/utils/network_utils.py:L283-L342 make_zmq_socket — SUBTRACTED:
# router_handover 参数链 (elastic EP, delete 项 2) 与 XPUB_VERBOSE (DP
# coordinator, delete 项 1) 及 IPv6 检测尾段 (ch05 域); HWM=0/缓冲/bind 默认
# 规则逐字
# SOURCE: vllm/utils/network_utils.py:L283-L342 make_zmq_socket
def make_zmq_socket(
    ctx: zmq.Context,
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
) -> zmq.Socket:
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    sock = ctx.socket(socket_type)

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
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        sock.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        sock.setsockopt(zmq.LINGER, linger)

    if bind:
        sock.bind(path)
    else:
        sock.connect(path)

    return sock


# SOURCE: vllm/utils/network_utils.py:L346-L370 zmq_socket_ctx (逐字, minus
# router_handover)
# SOURCE: vllm/utils/network_utils.py:L346-L370 zmq_socket_ctx
@contextlib.contextmanager
# SOURCE: vllm/utils/network_utils.py:L346-L370 zmq_socket_ctx
def zmq_socket_ctx(
    path: str, socket_type: Any, bind: bool | None = None
) -> Any:
    ctx = zmq.Context()
    sock = make_zmq_socket(ctx, path, socket_type, bind=bind)
    try:
        yield sock
    finally:
        close_sockets([sock])
        ctx.term()


# ============================================================================
# vllm/v1/serial_utils.py — wire codec subset (ch05 owns the full stack)
# ============================================================================


# SOURCE: vllm/v1/serial_utils.py:L54 bytestr (逐字)
bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


# SOURCE: vllm/v1/serial_utils.py:L129-L134 UtilityResult (逐字)
class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""

    # SOURCE: vllm/v1/serial_utils.py:L132-L134 UtilityResult.__init__ (逐字)
    def __init__(self, r: Any = None):
        self.result = r


# SOURCE: vllm/v1/serial_utils.py:L136-L310 MsgpackEncoder — SUBTRACTED:
# aux_buffer 多帧零拷贝 (_encode_tensor/_encode_ndarray/enc_hook 的张量分支,
# delete 项 4 tensor IPC 邻域 — 全量编码器是 ch05 的精简版); 本章线载荷全是
# msgpack 原生类型, 单帧足矣; pickle/cloudpickle 回退一并删 (观测逃生舱)
class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    # SOURCE: vllm/v1/serial_utils.py:L149-L164 MsgpackEncoder.__init__ —
    # SUBTRACTED: oob_tensor_consumer (项 4)
    # SOURCE: vllm/v1/serial_utils.py:L149-L164 MsgpackEncoder.__init__
    def __init__(self):
        self.encoder = msgpack_ext.Encoder(enc_hook=self.enc_hook)

    # SOURCE: vllm/v1/serial_utils.py:L166-L178 encode (逐字; encoder 为 host
    # seam, aux_buffers 单帧化)
    # SOURCE: vllm/v1/serial_utils.py:L166-L178 encode
    def encode(self, obj: Any) -> Sequence[bytestr]:
        # This `bufs` list allows us to collect direct pointers to backing
        # buffers of tensors and np arrays, and return them along with the
        # top-level encoded buffer instead of copying their data into the
        # new buffer.
        bufs = [b""]
        bufs[0] = self.encoder.encode(obj)
        return bufs

    # SOURCE: vllm/v1/serial_utils.py:L180-L189 encode_into (逐字) — 复用
    # buffer 的编码面 (m6: 输出 IO 线程的 bytearray 回收池)
    # SOURCE: vllm/v1/serial_utils.py:L180-L189 encode_into
    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        bufs = [buf]
        self.encoder.encode_into(obj, buf)
        return bufs

    # SOURCE: vllm/v1/serial_utils.py:L191-L235 enc_hook — SUBTRACTED: tensor/
    # ndarray 分支 (项 4) 与 insecure pickle 分支; UtilityResult 安全分支逐字
    # SOURCE: vllm/v1/serial_utils.py:L191-L235 enc_hook
    def enc_hook(self, obj: Any) -> Any:
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

        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable"
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based serialization."
            )
        raise AssertionError("unreachable: insecure serialization subtracted")


# SOURCE: vllm/v1/serial_utils.py:L313-L348 MsgpackDecoder — SUBTRACTED:
# tensor/ndarray/OOB 解码 (项 4 → ch05); typed decode + dec_hook + ext 逐字
class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization."""

    # SOURCE: vllm/v1/serial_utils.py:L323-L338 MsgpackDecoder.__init__ —
    # SUBTRACTED: share_mem/pin_tensors/oob_tensor_provider (项 4)
    # SOURCE: vllm/v1/serial_utils.py:L323-L338 MsgpackDecoder.__init__
    def __init__(self, t: Any | None = None):
        args = () if t is None else (t,)
        self.decoder = msgpack_ext.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )

    # SOURCE: vllm/v1/serial_utils.py:L340-L348 decode (逐字; aux_buffers 单帧化)
    # SOURCE: vllm/v1/serial_utils.py:L340-L348 decode
    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):  # type: ignore
            return self.decoder.decode(bufs)
        return self.decoder.decode(bufs[0])

    # SOURCE: vllm/v1/serial_utils.py:L350-L365 dec_hook — SUBTRACTED: mm 分支
    # (项 4) 与 ndarray/tensor 分支; slice/UtilityResult 逐字
    # SOURCE: vllm/v1/serial_utils.py:L350-L365 dec_hook
    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if t is slice:
                return slice(*obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
        return obj

    # SOURCE: vllm/v1/serial_utils.py:L367-L379 _decode_utility_result (逐字)
    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        result_type, result = obj
        if result_type is not None:
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise TypeError(
                    "VLLM_ALLOW_INSECURE_SERIALIZATION must "
                    "be set to use custom utility result types"
                )
        return UtilityResult(result)

    # SOURCE: vllm/v1/serial_utils.py:L426-L444 ext_hook — SUBTRACTED:
    # CUSTOM_TYPE_RAW_VIEW 张量重建 (项 4); 未知 ext 拒收逐字
    # SOURCE: vllm/v1/serial_utils.py:L426-L444 ext_hook
    @staticmethod
    # SOURCE: vllm/v1/serial_utils.py:L426-L444 ext_hook
    def ext_hook(code: int, data: memoryview) -> Any:
        raise NotImplementedError(f"Unknown ext code {code} (tensor wire is ch05's)")


# ============================================================================
# vllm/v1/engine/__init__.py — wire payload types
# ============================================================================

# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
# SOURCE: vllm/v1/engine/__init__.py:L23-L27 PauseMode (逐字)
PauseMode = "abort"  # HOST SEAM of the Literal type alias


# SOURCE: vllm/v1/engine/__init__.py:L29-L31 FINISH_REASON_STRINGS (逐字)
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

    # SOURCE: vllm/v1/engine/__init__.py:L64-L65 __str__ (逐字)
    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


# SUBTRACTED: vllm/v1/engine/__init__.py:L33-L40 EEP notification (项 2 → ch39)


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
    kv_events_config: Any | None = None  # KVEventsConfig seam (ch37 域)


# SOURCE: vllm/v1/engine/__init__.py:L97-L154 EngineCoreRequest — 字段全保留
# (线格式 schema 契约; mm/lora/embeds 等他章域字段类型放宽为 Any)
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: Any | None
    sampling_params: "SamplingParams | None"
    pooling_params: "PoolingParams | None"
    arrival_time: float
    lora_request: Any | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: Any | None = None

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


# SUBTRACTED: vllm/v1/engine/__init__.py:L157-L181 EngineCoreEventType/Event
# (事件簿记, 观测域 → delete 项 9)


# SOURCE: vllm/v1/engine/__init__.py:L184-L215 EngineCoreOutput — 字段全保留
# (线格式 schema 契约; logprobs/pooling/trace 等他章域字段类型放宽为 Any)
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: Any | None = None
    new_prompt_logprobs_tensors: Any | None = None

    pooling_output: Any | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: Any | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None

    prefill_stats: Any | None = None

    routed_experts: Any | None = None
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
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    result: UtilityResult | None = None


# SOURCE: vllm/v1/engine/__init__.py:L230-L258 EngineCoreOutputs — 字段全保留
# (DP wave 两字段随消费分支删但线位保留)
class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: Any | None = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None

    # SOURCE: vllm/v1/engine/__init__.py:L256-L258 __post_init__ (逐字)
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


# SOURCE: vllm/v1/engine/__init__.py:L261-L274 EngineCoreRequestType (逐字,
# minus START_DP_WAVE — delete 项 1 → ch34)
# SOURCE: vllm/v1/engine/__init__.py:L261-L274 EngineCoreRequestType
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
# / ReconfigureRankType / EngineStatusType (elastic EP + FT 控制面, delete 项 2/3)


# ============================================================================
# Config / request seams — the full assembly line is the ch03 product; only
# the fields the kept engine code reads are mirrored here.
# ============================================================================


# SOURCE: vllm/sampling_params.py SamplingParams — 字段 seam (跨线 dataclass 载荷;
# stop 面字段供 check_stop; structured_output_request 为 ch30 域注入位)
@dataclass
# SOURCE: vllm/sampling_params.py SamplingParams
class SamplingParams:  # HOST SEAM
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    min_tokens: int = 0
    eos_token_id: int | None = None
    stop_token_ids: list[int] | None = None
    structured_output_request: Any | None = None  # ENGINE SEAM (ch30 注入位)


# SOURCE: vllm/pooling_params.py PoolingParams — seam 占位 (ch6 域)
@dataclass
# SOURCE: vllm/pooling_params.py PoolingParams
class PoolingParams:  # HOST SEAM
    task: str = "embed"


# SOURCE: vllm/v1/metrics/stats.py SchedulerStats — metrics 轴 seam (消费端已删,
# 字段保留供 EngineCoreOutputs.scheduler_stats 注解)
@dataclass
# SOURCE: vllm/v1/metrics/stats.py SchedulerStats
class SchedulerStats:  # HOST SEAM
    num_running_reqs: int = 0


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
    local_engines_only: bool = False
    world_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    decode_context_parallel_size: int = 1


# SOURCE: vllm/config.py CacheConfig — 字段 seam (ready 回传面)
@dataclass
# SOURCE: vllm/config.py CacheConfig
class CacheConfig:  # HOST SEAM
    num_gpu_blocks: int | None = None
    block_size: int = 16
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None


# SOURCE: vllm/config.py SchedulerConfig — 字段 seam (async_scheduling 是 m11
# 的锚; get_scheduler_cls 返回本章 seam Scheduler)
@dataclass
# SOURCE: vllm/config.py SchedulerConfig
class SchedulerConfig:  # HOST SEAM
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True
    async_scheduling: bool = False  # v0.27.1 服务默认 True — 本章教学走同步 step()

    # SOURCE: vllm/config.py SchedulerConfig.get_scheduler_cls — ENGINE SEAM
    def get_scheduler_cls(self):
        return Scheduler


# SOURCE: vllm/config.py ModelConfig — 字段 seam
@dataclass
# SOURCE: vllm/config.py ModelConfig
class ModelConfig:  # HOST SEAM
    max_model_len: int = 4096
    dtype: str = "torch.float32"
    runner_type: str = "generate"
    is_moe: bool = False


# SOURCE: vllm/config.py VllmConfig — 字段 seam (装配线是 ch03 章产物)
@dataclass
class VllmConfig:  # HOST SEAM
    model_config: ModelConfig
    cache_config: CacheConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    instance_id: str
    shutdown_timeout: int = 0
    max_concurrent_batches: int = 1  # batch_queue_size 的落点 (m11)
    kv_transfer_config: Any | None = None

    # SOURCE: vllm/v1/engine/core.py:L1192 vllm_config.__post_init__() (握手后回调)
    def __post_init__(self):
        pass


# ============================================================================
# vllm/v1/request.py — the request bookkeeping the seam scheduler consumes
# ============================================================================


# SOURCE: vllm/v1/request.py:L348-L375 RequestStatus (逐字, minus streaming/
# remote-KVS 两状态的消费分支 — 状态本身保留: is_finished 的序数判据需要)
# SOURCE: vllm/v1/request.py:L348-L366 RequestStatus
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    # SOURCE: vllm/v1/request.py:L367-L368 __str__ (逐字)
    def __str__(self) -> str:
        return self.name

    # SOURCE: vllm/v1/request.py:L370-L372 is_finished (逐字)
    @staticmethod
    # SOURCE: vllm/v1/request.py:L370-L372 is_finished (逐字)
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    # SOURCE: vllm/v1/request.py:L374-L375 get_finished_reason (逐字)
    @staticmethod
    # SOURCE: vllm/v1/request.py:L374-L375 get_finished_reason (逐字)
    def get_finished_reason(status: "RequestStatus") -> "FinishReason | None":
        return _FINISHED_REASON_MAP.get(status)


# SOURCE: vllm/v1/request.py:L378-L390 _FINISHED_REASON_MAP (子集逐字; 映射体
# L383-L388 — streaming 域两行随消费分支删)
# SOURCE: vllm/v1/request.py:L378-L390 _FINISHED_REASON_MAP
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}


# SOURCE: vllm/v1/request.py StructuredOutputRequest — ENGINE SEAM (ch30 边界):
# 真实读 guided 解码配置并起异步编译; 本章只保留『有无结构化输出』判据
class StructuredOutputRequest:  # ENGINE SEAM
    # SOURCE: vllm/v1/request.py StructuredOutputRequest.from_sampling_params —
    # ENGINE SEAM: 测试经 SamplingParams.structured_output_request 注入可过线
    # 的真值标记 (真实是 guided 解码配置字段)
    @classmethod
    # SOURCE: vllm/v1/request.py StructuredOutputRequest.from_sampling_params —
    def from_sampling_params(cls, sampling_params) -> "StructuredOutputRequest | None":
        if getattr(sampling_params, "structured_output_request", None):
            return cls()
        return None


# SOURCE: vllm/v1/request.py:L60-L247 Request — 字段 seam: 本章调度循环触达的
# 记账字段全真 (num_computed_tokens/is_prefill_chunk/append/判停), 其余域字段收窄
class Request:  # HOST SEAM
    # SOURCE: vllm/v1/request.py:L60-L120 Request.__init__ — 字段 seam
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        client_index: int = 0,
        prompt_embeds: Any | None = None,
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
        self.mm_features = mm_features or []
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

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        # SOURCE: vllm/v1/request.py:L140-L143 token 账 (逐字语义)
        self.num_prompt_tokens = len(prompt_token_ids or ())
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(prompt_token_ids or [])

        # SOURCE: vllm/v1/request.py:L152-L173 记账字段 (逐字语义; 量测位收窄)
        self.num_output_placeholders = 0
        self.num_in_flight_tokens = 0

        self.num_computed_tokens = 0
        self.stop_reason: int | str | None = None
        self.status = RequestStatus.WAITING

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

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
            structured_output_request=(
                StructuredOutputRequest.from_sampling_params(request.sampling_params)
                if request.sampling_params is not None
                else None
            ),
        )

    # SOURCE: vllm/v1/request.py:L249-L259 append_output_token_ids (子集逐字;
    # block hash 更新归 ch13 前缀缓存域)
    # SOURCE: vllm/v1/request.py:L249-L259 append_output_token_ids (子集逐字;
    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

    # SOURCE: vllm/v1/request.py:L267-L269 use_structured_output property (逐字)
    @property
    # SOURCE: vllm/v1/request.py:L267-L269 use_structured_output property (逐字)
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    # SOURCE: vllm/v1/request.py:L271-L273 num_tokens property (逐字)
    @property
    # SOURCE: vllm/v1/request.py:L271-L273 num_tokens property (逐字)
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    # SOURCE: vllm/v1/request.py:L279-L281 num_output_tokens property (逐字)
    @property
    # SOURCE: vllm/v1/request.py:L279-L281 num_output_tokens property (逐字)
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    # SOURCE: vllm/v1/request.py:L182 output_token_ids 只读视图 (逐字语义;
    # ConstantList 以 list 代行)
    @property
    # SOURCE: vllm/v1/request.py:L182 output_token_ids 只读视图 (逐字语义;
    def output_token_ids(self) -> list[int]:
        return self._output_token_ids

    # SOURCE: vllm/v1/request.py:L303-L304 is_finished (逐字)
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    # SOURCE: vllm/v1/request.py:L306-L307 get_finished_reason (逐字)
    def get_finished_reason(self) -> "FinishReason | None":
        return RequestStatus.get_finished_reason(self.status)


# SOURCE: vllm/v1/core/sched/utils.py:L94-L136 check_stop — SUBTRACTED:
# repetition_detection 分支 (检测器域); stop/eos/max_tokens/min_tokens 主线逐字
# SOURCE: vllm/v1/core/sched/utils.py:L94-L136 check_stop
def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # SUBTRACTED: vllm/v1/core/sched/utils.py:L123-L135 repetition 检测分支

    return False


# ============================================================================
# vllm/v1/core/sched/output.py — what ① produces and ⑤ consumes
# ============================================================================


# SOURCE: vllm/v1/core/sched/output.py:L33-L75 NewRequestData (子集逐字)
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list
    sampling_params: Any | None
    pooling_params: Any | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: Any | None
    prompt_embeds: Any | None = None
    prompt_is_token_ids: list[bool] | None = None

    # Version of __repr__ with the prompt data obfuscated
    # SOURCE: vllm/v1/core/sched/output.py:L70-L75 anon_repr (逐字)
    def anon_repr(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={len(self.prompt_token_ids) if self.prompt_token_ids is not None else None},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params is not None},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request is not None},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            f")"
        )

    # SOURCE: vllm/v1/core/sched/output.py:L48-L67 from_request (逐字)
    @classmethod
    # SOURCE: vllm/v1/core/sched/output.py:L48-L67 from_request (逐字)
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
        )


# SOURCE: vllm/v1/core/sched/output.py:L139-L163 CachedRequestData (逐字;
# all_token_ids 注释为 MRV1 域)
@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L139-L163 CachedRequestData (逐字;
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used
    # as the request's block IDs instead of appending to the existing ones.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # MRV1-only: For requests not scheduled in the last step, propagate the token ids
    # to the connector. Won't contain requests scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]


# SOURCE: vllm/v1/core/sched/output.py:L192-L286 SchedulerOutput — SUBTRACTED:
# 增量协议的 ch18 域尾段 (preempted_req_ids/pending_structured_output_tokens/
# num_invalid_spec_tokens/kv_connector/ec_manager/block zero/CoW/partial tail/
# num_spec_tokens_to_schedule — delete 项 3/7/11 邻域); 五拍消费的字段与注释逐字
@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L192-L286 SchedulerOutput — SUBTRACTED:
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that request's 0-th and 1-st images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False


# SOURCE: vllm/v1/core/sched/output.py:L287-L291 GrammarOutput (逐字)
@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L287-L291 GrammarOutput (逐字)
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: Any  # npt.NDArray[np.int32]


# ============================================================================
# vllm/v1/core/sched/interface.py — the ①拍 contract (m13)
# ============================================================================


# SOURCE: vllm/v1/core/sched/interface.py:L22-L35 PauseState (逐字)
class PauseState(enum.IntEnum):
    """State of the scheduler.

    - UNPAUSED: Normal scheduling
    - PAUSED_NEW: No new requests are scheduled (existing requests continue)
    - PAUSE_ALL: No requests are scheduled
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


# SOURCE: vllm/v1/core/sched/interface.py:L38 SchedulerInterface — SUBTRACTED:
# 抽象 __init__/draft 更新/pause 面/reset 面/get_request_counts (delete 项 5/7);
# 本章消费的契约面逐字
class SchedulerInterface(ABC):
    # SOURCE: vllm/v1/core/sched/interface.py:L53-L83 schedule (docstring 逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L53-L83 schedule (docstring 逐字)
    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.

        Args:
            throttle_prefills: DP prefill balancing. When True (set by the DP
                engine core on non-cadence-aligned steps), new prefill compute is
                deferred to a later step so prefills stay aligned across DP ranks;
                automatically overridden when the rank is saturated.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            requests.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L85-L89 get_grammar_bitmask (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L85-L89 get_grammar_bitmask (逐字)
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L91-L109 update_from_output (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L91-L109 update_from_output (逐字)
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L135-L142 add_request (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L135-L142 add_request (逐字)
    def add_request(self, request: "Request") -> None:
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L144-L166 finish_requests (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L144-L166 finish_requests (逐字)
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "RequestStatus",
    ) -> "list[Request]":
        """Finish the requests in the scheduler's internal queue. If the request
        is not in the queue, this method will do nothing for that request.

        This method is called in two cases:
        1. When the request is aborted by the client.
        2. When the frontend process detects a stop string of the request after
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs, or None to finish all.
            finished_status: The finished status of the given requests.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L168-L171 get_num_unfinished_requests (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L168-L171 get_num_unfinished_requests (逐字)
    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in the scheduler's internal queue."""
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L173-L176 has_unfinished_requests (逐字)
    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        return self.get_num_unfinished_requests() > 0

    # SOURCE: vllm/v1/core/sched/interface.py:L178-L191 has_finished_requests (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L178-L191 has_finished_requests (逐字)
    def has_finished_requests(self) -> bool:
        """Returns True if there are finished requests that need to be cleared.
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in
        the previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L193-L196 has_requests (逐字)
    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests() or self.has_finished_requests()

    # SOURCE: vllm/v1/core/sched/interface.py:L242-L248 make_stats (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L242-L248 make_stats (逐字)
    def make_stats(self) -> "SchedulerStats | None":
        """Make a SchedulerStats object for logging.

        The SchedulerStats object is created for every scheduling step.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L250-L253 shutdown (逐字)
    @abstractmethod
    # SOURCE: vllm/v1/core/sched/interface.py:L250-L253 shutdown (逐字)
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        raise NotImplementedError


# ============================================================================
# vllm/v1/structured_output/__init__.py — the ch30 boundary (F6 埋点)
# ============================================================================


# SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager —
# ENGINE SEAM (ch30 边界): grammar_init 在输入线程启动异步编译 (不占忙循环);
# grammar_bitmask 是 ③拍 的计算位 (语法→FSM→位掩码归 ch30)。本章测试经
# UTILITY 薄 RPC 注入位掩码行 (structured_output_request_ids 顺序)。
class StructuredOutputManager:  # ENGINE SEAM
    # SOURCE: vllm/v1/structured_output/__init__.py:L114 grammar_init — ENGINE SEAM
    def __init__(self, vllm_config: Any | None = None):
        self.trace: list[str] = []  # ENGINE SEAM observation
        self._scripted_bitmasks: deque = deque()  # ENGINE SEAM script queue

    # SOURCE: vllm/v1/structured_output/__init__.py:L114 grammar_init — ENGINE SEAM
    # (异步编译启动; 真实在输入线程调用)
    # SOURCE: vllm/v1/structured_output/__init__.py:L114 grammar_init — ENGINE SEAM
    def grammar_init(self, request: Request) -> None:
        self.trace.append(f"grammar_init:{request.request_id}")
        return None

    # SOURCE: vllm/v1/structured_output/__init__.py grammar_bitmask — ENGINE SEAM
    # (③拍 计算位; 真实编译 FSM 并产位掩码 — ch30)
    # SOURCE: vllm/v1/structured_output/__init__.py grammar_bitmask — ENGINE SEAM
    def grammar_bitmask(
        self,
        requests: dict[str, Request],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Any:
        self.trace.append("grammar_bitmask")
        if not self._scripted_bitmasks:
            raise RuntimeError(
                "grammar_bitmask script queue is empty (the real FSM compiler "
                "is the ch30 boundary; tests must script the bitmask rows)"
            )
        rows = self._scripted_bitmasks.popleft()
        if len(rows) != len(structured_output_request_ids):
            raise RuntimeError(
                f"scripted bitmask has {len(rows)} rows, "
                f"{len(structured_output_request_ids)} expected"
            )
        return np.asarray(rows, dtype=np.int32)

    # SOURCE: vllm/v1/structured_output/__init__.py:L488 clear_backend — ENGINE SEAM
    def clear_backend(self) -> None:
        return None

    # ENGINE SEAM test hook: 位掩码注入位 (经真实 UTILITY 反射 RPC 过线)
    # SOURCE: vllm/v1/engine/core.py:L597 get_grammar_bitmask 调用位 — ENGINE SEAM 注入位
    def enqueue_bitmask(self, rows: list) -> None:
        self._scripted_bitmasks.append(list(rows))


# ============================================================================
# vllm/v1/core/sched/scheduler.py — the seam Scheduler (ch10/ch11 boundary)
# ============================================================================


# SOURCE: vllm/v1/core/sched/scheduler.py Scheduler — ENGINE SEAM (ch10/ch11
# 边界): schedule() 保留真头 (woosuk 注释逐字) 与真尾 (_update_after_schedule
# 记账逐字), 循环体换成最小 token 账; update_from_output 保留热循环骨架
# (跳过已完成的真实判据、判停、分桶、finished 簿记逐字), 深水 (抢占/spec/
# KV connector/事件/统计) SUBTRACTED。两段式契约在这只桩上同样可观察
# (dossier delete 项 11 批准的『同签名最小桩』)。
class Scheduler(SchedulerInterface):  # ENGINE SEAM
    # SOURCE: vllm/v1/core/sched/scheduler.py Scheduler.__init__ (ch10) — ENGINE SEAM
    def __init__(
        self,
        vllm_config: Any | None = None,
        kv_cache_config: Any | None = None,
        structured_output_manager: StructuredOutputManager | None = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
        block_size: int = 16,
        hash_block_size: int = 16,
    ):
        self.vllm_config = vllm_config
        self.structured_output_manager = structured_output_manager
        # SOURCE: vllm/v1/core/sched/scheduler.py:L172 current_step (逐字语义)
        self.current_step = 0
        # SOURCE: vllm/v1/core/sched/scheduler.py:L196-L202 簿记 (逐字语义)
        self.finished_req_ids: set[str] = set()
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            {} if include_finished_set else None
        )
        self.requests: dict[str, Request] = {}
        # ENGINE SEAM: RequestQueue 以朴素 list 代行 (优先级队列归 ch10)
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self._pause_state = PauseState.UNPAUSED
        cfg = vllm_config
        self.max_num_scheduled_tokens = (
            cfg.scheduler_config.max_num_batched_tokens if cfg else 8192
        )
        self.max_model_len = cfg.model_config.max_model_len if cfg else 4096
        # ENGINE SEAM observation (调用序/最后一次调度决定/abort 批账)
        self.trace: list[tuple[int, str]] = []
        self.last_output: SchedulerOutput | None = None
        self.finish_calls: list[list[str]] = []

    # SOURCE: vllm/v1/core/sched/scheduler.py:L439-L450 schedule 头 (逐字)
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L452-L1323 RUNNING/WAITING
        # 循环体与抢占 (delete 项 11 → ch10/ch11 精简版持有)
        # ENGINE SEAM (ch10 boundary): the minimal token account — running
        # requests catch up their remaining prompt or decode 1 token; waiting
        # requests prefill subject to the token budget; finished ids ride the
        # output for the worker.
        self.trace.append((time.perf_counter_ns(), "schedule"))
        scheduled_new_reqs: list[NewRequestData] = []
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # First, schedule the RUNNING requests.
        still_running: list[Request] = []
        for request in self.running:
            num_to_catch_up = request.num_tokens - request.num_computed_tokens
            num = min(max(num_to_catch_up, 1), token_budget)
            if num <= 0:
                still_running.append(request)
                continue
            num_scheduled_tokens[request.request_id] = num
            token_budget -= num
            still_running.append(request)
        self.running = still_running

        # Then, schedule the WAITING requests.
        while self.waiting and token_budget > 0:
            request = self.waiting[0]
            num_to_catch_up = request.num_tokens - request.num_computed_tokens
            num = min(max(num_to_catch_up, 1), token_budget)
            num_scheduled_tokens[request.request_id] = num
            token_budget -= num
            self.waiting.pop(0)
            self.running.append(request)
            # SOURCE: vllm/v1/core/sched/scheduler.py 入批置 RUNNING (ch10 语义)
            request.status = RequestStatus.RUNNING
            scheduled_new_reqs.append(NewRequestData.from_request(request, ()))

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[],
                num_computed_tokens=[],
                num_output_tokens=[],
            ),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=[],
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1325-L1341 _update_after_
        # schedule 记账 (逐字 minus defer_block_free/routed/inflight_prefills)
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()

        self.last_output = scheduler_output  # ENGINE SEAM observation
        return scheduler_output

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1646-L1668 get_grammar_bitmask (逐字)
    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        self.trace.append((time.perf_counter_ns(), "get_grammar_bitmask"))
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1670-L2033 update_from_output —
    # SUBTRACTED: 热循环外的深水 (defer_free/perf/kv invalid/routed/stats/
    # grammar-compile-error/kv connector/events — delete 项 3/7/9/30) 与热循环内
    # 的 spec/抢占分支 (项 7/11); 骨架/判停/分桶/finished 簿记逐字
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1670-L2033 update_from_output
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, EngineCoreOutputs]:
        self.trace.append((time.perf_counter_ns(), "update_from_output"))
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1676-L1726 logprobs/pooler/
        # nans/kv_connector/cudagraph 读取与 defer_free/perf/routed/kv 预处理

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1738-L1743 in-flight/
            # stale drain (项 11 → ch11); L1744-L1746 failed_kv_load (项 3)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                continue

            # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1757-L1759 drop-mode
            # stale output (项 11); L1766-L1791 spec 接受算术 (项 7); L1793-L1795
            # encoder 释放 (项 4 邻域)

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            stopped = False
            new_token_ids = generated_token_ids
            status_before_stop = request.status
            # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1804-L1805 记账位

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1812-L1815 pooling 停
            # (项 8); L1817-L1843 grammar advance (ch30); L1845-L1893 routed
            # (项 9)

            should_emit_output = bool(new_token_ids or stopped)

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    # SUBTRACTED: kv/ec 参数回传解包 (项 3)
                    self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1906-L1907
                # stopped_preempted_reqs (项 11 → ch11)

            # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1909-L1918 logprobs/
            # nans 采样簿记 (ch8/项 9); L1920-L1921 prompt logprobs (ch8)
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        stop_reason=request.stop_reason,
                    )
                )
                # SUBTRACTED: 其余实参 (events/prefill_stats/kv/ec/trace/
                # routed/nans — 项 3/9)

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            # SOURCE: vllm/v1/core/sched/scheduler.py remove_all (ch10 域工具,
            # 朴素过滤代行 — ENGINE SEAM)
            self.running = [r for r in self.running if r not in stopped_running_reqs]

        # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L1946-L2010 停止抢占请求
        # 移除/grammar 编译错误/KV connector 更新/KV events 发布 (项 3/11/30)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2019-L2033 finished 簿记冲刷
        # (逐字)
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        return engine_core_outputs

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213-L2235 add_request — SUBTRACTED:
    # 流式输入会话分支 (项 5); 新请求入队主线逐字
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213-L2235 add_request
    def add_request(self, request: Request) -> None:
        # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L2214-L2226 streaming 会话
        # (项 5); L2228-L2229 resumable 队列 (项 5); L2232-L2235 connector 钩子
        # (项 3) 与事件 (项 9)
        self._enqueue_waiting_request(request)
        self.requests[request.request_id] = request

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2237-L2298 finish_requests —
    # SUBTRACTED: 流式/WAITING_FOR_REMOTE_KVS/delay_free (项 3/5); 主线逐字
    # (幂等判据 L2265-L2268: 已完成/未知 id 直接跳过)
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2237-L2298 finish_requests
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()
        self.finish_calls.append(list(request_ids))  # ENGINE SEAM observation

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = [
                r for r in self.running if r not in running_requests_to_remove
            ]
        if waiting_requests_to_remove:
            for r in waiting_requests_to_remove:
                if r in self.waiting:
                    self.waiting.remove(r)

        # Second pass: set status and free requests
        for request in valid_requests:
            # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L2287-L2293 remote-KVS
            # delay_free (项 3)
            request.status = finished_status
            self._free_request(request)

        return valid_requests

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2300-L2327 _free_request —
    # SUBTRACTED: encoder cache/connector 钩子/参数回传 (项 3/4); finished
    # 簿记与释放逐字
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2300-L2327 _free_request
    def _free_request(self, request: Request):
        assert request.is_finished()

        # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L2305-L2317 inflight/
        # connector/ec 钩子与 encoder cache (项 3/4)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)
        # SUBTRACTED: delay_free_blocks 栅栏 (项 3) — 无 connector 即时释放
        self._free_blocks(request)
        return None, None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2329-L2332 _free_blocks (逐字语义;
    # KV 块释放归 ch13)
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2329-L2332 _free_blocks (逐字语义;
    def _free_blocks(self, request: Request):
        assert request.is_finished()
        del self.requests[request.request_id]

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2058-L2062 _enqueue_waiting_request —
    # SUBTRACTED: skipped_waiting 分流 (ch10)
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2058-L2062 _enqueue_waiting_request —
    def _enqueue_waiting_request(self, request: Request) -> None:
        self.waiting.append(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2076-L2092 _handle_stopped_request —
    # SUBTRACTED: resumable/streaming 续会话 (项 5)
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2076-L2092 _handle_stopped_request
    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True
        return True  # SUBTRACTED: 流式续会话 (项 5) — resumable 恒即完

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2094-L2111 _update_request_with_
    # output (逐字)
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2094-L2111 _update_request_with_
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2334-L2339 pause_state (逐字)
    @property
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2334-L2339 pause_state (逐字)
    def pause_state(self) -> PauseState:
        return self._pause_state

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2338-L2339 set_pause_state (逐字)
    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2383-L2394 get_num_unfinished_
    # requests — SUBTRACTED: pause 两分支 (项 5)
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2383-L2394 get_num_unfinished_
    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2392-L2404 has_finished_requests —
    # SUBTRACTED: connector pending 分支 (项 3); 主判据逐字
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2392-L2404 has_finished_requests —
    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        return False

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2406-L2421 has_requests —
    # SUBTRACTED: connector/ec pending 两读数 (项 3); 主判据逐字
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2406-L2421 has_requests —
    def has_requests(self) -> bool:
        self.trace.append((time.perf_counter_ns(), "has_requests"))
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py make_stats — ENGINE SEAM (项 9)
    def make_stats(self) -> "SchedulerStats | None":
        return None

    # SOURCE: vllm/v1/core/sched/scheduler.py shutdown (ch10) — ENGINE SEAM
    def shutdown(self) -> None:
        return None


# ============================================================================
# vllm/v1/outputs.py — worker→scheduler result carriers
# ============================================================================


# SOURCE: vllm/v1/outputs.py:L212-L219 SamplerOutput (逐字; logprobs 面收窄)
@dataclass
# SOURCE: vllm/v1/outputs.py:L212-L219 SamplerOutput (逐字; logprobs 面收窄)
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Any | None


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
# SOURCE: vllm/v1/outputs.py:L258-L308 ModelRunnerOutput — SUBTRACTED: logprobs/
# pooling/kv/ec/cudagraph/routed 字段 (ch8/项 3/8/9); 消费面字段逐字
@dataclass
# SOURCE: vllm/v1/outputs.py:L258-L308 ModelRunnerOutput — SUBTRACTED: logprobs/
class ModelRunnerOutput:
    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # SUBTRACTED: vllm/v1/outputs.py:L276-L308 logprobs/prompt_logprobs/pooler/
    # kv/ec/nans/cudagraph/routed 字段 (他章域)

    # SOURCE: vllm/v1/outputs.py:L341-L374 EMPTY_MODEL_RUNNER_OUTPUT (逐字语义)
# SOURCE: vllm/v1/outputs.py:L375 EMPTY_MODEL_RUNNER_OUTPUT
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})


# SOURCE: vllm/v1/outputs.py AsyncModelRunnerOutput — ABC (get_output 契约)
class AsyncModelRunnerOutput(ABC):
    # SOURCE: vllm/v1/outputs.py AsyncModelRunnerOutput.get_output (契约逐字)
    @abstractmethod
    # SOURCE: vllm/v1/outputs.py AsyncModelRunnerOutput.get_output (契约逐字)
    def get_output(self) -> ModelRunnerOutput:
        raise NotImplementedError


# ============================================================================
# vllm/v1/worker/gpu_input_batch.py — the persistent batch (ch17 boundary)
# ============================================================================


# SOURCE: vllm/v1/worker/gpu_input_batch.py InputBatch — ENGINE SEAM (ch17 边界):
# 真实持久批是 GPU 上的行态批 (block table/attn metadata); 本章保它的
# 消费面 — req_ids 顺序 (apply_grammar_bitmask 的重排基准) 与新请求落批/
# 完成清退, 供位掩码窗口与采样行定位。
class InputBatch:  # ENGINE SEAM
    # SOURCE: vllm/v1/worker/gpu_input_batch.py InputBatch.__init__ (ch17) — SEAM
    def __init__(self):
        self.req_ids: list[str] = []
        self.num_prompt_tokens: dict[str, int] = {}
        self.num_computed_tokens: dict[str, int] = {}

    # SOURCE: vllm/v1/worker/gpu_model_runner.py _update_states (ch17) — SEAM:
    # 新请求按调度序落批, 完成请求清退
    # SOURCE: vllm/v1/worker/gpu_model_runner.py _update_states (ch17) — SEAM:
    def update(self, scheduler_output: SchedulerOutput) -> None:
        for rid in scheduler_output.finished_req_ids:
            if rid in self.req_ids:
                self.req_ids.remove(rid)
            self.num_prompt_tokens.pop(rid, None)
            self.num_computed_tokens.pop(rid, None)
        for new_req in scheduler_output.scheduled_new_reqs:
            self.req_ids.append(new_req.req_id)
            self.num_prompt_tokens[new_req.req_id] = len(new_req.prompt_token_ids or ())
            # ENGINE SEAM (ch17): NewRequestData 快照在调度记账前拍 — 补上本拍
            # 调度的 token 才是执行后的 computed (真实由 _update_states 落位)
            self.num_computed_tokens[new_req.req_id] = (
                new_req.num_computed_tokens
                + scheduler_output.num_scheduled_tokens.get(new_req.req_id, 0)
            )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py logits_indices (ch17) — SEAM:
    # 非末块 prefill 无采样行 (真实由 logits_indices 决定)
    # SOURCE: vllm/v1/worker/gpu_model_runner.py logits_indices (ch17) — SEAM:
    def num_sampling_rows(self, rid: str) -> int:
        computed = self.num_computed_tokens[rid]
        prompt = self.num_prompt_tokens[rid]
        return 0 if computed < prompt else 1


# ============================================================================
# vllm/v1/structured_output/utils.py — apply_grammar_bitmask (worker half of F6)
# ============================================================================


# HOST SEAM of `import xgrammar as xgr`: the CPU kernel semantics are the
# documented xgrammar contract (disallowed token → -inf); inside the pin
# container the real kernel takes over automatically.
class _XgrammarSeam:  # HOST SEAM
    # SOURCE: xgrammar.apply_token_bitmask_inplace — HOST SEAM (CPU 内核语义:
    # 位清零的 token 置 -inf; indices=None 时全行)
    @staticmethod
    # SOURCE: xgrammar.apply_token_bitmask_inplace — HOST SEAM (CPU 内核语义:
    def apply_token_bitmask_inplace(logits: torch.Tensor, bitmask, indices=None):
        rows = range(logits.shape[0]) if indices is None else indices
        for i in rows:
            row = logits[i]
            words = bitmask[i]
            for w, word in enumerate(words):
                word = int(word)
                base = w * 32
                for b in range(32):
                    t = base + b
                    if t >= row.shape[0]:
                        break
                    if not (word >> b) & 1:
                        row[t] = float("-inf")


try:  # HOST SEAM: 真 xgrammar 优先 (容器内)
    import xgrammar as xgr  # type: ignore
except ImportError:
    xgr = _XgrammarSeam()  # type: ignore[assignment]


# SOURCE: vllm/v1/structured_output/utils.py:L86-L175 apply_grammar_bitmask —
# SUBTRACTED: GPU 路径的 async H2D/非阻塞拷贝 (L151-L161 — CPU host 无 copy
# stream); 排序/重排主线逐字
# SOURCE: vllm/v1/structured_output/utils.py:L86-L175 apply_grammar_bitmask
def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """
    Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.
    """
    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    grammar_bitmask = grammar_output.grammar_bitmask

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices = []

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask_tensor = torch.full(
        (logits.shape[0], grammar_bitmask.shape[1]),
        -1,
        dtype=torch.from_numpy(grammar_bitmask[:0]).dtype,
        pin_memory=PIN_MEMORY,
    )
    sorted_bitmask = sorted_bitmask_tensor.numpy()
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    # Copy async to device. (HOST SEAM: CPU 上即原地)
    grammar_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    # SUBTRACTED: vllm/v1/structured_output/utils.py:L151-L161 GPU 分支
    # (async H2D index tensor + xgr GPU kernel)

    # CPU case, use list for indices.
    indices = None if skip_out_indices else out_indices
    # Handle dtype conversion for CPU (older xgrammar CPU kernels require float32)
    # See: https://github.com/vllm-project/vllm/issues/31901
    if logits.dtype != torch.float32:
        # Convert to float32, apply bitmask, then convert back
        logits_fp32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(logits_fp32, grammar_bitmask, indices=indices)
        # Copy the modified values back to the original tensor
        logits.copy_(logits_fp32.to(logits.dtype))
    else:
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=indices)


# ============================================================================
# vllm/v1/worker/gpu_model_runner.py — the two-phase contract (m3/m4)
# ============================================================================


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L437-L450 ExecuteModelState (逐字)
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: Any | None
    spec_decode_common_attn_metadata: Any | None
    hidden_states: Any | None
    sample_hidden_states: Any | None
    aux_hidden_states: Any | None
    ec_connector_output: Any | None
    cudagraph_stats: Any | None
    slot_mappings: Any | None


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L259-L334 AsyncGPUModelRunnerOutput —
# SUBTRACTED: CUDA copy stream/routed/ep-fault/logprobs_cpu 深水 (ch12/17);
# ENGINE SEAM (CPU host): threading.Event 站在 async_copy_ready_event 的位置
# (同一 get_output 语义: 等 D2H 完成才交出 ModelRunnerOutput)
# SOURCE: vllm/v1/worker/gpu_model_runner.py:L259-L334 AsyncGPUModelRunnerOutput
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L261-L296 __init__ — SEAM
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: Any | None,
        invalid_req_indices: list[int],
        vocab_size: int = 0,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.async_copy_ready_event = threading.Event()  # ENGINE SEAM (CPU)

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        # ENGINE SEAM (CPU): the pinned host buffer the D2H copy targets
        # (真实: copy stream 上的 non_blocking 拷贝; 完成由 event 标记)
        self.sampled_token_ids_cpu = sampled_token_ids.cpu()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L309-L334 get_output — SEAM
    # (逐字语义: 阻塞至拷贝完成、invalid 行清空、回填 sampled_token_ids)
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L309-L334 get_output — SEAM
    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.wait()

        # Release the device tensors once the copy has completed.
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output


# SOURCE: vllm/v1/sample/sampler.py:L20-L58 Sampler — SUBTRACTED: 整个采样栈
# (logprobs/temperature/topk-topp/penalties/bad-words —— ch08/sampler 域);
# greedy_sample 逐字保留 (temperature=0 分支, 本章测试全用贪心)
class Sampler:  # ENGINE SEAM (sampling stack boundary)
    # SOURCE: vllm/v1/sample/sampler.py:L239-L241 greedy_sample (逐字)
    @staticmethod
    # SOURCE: vllm/v1/sample/sampler.py:L239-L241 greedy_sample (逐字)
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L453 GPUModelRunner — SUBTRACTED:
# 前向深水区 (input prep/模型执行/compute_logits/cudagraph —— ch17 执行域);
# 两段式契约面 (State error/暂存/解包/掩码/采样调用位) 逐字; 前向本体是
# ENGINE SEAM (测试经真实 UTILITY 薄 RPC 脚本化 logits 行, 不在环内伪造 forward)
class GPUModelRunner:  # ENGINE SEAM (ch17 boundary)
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456 GPUModelRunner.__init__ —
    # SUBTRACTED: 模型/缓存/采样器装配 (ch17); 契约字段与持久批 seam
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456 GPUModelRunner.__init__
    def __init__(self, vllm_config: Any, device: Any | None = None):
        self.vllm_config = vllm_config
        self.use_async_scheduling = (
            vllm_config.scheduler_config.async_scheduling
            if vllm_config is not None
            else False
        )
        self.execute_model_state: ExecuteModelState | None = None
        # SOURCE: vllm/v1/worker/gpu_model_runner.py 持久 InputBatch (ch17) — SEAM
        self.input_batch = InputBatch()
        # ENGINE SEAM (ch17): scripted forward — each step is a dict of
        # {request_id: logits row}; the busy loop blocks here until a script
        # arrives (the real engine blocks on the GPU) and dies on the real
        # error path if none does (bounded wait).
        self._scripted_logits: deque = deque()
        self._script_cond = threading.Condition()
        self._pending_async: list[AsyncGPUModelRunnerOutput] = []
        self.trace: list[tuple[int, str]] = []  # ENGINE SEAM observation

    # ENGINE SEAM test hook (经真实 UTILITY 反射 RPC 过线)
    # SOURCE: vllm/v1/engine/core.py:L596 execute_model 调用位 — ENGINE SEAM 注入位
    def enqueue_logits(self, steps: list) -> None:
        with self._script_cond:
            for step in steps:
                self._scripted_logits.append(
                    {rid: list(row) for rid, row in dict(step).items()}
                )
            self._script_cond.notify_all()

    # ENGINE SEAM test hook: 完成 D2H (CPU 上即事件置位)
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L309-L334 D2H 拷贝完成位 — ENGINE SEAM
    def release_async_copies(self) -> None:
        for out in self._pending_async:
            out.async_copy_ready_event.set()
        self._pending_async.clear()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4165-L4233 execute_model 入口 —
    # State error 防御 (L4171-L4175) 逐字; 空批早退 (L4218-L4233) 逐字;
    # SUBTRACTED: routed capturer/ngram copy/kv preempt (L4177-L4200) 与
    # input prep/模型前向/compute_logits (L4202-L4515 → ch17)
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4165-L4233 execute_model
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Any | None = None,
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | Any | None":
        self.trace.append((time.perf_counter_ns(), "execute_model"))
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4177-L4195 routed
        # capturer 清理 + ngram_gpu scheduler_output 拷贝 (项 7/9)
        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4197-L4200 KV
        # connector preemption (项 3)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4207-L4208 持久批同步
        # (_update_states — ch17 GPU 面; ENGINE SEAM 镜像: 新请求落批/完成清退,
        # 空拍也要同步 — 真实顺序在 0-token 早退之前)
        self.input_batch.update(scheduler_output)

        if not num_scheduled_tokens:
            # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4218-L4230 external
            # launcher DP dummy run (项 1)
            # Return empty ModelRunnerOutput if no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        # ENGINE SEAM (ch17 boundary): the scripted forward stands in for
        # model(...)+compute_logits — the loop consumes exactly what the test
        # scripts, through the same UTILITY dispatch vLLM uses.
        logits = self._seam_logits(scheduler_output)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4516-L4527 暂存 (逐字;
        # hidden_states 等深水字段为 None — 前向本体已删)
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            None,  # spec_decode_metadata (项 7)
            None,  # spec_decode_common_attn_metadata (项 7)
            None,  # hidden_states (ch17)
            None,  # sample_hidden_states (ch17)
            None,  # aux_hidden_states (ch17)
            None,  # ec_connector_output (项 3)
            None,  # cudagraph_stats (项 9)
            None,  # slot_mappings (ch17)
        )

        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4530-L4533 deferred
        # state corrections (ch12)

        return None

    # ENGINE SEAM (ch17): scripted forward — request-keyed rows; waits like
    # the engine waits on the GPU; a dry script raises on the real error path
    # (busy loop → ENGINE_CORE_DEAD). The 5ms sleep models one forward pass
    # (真实一步前向 ~几毫秒——环的节拍与抢占时序都以此为底).
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 模型前向 (ch17) — ENGINE SEAM 站位
    def _seam_logits(self, scheduler_output: SchedulerOutput) -> torch.Tensor:
        step = self._pop_scripted_rows()
        rows = []
        for rid in self.input_batch.req_ids:
            if rid not in step:
                raise RuntimeError(
                    f"no scripted logits row for request {rid!r} "
                    "(ch17 boundary: script each request's row)"
                )
            rows.append(step[rid])
        time.sleep(0.005)  # ENGINE SEAM: one forward pass ≈ 5ms
        return torch.tensor(rows, dtype=torch.float32)

    # ENGINE SEAM (ch17): bounded wait for the next scripted step
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4515 前向的 GPU 等待位 — ENGINE SEAM
    def _pop_scripted_rows(self) -> list:
        with self._script_cond:
            deadline = time.monotonic() + 5.0
            while not self._scripted_logits:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "scripted forward ran dry (ch17 boundary: the real "
                        "engine waits on the GPU here)"
                    )
                self._script_cond.wait(remaining)
            return self._scripted_logits.popleft()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4552-L4589 sample_tokens —
    # 解包→清态→掩码→_sample 调用位逐字; SUBTRACTED: PP/kv-conn 特例分支
    # (L4556-L4564, 项 3)、record_function 探针 (项 9)
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4552-L4589 sample_tokens
    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | Any":
        self.trace.append((time.perf_counter_ns(), "sample_tokens"))
        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4556-L4564
        # execute_model_state 为 None 的 PP/kv-conn 特例 (项 3)

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            self.trace.append((time.perf_counter_ns(), "apply_bitmask"))
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4588-L4589 _sample 调用位
        # (逐字 minus record_function 探针)
        sampler_output = self._sample(logits, spec_decode_metadata)

        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4591-L4781 _update_
        # states_after_model_execute / PP 广播 / spec drafter / bookkeeping
        # (ch17/18 + 项 7)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4766-L4780 ModelRunnerOutput
        # 组装 (逐字语义; logprobs/kv/ec/nans/cudagraph/routed 实参已删)
        req_ids = self.input_batch.req_ids
        sampled = sampler_output.sampled_token_ids
        rows = sampled.tolist()
        # ENGINE SEAM (ch17): 非末块 prefill 无采样行 —— 真实由 logits_indices
        # 决定; seam 按持久批判据清行
        output = ModelRunnerOutput(
            req_ids=list(req_ids),
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[
                rows[i] if self.input_batch.num_sampling_rows(rid) else []
                for i, rid in enumerate(req_ids)
            ],
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4782-L4840 异步包裹骨架 —
        # SUBTRACTED: routed 快照/ep fault/PP 面 (项 7/9); async 调度下把输出
        # 包进 AsyncGPUModelRunnerOutput (④拍『只等 D2H』的 worker 半边)
        if not self.use_async_scheduling:
            return output

        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            invalid_req_indices=[],
        )
        self._pending_async.append(async_output)
        # ENGINE SEAM (CPU): 模拟 D2H 在拷贝流上 pending (真实: copy stream +
        # event.record; 测试经 release_async_copies 完成)
        return async_output

    # SOURCE: vllm/v1/worker/gpu_model_runner.py _sample 调用位 (L4588-L4589) —
    # SUBTRACTED: self.sampler(logits, sampling_metadata) 全采样栈 (sampler 域);
    # ENGINE SEAM: 本章测试全 temperature=0 → greedy_sample 是该栈的 argmax 分支
    # SOURCE: vllm/v1/worker/gpu_model_runner.py _sample 调用位 (L4588-L4589) —
    def _sample(
        self, logits: torch.Tensor, spec_decode_metadata: Any | None = None
    ) -> SamplerOutput:
        self.trace.append((time.perf_counter_ns(), "greedy_sample"))
        sampled = Sampler.greedy_sample(logits)
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)


# ============================================================================
# vllm/v1/executor/uniproc_executor.py — the executor seam (②/④ 转发位)
# ============================================================================


# SOURCE: vllm/v1/executor/abstract.py:L44 Executor — SUBTRACTED: 工厂/分布式
# 面 (ch03/ch17); executors 的公共骨架
class Executor(ABC):  # HOST SEAM
    # SOURCE: vllm/v1/executor/abstract.py Executor.__init__ (骨架逐字)
    def __init__(self, vllm_config: Any, *args, **kwargs):
        self.vllm_config = vllm_config
        self._init_executor(*args, **kwargs)


# SOURCE: vllm/v1/executor/uniproc_executor.py:L26-L42 AsyncOutputFuture (逐字)
class AsyncOutputFuture(Future):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L27-L30 __init__ (逐字)
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L32-L42 result (逐字)
    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not super().done():
            try:
                output = self.async_output.get_output()
                self.set_result(output if self.single_value else [output])
            except Exception as e:
                self.set_exception(e)
        return super().result()


# SOURCE: vllm/v1/executor/uniproc_executor.py:L45-L147 UniProcExecutor —
# SUBTRACTED: worker 生命周期 (WorkerWrapperBase/init_device/load_model/net
# device — ch17); collective_rpc/execute_model/sample_tokens 转发面逐字
# (run_method 为 seam)
class UniProcExecutor(Executor):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L46-L69 _init_executor —
    # ENGINE SEAM: driver_worker = GPUModelRunner (worker 生命周期归 ch17);
    # num_gpu_blocks 的确定 (真实 determine_available_memory 剖析) 由 seam 给定值代行
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L46-L69 _init_executor —
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = GPUModelRunner(self.vllm_config)
        self.supported_tasks = ("generate",)
        self._failure_callback: Callable[[], None] | None = None
        if self.vllm_config.cache_config.num_gpu_blocks is None:
            # ENGINE SEAM of determine_available_memory (ch17)
            self.vllm_config.cache_config.num_gpu_blocks = 128

    # SOURCE: vllm/v1/engine/core.py:L134-L135 register_failure_callback (逐字位置)
    def register_failure_callback(self, callback: Callable[[], None]) -> None:
        self._failure_callback = callback

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L79-L106 collective_rpc (逐字)
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            return result if single_value else [result]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, single_value)
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L108-L121 execute_model (逐字)
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # In non-blocking mode, surface any exception as early as possible.
        if non_block and output.done():
            # Raise the exception in-line if the task failed.
            output.result()
        return output

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L123-L131 sample_tokens (逐字)
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L141-L143 shutdown (逐字)
    def shutdown(self) -> None:
        return None


# ============================================================================
# vllm/v1/engine/core.py — EngineCore (the five beats) + EngineCoreProc
# ============================================================================


# SOURCE: vllm/v1/engine/core.py:L98 HANDSHAKE_TIMEOUT_MINS (逐字)
HANDSHAKE_TIMEOUT_MINS = 5


# SOURCE: vllm/v1/engine/core.py:L103-L104 EngineCore (docstring 逐字)
class EngineCore:
    """Inner loop of vLLM's Engine."""

    # SOURCE: vllm/v1/engine/core.py:L106-L247 __init__ — SUBTRACTED: KV cache
    # 剖析装配 (_initialize_kv_caches — ch13/17; num_gpu_blocks 由 executor seam
    # 给定)、EEP 唤醒 (项 2)、spec 旗标 (项 7)、mm registry (项 4)、KV/EC
    # connector 握手 (项 3)、ec_consumer/pooling (项 3/8)、前缀哈希器装配
    # (ch13 — 字段保留 None)、idle 回调表 (项 5)、GC debug 钩子 (项 9);
    # 插件/日志/批量队列/step_fn 绑定/aborts_queue/GC 冻结/envs 缓存逐字
    # SOURCE: vllm/v1/engine/core.py:L106-L247 __init__
    def __init__(
        self,
        vllm_config: Any,
        executor_class: type,
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats
        # Opaque weight version supplied by the caller.
        self._weight_version = "default"

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # SUBTRACTED: vllm/v1/engine/core.py:L139-L140 EEP scale-up (项 2)
        # SUBTRACTED: vllm/v1/engine/core.py:L142-L143 KV cache 剖析与装配
        # (ch13/17; num_gpu_blocks 已由 executor seam 代行确定)

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        # SUBTRACTED: vllm/v1/engine/core.py:L149-L154 无 KV cache 模型的
        # chunked prefill 关闭 (ch13 域)

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=None,  # ENGINE SEAM (ch13)
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
        )
        # SUBTRACTED: vllm/v1/engine/core.py:L169-L174 spec 旗标与 connector
        # 聚合器 (项 3/7)
        # SUBTRACTED: vllm/v1/engine/core.py:L176-L200 mm registry/xfer 握手 (项 3/4)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        # SOURCE: vllm/v1/engine/core.py:L202-L212 (逐字)
        self.batch_queue_size = vllm_config.max_concurrent_batches
        self.batch_queue: Any | None = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # SUBTRACTED: vllm/v1/engine/core.py:L214-L218 is_ec_consumer /
        # is_pooling_model (项 3/8)

        self.request_block_hasher: Callable[[Request], list] | None = None
        # SUBTRACTED: vllm/v1/engine/core.py:L221-L229 前缀缓存哈希器装配
        # (ch13 域; 字段保留 None)

        # SOURCE: vllm/v1/engine/core.py:L231-L233 step_fn 静态绑定 (逐字;
        # step_with_batch_queue 本体是 ch12 的精简版 — 项 8)
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        # SOURCE: vllm/v1/engine/core.py:L234 async_scheduling (逐字)
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # SOURCE: vllm/v1/engine/core.py:L236 aborts_queue (逐字)
        self.aborts_queue = queue.Queue[list[str]]()

        # SUBTRACTED: vllm/v1/engine/core.py:L238 _idle_state_callbacks (项 5)

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        # SOURCE: vllm/v1/engine/core.py:L240-L242 freeze_gc_heap (逐字)
        freeze_gc_heap()
        # SUBTRACTED: vllm/v1/engine/core.py:L243-L244 GC debug 回调 (项 9)
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        # SOURCE: vllm/v1/engine/core.py:L245-L247 enable_envs_cache (逐字)
        enable_envs_cache()

    # SUBTRACTED: vllm/v1/engine/core.py:L249-L359 _initialize_kv_caches (ch13/17)

    # SOURCE: vllm/v1/engine/core.py:L361-L364 get_supported_tasks — SUBTRACTED:
    # _log_pooler_config (pooling 日志, 项 8/9)
    # SOURCE: vllm/v1/engine/core.py:L361-L364 get_supported_tasks
    def get_supported_tasks(self) -> tuple:
        supported_tasks = self.model_executor.supported_tasks
        return supported_tasks

    # SUBTRACTED: vllm/v1/engine/core.py:L366-L437 _log_pooler_config /
    # get_kv_cache_group_metadata (项 8/9 观测面)

    # SOURCE: vllm/v1/engine/core.py:L439-L483 add_request — SUBTRACTED: pooling
    # task 校验 (项 8)、kv/ec transfer 警告 (项 3)、abort_immediately (项 3);
    # request_id 类型校验与转交逐字
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

        # SUBTRACTED: vllm/v1/engine/core.py:L451-L477 pooling/kv/ec 校验 (项 3/8)

        self.scheduler.add_request(request)
        # SUBTRACTED: vllm/v1/engine/core.py:L480-L483 abort_immediately (项 3)

    # SOURCE: vllm/v1/engine/core.py:L485-L491 abort_requests (逐字)
    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # SOURCE: vllm/v1/engine/core.py:L493-L507 log_error_detail (逐字;
    # dump_engine_exception 为 seam)
    @contextmanager
    # SOURCE: vllm/v1/engine/core.py:L493-L507 log_error_detail (逐字;
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    # SUBTRACTED: vllm/v1/engine/core.py:L509-L577 capture_iteration_details /
    # _make_iteration_details_stats / _attach_iteration_details (项 9 观测面;
    # DP LB 统计附件)

    # SOURCE: vllm/v1/engine/core.py:L579-L582 _should_throttle_prefills (逐字)
    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        return False

    # SOURCE: vllm/v1/engine/core.py:L584-L614 step — 五拍全真 (F1 兑现处) —
    # SUBTRACTED: capture_iteration_details/_attach_iteration_details 观测附件
    # (项 9); 其余逐字
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
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        # SUBTRACTED: vllm/v1/engine/core.py:L598-L601 capture_iteration_details
        # 观测附件 (项 9)
        with (self.log_error_detail(scheduler_output),):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        # SUBTRACTED: vllm/v1/engine/core.py:L612 _attach_iteration_details (项 9)

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    # SOURCE: vllm/v1/engine/core.py:L616-L623 post_step — SUBTRACTED: draft
    # token 更新分支 (项 7 → ch32/33); 直通保留 (签名与调用位真实)
    # SOURCE: vllm/v1/engine/core.py:L616-L623 post_step
    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        # SUBTRACTED: vllm/v1/engine/core.py:L620-L623 draft 分支 (项 7)
        return None

    # SUBTRACTED: vllm/v1/engine/core.py:L625-L739 step_with_batch_queue (项 8
    # → ch12 的精简版; 精简版固定 batch_queue_size=1 走 step())

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

    # SOURCE: vllm/v1/engine/core.py:L751-L767 shutdown — SUBTRACTED: 无 (gc
    # unfreeze 与清理逐字; cleanup_dist_env_and_memory 为 host seam)
    # SOURCE: vllm/v1/engine/core.py:L751-L767 shutdown
    def shutdown(self):
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

        # Undo the gc.freeze() from __init__ so that the objects allocated
        # during engine startup (model weights, KV caches, etc.) become
        # visible to the garbage collector again. Without this, deleting
        # the engine in-process (e.g. unit tests) leaks GPU memory.
        gc.unfreeze()
        # Tear down distributed state initialized in this EngineCore process
        # before it exits and release cached memory.
        cleanup_dist_env_and_memory()
        logger.debug_once("[shutdown] EngineCore: local resource teardown complete")

    # SUBTRACTED: vllm/v1/engine/core.py:L769-L866 profile/reset_mm_cache/
    # reset_prefix_cache/reset_encoder_cache/_reset_caches/pause_scheduler/
    # resume_scheduler/is_scheduler_paused/sleep/wake_up/is_sleeping (项 5/6
    # 具体方法体; 分派骨架在 _handle_client_request 的 UTILITY 反射里保留)
    # SUBTRACTED: vllm/v1/engine/core.py:L868-L966 execute_dummy_batch/lora 四方法/
    # save_sharded_state/collective_rpc/set_weight_version (项 6 具体方法体)

    # SOURCE: vllm/v1/engine/core.py:L969-L991 preprocess_add_request — SUBTRACTED:
    # mm_receiver_cache 块 (项 4); 其余逐字 (含两段线程安全注释)
    # SOURCE: vllm/v1/engine/core.py:L969-L991 preprocess_add_request
    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        # SUBTRACTED: vllm/v1/engine/core.py:L978-L981 mm 接收缓存 (项 4)

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    # SUBTRACTED: vllm/v1/engine/core.py:L993-L999 _eep_* (项 2)

    # ── ENGINE SEAM observation / injection hooks (ch17/ch30 boundaries) ──
    # 测试与解说经真实 UTILITY 薄 RPC (getattr 反射分派) 扮演模型与语法编译器:
    # 引擎环内不伪造 forward —— 只消费脚本喂进来的行。

    # SOURCE: vllm/v1/engine/core.py (busy loop 产出注入位 — 前向产物) — SEAM
    def enqueue_forward_logits(self, steps: list) -> None:
        """Script the logits rows of upcoming forward passes (test
        instrumentation; the real forward is the ch17 boundary). One entry
        per step: a dict {request_id: logits row} — batch-composition
        races are thereby irrelevant; a missing row raises on the real
        error path."""
        self.model_executor.driver_worker.enqueue_logits(steps)

    # SOURCE: vllm/v1/engine/core.py (③拍 产物注入位) — SEAM
    def enqueue_grammar_bitmask(self, rows: list) -> None:
        """Script one ③拍 bitmask (test instrumentation; the real FSM
        compiler is the ch30 boundary). Rows follow structured_output_
        request_ids order."""
        self.structured_output_manager.enqueue_bitmask(rows)

    # SOURCE: vllm/v1/engine/core.py (busy loop 产出注入位) — SEAM 测试隔离钩:
    # 清空脚本队列 (跨测试共享同一引擎进程时, 消灭前测试的脚本残留)
    # SOURCE: vllm/v1/engine/core.py (busy loop 产出注入位) — SEAM 测试隔离钩:
    def clear_forward_scripts(self) -> None:
        runner = self.model_executor.driver_worker
        with runner._script_cond:
            runner._scripted_logits.clear()
        self.structured_output_manager._scripted_bitmasks.clear()

    # SOURCE: vllm/v1/engine/core.py (request 簿记观察口) — SEAM
    def get_step_count(self) -> int:
        return self.scheduler.current_step

    # SOURCE: vllm/v1/engine/core.py (request 簿记观察口) — SEAM
    def get_request_info(self, request_id: str) -> dict | None:
        request = self.scheduler.requests.get(request_id)
        if request is None:
            return None
        return {
            "status": request.status.name,
            "num_computed_tokens": request.num_computed_tokens,
            "num_output_tokens": request.num_output_tokens,
            "client_index": request.client_index,
            "is_prefill_chunk": request.is_prefill_chunk,
        }

    # SOURCE: vllm/v1/engine/core.py:L1583-L1585 (utility 失败路径) — SEAM 测试钩
    def boom_method(self):
        raise RuntimeError("boom_method exploded (test hook)")


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
    addresses: Any

    # SOURCE: vllm/v1/engine/core.py:L1015-L1127 __init__ — SUBTRACTED:
    # has_coordinator/internal_dp_balancing/publish_dp_lb_stats/last_counts
    # (DP stats, 项 1)、tensor IPC receiver (项 4)、_init_data_parallel (项 1)、
    # fault tolerance 哨兵 (项 3)、client_handshake_address (external LB, 项 1)、
    # coordinator READY 等待 (项 1); 双队列/identity/哨兵注入/握手窗口/双 IO
    # 线程 (含注释逐字) 保留
    # SOURCE: vllm/v1/engine/core.py:L1015-L1127 __init__
    def __init__(
        self,
        vllm_config: Any,
        local_client: bool,
        handshake_address: str,
        executor_class: type,
        log_stats: bool,
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

        # SUBTRACTED: vllm/v1/engine/core.py:L1038-L1042 TensorIpcReceiver (项 4)

        # SOURCE: vllm/v1/engine/core.py:L1044-L1050 (逐字, minus tensor_queue)
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
            # internal_dp_balancing — 项 1)
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

            # Don't complete handshake until DP coordinator ready message is
            # received.
            # SUBTRACTED: vllm/v1/engine/core.py:L1121-L1127 coordinator READY
            # 等待循环 (项 1)

    # SOURCE: vllm/v1/engine/core.py:L1129-L1192 _perform_handshakes — SUBTRACTED:
    # external-LB 双握手分支 (client_handshake_address, 项 1); 单握手主线逐字
    @contextmanager
    # SOURCE: vllm/v1/engine/core.py:L1129-L1192 _perform_handshakes
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: Any,
    ):
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
    # DP config-hash 附着 (项 1); 握手 socket 生命周期/HELLO→yield→READY 逐字
    @contextmanager
    # SOURCE: vllm/v1/engine/core.py:L1194-L1231 _perform_handshake
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: Any,
        parallel_config_to_update: Any | None = None,
    ):
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
            # SUBTRACTED: vllm/v1/engine/core.py:L1225-L1229 parallel_config_hash
            # (项 1 — MoE DP 配置校验)

            handshake_socket.send(msgpack_ext.encode(ready_msg))

    # SOURCE: vllm/v1/engine/core.py:L1233-L1269 startup_handshake (逐字)
    @staticmethod
    # SOURCE: vllm/v1/engine/core.py:L1233-L1269 startup_handshake (逐字)
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: Any | None = None,
    ) -> Any:
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
        init_message: Any = msgpack_ext.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    # SOURCE: vllm/v1/engine/core.py:L1271-L1360 run_engine_core — SUBTRACTED:
    # set_process_title/maybe_init_worker_tracer/decorate_logs/numa (项 9)、
    # kv_transfer_config dp 改名 (项 3)、DPEngineCoreProc MoE 分支 (项 1)、
    # process_title 变量; 信号注册/唤醒哨兵/忙循环/死讯/善后逐字
    # SOURCE: vllm/v1/engine/core.py:L1271-L1360 run_engine_core
    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: Any = kwargs["vllm_config"]
            parallel_config = vllm_config.parallel_config
            parallel_config.data_parallel_index = dp_rank
            # SUBTRACTED: vllm/v1/engine/core.py:L1283-L1304 DP 标题/tracer/
            # numa/kv_transfer 改名 (项 1/3/9)
            # Non-MoE DP ranks are completely independent, so treat like DP=1.
            # Note that parallel_config.data_parallel_index will still reflect
            # the original DP rank.
            # SUBTRACTED: vllm/v1/engine/core.py:L1307-L1310 DPEngineCoreProc
            # MoE 分支 (项 1)
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            # SOURCE: vllm/v1/engine/core.py:L1322-L1326 wakeup_engine (逐字)
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

    # SUBTRACTED: vllm/v1/engine/core.py:L1362-L1363 _init_data_parallel (项 1)

    # SOURCE: vllm/v1/engine/core.py:L1365-L1371 has_work (逐字 — 三来源判定;
    # engines_running 在 DP 删除后恒 False、batch_queue 恒 None, 判据保留)
    # SOURCE: vllm/v1/engine/core.py:L1365-L1371 has_work (逐字 — 三来源判定;
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    # SOURCE: vllm/v1/engine/core.py:L1373-L1375 is_running (逐字)
    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    # SOURCE: vllm/v1/engine/core.py:L1377-L1389 run_busy_loop — SUBTRACTED:
    # @fault_tolerant_wrapper (项 3) 与 _maybe_publish_request_counts 双调用
    # (DP LB stats, 项 1); 四拍骨架与 raise SystemExit 逐字
    # SOURCE: vllm/v1/engine/core.py:L1377-L1389 run_busy_loop
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # SUBTRACTED: vllm/v1/engine/core.py:L1383-L1384 publish request
            # counts (DP LB stats, 项 1 → ch34)
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
            # SUBTRACTED: vllm/v1/engine/core.py:L1387 publish request counts (项 1)

        raise SystemExit

    # SUBTRACTED: vllm/v1/engine/core.py:L1391-L1402 _maybe_publish_request_
    # counts (DP LB stats 发布, delete 项 1 → ch34)

    # SOURCE: vllm/v1/engine/core.py:L1404-L1433 _process_input_queue —
    # SUBTRACTED: _notify_idle_state_callbacks 调用点 (项 5); 其余逐字
    # (空闲清 aborts_queue/阻塞 get/积压清空)
    # SOURCE: vllm/v1/engine/core.py:L1404-L1433 _process_input_queue
    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            # SUBTRACTED: vllm/v1/engine/core.py:L1410 idle 回调 (项 5)
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

    # SOURCE: vllm/v1/engine/core.py:L1435-L1452 _process_engine_step (逐字 —
    # 含无执行有活时 1ms GIL 让渡: 无 connector 时该分支不可达但行为无害)
    # SOURCE: vllm/v1/engine/core.py:L1435-L1452 _process_engine_step (逐字 —
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there is still scheduler work
        # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
        # the GIL briefly to allow background transfer threads to make progress.
        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        return model_executed

    # SUBTRACTED: vllm/v1/engine/core.py:L1454-L1457 _notify_idle_callbacks (项 5)

    # SOURCE: vllm/v1/engine/core.py:L1459-L1505 _handle_shutdown — SUBTRACTED:
    # drain-timeout 计时语义的 while 等待面 (项 5 邻域; abort/drain 两模式的
    # 分流判据与请求处理逐字)
    # SOURCE: vllm/v1/engine/core.py:L1459-L1505 _handle_shutdown
    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout
            mode = "abort" if shutdown_timeout == 0 else "drain"

            logger.info(
                "[shutdown] EngineCore: start mode=%s timeout=%ds",
                mode,
                shutdown_timeout,
            )

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: aborting in-flight requests count=%d",
                        num_requests,
                    )
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: draining in-flight requests "
                        "count=%d timeout=%ds",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info(
                "[shutdown] EngineCore: request processing complete; "
                "starting resource teardown"
            )
            return False

        return True

    # SOURCE: vllm/v1/engine/core.py:L1507-L1540 _handle_client_request (逐字 —
    # ADD/ABORT/UTILITY/WAKEUP/EXECUTOR_FAILED 五路分派)
    # SOURCE: vllm/v1/engine/core.py:L1507-L1540 _handle_client_request (逐字 —
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
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

    # SOURCE: vllm/v1/engine/core.py:L1542-L1551 _reject_add_in_shutdown (逐字)
    def _reject_add_in_shutdown(self, request: Request) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.debug(
            "[shutdown] EngineCore: rejecting new request request_id=%s",
            request.request_id,
        )
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    # SOURCE: vllm/v1/engine/core.py:L1553-L1567 _reject_utility_in_shutdown (逐字)
    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning(
            "[shutdown] EngineCore: rejecting utility call method=%s",
            method_name,
        )
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

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

    # SOURCE: vllm/v1/engine/core.py:L1588-L1603 _convert_msgspec_args (逐字;
    # msgspec 为 seam)
    @staticmethod
    # SOURCE: vllm/v1/engine/core.py:L1588-L1603 _convert_msgspec_args (逐字;
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
    # dp_stats_address (项 1) 与 kv_events_config (ch37 邻域); post-init 配置
    # 回传字段逐字
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
    # coordinator XSUB 订阅与 READY 等待 (项 1)、b"READY" 忽略分支 (项 1)、
    # FT_UTILITY_METHOD 拦截 (项 3)、oob_tensor_provider (项 4); 主循环/ABORT
    # 双投递注释逐字
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
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            # SUBTRACTED: vllm/v1/engine/core.py:L1669-L1682 coordinator XSUB (项 1)

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

            # SUBTRACTED: vllm/v1/engine/core.py:L1695-L1698 coordinator READY (项 1)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # SUBTRACTED: vllm/v1/engine/core.py:L1706-L1710 b"READY"
                    # 忽略分支 (项 1)
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
    # coord_socket 与 client_index==-1 哨兵分支 (项 1 → ch34); linger=4000/
    # 盖章/encode_into 复用/首帧 tracker 逐字
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

    # SUBTRACTED: vllm/v1/engine/core.py:L1838-L1885 pause_scheduler/_pause_
    # complete (项 5 — pause/sleep 全族)

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
# EngineCoreActorMixin / DPMoEEngineCoreActor / EngineCoreActor (delete 项 1
# → ch34/39)


# ============================================================================
# vllm/v1/engine/utils.py — process orchestration (the front half of m8)
# ============================================================================


# SOURCE: vllm/v1/engine/utils.py:L40-L43 STARTUP_POLL_PERIOD_MS (逐字)
STARTUP_POLL_PERIOD_MS = 10000


# SOURCE: vllm/v1/engine/utils.py:L45-L48 CoreEngineState (逐字)
class CoreEngineState(enum.Enum):
    NEW = enum.auto()
    CONNECTED = enum.auto()
    READY = enum.auto()


# SOURCE: vllm/v1/engine/utils.py:L51-L58 CoreEngine (逐字)
class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    # SOURCE: vllm/v1/engine/utils.py:L54-L58 CoreEngine.__init__ (逐字)
    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


# SOURCE: vllm/v1/engine/utils.py:L61-L74 EngineZmqAddresses (逐字, minus
# coordinator 注释段随消费分支删 — 字段保留)
@dataclass
# SOURCE: vllm/v1/engine/utils.py:L61-L74 EngineZmqAddresses (逐字, minus
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
# GPU 绑定/NUMA 亲和包装 (平台绑定, 项 9 邻域); spawn/哨兵/监控主线逐字
# SOURCE: vllm/v1/engine/utils.py:L120-L250 CoreEngineProcManager
class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    # SOURCE: vllm/v1/engine/utils.py:L126-L171 CoreEngineProcManager.__init__
    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: Any,
        local_client: bool,
        handshake_address: str,
        executor_class: type,
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Any | None = None,
    ):
        context = get_mp_context()
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            # SUBTRACTED: vllm/v1/engine/utils.py:L146 tensor_queue (项 4)
        }

        # SUBTRACTED: vllm/v1/engine/utils.py:L149-L150 client_handshake_address
        # (external LB, 项 1)

        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        # SUBTRACTED: vllm/v1/engine/utils.py:L154 单模块化的 import 注释
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
                # SUBTRACTED: vllm/v1/engine/utils.py:L183-L209 GPU 绑定与 NUMA
                # 子进程包装 (平台绑定)
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


# SUBTRACTED: vllm/v1/engine/utils.py:L280-L1003 GPU 绑定/Ray 后端/
# CoreEngineActorManager (delete 项 1/9)


# SOURCE: vllm/v1/engine/utils.py:L253-L277 SignalCallback (逐字)
class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""

    # SOURCE: vllm/v1/engine/utils.py:L256-L264 SignalCallback.__init__ (逐字)
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
# 走 psutil (真实: vllm.utils 的进程树击杀); 主线逐字
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


# SOURCE: vllm/v1/engine/utils.py:L1005-L1048 get_engine_zmq_addresses — SUBTRACTED:
# defer_api_server_ports 参数 (Rust 前端, ch05 域) 与 elastic EP 翻转 (项 2)
# SOURCE: vllm/v1/engine/utils.py:L1005-L1048 get_engine_zmq_addresses
def get_engine_zmq_addresses(
    vllm_config: Any,
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
# DPCoordinator 运行段/ray 后端/external-LB 双握手链/tensor_queue 创建 (项 1/4);
# 单引擎自管路径逐字
# SOURCE: vllm/v1/engine/utils.py:L1053-L1203 launch_core_engines
@contextlib.contextmanager
# SOURCE: vllm/v1/engine/utils.py:L1053-L1203 launch_core_engines
def launch_core_engines(
    vllm_config: Any,
    executor_class: type,
    log_stats: bool,
    addresses: EngineZmqAddresses,
):
    """Launch engine and DP coordinator processes as needed."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    # SUBTRACTED: vllm/v1/engine/utils.py:L1085-L1110 tensor IPC 队列 (项 4)
    # SUBTRACTED: vllm/v1/engine/utils.py:L1087-L1110 run_coordinator (项 1)
    # SUBTRACTED: vllm/v1/engine/utils.py:L1112-L1123 ray backend (项 1)

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

    # SUBTRACTED: vllm/v1/engine/utils.py:L1153-L1155 elastic EP 翻转 (项 2)

    # Preserve "port=0 means auto-pick" for the handshake address, which
    # is consumed by engines spawned in this process and so cannot defer
    # port resolution to bind time.
    rpc_port = parallel_config.data_parallel_rpc_port or get_open_port()
    handshake_address = get_engine_client_zmq_addr(handshake_local_only, host, rpc_port)

    if local_engines_only and dp_rank > 0:
        # SUBTRACTED: vllm/v1/engine/utils.py:L1163-L1166 external-LB 本地握手 (项 1)
        raise AssertionError(
            "SUBTRACTED path: local_engines_only rank>0 (ch34). Unreachable with dp=1."
        )
    local_handshake_address = handshake_address

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
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
            )
        else:
            local_engine_manager = None

        # SUBTRACTED: tensor_queue (项 4) — None 占位, 保真实三元组 yield 形状
        yield local_engine_manager, addresses, None

        # Now wait for engines to start.
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            False,
            local_engine_manager,
        )


# SOURCE: vllm/v1/engine/utils.py:L1206-L1346 wait_for_engine_startup — SUBTRACTED:
# coordinator 进程 sentinel/remote headless 校验/MoE DP config-hash 校验 (项 1);
# **win32 哨兵 HOST SEAM** (spawn 哨兵是裸管道 HANDLE, zmq.Poller 误报可读 —
# 以 finished_procs() 轮询代行, 同一可观察契约); HELLO→init→READY 主线逐字
# SOURCE: vllm/v1/engine/utils.py:L1206-L1346 wait_for_engine_startup
def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: Any,
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
                    parallel_config={},
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
# vllm/v1/engine/core_client.py — InprocClient (the m10 contrast)
# ============================================================================


# SOURCE: vllm/v1/engine/core_client.py:L78 EngineCoreClient — SUBTRACTED: 具体
# utility 方法面 (项 6) / DP / elastic EP / FT; 基类壳保留 (InprocClient 的父类)
# SOURCE: vllm/v1/engine/core_client.py:L78 EngineCoreClient — SUBTRACTED: 具体
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    """


# SUBTRACTED: vllm/v1/engine/core_client.py SyncMPClient/AsyncMPClient/DP 系
# (delete 项 12 — 前端半边是 ch05 的精简版)


# SOURCE: vllm/v1/engine/core_client.py:L306-L336 InprocClient — SUBTRACTED:
# profile/save_sharded_state 等管理面 (项 12); 四方法面 + docstring 逐字
# SOURCE: vllm/v1/engine/core_client.py:L306-L336 InprocClient
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # SOURCE: vllm/v1/engine/core_client.py:L316-L317 InprocClient.__init__ (逐字)
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    # SOURCE: vllm/v1/engine/core_client.py:L319-L322 get_output (逐字)
    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

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

