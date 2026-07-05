# examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
#   —— subtract-only companion（ch10 第 3 层：proxy/router 负载均衡分发）
#
# proxy 的请求分发调度：SharedProxyScheduler 维护 PREFILL/DECODE 两个 RolePools，各是一个
#   惰性删除最小堆（least-loaded）。_priority：prefill = active_tokens + 0.3*active_kv_cache，
#   decode = active_tokens。一条请求的命：assign_instances 挑最轻 prefiller → build_prefill_request
#   盖章（do_remote_decode、max_tokens=1）→ send_request_to_service POST → 读回 kv_transfer_params
#   握手 → 挑最轻 decoder。
#
# 这是真实示例脚本，纯 Python，host 可跑核心分发逻辑（堆/打分/盖章）。被删的是部署弹性脚手架
# （NodeListener / 多进程 BaseManager / WorkerRuntime 客户端池 / lifespan / auth）。
import argparse
import asyncio
import heapq
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from runtime_stub import logger

# SUBTRACTED: import ipaddress / json / os / signal / time / Path / FastAPI / uvicorn /
#   BaseManager 等（原文件头）—— 仅服务被删的 HTTP 应用/多进程/弹性脚手架。
#   原 examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py 顶部导入


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L150
class ServerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceInfo:  # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L155
    request_id: str
    prefiller_key: str
    prefiller_score: float
    decoder_key: str
    decoder_score: float
    decoder_host: str
    decoder_port: int


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L166
TAINT_PRIORITY = 1e15


@dataclass
class BackendServer:  # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L173
    host: str
    port: int
    ordinal: int
    active_tokens: float = 0.0
    active_kv_cache: float = 0.0
    heap_seq: int = 0


@dataclass
class RolePools:  # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L183
    """Per-role scheduling state: live servers, priority heap, and drain-isolated keys."""

    servers: dict[str, BackendServer] = field(default_factory=dict)
    heap: list[tuple[float, int, int, str]] = field(default_factory=list)
    tainted: set[str] = field(default_factory=set)


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L201
def next_req_id() -> str:
    return str(uuid.uuid4())


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L205
def calculate_prefill_score(request_length: int) -> float:
    length_score = request_length / 4.0
    return length_score * 0.0345 + 120.0745


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L210
def calculate_decode_score(request_length: int) -> float:
    return request_length


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L218
def server_key(host: str, port: int) -> str:
    # SUBTRACTED: normalize_host 的 localhost/127.0.0.1 → 0.0.0.0 改写（L214-L215）—— 地址规范化细节。
    return f"{host}:{int(port)}"


# SUBTRACTED: build_server_url / build_base_url （L222-L234）—— IPv6 URL 成形，HTTP 客户端用。


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L237
class SharedProxyScheduler:
    """Centralized mutable scheduling state shared by all uvicorn workers.

    Uses lazy-deletion min-heap: on priority change, push a new entry and
    bump the server's ``heap_seq`` counter; stale entries (whose seq does
    not match) are skipped on pop.
    """

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L245
    def __init__(self, prefiller_instances, decoder_instances):
        self._lock = threading.RLock()
        self.request_num = 0
        self.waiting_nodes: dict[str, tuple[str, tuple[str, int], int]] = {}
        self._pools: dict[ServerRole, RolePools] = {
            ServerRole.PREFILL: RolePools(),
            ServerRole.DECODE: RolePools(),
        }
        self._ordinal = 0

        for host, port in prefiller_instances:
            self._add_server_no_lock(ServerRole.PREFILL, host, port)
        for host, port in decoder_instances:
            self._add_server_no_lock(ServerRole.DECODE, host, port)

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L260
    def _pool(self, role: ServerRole) -> RolePools:
        return self._pools[role]

    def _next_ordinal(self) -> int:  # SOURCE: load_balance_proxy_server_example.py:L271
        ordinal = self._ordinal
        self._ordinal += 1
        return ordinal

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L276
    def _priority(self, role: ServerRole, entry: BackendServer, key: str) -> float:
        if key in self._pool(role).tainted:
            return TAINT_PRIORITY
        if role is ServerRole.PREFILL:
            return entry.active_tokens + entry.active_kv_cache * 0.3
        return entry.active_tokens

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L283
    def _push_heap(self, role: ServerRole, key: str) -> None:
        pool = self._pool(role)
        entry = pool.servers[key]
        entry.heap_seq += 1
        heapq.heappush(pool.heap, (self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        if len(pool.heap) > 2 * len(pool.servers):
            self._reset_heap(role)

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L291
    def _pop_valid(self, role: ServerRole) -> str:
        pool = self._pool(role)
        while pool.heap:
            _, _, seq, key = heapq.heappop(pool.heap)
            if key not in pool.servers:
                continue
            entry = pool.servers[key]
            if entry.heap_seq == seq:
                return key
        raise RuntimeError(f"No available {role.value} servers")

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L302
    def _reset_heap(self, role: ServerRole, *, bump_seq: bool = False) -> None:
        pool = self._pool(role)
        heap = []
        for key, entry in pool.servers.items():
            if bump_seq:
                entry.heap_seq += 1
            heap.append((self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        heapq.heapify(heap)
        pool.heap = heap

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L312
    def _add_server_no_lock(self, role: ServerRole, host: str, port: int) -> bool:
        key = server_key(host, port)
        pool = self._pool(role)
        if key in pool.servers:
            return False
        pool.servers[key] = BackendServer(host, int(port), self._next_ordinal())
        self._push_heap(role, key)
        return True

    # SUBTRACTED: get_snapshot / log_status / healthcheck （L321-L350）—— 运维快照/日志。

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L352
    def _pick_server(
        self,
        role: ServerRole,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> dict[str, Any]:
        key = self._pop_valid(role)
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens += load
        if kv_cache:
            entry.active_kv_cache += load
        self._push_heap(role, key)
        return {"key": key, "host": entry.host, "port": entry.port}

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L369
    def _release_load(
        self,
        role: ServerRole,
        key: str | None,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> None:
        if not key or key not in self._pool(role).servers:
            return
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens -= load
        if kv_cache:
            entry.active_kv_cache = max(0.0, entry.active_kv_cache - load)
        self._push_heap(role, key)

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L387
    def begin_request(self, load: float) -> dict[str, Any]:
        """Pick a prefiller, reserve KV pressure, and count this as an active request."""
        with self._lock:
            picked = self._pick_server(ServerRole.PREFILL, load, kv_cache=True)
            self.request_num += 1
            return picked

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L394
    def reserve_prefill_kv(self, load: float) -> dict[str, Any]:
        """Pick a prefiller for recompute without bumping the active request count."""
        with self._lock:
            return self._pick_server(ServerRole.PREFILL, load, kv_cache=True)

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L399
    def pick_decoder(self, load: float) -> dict[str, Any]:
        with self._lock:
            return self._pick_server(ServerRole.DECODE, load, active_tokens=True)

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L403
    def release_prefill_kv(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.PREFILL, key, load, kv_cache=True)

    # SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L407
    def release_decoder(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.DECODE, key, load, active_tokens=True)

    # SUBTRACTED: finish_request / get_waiting_nodes / add_instances / mark_waiting_retry /
    #   activate_waiting_instance / drop_waiting_instance / remove_instances /
    #   finalize_tainted_instances （L411-L500）—— 实例增删/drain/taint 弹性管理。


# SUBTRACTED: SchedulerManager / WorkerRuntime / NodeListener / manager_config IO / parse_args /
#   lifespan / create_app / with_cancellation / auth_headers （L502-L788）—— 多进程共享、客户端
#   连接池、HTTP 应用脚手架。下面保留 get_runtime/get_global_args 的最小接缝供 assign_instances 引用。


# SUBTRACTED: get_runtime / get_global_args 真实体（多进程 runtime 单例 / argparse 全局）——
#   这里给出最小接缝，让 assign_instances 的编排骨架可读。
def get_runtime():  # SOURCE: load_balance_proxy_server_example.py:L567 get_runtime
    raise NotImplementedError("WorkerRuntime is process-shared scaffolding (subtracted).")


def get_global_args() -> argparse.Namespace:  # SOURCE: load_balance_proxy_server_example.py:L686
    raise NotImplementedError("Global argparse namespace is app scaffolding (subtracted).")


def auth_headers(request_id: str) -> dict[str, str]:  # SOURCE: load_balance_proxy_server_example.py:L783
    # SUBTRACTED: 真实体注入鉴权头；分发逻辑无关。
    return {"X-Request-Id": request_id}


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L790
def build_prefill_request(req_data: dict) -> dict:
    payload = req_data.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L809
async def send_request_to_service(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    req_data = build_prefill_request(req_data)
    headers = auth_headers(request_id)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc


# SUBTRACTED: stream_service_response_with_retry / _abort_prefill_selection / _finish_instance
#   （L835-L893）—— 流式回传重试与选型回滚的边角。


# SOURCE: examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:L896
async def assign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    is_initial_request: bool,
) -> InstanceInfo:
    runtime = get_runtime()
    args = get_global_args()
    prefiller_score = calculate_prefill_score(request_length)
    decoder_score = calculate_decode_score(request_length)
    request_id = next_req_id()
    pick_prefill = "begin_request" if is_initial_request else "reserve_prefill_kv"
    prefiller = await runtime.schedule(pick_prefill, prefiller_score)
    prefiller_key = prefiller["key"]

    response = await send_request_to_service(
        await runtime.get_client(ServerRole.PREFILL, prefiller_key),
        api,
        req_data,
        request_id,
        max_retries=args.max_retries,
        base_delay=args.retry_delay,
    )

    kv_transfer_params = response.json().get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    decoder = await runtime.schedule("pick_decoder", decoder_score)

    return InstanceInfo(
        request_id=request_id,
        prefiller_key=prefiller_key,
        prefiller_score=prefiller_score,
        decoder_key=decoder["key"],
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
    )


# SUBTRACTED: reassign_instances / handle_completions_impl / adjust_instances_impl 及全部 FastAPI
#   路由（L949-end）—— HTTP 入口与「KV 驱逐重算重选」重试编排，分发主线之外。
