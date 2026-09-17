# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSecondaryTierManager: Secondary tier for P2P KV cache sharing.

Owns transports and a single bidirectional P2PSession per remote peer.
"""
# ch38 P2P 层主干：三角色键解析 + 协议消费面 + PYTHONHASHSEED 硬门。
#
# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L1-L837
# SUBTRACTED（按减法计划删除项 9：p2p 运维细节）：
#   · ZmqTransport/NixlTransport 的构造与 _get_or_create_session/
#     _accept_new_peers（session 机器与传输后端实现体——教学面是三角色键
#     解析 + 协议消息类 + 对称 peer 语义；_sessions 恒空 → lookup 的
#     do_probe 分支/submit_store 的绑定快路自然走 MISS/停靠退化分支）。
#   · _reap_dead_sessions/_reap_unbound_stores 心跳收割与
#     _drain_inflight_for_shutdown 双模式排水（可靠性收尾，正文散文交代）。
#   · _poll_once 的控制面轮询体（无传输后端可轮）。

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import override

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
# SUBTRACTED: FileMapper 导入——config_fingerprint 随 NixlTransport 删除。
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
# SUBTRACTED: ZmqTransport/NixlTransport/P2PSession 导入——传输后端与
#   session 机器实现体未进精简版（契约面 control/base.py、data/base.py、
#   协议面 session/protocol.py 保留）。

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.base import ParentManager
    from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection

logger = init_logger(__name__)

# Reap unbound store batches that have been parked without a FetchMsg
# binding them to a session for longer than this. Protects against the
# prefiller buffering blocks for a decoder that never asks (decoder died,
# network partition, lost kv_request_id). Must be longer than the per-store
# deadline so the store-timeout path fires first for individual jobs.
_UNBOUND_STORE_TIMEOUT_S = 60.0

# SUBTRACTED: _SHUTDOWN_DRAIN_TIMEOUT_S / _DRAIN_SLEEP_S——排水循环删除
#   （删除项 9）后无消费点。


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L64-L72
def _remote_prefiller_params(kv_params: dict | None) -> dict | None:
    """Return the ``remote_prefiller`` sub-dict, or None if absent.

    Set on decoder requests to name the remote prefiller they pull from;
    carries kv_request_id, remote_host, remote_port.
    """
    if not kv_params:
        return None
    return kv_params.get("remote_prefiller")


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L75-L83
def _remote_decoder_params(kv_params: dict | None) -> dict | None:
    """Return the ``remote_decoder`` sub-dict, or None if absent.

    Set on prefiller requests to name the remote decoder they serve;
    carries kv_request_id.
    """
    if not kv_params:
        return None
    return kv_params.get("remote_decoder")


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L86-L94
def _remote_kv_source_params(kv_params: dict | None) -> dict | None:
    """Return the ``remote_kv_source`` sub-dict, or None if absent.

    Set on symmetric-P2P consumer requests to name the remote source they
    pull from; carries kv_request_id, remote_host, remote_port.
    """
    if not kv_params:
        return None
    return kv_params.get("remote_kv_source")


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L97-L103
def _peer_id_from_params(role_params: dict) -> str | None:
    """Build ``host:port`` peer_id from a role-scoped sub-dict, or None."""
    host = role_params.get("remote_host")
    port = role_params.get("remote_port")
    if host and port:
        return f"{host}:{port}"
    return None


@dataclass(slots=True)
class P2PSourceInfo:
    # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L107-L112
    """Consumer side: this request fetches from a remote (prefiller or peer)."""

    kv_request_id: str
    peer_id: str
    do_probe: bool  # False for remote_prefiller (PD), True for remote_kv_source


@dataclass(slots=True)
class P2PDestInfo:
    # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L116-L124
    """Producer side: a remote fetches this request's blocks from us.

    ``kv_request_id`` is None when the ``remote_decoder`` block is present
    but malformed (no id); the block's presence still marks the request as
    remote-decode, so submit_store must fail rather than store locally.
    """

    kv_request_id: str | None


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L127-L145
def _parse_source(kv_params: dict | None) -> P2PSourceInfo | None:
    """Parse the consumer sub-dict (PD ``remote_prefiller`` or symmetric
    ``remote_kv_source``) into a ``P2PSourceInfo``, or None if absent/incomplete."""
    role = _remote_prefiller_params(kv_params)
    do_probe = False
    if role is None:
        role = _remote_kv_source_params(kv_params)
        do_probe = True
    if not role:
        return None
    peer_id = _peer_id_from_params(role)
    kv_request_id = role.get("kv_request_id")
    if peer_id is None or not kv_request_id:
        return None
    return P2PSourceInfo(
        kv_request_id=kv_request_id,
        peer_id=peer_id,
        do_probe=do_probe,
    )


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L148-L154
def _parse_dest(kv_params: dict | None) -> P2PDestInfo | None:
    """Parse the producer ``remote_decoder`` sub-dict into a ``P2PDestInfo``,
    or None if the block is absent (not a remote-decode request)."""
    role = _remote_decoder_params(kv_params)
    if role is None:
        return None
    return P2PDestInfo(kv_request_id=role.get("kv_request_id") or None)


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L157-L169
def _annotate_req_context(req_context: ReqContext) -> None:
    """Parse kv_transfer_params once and cache the P2P routing state.

    Called from ``on_new_request``; later calls for the same request read
    the cached ``P2PSourceInfo``/``P2PDestInfo`` via ``get_state`` instead
    of re-parsing.
    """
    source = _parse_source(req_context.kv_transfer_params)
    if source is not None:
        req_context.set_state(source)
    dest = _parse_dest(req_context.kv_transfer_params)
    if dest is not None:
        req_context.set_state(dest)


@dataclass
class _UnboundStoreBatch:
    # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L173-L185
    """A submit_store batch parked at the manager before any peer has fetched.

    Indexed by kv_request_id only — the prefiller no longer learns the peer
    identity at store time. When a FetchMsg(kv_request_id) arrives on some
    session, the manager binds the kv_request_id to that session and replays
    every parked batch into ServerRole via session.add_stored_blocks.
    """

    job_id: int
    keys: list[OffloadKey]
    block_ids: Sequence[int]
    submitted_at: float = field(default_factory=time.monotonic)


# SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L188-L837
class P2PSecondaryTierManager(SecondaryTierManager):
    """Secondary tier for P2P KV cache sharing.

    A single P2PSession per remote peer handles both client-role (loading
    blocks from the peer) and server-role (serving blocks to the peer)
    over the same control connection.

    Single-threaded: every public method runs on the scheduler thread, and
    the engine drives polling via ``get_finished_jobs()`` once per step.
    ``has_pending_work()`` keeps the engine ticking so the control transport
    and existing sessions are polled even when no requests are scheduled.
    """

    # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L201-L322
    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str = "p2p",
        host: str | None = None,
        port: int | None = None,
        backends: list[str] | None = None,
        num_threads: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the P2P secondary tier manager.

        All keyword arguments after ``primary_kv_view`` come from the
        ``secondary_tiers`` entry in ``kv_connector_extra_config``. See
        ``docs/features/kv_offloading_usage.md`` for the user-facing
        configuration reference.

        Args:
            offloading_spec: Owning ``OffloadingSpec`` (provides normalized
                model, parallel, and cache layout configuration).
            primary_kv_view: Memoryview over the CPU primary tier; the
                NIXL agent registers this region for RDMA transfers.
            tier_type: Tier identifier (defaults to ``"p2p"``).
            host: Address the ZMQ control socket binds to, used verbatim
                as both the bind address and the identity peers dial back
                (mirrors the NIXL connector's ``VLLM_NIXL_SIDE_CHANNEL_HOST``;
                no auto-detection). Defaults to
                ``VLLM_P2P_SIDE_CHANNEL_HOST`` (``localhost``) when not set;
                must be set to the node's routable IP for cross-host P2P so
                remote peers can reach the socket.
            port: Base port for the ZMQ control socket. Must be
                reachable from peers. Defaults to
                ``VLLM_P2P_SIDE_CHANNEL_PORT`` (``5710``) when not set.
                The bound port is ``base + data_parallel_index`` so each
                DP replica gets a distinct port (one socket per replica,
                like NIXL); for DP=1 the offset is 0.
            backends: NIXL transport backends (e.g. ``["UCX"]``,
                ``["MOONCAKE"]``, ``["LIBFABRIC"]``). Defaults to
                ``["UCX"]``. When any non-UCX backend is requested, the
                NIXL agent is initialized with ``backends=...``;
                otherwise it falls back to a UCX-only agent with
                ``num_threads`` threads.
            num_threads: NIXL agent worker threads for the UCX-only
                branch. Ignored when ``backends`` contains a non-UCX
                entry.
            **kwargs: Reserved for future tier-specific options.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        # Block hashes chain from NONE_HASH, seeded from PYTHONHASHSEED
        # (see init_none_hash in v1/core/kv_cache_utils.py). Peers with
        # different seeds compute different hashes for identical content, so
        # lookups silently miss and no KV crosses the wire. Require it here so
        # a misconfigured P2P instance fails at startup rather than degrading
        # silently; the value is also verified against each peer on handshake.
        hash_seed = os.getenv("PYTHONHASHSEED")
        if hash_seed is None:
            raise ValueError(
                "PYTHONHASHSEED must be set for P2P KV offload so that block "
                "hashes match across instances. Set it to a fixed value (e.g. "
                "PYTHONHASHSEED=0) on every P2P peer."
            )
        self._hash_seed = hash_seed
        if host is None:
            host = envs.VLLM_P2P_SIDE_CHANNEL_HOST
        if port is None:
            port = envs.VLLM_P2P_SIDE_CHANNEL_PORT
        # One control socket per DP replica: offset the base by the global
        # data-parallel index so replicas on a host don't collide (mirrors
        # NIXL). For DP=1 the index is 0, leaving the base port unchanged.
        dp_index = offloading_spec.config.parallel.data_parallel_index
        port = int(port) + dp_index
        # Two decoupled identities:
        #   _local_id (``host:port``): the ZMQ control identity that peers
        #     dial back, used verbatim (the socket binds this host/port and
        #     the address is parsed back into host:port by the remote).
        #   _nixl_agent_name (uuid4): the NIXL agent name. It is never dialed
        #     — it travels opaquely inside the agent metadata blob — so it
        #     only needs to be globally unique. A per-process uuid guarantees
        #     that even for peers sharing a host:port (mirrors the NIXL
        #     connector; avoids the "remote agent name equals local" reject).
        self._local_id = f"{host}:{port}"
        self._nixl_agent_name = str(uuid.uuid4())

        # SUBTRACTED: config_fingerprint 推导与 NixlTransport/ZmqTransport
        #   构造（L285-L298）——删除项 9：传输后端实现体；对称 peer 的
        #   会话簿记保留（恒空 → 全部走无对端退化分支）。
        self._sessions: dict[str, object] = {}
        # kv_request_id → session, set when the bound session has received
        # FetchMsg for that id. submit_store after binding routes directly
        # to the session; before binding, batches are parked in
        # _unbound_stores below. Stays in sync with _sessions: entries
        # pointing to a reaped session are purged in _reap_dead_sessions.
        self._kv_to_session: dict[str, P2PSession] = {}
        # kv_request_id → list of batches submit_store'd before any peer
        # asked for that id. Drained into a session by _on_session_fetch
        # when the corresponding FetchMsg arrives, or surfaced as failures
        # by _reap_unbound_stores after _UNBOUND_STORE_TIMEOUT_S.
        self._unbound_stores: dict[str, list[_UnboundStoreBatch]] = {}

        self._finished_jobs: list[JobResult] = []
        # kv_request_ids that hit a transport/session failure; On load lookup()
        # rejects them so the request falls back to local prefill.
        self._failed_req_ids: set[str] = set()
        # Synthetic lookup ctxs from reaped sessions still owing a
        # ``parent.on_request_finished`` (the session's failed_serves). The
        # dead session had no parent handle at teardown; these are flushed
        # at the top of the next ``serve_external_requests`` where the
        # handle is valid.
        self._failed_serve_ctxs: list[ReqContext] = []

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L329-L355
        source = req_context.get_state(P2PSourceInfo)
        if source is None:
            return LookupResult.MISS
        if source.kv_request_id in self._failed_req_ids:
            return LookupResult.MISS

        # Symmetric-P2P consumer (``remote_kv_source`` sub-dict): probe the
        # peer asynchronously. First call registers the (kv_request_id,
        # key) entry and returns RETRY; flush_pending_lookups()
        # in on_schedule_end batches the LookupMsg; a later step's
        # lookup() returns HIT/MISS once LookupRespMsg has arrived.
        # PD path (``remote_prefiller`` sub-dict only) keeps the eager HIT.
        if source.do_probe:
            session = self._sessions.get(source.peer_id)
            if session is None:
                return LookupResult.MISS
            result = session.register_lookup(source.kv_request_id, key)
            if result is True:
                return LookupResult.HIT
            if result is False:
                return LookupResult.MISS
            return LookupResult.RETRY

        # PD consumer (we are the decoder): all kv blocks should be on the
        # prefiller side. Return HIT immediately.
        return LookupResult.HIT

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L358-L374
        """Parse kv_transfer_params once and open the outbound session.

        Parses the P2P routing state onto ``req_context`` (cached for the
        later lookup/submit/finish calls). On the consumer side
        (``remote_prefiller`` for PD or ``remote_kv_source`` for symmetric
        P2P), open a session toward the producer at remote_host:remote_port
        so submit_load can issue FetchMsg as soon as it fires. On the
        prefiller side, sessions are created when the consumer's inbound
        connection arrives in _accept_new_peers — submit_store no longer
        pre-creates anything.
        """
        _annotate_req_context(req_context)
        source = req_context.get_state(P2PSourceInfo)
        # SUBTRACTED: self._get_or_create_session(source.peer_id)——删除项 9
        #   （消费者侧向 producer 开会话；无传输后端不建）。
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L377-L411
        """Cancels pending loads and prunes session-scoped state.

        Consumer side (``remote_prefiller`` for PD or ``remote_kv_source``
        for symmetric-P2P): looks up the session by peer_id because the
        producer's address is what addresses the client-role load to
        cancel; also drops any pending symmetric-P2P lookup state via
        ``session.finish_request``.
        Prefiller side (``remote_decoder`` set): looks up via kv_request_id
        because peer_id is no longer carried on store-time
        kv_transfer_params; if a session has bound the id, finish it. If
        no session has bound the id yet, this is a no-op: parked batches
        in `_unbound_stores` are left in place and cleaned up only by
        `_reap_unbound_stores` after `_UNBOUND_STORE_TIMEOUT_S`.
        """
        source = req_context.get_state(P2PSourceInfo)
        dest = req_context.get_state(P2PDestInfo)
        kv_request_id = source.kv_request_id if source is not None else None
        if kv_request_id is None and dest is not None:
            kv_request_id = dest.kv_request_id
        if not kv_request_id:
            return
        self._failed_req_ids.discard(kv_request_id)

        if source is not None:
            session = self._sessions.get(source.peer_id)
            if session is not None:
                session.finish_request(kv_request_id)
            return

        # Prefiller-side finish: identify the session via kv_request_id.
        session = self._kv_to_session.pop(kv_request_id, None)
        if session is not None:
            session.finish_request(kv_request_id)
            return

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L414-L470
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        assert len(keys) == len(block_ids)

        dest = job_metadata.req_context.get_state(P2PDestInfo)
        logger.debug(
            "P2P %s: submit_store ENTRY job_id=%d blocks=%d "
            "remote_decoder=%s kv_request_id=%s",
            self._local_id,
            job_id,
            len(block_ids),
            dest is not None,
            dest.kv_request_id if dest is not None else None,
        )
        # Absent ``remote_decoder`` block => not a remote-decode request:
        # succeed locally without parking. An empty/malformed dict is still
        # a remote-decode signal and must fail the missing-id check below.
        if dest is None:
            self._finished_jobs.append(JobResult(job_id=job_id, success=True))
            return

        kv_request_id = dest.kv_request_id
        if not kv_request_id:
            logger.warning(
                "P2P %s: submit_store missing kv_request_id",
                self._local_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        # Fast path: a session has already received FetchMsg for this id,
        # so we can route the batch straight into its ServerRole.
        session = self._kv_to_session.get(kv_request_id)
        if session is not None:
            session.add_stored_blocks(kv_request_id, keys, block_ids, job_id)
            return

        # No session bound yet — park the batch keyed by kv_request_id.
        # _on_session_fetch drains it on the first FetchMsg; if no peer
        # ever asks, _reap_unbound_stores surfaces the job as failed.
        self._unbound_stores.setdefault(kv_request_id, []).append(
            _UnboundStoreBatch(
                job_id=job_id,
                keys=keys,
                block_ids=block_ids,
            )
        )
        logger.debug(
            "P2P %s: parked submit_store kv_request_id=%s job_id=%d blocks=%d",
            self._local_id,
            kv_request_id,
            job_id,
            len(block_ids),
        )

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L473-L529
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        source = job_metadata.req_context.get_state(P2PSourceInfo)
        logger.debug(
            "P2P %s: submit_load ENTRY job_id=%d blocks=%d kv_request_id=%s peer=%s",
            self._local_id,
            job_id,
            len(block_ids),
            source.kv_request_id if source is not None else None,
            source.peer_id if source is not None else None,
        )
        if source is None:
            logger.debug(
                "P2P %s: submit_load job_id=%d FAILED missing consumer params",
                self._local_id,
                job_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            return

        kv_request_id = source.kv_request_id
        peer_id = source.peer_id

        if not keys:
            logger.debug(
                "P2P %s: submit_load job_id=%d short-circuit success (no keys)",
                self._local_id,
                job_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=True))
            return

        session = self._sessions.get(peer_id)
        if session is None:
            logger.warning(
                "P2P %s: submit_load job_id=%d NO SESSION for peer=%s",
                self._local_id,
                job_id,
                peer_id,
            )
            self._finished_jobs.append(JobResult(job_id=job_id, success=False))
            self._failed_req_ids.add(kv_request_id)
            return
        logger.debug(
            "P2P %s: submit_load job_id=%d -> request_blocks peer=%s "
            "kv_request_id=%s blocks=%d session_ready=%s",
            self._local_id,
            job_id,
            peer_id,
            kv_request_id,
            len(block_ids),
            session.ready,
        )
        session.request_blocks(job_id, kv_request_id, keys, block_ids)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        # Drive one polling sweep on the scheduler thread, then hand off
        # whatever has accumulated. The engine calls this once per step
        # (and keeps stepping while has_pending_work() is True).
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L532-L539
        self._poll_once()
        result = self._finished_jobs
        self._finished_jobs = []
        return result

    @override
    def has_pending_work(self) -> bool:
        # The engine tick is the only driver of _control.poll() and
        # session.poll(); without it we miss new peer connects and
        # inbound fetch messages on existing sessions. Keep the engine
        # ticking for the lifetime of this manager.
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L542-L547
        return True

    @override
    def drain_jobs(self) -> None:
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L550-L573
        """Block until every submitted load/store job has completed or failed.

        Loops calling ``_poll_once()`` until no session has outstanding
        inbound loads or in-flight outbound stores. Mid-flight transfers
        are NOT cancelled — the caller (``TieringOffloadingManager.reset_cache``)
        needs the primary memoryview to be quiescent, not aborted. Results
        accumulate in ``_finished_jobs`` and are surfaced by the next
        ``get_finished_jobs()`` call.
        """
        # SUBTRACTED: 有界排水循环（L560-L573）——删除项 9：无会话即无在飞
        #   传输可排（骨架保留供 reset_cache 调用位）。
        return None

    @override
    def serve_external_requests(self, parent: ParentManager) -> None:
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L576-L591
        """Serve inbound peer lookups against the tiering manager.

        Called once per scheduler step (before this tier's
        ``on_schedule_end``) with a ``parent`` handle valid only for the
        duration of the call — the sole window in which the P2P server
        role may query the tiering manager. First release bookkeeping for
        the failed serves left by a reaped session, then let every live
        session resolve its enqueued inbound LookupMsgs.
        """
        if self._failed_serve_ctxs:
            for ctx in self._failed_serve_ctxs:
                parent.on_request_finished(ctx)
            self._failed_serve_ctxs = []
        for session in self._sessions.values():
            session.serve_external_requests(parent)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        # Flush any p2p lookups aggregated during this step.
        # One LookupMsg per (peer, kv_request_id) with unsent entries;
        # send-gating happens inside the session if not yet ready.
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L594-L599
        for session in self._sessions.values():
            session.flush_pending_lookups()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # SUBTRACTED: _get_or_create_session（L605-L628）——出站 ControlConnection
    #   与 P2PSession 构建（删除项 9：session 机器）。

    # SUBTRACTED: _accept_new_peers / _reap_dead_sessions / _reap_unbound_stores
    #   （L630-L731）——删除项 9：入站对端接纳与心跳收割（可靠性收尾）。


    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L737-L780
    def _poll_once(self) -> None:
        """One sweep of the polling work.

        Drains the control transport, polls every session, accumulates
        their results into ``_finished_jobs``, and reaps any dead sessions.
        Runs on the scheduler thread.
        """
        # SUBTRACTED: 控制面 poll / 会话 poll / 收割（L744-L780）——删除项 9
        #   （无传输后端与会话可轮；骨架保留供 get_finished_jobs 驱动位）。
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @override
    def shutdown(self) -> None:
        # SUBTRACTED: _drain_inflight_for_shutdown 的有界排水与逐会话 close
        #   （L787-L837）——删除项 9：双模式排水细节；簿记就地清空。
        # SOURCE: vllm/v1/kv_offload/tiering/p2p/manager.py:L787-L805
        self._sessions.clear()
        self._kv_to_session.clear()
        for batches in self._unbound_stores.values():
            for batch in batches:
                self._finished_jobs.append(
                    JobResult(job_id=batch.job_id, success=False)
                )
        self._unbound_stores.clear()
