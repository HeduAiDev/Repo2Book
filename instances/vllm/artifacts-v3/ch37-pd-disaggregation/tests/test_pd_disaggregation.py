# ch37《P/D 分离》精简版测试 —— TDD（先于实现）。
#
# 目标代码仓真实行为（vLLM v0.27.1, 6e448d0ea）的可观察复现：
#   - 部署门：两台完整引擎 P=kv_producer / D=kv_consumer；facade 按 role 只建半边
#   - disaggregator 控制回路：kv_transfer_params 回执信封的双跳搬运
#   - P 终局：request_finished 双条件 + delay_free + 租约戳 + remote_* 回执字典
#   - D 侧：do_remote_prefill → 全 prompt 异步拉；登记未哈希块 + 一次性翻转
#   - 带外握手：真 ZMQ side channel + compat hash 门 + RTT 中点时钟偏移
#   - RDMA 单边 READ：D 从 P 拉块；全命中 notif-only；消费者计数
#   - 完成回收：D 轮询 handle vs P 收 notif 的不对称 + TP 聚合器
#   - 租约心跳：scheduler_clock 重基准 / lease×2/3 只增不减 / 到期放块自愈
#   - 推模式：PUSH_REG 注册 + nixl-push-writer 配对 + WRITE 直写 D 预分配块
#   - 拒绝回执：serving 拒收 → abort_immediately 占位请求 → P 空 recv → notif 放块
#
# 运行：cd instances/vllm/artifacts-v3/ch37-pd-disaggregation
#       python -m pytest tests/ -q
from __future__ import annotations

import asyncio
import itertools
import time

import pytest
import torch

from _pd_harness import (
    BLOCK_SIZE,
    LAYER_NAMES,
    NUM_BLOCKS,
    Engine,
    block_checksum,
    fill_blocks,
    make_engine,
    make_kv_caches,
    make_kv_config,
    make_vllm_config,
    pump_until,
    side_channel,
)


# ═══════════════════════ 共用装配：一个请求跨 P/D 的一生 ═══════════════════════


def make_p_request(request_id: str, num_tokens: int = 16):
    """P 腿请求：do_remote_decode=True——P 由此知道自己只算 prefill。"""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                }
            },
        ),
    )


def make_d_request(request_id: str, receipt: dict):
    """D 腿请求：proxy 把 P 的回执信封原样附加（do_remote_prefill=True）。"""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    params = dict(receipt)
    params["do_remote_prefill"] = True
    params["do_remote_decode"] = False
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(
            max_tokens=8, extra_args={"kv_transfer_params": params}
        ),
    )


class KVCacheBlocksStub:
    """KVCacheBlocks 的测试替身：只提供 connector 用到的两个查询面。"""

    def __init__(self, groups):
        self.blocks = tuple(
            [_Block(i) for i in group] for group in groups
        )

    def get_unhashed_block_ids_all_groups(self):
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        return KVCacheBlocks.get_unhashed_block_ids_all_groups(self)


class _Block:
    def __init__(self, block_id, block_hash=None):
        self.block_id = block_id
        self.block_hash = block_hash
        self.is_null = False


def pair_of_engines(**kw):
    """两台完整引擎 + side channel 装配（P 先发布握手元数据）。"""
    p = make_engine("p", kv_role="kv_producer", **kw)
    d = make_engine("d", kv_role="kv_consumer", **kw)
    return p, d


# ═══════════════════ m1 部署形态：角色门 + facade 半边构建 ═══════════════════


def test_nixl_connector_is_pull_alias_and_factory_resolves_three_names():
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
        NixlPullConnector,
        NixlPushConnector,
    )

    assert NixlConnector is NixlPullConnector
    for name, cls in (
        ("NixlConnector", NixlPullConnector),
        ("NixlPullConnector", NixlPullConnector),
        ("NixlPushConnector", NixlPushConnector),
    ):
        cfg = make_vllm_config(engine_id="e", kv_role="kv_producer", connector=name)
        assert KVConnectorFactory.get_connector_class(cfg.kv_transfer_config) is cls


def test_facade_builds_only_one_half_per_role():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlPullConnector,
    )

    cfg = make_vllm_config(engine_id="e", kv_role="kv_producer")
    kv_config = make_kv_config()
    sched = NixlPullConnector(cfg, KVConnectorRole.SCHEDULER, kv_config)
    worker = NixlPullConnector(cfg, KVConnectorRole.WORKER, kv_config)
    assert sched.connector_worker is None and sched.connector_scheduler is not None
    assert worker.connector_scheduler is None and worker.connector_worker is not None
    assert sched.engine_id == "e"


def test_kv_role_is_the_deployment_gate():
    from vllm.config.kv_transfer import KVTransferConfig

    a = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_producer")
    b = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_producer")
    assert a.engine_id and a.engine_id != b.engine_id  # 默认 uuid4，各引擎独立
    assert a.is_kv_producer and not a.is_kv_consumer

    with pytest.raises(ValueError, match="Please specify kv_role"):
        KVTransferConfig(kv_connector="NixlConnector")
    with pytest.raises(ValueError, match="Unsupported kv_role"):
        KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_watcher")


def test_nixl_requires_hnd_layout_and_warns_on_kv_both():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlPullConnector,
    )

    cfg = make_vllm_config(engine_id="e", kv_role="kv_producer")
    assert NixlPullConnector.get_required_kvcache_layout(cfg) == "HND"
    cfg.model_config.use_mla = True
    assert NixlPullConnector.get_required_kvcache_layout(cfg) is None

    both = make_vllm_config(engine_id="e2", kv_role="kv_both")
    conn = NixlPullConnector(both, KVConnectorRole.SCHEDULER, make_kv_config())
    assert conn.connector_scheduler is not None  # kv_both 仍可用，但已弃用（告警）


# ═══════════ m2 控制回路：kv_transfer_params 回执信封的双跳搬运 ═══════════


def test_proxy_rewrites_request_for_prefill_leg():
    from toy_proxy_server import send_request_to_service

    class _Resp:
        def raise_for_status(self):
            return None

        async def aread(self):
            return b"{}"

    class _Client:
        def __init__(self):
            self.sent = None

        async def post(self, endpoint, json=None, headers=None):
            # 快照在 post 时刻（发单后再改本地 req_data 不影响已发出的载荷）
            self.sent = (endpoint, dict(json))
            return _Resp()

    client = _Client()
    req_data = {"model": "m", "prompt": "hi", "max_tokens": 7, "min_tokens": 2}

    asyncio.run(
        send_request_to_service(
            {"client": client}, "/completions", req_data, request_id="r1"
        )
    )
    sent = client.sent[1]
    assert sent["kv_transfer_params"]["do_remote_decode"] is True
    assert sent["max_tokens"] == 1  # P 只算 prefill 一个 token
    assert sent["stream"] is False  # 终局要掏回执，不能流式
    assert "min_tokens" not in sent  # P 不支持的参数被摘掉
    assert req_data["min_tokens"] == 2  # 但 D 腿还要用，发完单再放回去
    assert req_data["max_tokens"] == 7  # 原始请求体不被就地改写（copy 后改）


def test_proxy_forwards_prefill_receipt_to_decode_leg():
    from fastapi import FastAPI
    from starlette.requests import Request as StarletteRequest

    from toy_proxy_server import _handle_completions

    receipt = {
        "do_remote_prefill": True,
        "remote_block_ids": [[0, 1]],
        "remote_engine_id": "engine-p",
        "remote_request_id": "req-1",
        "remote_host": "127.0.0.1",
        "remote_port": 5600,
        "tp_size": 1,
        "remote_num_tokens": 16,
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"kv_transfer_params": receipt}

        async def aread(self):
            return b""

        async def aclose(self):
            return None

    class _Client:
        def __init__(self, payload=None, chunks=()):
            self.payload = payload
            self.chunks = chunks
            self.sent = None

        async def post(self, endpoint, json=None, headers=None):
            self.sent = (endpoint, json)
            return _Resp() if self.payload is None else _JsonResp(self.payload)

        def stream(self, method, endpoint, json=None, headers=None):
            self.sent = (endpoint, json)
            return _StreamCtx(self.chunks)

    class _JsonResp(_Resp):
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class _StreamCtx:
        def __init__(self, chunks):
            self.chunks = chunks

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            for chunk in self.chunks:
                yield chunk

    prefill_client, decode_client = _Client(), _Client(chunks=[b"data: {}\n\n"])
    app = FastAPI()
    app.state.prefill_clients = [{"client": prefill_client, "host": "h", "port": 1, "id": 0}]
    app.state.decode_clients = [{"client": decode_client, "host": "h", "port": 2, "id": 0}]
    app.state.prefill_iterator = itertools.cycle(range(1))
    app.state.decode_iterator = itertools.cycle(range(1))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/completions",
        "headers": [],
        "app": app,
    }
    starlette_req = StarletteRequest(scope)

    async def _body():
        return {"model": "m", "prompt": "hi", "stream": True}

    starlette_req.json = _body  # type: ignore[method-assign]

    async def run():
        resp = await _handle_completions("/completions", starlette_req)
        body = b"".join([chunk async for chunk in resp.body_iterator])
        return body

    body = asyncio.run(run())
    assert body == b"data: {}\n\n"
    # D 腿请求体里就是 P 的回执原件（控制面的全部工作 = 搬运这只信封）
    assert decode_client.sent[1]["kv_transfer_params"] == receipt


def test_request_takes_kv_transfer_params_off_extra_args():
    req = make_p_request("req-1")
    assert req.kv_transfer_params["do_remote_decode"] is True
    assert req.kv_transfer_params["remote_engine_id"] is None


# ═══════════════ m3 P 终局：租约钉住 + 回执字典（F8 回收①） ═══════════════


def test_p_terminal_pins_blocks_with_lease_and_returns_receipt():
    p, _d = pair_of_engines()
    try:
        req = make_p_request("req-p")
        req.status = __import__("vllm.v1.request", fromlist=["RequestStatus"]).RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 16
        sched = p.scheduler
        t0 = time.perf_counter()
        delay_free, receipt = sched.request_finished(req, ([0, 1, 2, 3],))
        assert delay_free is True  # 块交给 connector 异步释放
        assert req.request_id in sched._reqs_need_send
        deadline = sched._reqs_need_send[req.request_id]
        assert sched._kv_lease_duration == 30
        assert abs(deadline - (t0 + 30)) < 1.0
        assert receipt["remote_block_ids"] == ([0, 1, 2, 3],)
        assert receipt["remote_engine_id"] == p.engine_id
        assert receipt["remote_request_id"] == "req-p"
        assert receipt["remote_host"] == "127.0.0.1"
        assert receipt["remote_port"] == sched.side_channel_port
        assert receipt["tp_size"] == 1
        assert receipt["remote_num_tokens"] == 16
        assert receipt["do_remote_prefill"] is True  # 对 D 而言：prefill 在远端
        assert receipt["do_remote_decode"] is False
    finally:
        p.close()


def test_p_terminal_non_handoff_status_does_not_hand_over():
    from vllm.v1.request import RequestStatus

    p, _d = pair_of_engines()
    try:
        req = make_p_request("req-abort")
        req.status = RequestStatus.FINISHED_ABORTED
        sched = p.scheduler
        delay_free, receipt = sched.request_finished(req, ([0, 1],))
        assert delay_free is False and receipt is None
        assert "req-abort" in sched._reqs_not_processed
        assert "req-abort" not in sched._reqs_need_send
    finally:
        p.close()


def test_p_finished_request_never_scheduled_registers_empty_recv():
    """拒绝回执：do_remote_prefill 仍为 True ⇒ 从未被调度 ⇒ 空 recv 让 worker 放块。"""
    p, d = pair_of_engines()
    try:
        receipt = {
            "do_remote_prefill": True,
            "remote_block_ids": [[0, 1]],
            "remote_engine_id": p.engine_id,
            "remote_request_id": "req-p",
            "remote_host": "127.0.0.1",
            "remote_port": p.scheduler.side_channel_port,
            "tp_size": 1,
            "remote_num_tokens": 16,
        }
        req = make_d_request("req-rejected", receipt)
        sched = d.scheduler
        delay_free, ret = sched.request_finished(req, ([],))
        assert (delay_free, ret) == (False, None)
        assert sched._reqs_need_recv["req-rejected"] == (req, [])  # 空块列表
        assert req.kv_transfer_params["do_remote_prefill"] is False  # 不重复触发
    finally:
        p.close()
        d.close()


# ═══════════════ m4 D 查命中：全 prompt 异步拉 + 一次性翻转 ═══════════════


def test_d_remote_prefill_pulls_all_prompt_blocks_asynchronously():
    p, d = pair_of_engines()
    try:
        receipt = _receipt_from_p(p, "req-1")
        req = make_d_request("req-1", receipt)
        sched = d.scheduler
        count, async_load = sched.get_num_new_matched_tokens(req, 0)
        assert (count, async_load) == (16, True)  # 全 prompt，异步（不阻塞准入）
        assert sched.get_num_new_matched_tokens(req, 12) == (4, True)
    finally:
        p.close()
        d.close()


def test_d_update_state_after_alloc_registers_unhashed_blocks_once():
    p, d = pair_of_engines()
    try:
        receipt = _receipt_from_p(p, "req-1")
        req = make_d_request("req-1", receipt)
        sched = d.scheduler
        blocks = KVCacheBlocksStub([[0, 1, 2, 3]])
        sched.update_state_after_alloc(req, blocks, 16)
        assert sched._reqs_need_recv["req-1"] == (req, [[0, 1, 2, 3]])
        # 一次性翻转：抢占/重入不重发第二次传输
        sched.update_state_after_alloc(req, KVCacheBlocksStub([[4, 5]]), 16)
        assert sched._reqs_need_recv["req-1"] == (req, [[0, 1, 2, 3]])
    finally:
        p.close()
        d.close()


# ═══════════════ m5 带外握手：side channel + compat hash 门 ═══════════════


def test_engine_core_startup_publishes_worker_handshake_metadata():
    p = make_engine("p", kv_role="kv_producer")
    try:
        sched = p.scheduler
        assert sched._nixl_handshake_listener_t is not None
        assert sched._nixl_handshake_listener_t.is_alive()  # ROUTER 监听已就位
        payload = p.worker_connector.get_handshake_metadata()
        assert payload.compatibility_hash == p.worker.compat_hash
    finally:
        p.close()


def test_handshake_over_zmq_passes_compat_gate_and_estimates_clock_offset():
    p, d = pair_of_engines()
    try:
        d_worker = d.worker
        p_sched = p.scheduler
        agents, clock_offset = d_worker._nixl_handshake(
            p_sched.side_channel_host,
            p_sched.side_channel_port,
            p.worker.world_size,
            p.engine_id,
        )
        assert set(agents) == {(0, 0)}  # (pp_rank, tp_rank) -> agent 名
        assert abs(clock_offset) < 1.0  # 同机：偏移≈0
        # 远端注册由 _ensure_handshake 的完成回调落账（本函数只负责握手本身），
        # 见 test_ensure_handshake_is_single_flight。
    finally:
        p.close()
        d.close()


def test_handshake_rejects_incompatible_configuration():
    p, d = pair_of_engines()
    try:
        d_worker = d.worker
        d_worker.compat_hash = "0" * 64
        p_sched = p.scheduler
        with pytest.raises(RuntimeError, match="compatibility hash mismatch"):
            d_worker._nixl_handshake(
                p_sched.side_channel_host,
                p_sched.side_channel_port,
                p.worker.world_size,
                p.engine_id,
            )
    finally:
        p.close()
        d.close()


def test_ensure_handshake_is_single_flight():
    p, d = pair_of_engines()
    try:
        d_worker = d.worker
        p_sched = p.scheduler
        fut = d_worker._ensure_handshake(
            p.engine_id, p_sched.side_channel_host, p_sched.side_channel_port, 1
        )
        assert fut is not None
        fut.result(timeout=10)
        # 握手完成后：单飞表清空、远端已注册、再问一次直接返回 None
        assert d_worker._ensure_handshake(
            p.engine_id, p_sched.side_channel_host, p_sched.side_channel_port, 1
        ) is None
        assert p.engine_id in d_worker._remote_agents
        assert p.engine_id in d_worker._engine_clock_offset
    finally:
        p.close()
        d.close()


# ═══════════════ m6 RDMA 单边 READ：D 直接从 P 显存拉块 ═══════════════


def test_read_pulls_remote_blocks_into_local_cache():
    p, d = pair_of_engines()
    try:
        receipt = _receipt_from_p(p, "req-1", blocks=[0, 1, 2, 3])
        fill_blocks(p.kv_caches, [0, 1, 2, 3], 7.0)
        fill_blocks(d.kv_caches, [0, 1, 2, 3], 0.0)

        req = make_d_request("req-1", receipt)
        d.scheduler.update_state_after_alloc(
            req, KVCacheBlocksStub([[0, 1, 2, 3]]), 16
        )
        d.worker_step()  # 首遇远端 → 后台握手（请求进 _ready_requests）
        pump_until(lambda: d.worker._remote_agents.get(p.engine_id))
        # 握手就绪后的下一步排空 _ready_requests → 发 READ；再下一步收完成
        _, done_recving = pump_until(
            lambda: (lambda r: r if r[1] else None)(d.worker_step())
        )
        assert done_recving == {"req-1"}
        for b in (0, 1, 2, 3):
            assert torch.equal(d.kv_caches[LAYER_NAMES[0]][b], p.kv_caches[LAYER_NAMES[0]][b])
        assert block_checksum(d.kv_caches, 0) == pytest.approx(
            float(p.kv_caches[LAYER_NAMES[0]][0].numel()) * 7.0
        )
    finally:
        p.close()
        d.close()


def test_full_prefix_cache_hit_sends_notif_only():
    p, d = pair_of_engines()
    try:
        receipt = _receipt_from_p(p, "req-1", blocks=[0, 1])
        p.worker._reqs_to_process.add("req-1")
        p.worker._reqs_to_send["req-1"] = time.perf_counter() + 30

        req = make_d_request("req-1", receipt)
        d.scheduler.update_state_after_alloc(
            req, KVCacheBlocksStub([[]]), 0  # 本地全命中：未哈希块为空
        )
        d.worker_step()
        pump_until(lambda: d.worker._remote_agents.get(p.engine_id))
        d.worker_step()  # 排空 _ready_requests → 全命中路径只发 notif
        # 没有发生任何 READ（D 侧没有任何 handle）
        assert not d.worker._recving_transfers.get("req-1")
        # P 侧收到 notif 即放块
        done_sending = pump_until(
            lambda: (lambda r: r[0] if r[0] else None)(p.worker_step())
        )
        assert done_sending == {"req-1"}
        assert "req-1" not in p.worker._reqs_to_send
    finally:
        p.close()
        d.close()


def test_partial_prefix_hit_trims_remote_block_tail():
    p, d = pair_of_engines()
    try:
        w = d.worker
        local, remote = w._apply_prefix_caching([[0, 1]], [[0, 1, 2, 3]], 1)
        assert local == [[0, 1]] and remote == [[2, 3]]  # 只拉未命中的尾段
    finally:
        p.close()
        d.close()


# ═══════════════ m7 完成回收：双端信号不对称 + TP 聚合 ═══════════════


def test_done_recving_from_handle_poll_and_done_sending_from_notif():
    p, d = pair_of_engines()
    try:
        receipt = _receipt_from_p(p, "req-1", blocks=[0, 1])
        req = make_d_request("req-1", receipt)
        d.scheduler.update_state_after_alloc(
            req, KVCacheBlocksStub([[0, 1]]), 8
        )
        # P 腿：读端在批集合 + 租约戳（P 才能接受 notif）
        p.worker._reqs_to_process.add("req-1")
        p.worker._reqs_to_send["req-1"] = time.perf_counter() + 30

        d.worker_step()
        pump_until(lambda: d.worker._remote_agents.get(p.engine_id))
        pump_until(lambda: (lambda r: r if r and r[1] else None)(d.worker_step()))

        done_sending, _ = pump_until(
            lambda: (lambda r: r if r[0] else None)(p.worker_step())
        )
        assert done_sending == {"req-1"}
        assert "req-1" not in p.worker._reqs_to_send  # 块已放
    finally:
        p.close()
        d.close()


def test_kv_output_aggregator_reports_only_after_every_worker():
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    agg = KVOutputAggregator(expected_finished_count=2)

    def out(finished_recving):
        return ModelRunnerOutput(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            sampled_token_ids=[[0]],
            kv_connector_output=KVConnectorOutput(
                finished_recving=set(finished_recving) or None
            ),
        )

    first = agg.aggregate([out({"r1"}), out(set())])
    assert first.kv_connector_output.finished_recving is None  # 只报了一次
    second = agg.aggregate([out({"r1"}), out({"r1"})])
    assert second.kv_connector_output.finished_recving == {"r1"}  # 齐了才上报
    second = agg.aggregate([out(set()), out({"r1"})])
    assert second.kv_connector_output.finished_recving == {"r1"}


def test_scheduler_promotes_and_frees_on_kv_xfer_finished():
    from vllm.v1.outputs import KVConnectorOutput

    p = make_engine("p", kv_role="kv_producer")
    try:
        sched = p.core.scheduler
        req = make_p_request("req-p")
        req.status = sched_request_status("WAITING_FOR_REMOTE_KVS")
        sched.requests["req-p"] = req
        sched._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={"req-p"})
        )
        assert "req-p" in sched.finished_recving_kv_req_ids

        done = make_p_request("req-done")
        done.status = sched_request_status("FINISHED_STOPPED")
        sched.requests["req-done"] = done
        sched._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_sending={"req-done"})
        )
        assert "req-done" not in sched.requests  # P 侧块已放
    finally:
        p.close()


def sched_request_status(name):
    from vllm.v1.request import RequestStatus

    return getattr(RequestStatus, name)


def test_scheduler_has_requests_includes_pending_push_work():
    p = make_engine("p", kv_role="kv_producer", connector="NixlPushConnector")
    try:
        sched = p.core.scheduler
        assert sched.has_requests() is False
        sched.connector.connector_scheduler._finished_request_blocks["r"] = ([0],)
        assert sched.has_requests() is True  # 无活请求也保活到 WRITE 收尾
    finally:
        p.close()


# ═══════════════ m8 租约心跳与过期：跨进程时钟重基准 + 自愈 ═══════════════


def test_lease_deadline_is_rebased_onto_worker_clock():
    from vllm.v1.core.sched.output import SchedulerOutput

    p, d = pair_of_engines()
    try:
        w = d.worker
        meta = d.build_meta(SchedulerOutput())
        # 伪造「另一个进程」的时钟基准：截止时刻用的是 100000，基准 99000
        meta.scheduler_clock = 99000.0
        meta.reqs_to_send = {"req-1": 100000.0}
        w._reqs_to_process.add("req-1")
        t0 = time.perf_counter()
        w.start_load_kv(meta)
        assert 990 < w._reqs_to_send["req-1"] - t0 < 1010  # 剩余 TTL 原样搬到本机时钟
    finally:
        p.close()
        d.close()


def test_worker_without_scheduler_clock_keeps_deadline():
    from vllm.v1.core.sched.output import SchedulerOutput

    p, d = pair_of_engines()
    try:
        w = d.worker
        meta = d.build_meta(SchedulerOutput())
        meta.reqs_to_send = {"req-1": 100000.0}
        meta.scheduler_clock = 0.0  # 旧式 metadata
        w._reqs_to_process.add("req-1")
        w.start_load_kv(meta)
        assert w._reqs_to_send["req-1"] == 100000.0
    finally:
        p.close()
        d.close()


def test_heartbeat_extends_lease_and_never_shortens():
    p, d = pair_of_engines()
    try:
        w = p.worker
        w._reqs_to_send["req-1"] = time.perf_counter() + 1.0
        before = w._reqs_to_send["req-1"]
        w._handle_heartbeat("req-1,req-unknown")
        assert w._reqs_to_send["req-1"] > before + 19  # 续 2/3 个租约
        extended = w._reqs_to_send["req-1"]
        w._handle_heartbeat("req-1")
        assert w._reqs_to_send["req-1"] >= extended  # 只增不减
    finally:
        p.close()
        d.close()


def test_scheduler_packages_heartbeats_throttled_by_interval():
    p, d = pair_of_engines()
    try:
        receipt = _receipt_from_p(p, "req-1")
        req = make_d_request("req-1", receipt)
        sched = d.scheduler
        sched.on_new_request(req)
        assert p.engine_id in sched._heartbeat_by_engine
        assert sched._heartbeat_by_engine[p.engine_id].req_ids == {"req-1"}
        meta = d.build_meta()
        assert meta.heartbeat_by_engine[p.engine_id].req_ids == {"req-1"}
        # 节流：间隔内不再打包
        assert d.build_meta().heartbeat_by_engine == {}
        sched.update_connector_output(_kv_out(finished_recving={"req-1"}))
        assert sched._heartbeat_by_engine == {}  # 传输完成即停跳
    finally:
        p.close()
        d.close()


def test_expired_lease_releases_blocks_without_anyone_collecting():
    p, _d = pair_of_engines()
    try:
        w = p.worker
        w._reqs_to_send["req-dead"] = time.perf_counter() - 1.0
        w._reqs_to_process.add("req-dead")
        done_sending, _ = w.get_finished()
        assert done_sending == {"req-dead"}  # 到期强制放块
        assert "req-dead" not in w._reqs_to_send
    finally:
        p.close()


def _kv_out(**kw):
    from vllm.v1.outputs import KVConnectorOutput

    return KVConnectorOutput(**kw)


# ═══════════════ m9 推模式：PUSH_REG 注册 + writer 配对 + WRITE ═══════════════


def test_push_mode_registers_then_writes_into_decode_blocks():
    p, d = pair_of_engines(connector="NixlPushConnector")
    try:
        receipt = _receipt_from_p(p, "req-1", push=True, blocks=[0, 1])
        fill_blocks(p.kv_caches, [0, 1], 3.0)
        fill_blocks(d.kv_caches, [0, 1], 0.0)

        # D 腿：分配即注册（不等自己被调度）
        req = make_d_request("req-1", receipt)
        d.scheduler.update_state_after_alloc(
            req, KVCacheBlocksStub([[0, 1]]), 8
        )
        meta = d.build_meta()
        assert meta.push_registrations["req-1"]["decode_engine_id"] == d.engine_id
        d.worker_step(meta)

        # P 腿：终局产出待推送块
        p_req = make_p_request("req-1")
        p_req.status = sched_request_status("FINISHED_LENGTH_CAPPED")
        p_req.num_computed_tokens = 8
        p_sched = p.scheduler
        delay_free, _ = p_sched.request_finished(p_req, ([0, 1],))
        assert delay_free is True
        assert p_sched.has_pending_push_work() is True
        p_meta = p.build_meta()
        assert p_meta.push_finished_blocks["req-1"] == ([0, 1],)
        p.worker_step(p_meta)

        # writer 线程配对后 WRITE 直写 D 预分配块
        pump_until(lambda: block_checksum(d.kv_caches, 0) != 0.0)
        assert block_checksum(d.kv_caches, 0) == block_checksum(p.kv_caches, 0)
        assert block_checksum(d.kv_caches, 1) == block_checksum(p.kv_caches, 1)
    finally:
        p.close()
        d.close()


def test_push_done_recving_materialises_without_local_handle():
    p, d = pair_of_engines(connector="NixlPushConnector")
    try:
        receipt = _receipt_from_p(p, "req-1", push=True, blocks=[0, 1])
        req = make_d_request("req-1", receipt)
        d.scheduler.update_state_after_alloc(
            req, KVCacheBlocksStub([[0, 1]]), 8
        )
        d.worker_step(d.build_meta())

        p_req = make_p_request("req-1")
        p_req.status = sched_request_status("FINISHED_LENGTH_CAPPED")
        p_req.num_computed_tokens = 8
        p.scheduler.request_finished(p_req, ([0, 1],))
        p.worker_step(p.build_meta())

        done_sending = pump_until(
            lambda: (lambda r: r[0] if r[0] else None)(p.worker_step())
        )
        assert done_sending == {"req-1"}
        done_recving = pump_until(
            lambda: (lambda r: r[1] if r[1] else None)(d.worker_step())
        )
        assert done_recving == {"req-1"}  # D 无本地 handle，收 notif 即报完
    finally:
        p.close()
        d.close()


def test_push_matching_strips_random_request_suffix():
    p, d = pair_of_engines(connector="NixlPushConnector")
    try:
        w = p.worker
        w._pending_d_registrations["req-a"] = {"request_id": "req-a"}
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
            get_base_request_id,
        )

        assert get_base_request_id("req-a-1a2b3c4d") == "req-a"
        assert w._pop_matching_registration("REQ-A") is None
        w._push_finished_blocks["req-b"] = ([0],)
        assert w._pop_matching_finished_blocks("req-b") == ("req-b", ([0],))
        assert w._pop_matching_finished_blocks("req-b") is None
    finally:
        p.close()
        d.close()


def test_push_registration_watchdog_drops_stale_registrations():
    p, d = pair_of_engines(connector="NixlPushConnector")
    try:
        sched = d.scheduler
        sched._push_pending_registrations["req-stale"] = {"request_id": "req-stale"}
        sched._push_registration_deadlines["req-stale"] = time.perf_counter() - 1.0
        meta = d.build_meta()
        assert "req-stale" not in sched._push_pending_registrations
        assert "req-stale" not in meta.push_registrations
    finally:
        p.close()
        d.close()


# ═══════════════ m10 拒绝回执：D 拒收 → P 放块 ═══════════════


def test_serving_rejection_chain_releases_prefill_blocks():
    from vllm.entrypoints.generate.base.serving import ErrorResponse, GenerateBaseServing
    from vllm.v1.engine.async_llm import AsyncLLM

    p, d = pair_of_engines()
    try:
        # 1) P 腿：请求被调度（登记在批）→ 终局产出回执 → 钉住块
        p_req = make_p_request("req-1")
        p.scheduler.update_state_after_alloc(p_req, None, 0)
        p.worker_step(p.build_meta())  # reqs_in_batch → worker._reqs_to_process
        p_req.status = sched_request_status("FINISHED_LENGTH_CAPPED")
        p_req.num_computed_tokens = 8
        receipt = p.scheduler.request_finished(p_req, ([0, 1],))[1]
        p.worker_step(p.build_meta())  # reqs_to_send → 租约戳 + 心跳
        assert "req-1" in p.worker._reqs_to_send

        # 2) D serving 层拒收 → 占位请求
        async_llm = AsyncLLM(engine_core=d.core, vllm_config=d.vllm_config)
        serving = GenerateBaseServing(async_llm)
        assert serving.has_kv_connector is True  # 由 vllm_config 推出

        async def _rejected():
            # 拒收的判据：create_* 返回 ErrorResponse（请求没进引擎）
            return ErrorResponse("rejected by D")

        d_req = make_d_request("req-1", receipt)

        async def run():
            return await serving._with_kv_transfer_rejection_cleanup(
                _rejected(), d_req, None
            )

        asyncio.run(run())

        # 3) 占位请求已被立即 abort，但 request_finished 钩子跑过 → 空 recv
        sched = d.scheduler
        assert "req-1" in sched._reqs_need_recv
        assert sched._reqs_need_recv["req-1"][1] == []

        # 4) D worker 见空块列表 → notif-only → P 放块
        pump_until(lambda: d.worker._remote_agents.get(p.engine_id))
        done_sending = None
        for _ in range(6):
            d.worker_step()  # 排空 _ready_requests → 发 notif
            got = pump_until(
                lambda: (lambda r: r[0] if r and r[0] else None)(p.worker_step()),
                timeout=0.5,
                interval=0.01,
            )
            if got:
                done_sending = got
                break
        assert done_sending == {"req-1"}
    finally:
        p.close()
        d.close()


def test_rejection_cleanup_is_silent_without_remote_prefill():
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing

    serving = GenerateBaseServing(_SpyClient(), has_kv_connector=True)

    req = make_p_request("plain")  # do_remote_prefill 不为真 → 不该通知

    async def run():
        async def ok():
            return "ok"

        return await serving._with_kv_transfer_rejection_cleanup(ok(), req, None)

    assert asyncio.run(run()) == "ok"
    assert serving.engine_client.notified == []


class _SpyClient:
    def __init__(self):
        self.notified = []

    async def notify_kv_transfer_request_rejected(self, request_id, params, **kw):
        self.notified.append(request_id)


# ═══════════════ m11 失败处理：坏块上报 → 重算 ═══════════════


def test_failed_transfer_reports_invalid_blocks_and_req():
    p, d = pair_of_engines()
    try:
        receipt = _receipt_from_p(p, "req-1", blocks=[0, 1])
        req = make_d_request("req-1", receipt)
        d.scheduler.update_state_after_alloc(
            req, KVCacheBlocksStub([[0, 1]]), 8
        )
        d.worker_step()
        w = d.worker
        pump_until(lambda: "req-1" in w._recving_metadata)
        w._handle_failed_transfer("req-1", None)
        _, done_recving = w.get_finished()
        assert done_recving == {"req-1"}  # 失败请求并入 done_recving
        assert w.get_block_ids_with_load_errors() == {0, 1}
        assert w.get_block_ids_with_load_errors() == set()  # 取走即清空
    finally:
        p.close()
        d.close()


def test_build_connector_meta_clears_ledgers_and_stamps_clock():
    p, d = pair_of_engines()
    try:
        sched = d.scheduler
        receipt = _receipt_from_p(p, "req-1")
        req = make_d_request("req-1", receipt)
        sched.update_state_after_alloc(req, KVCacheBlocksStub([[0, 1]]), 8)
        t0 = time.perf_counter()
        meta = d.build_meta()
        assert meta.scheduler_clock >= t0
        assert "req-1" in meta.reqs_to_recv
        assert meta.reqs_to_recv["req-1"].remote.engine_id == p.engine_id
        assert sched._reqs_need_recv == {}  # 交给 worker 后即清账
        assert d.build_meta().reqs_to_recv == {}
    finally:
        p.close()
        d.close()


# ─────────────────────────── 测试内小工具 ───────────────────────────


def _receipt_from_p(p, request_id: str, blocks=(0, 1), push: bool = False):
    """P 终局产出的回执信封（控制面搬运的那只）。"""
    from vllm.v1.request import RequestStatus

    req = make_p_request(request_id)
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.num_computed_tokens = 8
    return p.scheduler.request_finished(
        req, (list(blocks),)
    )[1]
