"""测 NIXL RDMA connector 复现 vLLM 的可观察行为。

真实行为（vllm/.../nixl/connector.py + nixl/worker.py）：
  - facade：role=WORKER 只建 connector_worker，worker 契约方法全转发。
  - wait_for_layer_load / save_kv_layer 为 no-op（不逐层、不显式 save）。
  - start_load_kv：首遇远端先后台握手，握手完才发 RDMA READ（make_prepped_xfer
    'READ' + transfer），handle 存入 _recving_transfers。
  - get_finished 双源不对称：本地 handle 转 DONE = 收完成（_pop_done_transfers）；
    对端 read-done 通知 = 发完成（_get_new_notifs）。PROC 留着，全 DONE 才报。
"""
from implementation.base import KVConnectorRole
from implementation.nixl_connector import (
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorWorker,
    NixlReqMeta,
    _FakeNixlWrapper,
)


def _make_worker(world_size=1):
    wrapper = _FakeNixlWrapper(agent_name="D")
    worker = NixlConnectorWorker(engine_id="D-engine", nixl_wrapper=wrapper,
                                 world_size=world_size)
    return worker, wrapper


def test_facade_builds_only_worker_half():
    conn = NixlConnector(None, KVConnectorRole.WORKER,
                         connector_worker=NixlConnectorWorker("D", _FakeNixlWrapper("D")))
    assert conn.connector_worker is not None
    assert conn.connector_scheduler is None

    sched = NixlConnector(None, KVConnectorRole.SCHEDULER)
    assert sched.connector_worker is None
    assert sched.connector_scheduler is not None


def test_save_and_layer_load_are_noop():
    conn = NixlConnector(None, KVConnectorRole.WORKER,
                         connector_worker=NixlConnectorWorker("D", _FakeNixlWrapper("D")))
    conn.bind_connector_metadata(NixlConnectorMetadata())
    # 两个都是 no-op，不抛错、无副作用。
    assert conn.wait_for_layer_load("layer.0") is None
    assert conn.save_kv_layer("layer.0", None, None) is None


def test_first_contact_handshakes_then_reads():
    worker, wrapper = _make_worker()
    md = NixlConnectorMetadata()
    md.add_recv(NixlReqMeta("reqA", remote_engine_id="P-engine",
                            local_block_ids=[0, 1], remote_block_ids=[10, 11]))

    # 首遇 P-engine：握手前 _remote_agents 为空。
    assert "P-engine" not in worker._remote_agents
    worker.start_load_kv(md)

    # 握手完成（_remote_agents 记下），且 READ 已发起（handle 入 _recving_transfers）。
    assert "P-engine" in worker._remote_agents
    assert worker._recving_transfers["reqA"], "应已发起 RDMA READ"


def test_get_finished_recving_only_when_handle_done():
    worker, wrapper = _make_worker()
    md = NixlConnectorMetadata()
    md.add_recv(NixlReqMeta("reqA", "P-engine", [0], [10]))
    worker.start_load_kv(md)

    handle = worker._recving_transfers["reqA"][0]

    # handle 仍 PROC → get_finished 不报收完成（非阻塞 READ 仍在传）。
    sending, recving = worker.get_finished()
    assert "reqA" not in recving
    assert worker._recving_transfers["reqA"] == [handle]

    # 推进 handle 到 DONE → 这一步才报收完成。
    wrapper.complete(handle)
    sending, recving = worker.get_finished()
    assert recving == {"reqA"}
    assert "reqA" not in worker._recving_transfers


def test_get_finished_sending_from_peer_notif():
    """P 侧 worker：靠对端 D 的 read-done 通知判发完成（不对称完成信号）。"""
    p_wrapper = _FakeNixlWrapper(agent_name="P")
    d_wrapper = _FakeNixlWrapper(agent_name="D")
    # D 能向 P 的 agent 投递通知。
    d_wrapper.connect("P", p_wrapper)

    p_worker = NixlConnectorWorker("P-engine", p_wrapper, world_size=1)
    # P 侧登记一个待 D 来读的请求。
    p_worker._reqs_to_process.add("reqA")
    p_worker._reqs_to_send["reqA"] = float("inf")

    # 还没收到通知 → 不报发完成。
    sending, _ = p_worker.get_finished()
    assert "reqA" not in sending

    # D 读完后给 P 发通知（notif 格式 "req_id:world_size"）。
    d_wrapper.send_notif("P", notif_msg=b"reqA:1")
    sending, _ = p_worker.get_finished()
    assert sending == {"reqA"}
    # 通知到达后，P 侧把该请求从待处理集合移除（可释放块）。
    assert "reqA" not in p_worker._reqs_to_process


def test_full_prefix_hit_sends_notif_no_read():
    """local_block_ids 为空（全前缀命中）：不发 READ，只通知 P。"""
    worker, wrapper = _make_worker()
    worker.add_remote_agent("P-engine", agent_name="Pagent")
    peer = _FakeNixlWrapper("Pagent")
    wrapper.connect("Pagent", peer)

    worker._read_blocks(
        local_block_ids=[], remote_block_ids=[],
        dst_engine_id="P-engine", request_id="reqA", remote_request_id="reqA",
        remote_rank=0, local_xfer_side_handle=0, remote_xfer_side_handle=0,
    )
    # 没有发起 READ（无 handle），但向 P 发了通知。
    assert not worker._recving_transfers["reqA"]
    assert peer.get_new_notifs()  # P 收到了 read-done 通知
