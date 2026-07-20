"""测 P2P NCCL connector 复现 vLLM 的可观察行为。

真实行为（vllm/.../p2p/p2p_nccl_connector.py + p2p_nccl_engine.py）：
  - producer 在 save_kv_layer 里 extract_kv_from_layer + send_tensor 发；
    consumer 在 start_load_kv 里逐层 recv_tensor + inject_kv_into_layer 写回。
  - 对端地址编码在 request_id 里（parse_request_id）。
  - send_type=PUT_ASYNC：send_tensor 入队后台线程发，wait_for_sent 等队空。
  - FlashAttention layout（layer.shape[0]==2）沿第二维按 block_ids 切片/写回。
"""
import numpy as np

from implementation import runtime
from implementation.base import KVConnectorRole
from implementation.p2p_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
    P2pNcclEngine,
)


class _Layer:
    """模拟 attention 层对象：.kv_cache 是 (2, num_blocks, ...) 的 paged buffer。"""

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache


class _Config:
    class compilation_config:
        # P2P get_finished 用它数模型层数判传完没。
        static_forward_context = {"layer.0": object()}


def _make_engines():
    """两个 loopback 引擎：prefill(producer) ↔ decode(consumer)，互相 connect。"""
    p_engine = P2pNcclEngine(send_type="PUT_ASYNC")
    d_engine = P2pNcclEngine(send_type="PUT_ASYNC")
    p_engine.connect("127.0.0.1:5000", d_engine)  # producer 发往 decode 地址
    return p_engine, d_engine


def test_parse_request_id_decodes_peer_address():
    rid = "req1___prefill_addr_127.0.0.1:5000___decode_addr_127.0.0.1:5000"
    ip, port = P2pNcclConnector.parse_request_id(rid, is_prefill=True)
    assert (ip, port) == ("127.0.0.1", 5000)
    ip, port = P2pNcclConnector.parse_request_id(rid, is_prefill=False)
    assert (ip, port) == ("127.0.0.1", 5000)


def test_producer_save_consumer_load_roundtrip():
    p_engine, d_engine = _make_engines()
    rid = "req1___prefill_addr_127.0.0.1:5000___decode_addr_127.0.0.1:5000"
    block_ids = [0, 1]

    # Producer 侧 KV buffer：(2, num_blocks=4, head=3) FlashAttention layout。
    src = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(np.float32)
    # 把 block 0,1 的 KV 切出来发。
    expected = src[:, block_ids, ...].copy()

    producer = P2pNcclConnector(None, KVConnectorRole.WORKER, is_producer=True,
                                engine=p_engine)
    pmeta = P2pNcclConnectorMetadata()
    pmeta.add_request(rid, block_ids)
    producer.bind_connector_metadata(pmeta)
    producer.save_kv_layer("layer.0", src, attn_metadata=object())
    producer.wait_for_save()  # 等 PUT_ASYNC 后台线程把 send_queue 排空

    # Consumer 侧 KV buffer：(2, num_blocks=4, head=3)，待写回。
    dst = np.zeros((2, 4, 3), dtype=np.float32)
    consumer = P2pNcclConnector(None, KVConnectorRole.WORKER, is_producer=False,
                                engine=d_engine)
    cmeta = P2pNcclConnectorMetadata()
    cmeta.add_request(rid, block_ids)
    consumer.bind_connector_metadata(cmeta)

    fwd = runtime.ForwardContext(
        attn_metadata=object(),
        no_compile_layers={"layer.0": _Layer(dst)},
    )
    consumer.start_load_kv(fwd)

    # 收到的 KV 被写回 consumer 的 block 0,1。
    np.testing.assert_array_equal(dst[:, block_ids, ...], expected)
    # 未涉及的 block 仍为 0。
    np.testing.assert_array_equal(dst[:, 2:, ...], np.zeros((2, 2, 3)))


def test_get_finished_reports_after_send_and_recv():
    p_engine, d_engine = _make_engines()
    rid = "req1___prefill_addr_127.0.0.1:5000___decode_addr_127.0.0.1:5000"
    block_ids = [0]
    cfg = _Config()

    producer = P2pNcclConnector(cfg, KVConnectorRole.WORKER, is_producer=True,
                                engine=p_engine)
    pmeta = P2pNcclConnectorMetadata()
    pmeta.add_request(rid, block_ids)
    producer.bind_connector_metadata(pmeta)

    src = np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(np.float32)
    producer.save_kv_layer("layer.0", src, attn_metadata=object())
    producer.wait_for_save()

    # producer 发完一层（== 模型层数）→ get_finished 报 finished_sending。
    sending, recving = producer.get_finished({rid})
    assert sending == {rid}

    # consumer 收完一层 → 报 finished_recving。
    consumer = P2pNcclConnector(cfg, KVConnectorRole.WORKER, is_producer=False,
                                engine=d_engine)
    cmeta = P2pNcclConnectorMetadata()
    cmeta.add_request(rid, block_ids)
    consumer.bind_connector_metadata(cmeta)
    dst = np.zeros((2, 2, 1), dtype=np.float32)
    fwd = runtime.ForwardContext(
        attn_metadata=object(),
        no_compile_layers={"layer.0": _Layer(dst)},
    )
    consumer.start_load_kv(fwd)
    sending, recving = consumer.get_finished({rid})
    assert recving == {rid}


def test_consumer_only_loads_producer_only_saves():
    """is_producer 开关：producer 不 load、consumer 不 save。"""
    p_engine, d_engine = _make_engines()
    consumer = P2pNcclConnector(None, KVConnectorRole.WORKER, is_producer=False,
                                engine=d_engine)
    # consumer.save_kv_layer 直接返回（不是 producer）→ 引擎 send_queue 不增长。
    consumer.bind_connector_metadata(P2pNcclConnectorMetadata())
    consumer.save_kv_layer("layer.0", np.zeros((2, 1, 1)), attn_metadata=object())
    assert len(d_engine.send_queue) == 0

    producer = P2pNcclConnector(None, KVConnectorRole.WORKER, is_producer=True,
                                engine=p_engine)
    # producer.start_load_kv 直接返回（不 load）。
    producer.bind_connector_metadata(P2pNcclConnectorMetadata())
    fwd = runtime.ForwardContext(attn_metadata=object(), no_compile_layers={})
    producer.start_load_kv(fwd)  # 不抛错、什么也不做
