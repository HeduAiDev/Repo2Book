"""测 Offloading connector 复现 vLLM 的可观察行为。

真实行为（vllm/.../offloading_connector.py + offloading/worker.py）：
  - facade：role=WORKER 只建 connector_worker。
  - wait_for_save→prepare_store_kv 只把 store job 入队（不真发），推迟到下一步
    start_kv_transfers 开头才 transfer_async（避免拖慢本步采样）。
  - get_finished 只为 load 报 finished_recving；finished_sending 恒空（store 走
    completed_jobs + jobs_to_flush 围栏，不走 finished_sending）。
"""
from implementation.base import KVConnectorRole
from implementation.offloading_connector import (
    OffloadingConnector,
    OffloadingConnectorMetadata,
    _JobEntry,
)


def _worker_connector():
    return OffloadingConnector(None, KVConnectorRole.WORKER)


def test_facade_builds_only_worker_half():
    conn = _worker_connector()
    assert conn.connector_worker is not None
    assert conn.connector_scheduler is None

    sched = OffloadingConnector(None, KVConnectorRole.SCHEDULER)
    assert sched.connector_worker is None
    assert sched.connector_scheduler is not None


def test_store_is_deferred_to_next_step():
    conn = _worker_connector()
    worker = conn.connector_worker

    # 第 N 步：wait_for_save → prepare_store_kv 只入队，不真发。
    md1 = OffloadingConnectorMetadata()
    md1.store_jobs = {1: _JobEntry(transfer_spec="store-1")}
    conn.bind_connector_metadata(md1)
    conn.wait_for_save()
    assert worker._unsubmitted_store_jobs == [(1, "store-1")]
    assert worker.worker._inflight == []  # 还没真正 transfer_async

    # 第 N+1 步：start_load_kv → start_kv_transfers 开头才提交上一步排队的 store。
    md2 = OffloadingConnectorMetadata()
    conn.bind_connector_metadata(md2)
    conn.start_load_kv(forward_context=None)
    assert worker._unsubmitted_store_jobs == []  # 已清空提交
    assert len(worker.worker._inflight) == 1     # store 这才真正发起


def test_load_emits_finished_recving():
    conn = _worker_connector()
    worker = conn.connector_worker

    md = OffloadingConnectorMetadata()
    md.load_jobs = {7: _JobEntry(transfer_spec="load-7", req_id="reqX")}
    conn.bind_connector_metadata(md)
    conn.start_load_kv(forward_context=None)

    sending, recving = conn.get_finished({"reqX"})
    assert recving == {"reqX"}   # load 完成 → finished_recving
    assert sending == set()       # store 永不走 finished_sending


def test_store_completion_goes_through_worker_meta_not_finished_sending():
    conn = _worker_connector()
    worker = conn.connector_worker

    # 入队并提交一个 store job（无 req_id）。
    md1 = OffloadingConnectorMetadata()
    md1.store_jobs = {1: _JobEntry(transfer_spec="store-1")}
    conn.bind_connector_metadata(md1)
    conn.wait_for_save()
    conn.bind_connector_metadata(OffloadingConnectorMetadata())
    conn.start_load_kv(forward_context=None)

    # get_finished：store 完成 → finished_sending 恒空，但记入 completed_jobs。
    sending, recving = conn.get_finished(set())
    assert sending == set()
    assert recving == set()

    # store 完成经 build_connector_worker_meta 上报 completed_jobs（围栏依据）。
    meta = conn.build_connector_worker_meta()
    assert meta is not None
    assert 1 in meta.completed_jobs

    # 再次取走后清空（无新完成 → None）。
    assert conn.build_connector_worker_meta() is None
