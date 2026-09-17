# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MooncakeStore 的 worker 侧（本章消费面：『I/O 全押 get_finished』的对照
# 契约 + LookupKeyClient 管理通道）。
#
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1-L1986
# SUBTRACTED（减法计划删除项 6：Mooncake 实现体不进精简版——mooncake 传输
#   线程 KVCacheStoreSendingThread L450 / KVCacheStoreRecvingThread L928、
#   MooncakeDistributedStore 句柄与 master 协调、token 数据库、LookupKeyServer）：
#   保留 MooncakeStoreWorker 的三方法对照面（双 no-op + get_finished 的
#   compute-I/O overlap 契约，docstring 原话）与 LookupKeyClient 的
#   non_block=None『稍后再问』语义（调度器查询通道，host 可跑）。
import socket
from concurrent.futures import Future, ThreadPoolExecutor

import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (
    LOOKUP_MSG,
    RESET_MSG,
    RESP_OK,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:MooncakeStoreWorker 类头
class MooncakeStoreWorker:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:MooncakeStoreWorker 类头
    """Worker-side logic for MooncakeStoreConnector（骨架）。

    双传输线程与 MooncakeDistributedStore 句柄未进精简版；三方法对照面保留。
    """

    # SUBTRACTED: __init__（store 句柄/线程启动/token DB/LookupServer）——
    #   删除项 6；签名保留、构建即明示未实现（对照面方法见下）。
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:MooncakeStoreWorker.__init__
    def __init__(self, vllm_config, kv_cache_config):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:MooncakeStoreWorker.__init__
        raise NotImplementedError(
            "MooncakeStoreWorker requires the MooncakeDistributedStore handle "
            "and transfer threads (not part of this reduced build); see "
            "store/worker.py:L80-L1536."
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1538-L1543
    def start_load_kv(
        self,
        metadata,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1538-L1543
        """No-op: loads are issued in get_finished() for overlap."""
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1545-L1550
    def wait_for_save(
        self,
        metadata,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1545-L1550
        """No-op: stores are issued in get_finished() for overlap."""
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1552-L1611
    def get_finished(
        self,
        finished_req_ids: set[str],
        meta,
    ) -> tuple[set[str], set[str]]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1552-L1611
        """Issue all I/O and get completed send/recv request IDs.

        All load and store I/O requests are issued here (after model
        compute is launched on the compute stream) for better
        compute-I/O overlap.
        """
        # SUBTRACTED: load 发收（recv_request_queue.put）与 store 的 CUDA
        #   事件同步下发（L1563-L1591）——双传输线程实现体按删除项 6 未进
        #   精简版；契约与时机（compute 后重叠发 I/O）以 docstring 为准。
        raise NotImplementedError(
            "MooncakeStoreWorker.get_finished I/O issue path requires the "
            "MooncakeDistributedStore handle and transfer threads (not part "
            "of this reduced build); see store/worker.py:L1552-L1611."
        )


# ============================================================
# Lookup Key Client
# ============================================================


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1888-L1970
class LookupKeyClient:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1888-L1894
    """ZMQ client for the LookupKey admin channel.

    Routes both prefix-cache lookups and admin commands (currently:
    ``reset``) to ``LookupKeyServer`` on worker rank 0. The first frame
    of every request is a named tag from ``protocol.py``.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1896-L1910
    def __init__(self, vllm_config: VllmConfig):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1896-L1910
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

        # Async lookup support
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MooncakeLookupClient"
        )
        self.futures: dict[str, Future[int]] = {}

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1912-L1922
    def _lookup(self, num_tokens: int, block_hashes: list[BlockHash]) -> int:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1912-L1922
        hash_len = len(block_hashes[0]) if block_hashes else 0
        all_frames = (
            LOOKUP_MSG,
            num_tokens.to_bytes(4, byteorder="big"),
            hash_len.to_bytes(2, byteorder="big"),
            b"".join(block_hashes),
        )
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        return int.from_bytes(resp, "big")

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1924-L1945
    def lookup(
        self,
        req_id: str,
        num_tokens: int,
        block_hashes: list[BlockHash],
        non_block: bool = False,
    ) -> int | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1924-L1945
        """If non_block is True, will return None until the result is ready,
        so the caller retries on a later step."""
        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(self._lookup, num_tokens, list(block_hashes))
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as e:
            logger.error("Async Mooncake lookup failed for %s: %s", req_id, e)
            return 0
        finally:
            del self.futures[req_id]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1947-L1951
    def discard(self, req_id: str) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1947-L1951
        """Drop any cached/in-flight lookup for ``req_id`` (e.g. on abort)."""
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1953-L1963
    def _reset(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1953-L1963
        """Trigger ``store.remove_all(force=True)`` on worker rank 0.

        Ordering assumption: caller MUST ensure no in-flight Mooncake
        lookups or transfers when invoking reset. In RL workflows this
        holds naturally at the step boundary after weight updates and
        rollout drain. Returns True on ACK, False on NACK.
        """
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1965-L1966
    def reset(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1965-L1966
        return self.executor.submit(self._reset).result()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1968-L1970
    def close(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1968-L1970
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.socket.close(linger=0)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1973-L1986
def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:L1973-L1986
    """Construct IPC path for ZMQ lookup socket."""
    assert vllm_config.kv_transfer_config is not None
    # SUBTRACTED: get_mooncake_dp_engine_index 的 local_engines_only 分支
    #   （mooncake_utils:L20-L26）——本章 DP=1 主线，直接取 data_parallel_index。
    dp_rank = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    hostname = socket.gethostname()
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return (
        f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_host_{hostname}_dp_rank{dp_rank}"
    )
