# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py
#   —— subtract-only companion（★ 两端解耦的引擎：后台搬运线程）
#
# KVTransferThread 是基类：request_queue 单消费者 while 循环，把「谁来搬」(后台线程) 与「谁要搬」
#   (主循环 add_request) 解耦——主循环只管入队即返回，搬运在后台跑。子类实现 _handle_request：
#   ·SendingThread（save/put）：process_tokens 出 key → lookup 去重(池里已有的 chunk 跳过，跨请求
#     只存一次) → 仅 missing 块 prepare_value 取 (addr,size) → m_store.put；末尾 task_done() 解除
#     wait_for_save 的 request_queue.join() 背压屏障。
#   ·RecvingThread（async load/get）：组 key/addr/size → m_store.get → 失败块 record_failed_blocks
#     记入 _invalid_block_ids → set_finished_request（供 get_finished 上报 done_recving）。
#
# host 无 NPU/mooncake：m_store 用纯内存 Backend 替身（见 tests），队列解耦 / lookup 去重 /
#   契约调用顺序是纯 Python，可跑；实际 RDMA 搬运不真跑。
import queue
import threading
from collections import defaultdict
from typing import Any

from backend.backend import Backend
from config_data import ChunkedTokenDatabase, ReqMeta
from runtime_stub import logger


class KVTransferThread(threading.Thread):  # SOURCE: kv_transfer.py:L24
    def __init__(  # SOURCE: kv_transfer.py:L25
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.finished_requests: set[str] = set()
        # SUBTRACTED: ThreadPoolExecutor(max_workers=32)（仅 layerwise 用）+ kv_event_lock/kv_events
        #   （kv-event 旁路遥测）。原 kv_transfer.py:L45-L48

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:  # SOURCE: kv_transfer.py:L50
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(self, request: ReqMeta):  # SOURCE: kv_transfer.py:L57
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:  # SOURCE: kv_transfer.py:L63
        """Get and clear the requests that have been completed."""
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished_requests

    def set_finished_request(self, req_id):  # SOURCE: kv_transfer.py:L74
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):  # SOURCE: kv_transfer.py:L78
        """Run the thread to handle KV cache transfer requests."""
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request. This indicates queue shutdown or invalid request.")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error(
                    "Error in KVCacheTransferThread. type=%s, error=%s. Check thread state and request processing.",
                    type(e).__name__,
                    e,
                )

    def _handle_request(self, req_meta: Any):  # SOURCE: kv_transfer.py:L97
        pass

    def lookup(self, keys: list[str]) -> list[bool]:  # SOURCE: kv_transfer.py:L100
        """Check the existence of all keys from the cache engine.

        Returns a bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return [False] * len(keys)

    # SUBTRACTED: update_kv_event / get_kv_events（L122-L130）—— kv-event 旁路遥测，不参与搬运正确性。

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:  # SOURCE: kv_transfer.py:L132
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _process_tokens_with_block_ids(  # SOURCE: kv_transfer.py:L139
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
    ):
        # SUBTRACTED: 兼容旧 process_tokens(无 block_ids) 的 iter_with_legacy_process_tokens 回退分支
        #   （L149-L176）—— ChunkedTokenDatabase 已有 process_tokens_with_block_ids；走主路径。
        return self.token_database.process_tokens_with_block_ids(
            token_len,
            block_hashes,
            block_ids,
            mask_num,
            kv_cache_group_id=kv_cache_group_id,
            skip_null_blocks=skip_null_blocks,
            cache_role=cache_role,
        )

    def _prepare_value(  # SOURCE: kv_transfer.py:L178
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        return self.token_database.prepare_value(
            start,
            end,
            block_ids,
            kv_cache_group_id=kv_cache_group_id,
            cache_role=cache_role,
        )

    # SUBTRACTED: _decode_adaptor_prefill_pp（L197-L214）—— prefill PP 切分场景重写 key 的 @pp_rank。


class KVCacheStoreSendingThread(KVTransferThread):  # SOURCE: kv_transfer.py:L217
    def __init__(  # SOURCE: kv_transfer.py:L218
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        group_uses_align_state: list[bool],
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.group_uses_align_state = group_uses_align_state
        self.enable_kv_event = enable_kv_event
        # SUBTRACTED: completed_events_lock/completed_events（mamba 块异步释放回执）。原 kv_transfer.py:L239-L240

    def add_stored_request(self, req_id: str):  # SOURCE: kv_transfer.py:L242
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):  # SOURCE: kv_transfer.py:L246
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):  # SOURCE: kv_transfer.py:L251
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    # SUBTRACTED: mark_completed_events / get_completed_events（L256-L267）—— mamba 块释放事件回执。

    def _handle_request(self, req_meta: ReqMeta):  # SOURCE: kv_transfer.py:L269
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            if req_id not in self.stored_requests:
                self.request_queue.task_done()
                return

            for group_id in req_meta.kv_cache_group_ids or [0]:
                starts = []
                ends = []
                keys = []
                block_ids = req_meta.block_ids_by_group[group_id]

                for start, end, key, _ in self._process_tokens_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                ):
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_string())

                # SUBTRACTED: 按 tp_rank/put_step 对 starts/ends/keys 做 TP 分片（L303-L311）+ kv-event
                #   的 block_hashes 收集（L301）—— 多 TP rank 分摊 put 负载；单卡(tp_rank=0,put_step=1)恒等。

                if not keys:
                    continue

                # ★ 存前先 lookup 去重：池里已有的 chunk 跳过，只 put missing 块（跨请求复用的直接体现）
                exists_states = self.lookup(keys)
                missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

                if not missing_indices:
                    continue

                starts = [starts[index] for index in missing_indices]
                ends = [ends[index] for index in missing_indices]
                keys = [keys[index] for index in missing_indices]

                logger.info(
                    "Storing KV cache for %d blocks (missing_count=%d) for request %s in group %d",
                    len(keys),
                    len(missing_indices),
                    req_id,
                    group_id,
                )

                addrs = []
                sizes = []
                for index, start in enumerate(starts):
                    addr, size, _ = self._prepare_value(
                        start,
                        ends[index],
                        block_ids,
                        kv_cache_group_id=group_id,
                    )
                    addrs.append(addr)
                    sizes.append(size)
                    # SUBTRACTED: enable_kv_event 时构造 BlockStored 事件（L359-L379）—— kv-event 遥测。

                # SUBTRACTED: if self.kv_role == "kv_consumer": decode_adaptor_prefill_pp 重写 key（L381-L387）。

                if current_event is not None:
                    current_event.synchronize()
                self.m_store.put(keys, addrs, sizes)
                # SUBTRACTED: enable_kv_event 时 update_kv_event(stored_events)（L394-L395）。
        finally:
            # always free blocks
            # SUBTRACTED: self.mark_completed_events(req_meta.event_id)（L398）—— mamba 块释放回执。
            pass
        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):  # SOURCE: kv_transfer.py:L403
    def __init__(  # SOURCE: kv_transfer.py:L404
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        invalid_block_ids: set[int],
        invalid_block_ids_lock: threading.Lock,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )
        self._invalid_block_ids = invalid_block_ids
        self._invalid_block_ids_lock = invalid_block_ids_lock

    def _handle_request(self, req_meta: ReqMeta):  # SOURCE: kv_transfer.py:L421
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        addr_list = []
        size_list = []
        key_list = []
        block_id_list: list[int] = []
        group_ids = req_meta.kv_cache_group_ids or [0]
        for group_id in group_ids:
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            mask_num = (
                req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
                // group_block_size
                * group_block_size
            )
            for start, end, key, _ in self._process_tokens_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                addr, size, block_id = self._prepare_value(
                    start,
                    end,
                    block_ids,
                    kv_cache_group_id=group_id,
                )
                key_list.append(key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)
        if not key_list:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return
        # SUBTRACTED: 按 tp_rank 旋转 key/addr/size/block_id 列表（key_list_c = ...）（L459-L464）——
        #   多 TP rank 分摊 get 负载；单卡(tp_rank=0)旋转是恒等。直接用原列表。
        ret = self.m_store.get(key_list, addr_list, size_list)
        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(block_id_list, ret)
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(missing_block_ids)
        elif ret is None:
            missing_block_ids = record_failed_blocks(block_id_list, [1] * len(block_id_list))
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(missing_block_ids)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


# SUBTRACTED: class KVCacheStoreLayerSendingThread（L499-L678）+ class KVCacheStoreLayerRecvingThread
#   （L681-L741）—— layerwise 逐层流水搬运线程，与非 layerwise 整请求搬运并列的另一条模式分支。


def record_failed_blocks(  # SOURCE: kv_transfer.py:L744
    block_ids: list[int],
    ret_codes: list[int],
) -> set[int]:
    failed_blocks: set[int] = set()
    for block_id, code in zip(block_ids, ret_codes):
        if code != 0:
            failed_blocks.add(block_id)
    if failed_blocks:
        logger.error(
            "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
            len(failed_blocks),
            failed_blocks,
        )
    return failed_blocks
