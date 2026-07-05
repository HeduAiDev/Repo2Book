# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py
#   —— subtract-only companion（★「异步搬」那一端：池 worker）
#
# 运行在模型 worker 进程，持有 m_store 后端与 token_database。四件支柱事：
#   ①backend_map 动态选后端 —— 按 extra_config['backend'] importlib.import_module 出一个 Backend 子类，
#     存进 self.m_store（可插拔注入点）。
#   ②register_kv_caches —— 把本机 KV cache 显存区段注册进后端(register_buffer)，并按 role/load_async
#     起收/发后台搬运线程（两端解耦在 worker 侧的落地）。
#   ③start_load_kv / wait_for_save —— 取/存入口：load_async 时把请求丢进线程队列(主循环不阻塞)，
#     否则当场 m_store.get；wait_for_save 末尾 request_queue.join() 作背压屏障。
#   ④lookup_scheduler / get_finished —— lookup_scheduler 真正调 m_store.exists 算命中前缀(供 REP 服务端)；
#     get_finished 上报异步收/发完成，引擎据此放行 block。
#
# host 无 NPU/CANN/mooncake：分布式 rank / torch.npu.Event / 后端经 runtime_stub + FakeBackend 接住；
#   动态选后端 / 队列解耦 / 命中算术 / 契约调用顺序是纯 Python，可跑；实际显存指针搬运不真跑。
from __future__ import annotations

import importlib
import threading

from config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
)
from kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    record_failed_blocks,
)
from runtime_stub import (
    BlockHash,
    NpuEvent,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    logger,
)

# SUBTRACTED: import math（仅 lcm 算 cache_transfer_granularity 用，单 group 退化为 block_size）；
#   torch / vllm distributed pcp/dcp / FullAttentionSpec/MambaSpec/UniformTypeKVCacheSpecs / BlockStored /
#   config_data 的 LayerMultiBlockReqMeta/get_cache_family_granularity/infer_group_cache_families /
#   kv_transfer 的 KVCacheStoreLayer*Thread（L1-L45）—— 仅服务被删的 hybrid/mamba/compress/layerwise/
#   kv-event 路径。原 pool_worker.py:L1-L45

# ★ 可插拔后端注入点：后端名 → 模块路径 + 类名（worker 按 extra_config['backend'] 动态 import）。
backend_map = {  # SOURCE: pool_worker.py:L47
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}


class KVPoolWorker:  # SOURCE: pool_worker.py:L63
    # The main class for the cache engine.

    def __init__(  # SOURCE: pool_worker.py:L66
        self,
        vllm_config,
        use_layerwize: bool,
        kv_cache_config=None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.kv_cache_config = kv_cache_config
        # SUBTRACTED: hf_text_config/compress_ratios/use_compress/use_mla/use_sparse 推断（L75-L86）——
        #   DSV4 compress + MLA/sparse 模型判定；本主线走标准 dense full-attention。原 pool_worker.py:L75-L86
        self.use_compress = False
        self.use_mla = False
        self.use_sparse = False
        self.use_layerwise = use_layerwize
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size
        # SUBTRACTED: pcp/dcp（context-parallel）rank/size 推断（L93-L96）——多 CP 维并行；
        #   单卡退化 pcp=dcp 大小 1、rank 0。原 pool_worker.py:L93-L96
        self.pcp_rank = 0
        self.dcp_rank = 0
        self.dcp_size = 1

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self._invalid_block_ids: set[int] = set()
        self._invalid_block_ids_lock = threading.Lock()
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
        # SUBTRACTED: use_hybrid/use_mamba 推断 + cp_scale 缩放 + hash_block_size 整除校验 + lcm_block_size
        #   （L106-L135）—— hybrid/mamba KV 与多 CP 缩放；单 full-attention group、cp_scale=1 退化。
        self.original_block_size = [vllm_config.cache_config.block_size]
        self.grouped_block_size = list(self.original_block_size)
        self.block_size = self.grouped_block_size[0]
        self.num_kv_cache_groups = 1
        self.kv_cache_group_families = ["default"]
        self.group_uses_align_state = [False]
        # SUBTRACTED: cache_transfer_granularity = lcm(lcm_block_size, family granularities)（L124）——
        #   单 full-attention group 退化为 block_size。原 pool_worker.py:L124
        self.cache_transfer_granularity = self.block_size
        self.current_layer = 0
        self.num_layers = model_config.get_num_layers(parallel_config)

        # SUBTRACTED: use_mla 时 num_kv_head=1（L139-L142）—— MLA 单 latent head；dense 取 total kv heads。
        self.num_kv_head = model_config.get_total_num_kv_heads()
        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        # SUBTRACTED: prefill PP partition 推断（L151-L174）—— kv_consumer+consumer_is_to_put 的 prefill PP
        #   层切分；单 PP(prefill_pp_size=1) 不需要。原 pool_worker.py:L151-L174
        partitions = None

        self.metadata: list[KeyMetadata] = []
        for group_id in range(self.num_kv_cache_groups):
            group_tp_rank = self.tp_rank if self.group_uses_align_state[group_id] else self.head_or_tp_rank
            self.metadata.append(
                KeyMetadata(
                    model_config.model.rstrip("/").split("/")[-1],
                    group_tp_rank,
                    self.pcp_rank,
                    self.dcp_rank,
                    self.pp_rank,
                    group_id,
                )
            )

        self.token_database = ChunkedTokenDatabase(
            self.metadata, self.grouped_block_size, partitions, False, self.block_size
        )

        # ★ backend_map 动态选后端：按 extra_config['backend'] import 出一个 Backend 子类实例 → self.m_store
        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        backend_kwargs: dict = {}
        # SUBTRACTED: mooncake/memcache 的 lazy_init=use_compress（L204-L207）—— DSV4 compress 惰性建 store；
        #   非 compress 模型直接同步建。原 pool_worker.py:L204-L207
        self.m_store = real_backend(parallel_config, **backend_kwargs)
        # SUBTRACTED: enable_kv_events 读取（L212-L215）—— kv-event 旁路遥测，不参与搬运正确性。
        self.enable_kv_events = False

        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None

        self.finished_store_req: set[str] = set()

    # SUBTRACTED: _infer_group_families / _infer_group_block_sizes / _infer_group_uses_align_state /
    #   _infer_cache_transfer_granularity / _uses_hybrid_kv_cache / _uses_mamba_kv_cache（L222-L296）——
    #   hybrid/mamba/compress group 推断；单 full-attention group 全部退化为常量。原 pool_worker.py:L222-L296

    @staticmethod
    def _as_cache_tuple(cache_or_caches):  # SOURCE: pool_worker.py:L299
        import torch

        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _get_cache_block_metadata(self, cache):  # SOURCE: pool_worker.py:L304
        tensor_num_blocks = cache.shape[0]
        assert tensor_num_blocks % self.num_blocks == 0, (
            "The external block size must be an integer multiple of the kernel block size."
        )
        block_size_scale = tensor_num_blocks // self.num_blocks
        block_len = cache[0].numel() * cache.element_size() * block_size_scale
        block_stride = cache.stride(0) * cache.element_size() * block_size_scale
        region_len = (self.num_blocks - 1) * block_stride + block_len if self.num_blocks else 0
        return block_len, block_stride, region_len, block_size_scale

    @staticmethod
    def _get_storage_key(cache) -> int:  # SOURCE: pool_worker.py:L316
        try:
            return cache.untyped_storage().data_ptr()
        except AttributeError:
            return cache.storage().data_ptr()

    def _infer_cache_group_metadata(self, group_id: int, layer_names: list[str]):  # SOURCE: pool_worker.py:L322
        group_addrs: list[int] = []
        group_block_lens: list[int] = []
        group_block_strides: list[int] = []
        for layer_name in layer_names:
            cache_or_caches = self.kv_caches[layer_name]
            for cache in self._as_cache_tuple(cache_or_caches):
                base_addr = cache.data_ptr()
                block_len, block_stride, _, _ = self._get_cache_block_metadata(cache)
                group_addrs.append(base_addr)
                group_block_lens.append(block_len)
                group_block_strides.append(block_stride)
        self.group_kv_caches_base_addr[group_id] = group_addrs
        self.group_block_len[group_id] = group_block_lens
        self.group_block_stride[group_id] = group_block_strides
        self.group_num_layers[group_id] = len(layer_names)

    def register_kv_caches(self, kv_caches):  # SOURCE: pool_worker.py:L339
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache_tuple = self._as_cache_tuple(first_kv_cache_tuple)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        logger.info("num_blocks: %s", self.num_blocks)
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_block_stride: dict[int, list[int]] = {}
        self.kv_caches = kv_caches
        self.group_kv_cache_families: dict[int, str] = {
            group_id: "default" for group_id in range(self.num_kv_cache_groups)
        }
        self.group_num_layers: dict[int, int] = {}

        registered_regions: dict[int, tuple[int, int]] = {}
        for cache_or_caches in kv_caches.values():
            for cache in self._as_cache_tuple(cache_or_caches):
                base_addr = cache.data_ptr()
                _, _, region_len, _ = self._get_cache_block_metadata(cache)
                if not isinstance(region_len, int):
                    region_len = 0
                storage_key = self._get_storage_key(cache)
                start = base_addr
                end = base_addr + region_len
                if storage_key in registered_regions:
                    old_start, old_end = registered_regions[storage_key]
                    registered_regions[storage_key] = (min(old_start, start), max(old_end, end))
                else:
                    registered_regions[storage_key] = (start, end)

        ptrs = [start for start, _ in registered_regions.values()]
        lengths = [end - start for start, end in registered_regions.values()]

        # SUBTRACTED: if use_hybrid: 逐 group 推断 metadata（L384-L386）—— hybrid 多 group；
        #   单 full-attention group 走 group 0 一条路径。原 pool_worker.py:L384-L386
        self._infer_cache_group_metadata(0, list(kv_caches.keys()))

        self.m_store.register_buffer(ptrs, lengths)
        self.token_database.set_group_buffers(
            self.group_kv_caches_base_addr,
            self.group_block_len,
            self.group_block_stride,
            cache_role="kv",
            group_cache_families=self.group_kv_cache_families,
            group_num_layers=self.group_num_layers,
        )

        # SUBTRACTED: if use_layerwise: 起 KVCacheStoreLayer{Sending,Recving}Thread（L400-L429）——
        #   layerwise 逐层流水搬运线程；本主线走非 layerwise 整请求搬运。原 pool_worker.py:L400-L429
        # ★ 两端解耦在 worker 侧的落地：producer 起「发」线程、load_async 起「收」线程——独立后台线程，
        #   与模型前向主循环解耦。
        if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.dcp_size,
                self.put_step,
                self.kv_role,
                ready_event_sending,
                self.group_uses_align_state,
                self.enable_kv_events,
            )
            self.kv_send_thread.start()
        if self.load_async:
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreRecvingThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.dcp_size,
                ready_event,
                self._invalid_block_ids,
                self._invalid_block_ids_lock,
            )
            self.kv_recv_thread.start()
            ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):  # SOURCE: pool_worker.py:L461
        self.current_layer = 0
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  # load = 0
                continue
            request.skip_null_blocks_by_group = self.group_uses_align_state
            load_group_ids = request.kv_cache_group_ids or [0]
            token_len = request.token_len_chunk
            if (load_spec.kvpool_cached_tokens % self.cache_transfer_granularity != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = request.load_spec.kvpool_cached_tokens + 1
            else:
                token_len = request.load_spec.kvpool_cached_tokens
            request.load_spec.token_len = token_len
            # SUBTRACTED: if use_layerwise: retrieve_layer 逐层 next() 生成器（L495-L498）—— layerwise 路径。
            if self.load_async:
                # ★ 异步：只把 request 丢进 recv 线程队列(入队即返回，主循环不阻塞)；搬运在后台。
                self.kv_recv_thread.add_request(request)  # type: ignore[union-attr]
            else:
                # ★ 同步：当场组 key/addr/size 后 m_store.get（阻塞主循环）。
                addr_list = []
                size_list = []
                key_list = []
                block_id_list: list[int] = []
                for group_id in load_group_ids:
                    block_ids = request.block_ids_by_group[group_id]
                    group_block_size = self.grouped_block_size[group_id]
                    mask_num = request.load_spec.vllm_cached_tokens // group_block_size * group_block_size
                    skip_null = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
                    for start, end, key, _ in self.token_database.process_tokens_with_block_ids(
                        token_len,
                        request.block_hashes,
                        block_ids,
                        mask_num,
                        kv_cache_group_id=group_id,
                        skip_null_blocks=skip_null,
                    ):
                        addr, size, block_id = self.token_database.prepare_value(
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
                    continue
                # SUBTRACTED: 按 tp_rank 旋转 key/addr/size/block_id 列表（key_list_c = ...）（L536-L546）——
                #   多 TP rank 分摊 get 负载；单卡(tp_rank=0)旋转是恒等。直接用原列表。原 pool_worker.py:L536-L546
                ret = self.m_store.get(key_list, addr_list, size_list)
                if ret is not None and any(r != 0 for r in ret):
                    missing_block_ids = record_failed_blocks(block_id_list, ret)
                    self._invalid_block_ids.update(missing_block_ids)
                elif ret is None:
                    missing_block_ids = record_failed_blocks(block_id_list, [1] * len(block_id_list))
                    self._invalid_block_ids.update(missing_block_ids)

    # SUBTRACTED: wait_for_layer_load / save_kv_layer / retrieve_layer / store_layer（L576-L788 的 layerwise
    #   逐层取/存生成器）—— layerwise 逐层流水模式，与非 layerwise 整请求搬运并列的另一条分支。
    #   原 pool_worker.py:L576-L788

    def get_block_ids_with_load_errors(self) -> set[int]:  # SOURCE: pool_worker.py:L584
        with self._invalid_block_ids_lock:
            invalid_blocks = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_blocks

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata):  # SOURCE: pool_worker.py:L619
        current_event = None
        has_save_request = False
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            # SUBTRACTED: torch.npu.Event()（host 无 NPU）—— NpuEvent 替身记录显存写完事件。原 pool_worker.py:L626
            current_event = NpuEvent()
            current_event.record()
            break

        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.current_event = current_event
            self.kv_send_thread.add_stored_request(request.req_id)  # type: ignore[union-attr]
            self.kv_send_thread.add_request(request)  # type: ignore[union-attr]
            has_save_request = True

        if has_save_request:
            # vLLM expects wait_for_save() to make stores visible before the
            # request is reported as finished. Without this barrier a following
            # identical prompt can lookup before Mooncake put() has completed.
            self.kv_send_thread.request_queue.join()  # type: ignore[union-attr]

    def get_finished(  # SOURCE: pool_worker.py:L789
        self, finished_req_ids: set[str], meta: AscendConnectorMetadata
    ) -> tuple[set[str], set[str]]:
        done_sending = (
            self.get_and_clear_finished_requests(finished_req_ids, meta)
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()  # type: ignore[union-attr]
            if self.load_async
            else set()
        )
        return done_sending, done_recving

    def get_and_clear_finished_requests(  # SOURCE: pool_worker.py:L814
        self, finished_req_ids, meta: AscendConnectorMetadata
    ) -> set[str]:
        finished_sending = set()
        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)  # type: ignore[union-attr]
        for req_id in self.kv_send_thread.stored_requests.copy():  # type: ignore[union-attr]
            if self.kv_send_thread.stored_requests[req_id] == 0 and req_id in self.finished_store_req:  # type: ignore[union-attr]
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)  # type: ignore[union-attr]

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(req_id)  # type: ignore[union-attr]
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)  # type: ignore[union-attr]
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    # SUBTRACTED: lookup（L849-L920）—— worker 自身的 cache-hit 长度计算；与 lookup_scheduler 同构，
    #   但后者额外做跨 TP/PP 的 key 扩展(查别的 rank 命中)，是 LookupKeyServer 真正调的那个。保留
    #   lookup_scheduler。原 pool_worker.py:L849-L920
    # SUBTRACTED: _get_lookup_gate_group_ids / _is_lookup_gate_group / _get_group_num_kv_heads /
    #   get_group_tp_size（L922-L966）—— DSV4 c1/c4/c128 compress 的 lookup 门控 + 跨 TP 扩展尺寸；
    #   单 full-attention group + 单卡退化为「直接用原 group 集合、不扩展」。原 pool_worker.py:L922-L966

    def lookup_scheduler(  # SOURCE: pool_worker.py:L968
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
    ) -> int:
        """Check the existence of KV cache of the tokens from the cache engine.

        Returns an int indicating how many prefix tokens are cached.
        """
        try:
            hits = []
            kv_cache_group_ids = kv_cache_group_ids or [0]
            # SUBTRACTED: kv_cache_group_ids = self._get_lookup_gate_group_ids(...)（L983）—— compress 门控；
            #   单 full-attention group 直接用原集合。原 pool_worker.py:L983
            for group_id in kv_cache_group_ids:
                keys = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    return 0

                # SUBTRACTED: multi_tp_keys 的跨 TP(@head_or_tp_rank:i)/跨 PP(@pp_rank:i) key 扩展（L1005-L1020）
                #   —— 多 rank 命中查询；单卡(group_tp_size=1,pp_size=1)扩展集合 = keys。原 pool_worker.py:L1005-L1020
                res = self.m_store.exists(keys)
                num_block = len(keys)
                # SUBTRACTED: 按 group_tp_size*pp_size 把 res 切成 multi_tp_values 多组（L1027-L1030）——
                #   单卡只有一组。原 pool_worker.py:L1027-L1030
                multi_tp_values = [res]
                # SUBTRACTED: group_uses_align_state(mamba align) 的反向扫描分支（L1043-L1053）—— mamba 路径；
                #   走 dense 分支。原 pool_worker.py:L1043-L1053
                index = self.find_max_hit_index(multi_tp_values, num_block)
                if index == -1:
                    return 0
                else:
                    for hit_index in range(index, -1, -1):
                        if ends[hit_index] % self.cache_transfer_granularity == 0:
                            hits.append(ends[hit_index])
                            break
                    else:
                        return 0
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return 0
        return min(hits) if hits else 0

    # SUBTRACTED: check_all_layers_exists（L1088）—— layerwise 时把每层 exists 与成一个 chunk；非 layerwise 不用。

    def find_max_hit_index(self, arr, num_blocks: int):  # SOURCE: pool_worker.py:L1100
        for i in range(num_blocks):
            if any(row[i] != 1 for row in arr):
                return i - 1
        else:
            # if arr is not empty, all hits, else no hits
            return len(arr[0]) - 1 if arr else -1

    # SUBTRACTED: get_kv_events / build_connector_worker_meta（L1108-L1118）—— kv-event 遥测 +
    #   mamba 块异步释放回执。原 pool_worker.py:L1108-L1118
