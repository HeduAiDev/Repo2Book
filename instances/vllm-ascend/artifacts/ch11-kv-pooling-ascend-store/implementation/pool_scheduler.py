# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
#   —— subtract-only companion（★「搬什么」那一端：池调度器节拍）
#
# 运行在调度进程，无 KV 显存、无后端连接。三步节拍：
#   ①get_num_new_matched_tokens —— 经 LookupKeyClient.lookup（zmq REQ→worker 的 REP）问外部池
#     「本请求多少前缀 token 已在池中」，算 need_to_allocate = 命中 − 本地已算，登记 LoadSpec(can_load=False)。
#   ②update_state_after_alloc —— vLLM 分配好临时 block 后回填 can_load=True，并校验缺口一致。
#   ③build_connector_meta —— 每调度步把要 load(带 load_spec)/save(can_save) 的请求各打包成 ReqMeta，
#     聚成 AscendConnectorMetadata，随 SchedulerOutput 下发给 worker。
#
# host 无 vllm：配置/池 RPC socket 经 runtime_stub 接住；命中算术 + LoadSpec 节拍是纯 Python，可跑。
from typing import Any

import zmq

from config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    normalize_block_ids_by_group,
)
from runtime_stub import BlockHash, KVConnectorMetadata, MsgpackEncoder, logger, make_zmq_socket

# SUBTRACTED: import vllm.envs / FullAttentionSpec/MambaSpec/SlidingWindowSpec/UniformTypeKVCacheSpecs /
#   BlockPool / KVCacheBlocks / SchedulerOutput / Request / KVConnectorOutput 及 config_data 的
#   AscendStoreKVConnectorWorkerMetadata / get_cache_family_granularity / infer_group_cache_families
#   （L1-L35）—— 仅服务被删的 hybrid-cache / compress / mamba 块释放机制。原 pool_scheduler.py:L1-L35


class KVPoolScheduler:  # SOURCE: pool_scheduler.py:L38
    def __init__(self, vllm_config, use_layerwise, kv_cache_config=None):  # SOURCE: pool_scheduler.py:L39
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        # SUBTRACTED: hybrid/mamba/swa/compress 推断 + grouped_block_size/lcm/hash 粒度全套
        #   （compress_ratios/use_hybrid/_infer_*/num_swa_blocks/sending_events 等，L47-L118,L120-L222）——
        #   本主线是单 group(group_id=0) 标准 full-attention 池化路径。原 pool_scheduler.py:L47-L118
        self.kv_cache_group_ids = [0]
        self.kv_cache_group_families = ["default"]
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False
        )
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.client = LookupKeyClient(vllm_config)
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.original_block_size = [vllm_config.cache_config.block_size]
        # SUBTRACTED: cache_transfer_granularity = lcm(lcm_block_size, family granularities)（L102,L150-L159）——
        #   单 full-attention group 时退化为 block_size。原 pool_scheduler.py:L102
        self.cache_transfer_granularity = self.original_block_size[0]
        # request_id -> RequestTracker
        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        self._discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )
        self._unfinished_requests: dict[str, tuple[Any, list[list[int]]]] = {}
        self._unfinished_request_ids: set[str] = set()
        self._block_pool = None

    def _floor_to_cache_transfer_granularity(self, token_len: int) -> int:  # SOURCE: pool_scheduler.py:L161
        return token_len // self.cache_transfer_granularity * self.cache_transfer_granularity

    def get_num_new_matched_tokens(  # SOURCE: pool_scheduler.py:L224
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Check for external KV cache hit.

        Returns the number of tokens that can be loaded from the
        external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
            return 0, False

        if self._discard_partial_chunks:
            token_len = self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
        else:
            token_len = len(request.prompt_token_ids)

        if token_len < self.cache_transfer_granularity:
            return 0, False

        num_external_hit_tokens = self.client.lookup(
            token_len,
            request.block_hashes,
            self.kv_cache_group_ids,
        )

        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.debug(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        return need_to_allocate, self.load_async and not self.use_layerwise

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):  # SOURCE: pool_scheduler.py:L295
        """Update KVConnector state after temporary buffer alloc."""
        local_block_ids: list[list[int]] = [[] for _ in self.kv_cache_group_ids]
        if num_external_tokens > 0:
            local_block_ids = normalize_block_ids_by_group(blocks.get_block_ids())

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:  # SOURCE: pool_scheduler.py:L350
        """Attach the connector metadata to the request object.

        Packs this step's load/save requests into AscendConnectorMetadata
        and resets the connector state.
        """
        force_skip_save = self.kv_role == "kv_consumer" and not self.consumer_is_to_put

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._preempted_req_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)

        meta = AscendConnectorMetadata(self._unfinished_request_ids, scheduler_output.preempted_req_ids)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
            request_tuple = self._unfinished_requests.get(request.req_id)
            request_real = request_tuple[0]  # type: ignore[index]
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids_by_group=normalize_block_ids_by_group(request.block_ids),
                num_saved_tokens=0,
                token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
            )
            self._request_trackers[request.req_id] = request_tracker
            last_chunk_tokens_num = (
                self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
                if self._discard_partial_chunks
                else len(request.prompt_token_ids)
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self.cache_transfer_granularity,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                discard_partial_chunks=self._discard_partial_chunks,
                original_block_size=self.original_block_size,
                kv_cache_group_families=self.kv_cache_group_families,
            )
            if req_meta is not None:
                # SUBTRACTED: self.touch_sending_mamba_blocks(req_meta)（L417）—— mamba 块引用保活，hybrid 路径。
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                # resumed request
                new_block_ids = cached_reqs.new_block_ids[i]
                if not new_block_ids:
                    continue
                if req_id in self._preempted_req_ids:
                    self._preempted_req_ids.discard(req_id)
                    load_spec = self.load_specs.pop(req_id, None)
                    request_tuple = self._unfinished_requests.get(req_id)
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids_by_group=normalize_block_ids_by_group(new_block_ids),
                        num_saved_tokens=0,
                        token_ids=request_real.prompt_token_ids[:num_tokens_to_compute].copy(),
                    )
                    self._request_trackers[req_id] = request_tracker
                    last_chunk_tokens_num = (
                        self._floor_to_cache_transfer_granularity(len(request_real.prompt_token_ids))
                        if self._discard_partial_chunks
                        else len(request_real.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self.cache_transfer_granularity,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request_real.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                        kv_cache_group_families=self.kv_cache_group_families,
                    )

                # decode/chunked request
                else:
                    request_tracker = self._request_trackers[req_id]
                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        request = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = request.all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens]
                        request_tracker.token_len += len(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                        )
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    if num_computed_token >= len(request.prompt_token_ids):
                        continue
                    request_tracker.update(new_block_ids)

                    last_chunk_tokens_num = (
                        self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
                        if self._discard_partial_chunks
                        else len(request.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self.cache_transfer_granularity,
                        load_spec=None,
                        skip_save=force_skip_save,
                        block_hashes=request.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                        kv_cache_group_families=self.kv_cache_group_families,
                    )
                if req_meta is not None:
                    # SUBTRACTED: self.touch_sending_mamba_blocks(req_meta)（L496）—— mamba 块引用保活。
                    meta.add_request(req_meta)

        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (request, block_ids) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                if not load_spec:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                if (num_tokens_to_compute % self.cache_transfer_granularity != 0) and (
                    num_tokens_to_compute == len(request.prompt_token_ids) - 1
                ):
                    num_tokens_to_compute = num_tokens_to_compute + 1
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids_by_group=block_ids,
                    num_saved_tokens=0,
                )

                self._request_trackers[request_id] = request_tracker
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self.cache_transfer_granularity,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=request.block_hashes,
                    discard_partial_chunks=self._discard_partial_chunks,
                    kv_cache_group_families=self.kv_cache_group_families,
                )
                if req_meta is not None:
                    # SUBTRACTED: self.touch_sending_mamba_blocks(req_meta)（L528）—— mamba 块引用保活。
                    meta.add_request(req_meta)
        return meta

    # SUBTRACTED: get_sending_event_id / touch_sending_mamba_blocks / update_connector_output /
    #   request_finished / request_finished_all_groups / get_sw_clipped_blocks（L211-L222,L532-L625）——
    #   mamba 块异步释放回执、HMA 多 group finish、SWA 块裁剪。原 pool_scheduler.py:L532-L625

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:  # SOURCE: pool_scheduler.py:L627
        self._block_pool = gpu_block_pool


class LookupKeyClient:  # SOURCE: pool_scheduler.py:L631
    def __init__(self, vllm_config):  # SOURCE: pool_scheduler.py:L632
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(  # SOURCE: pool_scheduler.py:L643
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
    ) -> int:
        kv_cache_group_ids = kv_cache_group_ids or [0]
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        kv_group_frames = self.encoder.encode(kv_cache_group_ids)
        token_len_bytes = token_len.to_bytes(4, byteorder="big")
        all_frames = [token_len_bytes] + list(kv_group_frames) + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):  # SOURCE: pool_scheduler.py:L660
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config) -> str:  # SOURCE: pool_scheduler.py:L664
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    # SUBTRACTED: envs.VLLM_RPC_BASE_PATH 读取 + mooncake_rpc_port 弃用兼容告警（L666-L678）——
    #   端口解析细节；保留 ipc:// 路径成形（client/server 必须一致）。原 pool_scheduler.py:L664-L679
    rpc_port = 0
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    return f"ipc:///tmp/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"
