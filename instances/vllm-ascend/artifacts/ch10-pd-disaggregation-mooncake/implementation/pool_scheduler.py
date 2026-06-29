# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
#   —— subtract-only companion（★ ch10 高潮：KV 亲和 cache-hit-aware 路由）
#
# 本章只借用 KVPoolScheduler 的「命中查询」做亲和路由：get_num_new_matched_tokens 经
#   client.lookup 问外部 KV 池「这条请求有多少 prompt token 已经在池里命中」，据此只分配/加载
#   缺口（need_to_allocate = 命中 - 本地已算），把请求路由到 KV 所在处、最小化跨节点搬运。
#   其 store/池存取与池调度节拍留 ch11（前向引用，这里不展开）。
#
# host 无 vllm：配置/池 RPC socket 经 runtime_stub 接住；亲和判定本身是纯 Python，可跑。
from dataclasses import dataclass

import zmq

from runtime_stub import (
    BlockHash,
    MsgpackEncoder,
    logger,
    make_zmq_socket,
)

# SUBTRACTED: import vllm.envs / VllmConfig / KVConnectorMetadata / BlockPool / KVCacheBlocks /
#   FullAttentionSpec/MambaSpec/SlidingWindowSpec/UniformTypeKVCacheSpecs / KVConnectorOutput /
#   Request / SchedulerOutput 及 config_data 的 AscendConnectorMetadata/ReqMeta/RequestTracker 等
#   （L1-L35）—— 仅服务被删的 store/池节拍/hybrid-cache 机制（留 ch11）。
#   原 vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L1-L35


@dataclass
class LoadSpec:  # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:L490
    # 真实定义在 config_data.py，pool_scheduler 从那里 import；此处随章内联便于阅读。
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

    token_len: int = 0


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L38
class KVPoolScheduler:
    # SUBTRACTED: __init__ 中 hybrid/mamba/swa/block-size/hash 粒度 推断与 store 侧状态
    #   （compress_ratios / use_hybrid / _infer_* / grouped_block_size / sending_events 等，L44-L118）
    #   —— ch10 只用命中查询；池存取/节拍机制留 ch11。保留亲和判定读到的字段。
    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L39
    def __init__(self, vllm_config, use_layerwise, kv_cache_config=None):
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        self.kv_cache_group_ids = [0]
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False
        )
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.client = LookupKeyClient(vllm_config)
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        # SUBTRACTED: cache_transfer_granularity 由 _infer_cache_transfer_granularity 算（L102）——
        #   依赖 lcm_block_size + group families（被删的 hybrid 推断）。亲和算术只需粒度这个标量。
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        # Whether to discard partial chunks
        self._discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )

    # SUBTRACTED: _infer_group_families / _infer_group_block_sizes / _get_group_block_size /
    #   _get_group_family / _uses_hybrid_kv_cache / _infer_mamba_groups / _infer_swa_blocks /
    #   get_sw_clipped_blocks（L120-L222）—— hybrid/mamba/SWA 块推断，留 ch11。
    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L150
    def _infer_cache_transfer_granularity(self) -> int:
        # SUBTRACTED: 真实体按各 group family 取 lcm(lcm_block_size, family granularities)（依赖
        #   被删的 hybrid/block-size 推断，L150-L159）；亲和算术只需粒度这个标量。原 pool_scheduler.py:L150
        return getattr(self, "lcm_block_size", 1)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L161
    def _floor_to_cache_transfer_granularity(self, token_len: int) -> int:
        return token_len // self.cache_transfer_granularity * self.cache_transfer_granularity

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L224
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        """
        Check for external KV cache hit.

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

    # SUBTRACTED: update_state_after_alloc / build_connector_meta / get_sending_event_id /
    #   touch_sending_mamba_blocks / update_connector_output / request_finished* / bind_gpu_block_pool
    #   （L295-L631）—— store/save 侧与池节拍机制，整体留 ch11（前向引用）。


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L631
class LookupKeyClient:
    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L632
    def __init__(self, vllm_config):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,
            bind=False,
        )

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L643
    def lookup(
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

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L660
    def close(self):
        self.socket.close(linger=0)


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:L664
def get_zmq_rpc_path_lookup(vllm_config) -> str:
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    # SUBTRACTED: envs.VLLM_RPC_BASE_PATH 读取与 mooncake_rpc_port 兼容告警（L665-L678）——
    #   端口解析细节；保留 ipc:// 路径成形让 socket 路径可读。原 pool_scheduler.py:L664-L679
    rpc_port = 0
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    return f"ipc:///tmp/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"
