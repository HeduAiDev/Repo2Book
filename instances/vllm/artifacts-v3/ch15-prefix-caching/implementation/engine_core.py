# SOURCE: vllm/v1/engine/core.py
# **装配开关切面**（第 2 站）：enable_prefix_caching（默认 True）或装了
# connector 才建 request_block_hasher 并 init_none_hash 播种（随机 32 字节或
# PYTHONHASHSEED 派生）——关缓存则 _block_hasher=None、请求侧从不算哈希。
# ENGINE SEAM：EngineCore.__init__ 的 L220-L229 段抽出为函数以便单测
# （控制流逐字）。
# SUBTRACTED: EngineCore 的其余装配面（executor/connector/batch queue/GC
#   冻结等 L190-L247——ch03/ch05/ch09 各章切面）。
from typing import Callable

from .cache import CacheConfig
from .hashing import get_hash_fn_by_name
from .kv_cache_utils import BlockHash, get_request_block_hasher, init_none_hash
from .request import Request


# SOURCE: vllm/v1/engine/core.py:L220-L229 装配开关（ENGINE SEAM 抽出）
def assemble_block_hasher(
    cache_config: CacheConfig,
    hash_block_size: int,
    kv_connector: object | None = None,
) -> Callable[[Request], list[BlockHash]] | None:
    # SOURCE: vllm/v1/engine/core.py:L220 hasher 账位（默认 None）
    request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
    # SOURCE: vllm/v1/engine/core.py:L221-L229（开缓存或有 connector 才建：
    #   取哈希函数 → init_none_hash 播种 → 建请求侧增量 hasher）
    if cache_config.enable_prefix_caching or kv_connector is not None:
        caching_hash_fn = get_hash_fn_by_name(
            cache_config.prefix_caching_hash_algo
        )
        init_none_hash(caching_hash_fn)

        request_block_hasher = get_request_block_hasher(
            hash_block_size, caching_hash_fn
        )
    return request_block_hasher
