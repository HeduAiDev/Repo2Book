# SOURCE: vllm/config/cache.py
# 本章消费面：CacheConfig 的 block_size（ExampleConnector 的 slot 寻址
# 粒度）与 enable_prefix_caching（本地前缀缓存的开关——双查形态选择）。
# SUBTRACTED: 其余显存/量化/哈希算法配置面（ch03/14 各章切面）。
from dataclasses import dataclass


# SOURCE: vllm/config/cache.py:L~40 CacheConfig（切面直供——pydantic 装配
#   链归 ch03）
@dataclass
class CacheConfig:
    """Configuration for the KV cache（本章消费面）。"""

    # SOURCE: vllm/config/cache.py:L49 block_size
    block_size: int = 16
    # SOURCE: vllm/config/cache.py:L93 enable_prefix_caching（默认 True）
    enable_prefix_caching: bool = True
