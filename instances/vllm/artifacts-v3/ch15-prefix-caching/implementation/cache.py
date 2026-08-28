# SOURCE: vllm/config/cache.py
# CacheConfig 的**前缀缓存切面**（m12/m18）：prefix_match_unit（=代码里贯穿的
# hash_block_size——前缀命中可落的最细 token 边界，只控匹配粒度不控存储
# 频率）、enable_prefix_caching（总开关，默认 True）、prefix_caching_hash_algo
# （默认 sha256）、block_size（本章消费 cache_config.block_size 作块对齐）。
# SUBTRACTED: @config pydantic 装配面与派生标记（L43 注解）；量化/mamba_*/
# replayssm/offloading/gpu_memory_utilization 等其余配置位（ch03/ch13/ch14/
# ch27/ch16 各章切面）；compute_hash/metrics_info/校验器三方法（L202-L300）。
from typing import ClassVar


# SOURCE: vllm/config/cache.py:L43 CacheConfig
class CacheConfig:
    """Configuration for the KV cache."""

    # SOURCE: vllm/config/cache.py:L47 DEFAULT_BLOCK_SIZE
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16

    # SOURCE: vllm/config/cache.py:L49-L51 block_size
    # """Size of a contiguous cache block in number of tokens.
    # Accepts None (meaning "use default"). After construction, always int."""
    # SUBTRACTED: pydantic Field 校验面——None→默认的解析语义在构造期完成
    #   （真实在 config 装配器里）。
    def __init__(
        self,
        block_size: int | None = None,
        prefix_match_unit: int | None = None,
        enable_prefix_caching: bool = True,
        prefix_caching_hash_algo: str = "sha256",
        mamba_cache_mode: str = "none",
    ):
        # SOURCE: vllm/config/cache.py:L49-L51（None → DEFAULT_BLOCK_SIZE）
        self.block_size: int = (
            block_size if block_size is not None else self.DEFAULT_BLOCK_SIZE
        )
        # SOURCE: vllm/config/cache.py:L56-L67 prefix_match_unit——前缀命中可
        #   落的最细 token 边界；= 代码里贯穿的 hash_block_size
        self.prefix_match_unit: int | None = prefix_match_unit
        # SOURCE: vllm/config/cache.py:L93-L94 enable_prefix_caching（默认 True）
        self.enable_prefix_caching: bool = enable_prefix_caching
        # SOURCE: vllm/config/cache.py:L95 prefix_caching_hash_algo（默认
        #   sha256；cbor/xxhash 变体的取舍说明见 docstring L96-L110——精简版
        #   只实现 sha256 支）
        self.prefix_caching_hash_algo: str = prefix_caching_hash_algo
        # SOURCE: vllm/config/cache.py:L139-L147 mamba_cache_mode（语义账位：
        #   none=关缓存时 / all=每块边界全缓 / align=只缓调度步边界——
        #   scheduler 的 mamba 停点判定读它；mamba 状态管理深讲 → 邻章）
        self.mamba_cache_mode: str = mamba_cache_mode
