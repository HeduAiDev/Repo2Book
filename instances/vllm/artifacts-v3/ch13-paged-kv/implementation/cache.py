# SOURCE: vllm/config/cache.py
# CacheConfig——切块粒度的配置位（m13）：DEFAULT_BLOCK_SIZE = 16（一个块装
# 16 个 token 的 KV）；它同时是分配、哈希（ch15）、寻址的最小粒度。
# SUBTRACTED: @config pydantic 装配与 user_specified_* 派生标记（L43 注解、
#   L52-L55）及量化/前缀哈希/水位等其余配置位（ch03/ch14/ch15/ch27 各章）。
from typing import ClassVar


# SOURCE: vllm/config/cache.py:L43 CacheConfig
class CacheConfig:
    """Configuration for the KV cache."""

    # SOURCE: vllm/config/cache.py:L47 DEFAULT_BLOCK_SIZE
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16

    # SOURCE: vllm/config/cache.py:L49-L51 block_size
    # """Size of a contiguous cache block in number of tokens.
    # Accepts None (meaning "use default"). After construction, always int."""
    # SUBTRACTED: pydantic Field(default=None, gt=0) 校验面——None→默认的
    #   解析语义在构造期完成（真实在 config 装配器里）。
    def __init__(self, block_size: int | None = None):
        # SOURCE: vllm/config/cache.py:L49-L51（None → DEFAULT_BLOCK_SIZE）
        self.block_size: int = (
            block_size if block_size is not None else self.DEFAULT_BLOCK_SIZE
        )
