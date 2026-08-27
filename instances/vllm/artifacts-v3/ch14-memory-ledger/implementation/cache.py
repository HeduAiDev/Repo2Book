# SOURCE: vllm/config/cache.py
# CacheConfig——账本的配置面（m16/m4/m8/m9）：gpu_memory_utilization（0.92
# 默认、per-instance 语义）、num_gpu_blocks_override（人工凌驾+账本折算）、
# prefix_match_unit（哈希粒度覆盖）、num_gpu_blocks/kv_cache_size_tokens/
# kv_cache_max_concurrency（启动后写回、前端可见的账）。
# SUBTRACTED: @config pydantic 装配与派生标记（L43 注解、L52-L55）；量化/
#   前缀哈希算法/滑窗镜像/offloading/replayssm 等其余配置位（L76-L159、
#   L174-L236——ch03/ch15/ch27 各章切面）；mamba_* 三位（L119-L147——
#   mamba 状态管理 → 邻章，本章只在 resolve 的回退判定读 mamba_cache_mode
#   语义账位）；kv_offloading_size（L191-L194 → ch16）。
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
        gpu_memory_utilization: float = 0.92,
        prefix_match_unit: int | None = None,
        num_gpu_blocks_override: int | None = None,
        enable_prefix_caching: bool = True,
        kv_cache_memory_bytes: int | None = None,
    ):
        # SOURCE: vllm/config/cache.py:L49-L51（None → DEFAULT_BLOCK_SIZE）
        self.block_size: int = (
            block_size if block_size is not None else self.DEFAULT_BLOCK_SIZE
        )
        # SOURCE: vllm/config/cache.py:L56-L67 prefix_match_unit
        self.prefix_match_unit = prefix_match_unit
        # """The finest token boundary (in tokens) a prefix-cache hit can land
        # on. ... This equals to the `hash_block_size` used throughout the KV
        # cache code."""
        # SOURCE: vllm/config/cache.py:L68-L75 gpu_memory_utilization
        self.gpu_memory_utilization = gpu_memory_utilization
        # """The fraction of GPU memory to be used for the model executor...
        # This is a per-instance limit, and only applies to the current vLLM
        # instance. It does not matter if you have another vLLM instance
        # running on the same GPU."""
        # SOURCE: vllm/config/cache.py:L87-L89 num_gpu_blocks_override
        self.num_gpu_blocks_override = num_gpu_blocks_override
        # """Number of GPU blocks to use. This overrides the profiled
        # `num_gpu_blocks` if specified. Does nothing if `None`. Used for
        # testing preemption."""
        # SOURCE: vllm/config/cache.py:L93-L94 enable_prefix_caching
        # （本章只作为布尔出现——哈希机制 → ch15）
        self.enable_prefix_caching = enable_prefix_caching
        # SOURCE: vllm/config/cache.py:L161-L162 num_gpu_blocks（启动后写回）
        self.num_gpu_blocks: int | None = None
        # """The number of blocks to allocate for GPU memory."""
        # SOURCE: vllm/config/cache.py:L139-L147 mamba_cache_mode（账位：
        #   MambaSpec.max_memory_usage_bytes 三分支的输入；mamba 状态管理
        #   → 邻章，本章只消费 "none" 默认支）
        self.mamba_cache_mode: str = "none"
        # SOURCE: vllm/config/cache.py:L182-L189 kv_cache_memory_bytes（字段
        #   账位：手动凌驾旁路的输入，dossier.delete 第 2 条删其消费分支）
        self.kv_cache_memory_bytes: int | None = kv_cache_memory_bytes
        # Set after KV cache initialization.
        # SOURCE: vllm/config/cache.py:L167-L170 kv_cache_size_tokens
        self.kv_cache_size_tokens: int | None = None
        # """Per-DP-engine KV cache capacity in tokens (group-aware). Uses
        # group-aware capacity since num_gpu_blocks * block_size can be wrong
        # for hybrid models where requests occupy multiple KV cache groups."""
        # SOURCE: vllm/config/cache.py:L171-L172 kv_cache_max_concurrency
        self.kv_cache_max_concurrency: float | None = None
        # """Per-DP-engine maximum concurrency at max_model_len tokens."""
