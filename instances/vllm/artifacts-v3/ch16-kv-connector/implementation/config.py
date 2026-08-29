# SOURCE: vllm/config/vllm.py（+ model.py/parallel.py 消费面折入）
# VllmConfig——本章双面契约的公共入参：factory 读 kv_transfer_config 与
# scheduler_config.disable_hybrid_kv_cache_manager；base.__init__ 读
# kv_transfer_config；调度器读 max_concurrent_batches（defer_block_free 的
# 多在途批判定）与 cache_config（block_size/enable_prefix_caching）。
# SUBTRACTED: pydantic @config 装配链与 device/验证面（ch03 主场）；
#   speculative/ec_transfer 等其余子配置——账位以 None 直供。
from dataclasses import dataclass, field

from .cache import CacheConfig
from .kv_transfer import KVTransferConfig
from .scheduler_config import SchedulerConfig


# SOURCE: vllm/config/model.py:L717 ModelConfig（消费面折入：max_model_len）
@dataclass
class ModelConfig:
    # SOURCE: vllm/config/model.py:L717（max_model_len 账位）
    max_model_len: int


# SOURCE: vllm/config/parallel.py ParallelConfig（消费面折入：PP 尺寸——
#   worker 侧装配的 PP 广播分支用）
@dataclass
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py pipeline_parallel_size
    pipeline_parallel_size: int = 1


# SOURCE: vllm/config/vllm.py:L~200 VllmConfig（装配面切面）
@dataclass
class VllmConfig:
    """The global config（本章消费面：connector 双方装配的公共入参）。"""

    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # SOURCE: vllm/config/vllm.py kv_transfer_config 账位（None=无 connector）
    kv_transfer_config: KVTransferConfig | None = None
    # SUBTRACTED: 其余子配置装配（device/model_runner/compilation/spec/
    #   ec_transfer 等——各邻章）。compilation_config 以切面直供属性
    #   （static_forward_context——worker 逐层钩子从这取层实例）。

    # SOURCE: vllm/config/vllm.py max_concurrent_batches（默认 1；异步
    #   调度/PP > 1 时 > 1——defer_block_free 的判定输入）
    max_concurrent_batches: int = 1
    is_encoder_decoder: bool = False

    # SOURCE: vllm/config/vllm.py static_forward_context 账位
    #   （compilation_config.static_forward_context 的折入——层名→层实例）
    static_forward_context: dict = field(default_factory=dict)

    # SUBTRACTED: 其余属性面（trace/计算图哈希——ch03/19）。
