# SOURCE: vllm/config/vllm.py（+ model.py/parallel.py/compilation.py 消费面折入）
# VllmConfig——账本函数群的公共入参（get_num_blocks/get_kv_cache_groups/
# max_memory_usage_bytes 全拿它读 model_config.max_model_len 与 cache_config）。
# 折入：ModelConfig（max_model_len + original_max_model_len——auto-fit 的
# -1 哨兵账位，config/model.py:L717）、ParallelConfig（PP 尺寸 + DCP 乘子，
# dossier.delete 第 8 条单卡烘干恒 1）、CUDAGraphMode（cudagraph 门位，
# config/compilation.py:L53-L63）、compilation_config 账位（切面直供）。
# SUBTRACTED: pydantic @config 装配链与 device/验证面（ch03 的主场）；
#   speculative_config/ec_transfer_config 等其余子配置（各邻章）——账位
#   以 None 直供。
import enum
from dataclasses import dataclass, field

from .cache import CacheConfig
from .scheduler_config import SchedulerConfig


# SOURCE: vllm/config/compilation.py:L53 CUDAGraphMode
class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    # SOURCE: vllm/config/compilation.py:L59-L63
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)


# SOURCE: vllm/config/model.py:L717 original_max_model_len 账位（ModelConfig
#   消费面：max_model_len 可被 auto-fit 原位改小，original 记哨兵 -1）
@dataclass
class ModelConfig:
    """Model config (本章消费面：max_model_len / original_max_model_len)。"""

    # SOURCE: vllm/config/model.py:L717（构造期 original = max 的镜像）
    max_model_len: int
    original_max_model_len: int

    # SOURCE: vllm/config/model.py:L59 ModelConfig.__init__ 的构造面（切面
    #   dataclass 直供；hf_config/量化/多模态装配 → ch02/03）
    # SOURCE: vllm/config/model.py:L717（original = max 的镜像已由字段直供）
    def __post_init__(self):
        # SUBTRACTED: hf_config 解析与 max_model_len 推导链（ch02/03）——
        #   切面直供数值。
        pass


# SOURCE: vllm/config/parallel.py ParallelConfig（消费面折入：PP 尺寸 +
#   decode_context_parallel_size 乘子——dossier.delete 第 8 条单卡烘干恒 1）
@dataclass
class ParallelConfig:
    """Parallel config (本章消费面：pipeline_parallel_size / DCP 乘子)。"""

    # SOURCE: vllm/config/parallel.py pipeline_parallel_size（默认 1；
    #   get_kv_cache_configs 的 PP 取最小用它兜底单 worker）
    pipeline_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py decode_context_parallel_size
    # SUBTRACTED: DCP/PCP 乘子的真实装配与 CP 交错（dossier.delete 第 8 条
    #   ——上下文并行缩放，单卡恒 1 烘干：block_size 不放大、len 不除）
    decode_context_parallel_size: int = 1


# SOURCE: vllm/config/vllm.py:L~200 VllmConfig（装配面切面）
@dataclass
class VllmConfig:
    """The global config (本章消费面：账本函数群的公共入参)。"""

    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # SUBTRACTED: 其余子配置装配（device/model_runner/compilation 等）——
    #   compilation_config 由切面直供属性（cudagraph 门位）；
    #   kv_transfer_config 账位 None（resolve 的 connector 判定输入，双面
    #   契约 → ch16）；speculative_config 账位 None（eagle → ch33）。
    kv_transfer_config: object | None = None
    speculative_config: object | None = None

    # SOURCE: vllm/config/vllm.py:L553 max_in_flight_tokens
    @property
    def max_in_flight_tokens(self) -> int:
        # Upper bound on tokens that are scheduled but not yet settled (freed):
        # every concurrent batch may hold up to a full `max_num_batched_tokens`.
        # Recycling-aware KV cache specs (sliding-window, chunked-local) reserve
        # for this because out-of-window blocks are freed on the processed-token
        # basis, so in-flight steps transiently keep their blocks.
        # SOURCE: vllm/config/vllm.py:L559-L561
        return (
            self.max_concurrent_batches * self.scheduler_config.max_num_batched_tokens
        )

    # SOURCE: vllm/config/vllm.py max_concurrent_batches（默认 1；切面账位
    #   直供——批队列编排 → ch09/ch17）
    max_concurrent_batches: int = 1
