# SOURCE: vllm/config/vllm.py
# 本章两个配置真相源：max_concurrent_batches 深度仲裁 property（L539-L550——
# v0.27.1 唯一出处，v0.21 在 executor 侧）与 __post_init__ 里的 async 默认仲裁
# （L1057-L1143——显式 True 硬失败 + None→True 五类降级）。
# 真实 VllmConfig 是几十个字段的编排中心（ch03 全文已立）；此处以裸字段镜像
# 本章路径用到的输入面，仲裁逻辑逐字。
from __future__ import annotations

from dataclasses import dataclass, field

from .logger import init_logger
from .scheduler_config import SchedulerConfig

logger = init_logger(__name__)

# SOURCE: vllm/config/speculative.py:L55-L67 类型面（get_args 消费）：
# NgramGPUTypes = Literal["ngram_gpu"]；EagleModelTypes = Literal["eagle",
# "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes]。
# MTP/DFlash 成员清单归 ch33——保代表性成员，仲裁判定面一致。
EagleModelTypes = (
    "eagle",
    "eagle3",
    "extract_hidden_states",
    "dflash",
    "eagle_mtp",
)
# SOURCE: vllm/config/speculative.py:L61 NgramGPUTypes
NgramGPUTypes = ("ngram_gpu",)
# SOURCE: vllm/config/speculative.py:L63 DSparkModelTypes
DSparkModelTypes = ("dspark",)


# SOURCE: vllm/config/speculative.py SpeculativeConfig（字段面精简：仲裁用到的
# 两个输入——method 与 disable_padded_drafter_batch）
@dataclass
class SpeculativeConfig:
    # SUBTRACTED: num_speculative_tokens/drafter 装配（L70-L300s——ch33）。
    # SOURCE: vllm/config/speculative.py SpeculativeConfig method 字段
    method: str = "eagle"
    # SOURCE: vllm/config/speculative.py SpeculativeConfig disable_padded_drafter_batch
    disable_padded_drafter_batch: bool = False


# SOURCE: vllm/config/model.py ModelConfig（字段面精简）
@dataclass
class ModelConfig:
    # SOURCE: vllm/config/model.py runner_type（pooling 判定输入）
    runner_type: str = "generate"
    # SOURCE: vllm/config/model.py is_diffusion（check_for_draft_tokens 输入）
    is_diffusion: bool = False


# SOURCE: vllm/config/parallel.py ParallelConfig（字段面精简）
@dataclass
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py pipeline_parallel_size
    pipeline_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py distributed_executor_backend
    distributed_executor_backend: str = "uniproc"
    # SUBTRACTED: enable_dbo/all2all_backend 等 ROCm DeepEP 面（L1058-L1062
    #   的判定输入——HOST SEAM 恒 False：非 ROCm 平台组合不触发）。


# SOURCE: vllm/config/vllm.py:L69 VllmConfig（裸字段镜像）
@dataclass
class VllmConfig:
    # SUBTRACTED: 真实 VllmConfig 的几十个配置字段与 __post_init__ 全量校验
    #   （ch03 全文已立）——本章保 async 仲裁路径的输入面。
    # SOURCE: vllm/config/vllm.py scheduler_config 字段
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    # SOURCE: vllm/config/parallel.py pipeline_parallel_size 输入面
    pp_size: int = 1
    # SOURCE: vllm/config/vllm.py model_config（runner_type/is_diffusion）
    runner_type: str = "generate"
    # SOURCE: vllm/config/vllm.py speculative_config 输入面（None=无 spec）
    spec_method: str | None = None
    # SOURCE: vllm/config/speculative.py disable_padded_drafter_batch
    disable_padded_drafter_batch: bool = False
    # SOURCE: vllm/config/parallel.py distributed_executor_backend
    executor_backend: str = "uniproc"
    # SOURCE: vllm/config/vllm.py:L578-L581 use_v2_model_runner（真实是读
    #   VLLM_USE_V2_MODEL_RUNNER 的 property；HOST SEAM：压平为构造字段，
    #   默认 False——V2 runner 分支已随 dossier.delete 第 4 条删）
    use_v2_model_runner: bool = False
    # SOURCE: vllm/config/model.py max_model_len
    max_model_len: int = 4096

    # SOURCE: vllm/config/vllm.py:L563-L567 model_config property（装配面镜像）
    @property
    def model_config(self) -> ModelConfig:
        # SOURCE: vllm/config/vllm.py:L563-L567
        return ModelConfig(
            runner_type=self.runner_type,
            is_diffusion=False,  # SUBTRACTED: diffusion 面（L171 域）——恒 False
        )

    # SOURCE: vllm/config/vllm.py:L570-L576 speculative_config property
    @property
    def speculative_config(self) -> SpeculativeConfig | None:
        # SOURCE: vllm/config/vllm.py:L570-L576
        if self.spec_method is None:
            return None
        return SpeculativeConfig(
            method=self.spec_method,
            disable_padded_drafter_batch=self.disable_padded_drafter_batch,
        )

    # SOURCE: vllm/config/vllm.py:L539-L550 max_concurrent_batches（m2 深度
    # 仲裁——v0.27.1 唯一出处，逐字）
    @property
    def max_concurrent_batches(self) -> int:
        # SOURCE: vllm/config/vllm.py:L539-L550（逐字）
        # PP requires PP-size concurrent batches to fill the pipeline.
        # Async scheduling requires 2 concurrent batches to overlap.
        pp_size = self.parallel_config.pipeline_parallel_size
        if self.scheduler_config.async_scheduling:
            if self.use_v2_model_runner:
                return pp_size + 1
            # V1 Model Runner does not fully support async scheduling with PP.
            if pp_size <= 1:
                return 2
        return pp_size

    # SOURCE: vllm/config/vllm.py:L553-L561 max_in_flight_tokens（旁路消费面）
    @property
    def max_in_flight_tokens(self) -> int:
        # SOURCE: vllm/config/vllm.py:L559-L561
        return (
            self.max_concurrent_batches * self.scheduler_config.max_num_batched_tokens
        )

    # SOURCE: vllm/config/parallel.py parallel_config property（装配面镜像）
    @property
    def parallel_config(self) -> ParallelConfig:
        # SOURCE: vllm/config/parallel.py
        return ParallelConfig(
            pipeline_parallel_size=self.pp_size,
            distributed_executor_backend=self.executor_backend,
        )

    # SOURCE: vllm/config/vllm.py:L1057-L1143 __post_init__ 的 async 仲裁段
    # （显式 True 硬失败 L1064-L1094 + None 默认仲裁 L1095-L1143——逐字；
    # __post_init__ 其余千行校验已删，本方法只承裁这一段）
    def check_and_set_default_async_scheduling(self) -> None:
        # SOURCE: vllm/config/vllm.py:L1057 executor 支持位
        from .executor_factory import get_executor_class

        executor_class = get_executor_class(self)
        executor_supports_async_sched = executor_class.supports_async_scheduling()
        # SOURCE: vllm/config/vllm.py:L1058-L1062 ROCm DeepEP DBO 判定
        # （HOST SEAM：非 ROCm 平台恒 False）
        uses_rocm_deepep_ht_dbo = False

        if self.scheduler_config.async_scheduling:
            # Async scheduling explicitly enabled, hard fail any incompatibilities.
            # Currently, async scheduling only support eagle speculative
            # decoding.
            if uses_rocm_deepep_ht_dbo:
                # SOURCE: vllm/config/vllm.py:L1068-L1073
                raise ValueError(
                    "Async scheduling is not compatible with ROCm DeepEP "
                    "high-throughput DBO. Please use --no-async-scheduling or "
                    "select a different all2all backend."
                )
            if self.speculative_config is not None:
                # SOURCE: vllm/config/vllm.py:L1074-L1090
                if (
                    self.speculative_config.method not in EagleModelTypes
                    and self.speculative_config.method not in NgramGPUTypes
                    and self.speculative_config.method != "draft_model"
                    and self.speculative_config.method not in DSparkModelTypes
                ):
                    raise ValueError(
                        "Currently, async scheduling is only supported "
                        "with EAGLE/MTP/Draft Model/NGram GPU/DSpark kind of "
                        "speculative decoding"
                    )
                if self.speculative_config.disable_padded_drafter_batch:
                    raise ValueError(
                        "Async scheduling is not compatible with "
                        "disable_padded_drafter_batch=True."
                    )
            if not executor_supports_async_sched:
                # SOURCE: vllm/config/vllm.py:L1091-L1094
                raise ValueError(
                    f"`{self.executor_backend}` does not support async scheduling yet."
                )
        # SOURCE: vllm/config/vllm.py:L1095-L1143 默认仲裁（『默认心跳』出处）
        elif self.scheduler_config.async_scheduling is None:
            # Enable async scheduling unless there is an incompatible option.
            if (
                self.model_config is not None
                and self.model_config.runner_type == "pooling"
            ):
                # The current implementation of asynchronous scheduling negatively
                # impacts performance of pooling models, so we disable by default.
                logger.debug(
                    "Disabling asynchronous scheduling by default for pooling model."
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method not in EagleModelTypes
                and self.speculative_config.method not in NgramGPUTypes
                and self.speculative_config.method not in DSparkModelTypes
            ):
                logger.warning_once(
                    "Async scheduling not supported with %s-based "
                    "speculative decoding and will be disabled.",
                    self.speculative_config.method,
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.disable_padded_drafter_batch
            ):
                logger.warning_once(
                    "Async scheduling is not compatible with "
                    "disable_padded_drafter_batch=True and will be disabled.",
                )
                self.scheduler_config.async_scheduling = False
            elif not executor_supports_async_sched:
                logger.warning_once(
                    "Async scheduling will be disabled because it is not supported "
                    "with the `%s` distributed executor backend. ",
                    self.executor_backend,
                )
                self.scheduler_config.async_scheduling = False
            elif uses_rocm_deepep_ht_dbo:
                logger.warning_once(
                    "Async scheduling is disabled for ROCm DeepEP "
                    "high-throughput DBO because that combination can corrupt "
                    "DP+EP generation accuracy."
                )
                self.scheduler_config.async_scheduling = False
            else:
                # SOURCE: vllm/config/vllm.py:L1142-L1143 默认开启
                self.scheduler_config.async_scheduling = True
