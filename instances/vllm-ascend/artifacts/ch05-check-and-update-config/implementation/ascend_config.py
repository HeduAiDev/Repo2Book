# 章 ch05 精简版 —— vllm_ascend/ascend_config.py（subtract-only）
#
# 本章第二条主线：把开放 dict `vllm_config.additional_config` 解析成一组强类型子配置对象 +
# 标量字段（聚合根 AscendConfig），标量经 _get_config_value 走 additional_config→env→default 三级取值；
# 进程级懒加载单例由 init_ascend_config 管理（refresh / 同一 vllm_config / 完整性守卫）。
import os
from typing import TYPE_CHECKING, Any

# SUBTRACTED: 真源码 `from vllm.logger import logger`；host 无 vLLM，换成 _support 里的 stdlib 替身。
from _support import logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# SOURCE: vllm_ascend/ascend_config.py:L27-L283
class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        # SOURCE: vllm_ascend/ascend_config.py:L32-L283
        self.vllm_config = vllm_config
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        xlite_graph_config = additional_config.get("xlite_graph_config", {})
        self.xlite_graph_config = XliteGraphConfig(xlite_graph_config, vllm_config)

        ascend_compilation_config = additional_config.get("ascend_compilation_config", {})
        self.ascend_compilation_config = AscendCompilationConfig(**ascend_compilation_config)

        # SUBTRACTED: ascend_fusion_config / finegrained_tp_config / weight_prefetch_config /
        #             profiling_chunk_config（含 pp>1 交叉校验）等十余个『开放 dict 某键 → 对应强类型类/标量』
        #             的同范式解析（ascend_config.py:L42-L73, L95-L283 的大部分），删减计划批准。
        #             保留 eplb_config：init_ascend_config 的完整性守卫 _is_ascend_config_initialized
        #             探测该字段，删了会让单例缓存判定永远失败，破坏单例语义。
        eplb_config = additional_config.get("eplb_config", {})
        self.eplb_config = EplbConfig(eplb_config)

        # 三级取值的调用现场：标量字段经 _get_config_value 取，第4个实参 ascend_envs.VLLM_ASCEND_*
        # 在传入前就已被 envs.__getattr__ 求值（env 与 default 在 envs 表 lambda 内塌缩为单一 env_value）。
        # SUBTRACTED: 真源码 `from vllm_ascend import envs as ascend_envs`；改为本精简版同目录的 envs 模块。
        import envs as ascend_envs

        self.enable_balance_scheduling = self._get_config_value(
            additional_config,
            "enable_balance_scheduling",
            "VLLM_ASCEND_BALANCE_SCHEDULING",
            ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING,
        )
        self.enable_flashcomm1 = self._get_config_value(
            additional_config,
            "enable_flashcomm1",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1",
            ascend_envs.VLLM_ASCEND_ENABLE_FLASHCOMM1,
        )
        # SUBTRACTED: 同形态的 _get_config_value 标量约十次（enable_matmul_allreduce / enable_mlapo /
        #             weight_nz_mode 等）+ dump_config / layer_sharding / enable_shared_expert_dp 等
        #             标量与交叉校验（ascend_config.py:L89-L283），同范式重复，删减计划批准。

    # SOURCE: vllm_ascend/ascend_config.py:L284-L296
    @staticmethod
    def _get_config_value(additional_config: dict[str, Any], config_key: str, env_key: str, env_value: Any) -> Any:
        # SOURCE: vllm_ascend/ascend_config.py:L284-L296
        if config_key in additional_config:
            value = additional_config[config_key]
            logger.info_once(f"AscendConfig.{config_key} is set from additional_config with value {value}.")
            return value
        if env_key in os.environ:
            logger.info_once(
                f"AscendConfig.{config_key} falls back to environment variable {env_key} with value {env_value}. "
                f"Please use additional_config.{config_key} instead, because {env_key} will be removed in the "
                "next release."
            )
        return env_value


# SOURCE: vllm_ascend/ascend_config.py:L489-L538
class AscendCompilationConfig:
    """
    Configuration for controlling the behavior of Ascend graph optimization.
    """

    def __init__(
        self,
        enable_npugraph_ex: bool = True,
        enable_static_kernel: bool = False,
        fuse_norm_quant: bool = True,
        fuse_qknorm_rope: bool = True,
        fuse_allreduce_rms: bool = False,
        **kwargs,
    ):
        # SOURCE: vllm_ascend/ascend_config.py:L498-L538
        # SUBTRACTED: 原 __init__ 含一段逐项解释 5 个命名形参的长 docstring（L507-L530），正文已转述。
        self.fuse_norm_quant = fuse_norm_quant
        self.fuse_qknorm_rope = fuse_qknorm_rope
        self.fuse_allreduce_rms = fuse_allreduce_rms
        self.enable_npugraph_ex = enable_npugraph_ex
        self.enable_static_kernel = enable_static_kernel
        # **kwargs 是『无 schema 后门』的体现：未知键被静默吞掉（向前兼容），代价是拼错键名不报错。
        self.fuse_muls_add = kwargs.get("fuse_muls_add", True)
        if self.enable_static_kernel:
            assert self.enable_npugraph_ex, "Static kernel generation requires npugraph_ex to be enabled."


# SOURCE: vllm_ascend/ascend_config.py:L559-L582
class XliteGraphConfig:
    """
    Configuration Object for xlite_graph_config from additional_config
    """

    def __init__(self, xlite_graph_config, vllm_config):
        # SOURCE: vllm_ascend/ascend_config.py:L564-L582
        self.enabled = xlite_graph_config.get("enabled", False)
        self.full_mode = xlite_graph_config.get("full_mode", False)
        if self.enabled:
            if bool(vllm_config.speculative_config):
                raise RuntimeError(
                    "Xlite graph mode is not compatible with speculative decoding. Please disable speculative decoding."
                )
            if vllm_config.parallel_config.pipeline_parallel_size > 1:
                raise RuntimeError(
                    "Xlite graph mode is not compatible with pipeline parallelism. "
                    "Please set pipeline_parallel_size to 1."
                )
            if vllm_config.cache_config.block_size != 128:
                logger.warning(
                    "Current cache block size may not be optimal for xlite graph mode. "
                    "current_block_size=%d, recommended_block_size=128.",
                    vllm_config.cache_config.block_size,
                )


# SOURCE: vllm_ascend/ascend_config.py:L709-L769
class EplbConfig:
    """
    Configuration Object for xlite_graph_config from additional_config
    """

    _defaults = {
        "dynamic_eplb": False,
        "expert_map_path": None,
        "expert_heat_collection_interval": 400,
        "algorithm_execution_interval": 30,
        "expert_map_record_path": None,
        "num_redundant_experts": 0,
        "eplb_policy_type": 1,
    }

    def __init__(self, user_config: dict | None = None):
        # SOURCE: vllm_ascend/ascend_config.py:L724-L735
        if user_config is None:
            user_config = {}
        self.config = self._defaults.copy()
        if user_config and isinstance(user_config, dict):
            for key, value in user_config.items():
                if key in self.config:
                    self.config[key] = value
                else:
                    raise ValueError(f"Config has no attribute '{key}'")

        self._validate_config()

    def __getattr__(self, key):
        # SOURCE: vllm_ascend/ascend_config.py:L737-L740
        if key in self.config:
            return self.config[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def _validate_config(self):
        # SOURCE: vllm_ascend/ascend_config.py:L742-L769
        # SUBTRACTED: expert_map_path/.json 落盘校验、dynamic_eplb 环境变量断言等分支（L743-L766），
        #             与本章无关；保留区间/枚举校验代表性几条以示『子配置就地 _validate』范式。
        for key in ["expert_heat_collection_interval", "algorithm_execution_interval", "num_redundant_experts"]:
            if not isinstance(self.config[key], int):
                raise TypeError(f"{key} must be an integer")
            if self.config[key] < 0:  # type: ignore
                raise ValueError(f"{key} must greater than 0; got {self.config[key]} instead")
        if self.eplb_policy_type not in [0, 1, 2, 3]:
            raise ValueError("eplb_policy_type must in [0, 1, 2, 3]")


_ASCEND_CONFIG: AscendConfig | None = None


# SOURCE: vllm_ascend/ascend_config.py:L775-L785
def _is_ascend_config_initialized(config: AscendConfig | None) -> bool:
    """Check whether a config object has essential initialized fields.

    Some unit tests monkeypatch ``AscendConfig.__init__`` to bypass heavy
    initialization. In that case, the singleton cache can be polluted with a
    partially initialized instance. This guard prevents reusing such instances
    across tests.
    """
    if config is None:
        return False
    return hasattr(config, "ascend_compilation_config") and hasattr(config, "eplb_config")


# SOURCE: vllm_ascend/ascend_config.py:L788-L804
def init_ascend_config(vllm_config):
    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
    refresh = additional_config.get("refresh", False) if additional_config else False
    global _ASCEND_CONFIG
    if (
        _ASCEND_CONFIG is not None
        and not refresh
        and _is_ascend_config_initialized(_ASCEND_CONFIG)
        and getattr(_ASCEND_CONFIG, "vllm_config", None) is vllm_config
    ):
        return _ASCEND_CONFIG
    new_config = AscendConfig(vllm_config)
    if _is_ascend_config_initialized(new_config):
        _ASCEND_CONFIG = new_config
    else:
        logger.warning("Ascend config instance is not fully initialized. action: skip singleton cache update. ")
    return new_config


# SOURCE: vllm_ascend/ascend_config.py:L807-L812
def clear_ascend_config():
    global _ASCEND_CONFIG
    _ASCEND_CONFIG = None
    # SUBTRACTED: 真源码还 `from vllm_ascend.utils import clear_enable_sp; clear_enable_sp()`，
    #            清的是另一缓存（enable_sp），与本章单例无关。


# SOURCE: vllm_ascend/ascend_config.py:L815-L819
def get_ascend_config():
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is None or not _is_ascend_config_initialized(_ASCEND_CONFIG):
        raise RuntimeError("Ascend config is not initialized. Please call init_ascend_config first.")
    return _ASCEND_CONFIG
