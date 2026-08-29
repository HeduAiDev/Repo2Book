# ch19 精简版 config 子包（对应 vllm/config/__init__.py 的再出口面）。
#
# SOURCE: vllm/config/__init__.py（真源把 compilation/vllm 的公开名再出口
# 到 vllm.config 命名空间——custom_op/wrapper/backends 均从 vllm.config 取
# CUDAGraphMode/get_current_vllm_config 等；伴读版同构再出口）。
from .compilation import (  # noqa: F401
    CUDAGraphMode,
    CompilationConfig,
    CompilationMode,
    DynamicShapesConfig,
    DynamicShapesType,
    PassConfig,
)
from .vllm import (  # noqa: F401
    IS_DENSE,
    IS_QUANTIZED,
    OPTIMIZATION_LEVEL_00,
    OPTIMIZATION_LEVEL_01,
    OPTIMIZATION_LEVEL_02,
    OPTIMIZATION_LEVEL_03,
    OPTIMIZATION_LEVEL_TO_CONFIG,
    OptimizationLevel,
    VllmConfig,
    get_cached_compilation_config,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    set_current_vllm_config,
)
