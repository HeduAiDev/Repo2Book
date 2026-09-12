# SOURCE: vllm/models/deepseek_v4/__init__.py —— 减法门面（m9 布局证据：
# 平台分发骨架；子包本体（nvidia/amd/xpu 平台子类与 MoE）归 ch28 capstone）
from vllm.models.deepseek_v4.attention import (  # noqa: F401
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v4.compressor import (  # noqa: F401
    DeepseekCompressor,
)
