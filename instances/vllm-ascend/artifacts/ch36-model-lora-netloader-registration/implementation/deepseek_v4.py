# ch30 变体(1) 整模型特化 —— subtract-only 精简版
#
# 真实源码 vllm_ascend/models/deepseek_v4.py 共 1521 行：DeepSeek-V4 在 NPU 上的整模型
# 特化（昇腾唯一的整模型重写，对位 vLLM vllm/model_executor/models/deepseek_v2.py）。
#
# 本章定位是「注册 + 改动面概览」，不逐行讲模型。按 subtraction_plan.delete 批准项，
# 删去 ~1500 行的 layer/forward/load_weights 实现主体与全部辅助类
# （DeepseekV4Model / DeepseekV2MixtureOfExperts / DeepseekV4Attention / DeepseekV4MoE /
#  Ascend*Cache 等），只保留：
#   (a) imports 暴露出来的 NPU 特化「改动面」——同一个 DeepSeek-V4 在 NPU 上改了哪几类东西；
#   (b) AscendDeepseekV4ForCausalLM 的类签名（must_keep）——被 ModelRegistry 注册的对象。
# 删后此文件不可在 host 导入/运行（依赖 torch_npu / vllm 内部），与真仓一致只读控制面。

# ---------------------------------------------------------------------------
# 改动面（imports）：骨架与 vLLM deepseek_v2 同构，只把 attention/rope/cache/op 换成 NPU 版
# SOURCE: vllm_ascend/models/deepseek_v4.py:L26-L84
# ---------------------------------------------------------------------------
import torch_npu  # noqa: F401  —— NPU 运行时（host 不可用）
from torch import nn

# vLLM 侧通用层 / 接口（与 deepseek_v2 共用的骨架部件）
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    SupportsEagle,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.distributed import get_pp_group
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekV4SWACache as VllmDeepseekV4SWACache,
)

# 昇腾特化算子 —— 这正是「在 NPU 上改了哪几类东西」的清单：
#   稀疏注意力 / DSA、复指数 rope、若干 *Cache 子类、triton 融合 kernel
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules  # noqa: F401
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding  # noqa: F401
from vllm_ascend.ops.triton.mul_add import muls_add_triton  # noqa: F401

# SUBTRACTED: vllm_ascend/models/deepseek_v4.py:L113-L205 —— 三个昇腾 *Cache 子类
#   AscendCompressorStateCache(CompressorStateCache) / AscendDeepseekV4IndexerCache(
#   DeepseekV4IndexerCache) / AscendDeepseekV4SWACache(VllmDeepseekV4SWACache)：都是继承
#   vLLM 同名基类的薄壳，本章只点名「cache 也换了 NPU 版」，不逐行。

# SUBTRACTED: vllm_ascend/models/deepseek_v4.py:L308-L1170 —— 模型骨架各层实现
#   DeepseekV2MLP / DeepseekV4MoE / Indexer / Compressor / DeepseekV4Attention /
#   DeepseekV2DecoderLayer / DeepseekV4Model：与 vLLM deepseek_v2 同构，只把内部 op 换成
#   上面 import 的 NPU 版。本章不逐行讲前向（cartography 标为「中」，挑骨架）。

# SUBTRACTED: vllm_ascend/models/deepseek_v4.py:L1172-L1210 —— DeepseekV2MixtureOfExperts
#   (MixtureOfExperts) MoE 参数管理 mixin，AscendDeepseekV4ForCausalLM 继承它。


# ---------------------------------------------------------------------------
# 被 ModelRegistry 注册的对象：架构名 "DeepseekV4ForCausalLM" → 本类
# ---------------------------------------------------------------------------
# SOURCE: vllm_ascend/models/deepseek_v4.py:L1212-L1239
class AscendDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts, SupportsLoRA, SupportsEagle):  # noqa: F821
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls = DeepseekV4Model  # noqa: F821  —— SUBTRACTED 的 NPU 版模型骨架

    # SOURCE: vllm_ascend/models/deepseek_v4.py:L1218-L1239
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()

    # SUBTRACTED: vllm_ascend/models/deepseek_v4.py:L1241-L1521 —— set_moe_parameters /
    #   forward / compute_logits / load_weights 等整模型前向与权重加载（~280 行）。本章
    #   只讲「同一个 DeepSeek-V4 在 NPU 上改了哪几类东西 + 整模型被注册进 ModelRegistry」，
    #   不深入前向；删后控制流不影响「注册」这条主线。
