# 精简版（只做减法）— 对照真实源码 vllm_ascend/compilation/passes/norm_quant_fusion_pass.py
#
# 一个代表性融合 pass：把「add_rms_norm_bias → quantize」这条多算子子图，整体替换成单个
# 融合算子 npu_add_rms_norm_quant，减少 kernel 启动与中间张量往返。
#
# 减法：真实文件有 8 个同构 Pattern 变体（WithBias / SP（带 maybe_all_gather_and_maybe_unpad）/
# DynamicQuant 及其组合，原 :L89-L475），按 dossier 批准只保留最具代表性的 AddRMSNormQuantPattern
# 一例讲透融合机制；AddRMSNormQuantFusionPass.__init__ 里对那 7 个被删变体的 register 调用随之
# 一并 SUBTRACTED（引用已删类无法运行），只保留对 AddRMSNormQuantPattern 的注册。
import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import enable_custom_op


# SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L29
class AddRMSNormQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L30
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L33
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def get_pattern(self):
        # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L45
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L46
            """
            Pattern for AddRMSNormQuant fusion.
            """
            # 待匹配子图：add_rms_norm_bias（含残差加）之后接一个独立的 quantize 算子。
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L67
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L68
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            # 替换为单个融合算子：norm + 量化一次算完。
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


# SUBTRACTED: 7 个同构 Pattern 变体类（原 :L89-L475）——AddRMSNormQuantPatternWithBias /
#             AddRMSNormQuantSPPattern(WithBias) / AddRMSNormDynamicQuantPattern(SP/WithBias 组合)。
#             仅算子签名 / 是否带 bias / 是否插 maybe_all_gather_and_maybe_unpad 不同，结构同构，
#             保留 AddRMSNormQuantPattern 一例即可讲透 pattern→replacement 融合机制。


# SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L477
class AddRMSNormQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig):
        # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L482
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="rmsnorm_quant_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            # SUBTRACTED: 其余 7 个变体 Pattern 的 register（原 :L493-L501）——
            #             AddRMSNormDynamicQuantPattern / AddRMSNormDynamicQuantSPPattern 及
            #             enable_custom_op() 下的 SP/WithBias 组合；这些 Pattern 类已按上方减法删除。
            if enable_custom_op():
                AddRMSNormQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L503
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # SOURCE: vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L509
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
