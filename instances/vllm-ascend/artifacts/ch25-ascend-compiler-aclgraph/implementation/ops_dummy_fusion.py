# 精简版（只做减法）— 对照真实源码 vllm_ascend/ops/__init__.py（节选 dummy-fusion-op 部分）
#
# register_dummy_fusion_op 的「占位算子」巧思：融合 pattern 的 search_fn 里引用了
# torch.ops._C_ascend.* 这些符号；若不存在，pattern 注册/匹配阶段就拿不到可引用的算子对象。
# 启动期（worker.py 调用）给 torch.ops._C_ascend 挂一组同名 dummyFusionOp，让 pattern 匹配有锚点
# ——真实算子在别处实现，这里只挂同名属性供图捕获/匹配引用。
#
# 减法：原文件顶部还 import 了 layernorm / fused_moe / activation / rotary_embedding 等一批其它
# ops 子模块的注册（原 :L18-L34、L54 的 __all__），属他章范畴、与本章「占位算子」焦点无关，
# 故只截取 dummyFusionOp + register_dummy_fusion_op 两段。
import torch


# SOURCE: vllm_ascend/ops/__init__.py:L36
class dummyFusionOp:
    default = None

    def __init__(self, name=""):
        # SOURCE: vllm_ascend/ops/__init__.py:L39
        self.name = name


# SOURCE: vllm_ascend/ops/__init__.py:L43
def register_dummy_fusion_op() -> None:
    torch.ops._C_ascend.rms_norm = dummyFusionOp(name="rms_norm")
    torch.ops._C_ascend.fused_add_rms_norm = dummyFusionOp(name="fused_add_rms_norm")
    torch.ops._C_ascend.static_scaled_fp8_quant = dummyFusionOp(name="static_scaled_fp8_quant")
    torch.ops._C_ascend.dynamic_scaled_fp8_quant = dummyFusionOp(name="dynamic_scaled_fp8_quant")
    torch.ops._C_ascend.dynamic_per_token_scaled_fp8_quant = dummyFusionOp(name="dynamic_per_token_scaled_fp8_quant")
    torch.ops._C_ascend.rms_norm_static_fp8_quant = dummyFusionOp(name="rms_norm_static_fp8_quant")
    torch.ops._C_ascend.fused_add_rms_norm_static_fp8_quant = dummyFusionOp(name="fused_add_rms_norm_static_fp8_quant")
    torch.ops._C_ascend.rms_norm_dynamic_per_token_quant = dummyFusionOp(name="rms_norm_dynamic_per_token_quant")
