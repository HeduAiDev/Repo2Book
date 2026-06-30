"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/quant_parser.py

MXFP microscaling 系列的 dtype 映射表：给 W8A8_MXFP8 / W4A4_MXFP4 / W4A8_MXFP 指定
act/weight/scale dtype，其中 scale_dtype=FLOAT8_E8M0FNU（每组共享一个 e8m0 指数）——NPU 硬特化的
per-group 微缩放。注意：本文件【不】决定每层走哪个 scheme（逐层 scheme 决策在 modelslim_config.py
的 get_linear_quant_type / get_quant_type_for_layer），它只管 MXFP dtype 映射与 down_proj rollback。
"""

import torch

# SUBTRACTED: FLOAT4_E2M1FN_X2_DTYPE / ensure_mxfp4_dtype_available / ensure_mxfp8_scale_dtype_available
#   的 import（原 quant_parser.py:L3-L8）——仅被下方已减去的 W4A4/W4A8 表项与 parse_* 函数体使用。
from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE


# SOURCE: vllm_ascend/quantization/quant_parser.py:L11
class QuantTypeMapping:
    quant_configs = {
        "W8A8_MXFP8": {
            "act_quant_type": torch.float8_e4m3fn,
            "weight_quant_type": None,
            "scale_dtype": FLOAT8_E8M0FNU_DTYPE,
            "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
        },
        # SUBTRACTED: "W4A4_MXFP4" / "W4A8_MXFP" 两个表项（原 L19-L30）——同构 MXFP 配置，
        #   保留 W8A8_MXFP8 一行即足以说明 e8m0 共享指数 scale 的 microscaling 结构。
    }

    @staticmethod
    # SOURCE: vllm_ascend/quantization/quant_parser.py:L33
    def get_quant_settings():
        return QuantTypeMapping.quant_configs


# SUBTRACTED: get_rollback_quant_type / parse_mxfp_quant_params / parse_quant_moe_down_proj_params
#   （down_proj rollback 解析 + MXFP round_mode/dtype 出参，原 L38-L73）——MXFP 在本章是「点出 NPU
#   硬特化」的旁证而非主线 scheme，保留上面 dtype 映射表示例即可。
