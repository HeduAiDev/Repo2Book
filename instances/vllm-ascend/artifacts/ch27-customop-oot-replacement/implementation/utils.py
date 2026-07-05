# 只做减法的精简版 —— 活动实例 vllm-ascend
# 真实文件：vllm_ascend/utils.py
#
# 本章一号主角：register_ascend_customop —— 注册表总开关。一处调用，全模型算子被批量顶替。
# 外加第二层二分的开关 enable_custom_op（决定走不走编译出的 AscendC 融合 kernel）。
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# SOURCE: vllm_ascend/utils.py:L52
# 模块级映射表（vLLM CustomOp 子类的「类名字符串」→ 对应 Ascend 子类）；register_ascend_customop 内填充。
REGISTERED_ASCEND_OPS: dict = {}

# 幂等闸标志：注册全局且一次性，多次调用（多 worker / 测试）只生效一次。
_ASCEND_CUSTOMOP_IS_REIGISTERED = False

# enable_custom_op 的布尔缓存：None=未判，True=有融合 kernel，False=回退。
_CUSTOM_OP_ENABLED = None


def enable_custom_op():
    # SOURCE: vllm_ascend/utils.py:L357
    """
    Enable lazy init for vllm_ascend_C to avoid early initialization of CANN's RTS component.
    惰性 import 编译出的 AscendC 算子库；成功→可用融合 kernel，失败→回退原子算子。结果缓存只判一次。
    """
    global _CUSTOM_OP_ENABLED

    if _CUSTOM_OP_ENABLED is not None:
        return _CUSTOM_OP_ENABLED

    # SUBTRACTED: VLLM_BATCH_INVARIANT / A5 芯片强制回退特例（L369-376）—— 边缘场景，删后主二分不变。
    try:
        # SUBTRACTED: bootstrap_custom_op_env() 的 CANN 环境引导（L379-380）—— host 环境细节。
        # register custom ops into torch_library here
        import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
        import vllm_ascend.meta_registration  # type: ignore  # noqa: F401

        _CUSTOM_OP_ENABLED = True
    except ImportError as e:
        # SUBTRACTED: libcust_opapi.so 的 LD_LIBRARY_PATH rpath 二次重试块（L391-406）——
        #             兜底逻辑不改二分语义；host 无 CANN 本就 import 失败 → 走回退。
        _CUSTOM_OP_ENABLED = False
        logger.warning(
            "Failed to register custom ops, all custom ops will be disabled. error=%s.",
            e,
        )
    return _CUSTOM_OP_ENABLED


def register_ascend_customop(vllm_config: VllmConfig | None = None):
    # SOURCE: vllm_ascend/utils.py:L638
    """Register Ascend CustomOP

    NOTE: if the register branch requires model type, please use `vllm.config.get_current_vllm_config`.
    """
    global _ASCEND_CUSTOMOP_IS_REIGISTERED
    if _ASCEND_CUSTOMOP_IS_REIGISTERED:
        return
    from vllm.model_executor.custom_op import CustomOp

    from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul
    from vllm_ascend.ops.layernorm import AscendRMSNorm
    from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
    # SUBTRACTED: 其余 ~25 个 Ascend* 子类 import（linear / rotary_embedding / mla /
    #             vocab_parallel_embedding / conv / gdn / ... 见 L654-681）—— 同构重复，留 3 组代表即可演示全流程。

    global REGISTERED_ASCEND_OPS
    REGISTERED_ASCEND_OPS = {
        "QuickGELU": AscendQuickGELU,
        "SiluAndMul": AscendSiluAndMul,
        "RMSNorm": AscendRMSNorm,
        "FusedMoE": AscendFusedMoE,
        # SUBTRACTED: 其余约 24 项「vLLM 类名 → Ascend 子类」映射（L688-711）—— 每项同构，保留 4 则代表。
    }

    # SUBTRACTED: deepseek_mla 时追加 REGISTERED_ASCEND_OPS["GateLinear"]=AscendGateLinear（L714-724）——
    #             模型类型相关可选支线，不影响主顶替路径。
    # SUBTRACTED: is_310p() 用一批 *310 子类 REGISTERED_ASCEND_OPS.update(...) 覆盖部分键（L726-759）——
    #             310P 是另一芯片的同构覆盖（机制完全相同，只换实现类）；删后主路径行为不变。

    # 一次遍历，逐个把昇腾子类写进基座 op_registry_oot —— 一处调用，全模型算子换头。
    for name, op_cls in REGISTERED_ASCEND_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

    # NOTE: Keep this at last to ensure all custom actions are registered
    _ASCEND_CUSTOMOP_IS_REIGISTERED = True
