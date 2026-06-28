# 昇腾运行期配置/门控的 subtract-only 桩 —— 供 init_ascend_model_parallel 取参。
#
# 真实来源 vllm_ascend/ascend_config.py（AscendConfig / FinegrainedTPConfig /
# EplbConfig）与 vllm_ascend/utils.py（flashcomm2_enable）。这里只保留 init 实际
# 读到的字段，其余校验/旁支 SUBTRACTED，并提供 set_ 注入口让排布测试驱动不同并行度。


class FinegrainedTPConfig:
    # SOURCE: vllm_ascend/ascend_config.py:L429-L486
    """oproj/lmhead/embedding/mlp 各自的细粒度 TP 宽度（0=未启用）。"""

    def __init__(self, oproj=0, lmhead=0, embedding=0, mlp=0):
        # SOURCE: vllm_ascend/ascend_config.py:L434-L439
        self.oproj_tensor_parallel_size = oproj
        self.lmhead_tensor_parallel_size = lmhead
        self.embedding_tensor_parallel_size = embedding
        self.mlp_tensor_parallel_size = mlp
        # SUBTRACTED: olora_tensor_parallel_size —— o_lora 在 parallel_state.py 的 init
        #   里并未建组（只建 otp/lmtp/emtp/mlptp 四个），与本章无关。原:L439
        # SUBTRACTED: 各 size 的 graph-mode / PD-scenario / 整除 data_parallel_size 校验
        #   与「仅 MoE 模型可用」断言。原:L442-L486


class EplbConfig:
    # SOURCE: vllm_ascend/ascend_config.py:L_EplbConfig
    def __init__(self, dynamic_eplb=False):
        self.dynamic_eplb = dynamic_eplb


class AscendConfig:
    # SOURCE: vllm_ascend/ascend_config.py:L_AscendConfig
    def __init__(
        self,
        finegrained_tp_config=None,
        eplb_config=None,
        multistream_overlap_gate=False,
        enable_flashcomm2_parallel_size=0,
        flashcomm2_oproj_tensor_parallel_size=1,
    ):
        self.finegrained_tp_config = finegrained_tp_config or FinegrainedTPConfig()
        self.eplb_config = eplb_config or EplbConfig()
        self.multistream_overlap_gate = multistream_overlap_gate
        self.enable_flashcomm2_parallel_size = enable_flashcomm2_parallel_size
        self.flashcomm2_oproj_tensor_parallel_size = flashcomm2_oproj_tensor_parallel_size
        # SUBTRACTED: pd_tp_ratio / pd_head_ratio / layer_sharding —— 分别驱动已删的
        #   PD 分离 _P_TP 与 shard_weight 旁支。原 vllm_ascend/ascend_config.py:L97,L177


class ParallelConfig:
    # SOURCE: vllm/config/parallel.py:L_ParallelConfig
    # SUBTRACTED: 基座 ParallelConfig 全字段，仅留 init_ascend_model_parallel 读的 4 个 size。
    def __init__(
        self,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
    ):
        # SOURCE: vllm/config/parallel.py:L_ParallelConfig (仅留 4 个 size)
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.prefill_context_parallel_size = prefill_context_parallel_size


_ASCEND_CONFIG = AscendConfig()


def get_ascend_config():
    # SOURCE: vllm_ascend/ascend_config.py:L_get_ascend_config
    return _ASCEND_CONFIG


def set_ascend_config(cfg):
    # SOURCE: vllm_ascend/ascend_config.py:L_init_ascend_config（真实 init_ascend_config
    #   设模块级 _ascend_config 全局；此处为测试注入口，同样只改全局）
    global _ASCEND_CONFIG
    _ASCEND_CONFIG = cfg


def flashcomm2_enable():
    # SOURCE: vllm_ascend/utils.py:L1165-L1167
    config_val = get_ascend_config().enable_flashcomm2_parallel_size
    return config_val > 0
