# vllm_ascend/distributed/parallel_state.py —— subtract-only companion（ch08）
#
# 主线：init_ascend_model_parallel 是一次纯「加法式扩展」。它复用基座的
# init_model_parallel_group / GroupCoordinator（见 vllm_distributed_base.py 桩），
# 在基座已建好的 TP/PP/DP/PCP/DCP 之上，从同一张 all_ranks 5D 网格 reshape/
# transpose/slice 切出一批昇腾专属组（MC2 / 细粒度 TP / flashcomm2 …），不动基座任何组。
#
# 只验排布代数（reshape/transpose/slice，torch CPU 纯算）。真实 hccl 进程组创建被
# 基座桩 SUBTRACTED（host 无 NPU/CANN）。
import torch

# SUBTRACTED: from vllm.config import ParallelConfig, get_current_vllm_config
#   —— ParallelConfig 仅作类型注解；get_current_vllm_config 仅 PD 分离旁支(已删)用。
#   原 vllm_ascend/distributed/parallel_state.py:L2
# 复用基座符号（「复用而非替换」的字面证据）：
from vllm_distributed_base import (
    GroupCoordinator,
    get_tp_group,
    get_world_group,
    init_model_parallel_group,
)
from ascend_runtime_stub import flashcomm2_enable, get_ascend_config

# SUBTRACTED: from vllm_ascend.utils import enable_dsa_cp_with_layer_shard
#   —— 仅 shard_weight 旁支(已删)用。原 vllm_ascend/distributed/parallel_state.py:L6

# SOURCE: vllm_ascend/distributed/parallel_state.py:L8-L27
# Currently, mc2 op need their own group coordinator.
_MC2: "GroupCoordinator | None" = None

# Module specific tensor parallel groups
_MLP_TP: "GroupCoordinator | None" = None
_OTP: "GroupCoordinator | None" = None
_LMTP: "GroupCoordinator | None" = None
_EMBED_TP: "GroupCoordinator | None" = None

# flashcomm specific groups
_FLASHCOMM2_OTP: "GroupCoordinator | None" = None
_FLASHCOMM2_ODP: "GroupCoordinator | None" = None
_FC3_QUANT_X: "GroupCoordinator | None" = None

# SUBTRACTED: _SHARD_WEIGHT —— shard_weight 旁支组(已删)。原:L22-L23
# SUBTRACTED: _P_TP —— PD 分离旁支组(已删)。原:L25

_DYNAMIC_EPLB: "GroupCoordinator | None" = None


def init_ascend_model_parallel(
    parallel_config,
):
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L30-L52
    if model_parallel_initialized():
        return
    # SUBTRACTED: assert torch.distributed.is_initialized()；world_size =
    #   torch.distributed.get_world_size()；backend = get_backend(world_group.device_group)
    #   —— 真实 hccl 运行期。companion 用 world group 桩取 world_size、backend 固定 "hccl"。
    #   原 vllm_ascend/distributed/parallel_state.py:L35-L37
    world_size = get_world_group().world_size
    backend = "hccl"
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size
    global_pcp_size = parallel_config.prefill_context_parallel_size

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1,
        global_dp_size,
        global_pp_size,
        global_pcp_size,
        global_tp_size,
    )

    # SUBTRACTED: PD 分离的 _P_TP / alltoall 头复制组 —— 与本章四主题(MC2/细粒度TP/
    #   flashcomm/CP)正交，仅在 kv_transfer_config.is_kv_producer & pd_head_ratio>1 时触发。
    #   删去不影响 rank 排布代数主线（纯 Python 切分不依赖它）。
    #   原 vllm_ascend/distributed/parallel_state.py:L54-L82

    # EP like group ranks
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            global_dp_size * global_pcp_size * global_tp_size,
        )
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]

    global _MC2
    _MC2 = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="mc2")

    if get_ascend_config().eplb_config.dynamic_eplb:
        global _DYNAMIC_EPLB
        _DYNAMIC_EPLB = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dynamic_eplb"
        )

    if get_ascend_config().multistream_overlap_gate:
        global _FC3_QUANT_X
        _FC3_QUANT_X = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="fc3_quant_x"
        )

    # Initialize fine-grained TP process groups on Ascend for four components:
    # 1. LM Head: output logits projection (`lmhead_tensor_parallel_size`)
    # 2. O Proj: attention output projection (`oproj_tensor_parallel_size`)
    # 3. Embedding: The token embedding table at the input of the model (`embedding_tensor_parallel_size`)
    # 4. MLP: feed-forward network in transformer blocks (`mlp_tensor_parallel_size`)
    _group_cache = {}

    def _create_or_get_group(group_size: int, group_name: str) -> "GroupCoordinator":
        # SOURCE: vllm_ascend/distributed/parallel_state.py:L117-L133
        if group_size is None:
            return None
        if group_size not in _group_cache:
            rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
            num_chunks = global_dp_size // group_size
            group_ranks = []
            for pp_idx in range(global_pp_size):
                stage_ranks = rank_grid[pp_idx]  # (dp, tp)
                for chunk in range(num_chunks):
                    for tp_idx in range(global_tp_size):
                        group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                        group_ranks.append(group)
            pg = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=group_name)
            _group_cache[group_size] = pg

        return _group_cache[group_size]

    otp_size = get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size
    lmhead_tp_size = get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size
    embedding_tp_size = get_ascend_config().finegrained_tp_config.embedding_tensor_parallel_size
    mlp_tp_size = get_ascend_config().finegrained_tp_config.mlp_tensor_parallel_size

    global _OTP, _LMTP, _EMBED_TP, _MLP_TP

    if otp_size > 0:
        _OTP = _create_or_get_group(otp_size, "otp")
    if lmhead_tp_size > 0:
        _LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")
    if embedding_tp_size > 0:
        _EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
    if mlp_tp_size > 0:
        _MLP_TP = _create_or_get_group(mlp_tp_size, "mlptp")

    # TODO: Extract and unify the logic across different communication group.
    flashcomm2_otp_group_ranks = []
    if flashcomm2_enable():
        flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
        num_fc2_oproj_tensor_parallel_groups: int = global_tp_size // flashcomm2_otp_size
        global _FLASHCOMM2_OTP
        global _FLASHCOMM2_ODP

        _FLASHCOMM2_OTP = None
        _FLASHCOMM2_ODP = get_tp_group()

        if flashcomm2_otp_size > 1:
            odp_group_ranks: list = [
                [] for _ in range(flashcomm2_otp_size * global_dp_size * global_pp_size)
            ]
            for dp_group_index in range(global_dp_size):
                for pp_group_index in range(global_pp_size):
                    dp_pp_serial_index = dp_group_index * global_pp_size + pp_group_index
                    tp_base_rank = dp_pp_serial_index * global_tp_size
                    odp_base_index = dp_pp_serial_index * flashcomm2_otp_size

                    for i in range(num_fc2_oproj_tensor_parallel_groups):
                        ranks = []
                        for j in range(flashcomm2_otp_size):
                            tp_local_rank = i + j * num_fc2_oproj_tensor_parallel_groups
                            assert tp_local_rank < global_tp_size
                            global_rank = tp_base_rank + tp_local_rank
                            ranks.append(global_rank)

                            odp_group_index = odp_base_index + j
                            odp_group_ranks[odp_group_index].append(global_rank)
                        flashcomm2_otp_group_ranks.append(ranks)

            _FLASHCOMM2_OTP = init_model_parallel_group(
                flashcomm2_otp_group_ranks, get_world_group().local_rank, backend, group_name="flashcomm2_otp"
            )
            _FLASHCOMM2_ODP = init_model_parallel_group(
                odp_group_ranks, get_world_group().local_rank, backend, group_name="flashcomm2_odp"
            )

    # SUBTRACTED: shard_weight 组 / create_shard_weight_group —— layer_sharding 旁支辅助组，
    #   依赖 flashcomm2 / dsa_cp 配置，彼此独立，删去不破坏其余组排布。
    #   原 vllm_ascend/distributed/parallel_state.py:L191-L226


def model_parallel_initialized():
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L229-L230
    return _MC2 is not None


def get_mc2_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L233-L235
    assert _MC2 is not None, "mc2 group is not initialized"
    return _MC2


def get_mlp_tp_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L238-L240
    assert _MLP_TP is not None, "mlp group is not initialized"
    return _MLP_TP


def get_otp_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L243-L245
    assert _OTP is not None, "output tensor parallel group is not initialized"
    return _OTP


def get_lmhead_tp_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L248-L250
    assert _LMTP is not None, "lm head tensor parallel group is not initialized"
    return _LMTP


def get_embed_tp_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L253-L255
    assert _EMBED_TP is not None, "emtp group is not initialized"
    return _EMBED_TP


def get_flashcomm2_otp_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L258-L259
    # 故意不带断言：flashcomm2_otp_size==1 时合法地返回 None。
    return _FLASHCOMM2_OTP


def get_flashcomm2_odp_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L262-L264
    assert _FLASHCOMM2_ODP is not None, "output data parallel group for flashcomm2 is not initialized"
    return _FLASHCOMM2_ODP


# SUBTRACTED: get_shard_weight_group / get_p_tp_group —— 对应已删的 _SHARD_WEIGHT/_P_TP。
#   原 vllm_ascend/distributed/parallel_state.py:L267-L274


def get_fc3_quant_x_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L277-L279
    assert _FC3_QUANT_X is not None, "fc3 quant x group is not initialized"
    return _FC3_QUANT_X


def get_dynamic_eplb_group() -> "GroupCoordinator":
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L282-L284
    # 前向引用锚：动态专家负载均衡 → ch09。拓扑复用 MC2 group_ranks，本章不展开机制。
    assert _DYNAMIC_EPLB is not None, "Dynamic eplb group is not initialized"
    return _DYNAMIC_EPLB


def destroy_ascend_model_parallel():
    # SOURCE: vllm_ascend/distributed/parallel_state.py:L287-L341
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None

    global _MLP_TP
    if _MLP_TP:
        _MLP_TP.destroy()
    _MLP_TP = None

    global _LMTP
    if _LMTP:
        _LMTP.destroy()
    _LMTP = None

    global _EMBED_TP
    if _EMBED_TP:
        _EMBED_TP.destroy()
    _EMBED_TP = None

    global _OTP
    if _OTP:
        _OTP.destroy()
    _OTP = None

    # SUBTRACTED: _P_TP 销毁块 —— 对应已删的 PD 分离组。原:L313-L316

    global _FLASHCOMM2_OTP
    if _FLASHCOMM2_OTP and get_ascend_config().flashcomm2_oproj_tensor_parallel_size != 1:
        _FLASHCOMM2_OTP.destroy()
        _FLASHCOMM2_OTP = None

    global _FLASHCOMM2_ODP
    if _FLASHCOMM2_ODP and get_ascend_config().flashcomm2_oproj_tensor_parallel_size != 1:
        _FLASHCOMM2_ODP.destroy()
        _FLASHCOMM2_ODP = None

    # SUBTRACTED: _SHARD_WEIGHT 销毁块 —— 对应已删的 shard_weight 组。原:L328-L331

    global _FC3_QUANT_X
    if _FC3_QUANT_X:
        _FC3_QUANT_X.destroy()
    _FC3_QUANT_X = None

    global _DYNAMIC_EPLB
    if _DYNAMIC_EPLB:
        _DYNAMIC_EPLB.destroy()
    _DYNAMIC_EPLB = None
