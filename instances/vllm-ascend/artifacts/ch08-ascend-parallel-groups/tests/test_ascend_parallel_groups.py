"""ch08 — init_ascend_model_parallel 的 rank 排布代数测试。

测的是「复现昇腾真实源码的可观察行为」——即 dossier theory 里手推的 group_ranks
数值排布（样例 A/B/C + flashcomm2 strided），而非精简版自洽。真实 hccl 通信不跑，
只验 reshape/transpose/slice 切出的 group_ranks 与手推一致（host 纯 torch CPU）。
"""
import sys
from pathlib import Path

import pytest

IMPL = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import ascend_parallel_state as aps  # noqa: E402
import vllm_distributed_base as base  # noqa: E402
from ascend_runtime_stub import (  # noqa: E402
    AscendConfig,
    EplbConfig,
    FinegrainedTPConfig,
    ParallelConfig,
    set_ascend_config,
)


def _setup(world_size, dp=1, pp=1, pcp=1, tp=1, cfg=None, local_rank=0):
    base.reset_base_groups()
    aps.destroy_ascend_model_parallel()
    base.init_world_group(world_size, local_rank=local_rank)
    set_ascend_config(cfg or AscendConfig())
    return ParallelConfig(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=pcp,
    )


# ---------- 样例 A：world=8, dp=2, pp=1, pcp=1, tp=4 ----------

def test_sample_A_mc2_spans_all_dp_and_tp():
    pc = _setup(8, dp=2, tp=4)
    aps.init_ascend_model_parallel(pc)
    # MC2 = transpose(1,2).reshape(-1, dp*pcp*tp=8) → 单个 size-8 EP 组横跨两个 DP
    assert aps.get_mc2_group().group_ranks == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_sample_A_mlp_tp_is_orthogonal_column_slice():
    cfg = AscendConfig(finegrained_tp_config=FinegrainedTPConfig(mlp=2))
    pc = _setup(8, dp=2, tp=4, cfg=cfg)
    aps.init_ascend_model_parallel(pc)
    # rank_grid=reshape(pp=1,dp=2,tp=4), num_chunks=1 → 沿 DP 列向切，与全局 TP（行向）正交
    assert aps.get_mlp_tp_group().group_ranks == [[0, 4], [1, 5], [2, 6], [3, 7]]


# ---------- 样例 B：world=8, dp=4, tp=2, group_size=2 → num_chunks=2 ----------

def test_sample_B_finegrained_continuous_chunks_along_dp():
    cfg = AscendConfig(finegrained_tp_config=FinegrainedTPConfig(mlp=2))
    pc = _setup(8, dp=4, tp=2, cfg=cfg)
    aps.init_ascend_model_parallel(pc)
    # num_chunks=4//2=2：chunk0 行0:2 → [0,2],[1,3]；chunk1 行2:4 → [4,6],[5,7]
    assert aps.get_mlp_tp_group().group_ranks == [[0, 2], [1, 3], [4, 6], [5, 7]]


# ---------- 样例 C（CP 归口）：world=4, pcp=2, tp=2, dcp=2 ----------

def test_sample_C_base_cp_layout():
    # CP 组由【基座】建，本章作排布归口。dcp 复用 TP 的 GPU；PCP 沿 tp 跳取。
    base.reset_base_groups()
    base.init_world_group(4)
    base.initialize_model_parallel(
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
        tensor_model_parallel_size=2,
        world_size=4,
    )
    assert base.get_tp_group().group_ranks == [[0, 1], [2, 3]]
    assert base.get_dcp_group().group_ranks == [[0, 1], [2, 3]]  # dcp==tp，整组当 DCP
    assert base.get_pcp_group().group_ranks == [[0, 2], [1, 3]]  # transpose(3,4) 跳取


# ---------- flashcomm2 strided 重排 ----------

def test_flashcomm2_strided_otp_and_odp():
    cfg = AscendConfig(
        enable_flashcomm2_parallel_size=2,
        flashcomm2_oproj_tensor_parallel_size=2,
    )
    pc = _setup(8, dp=1, tp=8, cfg=cfg)
    base.initialize_model_parallel(1, 1, 1, 1, 8, world_size=8)  # 基座 TP（_FLASHCOMM2_ODP 默认复用）
    aps.init_ascend_model_parallel(pc)
    # global_tp=8, otp_size=2, num_groups=4：otp 跨步 i + j*4
    assert aps.get_flashcomm2_otp_group().group_ranks == [[0, 4], [1, 5], [2, 6], [3, 7]]
    # 同一 j 收进 odp 组
    assert aps.get_flashcomm2_odp_group().group_ranks == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_flashcomm2_size_one_reuses_base_tp_group():
    # otp_size==1 → _FLASHCOMM2_OTP 留 None、_FLASHCOMM2_ODP 直接复用基座 get_tp_group()
    base.reset_base_groups()
    aps.destroy_ascend_model_parallel()
    base.init_world_group(8)
    base.initialize_model_parallel(1, 1, 1, 1, 8, world_size=8)
    set_ascend_config(
        AscendConfig(enable_flashcomm2_parallel_size=1, flashcomm2_oproj_tensor_parallel_size=1)
    )
    pc = ParallelConfig(tensor_parallel_size=8, data_parallel_size=1,
                        pipeline_parallel_size=1, prefill_context_parallel_size=1)
    aps.init_ascend_model_parallel(pc)
    assert aps.get_flashcomm2_otp_group() is None
    assert aps.get_flashcomm2_odp_group() is base.get_tp_group()


# ---------- 复用 / 哨兵 / 幂等 ----------

def test_dynamic_eplb_reuses_mc2_group_ranks():
    cfg = AscendConfig(eplb_config=EplbConfig(dynamic_eplb=True))
    pc = _setup(8, dp=2, tp=4, cfg=cfg)
    aps.init_ascend_model_parallel(pc)
    # ch09 前向引用：_DYNAMIC_EPLB 复用同一份 group_ranks，只是独立 coordinator
    assert aps.get_dynamic_eplb_group().group_ranks == aps.get_mc2_group().group_ranks
    assert aps.get_dynamic_eplb_group() is not aps.get_mc2_group()


def test_mc2_is_initialization_sentinel():
    pc = _setup(8, dp=2, tp=4)
    assert aps.model_parallel_initialized() is False  # _MC2 None
    aps.init_ascend_model_parallel(pc)
    assert aps.model_parallel_initialized() is True  # _MC2 建好即视为已初始化


def test_idempotent_second_call_is_noop():
    pc = _setup(8, dp=2, tp=4)
    aps.init_ascend_model_parallel(pc)
    first = aps.get_mc2_group()
    aps.init_ascend_model_parallel(pc)  # 幂等守护：直接 return
    assert aps.get_mc2_group() is first


def test_all_groups_are_base_group_coordinator_instances():
    # 「复用而非替换」：昇腾各组都是基座 GroupCoordinator 的实例
    cfg = AscendConfig(finegrained_tp_config=FinegrainedTPConfig(mlp=2))
    pc = _setup(8, dp=2, tp=4, cfg=cfg)
    aps.init_ascend_model_parallel(pc)
    assert isinstance(aps.get_mc2_group(), base.GroupCoordinator)
    assert isinstance(aps.get_mlp_tp_group(), base.GroupCoordinator)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
