# ch22《slot_mapping 与 block_table》测试电池 —— TDD：先测真实 vLLM v0.27.1
# (6e448d0ea) 的可观察行为，精简版实现到通过为止。全部 host 可跑（纯单元，
# 不 import vllm、无 CUDA 上下文——kernel/CUDA 面以 HOST SEAM 镜像承载）。
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation._host_seams import (  # noqa: E402
    CUDAGraphMode,
    SchedulerOutputSeam,
    VllmConfigSeam,
    make_attn_group,
    make_kv_cache_config,
)
from implementation.attention import (  # noqa: E402
    get_attention_context,
    unified_attention_with_output,
    unified_kv_cache_update,
)
from implementation.backend import (  # noqa: E402
    AttentionBackend,
    CommonAttentionMetadata,
    MultipleOf,
)
from implementation.block_table import (  # noqa: E402
    PAD_SLOT_ID,
    BlockTable,
    MultiGroupBlockTable,
    SlotMappingMode,
    get_block_table_width,
)
from implementation.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    reshape_and_cache_flash,
)
from implementation.forward_context import (  # noqa: E402
    BatchDescriptor,
    set_forward_context,
)
from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    MambaSpec,
    get_kv_cache_spec_kind,
)
from implementation.worker_utils import (  # noqa: E402
    prepare_kernel_block_sizes,
    select_common_block_size,
)

CPU = torch.device("cpu")
LAYER0 = "model.layers.0.self_attn.attn"
LAYER1 = "model.layers.1.self_attn.attn"
MAMBA1 = "model.layers.1.mamba"


def make_bt(
    block_size: int = 16,
    kernel_block_size: int | None = None,
    max_num_reqs: int = 8,
    max_blocks_per_req: int = 64,
    max_num_batched_tokens: int = 128,
    mode: SlotMappingMode = SlotMappingMode.TOKEN_TO_KV_SLOT,
) -> BlockTable:
    return BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_blocks_per_req,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=False,
        device=CPU,
        kernel_block_size=kernel_block_size or block_size,
        cp_kv_cache_interleave_size=1,
        slot_mapping_mode=mode,
    )


# ── m1 槽位恒等式：slot = block_table[req][pos//bs] × bs + pos%bs ──────────
# dossier theory：block_size=16、prompt 100 token、第 100 个 token（pos=99）
# 落在第 99//16=6 个逻辑块的第 99%16=3 槽；块表行第 6 项=物理块 9 → slot=147。
def test_slot_identity_worked_example():
    bt = make_bt(block_size=16, max_blocks_per_req=16)
    # 7 个逻辑块，行内第 6 项刻意放物理块 9。
    bt.add_row([3, 8, 2, 7, 1, 5, 9], row_idx=0)
    bt.commit_block_table(1)
    positions = torch.arange(100, dtype=torch.int64)
    qsl = torch.tensor([0, 100], dtype=torch.int32)
    bt.compute_slot_mapping(1, qsl, positions)
    slots = bt.slot_mapping.gpu
    assert int(slots[99]) == 9 * 16 + 3 == 147
    # 正逆闭合：写侧 kernel 的逆分解 block_idx=slot//bs、offset=slot%bs。
    assert 147 // 16 == 9 and 147 % 16 == 3
    # 整行核对：每 token 的 slot 都服从恒等式。
    row = bt.block_table.np[0]
    for pos in range(100):
        expect = int(row[pos // 16]) * 16 + pos % 16
        assert int(slots[pos]) == expect


# ── m3 PAD 程序：第 num_reqs+1 个 program 每拍重填 [num_tokens, max) ───────
def test_pad_program_refills_tail_every_step():
    bt = make_bt(max_num_batched_tokens=128)
    # 7 个逻辑块（第 99 个 token 落第 6 块第 3 槽，行内第 6 项=5）。
    bt.add_row([4, 9, 8, 7, 6, 3, 5], row_idx=0)
    bt.commit_block_table(1)
    qsl = torch.tensor([0, 100], dtype=torch.int32)
    bt.compute_slot_mapping(1, qsl, torch.arange(100, dtype=torch.int64))
    slots = bt.slot_mapping.gpu
    assert int(slots[99]) == 5 * 16 + 3
    assert bool((slots[100:128] == PAD_SLOT_ID).all()) and PAD_SLOT_ID == -1
    # 持久 buffer 的尾部会残留上一拍的真数据——第二拍 token 数变少时必须重填。
    bt.compute_slot_mapping(1, torch.tensor([0, 10], dtype=torch.int32),
                            torch.arange(10, dtype=torch.int64))
    assert bool((bt.slot_mapping.gpu[10:128] == PAD_SLOT_ID).all())
    assert int(bt.slot_mapping.gpu[9]) == 4 * 16 + 9  # 前缀仍真（第 0 块）


# ── m10 SlotMappingMode.NONE：Mamba/GDN 组早退、不算 per-token slot ─────────
def test_none_mode_early_return():
    bt = make_bt(mode=SlotMappingMode.NONE)
    bt.add_row([11], row_idx=0)
    bt.slot_mapping.np[:] = 777  # 哨兵：early return 后必须原样
    bt.compute_slot_mapping(1, torch.tensor([0, 8], dtype=torch.int32),
                            torch.arange(8, dtype=torch.int64))
    # 早退：per-token slot 一律不算（CPU 侧哨兵原样、GPU 镜像不更新）。
    assert bool((bt.slot_mapping.np == 777).all())
    assert bool((bt.slot_mapping.gpu == 0).all())


# ── m8 CP 分片：I-token 交错归属、非本秩打 PAD、单卡退化恒等 ────────────────
def test_cp_interleave_worked_example():
    # dossier theory：W=2、I=2、block_size=16 → virtual_block_size=32；
    # pos=35 → vbi=1、voff=3；is_local=(3//2)%2==R → rank1 真；
    # rank1 local_block_offsets=(3//4)*2+3%2=1；block_indices=1；slot=行[1]*16+1。
    for rank, expect_local in ((1, True), (0, False)):
        bt = make_bt(block_size=16, max_blocks_per_req=16)
        bt.dcp_world_size = 2  # HOST SEAM 观测位：kernel 烘干值的 host 镜像
        bt.dcp_rank = rank
        bt.cp_kv_cache_interleave_size = 2
        bt.add_row([10, 20], row_idx=0)
        bt.commit_block_table(1)
        bt.compute_slot_mapping(1, torch.tensor([0, 1], dtype=torch.int32),
                                torch.tensor([35], dtype=torch.int64))
        if expect_local:
            assert int(bt.slot_mapping.gpu[0]) == 20 * 16 + 1
        else:
            assert int(bt.slot_mapping.gpu[0]) == PAD_SLOT_ID


def test_cp_single_gpu_degrades_to_identity():
    bt = make_bt(block_size=16)
    assert bt.dcp_world_size == 1 and bt.cp_kv_cache_interleave_size == 1
    bt.add_row([3, 9], row_idx=0)
    bt.commit_block_table(1)
    positions = torch.arange(40, dtype=torch.int64)
    bt.compute_slot_mapping(1, torch.tensor([0, 40], dtype=torch.int32), positions)
    for pos in range(40):
        expect = int(bt.block_table.np[0][pos // 16]) * 16 + pos % 16
        assert int(bt.slot_mapping.gpu[pos]) == expect


# ── m9 拆块算术：map_to_kernel_blocks docstring 例 [0,1,2]→[0,1,2,3,4,5] ────
def test_map_to_kernel_blocks_docstring_example():
    out = BlockTable.map_to_kernel_blocks(
        np.array([0, 1, 2]), 2, np.arange(0, 2).reshape(1, -1)
    )
    assert out.tolist() == [0, 1, 2, 3, 4, 5]
    # blocks_per_kv_block=1 直通
    src = np.array([7, 8])
    assert BlockTable.map_to_kernel_blocks(src, 1, None) is src


def test_hybrid_blocks_split_and_slots():
    # 32-token 内存块 × 16-token kernel 块：append_row([7]) 落行 [14,15]。
    bt = make_bt(block_size=32, kernel_block_size=16, max_blocks_per_req=8)
    assert bt.use_hybrid_blocks and bt.blocks_per_kv_block == 2
    assert bt.block_size == 16 and bt.kv_cache_block_size == 32
    assert bt.max_num_blocks_per_req == 16  # 表宽乘 blocks_per_kv_block
    bt.add_row([7], row_idx=0)
    assert bt.block_table.np[0, :2].tolist() == [14, 15]
    bt.commit_block_table(1)
    bt.compute_slot_mapping(1, torch.tensor([0, 32], dtype=torch.int32),
                            torch.arange(32, dtype=torch.int64))
    slots = bt.slot_mapping.gpu
    assert int(slots[0]) == 14 * 16 + 0 and int(slots[15]) == 14 * 16 + 15
    assert int(slots[16]) == 15 * 16 + 0 and int(slots[31]) == 15 * 16 + 15


# ── m11 块表宽度 128-token 对齐 + 细分定宽 ─────────────────────────────────
def test_get_block_table_width_alignment_and_split():
    # token_alignment=128：max_num_blocks=7、block_size=16 → block_alignment=
    # 128//gcd(128,16)=8 → ceil(7/8)*8=8 → 宽 8。
    assert get_block_table_width(7, 16) == 8
    # 对齐到 128-token 的块数：5 个 32-token 块（160 token）→ 对齐到 8 块 →
    # 细分 8*32//16=16。
    assert get_block_table_width(5, 32, 16) == 16
    # 已对齐不再放大：4 个 32-token 块恰 128 token → 4*32//16=8。
    assert get_block_table_width(4, 32, 16) == 8
    # 不对齐口径（NONE 组）：5*32//16=10。
    assert get_block_table_width(5, 32, 16, token_alignment=None) == 10
    # 已对齐不再放大：8 个 16-token 块恰好。
    assert get_block_table_width(8, 16) == 8
    # 校验支：kernel 块不整除 / token_alignment 非正。
    with pytest_raises(ValueError):
        get_block_table_width(4, 16, 7)
    with pytest_raises(ValueError):
        get_block_table_width(4, 16, token_alignment=0)


class pytest_raises:
    """极简 raises 断言（不引 pytest 依赖也能裸跑 python tests/...py）。"""

    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, tp, val, tb):
        assert tp is not None and issubclass(tp, self.exc), (
            f"expected {self.exc.__name__}, got {tp}")
        return True


# ── 站2 CPU 侧行维护五原语（含 m13 move_row 清空源行防 stale #49757）───────
def test_row_primitives():
    bt = make_bt(max_blocks_per_req=16)
    # append_row 差量追加：两次 append 接着写，num_blocks_per_row 记账。
    bt.append_row([1, 2], row_idx=0)
    bt.append_row([3], row_idx=0)
    assert bt.num_blocks_per_row[0] == 3
    assert bt.block_table.np[0, :3].tolist() == [1, 2, 3]
    # 空追加 no-op。
    bt.append_row([], row_idx=0)
    assert bt.num_blocks_per_row[0] == 3
    # add_row 重置重写。
    bt.add_row([9], row_idx=0)
    assert bt.num_blocks_per_row[0] == 1
    assert bt.block_table.np[0, 0] == 9
    # clear_row 清行。
    bt.clear_row(0)
    assert bt.num_blocks_per_row[0] == 0 and bt.block_table.np[0, 0] == 0
    # move_row 压实搬行——搬完清空源行（防 dummy-run 把陈旧行当 mamba 状态槽）。
    bt.add_row([4, 5, 6], row_idx=1)
    bt.move_row(1, 2)
    assert bt.block_table.np[2, :3].tolist() == [4, 5, 6]
    assert bt.num_blocks_per_row[2] == 3
    assert bt.num_blocks_per_row[1] == 0
    assert bool((bt.block_table.np[1, :3] == 0).all())
    # swap_row 交换重排。
    bt.add_row([7], row_idx=1)
    bt.swap_row(1, 2)
    assert bt.block_table.np[1, :3].tolist() == [4, 5, 6]
    assert bt.block_table.np[2, 0] == 7


def test_commit_copies_only_active_rows():
    bt = make_bt(max_blocks_per_req=8)
    bt.add_row([1, 2], row_idx=0)
    bt.add_row([3], row_idx=1)
    bt.commit_block_table(2)  # 只拷活跃前 2 行
    assert bt.block_table.gpu[0, :2].tolist() == [1, 2]
    assert bt.block_table.gpu[1, 0] == 3
    bt.block_table.np[2, :2] = [8, 8]  # 非活跃行后写——GPU 镜像不跟
    assert bt.block_table.gpu[2, 0] == 0


# ── m12 多组扇出：MultiGroupBlockTable 每组一张、NONE 组不对齐 ─────────────
def test_multi_group_fanout_and_none_alignment():
    mg = MultiGroupBlockTable(
        max_num_reqs=4,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=CPU,
        block_sizes=[16, 32],
        kernel_block_sizes=[16, 16],
        max_num_blocks=[7, 4],
        cp_kv_cache_interleave_size=1,
        slot_mapping_modes=[SlotMappingMode.TOKEN_TO_KV_SLOT, SlotMappingMode.NONE],
    )
    assert mg.block_tables[0].block_table.cpu.shape[1] == 8  # 128-token 对齐
    # NONE 组不对齐（4 块×32 token）→ 宽 4，再乘 BlockTable 内的细分乘子 2。
    assert mg.block_tables[1].block_table.cpu.shape[1] == 4 * 2
    assert mg[0].slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT
    assert mg[1].slot_mapping_mode == SlotMappingMode.NONE
    mg.append_row(([1, 2], [30]), row_idx=0)
    assert mg[0].block_table.np[0, :2].tolist() == [1, 2]
    assert mg[1].block_table.np[0, 0] == 30 * 2  # 32 内存块拆两个 16 kernel 块
    mg.compute_slot_mapping(1, torch.tensor([0, 4], dtype=torch.int32),
                            torch.arange(4, dtype=torch.int64))
    # 4 个 token 全在第 0 逻辑块（行内第 0 项=1）。
    assert int(mg[0].slot_mapping.gpu[3]) == 1 * 16 + 3
    assert bool((mg[1].slot_mapping.gpu == 0).all())  # NONE 组不算 slot


# ── m9 后端公共块尺寸：MultipleOf(16) 与 int 候选取公共 ─────────────────────
class _FALikeBackend:
    # 真实 FlashAttentionBackend 的两个语义位（flash_attn.py:L82-L86）。
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


class _DefaultBackend:
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]


def test_select_common_block_size():
    fa = _FALikeBackend
    # Case 1：管理块尺寸被全部后端支持 → 直接返回。
    assert select_common_block_size(16, [fa]) == 16
    assert select_common_block_size(32, [fa]) == 32
    # Case 1 混合后端：64 被 FA（64%16==0）与 int 后端（==64）都支持。
    assert select_common_block_size(64, [_FALikeBackend, _IntSizesBackend]) == 64
    # Case 2：96 不被 int 后端支持 → 候选降序（64 不整除 96 跳过）→ 32。
    assert select_common_block_size(96, [_FALikeBackend, _IntSizesBackend]) == 32
    with pytest_raises(ValueError):
        select_common_block_size(31, [_FALikeBackend, _IntSizesBackend])


class _IntSizesBackend:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32, 64]


def test_prepare_kernel_block_sizes_attention_and_mamba():
    cfg = make_kv_cache_config(
        groups=[(LAYER0, 16, "full"), (MAMBA1, 1, "mamba")]
    )
    attn_groups = [[make_attn_group(_FALikeBackend, [LAYER0], 0)], []]
    sizes = prepare_kernel_block_sizes(cfg, attn_groups)
    # full 组：16 恰被 FA 支持 → 16；mamba 组：不拆块、原样 block_size。
    assert sizes == [16, 1]


def test_fa_backend_semantics_bits():
    # 真实类的两个语义位（m7 裁决源 / m9 约束源）。
    assert FlashAttentionBackend.forward_includes_kv_cache_update is False
    assert AttentionBackend.forward_includes_kv_cache_update is True
    sizes = FlashAttentionBackend.get_supported_kernel_block_sizes()
    assert len(sizes) == 1 and isinstance(sizes[0], MultipleOf) and sizes[0].base == 16


# ── 站1/站8/站9：runner 切面（_update_states 块表线 + _prepare_inputs +
#    _get_slot_mappings 双口径 + _build_attention_metadata 收束）───────────
def _make_runner(num_tokens_buf=128, fa_separate=True):
    backend_cls = _FALikeBackend if fa_separate else _DefaultBackend
    cfg = VllmConfigSeam(num_seqs=8, max_batched_tokens=num_tokens_buf,
                         max_model_len=512)
    kv_cfg = make_kv_cache_config(groups=[(LAYER0, 16, "full"), (LAYER1, 16, "full")])
    return GPUModelRunner(
        vllm_config=cfg,
        kv_cache_config=kv_cfg,
        attn_groups=[[make_attn_group(backend_cls, [LAYER0], 0)],
                     [make_attn_group(backend_cls, [LAYER1], 0)]],
        device=CPU,
    )


class _ReqState:
    def __init__(self, req_id, block_ids):
        self.req_id = req_id
        self.block_ids = block_ids  # 每组一个 list
        self.num_computed_tokens = 0
        self.output_token_ids: list[int] = []
        self.num_tokens = 0


def _prep_two_reqs(runner, sched=(100, 20), computed=(0, 100),
                   new_block_ids=([4, 9], [5])):
    """摆两个在跑请求（预置块表行）→ _update_states → _prepare_inputs。"""
    runner.input_batch.num_reqs = 2
    runner.input_batch.req_ids = ["r0", "r1"]
    runner.input_batch.req_id_to_index = {"r0": 0, "r1": 1}
    runner.input_batch.num_computed_tokens_cpu[: len(computed)] = computed
    runner.input_batch.num_prompt_tokens_cpu_tensor[:2] = torch.tensor(
        [100, 120], dtype=torch.int32)
    runner.input_batch.block_table[0].add_row([3, 8, 2], 0)
    runner.input_batch.block_table[1].add_row([3, 8, 2], 0)
    # r1 已算 100 token：row 需覆盖 pos 100..119 所在的逻辑块 6/7（8 项）。
    runner.input_batch.block_table[0].add_row([7, 1, 5, 2, 9, 4, 6, 8], 1)
    runner.input_batch.block_table[1].add_row([7, 1, 5, 2, 9, 4, 6, 8], 1)
    # runner 配置是两个 KV 组：block_ids / new_block_ids 都按「每请求×每组」
    # 两层嵌套（两组表数据同构——断言面在组 0）。
    runner.requests = {
        "r0": _ReqState("r0", [[3, 8, 2], [3, 8, 2]]),
        "r1": _ReqState("r1", [[7, 1, 5, 2, 9, 4, 6, 8], [7, 1, 5, 2, 9, 4, 6, 8]]),
    }
    so = SchedulerOutputSeam(
        total_num_scheduled_tokens=sum(sched),
        num_scheduled_tokens={"r0": sched[0], "r1": sched[1]},
        req_ids=["r0", "r1"],
        # 真实协议：new_block_ids 每请求**每组**一个 list（CachedRequestData
        # 的 list[list[int]] 形状——单组即一层嵌套）。
        new_block_ids=[
            [list(new_block_ids[0]), list(new_block_ids[0])],
            [list(new_block_ids[1]), list(new_block_ids[1])],
        ],
        num_computed_tokens=list(computed),
        num_output_tokens=[0, 100],
    )
    runner._update_states(so)
    return runner._prepare_inputs(so, np.array(sched, dtype=np.int32))


def test_update_states_appends_block_line():
    runner = _make_runner()
    _prep_two_reqs(runner)
    # r0 差量追加 [4,9]、r1 追加 [5]——块号过线落 CPU 行。
    bt = runner.input_batch.block_table[0]
    assert bt.block_table.np[0, :5].tolist() == [3, 8, 2, 4, 9]
    assert bt.num_blocks_per_row[0] == 5
    assert bt.block_table.np[1, :3].tolist() == [7, 1, 5]
    assert bt.num_blocks_per_row[1] == 9  # 预置 8 项 + 差量追加 [5]
    # req_state.block_ids 同步 extend（差量调和的另一半）。
    assert runner.requests["r0"].block_ids[0] == [3, 8, 2, 4, 9]


def test_prepare_inputs_commit_first_and_gpu_positions():
    runner = _make_runner()
    logits_indices, spec = _prep_two_reqs(runner)
    assert spec is None
    # GPU 端 positions：num_computed[req_indices_gpu] + query_pos.gpu（站4）。
    positions = runner.positions[:120]
    assert int(positions[0]) == 0 and int(positions[99]) == 99
    assert int(positions[100]) == 100 and int(positions[119]) == 119
    # seq_lens = num_computed + scheduled；尾部清 0。
    assert runner.seq_lens[:2].tolist() == [100, 120]
    assert bool((runner.seq_lens[2:] == 0).all())
    # query_start_loc 非递减 pad（四件套之一）：[0,100,120,120,...]。
    qsl = runner.query_start_loc.gpu
    assert qsl[:3].tolist() == [0, 100, 120]
    assert bool((qsl[3:] == 120).all())
    # logits_indices = query_start_loc[1:] - 1（非 spec 支）。
    assert logits_indices.tolist() == [99, 119]
    # compute_slot_mapping 已派发：块表前缀恒等式成立（换算输入全程 GPU 张量）。
    slots = runner.input_batch.block_table[0].slot_mapping.gpu
    row = runner.input_batch.block_table[0].block_table.np
    # r0 的 token 99 与 r1 的 token 100/119（位置即 100/119——GPU positions）
    assert int(slots[99]) == int(row[0][99 // 16]) * 16 + 99 % 16
    assert int(slots[100]) == int(row[1][100 // 16]) * 16 + 100 % 16
    assert int(slots[119]) == int(row[1][119 // 16]) * 16 + 119 % 16
    # 尾部 PAD 由 kernel 的第 num_reqs+1 个 program 重填。
    assert bool((slots[120:128] == PAD_SLOT_ID).all())


def test_get_slot_mappings_padded_tail_fill_and_by_layer():
    runner = _make_runner()
    _prep_two_reqs(runner)  # 120 真 token
    by_gid, by_layer = runner._get_slot_mappings(
        num_tokens_padded=128, num_reqs_padded=8, num_tokens_unpadded=120
    )
    sm = by_gid[0]
    assert sm.shape[0] == 128
    assert bool((sm[120:128] == -1).all())  # 尾段 fill_(-1)（FULL cudagraph 用）
    assert int(sm[99]) == int(runner.input_batch.block_table[0].slot_mapping.gpu[99])
    # by-layer dict 供 ForwardContext 逐层取用：组 0 的层拿组 0 的表、
    # 组 1 的层拿组 1 的表（两组表值相同但张量各自独立——每组一张）。
    assert by_layer[LAYER0] is sm
    assert by_layer[LAYER1] is by_gid[1]
    assert by_layer[LAYER1] is not sm


def test_build_attention_metadata_null_block_tail_and_fa_metadata():
    runner = _make_runner()
    _prep_two_reqs(runner)
    by_gid, _ = runner._get_slot_mappings(
        num_tokens_padded=128, num_reqs_padded=8, num_tokens_unpadded=120
    )
    attn_md, spec_common = runner._build_attention_metadata(
        num_tokens=120, num_reqs=2, max_query_len=100,
        num_tokens_padded=128, num_reqs_padded=8,
        slot_mappings=by_gid,
    )
    assert spec_common is None
    md = attn_md[LAYER0]
    assert isinstance(md, FlashAttentionMetadata)
    # 块表尾行 [2,8) 填 NULL_BLOCK_ID=0（Block 0 is reserved for padding）。
    assert md.block_table.shape[0] == 8
    assert bool((md.block_table[2:] == 0).all())
    assert bool((md.block_table[:2] != 0).any())
    # 读腿表 + 写腿索引一起过桥（站10 收束）。
    assert md.slot_mapping.shape[0] == 128
    assert md.num_actual_tokens == 128 and md.max_query_len == 100
    # 块表张量就是块表缓冲的 GPU 镜像前缀（固定地址）。
    assert md.block_table.data_ptr() == (
        runner.input_batch.block_table[0].block_table.gpu.data_ptr())


def test_positions_tail_zeroed_in_preprocess():
    runner = _make_runner()
    so = SchedulerOutputSeam(total_num_scheduled_tokens=120)
    _prep_two_reqs(runner)  # 120 真 token
    runner.positions[120:128] = 999  # 模拟上一拍残留
    (input_ids, inputs_embeds, positions, _it, _kw, _ec) = runner._preprocess(so, 128)
    assert inputs_embeds is None
    assert positions.data_ptr() == runner.positions.data_ptr()
    assert bool((runner.positions[120:128] == 0).all())  # 四件套：positions 尾清零


def test_has_separate_kv_update_oracle():
    # has_separate_kv_update = 存在后端 forward_includes_kv_cache_update=False
    # （FA 系）→ True；默认后端全 True → False。
    assert _FALikeBackend.forward_includes_kv_cache_update is False
    assert _DefaultBackend.forward_includes_kv_cache_update is True


def test_execute_model_assembles_slot_mappings_and_forward_context():
    runner = _make_runner()
    # 注册一个 Attention 层进 static_forward_context（ch19 域装配位）——
    # 前向 seam 按真实调用序跑两算子（写腿 → 读腿）。
    pool = _make_pool()
    pool.fill_(1.0)  # 哨兵：写腿覆盖处归零、未触及处保持 1
    layer = _AttentionLayer(pool)
    runner.vllm_config.compilation_config.static_forward_context = {LAYER0: layer}
    # FULL cudagraph 口径（pad_attn=True）：slot_mappings 用 padded 维度。
    runner.seam_batch_desc = BatchDescriptor(num_tokens=128, num_reqs=8)
    runner.seam_cudagraph_mode = CUDAGraphMode.FULL
    _prep_two_reqs(runner, sched=(100, 20), computed=(0, 100),
                   new_block_ids=([4], []))
    # 上一拍残留（持久 buffer 的地址前提）。
    runner.positions[120:128] = 999
    so = SchedulerOutputSeam(
        total_num_scheduled_tokens=120,
        num_scheduled_tokens={"r0": 100, "r1": 20},
        req_ids=["r0", "r1"], new_block_ids=[[[4], [4]], [[], []]],
        num_computed_tokens=[0, 100], num_output_tokens=[0, 100],
    )
    out = runner.execute_model(so)
    assert out is None  # 两段式契约：execute_model 返回 None
    sm = runner.execute_model_state.slot_mappings
    assert sm[LAYER0].shape[0] == 128
    assert bool((sm[LAYER0][120:] == -1).all())
    # positions 尾部清零在 _preprocess 内发生（四件套收口）。
    assert bool((runner.positions[120:128] == 0).all())
    # set_forward_context 已把 by-layer dict 推进 ForwardContext（站11）——
    # 前向内按 layer_name 取表（seam 模型前向的观测位记录）。
    assert runner.seam_seen_slot_mapping is sm[LAYER0]
    assert runner.seam_seen_layer_name == LAYER0
    # 写腿在前向内真跑了：r0 的第 0 逻辑块（物理块 3）16 槽被 zero-K/V 散写
    # （槽位来自 slot_mapping）；不在任何块表行里的块（如 10）未被触及。
    assert bool((pool[3] == 0).all())
    assert bool((pool[10] == 1).all())


# ── 站12/13 写腿：unified_kv_cache_update → reshape_and_cache_flash ────────
def _make_pool(num_blocks=32, block_size=16, kv_heads=1, head_dim=8):
    # FA 主流布局 [num_blocks, num_kv_heads, block_size, 2*head_dim]（flash_attn
    # forward docstring 原文），K/V 打进内容维。
    return torch.zeros(num_blocks, kv_heads, block_size, 2 * head_dim,
                       dtype=torch.float32)


class _AttentionLayer:
    # 真实 Attention 层的缩放因子是常驻张量（forward 的 descale 展开读它们）。
    _k_scale = torch.tensor(1.0)
    _v_scale = torch.tensor(1.0)
    _q_scale = torch.tensor(1.0)

    def __init__(self, kv_cache, num_heads=2):
        self.kv_cache = kv_cache
        self.impl = FlashAttentionImpl(
            num_heads=num_heads, head_size=8, scale=1.0 / math.sqrt(8),
            num_kv_heads=1, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype="auto")


def test_reshape_and_cache_flash_skips_pad_and_decomposes_slot():
    pool = _make_pool()
    layer = _AttentionLayer(pool)
    key = torch.arange(4 * 1 * 8, dtype=torch.float32).reshape(4, 1, 8) / 8.0
    value = key + 100.0
    slot_mapping = torch.tensor([16 * 3 + 2, -1, 16 * 3 + 3, 16 * 3 + 4],
                                dtype=torch.int64)
    key_cache, value_cache = pool.transpose(1, 2).split(8, dim=-1)
    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping,
                            "auto", None, None)
    # token1 是 PAD（slot<0）→ 直接 return 不写；slot 50=块3偏移2 → 落页 3 行 2。
    assert bool((key_cache[3, 2] == key[0, 0]).all())
    assert bool((key_cache[3, 3] == key[2, 0]).all())
    assert bool((value_cache[3, 4] == value[3, 0]).all())
    assert bool((key_cache[3, 1] == 0).all())  # PAD token 没写这里
    # slot_mapping 的形状决定 token 数（woosuk NOTE）。
    reshape_and_cache_flash(key[:2], value[:2], key_cache, value_cache,
                            torch.tensor([16 * 3 + 5], dtype=torch.int64),
                            "auto", None, None)
    assert bool((key_cache[3, 5] == key[0, 0]).all())
    assert bool((key_cache[3, 6] == 0).all())


def test_write_leg_through_forward_context_ops():
    pool = _make_pool()
    layer = _AttentionLayer(pool)
    key = torch.ones(3, 1, 8) * 2.0
    value = torch.ones(3, 1, 8) * 3.0
    slot_mapping = torch.tensor([16 * 2, 16 * 2 + 1, 16 * 2 + 2], dtype=torch.int64)
    fc_slot_mappings = {LAYER0: slot_mapping}
    cfg = VllmConfigSeam(num_seqs=8, max_batched_tokens=128, max_model_len=512)
    cfg.compilation_config.static_forward_context = {LAYER0: layer}
    with set_forward_context({LAYER0: None}, cfg, slot_mapping=fc_slot_mappings):
        attn_md, attn_layer, kv_cache, layer_sm = get_attention_context(LAYER0)
        assert attn_layer is layer and kv_cache is pool
        assert layer_sm is slot_mapping
        dummy = unified_kv_cache_update(key, value, LAYER0)
        # 返回空张量作 dummy 数据依赖（保 torch.compile 顺序）。
        assert dummy.numel() == 0 and dummy.dtype == key.dtype
    key_cache, _ = pool.transpose(1, 2).split(8, dim=-1)
    assert bool((key_cache[2, 0] == 2.0).all())
    assert bool((key_cache[2, 2] == 2.0).all())


# ── 站14/F7 读腿：flash_attn 穿 block_table 间接寻址 —— 写直读间闭合 ───────
def test_read_leg_through_block_table_matches_dense_reference():
    torch.manual_seed(0)
    block_size, num_kv_heads, head_dim, num_heads = 16, 1, 8, 2
    pool = _make_pool(num_blocks=32, block_size=block_size, kv_heads=num_kv_heads,
                      head_dim=head_dim)
    layer = _AttentionLayer(pool, num_heads=num_heads)
    # 请求 0：块表行 [3, 8, 6]——前 32 个历史 token 落块 3/8，本拍 2 个新
    # token（pos 32/33）落逻辑块 2 → 物理块 6（slot = 6*16+{0,1}）。
    block_table = torch.tensor([[3, 8, 6]], dtype=torch.int32)
    q = torch.randn(2, num_heads, head_dim)
    k = torch.randn(2, num_kv_heads, head_dim)
    v = torch.randn(2, num_kv_heads, head_dim)
    slot_mapping = torch.tensor(
        [6 * block_size + 0, 6 * block_size + 1], dtype=torch.int64)
    md = FlashAttentionMetadata(
        num_actual_tokens=2, max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=34, seq_lens=torch.tensor([34], dtype=torch.int32),
        block_table=block_table, slot_mapping=slot_mapping,
        use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
        prefix_kv_lens=None, suffix_kv_lens=None,
    )
    # 先写腿：新 token 的 K/V 落进块 6 的前两槽（slot 直寻址散写）。
    layer.impl.do_kv_cache_update(layer, k, v, pool, slot_mapping)
    # 再读腿：attention 穿表间接寻址读历史 + 新写。
    out = torch.zeros(2, num_heads * head_dim)
    layer.impl.forward(layer, q, k, v, pool, md, output=out)
    # 稠密参照：把块 3/8/6 的 K/V 按块表顺序拼回逻辑序列，直接算 attention。
    key_cache, value_cache = pool.transpose(1, 2).split(head_dim, dim=-1)
    k_hist = torch.cat(
        [key_cache[3][:16], key_cache[8][:16], key_cache[6][:2]], dim=0
    ).reshape(-1, head_dim)
    v_hist = torch.cat(
        [value_cache[3][:16], value_cache[8][:16], value_cache[6][:2]], dim=0
    ).reshape(-1, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    for qi in range(2):
        q_i = q[qi].reshape(num_heads, head_dim)[0]
        logits = (k_hist @ q_i) * scale
        logits = logits[: 32 + qi + 1]  # causal
        p = torch.softmax(logits, dim=-1)
        expect = p @ v_hist[: 32 + qi + 1]
        got = out[qi].reshape(num_heads, head_dim)[0]
        assert torch.allclose(got, expect, atol=1e-5), (got, expect)


def test_unified_attention_with_output_reads_through_table():
    torch.manual_seed(1)
    pool = _make_pool()
    layer = _AttentionLayer(pool)
    slot_mapping = torch.tensor([16 * 5 + 0, 16 * 5 + 1], dtype=torch.int64)
    q = torch.randn(2, 2, 8)
    k = torch.randn(2, 1, 8)
    v = torch.randn(2, 1, 8)
    block_table = torch.tensor([[5]], dtype=torch.int32)
    md = FlashAttentionMetadata(
        num_actual_tokens=2, max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=2, seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=block_table, slot_mapping=slot_mapping,
        use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
        prefix_kv_lens=None, suffix_kv_lens=None,
    )
    sm = {LAYER0: slot_mapping}
    cfg = VllmConfigSeam(num_seqs=8, max_batched_tokens=128, max_model_len=512)
    cfg.compilation_config.static_forward_context = {LAYER0: layer}
    out = torch.zeros(2, 16)
    with set_forward_context({LAYER0: md}, cfg, slot_mapping=sm):
        dummy = unified_kv_cache_update(k, v, LAYER0)
        # dummy 作为 kv_cache_dummy_dep 传入 attention 算子——数据依赖保序。
        unified_attention_with_output(q, k, v, out, LAYER0,
                                      kv_cache_dummy_dep=dummy)
    # 写腿已把 K/V 落页，读腿穿表读回——两 token 的输出非零且有限。
    assert torch.isfinite(out).all() and float(out.abs().sum()) > 0


# ── WC2 D2H 之忌的成文纪律：deprecated 属性措辞 ────────────────────────────
def test_deprecated_cpu_props_warn_d2h_discipline():
    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        num_reqs=2, num_actual_tokens=5, max_query_len=3, max_seq_len=7,
        block_table_tensor=torch.zeros(2, 4, dtype=torch.int32),
        slot_mapping=torch.zeros(5, dtype=torch.int64),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seq_lens_cpu = cm.seq_lens_cpu
        nct_cpu = cm.num_computed_tokens_cpu
    joined = " ".join(str(x.message) for x in w)
    assert "avoid implicit H<>D sync" in joined
    assert "breaks full" in joined  # num_computed_tokens_cpu 措辞的后半句
    assert seq_lens_cpu.tolist() == [5, 7]
    # num_computed_tokens_cpu = seq_lens_cpu - query_lens（CPU 端推导）。
    assert nct_cpu.tolist() == [5 - 2, 7 - 3]


# ── 站1 配套：may_reinitialize_input_batch 按 spec kind 装配模式 ────────────
def test_may_reinitialize_input_batch_modes():
    runner = _make_runner()
    new_cfg = make_kv_cache_config(
        groups=[(LAYER0, 16, "full"), (MAMBA1, 1, "mamba")]
    )
    assert get_kv_cache_spec_kind(
        new_cfg.kv_cache_groups[0].kv_cache_spec).value == "full_attention"
    assert get_kv_cache_spec_kind(
        new_cfg.kv_cache_groups[1].kv_cache_spec).value == "mamba"
    assert isinstance(new_cfg.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec)
    assert isinstance(new_cfg.kv_cache_groups[1].kv_cache_spec, MambaSpec)
    runner.may_reinitialize_input_batch(new_cfg, [16, 1])
    modes = runner._init_slot_mapping_modes
    assert modes == [SlotMappingMode.TOKEN_TO_KV_SLOT, SlotMappingMode.NONE]
    # 块表随之重建：full 组 128-token 对齐、mamba 组 NONE。
    assert runner.input_batch.block_table[1].slot_mapping_mode == SlotMappingMode.NONE


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
