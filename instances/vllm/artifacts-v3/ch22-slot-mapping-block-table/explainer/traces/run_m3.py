# ch22 explainer 驱动脚本 · m3 PAD 语义三处分工
# 两段取证：
#   A. BlockTable 级两拍残留实验——持久 buffer 尾部在拍间残留上一拍真数据，
#      PAD 程序每拍重填 [num_tokens, max) 为 -1；
#   B. runner 级 FULL cudagraph 一拍——padding 四件套全量装配：
#      ①kernel PAD 尾 + ②_get_slot_mappings 尾段 fill_(-1)
#      ③块表尾行 NULL_BLOCK_ID=0 ④query_start_loc 非递减 + positions 尾清零。
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import (  # noqa: E402
    CUDAGraphMode,
    SchedulerOutputSeam,
    VllmConfigSeam,
    make_attn_group,
    make_kv_cache_config,
)
from implementation.block_table import BlockTable, PAD_SLOT_ID  # noqa: E402
from implementation.flash_attn import FlashAttentionImpl  # noqa: E402
from implementation.forward_context import BatchDescriptor  # noqa: E402
from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402

CPU = torch.device("cpu")

# ── A. BlockTable 级两拍残留实验 ───────────────────────────────────────────
bt = BlockTable(
    block_size=16, max_num_reqs=8, max_num_blocks_per_req=16,
    max_num_batched_tokens=128, pin_memory=False, device=CPU,
    kernel_block_size=16, cp_kv_cache_interleave_size=1,
)
ROW_A = [4, 9, 8, 7, 6, 3, 5, 2]
bt.add_row(ROW_A, row_idx=0)
bt.commit_block_table(1)

# 拍 1：100 token。kernel 尾部 PAD 后 [100,128) 应全为 -1。
qsl_a1 = torch.tensor([0, 100], dtype=torch.int32)
bt.compute_slot_mapping(1, qsl_a1, torch.arange(100, dtype=torch.int64))
slots_a1 = [int(x) for x in bt.slot_mapping.gpu]

# 拍 2：只剩 10 token。拍前 dump [10,20) 的残留（上一拍的真 slot），
# 拍后应被 PAD 程序重填为 -1。
stale_before = [int(x) for x in bt.slot_mapping.np[10:20]]
qsl_a2 = torch.tensor([0, 10], dtype=torch.int32)
bt.compute_slot_mapping(1, qsl_a2, torch.arange(10, dtype=torch.int64))
slots_a2 = [int(x) for x in bt.slot_mapping.gpu]

# ── B. runner 级 FULL cudagraph 一拍（padding 四件套） ────────────────────
LAYER0 = "model.layers.0.self_attn.attn"
LAYER1 = "model.layers.1.self_attn.attn"


class _FALikeBackend:
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]


class _AttentionLayer:
    _k_scale = torch.tensor(1.0)
    _v_scale = torch.tensor(1.0)
    _q_scale = torch.tensor(1.0)

    def __init__(self, kv_cache, num_heads=2):
        self.kv_cache = kv_cache
        self.impl = FlashAttentionImpl(
            num_heads=num_heads, head_size=8, scale=1.0 / math.sqrt(8),
            num_kv_heads=1, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype="auto")


class _ReqState:
    def __init__(self, req_id, block_ids):
        self.req_id = req_id
        self.block_ids = block_ids
        self.num_computed_tokens = 0
        self.output_token_ids = []
        self.num_tokens = 0


cfg = VllmConfigSeam(num_seqs=8, max_batched_tokens=128, max_model_len=512)
kv_cfg = make_kv_cache_config(
    groups=[(LAYER0, 16, "full"), (LAYER1, 16, "full")])
runner = GPUModelRunner(
    vllm_config=cfg, kv_cache_config=kv_cfg,
    attn_groups=[[make_attn_group(_FALikeBackend, [LAYER0], 0)],
                 [make_attn_group(_FALikeBackend, [LAYER1], 0)]],
    device=CPU,
)
pool = torch.zeros(32, 1, 16, 16)
pool.fill_(1.0)
layer0 = _AttentionLayer(pool)
cfg.compilation_config.static_forward_context = {LAYER0: layer0}

ROW_R0 = [3, 8, 2, 7, 1, 5, 9, 4]
ROW_R1 = [7, 1, 5, 2, 9, 4, 6, 8]
runner.input_batch.num_reqs = 2
runner.input_batch.req_ids = ["r0", "r1"]
runner.input_batch.req_id_to_index = {"r0": 0, "r1": 1}
runner.input_batch.num_computed_tokens_cpu[:2] = [0, 100]
runner.input_batch.num_prompt_tokens_cpu_tensor[:2] = torch.tensor(
    [100, 120], dtype=torch.int32)
runner.input_batch.block_table[0].add_row(ROW_R0, 0)
runner.input_batch.block_table[1].add_row(ROW_R0, 0)
runner.input_batch.block_table[0].add_row(ROW_R1, 1)
runner.input_batch.block_table[1].add_row(ROW_R1, 1)
runner.requests = {
    "r0": _ReqState("r0", [list(ROW_R0), list(ROW_R0)]),
    "r1": _ReqState("r1", [list(ROW_R1), list(ROW_R1)]),
}

# FULL cudagraph 口径（ch19 域 seam 观测位直供——真实 BatchDescriptor 查表
# 命中 128 token × 8 reqs 的捕获形状）。
runner.seam_batch_desc = BatchDescriptor(num_tokens=128, num_reqs=8)
runner.seam_cudagraph_mode = CUDAGraphMode.FULL

# 模拟上一拍残留：positions 尾部是旧真值。
runner.positions[120:128] = 999

so = SchedulerOutputSeam(
    total_num_scheduled_tokens=120,
    num_scheduled_tokens={"r0": 100, "r1": 20},
    req_ids=["r0", "r1"],
    new_block_ids=[[[], []], [[], []]],  # 每请求×每组：本拍无新块（行已预置）
    num_computed_tokens=[0, 100],
    num_output_tokens=[0, 100],
)
out_ret = runner.execute_model(so)
assert out_ret is None  # 两段式契约：execute_model 返回 None

bt_gpu0 = runner.input_batch.block_table[0].block_table.gpu
sm_by_layer = runner.execute_model_state.slot_mappings[LAYER0]
qsl_gpu = runner.query_start_loc.gpu.tolist()

out = {
    "mechanism": "m3 PAD 语义三处分工（-1 token 侧 / 0 行侧 + 四件套）",
    "trace_source": "run（精简版 host 镜像 + runner 切面全链）",
    "pad_constants": {
        "PAD_SLOT_ID": PAD_SLOT_ID,
        "NULL_BLOCK_ID": 0,
        "PAD_SLOT_ID_anchor": "vllm/v1/attention/backends/utils.py:L45",
        "NULL_BLOCK_ID_anchor": "vllm/v1/attention/backends/utils.py:L46",
    },
    "A_blocktable_two_step_residual": {
        "params": {"block_size": 16, "max_num_batched_tokens": 128,
                   "row": ROW_A},
        "step1_100_tokens": {
            "num_tokens": 100, "padded_to": 128,
            "slot_tail_100_128": slots_a1[100:128],
            "slot_9": slots_a1[9], "slot_10": slots_a1[10],
        },
        "step2_10_tokens": {
            "num_tokens": 10,
            "stale_tail_before": stale_before,
            "slot_tail_10_128_after": slots_a2[10:128],
            "slot_9": slots_a2[9],
            "tail_all_pad": bool(all(x == -1 for x in slots_a2[10:128])),
        },
    },
    "B_runner_full_cudagraph": {
        "params": {
            "num_reqs": 2, "num_reqs_padded": 8,
            "num_tokens_unpadded": 120, "num_tokens_padded": 128,
            "cudagraph_mode": "FULL",
            "block_table_rows": [ROW_R0, ROW_R1],
        },
        "slot_mapping_padded_len": int(sm_by_layer.shape[0]),
        "slot_tail_120_128": [int(x) for x in sm_by_layer[120:128]],
        "slot_99": int(sm_by_layer[99]),
        "slot_119": int(sm_by_layer[119]),
        "block_table_gpu_shape": list(bt_gpu0.shape),
        "block_table_tail_rows_2_8": [
            [int(x) for x in bt_gpu0[r, :8]] for r in range(2, 8)],
        "block_table_active_rows": [
            [int(x) for x in bt_gpu0[r, :8]] for r in range(0, 2)],
        "query_start_loc_full": qsl_gpu,
        "positions_tail_120_128_after_preprocess": [
            int(x) for x in runner.positions[120:128]],
        "positions_stale_before": [999] * 8,
        "pad_slots_written_by_pool_zero": 8,
        "padded_rows_filled_null": 6,
        "kv_pool_blocks_untouched_stay_one": bool((pool[10] == 1.0).all()),
        "kv_pool_block3_written": bool((pool[3] == 0).all()),
    },
}
path = os.path.join(os.path.dirname(__file__), "m3.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m3 trace written:", path)
