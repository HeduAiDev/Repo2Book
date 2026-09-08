# ch23 m7 驱动脚本 —— 分片 weight_loader 三型（QKV / Merged / Row）
# （linear.py:L1222-L1400 / L746-L889 / L1717-L1737）
# 跑精简版（implementation/，只做减法、三型分片算术主干逐字）取真实数值轨迹。
# 参考 = 测试电池 tests/test_model_layer_assembly.py::TestTPShardMath（同一注入位）。
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "implementation"))

import vllm.distributed as dist  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

VC = VllmConfig(
    model_config=ModelConfig(dtype=torch.float32),
    cache_config=CacheConfig(),
    load_config=LoadConfig(load_format="auto"),
    device_config=DeviceConfig(device="cpu"),
    compilation_config=CompilationConfig(),
    parallel_config=ParallelConfig(),
)

out = {"mechanism": "m7", "source": "run_m7.py @ implementation/ (vLLM v0.27.1 只做减法精简版, host CPU)"}


def mark_ranges(w_global):
    """全局权重每行打上自己的行号标记（乘 1 网格）——用于落地校验。"""
    r = torch.arange(w_global.shape[0], dtype=torch.float32).unsqueeze(1)
    return r @ torch.ones(1, w_global.shape[1])


# ── 型一 QKV：partition 分支（8q/2kv, head=16, hidden=64, tp=2, rank=1）──
dist._TP_STATE.update(rank=1, size=2, pp_rank=0, pp_size=1)
with set_current_vllm_config(VC):
    qkv = QKVParallelLinear(
        hidden_size=64, head_size=16,
        total_num_heads=8, total_num_kv_heads=2,
        bias=False, prefix="model.layers.0.self_attn.qkv_proj",
    )
q_global = mark_ranges(torch.zeros(8 * 16, 64))            # 全局 q：行号 0..127
k_global = mark_ranges(torch.zeros(2 * 16, 64)) + 1000.0   # 全局 k：1000..1031
v_global = mark_ranges(torch.zeros(2 * 16, 64)) + 2000.0   # 全局 v：2000..2031
qkv.weight_loader(qkv.weight, q_global.clone(), loaded_shard_id="q")
qkv.weight_loader(qkv.weight, k_global.clone(), loaded_shard_id="k")
qkv.weight_loader(qkv.weight, v_global.clone(), loaded_shard_id="v")
w = qkv.weight.data
om = qkv._get_shard_offset_mapping
out["type1_QKV_partition_8q_2kv_tp2_rank1"] = {
    "layer_attrs": {
        "num_heads": qkv.num_heads, "num_kv_heads": qkv.num_kv_heads,
        "num_kv_head_replicas": qkv.num_kv_head_replicas,
        "tp_rank": qkv.tp_rank, "tp_size": qkv.tp_size,
        "weight_shape_per_rank": list(w.shape),
        "global_q_rows": q_global.shape[0],
        "global_k_rows": k_global.shape[0],
        "global_v_rows": v_global.shape[0],
    },
    "shard_q": {
        "shard_offset": om("q"), "shard_size": qkv._get_shard_size_mapping("q"),
        "shard_rank": qkv.tp_rank, "start_idx": qkv.tp_rank * qkv._get_shard_size_mapping("q"),
        "global_rows_taken": [64, 128],
        "lands_in_fused_rows": [0, 64],
        "verify_w_equals_global_slice": bool(torch.equal(w[0:64], q_global[64:128])),
    },
    "shard_k": {
        "shard_offset": om("k"), "shard_size": qkv._get_shard_size_mapping("k"),
        "shard_rank": qkv.tp_rank // qkv.num_kv_head_replicas,
        "start_idx": (qkv.tp_rank // qkv.num_kv_head_replicas) * qkv._get_shard_size_mapping("k"),
        "global_rows_taken": [16, 32],
        "lands_in_fused_rows": [64, 80],
        "verify_w_equals_global_slice": bool(torch.equal(w[64:80], k_global[16:32])),
    },
    "shard_v": {
        "shard_offset": om("v"), "shard_size": qkv._get_shard_size_mapping("v"),
        "shard_rank": qkv.tp_rank // qkv.num_kv_head_replicas,
        "start_idx": (qkv.tp_rank // qkv.num_kv_head_replicas) * qkv._get_shard_size_mapping("v"),
        "global_rows_taken": [16, 32],
        "lands_in_fused_rows": [80, 96],
        "verify_w_equals_global_slice": bool(torch.equal(w[80:96], v_global[16:32])),
    },
    "fused_rows_total": int(w.shape[0]),
    "first_row_value_of_each_segment": [float(w[0, 0]), float(w[64, 0]), float(w[80, 0])],
}

# ── 型一 QKV：replicate 分支（8q/1kv → replicas=2）——GQA KV 头复制的装载语义 ──
dist._TP_STATE.update(rank=1, size=2, pp_rank=0, pp_size=1)
with set_current_vllm_config(VC):
    qkv_rep = QKVParallelLinear(
        hidden_size=64, head_size=16,
        total_num_heads=8, total_num_kv_heads=1,
        bias=False, prefix="model.layers.0.self_attn.qkv_proj",
    )
k1_global = mark_ranges(torch.zeros(1 * 16, 64)) + 5000.0  # 全局 k：5000..5015
qkv_rep.weight_loader(qkv_rep.weight, torch.zeros(8 * 16, 64), loaded_shard_id="q")
qkv_rep.weight_loader(qkv_rep.weight, k1_global.clone(), loaded_shard_id="k")
w_rep = qkv_rep.weight.data
replicas = qkv_rep.num_kv_head_replicas
out["type1_QKV_replicate_8q_1kv_tp2"] = {
    "num_kv_head_replicas": replicas,
    "num_kv_heads_per_rank": qkv_rep.num_kv_heads,
    "k_shard_rank_rank0": 0 // replicas,
    "k_shard_rank_rank1": 1 // replicas,
    "k_start_idx_rank1": (1 // replicas) * qkv_rep._get_shard_size_mapping("k"),
    "k_global_rows_taken_rank1": [0, 16],
    "lands_in_fused_rows": [64, 80],
    "verify_rank1_w_equals_global_first_segment": bool(torch.equal(w_rep[64:80], k1_global[0:16])),
    "replication_semantics": "两个 rank 的 shard_rank 都 = tp_rank//replicas = 0 → 同一段 k 权重被复制进两张卡",
}

# ── 型二 Merged：gate/up 两段（output_sizes=[32,32], tp=2, rank=1）──
dist._TP_STATE.update(rank=1, size=2, pp_rank=0, pp_size=1)
with set_current_vllm_config(VC):
    merged = MergedColumnParallelLinear(
        input_size=64, output_sizes=[32, 32], bias=False, prefix="model.layers.0.mlp.gate_up_proj",
    )
gate = mark_ranges(torch.zeros(32, 64)) + 1.0    # 行号+1：1..32
up = mark_ranges(torch.zeros(32, 64)) + 2.0      # 行号+2：2..33
merged.weight_loader(merged.weight, gate.clone(), loaded_shard_id=0)
merged.weight_loader(merged.weight, up.clone(), loaded_shard_id=1)
mw = merged.weight.data
out["type2_Merged_gate_up_tp2_rank1"] = {
    "layer_attrs": {
        "output_sizes": merged.output_sizes,
        "tp_rank": merged.tp_rank, "tp_size": merged.tp_size,
        "weight_shape_per_rank": list(mw.shape),
    },
    "shard_gate_id0": {
        "shard_offset_sum_then_div_tp": sum(merged.output_sizes[:0]) // 2,
        "shard_size_div_tp": merged.output_sizes[0] // 2,
        "start_idx": merged.tp_rank * (merged.output_sizes[0] // 2),
        "global_rows_taken": [16, 32],
        "lands_in_fused_rows": [0, 16],
        "verify_w_equals_global_slice": bool(torch.equal(mw[0:16], gate[16:32])),
    },
    "shard_up_id1": {
        "shard_offset_sum_then_div_tp": sum(merged.output_sizes[:1]) // 2,
        "shard_size_div_tp": merged.output_sizes[1] // 2,
        "start_idx": merged.tp_rank * (merged.output_sizes[1] // 2),
        "global_rows_taken": [16, 32],
        "lands_in_fused_rows": [16, 32],
        "verify_w_equals_global_slice": bool(torch.equal(mw[16:32], up[16:32])),
    },
}

# ── 型三 Row：沿 input 维行切（o_proj 形状 [32,64]，tp=2, rank=1）──
dist._TP_STATE.update(rank=1, size=2, pp_rank=0, pp_size=1)
with set_current_vllm_config(VC):
    row = RowParallelLinear(input_size=64, output_size=32, bias=False, prefix="model.layers.0.self_attn.o_proj")
row_global = mark_ranges(torch.zeros(32, 64)).T.contiguous().T * 0  # placeholder
# 列号标记：每列打上自己的列号（input 维 = dim1）
col_ids = torch.arange(64, dtype=torch.float32)
row_global = torch.ones(32, 64) * col_ids.unsqueeze(0)   # 元素值 = 所在列号 0..63
row.weight_loader(row.weight, row_global.clone())
rw = row.weight.data
out["type3_Row_input_dim_tp2_rank1"] = {
    "layer_attrs": {
        "input_size": row.input_size, "input_size_per_partition": row.input_size_per_partition,
        "output_size": row.output_size,
        "tp_rank": row.tp_rank, "tp_size": row.tp_size,
        "weight_shape_per_rank": list(rw.shape),
    },
    "slice": {
        "input_dim": 1,
        "shard_size": rw.shape[1],
        "start_idx": row.tp_rank * rw.shape[1],
        "global_cols_taken": [32, 64],
        "verify_w_equals_global_slice": bool(torch.equal(rw, row_global[:, 32:64])),
        "verify_min_col_value": float(rw.min()),
        "verify_max_col_value": float(rw.max()),
    },
    "forward_note": "行切的计算侧对偶：bias 只 rank0 加（bias_= None if tp_rank>0）、reduce_results 时 all_reduce（tp=1 恒等旁路）",
}

# ── 收尾：三型一致的字段名对照（供 figure/正文统一术语）──
out["shard_rank_formulas"] = {
    "QKV_q": "shard_rank = tp_rank",
    "QKV_k_v": "shard_rank = tp_rank // num_kv_head_replicas",
    "Merged": "start_idx = tp_rank * (output_sizes[id] // tp_size)",
    "Row": "start_idx = tp_rank * shard_size（shard_size = 本秩 input 行数）",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m7.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m7.json written:",
      "QKV-partition all-verified:",
      out["type1_QKV_partition_8q_2kv_tp2_rank1"]["shard_q"]["verify_w_equals_global_slice"]
      and out["type1_QKV_partition_8q_2kv_tp2_rank1"]["shard_k"]["verify_w_equals_global_slice"]
      and out["type1_QKV_partition_8q_2kv_tp2_rank1"]["shard_v"]["verify_w_equals_global_slice"],
      "replicate:", out["type1_QKV_replicate_8q_1kv_tp2"]["verify_rank1_w_equals_global_first_segment"],
      "merged:", out["type2_Merged_gate_up_tp2_rank1"]["shard_gate_id0"]["verify_w_equals_global_slice"]
      and out["type2_Merged_gate_up_tp2_rank1"]["shard_up_id1"]["verify_w_equals_global_slice"],
      "row:", out["type3_Row_input_dim_tp2_rank1"]["slice"]["verify_w_equals_global_slice"])
