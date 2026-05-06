"""Annotated runnable trace for Ch08 — Tensor Parallelism.

Five demos producing the verbatim numerics that the writer quotes
character-for-character (per the demo-numerics-verbatim hard gate, K17).

Run:
    python3 -m instances.vllm.artifacts.08-tensor-parallelism.implementation.demo

Sections:
    [1] Mathematical equivalence: column-parallel and row-parallel
        each reproduce the unsharded reference within fp32 tolerance.
    [2] Ring all-reduce α-β fit + step-by-step simulation correctness.
    [3] TP throughput / latency sweep across (tp_size × batch) on a
        Llama-7B-shaped MLP block; predicted vs measured all-reduce
        overhead from the α-β model.
    [4] GQA × TP boundary: KV head replication when tp_size exceeds the
        total KV head count. Memory floor demonstration.
    [5] End-to-end LlamaMLP TP correctness + per-forward collective count
        (the Megatron col→row pair = ONE all-reduce per block).

Numerics are organised under DEMO_RESULTS at the end so the writer can
quote them by key.

References:
- §1: linear.py:L410-L608 ColumnParallelLinear, L1394-L1577 RowParallelLinear
- §2: parallel_state.py:L502-L530 GroupCoordinator.all_reduce
- §3: comm_primitives.predict_block_overhead with α-β fits
- §4: linear.py:L1029-L1043 num_kv_head_replicas branch
- §5: llama.py:L81-L121 LlamaMLP
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np

from .tp_math import (
    column_parallel_forward,
    row_parallel_forward,
    column_then_row_block,
    verify_column_parallel_equivalence,
    verify_row_parallel_equivalence,
    verify_column_then_row_block,
)
from .comm_primitives import (
    AlphaBetaModel,
    HARDWARE_PROFILES,
    fit_alpha_beta,
    predict_block_overhead,
    ring_all_reduce_cost,
    simulate_all_reduce,
)
from .column_parallel import ColumnParallelLinear, MergedColumnParallelLinear
from .row_parallel import RowParallelLinear
from .qkv_parallel import QKVParallelLinear
from .mlp_block import LlamaMLPTP, reference_unsharded_mlp


def _hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Demo §1 — Mathematical equivalence test.
# ---------------------------------------------------------------------------

def demo_1_equivalence() -> dict:
    _hr("[1] Mathematical equivalence: TP forward == unsharded forward")
    out: dict = {}
    print("Column-parallel: split A along output dim, concat outputs.")
    for tp in (2, 4, 8):
        r = verify_column_parallel_equivalence(
            in_dim=128, out_dim=512, batch=4, tp_size=tp
        )
        print(f"  ColumnParallel  tp_size={tp}  max_abs_diff={r['max_abs_diff']:.3e}  "
              f"allclose={r['allclose']}  (tol={r['tolerance_used']:g})")
        out[f"col_tp{tp}_max_abs_diff"] = r["max_abs_diff"]
    print("\nRow-parallel: split A along input dim, X along last dim, sum partials.")
    for tp in (2, 4, 8):
        r = verify_row_parallel_equivalence(
            in_dim=128, out_dim=512, batch=4, tp_size=tp
        )
        print(f"  RowParallel     tp_size={tp}  max_abs_diff={r['max_abs_diff']:.3e}  "
              f"allclose={r['allclose']}  (tol={r['tolerance_used']:g})")
        out[f"row_tp{tp}_max_abs_diff"] = r["max_abs_diff"]
    print("\nColumn → Row composition (Megatron pair): one all-reduce per block.")
    for tp in (2, 4, 8):
        r = verify_column_then_row_block(
            hidden=128, ffn=512, batch=4, tp_size=tp
        )
        print(f"  Col→Row block   tp_size={tp}  num_collectives={r['num_collectives']}  "
              f"max_abs_diff={r['max_abs_diff']:.3e}  "
              f"allclose={r['allclose']}")
        out[f"colrow_tp{tp}_max_abs_diff"] = r["max_abs_diff"]
        out[f"colrow_tp{tp}_num_collectives"] = r["num_collectives"]
    return out


# ---------------------------------------------------------------------------
# Demo §2 — α-β microbench + ring all-reduce simulation.
# ---------------------------------------------------------------------------

def demo_2_alpha_beta() -> dict:
    _hr("[2] α-β model: fit (α, β) and verify ring simulation correctness")
    out: dict = {}
    # Step 1: simulate ring all-reduce produces the SAME result as a naive sum.
    rng = np.random.default_rng(7)
    P = 4
    base_shape = (16, 8)
    per_rank = [rng.standard_normal(base_shape).astype(np.float32) for _ in range(P)]
    target = np.sum(per_rank, axis=0)
    after = simulate_all_reduce(per_rank)
    sim_diff = max(float(np.max(np.abs(t - target))) for t in after)
    print(f"  Ring simulation P={P}  shape={base_shape}  "
          f"max diff vs naive sum = {sim_diff:.3e}  (must be 0)")
    out["ring_sim_max_diff"] = sim_diff

    # Step 2: synthesize "measurements" from a known model + noise, then fit.
    print("\n  α-β fit (synthetic noisy measurements from a known model):")
    true_ab = AlphaBetaModel(alpha_seconds=5e-6, beta_seconds_per_byte=1/(150e9))
    payloads = [1024, 16 * 1024, 256 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024]
    rng2 = np.random.default_rng(2)
    measured = [true_ab.predict(s) * (1 + rng2.normal(0, 0.02)) for s in payloads]
    fit = fit_alpha_beta(payloads, measured)
    print(f"    true:  α = {true_ab.alpha_seconds*1e6:6.2f} μs   "
          f"β bandwidth = {true_ab.bandwidth_GBps:6.1f} GB/s")
    print(f"    fit:   α = {fit.alpha_seconds*1e6:6.2f} μs   "
          f"β bandwidth = {fit.bandwidth_GBps:6.1f} GB/s")
    out["fit_alpha_us"] = fit.alpha_seconds * 1e6
    out["fit_bw_GBps"] = fit.bandwidth_GBps
    out["true_alpha_us"] = true_ab.alpha_seconds * 1e6
    out["true_bw_GBps"] = true_ab.bandwidth_GBps

    # Step 3: predict per-payload time using ring formula across P.
    print("\n  Ring all-reduce cost (NVLink_HSXM4 profile, P payloads × P ranks):")
    nvlink = HARDWARE_PROFILES["NVLink_HSXM4"]
    print(f"    α = {nvlink.alpha_seconds*1e6:.1f} μs, "
          f"β bandwidth = {nvlink.bandwidth_GBps:.0f} GB/s")
    out["nvlink_alpha_us"] = nvlink.alpha_seconds * 1e6
    out["nvlink_bw_GBps"] = nvlink.bandwidth_GBps
    print("    payload (B)        P=2          P=4          P=8")
    for s in payloads:
        row = [ring_all_reduce_cost(s, P_, nvlink) * 1e6 for P_ in (2, 4, 8)]
        print(f"    {s:>10d}  {row[0]:7.2f} μs  {row[1]:7.2f} μs  {row[2]:7.2f} μs")
        out[f"ring_us_S{s}_P2"] = row[0]
        out[f"ring_us_S{s}_P4"] = row[1]
        out[f"ring_us_S{s}_P8"] = row[2]
    return out


# ---------------------------------------------------------------------------
# Demo §3 — TP-sharded throughput sweep on a Llama-7B-shaped MLP block.
# ---------------------------------------------------------------------------

# REFERENCE: instances/vllm/source/vllm/model_executor/models/llama.py:L94-L121
# Demo §3 mirrors the LlamaMLP block under varying tp_size; production layer
# uses MergedColumnParallelLinear + RowParallelLinear configured by the
# transformer config. Our hidden=4096, ffn=11008 matches Llama-7B.
def demo_3_throughput_sweep() -> dict:
    _hr("[3] TP-sharded MLP throughput sweep (Llama-7B-shaped: H=4096, F=11008)")
    out: dict = {}
    H = 4096
    F = 11008
    seq = 512
    rng = np.random.default_rng(3)
    W_gate = rng.standard_normal((H, F)).astype(np.float32) * 0.01
    W_up = rng.standard_normal((H, F)).astype(np.float32) * 0.01
    W_down = rng.standard_normal((F, H)).astype(np.float32) * 0.01

    print(f"  shape: hidden={H}, ffn={F}, seq={seq}")
    print(f"  weights memory (full, fp16): "
          f"gate+up = {2*H*F*2 / 1e6:.1f} MB, down = {F*H*2 / 1e6:.1f} MB, "
          f"total per layer = {(2*H*F + F*H)*2 / 1e6:.1f} MB")
    out["weights_per_layer_MB_fp16"] = (2 * H * F + F * H) * 2 / 1e6

    for tp in (1, 2, 4):
        mlp = LlamaMLPTP(hidden_size=H, intermediate_size=F, tp_size=tp)
        mlp.load_weights(W_gate, W_up, W_down)
        # Per-rank weights memory (fp16). gate+up ffn dim halved by tp; down ffn dim halved by tp.
        wt_per_rank = ((H * F * 2) + (F * H)) * 2 / tp / 1e6
        out[f"weights_per_rank_tp{tp}_MB_fp16"] = wt_per_rank

        # Time forward (averaged).
        x = rng.standard_normal((seq, H)).astype(np.float32) * 0.05
        # Warm up.
        for _ in range(2):
            mlp.forward(x)
        N = 5
        t0 = time.perf_counter()
        for _ in range(N):
            mlp.forward(x)
        t1 = time.perf_counter()
        compute_time_per_forward = (t1 - t0) / N

        # α-β predicted all-reduce overhead per forward (ONE all-reduce in down_proj).
        if tp > 1:
            pred = predict_block_overhead(
                hidden=H, ffn=F, batch_seqs=seq, dtype_bytes=2,
                tp_size=tp, hardware="NVLink_HSXM4",
            )
            # MLP block has ONE all-reduce (only down_proj). predict_block_overhead
            # accounts for two (one per attn, one per mlp) — so divide by 2.
            pred_ar_per_forward_s = pred["predicted_seconds_per_block"] / 2
        else:
            pred_ar_per_forward_s = 0.0

        print(f"  tp={tp}  per-rank weights = {wt_per_rank:7.2f} MB   "
              f"compute_per_forward = {compute_time_per_forward*1e3:6.2f} ms   "
              f"AR overhead (predicted, NVLink) = {pred_ar_per_forward_s*1e6:.1f} μs   "
              f"collectives_per_forward = {1 if tp > 1 else 0}")
        out[f"compute_per_forward_tp{tp}_ms"] = compute_time_per_forward * 1e3
        out[f"predicted_AR_us_tp{tp}_NVLink"] = pred_ar_per_forward_s * 1e6
        out[f"collectives_per_forward_tp{tp}"] = 1 if tp > 1 else 0
    return out


# ---------------------------------------------------------------------------
# Demo §4 — GQA × TP boundary: KV head replication & memory floor.
# ---------------------------------------------------------------------------

# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L1031-L1036
# The GQA replication branch — when tp_size >= total_num_kv_heads, set
# num_kv_head_replicas = divide(tp_size, total_num_kv_heads). Demo §4 walks
# tp_size across that boundary and shows the memory floor.
def demo_4_gqa_tp_boundary() -> dict:
    _hr("[4] GQA × TP boundary: KV head replication and memory floor")
    out: dict = {}
    # Llama-3-70B numbers: 64 Q heads, 8 KV heads, head_size=128, hidden=8192.
    H = 8192
    head_size = 128
    total_q_heads = 64
    total_kv_heads = 8
    seq = 1024
    print(f"  Llama-3-70B-style: H={H}, head_size={head_size}, "
          f"Q heads={total_q_heads}, KV heads={total_kv_heads}, seq={seq}")
    print(f"  KV cache size per token (full, fp16) = "
          f"2 × {total_kv_heads} × {head_size} × 2 = "
          f"{2 * total_kv_heads * head_size * 2} bytes")
    full_kv_per_token_bytes = 2 * total_kv_heads * head_size * 2
    out["full_kv_per_token_bytes"] = full_kv_per_token_bytes

    print()
    print(f"  {'tp_size':>8s} {'kv_heads/rank':>14s} {'replicas':>10s} "
          f"{'KV/rank/token (B)':>20s} {'KV save factor':>16s}")
    for tp in (2, 4, 8, 16, 32):
        qkv = QKVParallelLinear(
            hidden_size=H, head_size=head_size,
            total_num_heads=total_q_heads, total_num_kv_heads=total_kv_heads,
            tp_size=tp,
        )
        s = qkv.per_rank_summary()
        kv_per_rank_bytes = 2 * s["num_kv_heads_per_rank"] * head_size * 2
        # Even when each rank holds 1 kv head, replicas mean total memory across
        # the replica-group is (num_kv_heads_per_rank * head_size * 2) * replicas
        # — so the EFFECTIVE save factor is total_kv_heads / num_kv_heads_per_rank,
        # capped at total_kv_heads (= 8 here).
        save_factor = full_kv_per_token_bytes / kv_per_rank_bytes
        print(f"  {tp:>8d} {s['num_kv_heads_per_rank']:>14d} "
              f"{s['num_kv_head_replicas']:>10d} {kv_per_rank_bytes:>20d} "
              f"{save_factor:>16.1f}×")
        out[f"tp{tp}_num_kv_heads_per_rank"] = s["num_kv_heads_per_rank"]
        out[f"tp{tp}_num_kv_head_replicas"] = s["num_kv_head_replicas"]
        out[f"tp{tp}_kv_bytes_per_rank_per_token"] = kv_per_rank_bytes
        out[f"tp{tp}_kv_save_factor"] = save_factor

    print(f"\n  Note: at tp={total_kv_heads}, each rank holds 1 KV head exactly. "
          f"Above tp={total_kv_heads}, KV is REPLICATED — memory savings cap at {total_kv_heads}×.")
    return out


# ---------------------------------------------------------------------------
# Demo §5 — End-to-end LlamaMLP TP correctness + collective accounting.
# ---------------------------------------------------------------------------

# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L1562-L1563
# RowParallelLinear's `if self.reduce_results and self.tp_size > 1:
#   output = tensor_model_parallel_all_reduce(output_parallel)`
# This is THE single collective the Megatron col→row pair performs.
def demo_5_llama_mlp_correctness() -> dict:
    _hr("[5] End-to-end LlamaMLP TP correctness (Megatron pair = 1 all-reduce per block)")
    out: dict = {}
    rng = np.random.default_rng(42)
    H = 1024
    F = 2752
    seq = 16
    x = rng.standard_normal((seq, H)).astype(np.float32) * 0.05
    W_gate = rng.standard_normal((H, F)).astype(np.float32) * 0.02
    W_up = rng.standard_normal((H, F)).astype(np.float32) * 0.02
    W_down = rng.standard_normal((F, H)).astype(np.float32) * 0.02

    y_ref = reference_unsharded_mlp(x, W_gate, W_up, W_down)
    print(f"  shape: hidden={H}, ffn={F}, seq={seq}")

    for tp in (1, 2, 4, 8):
        mlp = LlamaMLPTP(hidden_size=H, intermediate_size=F, tp_size=tp)
        mlp.load_weights(W_gate, W_up, W_down)
        mlp.reset_collective_count()
        N = 3
        for _ in range(N):
            y_tp, _ = mlp.forward(x)
        diff = float(np.max(np.abs(y_ref - y_tp)))
        per_call = mlp.count_collectives() / N
        print(f"  tp={tp}  max_abs_diff = {diff:.3e}  "
              f"avg_collectives_per_forward = {per_call:.1f}  "
              f"(expected: {0 if tp == 1 else 1})")
        out[f"mlp_tp{tp}_max_abs_diff"] = diff
        out[f"mlp_tp{tp}_collectives_per_forward"] = per_call
    return out


# ---------------------------------------------------------------------------
# Aggregator.
# ---------------------------------------------------------------------------

def run_demo() -> dict:
    print("=" * 72)
    print("Ch08 Tensor Parallelism — annotated trace (demo numerics)")
    print("=" * 72)
    print("Source pin: vLLM commit 98661fe at instances/vllm/source/")
    results = OrderedDict()
    results["§1_equivalence"] = demo_1_equivalence()
    results["§2_alpha_beta"] = demo_2_alpha_beta()
    results["§3_throughput_sweep"] = demo_3_throughput_sweep()
    results["§4_gqa_tp_boundary"] = demo_4_gqa_tp_boundary()
    results["§5_llama_mlp"] = demo_5_llama_mlp_correctness()

    # Final summary block — verbatim numerics for the writer.
    _hr("DEMO_RESULTS — verbatim numerics (writer quotes these character-for-character)")
    for section, vals in results.items():
        print(f"\n[{section}]")
        for k, v in vals.items():
            if isinstance(v, float):
                if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0):
                    print(f"  {k} = {v:.3e}")
                else:
                    print(f"  {k} = {v:.6g}")
            else:
                print(f"  {k} = {v}")
    return results


if __name__ == "__main__":
    run_demo()
