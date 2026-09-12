# ch25 m08 —— chunked prefill workspace 与 LSE merge：上投影大张量永不整段物化
# （mla_attention.py:L1803-L1831 定容 / L2301-L2422 分块循环 / merge_attn_states
#   LSE 精确合并——softmax 分块合并恒等式，ch20 online softmax 的块间推广）
# 场景一（手算档）：3 块 keys 的 LSE 逐轮合并 == 整块 softmax（精确恒等式数值表）。
# 场景二（实跑档）：上下文 150、workspace 64 → 3 块（64/64/22）+ suffix 140 的
#   真实 chunked prefill，与全量上投影 MHA 参照逐元素对照。
from __future__ import annotations

import math

import torch

from trace_common import MINI, dump, fmt, oracle_mha, run, setup

# ══ 场景一：3 块 LSE 合并的手算账（真实 merge_attn_states 算子）════════════
# 单头单 query：6 个 key 分 3 块（2+1+3），V=1 维——读者可手算复核每一步。
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

s1 = torch.tensor([[1.0, 2.0]])       # 块 1 的 2 个分数
v1 = torch.tensor([[10.0], [30.0]])
s2 = torch.tensor([[3.0]])            # 块 2 的 1 个分数
v2 = torch.tensor([[20.0]])
s3 = torch.tensor([[0.5, 0.2, 4.0]])  # 块 3 的 3 个分数
v3 = torch.tensor([[5.0], [6.0], [7.0]])


def block_out(s, v):
    """s [T,keys] / v [keys,V] → o [T,N=1,V] / lse [N=1,T]（merge 算子的形状约定）"""
    p = torch.softmax(s, dim=-1)
    o = (p @ v).unsqueeze(1)          # [T,1,V]
    lse = torch.logsumexp(s, dim=-1).unsqueeze(0)  # [1,T]
    return o, lse, p


o1, l1, p1 = block_out(s1, v1)
o2, l2, p2 = block_out(s2, v2)
o3, l3, p3 = block_out(s3, v3)


def mrg(po, pl, so, sl):
    out = torch.zeros_like(po)
    out_lse = torch.zeros_like(pl)
    merge_attn_states(output=out, prefix_output=po, prefix_lse=pl,
                      suffix_output=so, suffix_lse=sl, output_lse=out_lse)
    return out, out_lse


o12, l12 = mrg(o1, l1, o2, l2)
o123, l123 = mrg(o12, l12, o3, l3)
# 参照：整块 softmax
s_all = torch.cat([s1, s2, s3], dim=-1)
v_all = torch.cat([v1, v2, v3], dim=0)
ref, lref, p_all = block_out(s_all, v_all)
toy_max_diff = (o123 - ref).abs().max().item()
toy_lse_diff = (l123 - lref).abs().max().item()
# 逐轮合并权重（LSE 加权，e^{l} 归一）
e_l1, e_l2 = math.exp(float(l1[0, 0])), math.exp(float(l2[0, 0]))
w1 = e_l1 / (e_l1 + e_l2)
w2 = e_l2 / (e_l1 + e_l2)
e_l12, e_l3_ = math.exp(float(l12[0, 0])), math.exp(float(l3[0, 0]))
w12 = e_l12 / (e_l12 + e_l3_)
w3 = e_l3_ / (e_l12 + e_l3_)

# ══ 场景二：真实 chunked prefill（3 块上下文 + suffix 合并）════════════════
# block=16 / max_model_len=8 / max_num_seqs=1 → workspace = max(8×8, 4×1×16)=64
# → 上下文 150 分 3 块（64+64+22）；新 token 140 > 阈 128 → MHA 腿。
vllm_config, layer, inner, builder = setup(
    dims=MINI, max_model_len=8, max_num_seqs=1, block=16, seed=61)
dims = MINI
H, Lq, N, P, R, Lkv, V = (dims["hidden_size"], dims["q_lora_rank"],
                          dims["num_heads"], dims["qk_nope_head_dim"],
                          dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                          dims["v_head_dim"])
C, M = 150, 140

torch.manual_seed(61)
h_hist = torch.randn(C, H)
pos_hist = torch.arange(C)
run(layer, builder, vllm_config, [C], [C], h_hist, pos_hist, block=16)

pb = builder._prefill_backend
chunk_rec = []
orig_chunk = pb.run_prefill_context_chunk


def chunk_probe(chunk_idx, q, k, v):
    chunk_rec.append({"idx": int(chunk_idx), "context_keys": int(k.shape[0]),
                      "q_rows": int(q.shape[0])})
    return orig_chunk(chunk_idx, q, k, v)


pb.run_prefill_context_chunk = chunk_probe

# 记录 merge_attn_states 的调用（running 块间合并 + 终合并）
import vllm.model_executor.layers.attention.mla_attention as mla_mod
merge_calls = []
orig_merge = mla_mod.merge_attn_states


def merge_probe(output, prefix_output, prefix_lse, suffix_output,
                suffix_lse, output_lse=None, prefill_tokens_with_context=None,
                output_scale=None):
    merge_calls.append({
        "prefix_lse_max": fmt(prefix_lse.max().item(), 3),
        "suffix_lse_max": fmt(suffix_lse.max().item(), 3),
        "tokens_with_context": (None if prefill_tokens_with_context is None
                                else int(prefill_tokens_with_context)),
    })
    return orig_merge(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse, output_lse, prefill_tokens_with_context,
                      output_scale)


mla_mod.merge_attn_states = merge_probe

h_new = torch.randn(M, H)
pos_new = torch.arange(C, C + M)
out, md, common = run(layer, builder, vllm_config, [M], [C + M], h_new,
                      pos_new, block=16)
mla_mod.merge_attn_states = orig_merge

ref_out, _ = oracle_mha(layer, inner, common, h_new, pos_new, [M], [C + M],
                        block=16)
real_max_diff = (out - ref_out).abs().max().item()

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
)
ws_size = MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
    vllm_config)

# workspace 字节账（本例 fp32 实测 + 源码注释 64k 档口径）
ws_bytes_mini = ws_size * 576 * 4
up_full_bytes_mini = C * (N * (P + R) + N * V) * 4          # 不分块要整段物化
ws_bytes_64k_bf16 = 65536 * 576 * 2                          # 75497472 B = 72 MiB
comment_ws_claim = "源码注释自称 144mb（mla_attention.py:L1813-L1815 '2*(576)*(64*1024) = 144mb'）——按 2B×576×65536 精确积为 75497472 B（72 MiB），注释口径恰差 2 倍；上投影账 2*(192*128)*(64*1024)=3221225472 B（3 GB）同式成立"
up_64k_kb_only_bf16 = 65536 * (192 * 128) * 2                # 3221225472 B

trace = {
    "mechanism": "ch25-m08",
    "source": "run_m08.py @ implementation/（vLLM v0.27.1 只做减法精简版, host CPU, torch float32）",
    "code_anchor": "vllm/model_executor/layers/attention/mla_attention.py:L1803-L1831（workspace 定容）· L2301-L2422（分块循环）· vllm/v1/attention/ops/merge_attn_states.py:L9-L111（LSE 合并）",
    "toy_merge": {
        "setup": "单头单 query、6 个 key 分 3 块（2+1+3）、V=1、scale=1——每步可手算",
        "block1_scores": [1.0, 2.0], "block1_values": [10.0, 30.0],
        "block1_probs": [fmt(x) for x in p1[0].tolist()],
        "block1_output": fmt(float(o1[0, 0, 0])), "block1_lse": fmt(float(l1[0, 0])),
        "block2_scores": [3.0], "block2_values": [20.0],
        "block2_output": fmt(float(o2[0, 0, 0])), "block2_lse": fmt(float(l2[0, 0])),
        "block3_scores": [0.5, 0.2, 4.0], "block3_values": [5.0, 6.0, 7.0],
        "block3_probs": [fmt(x) for x in p3[0].tolist()],
        "block3_output": fmt(float(o3[0, 0, 0])), "block3_lse": fmt(float(l3[0, 0])),
        "merge_round1_weights": {"block1": fmt(w1), "block2": fmt(w2)},
        "merge_round1_output": fmt(float(o12[0, 0, 0])),
        "merge_round1_lse": fmt(float(l12[0, 0])),
        "merge_round2_weights": {"blocks12": fmt(w12), "block3": fmt(w3)},
        "merge_round2_output": fmt(float(o123[0, 0, 0])),
        "merge_round2_lse": fmt(float(l123[0, 0])),
        "full_softmax_reference": fmt(float(ref[0, 0, 0])),
        "full_softmax_lse": fmt(float(lref[0, 0])),
        "max_abs_diff_vs_full": fmt(toy_max_diff, 6),
        "lse_max_abs_diff": fmt(toy_lse_diff, 6),
    },
    "real_chunked_prefill": {
        "context_C": C, "new_M": M, "block_size": 16,
        "workspace_tokens": int(ws_size),                    # 64
        "workspace_formula": "min(max(8×max_model_len, 4×max_num_seqs×block), 65536) → max(64, 64)=64",
        "num_chunks": len(chunk_rec),                        # 3
        "chunks": chunk_rec,                                 # 64/64/22
        "chunk_total_context": [c["context_keys"] for c in chunk_rec],
        "merge_calls_inside_impl": merge_calls,
        "iters_note": "iters=3：块 1 直填 running → 块 2/块 3 各一次块间 merge → suffix 与 context 终合并（merge 共 3 次）",
        "vs_full_upprojected_oracle_max_abs_diff": fmt(real_max_diff, 6),
    },
    "workspace_byte_account": {
        "workspace_bytes_this_run_fp32": ws_bytes_mini,      # 64×576×4
        "full_upproject_bytes_this_run_fp32": up_full_bytes_mini,
        "workspace_bytes_64k_bf16": ws_bytes_64k_bf16,
        "upprojected_kb_only_bytes_64k_bf16": up_64k_kb_only_bf16,
        "source_comment_claim": comment_ws_claim,
        "up_projected_note": "3 GB 是仅 K（192×128）的账；加上 V（128×128）则 64k token 全量上投影 K+V ≈ 5.37 GB——workspace 只装潜向量，上投影在 64-token 块内现场做、用完即弃",
    },
    "outcome": {"out_shape": list(out.shape)},
}

assert [c["context_keys"] for c in chunk_rec] == [64, 64, 22]
assert len(merge_calls) == 3
assert toy_max_diff < 1e-5 and real_max_diff < 1e-4
dump("m08.json", trace)
