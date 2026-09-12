# ch25 m06 —— 同一前向混批分流：num_mqa_tokens=num_decode_tokens 在 token 维切一刀
# （mla_attention.py:L771-L772 分流决策 + L812-L829/L831-L949 两腿调用）
# 场景：混批 5 请求 = 3 decode（各 1 token）+ 2 prefill chunk（140 带 30 上下文 /
# 150 带 50 上下文）——同一次 forward 里前 3 token 走 MQA 吸收、后 290 token 走
# MHA 展开，两段写回同一 output 缓冲的不相交切片。
from __future__ import annotations

import torch

from trace_common import MINI, dump, fmt, oracle_mha, run, setup

vllm_config, layer, inner, builder = setup(seed=52)
dims = MINI
H, Lq, N, P, R, Lkv, V = (dims["hidden_size"], dims["q_lora_rank"],
                          dims["num_heads"], dims["qk_nope_head_dim"],
                          dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                          dims["v_head_dim"])

# ① 两条历史先写入 cache（decode 候选 hist 10/5/8 + prefill 候选已算 30/50）
hist = [10, 5, 8, 30, 50]
torch.manual_seed(52)
h_hist = torch.randn(sum(hist), H)
pos_hist = torch.cat([torch.arange(s) for s in hist])
run(layer, builder, vllm_config, hist, hist, h_hist, pos_hist)

# ② 混批一拍：3 decode + 2 prefill chunk
ql = [1, 1, 1, 140, 150]
seq = [11, 6, 9, 170, 200]     # 各自 = 已算 + 本拍新
total = sum(ql)
h_new = torch.randn(total, H)
pos_new = torch.tensor([10, 5, 8] + list(range(30, 170)) + list(range(50, 200)))

rec = {}
orig_fwd_mqa = inner.impl.forward_mqa
orig_fwd_mha = inner.impl.forward_mha
orig_vup = inner._v_up_proj


def fwd_mqa_probe(q, kv_cache, attn_metadata, lay):
    rec["mqa_rows"] = q[0].shape[0]
    rec["mqa_decode_seq_lens"] = [int(x) for x in attn_metadata.decode.seq_lens]
    return orig_fwd_mqa(q, kv_cache, attn_metadata, lay)


def fwd_mha_probe(q, kv_c_normed, k_pe, kv_cache, attn_metadata, k_scale,
                  output, output_scale=None):
    rec["mha_rows"] = q.shape[0]
    rec["mha_output_slice_shape"] = list(output.shape)
    return orig_fwd_mha(q, kv_c_normed, k_pe, kv_cache, attn_metadata,
                        k_scale, output, output_scale)


def vup_probe(x, out):
    rec["mqa_output_slice_shape"] = list(out.shape)
    return orig_vup(x, out)


inner.impl.forward_mqa = fwd_mqa_probe
inner.impl.forward_mha = fwd_mha_probe
inner._v_up_proj = vup_probe

out, md, common = run(layer, builder, vllm_config, ql, seq, h_new, pos_new)

ref_out, _ = oracle_mha(layer, inner, common, h_new, pos_new, ql, seq)
max_diff = (out - ref_out).abs().max().item()

qsl = common.query_start_loc.tolist()
token_rows = {f"req{i}": [qsl[i], qsl[i + 1]] for i in range(len(ql))}

trace = {
    "mechanism": "ch25-m06",
    "source": "run_m06.py @ implementation/（vLLM v0.27.1 只做减法精简版, host CPU, torch float32）",
    "code_anchor": "vllm/model_executor/layers/attention/mla_attention.py:L771-L772（num_mqa_tokens=num_decode_tokens / num_mha_tokens=其余）+ L812-L829（MHA 腿 q[num_mqa:]）+ L831-L838（MQA 腿 q[:num_mqa]）",
    "params": {
        "query_lens": ql, "seq_lens": seq, "total_tokens": total,
        "num_requests": len(ql),
        "context_of_prefills": [30, 50],
        "flashmla_reorder_threshold": 128,
    },
    "builder_counts": {
        "num_decodes": int(md.num_decodes),               # 3
        "num_prefills": int(md.num_prefills),             # 2
        "num_decode_tokens": int(md.num_decode_tokens),   # 3
        "num_prefill_tokens": int(md.num_actual_tokens) - int(md.num_decode_tokens),
        "query_start_loc": qsl,                           # [0,1,2,3,143,293]
    },
    "token_dim_cut": {
        "num_mqa_tokens": int(md.num_decode_tokens),      # 3 = num_decode_tokens
        "num_mha_tokens": total - int(md.num_decode_tokens),  # 290
        "mqa_slice_rows": rec["mqa_rows"],                # 3
        "mha_slice_rows": rec["mha_rows"],                # 290
        "mqa_output_slice": [0, int(md.num_decode_tokens)],        # output[0:3]
        "mha_output_slice": [int(md.num_decode_tokens), total],    # output[3:293]
        "mqa_output_slice_shape": rec["mqa_output_slice_shape"],
        "mha_output_slice_shape": rec["mha_output_slice_shape"],
        "cut_note": "切的是 token 维前缀/后缀（q[:3] / q[3:]），不是请求维——批已被站 7 的四区重排保证 decode 在前",
    },
    "per_request_rows": token_rows,
    "sq_over_skv_per_request": {
        "decode": [fmt(1 / s, 3) for s in seq[:3]],       # ≈0.09/0.17/0.11
        "prefill": [fmt(q / s, 3) for q, s in zip(ql[3:], seq[3:])],  # 0.82/0.75
        "note": "同一拍内 Sq/Skv 跨两个数量级——这就是一刀切两段数学的动机（文件头 L13-L21：prefill Sq/Skv≈1 走 compute-friendly、decode 小走 data-movement-friendly）",
    },
    "verification": {
        "vs_upprojected_mha_oracle_max_abs_diff": fmt(max_diff, 6),
        "note": "两段各自走 MQA 吸收 / MHA 上投影，合并输出与整批上投影 MHA 参照逐元素一致——同一数学、两种算法",
    },
    "outcome": {"out_shape": list(out.shape)},
}

assert rec["mqa_rows"] == 3 and rec["mha_rows"] == 290
assert qsl == [0, 1, 2, 3, 143, 293]
assert md.num_decode_tokens == 3 and md.num_prefills == 2
assert max_diff < 1e-4
dump("m06.json", trace)
