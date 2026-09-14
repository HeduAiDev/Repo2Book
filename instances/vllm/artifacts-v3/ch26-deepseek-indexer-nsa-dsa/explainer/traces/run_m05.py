# ch26-m05 prefill 打分核链 —— builder 真路径 + op 真入口：
# IndexCache（分页，物理位≠逻辑位）→ gather 进 workspace → fp8_fp4_mqa_logits
# （Eq.(1) O(L²) 本尊）→ top_k_per_row_prefill 因果边界选块 → buffer 行。
# 与暴力参考（独立参考数学）逐行对账。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import (_CommonMeta, DEV, dump, make_block_table,
                          make_small_hf_config, make_vllm_config,
                          ref_dsa_logits, ref_group_quant_ue8m0, ref_topk_desc)

doc = {"mechanism": "ch26-m05 prefill 打分核链：gather → mqa_logits → top_k_per_row",
       "source": "run_m05.py（host；打分/选核为 HOST SEAM 精确数学）",
       "code_anchor": "sparse_attn_indexer.py:L449-L528（主循环）"}

from vllm import _custom_ops as ops
from vllm.forward_context import ForwardContext, set_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

torch.manual_seed(5)
topk = 6
seq_lens, query_lens = (70, 6), (6, 4)   # req0 历史 64；req1 历史 2
block_size, first_block = 64, 2          # 物理块从 2 起：物理位≠逻辑位
hf = make_small_hf_config(index_topk=topk)
vllm_config = make_vllm_config(hf, max_model_len=128, max_num_batched_tokens=128)
n_tok = sum(query_lens)
spec = MLAAttentionSpec(block_size=block_size, num_kv_heads=1,
                        head_size=132, dtype=torch.uint8)
builder = DeepseekV32IndexerMetadataBuilder(
    kv_cache_spec=spec, layer_names=["t.k_cache"],
    vllm_config=vllm_config, device=DEV, block_table_width=4)
bt, n_blocks = make_block_table(list(seq_lens), block_size,
                                first_block=first_block)
qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(query_lens), 0)),
                   dtype=torch.int32)
slots = []
for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
    base = int(bt[b, 0].item()) * block_size
    slots.extend(range(base + S - Q, base + S))
common = _CommonMeta(num_reqs=2, num_actual_tokens=n_tok, query_start_loc=qsl,
                     seq_lens=torch.tensor(seq_lens),
                     max_query_len=max(query_lens), max_seq_len=max(seq_lens),
                     block_table=bt, slot_mapping=torch.tensor(slots,
                                                               dtype=torch.int64))
indexer_meta = builder.build(0, common)

# 历史 key 先入缓存（请求序），本拍 k 由 op 插入
kv_cache = torch.zeros(max(4, n_blocks), block_size, 132, dtype=torch.uint8)
k_all = torch.randn(sum(seq_lens), 128)
hist_slots, hist_rows = [], []
for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
    base = int(bt[b, 0].item()) * block_size
    hist_slots.extend(range(base, base + S - Q))
    hist_rows.extend(range(sum(seq_lens[:b]), sum(seq_lens[:b]) + S - Q))
ops.indexer_k_quant_and_cache(
    k_all[hist_rows], kv_cache, torch.tensor(hist_slots, dtype=torch.int64),
    128, "ue8m0")

q = torch.randn(n_tok, hf.index_n_heads, 128)
q_fp8, q_scale = ref_group_quant_ue8m0(q.reshape(-1, 128))
q_fp8 = q_fp8.view(n_tok, hf.index_n_heads, 128)
weights = (torch.randn(n_tok, hf.index_n_heads)
           * q_scale.view(n_tok, hf.index_n_heads))
k_cur = torch.cat([k_all[sum(seq_lens[:b]) + S - Q: sum(seq_lens[:b]) + S]
                   for b, (S, Q) in enumerate(zip(seq_lens, query_lens))])
buf = torch.full((128, topk), -7, dtype=torch.int32)
ctx = ForwardContext(no_compile_layers={}, attn_metadata={
    "t.k_cache": indexer_meta}, slot_mapping={})
with set_forward_context(ctx):
    out = sparse_attn_indexer(
        hidden_states=torch.zeros(n_tok, hf.hidden_size),
        k_cache_prefix="t.k_cache", kv_cache=kv_cache,
        q_quant=q_fp8, q_scale=None, k=k_cur,
        weights=weights, quant_block_size=128, scale_fmt="ue8m0",
        topk_tokens=topk, head_dim=128, max_model_len=128,
        total_seq_lens=sum(seq_lens), topk_indices_buffer=buf,
        skip_k_cache_insert=False, use_pcp=False,
        dense_mha_metadata_layer_name="")
assert out is buf

chunk = indexer_meta.prefill.chunks[0]
# ── A. chunk 元数据（因果边界账）────────────────────────────────────────────
doc["chunk_metadata"] = {
    "seq_lens": list(seq_lens), "query_lens": list(query_lens),
    "cu_seq_lens": chunk.cu_seq_lens.tolist(),
    "cu_seqlen_ks": chunk.cu_seqlen_ks.tolist(),
    "cu_seqlen_ke": chunk.cu_seqlen_ke.tolist(),
    "block_table": bt.tolist(),
    "local_total_seq_lens": int(chunk.local_total_seq_lens),
    "note": "cu_seqlen_ks=每行 K 起点（请求在 gathered workspace 的基址）；"
            "ke=ks+因果长（start_pos+1+offset）——req0 首 token 因果窗 [0,65)、"
            "req1 首 token 窗 [70,73)",
}

# ── B. 分页 gather 的间接寻址（物理位≠逻辑位，F7 在稀疏路径的再现）──────────
doc["paged_gather_slots"] = {
    "first_block": first_block, "block_size": block_size,
    "logical_to_physical": {
        "req0_pos63": int(bt[0, 0]) * 64 + 63,
        "req0_pos64": int(bt[0, 1]) * 64 + 0,
        "req0_pos65": int(bt[0, 1]) * 64 + 1,
        "req1_pos0": int(bt[1, 0]) * 64 + 0,
    },
    "note": "req0 跨块边界：逻辑 pos 63 在物理块 2（slot 191）、pos 64 起在"
            "物理块 3（slot 192）——gather 按 block_table 换算，请求序铺进 "
            "workspace [76, 132]",
}

# ── C. 打分矩阵与选块（逐行对账暴力参考）───────────────────────────────────
k_full = []
for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
    base = int(bt[b, 0].item()) * block_size
    for pos in range(S):
        row = kv_cache.reshape(-1, 132)[base + pos]
        k_full.append(row[:128].view(torch.float8_e4m3fn).float()
                      * row[128:].view(torch.float32).item())
k_full = torch.stack(k_full)
logits_full = ref_dsa_logits(q_fp8.float(), k_full, weights)
rows_account, echo_rows = [], []
tok = 0
for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
    start_pos = S - Q
    base = sum(seq_lens[:b])
    for o in range(Q):
        causal = start_pos + 1 + o
        ref = ref_topk_desc(logits_full[tok, base:base + causal].tolist(), topk)
        got = buf[tok, :topk].tolist()
        assert got == ref, f"req{b} tok{o}: {got} != {ref}"
        top_vals = sorted(logits_full[tok, base:base + causal].tolist(),
                          reverse=True)[:topk]
        sentinels = sum(1 for g in got if g == -1)
        rows_account.append({
            "row": tok, "req": b, "pos": start_pos + o,
            "ks": int(chunk.cu_seqlen_ks[tok]), "ke": int(chunk.cu_seqlen_ke[tok]),
            "candidates": causal, "buffer_row": got,
            "top_values_rounded": [f"{v:.3f}" for v in top_vals],
            "sentinels": sentinels})
        echo_rows.append(
            [f"r{tok}", f"req{b} pos{start_pos + o}",
             f"窗 [{int(chunk.cu_seqlen_ks[tok])}, {int(chunk.cu_seqlen_ke[tok])}) "
             f"共 {causal} 候选",
             f"选中 {got}" + (f"（-1 ×{sentinels}）" if sentinels else ""),
             f"top 分值 [{', '.join(f'{v:.3f}' for v in top_vals[:3])}, …]"])
        tok += 1
doc["per_row_account"] = rows_account
doc["all_rows_match_bruteforce"] = True
doc["logits_matrix_rounded3"] = [[f"{v:.3f}" for v in row]
                                 for row in logits_full.tolist()]
doc["workspace_layout_note"] = (
    "k_quant workspace [76, 128] + k_scale [76, 4]：prefill 打分不直接读分页 "
    "cache——先 cp_gather_indexer_k_quant_cache 收集成连续段（对比 m06 decode "
    "免 gather 直读）")
doc["table_rows_echo"] = echo_rows

dump("m05.json", doc)
print("m05.json written | rows match bruteforce:",
      doc["all_rows_match_bruteforce"], "| req1 first row sentinels:",
      rows_account[6]["sentinels"])
