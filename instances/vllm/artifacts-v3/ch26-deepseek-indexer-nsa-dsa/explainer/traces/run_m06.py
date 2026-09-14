# ch26-m06 decode paged 打分 + top-k 三核分派 —— builder decode 真路径 +
# op 真入口：fp8_fp4_paged_mqa_logits 免 gather 直读分页 IndexCache
# （block_table 寻址）；2D seq_lens 支持 spec 窗口；三核语义一致 + tie-break。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import (_CommonMeta, DEV, RADIX_WS, dump, make_small_hf_config,
                          make_vllm_config, ref_dsa_logits,
                          ref_group_quant_ue8m0, ref_topk_desc)

doc = {"mechanism": "ch26-m06 decode paged 打分 + top-k 三核分派",
       "source": "run_m06.py（host；paged 打分核/三核为 HOST SEAM 精确数学）",
       "code_anchor": "sparse_attn_indexer.py:L530-L688（decode 主路径）/ "
                      "L616-L665（三核分派条件）"}

from vllm import _custom_ops as ops
from vllm.forward_context import ForwardContext, set_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

torch.manual_seed(13)
topk = 6
B, next_n, L = 2, 2, 40          # spec 窗口 2：每拍 2 token
block_size = 16                  # 3 块/请求：分页间接寻址非平凡
hf = make_small_hf_config(index_topk=topk)
vllm_config = make_vllm_config(hf, max_model_len=64,
                               max_num_batched_tokens=64, spec_tokens=1)
spec = MLAAttentionSpec(block_size=block_size, num_kv_heads=1,
                        head_size=132, dtype=torch.uint8)
builder = DeepseekV32IndexerMetadataBuilder(
    kv_cache_spec=spec, layer_names=["t.k_cache"],
    vllm_config=vllm_config, device=DEV, block_table_width=8)
bt = torch.tensor([[5, 6, 7], [9, 10, 11]], dtype=torch.int32)
qsl = torch.tensor([0, next_n, 2 * next_n], dtype=torch.int32)
common = _CommonMeta(num_reqs=B, num_actual_tokens=B * next_n,
                     query_start_loc=qsl, seq_lens=torch.tensor([L, L]),
                     max_query_len=next_n, max_seq_len=L,
                     block_table=bt,
                     slot_mapping=torch.arange(B * next_n, dtype=torch.int64))
meta = builder.build(0, common)

# 历史量化 key 入缓存（两请求各自的物理 slot 集）
kv_cache = torch.zeros(16, block_size, 132, dtype=torch.uint8)
k_hist = torch.randn(L, 128)
for b in range(B):
    slots = torch.tensor([int(bt[b, p // block_size]) * block_size
                          + p % block_size for p in range(L)],
                         dtype=torch.int64)
    ops.indexer_k_quant_and_cache(k_hist, kv_cache, slots, 128, "ue8m0")
q = torch.randn(B * next_n, hf.index_n_heads, 128)
q_fp8, q_scale = ref_group_quant_ue8m0(q.reshape(-1, 128))
q_fp8 = q_fp8.view(B * next_n, hf.index_n_heads, 128)
weights = (torch.randn(B * next_n, hf.index_n_heads)
           * q_scale.view(B * next_n, hf.index_n_heads))
buf = torch.zeros(64, topk, dtype=torch.int32)
ctx = ForwardContext(no_compile_layers={},
                     attn_metadata={"t.k_cache": meta}, slot_mapping={})
with set_forward_context(ctx):
    sparse_attn_indexer(
        hidden_states=torch.zeros(B * next_n, 8), k_cache_prefix="t.k_cache",
        kv_cache=kv_cache, q_quant=q_fp8, q_scale=None,
        k=torch.randn(B * next_n, 128), weights=weights,
        quant_block_size=128, scale_fmt="ue8m0", topk_tokens=topk,
        head_dim=128, max_model_len=64, total_seq_lens=L,
        topk_indices_buffer=buf, skip_k_cache_insert=True, use_pcp=False,
        dense_mha_metadata_layer_name="")

# 参考对账：按 block_table 反读缓存成请求序 k
flat = kv_cache.reshape(-1, 132)
k_deq = torch.stack([
    flat[int(bt[0, p // block_size]) * block_size + p % block_size,
         :128].view(torch.float8_e4m3fn).float()
    * flat[int(bt[0, p // block_size]) * block_size + p % block_size,
           128:].view(torch.float32).item() for p in range(L)])
logits_full = ref_dsa_logits(q_fp8.float(), k_deq, weights)
rows, echo = [], []
for b in range(B):
    for j in range(next_n):
        r = b * next_n + j
        bound = L - next_n + j + 1
        ref = ref_topk_desc(logits_full[r, :bound].tolist(), topk)
        got = buf[r, :topk].tolist()
        assert got == ref
        top_vals = sorted(logits_full[r, :bound].tolist(), reverse=True)[:topk]
        rows.append({"row": r, "req": b, "j": j, "bound": bound,
                     "buffer_row": got,
                     "top_values_rounded": [f"{v:.3f}" for v in top_vals]})
        echo.append([f"r{r}", f"req{b} spec 位 {j}",
                     f"rowEnd = 40 - 2 + {j} + 1 = {bound}",
                     f"选中 {got}",
                     f"top3 [{', '.join(f'{v:.3f}' for v in top_vals[:3])}]"])

doc["decode_native_2d"] = {
    "B": B, "next_n": next_n, "L": L, "block_size": block_size,
    "block_table": bt.tolist(),
    "seq_lens_2d": meta.decode.seq_lens.tolist(),
    "seq_lens_2d_formula": "seq_lens[b, j] = L - next_n + j + 1",
    "requires_padding": bool(meta.decode.requires_padding),
    "paged_slot_examples": {
        "req0_pos15": int(bt[0, 0]) * 16 + 15,
        "req0_pos16": int(bt[0, 1]) * 16 + 0,
        "req0_pos20": int(bt[0, 1]) * 16 + 4,
        "req1_pos32": int(bt[1, 2]) * 16 + 0,
    },
    "no_gather_note": "decode 打分不经 gather——fp8_fp4_paged_mqa_logits 直接按 "
                      "block_table 读分页 cache（对比 m05 prefill 先 gather）",
    "all_rows_match_bruteforce": True,
    "rows": rows,
}

# ── 三核分派：语义一致 + tie-break + 分派条件账 ────────────────────────────
torch.manual_seed(17)
Nr, Nn, Kk = 3, 64, 8
logits3 = torch.randn(Nr, Nn)
seq_lens3 = torch.tensor([64, 40, 7], dtype=torch.int32)
outs = []
for call in (
    lambda o: ops.top_k_per_row_decode(
        logits3, 1, seq_lens3, o, Nr, logits3.stride(0), logits3.stride(1), Kk),
    lambda o: torch.ops._C.cooperative_topk(
        logits3, seq_lens3, o, torch.empty(RADIX_WS, dtype=torch.uint8), Kk, Nn),
    lambda o: torch.ops._C.persistent_topk(
        logits3, seq_lens3, o, torch.empty(RADIX_WS, dtype=torch.uint8), Kk, Nn),
):
    o = torch.full((Nr, Kk), -1, dtype=torch.int32)
    call(o)
    outs.append(o)
tie = torch.tensor([[5.0, 5.0, 1.0], [9.0, 2.0, 2.0]])
tie_buf = torch.full((2, 2), -1, dtype=torch.int32)
ops.top_k_per_row_decode(tie, 1, torch.tensor([2, 3], dtype=torch.int32),
                         tie_buf, 2, tie.stride(0), tie.stride(1), 2)
doc["three_kernels"] = {
    "per_row_equals_cooperative": bool(torch.equal(outs[0], outs[1])),
    "per_row_equals_persistent": bool(torch.equal(outs[0], outs[2])),
    "tie_break_input": [[5, 5, 1], [9, 2, 2]],
    "tie_break_output": tie_buf.tolist(),
    "tie_break_note": "行1 三个值 [9,2,2]：降序取 2 → [0,1]——9 在位 0，"
                      "两个 2 并列取小 index（位 1 先于位 2）——与 csrc 插入排序 "
                      "tie-break 同构",
    "host_dispatch_note": "host 无 CUDA（is_cuda=False）→ 走 per_row 兜底核；"
                          "三核语义一致由上方对账证明（取证环境差异，writer "
                          "须就近挑明）",
    "dispatch_conditions_sm90": [
        {"kernel": "cooperative_topk", "conditions":
            "topk∈{512,1024,2048} 且 num_rows≤32 且 stride0%4==0 且 "
            "capability≥90 且 非 sm120 家族",
         "example_topk": 2048, "example_rows": 32, "would_take": True},
        {"kernel": "persistent_topk", "conditions":
            "topk∈{512,1024,2048}（批大也走）",
         "example_topk": 2048, "example_rows": 33, "would_take": True},
        {"kernel": "top_k_per_row_decode", "conditions": "其余（兜底）",
         "example_topk": 3000, "example_rows": 32, "would_take": True},
    ],
}
doc["table_rows_echo"] = echo + [
    ["三核", "同输入 [3,64] logits、seq_lens [64,40,7]、k=8",
     "per_row == cooperative == persistent", "选中集合逐行相等=true"],
    ["tie", "行 [9,2,2] 取 top-2", "输出 [0,1]",
     "两个 2 并列取小 index——插入排序 tie-break"],
    ["分派", "topk=2048、rows=32、sm90", "cooperative=True",
     "rows=33 → cooperative 假、persistent 真；topk=3000 → per_row 兜底"],
]

dump("m06.json", doc)
print("m06.json written | 2D seq_lens:", meta.decode.seq_lens.tolist(),
      "| rows match:", doc["decode_native_2d"]["all_rows_match_bruteforce"],
      "| 3-kernel equal:", doc["three_kernels"]["per_row_equals_persistent"])
