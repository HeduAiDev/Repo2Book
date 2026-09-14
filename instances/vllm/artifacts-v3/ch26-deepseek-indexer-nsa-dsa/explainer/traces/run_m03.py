# ch26-m03 topk_indices_buffer —— 共享副作用缓冲的协议证据：
# 分配形状/裸共享身份、-1 哨兵两例、行=本拍 query token 整块重写、消费侧取行边界。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import (DEV, dump, make_small_hf_config, make_vllm_config)

doc = {"mechanism": "ch26-m03 topk_indices_buffer：全模型共享的副作用缓冲（-1 哨兵）",
       "source": "run_m03.py（host）",
       "code_anchor": "deepseek_v2.py:L1376-L1389（分配）/ mla.py:L189-L206（接线）"
                      " / sparse_attn_indexer.py:L426-L432（预清+落账）"}

# ── A. 分配：形状/dtype/裸共享身份 ──────────────────────────────────────────
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

hf = make_small_hf_config(num_hidden_layers=2, index_topk=2048)
vllm_config = make_vllm_config(hf, max_model_len=1024,
                               max_num_batched_tokens=512)
model = DeepseekV2Model(vllm_config=vllm_config, prefix="model")
buf = model.topk_indices_buffer
doc["allocation"] = {
    "shape": list(buf.shape),
    "dtype": str(buf.dtype).replace("torch.", ""),
    "max_num_batched_tokens": 512,
    "index_topk": 2048,
    "bytes": int(buf.numel() * 4),
    "mib": f"{buf.numel() * 4 / 1048576:g}",
    "example_at_8192_tokens_bytes": int(8192 * 2048 * 4),
    "example_at_8192_tokens_mib": f"{8192 * 2048 * 4 / 1048576:g}",
    "layer0_shares_same_object": bool(
        model.layers[0].self_attn.topk_indices_buffer is buf),
    "layer1_shares_same_object": bool(
        model.layers[1].self_attn.topk_indices_buffer is buf),
    "note": "同一块裸 buffer 无任何所有权封装——所有层的 indexer 写、稀疏 MLA 读"
            "（torch.empty 分配、不初始化）",
}

# 非 v32 无 buffer
hf_old = make_small_hf_config(num_hidden_layers=1)
del hf_old.index_topk
vllm_old = make_vllm_config(hf_old, max_model_len=1024)
model_old = DeepseekV2Model(vllm_config=vllm_old, prefix="model.old")
doc["allocation"]["non_v32_is_v32"] = bool(model_old.is_v32)
doc["allocation"]["non_v32_buffer"] = None if model_old.topk_indices_buffer is None \
    else "allocated"

# ── B. -1 哨兵两例 ─────────────────────────────────────────────────────────
from vllm import _custom_ops as ops

# 例1：top_k_per_row_decode 的 1D 因果自界——rowEnd = seq_len - next_n + j + 1
logits = torch.tensor([[9.0, 9.0, 1.0, 9.0]])  # 越界位 9 也不得入选
buf1 = torch.full((1, 2), -1, dtype=torch.int32)
ops.top_k_per_row_decode(logits, 2, torch.tensor([2], dtype=torch.int32),
                         buf1, 1, logits.stride(0), logits.stride(1), 2)
doc["sentinel_case_decode_bound"] = {
    "logits_row": [9, 9, 1, 9],
    "seq_len": 2, "next_n": 2,
    "rowEnd_formula": "seq_len - next_n + j + 1 = 2 - 2 + 0 + 1 = 1",
    "buffer_row": buf1[0].tolist(),
    "note": "只有 pos 0 有效（值 9）→ 第二位 -1：因果窗外的高分绝不入选",
}

# 例2：prefill local_total_seq_lens==0 → 整行 fill_(-1)
from vllm.forward_context import ForwardContext, set_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata, DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata)

hf2 = make_small_hf_config(index_topk=4)
vllm2 = make_vllm_config(hf2, max_model_len=64)
chunk = DeepseekV32IndexerPrefillChunkMetadata(
    block_table=torch.zeros(1, 1, dtype=torch.int32),
    cu_seqlen_ks=torch.zeros(2, dtype=torch.int32),
    cu_seqlen_ke=torch.zeros(2, dtype=torch.int32),
    cu_seq_lens=torch.zeros(2, dtype=torch.int32),
    token_to_seq=torch.zeros(0, dtype=torch.int32),
    total_seq_lens=0, token_start=0, token_end=2, num_reqs=1,
    local_cu_seq_lens=torch.zeros(2, dtype=torch.int32),
    local_total_seq_lens=0, max_local_total_seq_lens=0)
meta = DeepseekV32IndexerMetadata(
    seq_lens=torch.tensor([0]), max_seq_len=0,
    slot_mapping=torch.zeros(2, dtype=torch.int64),
    num_decodes=0, num_decode_tokens=0, num_prefills=1, num_prefill_tokens=2,
    prefill=DeepseekV32IndexerPrefillMetadata(chunks=[chunk]))
buf2 = torch.full((8, 4), -7, dtype=torch.int32)  # 预填 -7：预清由 op 做
ctx = ForwardContext(no_compile_layers={},
                     attn_metadata={"t.k_cache": meta}, slot_mapping={})
with set_forward_context(ctx):
    out = sparse_attn_indexer(
        hidden_states=torch.zeros(2, 8), k_cache_prefix="t.k_cache",
        kv_cache=torch.zeros(1, 64, 132, dtype=torch.uint8),
        q_quant=torch.zeros(2, 2, 128, dtype=torch.float8_e4m3fn),
        q_scale=None, k=None, weights=torch.ones(2, 2), quant_block_size=128,
        scale_fmt="ue8m0", topk_tokens=4, head_dim=128, max_model_len=64,
        total_seq_lens=0, topk_indices_buffer=buf2,
        skip_k_cache_insert=True, use_pcp=False,
        dense_mha_metadata_layer_name="")
doc["sentinel_case_empty_context"] = {
    "prefill_value_before": -7,
    "rows_all_minus1": bool((buf2[:2] == -1).all()),
    "note": "op 预清 -1 哨兵（sparse_attn_indexer.py:L426-L432）；无历史 → 整行 "
            "-1 直落 buffer（topk_indices.fill_(-1)）",
}

# ── C. 行 = 本拍 query token：跨拍整块重写 ─────────────────────────────────
# 用 decode top-k 核直接演示：拍1 写入 [5,3]；拍2 同一行换 q 后变 [1,4]——
# 历史行的旧值不残留（每拍整块重写）。
torch.manual_seed(23)
N, K = 8, 2
buf3 = torch.full((1, K), -100, dtype=torch.int32)  # -100 = 陈旧值
row1 = torch.randn(N)
ops.top_k_per_row_decode(row1.unsqueeze(0), 1,
                         torch.tensor([N], dtype=torch.int32), buf3, 1,
                         row1.stride(0), 1, K)
first = buf3[0].tolist()
row2 = torch.randn(N)
ops.top_k_per_row_decode(row2.unsqueeze(0), 1,
                         torch.tensor([N], dtype=torch.int32), buf3, 1,
                         row2.stride(0), 1, K)
second = buf3[0].tolist()
doc["row_rewrite_across_steps"] = {
    "stale_value_before": -100,
    "step1_row": first,
    "step1_top_values": sorted(row1.tolist(), reverse=True)[:K],
    "step2_row": second,
    "step2_top_values": sorted(row2.tolist(), reverse=True)[:K],
    "rows_differ_between_steps": first != second,
    "stale_gone": -100 not in first + second,
    "note": "buffer 行=本拍 query token：历史 token 的行上一拍已消费、本拍整块"
            "重写——『每轮只对新增 token 算 index』的载体（站 11）",
}

# ── D. 消费侧取行边界（站 12 契约）──────────────────────────────────────────
doc["consumer_contract"] = {
    "writer": "indexer（custom op 声明 mutates_args=['topk_indices_buffer']——"
              "写者是它的自我描述）",
    "reader": "FlashMLASparseImpl.forward_mqa 取前 num_actual_toks 行"
              "（flashmla_sparse.py:L858）；SparseMLACommonImpl 装配期接管 "
              "（skip 层无 indexer 也能读——显式传参兜底）",
    "return_value": "mla.py:L205-L206：indexer(...) 返回值无人接收——纯副作用",
}

dump("m03.json", doc)
print("m03.json written | shape =", buf.shape, "| bytes =", buf.numel() * 4,
      "| decode_bound_row =", buf1[0].tolist(),
      "| rewrite:", first, "->", second)
