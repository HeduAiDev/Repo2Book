# ch26-m09 复杂度诚实账 —— O(L²)→O(Lk) 与 indexer 自身仍 O(L²) 的数量级对账。
# 纯算术（无 GPU 计时）——数字全部由本脚本计算产出。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import dump

doc = {"mechanism": "ch26-m09 复杂度诚实账：O(L²)→O(Lk)，indexer 自身仍 O(L²)",
       "source": "run_m09.py（host；纯算术对账，无 GPU 计时——比例是结构性的）",
       "code_anchor": "sparse_attn_indexer.py:L449-L528（打分 O(L²) 本尊）/ "
                      "backends/mla/indexer.py:L442-L452（workspace 账）"}

H_main, D_main = 128, 576          # MLA 吸收态：128 头 × 576 维潜向量点积
H_idx, D_idx = 64, 128             # indexer 独立小头：64 头 × 128 维
L, k = 131072, 2048                # 128k 上下文、DSV3.2 index_topk
L32 = 163840                       # DSV3.2 max_model_len

dense_main = H_main * D_main * L
idx_cost = H_idx * D_idx * L
sparse_main = H_main * D_main * k
total_sparse = idx_cost + sparse_main

doc["decode_step_qk_macs"] = {
    "L": L, "k": k, "H_main": H_main, "D_main": D_main,
    "H_idx": H_idx, "D_idx": D_idx,
    "dense_mla_qk_macs": dense_main,
    "indexer_qk_macs": idx_cost,
    "sparse_main_qk_macs": sparse_main,
    "total_sparse_qk_macs": total_sparse,
    "main_attention_only_saving_x": L // k,
    "main_attention_only_saving_x_note": "主注意力每 token 读 L 条 → 读 k 条："
                                         f"131072/2048 = {L // k}",
    "indexer_vs_dense_main_ratio": f"{idx_cost / dense_main:.4f}",
    "indexer_vs_dense_main_percent": f"{idx_cost / dense_main * 100:.2f}",
    "indexer_per_pair_vs_main_per_pair": f"{(H_idx * D_idx) / (H_main * D_main):.4f}",
    "total_compute_reduction_x": f"{dense_main / total_sparse:.2f}",
    "fp8_note": "indexer QK 全程 FP8（值 1B、吞吐约 bf16 2×；FP4 再 2×）且无 "
                "反向/无 V 投影——MAC 之外还有常数因子优势；主注意力 MQA 在 "
                "576 维潜向量上做 bf16",
    "honest_note": "indexer 自身仍 O(L²)：131072² 级别的 (query, key) 对一个没"
                   "少——没消灭、换成便宜项（每对 8192 MAC vs 73728 MAC）",
}
doc["dsv32_scale"] = {
    "L": L32, "k": k,
    "saving_x": L32 // k,
    "dense_mla_qk_macs": H_main * D_main * L32,
    "indexer_qk_macs": H_idx * D_idx * L32,
}
# prefill 打分矩阵的峰值账（与 m04 对账）
M = 16384
doc["prefill_logits_peak"] = {
    "M": M, "N": L32,
    "unbounded_bytes": M * L32 * 4,
    "unbounded_mib": M * L32 * 4 / 1048576,
    "budget_mb": 512,
    "note": "O(L²) 的显存面：prefill 打分矩阵 M×N×4 B——双预算切块压峰值（m04）",
}
# IndexCache 的存在理由：打分要扫全历史 → 索引键只算一次
layers = 61
doc["index_cache_bytes"] = {
    "per_token_per_layer_bytes": 132,
    "main_kv_bf16_per_token_per_layer_bytes": 576 * 2,
    "layers": layers,
    "full_len_163840_indexer_total_bytes": 132 * layers * L32,
    "full_len_163840_indexer_total_gib": f"{132 * layers * L32 / 2 ** 30:.2f}",
    "full_len_163840_main_bf16_total_bytes": 576 * 2 * layers * L32,
    "full_len_163840_main_bf16_total_gib": f"{576 * 2 * layers * L32 / 2 ** 30:.2f}",
    "without_cache_note": "若无 IndexCache，decode 每步要对全部历史重算 "
                          "k 投影+量化（每 token 7168×192 GEMM）——打分原料"
                          "只算一次、量化缓存、跨拍复用是 decode 可行的前提",
}
doc["table_rows_echo"] = [
    ["主注意力", "dense MLA QK：128 头×576 维×131072 对",
     f"{dense_main} MAC", "每步每层都要读完全部历史"],
    ["换稀疏", "稀疏主注意力 QK：128×576×2048",
     f"{sparse_main} MAC", f"只算选中 k=2048 条：省 {L // k} 倍（主注意力侧）"],
    ["indexer", "打分仍扫全历史：64 头×128 维×131072 对",
     f"{idx_cost} MAC", f"为 dense 主注意力的 {idx_cost / dense_main * 100:.2f}%"
     "（FP8 再省）"],
    ["合计", "稀疏主注意力 + indexer",
     f"{total_sparse} MAC",
     f"vs dense {dense_main}：总算量降至 1/{dense_main / total_sparse:.2f}"],
    ["实尺", "L=163840（DSV3.2 max_model_len）", "163840/2048 = 80 倍",
     "打分矩阵峰值 16384×163840×4 B = 10240 MiB → 切块压住"],
    ["第二本账", "IndexCache 132B/token/layer × 61 层 × 163840",
     f"{132 * layers * L32} B（{132 * layers * L32 / 2 ** 30:.2f} GiB）",
     f"主 KV bf16 同规模 {576 * 2 * layers * L32 / 2 ** 30:.2f} GiB 的 "
     f"{132 / (576 * 2) * 100:.2f}%"],
]

dump("m09.json", doc)
print("m09.json written | saving:", L // k, "x | idx/dense:",
      f"{idx_cost / dense_main * 100:.2f}%", "| total reduction:",
      f"{dense_main / total_sparse:.2f}x")
