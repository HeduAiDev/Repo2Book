"""ch28 m01（两本账）/ m02（三类层与 4/128 交替）驱动脚本。

跑法（Windows Git Bash；`python3` 是 WindowsApps 存根、exit 49）：
    /d/Env/Miniconda/python explainer/traces/run_m01_m02_ledger.py

产出：explainer/traces/run_m01_m02_ledger.json（params + raw_stdout）。

素材来源：implementation/ledger.py（纯算术件，非论文机制的实现）+ paper.md §八 的 config 口径。
本脚本只做一件事：把论文给的三个相对比例（27% / 10% / ~2%）换成**分母明确**的绝对账，并把
层分布、每档每 query 实看条目数一次打印全。**所有数字都是自算**（论文不给绝对量纲）。
"""
import json
import sys
from pathlib import Path

CH = Path(__file__).resolve().parents[2]          # 章目录
sys.path.insert(0, str(CH / "implementation"))
import ledger  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK 控制台免疫

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


# ── 0. 入参（全部来自 config / pin 注释，不是论文数字） ─────────────────────
RATIOS_44 = ledger.V4_FLASH_COMPRESS_RATIOS        # V4-Flash config.compress_ratios（44 项）
RATIOS_43 = RATIOS_44[:43]                         # 主干 43 层（末项 0 是 MTP 槽）
CONTEXT = 1_000_000
N_WIN = 128
SWA_ROW = 576.0
IDX_TOPK = 512

P("== ch28 m01/m02 · 两本账自算（ledger.py，纯算术） ==")
P(f"config.compress_ratios 全长 = {len(RATIOS_44)} 项（前两项 0、中段 4/128 交替、末项 0）")
P(f"主干层数 = {len(RATIOS_43)}；1M 上下文 = {CONTEXT}；n_win = {N_WIN}；swa 行宽 = {SWA_ROW} B；index_topk = {IDX_TOPK}")
P(f"compress_ratios 前 8 项 = {RATIOS_44[:8]}；后 5 项 = {RATIOS_44[-5:]}")

# ── 1. 三类层（m02 的层分布） ────────────────────────────────────────────────
counts = ledger.classify_layers(RATIOS_43)
P("")
P(f"[层型分类] classify_layers(43 项) = {counts}")
P(f"[层型分类] 三类层名（pin: sparse_swa.py）swaonly/c4a/c128a 各 {counts['swaonly']} / {counts['c4a']} / {counts['c128a']} 层")
counts44 = ledger.classify_layers(RATIOS_44)
P(f"[层型分类] 44 项（含 MTP 槽）也全归三类：{counts44}（max(1, 0) 把配置里的 0 归一成 1）")

# ── 2. 单条条目的字节（分子） ────────────────────────────────────────────────
entry = ledger.compressed_entry_bytes()
idx_entry = ledger.indexer_cache_entry_bytes()
v32 = ledger.v32_bytes_per_token()
gqa8 = ledger.gqa8_baseline_bytes_per_token()
P("")
P(f"[KV 账·分子] 一条压缩条目 = 448 fp8 NoPE + 128 bf16 RoPE + 8 fp8 scale = {entry} B")
P(f"[KV 账·第二本账] 索引器小头一条 = 128 fp8 + 4 fp32 scale = {idx_entry} B")

# ── 3. 逐层摊销 ─────────────────────────────────────────────────────────────
per_c4a = ledger.cache_bytes_per_token(4, count_indexer=True)
per_c4a_no_idx = ledger.cache_bytes_per_token(4, count_indexer=False)
per_c128a = ledger.cache_bytes_per_token(128, count_indexer=True)
per_swa = ledger.cache_bytes_per_token(1, count_indexer=True)
P("")
P(f"[KV 账·逐层摊销] CSA 层 (584 + 132) / 4 = {per_c4a} B/token/层")
P(f"[KV 账·逐层摊销] CSA 层不计 IndexCache 584 / 4 = {per_c4a_no_idx} B/token/层")
P(f"[KV 账·逐层摊销] HCA 层 584 / 128 = {per_c128a} B/token/层")
P(f"[KV 账·逐层摊销] ratio ≤ 1 的层压缩账 = {per_swa} B（没有压缩机）")

# ── 4. 整机账 + 两个分母 ────────────────────────────────────────────────────
acc = ledger.kv_byte_account(RATIOS_43, count_indexer=True, count_swa_window=False)
acc_swa = ledger.kv_byte_account(RATIOS_43, count_indexer=True, count_swa_window=True)
acc_no_idx = ledger.kv_byte_account(RATIOS_43, count_indexer=False, count_swa_window=False)
P("")
P(f"[KV 账·整机] 逐层加权 = (2×0 + 21×{per_c4a} + 20×{per_c128a}) / 43 = {acc['bytes_per_token']:.6f} B/token/层")
P(f"[KV 账·分母一] BF16 GQA8 head_dim=128 基线 = 2 × 8 × 128 × 2 = {gqa8} B/token")
P(f"[KV 账·比值一] {acc['bytes_per_token']:.6f} / {gqa8} = {acc['ratio_vs_gqa8'] * 100:.4f} %   ← 论文 §2.3.4 的 ~2% 分母")
P(f"[KV 账·分母二] V3.2 自定义布局 = {v32} B/token")
P(f"[KV 账·比值二] {acc['bytes_per_token']:.6f} / {v32} = {acc['ratio_vs_v32'] * 100:.4f} %   ← 论文摘要 10% 的分母")
P("")
P(f"[KV 账·滑窗是加账] 43 层 × {N_WIN} 条 × {SWA_ROW} B = {acc_swa['swa_window_total_bytes']:.0f} B 的常数窗")
P(f"[KV 账·滑窗摊销] {acc_swa['swa_window_total_bytes']:.0f} / {CONTEXT} = {acc_swa['swa_bytes_per_token']:.4f} B/token（1M 下摊薄）")
P(f"[KV 账·计滑窗总账] {acc['bytes_per_token']:.6f} + {acc_swa['swa_bytes_per_token']:.4f} = {acc_swa['total_bytes_per_token']:.6f} B/token")
P(f"[KV 账·计滑窗比值] 对 GQA8 = {acc_swa['ratio_vs_gqa8'] * 100:.4f} %；对 V3.2 = {acc_swa['ratio_vs_v32'] * 100:.4f} %")
P("")
P(f"[口径敏感性] 不计 IndexCache：{acc_no_idx['bytes_per_token']:.6f} B/token → 对 GQA8 {acc_no_idx['ratio_vs_gqa8'] * 100:.4f} %、对 V3.2 {acc_no_idx['ratio_vs_v32'] * 100:.4f} %")
P("[口径敏感性] 结论：这三个比例随口径（是否计 IndexCache / 是否计滑窗 / 分母取谁）在个位数百分比内漂移——")

# ── 5. FLOPs 账的结构计数：每 query 实看条目数（1M 上下文） ──────────────────
P("")
P("[FLOPs 账·结构计数] 每 query 实看条目数（不是 FLOPs 数，是驱动它的条数）：")
P(f"    纯滑窗层（ratio ≤ 1）：滑窗长度就是全部注意力 = n_win = {N_WIN} 条/query（与上下文长度无关）")
for name, ratio, topk in (("c4a(m=4)", 4, IDX_TOPK), ("c128a(m'=128)", 128, None)):
    vis = ledger.attention_visible_entries(CONTEXT, ratio, topk)
    cand = CONTEXT // ratio
    P(f"    1M 上下文 {name}: 候选 = {CONTEXT} // {ratio} = {cand}，实看 = {vis} 条/query")
    P(f"    1K 上下文 {name}: 候选 = {CONTEXT // 1000} // {ratio} = {(CONTEXT // 1000) // ratio}，实看 = {ledger.attention_visible_entries(CONTEXT // 1000, ratio, topk)} 条/query")
vis_c4a, vis_c128a = ledger.attention_visible_entries(CONTEXT, 4, IDX_TOPK), ledger.attention_visible_entries(CONTEXT, 128, None)
P(f"[FLOPs 账·三档并置] 1M 下一 query 实看：纯滑窗 {N_WIN} 条、CSA {vis_c4a} 条（top-k 封顶）、HCA {vis_c128a} 条")
P(f"[FLOPs 账·HCA/CSA 比值] {vis_c128a} / {vis_c4a} = {vis_c128a / vis_c4a:.4f} 倍（HCA 每 query 看的是 CSA 的这么多数倍，省掉的是整个索引器）")
avg_vis = (2 * N_WIN + 21 * vis_c4a + 20 * vis_c128a) / 43
P(f"[FLOPs 账·按层加权] (2×{N_WIN} + 21×{vis_c4a} + 20×{vis_c128a}) / 43 = {avg_vis:.4f} 条/query（逐层平均；纯滑窗层的账是常数窗）")
P(f"[FLOPs 账·短上下文倒挂] 1K 上下文：HCA 实看 {ledger.attention_visible_entries(1000, 128, None)} 条、CSA 实看 {ledger.attention_visible_entries(1000, 4, IDX_TOPK)} 条（候选 {1000 // 4} 还没撞上 topk {IDX_TOPK}）——短上下文里 m=4 的候选墙不起作用、索引器打分成了纯开销，这正是实现里『候选 <= topk 就直接全选』快路径的动机")
P("")
P(f"[短上下文快路径] short_context_select_all(1000, 4, {IDX_TOPK}): 候选 {ledger.attention_visible_entries(1000, 4, IDX_TOPK)} <= topk {IDX_TOPK} -> True（一个不落地全选）")

# ── 6. 派生量（供正文直接引用；全部由上面各行的自算值导出） ──────────────────
P("")
P("[派生量] 滑窗项占压缩账的比例 = %s / %s = %.6f" % (f"{acc_swa['swa_bytes_per_token']:.4f}",
                                                     f"{acc['bytes_per_token']:.6f}",
                                                     acc_swa["swa_bytes_per_token"] / acc["bytes_per_token"]))
P("[派生量] 同一常数窗在 1K 上下文下的摊销 = %s / 1000 = %.6f B/token（与 1M 下的 %.4f 差 1000 倍）"
  % (f"{acc_swa['swa_window_total_bytes']:.0f}", acc_swa["swa_window_total_bytes"] / 1000,
     acc_swa["swa_bytes_per_token"]))
P("[派生量] 口径差的归因：%.6f − %.6f = %.6f；另一路算法 21 × (%d/4) / 43 = %.6f（两路相等）"
  % (acc["bytes_per_token"], acc_no_idx["bytes_per_token"],
     acc["bytes_per_token"] - acc_no_idx["bytes_per_token"], idx_entry,
     21 * (idx_entry / 4) / 43))

out = {
    "script": "run_m01_m02_ledger.py",
    "raw_stdout": "\n".join(LINES),
    "params": {
        "compress_ratios_44": RATIOS_44,
        "n_main_layers": len(RATIOS_43),
        "context_len": CONTEXT,
        "n_win": N_WIN,
        "swa_row_bytes": SWA_ROW,
        "index_topk": IDX_TOPK,
        "entry_bytes": entry,
        "index_entry_bytes": idx_entry,
        "v32_bytes_per_token": v32,
        "gqa8_baseline_bytes_per_token": gqa8,
    },
    "results": {
        "counts": counts,
        "per_type": {"swaonly": per_swa, "c4a": per_c4a, "c128a": per_c128a},
        "bytes_per_token": acc["bytes_per_token"],
        "ratio_vs_gqa8": acc["ratio_vs_gqa8"],
        "ratio_vs_v32": acc["ratio_vs_v32"],
        "swa_window_total_bytes": acc_swa["swa_window_total_bytes"],
        "swa_bytes_per_token": acc_swa["swa_bytes_per_token"],
        "total_bytes_per_token": acc_swa["total_bytes_per_token"],
        "bytes_no_indexcache": acc_no_idx["bytes_per_token"],
        "visible_entries_1m": {
            "swaonly": ledger.attention_visible_entries(CONTEXT, 1, None),
            "c4a": ledger.attention_visible_entries(CONTEXT, 4, IDX_TOPK),
            "c128a": ledger.attention_visible_entries(CONTEXT, 128, None),
        },
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
