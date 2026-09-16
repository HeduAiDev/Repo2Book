"""ch28 m08（HCA 重压缩与『压完不挑』Eq.20-26）驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m08_hca.py
产出：explainer/traces/run_m08_hca.json（params + raw_stdout）。

素材来源：implementation/compressor.py（论文忠实的小型参考实现）+ implementation/ledger.py（纯算术件）。
  A) 同构对照：同一组 H／同一组投影，跑 CSA（两组 C/Z、窗宽 2m、有重叠）与 HCA（一组 C/Z、窗宽 m'、
     无重叠），逐条对照——条目数、每条的输入个数、归一范围、i=0 是否特例。
  B) 「压完不挑」的成本论证（config 口径 m=4 / m'=128 / topk=512 / 1M 上下文）：候选数、每 query
     实看条数、multiply-add 口径的『省略掉的索引器』账、以及 HCA 层把 topk 缓冲全填的实现形态。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import compressor as cp  # noqa: E402
import ledger  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

# ── 玩具参数：n=12、c=2；CSA 的 m=4（窗宽 2m=8）、HCA 的 m'=4（窗宽 m'=4） ──────
M, C_DIM, N = 4, 2, 12
H = np.array([
    [1.0, 0.5, -1.0], [0.5, 1.0, 0.0], [-0.5, 0.5, 1.0], [1.0, -1.0, 0.5],
    [0.0, 1.0, 1.0], [1.0, 1.0, -0.5], [0.5, -0.5, 0.5], [-1.0, 0.0, 1.0],
    [1.0, 0.5, 0.5], [0.5, 0.0, -0.5], [0.0, 1.0, -1.0], [1.0, 0.5, 1.0],
])
W_kv_a = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
W_kv_b = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0]])
W_z_a = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
W_z_b = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
B_a = np.array([[0.5, 0.2], [0.4, 0.1], [0.3, 0.0], [0.2, -0.1]])
B_b = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

P("== ch28 m08 · HCA：同一套压缩机制的另一档（玩具 m'=4 + config 口径 m'=128） ==")
P(f"参数：n = {N}、c = {C_DIM}；CSA 的 m = {M}（窗宽 2m = {2 * M}）、HCA 的 m' = {M}（窗宽 m' = {M}）")

# ── A) 同构对照：同一组 H／同一组投影 ────────────────────────────────────────
P("")
P("── A) 同构对照：Eq.(20)-(23) 与 Eq.(9)-(12) 只差三件事 ──")
C_csa, S_csa = cp.csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, M)
C_hca, S_hca = cp.hca_compress(H, W_kv_a, W_z_a, B_a, M)
P(f"    [CSA] 条目数 = {N} // {M} = {N // M}；每条输入 = 2m = {2 * M} 个（当前窗 {M} + 前一个窗 {M}，重叠）")
P(f"    [HCA] 条目数 = {N} // {M} = {N // M}；每条输入 = m' = {M} 个（只有自己这一窗，无重叠）")
P(f"    [CSA] 条目 i=1 权重逐列和 = {R(S_csa[1].sum(axis=0))}（归一跨 {2 * M} 个元素）")
P(f"    [HCA] 条目 i=1 权重逐列和 = {R(S_hca[1].sum(axis=0))}（归一跨 {M} 个元素）——归一范围不同、『和为 1』相同")
P(f"    [CSA] 条目 i=1 的 b 半区（前一个窗）权重 = {R(S_csa[1][M:])} ← 非 0：重叠窗真的在看别人")
sets = cp.csa_entry_input_index_sets(M, N // M)
P(f"    [HCA] 条目 i=1 的输入 token = {list(range(M * 1, M * 2))}（单序列：窗即本条目的全部输入）")
P(f"    [CSA] 条目 i=1 的输入 token = a 半区 {sets[1][0]} + b 半区 {sets[1][1]}（与条目 i=0 共享 {M} 个）")
for i in range(N // M):
    same = bool(np.allclose(C_csa[i], C_hca[i]))
    P(f"    [对照] 条目 i={i}：CSA C^Comp = {R(C_csa[i])}｜HCA C^Comp = {R(C_hca[i])}"
      f"｜逐位相同 = {same}")
P(f"    ⚠️ i=0 两档**完全相同**（逐位相同 = True）——这不是巧合也不是 bug，是**结构性退化**：")
P(f"       i=0 时 CSA 的 b 半区被 −inf 填掉（权重 = {R(S_csa[0][M:])}）、只剩 a 半区那 {M} 个元素，")
P(f"       于是 CSA 的归一范围 8 → 实际只在自己窗内（等价 HCA 的 4）⇒ 第一个窗上两档是同一件事。")
P(f"       真实差别从 i≥1 出现：i=1 的逐位相同 = {bool(np.allclose(C_csa[1], C_hca[1]))}、"
  f"i=2 的逐位相同 = {bool(np.allclose(C_csa[2], C_hca[2]))}")
P(f"    ⇒ 讲对照时要用 i≥1 的条目（i=0 恰好是两者重合的那一格，拿它对比会得出错误结论）")
P(f"    [i=0 特例只在 CSA 有] CSA 条目 i=0 的 b 半区权重 = {R(S_csa[0][M:])}（−inf 填充 ⇒ 全 0）")
P(f"        HCA 没有 a/b 两半（Eq.(22) 只有一组 Z+B）⇒ 不存在 i=0 特例，每个窗都是自足归一")

# ── B) 「压完不挑」的成本论证（config 口径） ─────────────────────────────────
P("")
P("── B) 「压完不挑」的成本论证（config 口径：m=4、m'=128、topk=512、1M 上下文） ──")
CTX, TOPK, N_HI, C_I, C_MAIN = 1_000_000, 512, 64, 128, 512
cand_csa = CTX // 4
cand_hca = CTX // 128
P(f"    候选压缩块（= 已完成的块数）：CSA 层 = {CTX} // 4 = {cand_csa}；HCA 层 = {CTX} // 128 = {cand_hca}")
see_csa = ledger.attention_visible_entries(CTX, 4, TOPK)
see_hca = ledger.attention_visible_entries(CTX, 128)
P(f"    每 query 实看条目数：CSA = min({TOPK}, {cand_csa}) = {see_csa}（top-k 封顶）；HCA = {see_hca}（全看，无选择器）")
P(f"    HCA / CSA = {see_hca} / {see_csa} = {see_hca / see_csa:.6f} 倍（HCA 主注意力确实更贵）")
idx_macs_per_head = cand_csa * C_I          # 每个索引头：候选数 × 索引头维
hca_extra_per_head = (see_hca - see_csa) * C_MAIN   # 每个主头：多看的条目 × 主头维
P(f"    索引器打分的乘加数（每个索引头 / 每 query）= {cand_csa} × {C_I} = {idx_macs_per_head}")
P(f"    索引器打分的乘加数（config 的 {N_HI} 个索引头合计）= {idx_macs_per_head} × {N_HI} = {idx_macs_per_head * N_HI}")
P(f"    HCA『多看』的乘加数（每个主注意力头 / 每 query）= ({see_hca} - {see_csa}) × {C_MAIN} = {hca_extra_per_head}")
P(f"    ⇒ 单个索引头 vs 单个主头：{idx_macs_per_head} / {hca_extra_per_head} = {idx_macs_per_head / hca_extra_per_head:.6f} 倍")
P("        （保守口径：索引器有 64 个头、主注意力头数本包未取到，这里按『每个头』并排——即便只算一个头，")
P("          索引器打分也远贵于 HCA 多看的那部分；把 64 个头加起来差距再乘 64）")
P(f"    ① 选择器自身的『排序』账：CSA 要对 {cand_csa} 个分数做 top-{TOPK}（radix top-k）；HCA 一次排序都不做")
P(f"    ② 选择器的缓存账：CSA 层每层多一份 IndexCache = 132 B/条 ⇒ (584 + 132) / 4 = {(584 + 132) / 4:.6f} B/token")
P(f"       HCA 层只有主压缩条目：584 / 128 = {584 / 128:.6f} B/token（没有索引器对象）")
P(f"    ③ 选择器的盲区账：CSA 只能看 top-{TOPK} 条（1M 下候选 {cand_csa} 条里挑）；HCA 一条不落")
P("    ④ 代价：HCA 每 query 的条目数是 CSA 的 15.2578 倍——但压到 1/128 后这个『倍数』仍是小绝对值")

# ── C) 「全填」的实现形态：HCA 层的 topk 缓冲 ────────────────────────────────
P("")
P("── C) 『压完不挑』在实现里的形态：HCA 层把 topk 缓冲按序全填 ──")
M_PRIME, TOPK_BUF = 128, 8   # 玩具缓冲宽 8（真实 512），只为看清『全填』与『−1 哨兵』
for pos in (255, 256, 383, 1000):
    num_compressed = (pos + 1) // M_PRIME
    buf = [i if i < num_compressed else -1 for i in range(TOPK_BUF)]
    P(f"    pos={pos}: (pos+1)//{M_PRIME} = {num_compressed} 条已完成 ⇒ 缓冲（宽 {TOPK_BUF}）= {buf}")
P(f"    ⇒ HCA 的『选择』就是 seq(0, num_compressed) 一次全填（pin 的 _build_c128a_topk_metadata_kernel；")
P(f"      decode 版再查 block_table 换成全局槽位），没有打分、没有排序——indexer 对象根本不存在")
P(f"    短上下文同样成立：pos=1000 时候选只有 {1000 // M_PRIME} 条，全填 = 全看")

out = {
    "script": "run_m08_hca.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"m": M, "m_prime_toy": M, "c": C_DIM, "n": N, "m_prime_config": 128,
               "index_topk": TOPK, "context_len": CTX, "index_n_heads": N_HI,
               "index_head_dim": C_I, "head_dim": C_MAIN},
    "results": {
        "C_csa": C_csa.tolist(),
        "C_hca": C_hca.tolist(),
        "S_csa_entry1_colsum": S_csa[1].sum(axis=0).tolist(),
        "S_hca_entry1_colsum": S_hca[1].sum(axis=0).tolist(),
        "S_csa_entry0_bhalf": S_csa[0][M:].tolist(),
        "candidates_csa": cand_csa,
        "candidates_hca": cand_hca,
        "visible_csa": see_csa,
        "visible_hca": see_hca,
        "ratio_hca_over_csa": see_hca / see_csa,
        "indexer_macs_per_head": idx_macs_per_head,
        "indexer_macs_total": idx_macs_per_head * N_HI,
        "hca_extra_macs_per_head": hca_extra_per_head,
        "indexer_vs_hca_extra_ratio": idx_macs_per_head / hca_extra_per_head,
        "bytes_csa_layer": (584 + 132) / 4,
        "bytes_hca_layer": 584 / 128,
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
