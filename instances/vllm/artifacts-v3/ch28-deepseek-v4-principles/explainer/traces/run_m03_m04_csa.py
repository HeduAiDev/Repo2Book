"""ch28 m03（CSA 软池化压缩 Eq.9-12）/ m04（重叠窗与 i=0 边界）驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m03_m04_csa.py
产出：explainer/traces/run_m03_m04_csa.json（params + raw_stdout）。

素材来源：implementation/compressor.py（论文忠实的小型参考实现，NumPy 纯 CPU）。
玩具参数 m=4、c=2、n=12（12 个 token ⇒ 3 条条目）——小到能心算。
本脚本跑两遍：
  A) 手账版（widget）：直接把 8 个 (Z+B) 值与 8 个 C 向量摆出来，看 softmax 权重怎么挑人、
     与「直接平均」差多少；
  B) 完整版：走 csa_project → csa_entry_inputs → softmax_row → 加权和 全链，核每列权重和 = 1、
     i=0 的 b 半区权重恰为 0（−inf 填充）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import compressor as cp  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


np.set_printoptions(precision=6, suppress=True, linewidth=200)
M = 4          # CSA 压缩率（config 口径 m=4）
C_DIM = 2      # 头维 c（玩具；真实 512）
N = 12         # token 数

P("== ch28 m03/m04 · CSA 软池化（m=4, c=2, n=12） ==")
P(f"参数：m = {M}（每 {M} 个 token 压 1 条）、c = {C_DIM}、n = {N} ⇒ 条目数 = n // m = {N // M}")

# ── A) 手账版：8 个 (Z+B) 值 → softmax 权重 → 加权和 ────────────────────────
P("")
P("── A) 手账版（第 i=1 条条目、第 0 列）：把 2m 个 (Z+B) 值直接摆出来 ──")
z_col0 = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 0.0, 0.0, 0.0])          # a 半区(当前窗 4 个) + b 半区(前窗 4 个)
c_col0 = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])          # 对应的 C 值（同一条目的 8 个输入）
w = cp.softmax_row(z_col0.reshape(-1, 1), axis=0)[:, 0]
P(f"    2m 个 (Z+B) 值     = {[round(v, 6) for v in z_col0]}")
P(f"    exp(·)             = {[round(float(v), 6) for v in np.exp(z_col0)]}")
P(f"    softmax 权重 S     = {[round(float(v), 6) for v in w]}")
P(f"    权重和             = {w.sum():.6f}")
wavg = float((w * c_col0).sum())
P(f"    加权和 Σ S·C       = {wavg:.6f}")
P(f"    直接平均 (均匀 1/8) = {c_col0.mean():.6f}")
P(f"    差                 = {abs(wavg - c_col0.mean()):.6f}"
  f"（权重真的在挑人：最大的 {round(float(w.max()), 6)} 拿走了 {round(float(w.max()) * 100, 1)}% 的份额）")
P(f"    均匀权重           = {[round(1 / 8, 6)] * 8}")
P(f"    8 个输入 C           = {[round(float(v), 6) for v in c_col0]}")

# ── B) 完整链：投影 → 拼 2m 窗 → softmax → 加权和 ──────────────────────────
P("")
P("── B) 完整链（compressor.csa_compress，Eq.9-12） ──")
H = np.array([
    [1.0, 0.5, -1.0], [0.5, 1.0, 0.0], [-0.5, 0.5, 1.0], [1.0, -1.0, 0.5],
    [0.0, 1.0, 1.0], [1.0, 1.0, -0.5], [0.5, -0.5, 0.5], [-1.0, 0.0, 1.0],
    [1.0, 0.5, 0.5], [0.5, 0.0, -0.5], [0.0, 1.0, -1.0], [1.0, 0.5, 1.0],
])
W_kv_a = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
W_kv_b = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0]])
W_z_a = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
W_z_b = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
B_a = np.array([[0.5, 0.2], [0.4, 0.1], [0.3, 0.0], [0.2, -0.1]])       # 可学习位置偏置（窗内槽位 0..3）
B_b = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
C_comp, S = cp.csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, M)
P(f"    C^Comp 形状 = {C_comp.shape}（{N} 个 token → {N // M} 条，每条 {C_DIM} 维）")
for i in range(C_comp.shape[0]):
    P(f"    条目 i={i} C^Comp = {[round(float(v), 6) for v in C_comp[i]]}")
    P(f"    条目 i={i} softmax 权重 S（{2 * M}×{C_DIM}，逐列，6 位）："
      f" col0 = {[round(float(S[i][r][0]), 6) for r in range(2 * M)]}"
      f" col1 = {[round(float(S[i][r][1]), 6) for r in range(2 * M)]}")
    for r in range(2 * M):
        half = "a 半区(当前窗)" if r < M else "b 半区(前一个窗)"
        P(f"        slot {r} [{half}] = {[round(float(v), 6) for v in S[i][r]]}")
    P(f"    条目 i={i} 权重逐列和 = {[round(float(v), 6) for v in S[i].sum(axis=0)]}  ← Eq.11 的『跨 2m 个元素归一』")
P("")
P("── i=0 边界（唯一的特例：Z^b 填 -inf、C^b 填 0） ──")
P(f"    条目 i=0 的 b 半区权重（slot {M}..{2 * M - 1}）= {S[0][M:].tolist()} ← 全 0（exp(-inf)=0）")
P(f"    对照：条目 i=1 的 b 半区权重（slot {M}..{2 * M - 1}）= {S[1][M:].tolist()} ← 非 0（有前窗可看）")
P(f"    条目 i=0 的 a 半区权重逐列和 = {S[0][:M].sum(axis=0).tolist()} ← 『第一个窗只看自己』")
P("")
P("── 重叠（m04：为什么压到 1/m 而不是 1/(2m)） ──")
sets = cp.csa_entry_input_index_sets(M, N // M)
P(f"    csa_entry_input_index_sets(m={M}, count={N // M}) = {sets}")
for i, (a, b) in enumerate(sets):
    P(f"    条目 i={i}: a 半区 token {a}（当前窗）｜ b 半区 token {b}（前一个窗）")
P(f"    条目数 = {N} // {M} = {N // M}（不是 {N} // {2 * M} = {N // (2 * M)}）——相邻条目共享 {M} 个输入")
P(f"    共享验证：条目 i=1 的 b 半区 {sets[1][1]} 与 条目 i=0 的 a 半区 {sets[0][0]} 完全相同 -> {sets[1][1] == sets[0][0]}")
P(f"    每条条目的输入个数 = 2m = {2 * M}，但新增输入只有 m = {M} 个 ⇒ 序列压到 1/{M}")
P("")
P("── pin 侧的窗口算术（压缩核的 start/head_offset/mask_pos，CR=4 口径） ──")
CR, OVERLAP, HEAD_SIZE = 4, 1, 512
for position in (3, 7, 11):
    start = position - (1 + OVERLAP) * CR + 1
    tokens = np.arange((1 + OVERLAP) * CR)
    pos = start + tokens
    head_offset = (tokens >= CR).astype(int) * HEAD_SIZE
    P(f"    块尾 token position={position}: start = {position} - {1 + OVERLAP}*{CR} + 1 = {start}；pos = {pos.tolist()}"
      f"；mask_pos(pos>=0) = {[bool(p >= 0) for p in pos]}")
    P(f"        head_offset = {head_offset.tolist()}（前半 0=前窗 a 半区、后半 {HEAD_SIZE}=当前窗 b 半区；**代码的 a 半区装前窗**）")
    P(f"        (position+1)%CR = {(position + 1) % CR} ⇒ 只有块尾 token 干活")
P("")
P("── 标签陷阱（论文 vs 官方参考实现） ──")
P("    论文 Eq.11：a 半区 = 当前窗、b 半区 = 前一个窗")
P("    官方参考实现（DeepSeek-V4-Pro inference/model.py 的 Compressor，L296 注释原话 the first half of")
P("    dims is for overlapping compression, second half for normal）：前半区 [..., :head_dim] 装前一个窗、")
P("    后半区装当前窗 —— 与论文 Eq.(11) 的 a/b **相反**（pin 的压缩核 head_offset 同序）")
P("    （下面这条对账在**本章的 NumPy 实现**里做：csa_compress_reference_layout 是**译自**官方 model.py 的")
P("      Compressor（投影 L297-L298、窗内加工 L337-L342）；官方代码本身未在本机执行——host 无 tilelang/")
P("      GPU 路径，官方代码只作只读蓝本，见 implementation/impl-notes.md）")
C_ref, S_ref = cp.csa_compress_reference_layout(
    H,
    np.hstack([W_kv_b, W_kv_a]),      # 前半区 = 前一个窗的 KV（= 论文的 C^b）
    np.hstack([W_z_b, W_z_a]),        # 前半区 = 前一个窗的 gate
    np.hstack([B_b, B_a]),            # 前半区 = 前一个窗的位置偏置
    M,
)
swapped = np.concatenate([S_ref[:, M:], S_ref[:, :M]], axis=1)
P(f"    产出逐位等价验证：max|C_comp - C_ref| = {np.abs(C_comp - C_ref).max():.8f}")
P(f"    权重是同一组数、只是两半区顺序相反：把官方参考实现的半区调头后 max|S - S_ref_swapped| = {np.abs(S - swapped).max():.8f}")
P(f"    （直接对减 max|S - S_ref| = {np.abs(S - S_ref).max():.8f} 非零——正是『同一套数学、半区命名相反』的形态）")

out = {
    "script": "run_m03_m04_csa.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"m": M, "c": C_DIM, "n": N, "n_entries": N // M, "window_width": 2 * M,
               "CR_pin": CR, "HEAD_SIZE_pin": HEAD_SIZE},
    "results": {
        "widget_weights": w.tolist(),
        "widget_z": z_col0.tolist(),
        "widget_c": c_col0.tolist(),
        "widget_weighted_sum": wavg,
        "widget_uniform": float(c_col0.mean()),
        "C_comp": C_comp.tolist(),
        "S_entry0_colsum": S[0].sum(axis=0).tolist(),
        "S_entry0_bhalf": S[0][M:].tolist(),
        "index_sets": [[a, b] for a, b in sets],
        "layout_equivalence_maxdiff": float(np.abs(C_comp - C_ref).max()),
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
