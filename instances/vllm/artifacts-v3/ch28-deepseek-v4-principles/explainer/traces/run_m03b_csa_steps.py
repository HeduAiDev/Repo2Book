"""ch28 m03/m04 步进补充（CSA 压缩的「四组投影 / 位置偏置 / 伪码阶梯 / 跨拍续窗」）。

跑法：/d/Env/Miniconda/python explainer/traces/run_m03b_csa_steps.py
产出：explainer/traces/run_m03b_csa_steps.json（params + raw_stdout）。

为什么要补这一份：run_m03_m04_csa.json 讲的是「完整式跑出来的数」；本章按「密集小图」
规范（STYLE-dense-mini-figures.md）要把 CSA 那件拆成步进小图，其中有四步的数不在它里面：
  A) 四组投影各是什么（Eq.9-10）——哪些是要被混合的向量、哪些是决定怎么混合的分数原料；
  B) 位置偏置 B 的**非平凡性**：没有 B 时窗内 softmax 退化成等权平均（机制失效），
     有 B 时权重才分得开——四档伪码阶梯（sec1 等权 → sec2 +B → sec3 +Z 投影 → sec4 完整）
     在同一组输入上的输出对照；
  C) 跨拍续窗（状态维护）：把 16 个 token 拆成 12 + 4 两拍跑，第 4 条条目应与一次性 16 个
     token 跑出来的第 4 条**逐位相同**；未满一窗的 token 留在状态里等下一拍；
  D) 官方参考实现的状态布局转写（DeepSeek-V4-Pro inference/model.py:L303-L304 的 kv_state/
     score_state 注释、L347-L354 的 decode 分支、L350-L351 的两半区 cat）——本段是**逐行转写
     在本机 NumPy 上跑的**，不是官方 torch 代码执行（host 无 tilelang/GPU 路径）。

素材来源：implementation/compressor.py（论文忠实的小型参考实现）。
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


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

# ── 玩具参数：m=4、c=2、d=3、n=16（前 12 个 token 是第一拍、后 4 个是第二拍） ──
M, C_DIM, D, N_FIRST, N_ALL = 4, 2, 3, 12, 16
H = np.array([
    [1.0, 0.5, -1.0], [0.5, 1.0, 0.0], [-0.5, 0.5, 1.0], [1.0, -1.0, 0.5],
    [0.0, 1.0, 1.0], [1.0, 1.0, -0.5], [0.5, -0.5, 0.5], [-1.0, 0.0, 1.0],
    [1.0, 0.5, 0.5], [0.5, 0.0, -0.5], [0.0, 1.0, -1.0], [1.0, 0.5, 1.0],
    [0.5, -1.0, 0.0], [1.0, 0.0, 0.5], [-0.5, 1.0, 0.5], [0.5, 0.5, -1.0],
])
W_kv_a = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
W_kv_b = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0]])
W_z_a = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
W_z_b = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
B_a = np.array([[0.5, 0.2], [0.4, 0.1], [0.3, 0.0], [0.2, -0.1]])
B_b = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

P("== ch28 m03/m04 步进补充 · CSA 压缩（m=4, c=2, d=3, n=16 = 12 + 4 两拍） ==")
P(f"参数：m = {M}（每 {M} 个 token 压 1 条）、c = {C_DIM}、hidden d = {D}；第一拍 {N_FIRST} 个 token、"
  f"第二拍 {N_ALL - N_FIRST} 个 token")

# ── A) 四组投影：Eq.(9)(10) ───────────────────────────────────────────────────
P("")
P("── A) 四组投影：一条 hidden 变出两组『要被混合的向量』+ 两组『决定怎么混合的分数原料』──")
C_a, C_b, Z_a, Z_b = cp.csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)
P(f"    H 形状 = {H.shape}（n 个 token × d 维）；四组 W 形状 = {W_kv_a.shape}（d × c）")
P(f"    C^a = H·W^aKV 形状 = {C_a.shape}｜C^b = H·W^bKV 形状 = {C_b.shape}"
  f"（Eq.9：两组**要被混合**的向量）")
P(f"    Z^a = H·W^aZ 形状 = {Z_a.shape}｜Z^b = H·W^bZ 形状 = {Z_b.shape}"
  f"（Eq.10：两组**权重原料**——注意它不是权重，还要过 softmax）")
P(f"    token 4（第 2 个窗的第 1 个格子）的四个投影值：C^a = {R(C_a[4])}｜C^b = {R(C_b[4])}"
  f"｜Z^a = {R(Z_a[4])}｜Z^b = {R(Z_b[4])}")
sets = cp.csa_entry_input_index_sets(M, N_ALL // M)
P(f"    第 i=1 条条目取哪些行：C^a/Z^a 取当前窗 {sets[1][0]}、C^b/Z^b 取前一个窗 {sets[1][1]}")
P(f"    ⇒ 四组投影不是四个冗余，而是『a 半区 = 当前窗、b 半区 = 前一个窗』这一读法的原料"
  f"（pin 侧压成一条 GEMM：MergedColumnParallelLinear 一枪出 coff·head_dim × 2 组）")

# ── B) 位置偏置的非平凡性 + 四档伪码阶梯 ──────────────────────────────────────
P("")
P("── B) 位置偏置 B 的非平凡性：它是『窗内第几格』唯一的信号源 ──")
P("    （Z 是内容投影：同一窗内两个内容相同的格子，在 Z 眼里没有区别；要学出『最近的一格更重要』")
P("      只能靠 B —— 下面把 B 摘掉看会发生什么）")
z_equal = np.zeros((M, C_DIM))          # 窗内 4 个格子的分数原料**完全相同**
c_win = np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 9.0]])   # 窗内 4 个 token 的 C^a
w_no_b = cp.softmax_row(z_equal, axis=0)
out_no_b = np.sum(w_no_b * c_win, axis=0)
out_mean = c_win.mean(axis=0)
P(f"    窗内 4 个格子的 Z 相同（{R(z_equal[0])} ×4）且**不加 B** ⇒ softmax 权重 = {R(w_no_b[:, 0])}")
P(f"    4 列权重逐列全等 = {[bool(np.allclose(w_no_b[:, j], w_no_b[0, j])) for j in range(C_DIM)]}"
  f"｜加权和 = {R(out_no_b)}｜算术平均 = {R(out_mean)}｜逐位相同 = {bool(np.allclose(out_no_b, out_mean))}")
P(f"    ⇒ 摘掉 B 后『软池化』在等分分数上退化成平均池化：格子之间再无差别，学不出『哪格重要』")
z_with_b = z_equal + B_a
w_with_b = cp.softmax_row(z_with_b, axis=0)
out_with_b = np.sum(w_with_b * c_win, axis=0)
P(f"    同一组 Z 加上可学习位置偏置 B_a（行 = 窗内第几格）= {R(B_a[:, 0])} 后："
  f"权重 = {R(w_with_b[:, 0])}")
P(f"    权重不再全等 = {not bool(np.allclose(w_with_b[:, 0], w_with_b[0, 0]))}"
  f"｜输出 = {R(out_with_b)} vs 平均 {R(out_mean)}｜差 = {R(np.abs(out_with_b - out_mean))}")
P("")
P("    四档伪码阶梯（同一组输入 c_win，每档只加一件东西）：")
sec1 = c_win.mean(axis=0)
P(f"      sec1 等权平均（基础版）          = {R(sec1)}")
sec2 = np.sum(cp.softmax_row(np.full((M, C_DIM), 0.0) + B_a, axis=0) * c_win, axis=0)
P(f"      sec2 + 可学习位置偏置 B          = {R(sec2)}")
sec3 = np.sum(cp.softmax_row(Z_a[0:M], axis=0) * c_win, axis=0)
P(f"      sec3 + 分数投影 Z（不再恒定）    = {R(sec3)}")
C_comp, S_all = cp.csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, M)
sec4 = C_comp[0]
P(f"      sec4 完整（2m 交叠 + 跨 2m 归一）= {R(sec4)}（= csa_compress 的第 0 条条目）")
diffs = [float(np.abs(a - b).max()) for a, b in zip([sec1, sec2, sec3], [sec2, sec3, sec4])]
P(f"    四档两两差（max|·|）= {[round(d, 6) for d in diffs]} ⇒ 每加一件输出都变（非退化）")
P(f"    sec4 与 sec1 的差 = {round(float(np.abs(sec4 - sec1).max()), 6)}：这就是『学出来的权重』"
  f"相对『谁都一样』的位移量")
P(f"    2m 窗的权重逐列和（第 0 条条目）= {R(S_all[0].sum(axis=0))}（跨 {2 * M} 个元素归一）")

# ── C) 跨拍续窗：16 个 token 拆成 12 + 4 两拍 ─────────────────────────────────
P("")
P("── C) 跨拍续窗（状态维护）：第二拍的第 1 条条目 = 一次性 prefill 的第 4 条条目 ──")
C1, S1 = cp.csa_compress(H[:N_FIRST], W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, M)
P(f"    第一拍（前 {N_FIRST} 个 token）：产出条目 {C1.shape[0]} 条（{N_FIRST} // {M} = {N_FIRST // M}），"
  f"未满一窗的 token 数 = {N_FIRST} % {M}（整除 ⇒ 没有尾巴，尾巴那一路另看下面 E 段）")
prior_a = (C_b[N_FIRST - M:N_FIRST], Z_b[N_FIRST - M:N_FIRST])   # 上拍末窗的 b 投影
P(f"    交给第二拍的状态 = 上拍末窗（token {list(range(N_FIRST - M, N_FIRST))}）的 b 投影"
  f"（C^b/Z^b 各 {prior_a[0].shape}）——pin 侧就是压缩机 state_cache 里已落好的那半区")
C2, S2 = cp.csa_compress(H[N_FIRST:], W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, M, prior_a=prior_a)
C_full, S_full = cp.csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, M)
P(f"    第二拍（后 {N_ALL - N_FIRST} 个 token）：产出条目 {C2.shape[0]} 条"
  f"（{N_ALL - N_FIRST} // {M} = {(N_ALL - N_FIRST) // M}）⇒ 两拍合计 {C1.shape[0] + C2.shape[0]} 条，"
  f"与一次性 {N_ALL} 个 token 的 {C_full.shape[0]} 条相同")
same = bool(np.allclose(C2[0], C_full[3]))
P(f"    第二拍第 1 条（= 全局第 4 条）C^Comp = {R(C2[0])}")
P(f"    一次性 prefill 的第 4 条       C^Comp = {R(C_full[3])}")
P(f"    逐位相同 = {same}｜max|差| = {float(np.abs(C2[0] - C_full[3]).max()):.8f}"
  f" ⇒ 跨拍续窗与一次性 prefill 等价（状态就是那条『接缝』）")
C2_noprior, _ = cp.csa_compress(H[N_FIRST:], W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, M)
P(f"    对照（第二拍不给状态）：第 1 条 C^Comp = {R(C2_noprior[0])}"
  f"｜与有状态差 = {R(np.abs(C2_noprior[0] - C2[0]))} ⇒ 状态不是可选项，是接缝的一半")
P(f"    第二拍第 1 条的 b 半区权重逐列和 = {R(S2[0][M:].sum(axis=0))}"
  f"｜a 半区 = {R(S2[0][:M].sum(axis=0))}（两半区都在贡献）")

# ── E) 未满一窗的尾巴：留在状态里等下一拍（pin 的 state_cache 存量） ──────────
P("")
P("── E) 未满一窗的尾巴：攒不满就不产出，留在状态里等下一拍 ──")
acc = cp.WindowAccumulator(M, C_DIM)
P(f"    逐 token 喂 14 个 token（position 0..13）进 WindowAccumulator(m={M})：")
for pos in range(14):
    kv_t = np.array([[pos + 1.0, pos + 0.5]])
    gate_t = np.array([[0.1 * pos, -0.1 * pos]])
    chunk_kv, _, first_window_position, complete = acc.push(kv_t, gate_t, pos)
    if complete:
        P(f"      pos={pos}: 交出 {chunk_kv.shape[0] // M} 个完整窗（窗起点位置 = {first_window_position}）"
          f"｜已产出条目累计 = {acc.entry_count}｜buffer 里剩 {acc.buffered} 行")
P(f"    14 个 token 喂完：条目累计 = {acc.entry_count} 条（14 // {M} = {14 // M}）"
  f"｜buffer 里剩 {acc.buffered} 行（= 14 % {M}，未满一窗的那 {acc.buffered} 个 token 留在状态里）")
P(f"    条目槽位（压缩坐标系）：pos=13 的条目号 = {acc.entry_slot(13)}、pos=12 的 = {acc.entry_slot(12)}"
  f"（同一条条目覆盖 token {M * acc.entry_slot(12)}..{M * acc.entry_slot(12) + M - 1}）")
P(f"    ⇒ 『攒批时间差』的算术形态：最新的 {acc.buffered} 个 token 这一拍没有压缩条目可看，"
  f"只有滑窗看得见它们的原样 KV（本包跑出的对照见 run_m07_m09_attention.json 的 pos=2 行）")

# ── D) 官方参考实现的状态布局（逐行转写的本机 NumPy 仿真） ───────────────────
P("")
P("── D) 官方参考实现的**decode 分支**状态布局（转写自 DeepSeek-V4-Pro inference/model.py:L303-L304、L347-L354）──")
P("    （本段是**逐行转写在本机 NumPy 上跑**的：官方 torch 代码未在本机执行，host 无 tilelang/GPU 路径；")
P("      prefill 分支（L337-L342 的 unflatten + overlap_transform）已在 run_m03_m04_csa.json 里对过账）")
ratio, d = M, C_DIM
kv_state = np.zeros((2 * ratio, d))
score_state = np.full((2 * ratio, d), -np.inf)
# 块尾 token（position = 4i+3）才产出；decode 分支：先写当前窗半区（行 ratio+slot），
# 到块尾把『前一窗半区（行 :ratio）』与『当前窗半区（行 ratio:）』拼成 2m 行做一次 softmax。
rows_log = []
for block in range(2):
    for slot in range(ratio):
        pos = block * ratio + slot
        token_vec = np.array([pos + 1.0, pos + 0.5])          # 玩具：该 token 的 kv 值
        score_state[ratio + slot] = np.array([2.0 - slot, 0.5 * slot])
        kv_state[ratio + slot] = token_vec
        if slot == ratio - 1:                                  # 块尾：拼 2m 行 → 一条条目
            window = np.vstack([kv_state[:ratio], kv_state[ratio:]])
            swin = np.vstack([score_state[:ratio], score_state[ratio:]])
            w = cp.softmax_row(swin, axis=0)
            entry = np.sum(w * window, axis=0)
            rows_log.append(
                (pos, window[:, 0].tolist(), w[:, 0].tolist(), entry.tolist(),
                 kv_state[:ratio, 0].tolist(), kv_state[ratio:, 0].tolist())
            )
            kv_state[:ratio] = kv_state[ratio:]                # 当前窗滚成下一次的重叠窗
            score_state[:ratio] = score_state[ratio:]
P(f"    state 形状 = ({2 * ratio}, {d})：行 0..{ratio - 1} = 重叠窗（前一个窗）、"
  f"行 {ratio}..{2 * ratio - 1} = 当前窗（L303-L304 的注释口径）；宽度 = 2 × head_dim（L294 的 ape 同宽）")
for pos, win, w, entry, lo, hi in rows_log:
    P(f"    块尾 pos={pos}: 2m 行拼出的输入（第 0 列）= {R(win)}")
    P(f"                  其中行 0..{ratio - 1}（重叠窗，上一拍留下的）+ 行 {ratio}..{2 * ratio - 1}（当前窗）")
    P(f"                  = [{R(lo)}] + [{R(hi)}]（这两段来自 state 的**两个半区**）")
    P(f"                  softmax 权重（第 0 列）= {R(w)}（和 = {round(float(sum(w)), 6)}）"
      f" ⇒ 条目 = {R(entry)}")
P(f"    ⇒ 块尾 token 的那一拍：2m 个输入有一半来自**上一拍留在 state 里的那半区**"
  f"（L350-L351 的 torch.cat([state[:, :ratio, :d], state[:, ratio:, d:]], dim=1)）；")
P(f"      产出后执行 L353-L354 的滚动（state[:, :ratio] = state[:, ratio:]）——当前窗变成下一次的重叠窗。")

out = {
    "script": "run_m03b_csa_steps.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"m": M, "c": C_DIM, "d": D, "n_first": N_FIRST, "n_all": N_ALL,
               "window_width": 2 * M},
    "results": {
        "projection_shapes": {"C_a": list(C_a.shape), "C_b": list(C_b.shape),
                              "Z_a": list(Z_a.shape), "Z_b": list(Z_b.shape)},
        "c_a_token4": C_a[4].tolist(), "c_b_token4": C_b[4].tolist(),
        "z_a_token4": Z_a[4].tolist(), "z_b_token4": Z_b[4].tolist(),
        "weights_no_bias": w_no_b[:, 0].tolist(),
        "out_no_bias": out_no_b.tolist(), "out_mean": out_mean.tolist(),
        "same_no_bias_vs_mean": bool(np.allclose(out_no_b, out_mean)),
        "weights_with_bias": w_with_b[:, 0].tolist(),
        "out_with_bias": out_with_b.tolist(),
        "ladder": {"sec1_mean": sec1.tolist(), "sec2_ape": sec2.tolist(),
                   "sec3_z": sec3.tolist(), "sec4_full": sec4.tolist(),
                   "pairwise_max_abs_diff": [round(x, 6) for x in diffs]},
        "chain_entry_first_of_second_call": C2[0].tolist(),
        "full_prefill_entry3": C_full[3].tolist(),
        "chain_equals_full": same,
        "no_prior_entry": C2_noprior[0].tolist(),
        "entries_per_call": [int(C1.shape[0]), int(C2.shape[0])],
        "chunk_remainder_case": {"n_tokens": 14, "entries": acc.entry_count,
                                 "buffered": acc.buffered,
                                 "entry_slot_pos12": acc.entry_slot(12),
                                 "entry_slot_pos13": acc.entry_slot(13)},
        "state_rows": {"lower_half_pos7": rows_log[1][4], "upper_half_pos7": rows_log[1][5]},
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
