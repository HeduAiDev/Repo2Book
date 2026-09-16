"""ch28 m07（共享 KV 的 MQA Eq.18-19）/ m09（滑窗支路与压缩条目的合成）驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m07_m09_attention.py
产出：explainer/traces/run_m07_m09_attention.json（params + raw_stdout）。

素材来源：implementation/windowed_attention.py（论文忠实的小型参考实现）。
  A) m07：一条压缩条目**同时当 K 和 V**（同一个张量传两次）+ 多头 query 共享它 —— 「共享 KV」的字面含义；
     并给出 c^Q → 索引器 q（64×128）/ 主注意力 q（n_h×c=512）的分叉形状。
  B) m09：滑窗（最近 n_win 条**未压缩** KV）+ 压缩条目并进同一次注意力：两套口径（玩具 n_win=3 与
     config 口径 n_win=128）各自的 KV 轴组成，最后真跑一次合成后的注意力。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import windowed_attention as wa  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

P("== ch28 m07/m09 · 共享 KV 的 MQA + 滑窗与压缩条目的合成 ==")

# ── A) m07：一条条目两用 ────────────────────────────────────────────────────
P("")
P("── A) m07：CoreAttn(query=q, key=C^SprsComp, value=C^SprsComp)（Eq.19） ──")
q = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])          # 2 个 query 头 × 4 维
kv = np.array([[1.0, 2.0, 1.0, 0.0], [0.5, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 2.0]])   # 3 条压缩条目（K 与 V 同一份）
n_h, d = q.shape
P(f"    q 形状 = {q.shape}（n_h={n_h} 个 query 头 × c={d} 维）；kv 形状 = {kv.shape}（3 条压缩条目 × {d} 维）")
o, weights = wa.core_attention_mqa(q, kv)
P(f"    注意力权重（{n_h} 头 × 3 条）= {[R(r) for r in weights]}，每行和 = {R(weights.sum(axis=1))}")
P(f"    输出 o（{n_h} 头 × {d} 维）= {[R(r) for r in o]}")
P(f"    『共享 KV』的三个可观察后果：")
P(f"      ① kv 轴上没有 head 维（{kv.shape} 而不是 {n_h}×3×{d}）——条目在头之间共享；")
P(f"      ② K 与 V 是**同一个张量**：o 就是 weights @ kv，没有第二份 value 投影；")
kv2 = kv.copy()
kv2[0] += 1.0
o2, _ = wa.core_attention_mqa(q, kv2)
P(f"      ③ 动一条条目（kv[0] += 1.0）⇒ {n_h} 个头**同时**变（o 的变化 {R(o2 - o)}）——没有哪一头能绕过它。")
P("")
P(f"    c^Q 一次降维两处用（Eq.13 与 Eq.18）：同一个低秩潜向量 c^Q_t = h_t·W^DQ")
P(f"      → 索引器 q：c^Q · W^IUQ → n_h^I × c^I = 64 × 128（config: index_n_heads / index_head_dim）")
P(f"      → 主注意力 q：c^Q · W^UQ → n_h × c = n_h × 512（config: head_dim）")
P(f"    本例玩具：n_h^I=2、c^I=2、n_h={n_h}、c={d}（同构缩小）")

# ── B) m09：滑窗 + 压缩条目的合成 ──────────────────────────────────────────
P("")
P("── B) m09：KV = [最近 n_win 条未压缩 KV] + [压缩条目] ──")
M = 4
P("    玩具口径（n_win=3、m=4）与 config 口径（n_win=128）各走一遍：")
for n_win, pos_list in ((3, (0, 2, 8, 11)), (128, (8, 127, 128, 1000))):
    for pos in pos_list:
        r = wa.compose_attention_inputs(pos, M, n_win, rule="impl", topk=None)
        P(f"    n_win={n_win:3d} pos={pos:4d}: 滑窗 [{r['swa_start']}, {pos}] 共 {r['n_swa']} 条｜"
          f"候选压缩块 = ({pos}+1)//{M} = {r['candidates']} 条（本例 topk=None 不截断，HCA 读法）｜"
          f"KV 轴 = {r['n_swa']} + {r['n_compressed']} = {r['kv_len']} 条")
P("    两处口径要点：")
P(f"      ① 滑窗长度只看 n_win 与 pos（与压缩率 m 无关）：pos=8、n_win=3 ⇒ {wa.sliding_window_slice(8, 3)}；"
  f"n_win=128 ⇒ {wa.sliding_window_slice(8, 128)}（起手不足一窗就从头开始）")
P(f"      ② pos=2 时压缩条目 = {wa.compose_attention_inputs(2, M, 3)['n_compressed']} 条（(2+1)//4=0）："
  f"没攒满一块 ⇒ 这一拍只有滑窗看得见那 3 个 token 的原样 KV（『攒批时间差』）")
r_csa = wa.compose_attention_inputs(1_000_000 - 1, M, 128, rule="impl", topk=512)
P(f"      ③ CSA 层（topk=512）在 1M 处：候选 {r_csa['candidates']} 条 → 实看 min(512, 候选) = {r_csa['n_compressed']} 条 + 滑窗 {r_csa['n_swa']} 条")
r_hca = wa.compose_attention_inputs(1_000_000 - 1, 128, 128, rule="impl", topk=None)
P(f"      ④ HCA 层（无 topk）在 1M 处：候选 {r_hca['candidates']} 条 → 全看 {r_hca['n_compressed']} 条 + 滑窗 {r_csa['n_swa']} 条")
P(f"      ⑤ 滑窗是**加账**：每个带压缩机的层多背 128 条未压缩 KV（本例 KV 轴 {r_hca['kv_len']} 条里只有 {r_hca['n_compressed']} 条是压缩的）")
P("")
P("    合成后真跑一次（pos=8、m=4、n_win=3 ⇒ KV 轴 5 条）:")
POS, N_WIN = 8, 3
r = wa.compose_attention_inputs(POS, M, N_WIN)
P(f"      {r}")
kv_axis = np.array([
    [1.0, 0.0, 0.5, 0.0],   # 滑窗第 1 条（token 6，未压缩）
    [0.0, 1.0, 0.5, 0.5],   # 滑窗第 2 条（token 7）
    [0.5, 0.5, 1.0, 0.0],   # 滑窗第 3 条（token 8 = query 自己）
    [1.0, 1.0, 0.0, 0.0],   # 压缩条目 0（token 0-3 的软池化产物）
    [0.0, 0.5, 1.0, 0.5],   # 压缩条目 1（token 4-7）
])
q8 = np.array([[1.0, 0.0, 0.0, 1.0]])
o8, w8 = wa.core_attention_mqa(q8, kv_axis)
P(f"      KV 轴 5 条 = 3 条滑窗 + 2 条压缩条目（前者原样、后者是软池化产物，同一张表里没有区别）")
P(f"      权重（1 头 × 5 条）= {R(w8)}，和 = {R(w8.sum(axis=1))}")
P(f"      o = {R(o8)} ⇒ 3+2 两路在同一份权重里竞争（谁分多由 q·kv 决定，不分区）")

P("")
P("── 派生量（供正文直接引用） ──")
P(f"    1M·HCA 层的 KV 轴构成：{1_000_000 // 128} 条压缩 + 128 条滑窗 = {1_000_000 // 128 + 128} 条"
  f"（压缩条目占 {(1_000_000 // 128) / (1_000_000 // 128 + 128):.6f}）")
P(f"    1M·CSA 层的 KV 轴构成：512 条压缩 + 128 条滑窗 = 640 条（压缩条目占 {512 / 640:.6f}）")

out = {
    "script": "run_m07_m09_attention.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"n_heads": int(n_h), "head_dim": int(d), "n_entries": 3, "m": M, "n_win_toy": N_WIN, "n_win_real": 128},
    "results": {
        "weights": weights.tolist(),
        "o": o.tolist(),
        "compose_toy": {str(p): wa.compose_attention_inputs(p, M, N_WIN) for p in (0, 2, 8, 11)},
        "compose_real": {str(p): wa.compose_attention_inputs(p, M, 128) for p in (8, 127, 128, 1000)},
        "compose_csa_1m": wa.compose_attention_inputs(1_000_000 - 1, M, 128, topk=512),
        "compose_hca_1m": wa.compose_attention_inputs(1_000_000 - 1, 128, 128),
        "kv_axis_rows": int(kv_axis.shape[0]),
        "w8": w8.tolist(),
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
