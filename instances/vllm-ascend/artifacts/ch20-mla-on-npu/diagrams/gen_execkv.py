#!/usr/bin/env python3
"""ch20 exec_kv 融合算子：npu_kv_rmsnorm_rope_cache 一把做 RMSNorm+RoPE+写 KV cache，
decode/prefill 靠返回值位置区分用途（前两个 vs 后两个）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1280, 670
TXT = "#1e293b"
SUB = "#64748b"
IN = "#334155"    # 输入：灰蓝
OP = "#7c3aed"    # 融合算子：紫
DEC = "#15803d"   # decode 绿
PRE = "#b45309"   # prefill 橙

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-size="24" font-weight="bold" fill="{TXT}">npu_kv_rmsnorm_rope_cache：一个算子融合 RMSNorm + RoPE + 写分页 cache</text>')
L.append(f'<text x="{W/2}" y="64" text-anchor="middle" font-size="14" fill="{SUB}">decode 与 prefill 调同一个算子，只靠返回值的位置（前两个 / 后两个）区分用途</text>')


def box(cx, cy, w, h, fill, stroke, lines, fw="bold"):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 4) + s * 0.34
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw if i == 0 else "normal"}" fill="{stroke if i == 0 else TXT}">{esc(t)}</text>')


def varrow(x, y1, y2, label=None, color="#64748b"):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#ar)"/>')
    if label:
        L.append(f'<text x="{x + 10}" y="{(y1 + y2) / 2 + 4}" font-size="12" fill="#475569">{esc(label)}</text>')


def diagarrow(x1, y1, x2, y2, color="#64748b"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#ar)"/>')


CX = W / 2

# ---- 输入行：5 个入参盒 ----
IY, IH, IW, GAP = 100, 56, 226, 18
inputs = [
    [("kv_no_split", 14), ("[B,N,S=1,Lkv+R=576]", 12), ("(512+64)", 11)],
    [("kv_a_layernorm.weight", 13.5), ("RMSNorm 权重", 12)],
    [("cos, sin", 14), ("RoPE 旋转角", 12)],
    [("slots", 14), ("写 cache 的槽位", 12)],
    [("kv_cache[0]/[1]", 14), ("分页 KV 物理块", 12)],
]
n = len(inputs)
total_w = n * IW + (n - 1) * GAP
x0 = CX - total_w / 2
centers = []
for i, lines in enumerate(inputs):
    cx = x0 + IW / 2 + i * (IW + GAP)
    centers.append(cx)
    box(cx, IY, IW, IH, "#f1f5f9", IN, lines, fw="bold")

# ---- 汇聚到融合算子 ----
OPY, OPH, OPW = 220, 150, 980
merge_y = OPY - 24
for cx in centers:
    L.append(f'<line x1="{cx}" y1="{IY + IH}" x2="{cx}" y2="{merge_y}" stroke="#94a3b8" stroke-width="1.6"/>')
L.append(f'<line x1="{centers[0]}" y1="{merge_y}" x2="{centers[-1]}" y2="{merge_y}" stroke="#94a3b8" stroke-width="1.6"/>')
varrow(CX, merge_y, OPY, color=OP)

# 融合算子外框（标题贴顶部，不与内部子步骤共享垂直居中，避免相压）
L.append(f'<rect x="{CX - OPW / 2}" y="{OPY}" width="{OPW}" height="{OPH}" rx="9" fill="#f5f3ff" stroke="{OP}" stroke-width="1.6"/>')
L.append(f'<text x="{CX}" y="{OPY + 30}" text-anchor="middle" font-size="17" font-weight="bold" fill="{OP}">npu_kv_rmsnorm_rope_cache</text>')

# 算子内部三步（融合算子框内的三个子步骤）
sub_y = OPY + 56
sub_w, sub_gap = 300, 22
sub_labels = [
    ("① RMSNorm", "对 kv_c 做归一化（kv_a_layernorm）"),
    ("② RoPE", "对 k_pe 施加旋转位置编码（cos/sin）"),
    ("③ 写分页 cache", "按 slots 写入 kv_cache（cache_mode='PA'/'PA_NZ'）"),
]
sub_total = 3 * sub_w + 2 * sub_gap
sx0 = CX - sub_total / 2
sub_centers = []
for i, (a, b) in enumerate(sub_labels):
    scx = sx0 + sub_w / 2 + i * (sub_w + sub_gap)
    sub_centers.append(scx)
    box(scx, sub_y, sub_w, 68, "#ede9fe", OP, [(a, 14.5), (b, 11.5)], fw="bold")
for i in range(2):
    diagarrow(sub_centers[i] + sub_w / 2, sub_y + 34, sub_centers[i + 1] - sub_w / 2, sub_y + 34, color=OP)

# 算子返回值说明——放在紫框内部下沿，不与后续箭头/连线共享同一行
L.append(f'<text x="{CX}" y="{OPY + OPH - 14}" text-anchor="middle" font-size="12.5" fill="{OP}">算子共返回 4 个值：decode 只取前两个，prefill 传 is_output_kv=True 后取后两个</text>')

# ---- 单一返回值：4 个位置槽（k_pe, k_nope, k_pe, k_nope）----
ret_y = OPY + OPH + 46
varrow(CX, OPY + OPH, ret_y, color=OP)

ret_w, ret_gap = 190, 16
ret_labels = ["① k_pe", "② k_nope", "③ k_pe", "④ k_nope"]
ret_total = 4 * ret_w + 3 * ret_gap
rx0 = CX - ret_total / 2
ret_centers = []
for i, lab in enumerate(ret_labels):
    rcx = rx0 + ret_w / 2 + i * (ret_w + ret_gap)
    ret_centers.append(rcx)
    col = DEC if i < 2 else PRE
    fill = "#f0fdf4" if i < 2 else "#fff7ed"
    box(rcx, ret_y, ret_w, 44, fill, col, [(lab, 15)], fw="bold")

# ---- 分叉：decode 取①②，prefill 取③④ ----
fork_y = ret_y + 44 + 40
DX = (ret_centers[0] + ret_centers[1]) / 2
PX = (ret_centers[2] + ret_centers[3]) / 2
for c in ret_centers[:2]:
    L.append(f'<line x1="{c}" y1="{ret_y + 44}" x2="{DX}" y2="{fork_y - 10}" stroke="{DEC}" stroke-width="1.6"/>')
for c in ret_centers[2:]:
    L.append(f'<line x1="{c}" y1="{ret_y + 44}" x2="{PX}" y2="{fork_y - 10}" stroke="{PRE}" stroke-width="1.6"/>')
varrow(DX, fork_y - 10, fork_y + 10, color=DEC)
varrow(PX, fork_y - 10, fork_y + 10, color=PRE)

# ---- decode / prefill 结论盒 ----
concl_y = fork_y + 14
box(DX, concl_y, 420, 78,
    "#f0fdf4", DEC,
    [("decode：取第 1/2 个返回值", 15),
     ("cache_mode='PA'，无 is_output_kv 标志", 12),
     ("k_nope=写进 cache 的隐向量，直接当 K=V 做 MQA", 12.5)],
    fw="bold")
box(PX, concl_y, 420, 78,
    "#fff7ed", PRE,
    [("prefill：is_output_kv=True，取第 3/4 个返回值", 14.5),
     ("同一算子、多传一个标志", 12),
     ("k_nope=未量化输出 KV，交给 kv_b_proj 显式解压", 12.5)],
    fw="bold")

foot_y = concl_y + 78 + 34
L.append(f'<text x="{W/2}" y="{foot_y}" text-anchor="middle" font-size="12.5" fill="{SUB}">同一个 npu_kv_rmsnorm_rope_cache：exec_kv_decode 不传 is_output_kv（默认取前两个返回值），exec_kv_prefill 传 is_output_kv=True（取后两个）</text>')

L.append('</svg>')
open("npu-kv-rmsnorm-rope-cache.svg", "w").write('\n'.join(L))
print("wrote npu-kv-rmsnorm-rope-cache.svg")
