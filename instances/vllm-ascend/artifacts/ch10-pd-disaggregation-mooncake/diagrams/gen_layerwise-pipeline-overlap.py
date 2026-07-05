#!/usr/bin/env python3
"""逐层流水线重叠：prefiller 逐层 compute vs 后台串行 transfer 在时间轴上错位重叠。
数字来自 explainer/traces/layerwise.json（part_a_real_send_queue / part_b_timing_L4 / frac_hidden_by_L）。
"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


# ── 数据（全部来自 traces/layerwise.json，无即兴数字）──
L = 4
T_C = 10
T_X = 6
COMPUTE_WINDOWS = [(0, 10), (10, 20), (20, 30), (30, 40)]
TRANSFER_WINDOWS = [(10, 16), (20, 26), (30, 36), (40, 46)]
VERDICTS = ["hidden", "hidden", "hidden", "exposed"]
SEQ_TOTAL = 64
PIPE_TOTAL = 46
HIDDEN_N = 3
EXPOSED_N = 1
FRAC_L4 = 0.75
FRAC_L80 = 0.9875
NUM_SEND_TASKS = 4

# ── 版式常量：时间轴按 SEQ_TOTAL 定标，纵向布局全部由常量累加算出（零手写魔数）──
MARGIN_L = 190
PXU = 10.5  # px per time unit
T_MAX = SEQ_TOTAL


def xt(t):
    return MARGIN_L + t * PXU


TIMELINE_RIGHT = xt(T_MAX)
PANEL_X = TIMELINE_RIGHT + 55
PANEL_W = 300
W = int(PANEL_X + PANEL_W + 30)

TITLE_H = 34 + 24          # 标题两行
LEGEND_Y = TITLE_H + 46    # 图例基线
BASE_Y = LEGEND_Y + 32     # baseline 对照条顶
AXIS_Y = BASE_Y + 24 + 22  # 时间轴线
COMPUTE_Y = AXIS_Y + 20 + 46  # 轴刻度数字(AXIS_Y+20)与重叠标注(COMPUTE_Y-12)留足间距,避免文字相撞
COMPUTE_H = 56
TRANSFER_Y = COMPUTE_Y + COMPUTE_H + 54
TRANSFER_H = 44
TAIL_Y = TRANSFER_Y + TRANSFER_H + 22
PIPE_Y = TAIL_Y + 34
PANEL_Y0 = BASE_Y - 8
PANEL_H = PIPE_Y + 30 - PANEL_Y0
CONCL_Y = PANEL_Y0 + PANEL_H + 34
H = int(CONCL_Y + 24)

S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="arr" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto">'
          '<path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('<marker id="arrBlue" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto">'
          '<path d="M0,0 L10,3.5 L0,7 Z" fill="#1d4ed8"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def rect(x, y, w, h, fill, stroke, rx=6, sw=1.5, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' opacity="{opacity}"' if opacity is not None else ''
    S.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def txt(x, y, t, size=13, anchor="middle", weight="normal", fill="#1e293b", italic=False, mono=False):
    st = ' font-style="italic"' if italic else ''
    fam = ' font-family="monospace"' if mono else ''
    S.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
              f'font-weight="{weight}" fill="{fill}"{st}{fam}>{esc(t)}</text>')


def arrow_line(x1, y1, x2, y2, stroke="#334155", sw=1.5, dash=None, marker="arr"):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    S.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
              f'stroke="{stroke}" stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def line(x1, y1, x2, y2, stroke="#334155", sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    S.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
              f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


# ── 标题 ──
txt(W / 2, 34, "逐层流水线：第 i 层传输与第 i+1 层计算在时间轴上错位重叠", 20, "middle", "bold", "#0f172a")
txt(W / 2, 58, "save_kv_layer 每算完一层立刻推该层 KV 入后台串行发送线程（L=4 小例，t_c=10 / t_x=6）",
    13, "middle", "normal", "#64748b")

# ── 图例 ──
legend = [
    ("#bfdbfe", "#3b82f6", "prefiller compute"),
    ("#bbf7d0", "#22c55e", "transfer · hidden（被后继计算盖住）"),
    ("#fecaca", "#ef4444", "transfer · exposed（无后继计算可藏）"),
    ("#e2e8f0", "#94a3b8", "baseline：整段算完再一次性传输"),
]
lx = MARGIN_L
for fill, stroke, label in legend:
    rect(lx, LEGEND_Y - 12, 18, 14, fill, stroke, rx=3, sw=1.3)
    txt(lx + 24, LEGEND_Y, label, 11.5, "start", "normal", "#334155")
    lx += 24 + 11.5 * len(label) * 0.62 + 26

# ── baseline 对照条（顺序总时长 64）──
txt(MARGIN_L, BASE_Y - 8, "对照：整段 prompt 算完再统一发送（baseline，无重叠）", 12, "start", "italic", "#64748b")
rect(xt(0), BASE_Y, xt(SEQ_TOTAL) - xt(0), 24, "#e2e8f0", "#94a3b8", rx=4, sw=1.3)
txt((xt(0) + xt(SEQ_TOTAL)) / 2, BASE_Y + 17, f"顺序总时长 = {SEQ_TOTAL}", 12.5, "middle", "bold", "#475569")

# ── 时间轴 ──
line(xt(0), AXIS_Y, xt(T_MAX), AXIS_Y, "#94a3b8", 1.5)
for t in sorted(set([0, 10, 20, 30, 40, PIPE_TOTAL, SEQ_TOTAL])):
    line(xt(t), AXIS_Y - 5, xt(t), AXIS_Y + 5, "#64748b", 1.3)
    txt(xt(t), AXIS_Y + 20, str(t), 11, "middle", "normal", "#64748b")
txt(xt(T_MAX) + 8, AXIS_Y + 4, "t（墙钟）", 11.5, "start", "italic", "#64748b")

# ── 泳道标签 ──
txt(20, COMPUTE_Y + COMPUTE_H / 2 - 4, "Prefiller worker", 13.5, "start", "bold", "#1d4ed8")
txt(20, COMPUTE_Y + COMPUTE_H / 2 + 14, "逐层 compute", 12, "start", "normal", "#2563eb")
txt(20, TRANSFER_Y + TRANSFER_H / 2 - 12, "后台线程", 13.5, "start", "bold", "#b91c1c")
txt(20, TRANSFER_Y + TRANSFER_H / 2 + 4, "KVCacheSendingLayerThread", 11, "start", "normal", "#dc2626")
txt(20, TRANSFER_Y + TRANSFER_H / 2 + 20, f"串行 FIFO · SendTask×{NUM_SEND_TASKS}", 11, "start", "normal", "#dc2626")

# ── 重叠高亮带（先画，压在底层）：第1层 transfer[10,16] 落在第2层 compute[10,20] 内 ──
band_x0 = xt(TRANSFER_WINDOWS[0][0])
band_x1 = xt(COMPUTE_WINDOWS[1][1])
rect(band_x0, COMPUTE_Y - 6, band_x1 - band_x0, (TRANSFER_Y + TRANSFER_H) - (COMPUTE_Y - 6),
     "#0ea5e9", "#0284c7", rx=4, sw=1.6, dash="5,4", opacity=0.12)
txt((band_x0 + band_x1) / 2, COMPUTE_Y - 12,
    "重叠：第1层 transfer[10,16] 落在第2层 compute[10,20] 内", 11.5, "middle", "bold", "#0369a1")

# ── compute 条 + save_kv_layer 箭头 ──
for i, (c0, c1) in enumerate(COMPUTE_WINDOWS):
    x0, x1 = xt(c0), xt(c1)
    rect(x0, COMPUTE_Y, x1 - x0, COMPUTE_H, "#bfdbfe", "#3b82f6", rx=5, sw=1.6)
    txt((x0 + x1) / 2, COMPUTE_Y + COMPUTE_H / 2 - 3, f"layer {i}", 12.5, "middle", "bold", "#1e40af")
    txt((x0 + x1) / 2, COMPUTE_Y + COMPUTE_H / 2 + 14, f"compute [{c0},{c1}]", 10.5, "middle", "normal", "#1d4ed8")
    # 每条右端向下箭头 = save_kv_layer 入队
    arrow_line(x1, COMPUTE_Y + COMPUTE_H, x1, TRANSFER_Y - 4, "#1d4ed8", 1.6, dash="3,3", marker="arrBlue")
txt((xt(COMPUTE_WINDOWS[0][1]) + xt(COMPUTE_WINDOWS[-1][0])) / 2, COMPUTE_Y + COMPUTE_H + 20,
    "save_kv_layer：每层一算完就 put 进发送队列（fire-and-forget，×4）", 11, "middle", "italic", "#1d4ed8")

# ── transfer 条 ──
for i, (t0, t1) in enumerate(TRANSFER_WINDOWS):
    x0, x1 = xt(t0), xt(t1)
    hidden = VERDICTS[i] == "hidden"
    fill, stroke = ("#bbf7d0", "#22c55e") if hidden else ("#fecaca", "#ef4444")
    rect(x0, TRANSFER_Y, x1 - x0, TRANSFER_H, fill, stroke, rx=5, sw=1.6)
    txt((x0 + x1) / 2, TRANSFER_Y + TRANSFER_H / 2 - 2, f"transfer [{t0},{t1}]", 10.5, "middle", "bold",
        "#166534" if hidden else "#991b1b")
    txt((x0 + x1) / 2, TRANSFER_Y + TRANSFER_H / 2 + 14, "hidden" if hidden else "exposed", 10.5, "middle",
        "normal", "#15803d" if hidden else "#b91c1c")

# ── 末层暴露尾巴标注 ──
tail_x0, tail_x1 = xt(COMPUTE_WINDOWS[-1][1]), xt(TRANSFER_WINDOWS[-1][1])
line(tail_x0, TAIL_Y, tail_x1, TAIL_Y, "#b91c1c", 1.4)
line(tail_x0, TAIL_Y - 5, tail_x0, TAIL_Y + 5, "#b91c1c", 1.4)
line(tail_x1, TAIL_Y - 5, tail_x1, TAIL_Y + 5, "#b91c1c", 1.4)
txt((tail_x0 + tail_x1) / 2, TAIL_Y + 18, f"暴露尾巴 t_x={T_X}（末层无后继计算可藏）", 11, "middle", "bold", "#b91c1c")

# ── 流水线总长标注 ──
line(xt(0), PIPE_Y, xt(PIPE_TOTAL), PIPE_Y, "#0f172a", 1.6)
line(xt(0), PIPE_Y - 5, xt(0), PIPE_Y + 5, "#0f172a", 1.4)
line(xt(PIPE_TOTAL), PIPE_Y - 5, xt(PIPE_TOTAL), PIPE_Y + 5, "#0f172a", 1.4)
txt((xt(0) + xt(PIPE_TOTAL)) / 2, PIPE_Y + 18, f"逐层流水线总时长 = {PIPE_TOTAL}（省下 {SEQ_TOTAL - PIPE_TOTAL}）",
    12.5, "middle", "bold", "#0f172a")

# ── 右侧定量结果面板 ──
rect(PANEL_X, PANEL_Y0, PANEL_W, PANEL_H, "#f8fafc", "#64748b", rx=10, sw=1.8)
txt(PANEL_X + PANEL_W / 2, PANEL_Y0 + 24, "定量结果", 14.5, "middle", "bold", "#0f172a")
rows = [
    ("顺序总时长（baseline）", f"{SEQ_TOTAL}"),
    ("逐层流水线总时长", f"{PIPE_TOTAL}"),
    ("省下 = (L-1)×t_x", f"{SEQ_TOTAL - PIPE_TOTAL} = 3×{T_X}"),
    ("被盖住 / 暴露", f"{HIDDEN_N} hidden / {EXPOSED_N} exposed"),
    (f"隐藏比 (L-1)/L，L={L}", f"{HIDDEN_N}/{L} = {FRAC_L4}"),
]
ry = PANEL_Y0 + 50
for label, val in rows:
    txt(PANEL_X + 16, ry, label, 11.5, "start", "normal", "#334155")
    txt(PANEL_X + PANEL_W - 16, ry, val, 12.5, "end", "bold", "#0f172a", mono=True)
    ry += 25

rect(PANEL_X + 12, ry - 6, PANEL_W - 24, 60, "#ecfdf5", "#22c55e", rx=8, sw=1.5)
txt(PANEL_X + PANEL_W / 2, ry + 16, "放大到 L=80（示教值，非源码常量）", 11, "middle", "italic", "#15803d")
txt(PANEL_X + PANEL_W / 2, ry + 38, f"隐藏比 = 79/80 = {FRAC_L80} ≈ 99%", 14, "middle", "bold", "#166534")

# ── 底部结论 ──
txt(W / 2, CONCL_Y,
    "跨节点 KV 传输被藏进后续层计算：L=4 时 75% 传输被盖住，放大到 L=80 约 99% 被隐藏——layerwise 连接器的核心收益",
    13.5, "middle", "bold", "#0f172a")

S.append('</svg>')
open("layerwise-pipeline-overlap.svg", "w", encoding="utf-8").write("\n".join(S))
print("wrote layerwise-pipeline-overlap.svg")
