#!/usr/bin/env python3
"""ch30 §30.4 netloader.load_model 控制流：能拉就拉、拉不动就退。
照 vllm_ascend/model_loader/netloader/netloader.py load_model 源码分支绘制：
① source 三重有效性判断 → 无效则回退默认；② 有效则 initialize_model 建空壳 +
elastic_load 弹性拉权重；③ elastic_load 返回 None（拉取失败）同样回退默认。
两条回退路径都画出来。风格对齐同章 fig30-1。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# palette
P_FILL, P_STROKE, P_TC = "#f3e8ff", "#7c3aed", "#5b21b6"   # 处理框（紫）
D_FILL, D_STROKE, D_TC = "#fef3c7", "#d97706", "#92400e"   # 判定菱形（琥珀）
R_FILL, R_STROKE, R_TC = "#f1f5f9", "#94a3b8", "#475569"   # 回退框（灰）
G_FILL, G_STROKE, G_TC = "#dcfce7", "#16a34a", "#166534"   # 成功框（绿）

W, H = 1020, 800
cx = 290           # 主干中轴
bw = 360           # 主干框宽

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# title
L.append(f'<text x="{W/2}" y="44" text-anchor="middle" font-family="sans-serif" '
         f'font-size="28" font-weight="bold" fill="#1e293b">'
         f'{esc("netloader.load_model：能拉就拉，拉不动就优雅回退")}</text>')
L.append(f'<text x="{W/2}" y="74" text-anchor="middle" font-family="sans-serif" '
         f'font-size="16" fill="#64748b">'
         f'{esc("弹性网络加载是快路径；任何一条岔路都退回 vLLM 的 DefaultModelLoader")}</text>')


def rbox(x, y, w, h, fill, stroke, tc, lines, fs=16, mono=False, bold=True):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    ff = "monospace" if mono else "sans-serif"
    fw = "bold" if bold else "normal"
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * (fs + 4) / 2 + fs / 2 - 2
    for i, ln in enumerate(lines):
        L.append(f'<text x="{x+w/2}" y="{y0+i*(fs+4)}" text-anchor="middle" '
                 f'font-family="{ff}" font-size="{fs}" font-weight="{fw}" '
                 f'fill="{tc}">{esc(ln)}</text>')


def diamond(dcx, dcy, hw, hh, lines):
    pts = f'{dcx},{dcy-hh} {dcx+hw},{dcy} {dcx},{dcy+hh} {dcx-hw},{dcy}'
    L.append(f'<polygon points="{pts}" fill="{D_FILL}" stroke="{D_STROKE}" stroke-width="2"/>')
    n = len(lines)
    y0 = dcy - (n - 1) * 20 / 2 + 5
    for i, (ln, fs) in enumerate(lines):
        L.append(f'<text x="{dcx}" y="{y0+i*20}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="bold" '
                 f'fill="{D_TC}">{esc(ln)}</text>')


def arrow(x1, y1, x2, y2, marker="ap", stroke="#7c3aed"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
             f'stroke-width="2" marker-end="url(#{marker})"/>')


def alabel(x, y, t, fill="#7c3aed"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="{fill}">{esc(t)}</text>')


# --- 主干节点 ---
# A 入口
rbox(cx - bw / 2, 100, bw, 52, P_FILL, P_STROKE, P_TC,
     ["load_model(vllm_config, model_config)"], fs=17, mono=True)
# D1 source 三重有效性判断
diamond(cx, 250, 185, 66,
        [("source 有效性判断", 15),
         ("① 非 None  ② 是 list", 13),
         ("③ 本 rank 在 source 内", 13)])
# B initialize_model
rbox(cx - bw / 2, 356, bw, 52, P_FILL, P_STROKE, P_TC,
     ["initialize_model：先建一个空骨架"], fs=16)
# C elastic_load
rbox(cx - bw / 2, 440, bw, 52, P_FILL, P_STROKE, P_TC,
     ["elastic_load：从对等实例弹性拉权重填入"], fs=15)
# D2 拉取是否失败
diamond(cx, 576, 185, 60,
        [("elastic_load", 15), ("返回 None？", 15)])
# E 成功
rbox(cx - bw / 2, 692, bw, 56, G_FILL, G_STROKE, G_TC,
     ["拉到权重 → 返回填好的模型"], fs=16)

# --- 回退列 ---
fx, fw2 = 560, 440   # 560..1000
rbox(fx, 217, fw2, 66, R_FILL, R_STROKE, R_TC,
     ["回退①：revert_to_default", "→ DefaultModelLoader（load_format=auto）"], fs=15, bold=True)
rbox(fx, 543, fw2, 66, R_FILL, R_STROKE, R_TC,
     ["回退②：清 NPU cache 后 revert_to_default", "→ 同样走 DefaultModelLoader"], fs=15, bold=True)

# --- 箭头 ---
arrow(cx, 152, cx, 184)                      # A -> D1
arrow(cx, 316, cx, 356)                      # D1 -> B (有效)
alabel(cx + 46, 340, "有效")
arrow(cx + 185, 250, fx - 6, 250)            # D1 -> 回退① (无效)
alabel((cx + 185 + fx) / 2, 240, "无效（任一成立）")
arrow(cx, 408, cx, 440)                      # B -> C
arrow(cx, 492, cx, 516)                      # C -> D2
arrow(cx, 636, cx, 692)                      # D2 -> E (成功)
alabel(cx + 66, 668, "否（成功）")
arrow(cx + 185, 576, fx - 6, 576)            # D2 -> 回退② (None)
alabel((cx + 185 + fx) / 2, 566, "是（拉取失败）")

L.append('</svg>')
svg = '\n'.join(L)
with open('netloader-flow.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print("wrote netloader-flow.svg", W, H)
