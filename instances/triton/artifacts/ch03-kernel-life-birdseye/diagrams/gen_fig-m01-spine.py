#!/usr/bin/env python3
"""fig-m01-spine: 一个 kernel 的一生 —— run() 查缓存 -> miss 则 compile() 一次
5 级 for 循环 -> 回 run() 发射；命中则短路整条 compile。
每站标注它归哪一层 / 后面哪一章放大（Part 记号与 outline-final.json 一致）。
全部数字取自 explainer figure_specs['fig-m01-spine'].numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


STATIONS = [
    dict(key="query", title="run() 查缓存", lines=["cache key", "jit.py:L583"], kind="entry"),
    dict(key="make_ir", title="make_ir", lines=["追踪期 TTIR", "56 行"], tag="Part IV · ch16", kind="stage"),
    dict(key="make_ttir", title="make_ttir", lines=[".ttir", "38 行"], tag="Part IV", kind="stage", loop=True),
    dict(key="make_ttgir", title="make_ttgir", lines=[".ttgir", "39 行"], tag="Part VII · ch32", kind="stage", loop=True),
    dict(key="make_llir", title="make_llir", lines=[".llir", "150 行"], tag="Part VII", kind="stage", loop=True),
    dict(key="make_ptx", title="make_ptx", lines=[".ptx", "377 行"], tag="Part VII · ch35", kind="stage", loop=True),
    dict(key="make_cubin", title="make_cubin", lines=[".cubin", "9488 字节"], tag="Part VIII · ch37", kind="stage", loop=True),
    dict(key="dispatch", title="run() 发射", lines=["launcher", "jit.py:L638"], tag="Part III · ch11", kind="exit"),
]

COLOR = {
    "entry": ("#dcfce7", "#16a34a", "#14532d"),
    "exit": ("#ffedd5", "#f97316", "#7c2d12"),
    "stage": ("#dbeafe", "#3b82f6", "#1e3a5f"),
}

PAD, BOX_W, BOX_H, GAP = 40, 190, 76, 26
TITLE_Y, SUB_Y, BYPASS_Y, ROW_Y = 28, 50, 62, 112
COMPILE_BOX_BOTTOM = ROW_Y + BOX_H + 12
TAG_Y = COMPILE_BOX_BOTTOM + 22
BRACE_Y = TAG_Y + 22
LOOP_LBL_Y = BRACE_Y + 20
FOOT_Y = LOOP_LBL_Y + 34

N = len(STATIONS)
X = [PAD + i * (BOX_W + GAP) for i in range(N)]
w = PAD * 2 + N * BOX_W + (N - 1) * GAP
h = FOOT_Y + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">一个 kernel 的一生：run() 查缓存 → compile() 五级降级 → 发射</text>',
     f'<text x="{PAD}" y="{SUB_Y}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">vector-add · add_kernel · cuda sm_90 · num_warps=4（headless 实测）</text>']

# compile() 外框：包住 make_ir .. make_cubin 六站
c_x0 = X[1] - 18
c_x1 = X[6] + BOX_W + 18
c_y0 = ROW_Y - 18
c_y1 = COMPILE_BOX_BOTTOM
L.append(f'<rect x="{c_x0}" y="{c_y0}" width="{c_x1-c_x0}" height="{c_y1-c_y0}" rx="14" '
          'fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>')
L.append(f'<text x="{c_x0+10}" y="{c_y0-6}" font-family="sans-serif" font-size="12" '
          'font-weight="bold" fill="#475569">compile() · compiler.py（miss 才真开工）</text>')

# 缓存命中短路弧：query 顶部 -> dispatch 顶部
qx = X[0] + BOX_W / 2
dxp = X[7] + BOX_W / 2
L.append(f'<path d="M {qx},{ROW_Y} L {qx},{BYPASS_Y-24} L {dxp},{BYPASS_Y-24} L {dxp},{ROW_Y}" '
          'fill="none" stroke="#b91c1c" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#b)"/>')
L.append(f'<text x="{(qx+dxp)/2}" y="{BYPASS_Y-30}" text-anchor="middle" font-family="sans-serif" '
          'font-size="11.5" fill="#b91c1c">缓存命中 → 直接发射（短路，整条 compile 被跳过）</text>')

# 主线箭头（依次相连）+ miss 标签
for i in range(N - 1):
    x1, x2 = X[i] + BOX_W, X[i + 1]
    ay = ROW_Y + BOX_H / 2
    L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" stroke="#334155" '
              'stroke-width="1.5" marker-end="url(#a)"/>')
    if i == 0:
        L.append(f'<text x="{(x1+x2)/2}" y="{ay-8}" text-anchor="middle" font-family="sans-serif" '
                  'font-size="11" fill="#b45309" font-weight="bold">miss</text>')

# 站点框
for i, st in enumerate(STATIONS):
    x = X[i]
    fill, stroke, text_fill = COLOR[st["kind"]]
    L.append(f'<rect x="{x}" y="{ROW_Y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{ROW_Y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="{text_fill}">{esc(st["title"])}</text>')
    for k, line in enumerate(st["lines"]):
        L.append(f'<text x="{x+BOX_W/2}" y="{ROW_Y+40+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="{text_fill}">{esc(line)}</text>')
    tag = st.get("tag")
    if tag:
        L.append(f'<text x="{x+BOX_W/2}" y="{TAG_Y}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" fill="#6d28d9" font-weight="bold">→ {esc(tag)}</text>')

# for 循环括弧：跨 make_ttir .. make_cubin（5 级）
loop_x0 = X[2] - 8
loop_x1 = X[6] + BOX_W + 8
L.append(f'<path d="M {loop_x0},{BRACE_Y-6} L {loop_x0},{BRACE_Y} L {loop_x1},{BRACE_Y} '
          f'L {loop_x1},{BRACE_Y-6}" fill="none" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<text x="{(loop_x0+loop_x1)/2}" y="{LOOP_LBL_Y}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" font-weight="bold" fill="#1d4ed8">'
          f'{esc("compile() 一次 5 级 for 循环 · compiler.py:L278 · stages 注册数=5")}</text>')

L.append(f'<text x="{PAD}" y="{FOOT_Y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("绿=入口(查缓存) 橙=出口(发射) 蓝=五级降级阶段 · 紫色箭头=后续放大的章")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m01-spine.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
