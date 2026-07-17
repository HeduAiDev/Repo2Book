#!/usr/bin/env python3
"""fig-static-vs-range: before-after 模板 —— 同样循环 4 轮，static_range 追踪期
全展开成 0 个 scf.for / 8 个 arith.addi；range 发 1 个 scf.for 并挂
tt.num_stages=3 / tt.loop_unroll_factor=2 两个后端提示（性能旋钮）。
左右两栏同构对比，右栏挂后端提示标签高亮。全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "static_range vs range：追踪期就分道扬镳"
SUBTITLE = "同样循环 4 轮（make_ir 追踪期，任何 pass 之前，pin triton==3.2.0 实测）"

PANELS = [
    {
        "title": "tl.static_range(4)",
        "sub": "visit_For：真 Python range 跑 4 轮，constexpr(i) 塞进 lscope，逐轮复制循环体",
        "stats": [("scf.for", "0"), ("arith.addi", "8")],
        "note": "IR 里根本没有循环——编译期全展开",
        "hot": False,
    },
    {
        "title": "tl.range(0, 4, num_stages=3, loop_unroll_factor=2)",
        "sub": "visit_For：发 scf.for，取出提示 set_attr 挂到循环体上",
        "stats": [("scf.for", "1"), ("arith.addi", "2")],
        "note": "1 个 scf.for 保留循环，循环体只有一份",
        "hot": True,
        "attrs": [("tt.num_stages", "3"), ("tt.loop_unroll_factor", "2")],
    },
]

PANEL_W, PANEL_GAP, PAD, TOP = 420, 60, 40, 118
STAT_H, STAT_GAP = 56, 16

# 先算出每栏 note 落点的 y（右栏因多一个 attrs 框会比左栏更靠下），
# 取两栏最大值来定画布高度和页脚位置——避免固定公式在不对称面板下把页脚压穿。
note_ys = []
for panel in PANELS:
    sub_y = TOP + 34 + 20
    sub = panel["sub"]
    mid = len(sub) // 2
    cut = sub.rfind("，", 0, mid + 8)
    cut = mid if cut == -1 else cut + 1
    n_sub_lines = 2 if len(sub) > 26 else 1
    stats_top = sub_y + n_sub_lines * 16 + 14
    attrs_y = stats_top + len(panel["stats"]) * (STAT_H + STAT_GAP) - STAT_GAP + 26
    if panel["hot"] and panel.get("attrs"):
        note_ys.append(attrs_y + 40 + 26)
    else:
        note_ys.append(attrs_y + 4)
max_note_y = max(note_ys)

w = PAD * 2 + PANEL_W * 2 + PANEL_GAP
h = max_note_y + 56

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
          'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

for p, panel in enumerate(PANELS):
    px = PAD + p * (PANEL_W + PANEL_GAP)
    hot = panel["hot"]
    border = "#d97706" if hot else "#3b82f6"
    head_fill = "#fde68a" if hot else "#bfdbfe"

    # 标题条
    L.append(f'<rect x="{px}" y="{TOP}" width="{PANEL_W}" height="34" rx="6" '
              f'fill="{head_fill}" stroke="{border}" stroke-width="1.5"/>')
    L.append(f'<text x="{px+PANEL_W/2}" y="{TOP+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#1e293b">{esc(panel["title"])}</text>')

    sub_y = TOP + 34 + 20
    # 副标题（可能超宽，简单折成不超 2 行——按字符数粗切）
    sub = panel["sub"]
    mid = len(sub) // 2
    # 在中点附近找一个自然切分点（逗号/顿号/空格），否则直接切
    cut = sub.rfind("，", 0, mid + 8)
    if cut == -1:
        cut = mid
    else:
        cut += 1
    sub_lines = [sub[:cut], sub[cut:]] if len(sub) > 26 else [sub]
    for k, line in enumerate(sub_lines):
        L.append(f'<text x="{px+PANEL_W/2}" y="{sub_y+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" '
                  f'fill="#475569">{esc(line)}</text>')

    stats_top = sub_y + len(sub_lines) * 16 + 14
    for si, (label, value) in enumerate(panel["stats"]):
        sy = stats_top + si * (STAT_H + STAT_GAP)
        L.append(f'<rect x="{px}" y="{sy}" width="{PANEL_W}" height="{STAT_H}" rx="8" '
                  f'fill="{"#fef3c7" if hot else "#eff6ff"}" '
                  f'stroke="{border}" stroke-width="2"/>')
        L.append(f'<text x="{px+22}" y="{sy+STAT_H/2+5}" '
                  f'font-family="sans-serif" font-size="13.5" '
                  f'fill="#1e293b">{esc(label)}</text>')
        vfill = "#b45309" if hot else "#1d4ed8"
        L.append(f'<text x="{px+PANEL_W-22}" y="{sy+STAT_H/2+7}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="24" font-weight="bold" '
                  f'fill="{vfill}">{esc(value)}</text>')

    attrs_y = stats_top + len(panel["stats"]) * (STAT_H + STAT_GAP) - STAT_GAP + 26
    if hot and panel.get("attrs"):
        ax = px
        aw = PANEL_W
        ah = 40
        L.append(f'<rect x="{ax}" y="{attrs_y}" width="{aw}" height="{ah}" rx="6" '
                  'fill="#fff7ed" stroke="#c2410c" stroke-width="1.5" stroke-dasharray="4,3"/>')
        attr_text = "  ·  ".join(f"{k}={v}" for k, v in panel["attrs"])
        L.append(f'<text x="{ax+aw/2}" y="{attrs_y+18}" text-anchor="middle" '
                  'font-family="sans-serif" font-size="11" fill="#9a3412">'
                  '贴给后端 pass 的性能提示</text>')
        L.append(f'<text x="{ax+aw/2}" y="{attrs_y+35}" text-anchor="middle" '
                  'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="#c2410c">{esc(attr_text)}</text>')
        note_y = attrs_y + ah + 26
    else:
        note_y = attrs_y + 4

    L.append(f'<text x="{px+PANEL_W/2}" y="{note_y}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-style="italic" '
              f'fill="#334155">{esc(panel["note"])}</text>')

# 中间对照箭头
mid_y = TOP + 90
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{mid_y}" x2="{PAD+PANEL_W+PANEL_GAP-8}" y2="{mid_y}" '
          'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+PANEL_GAP/2}" y="{mid_y-10}" text-anchor="middle" '
          'font-family="sans-serif" font-size="10.5" fill="#64748b">对照</text>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          'fill="#64748b">code_generator.py visit_For 三分：边界是否 constexpr 决定能否在追踪期跑 Python for——num_stages/loop_unroll_factor 只在 range 分支才有意义。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-static-vs-range.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
