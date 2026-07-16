#!/usr/bin/env python3
"""tensor-flow 模板:从 grid 坐标到 block 张量的形状流。
链路:program_id -> pid*BLOCK -> +arange -> 本块坐标张量;每条边标 shape/具体值。
数字全部来自 dossier m1-grid-to-block worked_example(pid=1, BLOCK=8, num_programs=4)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "从 grid 坐标到 block 张量"
SUBTITLE = "pid×BLOCK + arange:kernel 首行『我是谁、我算哪一块』(pid=1, BLOCK=8, num_programs=4)"

# 主链节点:(标题行, 值行, IR 行)
NODES = [
    ("tl.program_id(0)", "pid = 1", "tt.get_program_id x : i32"),
    ("pid * BLOCK", "1 * 8 = 8", "arith.muli : i32 (标量)"),
    ("tl.arange(0, 8)", "[0,1,...,7]", "tt.make_range{0,8}:tensor<8xi32>"),
    ("本块坐标", "[8..15]", "tt.splat + arith.addi"),
]

BOX_W, BOX_H, HGAP, PAD, TOP = 220, 92, 70, 40, 108
w = PAD * 2 + len(NODES) * BOX_W + (len(NODES) - 1) * HGAP
h = TOP + BOX_H + 250

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1e3a8a"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#1e40af">{esc("从 grid 坐标到 block ")}'
     f'<tspan font-weight="normal">{esc("张量")}</tspan></text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="13" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

xs_ = [PAD + i * (BOX_W + HGAP) for i in range(len(NODES))]

for i, (title, val, ir) in enumerate(NODES):
    x = xs_[i]
    is_last = (i == len(NODES) - 1)
    fill = "#dbeafe" if is_last else "#eff6ff"
    stroke = "#1e40af" if is_last else "#3b82f6"
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if is_last else 1.5}"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#1e3a8a">{esc(title)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+50}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#b45309">{esc(val)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+72}" text-anchor="middle" font-family="monospace" '
              f'font-size="10.5" fill="#64748b">{esc(ir)}</text>')
    if i < len(NODES) - 1:
        ax1, ax2 = x + BOX_W, xs_[i+1]
        ay = TOP + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                  'stroke="#1e3a8a" stroke-width="2" marker-end="url(#a)"/>')

# arange 分支单独从下方汇入 "+arange" 节点(index 3),画一条竖线从 node[2]顶部弯到 node[3]
# 这里改成:node index2(arange)已经在主链里, node index1"pid*BLOCK"与 node index2"arange"
# 分别喂进 node3"本块坐标"。补一条从 node1 顶部绕到 node3 顶部的弧形辅助线,标注"+"
mid1x = xs_[1] + BOX_W / 2
mid3x = xs_[3] + BOX_W / 2
arc_y = TOP - 22
L.append(f'<path d="M {mid1x} {TOP} Q {(mid1x+mid3x)/2} {arc_y} {mid3x} {TOP}" '
          'fill="none" stroke="#0369a1" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#b)"/>')
L.append(f'<text x="{(mid1x+mid3x)/2}" y="{arc_y-6}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#0369a1">广播相加(逐元素 +)</text>')

# 覆盖区间图示(下方):4 个 program 各占 8 个坐标,无缝拼接 [0,32)
seg_top = TOP + BOX_H + 70
seg_w_total = w - PAD * 2
n_prog = 4
BLOCK = 8
seg_w = seg_w_total / (n_prog * BLOCK)
L.append(f'<text x="{PAD}" y="{seg_top-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">num_programs=4 个 program 无缝分掉 [0,32)——每程序覆盖 8 个坐标</text>')
prog_colors = ["#fecaca", "#fde68a", "#bbf7d0", "#bfdbfe"]
for p in range(n_prog):
    px = PAD + p * BLOCK * seg_w
    hl = (p == 1)
    L.append(f'<rect x="{px}" y="{seg_top}" width="{BLOCK*seg_w}" height="40" '
              f'fill="{prog_colors[p]}" stroke="{"#b45309" if hl else "#94a3b8"}" '
              f'stroke-width="{2.5 if hl else 1}"/>')
    label = f"program {p}" + ("(本例)" if hl else "")
    L.append(f'<text x="{px+BLOCK*seg_w/2}" y="{seg_top+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{px+BLOCK*seg_w/2}" y="{seg_top+33}" text-anchor="middle" '
              f'font-family="monospace" font-size="10" '
              f'fill="#334155">[{p*BLOCK},{(p+1)*BLOCK})</text>')
for p in range(n_prog + 1):
    tx = PAD + p * BLOCK * seg_w
    L.append(f'<line x1="{tx}" y1="{seg_top+40}" x2="{tx}" y2="{seg_top+48}" '
              'stroke="#64748b" stroke-width="1"/>')
    L.append(f'<text x="{tx}" y="{seg_top+62}" text-anchor="middle" font-family="monospace" '
              f'font-size="10.5" fill="#475569">{p*BLOCK}</text>')

foot_y = h - 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">结论:program_id 报编号、arange 发尺子,一乘一加锁定本 program 坐标区间'
          f'[8,16);区间紧邻不交,并集恰为 [0,32)——合并访存的整齐前提。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch07-grid-to-block.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
