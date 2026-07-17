#!/usr/bin/env python3
"""fig-ch32-five-stage-ladder: flow 模板。
add_stages 注册的五级降级阶梯(ttir/ttgir/llir/ptx/cubin),本章聚焦第一跳(ttir->ttgir)。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


# 节点:(标签行1, 标签行2, 是否本章聚焦)
NODES = [
    ("kernel 源码", "(编译入口)", False),
    ("ttir", "make_ttir", False),
    ("ttgir", "make_ttgir", True),
    ("llir", "make_llir", False),
    ("ptx", "make_ptx", False),
    ("cubin", "make_cubin", False),
]

# 每条弧的驱动函数标签 + 是否本章高亮
EDGES = [
    ("make_ttir", False, None),
    ("make_ttgir", True, "第一跳·本章"),
    ("make_llir", False, "ch33"),
    ("make_ptx", False, "ch34"),
    ("make_cubin", False, "ch35"),
]

BOX_W, BOX_H = 118, 60
GAP = 130
PAD = 40
TOP = 130
w = PAD * 2 + len(NODES) * BOX_W + (len(NODES) - 1) * GAP
h = TOP + BOX_H + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(
    f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#0f172a">{esc("五级降级阶梯:CUDABackend.add_stages 的登记(回收 f2)")}</text>'
)
L.append(
    f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
    f'fill="#475569">{esc("third_party/nvidia/backend/compiler.py:L384-L389 · 五行 stages[] 赋值,以同一 options/metadata 串联")}</text>'
)

# 节点中心坐标
centers = []
for i in range(len(NODES)):
    cx = PAD + BOX_W / 2 + i * (BOX_W + GAP)
    centers.append(cx)

cy = TOP + BOX_H / 2

# 边(先画,压在节点下层不必,箭头端点从框边缘算即可)
for i in range(len(NODES) - 1):
    x1 = centers[i] + BOX_W / 2
    x2 = centers[i + 1] - BOX_W / 2
    fn_label, hot, note = EDGES[i]
    color = "#2563eb" if hot else "#94a3b8"
    dash = "" if hot else ' stroke-dasharray="5,4"'
    width = 2.6 if hot else 1.6
    marker = "ah" if hot else "a"
    L.append(
        f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" stroke="{color}" '
        f'stroke-width="{width}"{dash} marker-end="url(#{marker})"/>'
    )
    midx = (x1 + x2) / 2
    L.append(
        f'<text x="{midx}" y="{cy-14}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11.5" font-weight="{"bold" if hot else "normal"}" '
        f'fill="{"#1d4ed8" if hot else "#64748b"}">{esc(fn_label)}</text>'
    )
    if note:
        for j, line in enumerate(note.split("\n")):
            L.append(
                f'<text x="{midx}" y="{cy+26+j*14}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="10.5" font-weight="{"bold" if hot else "normal"}" '
                f'fill="{"#1d4ed8" if hot else "#94a3b8"}">{esc(line)}</text>'
            )

# 节点框(画在边之上层)
for i, (label, sub, hot) in enumerate(NODES):
    cx = centers[i]
    x = cx - BOX_W / 2
    y = cy - BOX_H / 2
    is_source = i == 0
    fill = "#dbeafe" if hot else ("#f1f5f9" if is_source else "#e2e8f0")
    stroke = "#2563eb" if hot else "#64748b"
    sw = 2.6 if hot else 1.4
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    L.append(
        f'<text x="{cx}" y="{cy-4}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="15" font-weight="bold" fill="#0f172a">{esc(label)}</text>'
    )
    L.append(
        f'<text x="{cx}" y="{cy+16}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" fill="#475569">{esc(sub)}</text>'
    )

# 段落分类标注:前四段 = pass_manager;第五段 = shell 到 ptxas
band_y = cy + BOX_H / 2 + 62
x_left = centers[1] - BOX_W / 2
x_right4 = centers[4] + BOX_W / 2
L.append(f'<rect x="{x_left}" y="{band_y}" width="{x_right4-x_left}" height="22" rx="6" '
         f'fill="#ecfdf5" stroke="#22c55e"/>')
L.append(
    f'<text x="{(x_left+x_right4)/2}" y="{band_y+15}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" fill="#15803d">'
    f'{esc("前 4 段 = 各建一个 ir.pass_manager 跑 MLIR pass")}</text>'
)
x5_left = centers[4] + BOX_W / 2
x5_right = centers[5] + BOX_W / 2
L.append(f'<rect x="{x5_left}" y="{band_y}" width="{x5_right-x5_left}" height="22" rx="6" '
         f'fill="#fff7ed" stroke="#f97316"/>')
L.append(
    f'<text x="{(x5_left+x5_right)/2}" y="{band_y+15}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="10.5" fill="#c2410c">{esc("第 5 段 = shell 到 ptxas")}</text>'
)

# 底部:触发行原文
trig_y = band_y + 52
L.append(
    f'<rect x="{PAD}" y="{trig_y}" width="{w-2*PAD}" height="34" rx="8" '
    f'fill="#eff6ff" stroke="#2563eb" stroke-width="1.4"/>'
)
L.append(
    f'<text x="{w/2}" y="{trig_y+21}" text-anchor="middle" font-family="monospace" '
    f'font-size="12.5" fill="#1d4ed8">'
    f'{esc("第一跳触发行(make_ttgir 首个 pass): add_convert_to_ttgpuir(cuda:80, num_warps=4, threads_per_warp=32, num_ctas=1)")}</text>'
)

# 图例
leg_y = trig_y + 56
L.append(f'<line x1="{PAD}" y1="{leg_y}" x2="{PAD+28}" y2="{leg_y}" stroke="#2563eb" stroke-width="2.6" marker-end="url(#ah)"/>')
L.append(f'<text x="{PAD+36}" y="{leg_y+4}" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("本章:第一跳 ttir→ttgir")}</text>')
L.append(f'<line x1="{PAD+220}" y1="{leg_y}" x2="{PAD+248}" y2="{leg_y}" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+256}" y="{leg_y+4}" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("后续章:逐级走到 PTX 出口")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-five-stage-ladder.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
