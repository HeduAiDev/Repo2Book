#!/usr/bin/env python3
"""flow(分派树)模板:SharedEncodingAttr::get 按 dotOpEnc parent 类型
分派到各目标 mma 的反推公式,每条从 mma tile / 硬件常量算出 (vec,perPhase,maxPhase)。
语义色:普通反推分支(蓝)/ 不 swizzle 的守卫分支(灰,虚线框)/ 转到别的 builder 的
逃逸分支(橙,虚线框)。三色差异用图例说明。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROOT_LABEL = "SharedEncodingAttr::get(dotOpEnc, shape, order, ...)"
ROOT_SUB = "按 dotOpEnc.getParent() 类型分派(TritonGPUAttrDefs.td:L274-L409)"

# (edge_label, title, detail_lines, result, kind)
# kind: normal(蓝,反推公式) / guard(灰虚线,不 swizzle) / escape(橙虚线,转别的 builder)
BRANCHES = [
    ("parent = AMD MFMA",
     "AMD MFMA 分支",
     ["numBanks=32 / bankBitWidth=32 / SIMDWidth=16",
      "输入:fp16, K=32, kWidth=4"],
     ["(vec, perPhase, maxPhase)", "= (4, 2, 8)"],
     "normal", "TritonGPUAttrDefs.td:L288-L299"),
    ("parent = NVIDIA Ampere",
     "NVIDIA Ampere 分支",
     ["matShape = {8, 8, 4·kWidth}(mma.sync 一条指令的 M,N,K)",
      "输入:fp16, inner=32, kWidth=2, opA"],
     ["(vec, perPhase, maxPhase)", "= (8, 2, 4)"],
     "normal", "TritonGPUAttrDefs.td:L366-L380"),
    ("order[0] != kDim",
     "K 维不在最内圈(任意 parent)",
     ["天然不同 bank,无需摊开",
      "'accesses go in different banks even without swizzling'"],
     ["(vec, perPhase, maxPhase)", "= (1, 1, 1) 不 swizzle"],
     "guard", "TritonGPUAttrDefs.td:L306-L310"),
    ("parent = NVIDIA Hopper",
     "NVIDIA Hopper(MMAv3)分支",
     ["此 builder 内 llvm_unreachable",
      "改走 by-eltTy builder(另一套三档 + hasLeadingOffset)"],
     ["→ by-eltTy builder", "(非本分支产出)"],
     "escape", "TritonGPUAttrDefs.td:L401-L405"),
]

KIND_STYLE = {
    "normal": dict(fill="#dbeafe", stroke="#3b82f6", dash="none", text="#1e3a5f",
                   badge_fill="#3b82f6", badge_text="white"),
    "guard":  dict(fill="#f1f5f9", stroke="#64748b", dash="6,4", text="#334155",
                   badge_fill="#e2e8f0", badge_text="#334155"),
    "escape": dict(fill="#ffedd5", stroke="#f97316", dash="6,4", text="#7c2d12",
                   badge_fill="#f97316", badge_text="white"),
}

ROOT_W, ROOT_H = 560, 60
BOX_W, BOX_H = 300, 130
HGAP = 40
TOP, PAD = 150, 40
EDGE_ZONE = 46  # 根到分支之间留给边标签的高度

n = len(BRANCHES)
grid_w = n * BOX_W + (n - 1) * HGAP
w = PAD * 2 + max(grid_w, ROOT_W)
h = TOP + EDGE_ZONE + BOX_H + 70 + PAD

root_x = PAD + (w - PAD * 2 - ROOT_W) / 2
root_y = 46
branch_x0 = PAD + (w - PAD * 2 - grid_w) / 2
branch_y = TOP + EDGE_ZONE

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{26}" font-family="sans-serif" font-size="15" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("swizzle 参数由目标 mma 指令反推,不是自由填的")}</text>']

# 根节点
L.append(f'<rect x="{root_x}" y="{root_y}" width="{ROOT_W}" height="{ROOT_H}" rx="10" '
          f'fill="#0f172a" stroke="#334155"/>')
L.append(f'<text x="{root_x + ROOT_W/2}" y="{root_y + 26}" text-anchor="middle" '
          f'font-family="monospace" font-size="13" font-weight="bold" '
          f'fill="white">{esc(ROOT_LABEL)}</text>')
L.append(f'<text x="{root_x + ROOT_W/2}" y="{root_y + 46}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" '
          f'fill="#94a3b8">{esc(ROOT_SUB)}</text>')

root_bottom_y = root_y + ROOT_H
root_cx = root_x + ROOT_W / 2

for i, (edge_lbl, title, detail, result, kind, prov) in enumerate(BRANCHES):
    bx = branch_x0 + i * (BOX_W + HGAP)
    by = branch_y
    st = KIND_STYLE[kind]
    bcx = bx + BOX_W / 2
    # 边:根 -> 分支
    dash_attr = '' if st["dash"] == "none" else f' stroke-dasharray="{st["dash"]}"'
    L.append(f'<path d="M {root_cx} {root_bottom_y} L {root_cx} {root_bottom_y+16} '
              f'L {bcx} {by-16} L {bcx} {by}" fill="none" stroke="#334155" '
              f'stroke-width="1.5" marker-end="url(#a)"/>')
    ey = (root_bottom_y + by) / 2
    L.append(f'<rect x="{bcx-95}" y="{ey-11}" width="190" height="20" rx="4" '
              f'fill="white" stroke="#cbd5e1"/>')
    L.append(f'<text x="{bcx}" y="{ey+4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#475569">{esc(edge_lbl)}</text>')
    # 分支框
    L.append(f'<rect x="{bx}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="2"{dash_attr}/>')
    L.append(f'<text x="{bcx}" y="{by+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{st["text"]}">{esc(title)}</text>')
    dy = by + 44
    for line in detail:
        L.append(f'<text x="{bcx}" y="{dy}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="{st["text"]}">{esc(line)}</text>')
        dy += 15
    # 结果徽章(两行,避免长文本溢出)
    badge_y = by + BOX_H - 46
    L.append(f'<rect x="{bx+10}" y="{badge_y}" width="{BOX_W-20}" height="36" rx="6" '
              f'fill="{st["badge_fill"]}"/>')
    for k, line in enumerate(result):
        L.append(f'<text x="{bcx}" y="{badge_y+15+k*16}" text-anchor="middle" '
                  f'font-family="monospace" font-size="10.5" font-weight="bold" '
                  f'fill="{st["badge_text"]}">{esc(line)}</text>')
    # 出处
    L.append(f'<text x="{bcx}" y="{by+BOX_H+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" fill="#94a3b8">{esc(prov)}</text>')

# 图例
ly = h - 22
LEGEND = [("normal", "反推公式(读 mma tile / 硬件常量算参数)"),
          ("guard", "守卫分支(K 不在最内圈→不 swizzle)"),
          ("escape", "逃逸分支(转到另一 builder)")]
lx = PAD
for kind, label in LEGEND:
    st = KIND_STYLE[kind]
    L.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" rx="3" '
              f'fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="2"/>')
    L.append(f'<text x="{lx+20}" y="{ly}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 20 + 8 + len(label) * 6.6 + 30

L.append('</svg>')
out = Path(__file__).with_name("fig-mma-dispatch-derive.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
