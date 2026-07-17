#!/usr/bin/env python3
"""flow 模板:storeDistributedToShared 不手写换相,而是调
emitTransferBetweenRegistersAndShared——regLayout->invertAndCompose(sharedLayout)
把两张布局复合成一张寄存器->物理偏移表,再逐向量 applyLinearLayout 出偏移 gep+store。
主链横排,两个侧输入(order 来源/向量宽来源)从上方喂入对应节点。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CHAIN = [
    ("regLayout", "寄存器 → 逻辑元素"),
    ("invertAndCompose\n(sharedLayout)", "复合两张布局"),
    ("寄存器 → 物理偏移\n(复合 LinearLayout)", "一张表,无显式 urem/xor"),
    ("applyLinearLayout\n(i·vec, lane, warp)·stride", "逐向量取偏移"),
    ("gep(global_smem,\noffset) → store", "落盘,align=vecElems·bitwidth/8"),
]
CHAIN_LBL = ["", "", "", ""]

BOX_W, BOX_H, HGAP = 210, 74, 46
PAD, TOP = 40, 210
SIDE_BOX_H = 50
SIDE_DY = 70

n = len(CHAIN)
w = PAD * 2 + n * BOX_W + (n - 1) * HGAP
h = TOP + BOX_H + 150

X = [(PAD + i * (BOX_W + HGAP), TOP) for i in range(n)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#0f172a">{esc("storeDistributedToShared 不手写换相:两张布局复合成一张表")}</text>',
     f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("swizzle 已烘进 sharedLayout(见前一图);这里只做复合 + 逐向量 gep + store,读者看不到显式 urem/xor")}</text>']

# 主链节点
for i, (name, sub) in enumerate(CHAIN):
    x, y = X[i]
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.5"/>')
    lines = name.split("\n")
    ny0 = y + BOX_H/2 - (len(lines)-1)*8 - (6 if sub else 0)
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{ny0+k*16}" text-anchor="middle" '
                  f'font-family="monospace" font-size="12" font-weight="bold" '
                  f'fill="#0c4a6e">{esc(ln)}</text>')
    if sub:
        L.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H-10}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="#0369a1">{esc(sub)}</text>')

# 主链箭头
for i in range(n - 1):
    x1, y1 = X[i]
    x2, y2 = X[i+1]
    ay = y1 + BOX_H/2
    L.append(f'<line x1="{x1+BOX_W}" y1="{ay}" x2="{x2}" y2="{ay}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

# 侧输入 1:order 来源 -> 喂进 invertAndCompose(节点1)
side1_x, side1_y = X[1][0], TOP - SIDE_DY - SIDE_BOX_H
L.append(f'<rect x="{side1_x}" y="{side1_y}" width="{BOX_W}" height="{SIDE_BOX_H}" rx="8" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{side1_x+BOX_W/2}" y="{side1_y+20}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" font-weight="bold" fill="#92400e">'
          f'{esc("换相来源")}</text>')
L.append(f'<text x="{side1_x+BOX_W/2}" y="{side1_y+38}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#92400e">'
          f'{esc("SharedEncoding.getOrder()")}</text>')
L.append(f'<line x1="{side1_x+BOX_W/2}" y1="{side1_y+SIDE_BOX_H}" x2="{X[1][0]+BOX_W/2}" '
          f'y2="{TOP}" stroke="#0369a1" stroke-width="1.5" marker-end="url(#b)"/>')

# 侧输入 2:向量宽来源 -> 喂进 applyLinearLayout(节点3)
side2_x, side2_y = X[3][0], TOP - SIDE_DY - SIDE_BOX_H
L.append(f'<rect x="{side2_x}" y="{side2_y}" width="{BOX_W}" height="{SIDE_BOX_H}" rx="8" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{side2_x+BOX_W/2}" y="{side2_y+20}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" font-weight="bold" fill="#92400e">'
          f'{esc("向量宽 = 连续段长")}</text>')
L.append(f'<text x="{side2_x+BOX_W/2}" y="{side2_y+38}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#92400e">'
          f'{esc("getNumConsecutiveInOut")}</text>')
L.append(f'<line x1="{side2_x+BOX_W/2}" y1="{side2_y+SIDE_BOX_H}" x2="{X[3][0]+BOX_W/2}" '
          f'y2="{TOP}" stroke="#0369a1" stroke-width="1.5" marker-end="url(#b)"/>')

# 底部说明:store 对齐 + loadSharedToDistributed 同引擎
foot_y = TOP + BOX_H + 40
foot_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{foot_w}" height="66" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y+26}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("store 对齐 = vecElems · bitwidth/8 字节(Utility.cpp:L421-L422 setAlignment)")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+46}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("loadSharedToDistributed 是同一引擎、回调换成 load —— 换相逻辑不必再写一遍")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch34-m3-transfer-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
