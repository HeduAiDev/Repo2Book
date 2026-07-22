#!/usr/bin/env python3
"""fig-m5-rewriteaddptr-flow: rewriteAddPtr 控制流总装车间（flow 模板，竖排
主链 + 判定分岔 + 侧支引到 gather 回退）。worked_example 取自 lit 夹具
legal_stride.mlir：sizes=[4,1] strides_in=[4,0] → strides_out=[4,1]。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "rewriteAddPtr 控制流全景（BlockPtrAnalysis.cpp:L1125-L1214）"
SUBTITLE = "示例取自 legal_stride.mlir：sizes=[4,1] strides=[4,0] → 零 stride 修复 → strides=[4,1]（CHECK 逐位一致）"

MAIN_W, MAIN_H, GAP, PAD, TOP = 460, 58, 34, 40, 118
DIAMOND_H = 76

L = [
    ('parseAddPtr 逆向出 BlockData',
     '镜像 ch11（不重推）→ sizes=[4,1] strides=[4,0]，MemAccType=StrucMemAcc',
     '#dbeafe', '#2563eb', '#1e3a8a', 'L846-L894'),
]
STEP2_LABEL = '取 resultShape'
STEP2_DETAIL = 'resultShape = [4, 1]（来自 result 类型）'
STEP2_LOC = 'L1152-L1158'
STEP3_LABEL = 'known 存未修改态'
STEP3_DETAIL = 'known[result] = data（原始 stride=[4,0] 保留，供后续指针算术继续用）'
STEP3_LOC = 'L1160'
STEP4_LABEL = '逆序 stride 修复'
STEP4_DETAIL = 'i=1: size==1 且 stride==0 → 替成 inferedSize=1 → strides=[4,1]\ni=0: size=4≠1 → 不替；inferedSize 累积到 4'
STEP4_LOC = 'L1170-L1177'
STEP5_LABEL = 'createCastOp + replaceOp'
STEP5_DETAIL = 'reinterpret_cast offset:[%arg13] sizes:[4,1] strides:[%c4,%c1]（对齐 CHECK 输出）'
STEP5_LOC = 'L1195,L1200'

steps = [
    ('parseAddPtr 逆向出 BlockData',
     '镜像 ch11（不重推）→ sizes=[4,1] strides=[4,0]，MemAccType=StrucMemAcc',
     'L846-L894'),
]

w = PAD * 2 + MAIN_W + 320
# 纵向位置累加
y = TOP
positions = {}

def box(label, detail, loc, y0, height=MAIN_H, fill='#e2e8f0', stroke='#334155', tf='#0f172a'):
    x = PAD
    out = []
    out.append(f'<rect x="{x}" y="{y0}" width="{MAIN_W}" height="{height}" rx="8" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    dl = detail.split('\n')
    out.append(f'<text x="{x+MAIN_W/2}" y="{y0+20}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="12.5" font-weight="bold" fill="{tf}">{esc(label)}</text>')
    n = len(dl)
    dy0 = y0 + (20 if n == 1 else 16) + (18 if n > 1 else 20)
    if n == 1:
        out.append(f'<text x="{x+MAIN_W/2}" y="{y0+40}" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="11" fill="{tf}">{esc(dl[0])}</text>')
    else:
        base = y0 + 38
        for k, line in enumerate(dl):
            out.append(f'<text x="{x+MAIN_W/2}" y="{base+k*15}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="10.5" fill="{tf}">{esc(line)}</text>')
    out.append(f'<text x="{x+MAIN_W-10}" y="{y0+height-8}" text-anchor="end" '
               f'font-family="sans-serif" font-size="9.5" fill="#94a3b8">{esc(loc)}</text>')
    return out

body = []

# 步骤 1
body += box('① parseAddPtr 逆向出 BlockData',
             '镜像 ch11（不重推）→ sizes=[4,1] strides=[4,0]，MemAccType=StrucMemAcc',
             'L846-L894', y)
y1_bottom = y + MAIN_H
body.append(f'<line x1="{PAD+MAIN_W/2}" y1="{y1_bottom}" x2="{PAD+MAIN_W/2}" y2="{y1_bottom+GAP-4}" '
            'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
y = y1_bottom + GAP

# 决策菱形
dcx, dcy = PAD + MAIN_W / 2, y + DIAMOND_H / 2
dw, dh = 300, DIAMOND_H
body.append(f'<polygon points="{dcx-dw/2},{dcy} {dcx},{dcy-dh/2} {dcx+dw/2},{dcy} {dcx},{dcy+dh/2}" '
            'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
body.append(f'<text x="{dcx}" y="{dcy-4}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11.5" font-weight="bold" fill="#78350f">② isUnstructured()？</text>')
body.append(f'<text x="{dcx}" y="{dcy+14}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="10.5" fill="#78350f">StrucMemAcc → 否</text>')
body.append(f'<text x="{dcx}" y="{dcy+30}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="9.5" fill="#92400e">L1135-L1142</text>')

# 是分支 → 侧支 gather 回退
side_x = PAD + MAIN_W + 90
side_w = 240
side_y = dcy - 40
body.append(f'<line x1="{dcx+dw/2}" y1="{dcy}" x2="{side_x}" y2="{dcy}" '
            'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#a)"/>')
body.append(f'<text x="{(dcx+dw/2+side_x)/2}" y="{dcy-8}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10" fill="#94a3b8">是</text>')
body.append(f'<rect x="{side_x}" y="{side_y}" width="{side_w}" height="80" rx="8" '
            'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>')
body.append(f'<text x="{side_x+side_w/2}" y="{side_y+22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11" font-weight="bold" '
            f'fill="#475569">Unstructured：gather 回退</text>')
body.append(f'<text x="{side_x+side_w/2}" y="{side_y+40}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10" fill="#64748b">rewriteAddPtrToUnstrucMemAcc</text>')
body.append(f'<text x="{side_x+side_w/2}" y="{side_y+58}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10" fill="#64748b">（见下图 fig-m6）</text>')

# 否分支继续向下
y2_bottom = y + DIAMOND_H
body.append(f'<line x1="{dcx}" y1="{y2_bottom}" x2="{dcx}" y2="{y2_bottom+GAP-4}" '
            'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
body.append(f'<text x="{dcx+16}" y="{y2_bottom+GAP/2}" font-family="sans-serif" '
            f'font-size="10" fill="#334155">否（结构化路径）</text>')
y = y2_bottom + GAP

body += box('③ 取 resultShape', 'resultShape = [4, 1]（来自 result 类型）', 'L1152-L1158', y)
y_b = y + MAIN_H
body.append(f'<line x1="{dcx}" y1="{y_b}" x2="{dcx}" y2="{y_b+GAP-4}" '
            'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
y = y_b + GAP

body += box('④ known 存未修改态', 'known[result]=data（原始 stride=[4,0] 保留，供后续指针算术继续用）', 'L1160', y)
y_b = y + MAIN_H
body.append(f'<line x1="{dcx}" y1="{y_b}" x2="{dcx}" y2="{y_b+GAP-4}" '
            'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
y = y_b + GAP

h5 = MAIN_H + 16
body += box('⑤ 逆序 stride 修复',
            'i=1: size==1 且 stride==0 → 替成 inferedSize=1 → strides=[4,1]\ni=0: size=4≠1 → 不替；inferedSize 累积到 4',
            'L1170-L1177', y, height=h5, fill='#fef9c3', stroke='#ca8a04', tf='#713f12')
y_b = y + h5
body.append(f'<line x1="{dcx}" y1="{y_b}" x2="{dcx}" y2="{y_b+GAP-4}" '
            'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
y = y_b + GAP

h6 = MAIN_H + 14
body += box('⑥ createCastOp + replaceOp',
            'reinterpret_cast offset:[%arg13] sizes:[4,1] strides:[%c4,%c1]（对齐 CHECK 输出）',
            'L1195,L1200', y, height=h6, fill='#fef3c7', stroke='#d97706', tf='#78350f')
y = y + h6

h = y + PAD + 26
w = side_x + side_w + PAD

out_svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
           '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
           'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
           f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
           f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
           f'fill="#64748b">{esc(SUBTITLE)}</text>']
out_svg += body
foot_y = h - 14
out_svg.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
               f'fill="#0f172a">与 ch11 rewriteAddptrOp（TritonToStructured 侧，重发射规范 addptr）不同侧——本图终点是 memref.reinterpret_cast</text>')
out_svg.append('</svg>')

out = Path(__file__).with_name('fig-m5-rewriteaddptr-flow.svg')
out.write_text('\n'.join(out_svg), encoding='utf-8')
print(f'wrote {out}')
