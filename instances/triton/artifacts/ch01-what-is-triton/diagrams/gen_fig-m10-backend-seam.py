#!/usr/bin/env python3
"""fig-m10-backend-seam: flow 模板——libtriton 靠 main.cc 的 FOR_EACH_P 宏链缝合
后端。本 pin 树内后端只有 2 个 (nvidia, amd)(setup.py:L562);图右侧是一次
gcc -E 构造性实验,人为喂假想元组去探"宏链手写上限":N=4 仍正常展开,N=5 时第 5
个名被当宏名撞成未定义宏而编译失败——4 是宏容量上限,不是本 pin 的后端数。树外
后端靠 TRITON_PLUGIN_DIRS 从侧门把名字追加进同一个元组。全部坐标基于固定中心+
半宽显式计算,保证左右分支不出画布。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def box(cx, cy, w, h, fill, stroke, lines, bold_first=True, fs=12, sw=1.5, dashed=False):
    x, y = cx - w / 2, cy - h / 2
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="8" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>']
    n = len(lines)
    y0 = cy - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        fw = 'font-weight="bold" ' if (bold_first and k == 0) else ''
        this_fs = fs + 0.5 if (bold_first and k == 0) else fs
        out.append(f'<text x="{cx:.1f}" y="{y0+k*15:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{this_fs}" fill="#0f172a" {fw}>{esc(line)}</text>')
    return out

def vline(x, y1, y2, label=None):
    out = [f'<line x1="{x}" y1="{y1:.1f}" x2="{x}" y2="{y2:.1f}" '
           'stroke="#475569" stroke-width="1.5" marker-end="url(#a)"/>']
    if label:
        out.append(f'<text x="{x+10}" y="{(y1+y2)/2-4:.1f}" font-family="sans-serif" '
                    f'font-size="11" fill="#64748b">{esc(label)}</text>')
    return out

W = 1160
TRUNK_X = W / 2          # 580,主干居中
TRUNK_W = 540            # 主干框半宽 270,跨 310..850,余量足够

LEFT_CX = 260            # 左分支中心,半宽 220 → 跨 40..480
RIGHT_CX = 900           # 右分支中心,半宽 220 → 跨 680..1120
BRANCH_W = 440

NODE_H1, NODE_H2 = 58, 46
TOP = 60
VGAP = 50

y0 = TOP + NODE_H1 / 2
y1 = y0 + NODE_H1 / 2 + VGAP + NODE_H2 / 2
y_note = y1 + NODE_H2 / 2 + VGAP + 8       # 分岔说明两行文字的第一行 y
y_split = y_note + 46                       # 分岔点(说明文字之下)

BRANCH_H_LEFT = 74
BRANCH_H_RIGHT = 92
y_branch = y_split + 60

BOTTOM_W = 660
BOTTOM_H = 78
y_bottom = y_branch + max(BRANCH_H_LEFT, BRANCH_H_RIGHT) / 2 + VGAP + 30 + BOTTOM_H / 2

h = int(y_bottom + BOTTOM_H / 2 + 40)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
          '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#047857"/></marker>'
          '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
          '<marker id="o" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#a16207"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{h}" fill="white"/>')
L.append(f'<text x="30" y="28" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#1e293b">libtriton 的后端接缝:FOR_EACH_P 宏链手写上限 4 元(容量上限,非后端数)</text>')

# 节点0: TRITON_BACKENDS_TUPLE —— 本 pin 的真实值:2 个树内后端
L += box(TRUNK_X, y0, TRUNK_W, NODE_H1, "#e2e8f0", "#475569",
          ["本 pin 树内后端 = 2 个:TRITON_BACKENDS_TUPLE (nvidia, amd)",
           "setup.py:L562 / CMakeLists:L245,L253(带括号)"], fs=11.5)

# 节点1: REMOVE_PARENS -> FOR_EACH_P
L += vline(TRUNK_X, y0+NODE_H1/2, y1-NODE_H2/2)
L += box(TRUNK_X, y1, TRUNK_W, NODE_H2, "#dbeafe", "#1d4ed8",
          ["REMOVE_PARENS 脱括号 → FOR_EACH_P(...)  [main.cc:L24-L44]"], fs=11.5)

# 分岔说明(两行,居中于主干正下方,不与任何框重叠)
L += vline(TRUNK_X, y1+NODE_H2/2, y_note-18)
L.append(f'<text x="{TRUNK_X}" y="{y_note}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#0f172a" font-weight="bold">FOR_EACH_ARG_N 数参数个数 N,'
          f'CONCATENATE(FOR_EACH_, N) 拼宏名</text>')
L.append(f'<text x="{TRUNK_X}" y="{y_note+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">容量上限:只手写 FOR_EACH_1..FOR_EACH_4(main.cc:L7-L10)'
          f'——下面是一次 gcc -E 构造性实验,喂假想元组探这个上限</text>')

# 两条分支 = 同一次构造实验的 N=4 / N=5 两种结果(不是真实 vs 假想的对立)
L.append(f'<line x1="{TRUNK_X}" y1="{y_split}" x2="{LEFT_CX}" y2="{y_branch-BRANCH_H_LEFT/2}" '
          'stroke="#047857" stroke-width="1.5" marker-end="url(#g)"/>')
L.append(f'<line x1="{TRUNK_X}" y1="{y_split}" x2="{RIGHT_CX}" y2="{y_branch-BRANCH_H_RIGHT/2}" '
          'stroke="#b91c1c" stroke-width="1.5" marker-end="url(#r)"/>')
mid_l_x, mid_l_y = (TRUNK_X+LEFT_CX)/2, (y_split+y_branch-BRANCH_H_LEFT/2)/2
mid_r_x, mid_r_y = (TRUNK_X+RIGHT_CX)/2, (y_split+y_branch-BRANCH_H_RIGHT/2)/2
L.append(f'<text x="{mid_l_x:.1f}" y="{mid_l_y-8:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#047857" font-weight="bold">构造实验 · N=4</text>')
L.append(f'<text x="{mid_r_x:.1f}" y="{mid_r_y-8:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#b91c1c" font-weight="bold">同一实验 · N=5</text>')

L += box(LEFT_CX, y_branch, BRANCH_W, BRANCH_H_LEFT, "#ecfdf5", "#047857",
          ["N=4 → FOR_EACH_4 正常展开", "假想元组 (nvidia,amd,x,y),x/y 是占位名非树内后端"], fs=11)
L.append(f'<text x="{LEFT_CX:.1f}" y="{y_branch+BRANCH_H_LEFT/2+16:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#94a3b8">gcc -E 宏展开实测(Triton v3.2.0)</text>')

L += box(RIGHT_CX, y_branch, BRANCH_W, BRANCH_H_RIGHT, "#fee2e2", "#b91c1c",
          ["N=5:第 5 个名 fifth 被当成宏名", "CONCATENATE(FOR_EACH_, fifth)",
           "→ FOR_EACH_fifth 未定义宏 → 编译失败"], fs=10.5)
L.append(f'<text x="{RIGHT_CX:.1f}" y="{y_branch+BRANCH_H_RIGHT/2+16:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#94a3b8">gcc -E 宏展开实测(Triton v3.2.0)</text>')

# 底部:树外后端侧门(一条虚线从底框上到分岔前,表示"名字追加进同一个真实元组")
side_top = y_bottom - BOTTOM_H/2
side_head = y_branch + max(BRANCH_H_LEFT, BRANCH_H_RIGHT)/2 + 8
L.append(f'<line x1="{TRUNK_X}" y1="{side_top}" x2="{TRUNK_X}" y2="{side_head}" '
          'stroke="#a16207" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#o)"/>')
L.append(f'<text x="{TRUNK_X+12}" y="{(side_top+side_head)/2+4:.1f}" font-family="sans-serif" '
          f'font-size="10" fill="#a16207">树外后端也追加进同一元组(如姊妹篇 Triton-Ascend)</text>')
L += box(TRUNK_X, y_bottom, BOTTOM_W, BOTTOM_H, "#fef9c3", "#a16207",
          ["树外后端侧门:TRITON_PLUGIN_DIRS(is_external)",
           "读 <dir>/backend/name.conf → 追加进同一个后端元组"], fs=12, dashed=True)
L.append(f'<text x="{TRUNK_X:.1f}" y="{y_bottom+BOTTOM_H/2+18:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#94a3b8">CMakeLists.txt:L161-L175, L248</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m10-backend-seam.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={W}x{h}")
