#!/usr/bin/env python3
"""ch39-f3-session-fsm: proton 会话生命线(state-machine 改)。
claim: start 翻开全局 profiling 开关并挂钩子,finalize 关开关、写盘、摘钩子。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CHAIN = ["未启动\n(profiling_on=False)", "已启动\n(profiling_on=True)", "已终结\n(FINALIZED)"]
CHAIN_LBL = [
    ["start(hook='triton')", "set_profiling_on → register_triton_hook", "→ libproton.start"],
    ["finalize(None)", "finalize_all('hatchet') 写盘", "→ unregister_triton_hook"],
]
SIDE = ("已启动\n(profiling_on=True)", "activate/deactivate\n中间态", "activate()/deactivate()\n切换本 session 记录开关", "返回已启动状态")

BOX_W, BOX_H, HGAP, PAD, TOP, SIDE_DY = 250, 66, 150, 50, 130, 150
w = PAD * 2 + len(CHAIN) * BOX_W + (len(CHAIN) - 1) * HGAP
h = TOP + BOX_H + SIDE_DY + BOX_H + PAD + 40
X = {s: (PAD + i * (BOX_W + HGAP), TOP) for i, s in enumerate(CHAIN)}
anchor, side_name, down_lbl, up_lbl = SIDE
X[side_name] = (X[anchor][0], TOP + BOX_H + SIDE_DY)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

# title
L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" font-weight="bold" '
         f'fill="#1e40af">proton 会话生命线</text>')
L.append(f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12" fill="#64748b">'
         f'状态迁移: start → activate/deactivate → finalize (third_party/proton/proton/profile.py:L33-L140)</text>')

for name, (x, y) in X.items():
    lines = name.split("\n")
    is_side = (name == side_name)
    fill, stroke, tc = ("#fef3c7", "#b45309", "#78350f") if is_side else ("#e0f2fe", "#0369a1", "#0c4a6e")
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="22" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    y0 = y + BOX_H/2 - (n-1)*9 + 5
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*18}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="13" font-weight="bold" '
                 f'fill="{tc}">{esc(line)}</text>')

for i in range(len(CHAIN) - 1):  # 主线转移
    (x1, y1), (x2, y2) = X[CHAIN[i]], X[CHAIN[i + 1]]
    ay = y1 + BOX_H / 2
    L.append(f'<line x1="{x1+BOX_W}" y1="{ay}" x2="{x2}" y2="{ay}" '
             'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
    lbl_lines = CHAIN_LBL[i]
    ly0 = ay - 14 - (len(lbl_lines)-1)*13
    for k, line in enumerate(lbl_lines):
        wt = 'font-weight="bold" ' if k == 0 else ''
        col = "#1d4ed8" if k == 0 else "#475569"
        L.append(f'<text x="{(x1+BOX_W+x2)/2}" y="{ly0+k*13}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{11 if k else 12}" {wt}fill="{col}">{esc(line)}</text>')

# side branch (双向竖边)
(ax, ay0), (sx, sy) = X[anchor], X[side_name]
xl, xr = ax + BOX_W * 0.28, ax + BOX_W * 0.72
L.append(f'<line x1="{xl}" y1="{ay0+BOX_H}" x2="{xl}" y2="{sy}" '
         'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{xr}" y1="{sy}" x2="{xr}" y2="{ay0+BOX_H}" '
         'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
my = (ay0 + BOX_H + sy) / 2
L.append(f'<text x="{xl-10}" y="{my-4}" text-anchor="end" font-family="sans-serif" '
         f'font-size="11" fill="#334155">{esc(down_lbl)}</text>')
L.append(f'<text x="{xr+10}" y="{my-4}" font-family="sans-serif" '
         f'font-size="11" fill="#334155">{esc(up_lbl)}</text>')

# annotation: finalize default output format
fx, fy = X[CHAIN[2]]
L.append(f'<text x="{fx+BOX_W/2}" y="{fy+BOX_H+28}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#64748b">finalize 默认输出格式 = hatchet</text>')
L.append(f'<text x="{fx+BOX_W/2}" y="{fy+BOX_H+45}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#94a3b8">(third_party/proton/proton/profile.py:L120)</text>')

foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'落盘的 hatchet json 正是 roofline viewer 的输入。</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch39-f3-session-fsm.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
