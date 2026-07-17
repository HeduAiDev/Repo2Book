#!/usr/bin/env python3
"""f17-5-poison-lifecycle: 诱导变量三步走生命周期(单泳道时序):
create_poison 占位 -> dry-run 探 loop-carried/建 for_op -> get_induction_var 回填
replace_all_uses_with。仿 state-machine 单链写法,每步下方挂一个 IR/源码事实框。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

STEPS = [
    ("① 占位", "create_poison(iv_ir_type)", "set_value(target, poison)", "#fef9c3", "#a16207",
     "code_generator.py:L956-L958"),
    ("② 建 for_op", "dry-run 探 loop-carried", "create_for_op(iter_args=...)", "#dbeafe", "#1d4ed8",
     "%4 = ub.poison : i32(占位痕迹仍在追踪期 IR 里)"),
    ("③ 回填", "iv = for_op.get_induction_var()", "target.handle.replace_all_uses_with(iv)", "#dcfce7", "#15803d",
     "code_generator.py:L1016-L1023"),
]
BOX_W, BOX_H, HGAP, PAD, TOP = 300, 60, 90, 44, 96
DETAIL_H = 46
w = PAD * 2 + len(STEPS) * BOX_W + (len(STEPS) - 1) * HGAP
h = TOP + BOX_H + 30 + DETAIL_H + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#0f172a">{esc("诱导变量三步走:先用 poison 占位,for_op 建好后一次性替换")}</text>')

X = [PAD + i * (BOX_W + HGAP) for i in range(len(STEPS))]
for i, (badge, l1, l2, fill, stroke, detail) in enumerate(STEPS):
    x = X[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(badge)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#0f172a">{esc(l1)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+54}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#334155">{esc(l2)}</text>')
    dy = TOP + BOX_H + 26
    L.append(f'<rect x="{x}" y="{dy}" width="{BOX_W}" height="{DETAIL_H}" rx="6" '
              'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    words = detail
    # 自动换行:按字符宽度粗估,超过约 34 字符换行
    if len(words) > 30:
        mid = len(words) // 2
        # 找最近空格或标点断行
        cut = words.rfind('(', 0, mid + 8)
        if cut == -1:
            cut = mid
        line1, line2 = words[:cut].rstrip(), words[cut:].lstrip()
        L.append(f'<text x="{x+BOX_W/2}" y="{dy+18}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10" fill="#475569">{esc(line1)}</text>')
        L.append(f'<text x="{x+BOX_W/2}" y="{dy+34}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10" fill="#475569">{esc(line2)}</text>')
    else:
        L.append(f'<text x="{x+BOX_W/2}" y="{dy+DETAIL_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#475569">{esc(words)}</text>')
    if i < len(STEPS) - 1:
        y_arrow = TOP + BOX_H / 2
        L.append(f'<line x1="{x+BOX_W}" y1="{y_arrow}" x2="{X[i+1]}" y2="{y_arrow}" '
                  'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("poison 不是错误——是 SSA『先声明后定义』做不到时的合法占位;追踪期 IR 里 ub.poison 就是这枚占位符的痕迹")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-5-poison-lifecycle.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
