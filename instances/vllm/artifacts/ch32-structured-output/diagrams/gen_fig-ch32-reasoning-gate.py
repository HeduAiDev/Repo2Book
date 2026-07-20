#!/usr/bin/env python3
"""fig-ch32-reasoning-gate: 推理段内两道门都关着(不填掩码、不推进 FSM),
侦测到 </think> 的那一步只置标志位,约束从下一步才生效。
template: state-machine(主链 4 步 + 覆盖开关侧支)"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1400, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="p" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("推理段两道门都关着,约束从侦测到 </think> 的下一步才生效")}</text>')

PAD = 40
TOP = 100
BOX_W = 290
BOX_H = 100
GAP = 46

STEPS = [
    ("step 1", "吐出 50", "reasoning_ended(步前)=None", "should_fill=False / should_advance=False", "掩码:整行 -1(96 全允许)", "#e2e8f0", "#64748b"),
    ("step 2", "吐出 51", "reasoning_ended(步前)=False", "should_fill=False / should_advance=False", "掩码:整行 -1(96 全允许)", "#e2e8f0", "#64748b"),
    ("step 3", "吐出 99(</think>)", "reasoning_ended(步前)=False -> 步后 True", "should_fill=False / should_advance=False", "翻转步:置标志位,本步仍不推进", "#fef3c7", "#b45309"),
    ("step 4", "吐出 5", "reasoning_ended(步前)=True", "should_fill=True / should_advance=True", "掩码:只允许 5, 7;FSM 位置 -> 1", "#dcfce7", "#16a34a"),
]

sx = []
for i, (name, tok, re_, gate, mask, fill, stroke) in enumerate(STEPS):
    x = PAD + i * (BOX_W + GAP)
    sx.append(x)
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{stroke}">{esc(f"{name}:{tok}")}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#334155">{esc(re_)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+58}" text-anchor="middle" font-family="monospace" '
              f'font-size="9.5" fill="#334155">{esc(gate)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+80}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" font-weight="bold" fill="{stroke}">{esc(mask)}</text>')
    if i < len(STEPS) - 1:
        L.append(f'<line x1="{x+BOX_W}" y1="{TOP+BOX_H/2}" x2="{x+BOX_W+GAP-6}" y2="{TOP+BOX_H/2}" '
                  f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# 覆盖开关侧支:挂在 step1 下方
SIDE_Y = TOP + BOX_H + 90
sx0 = sx[0]
L.append(f'<line x1="{sx0+BOX_W*0.3}" y1="{TOP+BOX_H}" x2="{sx0+BOX_W*0.3}" y2="{SIDE_Y}" '
          f'stroke="#d97706" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#p)"/>')
L.append(f'<text x="{sx0+BOX_W*0.3+10}" y="{TOP+BOX_H+40}" font-family="sans-serif" font-size="10.5" '
          f'fill="#92400e">{esc("enable_in_reasoning=True")}</text>')
L.append(f'<rect x="{sx0}" y="{SIDE_Y}" width="{BOX_W}" height="76" rx="10" '
          f'fill="#fff7ed" stroke="#d97706" stroke-width="2" stroke-dasharray="4,3"/>')
L.append(f'<text x="{sx0+BOX_W/2}" y="{SIDE_Y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#92400e">{esc("覆盖:step1(enable_in_reasoning=True)")}</text>')
L.append(f'<text x="{sx0+BOX_W/2}" y="{SIDE_Y+42}" text-anchor="middle" font-family="monospace" '
          f'font-size="9.5" fill="#78350f">{esc("should_fill=True / should_advance=True")}</text>')
L.append(f'<text x="{sx0+BOX_W/2}" y="{SIDE_Y+62}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" font-weight="bold" fill="#92400e">{esc("第一步就受约束(只允许 5, 7)")}</text>')

FOOT_Y = SIDE_Y + 76 + 30
L.append(f'<rect x="{PAD}" y="{FOOT_Y}" width="{W-2*PAD}" height="66" rx="8" '
          f'fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+24}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("两道门的位置:should_fill_bitmask(填不填,__init__.py:L301-319)/ should_advance(推不推进,L321-357)")}</text>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+46}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("step3 侦测到 </think> 仍不推进——否则会拿 </think> 这个 token 去推进语法")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-reasoning-gate.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
