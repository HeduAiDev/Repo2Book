#!/usr/bin/env python3
"""一拍 schedule() 的 token 预算分配：先 RUNNING decode，再 WAITING prefill，直到预算耗尽。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 880, 460
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<rect width="%d" height="%d" fill="white"/>' % (w, h))
L.append(f'<text x="{w//2}" y="34" text-anchor="middle" font-size="21" '
         'font-weight="bold" fill="#0f172a">一拍 token_budget = 64 的分配（批大小 = token 数）</text>')

# budget bar geometry — by=130 gives enough room above for 3 staggered callout labels
bx, by, bw, bh = 60, 130, 760, 56
budget = 64
scale = bw / budget

# segments: (label, tokens, color, kind)
segs = [
    ("decode a", 1, "#22c55e", "RUNNING"),
    ("decode b", 1, "#22c55e", "RUNNING"),
    ("decode c", 1, "#22c55e", "RUNNING"),
    ("prefill d (chunk)", 40, "#3b82f6", "WAITING"),
    ("prefill e (chunk)", 21, "#3b82f6", "WAITING"),
]

# outline of full budget
L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="#f1f5f9" '
         'stroke="#cbd5e1" stroke-width="1.5"/>')

cur = bx
remaining = budget
small_idx = 0  # stagger small-segment labels to avoid collision
for label, tok, color, kind in segs:
    sw = tok * scale
    L.append(f'<rect x="{cur:.1f}" y="{by}" width="{sw:.1f}" height="{bh}" '
             f'fill="{color}" stroke="white" stroke-width="1.5"/>')
    # label inside or above
    cx = cur + sw / 2
    if sw > 70:
        L.append(f'<text x="{cx:.1f}" y="{by+24}" text-anchor="middle" font-size="12.5" '
                 f'fill="white" font-weight="bold">{esc(label)}</text>')
        L.append(f'<text x="{cx:.1f}" y="{by+42}" text-anchor="middle" font-size="12.5" '
                 f'fill="white">{tok} tok</text>')
    else:
        # small segment: leader line up to a staggered callout (avoid label collision)
        lift = by - 14 - small_idx * 18
        L.append(f'<line x1="{cx:.1f}" y1="{by}" x2="{cx:.1f}" y2="{lift+4:.1f}" '
                 'stroke="#16a34a" stroke-width="0.8"/>')
        L.append(f'<text x="{cx-4:.1f}" y="{lift:.1f}" text-anchor="end" font-size="10.5" '
                 f'fill="#166534">{esc(label)} ({tok})</text>')
        small_idx += 1
    cur += sw
    remaining -= tok

# red line at max
L.append(f'<line x1="{bx+bw}" y1="{by-14}" x2="{bx+bw}" y2="{by+bh+14}" '
         'stroke="#dc2626" stroke-width="2.5"/>')
L.append(f'<text x="{bx+bw}" y="{by-22}" text-anchor="end" font-size="13" '
         'font-weight="bold" fill="#dc2626">max_num_scheduled_tokens = 64</text>')

# budget countdown row
L.append(f'<text x="{bx}" y="{by+bh+50}" font-size="13" fill="#334155" font-weight="bold">'
         'token_budget 逐段递减：</text>')
steps = "64 → 63 → 62 → 61 → 21 → 0"
L.append(f'<text x="{bx}" y="{by+bh+74}" font-size="14" fill="#0f172a" '
         f'font-family="monospace">{esc(steps)}</text>')

# legend
ly = 340
L.append(f'<rect x="{bx}" y="{ly}" width="22" height="22" fill="#22c55e"/>')
L.append(f'<text x="{bx+30}" y="{ly+16}" font-size="13.5" fill="#0f172a">'
         'RUNNING 的 decode：每请求 1 token（追赶公式 = 1）</text>')
L.append(f'<rect x="{bx}" y="{ly+34}" width="22" height="22" fill="#3b82f6"/>')
L.append(f'<text x="{bx+30}" y="{ly+50}" font-size="13.5" fill="#0f172a">'
         'WAITING 的 prefill chunk：数十~数百 token，按剩余预算截断</text>')

# punchline
L.append(f'<text x="{w//2}" y="434" text-anchor="middle" font-size="13.5" '
         'fill="#475569" font-style="italic">'
         'prefill 的大块与 decode 的单 token 共享同一个预算池 —— 同一拍混批，不分相</text>')

L.append('</svg>')
open("13-token-centric-budget.svg", "w").write('\n'.join(L))
print("wrote 13-token-centric-budget.svg")
