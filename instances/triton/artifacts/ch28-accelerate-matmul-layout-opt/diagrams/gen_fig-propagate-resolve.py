#!/usr/bin/env python3
"""fig-propagate-resolve (flow 模板)
传播(setEncoding)两条规则 + 消冲突(resolveConflicts)两条偏好——
convert 被故意染成两端同色待删,多编码值按『访存偏 Blocked/计算偏 Mma』坍缩到单编码。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PAD = 50
TOP = 140
w = 1160
h = 460

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#6366f1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="44" font-family="sans-serif" font-size="18.5" '
          f'font-weight="bold" fill="#0f172a">{esc("传播(setEncoding) -> 消冲突(resolveConflicts):两步定出唯一编码")}</text>')
L.append(f'<text x="{PAD}" y="68" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("前向定点迭代给每个值攒编码集合,冲突时按值的用途(访存/计算)坍缩到单编码,重写才无歧义")}</text>')

STAGE_W = 500
STAGE_GAP = 100
S1_X = PAD
S2_X = PAD + STAGE_W + STAGE_GAP

# --- 阶段 1:传播 ---
L.append(f'<text x="{S1_X+STAGE_W/2}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14.5" font-weight="bold" fill="#1e3a5f">{esc("① 传播(setEncoding)")}</text>')

RULE_H = 74
r1_y = TOP + 20
box_style_1 = ("#eef2ff", "#6366f1", "#312e81")
L.append(f'<rect x="{S1_X}" y="{r1_y}" width="{STAGE_W}" height="{RULE_H}" rx="8" '
          f'fill="{box_style_1[0]}" stroke="{box_style_1[1]}" stroke-width="2"/>')
L.append(f'<text x="{S1_X+20}" y="{r1_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{box_style_1[2]}">{esc("遇 ConvertLayoutOp")}</text>')
L.append(f'<text x="{S1_X+20}" y="{r1_y+48}" font-family="sans-serif" font-size="12.5" '
          f'fill="{box_style_1[2]}">{esc("dstEncoding := 源编码(意图消掉它)")}</text>')
L.append(f'<text x="{S1_X+20}" y="{r1_y+66}" font-family="sans-serif" font-size="10.5" '
          f'fill="#818cf8">{esc("RemoveLayoutConversions.cpp:L217-L220")}</text>')

r2_y = r1_y + RULE_H + 20
L.append(f'<rect x="{S1_X}" y="{r2_y}" width="{STAGE_W}" height="{RULE_H}" rx="8" '
          f'fill="{box_style_1[0]}" stroke="{box_style_1[1]}" stroke-width="2"/>')
L.append(f'<text x="{S1_X+20}" y="{r2_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{box_style_1[2]}">{esc("遇其余 op(如 addf)")}</text>')
L.append(f'<text x="{S1_X+20}" y="{r2_y+48}" font-family="sans-serif" font-size="12.5" '
          f'fill="{box_style_1[2]}">{esc("dstEncoding := inferDstEncoding(op, encoding)")}</text>')
L.append(f'<text x="{S1_X+20}" y="{r2_y+66}" font-family="sans-serif" font-size="10.5" '
          f'fill="#818cf8">{esc("RemoveLayoutConversions.cpp:L221-L222")}</text>')

note_y = r2_y + RULE_H + 26
L.append(f'<text x="{S1_X}" y="{note_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("一个值可能被多条路径攒出多个编码 -> 需要下一步坍缩")}</text>')

# --- 阶段 2:消冲突 ---
L.append(f'<text x="{S2_X+STAGE_W/2}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14.5" font-weight="bold" fill="#1e3a5f">{esc("② 消冲突(resolveConflicts)")}</text>')

box_style_2 = ("#ecfdf5", "#059669", "#065f46")
L.append(f'<rect x="{S2_X}" y="{r1_y}" width="{STAGE_W}" height="{RULE_H}" rx="8" '
          f'fill="{box_style_2[0]}" stroke="{box_style_2[1]}" stroke-width="2"/>')
L.append(f'<text x="{S2_X+20}" y="{r1_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{box_style_2[2]}">{esc("值被 load/store(访存)消费")}</text>')
L.append(f'<text x="{S2_X+20}" y="{r1_y+48}" font-family="sans-serif" font-size="12.5" '
          f'fill="{box_style_2[2]}">{esc("冲突时坍缩偏好 -> Blocked")}</text>')
L.append(f'<text x="{S2_X+20}" y="{r1_y+66}" font-family="sans-serif" font-size="10.5" '
          f'fill="#34d399">{esc("RemoveLayoutConversions.cpp:L322-L325")}</text>')

L.append(f'<rect x="{S2_X}" y="{r2_y}" width="{STAGE_W}" height="{RULE_H}" rx="8" '
          f'fill="{box_style_2[0]}" stroke="{box_style_2[1]}" stroke-width="2"/>')
L.append(f'<text x="{S2_X+20}" y="{r2_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{box_style_2[2]}">{esc("值被计算 op(如 dot)消费")}</text>')
L.append(f'<text x="{S2_X+20}" y="{r2_y+48}" font-family="sans-serif" font-size="12.5" '
          f'fill="{box_style_2[2]}">{esc("冲突时坍缩偏好 -> Mma")}</text>')
L.append(f'<text x="{S2_X+20}" y="{r2_y+66}" font-family="sans-serif" font-size="10.5" '
          f'fill="#34d399">{esc("RemoveLayoutConversions.cpp:L322-L325")}</text>')

# 阶段间箭头(两条规则框分别指向"坍缩到单编码"汇聚点由中间箭头体现)
mid_y1 = r1_y + RULE_H / 2
mid_y2 = r2_y + RULE_H / 2
arrow_x1 = S1_X + STAGE_W + 14
arrow_x2 = S2_X - 14
L.append(f'<line x1="{arrow_x1}" y1="{(mid_y1+mid_y2)/2}" x2="{arrow_x2}" y2="{(mid_y1+mid_y2)/2}" '
          'stroke="#6366f1" stroke-width="2.4" marker-end="url(#a)"/>')
L.append(f'<text x="{(arrow_x1+arrow_x2)/2}" y="{(mid_y1+mid_y2)/2-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#4f46e5">{esc("多编码值")}</text>')

foot_y = note_y + 40
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="70" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+20}" y="{foot_y+26}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("两阶段合力:convert 被两端染同色(等着被删),多编码值坍缩到唯一编码——这是四阶段消 convert 中")}</text>')
L.append(f'<text x="{PAD+20}" y="{foot_y+48}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("真正只做分析、不改 IR 的两步(第 ④ 步 rewrite 才真正删除,见本章另一张四阶段图)。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-propagate-resolve.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
