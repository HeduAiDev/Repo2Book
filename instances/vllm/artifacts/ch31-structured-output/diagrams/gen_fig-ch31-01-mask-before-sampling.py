#!/usr/bin/env python3
"""fig-ch31-01: 约束解码的作用点在采样之前——fill_bitmask 是与采样的唯一接口。
template: flow（横向管线：语法状态机 -> 位掩码 -> logits -> 采样器）"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

def text_w(s, fs):
    # 粗略估算：CJK 宽度约等于字号，ASCII 约 0.58*字号
    w = 0.0
    for ch in s:
        w += fs if ord(ch) > 0x2E80 else fs * 0.58
    return w

W, H = 1420, 460
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
          f'font-weight="bold" fill="#0f172a">{esc("约束解码只在采样之前插一层掩码，采样算法本身一行不改")}</text>')

def box(x, y, w, h, fill, stroke, lines, fs=14, tw="bold"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * 0.5 * (fs + 5)
    for i, t in enumerate(lines):
        fw = 'bold' if (tw == 'bold' and i == 0) else 'normal'
        L.append(f'<text x="{x + w/2}" y="{cy + i*(fs+5) + fs*0.35:.0f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" fill="#1e293b">{esc(t)}</text>')

def arrow(x1, y1, x2, y2, label=None, label_dy=-15, color="#334155"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + label_dy
        lw = text_w(label, 12.5) + 16
        L.append(f'<rect x="{mx-lw/2:.1f}" y="{my-13}" width="{lw:.1f}" height="20" rx="4" fill="white" opacity="0.95"/>')
        L.append(f'<text x="{mx}" y="{my+2}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
                  f'font-weight="bold" fill="#b45309">{esc(label)}</text>')

TOP = 96
BH = 100
BW = 220
GAP = 140
xs_ = [60, 60 + (BW + GAP), 60 + 2*(BW + GAP), 60 + 3*(BW + GAP)]

# 1. 语法状态机
box(xs_[0], TOP, BW, BH, "#e0e7ff", "#6366f1",
    ["语法状态机", "StructuredOutputGrammar", "（哪些 token 现在合法）"], fs=13)

# 2. 位掩码
box(xs_[1], TOP, BW, BH, "#fee2e2", "#dc2626",
    ["位掩码", "每 token 1 bit：1=允许 / 0=禁止", "18.3 KB（|V|=150000）"], fs=13)

# 3. logits
box(xs_[2], TOP, BW, BH, "#dbeafe", "#2563eb",
    ["logits", "|V| 个 float32（ch30 采样输入）", "585.9 KB（|V|=150000）"], fs=13)

# 4. 采样器
box(xs_[3], TOP, BW, BH, "#dcfce7", "#16a34a",
    ["采样器（ch30）", "温度 / top-p / top-k", "逻辑一行不改"], fs=13)

ay = TOP + BH / 2
arrow(xs_[0] + BW, ay, xs_[1], ay, "fill_bitmask()", label_dy=-18)
arrow(xs_[1] + BW, ay, xs_[2], ay, "作用点：采样之前", label_dy=-18)
arrow(xs_[2] + BW, ay, xs_[3], ay, "流程不变", label_dy=-18)

# 六方法契约旁注
L.append(f'<text x="{W/2}" y="{TOP + BH + 40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" fill="#7c2d12" font-weight="bold">'
          f'{esc("六方法契约中，与采样交互的方法只有这 1 个：fill_bitmask(bitmask, batch_index)")}</text>')

# 下一章旁注（掩码怎么打 -inf）
note_y = TOP + BH + 66
note_x = xs_[1]
note_w = xs_[2] + BW - xs_[1]
L.append(f'<rect x="{note_x}" y="{note_y}" width="{note_w}" height="42" rx="8" '
          f'fill="#fefce8" stroke="#ca8a04" stroke-dasharray="5 3"/>')
L.append(f'<text x="{note_x+note_w/2}" y="{note_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#854d0e">'
          f'{esc("下一章：怎么把 0 位打成 -inf、上卡并行填充")}</text>')

# 数字对比条：18.3 KB vs 585.9 KB（约小 32 倍）
bar_y = note_y + 70
bar_x0 = xs_[1]
bar_w_max = 340
mask_w = bar_w_max * (18752 / 600000)
logits_w = bar_w_max
L.append(f'<text x="{bar_x0}" y="{bar_y-8}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155" font-weight="bold">{esc("同一行的字节数对比（约小 32 倍）")}</text>')
L.append(f'<rect x="{bar_x0}" y="{bar_y}" width="{mask_w:.1f}" height="20" rx="4" fill="#dc2626"/>')
L.append(f'<text x="{bar_x0+mask_w+8:.1f}" y="{bar_y+15}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("位掩码 18752 B（18.3 KB）")}</text>')
L.append(f'<rect x="{bar_x0}" y="{bar_y+28}" width="{logits_w:.1f}" height="20" rx="4" fill="#2563eb"/>')
L.append(f'<text x="{bar_x0+logits_w+8:.1f}" y="{bar_y+43}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("logits 600000 B（585.9 KB）")}</text>')

L.append('</svg>')
open("fig-ch31-01-mask-before-sampling.svg", "w", encoding="utf-8").write('\n'.join(L))
print("ok")
