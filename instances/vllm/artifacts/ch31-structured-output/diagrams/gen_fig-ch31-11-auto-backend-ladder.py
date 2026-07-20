#!/usr/bin/env python3
"""fig-ch31-11: auto 是一条"先试 xgrammar,失败才降级"的两级阶梯,从不选 lm-format-enforcer。
template: flow（纵向阶梯：前端校验期 try/except 降级链）"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, fs):
    w = 0.0
    for ch in s:
        w += fs if ord(ch) > 0x2E80 else fs * 0.58
    return w

W, H = 1260, 820
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
          '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16.5" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("auto 的降级阶梯发生在前端校验期，不是引擎期的选择")}</text>')

def box(x, y, w, h, fill, stroke, lines, fs=13, tw="bold"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    n = len(lines)
    cy = y + h/2 - (n-1)*0.5*(fs+4)
    for i, t in enumerate(lines):
        fw = 'bold' if (tw == 'bold' and i == 0) else 'normal'
        L.append(f'<text x="{x+w/2}" y="{cy+i*(fs+4)+fs*0.35:.0f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" fill="#1e293b">{esc(t)}</text>')

def arrow(x1, y1, x2, y2, label=None, color="#334155", marker="url(#a)", fs=12, side="right", dx=12):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8" marker-end="{marker}"/>')
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        lw = text_w(label, fs) + 12
        if side == "right":
            lx = max(x1, x2) + dx
            L.append(f'<rect x="{lx}" y="{my-11}" width="{lw:.1f}" height="22" rx="4" fill="white" opacity="0.96"/>')
            L.append(f'<text x="{lx+lw/2:.1f}" y="{my+3}" text-anchor="middle" font-family="sans-serif" font-size="{fs}" fill="{color}">{esc(label)}</text>')
        else:
            L.append(f'<rect x="{mx-lw/2:.1f}" y="{my-11}" width="{lw:.1f}" height="22" rx="4" fill="white" opacity="0.96"/>')
            L.append(f'<text x="{mx}" y="{my+3}" text-anchor="middle" font-family="sans-serif" font-size="{fs}" fill="{color}">{esc(label)}</text>')

CX = W/2 - 160
BW = 460

# 1. 入口
y1 = 60
box(CX-BW/2, y1, BW, 44, "#e0e7ff", "#6366f1",
    ["入口：SamplingParams._validate_structured_outputs（前端 Processor）"], fs=11.5)

# 显式指定分支（右侧岔开，独立说明，不与阶梯交叉）
side_x, side_y, side_w, side_h = CX + BW/2 + 90, y1 - 4, 340, 52
box(side_x, side_y, side_w, side_h, "#f1f5f9", "#64748b",
    ["backend 显式指定", "（xgrammar/guidance/outlines/lm-format-enforcer）", "→ 各自 validate_* 预检，不降级"], fs=10.5, tw="normal")
arrow(CX+BW/2, y1+22, side_x, side_y+side_h/2, None, color="#94a3b8")

# 2. auto 判断
y2 = y1 + 44 + 46
box(CX-BW/2, y2, BW, 44, "#fef9c3", "#ca8a04",
    ["backend == 'auto'"], fs=13)
arrow(CX, y1+44, CX, y2, None)

# 3. try xgrammar
y3 = y2 + 44 + 46
box(CX-BW/2, y3, BW, 50, "#dcfce7", "#16a34a",
    ["try: validate_xgrammar_grammar(self)", "→ _backend = 'xgrammar'"], fs=12)
arrow(CX, y2+44, CX, y3, "try", side="left")

# 4. except -> skip_guidance 判断
y4 = y3 + 50 + 56
box(CX-BW/2, y4, BW, 56, "#fee2e2", "#dc2626",
    ["except ValueError → 判 skip_guidance", "（非 tekken Mistral 分词器 或 schema 含", "guidance 不支持特性，如 patternProperties）"], fs=10.5)
arrow(CX, y3+50, CX, y4, "except ValueError", color="#dc2626", marker="url(#r)", side="left")

# 5a. skip_guidance=True -> outlines
y5 = y4 + 56 + 56
half = BW/2 - 15
box(CX-BW/2, y5, half, 50, "#dbeafe", "#2563eb",
    ["validate_..._outlines", "→ _backend = 'outlines'"], fs=11)
arrow(CX-BW/4, y4+56, CX-half/2-CX+CX-BW/2+half/2, y5, None)  # placeholder unused
# 精简：直接画两条从判断框底部引出的箭头
L.append(f'<line x1="{CX-40}" y1="{y4+56}" x2="{CX-BW/2+half/2}" y2="{y5}" stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
lw = text_w("skip_guidance=True", 11.5) + 12
L.append(f'<rect x="{CX-BW/2+half/2-lw/2:.1f}" y="{y4+56+6}" width="{lw:.1f}" height="18" rx="4" fill="white" opacity="0.96"/>')
L.append(f'<text x="{CX-BW/2+half/2:.1f}" y="{y4+56+19}" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("skip_guidance=True")}</text>')

# 5b. skip_guidance=False -> guidance
box(CX+15, y5, half, 50, "#ede9fe", "#7c3aed",
    ["validate_guidance_grammar", "→ _backend = 'guidance'"], fs=11)
L.append(f'<line x1="{CX+40}" y1="{y4+56}" x2="{CX+15+half/2}" y2="{y5}" stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
lw2 = text_w("否则", 11.5) + 12
L.append(f'<rect x="{CX+15+half/2-lw2/2:.1f}" y="{y4+56+6}" width="{lw2:.1f}" height="18" rx="4" fill="white" opacity="0.96"/>')
L.append(f'<text x="{CX+15+half/2:.1f}" y="{y4+56+19}" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("否则")}</text>')

# 6. 出口
y6 = y5 + 50 + 50
box(CX-BW/2, y6, BW, 50, "#e2e8f0", "#475569",
    ["出口：params._backend 落定 + _backend_was_auto=True", "引擎侧 grammar_init 只读 _backend，不做任何选择"], fs=11)
L.append(f'<line x1="{CX-BW/2+half/2}" y1="{y5+50}" x2="{CX}" y2="{y6}" stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{CX+15+half/2}" y1="{y5+50}" x2="{CX}" y2="{y6}" stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

# 旁注：lm-format-enforcer 不在阶梯上
note_x, note_y, note_w, note_h = CX - BW/2, y6 + 50 + 30, BW, 50
L.append(f'<rect x="{note_x}" y="{note_y}" width="{note_w}" height="{note_h}" rx="8" '
          f'fill="#fefce8" stroke="#ca8a04" stroke-dasharray="5 3"/>')
L.append(f'<text x="{note_x+note_w/2}" y="{note_y+22}" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#854d0e">{esc("旁注：lm-format-enforcer 不在这条阶梯上")}</text>')
L.append(f'<text x="{note_x+note_w/2}" y="{note_y+40}" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
          f'fill="#78350f">{esc("它只能被显式指定，auto 从不会降级选中它")}</text>')

# 底部数字条
foot_y = H - 46
facts = [
    "auto 阶梯涉及的后端数：3（xgrammar → guidance / outlines）",
    "阶梯从不选中的后端数：1（lm-format-enforcer）",
    "auto 走完后 _backend_was_auto：True",
    "全引擎允许的后端实例数：1",
]
L.append(f'<rect x="40" y="{foot_y-24}" width="{W-80}" height="56" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
for i, f in enumerate(facts):
    fx = 60 + (i % 2) * (W/2 - 20)
    fy = foot_y - 4 + (i // 2) * 22
    L.append(f'<text x="{fx}" y="{fy}" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("• " + f)}</text>')

L.append('</svg>')
out = Path("fig-ch31-11-auto-backend-ladder.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
