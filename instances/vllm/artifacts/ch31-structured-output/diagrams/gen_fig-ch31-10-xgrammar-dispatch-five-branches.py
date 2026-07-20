#!/usr/bin/env python3
"""fig-ch31-10: xgrammar 的 compile_grammar 只有五个分支——CHOICE 在校验期已被改写成 GRAMMAR。
template: flow。CHOICE/else 的关系被拆成入口行右侧的小旁支，完全不与五分支行交叉。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, fs):
    w = 0.0
    for ch in s:
        w += fs if ord(ch) > 0x2E80 else fs * 0.58
    return w

W, H = 1680, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="d" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
          '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("六个约束枚举，xgrammar 的 compile_grammar 却只有五个分支")}</text>')

def box(x, y, w, h, fill, stroke, lines, fs=13, tw="bold", dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" '
              f'stroke-width="1.8"{dash_attr}/>')
    n = len(lines)
    cy = y + h/2 - (n-1)*0.5*(fs+4)
    for i, t in enumerate(lines):
        fw = 'bold' if (tw == 'bold' and i == 0) else 'normal'
        L.append(f'<text x="{x+w/2}" y="{cy+i*(fs+4)+fs*0.35:.0f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" fill="#1e293b">{esc(t)}</text>')

def arrow(x1, y1, x2, y2, label=None, color="#334155", marker="url(#a)", dash=None, fs=11.5, ldy=0):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8" '
              f'marker-end="{marker}"{dash_attr}/>')
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + ldy
        lw = text_w(label, fs) + 12
        L.append(f'<rect x="{mx-lw/2:.1f}" y="{my-10}" width="{lw:.1f}" height="18" rx="4" fill="white" opacity="0.95"/>')
        L.append(f'<text x="{mx}" y="{my+3}" text-anchor="middle" font-family="sans-serif" font-size="{fs}" '
                  f'fill="{color}">{esc(label)}</text>')

# ---- 第 1 行：入口 + CHOICE 旁支（else），完全独立于五分支行 ----
# 框宽按文字实际估宽取(下限 420),否则这行长文字会压在圆角边框上、
# 右侧红色虚线箭头正好从压字处起笔。
entry_line = "入口：structured_output_key = (request_type, grammar_spec)"
entry_w, entry_h = max(420, text_w(entry_line, 12.5) + 40), 42
entry_x, entry_y = 90, 66
box(entry_x, entry_y, entry_w, entry_h, "#e0e7ff", "#6366f1", [entry_line], fs=12.5)

else_w, else_h = 430, 56
else_x, else_y = entry_x + entry_w + 130, entry_y - 7
box(else_x, else_y, else_w, else_h, "#fee2e2", "#dc2626",
    ["else → raise ValueError", "\"Validation should have already occurred\""], fs=11.5)
arrow(entry_x + entry_w, entry_y + entry_h/2, else_x, else_y + else_h/2,
      "CHOICE 若真调入这一层", color="#dc2626", marker="url(#d)", dash="5 3", ldy=-14)

# ---- 第 2 行：五个分支 ----
BRANCHES = [
    ("JSON", ["compile_json_schema(", "grammar_spec, any_whitespace=…)"]),
    ("JSON_OBJECT", ["compile_json_schema(", "'{\"type\": \"object\"}')"]),
    ("GRAMMAR", ["compiler.compile_grammar(spec)"]),
    ("REGEX", ["compiler.compile_regex(spec)"]),
    ("STRUCTURAL_TAG", ["compiler.compile_structural_tag(", "spec)"]),
]
BX_W, BX_H = 280, 84
gap = 30
total_w = len(BRANCHES) * BX_W + (len(BRANCHES) - 1) * gap
start_x = (W - total_w) / 2
by = 190
xs_list = []
for i, (name, call_lines) in enumerate(BRANCHES):
    bx = start_x + i * (BX_W + gap)
    xs_list.append(bx)
    fill, stroke = ("#fef9c3", "#ca8a04") if name == "GRAMMAR" else ("#dbeafe", "#2563eb")
    box(bx, by, BX_W, BX_H, fill, stroke, [f"分支：{name}"] + call_lines, fs=12)
    arrow(entry_x + 60 + i * (entry_w - 120) / (len(BRANCHES) - 1), entry_y + entry_h,
          bx + BX_W/2, by, None)

# ---- 第 3 行：GRAMMAR 分支出口 ----
exit_w, exit_h = 560, 46
exit_x, exit_y = W/2 - exit_w/2, by + BX_H + 70
box(exit_x, exit_y, exit_w, exit_h, "#dcfce7", "#16a34a",
    ["出口：XgrammarGrammar(matcher=GrammarMatcher(ctx, max_rollback_tokens=…))"], fs=11.5)
arrow(xs_list[2] + BX_W/2, by + BX_H, exit_x + exit_w/2, exit_y, "GRAMMAR 分支出口", color="#16a34a",
      marker="url(#g)", ldy=-12)

# ---- CHOICE 改写旁注（放在五分支行下方左侧，独立区块） ----
note_x, note_y, note_w, note_h = start_x, exit_y + exit_h + 30, 480, 96
L.append(f'<rect x="{note_x}" y="{note_y}" width="{note_w}" height="{note_h}" rx="10" '
          f'fill="#fefce8" stroke="#ca8a04" stroke-dasharray="5 3"/>')
L.append(f'<text x="{note_x+18}" y="{note_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#854d0e">{esc("旁注：CHOICE 不落到这五个分支里的任何一个")}</text>')
for k, line in enumerate([
    "validate_xgrammar_grammar 在前端校验期把 CHOICE",
    "原地改写成 EBNF 文法（choice=None, grammar=…），",
    "改写后 structured_output_key 走的是 GRAMMAR(=4) 分支",
]):
    L.append(f'<text x="{note_x+18}" y="{note_y+50+k*18}" font-family="sans-serif" font-size="12" '
              f'fill="#78350f">{esc(line)}</text>')

# ---- 底部数字条 ----
foot_y = H - 46
facts = [
    "枚举成员数：6",
    "compile_grammar 分支数：5（backend_xgrammar.py:L77-122）",
    "CHOICE 走到这里的结果：落 else → ValueError",
    "改写后实际命中的分支：GRAMMAR(=4)",
]
L.append(f'<rect x="40" y="{foot_y-24}" width="{W-80}" height="56" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
for i, f in enumerate(facts):
    fx = 60 + (i % 2) * (W/2 - 20)
    fy = foot_y - 4 + (i // 2) * 22
    L.append(f'<text x="{fx}" y="{fy}" font-family="sans-serif" font-size="12" fill="#334155">{esc("• " + f)}</text>')

L.append('</svg>')
out = Path("fig-ch31-10-xgrammar-dispatch-five-branches.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
