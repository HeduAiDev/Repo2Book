#!/usr/bin/env python3
"""flow 模板(定制,仿 ch14 driver-loop 主干+分支写法):visit_Call 四问级联分诊。
四个决策盒纵向级联,每个"命中"侧岔出到右侧结果叶子;"否"则继续往下一问。
改造点:QUESTIONS(四问 + 命中叶子文案)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def multiline(lines, cx, y0, size=12, weight=False, fill="#0f172a", lh=15):
    out = []
    wattr = 'font-weight="bold" ' if weight else ''
    for k, line in enumerate(lines):
        out.append(f'<text x="{cx}" y="{y0 + k * lh}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{size}" {wattr}'
                    f'fill="{fill}">{esc(line)}</text>')
    return out


# (badge, 问句, 命中叶子标题, 命中叶子细节行, 建op?, 叶子色调key)
QUESTIONS = [
    ("①", "fn ∈ statically_implemented_functions?",
     "static:编译期求值", ["返回 constexpr,不建 op", "如 static_assert/static_print/int/len"], "no-op"),
    ("②", "isinstance(fn, JITFunction)?",
     "JITFunction:内联", ["call_JitFunction 内联,建 tt.call", "如 _helper(x, BLOCK)"], "build"),
    ("③", "有 tensor __self__ 或 is_builtin(fn)?",
     "builtin:建 op", ["注入 _builder,建 tt.* op", "如 tl.load / x.to(tl.float32)"], "build"),
]
FALLBACK = ("④", "兜底:纯 Python callable",
            ["unwrap constexpr 后宿主直调", "不建 op,如 range(0, N)"], "no-op")

COLOR = {"no-op": ("#f1f5f9", "#475569"), "build": ("#dbeafe", "#1d4ed8")}
DECISION_FILL, DECISION_STROKE = "#fef9c3", "#a16207"

ENTRY_W, ENTRY_H = 420, 56
Q_W, Q_H = 420, 60
LEAF_W, LEAF_H = 380, 70
VGAP = 34
LEAF_GAP_X = 70

PAD_L, TOP = 60, 90
lane_cx = PAD_L + Q_W / 2 + 40
leaf_cx = lane_cx + Q_W / 2 + LEAF_GAP_X + LEAF_W / 2

w = leaf_cx + LEAF_W / 2 + 50
n_q = len(QUESTIONS)
h = TOP + ENTRY_H + VGAP + n_q * (Q_H + VGAP) + LEAF_H + VGAP + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
          '<marker id="s" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
          '</defs>')
L.append(f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>')
L.append(f'<text x="{PAD_L}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("visit_Call 四问级联分诊:命中即返回,顺序不可交换")}</text>')
L.append(f'<text x="{PAD_L}" y="56" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("回收 ch01 f4 心智模型:static / JITFunction / builtin 三岔 + 兜底纯 Python;static 必须先于 builtin 判定")}</text>')

# entry
y = TOP
L.append(f'<rect x="{lane_cx-ENTRY_W/2}" y="{y}" width="{ENTRY_W}" height="{ENTRY_H}" rx="10" '
          f'fill="#e2e8f0" stroke="#334155" stroke-width="1.6"/>')
L += multiline(["visit_Call(node): fn = visit(node.func)"], lane_cx, y + ENTRY_H/2 + 5, size=13, weight=True)
y += ENTRY_H + VGAP

q_centers = []
for badge, question, leaf_title, leaf_lines, kind in QUESTIONS:
    q_centers.append(y)
    L.append(f'<line x1="{lane_cx}" y1="{y-VGAP+4}" x2="{lane_cx}" y2="{y-4}" '
              'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<rect x="{lane_cx-Q_W/2}" y="{y}" width="{Q_W}" height="{Q_H}" rx="10" '
              f'fill="{DECISION_FILL}" stroke="{DECISION_STROKE}" stroke-width="1.6"/>')
    L.append(f'<circle cx="{lane_cx-Q_W/2+22}" cy="{y+Q_H/2}" r="15" fill="{DECISION_STROKE}"/>')
    L.append(f'<text x="{lane_cx-Q_W/2+22}" y="{y+Q_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{esc(badge)}</text>')
    L += multiline([question], lane_cx + 14, y + Q_H/2 + 5, size=12.5, weight=True, fill="#78350f")
    # 命中 -> 右侧叶子
    leaf_fill, leaf_stroke = COLOR[kind]
    marker = "b" if kind == "build" else "s"
    branch_y = y + Q_H / 2
    L.append(f'<line x1="{lane_cx+Q_W/2}" y1="{branch_y}" x2="{leaf_cx-LEAF_W/2}" y2="{branch_y}" '
              f'stroke="{leaf_stroke}" stroke-width="1.8" marker-end="url(#{marker})"/>')
    L.append(f'<text x="{(lane_cx+Q_W/2+leaf_cx-LEAF_W/2)/2}" y="{branch_y-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" fill="{leaf_stroke}">{esc("是")}</text>')
    L.append(f'<rect x="{leaf_cx-LEAF_W/2}" y="{branch_y-LEAF_H/2}" width="{LEAF_W}" height="{LEAF_H}" rx="10" '
              f'fill="{leaf_fill}" stroke="{leaf_stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{leaf_cx}" y="{branch_y-LEAF_H/2+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{leaf_stroke}">{esc(leaf_title)}</text>')
    L += multiline(leaf_lines, leaf_cx, branch_y - LEAF_H/2 + 40, size=10.5, fill="#334155")
    L.append(f'<text x="{lane_cx-8}" y="{y+Q_H+VGAP/2+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc("否 ↓")}</text>')
    y += Q_H + VGAP

# fallback (④,主干终点,直接在主干上,不侧岔)
badge, title, lines, kind = FALLBACK
fb_fill, fb_stroke = COLOR[kind]
L.append(f'<rect x="{lane_cx-Q_W/2}" y="{y}" width="{Q_W}" height="{LEAF_H}" rx="10" '
          f'fill="{fb_fill}" stroke="{fb_stroke}" stroke-width="1.8"/>')
L.append(f'<circle cx="{lane_cx-Q_W/2+22}" cy="{y+LEAF_H/2}" r="15" fill="{fb_stroke}"/>')
L.append(f'<text x="{lane_cx-Q_W/2+22}" y="{y+LEAF_H/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{esc(badge)}</text>')
L.append(f'<text x="{lane_cx+14}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="{fb_stroke}">{esc(title)}</text>')
L += multiline(lines, lane_cx + 14, y + 42, size=10.5, fill="#334155")

foot_y = y + LEAF_H + 40
L.append(f'<text x="{PAD_L}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("分派出口数=4(code_generator.py:L1099-L1126);static 表项数=4(int/len/static_assert/static_print,L1252-L1257);本例 6 个调用点中建 IR op 数=3")}</text>')
L.append(f'<text x="{PAD_L}" y="{foot_y + 20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("蓝=建 IR op(JITFunction 内联 / builtin);灰=不建 op(编译期折叠 / 兜底纯 Python);黄=决策问句")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("ch16-fig-visit-call-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w:.0f}x{h:.0f}")
