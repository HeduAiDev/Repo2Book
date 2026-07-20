#!/usr/bin/env python3
"""fig-ch31-18: 带约束请求的端到端接缝——五个接缝，本章讲到第五个交棒点为止。
template: swimlane（纵向流程：前端→引擎建请求→EngineCore→调度→交棒，每步一个泳道段）"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1180, 1180
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("带约束请求的端到端接缝：五步走到 get_grammar_bitmask 交棒为止")}</text>')

STEPS = [
    ("①", "前端", "#6366f1", "#e0e7ff",
     ["StructuredOutputsParams 六选一（__post_init__ 互斥校验）",
      "→ _validate_structured_outputs 定后端",
      "并可能原地改写请求（choice → EBNF）"], None),
    ("②", "引擎建请求", "#0891b2", "#cffafe",
     ["Request.__init__ 挂 StructuredOutputRequest",
      "初始 status = WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR(=2)"], None),
    ("③", "EngineCore（ch11）", "#7c3aed", "#ede9fe",
     ["preprocess_add_request：use_structured_output 为真",
      "→ grammar_init，编译扔进线程池"], None),
    ("④", "调度（ch13）", "#16a34a", "#dcfce7",
     ["阻塞态 → skipped_waiting",
      "grammar 就绪 → status 改 WAITING(=1)，可被调度"], None),
    ("⑤", "交棒", "#ea580c", "#ffedd5",
     ["get_grammar_bitmask 筛出本步的结构化请求 id",
      "→ 下一章：批装配 / 并行填充 / 上卡打 -inf"], "本章终点"),
]

LANE_X = 130
BOX_X, BOX_W = 210, 780
BOX_H = 92
TOP = 70
GAP = 44

y_positions = []
for i, (num, title, color, fill, lines, tag) in enumerate(STEPS):
    y = TOP + i * (BOX_H + GAP)
    y_positions.append(y)
    # 圆形序号节点
    L.append(f'<circle cx="{LANE_X}" cy="{y+BOX_H/2}" r="26" fill="{fill}" stroke="{color}" stroke-width="2.5"/>')
    L.append(f'<text x="{LANE_X}" y="{y+BOX_H/2+8}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="20" font-weight="bold" fill="{color}">{esc(num)}</text>')
    # 主体框
    L.append(f'<rect x="{BOX_X}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{color}" stroke-width="2"/>')
    L.append(f'<text x="{BOX_X+18}" y="{y+24}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="{color}">{esc(title)}</text>')
    for k, line in enumerate(lines):
        L.append(f'<text x="{BOX_X+18}" y="{y+46+k*19}" font-family="sans-serif" font-size="12" '
                  f'fill="#1e293b">{esc(line)}</text>')
    if tag:
        tag_w = len(tag) * 13 + 20
        L.append(f'<rect x="{BOX_X+BOX_W-tag_w-14}" y="{y+12}" width="{tag_w}" height="22" rx="5" '
                  f'fill="white" stroke="{color}" stroke-width="1.5"/>')
        L.append(f'<text x="{BOX_X+BOX_W-tag_w/2-14}" y="{y+27}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="{color}">{esc(tag)}</text>')
    # 连接线（序号圆心之间，箭头向下指，与①→⑤的阅读顺序一致）
    if i > 0:
        prev_cy = y_positions[i-1] + BOX_H/2
        cur_cy = y + BOX_H/2
        L.append(f'<line x1="{LANE_X}" y1="{prev_cy+26}" x2="{LANE_X}" y2="{cur_cy-26}" '
                  f'stroke="#94a3b8" stroke-width="2.5" marker-end="url(#a)"/>')

# 底部数字条
foot_y = y_positions[-1] + BOX_H + 46
facts = [
    "用户可选的约束形态：6（json/json_object/regex/choice/grammar/structural_tag，互斥）",
    "带约束请求的初始状态：WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR(=2)",
    "晋级后的状态：WAITING(=1)",
    "本章终点：scheduler.py:L1224-1246 get_grammar_bitmask",
]
L.append(f'<rect x="40" y="{foot_y}" width="{W-80}" height="70" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
for i, f in enumerate(facts):
    fx = 60 + (i % 2) * (W/2 - 30)
    fy = foot_y + 26 + (i // 2) * 24
    L.append(f'<text x="{fx}" y="{fy}" font-family="sans-serif" font-size="12" fill="#334155">{esc("• " + f)}</text>')

H2 = foot_y + 70 + 30
L.append('</svg>')
svg = '\n'.join(L).replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H2}"').replace(
    f'<rect width="{W}" height="{H}" fill="white"/>', f'<rect width="{W}" height="{H2}" fill="white"/>')
out = Path("fig-ch31-18-end-to-end-seam.svg")
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}, H={H2}")
