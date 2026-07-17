#!/usr/bin/env python3
"""fig-m1-binding-chain: swimlane 模板。
claim: 一次 _builder.create_make_range(0, 16) 依次穿过 Python / pybind11 / C++
TritonOpBuilder / MLIR OpBuilder 四条泳道，最终落成 1 个 tt.make_range op 并把
ir.value 递回 Python。
改造点：LANES 固定 4 条；ROWS 既支持跨泳道消息(MSG，箭头画在文本正下方)也支持
单泳道内注记(NOTE，表示"处理发生在这一层、不跨层"，画成一个顶部对齐的圆角框)。
布局采用"内容驱动、逐行累加"的栈式排布——每行的高度由自身文本行数算出，
不用对称居中，从根源上避免相邻行互相压字。全坐标由循环/常量计算，零魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["Python 前端", "pybind11 接缝 (.def)", "C++ TritonOpBuilder", "MLIR OpBuilder"]

# 行:("msg", from_lane, to_lane, [多行文本]) 或 ("note", lane, [多行文本])
ROWS = [
    ("msg", 0, 1, ['_builder.create_make_range(start=0, end=16)']),
    ("msg", 1, 2, ['按 .def("create_make_range") 匹配 lambda',
                   '(0, 16) → int start, int end']),
    ("note", 2, ['lambda: retType =',
                 'RankedTensorType::get({end-start=16}, i32)']),
    ("msg", 2, 3, ['self.create<MakeRangeOp>(retType, 0, 16)',
                   'create<OpTy>: loc=getLastLoc(); builder->create<MakeRangeOp>(loc,…)']),
    ("note", 3, ['落成 tt.make_range {start=0, end=16}',
                 ': tensor<16xi32>  —— 1 个 op']),
    ("msg", 3, 2, ['mlir::Value']),
    ("msg", 2, 1, ['转调回']),
    ("msg", 1, 0, ['return_value_policy 包成 ir.value']),
]

LANE_W = 300
PAD = 40
LANE_HEAD_H = 34
TOP = 90
LINE_H = 16          # 文本单行占用高度
MSG_TEXT_GAP = 10     # 文本块与箭头线之间的间距
MSG_BELOW_GAP = 26    # 箭头线到下一行内容的间距
NOTE_PAD_Y = 12       # note 框内上下各留白
NOTE_BELOW_GAP = 24   # note 框底到下一行内容的间距

w = PAD * 2 + LANE_W * (len(LANES) - 1) + 220
X = {i: PAD + 110 + i * LANE_W for i in range(len(LANES))}

# ---- 逐行栈式布局:content_top 是"这一行内容最上沿"的累加游标 ----
layout = []  # 每项: dict(kind=..., top=..., arrow_y=... / box=(y0,y1), ...)
cursor = TOP + 20
for row in ROWS:
    if row[0] == "msg":
        _, src, dst, lines = row
        n = len(lines)
        text_top = cursor
        arrow_y = text_top + n * LINE_H + MSG_TEXT_GAP
        layout.append({"kind": "msg", "src": src, "dst": dst, "lines": lines,
                        "text_top": text_top, "arrow_y": arrow_y})
        cursor = arrow_y + MSG_BELOW_GAP
    else:
        _, lane, lines = row
        n = len(lines)
        box_top = cursor
        box_h = NOTE_PAD_Y * 2 + n * LINE_H
        layout.append({"kind": "note", "lane": lane, "lines": lines,
                        "box_top": box_top, "box_h": box_h})
        cursor = box_top + box_h + NOTE_BELOW_GAP

lifeline_bottom = cursor - NOTE_BELOW_GAP + 10
h = lifeline_bottom + PAD + 34  # 底部留脚注一行

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
          f'font-size="17" font-weight="bold" fill="#0f172a">'
          f'{esc("create_make_range(0, 16) 的双语绑定链")}</text>')

# 泳道头 + 生命线
for i, name in enumerate(LANES):
    x = X[i]
    L.append(f'<rect x="{x-95}" y="{TOP-LANE_HEAD_H}" width="190" height="{LANE_HEAD_H}" rx="7" '
              'fill="#e2e8f0" stroke="#64748b" stroke-width="1.2"/>')
    L.append(f'<text x="{x}" y="{TOP-LANE_HEAD_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-2}" x2="{x}" y2="{lifeline_bottom}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

# 逐行绘制(按预算好的 layout)
for item in layout:
    if item["kind"] == "msg":
        src, dst, lines = item["src"], item["dst"], item["lines"]
        ry = item["arrow_y"]
        x1, x2 = X[src], X[dst]
        L.append(f'<line x1="{x1}" y1="{ry}" x2="{x2}" y2="{ry}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
        cx = (x1 + x2) / 2
        text_top = item["text_top"]
        for li, txt in enumerate(lines):
            ty = text_top + li * LINE_H + LINE_H - 4
            L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="11.5" fill="#334155">{esc(txt)}</text>')
    else:
        lane, lines = item["lane"], item["lines"]
        box_top, box_h = item["box_top"], item["box_h"]
        x = X[lane]
        box_w = 270
        bx = x - box_w / 2
        L.append(f'<rect x="{bx}" y="{box_top}" width="{box_w}" height="{box_h}" rx="6" '
                  'fill="#fef9c3" stroke="#ca8a04" stroke-width="1.2"/>')
        for li, txt in enumerate(lines):
            ty = box_top + NOTE_PAD_Y + li * LINE_H + LINE_H - 4
            L.append(f'<text x="{x}" y="{ty}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="11.5" fill="#713f12">{esc(txt)}</text>')

# 底部脚注:共享底座的数字(129→1)
foot_y = h - 20
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#475569">'
          f'{esc("pin v3.2.0：129 个 create_* 全部共用同一个 create<OpTy> 底座（ir.cc:L96），并非仅此一例")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-binding-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
