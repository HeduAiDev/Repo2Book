#!/usr/bin/env python3
"""flow 模板(定制,线性单链):divisibility 提示从 launch 期一路落到 tt.func 属性、
再被下游消费的端到端证据链。7 节点纵向单链,首节点标『回指』、尾节点标『前瞻』。
改造点:CHAIN(节点文案 + 分类色)。全坐标计算,零魔数。"""
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


# (标题, 细节行, kind) kind in {"past","chapter","future","mid"}
CHAIN = [
    ("launch 期:multiple_of 打标记 + 特化", ["回指第 9/12 章:算出哪些指针 16 对齐"], "past"),
    ("AttrsDescriptor._add_common_properties",
     ["property_values['tt.divisibility']=16", "arg_properties 收 16 对齐参数序号"], "mid"),
    ("get_fn_attrs()", ["打包成 {argIdx: [('tt.divisibility', 16)]}"], "mid"),
    ("ast_to_ttir 的 fn_attrs -> CodeGenerator.attributes", [], "mid"),
    ("visit_FunctionDef:set_arg_attr(idx, 'tt.divisibility', 16)",
     ["本章讲的那一步——提示真正落进 IR"], "chapter"),
    ("追踪期 tt.func:%argN {tt.divisibility = 16}", ["本例落属性的参数数 = 3"], "mid"),
    ("下游 AxisInfo / coalesce 消费属性", ["前瞻第 25 章:向量化访存的源头"], "future"),
]

COLOR = {
    "past": ("#f1f5f9", "#475569"),
    "mid": ("#dbeafe", "#1d4ed8"),
    "chapter": ("#fef3c7", "#b45309"),
    "future": ("#ede9fe", "#6d28d9"),
}

BOX_W, BOX_H, VGAP = 560, 66, 34
PAD_L, TOP = 60, 100
lane_cx = PAD_L + BOX_W / 2

w = PAD_L * 2 + BOX_W
n = len(CHAIN)
h = TOP + n * (BOX_H + VGAP) + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD_L}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("divisibility 提示的端到端证据链:链上任一环断,后端就分析不出对齐")}</text>')
L.append(f'<text x="{PAD_L}" y="56" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("性能落点收尾:launch 期算出的对齐信息,必须钉成 tt.func 参数属性,后端才读得到")}</text>')

y = TOP
centers = []
for i, (title, detail, kind) in enumerate(CHAIN):
    fill, stroke = COLOR[kind]
    centers.append(y)
    L.append(f'<rect x="{lane_cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    n_lines = 1 + len(detail)
    title_y = y + (BOX_H/2 + 5 if not detail else BOX_H/2 - (5 if len(detail) == 1 else 12))
    L += multiline([title], lane_cx, title_y, size=13, weight=True, fill=stroke)
    if detail:
        L += multiline(detail, lane_cx, title_y + 19, size=10.5, fill="#334155")
    if i < n - 1:
        y2 = y + BOX_H + VGAP
        L.append(f'<line x1="{lane_cx}" y1="{y+BOX_H}" x2="{lane_cx}" y2="{y2-4}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
    y += BOX_H + VGAP

foot_y = centers[-1] + BOX_H + 40
L.append(f'<text x="{PAD_L}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("divisibility 值=16(backends/compiler.py:L77)")}</text>')
L.append(f'<text x="{PAD_L}" y="{foot_y+20}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("本例落属性的参数数=3(traces/m8_functiondef_idx.json)")}</text>')
L.append(f'<text x="{PAD_L}" y="{foot_y+40}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("灰=已讲过的上游(回指);蓝=链路中段;橙=本章 CodeGenerator 落属性的那一步;紫=尚未讲到的下游消费者(前瞻)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("ch16-fig-divisibility-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
