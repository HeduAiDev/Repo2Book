#!/usr/bin/env python3
"""fig-ch14-m5-triggers — flow 模板:matchAndRewrite 的多级放行/兜底闸门。
入口 → discreteAttrName 判定(命中→已处理跳过,独立终止,不汇入兜底;未命中→继续)→
早退放行门(isStructured && !discreteMask && (!scalarLike||形状全1))→
真:结构化路径出口(绿);假:落到三条触发解释(mayDiscretememaccess 逃生口/触发组A/触发组B 对齐闸)→
汇入 Unstructure 标量化出口(橙)。
注意:discreteAttrName(命中即 return failure()、pass 完全不碰该 op)与
mayDiscretememaccess/checkUnstructureAnnotated(命中→强制置 unstructured、真正标量化)
是两个效果相反的检查点,本图故意画成两条不合流的支路,不共用一个「逃生口」框。
数据取自 explainer m5.figure_specs.numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W = 1600
H = 830
PAD = 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
     '<marker id="o" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker>'
     '<marker id="s" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="36" text-anchor="middle" font-family="sans-serif" '
     f'font-size="17" font-weight="bold" fill="#0f172a">'
     f'{esc("matchAndRewrite 的多级闸门:早退放行 vs 落入 Unstructure 兜底")}</text>',
     f'<text x="{W/2}" y="58" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12.5" fill="#475569">'
     f'{esc("只有整体 isStructured 且非离散 mask 且非广播 scalarLike,才交给 ch12/13 结构化路径")}</text>']

# 入口
ex, ey, ew, eh = W/2 - 130, 80, 260, 46
L.append(f'<rect x="{ex}" y="{ey}" width="{ew}" height="{eh}" rx="10" '
         'fill="#e2e8f0" stroke="#334155" stroke-width="1.5"/>')
L.append(f'<text x="{ex+ew/2}" y="{ey+eh/2+5}" text-anchor="middle" font-family="monospace" '
         f'font-size="13" fill="#0f172a">{esc("访存 op(load/store)")}</text>')

# discreteAttrName 判定(先于早退门:命中直接 return failure(),本 pass 完全不碰该 op)
dx, dy, dw, dh = W/2 - 240, ey + eh + 40, 480, 60
L.append(f'<line x1="{ex+ew/2}" y1="{ey+eh}" x2="{ex+ew/2}" y2="{dy}" stroke="#334155" '
         'stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{dx}" y="{dy}" width="{dw}" height="{dh}" rx="10" '
         'fill="#f1f5f9" stroke="#64748b" stroke-width="2"/>')
L.append(f'<text x="{dx+dw/2}" y="{dy+24}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#334155">'
         f'{esc("op->hasAttr(discreteAttrName)?")}</text>')
L.append(f'<text x="{dx+dw/2}" y="{dy+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">'
         f'{esc("UnstructureConversionPass.cpp:L251")}</text>')

# discreteAttrName 命中(是)→ 向右,独立终止,不汇入兜底
dcy = dy + dh / 2
sx, sw, sh = dx + dw + 90, 300, 62
sy = dcy - sh / 2
L.append(f'<line x1="{dx+dw}" y1="{dcy}" x2="{sx}" y2="{dcy}" '
         'stroke="#64748b" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#s)"/>')
L.append(f'<text x="{dx+dw+16}" y="{dcy-10}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="#475569">{esc("是")}</text>')
L.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="10" '
         'fill="#f8fafc" stroke="#64748b" stroke-width="2" stroke-dasharray="6,3"/>')
L.append(f'<text x="{sx+sw/2}" y="{sy+24}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#334155">'
         f'{esc("return failure():已处理,跳过")}</text>')
L.append(f'<text x="{sx+sw/2}" y="{sy+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#475569">'
         f'{esc("本 pass 完全不碰该 op(不进入标量化兜底)")}</text>')

# discreteAttrName 未命中(否)→ 向下继续到早退放行门
L.append(f'<line x1="{dx+dw/2}" y1="{dy+dh}" x2="{dx+dw/2}" y2="{dy+dh+38}" stroke="#334155" '
         'stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{dx+dw/2+14}" y="{dy+dh+22}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="#334155">{esc("否")}</text>')

# 早退放行门
gx, gy, gw, gh = W/2 - 330, dy + dh + 38 + 36, 660, 74
L.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="10" '
         'fill="#e0f2fe" stroke="#0369a1" stroke-width="2"/>')
L.append(f'<text x="{gx+gw/2}" y="{gy+28}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#0c4a6e">'
         f'{esc("早退放行门")}</text>')
L.append(f'<text x="{gx+gw/2}" y="{gy+50}" text-anchor="middle" font-family="monospace" '
         f'font-size="11.5" fill="#0c4a6e">'
         f'{esc("isStructured() && !isDiscreteMask && (!isScalarLike || 形状全为 1)")}</text>')

# 真(绿)出口 —— 结构化路径
gy2 = gy + gh
gcx = gx + gw / 2
L.append(f'<line x1="{gx+gw}" y1="{gy+gh/2}" x2="{gx+gw+130}" y2="{gy+gh/2}" '
         'stroke="#16a34a" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<text x="{gx+gw+30}" y="{gy+gh/2-10}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="#166534">{esc("真")}</text>')
oex, oey, oew, oeh = gx + gw + 130, gy + gh/2 - 34, 300, 68
L.append(f'<rect x="{oex}" y="{oey}" width="{oew}" height="{oeh}" rx="10" '
         'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<text x="{oex+oew/2}" y="{oey+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#166534">{esc("return failure()")}</text>')
L.append(f'<text x="{oex+oew/2}" y="{oey+46}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#166534">'
         f'{esc("交给结构化路径(ch12/13:memref + extract_slice)")}</text>')

# 假(橙)→ 向下继续到三条触发解释
L.append(f'<line x1="{gcx}" y1="{gy2}" x2="{gcx}" y2="{gy2+58}" stroke="#d97706" '
         'stroke-width="2" marker-end="url(#o)"/>')
L.append(f'<text x="{gcx+14}" y="{gy2+34}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="#b45309">{esc("假")}</text>')

trig_top = gy2 + 58
tw, th, tgap = 340, 130, 40
trig_total_w = 3 * tw + 2 * tgap
tx0 = (W - trig_total_w) / 2
tx = [tx0, tx0 + tw + tgap, tx0 + 2 * (tw + tgap)]
TRIGGERS = [
    ("mayDiscretememaccess 逃生口",
     "op 带 mayDiscretememaccess 标注\n(checkUnstructureAnnotated 命中)\n→ 强制置 unstructured",
     "UnstructureConversionPass.cpp:L258"),
    ("触发组 A",
     "forceScalarizeMode || ptr.isScalarLike()\n|| fromTensorArg[ptr] → setUnstructured(rank)",
     "UnstructureConversionPass.cpp:L303-306"),
    ("触发组 B(对齐闸)",
     "结构化尾部连续字节数 % 32 != 0\n→ setUnstructured(rank)",
     "UnstructureConversionPass.cpp:L334-342(常量32在L341)"),
]
for i, (name, body, prov) in enumerate(TRIGGERS):
    x = tx[i]
    L.append(f'<rect x="{x}" y="{trig_top}" width="{tw}" height="{th}" rx="10" '
              'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
    L.append(f'<text x="{x+tw/2}" y="{trig_top+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#92400e">{esc(name)}</text>')
    for li, line in enumerate(body.split("\n")):
        L.append(f'<text x="{x+tw/2}" y="{trig_top+44+li*17}" text-anchor="middle" '
                  f'font-family="monospace" font-size="10.5" fill="#78350f">{esc(line)}</text>')
    L.append(f'<text x="{x+tw/2}" y="{trig_top+th-10}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9.5" fill="#a16207">{esc(prov)}</text>')

# 从假箭头分叉到三个 trigger 顶部
branch_y = trig_top - 20
for i in range(3):
    cx = tx[i] + tw / 2
    L.append(f'<line x1="{gcx}" y1="{gy2+58}" x2="{cx}" y2="{branch_y}" '
              'stroke="#d97706" stroke-width="1.5" stroke-dasharray="4,3"/>')
    L.append(f'<line x1="{cx}" y1="{branch_y}" x2="{cx}" y2="{trig_top}" '
              'stroke="#d97706" stroke-width="1.5" marker-end="url(#o)"/>')

# 三条汇入 Unstructure 出口(discreteAttrName 的「已跳过」终止支路不参与汇合)
merge_y = trig_top + th + 60
mcx = W / 2
for i in range(3):
    cx = tx[i] + tw / 2
    y0 = trig_top + th
    L.append(f'<line x1="{cx}" y1="{y0}" x2="{mcx}" y2="{merge_y-20}" '
              'stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<line x1="{mcx}" y1="{merge_y-20}" x2="{mcx}" y2="{merge_y}" '
         'stroke="#d97706" stroke-width="2" marker-end="url(#o)"/>')

fex, fey, few, feh = mcx - 220, merge_y, 440, 74
L.append(f'<rect x="{fex}" y="{fey}" width="{few}" height="{feh}" rx="10" '
         'fill="#fed7aa" stroke="#c2410c" stroke-width="2"/>')
L.append(f'<text x="{fex+few/2}" y="{fey+28}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#9a3412">'
         f'{esc("Unstructure 标量化兜底")}</text>')
L.append(f'<text x="{fex+few/2}" y="{fey+50}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#9a3412">'
         f'{esc("触发维建 scf.for,逐元素/逐行访存(见 §14.6/§14.7)")}</text>')

foot_y = fey + feh + 46
L.append(f'<text x="{W/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#334155">'
         f'{esc("掉进兜底的常见写法:①load 出来的值当索引(gather/scatter) ②ptr 本身 scalarLike")}</text>')
L.append(f'<text x="{W/2}" y="{foot_y+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#334155">'
         f'{esc("③offset 溯源到形状不可知的 tensor 入参 ④结构化尾部不是 32 字节倍数(昇腾搬运粒度)")}</text>')
L.append(f'<text x="{W/2}" y="{foot_y+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">'
         f'{esc("discreteAttrName(已处理,直接跳过)与 mayDiscretememaccess(强制标量化)是两个相反的检查点,别混为一谈")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-m5-triggers.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
