#!/usr/bin/env python3
"""fig-blocked-to-mma-rewrite (before-after 模板)
BlockedToMMA 重写前后:tt.dot 输出编码 blocked -> NvidiaMmaEncodingAttr,
补 4 个 convert(acc/A/B/result),其中 result convert 是下一步 RemoveLayoutConversions 要消的对象。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PAD, TOP = 50, 132
w = 1560
h = 620

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="o" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="44" font-family="sans-serif" font-size="18.5" '
          f'font-weight="bold" fill="#0f172a">{esc("BlockedToMMA:换编码的同一步,引入了下一步要消的 convert")}</text>')
L.append(f'<text x="{PAD}" y="68" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("同构双面板,仅右侧新增节点高亮——tt.dot 的输出编码从 blocked 换成 NvidiaMmaEncodingAttr(AccelerateMatmul.cpp:L233-L349)")}</text>')

# ---- 左面板:重写前 ----
LEFT_X = PAD
LEFT_W = 300
L.append(f'<text x="{LEFT_X+LEFT_W/2}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14.5" font-weight="bold" fill="#0f172a">{esc("重写前")}</text>')

box_w, box_h = LEFT_W, 60
by = TOP + 30
L.append(f'<rect x="{LEFT_X}" y="{by}" width="{box_w}" height="{box_h}" rx="8" '
          'fill="#e2e8f0" stroke="#64748b" stroke-width="1.6"/>')
L.append(f'<text x="{LEFT_X+box_w/2}" y="{by+box_h/2-6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
          f'fill="#0f172a">{esc("tt.dot(A, B, acc)")}</text>')
L.append(f'<text x="{LEFT_X+box_w/2}" y="{by+box_h/2+14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("输出编码 = #blocked")}</text>')

L.append(f'<text x="{LEFT_X}" y="{by+box_h+60}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("A / B / acc 均沿用 blocked 编码,")}</text>')
L.append(f'<text x="{LEFT_X}" y="{by+box_h+80}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("不接触 Tensor Core。")}</text>')

# ---- 右面板:重写后 ----
RIGHT_X = LEFT_X + LEFT_W + 130
RIGHT_W = 560
L.append(f'<text x="{RIGHT_X+RIGHT_W/2}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14.5" font-weight="bold" fill="#0f172a">{esc("重写后 (versionMajor 已选定)")}</text>')

# 中间对比箭头
mid_y = by + box_h / 2
L.append(f'<line x1="{LEFT_X+box_w+14}" y1="{mid_y}" x2="{RIGHT_X-70}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="2.4" marker-end="url(#o)"/>')
L.append(f'<text x="{(LEFT_X+box_w+RIGHT_X-70)/2}" y="{mid_y-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#92400e">{esc("BlockedToMMA")}</text>')

# 右面板:3 个操作数输入(acc/A/B) 各配一个 convert,汇入 dot,再输出 convert
in_w, in_h = 150, 44
gap_y = 26
in_x = RIGHT_X
labels_in = [("acc(blocked)", "acc -> mma"), ("A(blocked)", "A -> dot_operand"), ("B(blocked)", "B -> dot_operand")]
conv_x = RIGHT_X + in_w + 60
dot_x = conv_x + in_w + 60
dot_w, dot_h = 150, 3 * in_h + 2 * gap_y
dot_y0 = by

for i, (src_label, conv_label) in enumerate(labels_in):
    yy = by + i * (in_h + gap_y)
    L.append(f'<rect x="{in_x}" y="{yy}" width="{in_w}" height="{in_h}" rx="6" '
              'fill="#e2e8f0" stroke="#64748b" stroke-width="1.4"/>')
    L.append(f'<text x="{in_x+in_w/2}" y="{yy+in_h/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#0f172a">{esc(src_label)}</text>')
    # convert 框
    L.append(f'<rect x="{conv_x}" y="{yy}" width="{in_w}" height="{in_h}" rx="6" '
              'fill="#fef3c7" stroke="#d97706" stroke-width="1.8"/>')
    L.append(f'<text x="{conv_x+in_w/2}" y="{yy+in_h/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#78350f">{esc("convert "+conv_label)}</text>')
    L.append(f'<line x1="{in_x+in_w}" y1="{yy+in_h/2}" x2="{conv_x-4}" y2="{yy+in_h/2}" '
              'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
    # convert -> dot(汇聚到 dot 框左边中点对应高度)
    target_y = dot_y0 + dot_h / 2
    L.append(f'<path d="M {conv_x+in_w} {yy+in_h/2} L {dot_x-16} {yy+in_h/2} L {dot_x-4} {target_y if i==1 else yy+in_h/2}" '
              f'fill="none" stroke="#64748b" stroke-width="1.4"/>')

# dot 框(纵向跨 3 行,居中对齐)
L.append(f'<rect x="{dot_x}" y="{dot_y0}" width="{dot_w}" height="{dot_h}" rx="8" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{dot_x+dot_w/2}" y="{dot_y0+dot_h/2-6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#1e3a5f">{esc("tt.dot")}</text>')
L.append(f'<text x="{dot_x+dot_w/2}" y="{dot_y0+dot_h/2+14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" '
          f'fill="#1e40af">{esc("输出 = mma")}</text>')

# 补齐 3 条箭头连到 dot 左边缘(覆盖前面简化路径的箭头终点)
for i in range(3):
    yy = by + i * (in_h + gap_y) + in_h / 2
    L.append(f'<line x1="{dot_x-20}" y1="{yy}" x2="{dot_x-2}" y2="{dot_y0+dot_h/2}" '
              'stroke="none"/>')

# dot -> 结果 convert -> store
res_conv_x = dot_x + dot_w + 60
res_conv_y = dot_y0 + dot_h / 2 - in_h / 2
L.append(f'<line x1="{dot_x+dot_w}" y1="{dot_y0+dot_h/2}" x2="{res_conv_x-4}" y2="{dot_y0+dot_h/2}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<rect x="{res_conv_x}" y="{res_conv_y}" width="{in_w+20}" height="{in_h}" rx="6" '
          'fill="#fee2e2" stroke="#dc2626" stroke-width="2.2"/>')
L.append(f'<text x="{res_conv_x+(in_w+20)/2}" y="{res_conv_y+in_h/2-1}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
          f'fill="#7f1d1d">{esc("convert result")}</text>')
L.append(f'<text x="{res_conv_x+(in_w+20)/2}" y="{res_conv_y+in_h/2+13}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" '
          f'fill="#7f1d1d">{esc("mma -> blocked")}</text>')

callout_y = res_conv_y + in_h + 30
L.append(f'<rect x="{res_conv_x-40}" y="{callout_y}" width="{in_w+100}" height="70" rx="8" '
          'fill="#fef2f2" stroke="#fca5a5"/>')
L.append(f'<text x="{res_conv_x+(in_w+20)/2}" y="{callout_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.8" fill="#7f1d1d">{esc("replaceOpWithNewOp<ConvertLayoutOp>")}</text>')
L.append(f'<text x="{res_conv_x+(in_w+20)/2}" y="{callout_y+40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.8" fill="#7f1d1d">{esc("(AccelerateMatmul.cpp:L346-L348)")}</text>')
L.append(f'<text x="{res_conv_x+(in_w+20)/2}" y="{callout_y+58}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.8" font-weight="bold" fill="#b91c1c">{esc("下一步被消除的对象")}</text>')

# 底部统计条
foot_y = by + dot_h + 130
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="66" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+20}" y="{foot_y+26}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("新增 convert 数 = 4:acc->mma、A->dot_operand、B->dot_operand、result mma->blocked")}</text>')
L.append(f'<text x="{PAD+20}" y="{foot_y+48}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc("『造 mma 编码』与『引入 convert』是同一步——引入的 convert 由下游 RemoveLayoutConversions / OptimizeDotOperands 收拾。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-blocked-to-mma-rewrite.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
