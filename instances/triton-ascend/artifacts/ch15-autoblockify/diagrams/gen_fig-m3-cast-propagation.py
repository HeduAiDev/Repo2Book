#!/usr/bin/env python3
"""flow 模板:m3 UnrealizedConversionCast 作『类型防火墙』沿 def-use 逐 op 下推。
顶部载体框 -> 水平分派干线 -> 3 条代表分支(20 路分派中的示例)-> 水平汇合干线 -> eraseOp 框。
全坐标计算,零魔数;fan-out/merge 走「竖-横-竖」折线,避免同点发散线退化成水平线。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TOP_BOX = ("UnrealizedConversionCastOp (2 inputs: blockifiedId, mask)",
           "matchAndRewrite 检查 inputs.size()==2 (L51-52),遍历全部 users")
DISPATCH_LABEL = "覆盖 20 类具名 Triton/MLIR op (白名单,非穷举,L67-L117)"
BRANCHES = [
    ("user = arith.addi (Elementwise trait)", "rewriteGeneraleOp 批处理化",
     ["重新包 1 层 cast,", "继续下推"], "#e2e8f0", "#64748b"),
    ("user = tt.load", "rewriteLoad 批处理化",
     ["重新包 1 层 cast,", "继续下推"], "#e2e8f0", "#64748b"),
    ("user = 另一 cast", "cast↔cast(1 类特例)",
     ["replaceOp(user,input)", "直接消解,无需重包 (L70-82)"], "#fef3c7", "#b45309"),
]
BOTTOM_BOX = ("载体最终动作", "全部 user 处理完 → rewriter.eraseOp(op) (L131)")

BOX_W, BOX_H = 780, 62
BR_W, BR_H = 250, 104
PAD, TOP = 40, 78
GAP1, GAP2 = 76, 76
cx = PAD + BOX_W / 2
w = PAD * 2 + BOX_W
top_y = TOP
br_y = top_y + BOX_H + GAP1
bot_y = br_y + BR_H + GAP2
h = bot_y + BOX_H + 60

br_gap = 40
br_total_w = len(BRANCHES) * BR_W + (len(BRANCHES) - 1) * br_gap
br_x0 = cx - br_total_w / 2
br_cx = [br_x0 + i * (BR_W + br_gap) + BR_W / 2 for i in range(len(BRANCHES))]

fan_y = top_y + BOX_H + GAP1 * 0.42       # horizontal distribution trunk (top -> branches)
merge_y = br_y + BR_H + GAP2 * 0.58       # horizontal collection trunk (branches -> bottom)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-18}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("cast 作载体、逐 op 下推(PropagateUnrealizedCastDown)")}</text>']

# top box
L.append(f'<rect x="{PAD}" y="{top_y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
          f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{cx}" y="{top_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(TOP_BOX[0])}</text>')
L.append(f'<text x="{cx}" y="{top_y+45}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">{esc(TOP_BOX[1])}</text>')

# label above the fan trunk (kept clear of the horizontal line itself)
L.append(f'<text x="{cx}" y="{fan_y-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#475569">{esc(DISPATCH_LABEL)}</text>')
# trunk: top box bottom -> fan_y (vertical), then horizontal across branch centers
L.append(f'<line x1="{cx}" y1="{top_y+BOX_H}" x2="{cx}" y2="{fan_y}" '
          'stroke="#64748b" stroke-width="1.6"/>')
L.append(f'<line x1="{br_cx[0]}" y1="{fan_y}" x2="{br_cx[-1]}" y2="{fan_y}" '
          'stroke="#64748b" stroke-width="1.6"/>')
for bx in br_cx:
    L.append(f'<line x1="{bx}" y1="{fan_y}" x2="{bx}" y2="{br_y-3}" '
              'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')

for i, (name, mid, actions, fill, stroke) in enumerate(BRANCHES):
    x = br_cx[i] - BR_W / 2
    L.append(f'<rect x="{x}" y="{br_y}" width="{BR_W}" height="{BR_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{br_cx[i]}" y="{br_y+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{br_cx[i]}" y="{br_y+42}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(mid)}</text>')
    for k, line in enumerate(actions):
        L.append(f'<text x="{br_cx[i]}" y="{br_y+62+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.6" fill="#475569">{esc(line)}</text>')

# merge trunk: branch bottoms -> merge_y horizontal -> down into bottom box
for bx in br_cx:
    L.append(f'<line x1="{bx}" y1="{br_y+BR_H}" x2="{bx}" y2="{merge_y}" '
              'stroke="#64748b" stroke-width="1.4"/>')
L.append(f'<line x1="{br_cx[0]}" y1="{merge_y}" x2="{br_cx[-1]}" y2="{merge_y}" '
          'stroke="#64748b" stroke-width="1.4"/>')
L.append(f'<line x1="{cx}" y1="{merge_y}" x2="{cx}" y2="{bot_y-3}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{cx}" y="{merge_y+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#475569">'
          f'{esc("非 cast↔cast 分支:处理完当前 user 后重新包 cast 送回上一步")}</text>')

L.append(f'<rect x="{PAD}" y="{bot_y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
          f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{cx}" y="{bot_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(BOTTOM_BOX[0])}</text>')
L.append(f'<text x="{cx}" y="{bot_y+45}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">{esc(BOTTOM_BOX[1])}</text>')

L.append(f'<text x="{w/2}" y="{h-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#64748b">'
          f'{esc("本图仅画 3 条代表分支:能批处理的走对应 rewrite 重新包 cast 继续下推,cast↔cast 特例直接消解;塞进 blockify 循环的分支见后文配图")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-m3-cast-propagation.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f"wrote {out}")
