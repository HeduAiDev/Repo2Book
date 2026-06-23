#!/usr/bin/env python3
"""四步 code→diagram 程序总览图（ch26 方法论主图）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 540
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="arrback" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# 标题
L.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-size="24" font-weight="bold" fill="#0f172a">从模型代码到架构图：四步可复用程序</text>')

stages = [
    ("P1", "读 __init__", "出模块树（嵌套框）", "self.&lt;name&gt; = Module(...)", "#1d4ed8", "#dbeafe"),
    ("P2", "读 forward", "出数据流（有向边）", "x = self.&lt;child&gt;(x)", "#047857", "#d1fae5"),
    ("P3", "标子系统边界", "算子 / 编译区 / 并行", "torch.ops.vllm.* · @torch.compile · *ParallelLinear", "#b45309", "#fef3c7"),
    ("P4", "用 svg-diagram", "渲染成图", "Python → xmllint → PNG", "#7c3aed", "#ede9fe"),
]

n = len(stages)
box_w, box_h = 230, 150
gap = (W - 80 - n * box_w) / (n - 1)
y0 = 90
cy = y0 + box_h / 2
xs_list = []
for i, (tag, title, sub, crit, stroke, fill) in enumerate(stages):
    x = 40 + i * (box_w + gap)
    xs_list.append((x, x + box_w))
    cx = x + box_w / 2
    L.append(f'<rect x="{x}" y="{y0}" width="{box_w}" height="{box_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
    L.append(f'<text x="{x+20}" y="{y0+34}" font-size="22" font-weight="bold" fill="{stroke}">{tag}</text>')
    L.append(f'<text x="{cx+18}" y="{y0+34}" text-anchor="middle" font-size="19" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<text x="{cx}" y="{y0+62}" text-anchor="middle" font-size="15" fill="#334155">{esc(sub)}</text>')
    # 源码判据小注
    L.append(f'<line x1="{x+20}" y1="{y0+80}" x2="{x+box_w-20}" y2="{y0+80}" stroke="{stroke}" stroke-width="1" stroke-dasharray="3,3"/>')
    L.append(f'<text x="{cx}" y="{y0+102}" text-anchor="middle" font-size="11.5" fill="#64748b" font-weight="bold">源码判据</text>')
    # 判据可能较长，按需断行
    words = crit
    if len(words) > 24:
        # 简单两行切
        mid = len(words) // 2
        # 在空格/点附近切
        cut = words.rfind(' ', 0, mid + 6)
        if cut < 0:
            cut = mid
        line1, line2 = words[:cut], words[cut:].lstrip()
        L.append(f'<text x="{cx}" y="{y0+122}" text-anchor="middle" font-size="12" fill="#475569" font-family="monospace">{line1}</text>')
        L.append(f'<text x="{cx}" y="{y0+139}" text-anchor="middle" font-size="12" fill="#475569" font-family="monospace">{line2}</text>')
    else:
        L.append(f'<text x="{cx}" y="{y0+128}" text-anchor="middle" font-size="12.5" fill="#475569" font-family="monospace">{words}</text>')

# 阶段间前向箭头
for i in range(n - 1):
    x1 = xs_list[i][1]
    x2 = xs_list[i + 1][0]
    L.append(f'<line x1="{x1+4}" y1="{cy}" x2="{x2-6}" y2="{cy}" stroke="#475569" stroke-width="3" marker-end="url(#arr)"/>')

# 底部验证回边
vy = y0 + box_h + 70
L.append(f'<rect x="40" y="{vy-30}" width="{W-80}" height="62" rx="10" fill="#fef2f2" stroke="#dc2626" stroke-width="2" stroke-dasharray="7,4"/>')
L.append(f'<text x="{W/2}" y="{vy-6}" text-anchor="middle" font-size="16" font-weight="bold" fill="#b91c1c">验证（回边）：每个图元逐一溯源回源码</text>')
L.append(f'<text x="{W/2}" y="{vy+18}" text-anchor="middle" font-size="13.5" fill="#7f1d1d">每个框 ⟶ 一条 self.X=Module ｜ 每条边 ⟶ 一句 forward 赋值 ｜ 每种着色 ⟶ 一个装饰器/类名/torch.ops。溯源不到的图元 = 杜撰，删。</text>')

# 从 P4 回到验证、再回到 P1 的回边（弯折线）
midx = (xs_list[-1][0] + xs_list[-1][1]) / 2
L.append(f'<path d="M {midx} {y0+box_h} L {midx} {vy-32}" fill="none" stroke="#b45309" stroke-width="2.5" marker-end="url(#arrback)"/>')
midx0 = (xs_list[0][0] + xs_list[0][1]) / 2
L.append(f'<path d="M 40 {vy} L {midx0} {vy} L {midx0} {y0+box_h+4}" fill="none" stroke="#b45309" stroke-width="2.5" marker-end="url(#arrback)"/>')

L.append('</svg>')
svg = '\n'.join(L)
out = __import__('pathlib').Path(__file__).parent / "procedure-overview.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}")
