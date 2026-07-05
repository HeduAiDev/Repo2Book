#!/usr/bin/env python3
"""fig28-3: 异步指数随机 —— 两条 stream 泳道，把 RNG 延迟藏进模型前向，临界路径少一段。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 520
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(
    '<defs>'
    '<marker id="sy" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
    '<path d="M0,0 L10,5 L0,10 Z" fill="#7c3aed"/></marker>'
    '</defs>'
)


def box(x, y, w, h, fill, stroke, rx=8, sw=1.6):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


text(W / 2, 40, "异步指数随机：让 q.exponential_() 与模型计算在两条 stream 上重叠", 22, "middle", "#0f172a", "bold")

# 串行（上）
text(70, 86, "同步串行（概念基线）", 15, "start", "#475569", "bold")
box(70, 100, 360, 48, "#eef2ff", "#6366f1")
text(250, 130, "模型前向 → logits → softmax", 13.5, "middle", "#3730a3")
box(440, 100, 240, 48, "#fef9c3", "#ca8a04")
text(560, 130, "q.exponential_()", 13.5, "middle", "#854d0e", mono=True)
box(690, 100, 200, 48, "#dcfce7", "#16a34a")
text(790, 130, "div_ / argmax", 13.5, "middle", "#166534", mono=True)
text(910, 130, "临界路径 = 三段相加", 13, "start", "#b91c1c")

# 异步（下）：两泳道
ly = 220
text(70, ly - 14, "异步重叠（昇腾）", 15, "start", "#15803d", "bold")
# 主 stream
box(70, ly, 60, 130, "#f1f5f9", "#94a3b8")
text(100, ly + 70, "主", 14, "middle", "#475569", "bold")
text(100, ly + 90, "stream", 11, "middle", "#475569")
box(150, ly + 10, 420, 50, "#eef2ff", "#6366f1")
text(360, ly + 40, "模型前向 → logits → softmax", 13.5, "middle", "#3730a3")
box(600, ly + 10, 220, 50, "#dcfce7", "#16a34a")
text(710, ly + 40, "probs.div_(q).argmax", 12.5, "middle", "#166534", mono=True)

# side stream
box(70, ly + 80, 60, 50, "#f1f5f9", "#94a3b8")
text(100, ly + 102, "global", 10.5, "middle", "#475569")
text(100, ly + 118, "stream", 10.5, "middle", "#475569")
box(150, ly + 80, 300, 50, "#fef9c3", "#ca8a04")
text(300, ly + 110, "q.exponential_()  并行发起", 13, "middle", "#854d0e", mono=True)

# 汇流箭头：side → 主（在 div_ 前）
L.append(f'<line x1="450" y1="{ly + 105}" x2="600" y2="{ly + 50}" stroke="#7c3aed" stroke-width="2.2" stroke-dasharray="5 4" marker-end="url(#sy)"/>')
text(470, ly + 150, "wait_stream / Event.synchronize：div_ 前汇流一次", 12.5, "start", "#7c3aed")

text(840, ly + 40, "临界路径 ≈ 模型前向 + div_/argmax", 13, "start", "#15803d", "bold")
text(840, ly + 62, "（指数随机的延迟被藏进前向，少一段）", 12.5, "start", "#166534")

text(W / 2, 470, "RNG 与本步 logits→probs 无数据依赖 → 可并行；只在最终 div_ 前同步一次。默认走 random_sample 内的两行包裹，"
                 "enable_async_exponential 更进一步用 Event 预算。",
     12.5, "middle", "#475569")

L.append('</svg>')
open("fig28-3-async.svg", "w").write('\n'.join(L))
print("wrote fig28-3-async.svg")
