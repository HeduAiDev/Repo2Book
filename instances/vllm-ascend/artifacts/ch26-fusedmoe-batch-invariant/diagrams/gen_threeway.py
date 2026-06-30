#!/usr/bin/env python3
"""fig26-2: MoE 通信三选一注册表 —— 枚举 → *CommImpl → (dispatcher, prepare_finalize) 配对。f10 回收。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1360, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, color="#94a3b8", sw=1.8, arrow=False):
    a = ' marker-end="url(#ar)"' if arrow else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{a}/>')


text(W / 2, 42, "MoE 通信方式三选一：从枚举到真正的算子配对", 23, "middle", "#0f172a", "bold")
text(W / 2, 70, "ch15 select_moe_comm_method 选枚举 → setup_moe_comm_method 建实例 → 每种配一对 dispatcher + prepare_finalize",
     13.5, "middle", "#64748b")

# 顶部：ch15 选择条
box(80, 96, W - 160, 40, "#eef2ff", "#6366f1", 10, 1.6)
text(W / 2, 121, "ch15 · select_moe_comm_method(num_tokens, soc, EP) ⇒ 返回一个 MoECommType",
     13.5, "middle", "#4338ca", "bold")

rows = [
    ("MoECommType.ALLGATHER", "AllGatherCommImpl", "TokenDispatcherWithAllGather", "PrepareAndFinalizeWithAllGather", "#10b981"),
    ("MoECommType.MC2", "MC2CommImpl", "TokenDispatcherWithMC2", "PrepareAndFinalizeWithMC2", "#0ea5e9"),
    ("MoECommType.ALLTOALL", "AlltoAllCommImpl", "TokenDispatcherWithAll2AllV", "PrepareAndFinalizeWithAll2All", "#f59e0b"),
    ("MoECommType.FUSED_MC2", "FusedMC2CommImpl", "TokenDispatcherWithMC2", "PrepareAndFinalizeWithMC2", "#ef4444"),
]

c1x, c1w = 80, 250
c2x, c2w = 410, 230
c3x, c3w = 720, 540
y0, rh, gap = 188, 88, 24

text(c1x + c1w / 2, y0 - 22, "枚举值", 14, "middle", "#334155", "bold")
text(c2x + c2w / 2, y0 - 22, "*CommImpl 实例", 14, "middle", "#334155", "bold")
text(c3x + c3w / 2, y0 - 22, "(TokenDispatcher* , PrepareAndFinalize*) 配对", 14, "middle", "#334155", "bold")

for i, (enum, impl, disp, pf, color) in enumerate(rows):
    y = y0 + i * (rh + gap)
    cy = y + rh / 2
    box(c1x, y, c1w, rh, "#f8fafc", color, 10, 2)
    text(c1x + c1w / 2, cy + 5, enum, 12.5, "middle", color, "bold", mono=True)
    box(c2x, y, c2w, rh, "white", color, 10, 2)
    text(c2x + c2w / 2, cy + 5, impl, 13, "middle", "#1e293b", "bold", mono=True)
    # pair box
    box(c3x, y, c3w, rh, "#fafafa", color, 10, 1.6)
    text(c3x + 18, cy - 8, "dispatcher: " + disp, 12, "start", "#334155", mono=True)
    text(c3x + 18, cy + 16, "prep/final: " + pf, 12, "start", "#334155", mono=True)
    line(c1x + c1w, cy, c2x, cy, color, 2, arrow=True)
    line(c2x + c2w, cy, c3x, cy, color, 2, arrow=True)

# FUSED_MC2 注记
text(c3x + c3w / 2, y0 + 4 * (rh + gap) - 6,
     "FUSED_MC2 额外覆写 fused_experts：dispatch+ffn+combine 融成单个 C++ 算子",
     12, "middle", "#b91c1c")

L.append('</svg>')
open("fig26-2-threeway.svg", "w").write('\n'.join(L))
print("wrote fig26-2-threeway.svg")
