#!/usr/bin/env python3
"""TopKTopPSampler 后端分发：init-time 绑 self.forward + call-time 二级分流。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

W, H = 1120, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def box(x, y, w, h, fill, stroke, lines, fs=14, tw="bold"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    cy = y + h/2 - (n-1)*0.5*(fs+3)
    for i, t in enumerate(lines):
        fw = 'bold' if (tw == 'bold' and i == 0) else 'normal'
        L.append(f'<text x="{x+w/2}" y="{cy + i*(fs+3) + fs*0.35:.0f}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" fill="#1e293b">{esc(t)}</text>')

def arrow(x1, y1, x2, y2, label=None):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        L.append(f'<rect x="{mx-26}" y="{my-12}" width="52" height="18" rx="4" fill="white" opacity="0.9"/>')
        L.append(f'<text x="{mx}" y="{my+1}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#b45309">{esc(label)}</text>')

# 标题分隔
L.append(f'<text x="280" y="26" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#334155">① __init__ 时绑定 self.forward（一次性）</text>')
L.append(f'<text x="850" y="26" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#334155">② 每次 call 时的二级分流</text>')
L.append(f'<line x1="560" y1="40" x2="560" y2="{H-20}" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6 4"/>')

# 左半：决策树
root = box(170, 50, 220, 50, "#e0e7ff", "#6366f1", ["__init__", "(device / platform / logprobs_mode)"], fs=14)
# 决策节点
def diamond_label(x, y, txt):
    box(x, y, 240, 44, "#fef9c3", "#ca8a04", [txt], fs=13, tw="normal")

dy = 130
nodes = [
    (40, dy,   "#dcfce7", "#16a34a", ["forward_cuda", "CUDA + flashinfer 可用"], "CUDA+FI"),
    (40, dy+78,"#dbeafe", "#2563eb", ["forward_native", "硬件不支持 FI → 静默回退"], "回退"),
    (40, dy+156,"#dbeafe","#2563eb", ["forward_native", "显式关闭 flashinfer"], "关闭"),
    (40, dy+234,"#fae8ff","#a21caf", ["forward_cpu", "CPU（x86 等）"], "CPU"),
    (40, dy+312,"#dbeafe","#2563eb", ["forward_native", "CPU RISCV/POWERPC"], "CPU特例"),
    (40, dy+390,"#ffedd5","#ea580c", ["forward_hip", "ROCm + aiter 启用"], "ROCm"),
    (40, dy+468,"#dbeafe","#2563eb", ["forward_native", "其余一切（含 host 无 GPU）"], "else"),
]
for (x, y, fill, st, lines, lab) in nodes:
    box(x, y, 260, 56, fill, st, lines, fs=13)
    arrow(390, 100, x+260, y+28 if x+260 < 390 else y+28, lab) if False else None
    # arrow from root bottom to each node left edge area
    arrow(280, 100, 50, y+28, lab) if False else None

# simpler radiating arrows from root bottom
for (x, y, fill, st, lines, lab) in nodes:
    L.append(f'<line x1="200" y1="100" x2="{x+260}" y2="{y+28}" stroke="#94a3b8" stroke-width="1.4" marker-end="url(#a)"/>')
    L.append(f'<rect x="{x+262}" y="{y+18}" width="0" height="0"/>')
    L.append(f'<text x="{x+200}" y="{y+12}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#b45309">{esc(lab)}</text>')

# 右半：call 时分流
cxx = 620
box(cxx, 60, 400, 46, "#e0e7ff", "#6366f1", ["TopKTopPSampler(...) 调用 = self.forward"], fs=14)

# 路 A: forward_cuda
box(cxx, 150, 400, 50, "#dcfce7", "#16a34a", ["forward_cuda", "无 k/p 或有 generator → 回退 native"], fs=13)
arrow(cxx+200, 106, cxx+200, 150)
box(cxx+40, 240, 320, 44, "#ecfccb", "#65a30d", ["flashinfer_sample（拒绝采样，不排序）"], fs=13, tw="normal")
arrow(cxx+200, 200, cxx+200, 240, "否则")

# 路 B: forward_native → apply_top_k_top_p 分流
box(cxx, 340, 400, 50, "#dbeafe", "#2563eb", ["forward_native → apply_top_k_top_p", "按 batch 规模二级分流"], fs=13)
arrow(cxx+200, 106, cxx+200, 340) if False else None
L.append(f'<line x1="{cxx+200}" y1="106" x2="{cxx+200}" y2="340" stroke="#475569" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#a)"/>')

box(cxx-30, 450, 250, 60, "#fee2e2", "#dc2626",
    ["apply_top_k_top_p_triton", "batch>=8 且有 Triton", "Qrita pivot-truncation（不排序整 vocab）"], fs=12)
box(cxx+260, 450, 250, 60, "#dbeafe", "#2563eb",
    ["apply_top_k_top_p_pytorch", "小 batch / 无 Triton", "sort + mask + scatter（教学主路径）"], fs=12)
arrow(cxx+120, 390, cxx+95, 450, "batch>=8")
arrow(cxx+280, 390, cxx+385, 450, "否则")

box(cxx+90, 560, 260, 46, "#ede9fe", "#7c3aed", ["random_sample（Gumbel/exp，无同步）"], fs=13, tw="normal")
arrow(cxx+95, 510, cxx+200, 560)
arrow(cxx+385, 510, cxx+250, 560)

L.append('</svg>')
open("02-backend-dispatch.svg", "w", encoding="utf-8").write('\n'.join(L))
print("ok")
