#!/usr/bin/env python3
"""ch22 profiling chunk 二次延迟模型：解 f(L+x)-f(L)=T 求增量 chunk x。"""
import math
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1180, 640
TXT = "#1e293b"
SUB = "#64748b"
CURVE = "#2563eb"
AREA = "#7c3aed"

# 坐标系
ox, oy = 130, 500   # 原点
axw, axh = 760, 400

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 12 10" refX="11" refY="5" markerWidth="11" markerHeight="9" orient="auto"><path d="M0,0 L12,5 L0,10 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="{TXT}">profiling chunk：用二次延迟模型反解「再算多少 token 恰好花掉目标延迟」</text>')

# 坐标轴
L.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+axw}" y2="{oy}" stroke="#475569" stroke-width="1.8" marker-end="url(#ar)"/>')
L.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-axh}" stroke="#475569" stroke-width="1.8" marker-end="url(#ar)"/>')
L.append(f'<text x="{ox+axw}" y="{oy+30}" text-anchor="middle" font-size="14" fill="{TXT}">序列长 l</text>')
L.append(f'<text x="{ox-12}" y="{oy-axh-8}" text-anchor="end" font-size="14" fill="{TXT}">单步延迟 f(l)</text>')

# 曲线 f(l)=a l^2 + b l + c
a, b, c = 0.00055, 0.05, 8.0
def fx(l): return a*l*l + b*l + c
lmax = 760
def sx(l): return ox + l/lmax*axw
def sy(v):
    vmax = fx(lmax)
    return oy - v/vmax*(axh-30)
pts = " ".join(f"{sx(l):.1f},{sy(fx(l)):.1f}" for l in range(0, lmax+1, 20))
L.append(f'<polyline points="{pts}" fill="none" stroke="{CURVE}" stroke-width="2.6"/>')
L.append(f'<text x="{sx(560)}" y="{sy(fx(560))-14}" font-size="14.5" font-weight="bold" fill="{CURVE}">f(l)=a·l²+b·l+c</text>')

# L 与 L+x
Lc = 240
xchunk = 300
Lx = Lc + xchunk
for val, lab, col in [(Lc, "L（已算）", "#0f766e"), (Lx, "L+x", "#b45309")]:
    L.append(f'<line x1="{sx(val)}" y1="{oy}" x2="{sx(val)}" y2="{sy(fx(val))}" stroke="{col}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    L.append(f'<line x1="{ox}" y1="{sy(fx(val))}" x2="{sx(val)}" y2="{sy(fx(val))}" stroke="{col}" stroke-width="1.2" stroke-dasharray="4 4"/>')
    L.append(f'<text x="{sx(val)}" y="{oy+24}" text-anchor="middle" font-size="13.5" font-weight="bold" fill="{col}">{esc(lab)}</text>')

# 阴影：纵向增量 = f(L+x)-f(L) = target_latency
L.append(f'<rect x="{ox-2}" y="{sy(fx(Lx))}" width="6" height="{sy(fx(Lc))-sy(fx(Lx))}" fill="{AREA}"/>')
midy = (sy(fx(Lc))+sy(fx(Lx)))/2
L.append(f'<line x1="{ox+4}" y1="{midy}" x2="{ox+70}" y2="{midy}" stroke="{AREA}" stroke-width="1.4"/>')
L.append(f'<text x="{ox+78}" y="{midy-6}" text-anchor="start" font-size="13.5" font-weight="bold" fill="{AREA}">增量延迟</text>')
L.append(f'<text x="{ox+78}" y="{midy+14}" text-anchor="start" font-size="13.5" font-weight="bold" fill="{AREA}">= target_latency T</text>')

# x 区间标注
L.append(f'<line x1="{sx(Lc)}" y1="{oy+44}" x2="{sx(Lx)}" y2="{oy+44}" stroke="#b45309" stroke-width="1.4" marker-end="url(#ar)"/>')
L.append(f'<text x="{sx((Lc+Lx)/2)}" y="{oy+62}" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#b45309">x = 本次 chunk size</text>')

# 公式块（右上角）
bx, by = 950, 130
L.append(f'<rect x="{bx-2}" y="{by}" width="220" height="220" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4"/>')
L.append(f'<text x="{bx+108}" y="{by+30}" text-anchor="middle" font-size="14" font-weight="bold" fill="{TXT}">反解 x</text>')
for i, t in enumerate([
    "f(L+x)−f(L)=T",
    "⟹ a·x²+(2aL+b)·x−T=0",
    "A=a, B=2aL+b, C=−T",
    "x=(−B+√(B²−4AC))/(2A)",
    "取正根 → 平滑",
    "→ 对齐 max(page,64)",
]):
    L.append(f'<text x="{bx+12}" y="{by+58+i*27}" font-size="12.5" fill="{TXT}">{esc(t)}</text>')

L.append(f'<text x="{W/2}" y="615" text-anchor="middle" font-size="13.5" fill="{SUB}">延迟随长度凸增（a≥0）：同样一段 T，越往后（L 越大）能塞进的 chunk x 越小——调度自动对长序列收窄分块。</text>')

L.append('</svg>')
open("quadratic.svg", "w").write('\n'.join(L))
print("wrote quadratic.svg")
