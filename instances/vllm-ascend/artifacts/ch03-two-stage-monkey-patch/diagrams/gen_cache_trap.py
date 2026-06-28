#!/usr/bin/env python3
"""from-import 缓存陷阱：名字绑定 vs 对象身份。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1200, 680
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" font-family="sans-serif" font-size="24" font-weight="bold" fill="#0f172a" text-anchor="middle">from-import 缓存陷阱：名字绑定 ≠ 对象身份</text>')


def namebox(x, y, label, sub, col, bg):
    w, h = 240, 56
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{bg}" stroke="{col}" stroke-width="1.8"/>')
    L.append(f'<text x="{x+w/2}" y="{y+24}" font-family="monospace" font-size="14" font-weight="bold" fill="{col}" text-anchor="middle">{esc(label)}</text>')
    L.append(f'<text x="{x+w/2}" y="{y+44}" font-family="sans-serif" font-size="11.5" fill="#64748b" text-anchor="middle">{esc(sub)}</text>')
    return (x, y, w, h)


def objbox(x, y, label, col, bg):
    w, h = 150, 50
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="25" fill="{bg}" stroke="{col}" stroke-width="2"/>')
    L.append(f'<text x="{x+w/2}" y="{y+30}" font-family="monospace" font-size="15" font-weight="bold" fill="{col}" text-anchor="middle">{esc(label)}</text>')
    return (x, y, w, h)


def arrow(x1, y1, x2, y2, col, marker, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.2"{d} marker-end="url(#{marker})"/>')


# ---------- Stage 1: from M import f ----------
L.append(f'<text x="40" y="92" font-family="sans-serif" font-size="16" font-weight="bold" fill="#1e293b">第 1 步　模块 M 定义 broadcast，调用方执行 from M import broadcast</text>')
n1 = namebox(70, 110, "M.broadcast", "顶层名字", "#2563eb", "#eff6ff")
n2 = namebox(70, 184, "caller.broadcast", "调用方缓存的名字", "#2563eb", "#eff6ff")
o1 = objbox(560, 147, "原 fn", "#475569", "#f1f5f9")
arrow(n1[0] + n1[2], n1[1] + n1[3] / 2, o1[0], o1[1] + 12, "#2563eb", "a")
arrow(n2[0] + n2[2], n2[1] + n2[3] / 2, o1[0], o1[1] + o1[3] - 12, "#2563eb", "a")
L.append(f'<text x="560" y="232" font-family="sans-serif" font-size="12.5" fill="#64748b">两个名字指向同一个函数对象</text>')

# divider
L.append(f'<line x1="40" y1="262" x2="{W-40}" y2="262" stroke="#e2e8f0" stroke-width="1.5"/>')

# ---------- Stage 2: only top name rebound (the trap) ----------
L.append(f'<text x="40" y="296" font-family="sans-serif" font-size="16" font-weight="bold" fill="#b91c1c">第 2 步　只改顶层名字 M.broadcast = w　→　调用方漏网（陷阱）</text>')
n3 = namebox(70, 314, "M.broadcast", "已改", "#16a34a", "#f0fdf4")
n4 = namebox(70, 388, "caller.broadcast", "仍指旧对象！", "#dc2626", "#fef2f2")
ow = objbox(560, 314, "wrapper w", "#16a34a", "#f0fdf4")
of = objbox(560, 388, "原 fn", "#dc2626", "#fef2f2")
arrow(n3[0] + n3[2], n3[1] + n3[3] / 2, ow[0], ow[1] + ow[3] / 2, "#16a34a", "ag")
arrow(n4[0] + n4[2], n4[1] + n4[3] / 2, of[0], of[1] + of[3] / 2, "#dc2626", "ar")
L.append(f'<text x="730" y="345" font-family="sans-serif" font-size="12.5" fill="#16a34a">新调用拿到 wrapper</text>')
L.append(f'<text x="730" y="419" font-family="sans-serif" font-size="12.5" fill="#dc2626">已缓存的调用仍走旧 fn → patch 漏网</text>')

# divider
L.append(f'<line x1="40" y1="462" x2="{W-40}" y2="462" stroke="#e2e8f0" stroke-width="1.5"/>')

# ---------- Stage 3: solution ----------
L.append(f'<text x="40" y="496" font-family="sans-serif" font-size="16" font-weight="bold" fill="#15803d">解药　把所有持有引用的别名都重绑为 w</text>')
sx, sy, sw, sh = 70, 514, 1060, 132
L.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="12" fill="#f8fafc" stroke="#16a34a" stroke-width="1.6"/>')
code = [
    "torch.distributed.broadcast              = broadcast310p_wrapper(torch.distributed.broadcast)",
    "torch.distributed.distributed_c10d.broadcast = broadcast310p_wrapper(torch.distributed.distributed_c10d.broadcast)",
]
for i, c in enumerate(code):
    L.append(f'<text x="{sx+24}" y="{sy+38+i*30}" font-family="monospace" font-size="13.5" fill="#0f172a">{esc(c)}</text>')
L.append(f'<text x="{sx+24}" y="{sy+112}" font-family="sans-serif" font-size="13" fill="#475569">顶层名字 + distributed_c10d 子模块别名「同时」重绑 —— 不管调用方从哪个名字 import，拿到的都是 wrapper。</text>')

L.append('</svg>')
open("cache_trap.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote cache_trap.svg")
