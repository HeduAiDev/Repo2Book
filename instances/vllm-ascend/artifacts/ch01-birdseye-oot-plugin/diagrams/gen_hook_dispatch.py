#!/usr/bin/env python3
"""分发范式聚焦：vLLM 各调用点 → current_platform.get_*_cls() → 昇腾 qualname → 昇腾实现类。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1300, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7.5" markerHeight="5.5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
    '<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7.5" markerHeight="5.5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="40" font-family="sans-serif" font-size="24" font-weight="bold" fill="#0f172a" text-anchor="middle">一处覆写，整链切换：vLLM 问 current_platform 要组件，拿到的全是昇腾版</text>')

# 中心节点
cx, cy, cw, ch = 520, 300, 260, 120
L.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="14" fill="#fef3c7" stroke="#b45309" stroke-width="2.6"/>')
L.append(f'<text x="{cx+cw/2}" y="{cy+42}" font-family="sans-serif" font-size="18" font-weight="bold" fill="#b45309" text-anchor="middle">current_platform</text>')
L.append(f'<text x="{cx+cw/2}" y="{cy+70}" font-family="monospace" font-size="15" fill="#92400e" text-anchor="middle">= NPUPlatform</text>')
L.append(f'<text x="{cx+cw/2}" y="{cy+98}" font-family="sans-serif" font-size="12.5" fill="#475569" text-anchor="middle">覆写一组 get_*_cls 工厂钩子</text>')


def lbox(x, y, w, h, t, col, bg, sz=13.5, mono=False):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{bg}" stroke="{col}" stroke-width="1.8"/>')
    tf = "monospace" if mono else "sans-serif"
    L.append(f'<text x="{x+w/2}" y="{y+h/2+5}" font-family="{tf}" font-size="{sz}" font-weight="bold" fill="{col}" text-anchor="middle">{esc(t)}</text>')
    return (x, y, w, h)


# 左侧：vLLM 调用点（入边）
L.append(f'<text x="200" y="140" font-family="sans-serif" font-size="15" font-weight="bold" fill="#2563eb" text-anchor="middle">vLLM 引擎的调用点</text>')
callers = [
    ("要注意力后端", 168),
    ("要设备通信器", 258),
    ("要图捕获包装", 348),
    ("要编译后端", 438),
]
for t, y in callers:
    b = lbox(60, y, 230, 52, t, "#2563eb", "#eff6ff")
    L.append(f'<line x1="{b[0]+b[2]}" y1="{y+26}" x2="{cx}" y2="{cy+ch/2}" stroke="#2563eb" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="405" y="546" font-family="sans-serif" font-size="12.5" fill="#64748b" text-anchor="middle">统一问 current_platform.get_*_cls()</text>')

# 右侧：昇腾实现（出边），标注返回的 qualname
L.append(f'<text x="1075" y="140" font-family="sans-serif" font-size="15" font-weight="bold" fill="#b45309" text-anchor="middle">昇腾实现类（返回的 qualname）</text>')
impls = [
    ("AscendAttentionBackend", "get_attn_backend_cls", 168),
    ("NPUCommunicator", "get_device_communicator_cls", 258),
    ("ACLGraphWrapper", "get_static_graph_wrapper_cls", 348),
    ("AscendCompiler", "get_compile_backend", 438),
]
for name, hook, y in impls:
    b = lbox(960, y, 280, 52, name, "#b45309", "#fef3c7", sz=13.5, mono=True)
    L.append(f'<line x1="{cx+cw}" y1="{cy+ch/2}" x2="{b[0]}" y2="{y+26}" stroke="#b45309" stroke-width="2" marker-end="url(#ao)"/>')
    L.append(f'<text x="{b[0]+b[2]/2}" y="{y+70}" font-family="monospace" font-size="11" fill="#94a3b8" text-anchor="middle">{esc(hook)} →</text>')

# 底部：对照基类默认
L.append(f'<rect x="60" y="596" width="{W-120}" height="96" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>')
L.append(f'<text x="84" y="626" font-family="sans-serif" font-size="14.5" font-weight="bold" fill="#334155">对照基座 vLLM：Platform 基类的默认答案</text>')
L.append(f'<text x="84" y="654" font-family="sans-serif" font-size="13" fill="#475569">· get_attn_backend_cls 默认返回空串 &quot;&quot;（子类必须覆写）　· get_device_communicator_cls 默认返回通用 DeviceCommunicatorBase</text>')
L.append(f'<text x="84" y="678" font-family="sans-serif" font-size="13" fill="#475569">NPUPlatform 把这些默认逐一换成昇腾 qualname——vLLM 源码一行未改，整条组件链就被顶替。</text>')

L.append('</svg>')
open("hook_dispatch.svg", "w").write("\n".join(L))
print("wrote hook_dispatch.svg")
