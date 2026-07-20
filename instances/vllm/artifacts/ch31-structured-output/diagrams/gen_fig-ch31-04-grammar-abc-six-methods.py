#!/usr/bin/env python3
"""fig-ch31-04: 语法状态机六方法契约——两种推进/一种回退/一种取掩码/一种问终态/一种归零。
template: state-machine（中心节点 + 六条辐射边，每条边标注方法与真实调用点）"""
import xml.sax.saxutils as xs
from math import cos, sin, pi
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1280, 1000
CX, CY = W / 2, 470
CORE_R = 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
          '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("请求级契约的六个方法，各自对应语法状态机上一种不可省略的操作")}</text>')

# 中心节点
L.append(f'<circle cx="{CX}" cy="{CY}" r="{CORE_R}" fill="#e0e7ff" stroke="#6366f1" stroke-width="2.5"/>')
L.append(f'<text x="{CX}" y="{CY-10}" text-anchor="middle" font-family="sans-serif" font-size="15" '
          f'font-weight="bold" fill="#312e81">{esc("语法状态机")}</text>')
L.append(f'<text x="{CX}" y="{CY+10}" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#4338ca">{esc("GrammarMatcher(xgrammar)")}</text>')
L.append(f'<text x="{CX}" y="{CY+27}" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#4338ca">{esc("LLMatcher(guidance)")}</text>')

# 六个方法节点，环形分布
METHODS = [
    {"name": "accept_tokens(request_id, tokens)", "role": "真推进",
     "detail": ["用这一步真采样出的 token 推进", "调用点：scheduler.py:L1363"],
     "color": "#16a34a", "fill": "#dcfce7"},
    {"name": "validate_tokens(tokens)", "role": "试走（不推进）",
     "detail": ["返回被接受的前缀", "调用点：scheduler.py:L1620 / L1650", "只喂 spec_token_ids"],
     "color": "#2563eb", "fill": "#dbeafe"},
    {"name": "rollback(num_tokens)", "role": "回退",
     "detail": ["投机解码专用的退回口子", "（下一章展开）"],
     "color": "#dc2626", "fill": "#fee2e2"},
    {"name": "fill_bitmask(bitmask, batch_index)", "role": "取掩码",
     "detail": ["与采样的唯一接口", "写自己那一行"],
     "color": "#ea580c", "fill": "#ffedd5"},
    {"name": "is_terminated()", "role": "问终态",
     "detail": ["语法是否已走到终态"],
     "color": "#7c3aed", "fill": "#ede9fe"},
    {"name": "reset()", "role": "归零",
     "detail": ["v0.21.0 全仓无 in-tree 调用者", "契约完整性存在、当前无人调用"],
     "color": "#64748b", "fill": "#f1f5f9"},
]

N = len(METHODS)
RADIUS = 330
NODE_W, NODE_H = 340, 108
angle0 = -pi / 2  # 从正上方开始，顺时针分布
for i, m in enumerate(METHODS):
    ang = angle0 + i * (2 * pi / N)
    nx = CX + RADIUS * cos(ang)
    ny = CY + RADIUS * sin(ang)
    # 边：从中心圆边缘到节点边缘
    ex = CX + CORE_R * cos(ang)
    ey = CY + CORE_R * sin(ang)
    # 节点边缘（矩形近似为椭圆交点）
    tx = nx - (NODE_W/2) * cos(ang) * 0.0  # 占位，稍后用简单裁剪
    marker = "url(#a)" if m["role"] in ("真推进", "回退") else "url(#b)"
    color = m["color"]
    if m["role"] == "回退":
        # 双向：语法机 <-> rollback
        L.append(f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{nx:.1f}" y2="{ny:.1f}" '
                  f'stroke="{color}" stroke-width="2" marker-end="url(#a)" marker-start="url(#a)"/>')
    else:
        L.append(f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{nx:.1f}" y2="{ny:.1f}" '
                  f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')

    x0 = nx - NODE_W / 2
    y0 = ny - NODE_H / 2
    L.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{NODE_W}" height="{NODE_H}" rx="10" '
              f'fill="{m["fill"]}" stroke="{color}" stroke-width="2"/>')
    L.append(f'<text x="{nx:.1f}" y="{y0+22:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{color}">{esc(m["role"])}</text>')
    L.append(f'<text x="{nx:.1f}" y="{y0+42:.1f}" text-anchor="middle" font-family="monospace" '
              f'font-size="11.5" font-weight="bold" fill="#1e293b">{esc(m["name"])}</text>')
    dy0 = y0 + 60
    for k, d in enumerate(m["detail"]):
        L.append(f'<text x="{nx:.1f}" y="{dy0+k*15:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" fill="#334155">{esc(d)}</text>')

# 底部数字条
foot_y = 940
L.append(f'<rect x="40" y="{foot_y-28}" width="{W-80}" height="70" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
facts = [
    "契约方法数：6（backend_types.py:L31-96）",
    "accept_tokens 调用点：1 处（scheduler.py:L1363）",
    "validate_tokens 调用点：2 处（scheduler.py:L1620 / L1650）",
    "reset() 的 in-tree 调用者：0",
]
for i, f in enumerate(facts):
    fx = 60 + (i % 2) * (W/2 - 60)
    fy = foot_y - 4 + (i // 2) * 26
    L.append(f'<text x="{fx}" y="{fy}" font-family="sans-serif" font-size="12.5" '
              f'fill="#334155">{esc("• " + f)}</text>')

L.append('</svg>')
out = Path("fig-ch31-04-grammar-abc-six-methods.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
