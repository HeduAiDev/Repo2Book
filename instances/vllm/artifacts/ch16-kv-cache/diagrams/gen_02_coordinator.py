#!/usr/bin/env python3
"""02-coordinator-three-states: factory decision tree + topology."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1120, 600
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def box(x, y, bw, bh, fill, stroke, lines, fs=14, tw="normal", tc="#0f172a"):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    cy = y + bh/2 - (n-1)*(fs+4)/2 + fs/2 - 2
    for i, t in enumerate(lines):
        fw = tw if i == 0 else "normal"
        L.append(f'<text x="{x+bw/2}" y="{cy+i*(fs+4)}" text-anchor="middle" font-size="{fs}" '
                 f'font-weight="{fw}" fill="{tc}">{esc(t)}</text>')

def txt(x, y, t, fs=14, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(t)}</text>')

txt(40, 40, "get_kv_cache_coordinator —— 构造期一次性解析拓扑差异", 21, "start", "#0f172a", "bold")

# root
rx, ry, rw, rh = 430, 70, 260, 56
box(rx, ry, rw, rh, "#e0e7ff", "#6366f1", ["get_kv_cache_coordinator(...)"], 15, "bold")

# three children
cy = 220
children = [
    (40, "#fef2f2", "#ef4444", "KVCacheCoordinatorNoPrefixCache",
     "!enable_caching", ["关前缀缓存的退化协调器", "支持任意组数（含 0 组）", "命中恒为 0"]),
    (420, "#ecfdf5", "#10b981", "UnitaryKVCacheCoordinator",
     "组数 == 1", ["单一注意力类型（最常见）", "find 直接委托唯一 manager", "命中长度 = 命中块数 × block_size"]),
    (800, "#eff6ff", "#3b82f6", "HybridKVCacheCoordinator",
     "组数 > 1", ["混合注意力（full + SWA + ...）", "verify_and_split 按 spec 分桶", "不动点迭代求一致命中长度"]),
]
cw = 280
for cx, fill, stroke, title, cond, body in children:
    box(cx, cy, cw, 56, fill, stroke, [title], 13.5, "bold")
    # condition label on edge
    midx = cx + cw/2
    L.append(f'<line x1="{rx+rw/2}" y1="{ry+rh}" x2="{midx}" y2="{cy}" stroke="#475569" stroke-width="1.6" marker-end="url(#a)"/>')
    lx = (rx+rw/2 + midx)/2
    ly = (ry+rh + cy)/2
    L.append(f'<rect x="{lx-50}" y="{ly-13}" width="100" height="22" rx="4" fill="white" stroke="#cbd5e1"/>')
    txt(lx, ly+3, cond, 12.5, "middle", "#b45309", "bold")
    # body box
    box(cx, cy+72, cw, 84, "white", stroke, [""]*0, 12)
    for i, t in enumerate(body):
        txt(cx+14, cy+96+i*22, "• " + t, 12.5, "start", "#334155")

# topology illustrations under Unitary and Hybrid
ty = 410
# Unitary topology
txt(420+cw/2, ty, "single_type_managers 拓扑", 13, "middle", "#065f46", "bold")
box(420+70, ty+16, cw-140, 44, "#d1fae5", "#10b981", ["manager[0]"], 13)
txt(420+cw/2, ty+90, "1 个 manager", 12, "middle", "#16a34a")

# Hybrid topology: full bucket first + SWA bucket
txt(800+cw/2, ty, "single_type_managers 按 spec 分桶", 13, "middle", "#1e40af", "bold")
box(800+10, ty+16, 120, 44, "#bfdbfe", "#2563eb", ["full 桶", "（排首）"], 12)
box(800+150, ty+16, 120, 44, "#fecaca", "#ef4444", ["SWA 桶"], 12)
txt(800+cw/2, ty+92, "命中长度对齐到 lcm_block_size", 12, "middle", "#1e40af")
txt(800+cw/2, ty+110, "（各 block_size 的最小公倍数）", 11.5, "middle", "#64748b")

svg = '\n'.join(L) + '\n</svg>\n'
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch16-kv-cache/diagrams/02-coordinator-three-states.svg", "w", encoding="utf-8").write(svg)
print("wrote 02")
