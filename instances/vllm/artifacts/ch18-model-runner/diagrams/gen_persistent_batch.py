#!/usr/bin/env python3
"""ch18-persistent-batch-update: 持久批次跨拍演化 打洞→复用 slot→压实."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1180, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>')
L.append('<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# colors
C_ACTIVE = "#dbeafe"   # active
C_BORDER = "#475569"
C_HOLE   = "#fee2e2"   # removed hole
C_REUSE  = "#dcfce7"   # reused slot
C_MOVE   = "#fef9c3"   # condensed moved
C_EMPTY  = "#f1f5f9"

NSLOT = 6
slot_h = 46
slot_w = 150
col_gap = 230
top = 110
left_label_x = 40

def title(x, y, t, sub=None):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="20" font-weight="bold" fill="#0f172a">{esc(t)}</text>')
    if sub:
        L.append(f'<text x="{x}" y="{y+22}" font-family="sans-serif" font-size="13" fill="#64748b">{esc(sub)}</text>')

def draw_col(cx, cells):
    # cells: list of (label, fillcolor, textcolor)
    for i, (lab, fc, tc) in enumerate(cells):
        y = top + i * slot_h
        L.append(f'<rect x="{cx}" y="{y}" width="{slot_w}" height="{slot_h-6}" rx="6" fill="{fc}" stroke="{C_BORDER}" stroke-width="1.5"/>')
        L.append(f'<text x="{cx-10}" y="{y+(slot_h-6)//2+5}" text-anchor="end" font-family="sans-serif" font-size="14" fill="#94a3b8">slot {i}</text>')
        L.append(f'<text x="{cx+slot_w//2}" y="{y+(slot_h-6)//2+5}" text-anchor="middle" font-family="sans-serif" font-size="15" fill="{tc}">{esc(lab)}</text>')

# 拍1
x1 = 160
title(x1-90, 70, "拍 1：四请求满载")
draw_col(x1, [
    ("A (active)", C_ACTIVE, "#1e3a8a"),
    ("B (active)", C_ACTIVE, "#1e3a8a"),
    ("C (active)", C_ACTIVE, "#1e3a8a"),
    ("D (active)", C_ACTIVE, "#1e3a8a"),
    ("—", C_EMPTY, "#94a3b8"),
    ("—", C_EMPTY, "#94a3b8"),
])

# 拍2
x2 = x1 + col_gap + 60
title(x2-100, 70, "拍 2：B、D 完成 + E、F、G 到来")
draw_col(x2, [
    ("A", C_ACTIVE, "#1e3a8a"),
    ("E  (复用 slot1)", C_REUSE, "#166534"),
    ("C", C_ACTIVE, "#1e3a8a"),
    ("F  (复用 slot3)", C_REUSE, "#166534"),
    ("—", C_EMPTY, "#94a3b8"),
    ("G  (追加 slot5)", C_ACTIVE, "#1e3a8a"),
])
# annotate removed->reused
L.append(f'<text x="{x2+slot_w//2}" y="{top - 12}" text-anchor="middle" font-family="sans-serif" font-size="12.5" fill="#b45309">remove 打洞(1,3) → pop_removed 复用，G 无洞可填则追加</text>')

# 拍3
x3 = x2 + col_gap + 70
title(x3-90, 70, "拍 3：C 完成留洞 + 无新请求")
# before condense: slot2 hole, slot5 has G (assume G arrived earlier filling tail)
# show condense: tail active slot5(G) slides into slot2
draw_col(x3, [
    ("A", C_ACTIVE, "#1e3a8a"),
    ("E", C_ACTIVE, "#1e3a8a"),
    ("G  ←slot5", C_MOVE, "#854d0e"),
    ("F", C_ACTIVE, "#1e3a8a"),
    ("—", C_EMPTY, "#94a3b8"),
    ("—  (G 搬走)", C_EMPTY, "#94a3b8"),
])
# arrow showing move from slot5 to slot2
y5 = top + 5*slot_h + (slot_h-6)//2
y2c = top + 2*slot_h + (slot_h-6)//2
L.append(f'<path d="M {x3+slot_w+8} {y5} C {x3+slot_w+70} {y5}, {x3+slot_w+70} {y2c}, {x3+slot_w+8} {y2c}" fill="none" stroke="#b45309" stroke-width="2" marker-end="url(#arr)"/>')
L.append(f'<text x="{x3+slot_w+78} " y="{(y5+y2c)//2+4}" font-family="sans-serif" font-size="12" fill="#b45309">condense</text>')

# legend
ly = top + NSLOT*slot_h + 40
items = [("active", C_ACTIVE), ("removed 空洞", C_HOLE), ("复用 slot", C_REUSE), ("condense 搬移", C_MOVE), ("空 slot", C_EMPTY)]
lx = 80
for lab, fc in items:
    L.append(f'<rect x="{lx}" y="{ly}" width="24" height="18" rx="3" fill="{fc}" stroke="{C_BORDER}"/>')
    L.append(f'<text x="{lx+30}" y="{ly+14}" font-family="sans-serif" font-size="13.5" fill="#334155">{esc(lab)}</text>')
    lx += 175 + (40 if lab=="removed 空洞" else 0)

# bottom note about batch_update_builder
L.append(f'<text x="80" y="{ly+58}" font-family="sans-serif" font-size="13.5" fill="#475569">batch_update_builder.removed 始终按降序存空洞；pop_removed() 总返回当前最小空 slot —— add 优先填洞，condense 兜底压实。</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch18-model-runner/diagrams/ch18-persistent-batch-update.svg","w").write('\n'.join(L))
print("ok")
