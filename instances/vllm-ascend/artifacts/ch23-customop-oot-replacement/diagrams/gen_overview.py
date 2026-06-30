#!/usr/bin/env python3
"""fig-oot-replacement-overview: 一处注册 → 全模型算子换头换身。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1480, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '<marker id="arR" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')


def text(x, y, s, size=16, anchor="start", fill="#1e293b", weight="normal", family="sans-serif", mono=False):
    fam = "monospace" if mono else family
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#64748b", marker="ar"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#{marker})"/>')


# Title
text(W / 2, 40, "一处注册，全模型算子被批量顶替", 24, "middle", "#0f172a", "bold")
text(W / 2, 70, "register_ascend_customop 建一张表 → 基座 __new__/dispatch_forward 顺势换身换头", 15, "middle", "#64748b")

# ---- Panel A: 模型定义（左）----
ax, ay, aw, ah = 40, 130, 300, 360
box(ax, ay, aw, ah, "#f8fafc", "#94a3b8")
text(ax + aw / 2, ay + 34, "模型定义", 18, "middle", "#0f172a", "bold")
text(ax + aw / 2, ay + 58, "（不知昇腾存在）", 13, "middle", "#64748b")
# code lines
cb_y = ay + 86
for i, ln in enumerate(["self.norm = RMSNorm(", "        hidden_size, eps)", "self.act = SiluAndMul()"]):
    text(ax + 24, cb_y + i * 26, ln, 14, "start", "#1e293b", mono=True)
text(ax + aw / 2, ay + ah - 70, "vLLM 主干源码", 13, "middle", "#475569", "bold")
text(ax + aw / 2, ay + ah - 46, "一行都不改", 15, "middle", "#b91c1c", "bold")
text(ax + aw / 2, ay + ah - 22, "（OOT：out-of-tree 插件）", 12, "middle", "#94a3b8")

# ---- Panel B: 启动期建表（中）----
bx, by, bw, bh = 430, 110, 470, 460
box(bx, by, bw, bh, "#eff6ff", "#3b82f6")
text(bx + bw / 2, by + 32, "启动期：register_ascend_customop()", 17, "middle", "#1d4ed8", "bold")
text(bx + bw / 2, by + 54, "worker 初始化时唯一一次调用", 12, "middle", "#64748b")

# table REGISTERED_ASCEND_OPS
tb_x, tb_y, tb_w = bx + 30, by + 76, bw - 60
text(tb_x, tb_y, "REGISTERED_ASCEND_OPS", 14, "start", "#1e293b", "bold", mono=True)
rows = [
    ("\"QuickGELU\"", "AscendQuickGELU"),
    ("\"SiluAndMul\"", "AscendSiluAndMul"),
    ("\"RMSNorm\"", "AscendRMSNorm"),
    ("\"FusedMoE\"", "AscendFusedMoE"),
    ("… 其余 23 项（共 27）", "Ascend*"),
]
row_h = 30
rt_y = tb_y + 16
box(tb_x, rt_y, tb_w, row_h * len(rows) + 8, "white", "#cbd5e1", 6)
for i, (k, v) in enumerate(rows):
    yy = rt_y + 24 + i * row_h
    text(tb_x + 14, yy, k, 13, "start", "#0369a1", mono=True)
    text(tb_x + 200, yy, "→", 14, "middle", "#94a3b8")
    text(tb_x + 220, yy, v, 13, "start", "#be123c", mono=True)

# loop register_oot
lp_y = rt_y + row_h * len(rows) + 40
text(bx + bw / 2, lp_y, "for name, op_cls in 表.items():", 13, "middle", "#334155", mono=True)
text(bx + bw / 2, lp_y + 22, "CustomOp.register_oot(op_cls, name)", 13, "middle", "#334155", mono=True)
arrow(bx + bw / 2, lp_y + 34, bx + bw / 2, lp_y + 62)

# op_registry_oot box
orx, ory, orw, orh = bx + 70, lp_y + 66, bw - 140, 60
box(orx, ory, orw, orh, "#dbeafe", "#2563eb", 8)
text(bx + bw / 2, ory + 26, "op_registry_oot", 15, "middle", "#1d4ed8", "bold", mono=True)
text(bx + bw / 2, ory + 46, "{ 'RMSNorm': AscendRMSNorm, … }", 12, "middle", "#475569", mono=True)

# ---- Panel C: 构造期（右）----
cx, cy, cw, ch = 980, 130, 460, 360
box(cx, cy, cw, ch, "#fef2f2", "#ef4444")
text(cx + cw / 2, cy + 32, "构造期：执行到 RMSNorm(...)", 16, "middle", "#b91c1c", "bold")

# 换身
sy = cy + 60
box(cx + 24, sy, cw - 48, 96, "white", "#f87171", 8)
text(cx + 44, sy + 26, "__new__ 查 op_registry_oot 命中", 13, "start", "#1e293b", mono=True)
text(cx + 44, sy + 48, "super().__new__(AscendRMSNorm)", 13, "start", "#be123c", mono=True)
text(cx + 44, sy + 76, "换身：真正造出的是昇腾子类实例", 14, "start", "#b91c1c", "bold")

# 换头
hy = sy + 120
box(cx + 24, hy, cw - 48, 96, "white", "#f87171", 8)
text(cx + 44, hy + 26, "__init__ → dispatch_forward()", 13, "start", "#1e293b", mono=True)
text(cx + 44, hy + 48, "_forward_method = forward_oot", 13, "start", "#be123c", mono=True)
text(cx + 44, hy + 76, "换头：forward 绑到 NPU 实现", 14, "start", "#b91c1c", "bold")

# arrows between panels
arrow(ax + aw, ay + ah / 2, bx, by + bh / 2)
text((ax + aw + bx) / 2, ay + ah / 2 - 12, "继承自", 12, "middle", "#64748b")
text((ax + aw + bx) / 2, ay + ah / 2 + 6, "基座", 12, "middle", "#64748b")
arrow(bx + bw, by + bh / 2, cx, cy + ch / 2, "#b91c1c", "arR")
text((bx + bw + cx) / 2, by + bh / 2 - 10, "查表", 12, "middle", "#b91c1c")

L.append('</svg>')
open("fig-overview.svg", "w").write('\n'.join(L))
print("wrote fig-overview.svg")
