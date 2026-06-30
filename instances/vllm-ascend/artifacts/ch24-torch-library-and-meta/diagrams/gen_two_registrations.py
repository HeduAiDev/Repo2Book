#!/usr/bin/env python3
"""fig24-1: 一个算子两份注册 —— 真实现(PrivateUse1,真算) vs meta(Meta,只推形状)。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1280, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '<marker id="arR" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#64748b", marker="ar"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#{marker})"/>')


text(W / 2, 40, "同一个算子，要登记两份实现", 25, "middle", "#0f172a", "bold")
text(W / 2, 68, "torch.ops._C_ascend.get_masked_input_and_mask", 15, "middle", "#475569", "normal", mono=True)

# central op node
cx, cy, cw, ch = 480, 96, 320, 60
box(cx, cy, cw, ch, "#eef2ff", "#6366f1", 12, 2.2)
text(cx + cw / 2, cy + 27, "一个 schema · 一个算子", 15, "middle", "#4338ca", "bold")
text(cx + cw / 2, cy + 48, "ops.def(\"...(Tensor input,...) -> (Tensor, Tensor)\")", 12, "middle", "#64748b", "normal", mono=True)

# left branch: real impl (warm)
lx, ly, lw, lh = 110, 250, 470, 260
box(lx, ly, lw, lh, "#fff7ed", "#f97316", 12, 2.2)
text(lx + lw / 2, ly + 34, "真实现 · 真算", 19, "middle", "#c2410c", "bold")
text(lx + lw / 2, ly + 64, "dispatch key = PrivateUse1（昇腾 NPU）", 14, "middle", "#9a3412", "bold")
text(lx + 28, ly + 100, "at::empty_like(input)  造输出壳", 13, "start", "#7c2d12", "normal", mono=True)
text(lx + 28, ly + 128, "input.data_ptr()       拿真实设备内存", 13, "start", "#7c2d12", "normal", mono=True)
text(lx + 28, ly + 156, "getCurrentNPUStream()  取 NPU 流", 13, "start", "#7c2d12", "normal", mono=True)
text(lx + 28, ly + 184, "OpCommand.Run()        提交 AscendC kernel", 13, "start", "#7c2d12", "normal", mono=True)
box(lx + 24, ly + 204, lw - 48, 40, "#ffedd5", "#fb923c", 8, 1.4)
text(lx + lw / 2, ly + 229, "→ 真往设备内存算并填数据", 14, "middle", "#9a3412", "bold")
arrow(cx + 80, cy + ch, lx + lw / 2, ly)

# right branch: meta impl (cool)
rx, ry, rw, rh = 700, 250, 470, 260
box(rx, ry, rw, rh, "#eff6ff", "#3b82f6", 12, 2.2)
text(rx + rw / 2, ry + 34, "meta 实现 · 只推形状", 19, "middle", "#1d4ed8", "bold")
text(rx + rw / 2, ry + 64, "dispatch key = Meta（at::kMeta 张量）", 14, "middle", "#1e40af", "bold")
text(rx + 28, ry + 100, "at::empty_like(input)  造输出壳", 13, "start", "#1e3a8a", "normal", mono=True)
text(rx + 28, ry + 128, "（无 data_ptr）        不碰设备内存", 13, "start", "#1e3a8a", "normal", mono=True)
text(rx + 28, ry + 156, "（无 NPU 流）          不取流", 13, "start", "#1e3a8a", "normal", mono=True)
text(rx + 28, ry + 184, "return {壳}            直接返回空壳", 13, "start", "#1e3a8a", "normal", mono=True)
box(rx + 24, ry + 204, rw - 48, 40, "#dbeafe", "#60a5fa", 8, 1.4)
text(rx + rw / 2, ry + 229, "→ 只给出 shape / dtype，不真算", 14, "middle", "#1e40af", "bold")
arrow(cx + cw - 80, cy + ch, rx + rw / 2, ry)

# bottom: two dispatch paths
by = 600
text(110, by - 18, "两条使用路径，按张量设备各派发到一边：", 15, "start", "#334155", "bold")

# inference path -> left
box(150, by, 390, 90, "#fefce8", "#eab308", 10, 1.6)
text(150 + 195, by + 30, "推理期：真实 NPU 张量", 15, "middle", "#854d0e", "bold")
text(150 + 195, by + 56, "torch 按设备 = NPU 派发", 13, "middle", "#475569")
text(150 + 195, by + 78, "→ 走左边真实现，真算出结果", 13, "middle", "#854d0e", "bold")
arrow(345, by, lx + lw / 2, ly + lh + 6, "#ca8a04")

# capture path -> right
box(740, by, 390, 90, "#f0fdf4", "#22c55e", 10, 1.6)
text(740 + 195, by + 30, "图捕获期：Meta 张量（假跑）", 15, "middle", "#15803d", "bold")
text(740 + 195, by + 56, "torch.compile / ACLGraph trace", 13, "middle", "#475569", "normal", mono=False)
text(740 + 195, by + 78, "→ 走右边 meta，只把形状推下去", 13, "middle", "#15803d", "bold")
arrow(935, by, rx + rw / 2, ry + rh + 6, "#16a34a")

# red break note (when meta missing)
box(rx + rw - 250, ry - 4, 250, 0.1, "#fff", "#fff", 0, 0)  # spacer no-op
text(W / 2, 720, "若右边缺 meta —— trace 到该算子推不出输出形状 —— 整张图捕获在此断", 15, "middle", "#dc2626", "bold")

L.append('</svg>')
open("fig24-1-two-registrations.svg", "w").write('\n'.join(L))
print("wrote fig24-1-two-registrations.svg")
