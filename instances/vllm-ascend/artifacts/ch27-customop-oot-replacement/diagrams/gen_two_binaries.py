#!/usr/bin/env python3
"""fig-two-binaries: 两层正交二分 —— dispatch 选 oot/native；forward_oot 内选 融合/回退。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 740
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#64748b"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#ar)"/>')


def label(x, y, s, fill="#475569"):
    text(x, y, s, 14, "middle", fill, "bold")


text(W / 2, 38, "两层正交二分：各判一次，互不干扰", 24, "middle", "#0f172a", "bold")

# ---- Layer 1: dispatch_forward ----
text(60, 84, "第一层 · dispatch_forward（构造期，选 forward 后端）", 16, "start", "#1d4ed8", "bold")

# decision node
d1x, d1y, d1w, d1h = 540, 100, 240, 64
box(d1x, d1y, d1w, d1h, "#eff6ff", "#3b82f6", 10)
text(d1x + d1w / 2, d1y + 28, "enabled()?", 16, "middle", "#1d4ed8", "bold", mono=True)
text(d1x + d1w / 2, d1y + 50, "算子启用了吗", 12, "middle", "#64748b")

# left: not enabled -> native
n1x, n1y, n1w, n1h = 200, 220, 300, 80
box(n1x, n1y, n1w, n1h, "#f1f5f9", "#94a3b8", 10)
text(n1x + n1w / 2, n1y + 30, "forward_native（compiled）", 14, "middle", "#334155", "bold", mono=True)
text(n1x + n1w / 2, n1y + 54, "基座 PyTorch 原生实现 · 不顶替", 12, "middle", "#64748b")
arrow(d1x + 40, d1y + d1h, n1x + n1w / 2, n1y)
label((d1x + 40 + n1x + n1w / 2) / 2 - 30, 200, "否")

# right: enabled & is_out_of_tree -> forward_oot
o1x, o1y, o1w, o1h = 820, 220, 320, 80
box(o1x, o1y, o1w, o1h, "#fef2f2", "#ef4444", 10, 2.2)
text(o1x + o1w / 2, o1y + 30, "forward_oot（昇腾覆写）", 14, "middle", "#b91c1c", "bold", mono=True)
text(o1x + o1w / 2, o1y + 54, "enabled 且 is_out_of_tree() == True", 12, "middle", "#64748b")
arrow(d1x + d1w - 40, d1y + d1h, o1x + o1w / 2, o1y)
label((d1x + d1w - 40 + o1x + o1w / 2) / 2 + 40, 200, "是 + 昇腾平台", "#b91c1c")

# divider
L.append(f'<line x1="60" y1="350" x2="{W-60}" y2="350" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="6 5"/>')
text(W / 2, 374, "—— 走进 forward_oot 后，AscendRMSNorm 内部还有第二层判定 ——", 13, "middle", "#94a3b8")

# ---- Layer 2: enable_custom_op ----
text(60, 416, "第二层 · enable_custom_op()（forward_oot 内，选具体 kernel）", 16, "start", "#b91c1c", "bold")

d2x, d2y, d2w, d2h = 540, 432, 240, 64
box(d2x, d2y, d2w, d2h, "#fff7ed", "#f97316", 10)
text(d2x + d2w / 2, d2y + 28, "enable_custom_op()?", 15, "middle", "#c2410c", "bold", mono=True)
text(d2x + d2w / 2, d2y + 50, "编译产物 vllm_ascend_C 在位吗", 12, "middle", "#64748b")

# left: fused
fx, fy, fw, fh = 160, 560, 420, 120
box(fx, fy, fw, fh, "#ecfdf5", "#10b981", 10, 2.2)
text(fx + fw / 2, fy + 30, "torch.ops._C_ascend.npu_add_rms_norm_bias", 13, "middle", "#047857", "bold", mono=True)
text(fx + fw / 2, fy + 56, "一颗 AscendC 融合 kernel", 14, "middle", "#065f46", "bold")
text(fx + fw / 2, fy + 80, "add + rms_norm + bias 编进同一 kernel", 12, "middle", "#475569")
text(fx + fw / 2, fy + 100, "中间量留片上，HBM 只读写一次", 12, "middle", "#475569")
arrow(d2x + 40, d2y + d2h, fx + fw / 2, fy)
label((d2x + 40 + fx + fw / 2) / 2 - 20, 540, "真 → 有库")

# right: fallback
rx, ry, rw, rh = 760, 560, 460, 120
box(rx, ry, rw, rh, "#f8fafc", "#94a3b8", 10)
text(rx + rw / 2, ry + 30, "torch_npu.npu_add_rms_norm + x.add_(bias)", 13, "middle", "#475569", "bold", mono=True)
text(rx + rw / 2, ry + 56, "多颗 torch_npu 原子算子拼", 14, "middle", "#334155", "bold")
text(rx + rw / 2, ry + 80, "每颗 kernel 各读写一遍 HBM", 12, "middle", "#475569")
text(rx + rw / 2, ry + 100, "结果正确，但访存更多 → 更慢", 12, "middle", "#b45309")
arrow(d2x + d2w - 40, d2y + d2h, rx + rw / 2, ry)
label((d2x + d2w - 40 + rx + rw / 2) / 2 + 30, 540, "假 → 回退", "#b45309")

L.append('</svg>')
open("fig-two-binaries.svg", "w").write('\n'.join(L))
print("wrote fig-two-binaries.svg")
