#!/usr/bin/env python3
"""可迁移性证明图：同一四步先跑最简 LlamaDecoderLayer。
左 __init__ → 中 模块树 → 右 forward 数据流。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1240, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#047857"/></marker>'
    '<marker id="map" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '<marker id="res" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">同一套四步先跑最简 LlamaDecoderLayer（可迁移性证明）</text>')

# 三栏标题
col_titles = [("① 读 __init__", 200, "#1d4ed8"), ("② 模块树（4 框）", 620, "#7c3aed"), ("③ 读 forward → 数据流", 1030, "#047857")]
for t, cx, c in col_titles:
    L.append(f'<text x="{cx}" y="68" text-anchor="middle" font-size="16.5" font-weight="bold" fill="{c}">{esc(t)}</text>')

# ── 左栏：__init__ 代码（4 条 self.X 赋值） ──
code_lines = [
    "self.self_attn = attn_layer_type(",
    "    ..., prefix=f\"{prefix}.self_attn\")",
    "self.mlp = LlamaMLP(",
    "    ..., prefix=f\"{prefix}.mlp\")",
    "self.input_layernorm = RMSNorm(...)",
    "self.post_attention_layernorm \\",
    "    = RMSNorm(...)",
]
cx0, cy0 = 30, 90
L.append(f'<rect x="{cx0}" y="{cy0}" width="340" height="210" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>')
for i, ln in enumerate(code_lines):
    L.append(f'<text x="{cx0+14}" y="{cy0+28+i*26}" font-size="12.5" font-family="monospace" fill="#334155">{esc(ln)}</text>')

# ── 中栏：4 个模块树框，按 self.X 一一对应 ──
tree_boxes = [
    ("self_attn", "LlamaAttention", "#1d4ed8", "#dbeafe"),
    ("mlp", "LlamaMLP", "#1d4ed8", "#dbeafe"),
    ("input_layernorm", "RMSNorm", "#475569", "#f1f5f9"),
    ("post_attention_layernorm", "RMSNorm", "#475569", "#f1f5f9"),
]
tx, tw, th = 470, 300, 44
ty0 = 90
tgap = 14
mid_centers = []
for i, (name, cls, st, fl) in enumerate(tree_boxes):
    y = ty0 + i * (th + tgap)
    L.append(f'<rect x="{tx}" y="{y}" width="{tw}" height="{th}" rx="7" fill="{fl}" stroke="{st}" stroke-width="2"/>')
    L.append(f'<text x="{tx+14}" y="{y+19}" font-size="14" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{tx+14}" y="{y+37}" font-size="12" fill="#64748b">({esc(cls)})</text>')
    mid_centers.append((tx, y + th / 2, tx + tw, y + th / 2))

# 父框 LlamaDecoderLayer 包住四个
L.append(f'<rect x="{tx-16}" y="{ty0-36}" width="{tw+32}" height="{4*(th+tgap)+28}" rx="10" fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6,4"/>')
L.append(f'<text x="{tx+tw/2}" y="{ty0-42}" text-anchor="middle" font-size="13" fill="#7c3aed" font-weight="bold">LlamaDecoderLayer</text>')

# 左→中映射线（每条代码 → 框）
left_pts = [(cy0 + 28 + 0 * 26 - 5), (cy0 + 28 + 2 * 26 - 5), (cy0 + 28 + 4 * 26 - 5), (cy0 + 28 + 5 * 26 - 5)]
for i, ly in enumerate(left_pts):
    L.append(f'<line x1="{cx0+340}" y1="{ly}" x2="{tx-2}" y2="{mid_centers[i][1]}" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#map)"/>')

# ── 右栏：forward 数据流（自上而下链） ──
flow = [
    ("input_layernorm", "#475569", "#f1f5f9"),
    ("self_attn", "#1d4ed8", "#dbeafe"),
    ("post_attention_layernorm", "#475569", "#f1f5f9"),
    ("mlp", "#1d4ed8", "#dbeafe"),
]
fx, fw, fh = 870, 280, 44
fy0 = 100
fgap = 38
flow_centers = []
for i, (name, st, fl) in enumerate(flow):
    y = fy0 + i * (fh + fgap)
    cx = fx + fw / 2
    L.append(f'<rect x="{fx}" y="{y}" width="{fw}" height="{fh}" rx="7" fill="{fl}" stroke="{st}" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{y+28}" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    flow_centers.append((cx, y, y + fh))

# 前向有向边
for i in range(len(flow) - 1):
    cx, _, ybot = flow_centers[i]
    _, ytop2, _ = flow_centers[i + 1]
    L.append(f'<line x1="{cx}" y1="{ybot+2}" x2="{cx}" y2="{ytop2-6}" stroke="#047857" stroke-width="2.5" marker-end="url(#arr)"/>')

# 残差回边（右侧外弯）：residual 加回到 attn 后、mlp 后
rx = fx + fw + 26
# 残差1：入口 residual → 跨过 self_attn（add-norm）
L.append(f'<path d="M {fx+fw} {flow_centers[0][1]+8} L {rx} {flow_centers[0][1]+8} L {rx} {flow_centers[2][1]+8} L {fx+fw} {flow_centers[2][1]+8}" fill="none" stroke="#b45309" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#res)"/>')
L.append(f'<text x="{rx+8}" y="{(flow_centers[0][1]+flow_centers[2][1])/2+10}" font-size="11.5" fill="#b45309" font-weight="bold">residual</text>')
L.append(f'<text x="{rx+8}" y="{(flow_centers[0][1]+flow_centers[2][1])/2+26}" font-size="11.5" fill="#b45309">add-norm</text>')

# 中→右映射（同名框对应）虚线极淡，避免太乱：只标一句
L.append(f'<text x="{(tx+tw+fx)/2}" y="{H-70}" text-anchor="middle" font-size="12.5" fill="#64748b">同名框：模块树「拥有」（静态）→ 数据流「调用」（动态），forward 决定调用次序</text>')

# 底部结论条
L.append(f'<rect x="40" y="{H-50}" width="{W-80}" height="36" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
L.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" font-size="15" font-weight="bold" fill="#3730a3">同样四步，DeepSeek-V4 只是框更多、判据不变</text>')

L.append('</svg>')
svg = '\n'.join(L)
out = __import__('pathlib').Path(__file__).parent / "llama-baseline.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}")
