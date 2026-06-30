#!/usr/bin/env python3
"""fig29-3: 提议-验证闭环 —— 本章管提议侧（_propose / prepare_inputs），ch28 管验证侧（拒绝采样）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 540
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#6d28d9"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=12, sw=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, color, sw=2.2, marker="ab", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{marker})"{d}/>')


PURPLE = "#6d28d9"
text(W / 2, 42, "提议-验证闭环：本章管提议侧，第 28 章管验证侧", 23, "middle", "#0f172a", "bold")
text(W / 2, 68, "紫 = 本章（提出 draft token / 按拒绝数收缩输入）    灰 = 第 28 章验证侧", 13.5, "middle", "#64748b")

bw, bh = 440, 120
ax, ay = 80, 120     # A 提议
bx, by = 800, 120    # B target 前向
cx2, cy = 800, 360   # C 拒绝采样
dx, dy = 80, 360     # D prepare_inputs

# A：proposer 提议侧（本章）
box(ax, ay, bw, bh, "#f5f3ff", PURPLE, 12, 2.2)
text(ax + bw / 2, ay + 30, "①  proposer.propose / _propose", 15, "middle", PURPLE, "bold")
text(ax + bw / 2, ay + 56, "一步提出 k = num_speculative_tokens 个 draft token", 12.5, "middle", "#5b21b6")
text(ax + bw / 2, ay + 80, "薄壳：CPU 匹配近零成本", 12, "middle", "#7c3aed")
text(ax + bw / 2, ay + 100, "重量级：跑一次 draft 前向（ACLGraph 包裹）", 12, "middle", "#7c3aed")

# B：target 一次前向（中性）
box(bx, by, bw, bh, "#f8fafc", "#475569", 12, 2.0)
text(bx + bw / 2, by + 34, "②  target 模型一次前向", 15, "middle", "#334155", "bold")
text(bx + bw / 2, by + 62, "并行验证 k+1 个候选位", 12.5, "middle", "#475569")
text(bx + bw / 2, by + 88, "（一趟前向算出每个候选位的真实分布）", 12, "middle", "#64748b")

# C：拒绝采样（ch28）
box(cx2, cy, bw, bh, "#f8fafc", "#94a3b8", 12, 2.0, dash="6 4")
text(cx2 + bw / 2, cy + 30, "③  AscendRejectionSampler  (第 28 章)", 15, "middle", "#475569", "bold")
text(cx2 + bw / 2, cy + 56, "接受最长合法前缀 + 1 个 bonus token", 12.5, "middle", "#475569")
text(cx2 + bw / 2, cy + 82, "其余 draft token 被拒绝", 12, "middle", "#64748b")
text(cx2 + bw / 2, cy + 102, "例：k=3 提出 → 接受 2 + bonus → 拒绝 1", 12, "middle", "#64748b")

# D：prepare_inputs（本章）
box(dx, dy, bw, bh, "#f5f3ff", PURPLE, 12, 2.2)
text(dx + bw / 2, dy + 30, "④  prepare_inputs", 15, "middle", PURPLE, "bold")
text(dx + bw / 2, dy + 56, "按拒绝数收缩 query_start_loc / seq_lens", 12.5, "middle", "#5b21b6")
text(dx + bw / 2, dy + 82, "np.cumsum + np.repeat 构造 token_indices", 12, "middle", "#7c3aed", mono=False)
text(dx + bw / 2, dy + 102, "只把未拒绝部分喂回下一步 speculator", 12, "middle", "#7c3aed")

# 箭头 A→B（顶部）
line(ax + bw, ay + bh / 2, bx, by + bh / 2, PURPLE, 2.4, "ab")
text((ax + bw + bx) / 2, ay + bh / 2 - 12, "k 个 draft token", 12.5, "middle", PURPLE, "bold")
# B→C（右侧下行，灰）
line(bx + bw / 2, by + bh, cx2 + bw / 2, cy, "#94a3b8", 2.2, "ag")
text(bx + bw / 2 + 110, (by + bh + cy) / 2 + 4, "logits → 验证", 12, "middle", "#64748b")
# C→D（底部，灰）
line(cx2, cy + bh / 2, dx + bw, dy + bh / 2, "#94a3b8", 2.2, "ag")
text((cx2 + dx + bw) / 2, cy + bh / 2 - 12, "已接受 token + 各请求拒绝数", 12.5, "middle", "#64748b")
# D→A（左侧上行，紫，闭环）
line(dx + bw / 2, dy, ax + bw / 2, ay + bh, PURPLE, 2.4, "ab")
text(dx + bw / 2 - 96, (dy + ay + bh) / 2 + 4, "下一步收缩后的输入", 12, "middle", PURPLE)

# 中心标注
text(W / 2, 300, "一步投机 = 提议 k 个 → 一趟验证 → 接受前缀 → 收缩输入再来", 13.5, "middle", "#0f172a", "bold")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch29-speculative-decode-npu/diagrams/fig29-3-loop.svg", "w").write('\n'.join(L))
print("wrote fig29-3-loop.svg")
