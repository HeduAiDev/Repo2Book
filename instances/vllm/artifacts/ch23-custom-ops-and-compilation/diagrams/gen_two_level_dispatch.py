#!/usr/bin/env python3
"""两级 dispatch 对照图：左=单算子构造期 dispatch，右=整图首次前向 dispatch。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1040, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke="#475569", rx=8):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def text(x, y, s, size=15, anchor="middle", weight="normal", fill="#1e293b", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
             f'font-weight="{weight}" fill="{fill}" font-family="{fam}">{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color="#475569"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
             f'stroke-width="1.8" marker-end="url(#a)"/>')


# 列分隔
LCX, RCX = 260, 780      # 两栏中心
COLW = 460

# 标题
text(LCX, 36, "第 1 级　单算子 dispatch（构造期一次性）", 18, weight="bold", fill="#0f172a")
text(RCX, 36, "第 2 级　整图 dispatch（首次前向）", 18, weight="bold", fill="#0f172a")
L.append(f'<line x1="520" y1="50" x2="520" y2="{H-30}" stroke="#cbd5e1" '
         f'stroke-width="1" stroke-dasharray="5 5"/>')

# ---- 左栏 ----
box(LCX - 150, 64, 300, 40, "#e0e7ff")
text(LCX, 89, "RMSNorm(...)  →  CustomOp.__init__", 14, mono=True)
arrow(LCX, 104, LCX, 128)

box(LCX - 150, 130, 300, 100, "#e0e7ff")
text(LCX, 158, "dispatch_forward()", 14, mono=True, weight="bold")
text(LCX, 192, "读 CompilationConfig.custom_ops", 13, fill="#475569")
text(LCX, 213, "调 enabled() / default_on()", 13, fill="#475569")

# 分叉
arrow(LCX, 230, LCX - 130, 248)
arrow(LCX, 230, LCX + 130, 248)

box(LCX - 240, 250, 220, 96, "#dcfce7", "#16a34a")
text(LCX - 130, 274, "enabled 且 CUDA", 13, weight="bold", fill="#166534")
text(LCX - 130, 296, "→ forward_cuda", 13, mono=True, fill="#166534")
text(LCX - 130, 318, "预编译融合 kernel", 12, fill="#475569")
text(LCX - 130, 336, "对 Inductor 不透明", 12, fill="#475569")

box(LCX + 20, 250, 220, 96, "#fef9c3", "#ca8a04")
text(LCX + 130, 274, "disabled（默认 none）", 13, weight="bold", fill="#854d0e")
text(LCX + 130, 296, "→ forward_native", 13, mono=True, fill="#854d0e")
text(LCX + 130, 318, "一串纯 torch 算子", 12, fill="#475569")
text(LCX + 130, 336, "留给 Inductor 融合", 12, fill="#475569")

arrow(LCX - 130, 346, LCX, 380)
arrow(LCX + 130, 346, LCX, 380)
box(LCX - 170, 382, 340, 40, "#f1f5f9")
text(LCX, 407, "self._forward_method  定死，运行期零开销转发", 13, mono=True)

text(LCX, 452, "粒度：单个算子选哪份实现", 13, fill="#64748b", weight="bold")

# ---- 右栏 ----
box(RCX - 170, 64, 340, 40, "#e0e7ff")
text(RCX, 89, "@support_torch_compile 模型 forward", 13, mono=True)
arrow(RCX, 104, RCX, 128)

box(RCX - 170, 130, 340, 40, "#e0e7ff")
text(RCX, 155, "Dynamo trace 整张计算图", 14, weight="bold")
arrow(RCX, 170, RCX, 194)

box(RCX - 200, 196, 400, 52, "#fee2e2", "#dc2626")
text(RCX, 218, "遇 unified_attention_with_output", 13, mono=True, fill="#991b1b")
text(RCX, 238, "不透明 op：不内联、不 graph break（回收 f17）", 12, fill="#991b1b")
arrow(RCX, 248, RCX, 272)

box(RCX - 170, 274, 340, 40, "#e0e7ff")
text(RCX, 299, "VllmBackend.split_graph 在此处切图", 13, mono=True, weight="bold")

arrow(RCX, 314, RCX - 130, 346)
arrow(RCX, 314, RCX + 130, 346)

box(RCX - 240, 348, 220, 86, "#dcfce7", "#16a34a")
text(RCX - 130, 372, "规整段", 13, weight="bold", fill="#166534")
text(RCX - 130, 394, "Inductor 编译", 12, fill="#475569")
text(RCX - 130, 414, "+ CUDA graph 重放", 12, fill="#475569")

box(RCX + 20, 348, 220, 86, "#fee2e2", "#dc2626")
text(RCX + 130, 372, "attention 段", 13, weight="bold", fill="#991b1b")
text(RCX + 130, 394, "保持 eager", 12, fill="#475569")
text(RCX + 130, 414, "跑真实 kernel", 12, fill="#475569")

text(RCX, 452, "粒度：整段图捕获/融合策略", 13, fill="#64748b", weight="bold")

# 底部说明
box(60, 500, W - 120, 120, "#f8fafc", "#cbd5e1")
text(W / 2, 528, "两级 dispatch 是两种不同粒度的选择", 16, weight="bold", fill="#0f172a")
text(W / 2, 558, "第 1 级在「构造期」按平台/配置把每个算子定到 forward_cuda 或 forward_native——单算子粒度。", 14, fill="#334155")
text(W / 2, 582, "第 2 级在「首次前向」把整图按 attention 切成段——规整段融合 + 图捕获，attention 段 eager——整图粒度。", 14, fill="#334155")
text(W / 2, 606, "选 forward_native（第 1 级 disabled），正是为了让纯 torch 算子暴露给第 2 级的 Inductor 去融合。", 14, fill="#334155")

L.append('</svg>')
with open("two-level-dispatch.svg", "w", encoding="utf-8") as f:
    f.write('\n'.join(L))
print("wrote two-level-dispatch.svg")
