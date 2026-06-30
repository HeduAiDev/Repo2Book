#!/usr/bin/env python3
"""fig27-1: 三入口注册 + 一张 scheme 注册表 + 三个 wrapper —— OOT 量化接入全景。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1380, 880
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="arb" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#475569", sw=2.0, mk="ar"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mk})"/>')


text(W / 2, 42, "把昇腾量化整套接进 vLLM：不改 vLLM 量化框架一行", 24, "middle", "#0f172a", "bold")
text(W / 2, 70, "三入口注册 → 一张 (quant_type, layer_type) 注册表 → 三个适配 vLLM 基类的 wrapper", 14, "middle", "#64748b")

# ── 列1：三入口 Config → vLLM 量化注册表 ──
text(230, 118, "① 三入口：注册 Config", 16, "middle", "#0f172a", "bold")
entries = [
    (140, "AscendModelSlimConfig", "@register_quantization_config('ascend')", "#ecfdf5", "#10b981"),
    (212, "AscendCompressedTensorsConfig", "先删 vLLM 原版 → 同名注册顶替", "#fff7ed", "#f59e0b"),
    (284, "AscendFp8Config", "先删 fp8/deepseek_v4_fp8 → 顶替", "#fff7ed", "#f59e0b"),
]
for ey, name, sub, fill, stroke in entries:
    box(40, ey, 380, 58, fill, stroke, 10, 1.8)
    text(56, ey + 26, name, 14.5, "start", "#0f172a", "bold", mono=True)
    text(56, ey + 47, sub, 11.5, "start", "#475569")

# vLLM 量化注册表
box(70, 372, 320, 60, "#eef2ff", "#6366f1", 10, 1.8)
text(230, 397, "vLLM 量化注册表", 14, "middle", "#0f172a", "bold")
text(230, 418, "QUANTIZATION_METHODS（已注册方法名列表）", 11.5, "middle", "#475569")
for ey, *_ in entries:
    arrow(230, ey + 58, 230, 372, "#475569", 1.6)

# ── 列2：分发枢纽 get_quant_method ──
text(W / 2, 118, "② 按层分发：get_quant_method(layer, prefix)", 16, "middle", "#0f172a", "bold")
box(500, 140, 380, 230, "#f8fafc", "#94a3b8", 12, 1.8)
text(690, 168, "isinstance(layer, ...) 四岔", 14, "middle", "#334155", "bold")
forks = [
    "LinearBase            → 'linear'",
    "AttentionLayerBase    → 'attention'",
    "FusedMoE              → 'moe'  (ch26)",
    "VocabParallelEmbedding→ 'linear'",
    "FLOAT 层 → 未量化方法（跳过）",
]
for i, f in enumerate(forks):
    text(520, 200 + i * 30, "• " + f, 12.5, "start", "#475569", mono=True)
text(690, 358, "→ create_scheme_for_layer(...)", 12.5, "middle", "#334155", mono=True)

# scheme 注册表
box(500, 410, 380, 150, "#f5f3ff", "#7c3aed", 12, 2)
text(690, 438, "③ scheme 注册表 _SCHEME_REGISTRY", 14, "middle", "#5b21b6", "bold")
text(690, 462, "dict[(quant_type, layer_type)] → SchemeClass", 11.5, "middle", "#6d28d9", mono=True)
rows = [
    "('W8A8_DYNAMIC','linear') → AscendW8A8DynamicLinearMethod",
    "('W8A8_DYNAMIC','moe')    → AscendW8A8DynamicFusedMoEMethod",
    "('W4A8_DYNAMIC','linear') → AscendW4A8DynamicLinearMethod",
    "… @register_scheme 装饰器逐项落表 …",
]
for i, r in enumerate(rows):
    text(516, 488 + i * 18, r, 10, "start", "#5b21b6", mono=True)
arrow(690, 370, 690, 410, "#7c3aed", 1.8, "arb")

# get_scheme_class 查表
text(690, 588, "get_scheme_class(quant_type, layer_type) 查表 → 实例化 scheme", 11.5, "middle", "#5b21b6", mono=True)

# ── 列3：三个 wrapper ──
text(1150, 118, "④ 三个 wrapper（持 scheme）", 16, "middle", "#0f172a", "bold")
wraps = [
    (150, "AscendLinearMethod", "(LinearMethodBase)", "linear / embedding 层"),
    (300, "AscendKVCacheMethod", "(BaseKVCacheMethod)", "attention / KV cache 量化"),
    (450, "AscendFusedMoEMethod", "(FusedMoEMethodBase)", "量化版 MoE ← ch26 FusedMoE"),
]
for wy, name, base, desc in wraps:
    box(960, wy, 380, 110, "#ecfeff", "#06b6d4", 12, 2)
    text(980, wy + 30, name, 15, "start", "#0e7490", "bold", mono=True)
    text(980, wy + 54, base, 12.5, "start", "#0891b2", mono=True)
    text(980, wy + 80, desc, 12.5, "start", "#475569")
    text(980, wy + 100, "create_weights / apply → 全转交 scheme", 11, "start", "#64748b", mono=True)
    arrow(880, wy + 55, 960, wy + 55, "#06b6d4", 1.8)

text(690, 658, "scheme 包进对应 wrapper，作为该层 quant_method 返回 vLLM", 12.5, "middle", "#334155")
arrow(690, 600, 690, 632, "#06b6d4", 1.8)

# 底注
box(70, 700, 1240, 140, "#f8fafc", "#cbd5e1", 12, 1.6, dash="6 5")
text(690, 728, "一张表 + 三个 wrapper：换一种量化方案只是给注册表加一行装饰器，分发与适配代码零改动", 15, "middle", "#0f172a", "bold")
notes = [
    "• 入口①直接继承 vLLM 基类 QuantizationConfig；入口②③名字被 vLLM 内置占用，故「先删原版、再同名顶替」。",
    "• 适配器只搬运：create_weights 按 scheme 给的形状逐个 register_parameter；apply 把前向原样转交 scheme.apply。",
    "• scheme 才决定一切：W8A8 int8+per-channel、W4A8 4bit+per-group、MXFP e8m0 微缩放——真实量化算子走 torch_npu（NPU/CANN）。",
]
for i, n in enumerate(notes):
    text(96, 760 + i * 26, n, 12.5, "start", "#475569")

L.append('</svg>')
open("fig27-1-overview.svg", "w").write('\n'.join(L))
print("wrote fig27-1-overview.svg")
