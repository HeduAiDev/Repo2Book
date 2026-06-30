#!/usr/bin/env python3
"""fig27-2: W8A8_DYNAMIC 一条全链 —— 从两张表落地到 npu_quant_matmul。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 970
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
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


def varrow(x, y1, y2, color="#475569", sw=2.0):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#ar)"/>')


text(W / 2, 42, "走通一条全链：W8A8_DYNAMIC linear 层", 24, "middle", "#0f172a", "bold")
text(W / 2, 70, "导入期填两张表 → 加载期读 json → 建层期选 scheme → 前向跑 NPU 量化算子", 14, "middle", "#64748b")

cx = W / 2
bw = 720
bx = cx - bw / 2

stages = [
    ("#ecfdf5", "#10b981", "导入期 · 两张表就位",
     ["@register_quantization_config('ascend') → AscendModelSlimConfig 进 vLLM 量化注册表",
      "@register_scheme('W8A8_DYNAMIC','linear') → (键)→AscendW8A8DynamicLinearMethod 进 _SCHEME_REGISTRY"]),
    ("#eff6ff", "#3b82f6", "加载期 · 读逐层量化描述",
     ["读模型目录 quant_model_description.json → 填 quant_description",
      "quant_description['…down_proj.weight'] == 'W8A8_DYNAMIC'（逐层一条）"]),
    ("#f5f3ff", "#7c3aed", "建层期 · 分发 + 选 scheme",
     ["get_quant_method(layer, prefix): isinstance → 'linear'",
      "get_linear_quant_type → 'W8A8_DYNAMIC'；get_scheme_class 查表实例化 scheme",
      "AscendLinearMethod(scheme) ← wrapper 持有 scheme"]),
    ("#fefce8", "#ca8a04", "建权重 · scheme 决定形状",
     ["scheme.get_weight → weight: int8 [out, in]",
      "scheme.get_perchannel_param → weight_scale/offset: [out, 1]（每输出通道一个）",
      "wrapper 逐个 register_parameter；process_weights_after_loading：transpose + 转 NZ 排布"]),
    ("#fef2f2", "#ef4444", "前向 apply · NPU 硬特化算子",
     ["x → npu_dynamic_quant(x) → (int8 quantized_x, pertoken_scale 每 token 一个)",
      "npu_quant_matmul(quantized_x, weight, weight_scale, pertoken_scale, output_dtype=x.dtype)",
      "out = (Wq · xq) ⊙ weight_scale ⊙ pertoken_scale，融合反量化，回到 x.dtype"]),
]

y = 108
sh = 138
gap = 26
for fill, stroke, title, lines in stages:
    box(bx, y, bw, sh, fill, stroke, 12, 2)
    text(bx + 22, y + 32, title, 16, "start", "#0f172a", "bold")
    for i, ln in enumerate(lines):
        text(bx + 30, y + 62 + i * 26, ln, 12, "start", "#334155", mono=True)
    if y + sh + gap < 108 + 5 * (sh + gap):
        varrow(cx, y + sh, y + sh + gap, "#475569", 2.2)
    y += sh + gap

text(cx, y + 4, "npu_dynamic_quant / npu_quant_matmul 由 torch_npu 提供，真实量化 GEMM 需 NPU/CANN，host 不真跑",
     13, "middle", "#b91c1c", "bold")

L.append('</svg>')
open("fig27-2-fullchain.svg", "w").write('\n'.join(L))
print("wrote fig27-2-fullchain.svg")
