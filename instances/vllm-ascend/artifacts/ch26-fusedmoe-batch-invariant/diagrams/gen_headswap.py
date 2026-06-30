#!/usr/bin/env python3
"""fig26-1: AscendFusedMoE 换头不换身 —— 身(灰，super().__init__ 复用) vs 三个头(红，被替换)。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
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


text(W / 2, 44, "AscendFusedMoE：继承 vLLM FusedMoE，只换三个头", 24, "middle", "#0f172a", "bold")
text(W / 2, 74, "super().__init__() 复用整副身体；昇腾只替 quant_method / runner / forward_impl", 14, "middle", "#64748b")

# 身体（灰，左侧大框）
bx, by, bw, bh = 60, 110, 540, 590
box(bx, by, bw, bh, "#f8fafc", "#cbd5e1", 14, 2, dash="6 5")
text(bx + bw / 2, by + 34, "身 · super().__init__() 继承复用，一行不改", 16, "middle", "#64748b", "bold")
body_lines = [
    "权重创建（w13_weight / w2_weight 参数与形状）",
    "路由参数（top_k / renormalize / topk_group …）",
    "EP / DP / TP 并行配置与 group 句柄",
    "weight_loader（按 TP/EP 切分加载权重）",
    "FusedMoE 调用约定 forward(hidden, router_logits)",
    "容错 / EPLB 接口（expert_map / log2phy 占位）",
]
for i, ln in enumerate(body_lines):
    text(bx + 30, by + 90 + i * 56, "• " + ln, 14, "start", "#475569")

# 三个头（红，右侧三框）
hx, hw = 700, 560
heads = [
    (140, "头①  quant_method + base_quant_method",
     ["→ AscendUnquantizedFusedMoEMethod", "apply(): select_experts → fused_experts",
      "base_quant_method 必须同步换，", "否则基类 dispatch 回上游会 raise"]),
    (320, "头②  runner",
     ["→ AscendMoERunner（只改 2 个旋钮）", "use_dp_chunking=False  ⇒ 走 forward_impl",
      "_fused_output_is_reduced：告诉基类", "MC2/AllToAll 的 finalize 已 all-reduce"]),
    (500, "头③  forward_impl（覆写）",
     ["prepare → apply → finalize 三段", "① comm.prepare 通信前置",
      "② quant_method.apply 选专家+算", "③ comm.finalize 通信后置"]),
]
for hy, title, lines in heads:
    box(hx, hy, hw, 158, "#fef2f2", "#ef4444", 12, 2)
    text(hx + 20, hy + 30, title, 15, "start", "#b91c1c", "bold")
    for i, ln in enumerate(lines):
        text(hx + 24, hy + 60 + i * 24, ln, 12.5, "start", "#7f1d1d", mono=(i == 0))
    # 换头箭头：身 → 头
    ay = hy + 79
    L.append(f'<line x1="{bx + bw}" y1="{ay}" x2="{hx}" y2="{ay}" stroke="#b91c1c" stroke-width="2.2" marker-end="url(#ar)"/>')

text(hx + hw / 2, 130, "替换（换头）", 13, "middle", "#b91c1c", "bold")
text((bx + bw + hx) / 2, 712, "身不变 = 继承；头被替 = 覆写。这是 ch23『换头不换身』在全书最大单体算子上的压力测试。",
     13, "middle", "#64748b")

L.append('</svg>')
open("fig26-1-headswap.svg", "w").write('\n'.join(L))
print("wrote fig26-1-headswap.svg")
