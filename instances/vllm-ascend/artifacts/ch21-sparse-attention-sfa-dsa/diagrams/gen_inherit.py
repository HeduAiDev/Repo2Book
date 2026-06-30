#!/usr/bin/env python3
"""ch21 SFA / DSA 继承对照：复用 vLLM MLA vs 自起一套，共用 DeviceOperator 底座。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1340, 720
TXT = "#1e293b"
SUB = "#64748b"
SFA = "#1d4ed8"   # SFA 蓝
DSA = "#b45309"   # DSA 橙
VLLM = "#0f766e"  # vLLM 基类 青
ASC = "#7c3aed"   # 昇腾自有 紫
DEV = "#475569"   # 门面 灰

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 12 10" refX="11" refY="5" markerWidth="11" markerHeight="9" orient="auto"><path d="M0,0 L12,5 L0,10 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">SFA 复用 vLLM MLA，DSA 自起一套——两条线都「建在 MLA 之上」</text>')

# 顶部横幅
L.append(f'<rect x="290" y="62" width="760" height="40" rx="20" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>')
L.append(f'<text x="{W/2}" y="87" text-anchor="middle" font-size="15" font-weight="bold" fill="#dc2626">vLLM 主干无对位后端 —— 这是插件「加法式扩展」出的能力（ch08 母题在注意力子系统的再现），不是顶替某个内核</text>')


def box(cx, cy, w, h, fill, stroke, lines, fw="bold", sw=1.7):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 5) + s * 0.34
        col = stroke if i == 0 else TXT
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw if i==0 else "normal"}" fill="{col}">{esc(t)}</text>')


def inherit(cx, y1, y2, color):
    # 子类 → 基类（继承指向上方）
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#ar)"/>')
    L.append(f'<text x="{cx+12}" y="{(y1+y2)/2+4}" font-size="12" fill="{color}">继承</text>')


# ===== 左：SFA =====
LX = 340
L.append(f'<text x="{LX}" y="138" text-anchor="middle" font-size="18" font-weight="bold" fill="{SFA}">SFA · 直接复用 MLA 全套</text>')
L.append(f'<text x="{LX}" y="160" text-anchor="middle" font-size="12.5" fill="{SUB}">ch18 路由 (use_mla,use_sparse,use_compress)=(T,T,F)</text>')

# vLLM 基类（上）
box(LX-160, 185, 300, 50, "#ecfeff", VLLM, [("MLACommonMetadataBuilder", 13.5), ("（vLLM MLA 元数据 builder）", 11.5)])
box(LX+170, 185, 280, 50, "#ecfeff", VLLM, [("MLAAttentionImpl", 14), ("（vLLM MLA 实现基类）", 11.5)])
# SFA 子类（下）
box(LX-160, 320, 300, 52, "#eff6ff", SFA, [("AscendSFAMetadataBuilder", 13.5), ("复用 MLA 元数据装配", 11.5)])
box(LX+170, 320, 280, 52, "#eff6ff", SFA, [("AscendSFAImpl", 14), ("叠两段式稀疏选择", 11.5)])
inherit(LX-160, 320, 237, VLLM)
inherit(LX+170, 320, 237, VLLM)

# ===== 右：DSA =====
RX = 1000
L.append(f'<text x="{RX}" y="138" text-anchor="middle" font-size="18" font-weight="bold" fill="{DSA}">DSA · 自起一套，内联 MLA prolog</text>')
L.append(f'<text x="{RX}" y="160" text-anchor="middle" font-size="12.5" fill="{SUB}">ch18 路由 (use_mla,use_sparse,use_compress)=(T,F,T)</text>')

box(RX-165, 185, 300, 50, "#f0fdfa", VLLM, [("AttentionMetadataBuilder", 13.5), ("（vLLM 通用 builder，非 MLA）", 11.5)])
box(RX+175, 185, 270, 50, "#f5f3ff", ASC, [("DSAAttentionImpl", 14), ("（昇腾自有 abstract.py）", 11.5)])
box(RX-165, 320, 300, 52, "#fff7ed", DSA, [("AscendDSAMetadataBuilder", 13.5), ("自带 prefill/decode 元数据", 11.5)])
box(RX+175, 320, 270, 52, "#fff7ed", DSA, [("AscendDSAImpl", 14), ("内联 wq_a/wq_b/wkv 低秩 prolog", 11)])
inherit(RX-165, 320, 237, VLLM)
inherit(RX+175, 320, 237, ASC)

# 中缝分隔
L.append(f'<line x1="{W/2}" y1="175" x2="{W/2}" y2="400" stroke="#e2e8f0" stroke-width="1.4" stroke-dasharray="5 5"/>')

# ===== 共用底座 DeviceOperator =====
by = 470
box(W/2, by, 880, 110, "#f8fafc", DEV,
    [("共用底座 · DeviceOperator 设备算子门面（device_op.py）", 16),
     ("DeviceOperator = get_device_adaptor()：import 期按 AscendDeviceType 选 BaseDeviceAdaptor(A2/A3) 或 A5DeviceAdaptor", 12.5),
     ("封装 reshape_and_cache / 稀疏内核选择 / KV cache 解包 —— 注意力各章（ch19/ch20/ch21）写一份设备无关控制流", 12.5)])
# 两条 impl 连到门面
L.append(f'<line x1="{LX+170}" y1="372" x2="{LX+170}" y2="430" stroke="{DEV}" stroke-width="1.8"/>')
L.append(f'<line x1="{LX+170}" y1="430" x2="{W/2}" y2="430" stroke="{DEV}" stroke-width="1.8"/>')
L.append(f'<line x1="{RX+175}" y1="372" x2="{RX+175}" y2="430" stroke="{DEV}" stroke-width="1.8"/>')
L.append(f'<line x1="{RX+175}" y1="430" x2="{W/2}" y2="430" stroke="{DEV}" stroke-width="1.8"/>')
L.append(f'<line x1="{W/2}" y1="430" x2="{W/2}" y2="{by}" stroke="{DEV}" stroke-width="2" marker-end="url(#ar)"/>')

# 底注
L.append(f'<text x="{W/2}" y="640" text-anchor="middle" font-size="13.5" fill="{SUB}">两条线分歧在「复用 vs 自起一套」，但都以 ch20 的 MLA 低秩 KV 压缩为底，再叠一层「只对 top-k 算注意力」的稀疏选择。</text>')

L.append('</svg>')
open("inherit.svg", "w").write('\n'.join(L))
print("wrote inherit.svg")
