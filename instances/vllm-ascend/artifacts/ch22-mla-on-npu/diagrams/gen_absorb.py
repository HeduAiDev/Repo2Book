#!/usr/bin/env python3
"""ch20 权重吸收对照：左=朴素 MHA 解码（每步解压 KV），右=吸收后（q 投进 latent，对 kv_c 做 MQA）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1240, 660
TXT = "#1e293b"
SUB = "#64748b"
LB = "#b91c1c"   # 朴素：红（费）
RB = "#15803d"   # 吸收：绿（省）
CACHE = "#1d4ed8"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">权重吸收：把解码期对 KV 的解压，移到 query 侧并预先合并</text>')
L.append(f'<text x="{W/2}" y="68" text-anchor="middle" font-size="14.5" fill="{SUB}">KV cache 只存隐向量 kv_c（512 维）+ 解耦位置 k_pe（64 维）—— 约为满维 K/V 的 1/57</text>')

# 分隔线
L.append(f'<line x1="{W/2}" y1="92" x2="{W/2}" y2="{H-46}" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="5 5"/>')

# 列头
L.append(f'<text x="310" y="116" text-anchor="middle" font-size="17" font-weight="bold" fill="{LB}">朴素 MHA 解码 · 每步都解压</text>')
L.append(f'<text x="930" y="116" text-anchor="middle" font-size="17" font-weight="bold" fill="{RB}">吸收后 · 解码不碰满维 K/V</text>')


def box(cx, cy, w, h, fill, stroke, lines, fs=14.5, fw="normal"):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 4) + s * 0.34
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw}" fill="{stroke if i==0 else TXT}">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, label=None):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="2" marker-end="url(#ar)"/>')
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        L.append(f'<text x="{mx+12}" y="{my+4}" font-size="12.5" fill="#475569">{esc(label)}</text>')


# ---- 左：朴素 ----
LX = 310
box(LX, 140, 250, 46, "#eff6ff", CACHE, [("缓存 kv_c (B,Lkv=512)", 14)], fw="bold")
arrow(LX, 186, LX, 214, "W_UK 解压")
box(LX, 214, 250, 46, "#fef2f2", LB, [("k_nope (B,N,P=128)  每步重算", 13.5)])
arrow(LX, 260, LX, 288, "W_UV 解压")
box(LX, 288, 250, 46, "#fef2f2", LB, [("value (B,N,V=128)  每步重算", 13.5)])
arrow(LX, 334, LX, 366)
box(LX, 366, 260, 70, "#fff1f2", LB,
    [("MHA(q, k_nope, value)", 15), ("满维 K/V，N 头各一份", 12.5)], fw="bold")
L.append(f'<text x="{LX}" y="472" text-anchor="middle" font-size="13.5" fill="{LB}">代价：每个缓存 token 每步都解压 → 算力 + 带宽双费</text>')

# ---- 右：吸收 ----
RX = 930
box(RX, 140, 250, 46, "#eff6ff", CACHE, [("缓存 kv_c (B,Lkv=512)", 14)], fw="bold")
# q 侧
box(RX - 150, 214, 200, 46, "#f0fdf4", RB, [("q_nope (B,N,P=128)", 13.5)])
arrow(RX - 150, 260, RX - 150, 300, "bmm(·, W_UK_T)")
box(RX - 150, 300, 200, 50, "#dcfce7", RB, [("ql_nope (B,N,Lkv)", 14), ("已投进 latent 空间", 12)], fw="bold")
arrow(RX - 50, 325, RX + 60, 380)
arrow(RX, 186, RX, 380, "K=V=kv_c")
box(RX, 380, 270, 70, "#f0fdf4", RB,
    [("MQA(ql_nope, kv_c, kv_c)", 15), ("只对 512 维隐向量做注意力", 12)], fw="bold")
arrow(RX, 450, RX, 486, "_v_up_proj · W_UV")
box(RX, 486, 250, 44, "#dcfce7", RB, [("输出投回 V (B,N,V)", 13.5)])
L.append(f'<text x="{RX}" y="560" text-anchor="middle" font-size="13.5" fill="{RB}">数学等价：(q·Wᵤₖᵀ)·kv_c = q·(Wᵤₖᵀ·kv_c) = q·k_nope，但把 Wᵤₖ 吸进 query 侧</text>')

L.append(f'<text x="{W/2}" y="{H-18}" text-anchor="middle" font-size="13" fill="{SUB}">加载期 process_weights_after_loading 拆出 W_UK/W_UV、permute 成 W_UK_T；运行期 _q_proj_and_k_up_proj 一次 bmm 完成吸收</text>')

L.append('</svg>')
open("absorb.svg", "w").write('\n'.join(L))
print("wrote absorb.svg")
