#!/usr/bin/env python3
"""顿悟头图(§2.5 五步法)——落差揭示。
那一下:生成 1 个 token,dense 要回看全部 1,000,000 条 KV;V4 的 CSA 核注意力
只真读固定的 3,072 条(top-k 2048 + 滑窗 1024)≈ 1/326,且不随上下文再涨。
视觉主轴 = 两根同起点的条的长度落差(真实 326:1 比例);V4 那根几乎成一根发丝,
右侧一整段空白就是「根本不读」的证据。削到只剩这一个对比,不画数据流、不画模块。
全部数字来自条目 numbers(带溯源),全坐标计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# ---- 数字(全部来自 figure-requests numbers[],带 provenance)----
N_DENSE = 1000000     # explainer trace efficiency-account-27-10 params.L / dense KV 存量
K_TOPK  = 2048        # params.k
N_WIN   = 1024        # params.n_win
N_READ  = K_TOPK + N_WIN  # 3072,正文 (2048+1024)×128=393216 同源
RATIO   = round(N_DENSE / N_READ)  # 326 落差倍数

def fmt(n): return f"{n:,}"

# ---- 画布 ----
XO   = 72                 # 两根条共同的左起点
BAR_FULL = 1300           # dense 条满宽
BAR_H = 62
w = XO + BAR_FULL + 68
DENSE_Y = 150
V4_Y    = 372
h = 640

sliver_w = BAR_FULL * N_READ / N_DENSE   # 真实 326:1 → ≈4px 的发丝
sliver_w = max(sliver_w, 3.5)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
     'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

# ---- 标题 ----
L.append(f'<text x="{XO}" y="46" font-family="sans-serif" font-size="22" '
         f'font-weight="bold" fill="#1e40af">同样生成 1 个 token,要真读多少条 KV?</text>')
L.append(f'<text x="{XO}" y="76" font-family="sans-serif" font-size="14" '
         f'fill="#64748b">1M 上下文里 —— 两根条同一个起点,长度按真实 {RATIO}:1 画。看长度差,就是这一章的那一下。</text>')

# ======== 上半:朴素 dense ========
L.append(f'<text x="{XO}" y="{DENSE_Y-14}" font-family="sans-serif" font-size="15" '
         f'font-weight="bold" fill="#b91c1c">朴素做法(dense):每一步都回看全部</text>')
L.append(f'<rect x="{XO}" y="{DENSE_Y}" width="{BAR_FULL}" height="{BAR_H}" rx="5" '
         f'fill="#fecaca" stroke="#dc2626" stroke-width="2"/>')
# 条内密集刻度暗示「一整摞」
n_ticks = 52
for i in range(1, n_ticks):
    tx = XO + BAR_FULL * i / n_ticks
    L.append(f'<line x1="{tx:.1f}" y1="{DENSE_Y+6}" x2="{tx:.1f}" y2="{DENSE_Y+BAR_H-6}" '
             f'stroke="#f4a3a3" stroke-width="1"/>')
L.append(f'<text x="{XO+BAR_FULL/2}" y="{DENSE_Y+BAR_H/2+6}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="18" font-weight="bold" fill="#7f1d1d">'
         f'{esc(fmt(N_DENSE))} 条 KV,逐条回看</text>')
# 端点标注
L.append(f'<text x="{XO}" y="{DENSE_Y+BAR_H+18}" font-family="sans-serif" font-size="11.5" '
         f'fill="#94a3b8">第 1 条</text>')
L.append(f'<text x="{XO+BAR_FULL}" y="{DENSE_Y+BAR_H+18}" text-anchor="end" '
         f'font-family="sans-serif" font-size="11.5" fill="#b91c1c">读到第 {esc(fmt(N_DENSE))} 条</text>')

# ======== 落差竖直落线:dense 右端 → V4 行,点出「V4 早已读完」 ========
drop_x = XO + BAR_FULL
L.append(f'<line x1="{drop_x}" y1="{DENSE_Y+BAR_H}" x2="{drop_x}" y2="{V4_Y+BAR_H+8}" '
         f'stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="5 4"/>')

# ======== 下半:V4 CSA(发丝 + 右侧一整段空白 = 根本不读) ========
L.append(f'<text x="{XO}" y="{V4_Y-14}" font-family="sans-serif" font-size="15" '
         f'font-weight="bold" fill="#047857">V4 的 CSA 核注意力:只真读固定的一小撮</text>')
# V4 那根发丝(真实比例 ≈4px)
L.append(f'<rect x="{XO}" y="{V4_Y}" width="{sliver_w:.1f}" height="{BAR_H}" rx="1.5" '
         f'fill="#059669" stroke="#065f46" stroke-width="1"/>')
# 发丝旁一条极短提示线
L.append(f'<text x="{XO}" y="{V4_Y+BAR_H+18}" font-family="sans-serif" font-size="11.5" '
         f'fill="#047857">↑ 就这么一根(≈ 满宽的 1/{RATIO})</text>')
# 右侧「根本不读」的空白跨度虚线
notread_x1 = XO + 150
notread_x2 = drop_x - 6
ny = V4_Y + BAR_H/2
L.append(f'<line x1="{notread_x1}" y1="{ny}" x2="{notread_x2}" y2="{ny}" '
         f'stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="7 5"/>')
L.append(f'<text x="{(notread_x1+notread_x2)/2}" y="{ny-12}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" fill="#94a3b8">'
         f'这一整段(dense 还在逐条读的)—— V4 根本不碰</text>')

# ======== 放大镜:把发丝拆成 top-k 2048 + 滑窗 1024 ========
ZX, ZY, ZW, ZH = 360, V4_Y + 96, 560, 82
# 引出扇线:发丝右缘上下角 → 放大框左上/左下
L.append(f'<line x1="{XO+sliver_w:.1f}" y1="{V4_Y}" x2="{ZX}" y2="{ZY}" '
         f'stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 3"/>')
L.append(f'<line x1="{XO+sliver_w:.1f}" y1="{V4_Y+BAR_H}" x2="{ZX}" y2="{ZY+ZH}" '
         f'stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 3"/>')
L.append(f'<text x="{ZX}" y="{ZY-10}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#047857">放大这根发丝 —— 它其实是两块加起来:</text>')
seg_topk = ZW * K_TOPK / N_READ
seg_win  = ZW * N_WIN / N_READ
# top-k 段
L.append(f'<rect x="{ZX}" y="{ZY}" width="{seg_topk:.1f}" height="{ZH}" rx="4" '
         f'fill="#10b981" stroke="#065f46" stroke-width="1.5"/>')
L.append(f'<text x="{ZX+seg_topk/2:.1f}" y="{ZY+ZH/2-4}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">'
         f'top-k 选中块</text>')
L.append(f'<text x="{ZX+seg_topk/2:.1f}" y="{ZY+ZH/2+18}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="17" font-weight="bold" fill="white">'
         f'{esc(fmt(K_TOPK))} 条</text>')
# 滑窗段
wx = ZX + seg_topk
L.append(f'<rect x="{wx:.1f}" y="{ZY}" width="{seg_win:.1f}" height="{ZH}" rx="4" '
         f'fill="#6ee7b7" stroke="#065f46" stroke-width="1.5"/>')
L.append(f'<text x="{wx+seg_win/2:.1f}" y="{ZY+ZH/2-4}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" fill="#064e3b">'
         f'滑窗</text>')
L.append(f'<text x="{wx+seg_win/2:.1f}" y="{ZY+ZH/2+18}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="17" font-weight="bold" fill="#064e3b">'
         f'{esc(fmt(N_WIN))} 条</text>')
# 合计标注
L.append(f'<text x="{ZX+ZW+16}" y="{ZY+ZH/2+6}" font-family="sans-serif" font-size="16" '
         f'font-weight="bold" fill="#047857">= {esc(fmt(N_READ))} 条</text>')

# ======== 落差大徽标:≈ 1/326 ========
bx, by, bw, bh = drop_x - 250, DENSE_Y + BAR_H + 42, 250, 96
L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="12" '
         f'fill="#fef2f2" stroke="#b91c1c" stroke-width="2.5"/>')
L.append(f'<text x="{bx+bw/2}" y="{by+34}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="#b91c1c">读的条数落差</text>')
L.append(f'<text x="{bx+bw/2}" y="{by+72}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="38" font-weight="bold" fill="#b91c1c">≈ {RATIO}×</text>')

# ======== 底部:那一下的收口(钉成常数,不随上下文涨)========
fy = h - 62
L.append(f'<rect x="{XO}" y="{fy}" width="{BAR_FULL}" height="46" rx="8" '
         f'fill="#ecfdf5" stroke="#047857" stroke-width="1.8"/>')
L.append(f'<text x="{XO+18}" y="{fy+29}" font-family="sans-serif" font-size="14.5" '
         f'font-weight="bold" fill="#065f46">'
         f'那一下:上下文再变长,红条一直伸;绿条被 top-k+滑窗钉死,始终 {esc(fmt(N_READ))} 条 —— '
         f'真读的条数与序列长 L 解耦。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-0-epiphany-pinned-readout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
