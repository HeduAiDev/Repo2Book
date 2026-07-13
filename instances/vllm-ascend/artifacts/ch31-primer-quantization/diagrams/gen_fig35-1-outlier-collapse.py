#!/usr/bin/env python3
"""fig35-1-outlier-collapse — 顿悟头图（落差揭示，spec §2.5 五步法）。
视觉主轴 = 刻度落差：以为每通道独享 256 级满刻度尺，per-tensor 下一个 outlier
把共用 absmax 撑到头，邻居通道可用刻度塌成一条缝（3.84 → 1.536 级）。
一图一拳：险的不是位宽，是刻度被偷。数字全部来自 figure-requests.json numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# 渲染环境缺陷：字符"量"在 synthetic-bold 下会被错渲成实心方块。
# 所有粗体文本一律经 btext() 拆 tspan，把"量"单独降为 normal 字重规避。
_BOLD_BREAK = {"量"}
def btext(s):
    parts, buf = [], ""
    for ch in s:
        if ch in _BOLD_BREAK:
            if buf:
                parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>'); buf = ""
            parts.append(f'<tspan font-weight="normal">{esc(ch)}</tspan>')
        else:
            buf += ch
    if buf:
        parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
    return "".join(parts)

# ---- 数字（全部来自 figure-requests.json 条目 numbers，带溯源）----
FULL_LEVELS = 256.0          # INT8 满量程级数
BASE_LEVELS = 3.84           # 基准 m=10.0 → 256·0.15/10.0
EXTREME_LEVELS = 1.536       # 极端 m=25.0 → 256·0.15/25.0
GAP_RATIO = 167              # 256 : 1.536 ≈ 167×

W, H = 1080, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '<marker id="al" viewBox="0 0 10 6" refX="1" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M10,0 L0,3 L10,6 Z" fill="#b91c1c"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# ---- 标题 + 副标题 ----
L.append(f'<text x="60" y="46" font-family="sans-serif" font-size="25" '
         f'fill="#1e40af">{btext("险的不是位宽，是刻度被偷")}</text>')
L.append(f'<text x="60" y="76" font-family="sans-serif" font-size="14.5" fill="#475569">'
         f'{esc("per-tensor 量化：一个 outlier 把 absmax 撑到头，邻居通道的可用刻度被挤成一条缝")}</text>')

# ---- 病因带（为什么刻度被撑满）----
cy = 100
L.append(f'<rect x="60" y="{cy}" width="960" height="40" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="78" y="{cy+25}" font-family="sans-serif" font-size="13.5" fill="#334155">'
         f'{esc("三通道 absmax  m_i = [0.15, 0.2, 10.0]：一个 outlier(10.0，极端 25.0) 钉死共用刻度 m")}'
         f'{esc("　→　有效级数  ℓ_i = 2^8 · m_i / m")}</text>')

# ---- 刻度尺落差区（视觉主轴）----
X0 = 300                 # 三条尺共同左原点
RULER_W = 660            # 满刻度 256 级对应像素宽
XEND = X0 + RULER_W
BAR_H = 50

def sliver_w(levels):    # 级数 → 像素宽（与 256 满尺严格同比例）
    return RULER_W * levels / FULL_LEVELS

rows = [
    # (y,          row_label,                 fill,       track?, levels_or_None, val_label)
    (185, "以为：每通道独立量化", "#10b981", False, None,           "256 级"),
    (285, "其实 · per-tensor 基准 m=10", "#f59e0b", True, BASE_LEVELS,    "3.84 级"),
    (385, "其实 · per-tensor 极端 m=25", "#ef4444", True, EXTREME_LEVELS, "1.536 级"),
]

for y, label, fill, track, levels, val in rows:
    # 行标签（右对齐，落在原点左侧）
    L.append(f'<text x="{X0-18}" y="{y+BAR_H/2+5}" text-anchor="end" font-family="sans-serif" '
             f'font-size="14" fill="#374151">{btext(label)}</text>')
    if track:
        # 灰色"以为的满刻度"底槽 —— 空掉的部分 = 被 outlier 偷走的刻度
        L.append(f'<rect x="{X0}" y="{y}" width="{RULER_W}" height="{BAR_H}" rx="6" '
                 f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="5 4"/>')
        sw = max(sliver_w(levels), 3.5)
        L.append(f'<rect x="{X0}" y="{y}" width="{sw:.2f}" height="{BAR_H}" rx="2" fill="{fill}"/>')
        # 可用级数标注（紧贴细缝右侧，落在空槽上，可读）
        L.append(f'<text x="{X0+sw+16:.1f}" y="{y+BAR_H/2+5}" font-family="sans-serif" '
                 f'font-size="15" fill="{fill}" font-weight="bold">{esc(val)}</text>')
        L.append(f'<text x="{X0+sw+16:.1f}" y="{y+BAR_H/2+22}" font-family="sans-serif" '
                 f'font-size="11.5" fill="#94a3b8">{esc("可用刻度（其余被偷）")}</text>')
    else:
        # 满绿：以为整条 256 级都归自己
        L.append(f'<rect x="{X0}" y="{y}" width="{RULER_W}" height="{BAR_H}" rx="6" '
                 f'fill="{fill}" stroke="#047857" stroke-width="1.5"/>')
        L.append(f'<text x="{XEND-16}" y="{y+BAR_H/2+6}" text-anchor="end" font-family="sans-serif" '
                 f'font-size="17" fill="white" font-weight="bold">{esc(val)}</text>')

# 原点 "0" 刻度
L.append(f'<line x1="{X0}" y1="175" x2="{X0}" y2="445" stroke="#94a3b8" stroke-width="1.2"/>')
L.append(f'<text x="{X0-4}" y="168" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#64748b">{esc("0")}</text>')
# 满刻度 256 参考虚线
L.append(f'<line x1="{XEND}" y1="175" x2="{XEND}" y2="445" stroke="#0f172a" stroke-width="1.4" '
         f'stroke-dasharray="6 4"/>')
L.append(f'<text x="{XEND}" y="168" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#0f172a">{esc("满刻度 256")}</text>')

# ---- 落差量化：×167 双箭头（红），量在被偷走的空档上 ----
gy = 480
sx = X0 + sliver_w(EXTREME_LEVELS)
L.append(f'<line x1="{sx:.1f}" y1="{gy}" x2="{XEND}" y2="{gy}" stroke="#b91c1c" stroke-width="2" '
         f'marker-start="url(#al)" marker-end="url(#ar)"/>')
L.append(f'<text x="{(sx+XEND)/2:.1f}" y="{gy-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="16" fill="#b91c1c">'
         f'{btext(f"×{GAP_RATIO} 落差：256 → 1.536 级，可用刻度被 outlier 偷走 {GAP_RATIO} 倍")}</text>')

# ---- 结论图注 ----
L.append(f'<text x="60" y="560" font-family="sans-serif" font-size="13" fill="#475569">'
         f'{esc("绿 = 以为每通道独享的满刻度 256 级；橙/红 = per-tensor 下邻居通道实得的可用刻度。")}</text>')
L.append(f'<text x="60" y="584" font-family="sans-serif" font-size="13" fill="#475569">'
         f'{esc("极端 outlier(25×) 下通道 0 只剩 1.536 级——不足 2 档，8-bit 名存实亡。险的不是位宽，是 per-tensor 把刻度让给了 outlier。")}</text>')
L.append(f'<text x="60" y="612" font-family="sans-serif" font-size="11.5" fill="#94a3b8">'
         f'{esc("公式：SmoothQuant §3（arXiv:2211.10438）有效量化级数 ℓ_i = 2^N·m_i/m；256×0.15/10.0=3.84，256×0.15/25.0=1.536。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-1-outlier-collapse.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
