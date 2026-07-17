#!/usr/bin/env python3
"""fig-ab-operand-structure (layout 模板,两面板)
A/B 操作数 fragment 的结构骨架——honesty_boundary(exp-0715-1):只画源码算式钉死的结构
(每线程元素数、『4 threads per row 沿 K 分担』),逐 lane 精确坐标一律不填,标『待核』回指 PTX ISA。

修订(figure-requests replace 第2轮):
- 不再把 M(A)/K(B) 沿轴切 4 条『每带 4 threads』的横带——那既暗示了源码未证明的 M 轴切分,
  又让线程数合计成 4x4=16,与『全 32 lane 参与』脚注矛盾。
- 改画:一行(A,M 固定)/一列(B,N 固定)沿 K=16 由 4 个线程分担、每线程持 4 个 K 元素
  (连续段长=kWidth=2),并明标『全 32 lane 各持 A 8 个 / B 4 个 f16』守恒。图内不出现合计与 32
  矛盾的线程头数:唯一的『4』= 沿 K 分担一行/一列的线程数(源码锚定事实),非任何轴的总切块数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PANEL_W = 470
PANEL_H = 500
GAP = 50
PAD = 50
TOP = 120

w = PAD * 2 + PANEL_W * 2 + GAP
h = TOP + PANEL_H + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="19" '
         f'font-weight="bold" fill="#0f172a">{esc("A/B 操作数 fragment:结构可核,精确坐标待核")}</text>')
L.append(f'<text x="{PAD}" y="76" font-family="sans-serif" font-size="13" '
         f'fill="#475569">{esc("每线程元素数由源码算式钉死(可验算);逐 lane 的精确 (row,k) 坐标权威在 PTX ISA,本仓未联网核对,不编造")}</text>')

# 每面板:16x16(A,M x K) / 16x8(B,K x N)。K 轴各由 4 线程分担、每线程 4 个 K 元素。
PANELS = [
    {
        "title": "A 操作数(opIdx=0)  16x16 (M x K)",
        "n_rows": 16, "n_cols": 16,          # 行=M,列=K
        "row_label": "M = 16", "col_label": "K = 16",
        "k_axis": "cols",                    # A:K 沿水平(列)
        "strip_span": 2,                     # 高亮 M 向 2 行(每线程 M 向 2)
        "share_line1": "同一行(M 固定)沿 K=16 由 4 个线程分担",
        "share_line2": "→ 每线程持 4 个 K 元素(连续段长 = kWidth = 2)",
        "per_thread": "每线程 8 个 f16 = [M 向 2, K 向 2*kWidth=4]",
        "conserve": "全 32 lane 都参与:32 x 8 = 256 守恒",
        "ptx": "逐 lane (row,K) 精确坐标:待核 PTX ISA #mma-16816-a-f16",
        "fill": "#dbeafe", "strip_fill": "#93c5fd", "stroke": "#2563eb",
    },
    {
        "title": "B 操作数(opIdx=1)  16x8 (K x N)",
        "n_rows": 16, "n_cols": 8,           # 行=K,列=N
        "row_label": "K = 16", "col_label": "N = 8",
        "k_axis": "rows",                    # B:K 沿竖直(行)
        "strip_span": 1,                     # 高亮 N 向 1 列(每线程 N 向 1)
        "share_line1": "同一列(N 固定)沿 K=16 由 4 个线程分担",
        "share_line2": "→ 每线程持 4 个 K 元素(连续段长 = kWidth = 2)",
        "per_thread": "每线程 4 个 f16 = [K 向 2*kWidth=4, N 向 1]",
        "conserve": "全 32 lane 都参与:32 x 4 = 128 守恒",
        "ptx": "逐 lane (K,N) 精确坐标:待核 PTX ISA #mma-16816-b-f16",
        "fill": "#fef3c7", "strip_fill": "#fcd34d", "stroke": "#b45309",
    },
]

MAT_W, MAT_H = 340, 210
N_SHARE = 4   # 沿 K 分担一行/一列的线程数(源码锚定:AccelerateMatmul.cpp:L504『4 threads per row』)

for p, spec in enumerate(PANELS):
    px = PAD + p * (PANEL_W + GAP)
    L.append(f'<rect x="{px}" y="{TOP}" width="{PANEL_W}" height="{PANEL_H}" rx="10" '
              'fill="#f8fafc" stroke="#cbd5e1"/>')
    L.append(f'<text x="{px+22}" y="{TOP+32}" font-family="sans-serif" font-size="15" '
              f'font-weight="bold" fill="#0f172a">{esc(spec["title"])}</text>')

    mat_x = px + (PANEL_W - MAT_W) / 2
    mat_y = TOP + 56
    cw = MAT_W / spec["n_cols"]
    ch = MAT_H / spec["n_rows"]

    # 矩阵底：整块浅色 + 稀疏网格(给出真实 shape 的观感,不喧宾夺主)
    L.append(f'<rect x="{mat_x}" y="{mat_y}" width="{MAT_W}" height="{MAT_H}" rx="3" '
              f'fill="{spec["fill"]}" stroke="{spec["stroke"]}" stroke-width="2"/>')
    for c in range(1, spec["n_cols"]):
        gx = mat_x + c * cw
        L.append(f'<line x1="{gx}" y1="{mat_y}" x2="{gx}" y2="{mat_y+MAT_H}" '
                  f'stroke="{spec["stroke"]}" stroke-width="0.4" opacity="0.35"/>')
    for r in range(1, spec["n_rows"]):
        gy = mat_y + r * ch
        L.append(f'<line x1="{mat_x}" y1="{gy}" x2="{mat_x+MAT_W}" y2="{gy}" '
                  f'stroke="{spec["stroke"]}" stroke-width="0.4" opacity="0.35"/>')

    # 高亮一个代表性的『线程 tile 所在行/列』,并沿 K 拆成 N_SHARE 个线程段。
    if spec["k_axis"] == "cols":
        # A:K 在水平。高亮顶部 strip_span 行(M 向),沿 K 拆 4 段。
        sx, sy = mat_x, mat_y
        sw, sh = MAT_W, spec["strip_span"] * ch
        L.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" '
                  f'fill="{spec["strip_fill"]}" stroke="{spec["stroke"]}" stroke-width="1.5" opacity="0.9"/>')
        seg = MAT_W / N_SHARE
        for t in range(N_SHARE):
            tx0 = sx + t * seg
            if t > 0:
                L.append(f'<line x1="{tx0}" y1="{sy}" x2="{tx0}" y2="{sy+sh}" '
                          f'stroke="{spec["stroke"]}" stroke-width="1.8"/>')
            # kWidth=2 连续段内分:每段中点一条细虚线(2+2)
            L.append(f'<line x1="{tx0+seg/2}" y1="{sy}" x2="{tx0+seg/2}" y2="{sy+sh}" '
                      f'stroke="{spec["stroke"]}" stroke-width="0.7" stroke-dasharray="3,3" opacity="0.7"/>')
            L.append(f'<text x="{tx0+seg/2}" y="{sy+sh/2+5}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                      f'fill="#0f172a">{esc(f"线程 t{t}")}</text>')
        note_y = sy + sh + 16
        L.append(f'<text x="{mat_x+MAT_W/2}" y="{note_y}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#334155">'
                  f'{esc("↑ 这一行(M 固定)沿 K 被 t0..t3 共 4 个线程分担;其余各行同理,全 32 lane 平铺")}</text>')
    else:
        # B:K 在竖直。高亮最左 strip_span 列(N 向),沿 K 拆 4 段。
        sx, sy = mat_x, mat_y
        sw, sh = spec["strip_span"] * cw, MAT_H
        L.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" '
                  f'fill="{spec["strip_fill"]}" stroke="{spec["stroke"]}" stroke-width="1.5" opacity="0.9"/>')
        seg = MAT_H / N_SHARE
        for t in range(N_SHARE):
            ty0 = sy + t * seg
            if t > 0:
                L.append(f'<line x1="{sx}" y1="{ty0}" x2="{sx+sw}" y2="{ty0}" '
                          f'stroke="{spec["stroke"]}" stroke-width="1.8"/>')
            L.append(f'<line x1="{sx}" y1="{ty0+seg/2}" x2="{sx+sw}" y2="{ty0+seg/2}" '
                      f'stroke="{spec["stroke"]}" stroke-width="0.7" stroke-dasharray="3,3" opacity="0.7"/>')
            L.append(f'<text x="{sx+sw/2}" y="{ty0+seg/2+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" font-weight="bold" '
                      f'fill="#0f172a">{esc(f"t{t}")}</text>')
        note_y = mat_y + MAT_H + 16
        L.append(f'<text x="{mat_x+MAT_W/2}" y="{note_y}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#334155">'
                  f'{esc("← 这一列(N 固定)沿 K 被 t0..t3 共 4 个线程分担;其余各列同理,全 32 lane 平铺")}</text>')

    # 轴标签
    L.append(f'<text x="{mat_x-10}" y="{mat_y+MAT_H/2}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#64748b" '
              f'transform="rotate(-90 {mat_x-10} {mat_y+MAT_H/2})">{esc(spec["row_label"])}</text>')
    L.append(f'<text x="{mat_x+MAT_W/2}" y="{mat_y+MAT_H+34}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#64748b">{esc(spec["col_label"])}</text>')

    # 文案块
    ty = mat_y + MAT_H + 60
    L.append(f'<text x="{px+22}" y="{ty}" font-family="sans-serif" font-size="12.5" '
              f'fill="#0f172a">{esc(spec["share_line1"])}</text>')
    L.append(f'<text x="{px+22}" y="{ty+21}" font-family="sans-serif" font-size="12.5" '
              f'fill="#0f172a">{esc(spec["share_line2"])}</text>')
    L.append(f'<text x="{px+22}" y="{ty+45}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="#0f172a">{esc(spec["per_thread"])}</text>')
    L.append(f'<text x="{px+22}" y="{ty+67}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="#059669">{esc(spec["conserve"])}</text>')

    # 待核 ribbon
    rib_y = TOP + PANEL_H - 34
    L.append(f'<rect x="{px+16}" y="{rib_y}" width="{PANEL_W-32}" height="26" rx="6" '
              'fill="#fee2e2" stroke="#dc2626" stroke-width="1.2"/>')
    L.append(f'<text x="{px+PANEL_W/2}" y="{rib_y+17.5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="#991b1b">{esc(spec["ptx"])}</text>')

cap = ("『4 threads per row』= 一行/一列沿 K 由 4 线程分担(非 M/N 轴切 4 组);"
       "元素数守恒:A=8x32=256、B=4x32=128,全 32 lane 参与;逐 lane 精确坐标不编造,回指 PTX ISA。")
L.append(f'<text x="{PAD}" y="{h-16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(cap)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ab-operand-structure.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
