#!/usr/bin/env python3
"""论文精髓图重绘:arXiv:2205.14135 Fig.2 —
左:GPT-2 medium(N=1024,d=64,16 heads,batch 64)上标准注意力与 FlashAttention 的实测对比表
(GFLOPs 更多但 HBM 读写更少、runtime 更快——验证"决定 runtime 的是访存量,不是 FLOP 数");
中:分块大小 Bc 对 HBM 访问量与前向耗时的影响;右:block-sparse FlashAttention 相对稀疏度的加速。
左表数字取自 ar5iv 抓到的原始 HTML 表格(S3.F2 节点,精确值);中/右两条曲线读图近似
(原始像素图 assets/x2.png),仅用于还原趋势形状。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

FIG_ID = "paper-fig-2"
TITLE = "重绘自 arXiv:2205.14135 Fig.2"
SUBTITLE = "GPT-2 medium 实测:HBM 访存量(不是 FLOP 数)决定 runtime——FlashAttention FLOP 更多却更快"

PAD, TOP = 40, 110
PANEL_GAP = 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">']  # placeholder, viewBox 稍后回填
DEFS = ('<defs>'
        '<marker id="gray" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
        'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
        '</defs>')

body = []
body.append(f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
            f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
body.append(f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12.5" '
            f'fill="#475569">{esc(SUBTITLE)}</text>')

# ================= 面板 A:实测对比表(左) =================
tbl_x, tbl_y = PAD, TOP
row_h, label_w, col_w = 34, 130, 108
rows = [("GFLOPs", "66.6", "75.2", False),
        ("HBM 读写 (GB)", "40.3", "4.4", True),
        ("Runtime (ms)", "41.7", "7.3", True)]
tbl_w = label_w + col_w * 2
header_h = 30

body.append(f'<text x="{tbl_x+tbl_w/2}" y="{tbl_y-14}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="13.5" font-weight="bold" fill="#0f172a">'
            f'{esc("A100 实测:Standard vs FlashAttention")}</text>')
body.append(f'<text x="{tbl_x+tbl_w/2}" y="{tbl_y+2}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="10.5" fill="#64748b">{esc("GPT-2 medium, N=1024, d=64, 16 heads, batch 64")}</text>')

hy = tbl_y + 16
body.append(f'<rect x="{tbl_x+label_w}" y="{hy}" width="{col_w}" height="{header_h}" '
            'fill="#e2e8f0" stroke="#334155" stroke-width="1.2"/>')
body.append(f'<text x="{tbl_x+label_w+col_w/2}" y="{hy+header_h/2+4}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" font-weight="bold" '
            f'fill="#334155">{esc("Standard")}</text>')
body.append(f'<rect x="{tbl_x+label_w+col_w}" y="{hy}" width="{col_w}" height="{header_h}" '
            'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.2"/>')
body.append(f'<text x="{tbl_x+label_w+col_w*1.5}" y="{hy+header_h/2+4}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" font-weight="bold" '
            f'fill="#1e3a8a">{esc("FlashAttn")}</text>')

for i, (name, std, fa, highlight) in enumerate(rows):
    ry = hy + header_h + i * row_h
    body.append(f'<text x="{tbl_x+label_w-10}" y="{ry+row_h/2+4}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12" fill="#374151">{esc(name)}</text>')
    fill = "#ecfdf5" if highlight else "white"
    stroke = "#047857" if highlight else "#94a3b8"
    body.append(f'<rect x="{tbl_x+label_w}" y="{ry}" width="{col_w}" height="{row_h}" '
                f'fill="white" stroke="#94a3b8" stroke-width="1"/>')
    body.append(f'<text x="{tbl_x+label_w+col_w/2}" y="{ry+row_h/2+4}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12.5" fill="#374151">{esc(std)}</text>')
    body.append(f'<rect x="{tbl_x+label_w+col_w}" y="{ry}" width="{col_w}" height="{row_h}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if highlight else 1}"/>')
    body.append(f'<text x="{tbl_x+label_w+col_w*1.5}" y="{ry+row_h/2+4}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12.5" font-weight="{"bold" if highlight else "normal"}" '
                f'fill="{"#047857" if highlight else "#374151"}">{esc(fa)}</text>')

tbl_bottom = hy + header_h + len(rows) * row_h
note_y = tbl_bottom + 26
body.append(f'<rect x="{tbl_x}" y="{note_y}" width="{tbl_w}" height="50" rx="6" '
            'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
body.append(f'<text x="{tbl_x+tbl_w/2}" y="{note_y+20}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11" font-weight="bold" fill="#1e3a8a">{esc("FLOP 更多(75.2>66.6)反而更快")}</text>')
body.append(f'<text x="{tbl_x+tbl_w/2}" y="{note_y+38}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11" font-weight="bold" fill="#1e3a8a">{esc("因为 HBM 读写少了 9.2×")}</text>')

panelA_bottom = note_y + 50

# ================= 面板 B:分块大小的影响(中) =================
chartB_x = tbl_x + tbl_w + PANEL_GAP + 40
chartB_w, chartB_h = 260, 220
chartB_y = TOP + 30
BLOCK_SIZES = [64, 128, 256, 512]
HBM_GB = [6.7, 3.3, 1.8, 1.1]        # 左轴,读图近似(GB)
RUNTIME_MS = [6.8, 3.6, 2.5, 2.4]     # 右轴,读图近似(ms)——与左轴同刻度位置仅为并排对照,非同一物理量
axis_max = 7.0

body.append(f'<text x="{chartB_x+chartB_w/2}" y="{chartB_y-16}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
            f'fill="#0f172a">{esc("Effect of Block Size")}</text>')

bx0 = chartB_x + 10
by_bottom = chartB_y + chartB_h
n = len(BLOCK_SIZES)
step = chartB_w - 40
xs_pts = [bx0 + 10 + i * step / (n - 1) for i in range(n)]

body.append(f'<line x1="{bx0}" y1="{chartB_y}" x2="{bx0}" y2="{by_bottom}" stroke="#0f172a" stroke-width="1.4"/>')
body.append(f'<line x1="{bx0}" y1="{by_bottom}" x2="{bx0+chartB_w-20}" y2="{by_bottom}" stroke="#0f172a" stroke-width="1.4"/>')
for tick in (2, 4, 6):
    ty = by_bottom - tick / axis_max * chartB_h
    body.append(f'<line x1="{bx0-4}" y1="{ty}" x2="{bx0}" y2="{ty}" stroke="#0f172a" stroke-width="1"/>')
    body.append(f'<text x="{bx0-8}" y="{ty+4}" text-anchor="end" font-family="sans-serif" '
                f'font-size="10" fill="#059669">{tick}</text>')
body.append(f'<text x="{bx0-30}" y="{(chartB_y+by_bottom)/2}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="10" fill="#059669" transform="rotate(-90 {bx0-30} {(chartB_y+by_bottom)/2})">'
            f'{esc("HBM Accesses (GB)")}</text>')

hbm_pts = " ".join(f"{xs_pts[i]},{by_bottom - HBM_GB[i]/axis_max*chartB_h}" for i in range(n))
rt_pts = " ".join(f"{xs_pts[i]},{by_bottom - RUNTIME_MS[i]/axis_max*chartB_h}" for i in range(n))
body.append(f'<polyline points="{hbm_pts}" fill="none" stroke="#059669" stroke-width="2.4"/>')
body.append(f'<polyline points="{rt_pts}" fill="none" stroke="#2563eb" stroke-width="2.4"/>')
for i in range(n):
    hx, hy_ = xs_pts[i], by_bottom - HBM_GB[i]/axis_max*chartB_h
    rx_, ry_ = xs_pts[i], by_bottom - RUNTIME_MS[i]/axis_max*chartB_h
    body.append(f'<circle cx="{hx}" cy="{hy_}" r="3.4" fill="#059669"/>')
    body.append(f'<circle cx="{rx_}" cy="{ry_}" r="3.4" fill="#2563eb"/>')
    body.append(f'<text x="{xs_pts[i]}" y="{by_bottom+16}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="10.5" fill="#334155">{BLOCK_SIZES[i]}</text>')
label_mid_x = (xs_pts[2] + xs_pts[3]) / 2
rt_mid_val = (RUNTIME_MS[2] + RUNTIME_MS[3]) / 2
hbm_mid_val = (HBM_GB[2] + HBM_GB[3]) / 2
body.append(f'<text x="{label_mid_x}" y="{by_bottom - rt_mid_val/axis_max*chartB_h - 14}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#2563eb" '
            f'font-weight="bold">{esc("Fwd Runtime")}</text>')
body.append(f'<text x="{label_mid_x}" y="{by_bottom - hbm_mid_val/axis_max*chartB_h + 20}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#059669" '
            f'font-weight="bold">{esc("HBM Accesses")}</text>')
body.append(f'<text x="{chartB_x+chartB_w/2}" y="{by_bottom+38}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11.5" fill="#334155">{esc("Block Size")}</text>')

panelB_bottom = by_bottom + 38

# ================= 面板 C:稀疏度加速(右) =================
chartC_x = chartB_x + chartB_w + PANEL_GAP
chartC_w, chartC_h = 260, 220
chartC_y = chartB_y

body.append(f'<text x="{chartC_x+chartC_w/2}" y="{chartC_y-16}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
            f'fill="#0f172a">{esc("Sparsity Speedup")}</text>')

cx0 = chartC_x + 10
cy_bottom = chartC_y + chartC_h
c_axis_max = 190.0
body.append(f'<line x1="{cx0}" y1="{chartC_y}" x2="{cx0}" y2="{cy_bottom}" stroke="#0f172a" stroke-width="1.4"/>')
body.append(f'<line x1="{cx0}" y1="{cy_bottom}" x2="{cx0+chartC_w-20}" y2="{cy_bottom}" stroke="#0f172a" stroke-width="1.4"/>')
for tick in (50, 100, 150):
    ty = cy_bottom - tick / c_axis_max * chartC_h
    body.append(f'<line x1="{cx0-4}" y1="{ty}" x2="{cx0}" y2="{ty}" stroke="#0f172a" stroke-width="1"/>')
    body.append(f'<text x="{cx0-8}" y="{ty+4}" text-anchor="end" font-family="sans-serif" '
                f'font-size="10" fill="#334155">{tick}</text>')
# 贴轴顶放一个不旋转的小标签,避开与「100」刻度数字的几何碰撞
body.append(f'<text x="{cx0-6}" y="{chartC_y-8}" text-anchor="end" font-family="sans-serif" '
            f'font-size="10" fill="#334155">{esc("Fwd+Bwd(ms)")}</text>')

sparsity_pct = [10, 20, 40, 60, 80, 95]
dense_ms = 175.0
sparse_ms = [25, 42, 78, 112, 148, 172]
c_x_max = 100.0
c_xs = [cx0 + 10 + p/c_x_max*(chartC_w-40) for p in sparsity_pct]

dense_y = cy_bottom - dense_ms/c_axis_max*chartC_h
body.append(f'<line x1="{cx0}" y1="{dense_y}" x2="{cx0+chartC_w-20}" y2="{dense_y}" '
            'stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>')
body.append(f'<text x="{cx0+30}" y="{dense_y-10}" font-family="sans-serif" font-size="10.5" '
            f'font-weight="bold" fill="#dc2626">{esc("Dense FlashAttention")}</text>')

sparse_pts = " ".join(f"{c_xs[i]},{cy_bottom - sparse_ms[i]/c_axis_max*chartC_h}" for i in range(len(sparsity_pct)))
body.append(f'<polyline points="{sparse_pts}" fill="none" stroke="#2563eb" stroke-width="2.4"/>')
for i in range(len(sparsity_pct)):
    py = cy_bottom - sparse_ms[i]/c_axis_max*chartC_h
    body.append(f'<circle cx="{c_xs[i]}" cy="{py}" r="3.2" fill="#2563eb"/>')
body.append(f'<text x="{c_xs[3]}" y="{cy_bottom - sparse_ms[3]/c_axis_max*chartC_h + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
            f'fill="#2563eb">{esc("Block-Sparse FlashAttention")}</text>')
for p in (20, 60):
    idx = sparsity_pct.index(p) if p in sparsity_pct else None
    tx = cx0 + 10 + p/c_x_max*(chartC_w-40)
    body.append(f'<text x="{tx}" y="{cy_bottom+16}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="10.5" fill="#334155">{p}</text>')
body.append(f'<text x="{chartC_x+chartC_w/2}" y="{cy_bottom+38}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11.5" fill="#334155">{esc("% Non-Zero Blocks")}</text>')

panelC_bottom = cy_bottom + 38

w = int(chartC_x + chartC_w + 60)
h = int(max(panelA_bottom, panelB_bottom, panelC_bottom) + 70)

foot_y = h - 16
FOOT = ("左表数字取自论文 HTML 表格原值(精确);中/右两条曲线为读图近似(还原趋势形状,"
        "非逐点复刻论文数据)。")
body.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
            f'fill="#64748b">{esc(FOOT)}</text>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', DEFS,
     f'<rect width="{w}" height="{h}" fill="white"/>'] + body + ['</svg>']
out = Path(__file__).with_name(f"{FIG_ID}.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, canvas {w}x{h}")
