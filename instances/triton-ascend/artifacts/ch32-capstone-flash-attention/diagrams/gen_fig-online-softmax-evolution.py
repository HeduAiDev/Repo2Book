#!/usr/bin/env python3
"""state-table 模板改写:在线 softmax 递推,列=块(块0/块1/收尾),行=追踪变量。
高亮行 = alpha(重标定因子),块0 触发"清零初值"、块1 触发"真实缩水"。
数据来源 explainer/traces/run_online_softmax.json(host numpy 复现,与一次性物化 softmax 对拍差 0)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "在线 softmax 递推 — 两块 K/V 的 m/l/acc 演化"
SUBTITLE = "单查询 q=[1.0,0.0],sm_scale=1.0(教学取值);初值 m_i=-inf, l_i=1.0(06-fused-attention.py:L211-L212)"
COLS = ["块0  K0V0", "块1  K1V1", "收尾"]
ROW_LABELS = ["qk·scale", "m_ij (running max)", "alpha (重标定)", "l_i (分母累加)", "acc / 归一输出"]
CELLS = {
    "qk·scale": ["[1.0, 0.0]", "[2.0, 0.5]", "—"],
    "m_ij (running max)": ["1.0", "2.0", "m_i=2.0\nlogsumexp=log(l_i)=2.546"],
    "alpha (重标定)": ["0.0\nm_prev=-inf → 清零 l_i 初值 1.0", "0.3679\nrunning max 1.0→2.0,旧累加器缩水", "—"],
    "l_i (分母累加)": ["1.3679", "1.7263", "l_i=1.7263"],
    "acc / 归一输出": ["[1.0, 0.3679]", "[1.8141, 1.1353]", "acc/l_i=[1.0509, 0.6577]\n参考softmax输出=[1.0509, 0.6577]\n在线vs参考最大差=0.0"],
}
HIGHLIGHT_ROW = "alpha (重标定)"
STATUS = {"alpha (重标定)": ["init", "trigger", None]}
COLOR = {"init": ("#eff6ff", "#1d4ed8"), "trigger": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 260, 66, 36, 100, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 20
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签 + 单元格
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-10}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            fs = 12 if len(line) < 30 else 10.5
            L.append(f'<text x="{cx+(COL_W-10)/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="{fs}" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

foot_y = h - PAD + 8
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">蓝=running max 首次确立(清零陈旧初值);红=running max 被后块刷新,触发 alpha 缩水旧累加器 · '
          f'数值按源码初值与更新序 host 复现,与一次性物化 softmax 对拍验证</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-online-softmax-evolution.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
