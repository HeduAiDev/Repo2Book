#!/usr/bin/env python3
"""fig-m8-granularity-landed：三档调度粒度对比表（决策粒度 x 是否落地），
自定义三行卡片式对比（非严格 state-table 网格，因每行内容长度差异大）。
行内文本手工预分行（非字符截断），保证列宽内不溢出。
数据取自 explainer.json m8 figure_specs.numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROWS = [
    {
        "name": "论文 Algorithm 1",
        "gran_lines": ["逐请求 · 逐位置置信度", "c_k 贪心早停"],
        "landed": "仅论文侧，未落地",
        "detail_lines": ["paper.md §五", "dossier honest_gaps 第 2 条"],
        "color": ("#f8fafc", "#94a3b8", "#64748b"),
        "dashed": True,
    },
    {
        "name": "本 PR #46995",
        "gran_lines": ["每步固定", "N=num_speculative_steps，无早停"],
        "landed": "已落地",
        "detail_lines": ["speculator.py:L74-L113", "for i in range(n_spec)", "无提前退出分支"],
        "color": ("#dcfce7", "#15803d", "#15803d"),
        "dashed": False,
    },
    {
        "name": "上游最接近",
        "gran_lines": ["num_speculative_tokens_", "per_batch_size 按批大小", "区间静态查表"],
        "landed": "已落地（粗粒度雏形）",
        "detail_lines": ["config/speculative.py:", "L164-L169"],
        "color": ("#dbeafe", "#1d4ed8", "#1d4ed8"),
        "dashed": False,
    },
]

W, PAD, TOP = 1280, 40, 100
NAME_W, GRAN_W, LAND_W = 190, 400, 230
DETAIL_W = W - 2*PAD - NAME_W - GRAN_W - LAND_W
ROW_H = 116
HEADER_H = 34

H = TOP + HEADER_H + ROW_H * 3 + 70
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">三档调度粒度：决策粒度 × 是否落地</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12.5" fill="#475569">'
     f'同一坐标摆开三种调度机制——读者勿把论文 Algorithm 1 与本 PR 实际行为混为一谈</text>']

HEADERS = [("方案", NAME_W), ("决策粒度", GRAN_W), ("落地状态", LAND_W), ("依据（源码/文档锚点）", DETAIL_W)]
hx = PAD
for name, wcol in HEADERS:
    L.append(f'<rect x="{hx}" y="{TOP}" width="{wcol-6}" height="{HEADER_H}" rx="4" '
             f'fill="#334155"/>')
    L.append(f'<text x="{hx+10}" y="{TOP+HEADER_H/2+5}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="white">{esc(name)}</text>')
    hx += wcol

for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H + 10
    bg, border, text_color = row["color"]
    dash = ' stroke-dasharray="6,4"' if row["dashed"] else ""
    x = PAD
    row_w = W - 2 * PAD - 6
    box_h = ROW_H - 16
    L.append(f'<rect x="{x}" y="{ry}" width="{row_w}" height="{box_h}" rx="6" '
             f'fill="{bg}" stroke="{border}" stroke-width="2"{dash}/>')

    cx = x + 12
    L.append(f'<text x="{cx}" y="{ry+box_h/2+5}" font-family="sans-serif" font-size="13.5" '
             f'font-weight="bold" fill="{text_color}">{esc(row["name"])}</text>')

    cx = x + NAME_W
    lines = row["gran_lines"]
    ly0 = ry + box_h/2 - (len(lines)-1)*9 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{ly0+k*18}" font-family="sans-serif" font-size="12.5" '
                 f'fill="{text_color}">{esc(ln)}</text>')

    cx = x + NAME_W + GRAN_W
    badge_w = LAND_W - 16
    L.append(f'<rect x="{cx}" y="{ry+box_h/2-16}" width="{badge_w}" height="32" rx="6" '
             f'fill="white" stroke="{border}" stroke-width="1.4"/>')
    L.append(f'<text x="{cx+badge_w/2}" y="{ry+box_h/2+5}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" font-weight="bold" '
             f'fill="{text_color}">{esc(row["landed"])}</text>')

    cx = x + NAME_W + GRAN_W + LAND_W
    dlines = row["detail_lines"]
    dly0 = ry + box_h/2 - (len(dlines)-1)*8 + 4
    for k, ln in enumerate(dlines):
        L.append(f'<text x="{cx}" y="{dly0+k*16}" font-family="sans-serif" font-size="10.5" '
                 f'fill="{text_color}">{esc(ln)}</text>')

foot_y = TOP + HEADER_H + ROW_H * 3 + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">虚线框=仅论文侧描述，尚无对应实现代码；实线框=已合入 vLLM 主线 PR #46995 的实际行为。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m8-granularity-landed.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
