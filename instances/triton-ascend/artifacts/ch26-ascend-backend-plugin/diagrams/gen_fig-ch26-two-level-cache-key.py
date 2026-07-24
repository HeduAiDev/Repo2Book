#!/usr/bin/env python3
"""fig-ch26-two-level-cache-key：编译缓存命中要求 backend_hash 与 options_hash 两把锁
同时对上；arch 变化同时拨动两把锁，num_warps 只拨动 options 一把。
state-table 模板改造为「场景×字段」矩阵，changed/unchanged 语义上色。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "两级缓存键正交合取：backend_hash × options_hash 两把锁"
SUBTITLE = "third_party/ascend/backend/compiler.py:L971-974(backend_hash) + L810-812(options_hash) —— 两锁皆合才复用编译产物"

COL_LABELS = ["场景", "backend_hash = str(self.target)", "options_hash 前 8 位", "缓存命中？"]
ROWS = [
    {
        "label": "基准\n(910B, num_warps=32)",
        "backend_hash": "GPUTarget(backend='npu',\n arch='Ascend910B', warp_size=0)",
        "options_hash": "27b4ce00",
        "hit": "—（基准）",
        "backend_status": "base",
        "options_status": "base",
        "hit_status": "base",
    },
    {
        "label": "改 arch\n910B → 950",
        "backend_hash": "GPUTarget(backend='npu',\n arch='Ascend950', warp_size=0)",
        "options_hash": "7f55aa82",
        "hit": "miss\n（两指纹都变）",
        "backend_status": "changed",
        "options_status": "changed",
        "hit_status": "miss",
    },
    {
        "label": "只改 num_warps\n32 → 16",
        "backend_hash": "GPUTarget(backend='npu',\n arch='Ascend910B', warp_size=0)\n[不变]",
        "options_hash": "6c652952",
        "hit": "miss\n（只 options 指纹变）",
        "backend_status": "unchanged",
        "options_status": "changed",
        "hit_status": "miss",
    },
]

COLOR = {
    "base": ("#f8fafc", "#475569", "#334155"),
    "changed": ("#fee2e2", "#b91c1c", "#7f1d1d"),
    "unchanged": ("#dcfce7", "#15803d", "#14532d"),
    "miss": ("#fee2e2", "#b91c1c", "#7f1d1d"),
}

LABEL_W, C1_W, C2_W, C3_W = 200, 340, 190, 220
COL_W = [LABEL_W, C1_W, C2_W, C3_W]
HEADER_H = 42
ROW_H = 96
PAD, TOP = 40, 128

w = PAD * 2 + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + 110

col_x = [PAD]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 表头
for j, name in enumerate(COL_LABELS):
    x = col_x[j]
    cw = COL_W[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-4}" height="{HEADER_H-6}" rx="4" '
              'fill="#334155" stroke="#1e293b" stroke-width="1"/>')
    L.append(f'<text x="{x+(cw-4)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

row_top0 = TOP + HEADER_H
for i, row in enumerate(ROWS):
    ry = row_top0 + i * ROW_H
    # 场景 label（无语义色，纯行标签底）
    lx = col_x[0]
    L.append(f'<rect x="{lx}" y="{ry+4}" width="{COL_W[0]-8}" height="{ROW_H-8}" rx="6" '
              'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
    lines = row["label"].split("\n")
    y0 = ry + ROW_H / 2 - (len(lines) - 1) * 9 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{lx+(COL_W[0]-8)/2}" y="{y0+k*18}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="#334155">{esc(line)}</text>')

    for j, (key, status_key) in enumerate([("backend_hash", "backend_status"),
                                            ("options_hash", "options_status"),
                                            ("hit", "hit_status")], start=1):
        x = col_x[j]
        cw = COL_W[j]
        status = row[status_key]
        fill, stroke_c, text_fill = COLOR[status]
        L.append(f'<rect x="{x}" y="{ry+4}" width="{cw-8}" height="{ROW_H-8}" rx="6" '
                  f'fill="{fill}" stroke="{stroke_c}" stroke-width="1.8"/>')
        lines = row[key].split("\n")
        mono = key != "hit"
        ff = "monospace" if mono else "sans-serif"
        fs = 11 if mono else 12.5
        y0 = ry + ROW_H / 2 - (len(lines) - 1) * 9 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{x+(cw-8)/2}" y="{y0+k*18}" text-anchor="middle" '
                      f'font-family="{ff}" font-size="{fs}" fill="{text_fill}" '
                      f'font-weight="bold">{esc(line)}</text>')

table_bottom = row_top0 + ROW_H * len(ROWS)
legend_y = table_bottom + 26
L.append(f'<rect x="{PAD}" y="{legend_y-14}" width="14" height="14" rx="3" '
          f'fill="{COLOR["unchanged"][0]}" stroke="{COLOR["unchanged"][1]}"/>')
L.append(f'<text x="{PAD+20}" y="{legend_y-3}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("不变（与基准逐字相同）")}</text>')
L.append(f'<rect x="{PAD+220}" y="{legend_y-14}" width="14" height="14" rx="3" '
          f'fill="{COLOR["changed"][0]}" stroke="{COLOR["changed"][1]}"/>')
L.append(f'<text x="{PAD+240}" y="{legend_y-3}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("变化（与基准不同 → 该锁不合）")}</text>')

note_lines = [
    "arch 同时出现在两把锁里（parse_options 用 setdefault 把 target.arch 复制进 options.arch）",
    "——只有它能一次拨动两把锁；num_warps 只在 options 里，backend_hash 逐字节不变。",
]
for i, line in enumerate(note_lines):
    L.append(f'<text x="{PAD}" y="{legend_y+24+i*20}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch26-two-level-cache-key.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w} h={h}")
