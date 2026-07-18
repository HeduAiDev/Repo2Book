#!/usr/bin/env python3
"""fig-ch04-m2-routing-trace — state-table 模板（三例实测路由表）。
三个被调对象喂进真实 visit_Call：al.sub_vec_id 路由到 ascend_builder、tl_load 路由
到 builder、plain_python 落兜底裸调用——三条路径各命中一次。数字全部取自本章实测
（host 站位 FakeBuilder/FakeAscendBuilder，记录路由落点，不模拟 MLIR 语义）。
全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "visit_Call 实测路由表 — 三例各命中一条路径"
SUBTITLE = "Triton-Ascend 实测（host 站位 FakeBuilder/FakeAscendBuilder，记录路由落点，非 MLIR 语义）"

COLS = ["入口门\nis_builtin(fn)", "选路\nextension.is_builtin", "路由落点 _builder", "调用返回"]
ROWS = ["al.sub_vec_id", "tl_load", "plain_python"]
CELLS = {
    "al.sub_vec_id": ["通过", "命中", "ascend_builder\n（create_get_sub_vec_id）", "sub-vec-id-handle"],
    "tl_load":       ["通过", "不命中", "builder", "loaded"],
    "plain_python":  ["拦下", "不命中", "兜底裸调用\n（无 _builder）", "14"],
}
STATUS = {
    "al.sub_vec_id": ["neutral", "hit", "ascend", "neutral"],
    "tl_load":       ["neutral", "miss", "builder", "neutral"],
    "plain_python":  ["gate-block", "miss", "fallback", "neutral"],
}
COLOR = {
    "hit": ("#dcfce7", "#15803d"),
    "miss": ("#f1f5f9", "#475569"),
    "ascend": ("#dcfce7", "#15803d"),
    "builder": ("#dbeafe", "#1d4ed8"),
    "fallback": ("#fee2e2", "#b91c1c"),
    "gate-block": ("#fee2e2", "#b91c1c"),
    "neutral": (None, "#334155"),
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 150, 190, 60, 46, 100, 32
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 96
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

L.append(f'<text x="{PAD+LABEL_W-16}" y="{TOP+HEADER_H-16}" text-anchor="end" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#334155">{esc("被调对象")}</text>')

for j, name in enumerate(COLS):  # 列头（支持两行）
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("\n")
    n = len(lines)
    y0 = TOP + (HEADER_H - 6) / 2 - (n - 1) * 8 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(ln)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS[row]
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j]
        fill, stroke = COLOR[status]
        if fill:
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, ln in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="{stroke}" '
                      f'font-weight="bold">{esc(ln)}</text>')

foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("三例覆盖谓词代数三个非退化区，没有两例走同一分支：")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("al.sub_vec_id 与 tl_load 两例都发生插入点搬运（ip_synced_from_main_to_selected_builder = true）；")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+40}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("plain_python 返回 14（= 7×2）且两个 builder 都无调用，证明它走了最后一行 fn(*args, **kws) 兜底、未被塞 _builder。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch04-m2-routing-trace.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
