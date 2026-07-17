#!/usr/bin/env python3
"""fig-m1-base-backend-contract-table：BaseBackend 是一张"填空表"——
6 个 @abstractmethod 必填 + 2 个可覆写钩子选填，NVIDIA 与 AMD 各填一份。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "BaseBackend 契约面：一张填空表，两份落地"
SUBTITLE = "6 个 @abstractmethod 必填 + 2 个可覆写钩子选填 —— NVIDIA、AMD 各填一份，编译总控 compile() 一行不改"

REQUIRED_ROWS = [
    ("supports_target", "已填", "已填"),
    ("hash", "已填", "已填"),
    ("parse_options", "已填 → CUDAOptions", "已填 → HIPOptions"),
    ("add_stages", "已填(ttir/ttgir/llir/ptx/cubin)", "已填(ttir/ttgir/llir/amdgcn/hsaco)"),
    ("load_dialects", "已填(nvidia dialect)", "已填(amd dialect)"),
    ("get_module_map", "已填", "已填"),
]
OPTIONAL_ROWS = [
    ("get_attrs_descriptor", "用默认(AttrsDescriptor)", "覆写 → HIPAttrsDescriptor"),
    ("compute_spec_key", "用默认", "用默认"),
]

NAME_W, COL_W, ROW_H, HEADER_H, GROUP_GAP = 300, 280, 34, 34, 30
PAD, TOP = 40, 128
LEGEND_H = 48

n_req = len(REQUIRED_ROWS)
n_opt = len(OPTIONAL_ROWS)
table_w = NAME_W + COL_W * 2
w = PAD * 2 + table_w
group1_h = HEADER_H + n_req * ROW_H
group2_top = TOP + HEADER_H + group1_h + GROUP_GAP
group2_h = HEADER_H + n_opt * ROW_H
h = group2_top + group2_h + LEGEND_H + PAD + 30

col_x = [PAD + NAME_W, PAD + NAME_W + COL_W]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

# subtitle wraps if long — split manually at natural point
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')


def draw_group(top, rows, group_label, badge_text, badge_color):
    header_y = top
    body_top = top + HEADER_H
    # group label chip
    L.append(f'<rect x="{PAD}" y="{header_y}" width="{NAME_W}" height="{HEADER_H-6}" rx="4" '
              f'fill="{badge_color[0]}" stroke="{badge_color[1]}" stroke-width="1.5"/>')
    L.append(f'<text x="{PAD+14}" y="{header_y+(HEADER_H-6)/2+5}" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{badge_color[1]}">{esc(group_label)} · {esc(badge_text)}</text>')
    for j, colname in enumerate(["CUDABackend（NVIDIA）", "HIPBackend（AMD）"]):
        x = col_x[j]
        L.append(f'<rect x="{x}" y="{header_y}" width="{COL_W-6}" height="{HEADER_H-6}" rx="4" '
                  'fill="#334155" stroke="#1e293b" stroke-width="1"/>')
        L.append(f'<text x="{x+(COL_W-6)/2}" y="{header_y+(HEADER_H-6)/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(colname)}</text>')
    for i, (name, nv, amd) in enumerate(rows):
        ry = body_top + i * ROW_H
        row_fill = "#f8fafc" if i % 2 == 0 else "white"
        L.append(f'<rect x="{PAD}" y="{ry}" width="{table_w}" height="{ROW_H}" '
                  f'fill="{row_fill}" stroke="#e2e8f0" stroke-width="1"/>')
        L.append(f'<text x="{PAD+14}" y="{ry+ROW_H/2+4}" font-family="monospace" '
                  f'font-size="12.5" fill="#0f172a">{esc(name)}</text>')
        for j, val in enumerate((nv, amd)):
            x = col_x[j]
            L.append(f'<text x="{x+(COL_W-6)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" fill="#1e293b">{esc(val)}</text>')
    return body_top + len(rows) * ROW_H


bottom1 = draw_group(TOP, REQUIRED_ROWS, "必填", "@abstractmethod × 6",
                      ("#dbeafe", "#1e40af"))
bottom2 = draw_group(group2_top, OPTIONAL_ROWS, "选填", "可覆写钩子 × 2（默认=pass）",
                      ("#fef3c7", "#b45309"))

legend_y = bottom2 + 24
L.append(f'<text x="{PAD}" y="{legend_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">已落地 2 份：CUDABackend（third_party/nvidia）+ HIPBackend（third_party/amd）——姊妹篇 ascend 后端是第 3 份。</text>')
L.append(f'<text x="{PAD}" y="{legend_y+18}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">可覆写钩子默认实现即一行 pass（python/triton/backends/compiler.py:L100-L102）。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-base-backend-contract-table.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
