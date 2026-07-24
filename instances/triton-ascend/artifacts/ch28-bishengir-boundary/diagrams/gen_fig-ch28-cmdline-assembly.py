#!/usr/bin/env python3
"""state-table 模板改造:按 metadata 开关条件拼命令行(compiler.py:L310-L459,
910_95 分支)。行=9 个考察的开关,列=取值/判定/落地参数;判定按语义上色
(命中→蓝色,跳过→灰色)。`enable_auto_bind_sub_block`(用户开关)与
`auto_tile_and_bind_subblock`(模块读到的值,compiler.py:L218)拆成相邻两行——
前者 None 时"跳过"(不独立落地),实际落地参数取后者的值,避免两个 metadata 键
被合并成一行、让"9 考察"数不出行、"3 跳过"对不上标记。底部脚注补全局统计口径
(9 考察/6 落地/约 30 个开关)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "按 metadata 开关条件拼命令行 — 910_95 分支(compiler.py:L310-L459)"
SUBTITLE = "None 开关整行跳过,只有显式设值才 append 进 cmd_list"
COLS = ["本例取值", "判定", "落地参数"]
ROWS = [
    ("get_common → target", "Ascend910B", "恒加", "--target=Ascend910B", "emit"),
    ("multibuffer", "2", "is not None → 加", "--enable-auto-multi-buffer=2", "emit"),
    ("disable_tightly_coupled_buffer_reuse", "False", "假值 → 跳过", "(无)", "skip"),
    ("enable_auto_bind_sub_block\n(用户开关)", "None", "is None →\n跳过,改用模块值", "(无,取下一行)", "skip"),
    ("auto_tile_and_bind_subblock\n(模块读到的值)", "True", "用户为 None →\n采用此值", "--enable-auto-bind-sub-block=True", "emit"),
    ("sync_solver", "None", "is None → 跳过", "(无)", "skip"),
    ("unit_flag", "1", "is not None → 加", "--enable-hivm-unit-flag-sync=1", "emit"),
    ("enable_vf_fusion", "True", "真值 → 裸 flag", "--enable-vf-fusion", "emit"),
    ("bitcodes(循环)", "libdevice.bc", "逐个 → 加", "--link-aicore-bitcode=libdevice.bc", "emit"),
]
COLOR = {"emit": ("#eff6ff", "#1e40af"), "skip": ("#f8fafc", "#94a3b8")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 300, 320, 58, 34, 100, 32
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 70
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>',
     f'<text x="{PAD}" y="{TOP-14}" font-family="sans-serif" font-size="12" '
     f'font-weight="bold" fill="#334155">开关(metadata 键)</text>']

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, (label, val, decision, emitted, kind) in enumerate(ROWS):
    ry = row_y[i]
    fill, stroke = COLOR[kind]
    L.append(f'<rect x="{PAD}" y="{ry+4}" width="{LABEL_W-8}" height="{ROW_H-8}" rx="4" '
              'fill="#f1f5f9" stroke="#94a3b8"/>')
    label_lines = label.split("\n")
    ln0 = len(label_lines)
    ly0 = ry + ROW_H / 2 - (ln0 - 1) * 8 + 4
    for k, line in enumerate(label_lines):
        L.append(f'<text x="{PAD+16}" y="{ly0+k*15}" font-family="monospace" '
                  f'font-size="12" font-weight="bold" fill="#0f172a">{esc(line)}</text>')
    cells = [val, decision, emitted]
    for j, cell in enumerate(cells):
        cx = col_x[j]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        lines = cell.split("\n")
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="monospace" font-size="11.5" fill="{stroke}">{esc(line)}</text>')

foot_top = TOP + HEADER_H + ROW_H * len(ROWS) + 22
L.append(f'<rect x="{PAD}" y="{foot_top}" width="{w-PAD*2}" height="46" rx="6" '
          'fill="#fefce8" stroke="#ca8a04"/>')
L.append(f'<text x="{PAD+16}" y="{foot_top+29}" font-family="sans-serif" '
          f'font-size="12.5" fill="#854d0e">'
          f'{esc("本例 9 个开关考察、6 个参数落地；910_95 分支全体约 30 个条件开关(compiler.py:L312-L446)")}'
          f'</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch28-cmdline-assembly.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
