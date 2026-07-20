#!/usr/bin/env python3
"""fig-ch32-parallel-gate: 并行填充要连过三道门(构造期阈值、运行期批量、无投机),
第一道门在 max_num_seqs <= 128 的部署里直接把这条分支变成死代码。
template: flow"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1220, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("并行填充要连过三道门,第一道门在 max_num_seqs <= 128 时就把它变成死代码")}</text>')

PAD = 40
TOP = 80
GATE_W, GATE_H = 280, 90
GAP = 60
GATES = [
    ("① 构造期阈值", "128 < max_num_seqs ?", "vllm/v1/structured_output/__init__.py:L61-62"),
    ("② 运行期批量", "len(ids) > 128 ?", "同上,threshold=128"),
    ("③ 无投机", "max_num_spec_tokens == 0 ?", "行间有推进/回滚顺序依赖时不可并发"),
]
gx = []
for i, (title, cond, prov) in enumerate(GATES):
    x = PAD + i * (GATE_W + GAP)
    gx.append(x)
    L.append(f'<rect x="{x}" y="{TOP}" width="{GATE_W}" height="{GATE_H}" rx="10" '
              f'fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>')
    L.append(f'<text x="{x+GATE_W/2}" y="{TOP+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#713f12">{esc(title)}</text>')
    L.append(f'<text x="{x+GATE_W/2}" y="{TOP+48}" text-anchor="middle" font-family="monospace" '
              f'font-size="12.5" fill="#422006">{esc(cond)}</text>')
    L.append(f'<text x="{x+GATE_W/2}" y="{TOP+70}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#a16207">{esc(prov)}</text>')
    if i < len(GATES) - 1:
        L.append(f'<line x1="{x+GATE_W}" y1="{TOP+GATE_H/2}" x2="{x+GATE_W+GAP-6}" y2="{TOP+GATE_H/2}" '
                  f'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')
        L.append(f'<text x="{x+GATE_W+GAP/2}" y="{TOP+GATE_H/2-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#059669">{esc("是")}</text>')

# 结论:全过 -> 并行;任一不过 -> 串行
OUT_Y = TOP + GATE_H + 70
BW, BH = 200, 56
para_x = gx[-1] + GATE_W - BW
serial_x = PAD
L.append(f'<rect x="{para_x}" y="{OUT_Y}" width="{BW}" height="{BH}" rx="10" '
          f'fill="#dcfce7" stroke="#16a34a" stroke-width="2.5"/>')
L.append(f'<text x="{para_x+BW/2}" y="{OUT_Y+34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#166534">{esc("并行填充")}</text>')
L.append(f'<line x1="{gx[-1]+GATE_W/2}" y1="{TOP+GATE_H}" x2="{para_x+BW/2}" y2="{OUT_Y}" '
          f'stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<rect x="{serial_x}" y="{OUT_Y}" width="{BW}" height="{BH}" rx="10" '
          f'fill="#f1f5f9" stroke="#64748b" stroke-width="2"/>')
L.append(f'<text x="{serial_x+BW/2}" y="{OUT_Y+34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#334155">{esc("串行填充")}</text>')
for x in gx:
    L.append(f'<line x1="{x+GATE_W*0.25}" y1="{TOP+GATE_H}" x2="{serial_x+BW/2}" y2="{OUT_Y}" '
              f'stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="4,3" marker-end="url(#a)"/>')
L.append(f'<text x="{(serial_x+BW+PAD)/2+40}" y="{OUT_Y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("任一门为否")}</text>')

# 常量注记条
CONST_Y = OUT_Y + BH + 30
L.append(f'<rect x="{PAD}" y="{CONST_Y}" width="{W-2*PAD}" height="42" rx="8" '
          f'fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{W/2}" y="{CONST_Y+26}" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#3730a3">{esc("fill_bitmask_parallel_threshold=128 / fill_bitmask_parallel_batch_size=16 / 线程数=max(1,min(cpu_count//2,8));取证机 cpu_count=4 -> 2 线程")}</text>')

# --- 4 个 cfg 的过门情况表 ---
TAB_Y = CONST_Y + 70
L.append(f'<text x="{PAD}" y="{TAB_Y}" font-family="sans-serif" font-size="13.5" font-weight="bold" '
          f'fill="#1e293b">{esc("四个配置各自过门情况")}</text>')

COLS = ["配置", "max_num_seqs", "门①构造期", "本步请求数", "门②运行期", "门③无投机", "结果"]
ROWS = [
    ["cfg1", "128", "否(128<128 假)", "128", "—", "—", "串行"],
    ["cfg2", "256", "是", "128", "否(128>128 假)", "—", "串行"],
    ["cfg3", "256", "是", "256", "是", "是(num_spec=0)", "并行,16 任务x16 请求"],
    ["cfg4", "256", "是", "256", "是", "否(num_spec=2)", "串行"],
]
COL_W = [70, 110, 150, 110, 160, 160, 210]
row_h = 34
tx0 = PAD
ty0 = TAB_Y + 16
# header
cx = tx0
for j, name in enumerate(COLS):
    L.append(f'<rect x="{cx}" y="{ty0}" width="{COL_W[j]-4}" height="{row_h}" fill="#3b82f6"/>')
    L.append(f'<text x="{cx+(COL_W[j]-4)/2}" y="{ty0+row_h/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" fill="white">{esc(name)}</text>')
    cx += COL_W[j]
for i, row in enumerate(ROWS):
    ry = ty0 + row_h * (i + 1)
    is_result_row = row[-1].startswith("并行")
    cx = tx0
    for j, val in enumerate(row):
        fill = "#dcfce7" if (j == len(row)-1 and is_result_row) else ("#f8fafc" if i % 2 == 0 else "white")
        L.append(f'<rect x="{cx}" y="{ry}" width="{COL_W[j]-4}" height="{row_h}" fill="{fill}" '
                  f'stroke="#e2e8f0"/>')
        fs = 10.5 if len(val) > 14 else 11.5
        L.append(f'<text x="{cx+(COL_W[j]-4)/2}" y="{ry+row_h/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="#1e293b">{esc(val)}</text>')
        cx += COL_W[j]

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-parallel-gate.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
