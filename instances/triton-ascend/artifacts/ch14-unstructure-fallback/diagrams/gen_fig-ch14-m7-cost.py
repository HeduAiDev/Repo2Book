#!/usr/bin/env python3
"""fig-ch14-m7-cost — before-after 模板扩展为三态代价柱状对比:
结构化基线(1 次)/ 部分标量化(16 次)/ 完全标量化(128 次)。柱高按线性比例(×2 缩放)
直接反映 1:16:128 的真实量级差,不做对数失真。底部单列对齐闸(32 字节)规则说明
为何 unstructure_mix 停在 16 次而非退化到 128 次。数据取自 explainer m7.figure_specs.numbers。
NB:font-weight="bold" 与 CJK "量" 字组合在本渲染环境下出 tofu(见 fig-ch14-m6 教训)——
含"量"字的文本一律不用 bold。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

SCALE = 1.9   # 每次搬运对应的像素高度(线性,不失真)
BASE_H = 10   # 最小可见高度(结构化基线=1 次时仍要看得见柱子)

BARS = [
    ("结构化基线", 1, "#16a34a", "#166534", "整块 16×8 memref 一次连续搬运"),
    ("部分标量化", 16, "#d97706", "#92400e", "dim0 循环 16 次,每次 1×8 连续 f32"),
    ("完全标量化", 128, "#dc2626", "#7f1d1d", "单维(128)全离散,每次 1 个 i32 单元素"),
]

BAR_W = 220
GAP = 110
PAD = 60
TOP = 150
BASELINE_EXTRA = 40
CHART_H = 128 * SCALE + BASE_H + 20

w = PAD * 2 + len(BARS) * BAR_W + (len(BARS) - 1) * GAP
h = TOP + CHART_H + 230

baseline_y = TOP + CHART_H

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
     f'font-size="17" fill="#0f172a">'
     f'{esc("同一份 16×8 数据,掉进兜底的代价:柱高 = 实际搬运次数(线性,1:16:128)")}</text>',
     f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12.5" fill="#475569">'
     f'{esc("结构化 O(1) 元数据描述 → 标量化 O(∏unstructured 维 size)逐元素/逐行访存")}</text>']

# y 轴刻度参考线(1/16/128)
for val in (1, 16, 128):
    y = baseline_y - (BASE_H + val * SCALE)
    L.append(f'<line x1="{PAD-10}" y1="{y}" x2="{w-PAD+10}" y2="{y}" '
              'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>')
    L.append(f'<text x="{PAD-16}" y="{y+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11" fill="#94a3b8">{val}</text>')

for i, (name, count, fill, dark, note) in enumerate(BARS):
    x = PAD + i * (BAR_W + GAP)
    bh = BASE_H + count * SCALE
    y = baseline_y - bh
    L.append(f'<rect x="{x}" y="{y}" width="{BAR_W}" height="{bh}" rx="6" '
              f'fill="{fill}"/>')
    L.append(f'<text x="{x+BAR_W/2}" y="{y-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" fill="{dark}">{esc(f"{count} 次搬运")}</text>')
    L.append(f'<text x="{x+BAR_W/2}" y="{baseline_y+26}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{x+BAR_W/2}" y="{baseline_y+48}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#64748b">{esc(note)}</text>')

L.append(f'<line x1="{PAD-10}" y1="{baseline_y}" x2="{w-PAD+10}" y2="{baseline_y}" '
          'stroke="#334155" stroke-width="1.5"/>')

gate_y = baseline_y + 90
gate_x0, gate_w = PAD, w - PAD * 2
L.append(f'<rect x="{gate_x0}" y="{gate_y}" width="{gate_w}" height="70" rx="10" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<text x="{gate_x0+gate_w/2}" y="{gate_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#1e3a8a">'
          f'{esc("对齐闸(UnstructureConversionPass.cpp:L334-342,常量 32 在 L341):尾部连续字节 % 32")}</text>')
L.append(f'<text x="{gate_x0+gate_w/2}" y="{gate_y+48}" text-anchor="middle" '
          f'font-family="monospace" font-size="11.5" fill="#1e40af">'
          f'{esc("8×f32=32 字节,32%32=0 → 保向量(16 次);1 个元素 4 字节,4%32≠0 → 强制标量化(128 次)")}</text>')

foot_y = gate_y + 70 + 34
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">'
          f'{esc("离散维每多一个,代价单调乘增:一个 tl.load 当索引,就把 O(1) 变成两个数量级的 O(N)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-m7-cost.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
