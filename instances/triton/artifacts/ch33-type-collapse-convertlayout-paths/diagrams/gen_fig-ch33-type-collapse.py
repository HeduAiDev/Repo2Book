#!/usr/bin/env python3
"""fig-ch33-type-collapse: before-after 类型塌缩——TTGIR 带布局张量 -> LLVM 每线程 struct。
底部一行对照 shared/memdesc 的塌缩形态（不是 struct-of-N，是寻址便签）。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, H = 980, 460
PAD = 40
PANEL_W = 380
PANEL_GAP = 140
TOP = 100

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">类型塌缩:带布局张量 -&gt; LLVM 每线程 struct</text>',
     f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'TritonGPUToLLVMTypeConverter::convertTritonTensorType(TypeConverter.cpp:L112-L114)</text>']

# ---- 左面板：TTGIR 带布局张量 ----
LX = PAD
L.append(f'<text x="{LX+PANEL_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#1e40af">TTGIR:带布局张量</text>')
L.append(f'<rect x="{LX}" y="{TOP}" width="{PANEL_W}" height="220" rx="10" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<text x="{LX+PANEL_W/2}" y="{TOP+34}" text-anchor="middle" font-family="sans-serif" '
          'font-size="15" font-weight="bold" fill="#1e3a8a">tensor&lt;16x16xf32, #blocked&gt;</text>')

# 16x16 网格缩成 8x8 可视格代表整体张量（示意，不逐格对应线程）
GRID_N = 8
CELL = 18
gx0 = LX + PANEL_W/2 - GRID_N*CELL/2
gy0 = TOP + 54
for r in range(GRID_N):
    for c in range(GRID_N):
        L.append(f'<rect x="{gx0+c*CELL}" y="{gy0+r*CELL}" width="{CELL-2}" height="{CELL-2}" '
                  'fill="#bfdbfe" stroke="#3b82f6" stroke-width="0.6"/>')
L.append(f'<text x="{LX+PANEL_W/2}" y="{gy0+GRID_N*CELL+22}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" fill="#1e3a8a">'
          '整体视角:一整块带布局张量</text>')
L.append(f'<text x="{LX+PANEL_W/2}" y="{TOP+220-14}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" font-weight="bold" fill="#1e40af">'
          '256 元素 / 128 线程 (4 warp x 32)</text>')

# ---- 右面板：LLVM 每线程 struct ----
RX = LX + PANEL_W + PANEL_GAP
L.append(f'<text x="{RX+PANEL_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#b45309">LLVM:每线程视角</text>')
L.append(f'<rect x="{RX}" y="{TOP}" width="{PANEL_W}" height="220" rx="10" '
          'fill="#fffbeb" stroke="#b45309" stroke-width="1.5"/>')
L.append(f'<text x="{RX+PANEL_W/2}" y="{TOP+34}" text-anchor="middle" font-family="sans-serif" '
          'font-size="15" font-weight="bold" fill="#78350f">!llvm.struct&lt;(f32, f32)&gt;</text>')

# 单线程 struct 的两个字段
FIELD_W, FIELD_H = 130, 60
fx0 = RX + PANEL_W/2 - FIELD_W
fy0 = TOP + 66
for i in range(2):
    fx = fx0 + i*FIELD_W
    L.append(f'<rect x="{fx}" y="{fy0}" width="{FIELD_W-8}" height="{FIELD_H}" rx="6" '
              'fill="#fde68a" stroke="#b45309" stroke-width="1.5"/>')
    L.append(f'<text x="{fx+(FIELD_W-8)/2}" y="{fy0+FIELD_H/2-4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#78350f">字段 {i}</text>')
    L.append(f'<text x="{fx+(FIELD_W-8)/2}" y="{fy0+FIELD_H/2+14}" text-anchor="middle" '
              'font-family="sans-serif" font-size="12" font-weight="bold" fill="#78350f">f32</text>')
L.append(f'<text x="{RX+PANEL_W/2}" y="{fy0+FIELD_H+22}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" fill="#78350f">'
          '单个线程私有:N 个标量寄存器</text>')
L.append(f'<text x="{RX+PANEL_W/2}" y="{TOP+220-14}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" font-weight="bold" fill="#b45309">'
          '每线程字段 N = 2 = getTotalElemsPerThread</text>')

# ---- 中间箭头 ----
midy = TOP + 100
L.append(f'<line x1="{LX+PANEL_W+10}" y1="{midy}" x2="{RX-10}" y2="{midy}" '
          'stroke="#334155" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(LX+PANEL_W+RX)/2}" y="{midy-14}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" font-weight="bold" fill="#334155">塌缩</text>')
L.append(f'<text x="{(LX+PANEL_W+RX)/2}" y="{midy+22}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#64748b">256/128=2</text>')

# ---- 底部：shared/memdesc 对照条 ----
FY = TOP + 260
L.append(f'<rect x="{PAD}" y="{FY}" width="{W-2*PAD}" height="90" rx="10" '
          'fill="#f1f5f9" stroke="#64748b" stroke-width="1.2"/>')
L.append(f'<text x="{PAD+20}" y="{FY+28}" font-family="sans-serif" font-size="13" '
          'font-weight="bold" fill="#334155">对照:shared / memdesc 的塌缩(TypeConverter.cpp:L117)</text>')
L.append(f'<text x="{PAD+20}" y="{FY+54}" font-family="sans-serif" font-size="13" '
          'fill="#334155">没有“每线程持有几个”的说法 -&gt; 塌成寻址便签:</text>')
L.append(f'<rect x="{PAD+560}" y="{FY+34}" width="330" height="34" rx="6" '
          'fill="#e0e7ff" stroke="#6366f1" stroke-width="1.2"/>')
L.append(f'<text x="{PAD+560+165}" y="{FY+56}" text-anchor="middle" font-family="sans-serif" '
          'font-size="13" font-weight="bold" fill="#3730a3">'
          '{ptr} + rank x 2 个 i32(offsets, strides)</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch33-type-collapse.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
