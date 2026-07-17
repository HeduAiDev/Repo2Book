#!/usr/bin/env python3
"""fig-m1-shared-registry: flow 模板(自定义 fan-in/fan-out)。
四个调试工具各建空 registry -> 都调同一个 registerTritonDialects 填满
-> 交给各自的 MLIR 驱动。承重全在中间那份共享注册表。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TOOLS = ["triton-opt", "triton-reduce", "triton-lsp", "triton-tensor-layout"]
DRIVERS = ["MlirOptMain", "MlirReduceMain", "MlirLspServerMain", "layoutPrint"]

TOOL_W, TOOL_H = 168, 44
HUB_W, HUB_H = 300, 92
DRV_W, DRV_H = 168, 44
COL_GAP = 26
PAD = 40
TOP = 74
ROW_GAP = 46  # 垂直间距: tools -> hub -> drivers

n = len(TOOLS)
tools_total_w = n * TOOL_W + (n - 1) * COL_GAP
w = PAD * 2 + max(tools_total_w, HUB_W + 260)
h = TOP + TOOL_H + ROW_GAP + HUB_H + ROW_GAP + DRV_H + PAD + 70

tools_x0 = PAD + (w - PAD * 2 - tools_total_w) / 2
TOOL_X = [tools_x0 + i * (TOOL_W + COL_GAP) for i in range(n)]
TOOL_Y = TOP

hub_x = PAD + (w - PAD * 2 - HUB_W) / 2
hub_y = TOOL_Y + TOOL_H + ROW_GAP
DRV_Y = hub_y + HUB_H + ROW_GAP
DRV_X = TOOL_X  # 对齐同一列

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(f'<text x="{PAD}" y="{34}" font-family="sans-serif" font-size="16" '
         f'font-weight="bold" fill="#0f172a">'
         f'{esc("四个调试工具共用一份 DialectRegistry")}</text>')

# 顶行: 四个工具薄壳
for i, name in enumerate(TOOLS):
    x = TOOL_X[i]
    L.append(f'<rect x="{x}" y="{TOOL_Y}" width="{TOOL_W}" height="{TOOL_H}" rx="8" '
              f'fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>')
    L.append(f'<text x="{x+TOOL_W/2}" y="{TOOL_Y+TOOL_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#1e3a8a">{esc(name)}</text>')
    # 箭头: 工具底部 -> 汇入 hub 顶部
    x1, y1 = x + TOOL_W / 2, TOOL_Y + TOOL_H
    x2, y2 = hub_x + HUB_W / 2, hub_y
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
              'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)" opacity="0.75"/>')

# 中间: 共享 registry(hub)
L.append(f'<rect x="{hub_x}" y="{hub_y}" width="{HUB_W}" height="{HUB_H}" rx="12" '
          f'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{hub_x+HUB_W/2}" y="{hub_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#78350f">{esc("registerTritonDialects(registry)")}</text>')
L.append(f'<text x="{hub_x+HUB_W/2}" y="{hub_y+48}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" '
          f'fill="#78350f">{esc("注册 13 个 dialect")}</text>')
L.append(f'<text x="{hub_x+HUB_W/2}" y="{hub_y+68}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" '
          f'fill="#78350f">{esc("+ Triton/GPU/NvidiaGPU/AMD 全部 pass")}</text>')

# 底行: 各自驱动
for i, name in enumerate(DRIVERS):
    x = DRV_X[i]
    L.append(f'<rect x="{x}" y="{DRV_Y}" width="{DRV_W}" height="{DRV_H}" rx="8" '
              f'fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>')
    L.append(f'<text x="{x+DRV_W/2}" y="{DRV_Y+DRV_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#14532d">{esc(name)}</text>')
    x1, y1 = hub_x + HUB_W / 2, hub_y + HUB_H
    x2, y2 = x + DRV_W / 2, DRV_Y
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
              'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)" opacity="0.75"/>')

# 底部标注: 接缝
note_y = DRV_Y + DRV_H + 34
L.append(f'<text x="{PAD}" y="{note_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("第三方后端接入调试链:只需在 registerTritonDialects 里加一行注册自家 dialect + pass。")}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m1-shared-registry.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
