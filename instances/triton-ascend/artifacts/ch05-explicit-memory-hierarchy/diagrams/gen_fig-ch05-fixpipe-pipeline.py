#!/usr/bin/env python3
"""fig-ch05-fixpipe-pipeline — flow 模板（al.fixpipe：L0C Fractal NZ → UB ND）。
横向主链：源(L0C tensor)→结构校验→目的(UB buffer)，校验通过后向下落一个
六参 create_fixpipe op 框（前端固定 NO_QUANT/NO_RELU，无用户入口）。
全部坐标由常量/循环计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "al.fixpipe：L0C → UB 的专用搬运 + 布局转换通路"
SUBTITLE = "校验通过后向下方落地一个六参 create_fixpipe op；前端固定 NO_QUANT/NO_RELU，用户无入口"

PAD = 40
TOP = 106
NODE_W, NODE_H = 260, 108
GATE_W, GATE_H = 300, 108
GAP = 100

src_x = PAD
gate_x = src_x + NODE_W + GAP
dst_x = gate_x + GATE_W + GAP
row_y = TOP

w = dst_x + NODE_W + PAD

PARAM_W = w - 2 * PAD
PARAM_Y = row_y + NODE_H + 92
PARAM_H = 96

h = PARAM_Y + PARAM_H + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 源：L0C 上的 tl.tensor（Fractal NZ）
L.append(f'<rect x="{src_x}" y="{row_y}" width="{NODE_W}" height="{NODE_H}" rx="10" '
          f'fill="#ffedd5" stroke="#c2410c" stroke-width="2.2"/>')
L.append(f'<text x="{src_x+NODE_W/2}" y="{row_y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#7c2d12">{esc("源：L0C 上的 tl.tensor")}</text>')
L.append(f'<text x="{src_x+NODE_W/2}" y="{row_y+48}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#9a3412">{esc("Fractal NZ 分形布局(cube 累加输出)")}</text>')
L.append(f'<text x="{src_x+NODE_W/2}" y="{row_y+70}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#c2410c">{esc("须为 tl.tensor，否则拒")}</text>')
L.append(f'<text x="{src_x+NODE_W/2}" y="{row_y+92}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="#c2410c" font-weight="bold">'
          f'{esc("core.py:L286-289,L296")}</text>')

# 结构校验节点
L.append(f'<rect x="{gate_x}" y="{row_y}" width="{GATE_W}" height="{GATE_H}" rx="10" '
          f'fill="#f1f5f9" stroke="#334155" stroke-width="2" stroke-dasharray="7,5"/>')
L.append(f'<text x="{gate_x+GATE_W/2}" y="{row_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("结构校验(顺序短路)")}</text>')
gate_lines = ["① is_910_95", "② src 须 tl.tensor", "③ dst 须 bl.buffer", "④ dst.space 须 UB"]
for i, line in enumerate(gate_lines):
    L.append(f'<text x="{gate_x+18}" y="{row_y+44+i*16}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(line)}</text>')

# 目的：UB 上的 bl.buffer
L.append(f'<rect x="{dst_x}" y="{row_y}" width="{NODE_W}" height="{NODE_H}" rx="10" '
          f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.2"/>')
L.append(f'<text x="{dst_x+NODE_W/2}" y="{row_y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#1e3a8a">{esc("目的：UB 上的 bl.buffer")}</text>')
L.append(f'<text x="{dst_x+NODE_W/2}" y="{row_y+48}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#1e40af">{esc("常规 ND 布局(dma_mode=NZ2ND 还原)")}</text>')
L.append(f'<text x="{dst_x+NODE_W/2}" y="{row_y+70}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#1d4ed8">{esc("须位于 UB，否则拒")}</text>')
L.append(f'<text x="{dst_x+NODE_W/2}" y="{row_y+92}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="#1d4ed8" font-weight="bold">{esc("core.py:L299-300")}</text>')

# 横向箭头：源 -> 校验 -> 目的
mid_y = row_y + NODE_H / 2
L.append(f'<line x1="{src_x+NODE_W}" y1="{mid_y}" x2="{gate_x}" y2="{mid_y}" '
          f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(src_x+NODE_W+gate_x)/2}" y="{mid_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#334155">'
          f'{esc("al.fixpipe")}</text>')
L.append(f'<line x1="{gate_x+GATE_W}" y1="{mid_y}" x2="{dst_x}" y2="{mid_y}" '
          f'stroke="#15803d" stroke-width="2.2" marker-end="url(#g)"/>')
L.append(f'<text x="{(gate_x+GATE_W+dst_x)/2}" y="{mid_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#15803d">'
          f'{esc("全通过")}</text>')

# 竖向箭头：校验节点 -> 下方 create_fixpipe 六参框
gate_cx = gate_x + GATE_W / 2
L.append(f'<line x1="{gate_cx}" y1="{row_y+NODE_H}" x2="{gate_cx}" y2="{PARAM_Y}" '
          f'stroke="#15803d" stroke-width="2.2" marker-end="url(#g)"/>')
L.append(f'<text x="{gate_cx+12}" y="{(row_y+NODE_H+PARAM_Y)/2}" font-family="sans-serif" '
          f'font-size="11" fill="#15803d">{esc("落地建 op")}</text>')

# create_fixpipe 六参框
L.append(f'<rect x="{PAD}" y="{PARAM_Y}" width="{PARAM_W}" height="{PARAM_H}" rx="10" '
          f'fill="#ecfdf5" stroke="#15803d" stroke-width="2.2"/>')
L.append(f'<text x="{PAD+18}" y="{PARAM_Y+26}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#14532d">'
          f'{esc("create_fixpipe(src, dst, dma_mode, dual_dst_mode, pre_quant, pre_relu)")}</text>')

PARAMS = [("src", "L0C tensor"), ("dst", "UB buffer"), ("dma_mode", "NZ2ND(默认)"),
          ("dual_dst_mode", "NO_DUAL"), ("pre_quant", "NO_QUANT(前端硬编码)"),
          ("pre_relu", "NO_RELU(前端硬编码)")]
pw = (PARAM_W - 36) / len(PARAMS)
for i, (name, val) in enumerate(PARAMS):
    px = PAD + 18 + i * pw
    L.append(f'<text x="{px}" y="{PARAM_Y+52}" font-family="sans-serif" font-size="11" '
              f'fill="#166534" font-weight="bold">{esc(name)}</text>')
    L.append(f'<text x="{px}" y="{PARAM_Y+70}" font-family="sans-serif" font-size="11" '
              f'fill="#14532d">{esc(val)}</text>')
L.append(f'<text x="{PAD+18}" y="{PARAM_Y+PARAM_H-10}" font-family="sans-serif" font-size="10.5" '
          f'fill="#15803d">{esc("6 参签名后两位(pre_quant/pre_relu)前端硬编码固定，量化/ReLU 融合语言层永不开启")}</text>')

# 脚注
foot_y = PARAM_Y + PARAM_H + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("fixpipe 是 L0C → 其它内存层级的专用数据通路(本芯片系列仅支持 L0C→UB)；NZ→ND 分形下降细节归 P5(ch23 HIVM 方言)。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("数据来自 host 实测；固定参数见 third_party/ascend/language/cann/extension/core.py:L331。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch05-fixpipe-pipeline.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
