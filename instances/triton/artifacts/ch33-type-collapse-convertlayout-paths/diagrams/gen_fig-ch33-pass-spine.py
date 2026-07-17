#!/usr/bin/env python3
"""fig-ch33-pass-spine: swimlane 变体——TTGIR->LLVM 两阶段 applyPartialConversion。
上泳道=阶段1 func;下泳道=阶段2 ops(几十个 populate*Patterns 按 PatternBenefit 排序)。
右侧小盒:TargetInfo 作为构造参数注入两阶段（后端接缝）。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PAD = 40
# 泳道宽度由阶段2的3个 pattern 盒子(pbw=250,gap=30)反推,保证不溢出
PBX0_OFFSET = 190   # pattern 盒左起相对泳道左边的偏移
PBW, PBH, PGAP = 250, 96, 30
N_PATTERNS = 3
PATTERNS_SPAN = N_PATTERNS * PBW + (N_PATTERNS - 1) * PGAP
LANE_W_ALL = PBX0_OFFSET + PATTERNS_SPAN + 20   # 泳道内容宽度(含右侧留白)
TI_W = 160
TI_GAP = 40
W = PAD + LANE_W_ALL + TI_GAP + TI_W + PAD
H = 460

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     'fill="#0f172a">TTGIR-&gt;LLVM 两阶段 applyPartialConversion</text>',
     f'<text x="{PAD}" y="55" font-family="sans-serif" font-size="12" fill="#64748b">'
     'ConvertTritonGPUToLLVM::runOnOperation (TritonGPUToLLVM.cpp:L110-L129)</text>']

# ---- 泳道1: func 阶段 ----
LANE1_Y = 84
LANE1_H = 140
L.append(f'<rect x="{PAD}" y="{LANE1_Y}" width="{LANE_W_ALL}" height="{LANE1_H}" rx="10" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<rect x="{PAD+10}" y="{LANE1_Y+10}" width="150" height="26" rx="5" '
          'fill="#1d4ed8"/>')
L.append(f'<text x="{PAD+85}" y="{LANE1_Y+28}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12" font-weight="bold" fill="white">阶段 1:func</text>')

step1_boxes = [
    ("populateFuncOpConversionPattern", "函数签名张量塌成 struct"),
    ("initSharedMemory", "建 global_smem(call op 需先知\n各函数 shmem 基址)"),
]
BOX_W1, BOX_H1 = 260, 76
sx = PAD + 190
sy = LANE1_Y + 34
for i, (name, note) in enumerate(step1_boxes):
    x = sx + i * (BOX_W1 + 60)
    L.append(f'<rect x="{x}" y="{sy}" width="{BOX_W1}" height="{BOX_H1}" rx="8" '
              'fill="#bfdbfe" stroke="#1d4ed8" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W1/2}" y="{sy+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#1e3a8a">{esc(name)}</text>')
    for k, line in enumerate(note.split("\n")):
        L.append(f'<text x="{x+BOX_W1/2}" y="{sy+42+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#1e3a8a">{esc(line)}</text>')
    if i < len(step1_boxes) - 1:
        ax1 = x + BOX_W1
        ax2 = x + BOX_W1 + 60
        ay = sy + BOX_H1/2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2-4}" y2="{ay}" '
                  'stroke="#1d4ed8" stroke-width="2" marker-end="url(#a)"/>')

# ---- 阶段间箭头 ----
mid_y1 = LANE1_Y + LANE1_H
mid_y2 = mid_y1 + 40
L.append(f'<line x1="{PAD+120}" y1="{mid_y1}" x2="{PAD+120}" y2="{mid_y2-4}" '
          'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+130}" y="{(mid_y1+mid_y2)/2+4}" font-family="sans-serif" '
          'font-size="11" fill="#334155">func 全转完 -&gt;</text>')

# ---- 泳道2: ops 阶段 ----
LANE2_Y = mid_y2
LANE2_H = 190
L.append(f'<rect x="{PAD}" y="{LANE2_Y}" width="{LANE_W_ALL}" height="{LANE2_H}" rx="10" '
          'fill="#fffbeb" stroke="#b45309" stroke-width="1.5"/>')
L.append(f'<rect x="{PAD+10}" y="{LANE2_Y+10}" width="150" height="26" rx="5" '
          'fill="#b45309"/>')
L.append(f'<text x="{PAD+85}" y="{LANE2_Y+28}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12" font-weight="bold" fill="white">阶段 2:ops</text>')

L.append(f'<text x="{PAD+190}" y="{LANE2_Y+22}" font-family="sans-serif" font-size="12" '
          'fill="#78350f">几十个 populate*Patterns 塞进同一 RewritePatternSet,'
          '按 PatternBenefit 从高到低试:</text>')

patterns = [
    ("local_alloc(优化):\nLocalAllocOpConversion", 20, "#166534", "#dcfce7"),
    ("convert_layout\n优先于 LLVM 转换的规则", 10, "#78350f", "#fef3c7"),
    ("elementwise / load-store / \ndot / reduce 等默认规则", 1, "#7f1d1d", "#fee2e2"),
]
pbx0 = PAD + PBX0_OFFSET
pby = LANE2_Y + 40
pbw, pbh = PBW, PBH
for i, (name, benefit, tcolor, bg) in enumerate(patterns):
    x = pbx0 + i * (pbw + PGAP)
    L.append(f'<rect x="{x}" y="{pby}" width="{pbw}" height="{pbh}" rx="8" '
              f'fill="{bg}" stroke="{tcolor}" stroke-width="1.5"/>')
    for k, line in enumerate(name.split("\n")):
        L.append(f'<text x="{x+pbw/2}" y="{pby+22+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="{tcolor}">{esc(line)}</text>')
    L.append(f'<rect x="{x+pbw/2-38}" y="{pby+pbh-34}" width="76" height="24" rx="12" '
              f'fill="{tcolor}"/>')
    L.append(f'<text x="{x+pbw/2}" y="{pby+pbh-17}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="white">benefit={benefit}</text>')
    if i < len(patterns) - 1:
        ax1 = x + pbw
        ax2 = x + pbw + PGAP
        ay = pby + pbh/2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2-4}" y2="{ay}" '
                  'stroke="#64748b" stroke-width="1.5" stroke-dasharray="4,3" '
                  'marker-end="url(#a)"/>')

L.append(f'<text x="{pbx0}" y="{pby+pbh+22}" font-family="sans-serif" font-size="11" '
          'fill="#78350f">20 &gt; 10 &gt; 1:先命中 benefit 高者(贪心匹配)</text>')

# ---- 右侧 TargetInfo 接缝 ----
TIX = PAD + LANE_W_ALL + TI_GAP
TIY = 84
TIH = LANE2_Y + LANE2_H - TIY
TICX = TIX + TI_W/2
L.append(f'<rect x="{TIX}" y="{TIY}" width="{TI_W}" height="{TIH}" rx="10" '
          'fill="#f1f5f9" stroke="#475569" stroke-width="1.5" stroke-dasharray="5,3"/>')
L.append(f'<text x="{TICX}" y="{TIY+26}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12" font-weight="bold" fill="#334155">TargetInfo</text>')
L.append(f'<text x="{TICX}" y="{TIY+46}" text-anchor="middle" font-family="sans-serif" '
          'font-size="11" fill="#475569">后端接缝</text>')
L.append(f'<text x="{TICX}" y="{TIY+72}" text-anchor="middle" font-family="sans-serif" '
          'font-size="10" fill="#475569">构造参数注入</text>')
L.append(f'<text x="{TICX}" y="{TIY+88}" text-anchor="middle" font-family="sans-serif" '
          'font-size="10" fill="#475569">两阶段共用</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch33-pass-spine.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
