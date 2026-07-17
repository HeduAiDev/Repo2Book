#!/usr/bin/env python3
"""fig-m1-tripartition: flow 模板。
compile.py 把命令行签名字符串"*fp32:16, i32:16, 1024, i32"按『有无冒号 x 整段能否转数』
三分成 hints / constants / signature 三张不相交的表。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PAD = 40
TOP = 100

TITLE = "签名三分:一行命令行签名字符串切成 hints / constants / signature 三桶"
SUBTITLE = "python/triton/tools/compile.py:L81-L102 —— AOT 把 JIT 特化钉到命令行的入口"

# 4 个签名段(位置, 参数名, 冒号前, 冒号后/整段, 去向桶列表)
SEGMENTS = [
    {"seg": "*fp32:16", "pos": "pos0 · X", "targets": ["hints", "signature"]},
    {"seg": "i32:16", "pos": "pos1 · N", "targets": ["hints", "signature"]},
    {"seg": "1024", "pos": "pos2 · BLOCK", "targets": ["constants"]},
    {"seg": "i32", "pos": "pos3 · stride", "targets": ["signature"]},
]

BUCKET_STYLE = {
    "hints": dict(fill="#fef3c7", stroke="#b45309", text="#78350f",
                  title="hints(整除性提示)", content="{0:16, 1:16}"),
    "constants": dict(fill="#dcfce7", stroke="#15803d", text="#166534",
                       title="constants(编译期常量)", content="{BLOCK:1024}"),
    "signature": dict(fill="#e0f2fe", stroke="#0369a1", text="#0c4a6e",
                       title="signature(运行期参数)", content="{X:*fp32, N:i32, stride:i32}"),
}
BUCKET_ORDER = ["hints", "constants", "signature"]

SEG_W, SEG_H = 168, 56
SEG_GAP = 26
seg_row_w = len(SEGMENTS) * SEG_W + (len(SEGMENTS) - 1) * SEG_GAP

BUCKET_W = {"hints": 210, "constants": 220, "signature": 360}
BUCKET_H = 74
BUCKET_GAP = 40
bucket_row_w = sum(BUCKET_W[b] for b in BUCKET_ORDER) + BUCKET_GAP * (len(BUCKET_ORDER) - 1)

w = PAD * 2 + max(seg_row_w, bucket_row_w)
seg_x0 = PAD + (w - PAD * 2 - seg_row_w) / 2
bucket_x0 = PAD + (w - PAD * 2 - bucket_row_w) / 2

seg_y = TOP + 20
fan_gap = 96
bucket_y = seg_y + SEG_H + fan_gap

elems = []


def add(s):
    elems.append(s)


# 段位置
seg_pos = {}
for i, seg in enumerate(SEGMENTS):
    x = seg_x0 + i * (SEG_W + SEG_GAP)
    seg_pos[i] = (x, seg_y)

# 桶位置
bucket_pos = {}
bx = bucket_x0
for b in BUCKET_ORDER:
    bucket_pos[b] = (bx, bucket_y)
    bx += BUCKET_W[b] + BUCKET_GAP

# 段行标题
add(f'<text x="{PAD:.0f}" y="{TOP-14:.0f}" font-family="sans-serif" font-size="13.5" '
    f'font-weight="bold" fill="#334155">输入(4 段):*fp32:16, i32:16, 1024, i32</text>')

# 段框
for i, seg in enumerate(SEGMENTS):
    x, y = seg_pos[i]
    add(f'<rect x="{x:.0f}" y="{y:.0f}" width="{SEG_W}" height="{SEG_H}" rx="8" '
        f'fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>')
    add(f'<text x="{x+SEG_W/2:.0f}" y="{y+24:.0f}" text-anchor="middle" font-family="monospace" '
        f'font-size="13" font-weight="bold" fill="#0f172a">{esc(seg["seg"])}</text>')
    add(f'<text x="{x+SEG_W/2:.0f}" y="{y+42:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#475569">{esc(seg["pos"])}</text>')

# 桶框(先画,箭头压在上面更清楚需先画箭头再画桶框——反过来:先桶框后箭头会盖住桶框顶边,
# 所以箭头端点在桶框顶边、桶框整体画好后箭头自然连到边缘,顺序不影响视觉,这里先画箭头线
# 再画桶框,保证箭头终点被桶框上边缘"咬住"而不穿透太多)
arrow_lines = []
for i, seg in enumerate(SEGMENTS):
    sx, sy = seg_pos[i]
    scx = sx + SEG_W / 2
    for tgt in seg["targets"]:
        tx, ty = bucket_pos[tgt]
        tcx = tx + BUCKET_W[tgt] / 2
        color = {"hints": "#b45309", "constants": "#15803d", "signature": "#0369a1"}[tgt]
        arrow_lines.append((scx, sy + SEG_H, tcx, ty, color))

add('<defs>'
    '<marker id="a-h" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '<marker id="a-c" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
    '<marker id="a-s" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker>'
    '</defs>')
MARKER = {"#b45309": "url(#a-h)", "#15803d": "url(#a-c)", "#0369a1": "url(#a-s)"}
for x1, y1, x2, y2, color in arrow_lines:
    add(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="1.8" stroke-opacity="0.75" '
        f'marker-end="{MARKER[color]}"/>')

# 桶框
for b in BUCKET_ORDER:
    x, y = bucket_pos[b]
    st = BUCKET_STYLE[b]
    add(f'<rect x="{x:.0f}" y="{y:.0f}" width="{BUCKET_W[b]}" height="{BUCKET_H}" rx="10" '
        f'fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="2"/>')
    add(f'<text x="{x+BUCKET_W[b]/2:.0f}" y="{y+26:.0f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13" font-weight="bold" '
        f'fill="{st["text"]}">{esc(st["title"])}</text>')
    add(f'<text x="{x+BUCKET_W[b]/2:.0f}" y="{y+50:.0f}" text-anchor="middle" '
        f'font-family="monospace" font-size="12.5" fill="{st["text"]}">{esc(st["content"])}</text>')

bottom_y = bucket_y + BUCKET_H

# 图例:同一段可落入两个桶(hints 与 signature 正交、不互斥)
legend_y = bottom_y + 34
add(f'<text x="{PAD:.0f}" y="{legend_y:.0f}" font-family="sans-serif" font-size="11.5" '
    f'fill="#64748b">箭头颜色 = 去向桶(琥珀→hints / 绿→constants / 蓝→signature);同一段可同时有两条箭头'
    f'(如 pos0 既进 hints 又留在 signature)</text>')

# 底部注解
note_lines = [
    "constants 与 signature 互斥且共同覆盖被 arg_names 认领的段;hints 是正交的位置索引集,可与 signature 重叠。",
    "本例 4 段 → hints 2 项 + constants 1 项(BLOCK) + signature 3 项(X, N, stride)——BLOCK 从运行期参数中除名。",
]
note_top = legend_y + 22
note_h = 24 * len(note_lines) + 20
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+24+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12.5" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-m1-tripartition.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
