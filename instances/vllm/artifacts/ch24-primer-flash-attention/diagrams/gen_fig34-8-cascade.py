#!/usr/bin/env python3
"""swimlane 模板(定制为两泳道汇流):cascade attention 把共享前缀请求拆成
前缀段(causal=False,算一遍复用)+ 后缀段(causal=True,私有),两段各带 lse,
汇入 merge_attn_states 合并。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "cascade attention — 共享前缀两段泳道汇入 LSE 合并"
SUBTITLE = "前缀段对全批共享前缀只算一遍(causal=False);后缀段各请求算私有 KV(causal=True);两段各带 softmax_lse,汇入 merge_attn_states"

PAD, TOP = 40, 100
LANE_LABEL_W = 130
LANE_BOX_W = 620
LANE_H = 64
LANE_GAP = 46

lane_x = PAD + LANE_LABEL_W
prefix_y = TOP
suffix_y = TOP + LANE_H + LANE_GAP

LANES = [
    ("前缀段", "causal=False", "#dbeafe", "#1d4ed8",
     ["block_table[:1]", "return_softmax_lse=True"]),
    ("后缀段", "causal=True", "#fef3c7", "#b45309",
     ["block_table[:, num_common_kv_blocks:]", "return_softmax_lse=True"]),
]

merge_x = lane_x
merge_y = suffix_y + LANE_H + 74
merge_w = LANE_BOX_W
MERGE_LINES = ["merge_attn_states(output,", "  prefix_output, prefix_lse,", "  suffix_output, suffix_lse)"]
merge_h = 26 + 20 + (len(MERGE_LINES) - 1) * 18 + 16

w = lane_x + LANE_BOX_W + PAD
h = merge_y + merge_h + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

for (name, causal, fill, stroke, lines), y in zip(LANES, [prefix_y, suffix_y]):
    # 泳道标签
    L.append(f'<rect x="{PAD}" y="{y}" width="{LANE_LABEL_W-14}" height="{LANE_H}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{PAD+(LANE_LABEL_W-14)/2}" y="{y+LANE_H/2-4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{stroke}">{esc(name)}</text>')
    L.append(f'<text x="{PAD+(LANE_LABEL_W-14)/2}" y="{y+LANE_H/2+14}" text-anchor="middle" '
              f'font-family="monospace" font-size="11" fill="{stroke}">{esc(causal)}</text>')
    # 泳道内容框
    L.append(f'<rect x="{lane_x}" y="{y}" width="{LANE_BOX_W}" height="{LANE_H}" rx="6" '
              f'fill="white" stroke="{stroke}" stroke-width="1.5" stroke-dasharray="5,3"/>')
    for i, line in enumerate(lines):
        L.append(f'<text x="{lane_x+16}" y="{y+24+i*22}" font-family="monospace" '
                  f'font-size="12" fill="#334155">{esc(line)}</text>')

# 汇流箭头:两泳道 -> 合并框(路径刻意走两个泳道框的右半空白区,不压文字——
# 文字左对齐止于约 x=420;两条线保持右线常在左线之右,互不交叉)
merge_top_center = merge_x + merge_w / 2
L.append(f'<line x1="{lane_x+merge_w*0.72}" y1="{prefix_y+LANE_H}" '
          f'x2="{merge_top_center-40}" y2="{merge_y}" stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{lane_x+merge_w*0.62}" y1="{suffix_y+LANE_H}" '
          f'x2="{merge_top_center+40}" y2="{merge_y}" stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')

# 合并框
L.append(f'<rect x="{merge_x}" y="{merge_y}" width="{merge_w}" height="{merge_h}" rx="8" '
          'fill="#ecfdf5" stroke="#047857" stroke-width="2.2"/>')
yy = merge_y + 24
L.append(f'<text x="{merge_x+merge_w/2}" y="{yy}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#047857">LSE 合并</text>')
yy += 22
for line in MERGE_LINES:
    L.append(f'<text x="{merge_x+merge_w/2}" y="{yy}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" fill="#065f46">{esc(line)}</text>')
    yy += 18

# 底部结论框
concl_y = merge_y + merge_h + 26
L.append(f'<rect x="{PAD}" y="{concl_y}" width="{w-2*PAD}" height="34" rx="6" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
CONCL = "合并结果与不拆分的精确注意力差在浮点舍入内(见 lse-merge trace,约 2e-16)"
L.append(f'<text x="{w/2}" y="{concl_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#1e3a8a">{esc(CONCL)}</text>')

foot_y = h - 16
FOOT = "前缀段算一次、复用给全批共享该前缀的请求;后缀段各请求私有——这是 LSE 合并(⊕ 算子)在 vLLM 推理期调用现场的落地。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc(FOOT)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig34-8-cascade.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
