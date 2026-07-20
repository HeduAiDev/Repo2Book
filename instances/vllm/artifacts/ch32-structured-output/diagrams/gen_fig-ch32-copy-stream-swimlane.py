#!/usr/bin/env python3
"""fig-ch32-copy-stream-swimlane: 掩码与行映射走独立 copy_stream 搬上卡,
前后各一次 wait_stream 把「数据就位」与「缓冲用完」钉死,拷贝才能安全地与计算流重叠。
template: swimlane"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LANES = ["copy_stream", "current_stream(计算)"]
LANE_COLOR = {"copy_stream": "#0891b2", "current_stream(计算)": "#16a34a"}

PAD = 40
MARGIN = 30
TOP = 130
STEP = 108
BOX_W = 420
BH = 74
LANE_W = BOX_W + 140  # 留出中间走廊给跨泳道 wait_stream 箭头与标签
HEAD_W = 300
W = MARGIN * 2 + BOX_W + LANE_W + BOX_W
H = TOP + STEP * 6 + PAD + 40
X = {name: MARGIN + BOX_W/2 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
          '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16.5" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("两条流的泳道:拷贝与计算重叠,两次 wait_stream 方向相反")}</text>')
L.append(f'<text x="{W/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("掩码计算本身早在 execute_model 非阻塞发车之后就在 CPU 侧完成,不占这条时间线")}</text>')

HEAD_H = 40
for name, x in X.items():
    color = LANE_COLOR[name]
    head_top = TOP - 18 - HEAD_H
    L.append(f'<rect x="{x-HEAD_W/2}" y="{head_top}" width="{HEAD_W}" height="{HEAD_H}" rx="6" '
              f'fill="{color}" opacity="0.15" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{x}" y="{head_top+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{color}">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-18}" x2="{x}" y2="{H-PAD}" '
              f'stroke="{color}" stroke-width="1.2" stroke-dasharray="4,4" opacity="0.5"/>')

EVENTS = [
    ("copy_stream", ["异步 H2D ①:掩码本体", "4 行 x 3 列 int32(本例)",
                      "真实规模 4752 列 = 18.5625 KiB/行",
                      "vllm/v1/worker/gpu/structured_outputs.py:L34-37"]),
    ("copy_stream", ["异步 H2D ②:行映射张量", "长度 = num_masks = 4(映射构造在 L39-48)",
                      "structured_outputs.py:L51-57"]),
    ("current_stream(计算)", ["current_stream.wait_stream(copy_stream)",
                      "等两次 H2D 都落地才启动 kernel",
                      "structured_outputs.py:L59-61"]),
    ("current_stream(计算)", ["_apply_grammar_bitmask_kernel", "读预分配缓冲",
                      "[max_num_logits, ceil(V/32)]"]),
    ("copy_stream", ["copy_stream.wait_stream(current_stream)",
                      "等 kernel 读完这块缓冲才允许下一步覆写",
                      "structured_outputs.py:L78-80"]),
    ("copy_stream", ["下一步:覆写前 num_masks 行", "缓冲跨步复用,不新分配"]),
]

centers = {}
for i, (lane, lines) in enumerate(EVENTS):
    y = TOP + STEP * i
    x = X[lane]
    color = LANE_COLOR[lane]
    L.append(f'<rect x="{x-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BH}" rx="8" '
              f'fill="white" stroke="{color}" stroke-width="2"/>')
    n = len(lines)
    cy0 = y + BH/2 - (n-1)*8
    for k, t in enumerate(lines):
        fw = "bold" if k == 0 else "normal"
        fs = 12 if k == 0 else 10.5
        fc = "#0f172a" if k == 0 else "#64748b"
        L.append(f'<text x="{x}" y="{cy0+k*16:.0f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fs}" font-weight="{fw}" fill="{fc}">{esc(t)}</text>')
    L.append(f'<text x="{PAD}" y="{y+BH/2+4}" font-family="sans-serif" font-size="11" '
              f'fill="#94a3b8">{esc(f"t{i+1}")}</text>')
    centers[i] = (x, y, BOX_W, BH, lane)

# 同泳道时序箭头
for i in range(len(centers) - 1):
    x1, y1, w1, h1, lane1 = centers[i]
    x2, y2, w2, h2, lane2 = centers[i + 1]
    if lane1 == lane2:
        L.append(f'<line x1="{x1}" y1="{y1+h1}" x2="{x2}" y2="{y2}" '
                  f'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')

# 跨泳道 wait_stream 箭头(显式标注方向,红色高亮,避免默认汇聚被误读为因果不明)
def cross_arrow(i_src, i_dst, label):
    xs_, ys_, ws_, hs_, ls_ = centers[i_src]
    xd_, yd_, wd_, hd_, ld_ = centers[i_dst]
    x1 = xs_ + (ws_/2 if xd_ > xs_ else -ws_/2)
    x2 = xd_ + (-wd_/2 if xd_ > xs_ else wd_/2)
    y1 = ys_ + hs_/2
    y2 = yd_ + hd_/2
    L.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="#dc2626" '
              f'stroke-width="2" stroke-dasharray="6,3" marker-end="url(#r)"/>')
    mx, my = (x1+x2)/2, (y1+y2)/2 - 10
    tw = 10 + len(label) * 6
    L.append(f'<rect x="{mx-tw/2}" y="{my-11}" width="{tw}" height="22" rx="5" '
              f'fill="white" stroke="#dc2626"/>')
    L.append(f'<text x="{mx}" y="{my+4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#991b1b">{esc(label)}</text>')

# 第 2 个 H2D(idx1) -> 第 3 个事件(current_stream.wait,idx2):数据就位方向
cross_arrow(1, 2, "wait:数据已就位")
# kernel(idx3) -> copy_stream.wait(idx4):缓冲用完方向
cross_arrow(3, 4, "wait:缓冲已用完")

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-copy-stream-swimlane.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
