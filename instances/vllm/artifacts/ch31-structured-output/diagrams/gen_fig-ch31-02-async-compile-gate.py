#!/usr/bin/env python3
"""fig-ch31-02: 异步语法编译门——三条泳道，调度线程每轮只花 100us 问一句"好了没"。
template: swimlane"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LANES = ["输入处理线程", "线程池工作线程", "调度线程"]
LANE_COLOR = {"输入处理线程": "#6366f1", "线程池工作线程": "#0891b2", "调度线程": "#16a34a"}
# 线程池规模是这条泳道本身的属性(池在建后端时构造,不属于某一轮事件),
# 所以挂在泳道标题下,而不是挂在「编译完成」那一行。
LANE_NOTE = {"线程池工作线程": "max_workers=2（本机观测）"}

# (round_label, lane, box_text_lines, note)
EVENTS = [
    ("入队", "输入处理线程",
     ["preprocess_add_request", "grammar_init →", "executor.submit(_create_grammar)", "立刻返回"], None),
    ("提交编译", "线程池工作线程",
     ["_create_grammar", "backend.compile_grammar", "(request_type, grammar_spec)"], "编译中（Future 未完成）"),
    # 调度线程每轮是两个函数串联:先判是不是阻塞态,再跑晋级检查。两者各有一个
    # 布尔量,不能挤进同一格(正文专门澄清过轮 4 的两个 False 含义不同)。
    ("轮 1", "调度线程",
     ["_is_blocked_waiting_status=True", "→ _try_promote_blocked_waiting_request",
      "读 grammar=None → 晋级检查 False", "status=WAITING_FOR_..._GRAMMAR(=2)"], "轮询预算 100us"),
    ("轮 2", "调度线程",
     ["_is_blocked_waiting_status=True", "→ _try_promote_blocked_waiting_request",
      "读 grammar=None → 晋级检查 False", "status=WAITING_FOR_..._GRAMMAR(=2)"], "轮询预算 100us"),
    ("编译完成", "线程池工作线程",
     ["成品语法对象", "写回 Future"], None),
    ("轮 3", "调度线程",
     ["_is_blocked_waiting_status=True", "→ _try_promote_blocked_waiting_request",
      "读 grammar=成品 → 晋级检查 True", "status 改为 WAITING(=1)"], "晋级"),
    ("轮 4", "调度线程",
     ["_is_blocked_waiting_status=False", "晋级检查也返回 False,但另有原因:",
      "if 分支没命中（与轮 1/2 的 False 不同因）"], None),
]

LANE_W = 380
TOP = 124
STEP = 110
PAD = 40
BOX_W = 320

w = PAD * 2 + LANE_W * (len(LANES) - 1) + 480
h = TOP + STEP * len(EVENTS) + PAD + 40
X = {name: PAD + 150 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
     '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("异步编译门：编译在线程池里跑，调度线程每轮只花 100 微秒问一句好了没")}</text>')

HEAD_H = 50  # 各泳道统一高度:留出第二行给泳道自身的属性注记(如线程池规模)
for name, x in X.items():
    color = LANE_COLOR[name]
    head_top = TOP - 18 - HEAD_H
    L.append(f'<rect x="{x-115}" y="{head_top}" width="230" height="{HEAD_H}" rx="6" '
              f'fill="{color}" opacity="0.15" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{x}" y="{head_top+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{color}">{esc(name)}</text>')
    if name in LANE_NOTE:
        L.append(f'<text x="{x}" y="{head_top+40}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" fill="{color}">{esc(LANE_NOTE[name])}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-18}" x2="{x}" y2="{h-PAD}" '
              f'stroke="{color}" stroke-width="1.2" stroke-dasharray="4,4" opacity="0.5"/>')

BH = 78  # 统一盒高，简化连接线计算
centers = []
for i, (rlabel, lane, lines, note) in enumerate(EVENTS):
    y = TOP + STEP * i
    x = X[lane]
    color = LANE_COLOR[lane]
    L.append(f'<rect x="{x-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BH}" rx="8" '
              f'fill="white" stroke="{color}" stroke-width="2"/>')
    n = len(lines)
    cy0 = y + BH/2 - (n-1)*8
    for k, t in enumerate(lines):
        fw = "bold" if k == 0 else "normal"
        L.append(f'<text x="{x}" y="{cy0+k*16:.0f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11.5" font-weight="{fw}" fill="#1e293b">{esc(t)}</text>')
    L.append(f'<text x="{PAD+10}" y="{y+BH/2+4}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="#64748b">{esc(rlabel)}</text>')
    if note:
        # 按 CJK/半角逐字估宽,不用 len()*常数——否则注记框被算得过宽,右边压到隔壁泳道的生命线上
        note_w = sum(11.5 if ord(c) > 0x2E80 else 11.5 * 0.58 for c in note) + 16
        nx = x + BOX_W/2 + 14
        L.append(f'<rect x="{nx}" y="{y+BH/2-11}" width="{note_w}" height="22" rx="5" '
                  f'fill="#fef3c7" stroke="#b45309"/>')
        L.append(f'<text x="{nx+8}" y="{y+BH/2+4}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#92400e">{esc(note)}</text>')
    centers.append((x, y, BOX_W, BH, lane))

# 连接箭头：按事件时间顺序，画从上一个盒子边缘到下一个盒子边缘的折线
for i in range(len(centers) - 1):
    x1, y1, w1, h1, lane1 = centers[i]
    x2, y2, w2, h2, lane2 = centers[i + 1]
    src_y = y1 + h1
    dst_y = y2
    if lane1 == lane2:
        L.append(f'<line x1="{x1}" y1="{src_y}" x2="{x2}" y2="{dst_y}" '
                  f'stroke="#94a3b8" stroke-width="1.6" marker-end="url(#a)"/>')
    else:
        mid_y = (src_y + dst_y) / 2
        L.append(f'<path d="M {x1} {src_y} L {x1} {mid_y} L {x2} {mid_y} L {x2} {dst_y}" '
                  f'fill="none" stroke="#94a3b8" stroke-width="1.6" marker-end="url(#a)"/>')

L.append('</svg>')
out = Path("fig-ch31-02-async-compile-gate.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
