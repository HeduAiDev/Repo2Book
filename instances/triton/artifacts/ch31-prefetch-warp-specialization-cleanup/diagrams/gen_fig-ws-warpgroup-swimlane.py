#!/usr/bin/env python3
"""fig-ws-warpgroup-swimlane: swimlane 模板——num_consumer_groups>0 时,WS 五个 pass 把循环
按 async task id 拆到 producer/consumer 两条泳道,token 做跨 warpgroup 流控;
WSLowering 把 async task id 落成 warpId/4。默认路径下五个 WS pass 全部入口即早退、IR 不变。
数字来源见 explainer.json mechanism ws-token-landing。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Hopper Warp Specialization —— async task id 落到 producer/consumer 两条泳道(选读)"
GATE_NOTE_LINES = [
    "默认 num_consumer_groups = 0:WSTaskPartition 等五个 WS pass 入口即早退,IR 不变。",
    "以下为 num_consumer_groups > 0 时的落地形态：",
]

LANES = ["Producer warpgroup\n(asyncTaskId=0)", "Consumer warpgroup\n(asyncTaskId=1)"]
EVENTS = [
    ("P", "P", "load(global→shared)  异步搬数据"),
    ("P", "C", "ProducerCommit(token)  『料好了』"),
    ("C", "C", "ConsumerWait(token) → wgmma 算 dot"),
    ("C", "P", "ConsumerRelease(token)  『可以搬下一批』"),
]
LANE_KEY = {"P": 0, "C": 1}

LANE_W, TOP, STEP, PAD = 460, 210, 74, 50
w = PAD * 2 + LANE_W + 260
h = TOP + STEP * (len(EVENTS) + 1) + PAD + 130

X = [PAD + 140 + i * LANE_W for i in range(2)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>']

# 门控提示条(两行,避免长句溢出)
L.append(f'<rect x="{PAD}" y="42" width="{w-2*PAD}" height="60" rx="8" '
          'fill="#fef3c7" stroke="#d97706"/>')
for gi, gline in enumerate(GATE_NOTE_LINES):
    L.append(f'<text x="{w/2}" y="{64+gi*22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" fill="#78350f">{esc(gline)}</text>')

# 泳道头 + 生命线
LANE_COLOR = ["#3b82f6", "#f59e0b"]
for i, name in enumerate(LANES):
    x = X[i]
    L.append(f'<rect x="{x-130}" y="{TOP-52}" width="260" height="40" rx="8" '
              f'fill="#e2e8f0" stroke="{LANE_COLOR[i]}" stroke-width="2"/>')
    for li, line in enumerate(name.split("\n")):
        L.append(f'<text x="{x}" y="{TOP-33+li*16}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(line)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-10}" x2="{x}" y2="{h-PAD-110}" '
              f'stroke="{LANE_COLOR[i]}" stroke-width="1.5" stroke-dasharray="4,4" opacity="0.5"/>')

for i, (src, dst, label) in enumerate(EVENTS):
    y = TOP + STEP * (i + 1)
    x1, x2 = X[LANE_KEY[src]], X[LANE_KEY[dst]]
    if src == dst:
        # 同泳道内部动作:短横线加圆点;最右侧泳道文字向左长出,避免出画布
        is_rightmost = LANE_KEY[src] == len(LANES) - 1
        L.append(f'<circle cx="{x1}" cy="{y}" r="5" fill="{LANE_COLOR[LANE_KEY[src]]}"/>')
        if is_rightmost:
            L.append(f'<text x="{x1-16}" y="{y+4}" text-anchor="end" font-family="sans-serif" '
                      f'font-size="12.5" fill="#334155">{esc(label)}</text>')
        else:
            L.append(f'<text x="{x1+16}" y="{y+4}" font-family="sans-serif" font-size="12.5" '
                      f'fill="#334155">{esc(label)}</text>')
    else:
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" '
                  'stroke-width="1.5" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{PAD+30}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">t{i+1}</text>')

# 小字注:图为简化三步握手,ProducerAcquire 未画
L.append(f'<text x="{w/2}" y="{TOP + STEP * (len(EVENTS) + 1) + 22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#64748b">'
          f'图为简化的三步握手(Commit/Wait/Release)；ProducerAcquire「申请空位」发生在 load 之前，词汇见第 24 章。</text>')

# 底部数字条
NUMS = [
    ("WARPS_PER_TASK", "4"),
    ("THREADS_PER_TASK", "128"),
    ("task id = warpId / 4", "4"),
    ("默认 num_consumer_groups", "0"),
]
ny = h - 110
L.append(f'<rect x="{PAD}" y="{ny}" width="{w-2*PAD}" height="56" rx="8" '
          'fill="#eff6ff" stroke="#93c5fd"/>')
seg_w = (w - 2*PAD) / len(NUMS)
for i, (label, val) in enumerate(NUMS):
    cx = PAD + seg_w * i + seg_w / 2
    L.append(f'<text x="{cx}" y="{ny+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="16" font-weight="bold" fill="#1e40af">{esc(val)}</text>')
    L.append(f'<text x="{cx}" y="{ny+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#334155">{esc(label)}</text>')

CAPTION_LINES = [
    "producer 泳道搬 load、consumer 泳道算 dot,token 握手保序——这正是第 24 章 create_token/",
    "ProducerCommit/ConsumerWait/ConsumerRelease 词汇被 pass 落地的样子,别夸大它在默认路径的作用。",
]
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{w/2}" y="{h-24+i*17}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#475569">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-ws-warpgroup-swimlane.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
