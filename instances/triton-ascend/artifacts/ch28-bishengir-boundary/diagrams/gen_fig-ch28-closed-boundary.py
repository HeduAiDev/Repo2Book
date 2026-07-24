#!/usr/bin/env python3
"""swimlane 模板改造:开源侧 Python 与闭源二进制 bishengir-compile 之间的一次
subprocess 往返(compiler.py:L448-L499)。中段插一个「黑盒」矩形贴在闭源泳道上,
标注书读到此为止、内部不猜。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["开源侧 (compiler.py)", "闭源二进制 (bishengir-compile)"]
EVENTS = [  # (from_lane, to_lane, label lines)
    (0, 1, ["subprocess.run(cmd_list, env,",
            "stdout=PIPE, stderr=PIPE, check=True)"]),
    (1, 0, ["退出码 + stdout/stderr",
            "+ kernel.o / libkernel.so 落盘"]),
]
CMD_NOTE = "cmd_list = [编译器, ttadapter_path] + 选项 + ['-o', bin_file]（compiler.py:L448-L452）"
CALL_NOTE = "subprocess.run 调用点（compiler.py:L465-L471）——本章读到这里为止"
BLACKBOX_TEXT = ["闭源内部（本书不猜、无源码可读）：", "Linalg → HFusion → HIVM → NPU 二进制"]

LANE_W, TOP, STEP, PAD = 560, 130, 130, 44
BOX_W_BLACK = 360  # 黑盒宽度,决定右侧留白(比泳道头 300 宽,是画布右边界的约束项)
X = {i: PAD + 130 + i * LANE_W for i in range(len(LANES))}
w = X[len(LANES) - 1] + BOX_W_BLACK / 2 + PAD
h = TOP + STEP * len(EVENTS) + 200

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("subprocess.run 是开源可读与闭源黑盒的诚实分界")}</text>']

lane_bottom = h - 46
for i, name in enumerate(LANES):
    x = X[i]
    is_closed = (i == 1)
    head_fill = "#fee2e2" if is_closed else "#e0f2fe"
    head_stroke = "#b91c1c" if is_closed else "#0369a1"
    L.append(f'<rect x="{x-150}" y="{TOP-46}" width="300" height="32" rx="6" '
              f'fill="{head_fill}" stroke="{head_stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x}" y="{TOP-25}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-14}" x2="{x}" y2="{lane_bottom}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

# 事件箭头
for i, (src, dst, lines) in enumerate(EVENTS):
    y = TOP + STEP * (i + 1) - 30
    x1, x2 = X[src], X[dst]
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" '
              'stroke-width="1.8" marker-end="url(#a)"/>')
    n = len(lines)
    y0 = y - 10 - (n - 1) * 14
    for k, line in enumerate(lines):
        L.append(f'<text x="{(x1+x2)/2}" y="{y0+k*14}" text-anchor="middle" '
                  f'font-family="monospace" font-size="11.5" fill="#334155">{esc(line)}</text>')

# 黑盒:贴在闭源泳道生命线上,在两个事件箭头之间
box_y = TOP + STEP - 6
box_w, box_h = BOX_W_BLACK, 74
bx = X[1] - box_w / 2
L.append(f'<rect x="{bx}" y="{box_y}" width="{box_w}" height="{box_h}" rx="8" '
          'fill="#1e293b" stroke="#0f172a" stroke-width="2"/>')
for k, line in enumerate(BLACKBOX_TEXT):
    L.append(f'<text x="{X[1]}" y="{box_y+30+k*22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" '
              f'font-weight="{"bold" if k==0 else "normal"}" fill="#fbbf24">{esc(line)}</text>')

# 底部说明条
note_top = h - 110
L.append(f'<rect x="{PAD}" y="{note_top}" width="{w-PAD*2}" height="66" rx="6" '
          'fill="#eff6ff" stroke="#3b82f6"/>')
L.append(f'<text x="{PAD+16}" y="{note_top+26}" font-family="monospace" '
          f'font-size="11.5" fill="#1e3a8a">{esc(CMD_NOTE)}</text>')
L.append(f'<text x="{PAD+16}" y="{note_top+48}" font-family="sans-serif" '
          f'font-size="12" fill="#1e3a8a">{esc(CALL_NOTE)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch28-closed-boundary.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
