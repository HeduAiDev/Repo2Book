#!/usr/bin/env python3
"""swimlane 模板(定制):Python -> C++ launcher 跨语言发射一跳。
两泳道:左 Python(jit.py run 收尾),右 后端 C++ launcher(driver 子系统);
中间竖直粗虚线标『Python | C++』分界(需真设备)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LANES = ["Python（jit.py run 收尾）", "后端 C++ launcher（driver 子系统，见第十二章）"]
STEPS_LEFT = [
    ("launch_metadata = kernel.launch_metadata(", "grid, stream, *non_constexpr_vals)",
     "常态无 hook -> None"),
    ("kernel.run(grid_0,grid_1,grid_2, stream,", "function, packed_metadata, launch_metadata,",
     "launch_enter/exit_hook, *non_constexpr_vals)"),
]
RIGHT_STEP = ("接收上述参数 -> cuLaunchKernel", "GPU 上出现一个 kernel 实例", "")

LANE_W, PAD, TOP = 560, 44, 110
LANE_GAP = 150
w = PAD * 2 + LANE_W * 2 + LANE_GAP
h = 430
LX = PAD + LANE_W / 2
RX = PAD + LANE_W + LANE_GAP + LANE_W / 2
boundary_x = PAD + LANE_W + LANE_GAP / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
          'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("跨语言发射一跳：kernel.run(...) 把接力棒交给 C++ launcher")}</text>')

# 泳道头
for name, cx, color in [(LANES[0], LX, "#1d4ed8"), (LANES[1], RX, "#64748b")]:
    L.append(f'<rect x="{cx-LANE_W/2+10}" y="{TOP-46}" width="{LANE_W-20}" height="30" rx="6" '
              f'fill="#e2e8f0" stroke="{color}" stroke-width="1.4"/>')
    L.append(f'<text x="{cx}" y="{TOP-25}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{color}">{esc(name)}</text>')
    L.append(f'<line x1="{cx}" y1="{TOP-10}" x2="{cx}" y2="{h-70}" stroke="#94a3b8" stroke-dasharray="4,4"/>')

y = TOP
box_w = LANE_W - 60
first_box_y = None
for i, lines in enumerate(STEPS_LEFT):
    box_h = 20 * len(lines) + 26
    if first_box_y is None:
        first_box_y = y
    L.append(f'<rect x="{LX-box_w/2}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
              'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/>')
    for k, line in enumerate(lines):
        weight = 'font-weight="bold" ' if k == 0 else ''
        fill = "#1e3a8a" if k == 0 else "#334155"
        size = 12 if k == 0 else 11
        L.append(f'<text x="{LX}" y="{y+20+k*18}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{size}" {weight}fill="{fill}">{esc(line)}</text>')
    y += box_h + 26

last_box_bottom = y - 26  # bottom of kernel.run box

# 跨语言粗箭头(从左泳道最后一个框指向右泳道)
mid_y = last_box_bottom - 30
L.append(f'<line x1="{LX+box_w/2}" y1="{mid_y}" x2="{RX-box_w/2}" y2="{mid_y}" '
          'stroke="#b91c1c" stroke-width="3" marker-end="url(#r)"/>')
L.append(f'<text x="{(LX+box_w/2+RX-box_w/2)/2}" y="{mid_y-14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#b91c1c">'
          f'{esc("⚡ 双语断点")}</text>')
L.append(f'<text x="{(LX+box_w/2+RX-box_w/2)/2}" y="{mid_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#b91c1c">'
          f'{esc("（需真设备）")}</text>')

L.append(f'<rect x="{RX-box_w/2}" y="{mid_y-24}" width="{box_w}" height="72" rx="8" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.6"/>')
L.append(f'<text x="{RX}" y="{mid_y-2}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#7f1d1d">{esc(RIGHT_STEP[0])}</text>')
L.append(f'<text x="{RX}" y="{mid_y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">{esc(RIGHT_STEP[1])}</text>')
L.append(f'<text x="{RX}" y="{mid_y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#b91c1c">{esc("本书 headless 未执行此跳（无 GPU）")}</text>')

foot_y = h - 44
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("数字核对：grid 三维 (4, 1, 1)；运行期实参 non_constexpr 4 个（3 指针 + 1 i32）；")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("launch_enter_hook / launch_exit_hook 常态 None，launch_metadata 直接 None，路径零额外开销。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch11-emission-crosslang.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
