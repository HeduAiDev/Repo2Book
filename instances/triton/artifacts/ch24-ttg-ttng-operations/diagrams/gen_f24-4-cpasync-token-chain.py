#!/usr/bin/env python3
"""f24-4-cpasync-token-chain: swimlane 模板。
两泳道:Warp 指令流 / cp.async 引擎(后台)。事件串 async.token,展示访存与计算重叠。
底部信息条给硬件门槛/字节宽/等待参数/token 类型(numbers 全覆盖)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["Warp 指令流", "cp.async 引擎（后台）"]
EVENTS = [
    ("Warp 指令流", "cp.async 引擎（后台）", "async_copy_global_to_local(dst, src) → %t0"),
    ("Warp 指令流", "cp.async 引擎（后台）", "async_commit_group(%t0) → %t1（打成一组）"),
    ("Warp 指令流", "Warp 指令流", "（继续算上一轮，访存与计算重叠）"),
    ("Warp 指令流", "cp.async 引擎（后台）", "async_wait(%t1, num=0)"),
    ("cp.async 引擎（后台）", "Warp 指令流", "未完成组 ≤ num：数据就绪，放行"),
]
LANE_W, TOP, STEP, PAD = 560, 90, 66, 50
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 160
h = TOP + STEP * (len(EVENTS) + 1) + 150

X = {name: PAD + 80 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="s" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
         f'font-size="17" font-weight="bold" fill="#0f172a">cp.async 三件套：用 async.token 串成 SSA 依赖链</text>')

for name, x in X.items():
    L.append(f'<rect x="{x-110}" y="{TOP-40}" width="220" height="30" rx="6" '
             'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x}" y="{TOP-19}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-9}" x2="{x}" y2="{h-90}" '
             'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for i, (src, dst, label) in enumerate(EVENTS):
    y = TOP + STEP * (i + 1)
    x1, x2 = X[src], X[dst]
    if src == dst:  # 注记(非消息):Warp 侧继续做别的事,与拷贝重叠——画一个虚线note框,不用箭头
        note_w, note_h = 340, 30
        nx, ny = x1 + 30, y - note_h / 2
        L.append(f'<rect x="{nx}" y="{ny}" width="{note_w}" height="{note_h}" rx="6" '
                 'fill="#f0fdf4" stroke="#16a34a" stroke-width="1.2" stroke-dasharray="4,3"/>')
        L.append(f'<text x="{nx+note_w/2}" y="{y+4}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="11.5" fill="#15803d">{esc(label)}</text>')
    else:
        color = "#0369a1" if i < 2 else "#334155"
        marker = "url(#s)" if i < 2 else "url(#a)"
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" '
                 f'stroke-width="1.8" marker-end="{marker}"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-8}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12" fill="{color}">{esc(label)}</text>')
    L.append(f'<text x="{PAD-16}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#64748b">t{i+1}</text>')

# 底部信息条(numbers 全覆盖)
info_top = h - 120
info_lines = [
    "cp.async 硬件门槛：computeCapability ≥ 80（Ampere/sm80 起）；合法单次异步拷贝字节宽：{4, 8, 16} bytes",
    "async_wait 等待参数：num（未完成组数，I32Attr）；三件套均吞吐/产出 async.token——串依赖的粘合剂",
]
info_h = 24 + len(info_lines) * 24 + 10
L.append(f'<rect x="{PAD}" y="{info_top}" width="{w-2*PAD}" height="{info_h}" rx="8" '
         'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
for i, line in enumerate(info_lines):
    L.append(f'<text x="{PAD+18}" y="{info_top+24+i*24}" font-family="sans-serif" '
             f'font-size="11.5" fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f24-4-cpasync-token-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
