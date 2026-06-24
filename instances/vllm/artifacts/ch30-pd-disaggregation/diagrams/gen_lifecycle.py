#!/usr/bin/env python3
"""ch30-worker-lifecycle: 时间轴生命周期泳道图。
三条泳道：connector 调用 / model forward / 后台传输；标出 load 与 compute 重叠区。"""
import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 1300, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('<marker id="arR" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# title
L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-size="21" font-weight="bold" fill="#0f172a">_get_kv_connector_output：把 forward 夹在 connector 生命周期里</text>')

# lane geometry
lane_x = 250
lane_w = w - lane_x - 30
lanes = [
    ("connector 调用", 90, "#eff6ff"),
    ("model forward", 230, "#f0fdf4"),
    ("后台传输", 370, "#fef2f2"),
]
lane_h = 110
for name, y, bg in lanes:
    L.append(f'<rect x="{lane_x}" y="{y}" width="{lane_w}" height="{lane_h}" fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<text x="{lane_x-14}" y="{y+lane_h/2+5}" text-anchor="end" font-size="15" font-weight="bold" fill="#334155">{esc(name)}</text>')

# time axis bottom
ty = 500
L.append(f'<line x1="{lane_x}" y1="{ty}" x2="{lane_x+lane_w}" y2="{ty}" stroke="#475569" stroke-width="1.5" marker-end="url(#ar)"/>')
L.append(f'<text x="{lane_x+lane_w}" y="{ty+22}" text-anchor="end" font-size="13" fill="#64748b">时间 →</text>')

def box(x, y, bw, bh, text, fill, stroke, fs=13, tcol="#0f172a", bold=False):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    fw = ' font-weight="bold"' if bold else ''
    lines = text.split("\n")
    y0 = y + bh/2 - (len(lines)-1)*8 + 4
    for i, ln in enumerate(lines):
        L.append(f'<text x="{x+bw/2}" y="{y0+i*16}" text-anchor="middle" font-size="{fs}"{fw} fill="{tcol}">{esc(ln)}</text>')

# connector lane events
c_y = 110
box(lane_x+10, c_y, 130, 70, "bind_connector\n_metadata", "#dbeafe", "#3b82f6")
box(lane_x+155, c_y, 150, 70, "start_load_kv\n(异步发起,\n立即返回)", "#dbeafe", "#3b82f6")
box(lane_x+800, c_y, 130, 70, "wait_for_save\n(阻塞收齐)", "#fee2e2", "#ef4444", tcol="#b91c1c", bold=True)
box(lane_x+945, c_y, 95, 70, "get_finished\n→ clear", "#dbeafe", "#3b82f6")

# forward lane: layers
f_y = 250
nlayers = 5
lx0 = lane_x + 330
lw = 80
gap = 14
for i in range(nlayers):
    x = lx0 + i*(lw+gap)
    fill = "#bbf7d0" if i in (0,) else "#dcfce7"
    box(x, f_y, lw, 70, f"层 {i}", fill, "#22c55e", fs=14)
# wait_for_layer_load marker on layer 2 (where KV first needed)
wll_x = lx0 + 2*(lw+gap)
# callout placed to the RIGHT, below forward lane, clear of the load bar
box(wll_x+lw+8, 250, 170, 64, "wait_for_layer_load\n(阻塞到该层 KV 到位,\n与前面层计算重叠)", "#fee2e2", "#ef4444", fs=11, tcol="#b91c1c")
L.append(f'<line x1="{wll_x+lw+8}" y1="282" x2="{wll_x+lw+1}" y2="{f_y+35}" stroke="#b91c1c" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#arR)"/>')
# save_kv_layer marker on layer exit
skl_x = lx0 + 4*(lw+gap)
L.append(f'<line x1="{skl_x+lw/2}" y1="{f_y+72}" x2="{skl_x+lw/2}" y2="{390}" stroke="#0891b2" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#ar)"/>')

# background transfer lane: overlap region
t_y = 390
# load bar overlapping with forward
load_x0 = lane_x+230
load_x1 = lx0 + 2*(lw+gap) + lw/2
L.append(f'<rect x="{load_x0}" y="{t_y}" width="{load_x1-load_x0}" height="44" rx="6" fill="#fde68a" stroke="#f59e0b" stroke-width="1.5"/>')
L.append(f'<text x="{(load_x0+load_x1)/2}" y="{t_y+27}" text-anchor="middle" font-size="13" font-weight="bold" fill="#92400e">后台 load KV（与计算重叠）</text>')
# save bar
save_x0 = skl_x+lw/2
save_x1 = lane_x+800
L.append(f'<rect x="{save_x0}" y="{t_y}" width="{save_x1-save_x0}" height="44" rx="6" fill="#a5f3fc" stroke="#06b6d4" stroke-width="1.5"/>')
L.append(f'<text x="{(save_x0+save_x1)/2}" y="{t_y+27}" text-anchor="middle" font-size="12" fill="#155e75">后台 save</text>')

# arrow start_load_kv -> load bar
L.append(f'<line x1="{lane_x+230}" y1="180" x2="{load_x0+4}" y2="{t_y-2}" stroke="#475569" stroke-width="1.3" marker-end="url(#ar)"/>')

# bottom note
L.append(f'<text x="{lane_x}" y="540" font-size="13" fill="#334155">enter：bind → start_load_kv（发起即返回）。finally：wait_for_save → get_finished → clear。异常路径也保证收尾不漏块。</text>')
L.append(f'<text x="{lane_x}" y="562" font-size="13" fill="#334155">输出 KVConnectorOutput（finished_sending / finished_recving）随 ModelRunnerOutput 回传 scheduler，接第 29 章的提升 / 释放闭环。</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch30-pd-disaggregation/diagrams/ch30-worker-lifecycle.svg","w").write('\n'.join(L))
print("ok")
