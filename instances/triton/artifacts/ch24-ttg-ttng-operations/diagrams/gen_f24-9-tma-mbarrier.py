#!/usr/bin/env python3
"""f24-9-tma-mbarrier: swimlane 模板(3 泳道)。
Warp / mbarrier(共享内存) / TMA 引擎。展示 init_barrier -> barrier_expect -> async_tma_copy
-> TMA 到达通知 -> wait_barrier 阻塞到相位翻转 的整块搬运协议。
底部信息条给硬件门槛/操作数/降级 PTX/粒度对照(numbers 全覆盖)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["Warp（发起线程）", "mbarrier（共享内存）", "TMA 引擎"]
EVENTS = [
    ("Warp（发起线程）", "mbarrier（共享内存）", "init_barrier"),
    ("Warp（发起线程）", "mbarrier（共享内存）", "barrier_expect(bytes=N)（声明期望字节）"),
    ("Warp（发起线程）", "TMA 引擎", "async_tma_copy_global_to_local(desc_ptr, coord, barrier, result)"),
    ("TMA 引擎", "mbarrier（共享内存）", "整块搬运完成 → arrive（通知）"),
    ("Warp（发起线程）", "mbarrier（共享内存）", "wait_barrier（阻塞到相位翻转）"),
    ("mbarrier（共享内存）", "Warp（发起线程）", "相位完成：数据齐，放行"),
]
LANE_W, TOP, STEP, PAD = 430, 90, 66, 50
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 220
h = TOP + STEP * (len(EVENTS) + 1) + 160

X = {name: PAD + 90 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="s" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
         f'font-size="17" font-weight="bold" fill="#0f172a">TMA + mbarrier：整块搬运交专用引擎，报数式完成同步</text>')

for name, x in X.items():
    L.append(f'<rect x="{x-115}" y="{TOP-40}" width="230" height="30" rx="6" '
             'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x}" y="{TOP-19}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-9}" x2="{x}" y2="{h-100}" '
             'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for i, (src, dst, label) in enumerate(EVENTS):
    y = TOP + STEP * (i + 1)
    x1, x2 = X[src], X[dst]
    color = "#7c3aed" if "TMA" in src or "TMA" in dst else "#334155"
    marker = "url(#s)" if "TMA" in src or "TMA" in dst else "url(#a)"
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" '
             f'stroke-width="1.8" marker-end="{marker}"/>')
    anchor = "middle"
    tx = (x1 + x2) / 2
    L.append(f'<text x="{tx}" y="{y-8}" text-anchor="{anchor}" '
             f'font-family="sans-serif" font-size="11.5" fill="{color}">{esc(label)}</text>')
    L.append(f'<text x="{PAD-16}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#64748b">t{i+1}</text>')

# 底部信息条(numbers 全覆盖)
info_top = h - 130
info_lines = [
    "TMA / mbarrier 硬件门槛：computeCapability ≥ 90（Hopper/sm90）",
    "async_tma_copy 定位操作数：desc_ptr + coord（Variadic<I32>）+ barrier(memdesc) + result(memdesc) + pred",
    "init_barrier 降级 PTX：mbarrier.init.shared::cta.b64；wait_barrier 降级 PTX：mbarrier.try_wait.parity.shared.b64",
    "对照 cp.async 粒度：cp.async 线程级逐指针（sm80）vs TMA 张量块级整块（sm90）",
]
info_h = 24 + len(info_lines) * 22 + 10
L.append(f'<rect x="{PAD}" y="{info_top}" width="{w-2*PAD}" height="{info_h}" rx="8" '
         'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
for i, line in enumerate(info_lines):
    L.append(f'<text x="{PAD+18}" y="{info_top+24+i*22}" font-family="sans-serif" '
             f'font-size="11.5" fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f24-9-tma-mbarrier.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
