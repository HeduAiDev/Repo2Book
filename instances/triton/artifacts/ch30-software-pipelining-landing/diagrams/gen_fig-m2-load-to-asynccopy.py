#!/usr/bin/env python3
"""fig-m2-load-to-asynccopy: before-after — 一条同步 tt.load 被 pipeliner 就地
拆成『选槽 subview + 异步 cp.async + 封组 commit + 下游 wait』四件套。
数字来自 explainer.json m2.figure_specs[0].numbers:
  2  = sm90_ns3.ttir_tt_load(前)
  0  = sm90_ns3.ttgir_tt_load(后)
  6  = sm90_ns3.ttgir_async_copy
  3  = matmul_sm90_ns3.ttgir.mlir:L61 memdesc<3x128x64xf16> 环形缓冲深度
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LEFT_TITLE = "优化前:同步 tt.load"
LEFT_BADGE = "tt.load 计数 = 2"
LEFT_STEPS = [
    "tt.load %a_ptrs\n-> tensor<128x64xf16>",
    "local_alloc\n-> 共享内存",
    "warp_group_dot",
]

RIGHT_TITLE = "优化后:async copy 四件套"
RIGHT_BADGE = "tt.load 计数 = 0"
RIGHT_STEPS = [
    "memdesc_subview 选槽\n(环形缓冲深度 3)",
    "async_copy_global_to_local\n(cp.async,共 6 次)",
    "async_commit_group\n(封组)",
    "async_wait(下游)\n-> warp_group_dot",
]
RIGHT_HOT = 1  # cp.async 是核心差异,高亮

BOX_W, BOX_H, VGAP = 300, 60, 26
PAD, TOP, PANEL_GAP = 44, 118, 110

left_h = len(LEFT_STEPS) * (BOX_H + VGAP) - VGAP
right_h = len(RIGHT_STEPS) * (BOX_H + VGAP) - VGAP
body_h = max(left_h, right_h)

w = PAD * 2 + BOX_W * 2 + PANEL_GAP
h = TOP + body_h + 68

lx = PAD
rx = PAD + BOX_W + PANEL_GAP
lcx = lx + BOX_W / 2
rcx = rx + BOX_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">循环里喂 dot 的 load:从同步搬运变异步预取</text>',
     f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">fp16 matmul,num_stages=3,sm90(make_ttgir 之后)</text>']


def panel(cx, x0, title, badge, steps, hot):
    L.append(f'<text x="{cx}" y="{TOP-46}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<rect x="{cx-90}" y="{TOP-38}" width="180" height="24" rx="12" '
              'fill="#eef2ff" stroke="#6366f1"/>')
    L.append(f'<text x="{cx}" y="{TOP-21}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#4338ca">{esc(badge)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        is_hot = (hot is not None and i == hot)
        fill = "#fef3c7" if is_hot else "#e2e8f0"
        stroke = "#d97706" if is_hot else "#64748b"
        sw = 2.5 if is_hot else 1
        L.append(f'<rect x="{x0}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines = step.split("\n")
        n = len(lines)
        y0 = y + BOX_H / 2 - (n - 1) * 9 + 5
        for k, line in enumerate(lines):
            fw = 'font-weight="bold" ' if (is_hot and k == 0) else ''
            L.append(f'<text x="{cx}" y="{y0+k*18}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12.5" fill="#0f172a" {fw}>'
                      f'{esc(line)}</text>')
        if i < len(steps) - 1:
            y2 = y + BOX_H
            L.append(f'<line x1="{cx}" y1="{y2}" x2="{cx}" y2="{y2+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')


panel(lcx, lx, LEFT_TITLE, LEFT_BADGE, LEFT_STEPS, None)
panel(rcx, rx, RIGHT_TITLE, RIGHT_BADGE, RIGHT_STEPS, RIGHT_HOT)

# 中央大箭头:优化前 -> 优化后
mid_y = TOP + body_h / 2 - 30
L.append(f'<line x1="{lx+BOX_W+10}" y1="{mid_y}" x2="{rx-10}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="3" marker-end="url(#ah)"/>')
L.append(f'<text x="{(lx+BOX_W+rx)/2}" y="{mid_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#d97706">'
          f'pipeliner 改写</text>')

foot_y = TOP + body_h + 46
L.append(f'<line x1="{PAD}" y1="{foot_y-24}" x2="{w-PAD}" y2="{foot_y-24}" stroke="#e2e8f0"/>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#374151">'
          f'同一循环净变化:tt.load 2→0,async_copy 出现 6 次(2 个 load×2 段 prologue+1 段稳态),'
          f'环形缓冲深度 3。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-m2-load-to-asynccopy.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
