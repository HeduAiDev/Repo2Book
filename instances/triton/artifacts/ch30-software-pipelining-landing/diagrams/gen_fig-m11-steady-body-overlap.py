#!/usr/bin/env python3
"""fig-m11-steady-body-overlap: 稳态循环单次迭代同体并行跑 S0(I+2) 取数与
S2(I) 计算,操作数各接各拍版本。
数字来自 explainer.json m11.figure_specs[0].numbers:
  2 = 重叠深度 = maxStage = 2
  3 = matmul_sm90_ns3.ttgir.mlir:L61 numBuffers = 3
  7 = matmul_sm90_ns3.ttgir.mlir:L83 %64:7 稳态 iter_args
  (wgmma wait pendings 水位细节留给本章末 Hopper 收尾一节交代)
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LEFT_TITLE = "S2:计算第 I 拍(读旧数据)"
LEFT_STEPS = [
    "extractIdx 推进\n= (旧+1) % 3",
    "memdesc_subview 读槽\n(2拍前写入的槽)",
    "warp_group_dot(%86)",
]

RIGHT_TITLE = "S0:预取第 I+2 拍(写新指针)"
RIGHT_STEPS = [
    "insertIdx 推进\n= (旧+1) % 3",
    "memdesc_subview 写槽\n(本拍推进后指针%88/%89)",
    "async_copy(mask)+commit\n(%92,%93,%95,%96)",
]

BOX_W, BOX_H, VGAP = 300, 62, 20
PAD, TOP, PANEL_GAP = 44, 150, 70

body_h = len(LEFT_STEPS) * (BOX_H + VGAP) - VGAP
w = PAD * 2 + BOX_W * 2 + PANEL_GAP
merge_h = 84
h = TOP + body_h + 40 + merge_h + 70

lx = PAD
rx = PAD + BOX_W + PANEL_GAP
lcx = lx + BOX_W / 2
rcx = rx + BOX_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">稳态循环一拍,同体并行两个逻辑迭代</text>',
     f'<text x="{w/2}" y="54" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">重叠深度 = maxStage = 2;numBuffers = 3(matmul_sm90_ns3.ttgir.mlir)</text>']

# 外框:同一拍
outer_top = TOP - 46
outer_h = body_h + 60
L.append(f'<rect x="{PAD-18}" y="{outer_top}" width="{w-2*(PAD-18)}" height="{outer_h}" '
          'rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6,4"/>')
L.append(f'<text x="{w/2}" y="{outer_top+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#475569">同一拍(本次迭代)</text>')


def panel(cx, x0, title, steps, color):
    fill, stroke = color
    L.append(f'<text x="{cx}" y="{TOP-4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + 18 + i * (BOX_H + VGAP)
        L.append(f'<rect x="{x0}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        lines = step.split("\n")
        n = len(lines)
        y0 = y + BOX_H / 2 - (n - 1) * 9 + 5
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx}" y="{y0+k*18}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="#0f172a">'
                      f'{esc(line)}</text>')
        if i < len(steps) - 1:
            y2 = y + BOX_H
            L.append(f'<line x1="{cx}" y1="{y2}" x2="{cx}" y2="{y2+VGAP-3}" '
                      'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
    return TOP + 18 + len(steps) * (BOX_H + VGAP) - VGAP


bottom_l = panel(lcx, lx, LEFT_TITLE, LEFT_STEPS, ("#dbeafe", "#1d4ed8"))
bottom_r = panel(rcx, rx, RIGHT_TITLE, RIGHT_STEPS, ("#fef3c7", "#d97706"))
box_bottom = max(bottom_l, bottom_r)

# 底部合流:scf.yield
merge_y = box_bottom + 26
merge_w = BOX_W * 2 + PANEL_GAP
L.append(f'<rect x="{lx}" y="{merge_y}" width="{merge_w}" height="{merge_h}" rx="10" '
          'fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>')
L.append(f'<text x="{lx+merge_w/2}" y="{merge_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#5b21b6">scf.yield 回灌(iter_args = 7)</text>')
L.append(f'<text x="{lx+merge_w/2}" y="{merge_y+48}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#5b21b6">'
          f'acc、a_ptrs/b_ptrs、insertIdx/extractIdx、2 个 async.token 版本各前移一格</text>')
L.append(f'<text x="{lx+merge_w/2}" y="{merge_y+66}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#7c3aed">'
          f'-&gt; 交给下一拍,循环重复直至谓词关闭(见 prologue/epilogue 图)</text>')
for cx in (lcx, rcx):
    L.append(f'<line x1="{cx}" y1="{box_bottom}" x2="{cx}" y2="{merge_y}" '
              'stroke="#94a3b8" stroke-width="1.3" marker-end="url(#a)"/>')

foot_y = merge_y + merge_h + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#374151">同一段代码里,S0 用本拍刚推进的新指针写入、S2 用 2 拍前存下的旧槽读出'
          f'——取数延迟被藏进 2 拍计算之后。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-m11-steady-body-overlap.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
