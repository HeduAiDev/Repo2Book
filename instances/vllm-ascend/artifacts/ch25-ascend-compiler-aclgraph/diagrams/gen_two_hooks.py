#!/usr/bin/env python3
"""fig25-1: 三个返回字符串的平台钩子，把 vLLM 编译三件套整体换成 NPU 版。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

w, h = 1180, 540
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# title
L.append(f'<text x="{w/2}" y="42" text-anchor="middle" font-size="26" font-weight="bold" fill="#1e293b">三处平台钩子，换掉 vLLM 整套编译 + 图捕获栈</text>')
L.append(f'<text x="{w/2}" y="74" text-anchor="middle" font-size="16" fill="#64748b">vLLM 编译框架一行不改：current_platform 钩子返回类路径字符串，vLLM resolve_obj_by_qualname 导入它</text>')

rows = [
    ("InductorAdaptor", "get_compile_backend()", "AscendCompiler", "二选一编译后端"),
    ("CUDAGraphWrapper", "get_static_graph_wrapper_cls()", "ACLGraphWrapper", "NPUGraph 捕获 / 重放"),
    ("PostGradPassManager", "get_pass_manager_cls()", "GraphFusionPassManager", "串 pattern 融合 pass"),
]

box_w, box_h = 300, 86
left_x = 60
right_x = w - 60 - box_w
y0 = 130
gap = 130

# column headers
L.append(f'<text x="{left_x + box_w/2}" y="{y0-18}" text-anchor="middle" font-size="17" font-weight="bold" fill="#475569">vLLM 原件（torch / CUDA 版）</text>')
L.append(f'<text x="{right_x + box_w/2}" y="{y0-18}" text-anchor="middle" font-size="17" font-weight="bold" fill="#7c3aed">NPU 版（vllm_ascend）</text>')

for i, (lname, hook, rname, note) in enumerate(rows):
    y = y0 + i * gap
    cy = y + box_h/2
    # left box
    L.append(f'<rect x="{left_x}" y="{y}" width="{box_w}" height="{box_h}" rx="12" fill="#f1f5f9" stroke="#94a3b8" stroke-width="2"/>')
    L.append(f'<text x="{left_x + box_w/2}" y="{cy+7}" text-anchor="middle" font-size="20" font-weight="bold" fill="#334155">{esc(lname)}</text>')
    # right box
    L.append(f'<rect x="{right_x}" y="{y}" width="{box_w}" height="{box_h}" rx="12" fill="#f3e8ff" stroke="#7c3aed" stroke-width="2.5"/>')
    L.append(f'<text x="{right_x + box_w/2}" y="{cy}" text-anchor="middle" font-size="20" font-weight="bold" fill="#6d28d9">{esc(rname)}</text>')
    L.append(f'<text x="{right_x + box_w/2}" y="{cy+24}" text-anchor="middle" font-size="14" fill="#7c3aed">{esc(note)}</text>')
    # arrow left->right
    ax0 = left_x + box_w
    ax1 = right_x
    L.append(f'<line x1="{ax0+6}" y1="{cy}" x2="{ax1-8}" y2="{cy}" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#a)"/>')
    # hook label above arrow
    mid = (ax0 + ax1) / 2
    L.append(f'<rect x="{mid-150}" y="{cy-30}" width="300" height="26" rx="6" fill="#ede9fe" stroke="#c4b5fd" stroke-width="1"/>')
    L.append(f'<text x="{mid}" y="{cy-12}" text-anchor="middle" font-size="15" font-family="monospace" fill="#6d28d9">{esc(hook)}</text>')

L.append('</svg>')
open("fig25-1-two-hooks-override.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote fig25-1-two-hooks-override.svg")
