#!/usr/bin/env python3
"""fig25-2b: AscendCompiler.compile() 的二选一编译控制流。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

w, h = 1180, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="#1e293b">AscendCompiler.compile()：一次编译，两条路</text>')

def box(x, y, bw, bh, fill, stroke, lines, fs=15, tcol="#1e293b", rx=12, bold0=True):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    n = len(lines)
    total = n * (fs + 5)
    sy = y + (bh - total)/2 + fs
    for i, ln in enumerate(lines):
        fw = 'bold' if (i == 0 and bold0) else 'normal'
        L.append(f'<text x="{x+bw/2}" y="{sy + i*(fs+5)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcol}">{esc(ln)}</text>')

def diamond(cx, cy, rw, rh, lines, fs=15):
    pts = f"{cx},{cy-rh} {cx+rw},{cy} {cx},{cy+rh} {cx-rw},{cy}"
    L.append(f'<polygon points="{pts}" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>')
    n = len(lines)
    sy = cy - (n-1)*(fs+3)/2 + fs/2 - 2
    for i, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{sy + i*(fs+3)}" text-anchor="middle" font-size="{fs}" fill="#713f12">{esc(ln)}</text>')

def varrow(x, y1, y2, col="#475569"):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{col}" stroke-width="2.2" marker-end="url(#a)"/>')

def label(x, y, t, col="#475569", fs=14, anchor="middle"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" fill="{col}">{esc(t)}</text>')

cx = w/2
# entry
box(cx-200, 58, 400, 60, "#e0f2fe", "#0284c7",
    ["compile(graph, example_inputs, …)",
     "deepcopy(graph) + fake_mode 重绑 inputs"], fs=14)
varrow(cx, 118, 144)

# diamond enable_npugraph_ex?
diamond(cx, 188, 165, 44, ["enable_npugraph_ex ?"])

# LEFT = True branch (npugraph_ex)
lx = 70
L.append(f'<line x1="{cx-165}" y1="188" x2="{lx+200}" y2="188" stroke="#475569" stroke-width="2.2"/>')
varrow(lx+200, 188, 250)
label(cx-260, 174, "True：图编译优化", "#16a34a", 14, "middle")
box(lx, 250, 400, 52, "#dcfce7", "#16a34a", ["npugraph_ex_compile"], fs=17, tcol="#15803d")
varrow(lx+200, 302, 336)
diamond(lx+200, 380, 165, 44, ["import npugraph_ex ?"])
# ok -> nge backend
varrow(lx+200, 424, 470)
label(lx+200, 452, "成功", "#16a34a", 13)
box(lx, 470, 400, 56, "#f0fdf4", "#16a34a",
    ["nge.get_npu_backend(config)", "（npugraph_ex 走扁平 options 字典）"], fs=13.5, tcol="#15803d")
# ImportError -> torchair
L.append(f'<line x1="{lx+200+165}" y1="380" x2="{lx+430}" y2="380" stroke="#475569" stroke-width="2.2"/>')
L.append(f'<line x1="{lx+430}" y1="380" x2="{lx+430}" y2="566" stroke="#475569" stroke-width="2.2"/>')
varrow_x = lx+430
L.append(f'<line x1="{varrow_x}" y1="566" x2="{lx+400}" y2="566" stroke="#475569" stroke-width="2.2" marker-end="url(#a)"/>')
label(lx+440, 372, "ImportError 回退", "#b45309", 13, "start")
box(lx, 552, 400, 56, "#fff7ed", "#ea580c",
    ["torchair.get_npu_backend(config)", "config.mode = 'reduce-overhead'（aclgraph）"], fs=13.5, tcol="#9a3412")
label(lx+200, 640, "两侧都经 _configure_backend 配成 aclgraph 模式", "#475569", 13.5)

# RIGHT = False branch (fusion pass)
rx = w - 70 - 400
L.append(f'<line x1="{cx+165}" y1="188" x2="{rx+200}" y2="188" stroke="#475569" stroke-width="2.2"/>')
varrow(rx+200, 188, 250)
label(cx+265, 174, "False：自家融合 pass", "#b45309", 14, "middle")
box(rx, 250, 400, 52, "#fef2f2", "#dc2626", ["fusion_pass_compile"], fs=17, tcol="#b91c1c")
varrow(rx+200, 302, 348)
box(rx, 348, 400, 56, "#fff1f2", "#e11d48",
    ["compile_fx + aot_autograd", "（graph_returns_tuple 适配输出）"], fs=13.5, tcol="#be123c")
varrow(rx+200, 404, 450)
box(rx, 450, 400, 76, "#fdf2f8", "#db2777",
    ["inner_compile 取", "compiler_config[COMPILATION_PASS_KEY]", "= GraphFusionPassManager 跑融合 pass"], fs=13.5, tcol="#9d174d")
label(rx+200, 600, "torch_npu 暂不支持 triton →", "#475569", 13.5)
label(rx+200, 620, "自定义 pass manager 只跑 pattern matcher", "#475569", 13.5)

L.append('</svg>')
open("fig25-2b-compile-twofold.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote fig25-2b-compile-twofold.svg")
