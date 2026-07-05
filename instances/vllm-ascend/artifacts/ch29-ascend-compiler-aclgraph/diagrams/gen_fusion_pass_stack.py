#!/usr/bin/env python3
"""fig25-3: GraphFusionPassManager.configure() 按开关 append 的 6 个融合 Pass 全景。

源码核实：vllm_ascend/compilation/graph_fusion_pass_manager.py:L49-79（6 个 pass、顺序、开关）。
"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

w, h = 1200, 700
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('<marker id="p" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>')
L.append('<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# title
L.append(f'<text x="{w/2}" y="38" text-anchor="middle" font-size="24" font-weight="bold" fill="#1e293b">GraphFusionPassManager.configure()：按 config 开关 append 的 6 个融合 Pass</text>')
L.append(f'<text x="{w/2}" y="66" text-anchor="middle" font-size="15" fill="#64748b">self.passes 按源码顺序逐个 append；每个 Pass 挂一个开关，全关则该 pass 不进栈</text>')

def box(x, y, bw, bh, fill, stroke, lines, fs=15, tcol="#1e293b", rx=10, bold0=True, sw=2):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    n = len(lines)
    total = n * (fs + 4)
    sy = y + (bh - total)/2 + fs
    for i, ln in enumerate(lines):
        fw = 'bold' if (i == 0 and bold0) else 'normal'
        L.append(f'<text x="{x+bw/2}" y="{sy + i*(fs+4)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcol}">{esc(ln)}</text>')

def label(x, y, t, col="#475569", fs=14, anchor="middle", mono=False, fw="normal"):
    fam = ' font-family="monospace"' if mono else ''
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}"{fam} font-weight="{fw}" fill="{col}">{esc(t)}</text>')

# ---- center pass stack ----
sx, sw_box = 415, 370
pass_h, gap = 52, 13
y0 = 108
passes = [
    ("① AddRMSNormQuantFusionPass", "fuse_norm_quant · 非 310P"),
    ("② QKNormRopeFusionPass", "fuse_qknorm_rope"),
    ("③ MatmulAllReduceAddRMSNormPass", "fuse_allreduce_rms"),
    ("④ MulsAddFusionPass", "fuse_muls_add · 非 310P"),
    ("⑤ SequenceParallelismPass", "enable_sp"),
    ("⑥ SequenceParallelismMoePass", "enable_sp"),
]
# group label for the stack
L.append(f'<text x="{sx+sw_box/2}" y="{y0-14}" text-anchor="middle" font-size="15" font-weight="bold" fill="#7c3aed">self.passes（按序 append）</text>')

pass_ys = []
for i, (name, sw) in enumerate(passes):
    y = y0 + i*(pass_h+gap)
    pass_ys.append(y)
    # SP passes (last two) tinted differently
    if i >= 4:
        fill, stroke, tcol, scol = "#eef2ff", "#6366f1", "#3730a3", "#4f46e5"
    else:
        fill, stroke, tcol, scol = "#f3e8ff", "#7c3aed", "#6d28d9", "#7c3aed"
    L.append(f'<rect x="{sx}" y="{y}" width="{sw_box}" height="{pass_h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{sx+16}" y="{y+22}" text-anchor="start" font-size="15.5" font-weight="bold" fill="{tcol}">{esc(name)}</text>')
    L.append(f'<text x="{sx+16}" y="{y+42}" text-anchor="start" font-size="13" font-family="monospace" fill="{scol}">{esc("if "+sw)}</text>')

# enable_sp bracket for 5 & 6 (slim bracket, no overlapping text — boxes already show "if enable_sp")
by0 = pass_ys[4]
by1 = pass_ys[5] + pass_h
bx = sx + sw_box + 6
L.append(f'<path d="M{bx},{by0} q8,0 8,8 L{bx+8},{(by0+by1)/2-8} q0,8 6,8 q-6,0 -6,8 L{bx+8},{by1-8} q0,8 -8,8" fill="none" stroke="#6366f1" stroke-width="1.6"/>')

# ---- left: dummy anchors ----
dx, dw, dy, dh = 40, 320, 150, 168
box(dx, dy, dw, dh, "#fff7ed", "#ea580c",
    ["register_dummy_fusion_op",
     "挂 torch.ops._C_ascend.* 空壳占位算子：",
     "rms_norm / fused_add_rms_norm /",
     "static_scaled_fp8_quant / …",
     "",
     "→ pattern 注册时有锚点可抓，"
     ,"  真实算子在别处由 torch.library 装配"],
    fs=12.8, tcol="#9a3412", bold0=True)
# arrow dummy -> stack
ay = dy + dh/2
L.append(f'<line x1="{dx+dw}" y1="{ay}" x2="{sx-6}" y2="{ay}" stroke="#ea580c" stroke-width="2.2" marker-end="url(#a)"/>')
label((dx+dw+sx)/2, ay-8, "提供占位锚点", "#ea580c", 12.5)

# ---- right top: double registration ----
rx, rw = 825, 335
reg_y, reg_h = 108, 96
box(rx, reg_y, rw, reg_h, "#f5f3ff", "#7c3aed",
    ["BasePattern.register：一份 pattern 双注册",
     "（每个 pass 内的 pattern 都走这里）",
     "同一对 pattern_fn / replacement_fn →"], fs=13, tcol="#6d28d9", bold0=True)
# two targets
ty = reg_y + reg_h + 28
box(rx, ty, rw, 50, "#eff6ff", "#2563eb",
    ["pm.register_replacement(...)", "→ torch Inductor（供分支 B 用）"], fs=12.8, tcol="#1d4ed8")
box(rx, ty+62, rw, 50, "#ecfdf5", "#16a34a",
    ["nge.register_replacement(..., extra_check)", "→ npugraph_ex（供分支 A 用，跨 stream 拒融）"], fs=12.5, tcol="#15803d")
# arrows from reg to two targets
L.append(f'<line x1="{rx+rw/2}" y1="{reg_y+reg_h}" x2="{rx+rw/2}" y2="{ty-4}" stroke="#7c3aed" stroke-width="2" marker-end="url(#p)"/>')

# ---- right lower: side note ----
ny = ty + 130
box(rx, ny, rw, 120, "#f8fafc", "#94a3b8",
    ["目录里有、但不进 self.passes：",
     "· noop_elimination / allgather_chunk_noop",
     "  → 在 SP pass 内部被串起来调用",
     "· base_pattern → 所有 pattern 的基类，",
     "  不是独立 pass"], fs=12.6, tcol="#475569", bold0=True)

# ---- bottom: concrete fusion example ----
ex_y = 545
L.append(f'<rect x="40" y="{ex_y}" width="{w-80}" height="120" rx="12" fill="#fefce8" stroke="#ca8a04" stroke-width="2"/>')
L.append(f'<text x="62" y="{ex_y+26}" text-anchor="start" font-size="15" font-weight="bold" fill="#854d0e">一个融合实例（AddRMSNormQuantPattern）：2 kernel + 1 中间张量 → 1 kernel + 0</text>')
# pattern box
pbx, pbw = 62, 470
box(pbx, ex_y+40, pbw, 64, "#fff1f2", "#e11d48",
    ["pattern：npu_add_rms_norm_bias(...) → out0=output[0]",
     "再 quantize(out0, scale, …)　【2 算子 + 中间张量 out0】"], fs=12.5, tcol="#be123c", bold0=False)
# arrow
mx = pbx+pbw+58
L.append(f'<line x1="{pbx+pbw+8}" y1="{ex_y+72}" x2="{mx+30}" y2="{ex_y+72}" stroke="#16a34a" stroke-width="2.4" marker-end="url(#g)"/>')
label((pbx+pbw+8+mx+30)/2, ex_y+62, "融合", "#16a34a", 13, fw="bold")
# replacement box
rbx = mx+40
rbw = w-40-12 - rbx
box(rbx, ex_y+40, rbw, 64, "#ecfdf5", "#16a34a",
    ["replacement：npu_add_rms_norm_quant(...)",
     "【1 融合算子 + 0 中间张量落地】"], fs=12.8, tcol="#15803d", bold0=False)

L.append('</svg>')
open("fusion_pass_stack.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote fusion_pass_stack.svg")
