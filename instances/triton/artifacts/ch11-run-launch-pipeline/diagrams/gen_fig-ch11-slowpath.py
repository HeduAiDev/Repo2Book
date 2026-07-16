#!/usr/bin/env python3
"""flow 模板(定制):内存 cache 未命中慢路径。横向流水 5 步 -> compile(灰框内 5 段
lowering,标第十四章不展开) -> CompiledKernel -> 回填 cache。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

STEPS = [
    ("parse_options", "CUDAOptions\nnum_warps=4\nnum_stages=3"),
    ("None->*i8\n签名修正", "本例无 None\n全直通"),
    ("get_attrs_descriptor\n+get_constants", "configs[0]\nconstant_params"),
    ("组 constants", "含 BLOCK_SIZE=256"),
    ("ASTSource", "src"),
]
IR_STAGES = ["ttir", "ttgir", "llir", "ptx", "cubin"]

BOX_W, BOX_H, GAP, PAD, TOP = 150, 92, 26, 40, 110
n = len(STEPS)
COMPILE_W = 430
CK_W = 190
w = PAD * 2 + n * (BOX_W + GAP) + COMPILE_W + GAP * 2 + CK_W
h = 335

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="38" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("未命中慢路径：run() 前台派单，备齐 5 样输入交给 compile")}</text>')
L.append(f'<text x="{PAD}" y="60" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc("热进程实测：本段 ≈ 98.379 ms（相对内存命中 4.398 μs 贵约 4 个数量级）")}</text>')

cy = TOP + BOX_H / 2
x = PAD
centers = []
for i, (title, detail) in enumerate(STEPS):
    centers.append(x)
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
    for k, line in enumerate(title.split("\n")):
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+20+k*15}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" font-weight="bold" fill="#7c2d12">{esc(line)}</text>')
    dy = TOP + 20 + len(title.split("\n")) * 15 + 6
    for k, line in enumerate(detail.split("\n")):
        L.append(f'<text x="{x+BOX_W/2}" y="{dy+k*14}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="#334155">{esc(line)}</text>')
    if i < n - 1:
        L.append(f'<line x1="{x+BOX_W}" y1="{cy}" x2="{x+BOX_W+GAP-4}" y2="{cy}" '
                  'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    x += BOX_W + GAP

# 箭头进入 compile 灰框
compile_x = x + GAP / 2
L.append(f'<line x1="{x}" y1="{cy}" x2="{compile_x-4}" y2="{cy}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

L.append(f'<rect x="{compile_x}" y="{TOP-14}" width="{COMPILE_W}" height="{BOX_H+28}" rx="10" '
          'fill="#e2e8f0" stroke="#64748b" stroke-width="1.6"/>')
L.append(f'<text x="{compile_x+COMPILE_W/2}" y="{TOP-22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#475569">'
          f'{esc("self.compile —— 第十四章：五段驱动主循环，本章不展开")}</text>')
ir_n = len(IR_STAGES)
ir_pad = 16
ir_w = (COMPILE_W - ir_pad * 2 - (ir_n - 1) * 8) / ir_n
for i, stage in enumerate(IR_STAGES):
    ix = compile_x + ir_pad + i * (ir_w + 8)
    L.append(f'<rect x="{ix}" y="{TOP+14}" width="{ir_w}" height="{BOX_H-28}" rx="6" '
              'fill="#cbd5e1" stroke="#94a3b8"/>')
    L.append(f'<text x="{ix+ir_w/2}" y="{TOP+14+(BOX_H-28)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#334155">{esc(stage)}</text>')
    if i < ir_n - 1:
        L.append(f'<line x1="{ix+ir_w}" y1="{TOP+14+(BOX_H-28)/2}" x2="{ix+ir_w+7}" y2="{TOP+14+(BOX_H-28)/2}" '
                  'stroke="#94a3b8" stroke-width="1.2" marker-end="url(#a)"/>')

# CompiledKernel 出结果框
ck_x = compile_x + COMPILE_W + GAP
L.append(f'<line x1="{compile_x+COMPILE_W}" y1="{cy}" x2="{ck_x-4}" y2="{cy}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{ck_x}" y="{TOP}" width="{CK_W}" height="{BOX_H}" rx="8" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.6"/>')
L.append(f'<text x="{ck_x+CK_W/2}" y="{TOP+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#1e3a8a">{esc("CompiledKernel")}</text>')
for k, line in enumerate(["asm 5 段", "ttir 3882 字符", "shared=0 字节"]):
    L.append(f'<text x="{ck_x+CK_W/2}" y="{TOP+42+k*16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(line)}</text>')

# 回填 cache: 从 CompiledKernel 底部回箭到最左侧顶部,标注"回填 cache[key]"
loop_y = TOP + BOX_H + 46
L.append(f'<path d="M {ck_x+CK_W/2},{TOP+BOX_H} L {ck_x+CK_W/2},{loop_y} '
          f'L {centers[0]+BOX_W/2},{loop_y} L {centers[0]+BOX_W/2},{TOP+BOX_H+4}" '
          'fill="none" stroke="#b45309" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#b)"/>')
L.append(f'<text x="{(ck_x+CK_W/2+centers[0]+BOX_W/2)/2}" y="{loop_y-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#b45309">'
          f'{esc("回填 cache[device][key]——此键从此走快路径")}</text>')

foot_y = h - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
          f'{esc("run 只是前台派单员：备齐 options/签名/特化描述子/constants/ASTSource 后一次调用 compile；灰框内部留给第十四章。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch11-slowpath.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
