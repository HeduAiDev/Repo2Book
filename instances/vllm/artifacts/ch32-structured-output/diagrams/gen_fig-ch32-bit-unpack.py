#!/usr/bin/env python3
"""fig-ch32-bit-unpack: 一个 program 把 256 个打包 int32 广播移位成 [256,32] 的位矩阵、
reshape 成 8192 个布尔量,bit==0 的位置写 -inf。
template: tiling(流水线 + 位分解 zoom)
注意:这是 V2(opt-in)路径的 kernel,默认部署走 m11 的 xgrammar 库函数——本图标题需清楚标注。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1300, 460
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("_apply_grammar_bitmask_kernel:打孔卡抖开成位矩阵,没打孔的位置盖 -inf")}</text>')
L.append(f'<rect x="{W/2-330}" y="40" width="660" height="24" rx="12" fill="#fef3c7" stroke="#b45309"/>')
L.append(f'<text x="{W/2}" y="57" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
          f'font-weight="bold" fill="#92400e">'
          f'{esc("V2(opt-in)路径的 kernel;默认部署走 xgrammar 库函数,见下一张图")}</text>')

PAD = 50
TOP = 100
BOX_H = 60
STAGE_W = 210
GAP = 46
STAGES = [
    ["打包 int32[256]", "每 program 载入", "8192//32=256 个字"],
    ["广播右移", "(packed>>arange(32))&1", "[256,1]>>[1,32]"],
    ["位矩阵 [256,32]", "reshape", "-> 展平 8192 个布尔量"],
    ["谓词 store(-inf)", "mask=bit==0 &", "block_offset<vocab_size"],
]
sx = []
for i, lines in enumerate(STAGES):
    x = PAD + i * (STAGE_W + GAP)
    sx.append(x)
    L.append(f'<rect x="{x}" y="{TOP}" width="{STAGE_W}" height="{BOX_H}" rx="8" '
              f'fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>')
    for k, t in enumerate(lines):
        fw = "bold" if k == 0 else "normal"
        fs = 12.5 if k == 0 else 10.5
        fc = "#312e81" if k == 0 else "#4338ca"
        L.append(f'<text x="{x+STAGE_W/2}" y="{TOP+18+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" '
                  f'fill="{fc}">{esc(t)}</text>')
    if i < len(STAGES) - 1:
        L.append(f'<line x1="{x+STAGE_W}" y1="{TOP+BOX_H/2}" x2="{x+STAGE_W+GAP-6}" y2="{TOP+BOX_H/2}" '
                  f'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

# --- zoom-in:packed=160 的位分解(与阶段 2/3 对齐) ---
ZOOM_Y = TOP + BOX_H + 70
L.append(f'<text x="{PAD}" y="{ZOOM_Y-16}" font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#1e293b">{esc("放大看掩码行 1 的第 0 个 int32:packed = 160 = 0b10100000")}</text>')
BIT_W, BIT_H = 46, 40
N_SHOW = 8  # 只示意低 8 位(bit0..bit7),其余 24 位省略但不影响论点
ONE_BITS = {5, 7}
for b in range(N_SHOW):
    x = PAD + b * (BIT_W + 4)
    bit_idx = N_SHOW - 1 - b  # 从左到右 bit7 -> bit0
    is_one = bit_idx in ONE_BITS
    fill = "#bbf7d0" if is_one else "#fecaca"
    stroke = "#16a34a" if is_one else "#dc2626"
    L.append(f'<rect x="{x}" y="{ZOOM_Y}" width="{BIT_W}" height="{BIT_H}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BIT_W/2}" y="{ZOOM_Y+18}" text-anchor="middle" font-family="monospace" '
              f'font-size="14" font-weight="bold" '
              f'fill="{"#14532d" if is_one else "#7f1d1d"}">{esc(str(int(is_one)))}</text>')
    L.append(f'<text x="{x+BIT_W/2}" y="{ZOOM_Y+34}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="9.5" fill="#64748b">{esc(f"bit{bit_idx}")}</text>')
L.append(f'<text x="{PAD + N_SHOW*(BIT_W+4) + 10}" y="{ZOOM_Y+BIT_H/2+5}" font-family="sans-serif" '
          f'font-size="11.5" fill="#64748b">{esc("… 高 24 位省略(该行其余 token 全非法)")}</text>')

LEGEND_Y = ZOOM_Y + BIT_H + 26
L.append(f'<rect x="{PAD}" y="{LEGEND_Y}" width="18" height="18" rx="3" fill="#bbf7d0" stroke="#16a34a"/>')
L.append(f'<text x="{PAD+26}" y="{LEGEND_Y+14}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("bit==1:合法(不写,logit 原样)")}</text>')
L.append(f'<rect x="{PAD+280}" y="{LEGEND_Y}" width="18" height="18" rx="3" fill="#fecaca" stroke="#dc2626"/>')
L.append(f'<text x="{PAD+306}" y="{LEGEND_Y+14}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("bit==0:非法(tl.store 写 -inf)")}</text>')

# --- 结果条:kernel 跑完 logits 行 1 的存活 token ---
RES_Y = LEGEND_Y + 50
L.append(f'<rect x="{PAD}" y="{RES_Y}" width="{W-2*PAD}" height="76" rx="8" '
          f'fill="#f0fdf4" stroke="#16a34a"/>')
L.append(f'<text x="{W/2}" y="{RES_Y+26}" text-anchor="middle" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#166534">'
          f'{esc("真机 Triton kernel 实测:解出 bit5、bit7,logits 行 1 跑完后只剩 token 5、7 有限")}</text>')
L.append(f'<text x="{W/2}" y="{RES_Y+48}" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#166534">{esc("其余 94 个 token 均为 -inf(isneginf 逐元素核实为真)")}</text>')
L.append(f'<text x="{W/2}" y="{RES_Y+68}" text-anchor="middle" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("真实规模:BLOCK_SIZE=8192,|V|=152064 时 grid 第二维=19,batch=256 无投机则 grid=(256,19)=4864 个 program(按源码常量推算)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-bit-unpack.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
