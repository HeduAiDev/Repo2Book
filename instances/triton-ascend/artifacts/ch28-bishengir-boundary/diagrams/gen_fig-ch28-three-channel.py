#!/usr/bin/env python3
"""flow 模板改造:一个源(闭源编译产物)分三条通道回收元数据(compiler.py:L480-L499)。
根节点在顶部居中,三条通道纵向排列、各自终点标明下游用途。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROOT = "bishengir-compile 编译完成\n（kernel.o + stdout + libkernel.so）"
CHANNELS = [
    {
        "title": "通道①：产物二进制",
        "steps": ["Path(bin_path).read_bytes()", "（compiler.py:L499）"],
        "result": "npubin 字节",
        "usage": "driver 发射到 NPU",
        "color": ("#eff6ff", "#1e40af"),
    },
    {
        "title": "通道②：stdout 正则",
        "steps": ["re.search(r'UB\\s+size\\s*=\\s*(\\d+)\\s*bits')", "（compiler.py:L481-L484）"],
        "result": "required_ub_bits",
        "usage": "inductor autotune",
        "color": ("#f0fdf4", "#15803d"),
    },
    {
        "title": "通道③：dlopen 回调",
        "steps": ["ctypes.CDLL(libkernel.so) + 4 个回调", "（compiler.py:L492-L497）"],
        "result": "bs_task_type / workspace_size /\nlock_num / lock_init_val",
        "usage": "sync/task 元数据",
        "color": ("#fdf4ff", "#a21caf"),
    },
]

PAD = 40
ROOT_W, ROOT_H = 460, 64
COL_W, COL_H = 380, 176
GAP = 40
TOP_ROOT = 60
TOP_CHAN = TOP_ROOT + ROOT_H + 70
w = PAD * 2 + COL_W * len(CHANNELS) + GAP * (len(CHANNELS) - 1)
h = TOP_CHAN + COL_H + 60

root_cx = w / 2
root_x = root_cx - ROOT_W / 2
col_x = [PAD + i * (COL_W + GAP) for i in range(len(CHANNELS))]
col_cx = [x + COL_W / 2 for x in col_x]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("闭源编译产物分三条通道回收——不同用途走不同路径")}</text>']

# 根节点
L.append(f'<rect x="{root_x}" y="{TOP_ROOT}" width="{ROOT_W}" height="{ROOT_H}" rx="10" '
          'fill="#1e293b" stroke="#0f172a" stroke-width="2"/>')
for k, line in enumerate(ROOT.split("\n")):
    L.append(f'<text x="{root_cx}" y="{TOP_ROOT+26+k*20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="white">{esc(line)}</text>')

root_bottom_y = TOP_ROOT + ROOT_H
for i, ch in enumerate(CHANNELS):
    cx = col_cx[i]
    x = col_x[i]
    fill, stroke = ch["color"]
    # 分支箭头:从根节点底边到该通道顶部
    L.append(f'<line x1="{root_cx}" y1="{root_bottom_y}" x2="{cx}" y2="{TOP_CHAN-6}" '
              'stroke="#475569" stroke-width="1.6" marker-end="url(#a)"/>')
    # 通道卡片
    L.append(f'<rect x="{x}" y="{TOP_CHAN}" width="{COL_W}" height="{COL_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{cx}" y="{TOP_CHAN+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="{stroke}">{esc(ch["title"])}</text>')
    sy = TOP_CHAN + 50
    for k, line in enumerate(ch["steps"]):
        L.append(f'<text x="{cx}" y="{sy+k*16}" text-anchor="middle" font-family="monospace" '
                  f'font-size="10.5" fill="#334155">{esc(line)}</text>')
    # 结果 -> 用途(结果行数可变,用途 y 随结果行数下移,避免二者重叠)
    res_y = TOP_CHAN + 96
    L.append(f'<line x1="{cx}" y1="{res_y-14}" x2="{cx}" y2="{res_y+2}" '
              'stroke="#94a3b8" stroke-width="1.3" marker-end="url(#a)"/>')
    res_lines = ch["result"].split("\n")
    for k, line in enumerate(res_lines):
        L.append(f'<text x="{cx}" y="{res_y+16+k*15}" text-anchor="middle" '
                  f'font-family="monospace" font-size="11" font-weight="bold" '
                  f'fill="{stroke}">{esc(line)}</text>')
    usage_y = res_y + 16 + (len(res_lines) - 1) * 15 + 26
    L.append(f'<text x="{cx}" y="{usage_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#475569">→ {esc(ch["usage"])}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch28-three-channel.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
