#!/usr/bin/env python3
"""fig-m04-ladder: 五级降级阶梯 —— ttir->ttgir->llir->ptx->cubin，每级一个
make_* 入口、跨越一道边界、对应一个后续 Part。垂直台阶版式，逐级右移体现
"降级"方向。数字取自 explainer figure_specs['fig-m04-ladder'].numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


# (阶段名, 产物/规模, 额外证据(可为 None), 源码入口(可为 None), Part 记号)
STEPS = [
    ("追踪期 TTIR", "阶梯起点 · 56 行", None, None, "Part IV · ch16"),
    (".ttir", "make_ttir 之后 · 38 行", None, "compiler.py:L187", "Part IV"),
    (".ttgir", "首次贴布局 · 39 行", "num-warps=4", "compiler.py:L203", "Part VII · ch32"),
    (".llir", "跨到 LLVM 世界 · 150 行", None, "compiler.py:L256", "Part VII"),
    (".ptx", "虚拟汇编文本 · 377 行", ".target sm_90a", "compiler.py:L317", "Part VII · ch35"),
    (".cubin", "唯一二进制终点 · 9488 字节", None, "compiler.py:L339", "Part VIII · ch37"),
]

PAD, BOX_W, BOX_H, ROW_GAP, DX = 40, 480, 68, 22, 70
TITLE_Y, SUB_Y, TOP = 28, 50, 90

n = len(STEPS)
w = PAD * 2 + (n - 1) * DX + BOX_W + 260
h = TOP + n * (BOX_H + ROW_GAP) + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">五级降级阶梯：ttir → ttgir → llir → ptx → cubin</text>',
     f'<text x="{PAD}" y="{SUB_Y}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("每级一个 make_* 入口、跨一道语言边界；越往下越啰嗦（56→38→39→150→377 行 → 9488 字节机器码）")}</text>']

xs_ = [PAD + i * DX for i in range(n)]
ys_ = [TOP + i * (BOX_H + ROW_GAP) for i in range(n)]

# 台阶连接线（前一级右下角 -> 下一级左上角附近）
for i in range(n - 1):
    x1, y1 = xs_[i] + 60, ys_[i] + BOX_H
    x2, y2 = xs_[i + 1] + 60, ys_[i + 1]
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#1d4ed8" '
              'stroke-width="1.5" marker-end="url(#a)"/>')

for i, (name, sub, extra, entry, part) in enumerate(STEPS):
    x, y = xs_[i], ys_[i]
    is_last = (i == n - 1)
    fill, stroke = ("#dbeafe", "#3b82f6") if not is_last else ("#ffedd5", "#f97316")
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+18}" y="{y+24}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{x+18}" y="{y+43}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(sub)}</text>')
    if extra:
        L.append(f'<text x="{x+18}" y="{y+59}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#b45309" font-weight="bold">{esc(extra)}</text>')
    if entry:
        L.append(f'<text x="{x+BOX_W-16}" y="{y+24}" text-anchor="end" font-family="sans-serif" '
                  f'font-size="11.5" fill="#475569">{esc(entry)}</text>')
    # Part 记号：放在阶梯右侧固定列，随行下移
    L.append(f'<text x="{x+BOX_W+30}" y="{y+BOX_H/2+4}" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#6d28d9">→ {esc(part)}</text>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("蓝=中间级 橙=终点(唯一二进制) · 台阶逐级右移=降级方向 · 右列=对应后续 Part/章")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m04-ladder.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
