#!/usr/bin/env python3
"""fig-m13-locator: kernel 慢时先看哪层 —— 三类症状 -> 三个旋钮层 -> 源码位置。
把结构主线(一路降 PTX)翻译成读者收益：按症状定位该拧哪一层的旋钮。
数字取自 explainer figure_specs['fig-m13-locator'].numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


ROWS = [
    ("(a)", "换 shape / dtype 就变慢", "编译期特化层",
     "jit.py:L583（cache key）", "self.cache 命中短路 · jit.py:L584（第一次慢，之后快）"),
    ("(b)", "算得慢 / 带宽打不满", "优化 pass 层",
     "compiler.py:L203（TTGIR 贴布局）", "布局旋钮实证：ttgir 里 num-warps=4"),
    ("(c)", "小 kernel 频繁调用，host 开销大", "发射层",
     "jit.py:L638（每次调用都走 launcher）", "缓存只省编译，不省发射"),
]
COLS = ["", "症状", "旋钮层", "源码锚点", "证据 / 备注"]
COL_W = [70, 300, 210, 320, 420]
ROW_H = 78
LABEL_ROW_H = 34
PAD, TOP = 44, 96

col_x = [PAD]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)
w = PAD * 2 + sum(COL_W)
h = TOP + LABEL_ROW_H + ROW_H * len(ROWS) + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">kernel 慢时先看哪层：三类症状对应三个旋钮层</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("把结构主线（一路降 PTX）翻译成读者收益——先分清是编译期特化、优化 pass、还是发射开销")}</text>']

# 表头
for j, name in enumerate(COLS):
    x = col_x[j]
    cw = COL_W[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-4}" height="{LABEL_ROW_H}" '
              'fill="#3b82f6" stroke="#1e3a5f"/>')
    L.append(f'<text x="{x+(cw-4)/2}" y="{TOP+LABEL_ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="white">{esc(name)}</text>')

ROW_COLOR = ["#eff6ff", "#fef3c7", "#fee2e2"]
ROW_STROKE = ["#3b82f6", "#d97706", "#dc2626"]

for i, (tag, symptom, layer, anchor, evid) in enumerate(ROWS):
    y = TOP + LABEL_ROW_H + i * ROW_H
    fill, stroke = ROW_COLOR[i], ROW_STROKE[i]
    for j, cw in enumerate(COL_W):
        x = col_x[j]
        L.append(f'<rect x="{x}" y="{y}" width="{cw-4}" height="{ROW_H-6}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{col_x[0]+(COL_W[0]-4)/2}" y="{y+ROW_H/2}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="15" font-weight="bold" '
              f'fill="{stroke}">{esc(tag)}</text>')

    def wrap_text(x0, cw, text, y0, fs=12.5, weight="", fill_c="#0f172a", max_chars=13):
        words = text
        lines = []
        cur = ""
        for ch in words:
            cur += ch
            if len(cur) >= max_chars and ch in "，、 ":
                lines.append(cur)
                cur = ""
        if cur:
            lines.append(cur)
        n = len(lines)
        y_start = y0 - (n - 1) * 8
        out = []
        for k, ln in enumerate(lines):
            out.append(f'<text x="{x0+cw/2}" y="{y_start+k*16}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="{fs}" {weight} '
                        f'fill="{fill_c}">{esc(ln)}</text>')
        return out

    L += wrap_text(col_x[1], COL_W[1] - 4, symptom, y + ROW_H / 2, fs=13, weight='font-weight="bold"')
    L += wrap_text(col_x[2], COL_W[2] - 4, layer, y + ROW_H / 2, fs=13.5, weight='font-weight="bold"', fill_c=stroke)
    L += wrap_text(col_x[3], COL_W[3] - 4, anchor, y + ROW_H / 2, fs=12, max_chars=15)
    L += wrap_text(col_x[4], COL_W[4] - 4, evid, y + ROW_H / 2, fs=11.5, max_chars=17)

foot_y = h - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("蓝=(a)编译期特化 黄=(b)优化 pass 红=(c)发射 · 每格标了源码锚点，供你按症状定位该拧哪一层的旋钮")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m13-locator.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
