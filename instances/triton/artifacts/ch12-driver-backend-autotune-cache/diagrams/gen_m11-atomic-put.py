#!/usr/bin/env python3
"""figure m11-atomic-put: FileCacheManager.put 写临时目录 + os.replace 原子改名落盘。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

TITLE = "FileCacheManager.put：原子落盘，读者永远看不到半截文件"
SUBTITLE = "python/triton/runtime/cache.py:L112-L136 —— 磁盘缓存与内存 launch 缓存正交（回指上一章）"

STEPS = [
    ("写临时目录", "cache_dir/tmp.pid_<pid>_<uuid>/<filename>", "#dbeafe", "#1d4ed8", "#1e3a5f"),
    ("原子改名", "os.replace(temp_path, filepath)", "#fef9c3", "#ca8a04", "#854d0e"),
    ("正本落地", "cache_dir/<filename>", "#dcfce7", "#15803d", "#14532d"),
    ("清理临时目录", "os.removedirs(temp_dir)", "#f1f5f9", "#64748b", "#334155"),
]

PAD = 40
BOX_H = 84
GAP = 60
TOP = 100

box_widths = [max(cjk_w(t, 14), cjk_w(s, 13)) + 44 for t, s, *_ in STEPS]
w_content = sum(box_widths) + GAP * (len(STEPS) - 1)

note_lines = [
    "cache_dir = <TRITON_CACHE_DIR 或 ~/.triton/cache>/<key>；os.replace 是 POSIX 原子 rename——",
    "并发读者要么看到旧正本、要么看到完整新正本，绝不会读到写了一半的残缺文件。",
    "这层跨进程免重编的磁盘缓存，与上一章内存 launch 缓存的 cache[device][key] 三桶正交，互不替代。",
]
note_w_needed = max(cjk_w(s, 12.5) for s in note_lines) + 32
subtitle_w = cjk_w(SUBTITLE, 12) + 20
w = PAD * 2 + max(w_content, note_w_needed, subtitle_w)

# 若内容比 note 窄，居中排布 steps
x_start = PAD + (w - 2 * PAD - w_content) / 2

h = TOP + BOX_H + 40 + 24 * len(note_lines) + 24 + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

xs_ = []
cx = x_start
for bw in box_widths:
    xs_.append(cx)
    cx += bw + GAP

y = TOP
for i, ((title, sub, fill, stroke, tf), bx, bw) in enumerate(zip(STEPS, xs_, box_widths)):
    L.append(f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="{BOX_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{bx+bw/2:.0f}" y="{y+32:.0f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="{tf}">{esc(title)}</text>')
    L.append(f'<text x="{bx+bw/2:.0f}" y="{y+58:.0f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" '
              f'fill="{tf}">{esc(sub)}</text>')
    if i > 0:
        px = xs_[i - 1] + box_widths[i - 1]
        ay = y + BOX_H / 2
        L.append(f'<line x1="{px:.0f}" y1="{ay:.0f}" x2="{bx:.0f}" y2="{ay:.0f}" '
                  'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

note_top = y + BOX_H + 40
note_h = 24 * len(note_lines) + 24
L.append(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
          'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    L.append(f'<text x="{PAD+16}" y="{note_top+26+i*24:.0f}" font-family="sans-serif" '
              f'font-size="12.5" fill="#1e3a5f">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("m11-atomic-put.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
