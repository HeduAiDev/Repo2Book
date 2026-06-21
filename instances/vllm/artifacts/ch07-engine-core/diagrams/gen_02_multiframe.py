#!/usr/bin/env python3
"""02-byte-tag-multiframe-message: ZMQ 多帧消息逐帧布局。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

w, h = 980, 520
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def box(x, y, bw, bh, fill, stroke, rx=6):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def txt(x, y, s, size=14, anchor="middle", fill="#1e293b", weight="normal"):
    L.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')

def frame(x, y, bw, label1, label2, fill, stroke):
    box(x, y, bw, 78, fill, stroke)
    txt(x + bw / 2, y + 32, label1, 14, weight="bold")
    txt(x + bw / 2, y + 54, label2, 12, fill="#475569")

# 例一：带张量 ADD 请求（>3 帧 -> track=True）
txt(40, 60, "例一：带张量的 ADD 请求（aux_buffers 非空 → send_multipart(copy=False, track=True)）",
    16, anchor="start", weight="bold", fill="#1d4ed8")
x = 40
y = 90
widths = [(150, "frame0", "engine identity 2B LE", "#dbeafe", "#2563eb"),
          (150, "frame1", "type tag  b'\\x00' (ADD)", "#fee2e2", "#dc2626"),
          (220, "frame2  bufs[0]", "msgpack 主结构帧", "#dcfce7", "#16a34a"),
          (180, "frame3", "aux_buffers[1] 张量 backing", "#fef9c3", "#ca8a04"),
          (140, "frame4 …", "更多张量 buffer", "#fef9c3", "#ca8a04")]
for bw, l1, l2, fill, stroke in widths:
    frame(x, y, bw, l1, l2, fill, stroke)
    x += bw + 10
txt(490, 200, "↑ frame2 主帧里只存 backing buffer 的索引（int），张量本体在 frame3.. 零拷贝直发",
    12, fill="#92400e", weight="bold")

# 例二：无张量请求（=3 帧, copy=False 快路径）
txt(40, 290, "例二：无张量请求 / ABORT（len(msg) ≤ 3 → 直接 send_multipart(copy=False)，无需 track）",
    16, anchor="start", weight="bold", fill="#15803d")
x = 40
y = 320
widths2 = [(150, "frame0", "engine identity 2B LE", "#dbeafe", "#2563eb"),
           (150, "frame1", "type tag  b'\\x01' (ABORT)", "#fee2e2", "#dc2626"),
           (260, "frame2  bufs[0]", "msgpack 主帧（全部内联）", "#dcfce7", "#16a34a")]
for bw, l1, l2, fill, stroke in widths2:
    frame(x, y, bw, l1, l2, fill, stroke)
    x += bw + 10
txt(40, 440, "engine 侧 process_input_sockets：type_frame, *data_frames = recv_multipart(copy=False)",
    13, anchor="start", fill="#475569")
txt(40, 466, "→ EngineCoreRequestType(bytes(type_frame.buffer)) 一步还原类型 → 按类型选 decoder 解 data_frames",
    13, anchor="start", fill="#475569")

L.append('</svg>')
open("02-byte-tag-multiframe-message.svg", "w").write('\n'.join(L))
print("wrote 02-byte-tag-multiframe-message.svg")
