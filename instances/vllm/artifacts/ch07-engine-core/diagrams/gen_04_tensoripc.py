#!/usr/bin/env python3
"""04-tensor-ipc-zerocopy: 普通 ZMQ aux 帧路径 vs OOB 共享内存旁路对照。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

w, h = 980, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#ca8a04"/></marker>')
L.append('<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#9333ea"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def box(x, y, bw, bh, fill, stroke, rx=8):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def txt(x, y, s, size=13, anchor="middle", fill="#1e293b", weight="normal"):
    L.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')

# 上半：普通 aux_buffers 帧路径（一次进程间拷贝）
box(30, 40, 920, 200, "#fffbeb", "#ca8a04")
txt(490, 64, "路径 A：aux_buffers 多帧（默认）— 一次进程间拷贝", 16, weight="bold", fill="#92400e")
box(70, 100, 200, 80, "#fef9c3", "#ca8a04")
txt(170, 130, "_encode_tensor", 14, weight="bold")
txt(170, 152, "append 到 aux_buffers", 12, fill="#475569")
box(390, 100, 220, 80, "#fde68a", "#ca8a04")
txt(500, 130, "ZMQ 数据帧", 14, weight="bold")
txt(500, 152, "send_multipart(copy=False)", 11, fill="#475569")
box(720, 100, 200, 80, "#fef9c3", "#ca8a04")
txt(820, 130, "_decode_tensor", 14, weight="bold")
txt(820, 152, "aux_buffers[idx] 还原", 12, fill="#475569")
L.append('<line x1="270" y1="140" x2="390" y2="140" stroke="#ca8a04" stroke-width="2.5" marker-end="url(#ag)"/>')
L.append('<line x1="610" y1="140" x2="720" y2="140" stroke="#ca8a04" stroke-width="2.5" marker-end="url(#ag)"/>')
txt(490, 215, "张量字节流随 ZMQ 帧跨进程传输（内核拷贝一次），解码端 clone 进 PyTorch 内存",
    12, fill="#92400e")

# 下半：OOB 共享内存旁路（零进程间拷贝）
box(30, 290, 920, 230, "#faf5ff", "#9333ea")
txt(490, 314, "路径 B：OOB 共享内存旁路（mm_tensor_ipc=='torch_shm'）— 零进程间拷贝", 16, weight="bold", fill="#7e22ce")
box(70, 350, 210, 90, "#f3e8ff", "#9333ea")
txt(175, 378, "_encode_tensor 见大张量", 13, weight="bold")
txt(175, 400, "TensorIpcSender.__call__", 12, fill="#475569")
txt(175, 420, "share_memory_() + put", 11, fill="#475569")
box(390, 350, 220, 90, "#ede9fe", "#7c3aed")
txt(500, 374, "msgpack 主帧", 13, weight="bold")
txt(500, 396, "{sender_id, message_id,", 11, fill="#475569")
txt(500, 414, "tensor_id} 占位 dict", 11, fill="#475569")
box(720, 350, 210, 90, "#f3e8ff", "#9333ea")
txt(825, 378, "_decode_tensor 见 dict", 13, weight="bold")
txt(825, 400, "TensorIpcReceiver", 12, fill="#475569")
txt(825, 420, "drain-and-buffer 取回", 11, fill="#475569")
L.append('<line x1="280" y1="395" x2="390" y2="395" stroke="#9333ea" stroke-width="2.5" marker-end="url(#ap)"/>')
L.append('<line x1="610" y1="395" x2="720" y2="395" stroke="#9333ea" stroke-width="2.5" marker-end="url(#ap)"/>')
# shared memory channel – U-shaped dashed path connecting left box → shm box → right box
box(280, 470, 440, 36, "#ddd6fe", "#7c3aed")
txt(500, 493, "torch.mp.Queue 共享内存（张量本体走此通道，不进 ZMQ 帧）", 12, weight="bold", fill="#5b21b6")
# U-shaped path: left box bottom (175,440) → down to y=510 → right to (825,510) → up to right box bottom (825,440)
L.append('<path d="M175,440 L175,510 L825,510 L825,440" stroke="#9333ea" stroke-width="2" stroke-dasharray="3 3" fill="none" marker-end="url(#ap)"/>')

L.append('</svg>')
open("04-tensor-ipc-zerocopy.svg", "w").write('\n'.join(L))
print("wrote 04-tensor-ipc-zerocopy.svg")
