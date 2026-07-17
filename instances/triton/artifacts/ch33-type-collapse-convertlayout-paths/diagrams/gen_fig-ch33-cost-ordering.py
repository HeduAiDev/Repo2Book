#!/usr/bin/env python3
"""fig-ch33-cost-ordering: state-table——三条搬运路径按跨线程流量严格递增排序。
判据 cvtReordersRegisters / cvtNeedsWarpShuffle / cvtNeedsSharedMemory 互斥完备
(lib/Analysis/Utility.cpp:L672-L705)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "convert_layout 三条搬运路径——代价按跨线程流量严格递增"
SUBTITLE = "构造示例:T=8 元素/线程,32 lane/warp,f32(4 字节);判据互斥完备(Utility.cpp:L672-L705)"
COLS = ["跨线程搬运", "barrier", "shmem 往返", "相对代价"]
ROW_LABELS = ["寄存器重排\n(register)", "warp shuffle\n(lane)", "共享内存往返\n(warp/block)"]
CELLS = {
    "寄存器重排\n(register)": ["0", "0", "0 字节", "最低(基准)"],
    "warp shuffle\n(lane)": ["约 log2(32)=5 次 shfl", "0", "0 字节",
                              "中(v3.2.0 专用实现\nTODO,暂落 shmem)"],
    "共享内存往返\n(warp/block)": ["4 store + 4 load 指令\n(共 8 元素)", "3",
                                    "32 字节/线程\n(两迭代合计)", "最高"],
}
STATUS = {
    "寄存器重排\n(register)": ["cheap", "cheap", "cheap", "cheap"],
    "warp shuffle\n(lane)": ["mid", "cheap", "cheap", "mid"],
    "共享内存往返\n(warp/block)": ["expensive", "expensive", "expensive", "expensive"],
}
COLOR = {
    "cheap": ("#ecfdf5", "#047857"),
    "mid": ("#fef9c3", "#a16207"),
    "expensive": ("#fee2e2", "#b91c1c"),
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 170, 230, 66, 40, 100, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 46
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    label_lines = row.split("\n")
    ly0 = ry + ROW_H/2 - (len(label_lines)-1)*8
    for k, line in enumerate(label_lines):
        L.append(f'<text x="{PAD+LABEL_W-16}" y="{ly0+k*16+4}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="#374151">{esc(line)}</text>')
    statuses = STATUS[row]
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j]
        fill, stroke = COLOR[status]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        n = len(lines)
        y0 = ry + ROW_H/2 - (n-1)*8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="{stroke}" '
                      f'font-weight="bold">{esc(line)}</text>')

foot_y = TOP + HEADER_H + ROW_H*len(ROW_LABELS) + 28
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          'fill="#64748b">绿=最低代价档,黄=中间档(当前版本降级落最高档),'
          '红=最高代价档 — dump 里认出 st.shared/ld.shared+bar.sync 即命中红档</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          'fill="#64748b">回指第 24 章:convert_layout=唯一跨线程搬运;'
          '第 28 章 RemoveLayoutConversions 尝试消掉红档</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch33-cost-ordering.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
