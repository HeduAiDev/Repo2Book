#!/usr/bin/env python3
"""ch30-three-backends: 三列对照表图。
列=P2P NCCL / NIXL RDMA / Offloading；行=结构/start_load_kv/save 路径/wait_for_save/get_finished/介质·瓶颈。"""
import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

cols = ["P2P NCCL\n(点对点)", "NIXL RDMA\n(高性能)", "Offloading\n(CPU·磁盘)"]
rows = [
    "结构",
    "start_load_kv",
    "save 路径",
    "wait_for_save",
    "get_finished\n(完成信号)",
    "介质·瓶颈",
]
cells = [
    # P2P                                 NIXL                                  Offloading
    ["单类自包含\nis_producer 分 P/D",   "facade：按 role 建\nworker 子对象",   "facade：按 role 建\nworker 子对象"],
    ["consumer 逐请求逐层\nrecv_tensor + inject", "发非阻塞 RDMA READ\n(先握手再 READ)", "提交本步 load +\n上步 store 的 transfer_async"],
    ["producer 逐层\nextract + send_tensor", "no-op\n(保存 = 让 D 来读)",        "wait_for_save 只入队\nstore，推迟到下一步"],
    ["wait_for_sent\n等 send_queue 排空",  "仅 host buffer 模式\n才回拷设备",     "prepare_store_kv\n只入队，不真等"],
    ["引擎双向记账\n(收发都本地可见)",     "收 = handle 转 DONE\n发 = 对端通知",  "仅 load 报 recving；\nstore 走 completed_jobs 围栏"],
    ["GPU↔GPU NCCL\n带宽 / 队列串行",      "网卡单边读对端显存\nRTT / P 端零拷贝", "GPU↔CPU·磁盘\nPCIe / 推迟避开关键路径"],
]
colors = ["#dbeafe", "#dcfce7", "#fef3c7"]
cstroke = ["#3b82f6", "#22c55e", "#f59e0b"]

label_w = 150
col_w = 290
row_h = 78
x0 = label_w + 14
y_head = 96
y0 = y_head + 64

w = x0 + col_w*3 + 20
h = y0 + row_h*len(rows) + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-size="21" font-weight="bold" fill="#0f172a">三类传输后端，各填同一套 worker 契约</text>')

# top banner spanning all columns
L.append(f'<rect x="{x0}" y="56" width="{col_w*3}" height="34" rx="6" fill="#1e293b"/>')
L.append(f'<text x="{x0+col_w*3/2}" y="78" text-anchor="middle" font-size="15" font-weight="bold" fill="#f8fafc">同一套 KVConnectorBase_V1 worker 契约</text>')

def multitext(cx, cy, text, fs, fill, bold=False):
    lines = text.split("\n")
    yy = cy - (len(lines)-1)*8
    fw = ' font-weight="bold"' if bold else ''
    for i, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{yy+i*16+4}" text-anchor="middle" font-size="{fs}"{fw} fill="{fill}">{esc(ln)}</text>')

# column headers
for c in range(3):
    cx = x0 + c*col_w + col_w/2
    L.append(f'<rect x="{x0+c*col_w+4}" y="{y_head}" width="{col_w-8}" height="56" rx="7" fill="{colors[c]}" stroke="{cstroke[c]}" stroke-width="2"/>')
    multitext(cx, y_head+28, cols[c], 16, "#0f172a", bold=True)

# rows
for r in range(len(rows)):
    ry = y0 + r*row_h
    # row label
    multitext(label_w/2+6, ry+row_h/2, rows[r], 14, "#334155", bold=True)
    for c in range(3):
        cx = x0 + c*col_w + col_w/2
        L.append(f'<rect x="{x0+c*col_w+4}" y="{ry+4}" width="{col_w-8}" height="{row_h-8}" rx="6" fill="{colors[c]}" fill-opacity="0.35" stroke="{cstroke[c]}" stroke-width="1.2"/>')
        multitext(cx, ry+row_h/2, cells[r][c], 12.5, "#0f172a")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch30-pd-disaggregation/diagrams/ch30-three-backends.svg","w").write('\n'.join(L))
print("ok")
