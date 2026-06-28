#!/usr/bin/env python3
"""fig-ch06-pyhccl-vs-pynccl: 逐行对位 pynccl(GPU) ↔ pyhccl(NPU)。
照搬控制流、只换符号；少数 cell 是真差异（尺寸/枚举值/句柄/开关）。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

# 列：步骤 | pynccl(GPU/NCCL) | pyhccl(NPU/HCCL)
# diff=True 表示该行左右是「真差异」（强调色），否则结构同构（中性）
rows = [
    ("控制流步骤",            "pynccl · GPU/NCCL",            "pyhccl · NPU/HCCL",            "head"),
    ("rank0 取 root info",    "ncclGetUniqueId()",           "hcclGetUniqueId()",           "sym"),
    ("unique_id 缓冲尺寸",    "ncclUniqueId  128 字节",       "hcclUniqueId  4108 字节",      "diff"),
    ("dist.broadcast 散 id",  "dist.broadcast(...)",          "dist.broadcast(...)",         "same"),
    ("建通信子",              "ncclCommInitRank",            "hcclCommInitRank",            "sym"),
    ("warmup all_reduce",     "all_reduce(zeros(1))",         "all_reduce(zeros(1))",        "same"),
    ("tensor → ctypes 缓冲",  "buffer_type(data_ptr())",      "buffer_type(data_ptr())",     "same"),
    ("stream 句柄",           "stream.cuda_stream",          "stream.npu_stream",          "diff"),
    ("float16 枚举值",        "ncclFloat16 = 6",             "hcclFloat16 = 3",            "diff"),
    ("单卡 / 缺库降级",       "disabled = True",              "disabled = True",             "same"),
    ("额外停用开关",          "VLLM_DISABLE_PYNCCL",         "（无）",                       "diff"),
]

# 几何
pad = 24
col_w = [220, 300, 300]
row_h = 46
head_h = 52
W = pad * 2 + sum(col_w)
H = pad * 2 + head_h + (len(rows) - 1) * row_h + 70  # +图例

x0 = pad
xs_col = [x0]
for w in col_w[:-1]:
    xs_col.append(xs_col[-1] + w)

C = {
    "same":  ("#ecfdf5", "#10b981", "#065f46"),   # 结构同构=绿
    "sym":   ("#eff6ff", "#3b82f6", "#1e3a8a"),   # 仅换符号=蓝
    "diff":  ("#fef2f2", "#ef4444", "#991b1b"),   # 真差异=红
    "head":  ("#1e293b", "#1e293b", "#ffffff"),
    "label": ("#f1f5f9", "#cbd5e1", "#334155"),
}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

y = pad
for ri, (c0, c1, c2, _kind) in enumerate(rows):
    cells = [c0, c1, c2]
    is_head = (ri == 0)
    h = head_h if is_head else row_h
    kind = rows[ri][3]
    for ci, text in enumerate(cells):
        x = xs_col[ci]
        w = col_w[ci]
        if is_head:
            fill, stroke, tcol = C["head"]
            fw = "bold"
        elif ci == 0:
            fill, stroke, tcol = C["label"]
            fw = "bold"
        else:
            fill, stroke, tcol = C[kind]
            fw = "normal"
        L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.4" rx="6"/>')
        fs = 16 if is_head else 15
        L.append(f'<text x="{x + w/2}" y="{y + h/2 + 5}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcol}">{esc(text)}</text>')
    y += h

# 图例（三项均分整宽，避免溢出）
ly = y + 22
items = [("结构同构 · 逐字相同", C["same"][1]), ("仅换符号 · NCCL→HCCL", C["sym"][1]), ("真差异 · 需查头文件", C["diff"][1])]
seg = (W - pad * 2) / 3
for i, (label, col) in enumerate(items):
    lx = pad + i * seg
    L.append(f'<rect x="{lx}" y="{ly-12}" width="18" height="18" fill="{col}" rx="3"/>')
    L.append(f'<text x="{lx+26}" y="{ly+2}" font-size="14" fill="#334155">{esc(label)}</text>')

L.append('</svg>')
open("pyhccl_vs_pynccl.svg", "w").write('\n'.join(L))
print("wrote pyhccl_vs_pynccl.svg", W, H)
