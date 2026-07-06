#!/usr/bin/env python3
"""02-common-to-backend-metadata-translation: Common → FA metadata 翻译。"""
import xml.sax.saxutils as xs

OUT = "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch24-attention/diagrams/02-common-to-backend-metadata-translation.svg"


def esc(s):
    return xs.escape(str(s))


W, H = 1000, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
    'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def field(x, y, w, t, fill, stroke, fs=12.5, weight="normal", tcol="#0f172a"):
    h = 30
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{x+10}" y="{y+20}" font-family="sans-serif" font-size="{fs}" '
             f'font-weight="{weight}" text-anchor="start" fill="{tcol}">{esc(t)}</text>')


def label(x, y, t, fs=12, col="#475569", weight="normal", anchor="middle"):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{fs}" '
             f'font-weight="{weight}" text-anchor="{anchor}" fill="{col}">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, col="#475569"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
             f'stroke-width="1.8" marker-end="url(#a)"/>')


label(W/2, 30, "两层 metadata：CommonAttentionMetadata → FlashAttentionMetadata", 18, "#0f172a", "bold")

# 上框 Common
CX, CY, CW = 60, 60, 380
L.append(f'<rect x="{CX-12}" y="{CY-8}" width="{CW+24}" height="280" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>')
label(CX + CW/2, CY + 12, "CommonAttentionMetadata（核心 8 字段）", 13, "#1d4ed8", "bold")
label(CX + CW/2, CY + 30, "model_runner 那头每步组装一次", 11, "#64748b")

common_fields = [
    ("query_start_loc (+_cpu)", "shared"),
    ("seq_lens", "shared"),
    ("num_reqs / num_actual_tokens", "shared"),
    ("max_query_len / max_seq_len", "shared"),
    ("block_table_tensor", "key"),
    ("slot_mapping", "key"),
    ("causal", "shared"),
]
fy = CY + 44
for name, kind in common_fields:
    if kind == "key":
        field(CX, fy, CW, name, "#fde68a", "#f59e0b", weight="bold", tcol="#92400e")
    else:
        field(CX, fy, CW, name, "#dbeafe", "#3b82f6")
    fy += 34
label(CX + CW + 90, CY + 230, "block_table_tensor / slot_mapping", 11.5, "#92400e", "bold")
label(CX + CW + 90, CY + 248, "= 两级页表，所有后端/所有层共用", 11, "#64748b")

# 翻译箭头
arrow(CX + CW/2, CY + 272, CX + CW/2, CY + 330, col="#0f172a")
label(CX + CW/2 + 130, CY + 305, "builder.build(common_prefix_len, common)", 12.5, "#0f172a", "bold")
label(CX + CW/2 + 130, CY + 322, "唯一中心翻译入口", 11, "#64748b")

# 下框 FA metadata
FX, FY, FW = 60, CY + 340, 880
L.append(f'<rect x="{FX-12}" y="{FY-8}" width="{FW+24}" height="230" rx="10" fill="#f0fdf4" stroke="#bbf7d0"/>')
label(FX + FW/2, FY + 12, "FlashAttentionMetadata（FlashAttention 专属）", 13, "#15803d", "bold")
label(FX + FW/2, FY + 30, "backend 这头 build 出来、按 layer_name 装进 forward_context", 11, "#64748b")

# 左列：共享字段直接搬
ly = FY + 48
label(FX + 190, FY + 46, "共享字段：直接搬（含改名）", 12, "#1d4ed8", "bold")
shared = [
    "block_table  ←  block_table_tensor",
    "slot_mapping  ←  slot_mapping",
    "seq_lens / query_start_loc",
    "max_query_len / max_seq_len / causal",
]
sy = FY + 60
for t in shared:
    fill = "#fde68a" if "←" in t else "#dbeafe"
    stroke = "#f59e0b" if "←" in t else "#3b82f6"
    weight = "bold" if "←" in t else "normal"
    tcol = "#92400e" if "←" in t else "#0f172a"
    field(FX + 8, sy, 370, t, fill, stroke, fs=12, weight=weight, tcol=tcol)
    sy += 34

# 右列：FA 特有新增
label(FX + 640, FY + 46, "FA 特有：build 过程新增", 12, "#15803d", "bold")
extra = [
    "use_cascade / common_prefix_len",
    "scheduler_metadata（AOT 调度）",
    "cu_prefix_query_lens",
    "prefix_kv_lens / suffix_kv_lens",
]
ey = FY + 60
for t in extra:
    field(FX + 470, ey, 380, t, "#dcfce7", "#22c55e", fs=12)
    ey += 34

label(W/2, H - 12, "翻译模式：共享量只算一次（省重复）+ 各后端按需补 kernel 要的特有字段", 12.5, "#475569")

L.append('</svg>')
open(OUT, "w").write('\n'.join(L))
print("ok", OUT)
