#!/usr/bin/env python3
"""state-table 模板(定制为二列规格表):flash_attn_varlen_func 调用面——
字段/形参 -> 语义与取值,底部标主调用点。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "flash_attn_varlen_func 的调用面"
SUBTITLE = "vLLM 把变长打平的 prefill+decode 批次,连同分页 KV 缓存的块表,一次喂给这个当作黑盒 import 的函数"

FIELD_COL_W, VALUE_COL_W, ROW_H, HEADER_H, TOP, PAD = 210, 560, 52, 34, 96, 30
ROWS = [
    ("q / k / v", "形状 (total_tokens, nheads, headdim) —— 不等长序列首尾相接打平"),
    ("cu_seqlens_q / cu_seqlens_k", "长度 = batch+1 的前缀和数组,切出每条序列的边界"),
    ("softmax_scale", "默认 1/sqrt(headdim)"),
    ("seqused_k / block_table", "指向分页 KV 缓存:每条序列实际长度 + 物理块号"),
    ("softmax_lse", "return_softmax_lse=True 时返回,形状 (nheads, total_q) —— 每行 QKᵀ·scale 的 logsumexp"),
]

w = PAD * 2 + FIELD_COL_W + VALUE_COL_W
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 66
field_x = PAD
value_x = PAD + FIELD_COL_W
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 表头
L.append(f'<rect x="{field_x}" y="{TOP}" width="{FIELD_COL_W-6}" height="{HEADER_H-6}" rx="3" '
          'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
L.append(f'<text x="{field_x+(FIELD_COL_W-6)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12.5" fill="white" font-weight="bold">字段/形参</text>')
L.append(f'<rect x="{value_x}" y="{TOP}" width="{VALUE_COL_W}" height="{HEADER_H-6}" rx="3" '
          'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
L.append(f'<text x="{value_x+VALUE_COL_W/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12.5" fill="white" font-weight="bold">语义 / 取值</text>')

for i, (field, value) in enumerate(ROWS):
    ry = row_y[i]
    is_lse = "lse" in field
    row_fill = "#ecfdf5" if is_lse else ("#f8fafc" if i % 2 == 0 else "white")
    row_stroke = "#047857" if is_lse else "#cbd5e1"
    L.append(f'<rect x="{field_x}" y="{ry}" width="{FIELD_COL_W-6}" height="{ROW_H-6}" rx="4" '
              f'fill="{row_fill}" stroke="{row_stroke}" stroke-width="{2 if is_lse else 1}"/>')
    L.append(f'<text x="{field_x+12}" y="{ry+(ROW_H-6)/2+5}" font-family="monospace" '
              f'font-size="12" font-weight="bold" '
              f'fill="{"#047857" if is_lse else "#0f172a"}">{esc(field)}</text>')
    L.append(f'<rect x="{value_x}" y="{ry}" width="{VALUE_COL_W}" height="{ROW_H-6}" rx="4" '
              f'fill="{row_fill}" stroke="{row_stroke}" stroke-width="{2 if is_lse else 1}"/>')
    L.append(f'<text x="{value_x+12}" y="{ry+(ROW_H-6)/2+5}" font-family="sans-serif" '
              f'font-size="12" fill="#334155">{esc(value)}</text>')

box_y = row_y[-1] + ROW_H + 20
L.append(f'<rect x="{PAD}" y="{box_y}" width="{w-2*PAD}" height="34" rx="6" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
CALL_SITE = "主路径调用:vllm/v1/attention/backends/flash_attn.py:L809-L832 — 一次调用吃整批 prefill+decode"
L.append(f'<text x="{w/2}" y="{box_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#1e3a8a">{esc(CALL_SITE)}</text>')

foot_y = h - 16
FOOT = "回指注意力章(ch24);causal / window_size / softcap 控制掩码与缩放(表未逐一列出)。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig34-7-varlen-call-surface.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
