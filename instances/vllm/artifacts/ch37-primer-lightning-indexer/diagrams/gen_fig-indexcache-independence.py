#!/usr/bin/env python3
"""fig-indexcache-independence: layout 模板。
重绘自 arXiv:2606.19348 Figure 3(DeepSeek-V4 CSA 核心架构图):KV 先压缩到 1/m,
压缩索引键 K^IComp 与压缩 KV 条目 C^Comp 由同一套压缩操作并行产出,indexer 在压缩后的键上
打分选 top-k。布局对齐原图的"一次压缩、两路分叉、各自独立缓存"结构,配色区分主 KV(蓝) /
IndexCache(绿),文字换成 vLLM 里 DeepseekV32/V4IndexerCache 的真实字段名与真实字节数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text(x, y, s, size=13, anchor="middle", weight="normal", fill="#0f172a"):
    fw = f' font-weight="{weight}"' if weight != "normal" else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" '
            f'font-size="{size}"{fw} fill="{fill}">{esc(s)}</text>')

def lines(x, y, items, size=12, anchor="middle", fill="#334155", lh=16):
    return [text(x, y + i * lh, s, size=size, anchor=anchor, fill=fill) for i, s in enumerate(items)]

def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def varrow(x, y1, y2, color="#64748b", sw=1.8, marker="a"):
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})"/>')

W = 1180
PAD = 40
TOP = 92

BLUE_FILL, BLUE_STROKE = "#dbeafe", "#3b82f6"
GREEN_FILL, GREEN_STROKE = "#dcfce7", "#16a34a"
GRAY_FILL, GRAY_STROKE = "#e2e8f0", "#64748b"
AMBER_FILL, AMBER_STROKE = "#fef3c7", "#d97706"

# ---- Tier 布局 ----
y_src = TOP                          # 原始条目流
h_src = 60
y_comp = y_src + h_src + 46          # 压缩操作(并行产出)
h_comp = 70
y_cache = y_comp + h_comp + 60       # 两个独立缓存池
h_cache = 190
y_note = y_cache + h_cache + 40      # V4 页对齐 note
h_note = 56
H = y_note + h_note + 30

CACHE_W = (W - 2 * PAD - 60) / 2
CACHE_X = [PAD, PAD + CACHE_W + 60]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
          '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#3b82f6"/></marker>'
          '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(text(W / 2, 32, "IndexCache 与主 KV cache 各自独立分配、独立布局", size=16.5, weight="bold"))
L.append(text(W / 2, 54,
              "重绘自 arXiv:2606.19348 Figure 3:压缩操作并行产出 C^Comp 与 K^IComp,两者写进互不引用的独立缓存",
              size=11.5, fill="#475569"))

# ---- Tier 0: 原始条目流 ----
src_w = W - 2 * PAD
L.append(box(PAD, y_src, src_w, h_src, GRAY_FILL, GRAY_STROKE))
L.append(text(PAD + src_w / 2, y_src + 26, "原始 latent KV 条目流(每 m 个压缩为 1 个)", size=13, weight="bold"))
L.append(text(PAD + src_w / 2, y_src + 46, "同一段历史,同一次压缩操作读取", size=10.8, fill="#64748b"))

# ---- Tier 1: 压缩操作(并行产出) ----
comp_w = 460
comp_x = (W - comp_w) / 2
L.append(box(comp_x, y_comp, comp_w, h_comp, AMBER_FILL, AMBER_STROKE))
L.append(text(comp_x + comp_w / 2, y_comp + 26, "DeepseekCompressor:压缩操作(并行产出)", size=13, weight="bold", fill="#92400e"))
L.append(text(comp_x + comp_w / 2, y_comp + 48, "C^Comp(主压缩 KV) 与 K^IComp(索引器压缩键) 同一步算出", size=11, fill="#334155"))
L.append(varrow(PAD + src_w / 2, y_src + h_src, y_comp))

# 压缩 -> 两个缓存池的分叉箭头
mid_comp_x = comp_x + comp_w / 2
mid_comp_y = y_comp + h_comp
target0 = CACHE_X[0] + CACHE_W / 2
target1 = CACHE_X[1] + CACHE_W / 2
L.append(f'<path d="M{mid_comp_x},{mid_comp_y} L{target0},{y_cache}" stroke="#3b82f6" '
          f'stroke-width="2.2" fill="none" marker-end="url(#ab)"/>')
L.append(f'<path d="M{mid_comp_x},{mid_comp_y} L{target1},{y_cache}" stroke="#16a34a" '
          f'stroke-width="2.2" fill="none" marker-end="url(#ag)"/>')
L.append(text(target0 - 34, (mid_comp_y + y_cache) / 2 - 6, "C^Comp", size=11, fill="#1e3a8a", anchor="end"))
L.append(text(target1 + 34, (mid_comp_y + y_cache) / 2 - 6, "K^IComp", size=11, fill="#166534", anchor="start"))

# ---- Tier 2: 两个独立缓存池 ----
# 主 KV cache(左,蓝)
mx = CACHE_X[0]
L.append(box(mx, y_cache, CACHE_W, h_cache, BLUE_FILL, BLUE_STROKE))
L.append(text(mx + CACHE_W / 2, y_cache + 26, "主 KV cache(DeepseekV32/V4 主注意力)", size=13, weight="bold", fill="#1e3a8a"))
main_cells = ["C^Comp[0]", "C^Comp[1]", "C^Comp[2]", "…"]
cell_w = (CACHE_W - 60) / len(main_cells)
for i, c in enumerate(main_cells):
    cx = mx + 30 + i * cell_w
    L.append(box(cx, y_cache + 46, cell_w - 8, 52, "#eff6ff", BLUE_STROKE, rx=6, sw=1.2))
    L.append(text(cx + (cell_w - 8) / 2, y_cache + 76, c, size=10.5, fill="#1e3a8a"))
L.append(text(mx + CACHE_W / 2, y_cache + 122, "布局由主注意力头维/头数决定,与 indexer 无关", size=10.8, fill="#334155"))
L.append(text(mx + CACHE_W / 2, y_cache + 146, "MLAAttentionSpec(主注意力 head_size)", size=10.5, fill="#64748b"))
L.append(text(mx + CACHE_W / 2, y_cache + 168, "分开分配,不被 IndexCache 引用/复用", size=10.5, fill="#64748b"))

# IndexCache(右,绿)
ix = CACHE_X[1]
L.append(box(ix, y_cache, CACHE_W, h_cache, GREEN_FILL, GREEN_STROKE))
L.append(text(ix + CACHE_W / 2, y_cache + 26, "IndexCache(DeepseekV32/V4IndexerCache)", size=13, weight="bold", fill="#166534"))
# 单条 entry 的字节拆分:128B fp8 值 + 4B fp32 scale = 132B
entry_x = ix + 30
entry_w = CACHE_W - 60
val_w = entry_w * 128 / 132
scale_w = entry_w * 4 / 132
L.append(box(entry_x, y_cache + 46, val_w, 40, "#ecfdf5", GREEN_STROKE, rx=4, sw=1.2))
L.append(text(entry_x + val_w / 2, y_cache + 66, "128B fp8 值", size=10.5, fill="#166534"))
L.append(box(entry_x + val_w, y_cache + 46, scale_w, 40, "#fef3c7", "#d97706", rx=4, sw=1.2))
L.append(text(entry_x + val_w + scale_w / 2, y_cache + 66, "4B", size=9, fill="#92400e"))
L.append(text(ix + CACHE_W / 2, y_cache + 104, "每条 K^IComp = 128 + 4 = 132 字节/head_dim", size=11, weight="bold", fill="#166534"))
L.append(text(ix + CACHE_W / 2, y_cache + 124, "scale = head_dim/quant_block_size×4 = 128/128×4 = 4B", size=10.3, fill="#334155"))
L.append(text(ix + CACHE_W / 2, y_cache + 146, "num_kv_heads = 1(MQA 式,跨头共享 1 份)", size=10.8, fill="#334155"))
L.append(text(ix + CACHE_W / 2, y_cache + 168, "与主 KV cache 分开分配,互不引用", size=10.5, fill="#64748b"))

# ---- Tier 3: V4 页对齐 note ----
note_w = CACHE_W
nx = CACHE_X[1]
L.append(box(nx, y_note, note_w, h_note, "#f1f5f9", "#94a3b8", dash="5,4"))
L.append(text(nx + note_w / 2, y_note + 22, "V4 版页对齐 = 576 字节", size=12, weight="bold", fill="#334155"))
L.append(text(nx + note_w / 2, y_note + 42, "132B 条目 + 空位,便于与 compressor 状态一起打包", size=10.3, fill="#64748b"))
L.append(varrow(ix + CACHE_W / 2, y_cache + h_cache, y_note, color="#16a34a"))

# 左侧留白处补一句说明,呼应主 KV 无需页对齐约束
L.append(text(mx + CACHE_W / 2, y_note + 22, "主 KV cache 页对齐由主注意力后端自定,不受", size=10.3, fill="#94a3b8"))
L.append(text(mx + CACHE_W / 2, y_note + 40, "IndexCache 的 576 字节约束影响", size=10.3, fill="#94a3b8"))

L.append('</svg>')
out = Path(__file__).with_name("fig-indexcache-independence.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
