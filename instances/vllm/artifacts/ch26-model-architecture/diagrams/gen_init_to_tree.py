#!/usr/bin/env python3
"""P1 worked example：DeepseekV4 的 __init__ → 嵌套模块树。
着色按 Column/Row/Merged ParallelLinear 区分并行子系统。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1280, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">P1：读 __init__ → DeepSeek-V4 嵌套模块树</text>')
L.append(f'<text x="{W/2}" y="60" text-anchor="middle" font-size="14" fill="#64748b">每条 self.&lt;name&gt; = Module(prefix=f"{{prefix}}.&lt;name&gt;") = 一个子框 + 一条父→子嵌套边</text>')


def box(x, y, w, h, title, sub, stroke, fill, tfs=15):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+w/2}" y="{y+(22 if sub else h/2+5)}" text-anchor="middle" font-size="{tfs}" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    if sub:
        L.append(f'<text x="{x+w/2}" y="{y+40}" text-anchor="middle" font-size="11.5" fill="#64748b">{esc(sub)}</text>')


# 颜色：列并行=蓝，行并行=绿，合并列并行(replicated)=紫，普通模块=灰，嵌入/头=橙
COL = ("#1d4ed8", "#dbeafe")   # ColumnParallelLinear
ROW = ("#047857", "#d1fae5")   # RowParallelLinear
MRG = ("#7c3aed", "#ede9fe")   # MergedColumnParallelLinear (replicated)
PLN = ("#475569", "#f1f5f9")   # 普通 nn.Module (RMSNorm/gate/...)
EMB = ("#b45309", "#fef3c7")   # 嵌入 / lm_head (Vocab/ParallelLMHead)

# 最外层：DeepseekV4ForCausalLM
ox, oy, ow, oh = 30, 80, W - 60, H - 130
L.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="12" fill="none" stroke="#0f172a" stroke-width="2.5"/>')
L.append(f'<text x="{ox+18}" y="{oy+26}" font-size="16" font-weight="bold" fill="#0f172a">DeepseekV4ForCausalLM  (从这里开读)</text>')

# 三个顶层框：model / lm_head / logits_processor
# lm_head + logits_processor 放右侧
box(ox + ow - 250, oy + 50, 230, 50, "lm_head", "ParallelLMHead", *EMB, tfs=15)
box(ox + ow - 250, oy + 115, 230, 50, "logits_processor", "LogitsProcessor", *PLN, tfs=14)

# model 容器（占左 + 中）
mx, my, mw, mh = ox + 20, oy + 50, ow - 290, oh - 70
L.append(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="10" fill="#fafafa" stroke="#334155" stroke-width="2"/>')
L.append(f'<text x="{mx+14}" y="{my+24}" font-size="15" font-weight="bold" fill="#0f172a">model  (DeepseekV4Model · @support_torch_compile)</text>')

# model 内：embed_tokens / layers×N / norm
inner_y = my + 40
box(mx + 16, inner_y, 200, 48, "embed_tokens", "VocabParallelEmbedding", *EMB, tfs=14)
box(mx + 16, my + mh - 64, 200, 48, "norm", "RMSNorm", *PLN, tfs=15)

# layers × N 复数框（堆叠层信号：make_layers）—— 用三层叠影
lx, ly, lw, lh = mx + 250, inner_y, mw - 270, mh - 56
for k in range(2, -1, -1):
    off = k * 7
    L.append(f'<rect x="{lx+off}" y="{ly+off}" width="{lw}" height="{lh}" rx="10" fill="#fff7ed" stroke="#9a3412" stroke-width="2"/>')
L.append(f'<text x="{lx+18}" y="{ly+26}" font-size="15" font-weight="bold" fill="#9a3412">layers  × num_hidden_layers</text>')
L.append(f'<text x="{lx+18}" y="{ly+44}" font-size="11.5" fill="#9a3412">make_layers(lambda: DeepseekV4DecoderLayer(...))</text>')

# 单个 DecoderLayer 内：attn / ffn (+ attn_norm/ffn_norm)
dl_x, dl_y, dl_w, dl_h = lx + 18, ly + 56, lw - 36, lh - 70
L.append(f'<rect x="{dl_x}" y="{dl_y}" width="{dl_w}" height="{dl_h}" rx="9" fill="#fffbeb" stroke="#b45309" stroke-width="1.8"/>')
L.append(f'<text x="{dl_x+12}" y="{dl_y+20}" font-size="13.5" font-weight="bold" fill="#92400e">DeepseekV4DecoderLayer (单层)</text>')

# attn 子树
ax, ay, aw, ah = dl_x + 12, dl_y + 32, (dl_w - 36) / 2, dl_h - 50
L.append(f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" rx="8" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.8"/>')
L.append(f'<text x="{ax+aw/2}" y="{ay+20}" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1e40af">attn (DeepseekV4Attention)</text>')
mla_leaves = [
    ("fused_wqa_wkv", "Merged·repl", MRG),
    ("q_norm", "RMSNorm", PLN),
    ("wq_b", "Column", COL),
    ("kv_norm", "RMSNorm", PLN),
    ("wo_a", "Column", COL),
    ("wo_b", "Row", ROW),
]
leaf_w = (aw - 30) / 2
for i, (nm, tag, (st, fl)) in enumerate(mla_leaves):
    r, c = divmod(i, 2)
    bx = ax + 10 + c * (leaf_w + 6)
    by = ay + 30 + r * 42
    L.append(f'<rect x="{bx}" y="{by}" width="{leaf_w}" height="36" rx="5" fill="{fl}" stroke="{st}" stroke-width="1.6"/>')
    L.append(f'<text x="{bx+leaf_w/2}" y="{by+15}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#0f172a">{esc(nm)}</text>')
    L.append(f'<text x="{bx+leaf_w/2}" y="{by+29}" text-anchor="middle" font-size="9.5" fill="#64748b">{esc(tag)}</text>')

# ffn 子树（MoE）
fx2 = ax + aw + 12
L.append(f'<rect x="{fx2}" y="{ay}" width="{aw}" height="{ah}" rx="8" fill="#f0fdf4" stroke="#047857" stroke-width="1.8"/>')
L.append(f'<text x="{fx2+aw/2}" y="{ay+20}" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#065f46">ffn (DeepseekV4MoE)</text>')
moe_leaves = [
    ("gate", "GateLinear", PLN),
    ("shared_experts?", "if n_shared", ("#ca8a04", "#fef9c3")),
    ("experts", "FusedMoE ⟂ MegaMoE", ("#be123c", "#ffe4e6")),
]
for i, (nm, tag, (st, fl)) in enumerate(moe_leaves):
    by = ay + 32 + i * 50
    L.append(f'<rect x="{fx2+12}" y="{by}" width="{aw-24}" height="42" rx="6" fill="{fl}" stroke="{st}" stroke-width="1.8"/>')
    L.append(f'<text x="{fx2+aw/2}" y="{by+18}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#0f172a">{esc(nm)}</text>')
    L.append(f'<text x="{fx2+aw/2}" y="{by+33}" text-anchor="middle" font-size="10" fill="#64748b">{esc(tag)}</text>')

# 图例
lg_y = H - 26
legend = [("Column 列并行", COL), ("Row 行并行", ROW), ("Merged·repl", MRG), ("嵌入/头", EMB), ("普通模块", PLN), ("条件框", ("#ca8a04", "#fef9c3")), ("互斥后端", ("#be123c", "#ffe4e6"))]
lx0 = 30
for nm, (st, fl) in legend:
    L.append(f'<rect x="{lx0}" y="{lg_y-12}" width="16" height="14" rx="3" fill="{fl}" stroke="{st}" stroke-width="1.6"/>')
    L.append(f'<text x="{lx0+22}" y="{lg_y}" font-size="12" fill="#334155">{esc(nm)}</text>')
    lx0 += 36 + len(nm) * 13

L.append('</svg>')
svg = '\n'.join(L)
out = __import__('pathlib').Path(__file__).parent / "init-to-tree.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}")
