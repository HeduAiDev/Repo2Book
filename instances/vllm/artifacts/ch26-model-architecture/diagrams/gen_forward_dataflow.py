#!/usr/bin/env python3
"""P2+P3 worked example：DeepseekV4 的 forward → 有向数据流 + 形状注记 + 子系统着色。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 880
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
    '<marker id="res" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '<marker id="byp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#9333ea"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">P2+P3：读 forward → 主干数据流（形状注记 + 子系统着色）</text>')

cx = 420  # 主干中线
bw = 280
bx = cx - bw / 2


def node(y, name, sub, stroke, fill, h=46, dashed=False):
    da = ' stroke-dasharray="6,4"' if dashed else ''
    L.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"{da}/>')
    L.append(f'<text x="{cx}" y="{y+(20 if sub else h/2+5)}" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    if sub:
        L.append(f'<text x="{cx}" y="{y+37}" text-anchor="middle" font-size="11" fill="#64748b">{esc(sub)}</text>')
    return y, y + h


def edge(y1, y2, label=None, shape=None):
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2-6}" stroke="#334155" stroke-width="2.5" marker-end="url(#arr)"/>')
    midy = (y1 + y2) / 2
    if shape:
        L.append(f'<rect x="{cx+8}" y="{midy-12}" width="{len(shape)*8.2+12}" height="22" rx="5" fill="#fff" stroke="#0369a1" stroke-width="1.3"/>')
        L.append(f'<text x="{cx+14}" y="{midy+4}" font-size="12.5" font-family="monospace" fill="#0369a1">{esc(shape)}</text>')
    if label:
        L.append(f'<text x="{cx-14}" y="{midy+4}" text-anchor="end" font-size="11.5" fill="#475569">{esc(label)}</text>')


COL = ("#334155", "#f1f5f9")
OP = ("#b45309", "#fef3c7")     # torch.ops.vllm.* 算子框
y = 60
# embed
_, b = node(y, "embed_input_ids", "VocabParallelEmbedding", "#b45309", "#fef3c7")
top_embed = y
y = b + 34
edge(b, y, label="unsqueeze(-2).repeat(hc_mult)", shape="[T,H]→[T,hc_mult,H]")

# 编译区容器开始（DeepseekV4Model @support_torch_compile 包住主干）
comp_top = y - 8

# 逐层堆叠（×N）容器
ly = y
lay_h = 490
L.append(f'<rect x="{bx-60}" y="{ly}" width="{bw+170}" height="{lay_h}" rx="12" fill="#fff7ed" stroke="#9a3412" stroke-width="2"/>')
L.append(f'<text x="{bx-50}" y="{ly+22}" font-size="13.5" font-weight="bold" fill="#9a3412">for layer in layers   (× num_hidden_layers)</text>')

y = ly + 40
# 层内：hc_pre → attn_norm → attn → hc_post → hc_pre → ffn_norm → ffn → hc_post
seq = [
    ("hc_pre", "torch.ops.vllm.mhc_pre", OP),
    ("attn_norm", "RMSNorm", COL),
    ("attn", "DeepseekV4Attention (MLA)", ("#1d4ed8", "#dbeafe")),
    ("hc_post", "torch.ops.vllm.mhc_post", OP),
    ("hc_pre", "torch.ops.vllm.mhc_pre", OP),
    ("ffn_norm", "RMSNorm", COL),
    ("ffn", "DeepseekV4MoE", ("#047857", "#d1fae5")),
    ("hc_post", "torch.ops.vllm.mhc_post", OP),
]
seg_centers = []
for i, (nm, sub, (st, fl)) in enumerate(seq):
    t, bb = node(y, nm, sub, st, fl, h=38)
    seg_centers.append((y, bb))
    if i < len(seq) - 1:
        y = bb + 18
        L.append(f'<line x1="{cx}" y1="{bb}" x2="{cx}" y2="{y-6}" stroke="#334155" stroke-width="2" marker-end="url(#arr)"/>')
    else:
        y = bb

# 残差回边：residual=x → hc_post（两段）
ratt0 = seg_centers[0][0]
ratt1 = seg_centers[3][1]
rx = bx - 68
L.append(f'<path d="M {bx} {ratt0+6} L {rx} {ratt0+6} L {rx} {ratt1-6} L {bx} {ratt1-6}" fill="none" stroke="#b45309" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#res)"/>')
L.append(f'<text x="{rx-4}" y="{(ratt0+ratt1)/2}" text-anchor="end" font-size="11" fill="#b45309" font-weight="bold">residual</text>')
rffn0 = seg_centers[4][0]
rffn1 = seg_centers[7][1]
L.append(f'<path d="M {bx} {rffn0+6} L {rx} {rffn0+6} L {rx} {rffn1-6} L {bx} {rffn1-6}" fill="none" stroke="#b45309" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#res)"/>')
L.append(f'<text x="{rx-4}" y="{(rffn0+rffn1)/2}" text-anchor="end" font-size="11" fill="#b45309" font-weight="bold">residual</text>')

# 离开层循环
y = ly + lay_h + 26
L.append(f'<line x1="{cx}" y1="{ly+lay_h}" x2="{cx}" y2="{y-6}" stroke="#334155" stroke-width="2.5" marker-end="url(#arr)"/>')

# copy_ 旁路 → _mtp_hidden_buffer
mtp_y = y + 4
_, mb = node(y, "hc_head", "@torch.compile · flatten→view→sum", "#7c3aed", "#ede9fe")
y2 = mb + 34
edge(mb, y2, shape="[T,hc_mult,H]→[T,H]")

# 旁路框（右侧）
byp_x = cx + 220
L.append(f'<rect x="{byp_x}" y="{mtp_y}" width="200" height="50" rx="8" fill="#faf5ff" stroke="#9333ea" stroke-width="2" stroke-dasharray="6,4"/>')
L.append(f'<text x="{byp_x+100}" y="{mtp_y+20}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#6b21a8">_mtp_hidden_buffer</text>')
L.append(f'<text x="{byp_x+100}" y="{mtp_y+38}" text-anchor="middle" font-size="10.5" fill="#7e22ce">copy_(flatten(1)) 旁路去 MTP</text>')
L.append(f'<path d="M {cx+bw/2} {mtp_y+10} L {byp_x-4} {mtp_y+25}" fill="none" stroke="#9333ea" stroke-width="2" marker-end="url(#byp)"/>')

# norm 收尾
_, nb = node(y2, "norm", "RMSNorm", "#334155", "#f1f5f9")
y3 = nb + 30
edge(nb, y3)
node(y3, "lm_head → logits", "ParallelLMHead", "#b45309", "#fef3c7")

# 编译区虚线容器（包 hc_head 到 norm 这段说明编译边界来自装饰器）
comp_y2 = nb + 6
L.append(f'<rect x="{bx-12}" y="{mtp_y-8}" width="{bw+24}" height="{comp_y2-mtp_y+16}" rx="10" fill="none" stroke="#7c3aed" stroke-width="1.6" stroke-dasharray="4,4"/>')
L.append(f'<text x="{bx-16}" y="{mtp_y-14}" font-size="11" fill="#7c3aed" font-weight="bold">torch.compile 编译区</text>')

# 图例
lg_y = H - 10
legend = [("普通 nn.Module", COL), ("MLA 注意力", ("#1d4ed8", "#dbeafe")), ("MoE", ("#047857", "#d1fae5")), ("torch.ops.vllm.* 算子", OP), ("编译区/旁路", ("#7c3aed", "#ede9fe"))]
lx0 = 30
for nm, (st, fl) in legend:
    L.append(f'<rect x="{lx0}" y="{lg_y-12}" width="16" height="14" rx="3" fill="{fl}" stroke="{st}" stroke-width="1.6"/>')
    L.append(f'<text x="{lx0+22}" y="{lg_y}" font-size="11.5" fill="#334155">{esc(nm)}</text>')
    lx0 += 50 + len(nm) * 13

L.append('</svg>')
svg = '\n'.join(L)
out = __import__('pathlib').Path(__file__).parent / "forward-dataflow.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}")
