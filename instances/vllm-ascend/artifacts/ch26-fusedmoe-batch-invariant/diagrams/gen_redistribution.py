#!/usr/bin/env python3
"""fig26-3: token 按专家重分发 —— dispatch(all2all-v 不等长) → 每专家 MLP → combine。回收 f3。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1380, 780
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, color="#64748b", sw=2, arrow=True, dash=None):
    a = ' marker-end="url(#ar)"' if arrow else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{a}{d}/>')


text(W / 2, 42, "MoE 难点：token 按专家跨卡重分发（all2all-v 不等长）", 23, "middle", "#0f172a", "bold")
text(W / 2, 70, "各专家收到的 token 数天然不均 → input_splits/output_splits 给每卡不同条数；combine 是其逆置换+加权",
     13, "middle", "#64748b")

ep = ["卡0 专家E0,E1", "卡1 专家E2,E3", "卡2 专家E4,E5", "卡3 专家E6,E7"]
colors = ["#10b981", "#0ea5e9", "#f59e0b", "#ef4444"]
n = 4
cw, ch_ = 210, 70
gap = 48
x0 = 330  # 左侧留出阶段标签的横向空间

# 行1：N tokens（每卡本地一批，颜色=该 token 选中的专家所在卡）
y1 = 110
text(70, y1 + 30, "① select_experts", 13, "start", "#334155", "bold")
text(70, y1 + 50, "每 token 选 top_k 专家", 11.5, "start", "#64748b")
for i in range(n):
    x = x0 + i * (cw + gap)
    box(x, y1, cw, ch_, "#f8fafc", "#94a3b8", 10, 1.6)
    text(x + cw / 2, y1 + 28, f"卡{i}：本地 token 批", 12.5, "middle", "#334155", "bold")
    # 小色块表示混合归属
    bw = 22
    for j in range(8):
        bx = x + 14 + j * (bw + 4)
        if bx + bw < x + cw - 10:
            L.append(f'<rect x="{bx}" y="{y1+40}" width="{bw}" height="16" rx="3" fill="{colors[(i+j)%n]}" opacity="0.8"/>')

# dispatch 箭头（交叉，示意 all2all-v）
y2 = 280
text(70, y2 + 30, "② token_dispatch", 13, "start", "#b91c1c", "bold")
text(70, y2 + 50, "all2all-v 按专家归属", 11.5, "start", "#64748b")
for i in range(n):
    sx = x0 + i * (cw + gap) + cw / 2
    for k in range(n):
        tx = x0 + k * (cw + gap) + cw / 2
        line(sx, y1 + ch_, tx, y2, colors[k], 1.2, arrow=True)

# 行2：各卡收到不等长（按专家归属聚拢）
recv = ["收 6 条→E0,E1", "收 3 条→E2,E3", "收 4 条→E4,E5", "收 3 条→E6,E7"]
for i in range(n):
    x = x0 + i * (cw + gap)
    box(x, y2, cw, ch_, "white", colors[i], 10, 2)
    text(x + cw / 2, y2 + 28, f"卡{i} · {recv[i]}", 12.5, "middle", colors[i], "bold")
    text(x + cw / 2, y2 + 50, "(input/output_splits 不等)", 11, "middle", "#64748b")

# 行3：per-expert MLP
y3 = 440
text(70, y3 + 30, "③ per-expert MLP", 13, "start", "#334155", "bold")
text(70, y3 + 50, "gmm1 → swiglu → gmm2", 11.5, "start", "#64748b")
for i in range(n):
    x = x0 + i * (cw + gap)
    line(x + cw / 2, y2 + ch_, x + cw / 2, y3, colors[i], 1.8, arrow=True)
    box(x, y3, cw, ch_, "#fef9c3", colors[i], 10, 1.8)
    text(x + cw / 2, y3 + 28, f"专家 E{2*i},E{2*i+1} 各算自己", 12, "middle", "#334155", "bold")
    text(x + cw / 2, y3 + 50, "gmm1→swiglu→gmm2", 11, "middle", "#64748b", mono=True)

# 行4：combine 聚回
y4 = 600
text(70, y4 + 30, "④ token_combine", 13, "start", "#b91c1c", "bold")
text(70, y4 + 50, "逆置换 + topk_weights 加权", 11.5, "start", "#64748b")
for i in range(n):
    sx = x0 + i * (cw + gap) + cw / 2
    for k in range(n):
        tx = x0 + k * (cw + gap) + cw / 2
        line(sx, y3 + ch_, tx, y4, colors[i], 1.0, arrow=True, dash="3 3")
for i in range(n):
    x = x0 + i * (cw + gap)
    box(x, y4, cw, ch_, "#f8fafc", "#94a3b8", 10, 1.6)
    text(x + cw / 2, y4 + 30, f"卡{i}：还原回 N 个 token", 12.5, "middle", "#334155", "bold")
    text(x + cw / 2, y4 + 52, "顺序/形状与输入一致", 11, "middle", "#64748b")

# 底部三实现注记
box(70, 700, W - 140, 56, "#f1f5f9", "#cbd5e1", 10, 1.4)
text(90, 722, "三种落地：", 12.5, "start", "#334155", "bold")
text(90, 744, "MC2 = npu_moe_distribute_dispatch/combine 融合算子   ·   All2AllV = async_all_to_all(dist.all_to_all_single)   ·   AllGather = 全局复制后本地 npu_moe_init_routing 排序",
     11.5, "start", "#475569")
text(W - 90, 722, "②④ 的 all2all 正是 ch06 留给 MoE 的用武之地", 11.5, "end", "#b91c1c")

L.append('</svg>')
open("fig26-3-redistribution.svg", "w").write('\n'.join(L))
print("wrote fig26-3-redistribution.svg")
