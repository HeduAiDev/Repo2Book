#!/usr/bin/env python3
"""9 步采样流水线总览：竖向 step 方块 + step7 处分叉 greedy 快路 / 随机路径。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

W, H = 1040, 1160
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
         '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def box(x, y, w, h, fill, stroke, lines, fs=15, tw="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    cy = y + h/2 - (n-1)*0.5*(fs+4)
    for i, t in enumerate(lines):
        fw = 'bold' if (tw == 'bold' and i == 0) else 'normal'
        L.append(f'<text x="{x+w/2}" y="{cy + i*(fs+4) + fs*0.35:.0f}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" fill="#1e293b">{esc(t)}</text>')

def varrow(x, y1, y2, marker="a"):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#475569" stroke-width="2" marker-end="url(#{marker})"/>')

cx = 300
bw = 380
# 上半：线性 step1-6（forward + apply_logits_processors）
steps = [
    ("step1  抽取 raw logprobs", "在任何惩罚/温度之前 · log_softmax(原始 logits)", "#eef2ff", "#6366f1"),
    ("step2  logits → float32", "全程 fp32 算，避免半精度数值问题", "#eef2ff", "#6366f1"),
    ("step3  allowed token 白名单", "非白名单位置 masked_fill_ 置 -inf", "#fef2f2", "#ef4444"),
    ("step4  bad words 屏蔽", "前缀命中→末 token 置 -inf", "#fef2f2", "#ef4444"),
    ("step5  non-argmax-invariant 处理器", "min_tokens / logit_bias（能改 argmax）", "#fef2f2", "#ef4444"),
    ("step6  penalties", "repetition / frequency / presence", "#fef2f2", "#ef4444"),
]
y = 30
bh = 56
gap = 24
for i, (t, sub, fill, st) in enumerate(steps):
    box(cx, y, bw, bh, fill, st, [t, sub], fs=14, tw="bold")
    if i < len(steps)-1:
        varrow(cx+bw/2, y+bh, y+bh+gap)
    y += bh + gap

# 左侧 region 标注
L.append(f'<rect x="40" y="146" width="22" height="{(steps and (56+24))*4-24}" rx="6" fill="#fee2e2" stroke="#ef4444" stroke-width="1.2"/>')
L.append(f'<text x="51" y="290" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold" '
         f'fill="#b91c1c" transform="rotate(-90 51 290)">step3-6 改 argmax</text>')

# step7 分叉节点
y7 = y
box(cx, y7, bw, 50, "#f0fdf4", "#16a34a", ["step7  sample —— 在这里分叉", ""], fs=15, tw="bold")
varrow(cx+bw/2, y-gap+bh if False else y7, y7)  # connector from step6 already drawn above? draw explicit
# explicit arrow from step6 bottom to step7
# step6 bottom y was y - bh - gap + bh = y-gap ; recompute
# Simpler: draw arrow from (cx+bw/2, step6_bottom) to step7 top
step6_bottom = 30 + 6*(bh+gap) - gap
varrow(cx+bw/2, step6_bottom, y7)

# 分叉两路
yb = y7 + 50 + 40
# 左路 greedy 快路
gx = 70
gw = 320
L.append(f'<line x1="{cx+bw/2}" y1="{y7+50}" x2="{gx+gw/2}" y2="{yb}" stroke="#15803d" stroke-width="2" marker-end="url(#ag)"/>')
box(gx, yb, gw, 70, "#dcfce7", "#15803d",
    ["greedy 快路（all_greedy 早退）", "greedy_sample = argmax", "跳过温度 / min_p / top-k / top-p"], fs=13, tw="bold")

# 右路 随机
rx = 560
rw = 410
L.append(f'<line x1="{cx+bw/2}" y1="{y7+50}" x2="{rx+rw/2}" y2="{yb}" stroke="#b45309" stroke-width="2" marker-end="url(#ar)"/>')
rsteps = [
    ("7b  apply_temperature", "logits /= T（temp<eps 替 1.0 防除零）"),
    ("7c  argmax-invariant 处理器", "默认 min_p（永不改 argmax）"),
    ("7d  top-k / top-p 截断", "TopKTopPSampler 多后端"),
    ("7e  随机采样", "random_sample / flashinfer"),
]
ry = yb
rbh = 52
for i, (t, sub) in enumerate(rsteps):
    box(rx, ry, rw, rbh, "#fffbeb", "#b45309", [t, sub], fs=13, tw="bold")
    if i < len(rsteps)-1:
        L.append(f'<line x1="{rx+rw/2}" y1="{ry+rbh}" x2="{rx+rw/2}" y2="{ry+rbh+18}" stroke="#b45309" stroke-width="2" marker-end="url(#ar)"/>')
    ry += rbh + 18

# 汇合 torch.where
merge_y = max(yb+70, ry) + 36
mx = cx
mw = bw
box(mx, merge_y, mw, 50, "#ede9fe", "#7c3aed",
    ["torch.where(temp<eps, greedy, random)", "混合批逐请求合并"], fs=14, tw="bold")
# arrows into merge
L.append(f'<line x1="{gx+gw/2}" y1="{yb+70}" x2="{mx+mw*0.32}" y2="{merge_y}" stroke="#15803d" stroke-width="2" marker-end="url(#ag)"/>')
L.append(f'<line x1="{rx+rw/2}" y1="{ry-18}" x2="{mx+mw*0.68}" y2="{merge_y}" stroke="#b45309" stroke-width="2" marker-end="url(#ar)"/>')

# step8 / step9
y8 = merge_y + 50 + 30
box(cx, y8, bw, 50, "#eef2ff", "#6366f1",
    ["step8  gather_logprobs", "对 step1 的 raw_logprobs 取 topk + 采样 token + rank"], fs=14, tw="bold")
varrow(cx+bw/2, merge_y+50, y8)
y9 = y8 + 50 + 30
box(cx, y9, bw, 46, "#eef2ff", "#6366f1", ["step9  返回 SamplerOutput", ""], fs=14, tw="bold")
varrow(cx+bw/2, y8+50, y9)

# 方法归属标注（右侧）
L.append(f'<text x="700" y="50" font-family="sans-serif" font-size="13" fill="#64748b">forward()</text>')
L.append(f'<text x="700" y="200" font-family="sans-serif" font-size="13" fill="#64748b">apply_logits_processors()</text>')
L.append(f'<text x="{rx+rw+8}" y="{yb+20}" font-family="sans-serif" font-size="13" fill="#64748b">sample()</text>')

L.append('</svg>')
open("01-sampling-pipeline.svg", "w", encoding="utf-8").write('\n'.join(L))
print("ok")
