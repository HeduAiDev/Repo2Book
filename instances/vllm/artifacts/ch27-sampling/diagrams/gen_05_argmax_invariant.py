#!/usr/bin/env python3
"""argmax-invariant 分类表：所有 logits 处理器按是否改变 argmax 归类，
标注各自在流水线中的位置（step3-6 前置 vs. step7c 随机路）。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

W, H = 880, 500
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---- title ----
L.append(f'<text x="{W//2}" y="34" text-anchor="middle" font-family="sans-serif" '
         f'font-size="17" font-weight="bold" fill="#1e293b">logits 处理器按 argmax-invariance 分类</text>')
L.append(f'<text x="{W//2}" y="56" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="#64748b">'
         f'is_argmax_invariant() == False → 前置 step3-6（可改 argmax，greedy 必须看到它们）'
         f'</text>')
L.append(f'<text x="{W//2}" y="74" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="#64748b">'
         f'is_argmax_invariant() == True  → 推迟 step7c（不改 argmax，greedy 可安全跳过）'
         f'</text>')

# ---- column headers ----
col_l_x = 60
col_r_x = 470
col_w = 360
hdr_y = 96
hdr_h = 38

def hdr(x, fill, stroke, txt, sub):
    L.append(f'<rect x="{x}" y="{hdr_y}" width="{col_w}" height="{hdr_h}" rx="6" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{x + col_w//2}" y="{hdr_y+16}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="14" font-weight="bold" fill="#1e293b">{esc(txt)}</text>')
    L.append(f'<text x="{x + col_w//2}" y="{hdr_y+30}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" fill="#64748b">{esc(sub)}</text>')

hdr(col_l_x, "#fee2e2", "#ef4444",
    "NOT argmax-invariant  →  前置（step 3–6）",
    "能改 greedy 结果，必须在 all_greedy 早退之前生效")
hdr(col_r_x, "#dcfce7", "#16a34a",
    "argmax-invariant  →  推迟（step 7c）",
    "不改 greedy 结果，greedy 早退可安全跳过")

# ---- left column items ----
left_items = [
    ("step3  allowed token 白名单",
     "masked_fill_(-inf) 把非白名单 token 摁死",
     "→ 能把原 argmax token 直接删掉"),
    ("step4  bad words 屏蔽",
     "前缀命中时把末 token 置 -inf",
     "→ 能让原最高分 token 失效"),
    ("step5a  min_tokens 处理器",
     "在未到最小长度前摁死 EOS/stop",
     "→ 若 EOS 原为 argmax 则座次换人"),
    ("step5b  logit_bias 处理器",
     "给指定 token 加偏置（可正可负）",
     "→ 加够了能把其他 token 顶到第一"),
    ("step6a  repetition penalty",
     "出现过的 token：正 logit 除、负 logit 乘",
     "→ 够重时能压低原 argmax"),
    ("step6b  frequency penalty",
     "logit -= freq × 出现次数（线性比例）",
     "→ 高频出现可导致座次变化"),
    ("step6c  presence penalty",
     "出现过即扣固定值，不管次数",
     "→ 能把原 argmax 拉出前位"),
]

right_items = [
    ("step7c  min_p 处理器",
     "砍掉概率 < min_p × max_prob 的尾部",
     "max_prob 本身满足 max_prob ≥ min_p×max_prob"),
    ("step7b  temperature 缩放",
     "logits /= T（T>0 单调变换）",
     "除以正数不改任何两 logit 的大小关系"),
    ("step7d  top-k 截断",
     "只砍升序最小的 V-k 个",
     "最大值 token 永远落在留下的 k 个里"),
    ("step7d  top-p 截断",
     "砍掉累积概率尾部",
     "最大概率 token 永远排在累积首位"),
]

row_y0 = hdr_y + hdr_h + 10
row_h = 56
row_gap = 6

def item_box(x, idx, name, desc, proof, fill, stroke):
    y = row_y0 + idx * (row_h + row_gap)
    L.append(f'<rect x="{x}" y="{y}" width="{col_w}" height="{row_h}" rx="5" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    L.append(f'<text x="{x+12}" y="{y+17}" font-family="sans-serif" font-size="12.5" '
             f'font-weight="bold" fill="#1e293b">{esc(name)}</text>')
    L.append(f'<text x="{x+12}" y="{y+31}" font-family="sans-serif" font-size="11" '
             f'fill="#475569">{esc(desc)}</text>')
    L.append(f'<text x="{x+12}" y="{y+46}" font-family="sans-serif" font-size="10.5" '
             f'fill="#b45309" font-style="italic">{esc(proof)}</text>')

for i, (name, desc, proof) in enumerate(left_items):
    item_box(col_l_x, i, name, desc, proof, "#fff1f2", "#fca5a5")

for i, (name, desc, proof) in enumerate(right_items):
    item_box(col_r_x, i, name, desc, proof, "#f0fdf4", "#86efac")

# ---- divider ----
mid_x = (col_l_x + col_w + col_r_x) // 2
L.append(f'<line x1="{mid_x}" y1="{hdr_y}" x2="{mid_x}" y2="{H-20}" '
         f'stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6 3"/>')

# ---- footer ----
footer_y = H - 18
L.append(f'<text x="{W//2}" y="{footer_y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#94a3b8">'
         f'vllm/v1/sample/logits_processor/state.py:L148  ·  builtin.py:L46, L130, L186'
         f'</text>')

L.append('</svg>')

out = '/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch27-sampling/diagrams/05-argmax-invariant.svg'
open(out, 'w', encoding='utf-8').write('\n'.join(L))
print("ok:", out)
