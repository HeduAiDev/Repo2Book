#!/usr/bin/env python3
"""节拍时间线表：缩小例 interval=4, algo=2, num_moe_layers=3。
逐拍 cur_iterations 演示三个 flag 何时触发、forward_before / forward_end 各干什么。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


# 缩小例常量
INTERVAL = 4   # expert_heat_collection_interval
ALGO = 2       # algorithm_execution_interval
L = 3          # num_moe_layers
CYCLE = INTERVAL + ALGO + L  # 9

rows = [
    # cur, forward_before, forward_end, phase, color
    (0, "—", "—", "累积本地负载", "idle"),
    (1, "—", "—", "累积本地负载", "idle"),
    (2, "—", "—", "累积本地负载", "idle"),
    (3, "—", "wakeup ✓\nall_gather → shared_dict\nplanner_q.put(1)", "interval-1：gather + 点火子进程", "wake"),
    (4, "—", "—", "子进程规划中（并行）", "plan"),
    (5, "get_update ✓\nblock_update_q.get()", "—", "interval+algo-1：取回规划", "get"),
    (6, "weight ✓ L0\ngenerate + asyn\nisend/irecv 发车", "weight ✓ L0\nreq.wait + copy_ 落地", "搬第 0 层权重", "move"),
    (7, "weight ✓ L1\n发车", "weight ✓ L1\n收车落地", "搬第 1 层权重", "move"),
    (8, "weight ✓ L2\n发车", "weight ✓ L2\n收车落地", "搬第 2 层；末拍 update_iteration → 9==9 归零", "move"),
]

fill = {
    "idle": "#f1f5f9", "wake": "#fde68a", "plan": "#e0e7ff",
    "get": "#bae6fd", "move": "#bbf7d0",
}

# 几何
x0 = 30
col_w = [70, 250, 250, 360]   # cur / forward_before / forward_end / 节拍说明
xs_col = [x0]
for w in col_w:
    xs_col.append(xs_col[-1] + w)
W = xs_col[-1] + 30
header_h = 56
row_h = 78
H = header_h + 70 + len(rows) * row_h + 40

S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# 标题
S.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">节拍状态机：一整轮 = interval(4) + algo(2) + num_moe_layers(3) = 9 拍</text>')
S.append(f'<text x="{W/2}" y="58" text-anchor="middle" font-size="14" fill="#64748b">真实默认 interval=400 / algo=30；这里用缩小例把「哪一拍干什么」走清</text>')

ty = 84
# 表头
headers = ["cur", "forward_before（step 前）", "forward_end（step 后）", "节拍说明"]
S.append(f'<rect x="{x0}" y="{ty}" width="{W-2*x0}" height="{header_h}" fill="#1e293b" rx="6"/>')
for i, htext in enumerate(headers):
    cx = (xs_col[i] + xs_col[i+1]) / 2
    S.append(f'<text x="{cx}" y="{ty+35}" text-anchor="middle" font-size="15" font-weight="bold" fill="white">{esc(htext)}</text>')

y = ty + header_h + 6
for cur, fb, fe, phase, ckey in rows:
    rh = row_h
    # cur 单元格
    S.append(f'<rect x="{xs_col[0]}" y="{y}" width="{col_w[0]}" height="{rh}" fill="{fill[ckey]}" stroke="#cbd5e1"/>')
    S.append(f'<text x="{(xs_col[0]+xs_col[1])/2}" y="{y+rh/2+8}" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">{cur}</text>')

    def cell(idx, text):
        cx = (xs_col[idx] + xs_col[idx+1]) / 2
        S.append(f'<rect x="{xs_col[idx]}" y="{y}" width="{col_w[idx]}" height="{rh}" fill="{fill[ckey]}" stroke="#cbd5e1"/>')
        lines = text.split("\n")
        n = len(lines)
        start_y = y + rh/2 - (n-1)*9 + 5
        for j, ln in enumerate(lines):
            fw = 'bold' if ('✓' in ln) else 'normal'
            col = '#15803d' if '✓' in ln else '#334155'
            S.append(f'<text x="{cx}" y="{start_y + j*18}" text-anchor="middle" font-size="13" font-weight="{fw}" fill="{col}">{esc(ln)}</text>')

    cell(1, fb)
    cell(2, fe)
    # 节拍说明左对齐
    S.append(f'<rect x="{xs_col[3]}" y="{y}" width="{col_w[3]}" height="{rh}" fill="{fill[ckey]}" stroke="#cbd5e1"/>')
    S.append(f'<text x="{xs_col[3]+14}" y="{y+rh/2+5}" text-anchor="start" font-size="13.5" fill="#1e293b">{esc(phase)}</text>')
    y += rh

S.append('</svg>')
open("cadence_table.svg", "w").write("\n".join(S))
print("wrote cadence_table.svg")
