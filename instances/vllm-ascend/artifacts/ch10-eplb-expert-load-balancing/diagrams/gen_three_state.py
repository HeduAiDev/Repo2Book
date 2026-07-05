#!/usr/bin/env python3
"""D2DExpertWeightLoader 三态机：WAITING→READY→TRANSFERRING→WAITING，每条边标触发函数。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1180, 660
S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="a" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')
S.append(f'<text x="{W/2}" y="42" text-anchor="middle" font-size="23" font-weight="bold" fill="#0f172a">D2DExpertWeightLoader 三态机：发车与落地隔开，防上一层没搬完就接新任务</text>')

states = [
    (110, "WAITING", "0", "#fee2e2", "#dc2626", "#991b1b", "等 EplbWorker 的新 expert_map"),
    (490, "READY", "1", "#fef9c3", "#ca8a04", "#854d0e", "P2POp 已攒好，待发车"),
    (870, "TRANSFERRING", "2", "#dcfce7", "#16a34a", "#166534", "isend/irecv 已发，待回收落地"),
]
bw, bh, by = 200, 130, 240
cx_list = []
for x, name, num, fill, stroke, txtc, sub in states:
    cx = x + bw/2
    cx_list.append(cx)
    S.append(f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    S.append(f'<text x="{cx}" y="{by+50}" text-anchor="middle" font-size="20" font-weight="bold" fill="{txtc}">{esc(name)}</text>')
    S.append(f'<text x="{cx}" y="{by+78}" text-anchor="middle" font-size="14" fill="{txtc}">state = {esc(num)}</text>')
    S.append(f'<text x="{cx}" y="{by+106}" text-anchor="middle" font-size="12.5" fill="{stroke}">{esc(sub)}</text>')

# 边 1：WAITING → READY（标签置于上方空白区）
y_edge = by + bh/2
S.append(f'<line x1="{states[0][0]+bw}" y1="{y_edge}" x2="{states[1][0]-6}" y2="{y_edge}" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
mx = (states[0][0]+bw + states[1][0])/2
S.append(f'<text x="{mx}" y="155" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1e293b">generate_expert_d2d_transfer_task</text>')
S.append(f'<text x="{mx}" y="177" text-anchor="middle" font-size="12.5" fill="#475569">攒 P2POp</text>')
S.append(f'<text x="{mx}" y="195" text-anchor="middle" font-size="12.5" fill="#475569">（每 expert 多块张量各挂 isend/irecv）</text>')

# 边 2：READY → TRANSFERRING
S.append(f'<line x1="{states[1][0]+bw}" y1="{y_edge}" x2="{states[2][0]-6}" y2="{y_edge}" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
mx2 = (states[1][0]+bw + states[2][0])/2
S.append(f'<text x="{mx2}" y="155" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1e293b">asyn_expert_weight_transfer</text>')
S.append(f'<text x="{mx2}" y="177" text-anchor="middle" font-size="12.5" fill="#475569">batch_isend_irecv</text>')
S.append(f'<text x="{mx2}" y="195" text-anchor="middle" font-size="12.5" fill="#475569">异步发车（不等）</text>')

# 边 3：TRANSFERRING → WAITING（下方回弧）
y_arc = by + bh + 110
S.append(f'<path d="M {cx_list[2]} {by+bh} '
         f'C {cx_list[2]} {y_arc}, {cx_list[0]} {y_arc}, {cx_list[0]} {by+bh+4}" '
         f'stroke="#475569" stroke-width="2.5" fill="none" marker-end="url(#a)"/>')
S.append(f'<text x="{W/2}" y="{y_arc-6}" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1e293b">update_expert_map_and_weight</text>')
S.append(f'<text x="{W/2}" y="{y_arc+14}" text-anchor="middle" font-size="12.5" fill="#475569">req.wait() 收车 → copy_ 落地新权重 → 更新 expert_map / log2phy_map → 状态回 WAITING（闭环）</text>')

# 自守卫标注
S.append(f'<rect x="350" y="{by+bh+150}" width="480" height="46" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>')
S.append(f'<text x="590" y="{by+bh+170}" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">互斥保护：非 WAITING 态下 generate 直接 return（warning_once）</text>')
S.append(f'<text x="590" y="{by+bh+188}" text-anchor="middle" font-size="12" fill="#64748b">上一层的搬运没落地，绝不接下一层的新任务</text>')

S.append('</svg>')
open("three_state.svg", "w").write("\n".join(S))
print("wrote three_state.svg")
