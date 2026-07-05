#!/usr/bin/env python3
"""compose_expert_update_info_greedy：用两张 expert_map 的 -1 做差集，推出 send/recv 对。
缩小例：2 卡 4 expert，把 e1（rank0）与 e3（rank1）对调。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1180, 600
S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="a" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')
S.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">从 expert_map 的 -1 差集，反推「谁发给谁哪个 expert」</text>')
S.append(f'<text x="{W/2}" y="66" text-anchor="middle" font-size="14" fill="#64748b">缩小例：2 卡 4 expert，把 rank0 的 e1 与 rank1 的 e3 对调（-1 = 该卡不持有该 expert）</text>')

experts = ["e0", "e1", "e2", "e3"]
# expert_map[rank][global_expert] = 本地槽位 or -1
current = [[0, 1, -1, -1], [-1, -1, 0, 1]]
updated = [[0, -1, -1, 1], [-1, 1, 0, -1]]


def draw_table(x0, y0, title, data, diffmark):
    cw, ch = 70, 44
    label_w = 80
    S.append(f'<text x="{x0 + label_w + 2*cw}" y="{y0-14}" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">{esc(title)}</text>')
    # 列头
    for j, e in enumerate(experts):
        cx = x0 + label_w + j*cw + cw/2
        S.append(f'<text x="{cx}" y="{y0+ch-14}" text-anchor="middle" font-size="13" font-weight="bold" fill="#475569">{esc(e)}</text>')
    for i in range(2):
        ry = y0 + ch + i*ch
        S.append(f'<rect x="{x0}" y="{ry}" width="{label_w}" height="{ch}" fill="#f1f5f9" stroke="#cbd5e1"/>')
        S.append(f'<text x="{x0+label_w/2}" y="{ry+ch/2+5}" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">rank{i}</text>')
        for j in range(4):
            cx = x0 + label_w + j*cw
            v = data[i][j]
            mark = diffmark.get((i, j))
            fill = "#ffffff"
            tcol = "#94a3b8" if v == -1 else "#0f172a"
            if mark == "recv":
                fill = "#dcfce7"
            elif mark == "send":
                fill = "#fee2e2"
            S.append(f'<rect x="{cx}" y="{ry}" width="{cw}" height="{ch}" fill="{fill}" stroke="#cbd5e1"/>')
            S.append(f'<text x="{cx+cw/2}" y="{ry+ch/2+5}" text-anchor="middle" font-size="14" font-weight="bold" fill="{tcol}">{esc(v)}</text>')


# current 表
draw_table(60, 130, "current_expert_maps", current, {})
# updated 表：标差异
diff_up = {(0, 3): "recv", (1, 1): "recv", (0, 1): "send", (1, 3): "send"}
draw_table(660, 130, "updated_expert_maps", updated, diff_up)

# 图例
S.append(f'<rect x="660" y="290" width="22" height="18" fill="#dcfce7" stroke="#cbd5e1"/>')
S.append(f'<text x="690" y="304" text-anchor="start" font-size="12.5" fill="#166534">current==-1 &amp; updated!=-1 → 该卡要「收」</text>')
S.append(f'<rect x="660" y="314" width="22" height="18" fill="#fee2e2" stroke="#cbd5e1"/>')
S.append(f'<text x="690" y="328" text-anchor="start" font-size="12.5" fill="#991b1b">current!=-1 &amp; updated==-1 → 该卡要「发」</text>')

# 差集箭头
S.append(f'<line x1="380" y1="218" x2="650" y2="218" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
S.append(f'<text x="515" y="208" text-anchor="middle" font-size="13" font-weight="bold" fill="#1e293b">torch.where 取差集</text>')

# 推出的 send/recv 对
y_pair = 380
S.append(f'<rect x="60" y="{y_pair}" width="1060" height="170" rx="12" fill="#faf5ff" stroke="#9333ea" stroke-width="2"/>')
S.append(f'<text x="590" y="{y_pair+30}" text-anchor="middle" font-size="16" font-weight="bold" fill="#7e22ce">逐 expert 配对：候选源 = current 里仍持有它的卡，贪心挑第一个</text>')

pairs = [
    ("e3", "rank1", "rank0", "rank0 收 e3 → 源是 current 里持 e3 的 rank1"),
    ("e1", "rank0", "rank1", "rank1 收 e1 → 源是 current 里持 e1 的 rank0"),
]
for k, (eid, src, dst, note) in enumerate(pairs):
    yy = y_pair + 60 + k*48
    S.append(f'<rect x="100" y="{yy}" width="120" height="34" rx="8" fill="#ede9fe" stroke="#a855f7"/>')
    S.append(f'<text x="160" y="{yy+22}" text-anchor="middle" font-size="14" font-weight="bold" fill="#6b21a8">{esc(src)}</text>')
    S.append(f'<line x1="225" y1="{yy+17}" x2="305" y2="{yy+17}" stroke="#9333ea" stroke-width="2.5" marker-end="url(#a)"/>')
    S.append(f'<text x="265" y="{yy+10}" text-anchor="middle" font-size="12" font-weight="bold" fill="#7e22ce">{esc(eid)}</text>')
    S.append(f'<rect x="310" y="{yy}" width="120" height="34" rx="8" fill="#ede9fe" stroke="#a855f7"/>')
    S.append(f'<text x="370" y="{yy+22}" text-anchor="middle" font-size="14" font-weight="bold" fill="#6b21a8">{esc(dst)}</text>')
    S.append(f'<text x="455" y="{yy+22}" text-anchor="start" font-size="13" fill="#334155">{esc(note)}</text>')

S.append(f'<text x="590" y="{y_pair+158}" text-anchor="middle" font-size="12.5" fill="#64748b">收方先 irecv 进预分配 buffer，全部 req.wait() 后再 copy_ 原位替换 expert 槽——先收齐再落地，避免半搬运被前向读到</text>')

S.append('</svg>')
open("map_diff.svg", "w").write("\n".join(S))
print("wrote map_diff.svg")
