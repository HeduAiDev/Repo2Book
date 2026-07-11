#!/usr/bin/env python3
"""flow 模板:DDTree best-first 堆按前缀概率非增序弹出,预算 B=4 内构出最优草稿树。
数字来自 explainer/traces/ddtree.json(mechanism ddtree-tree-verification)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "best-first 堆按前缀概率非增序弹出,预算 B=4 内构出可证明最优的草稿树"
SUBTITLE = "弹出序:[1] -> [1,1] -> [2] -> [1,1,1]——树 surrogate 累计 1.756,高于单轨迹链 1.356(约 30%)"

STEPS = [
    ("[1]", "0.6", "0.6"),
    ("[1, 1]", "0.42", "1.02"),
    ("[2]", "0.4", "1.42"),
    ("[1, 1, 1]", "0.336", "1.756"),
]

BOX_W, BOX_H, GAP_X, PAD, TOP = 190, 88, 56, 46, 130
w = PAD * 2 + len(STEPS) * BOX_W + (len(STEPS) - 1) * GAP_X
h = TOP + BOX_H + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

x = PAD
for i, (rank_tuple, prob, cum) in enumerate(STEPS):
    highlight = (i == len(STEPS) - 1)
    fill = "#dcfce7" if highlight else "#dbeafe"
    stroke = "#16a34a" if highlight else "#2563eb"
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">pop {i+1}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+50}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="16" font-weight="bold" fill="#0f172a">{esc(rank_tuple)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+72}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">该前缀概率 {prob}</text>')
    # cumulative surrogate below box
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+BOX_H+26}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{stroke}">累计 surrogate = {cum}</text>')
    if i < len(STEPS) - 1:
        ax1 = x + BOX_W
        ax2 = ax1 + GAP_X
        ay = TOP + BOX_H/2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                  'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
    x += BOX_W + GAP_X

foot_y = TOP + BOX_H + 70
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">堆每次弹最高分前缀,推入同层下一 rank 的兄弟与下探一层 rank1 的孩子,二者得分都 &lt;= 已弹出前缀——弹出序即全局非增序。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">与暴力枚举 top-4 前缀集合完全一致(best_first_matches_bruteforce=true)——验证仍是 target 一次前向(祖先-only 掩码)。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+44}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">对比:vanilla 单轨迹只走 [1],[1,1],[1,1,1] 三步,累计 surrogate 仅 1.356——树验证多花的只是把预算铺成 4 个节点。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ddtree-best-first.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
