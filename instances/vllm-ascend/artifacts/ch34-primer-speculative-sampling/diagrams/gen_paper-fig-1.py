#!/usr/bin/env python3
"""论文精髓图重绘 —— arXiv:2211.17192 Figure 1(§0 开篇例图)。
真实训练的 6M 草稿模型 / 97M 目标模型,在 lm1b 上跑出的一句生成实例。
每行 = 目标模型一次前向;本行新增的草稿 token 逐个标色:
绿 = 被接受,红(删除线) = 被拒绝,蓝 = 目标模型给出的修正(或全部接受时的兜底)。
黑色 = 此前几趟已经落定、不再改变的前缀。
9 行(=9 次目标前向)、每行接受/拒绝/修正的分段与颜色经像素级核对原图
(ar5iv assets/figure1.png 的分段坐标)复原;为可读性按完整单词合并展示
(原图按 BPE 子词切分,论文正文称共 38 个 token —— 本图按词计,末行脚注同时
标出两个数字,不混为一谈)。
provenance = 论文原图本身(key_figure 重绘,豁免 explainer/spec.numbers 通道)。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


BLACK = "#334155"
GREEN = "#16a34a"
RED = "#dc2626"
BLUE = "#2563eb"

# 每趟:本趟新增的绿色(接受)token 列表 + 被拒的红色 token(或 None)+ 修正/兜底的蓝色 token(或 None)
ROUNDS = [
    {"green": ["japan", "'", "s", "benchmark"], "reject": "bond", "correct": "n"},
    {"green": ["ikkei", "22"], "reject": "7", "correct": "5"},
    {"green": ["index", "rose", "22"], "reject": "1", "correct": "6"},
    {"green": [".", "69"], "reject": "7", "correct": "points"},
    {"green": [",", "or"], "reject": "0", "correct": "1"},
    {"green": [".", "5", "percent", ",", "to", "10", ",", "98"], "reject": "5", "correct": "9"},
    {"green": [".", "79"], "reject": "1", "correct": "in"},
    {"green": [], "reject": "tokyo", "correct": "late"},
    {"green": ["morning", "trading", ".", "[END]"], "reject": None, "correct": None},
]
INITIAL_CONTEXT = ["[START]"]

FONT_SIZE = 11
TOK_GAP = 7


def _char_w(c):
    if c in "iIl.,'!:;|jt ":
        return 4.1
    if c in "mMWw@":
        return 9.3
    if c.isupper():
        return 7.0
    if c.isdigit():
        return 6.4
    if c in "[]":
        return 4.9
    return 6.0


def tok_w(s):
    return max(14.0, sum(_char_w(c) for c in s) + 7.5)


# ---- 第一遍:纯模拟,只求每行 token 序列 + 每行像素宽度 + 累计计数(不出图) ----
kept = list(INITIAL_CONTEXT)
rows = []  # 每行: [(text, color, strike), ...]，以及 (new_count, cumulative_count)
for rnd in ROUNDS:
    row_tokens = [(t, BLACK, False) for t in kept]
    row_tokens += [(t, GREEN, False) for t in rnd["green"]]
    if rnd["reject"] is not None:
        row_tokens.append((rnd["reject"], RED, True))
    if rnd["correct"] is not None:
        row_tokens.append((rnd["correct"], BLUE, False))
    new_count = len(rnd["green"]) + (1 if rnd["correct"] else 0)
    kept = kept + rnd["green"] + ([rnd["correct"]] if rnd["correct"] else [])
    rows.append({"tokens": row_tokens, "new": new_count, "cum": len(kept)})

row_widths = []
for r in rows:
    x = 0.0
    for t, _, _ in r["tokens"]:
        x += tok_w(t) + TOK_GAP
    row_widths.append(x)
max_row_w = max(row_widths)

# ---- 画布几何 ----
PAD = 20
LABEL_W = 116
TOKENS_X = PAD + LABEL_W
BADGE_W = 126
w = TOKENS_X + max_row_w + BADGE_W + PAD

TOP = 96
LEGEND_H = 30
ROW_H = 40
ROW_PITCH = ROW_H + 6
rows_y0 = TOP + LEGEND_H + 14
h = rows_y0 + ROW_PITCH * len(rows) + 78

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">']
L.append(f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>')

TITLE = "arXiv:2211.17192 Fig.1 重绘：6M 草稿 / 97M 目标模型的真实生成实例"
SUBTITLE = "每行 = 目标模型一次前向；绿 = 草稿被接受、红(删除线) = 草稿被拒绝、蓝 = 目标模型给出的修正/兜底 token"
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+21}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# 图例
legend_y = PAD + 44
legend_items = [
    ("已确定前缀", BLACK, False),
    ("接受", GREEN, False),
    ("拒绝", RED, True),
    ("修正 / 兜底", BLUE, False),
]
lx = PAD
for label, color, strike in legend_items:
    L.append(f'<rect x="{lx}" y="{legend_y-12}" width="14" height="14" rx="3" fill="{color}"/>')
    tx = lx + 20
    L.append(f'<text x="{tx}" y="{legend_y}" font-family="sans-serif" font-size="12" '
              f'fill="{color}">{esc(label)}</text>')
    tw_lab = 7.0 * len(label) + 26
    if strike:
        L.append(f'<line x1="{tx}" y1="{legend_y-4}" x2="{tx+tw_lab-26}" y2="{legend_y-4}" '
                  f'stroke="{RED}" stroke-width="1.5"/>')
    lx = tx + tw_lab

# 逐行
for i, r in enumerate(rows):
    ry = rows_y0 + i * ROW_PITCH
    baseline = ry + ROW_H / 2 + 5
    label = f"第 {i+1} 次目标前向"
    L.append(f'<text x="{PAD}" y="{baseline}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    x = TOKENS_X
    for text, color, strike in r["tokens"]:
        tw_ = tok_w(text)
        L.append(f'<text x="{x:.1f}" y="{baseline:.1f}" font-family="sans-serif" '
                  f'font-size="{FONT_SIZE}" fill="{color}">{esc(text)}</text>')
        if strike:
            L.append(f'<line x1="{x-3:.1f}" y1="{baseline-5:.1f}" x2="{x+tw_-8:.1f}" '
                      f'y2="{baseline-5:.1f}" stroke="{RED}" stroke-width="1.6"/>')
        x += tw_ + TOK_GAP
    badge_x = TOKENS_X + max_row_w + 16
    badge = f"本趟新增 +{r['new']}"
    L.append(f'<rect x="{badge_x:.1f}" y="{ry+4}" width="{BADGE_W-24:.1f}" height="{ROW_H-8}" '
              f'rx="6" fill="#fef3c7" stroke="#b45309" stroke-width="1.2"/>')
    L.append(f'<text x="{badge_x+(BADGE_W-24)/2:.1f}" y="{baseline:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="#b45309">{esc(badge)}</text>')
    # 行分隔线
    if i < len(rows) - 1:
        L.append(f'<line x1="{PAD}" y1="{ry+ROW_H+3}" x2="{w-PAD:.1f}" y2="{ry+ROW_H+3}" '
                  f'stroke="#f1f5f9" stroke-width="1"/>')

final_words = rows[-1]["cum"]
foot_y0 = rows_y0 + ROW_PITCH * len(rows) + 26
L.append(f'<text x="{PAD}" y="{foot_y0:.1f}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">论文原文：该例句按 BPE 子词计 38 个 token(含起始符)，'
          f'仅用 9 次目标模型前向生成 —— 本图为可读性按完整单词合并展示，共 {final_words} 个词级单元</text>')
L.append(f'<text x="{PAD}" y="{foot_y0+20:.1f}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">收益从不均匀：第 6 趟连中 8 个草稿一次性净增 9 个词，'
          f'第 8 趟草稿全军覆没仍靠兜底净增 1 个 —— 这正是第一节“至少产出 1 个 token”下界的实例</text>')
L.append('</svg>')

out = Path(__file__).with_name("paper-fig-1.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w:.0f}x{h:.0f}  aspect={w/h:.2f}")
