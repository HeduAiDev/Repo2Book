#!/usr/bin/env python3
"""fig-ch31-13: 四后端同契约对照——能力矩阵藏在编译分派/回滚能力/终态语义/编译复用四列里。
template: state-table"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "四后端同契约对照：同一份六方法接口，能力矩阵藏在这四列里"
SUBTITLE = "调度器只认六方法接口，不需要知道每家后端内部怎么实现——这正是把契约切成 ABC 的价值"

ROWS = ["xgrammar", "guidance", "outlines", "lm-format-enforcer"]
COLS = ["编译分派支持的形态", "回滚能力", "is_terminated 语义", "编译复用"]

CELLS = {
    "xgrammar": [
        "5 个分支\n（CHOICE 由校验期\n改写并入 GRAMMAR）",
        "上限 = num_speculative_tokens\nmatcher.rollback(n)",
        "缓存标志位\n(accept_tokens/rollback 刷新)",
        "GrammarCompiler\n(cache_enabled=True)",
    ],
    "guidance": [
        "校验期只做 EBNF/\nJSON schema 预检",
        "有偏移：num_tokens\n− rollback_lag(∈{0,1})",
        "EOS 且 matcher.stopped\n→ rollback_lag 置 1",
        "0（无任何编译缓存）",
    ],
    "outlines": [
        "3 种\n(JSON / REGEX / CHOICE)",
        "库内支持，未在\n精简版覆盖",
        "返回上一次的\nis_finished()（延迟一步）",
        "自建 cache，键=\nf\"{vocab._hash}_{regex}\"",
    ],
    "lm-format-enforcer": [
        "4 种\n(JSON/JSON_OBJECT/\nREGEX/CHOICE)",
        "0：max_rollback_tokens>0\n直接 raise ValueError",
        "看 current_tokens_prefix\n末位是不是 EOS",
        "只 lru_cache 了\ntokenizer_data",
    ],
}
ROW_COLOR = {
    "xgrammar": "#16a34a", "guidance": "#7c3aed",
    "outlines": "#2563eb", "lm-format-enforcer": "#dc2626",
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 300, 100, 40, 110, 36
W = PAD * 2 + LABEL_W + COL_W * len(COLS)
H = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 90

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头
for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              f'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" font-weight="bold">{esc(name)}</text>')

# 行
for i, row in enumerate(ROWS):
    ry = row_y[i]
    color = ROW_COLOR[row]
    L.append(f'<rect x="{PAD}" y="{ry+4}" width="{LABEL_W-16}" height="{ROW_H-8}" rx="6" '
              f'fill="{color}" opacity="0.12" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{PAD+(LABEL_W-16)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="{color}">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
        n = len(lines)
        y0 = ry + ROW_H/2 - (n-1)*8.5 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*17:.0f}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(line)}</text>')

# 底部数字条
foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 34
facts = [
    "xgrammar 支持的形态：5 个分支（CHOICE 由校验期改写并入 GRAMMAR）",
    "outlines 支持的形态：3（JSON / REGEX / CHOICE）",
    "lm-format-enforcer 支持的形态：4（JSON/JSON_OBJECT/REGEX/CHOICE）",
    "lm-format-enforcer 的回滚能力：0（max_rollback_tokens>0 直接 raise）",
    "guidance 回滚时的偏移：num_tokens − rollback_lag（EOS 后 lag=1）",
    "guidance 的编译缓存：0（无）",
]
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{W-2*PAD}" height="70" rx="10" fill="#f1f5f9" stroke="#cbd5e1"/>')
for i, f in enumerate(facts):
    fx = PAD + 20 + (i % 2) * (W/2 - PAD - 10)
    fy = foot_y + 22 + (i // 2) * 20
    L.append(f'<text x="{fx}" y="{fy}" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("• " + f)}</text>')

L.append('</svg>')
out = Path("fig-ch31-13-four-backend-matrix.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
