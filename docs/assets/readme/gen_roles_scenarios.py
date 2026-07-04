#!/usr/bin/env python3
"""README 图 2:同一批角色的三种编排场景。
claim:新章走全流水线(码章/primer 只换实现关卡与门禁),回修外科式免修早退,审计只读不动刀。
出处:.claude/workflows/{chapter-pipeline,chapter-retrofit,book-gap-audit}.js。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


ROLE_COLOR = {
    "analyst": "#d946ef", "implementer": "#3b82f6", "tester": "#eab308",
    "explainer": "#f97316", "illustrator": "#a855f7", "writer": "#22c55e",
    "reviewer": "#ef4444", "archivist": "#06b6d4", "Lead": "#64748b",
}
# 每场景:(标题, 副标题, [(角色, 动作)], 尾注)
BANDS = [
    ("场景 A · 新章生产(chapter-pipeline)",
     "全 8 角色接力(见上图)。码章与 primer 原理章只换两处:",
     [("implementer", "码章:减法精简版·lint_fidelity"),
      ("implementer", "primer:论文参考实现·lint_paper_grounding"),
      ("reviewer", "primer 维度 0 换 paper-fidelity(对照论文包)")],
     "其余 6 阶段两种章完全同构——豁免与替代门禁成对出现"),
    ("场景 B · 存量回修(chapter-retrofit)",
     "外科式:只动图和算法段,禁整章重写",
     [("reviewer", "体检:逐机制评深度/看图,免修即早退"),
      ("explainer", "只对 flagged 机制补素材"),
      ("illustrator", "补缺图/换错图+盲审"),
      ("writer", "只许定点 Edit"),
      ("archivist", "登记 figures/trace")],
     "健康章零成本退出;评审缩编为 2 门控维"),
    ("场景 C · 全书 gap 审计(book-gap-audit)",
     "只读不动刀:概念首现须「本章建立/前章已立/有指路」三者居一",
     [("reviewer", "每章 1 审计员,并行扫全书"),
      ("reviewer", "……×N 章"),
      ("Lead", "拿 cliff/bump 清单 triage:回修/立原理章/接受")],
     "每 Part 收尾跑;原理篇写完后重跑作闭环验收"),
]

BAND_H, PAD, TOP, CHIP_H = 144, 30, 56, 56
w = 1180
h = TOP + len(BANDS) * (BAND_H + 16) + 8

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="32" font-family="sans-serif" font-size="19" font-weight="bold" '
     'fill="#0f172a">同一批角色,三种编排:生产全流水线 · 回修外科式 · 审计只读</text>']

for bi, (title, subtitle, chips, foot) in enumerate(BANDS):
    by = TOP + bi * (BAND_H + 16)
    L.append(f'<rect x="{PAD}" y="{by}" width="{w - PAD * 2}" height="{BAND_H}" rx="10" '
             'fill="#f8fafc" stroke="#cbd5e1"/>')
    L.append(f'<text x="{PAD + 14}" y="{by + 24}" font-family="sans-serif" font-size="14.5" '
             f'font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<text x="{PAD + 14}" y="{by + 44}" font-family="sans-serif" font-size="12" '
             f'fill="#64748b">{esc(subtitle)}</text>')
    cw = (w - PAD * 2 - 28 - (len(chips) - 1) * 34) / len(chips)
    cy = by + 54
    for ci, (role, action) in enumerate(chips):
        cx = PAD + 14 + ci * (cw + 34)
        color = ROLE_COLOR[role]
        L.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{CHIP_H}" rx="8" '
                 f'fill="white" stroke="{color}" stroke-width="1.8"/>')
        L.append(f'<text x="{cx + 10}" y="{cy + 20}" font-family="sans-serif" font-size="12" '
                 f'font-weight="bold" fill="{color}">{esc(role)}</text>')
        L.append(f'<text x="{cx + 10}" y="{cy + 40}" font-family="sans-serif" font-size="11.5" '
                 f'fill="#334155">{esc(action)}</text>')
        if ci < len(chips) - 1 and bi != 0:
            L.append(f'<line x1="{cx + cw}" y1="{cy + CHIP_H / 2}" x2="{cx + cw + 31}" '
                     f'y2="{cy + CHIP_H / 2}" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
        elif ci < len(chips) - 1:
            sep = 'vs' if ci == 0 else '+'
            L.append(f'<text x="{cx + cw + 16}" y="{cy + CHIP_H / 2 + 4}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="14" fill="#94a3b8">{sep}</text>')
    L.append(f'<text x="{w - PAD - 14}" y="{by + BAND_H - 12}" text-anchor="end" '
             f'font-family="sans-serif" font-size="11.5" fill="#64748b">{esc(foot)}</text>')

L.append('</svg>')
out = __file__.replace('gen_roles_scenarios.py', 'roles-scenarios.svg')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print(f'wrote {out}')
