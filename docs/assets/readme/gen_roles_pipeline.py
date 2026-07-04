#!/usr/bin/env python3
"""README 图 1:单章流水线的角色接力。
claim:一章书 = 8 角色按阶段接力——工件单向流动、三处有界回环(≤3 轮)、任何阶段可拉闸升级 Lead。
数字出处:回环上限/评审维度均见 .claude/workflows/chapter-pipeline.js。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (阶段, 角色, 角色色, 产物行1, 产物行2)
STAGES = [
    ("Dossier",    "analyst",     "#d946ef", "dossier.json", "机制账本+真源码片段"),
    ("Implement",  "implementer", "#3b82f6", "精简版(只删不增)", "primer:论文参考实现"),
    ("Test",       "tester",      "#eab308", "test-report", "复现真实行为,非自洽"),
    ("Explain",    "explainer",   "#f97316", "explainer.json", "经运行验证的数值轨迹"),
    ("Illustrate", "illustrator", "#a855f7", "diagrams+manifest", "渲染后亲眼看+盲审"),
    ("Write",      "writer",      "#22c55e", "chapter.md", "拿素材自由叙事"),
    ("Review",     "reviewer",    "#ef4444", "review-report", "4 门控维+haiku 读者"),
    ("Archive",    "archivist",   "#06b6d4", "bible / trace", "术语·伏笔·figures 登记"),
]
# 有界回环:(源阶段下标, 目标阶段下标, 标签)
LOOPS = [(2, 1, "REJECTED 回修 ≤3 轮"), (6, 5, "REVISE 定点改 ≤3 轮")]
SELF_LOOP = (4, "盲审 FAIL 重绘 ≤3 轮")

BOX_W, BOX_H, GAP, PAD, TOP = 168, 108, 18, 30, 132
n = len(STAGES)
w = PAD * 2 + n * BOX_W + (n - 1) * GAP
h = TOP + BOX_H + 168

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="loop" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker>'
     '<marker id="escm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

# 标题 + Lead 车道
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="19" font-weight="bold" '
         f'fill="#0f172a">一章书怎么造:8 角色按阶段接力,工件单向流动</text>')
lead_y = 46
L.append(f'<rect x="{PAD}" y="{lead_y}" width="{w - PAD * 2}" height="34" rx="8" '
         'fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="5,3"/>')
L.append(f'<text x="{w / 2}" y="{lead_y + 22}" text-anchor="middle" font-family="sans-serif" '
         'font-size="13" fill="#334155">Team Lead(主会话):Workflow 发车 → 后台监控 → '
         '逃生舱处理 → resumeFromRunId 断点续跑</text>')

# 阶段盒
X = []
for i, (stage, role, color, art1, art2) in enumerate(STAGES):
    x = PAD + i * (BOX_W + GAP)
    X.append(x)
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
             f'fill="white" stroke="{color}" stroke-width="2"/>')
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="26" rx="10" fill="{color}"/>')
    L.append(f'<rect x="{x}" y="{TOP + 13}" width="{BOX_W}" height="13" fill="{color}"/>')
    L.append(f'<text x="{x + BOX_W / 2}" y="{TOP + 18}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="13" font-weight="bold" fill="white">{esc(role)}</text>')
    L.append(f'<text x="{x + BOX_W / 2}" y="{TOP + 46}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#0f172a">{esc(stage)}</text>')
    L.append(f'<text x="{x + BOX_W / 2}" y="{TOP + 68}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(art1)}</text>')
    L.append(f'<text x="{x + BOX_W / 2}" y="{TOP + 86}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" fill="#64748b">{esc(art2)}</text>')
    if i < n - 1:
        L.append(f'<line x1="{x + BOX_W}" y1="{TOP + BOX_H / 2}" x2="{x + BOX_W + GAP - 3}" '
                 f'y2="{TOP + BOX_H / 2}" stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# 有界回环(下方折线)
loop_y = TOP + BOX_H + 26
def polyline(pts, marker):
    """三段折线:仅末段带箭头,端点全部取自框边——几何 linter 的 path 解析器不认 V/H,用 line 三连。"""
    for k in range(len(pts) - 1):
        (ax, ay), (bx2, by2) = pts[k], pts[k + 1]
        m = f' marker-end="url(#{marker})"' if k == len(pts) - 2 else ''
        L.append(f'<line x1="{ax}" y1="{ay}" x2="{bx2}" y2="{by2}" '
                 f'stroke="#d97706" stroke-width="1.6" fill="none"{m}/>')


for si, ti, label in LOOPS:
    x1 = X[si] + BOX_W * 0.35
    x2 = X[ti] + BOX_W * 0.65
    polyline([(x1, TOP + BOX_H), (x1, loop_y), (x2, loop_y), (x2, TOP + BOX_H + 3)], 'loop')
    L.append(f'<text x="{(x1 + x2) / 2}" y="{loop_y + 15}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11.5" fill="#b45309">{esc(label)}</text>')
# 盲审自环(Illustrate 顶部)
bx = X[SELF_LOOP[0]]
polyline([(bx + BOX_W * 0.7, TOP), (bx + BOX_W * 0.7, TOP - 18),
          (bx + BOX_W * 0.3, TOP - 18), (bx + BOX_W * 0.3, TOP - 3)], 'loop')
L.append(f'<text x="{bx + BOX_W / 2}" y="{TOP - 26}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11.5" fill="#b45309">{esc(SELF_LOOP[1])}</text>')

# 逃生舱带
esc_y = loop_y + 34
L.append(f'<rect x="{PAD}" y="{esc_y}" width="{w - PAD * 2}" height="36" rx="8" '
         'fill="#fef2f2" stroke="#dc2626" stroke-dasharray="5,3"/>')
L.append(f'<text x="{w / 2}" y="{esc_y + 23}" text-anchor="middle" font-family="sans-serif" '
         'font-size="13" fill="#b91c1c">逃生舱:任一阶段 BLOCKED / agent 失败 → 立即早停,升级 Lead'
         '(宁拉闸,不硬着头皮做错)</text>')
for i in (0, 3, 5, 7):
    cx = X[i] + BOX_W / 2
    L.append(f'<line x1="{cx}" y1="{TOP + BOX_H}" x2="{cx}" y2="{esc_y - 3}" '
             'stroke="#dc2626" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#escm)"/>')

# 图例
leg_y = esc_y + 56
items = [("#334155", "工件交接", False), ("#d97706", "有界回环(≤3 轮)", False), ("#dc2626", "逃生舱(升级 Lead)", True)]
lx = PAD
for color, label, dashed in items:
    dash = ' stroke-dasharray="4,3"' if dashed else ''
    L.append(f'<line x1="{lx}" y1="{leg_y}" x2="{lx + 34}" y2="{leg_y}" stroke="{color}" '
             f'stroke-width="2"{dash}/>')
    L.append(f'<text x="{lx + 42}" y="{leg_y + 4}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(label)}</text>')
    lx += 60 + len(label) * 12

L.append('</svg>')
out = __file__.replace('gen_roles_pipeline.py', 'roles-pipeline.svg')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print(f'wrote {out}')
