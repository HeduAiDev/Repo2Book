#!/usr/bin/env python3
"""ch29 原理图 · 压缩器怎么把 4 个 token 合成 1 条（ch29-fig-compressor-softpool）

原理示意图（论文级概念图）：不解释代码、不出现任何 vLLM 类名/文件名/行号/章号/
内部产物名。只回答「这一步的合成方式是什么」。

claim（自 figure-request 逐字）：
压缩器怎么把 4 个 token 合成 1 条压缩 KV 条目：每个 token 先投出「内容」与「打分」
两份，当前窗 4 个打分加前窗 4 个打分（共 2m=8 个）一起过 softmax，得到一组和为 1
的权重，再用这 8 个权重对 8 份内容加权求和，得到 1 条压缩条目；相邻两条共享一半输入
（窗重叠），所以每条实际吸收约 8 个 token 的信息、块边界不硬切，序列长度精确压到
1/4。权重是模型学出来的，这是加权求和，不是让模型写摘要。

numbers（逐字取自 figure-request，禁即兴新增）：
  1) 4:1——每 4 个 token 产出 1 条压缩条目；相邻条目共享一半输入，每条吸收约 8 个
     token（2m）的信息
  2) 2m=8 个位置参与同一次 softmax 归一化（当前窗 4 个 + 前窗 4 个）
  3) 同一个压缩器两档：表上记 4（CSA 层）与 128（HCA 层）
provenance：V4 技术报告 arXiv:2606.19348 §2.3.1 Eq.9-12；官方 config compress_ratios
（HF deepseek-ai/DeepSeek-V4-Flash，2026-09-16 抓取）。

配色走 book/cartography/l0_common.py 的角色常量（青 = KV/内容，橙 = 分数/权重等辅助
量）。坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 780
MX, BXR = 56, 1604

# ---------------- 语义配色 ----------------
C_PREV_F, C_PREV_S = '#f1f5f9', '#94a3b8'    # 前窗（上一窗）的位置
C_CUR_F, C_CUR_S = '#e0f2fe', '#0284c7'      # 当前窗（本次要合成的那 4 个）
C_CONT_F, C_CONT_S = lc.C_KV_F, lc.C_KV_S    # 内容（待合并的 KV 原料）
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'    # 打分（用来算权重的分数）
C_ENTRY_F, C_ENTRY_S = lc.C_KV_S, '#155e75'  # 压缩条目（合并结果）

EXTRA_DEFS = ('<defs>'
              '<marker id="wm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SCOR_S}"/></marker>'
              '<marker id="cy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_ENTRY_S}"/></marker>'
              '<marker id="tk" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="5.5" '
              f'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '</defs>')

# ---------------- 布局常量（坐标全部由此计算） ----------------
BX, CW, NC = 288, 84, 8          # 8 个位置的横向格子：左缘 / 格距 / 格数
BW = CW * NC                     # 672
BRX = BX + BW                    # 960（带右缘）
GAP = 4                          # 格内缩，格与格之间的缝
TY, TH = 104, 42                 # token 带
CY, CH = 180, 42                 # 内容带
RY, RH = 271, 6                  # 权重轨道（细长实心条）
SY, SH = 320, 42                 # 打分带
SBX0, SBX1, SBY0, SBY1 = 356, 892, 400, 476     # softmax 盒
EBX0, EBY0, EBX1, EBY1 = 1120, 171, 1470, 231   # 压缩条目条
TRUNK = 254                      # 「投出两份」分叉的竖干 x
PAN = (56, 500, 1548, 140)       # 底部「窗重叠」面板


def cellx(i):
    return BX + i * CW


def cellcx(i):
    return BX + (i + 0.5) * CW


# ---------------- 标题区 ----------------
lc.text(MX, 34, '压缩器怎么把 4 个 token 合成 1 条：一次学习加权的软池化', 16, lc.C_TXT,
        'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '每个 token 投出两份——内容与打分；8 个位置的打分一起归一，得到的 8 个权重再回到 8 份内容上加权求和',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')

# ---------------- 图例（三种以上语义色，必须给） ----------------
LGX, LGY, LGW, LGH = 1000, 90, 604, 74
lc.rect(LGX, LGY, LGW, LGH, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(LGX + 14, LGY + 18, '图例', 9.5, lc.C_MUTE, 'start', True, maxw=40, tag='lg:t')
LEG = [('prev', '前窗位置（上一窗 4 个）'), ('cur', '当前窗位置（本次合成 4 个）'),
       ('cont', '内容：待合并的 KV 原料'), ('scor', '打分：用来算权重的分数'),
       ('wt', '权重：softmax 归一后的 8 个'), ('entry', '压缩条目：合并出的 1 条')]
FILLS = {'prev': (C_PREV_F, C_PREV_S), 'cur': (C_CUR_F, C_CUR_S),
         'cont': (C_CONT_F, C_CONT_S), 'scor': (C_SCOR_F, C_SCOR_S),
         'wt': (C_SCOR_S, C_SCOR_S), 'entry': (C_ENTRY_F, C_ENTRY_S)}
COLS, ROWS = 3, 2
for k, (kind, lab) in enumerate(LEG):
    cx = LGX + 16 + (k % COLS) * 198
    cy = LGY + 34 + (k // COLS) * 22
    f, s = FILLS[kind]
    lc.rect(cx, cy - 10, 18, 12, f, s, rx=2, sw=1.1)
    lc.text(cx + 24, cy, lab, 8.5, '#334155', 'start', maxw=172, tag='lg:%s' % kind)

# ---------------- ① token 带（8 个位置：前窗 4 + 当前窗 4） ----------------
lc.text(MX, TY + 16, '输入 token', 11, lc.C_TXT, 'start', True, maxw=170, tag='tk:t')
lc.text(MX, TY + 32, '一行 8 个位置', 8.5, lc.C_MUTE, 'start', maxw=170, tag='tk:s')
for half, (lab, f, s) in enumerate([('前窗 m=4', C_PREV_F, C_PREV_S),
                                    ('当前窗 m=4', C_CUR_F, C_CUR_S)]):
    c0 = half * 4
    lc.text(BX + (c0 + 2) * CW, TY - 12, lab, 10, '#334155' if half == 0 else C_CUR_S,
            'middle', True, maxw=4 * CW - 8, tag='tk:h%d' % half)
    for i in range(c0, c0 + 4):
        lc.rect(cellx(i) + GAP / 2, TY, CW - GAP, TH, f, s, rx=3, sw=1.3)

# ---------------- ② 两条并行细带：内容 / 打分 ----------------
# 内容带（上带）
lc.text(MX, CY + 14, '内容', 11, lc.C_TXT, 'start', True, maxw=170, tag='ct:t')
lc.text(MX, CY + 28, '每个 token 投出的待合并内容', 8.5, lc.C_MUTE, 'start', maxw=170, tag='ct:s')
lc.text(MX, CY + 40, 'C^a / C^b', 8, lc.C_FAINT, 'start', maxw=170, tag='ct:n')
for i, sub in [(0, 'C^b'), (4, 'C^a')]:
    lc.text(cellcx(i + 1.5), CY - 8, sub, 8.5, C_CONT_S, 'middle', True, maxw=3 * CW, tag='ct:h%d' % i)
for i in range(NC):
    lc.rect(cellx(i) + GAP / 2, CY, CW - GAP, CH, C_CONT_F, C_CONT_S, rx=3, sw=1.3)
    lc.text(cellcx(i), CY + CH / 2 + 3.5, 'C', 11, C_CONT_S, 'middle', True, tag='ct:c%d' % i)

# 打分带（下带）
lc.text(MX, SY + 14, '打分', 11, lc.C_TXT, 'start', True, maxw=170, tag='sc:t')
lc.text(MX, SY + 28, '用来算权重的分数', 8.5, lc.C_MUTE, 'start', maxw=170, tag='sc:s')
lc.text(MX, SY + 40, 'Z^a / Z^b', 8, lc.C_FAINT, 'start', maxw=170, tag='sc:n')
for i, sub in [(0, 'Z^b'), (4, 'Z^a')]:
    lc.text(cellcx(i + 1.5), SY - 8, sub, 8.5, C_SCOR_S, 'middle', True, maxw=3 * CW, tag='sc:h%d' % i)
for i in range(NC):
    lc.rect(cellx(i) + GAP / 2, SY, CW - GAP, SH, C_SCOR_F, C_SCOR_S, rx=3, sw=1.3)
    lc.text(cellcx(i), SY + SH / 2 + 3.5, 'Z', 11, C_SCOR_S, 'middle', True, tag='sc:c%d' % i)

# ---------------- ③ 分叉：每个 token 投出两份 ----------------
lc.parrow([(BX, TY + TH / 2), (TRUNK, TY + TH / 2), (TRUNK, CY + CH / 2)], '#64748b',
          1.6, marker=None)
lc.parrow([(TRUNK, CY + CH / 2), (BX, CY + CH / 2)], '#64748b', 1.6, marker='tk')
lc.parrow([(TRUNK, CY + CH / 2), (TRUNK, SY + SH / 2), (BX, SY + SH / 2)], '#64748b',
          1.6, marker='tk')
lc.text(TRUNK + 8, TY + TH / 2 + 30, '每个 token', 8.5, lc.C_MUTE, 'start', maxw=120, tag='fk:l0')
lc.text(TRUNK + 8, TY + TH / 2 + 44, '各投出两份', 8.5, lc.C_MUTE, 'start', maxw=120, tag='fk:l1')

# ---------------- ④ softmax 盒 + 权重轨道 ----------------
# 打分带 → softmax 盒
lc.parrow([(BX + BW / 2, SY + SH), (BX + BW / 2, SBY0)], C_SCOR_S, 2.2, marker='wm')
lc.rect(SBX0, SBY0, SBX1 - SBX0, SBY1 - SBY0, '#ffffff', lc.C_KV_S, rx=8, sw=1.8)
lc.text((SBX0 + SBX1) / 2, SBY0 + 20, 'softmax', 13, C_ENTRY_S, 'middle', True, tag='sm:t')
lc.text((SBX0 + SBX1) / 2, SBY0 + 38, '8 个位置一起归一（当前窗 4 + 前窗 4 = 2m = 8）',
        10, lc.C_TXT, 'middle', maxw=SBX1 - SBX0 - 30, tag='sm:l0')
lc.text((SBX0 + SBX1) / 2, SBY0 + 54, '得到一组权重：权重和 = 1', 10, C_SCOR_S, 'middle', True,
        maxw=SBX1 - SBX0 - 30, tag='sm:l1')
lc.text(SBX0 + 12, SBY0 + 68, '（打分带里 8 个位置的分数，一起竞争出 8 个权重）', 8.5, lc.C_FAINT,
        'start', maxw=SBX1 - SBX0 - 24, tag='sm:l2')

# 权重轨道（细长实心条）+ 8 根权重箭头（回到 8 份内容上）
for i in range(NC):
    lc.seg(cellcx(i), RY + RH / 2, cellcx(i), CY + CH, C_SCOR_S, 2.0, marker='wm')
lc.rect(BX, RY, BW, RH, C_SCOR_S, C_SCOR_S, rx=2, sw=1)
# softmax → 权重的回程总线
lc.parrow([(SBX1, (SBY0 + SBY1) / 2), (1000, (SBY0 + SBY1) / 2), (1000, RY + RH / 2),
           (BRX, RY + RH / 2)], C_SCOR_S, 2.2, marker='wm')
lc.text(1012, (SBY0 + SBY1) / 2 - 12, '8 个权重', 9.5, C_SCOR_S, 'start', True, maxw=150, tag='wt:t')
lc.text(1012, (SBY0 + SBY1) / 2 + 4, '按位乘回', 8.5, lc.C_MUTE, 'start', maxw=150, tag='wt:s')
lc.text(1012, (SBY0 + SBY1) / 2 + 18, '8 份内容', 8.5, lc.C_MUTE, 'start', maxw=150, tag='wt:s2')

# ---------------- ⑤ 合并成 1 条压缩条目 ----------------
lc.parrow([(BRX, CY + CH / 2), (EBX0, CY + CH / 2)], C_ENTRY_S, 3.0, marker='cy')
lc.text((BRX + EBX0) / 2, CY + CH / 2 - 14, '8 份内容按这 8 个权重相加', 9.5, C_ENTRY_S,
        'middle', True, maxw=150, tag='mg:l')
lc.rect(EBX0, EBY0, EBX1 - EBX0, EBY1 - EBY0, C_ENTRY_F, C_ENTRY_S, rx=5, sw=1.6)
lc.text((EBX0 + EBX1) / 2, (EBY0 + EBY1) / 2 + 5, '1 条压缩 KV 条目', 13, '#ffffff', 'middle',
        True, maxw=EBX1 - EBX0 - 20, tag='en:t')
lc.text(EBX1 + 22, (EBY0 + EBY1) / 2 - 2, '4:1', 20, C_ENTRY_S, 'start', True, maxw=110, tag='en:r')
lc.text(EBX1 + 22, (EBY0 + EBY1) / 2 + 18, '每 4 个 token 出 1 条', 8.5, lc.C_MUTE, 'start',
        maxw=150, tag='en:r2')

# ---------------- ⑥ 底部面板：窗重叠 ----------------
PX, PY, PW, PH = PAN
lc.rect(PX, PY, PW, PH, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(PX + 16, PY + 24, '窗重叠：相邻两条压缩条目共享一半输入', 10.5, lc.C_TXT, 'start', True,
        maxw=240, tag='pn:t')
RTY, RTH = PY + 34, 28                                   # 面板内 12 格标尺（与前 8 格同一网格）
for i in range(NC + 4):
    f, s, dash = (C_PREV_F, C_PREV_S, False) if i < 4 else (
        (C_CUR_F, C_CUR_S, False) if i < NC else ('#ffffff', lc.C_FAINT, True))
    lc.rect(cellx(i) + GAP / 2, RTY, CW - GAP, RTH, f, s, rx=3, sw=1.3, dash=dash)
lc.text(cellcx(NC + 1.5), RTY - 6, '下一批 4 个 token', 8.5, lc.C_FAINT, 'middle',
        maxw=4 * CW - 10, tag='pn:nxt')
# 共享区高亮（先铺，压住之后画的括号）
lc.rect(cellx(4), RTY + 34, 4 * CW, 40, '#cffafe', C_CONT_S, rx=5, sw=1.2, dash=True)
# 两条错开 4 格的括号
BRY1, BRY2 = RTY + 42, RTY + 64
for row, (s0, n, y, lab, col) in enumerate([
        (0, NC, BRY1, '条目 i 的窗：前窗 4 + 当前窗 4 = 8 个位置', C_ENTRY_S),
        (4, NC, BRY2, '条目 i+1 的窗：右移 4 格，共享这 4 个位置', C_CONT_S)]):
    x0, x1 = cellx(s0), cellx(s0 + n)
    lc.parrow([(x0, y - 8), (x0, y), (x1, y), (x1, y - 8)], col, 1.6, marker=None)
    lc.text(x1 + 12, y + 4, lab, 8.5, col, 'start', maxw=300, tag='pn:br%d' % row)
lc.text(cellcx(6), RTY + 34 + 40 + 14, '共享 4 格', 9, C_CONT_S, 'middle', True, maxw=4 * CW,
        tag='pn:sh')
lc.text(1330, PY + 24, '每条压缩条目实际吸收约 8 个 token（2m）的信息', 9, '#334155', 'start',
        True, maxw=270, tag='pn:a0')
lc.text(1330, PY + 40, '块边界不硬切；序列长度精确压到 1/4', 8.5, lc.C_MUTE, 'start', maxw=270,
        tag='pn:a1')

# ---------------- ⑦ 一行小注（框内一行、不加粗） ----------------
NBX, NBY, NBW, NBH = MX, 656, BXR - MX, 48
lc.rect(NBX, NBY, NBW, NBH, '#f1f5f9', lc.C_MUTE, rx=8, sw=1.2)
lc.text(NBX + NBW / 2, NBY + 29, '权重由模型学出来：这是学习加权的软池化，不是让模型写摘要。',
        11.5, lc.C_TXT, 'middle', maxw=NBW - 30, tag='note')

# ---------------- 页脚 ----------------
lc.text(MX, 730, '同一个压缩器两档：表上记 4（CSA 层）与 128（HCA 层）——本图取 4:1 那一档为例，128:1 那一档合并方式相同。',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 748, '依据 DeepSeek-V4 技术报告 §2.3.1 压缩器（arXiv:2606.19348，Eq.9-12）；压缩比取值出自官方 config 的逐层表。',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-compressor-softpool.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
