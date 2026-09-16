#!/usr/bin/env python3
"""ch28 机制图 · CSA 软池化的四档阶梯（ch28-fig-csa-ladder）

claim：位置偏置 B 是『窗内第几格』唯一的信号源：摘掉它，等分分数上的 softmax 退化成等权
平均（权重 0.25×4、输出就是算术平均）；每补一件（B → 分数投影 Z → 2m 交叠跨窗归一），
输出就位移一次（两两差 0.249294 / 1.218941 / 6.500476）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  摘掉 B、窗内 Z 相同时：权重 = [0.25, 0.25, 0.25, 0.25]、加权和 = [4.0, 6.0] = 算术平均
  加上 B_a = [0.5, 0.4, 0.3, 0.2] 后：权重 = [0.288651, 0.261183, 0.236328, 0.213838]、
        输出 = [3.750706, 5.750706]（与平均差 [0.249294, 0.249294]）
  四档：sec1 [4.0, 6.0]｜sec2 [3.750706, 5.750706]｜sec3 [3.135686, 6.969646]｜
        sec4 [0.576481, 0.469171]
  sec4 的 2m 窗权重逐列和 = [1.0, 1.0]

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩缓存侧）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 806
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_BAR_F = '#cffafe'

EXTRA_DEFS = ('<defs>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '同一组输入、四档实现：每加一件输出就位移一次', 17, C_TXT, 'start', True, maxw=580,
        tag='title')
lc.text(MX, 56, '从「谁也不比谁重要」的等权平均，长到「2m 交叠 + 跨 2m 归一」的完整式——'
                '每一步只加一件东西，输出就动一次', 10, C_MUTE, 'start', maxw=1000, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』压缩段的渐进实现阶梯', 9, C_MUTE,
        'end', maxw=460, tag='l0:a')
lc.text(BXR, 64, '——它讲清「为什么非要有位置偏置那一项」，是软池化那张图的前置一格', 9, C_MUTE,
        'end', maxw=500, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, dash, lab in ((C_KV_F, C_KV_S, False, '窗内格子 / 实际权重条'),
                        ('#ffffff', C_KV_S, True, '本例未印出数值的一档（只有输出）'),
                        (C_ACC_S, C_ACC_S, False, '输出与位移量')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1, dash=dash)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=280, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 四档 =================
CW, GAP = 325, 40
COLS = [MX + i * (CW + GAP) for i in range(4)]
PAD = 16
INNER = CW - 2 * PAD

SECS = [
    dict(name='sec1 等权平均（基础版）',
         desc=['窗内 4 个格子谁也不比谁重要：', '没有 B、也没有分数投影'],
         ncells=4, cells=['0.0', '0.0', '0.0', '0.0'],
         card=['分数来源：无（四格分数全 0.0）', 'softmax 之后四格等高：0.25 × 4'],
         bars=[0.25, 0.25, 0.25, 0.25], bar_lab=['0.25', '0.25', '0.25', '0.25'],
         out='[4.0, 6.0]', delta='基准：四格等高，也就是算术平均'),
    dict(name='sec2 ＋位置偏置 B',
         desc=['B^a = [0.5, 0.4, 0.3, 0.2]', '（行 = 窗内第几格）'],
         ncells=4, cells=['0.5', '0.4', '0.3', '0.2'],
         card=['分数来源：B —— 「窗内第几格」', '唯一能携带这件信息的量'],
         bars=[0.288651, 0.261183, 0.236328, 0.213838],
         bar_lab=['0.288651', '0.261183', '0.236328', '0.213838'],
         out='[3.750706, 5.750706]', delta='位移 0.249294（与平均比）'),
    dict(name='sec3 ＋分数投影 Z',
         desc=['分数不再由人给，而是从输入', '学出来的投影（不再恒定）'],
         ncells=4, cells=['Z', 'Z', 'Z', 'Z'],
         card=['分数来源：Z（不再叠加 B）', 'softmax(Z) 直接作为权重'],
         bars=None, bar_lab=None,
         bar_note='权重由 softmax(Z) 直接给出',
         out='[3.135686, 6.969646]', delta='位移 1.218941（与 sec2 比）'),
    dict(name='sec4 完整（2m 交叠）',
         desc=['窗宽从 m 拉成 2m，', 'softmax 跨两个窗一起归一'],
         ncells=8, cells=['2m 窗'] * 8,
         card=['分数来源：Z^a/Z^b ＋ B', '跨 8 个元素一次 softmax'],
         bars=None, bar_lab=None,
         bar_note='权重由 softmax(Z ＋ B) 给出',
         out='[0.576481, 0.469171]', delta='位移 6.500476（与 sec3 比）'),
]

HY0, HY1 = 104, 184
CELL_Y, CELL_H = 204, 38
CARD_Y, CARD_H = 286, 60
BAR_LAB_Y = 360
BAR_BASE, BAR_SCALE = 466, 280
OUT_Y, OUT_H = 482, 46
DELTA_Y = 546

for cx, S in zip(COLS, SECS):
    # ① 档头
    lc.rect(cx, HY0, CW, HY1 - HY0, C_KV_F, C_KV_S, rx=10, sw=1.5)
    lc.text(cx + PAD, HY0 + 30, S['name'], 12.6, C_KV_DEEP, 'start', True, maxw=INNER, tag='hd')
    for i, s in enumerate(S['desc']):
        lc.text(cx + PAD, HY0 + 56 + i * 19, s, 9.2, '#334155', 'start', maxw=INNER, tag='hd:d')

    # ② 窗内格子
    n = S['ncells']
    gap = 5 if n == 4 else 4
    cw = (INNER - (n - 1) * gap) / n
    for i in range(n):
        x = cx + PAD + i * (cw + gap)
        half_b = (n == 8 and i >= 4)
        lc.rect(x, CELL_Y, cw, CELL_H, C_KV_F if not half_b else '#ffffff', C_KV_S, rx=4, sw=1.2)
        if n == 4:
            lc.text(x + cw / 2, CELL_Y + 24, S['cells'][i], 10, C_KV_DEEP, 'middle',
                    maxw=cw + 2, tag='cl')
        elif i < 4:
            lc.text(x + cw / 2, CELL_Y + 24, 'a', 9.5, C_KV_DEEP, 'middle', tag='cl:a')
        else:
            lc.text(x + cw / 2, CELL_Y + 24, 'b', 9.5, C_KV_DEEP, 'middle', tag='cl:b')
    if n == 4:
        lc.text(cx + PAD, CELL_Y - 8, '窗内 4 个格子（m = 4）', 8.6, C_MUTE, 'start', maxw=INNER,
                tag='cl:l4')
    else:
        lc.text(cx + PAD, CELL_Y - 8, '2m = 8 个格子（当前窗 4 + 前窗 4）', 8.6, C_MUTE,
                'start', maxw=INNER, tag='cl:l8')
        # 2m 括线
        lc.parrow([(cx + PAD, CELL_Y + CELL_H + 12), (cx + PAD + INNER, CELL_Y + CELL_H + 12)],
                  C_KV_S, 1.6, marker=None)
        lc.text(cx + PAD + INNER / 2, CELL_Y + CELL_H + 28, '2m 窗（跨这 8 个元素归一）', 8.8,
                C_KV_DEEP, 'middle', maxw=INNER + 20, tag='cl:br')

    # ③ 分数来源卡
    lc.rect(cx + PAD, CARD_Y, INNER, CARD_H, C_SOFT_F, C_SOFT_S, rx=8, sw=1.2)
    for i, s in enumerate(S['card']):
        lc.text(cx + PAD + 12, CARD_Y + 26 + i * 22, s, 9.2 if i == 0 else 8.8,
                '#334155' if i == 0 else C_MUTE, 'start', maxw=INNER - 24, tag='card')

    # ④ 权重条
    lc.text(cx + PAD, BAR_LAB_Y, '权重（按大小做深浅）' if S['bars'] else '权重（这一档不印数值）',
            8.6, C_MUTE, 'start', maxw=INNER, tag='bar:l')
    if S['bars']:
        n = len(S['bars'])
        gap = 5
        bw = (INNER - (n - 1) * gap) / n
        for i, wv in enumerate(S['bars']):
            hgt = wv * BAR_SCALE
            x = cx + PAD + i * (bw + gap)
            lc.rect(x, BAR_BASE - hgt, bw, hgt, C_BAR_F if i else C_SOFT_F, C_KV_S, rx=3, sw=1.2)
            lc.text(x + bw / 2, BAR_BASE - hgt - 6, S['bar_lab'][i], 7.6, C_KV_DEEP, 'middle',
                    maxw=bw + 8, tag='bar:v')
        lc.parrow([(cx + PAD, BAR_BASE), (cx + PAD + INNER, BAR_BASE)], C_SOFT_S, 1.4,
                  marker=None)
    else:
        lc.rect(cx + PAD, BAR_BASE - 84, INNER, 84, '#ffffff', C_KV_S, rx=6, sw=1.2, dash=True)
        lc.text(cx + PAD + 12, BAR_BASE - 52, '四格不再等高：', 9.2, C_KV_DEEP, 'start',
                maxw=INNER - 24, tag='bar:p0')
        lc.text(cx + PAD + 12, BAR_BASE - 30, S['bar_note'], 9.2, C_KV_DEEP,
                'start', maxw=INNER - 24, tag='bar:p1')

    # ⑤ 输出
    lc.rect(cx, OUT_Y, CW, OUT_H, '#fff7ed', C_ACC_S, rx=8, sw=1.3)
    lc.text(cx + PAD, OUT_Y + 29, '输出', 9.6, C_ACC_S, 'start', True, tag='out:l')
    lc.text(cx + PAD + 46, OUT_Y + 29, S['out'], 12.5, C_ACC_S, 'start', True, maxw=INNER - 46,
            tag='out:v')

    # ⑥ 位移
    lc.text(cx + PAD, DELTA_Y, S['delta'], 9.0, C_MUTE, 'start', maxw=INNER, tag='dl')

# ================= 底：B 的必要性 + sec4 的归一 =================
BY0, BY1 = 566, 750
lc.rect(MX, BY0, 700 - MX, BY1 - BY0, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 28, '为什么非要那一项 B：「窗内第几格」只能由它携带', 12, C_KV_DEEP,
        'start', True, maxw=620, tag='b1:t')
B1 = [('摘掉 B', '权重 = [0.25, 0.25, 0.25, 0.25]｜加权和 = [4.0, 6.0]',
       '与算术平均逐位相同 = True —— 软池化退化回平均池化'),
      ('加上 B', '权重 = [0.288651, 0.261183, 0.236328, 0.213838]',
       '输出 [3.750706, 5.750706]：格子之间终于分得开'),
      ('为什么', '分数投影 Z 只看内容——同一窗内两个内容相同的格子，在它眼里没有区别',
       '要学出「最近的一格更重要」，只能靠 B')]
for i, (a, b, c) in enumerate(B1):
    ty = BY0 + 62 + i * 42
    lc.text(MX + 16, ty, a, 9.6, C_KV_DEEP, 'start', True, tag='b1:a%d' % i)
    lc.text(MX + 90, ty, b, 9.6, '#334155', 'start', True, maxw=560, tag='b1:b%d' % i)
    lc.text(MX + 16, ty + 18, c, 8.6, C_MUTE, 'start', maxw=620, tag='b1:c%d' % i)

lc.rect(720, BY0, BXR - 720, BY1 - BY0, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
lc.text(736, BY0 + 28, 'sec4 为什么是「完整」：归一范围变了', 12, C_TXT, 'start', True,
        maxw=460, tag='b2:t')
B2 = [('sec1–sec3', 'softmax 只在 4 个元素上归一（自己这一窗）'),
      ('sec4', 'softmax 在 8 个元素上归一：权重逐列和 = [1.0, 1.0]'),
      ('差别', '2m 窗的输入有一半来自前一个窗——这就是「重叠」，也正是压缩率恰好是 1/m 而不是 1/2m 的原因')]
for i, (a, b) in enumerate(B2):
    ty = BY0 + 62 + i * 42
    lc.text(736, ty, a, 9.6, C_TXT, 'start', True, tag='b2:a%d' % i)
    lc.text(736 + 76, ty, b, 9.4, '#334155', 'start', maxw=640, tag='b2:b%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 774, '数字口径：玩具 m = 4、c = 2；四档共用同一组输入，'
                 '每档只多一件东西（B / 分数投影 Z / 2m 交叠与跨窗归一）。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 792, '依据 DeepSeek-V4 技术报告 §2.3.1 的 Eq.(10)-(12)（arXiv:2606.19348）；'
                 '四档数值与两两位移由本章算例实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-csa-ladder.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
