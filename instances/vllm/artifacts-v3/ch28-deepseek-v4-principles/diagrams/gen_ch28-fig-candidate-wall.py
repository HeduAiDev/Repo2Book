#!/usr/bin/env python3
"""ch28 机制图 · 候选墙：因果由候选集合本身保证（ch28-fig-candidate-wall）

claim：因果约束不需要掩码表：候选集合本身就是『已完成的块数』(t+1)//m——t=2 时候选 0 个
（一个压缩块都没攒满），只有滑窗看得见那 3 个 token；论文的 s < ⌊t/m⌋ 与代码的 (t+1)//m
只在块尾 token 上差一格（仍严格因果、不含未来 token）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  m = 4、t = 0..11 的候选表：t=0/1/2 可见块 []（滑窗 [0,0] / [0,1] / [0,2]）；
        t=3 可见块 [0]；t=7 可见块 [0, 1]；t=11 可见块 [0, 1, 2]
  论文口径（⌊t/m⌋）与代码口径（(t+1)//m）的差只出现在块尾 token：t = 3, 7, 11（差一格 = 1）
  t=2 时可见压缩块 = 0 个——还没攒满一块，只能靠滑窗

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩缓存侧）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 720
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_ZERO_F, C_ZERO_S = '#fef2f2', '#fca5a5'
C_TAIL_F, C_TAIL_S = '#fefce8', '#facc15'

# ---------------- 标题区 ----------------
lc.text(MX, 32, '候选墙：因果不需要一张掩码表', 17, C_TXT, 'start', True, maxw=480, tag='title')
lc.text(MX, 56, 'query 能挑的块，只有那些已经攒满的块——候选集合本身就是因果的；'
                '论文写 s < floor(t/m)（向下取整），代码写 (t+1)//m，两者只在块尾 token 上差一格',
        10, C_MUTE, 'start', maxw=1160, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』选择段的候选集合格', 9, C_MUTE,
        'end', maxw=440, tag='l0:a')
lc.text(BXR, 64, '——打分打完，这里决定那个分数的定义域', 9, C_MUTE, 'end', maxw=430,
        tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_ZERO_F, C_ZERO_S, '候选 0 个：只有滑窗看得见'),
                  (C_TAIL_F, C_TAIL_S, '块尾 token：两种口径差一格'),
                  (C_KV_F, C_KV_S, '已完成的块（可见）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 左：12 行候选表 =================
TX0 = MX
TX1 = 1000
TBY = 128
lc.text(TX0, TBY, 'm = 4，逐 token 看一遍（n_win = 3 的滑窗口径）', 11.5, C_KV_DEEP, 'start',
        True, maxw=560, tag='tb:t')
HDR_Y, HDR_H = TBY + 16, 26
COLS = [(TX0, 90, 't'), (TX0 + 90, 150, '论文 floor(t/m)'), (TX0 + 240, 160, '代码 (t+1)//m'),
        (TX0 + 400, TX1 - TX0 - 400, '可见的压缩块 ／ 滑窗里的 token')]
lc.rect(TX0, HDR_Y, TX1 - TX0, HDR_H, '#e2e8f0', '#94a3b8', rx=4, sw=1.1)
for x, w, lab in COLS:
    lc.text(x + 10, HDR_Y + 18, lab, 9.2, '#334155', 'start', True, maxw=w, tag='tb:h')

ROWS = [('0', '0', '0', '[]', '[0, 0]'),
        ('1', '0', '0', '[]', '[0, 1]'),
        ('2', '0', '0', '[]', '[0, 2]'),
        ('3', '0', '1', '[0]', '[1, 3]'),
        ('4', '1', '1', '[0]', '[2, 4]'),
        ('5', '1', '1', '[0]', '[3, 5]'),
        ('6', '1', '1', '[0]', '[4, 6]'),
        ('7', '1', '2', '[0, 1]', '[5, 7]'),
        ('8', '2', '2', '[0, 1]', '[6, 8]'),
        ('9', '2', '2', '[0, 1]', '[7, 9]'),
        ('10', '2', '2', '[0, 1]', '[8, 10]'),
        ('11', '2', '3', '[0, 1, 2]', '[9, 11]')]
ROW_H = 38
RY0 = HDR_Y + HDR_H
for i, (t, paper, impl, blocks, swa) in enumerate(ROWS):
    ry = RY0 + i * ROW_H
    fill = C_ZERO_F if t in ('0', '1', '2') else (C_TAIL_F if t in ('3', '7', '11') else '#ffffff')
    lc.rect(TX0, ry, TX1 - TX0, ROW_H, fill, C_KV_S, rx=3, sw=1.0)
    lc.text(TX0 + 12, ry + 22, t, 9.6, C_KV_DEEP, 'start', True, tag='tb:t%d' % i)
    lc.text(TX0 + 102, ry + 22, paper, 9.6, '#334155', 'start', tag='tb:p%d' % i)
    lc.text(TX0 + 252, ry + 22, impl, 9.6, '#334155', 'start', tag='tb:c%d' % i)
    lc.text(TX0 + 412, ry + 22, blocks, 9.6, C_KV_DEEP, 'start', True, maxw=120,
            tag='tb:b%d' % i)
    lc.text(TX0 + 560, ry + 22, swa, 9.6, C_MUTE, 'start', tag='tb:s%d' % i)
TY_BOT = RY0 + 12 * ROW_H
lc.text(TX0, TY_BOT + 22, '候选数关于 t 单调不减（每步 +0 或 +1），且永远 ≤ 已经处理完的 token 数'
                          '——因果不用额外掩码。', 8.8, C_MUTE, 'start', maxw=940, tag='tb:n')

# ================= 右：三张说明卡 =================
RX0, RX1 = 1030, BXR
lc.rect(RX0, TBY + 16, RX1 - RX0, 150, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
lc.text(RX0 + 16, TBY + 44, '① 前 3 个 token：一块都还没攒满', 11, C_TXT, 'start', True,
        maxw=400, tag='rc1:t')
for i, s in enumerate(['t = 0 / 1 / 2 时可见压缩块 = 0 个。',
                       '它们的全部上下文只有滑窗那几条（1 / 2 / 3 条）。',
                       '这是「压缩有延迟」的直接后果：',
                       '块没攒满，就没有条目可看。']):
    lc.text(RX0 + 16, TBY + 70 + i * 22, s, 9.2, '#334155' if i < 2 else C_MUTE, 'start',
            maxw=400, tag='rc1:r%d' % i)

lc.rect(RX0, TBY + 182, RX1 - RX0, 128, C_TAIL_F, C_TAIL_S, rx=10, sw=1.4)
lc.text(RX0 + 16, TBY + 210, '② 仅有的分歧：块尾那三行', 11, '#854d0e', 'start', True, maxw=400,
        tag='rc2:t')
for i, s in enumerate(['t = 3、7、11 是块尾 token：代码口径',
                       '允许它多看见「自己刚产出的那一块」——',
                       '仍严格因果（那一块不含未来 token）。',
                       '别断言两种口径等价：定义域差一格。']):
    lc.text(RX0 + 16, TBY + 236 + i * 22, s, 9.2, '#3f3f46' if i < 3 else C_ACC_S, 'start',
            maxw=400, tag='rc2:r%d' % i)

lc.rect(RX0, TBY + 326, RX1 - RX0, 148, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(RX0 + 16, TBY + 354, '③ 墙随 t 线性长，靠 top-k 封顶', 11, C_KV_DEEP, 'start', True,
        maxw=400, tag='rc3:t')
for i, s in enumerate(['候选 0 → 250000 条（1M 上下文、m = 4）',
                       '实看被 top-k 封在 512 条',
                       '封不住的是短上下文：候选少时',
                       '排序无意义，全选快路径直接接管。']):
    lc.text(RX0 + 16, TBY + 380 + i * 22, s, 9.2, '#334155' if i < 2 else C_MUTE, 'start',
            maxw=400, tag='rc3:r%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 672, '数字口径：玩具 m = 4、t = 0..11（三个块）；滑窗列按 n_win = 3 的玩具口径列出'
                 '（config 口径是 n_win = 128）。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:1')
lc.text(MX, 690, '依据 DeepSeek-V4 技术报告 Eq.(17) 与 §2.3.3（arXiv:2606.19348）；'
                 '两种因果口径的逐行对照由本章算例实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')
lc.text(MX, 708, '读法：先看左边那列「可见的压缩块」怎么从空变长，再看右边几张卡回答'
                 '「为什么不需要掩码表」。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-candidate-wall.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
