#!/usr/bin/env python3
"""ch28 机制图 · 重叠窗与边界：为什么恰好压到 1/m 而不是 1/(2m)（ch28-fig-overlap-window）

claim：相邻条目共享 m 个输入 → 序列恰好压到 1/m（不是 1/2m）；i=0 是唯一特例
（b 半区 −inf/0），而实现侧把同一件事写成『窗口起点越界 → score 取 −inf、kv 取 0』。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  三条条目的输入下标：i=0 [0,1,2,3]+[]；i=1 [4,5,6,7]+[0,1,2,3]；i=2 [8,9,10,11]+[4,5,6,7]
  条目数 12 // 4 = 3（不是 12 // 8 = 1）；共享验证 True
  pin 窗口算术（position=3）：start = 3 − 2*4 + 1 = −4；mask_pos = [F,F,F,F,T,T,T,T]
  head_offset = [0,0,0,0,512,512,512,512]（前半区 = 前窗、后半区 = 当前窗，与论文 a/b 相反）
  布局等价：max|C_comp − C_ref| = 0.00000000；调头后 max|S − S_ref_swapped| = 0.00000000；
  直接对减 max|S − S_ref| = 0.47154682

配色走 book/cartography/l0_common.py 的角色常量（KV 青标压缩产物、蓝=当前窗、灰=前一个窗）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_CUR_F, C_CUR_S = '#e0f2fe', '#0284c7'       # 当前窗（a 半区）
C_PRE_F, C_PRE_S = '#f1f5f9', '#94a3b8'       # 前一个窗（b 半区）
C_SHARE_F, C_SHARE_S = '#cffafe', '#0891b2'   # 共享区高亮

EXTRA_DEFS = ('<defs>'
              '<marker id="tk" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '<marker id="cy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '</defs>')

# ---------------- 布局常量 ----------------
CW, GAP = 72, 4
BX0 = 180
ROWS_Y = (176, 222, 268)          # 三条条目各自的格子行
CH = 30
SETS = [(0, (0, 1, 2, 3), ()), (1, (4, 5, 6, 7), (0, 1, 2, 3)), (2, (8, 9, 10, 11), (4, 5, 6, 7))]


def cx(i):
    return BX0 + i * CW


def ccx(i):
    return BX0 + (i + 0.5) * CW


# ---------------- 标题区 ----------------
lc.text(MX, 32, '两扇窗叠着推：为什么恰好压到 1/4，而不是 1/8', 17, C_TXT, 'start', True,
        maxw=620, tag='title')
lc.text(MX, 56, '每条条目要吃 2m = 8 个位置，但相邻两条共享其中一半——每条净增只有 m = 4 个，'
                '于是 12 个 token 得到 3 条（不是 1 条）', 10, C_MUTE, 'start', maxw=1000, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』内压缩段的一处细节', 9, C_MUTE,
        'end', maxw=450, tag='l0:a')
lc.text(BXR, 64, '——上游是软池化那张图的输入侧，下游是索引器的候选块', 9, C_MUTE, 'end',
        maxw=450, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_CUR_F, C_CUR_S, '当前窗 a 半区（本条目新吃进的 m 个）'),
                  (C_PRE_F, C_PRE_S, '前一个窗 b 半区（与上一条共享的 m 个）'),
                  (C_SHARE_F, C_SHARE_S, '共享区高亮')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=290, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ---------------- ① token 轴（12 格） ----------------
lc.text(BX0 - 14, ROWS_Y[0] - 24, 'token 轴', 10, C_TXT, 'end', True, tag='ax:t')
for i in range(12):
    lc.rect(cx(i) + GAP / 2, ROWS_Y[0] - 20, CW - GAP, 16, '#ffffff', lc.C_FAINT, rx=2, sw=0.9)
    lc.text(ccx(i), ROWS_Y[0] - 8, str(i), 8.5, C_MUTE, 'middle', tag='ax:%d' % i)

# 共享区高亮（先铺两层，压住之后画的窗格）
for r, (a0, a1) in ((0, (0, 3)), (1, (0, 3))):
    lc.rect(cx(0) + GAP / 2, ROWS_Y[r] - 3, (3 + 1) * CW - GAP, CH + 6, C_SHARE_F, C_SHARE_S,
            rx=4, sw=1.4, dash=True)
lc.rect(cx(4) + GAP / 2, ROWS_Y[1] - 3, 4 * CW - GAP, CH + 6 + (ROWS_Y[2] - ROWS_Y[1]),
        C_SHARE_F, C_SHARE_S, rx=4, sw=1.4, dash=True)

# 三条条目各自的两个半区
for idx, a, b in SETS:
    y = ROWS_Y[idx]
    lc.text(cx(0) - 14, y + CH - 8, '条目 i=%d' % idx, 9.5, C_TXT, 'end', True, tag='rw:l%d' % idx)
    for j in a:
        lc.rect(cx(j) + GAP / 2, y, CW - GAP, CH, C_CUR_F, C_CUR_S, rx=3, sw=1.3)
    for j in b:
        lc.rect(cx(j) + GAP / 2, y, CW - GAP, CH, C_PRE_F, C_PRE_S, rx=3, sw=1.3)
    if not b:
        lc.rect(cx(4) + GAP / 2, y, 4 * CW - GAP, CH, 'none', C_PRE_S, rx=3, sw=1.1, dash=True)
        lc.text(ccx(5.5), y + CH - 10, '没有前一个窗：−inf / 0 填充', 8.6, '#64748b', 'middle',
                maxw=4 * CW - 20, tag='rw:e%d' % idx)

# ---------------- 右侧读数 ----------------
RX0 = 1080
lc.rect(RX0, 120, BXR - RX0, 250, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(RX0 + 16, 144, '四处读数', 11.5, C_TXT, 'start', True, maxw=140, tag='rd:t')
RD = [('条目数', '12 // 4 = 3 条', '（不是 12 // 8 = 1）——重叠不改变条目数'),
      ('每条输入', '2m = 8 个', '前一个窗 4 + 当前窗 4'),
      ('每条净增', 'm = 4 个', 'b 半区正是上一条的 a 半区'),
      ('共享验证', '[0,1,2,3] == [0,1,2,3]', '→ True（论文的 overlapped 句）')]
for i, (a, b, c) in enumerate(RD):
    ty = 170 + i * 48
    lc.text(RX0 + 16, ty, a, 9.5, C_TXT, 'start', True, tag='rd:a%d' % i)
    lc.text(RX0 + 16, ty + 17, b, 11, C_KV_DEEP, 'start', True, tag='rd:b%d' % i)
    lc.text(RX0 + 16, ty + 32, c, 8.5, C_MUTE, 'start', maxw=320, tag='rd:c%d' % i)

# ---------------- 中带：读法 + 一句话结论 ----------------
BAND_W = 1000
lc.rect(MX, 330, BAND_W, 56, C_SHARE_F, C_SHARE_S, rx=8, sw=1.4)
lc.text(MX + 18, 352, '读法：每条目的 8 格 = 前一个窗 4 格（灰）+ 当前窗 4 格（蓝）；'
                      '虚线框是同一批 4 个 token 的两次使用（共享区）。', 9.5, '#0e7490',
        'start', True, maxw=BAND_W - 40, tag='mid:k')
lc.text(MX + 18, 374, '一句话：已消耗的 token 数 = m × 条目数（严格的线性对账）——'
                      '每新增一条只多花 m 个 token，所以序列恰好压到 1/m。', 10.5, '#0e7490',
        'start', True, maxw=BAND_W - 40, tag='mid:l')

# ---------------- 底左：pin 侧窗口算术 ----------------
PY0, PY1 = 448, 676
lc.rect(MX, PY0, 700, PY1 - PY0, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, PY0 + 24, '实现侧把同一件事写成窗口算术（块尾 token position=3）', 11, C_TXT,
        'start', True, maxw=440, tag='pA:t')
PA = [('窗口起点', 'start = position − (1+OVERLAP)·CR + 1 = 3 − 2×4 + 1 = −4'),
      ('8 个位置', 'pos = [−4, −3, −2, −1, 0, 1, 2, 3]'),
      ('越界处理', 'mask_pos（pos ≥ 0）= [F, F, F, F, T, T, T, T]'),
      ('落法', '越界位置 score 取 −inf、kv 取 0（与论文 i=0 的填充同一件事）')]
for i, (a, b) in enumerate(PA):
    ty = PY0 + 54 + i * 32
    lc.text(MX + 16, ty, a, 9.5, C_TXT, 'start', True, tag='pA:a%d' % i)
    lc.text(MX + 120, ty, b, 9.2, '#334155', 'start', maxw=560, tag='pA:b%d' % i)
lc.text(MX + 16, PY1 - 42, '只有块尾 token 干活：(position + 1) % CR == 0（pos 3、7、11…）',
        9, C_KV_DEEP, 'start', True, maxw=640, tag='pA:x')
lc.text(MX + 16, PY1 - 20, 'CR = 4 的一个 8 元素 gather 窗口就是压缩核每次要吃的那一口。',
        8.8, C_MUTE, 'start', maxw=640, tag='pA:y')

# ---------------- 底右：标签对照 ----------------
QX0 = 770
lc.rect(QX0, PY0, BXR - QX0, PY1 - PY0, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(QX0 + 16, PY0 + 24, '一个记号陷阱：两半区的命名在论文与实现里相反', 11, C_TXT, 'start',
        True, maxw=440, tag='pB:t')
lc.text(QX0 + 16, PY0 + 46, '论文 Eq.(11)：a 半区 = 当前窗、b 半区 = 前一个窗', 9.2, '#334155',
        'start', maxw=620, tag='pB:l0')
lc.text(QX0 + 16, PY0 + 64, '两侧实现：前半区装前一个窗、后半区装当前窗（head_offset = '
                            '[0,0,0,0,512,512,512,512]）', 9.2, '#334155', 'start', maxw=620,
        tag='pB:l1')
lc.text(QX0 + 16, PY0 + 82, '→ 同一套数学，两半区顺序相反——把任一边调头后逐位相同。', 9.2,
        '#155e75', 'start', True, maxw=620, tag='pB:l2')
PB = [('产出对账', 'max|C_comp − C_ref| = 0.00000000', '两边的压缩条目逐位一致'),
      ('权重调头', 'max|S − S_ref_swapped| = 0.00000000', '半区顺序换回来后逐位一致'),
      ('直接对减', 'max|S − S_ref| = 0.47154682', '不调头就对减 → 非零（这正是命名相反的形态）')]
for i, (a, b, c) in enumerate(PB):
    ty = PY0 + 116 + i * 34
    lc.text(QX0 + 16, ty, a, 9.5, C_TXT, 'start', True, tag='pB:a%d' % i)
    lc.text(QX0 + 96, ty, b, 9.5, C_KV_DEEP, 'start', True, tag='pB:b%d' % i)
    lc.text(QX0 + 16, ty + 15, c, 8.5, C_MUTE, 'start', maxw=620, tag='pB:c%d' % i)
lc.text(QX0 + 16, PY1 - 20, '对账在同一个槽位上做：把两半区当同一个张量的两个区段读。', 8.8,
        '#334155', 'start', maxw=620, tag='pB:z')

# ---------------- 底带：归纳论证 ----------------
IY0, IY1 = 700, 806
lc.rect(MX, IY0, BXR - MX, IY1 - IY0, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, IY0 + 24, '为什么是严格的 1/m：基例 + 归纳步各自只花 m 个 token', 11.5, C_TXT,
        'start', True, maxw=420, tag='in:t')
IN = [('基例 i=0', 'a 半区吃 token [0, m)，b 半区被 −inf / 0 填掉（不消耗 token）→ 第 0 条花 m 个'),
      ('归纳步 i → i+1', 'a 半区 [m(i+1), m(i+2)) 是新 token（+m）；b 半区 [mi, m(i+1)) 正是上一条的 a 半区（+0）'),
      ('所以', '每条净增 m 个 → 条目数 = n // m；两半区若互不共享就会是 n/(2m)（本例 12 // 8 = 1）')]
for i, (a, b) in enumerate(IN):
    ty = IY0 + 50 + i * 20
    lc.text(MX + 16, ty, a, 9.2, C_TXT, 'start', True, tag='in:a%d' % i)
    lc.text(MX + 150, ty, b, 9.2, '#334155', 'start', maxw=1270, tag='in:b%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 836, '数字口径：玩具 m=4、n=12（三条条目）；窗口算术取 CR = 4、HEAD_SIZE = 512 的一档，'
                 '块尾 token position = 3 / 7 / 11 各走一遍。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 854, '依据 DeepSeek-V4 技术报告 §2.3.1 的 Eq.(11)(12) 与 overlapped 句（arXiv:2606.19348）；'
                 '条目下标、共享验证与布局等价性由本章驱动脚本实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-overlap-window.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
