#!/usr/bin/env python3
"""ch28 机制图 · 压缩坐标系：谁产出条目、条目落在哪、未满的窗在哪（ch28-fig-compress-coords）

claim：压缩坐标系把『谁产出条目、条目落在哪个槽、未满的窗在哪』三件事一次说清：
条目号 = 位置 // m、只有块尾 token 产出、未满一窗留在压缩机 state 里
（与 KV 页共享物理张量，所以块大小被页尺寸约束）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  条目号 = pos // m；产出触发条件 (position + 1) % compress_ratio == 0（只有块尾 token 干活）
  三本缓存宽度：主压缩条目 584 B、索引器小头 132 B、滑窗 128 条未压缩 KV
  1M 下条目数：CSA 层 250000 条（1e6 // 4）、HCA 层 7812 条（1e6 // 128）
  块尾 token 的窗口算术：start = position − (1 + OVERLAP)·CR + 1

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 缓存账）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 800
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_SWA_F, C_SWA_S = '#cffafe', '#0e7490'
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'

EXTRA_DEFS = ('<defs>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '三把尺子量同一串 token：谁产出、落在哪、没攒满的在哪', 17, C_TXT, 'start',
        True, maxw=760, tag='title')
lc.text(MX, 56, '原始 token 轴、压缩块轴、分页槽位轴叠在一起看——压缩不是另存一份，'
                '而是把『每个 token 一页』换成『每 m 个 token 一页』', 10, C_MUTE, 'start',
        maxw=1050, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『调度 · 显存账本』的 KV 池块内部', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——它是软池化那张图的产出去向，也是三本账在缓存层的落点', 9, C_MUTE, 'end',
        maxw=470, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '原始 token（一位一格）'),
                  (C_KV_F, C_KV_S, '压缩条目（一块一条目）'),
                  (C_SWA_F, C_SWA_S, '未压缩的滑窗 KV')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ---------------- 网格常量（三把尺子共用一套列宽） ----------------
COLW = 62
GX0 = 200
NC = 12
M = 4


def colx(i):
    return GX0 + i * COLW


# ---------------- ① 原始 token 轴 ----------------
R1Y = 148
lc.text(GX0 - 12, R1Y + 20, '原始 token 轴', 10, C_TXT, 'end', True, tag='a1:t')
lc.text(GX0 - 12, R1Y + 36, '（位置 0 … n−1）', 8.6, C_MUTE, 'end', tag='a1:s')
for i in range(NC):
    tail = ((i + 1) % M == 0)
    lc.rect(colx(i) + 3, R1Y, COLW - 6, 34, C_GPU_F, C_GPU_S, rx=3, sw=1.2)
    lc.text(colx(i) + COLW / 2, R1Y + 21, str(i), 10, C_TXT, 'middle', True, tag='a1:%d' % i)
    if tail:
        lc.text(colx(i) + COLW / 2, R1Y - 8, '块尾产出', 8.6, C_SCOR_S, 'middle', True,
                tag='a1:t%d' % i)
lc.text(colx(NC) + 14, R1Y + 21, '只有块尾 token 干活：(position + 1) % m == 0', 9.2, C_SCOR_S,
        'start', True, maxw=330, tag='a1:n0')
lc.text(colx(NC) + 14, R1Y + 37, '其余位置早退——它们不产出条目，只喂统计量', 8.8, C_MUTE,
        'start', maxw=330, tag='a1:n1')

# ---------------- ② 压缩块轴 ----------------
R2Y = R1Y + 76
lc.text(GX0 - 12, R2Y + 20, '压缩块轴', 10, C_TXT, 'end', True, tag='a2:t')
lc.text(GX0 - 12, R2Y + 36, '（条目号 = 位置 // m）', 8.6, C_MUTE, 'end', tag='a2:s')
for b in range(NC // M):
    w = M * COLW - 6
    lc.rect(colx(b * M) + 3, R2Y, w, 34, C_KV_F, C_KV_S, rx=4, sw=1.4)
    lc.text(colx(b * M) + w / 2 + 3, R2Y + 21, '条目 %d' % b, 10, C_KV_DEEP, 'middle', True,
            tag='a2:b%d' % b)
lc.text(colx(NC) + 14, R2Y + 21, '12 个 token → 3 条（4 个 token 一条）', 9.2, C_KV_DEEP,
        'start', True, maxw=330, tag='a2:n0')
lc.text(colx(NC) + 14, R2Y + 37, '条目号与位置一一对应：pos // m', 8.8, C_MUTE, 'start',
        maxw=330, tag='a2:n1')
# 产出对齐虚线
for b in range(NC // M):
    x = colx(b * M + M - 1) + COLW / 2
    lc.seg(x, R1Y + 34, x, R2Y, '#cbd5e1', 1.2, dash=True)

# ---------------- ③ 分页槽位轴 ----------------
R3Y = R2Y + 76
lc.text(GX0 - 12, R3Y + 20, '分页槽位轴', 10, C_TXT, 'end', True, tag='a3:t')
lc.text(GX0 - 12, R3Y + 36, '（块号 + 槽内偏移）', 8.6, C_MUTE, 'end', tag='a3:s')
PAGE = 8                                     # 玩具里一页放几格（真实被页尺寸约束成 4 或 8）
for k in range(2):
    lc.rect(colx(k * PAGE) + 3, R3Y, PAGE * COLW - 6, 34, '#ffffff', '#94a3b8', rx=4, sw=1.4,
            dash=True)
    lc.text(colx(k * PAGE) + PAGE * COLW / 2, R3Y - 8, '分页块 %d（槽 0…%d）' % (k, PAGE - 1),
            9.5, '#334155', 'middle', True, maxw=PAGE * COLW - 14, tag='a3:p%d' % k)
for b in range(NC // M):
    x = colx(b * M) + 3 + (M * COLW - 6) / 2
    lc.text(x, R3Y + 21, '条目 %d 落位' % b, 8.8, C_KV_DEEP, 'middle', maxw=M * COLW - 12,
            tag='a3:b%d' % b)
lc.text(1203, R3Y + 14, '条目与页共享同一张物理张量（块大小被页尺寸约束）',
        9.0, '#155e75', 'start', True, maxw=250, tag='a3:n0')
lc.text(1203, R3Y + 30, '（config 里是 4 或 8 这一档）', 8.8, C_MUTE, 'start', maxw=250,
        tag='a3:n1')

# ---------------- ④ 未满的窗 ----------------
R4Y = R3Y + 62
lc.rect(colx(0) + 3, R4Y, M * COLW - 6, 40, '#f8fafc', C_MUTE, rx=4, sw=1.3, dash=True)
lc.text(colx(0) + (M * COLW - 6) / 2 + 3, R4Y + 24, '还没攒满的窗', 9.5, '#334155', 'middle',
        True, maxw=M * COLW - 14, tag='a4:b')
lc.text(colx(M) + 14, R4Y + 24, '未满一窗留在压缩机自己的 state 里等下一拍——不产出条目，'
                                '也不占页', 9.2, '#334155', 'start', True, maxw=620, tag='a4:n')

# ---------------- ⑤ 三本缓存宽度 ----------------
CY0 = 502
lc.rect(MX, CY0, BXR - MX, 116, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, CY0 + 26, '三本缓存宽度（1M 上下文、43 层）', 11.5, C_KV_DEEP, 'start', True,
        maxw=420, tag='cw:t')
CW = [('主压缩条目', '584 B 一条', '448 fp8 NoPE + 128 bf16 RoPE + 8 fp8 scale'),
      ('索引器小头', '132 B 一条', '128 fp8 + 4 fp32 scale（只有压缩率为 4 的层建它）'),
      ('滑窗窗口', '128 条未压缩 KV', '恒长：min(pos + 1, 128)，与上下文长度无关')]
for i, (a, b, c) in enumerate(CW):
    ty = CY0 + 52 + i * 22
    lc.text(MX + 16, ty, a, 9.5, C_TXT, 'start', True, tag='cw:a%d' % i)
    lc.text(MX + 130, ty, b, 9.5, C_KV_DEEP, 'start', True, maxw=140, tag='cw:b%d' % i)
    lc.text(MX + 290, ty, c, 9, '#334155', 'start', maxw=560, tag='cw:c%d' % i)
lc.text(MX + 880, CY0 + 52, '1M 下的条目数：', 9.2, C_TXT, 'start', True, tag='cw:d0')
lc.text(MX + 880, CY0 + 74, 'CSA 层 250000 条（1e6 // 4）', 9.2, C_KV_DEEP, 'start', True,
        maxw=280, tag='cw:d1')
lc.text(MX + 880, CY0 + 96, 'HCA 层 7812 条（1e6 // 128）', 9.2, C_KV_DEEP, 'start', True,
        maxw=280, tag='cw:d2')

# ---------------- ⑥ 三条动线小结 ----------------
SY0 = 634
lc.rect(MX, SY0, BXR - MX, 116, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, SY0 + 26, '三条动线各一句话', 11.5, C_TXT, 'start', True, maxw=280, tag='sm:t')
SM = [('谁产出条目', '只有 (position + 1) % compress_ratio == 0 的块尾 token 触发；'
                   '其余位置全早退'),
      ('条目落在哪', '同一张分页 KV 里：条目号 = 位置 // m，槽位 = 块号 + 槽内偏移'),
      ('未完成窗在哪', '留在压缩机的 state 里（与 KV 页共享物理张量，因此块大小被页尺寸约束）')]
for i, (a, b) in enumerate(SM):
    ty = SY0 + 52 + i * 22
    lc.text(MX + 16, ty, a, 9.5, C_TXT, 'start', True, tag='sm:a%d' % i)
    lc.text(MX + 150, ty, b, 9.2, '#334155', 'start', maxw=1240, tag='sm:b%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 770, '数字口径：玩具 n = 12、m = 4（3 条条目、一页 8 格）；'
                 'config 口径 1M 上下文、压缩率 4 与 128 各一批层。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 788, '依据 DeepSeek-V4 技术报告 §2.3.1 的 1/m 与产出单位口径（arXiv:2606.19348）；'
                 '条目字节数与 1M 条目数由本章驱动脚本按 config 纯算术实跑。', 8.5, C_MUTE,
        'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-compress-coords.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
