#!/usr/bin/env python3
"""ch15 机制图 10 · 块内部分命中与 CoW 换尾（figure_spec ch15-fig-partial-cow，模板 before-after）

放大自 L0 KV 账本列（kv_column）缓存区·命中主循环——「CoW 换尾」与「拷贝过线」两拍的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：块内部分命中后，请求把共享尾块原地换成私有 cow 块并登记 (source, cow) 拷贝对——
两个人各写各的块，缓存条目与在跑请求两不坏。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
GREEN = '#16a34a'
RED = '#dc2626'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '共享半截块谁接着写？各自拷一本：CoW 换尾——带宽级拷贝换掉整块重算',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'B 与 A 共享 48 token、落在 64-token 块内部：phase 1 满块链 miss，phase 2 探到 @48 边界命中；A 的块还在表里服务别人——B 写自己的 cow 拷贝',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 命中主循环「CoW 换尾」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96

# ---------------- 左：两级探测（B 80 token · 共享前 48） ----------------
LX, LW = MX, 620
lc.rect(LX, LY, LW, 420, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, LY + 22, '两级探测：full(64) + mamba(64) 混合 · hash_block_size=16', 11.5,
        lc.C_TXT, 'start', True, maxw=LW - 32, tag='lp:t')
# A 的块 1 状态条（64-token 块内部 48 已算）
by = LY + 42
BW3 = LW - 32
seg48 = BW3 * 48 / 64
lc.rect(LX + 16, by, seg48, 40, lc.C_KV_F, lc.C_KV_S, rx=0, sw=1.2)
lc.rect(LX + 16 + seg48, by, BW3 - seg48, 40, '#ffffff', GRAY, rx=0, sw=1.2, dash=True)
lc.text(LX + 16 + seg48 / 2, by + 25, 'A 已算 48 token（部分条目 @48 注册）', 8.8, lc.C_KV_S,
        'middle', True, maxw=seg48 - 6, tag='lp:a')
lc.text(LX + 16 + seg48 + (BW3 - seg48) / 2, by + 25, '未写', 8.2, GRAY, 'middle', maxw=60,
        tag='lp:b')
for t in (16, 32, 48, 64):
    xx = LX + 16 + BW3 * t / 64
    lc.seg(xx, by + 40, xx, by + 46, GRAY, 1.0)
    lc.text(xx, by + 58, '@%d' % t, 7.6, GRAY, 'middle', maxw=34, tag='bt%d' % t)
# 探测步骤（@64 的 miss 属 phase 1——它经 64 粗视图查链尾 hash[3]；phase 2 的探测序是
# range(max_partial_idx−1, first_partial_idx−1, −1)，零满块命中时从 fine_idx 2（@48）起）
PROBES = [
    ('① phase 1', '满块链查 @64 边界（hash[3]）', '✗ miss——A 只缓存到 48、没有 64 边界条目', RED),
    ('② phase 2', '自高向低探块内边界：@48（fine_idx 2）', '✓ 命中 48——块内边界命中（1 次查表；细粒度对齐 16）', GREEN),
]
py = by + 74
for i, (tag, act, res, col) in enumerate(PROBES):
    yy = py + i * 44
    if tag:
        lc.text(LX + 20, yy + 10, tag, 9.2, lc.C_TXT, 'start', True, maxw=66, tag='pb%d:t' % i)
    lc.text(LX + 88, yy + 10, act, 8.8, '#334155', 'start', maxw=290, tag='pb%d:a' % i)
    lc.text(LX + 20, yy + 27, res, 8.8, col, 'start', True, maxw=LW - 40, tag='pb%d:r' % i)
ry = py + 2 * 44 + 6
lc.rect(LX + 16, ry, LW - 32, 74, '#f0fdf4', GREEN, rx=6, sw=1.3)
lc.text(LX + 30, ry + 21, '命中 48 / 80 = 60% 的 prompt', 9.8, GREEN, 'start', True,
        maxw=LW - 60, tag='lp:r1')
lc.text(LX + 30, ry + 41, '但 A 的那块还在表里服务别人——B 不能接着写它：', 8.6, '#334155',
        'start', maxw=LW - 60, tag='lp:r2')
lc.text(LX + 30, ry + 59, '命中落进「块内部」，接写就要 CoW', 8.6, '#334155', 'start',
        maxw=LW - 60, tag='lp:r3')

# ---------------- 右：CoW 换尾（before-after） ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
RH = 420
lc.rect(RX, LY, RW, RH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, LY + 22, 'CoW 换尾：每组多预留的那 1 块成为私有拷贝', 11.5, lc.C_TXT, 'start',
        True, maxw=RW - 32, tag='rp:t')
CELL_W, CELL_H = 118, 46
GAPX = 14


def block_cell(x, y, label, sub, fill, stroke, tcol):
    lc.rect(x, y, CELL_W, CELL_H, fill, stroke, rx=6, sw=1.3, dash=(stroke == GRAY))
    lc.text(x + CELL_W / 2, y + 19, label, 9.6, tcol, 'middle', True, maxw=CELL_W - 8,
            tag='bc:' + label)
    lc.text(x + CELL_W / 2, y + 37, sub, 7.8, '#475569', 'middle', maxw=CELL_W - 8,
            tag='bcs:' + label)


# before
byy = LY + 44
lc.text(RX + 16, byy, '换尾前（准入后 · 部分命中记账）', 9.2, lc.C_TXT, 'start', True, maxw=240,
        tag='rp:b4')
block_cell(RX + 16, byy + 12, '块 1', '共享（A 的）', lc.C_KV_F, lc.C_KV_S, lc.C_KV_S)
lc.text(RX + 16 + CELL_W + GAPX + CELL_W / 2, byy + 12 + CELL_H / 2 + 4, '+', 12, GRAY,
        'middle', True, maxw=16, tag='rp:p1')
block_cell(RX + 16 + 2 * (CELL_W + GAPX), byy + 12, '块 2', '共享（A 的）', lc.C_KV_F, lc.C_KV_S,
           lc.C_KV_S)
lc.text(RX + 16 + 3 * (CELL_W + GAPX), byy + 30, 'B 的块表里挂着', 8, GRAY, 'start', maxw=RW - 16 - 3 * (CELL_W + GAPX) - 10,
        tag='rp:bn1')
lc.text(RX + 16 + 3 * (CELL_W + GAPX), byy + 46, 'A 的两个共享块', 8, GRAY, 'start', maxw=RW - 16 - 3 * (CELL_W + GAPX) - 10,
        tag='rp:bn2')
lc.text(RX + 16 + 3 * (CELL_W + GAPX), byy + 62, '（full 组 + mamba 组）', 8, GRAY, 'start', maxw=RW - 16 - 3 * (CELL_W + GAPX) - 10,
        tag='rp:bn3')
# 换尾动作箭头
ayy = byy + 12 + CELL_H + 8
lc.text(RX + 16, ayy + 12, '_apply_cow：块表原地换块 + 登记 (source, cow) 拷贝对', 8.8,
        lc.C_TXT, 'start', True, maxw=RW - 32, tag='rp:act')
lc.seg(RX + 16 + CELL_W / 2, ayy + 18, RX + 16 + CELL_W / 2, ayy + 40, GRAY, 1.4, 'std')
# after
ay = ayy + 44
lc.text(RX + 16, ay, '换尾后（B 只写自己的块）', 9.2, lc.C_TXT, 'start', True, maxw=240,
        tag='rp:a4')
block_cell(RX + 16, ay + 12, 'cow 块 3', '拷贝自 1 · ref_cnt=2', '#f0fdf4', GREEN, GREEN)
block_cell(RX + 16 + CELL_W + GAPX, ay + 12, '块 4', 'B 自有新块', '#ffffff', GRAY, lc.C_TXT)
block_cell(RX + 16 + 2 * (CELL_W + GAPX), ay + 12, 'cow 块 5', '拷贝自 2 · ref_cnt=2', '#f0fdf4',
           GREEN, GREEN)
lc.text(RX + 16 + 3 * (CELL_W + GAPX), ay + 30, 'full 组：[cow3, 4]', 8, GRAY, 'start',
        maxw=RW - 16 - 3 * (CELL_W + GAPX) - 10, tag='rp:an1')
lc.text(RX + 16 + 3 * (CELL_W + GAPX), ay + 46, 'mamba 组：[cow5]', 8, GRAY, 'start',
        maxw=RW - 16 - 3 * (CELL_W + GAPX) - 10, tag='rp:an2')
lc.text(RX + 16 + 3 * (CELL_W + GAPX), ay + 62, 'A 的块 1、2 原样', 8, GRAY, 'start',
        maxw=RW - 16 - 3 * (CELL_W + GAPX) - 10, tag='rp:an3')
# 过线
oy = ay + 12 + CELL_H + 14
lc.rect(RX + 16, oy, RW - 32, 78, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
lc.text(RX + 30, oy + 20, '拷贝过线：take_kv_cache_block_copies 打包 2 对 →', 9, lc.C_TXT,
        'start', True, maxw=RW - 60, tag='rp:o1')
lc.text(RX + 30, oy + 39, 'worker 在 GPU 上整块拷（带宽级）；两端引用保留到', 8.6, '#334155',
        'start', maxw=RW - 60, tag='rp:o2')
lc.text(RX + 30, oy + 57, '拷完（步序栅栏）——retained 4 · copies 2', 8.6, '#334155', 'start',
        maxw=RW - 60, tag='rp:o3')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + 420 + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不变量：CoW 后请求只写自己的私有块（写路径只看得见自己块表）——缓存条目与共享者两不坏；预算正确性：部分命中 +1 恰是 cow 目标那块、分配永不超发',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '对照无 partial-hit 的世界：命中只能落 64 边界 ⇒ 命中 0、整块重算 48 token 的 prefill（矩阵乘级）——拷一块（显存带宽级）远便宜于重算一块',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, dash, tcol, name in [
        (lc.C_KV_F, lc.C_KV_S, False, lc.C_KV_S, '共享块（A 的 · 仍在表里服务别人）'),
        ('#f0fdf4', GREEN, False, GREEN, 'cow 私有块（B 接着写）'),
        ('#ffffff', GRAY, True, GRAY, '未写区 / 自有新块')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=250, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '@N = 块内细粒度边界（hash_block_size=16 的格点）；探测成本：phase 2 至多 scale_factor−1 = 3 次查表',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/single_type_kv_cache_manager.py:L682-L777（phase 1/2 两级探测）· '
        'L226-L230（部分命中预算 +1）· L347-L357（消费 _partial_hit_reqs）· L405-L425（_apply_cow 原地换块+登记拷贝对）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, 'vllm/v1/core/sched/scheduler.py:L1181-L1190（take_kv_cache_block_copies 过线）· '
        'vllm/v1/worker/gpu_model_runner.py:L1223-L1228（GPU 整块拷）· 数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-partial-cow.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
