#!/usr/bin/env python3
"""ch15 机制图 4 · touch 挂块与引用计数（figure_spec ch15-fig-touch-refcount，模板 before-after）

放大自 L0 KV 账本列（kv_column）缓存区·命中主循环——「touch 挂块」一拍的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：touch 给命中块 ref_cnt+1 并在 ref_cnt==0 时把它从自由队列 O(1) 摘出——
同一物理块从此被多个请求共同引用，共享前缀才真的只存一份。

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
ORANGE = '#ea580c'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'touch 挂块：块 1 的 ref_cnt 走过 0→1→2——共享前缀不是拷贝，是引用计数',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '命中即 +1 且 ref_cnt==0 时 O(1) 出队救回；free 成对 −1、减到 0 才回 LRU 尾——中间人退租不伤共享者，最后一只手放开块才回队',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 命中主循环「挂」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96

# ---------------- 左：六个时点的演化表 ----------------
LX, LW = MX, 800
ROWS = [
    ('A 在跑（32 token）', '1', '1', '否', 'A 独占前 2 块', False),
    ('A 完成 free', '0', '0', '是', 'ref_cnt 归零但哈希仍在——块回队当驱逐候选', False),
    ('B 准入命中 + 分配', '1', '1', '否', 'touch：0→1、O(1) remove 出队救回（B 命中 32）', True),
    ('C 也进场', '2', '2', '否', '同一物理块被两请求共享——只存一份的物理基础', True),
    ('B 完成 free', '1', '1', '否', 'C 还引用——ref_cnt 2→1、不回队', False),
    ('C 完成 free', '0', '0', '是', '最后引用者放手 → 回 LRU 尾（尾段 4、2、1）', False),
]
COLS = [('时点', 206), ('块1 ref', 74), ('块2 ref', 74), ('在自由队列?', 96), ('说明', 330)]
HEAD_Y, ROW_H = LY + 44, 50
TB_H = HEAD_Y - LY + ROWS.__len__() * ROW_H + 14
lc.rect(LX, LY, LW, TB_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, LY + 22, '块 1、块 2（A 的 32-token 前缀）的引用计数演化 · B/C 各 48 token 共享前 32',
        11.5, lc.C_TXT, 'start', True, maxw=LW - 32, tag='lp:t')
cx0 = LX + 18
for (name, cwid) in COLS:
    lc.text(cx0, HEAD_Y, name, 9.2, lc.C_MUTE, 'start', True, maxw=cwid - 8, tag='th:' + name)
    cx0 += cwid
for i, (t, r1, r2, fq, note, hl) in enumerate(ROWS):
    yy = HEAD_Y + 14 + i * ROW_H
    if hl:
        lc.rect(LX + 10, yy - 13, LW - 20, ROW_H - 4, '#f0fdf4', GREEN, rx=5, sw=0.0)
    cx0 = LX + 18
    lc.text(cx0, yy + 14, t, 9.2, lc.C_TXT, 'start', True, maxw=COLS[0][1] - 8, tag='r%d:t' % i)
    cx0 += COLS[0][1]
    for j, val in enumerate((r1, r2)):
        lc.text(cx0, yy + 14, val, 10.4, GREEN if val == '2' else lc.C_TXT, 'start', True,
                maxw=COLS[1][1] - 8, tag='r%d:c%d' % (i, j))
        cx0 += COLS[1][1]
    lc.text(cx0, yy + 14, fq, 9.2, ORANGE if fq == '是' else GRAY, 'start', maxw=COLS[3][1] - 8,
            tag='r%d:f' % i)
    cx0 += COLS[3][1]
    lc.text(cx0, yy + 14, note, 8.6, '#334155', 'start', maxw=COLS[4][1] - 6, tag='r%d:n' % i)

# ---------------- 右：C 进场快照 + touch 解剖 + 队列尾段 ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
RH = TB_H
lc.rect(RX, LY, RW, RH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, LY + 22, '「C 也进场」快照：两个请求牵着同两块', 11.5, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='rp:t')
# 请求 chips
CHIP_W, CHIP_H = 150, 38
cy = LY + 36
for j, (nm, sub) in enumerate([('B · 48 token', '命中前 32 + 自有 16'), ('C · 48 token', '命中前 32 + 自有 16')]):
    xx = RX + 40 + j * (CHIP_W + 90)
    lc.rect(xx, cy, CHIP_W, CHIP_H, '#ffffff', lc.C_MUTE, rx=6, sw=1.3)
    lc.text(xx + CHIP_W / 2, cy + 16, nm, 9.6, lc.C_TXT, 'middle', True, maxw=CHIP_W - 8,
            tag='rq%d' % j)
    lc.text(xx + CHIP_W / 2, cy + 31, sub, 7.8, GRAY, 'middle', maxw=CHIP_W - 8, tag='rqs%d' % j)
# 共享块
by = LY + 94
BW2 = 132
lc.text(RX + 16, by - 8, 'B、C 各自的请求块表都指向这两块（touch 已各自 +1）', 8.2, GRAY,
        'start', maxw=RW - 32, tag='rp:mid')
for j in range(2):
    xx = RX + 60 + j * (BW2 + 56)
    lc.rect(xx, by, BW2, 54, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.4)
    lc.text(xx + BW2 / 2, by + 17, '物理块 %d' % (j + 1), 9.8, lc.C_KV_S, 'middle', True,
            maxw=BW2 - 8, tag='sb%d' % j)
    lc.text(xx + BW2 / 2, by + 34, 'ref_cnt = 2', 8.8, GREEN, 'middle', True, maxw=BW2 - 8,
            tag='sbr%d' % j)
    lc.text(xx + BW2 / 2, by + 47, '被 B、C 共同引用', 7.6, '#475569', 'middle', maxw=BW2 - 8,
            tag='sbc%d' % j)
lc.text(RX + 16, LY + 172, '共享 32-token 前缀：物理 2 块、显存 2 块（而非各存一份的', 8.6,
        '#334155', 'start', maxw=RW - 32, tag='rp:s1')
lc.text(RX + 16, LY + 188, '4 块）——共享比 2:1；k 个请求共享同一 system prompt 时', 8.6,
        '#334155', 'start', maxw=RW - 32, tag='rp:s2')
lc.text(RX + 16, LY + 204, 'ref_cnt=k、显存仍是 1 份', 8.6, '#334155', 'start', maxw=RW - 32,
        tag='rp:s3')
# touch 解剖
ty = LY + 222
lc.rect(RX + 16, ty, RW - 32, 52, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.2)
lc.text(RX + 30, ty + 20, 'touch：ref_cnt==0 → free_block_queue.remove(block)', 9.2, lc.C_KV_S,
        'start', True, maxw=RW - 60, tag='rp:tt')
lc.text(RX + 30, ty + 40, 'O(1)——侵入式双向链表摘结点，不是 list.remove 的 O(n)', 8.6,
        '#334155', 'start', maxw=RW - 60, tag='rp:ts')
# 队列尾段
qy = LY + 292
lc.text(RX + 16, qy, 'C 最后走完后的自由队列（头→尾）：', 9.2, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='rp:qt')
QCELL, QGAP = 34, 6
qx = RX + 16
for j, b in enumerate(['…', '3', '4', '2', '1']):
    hl = b in ('2', '1')
    lc.rect(qx, qy + 8, QCELL, 26, lc.C_KV_F if hl else '#ffffff', lc.C_KV_S if hl else GRAY,
            rx=4, sw=1.1, dash=(b == '…'))
    lc.text(qx + QCELL / 2, qy + 26, b, 9, lc.C_KV_S if hl else GRAY, 'middle', True,
            maxw=QCELL - 4, tag='q%d' % j)
    qx += QCELL + QGAP
lc.text(qx + 6, qy + 26, '← 驱逐先来', 8.2, ORANGE, 'start', maxw=90, tag='rp:qh')
lc.text(RX + 16, LY + 338, '尾段 4、2、1：B 的独有尾块 3 最先走；', 8.4, '#334155', 'start',
        maxw=RW - 32, tag='rp:q1')
lc.text(RX + 16, LY + 353, '共享前缀块 2、1 沉到 LRU 尾——最可复用端', 8.4, '#334155', 'start',
        maxw=RW - 32, tag='rp:q2')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + TB_H + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不变量：ref_cnt 恒等于引用该块的请求数；ref_cnt>0 的块绝不在自由队列（不可能被驱逐）',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '唯一的 +1 源是 touch（与挂块严格成对）、唯一的 −1 源是 free_blocks（与摘账成对）——两对镜像操作，无第三条路径可绕',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, dash, tcol, name in [
        ('#f0fdf4', GREEN, False, GREEN, 'touch 生效的时点 / ref_cnt=2'),
        (lc.C_KV_F, lc.C_KV_S, False, lc.C_KV_S, '物理块（KV 账本青）'),
        ('#ffffff', ORANGE, False, ORANGE, '回到自由队列（驱逐候选）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=210, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, 'B/C 各 48 token = 共享 32 + 各自 16（块 3 = B 的尾块、块 4 = C 的尾块）；自由队列头=先驱逐端',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/block_pool.py:L702-L717（touch：+1 与 ref_cnt==0 时 O(1) remove 出队）· '
        'L719-L742（free_blocks：−1 与归零入队）· vllm/v1/core/single_type_kv_cache_manager.py:L232-L289（add_local_computed_blocks 挂命中块）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（A 32 token 先跑完 free；B/C 各 48 token 依次进场）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=620, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-touch-refcount.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
