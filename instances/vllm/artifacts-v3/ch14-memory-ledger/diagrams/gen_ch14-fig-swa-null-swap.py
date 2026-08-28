#!/usr/bin/env python3
"""ch14 机制图 8 · SWA 窗外回收与 null 占位（figure_spec ch14-fig-swa-null-swap，模板 before-after）

放大自 L0 KV 账本列（池内动态）——本章 L2 章图南行站11「SWA 窗外回收 · null 占位」
的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：窗外整块 free 归池、原位换 null_block 占位：块表第 i 块仍对应第 i×block_size
个 token，位置对齐不断裂、注意力照表读。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；
文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '窗外整块归池、原位换 NULL：座位号不挪——第 i 块永远是第 i×4 个 token 起',
        16.5, lc.C_TXT, 'start', True, maxw=990, tag='title')
lc.text(MX, 58, 'remove_skipped_blocks 把 [0, num_skipped_blocks) 的整块逆序 free、原位赋 null_block（遇 null 早停）；表长只增不减、按号入座的对齐不断裂',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列池内动态 · L2 南行站11'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

PY0 = 92

# ---------------- 左：逐拍推进（window 4 · bs 4 · 16 token · 池 8） ----------------
LX, LW = MX, 840
lc.rect(LX, PY0, LW, 470, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, PY0 + 22, '逐拍推进：window 4 · block_size 4 · 16-token 请求持 4 块 · 池 8', 11.5,
        lc.C_TXT, 'start', True, maxw=LW - 32, tag='lp:t')
# 状态数据：(标签, processed, 各块状态 0=NULL 1=浅青 2=实心, 窗外token, 实持, free, 注记)
STATES = [
    ('初始', '0', [2, 1, 1, 1], '0', '4', '3', '窗口 0-3'),
    ('推进一', '7', [0, 2, 1, 1], '4', '3', '4', '窗外 4 token → b1 归池'),
    ('推进二', '11', [0, 0, 2, 1], '8', '2', '5', '窗外 8 token → b2 归池'),
    ('推进三', '15', [0, 0, 0, 2], '12', '1', '6', '窗外 12 token → b3 归池'),
]
CELL_W, CELL_H, CELL_GAP = 130, 56, 8
CX0 = LX + 150
ROW_GAP = 46
for si, (name, proc, cells, skip, held, free, note) in enumerate(STATES):
    ry = PY0 + 46 + si * (CELL_H + ROW_GAP)
    lc.text(LX + 16, ry + 22, name, 10.2, lc.C_TXT, 'start', True, maxw=80, tag='st%d:n' % si)
    lc.text(LX + 16, ry + 40, 'processed %s' % proc, 8.4, lc.C_MUTE, 'start', maxw=110,
            tag='st%d:p' % si)
    for ci, st in enumerate(cells):
        xx = CX0 + ci * (CELL_W + CELL_GAP)
        if st == 0:
            lc.rect(xx, ry, CELL_W, CELL_H, '#ffffff', GRAY, rx=5, sw=1.2, dash=True)
            lc.text(xx + CELL_W / 2, ry + 26, 'NULL', 11, GRAY, 'middle', True,
                    maxw=CELL_W - 10, tag='c%d%d' % (si, ci))
        else:
            fill = lc.C_KV_S if st == 2 else lc.C_KV_F
            tcol = '#ffffff' if st == 2 else lc.C_KV_S
            lc.rect(xx, ry, CELL_W, CELL_H, fill, lc.C_KV_S, rx=5, sw=1.2)
            lc.text(xx + CELL_W / 2, ry + 26, 'b%d' % (ci + 1), 11.5, tcol, 'middle', True,
                    maxw=CELL_W - 10, tag='c%d%d' % (si, ci))
        lc.text(xx + CELL_W / 2, ry + 46, 'tokens %d-%d' % (ci * 4, ci * 4 + 3), 7.6,
                '#ffffff' if st == 2 else '#64748b', 'middle', maxw=CELL_W - 8,
                tag='ct%d%d' % (si, ci))
    rx0 = CX0 + 4 * CELL_W + 3 * CELL_GAP + 14
    lc.text(rx0, ry + 22, '实持 %s · 池 free %s' % (held, free), 9.2, lc.C_TXT, 'start', True,
            maxw=LX + LW - rx0 - 12, tag='r%d:ro' % si)
    lc.text(rx0, ry + 40, note, 8.2, '#64748b', 'start', maxw=LX + LW - rx0 - 12,
            tag='r%d:note' % si)
    if si < 3:
        ay = ry + CELL_H
        lc.seg(LX + 60, ay + 3, LX + 60, ay + ROW_GAP - 3, lc.C_KV_S, 1.8, 'std')
        lc.text(LX + 70, ay + ROW_GAP / 2 + 3, '推进', 8.2, lc.C_MUTE, 'start', maxw=40,
                tag='adv%d' % si)
LB = PY0 + 470

# ---------------- 右：稳态 + 对照 ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
lc.rect(RX, PY0, RW, 470, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, PY0 + 22, '稳态：64-token · window 8 · 算到 60', 11.5, lc.C_TXT, 'start',
        True, maxw=RW - 32, tag='rp:t')
SCELL, SGAP = 28, 2
sx0 = RX + 16
sy = PY0 + 70
for i in range(16):
    xx = sx0 + i * (SCELL + SGAP)
    if i < 13:
        lc.rect(xx, sy, SCELL, 44, '#ffffff', GRAY, rx=3, sw=1.0, dash=True)
        if i in (0, 6, 12):
            lc.text(xx + SCELL / 2, sy + 27, 'N', 8, GRAY, 'middle', True, maxw=SCELL - 4,
                    tag='sc%d' % i)
    else:
        lc.rect(xx, sy, SCELL, 44, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
        lc.text(xx + SCELL / 2, sy + 27, str(i * 4), 8, lc.C_KV_S, 'middle', True,
                maxw=SCELL - 4, tag='sc%d' % i)
# 13×NULL 大括号线（简单画一条横线 + 标注）
brace_y = sy + 56
lc.seg(sx0, brace_y, sx0 + 13 * (SCELL + SGAP) - 2, brace_y, GRAY, 1.2)
for endx in (sx0, sx0 + 13 * (SCELL + SGAP) - 2):
    lc.seg(endx, brace_y - 4, endx, brace_y + 4, GRAY, 1.2)
lc.text(sx0 + 13 * (SCELL + SGAP) / 2, brace_y + 14, '13 块原位 NULL（窗外 53 token）', 8.4,
        '#475569', 'middle', maxw=RW - 32, tag='rp:brace')
lc.text(sx0 + 13 * (SCELL + SGAP) + 8, brace_y + 14, '← 实持 3 块', 8.6, lc.C_KV_S,
        'start', True, maxw=RW - 16 - (13 * (SCELL + SGAP)) - 12, tag='rp:held')
ly = brace_y + 36
for i, ln in enumerate([
        '64-token 序列算到第 60 个 token：',
        'skipped = max(0, 60 − 8 + 1) = 53 → 53 // 4 = 13 整块',
        '实持 3 块（tokens 52-63 · 窗口 53-60 在其中）',
        '池 64 块 · free 回升到 60——回收是连续小步，不是一次性']):
    lc.text(RX + 16, ly + i * 19, ln, 8.8, '#334155', 'start', maxw=RW - 30,
            tag='rp:l%d' % i)
# 对照盒
cby = ly + 4 * 19 + 16
lc.rect(RX + 16, cby, RW - 32, 108, '#f8fafc', GRAY, rx=7, sw=1.1, dash=True)
lc.text(RX + 28, cby + 18, '对照：另外两种 spec 的回收行为', 9.5, lc.C_TXT, 'start', True,
        maxw=RW - 56, tag='cp:t')
lc.text(RX + 28, cby + 38, 'full attention：恒 0 从不回收——', 8.6, '#334155', 'start',
        maxw=RW - 56, tag='cp:l1')
lc.text(RX + 28, cby + 55, '  全长持有到请求结束（本图左侧行为的对立面）', 8.6, '#334155',
        'start', maxw=RW - 56, tag='cp:l2')
lc.text(RX + 28, cby + 74, 'chunked-local：按 chunk 对齐收整 chunk——', 8.6, '#334155',
        'start', maxw=RW - 56, tag='cp:l3')
lc.text(RX + 28, cby + 91, '  computed 13→收 8 · 8→收 8 · 7→收 0', 8.6, '#334155',
        'start', maxw=RW - 56, tag='cp:l4')

# ---------------- 底部不变量条（全宽） ----------------
BY = PY0 + 492
lc.rect(MX, BY, BXR - MX, 60, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '位置不变量：块表第 i 项 ↔ 第 i×block_size 起的 block_size 个 token——回收只把 block_id 原位换成 null，表长与下标集合不变', 9.6,
        lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '凡被收走的块，其 token 全部满足 pos < computed − window + 1（窗外、后续 attention 不再读）——按表读的 kernel 语义不变', 9,
        '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 82
lx = MX
for fill, stroke, dash, tcol, name in [
        (lc.C_KV_S, lc.C_KV_S, False, '#ffffff', '当前窗口内'),
        (lc.C_KV_F, lc.C_KV_S, False, lc.C_KV_S, '已持有（未进窗 / 未写）'),
        ('#ffffff', GRAY, True, GRAY, '已回收（null_block 占位）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=180, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '左 b1-b4 = 块 id 示例；右列数字 = 该格起始 token 号（第 i 格 = 第 i×4 个 token 起）；「下一 token 的窗口」= computed .. computed+window−1',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/single_type_kv_cache_manager.py:L622-L659（remove_skipped_blocks 逆序 free + 原位 null）· '
        'L1057-L1083（get_num_skipped_tokens，docstring 自带 ASCII 例）· vllm/v1/core/block_pool.py:L187-L191（null 特判）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1', 8.2,
        lc.C_FAINT, 'start', maxw=500, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-swa-null-swap.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
