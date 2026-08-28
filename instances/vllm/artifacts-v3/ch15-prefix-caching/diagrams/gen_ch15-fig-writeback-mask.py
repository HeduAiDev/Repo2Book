#!/usr/bin/env python3
"""ch15 机制图 5 · 写回只登记新满块 + block_mask（figure_spec ch15-fig-writeback-mask，模板 state-table）

放大自 L0 KV 账本列（kv_column）缓存区·命中主循环——「写回」一拍的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：写回只登记新满块：不满的尾块永不出现在登记区间里，block_mask=False 的块被跳过
——永不能服务命中的块不占哈希表。

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
GRAY = '#94a3b8'
ORANGE = '#ea580c'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '写回只登记新满块：块 2 只有 8 个 token，满了才配拥有指纹',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '登记区间 = [num_cached_blocks, num_full_blocks)——进度账单调前进、幂等；block_mask=False 的块被 continue 跳过，永不能服务命中的块不占表',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 命中主循环「写回」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96


def ruler(x0, w, y, total, segs, ticks, h=36, tick_fs=8):
    for (t0, t1, fill, stroke, label, tcol) in segs:
        xx = x0 + w * t0 / total
        ww = w * (t1 - t0) / total
        lc.rect(xx, y, ww, h, fill, stroke, rx=0, sw=1.0)
        if label:
            lc.text(xx + ww / 2, y + h / 2 + 4, label, 8.6, tcol, 'middle', True, maxw=ww - 3,
                    tag='rl:' + label)
    for t in ticks:
        xx = x0 + w * t / total
        lc.text(xx, y + h + 13, str(t), tick_fs, lc.C_MUTE, 'middle', maxw=30, tag='tk%d' % t)


# ---------------- 左：40-token prompt 的写回 ----------------
LX, LW = MX, 770
lc.rect(LX, LY, LW, 372, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, LY + 22, '40-token prompt · block_size=16 → 2 个满块 + 8 token 尾', 11.5,
        lc.C_TXT, 'start', True, maxw=LW - 32, tag='lp:t')
rl_y = LY + 42
ruler(LX + 24, LW - 48, rl_y, 40,
      [(0, 16, '#f0fdf4', GREEN, '块 0（id 1）', GREEN),
       (16, 32, '#f0fdf4', GREEN, '块 1（id 2）', GREEN),
       (32, 40, '#ffffff', GRAY, '8 token 尾', GRAY)],
      [0, 8, 16, 24, 32, 40])
# 表：三块逐项
COLS = [('块', 96), ('覆盖 token', 110), ('满块?', 62), ('入表?', 62), ('_block_hash_num_tokens', 300)]
ty = rl_y + 66
cx0 = LX + 24
for (name, cwid) in COLS:
    lc.text(cx0, ty, name, 8.8, lc.C_MUTE, 'start', True, maxw=cwid + 40, tag='th:' + name[:6])
    cx0 += cwid
TROWS = [
    ('块 0（id 1）', '0-15', '是', '是', '16——登记覆盖 16 token 的边界哈希', GREEN),
    ('块 1（id 2）', '16-31', '是', '是', '32——链上下一个边界', GREEN),
    ('块 2（id 3）', '32-39', '否', '否', '无（null）——8 token 尾，写满才有指纹', GRAY),
]
for i, (blk, cov, full, inset, num, col) in enumerate(TROWS):
    yy = ty + 16 + i * 30
    cx0 = LX + 24
    for j, val in enumerate((blk, cov, full, inset)):
        lc.text(cx0, yy + 8, val, 8.8, col if j in (2, 3) and val in ('是', '否（8 token 尾）', '否')
                else lc.C_TXT, 'start', maxw=COLS[j][1] + 40, tag='tr%d:%d' % (i, j))
        cx0 += COLS[j][1]
    lc.text(cx0, yy + 8, num, 8.6, '#334155', 'start', maxw=COLS[4][1], tag='tr%d:4' % i)
my = ty + 16 + 3 * 30 + 14
lc.rect(LX + 24, my, LW - 48, 46, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.2)
lc.text(LX + 38, my + 19, 'map 条目数 = 满块数 = 2：写回成本 O(新满块数) 次 set_block_hash + insert',
        9.2, lc.C_KV_S, 'start', True, maxw=LW - 76, tag='lp:m1')
lc.text(LX + 38, my + 37, '尾块等它写满（后续 token 到齐跨过边界）才有指纹', 8.6, '#334155',
        'start', maxw=LW - 76, tag='lp:m2')

# ---------------- 右：mask 对照 ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
lc.rect(RX, LY, RW, 372, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, LY + 22, '对照：block_mask = [True, False] 再写回一遍', 11.5, lc.C_TXT, 'start',
        True, maxw=RW - 32, tag='rp:t')
ml_y = LY + 42
ruler(RX + 24, RW - 48, ml_y, 32,
      [(0, 16, '#f0fdf4', GREEN, 'mask=True → 入表', GREEN),
       (16, 32, '#ffffff', GRAY, 'mask=False → 跳过', GRAY)],
      [0, 8, 16, 24, 32])
my2 = ml_y + 66
lc.rect(RX + 24, my2, RW - 48, 54, '#fff7ed', ORANGE, rx=6, sw=1.2)
lc.text(RX + 38, my2 + 21, 'map 条目数 0 → 1：两个满块只进一个', 9.4, ORANGE, 'start', True,
        maxw=RW - 76, tag='rp:m1')
lc.text(RX + 38, my2 + 41, '被掩的块连表都不进——零哈希、零条目', 8.6, '#334155', 'start',
        maxw=RW - 76, tag='rp:m2')
ay = my2 + 70
for i, ln in enumerate([
        '为什么允许掩掉？有些块永不能服务命中：',
        '· SWA 窗外的块——窗外状态后续注意力不读',
        '· Mamba 对齐组的不可复用块（对齐规则内）',
        '它们占表只白费：哈希表内存 + 驱逐时',
        '的摘除成本——从源头不登记。']):
    lc.text(RX + 24, ay + i * 17, ln, 8.6, '#334155', 'start', maxw=RW - 48, tag='rp:a%d' % i)
py = ay + 5 * 17 + 10
lc.rect(RX + 24, py, RW - 48, 50, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
lc.text(RX + 38, py + 20, '幂等闸：cache_blocks 开头 num_cached_blocks >=', 8.6, lc.C_TXT,
        'start', True, maxw=RW - 76, tag='rp:p1')
lc.text(RX + 38, py + 38, 'num_full_blocks 即 return——同请求后续 chunk 只登记增量', 8.6,
        '#334155', 'start', maxw=RW - 76, tag='rp:p2')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + 372 + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不变量：只有新满块入表，且每块至多登记一次（num_cached_blocks 进度账单调前进）；入表的块 ⇔ 满且非 null 且可达',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '不满尾块永不在登记区间（num_full_blocks = num_tokens // block_size 向下取整）——「等写满」不是重算，是等下一个满块边界',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, dash, tcol, name in [
        ('#f0fdf4', GREEN, False, GREEN, '满块 · 登记入表'),
        ('#ffffff', GRAY, True, GRAY, '不满尾块 / 被掩块 · 不入表'),
        ('#fff7ed', ORANGE, False, ORANGE, 'mask 对照的结果')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=180, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '_block_hash_num_tokens = 该条目哈希盖住的 token 上界（链式边界的里程碑）',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/block_pool.py:L225-L342（cache_full_blocks：登记区间 + 进度账 + null/mask continue）· '
        'vllm/v1/core/single_type_kv_cache_manager.py:L427-L477（cache_blocks 写回入口 · reachable_block_mask）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（40 token · 2 满块 + 8 尾；mask 对照 0→1）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=640, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-writeback-mask.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
