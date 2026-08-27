#!/usr/bin/env python3
"""ch13 机制图 2 · 块的身份证与池的出生（figure_spec ch13-fig-block-id-card，模板 layout）

放大自 L0『调度 · 显存账本』列（kv_column）中 KVCacheManager/BlockPool 框内的块
元数据层——即本章 L2 章图北行『KVCacheBlock · 块的身份证』框的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：池的出生三件事 + 一张身份证：对象数组一次预构、自由队列整串互串、
null_block 从队头 popleft 占掉 block_id=0 贴封条——此后块 id 从 1 起出租，
且 get_usage 的分母永远减 1 记 null。

数字全部取自 figure_spec.numbers（配套精简版 host 实测：6 块池 ids [0..5]、
块 1 七字段快照、哨兵 block_id=−1、usage = 1 − 3/5 = 0.4 分母减 null）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '块的身份证与池的出生：七字段一次预构，0 号当场贴封条、永不出租',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'KVCacheBlock（@dataclass(slots=True)）七个字段——五个是本章主角（编号/租客数/前后邻居/封条位），'
                '两个哈希账位 ch15 才启用，本章当它空着不动',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L2 章图北行「KVCacheBlock · 块的身份证」框 · L0：显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：放大的身份证卡片 ----------------
CX, CY, CWD, HDR = 190, 92, 560, 34
FIELDS = [
    ('block_id', '= 1', '编号：块池的合成顺序整数（物理页另图展开）', True),
    ('ref_cnt', '= 0', '租客数：分配 +1 / 释放 −1，归零才回自由队列', True),
    ('_block_hash', '= 空', '缓存账位：链式哈希 ch15 才启用，本章恒空', False),
    ('_block_hash_num_tokens', '= 空', '缓存账位：块内部分命中要用，ch15 才启用', False),
    ('prev_free_block', '= −1', '前邻居：此处指队头哨兵——块 1 是队列第一个真实块', True),
    ('next_free_block', '= 2', '后邻居：指块 2——链表就长在这两个字段上', True),
    ('is_null', '= False', '封条位：null 块专属（→ ch14），普通块恒 False', True),
]
ROW_H = 44
CH_ = HDR + len(FIELDS) * ROW_H + 12
lc.rect(CX, CY, CWD, CH_, '#ffffff', lc.C_KV_S, rx=10, sw=2.0)
lc.rect(CX, CY, CWD, HDR, lc.C_KV_F, lc.C_KV_S, rx=10, sw=2.0)
lc.rect(CX, CY + HDR - 12, CWD, 12, lc.C_KV_F, 'none', rx=0, sw=0)
lc.text(CX + 18, CY + 23, 'KVCacheBlock · 块 1 的身份证（快照）', 11.5, lc.C_KV_S, 'start', True,
        maxw=CWD - 220, tag='card:t')
lc.text(CX + CWD - 16, CY + 23, 'kv_cache_utils.py:L118-L138', 8.5, lc.C_FAINT, 'end',
        maxw=240, tag='card:f')
for i, (name, val, note, hot) in enumerate(FIELDS):
    ry = CY + HDR + i * ROW_H
    lc.rect(CX + 8, ry + 3, CWD - 16, ROW_H - 6, lc.C_KV_F if hot else '#f8fafc', 'none', rx=6, sw=0)
    col = lc.C_TXT if hot else '#94a3b8'
    lc.text(CX + 22, ry + 19, name, 10.5, col, 'start', True, maxw=230, tag='fd%d' % i)
    lc.text(CX + 254, ry + 19, val, 10.5, col, 'start', True, maxw=80, tag='fv%d' % i)
    lc.text(CX + 330, ry + 19, note, 8.5, '#64748b' if hot else '#b6c2d2', 'start',
            maxw=CWD - 348, tag='fn%d' % i)
CARD_BOT = CY + CH_

# 卡片左缘的 prev/next 挂环（虚线连向左右邻居缩略图）
RING_R = 7
prev_y = CY + HDR + 4 * ROW_H + 12
next_y = CY + HDR + 5 * ROW_H + 12
for ry, lab in [(prev_y, 'prev'), (next_y, 'next')]:
    lc.circle(CX + 4, ry, RING_R, lc.C_KV_S, 1.8, dash=False)
    lc.text(CX + 2, ry - 16, lab, 8, lc.C_KV_S, 'middle', maxw=60, tag='ring' + lab)

# 左邻居：fake_head 哨兵（六边形）
HX, HY = 88, prev_y - 13          # 六边形右顶点 (HX+22, HY+13) 与 prev 挂环同高
lc.parrow([(HX + 24, HY + 13), (CX - RING_R - 3, prev_y)], lc.C_KV_S, 1.4, None, dash=True)
hexpts = ' '.join(f'{HX + dx:.1f},{HY + dy:.1f}' for dx, dy in
                  [(-22, 0), (-11, 13), (11, 13), (22, 0), (11, -13), (-11, -13)])
lc.ELEMS.append(((HX - 24, HY - 15, HX + 24, HY + 15),
                 f'<polygon points="{hexpts}" fill="#f1f5f9" stroke="#64748b" stroke-width="1.4"/>'))
lc.text(HX, HY + 4, 'fake_head', 8, '#475569', 'middle', maxw=44, tag='hx1')
lc.text(HX, HY + 32, 'block_id = −1', 7.5, '#64748b', 'middle', maxw=70, tag='hx2')
lc.text(HX, HY - 24, '队头哨兵', 8, lc.C_MUTE, 'middle', maxw=80, tag='hx0')

# 右邻居：块 2 缩略卡（next 挂环同高）
NX, NY = CX + CWD + 44, next_y - 15
lc.parrow([(CX + CWD + 3, next_y), (NX - 3, next_y)], lc.C_KV_S, 1.4, None, dash=True)
lc.rect(NX, NY, 96, 30, '#ffffff', lc.C_KV_S, rx=5, sw=1.3)
lc.text(NX + 48, NY + 13, '块 2', 9.5, lc.C_KV_S, 'middle', True, maxw=88, tag='nb')
lc.text(NX + 48, NY + 26, '下一任邻居', 7.5, '#64748b', 'middle', maxw=88, tag='nbs')

# ---------------- 右：池的出生三步 ----------------
BX_, BW_ = 950, BXR - 950
lc.text(BX_, CY + 4, '池的出生（BlockPool.__init__ 三件事）', 11.5, lc.C_TXT, 'start', True,
        maxw=BW_, tag='birth:t')
MINI_W, MINI_H, MINI_GAP, BOX_H = 58, 40, 14, 124
STEPS = [
    ('① 一次预构', '对象数组：KVCacheBlock(idx) for idx in range(6)——运行期零构造',
     'ids [0,1,2,3,4,5] · 6 块一次全发'),
    ('② 整串互串', 'FreeKVCacheBlockQueue(blocks)：相邻块 prev/next 互指，整串成链',
     'fake_head(−1) ↔ 0 ↔ 1 ↔ 2 ↔ 3 ↔ 4 ↔ 5 ↔ fake_tail(−1)'),
    ('③ 队头摘走 0 号', 'popleft() → null_block：block_id=0、贴 is_null 封条、ref_cnt 不维护',
     '此后首租从 1 号开始 · 自由队列 [1,2,3,4,5] · num_free = 5'),
]
sy0, pitch = CY + 26, BOX_H + 14
for si, (title, note, tail) in enumerate(STEPS):
    sy = sy0 + si * pitch
    last = si == len(STEPS) - 1
    lc.rect(BX_, sy, BW_, BOX_H, '#ffffff', lc.C_ABORT if last else '#94a3b8', rx=7, sw=1.3)
    lc.text(BX_ + 14, sy + 20, title, 10, lc.C_ABORT if last else lc.C_TXT, 'start', True,
            maxw=220, tag='bs' + title[:4])
    lc.text(BX_ + 14, sy + 38, note, 8.5, '#64748b', 'start', maxw=BW_ - 28, tag='bn' + title[:4])
    my = sy + 56
    for k in range(6):
        mx = BX_ + 18 + k * (MINI_W + MINI_GAP)
        if last and k == 0:
            lc.rect(mx, my - 14, MINI_W, MINI_H, '#fef2f2', lc.C_ABORT, rx=5, sw=1.6, dash=True)
            lc.seg(mx + 8, my - 6, mx + MINI_W - 8, my + MINI_H - 22, lc.C_ABORT, 1.4)
            lc.seg(mx + 8, my + MINI_H - 22, mx + MINI_W - 8, my - 6, lc.C_ABORT, 1.4)
            lc.text(mx + MINI_W / 2, my + 2, '0', 11, lc.C_ABORT, 'middle', True, tag='mz0')
            lc.text(mx + MINI_W / 2, my + 20, 'null 封条', 6.5, lc.C_ABORT, 'middle',
                    maxw=MINI_W - 4, tag='mzs0')
            lc.text(mx + MINI_W + 8, my - 2, '← popleft() 永不出租', 8, lc.C_ABORT, 'start',
                    maxw=150, tag='pull')
        else:
            free_after = last                      # ③ 之后 1..5 在自由队列里
            lc.rect(mx, my, MINI_W, MINI_H, '#ffffff' if free_after else lc.C_KV_F,
                    '#cbd5e1' if free_after else lc.C_KV_S, rx=5, sw=1.1)
            lc.text(mx + MINI_W / 2, my + 17, str(k), 11, '#64748b' if free_after else lc.C_KV_S,
                    'middle', True, tag='mk%d%d' % (si, k))
        if k < 5 and not (last and k == 0):
            lc.seg(mx + MINI_W + 2, my + MINI_H / 2, mx + MINI_W + MINI_GAP - 2, my + MINI_H / 2,
                   '#94a3b8', 1.1)
    lc.text(BX_ + 18, my + MINI_H + 18, tail, 8.5, lc.C_MUTE, 'start', maxw=BW_ - 30,
            tag='bt%d' % si)
BIRTH_BOT = sy0 + 2 * pitch + BOX_H

# ---------------- 底部：usage 口径 + 哨兵注记 ----------------
UY = max(CARD_BOT, BIRTH_BOT) + 22
lc.rect(MX, UY, 700, 66, '#ffffff', lc.C_KV_S, rx=7, sw=1.3)
lc.text(MX + 16, UY + 22, '观测口径 get_usage：分配 2 块后 usage = 1 − 3/5 = 0.4', 10, lc.C_KV_S,
        'start', True, maxw=660, tag='us:1')
lc.text(MX + 16, UY + 42, '分母 5 = 6 − 1：永远减掉 null 块——vLLM 运行日志「GPU KV cache usage」的出处',
        8.5, lc.C_MUTE, 'start', maxw=660, tag='us:2')
lc.text(MX + 16, UY + 58, '分配 2 块（块 1、块 2，ref_cnt 各 1）后空闲 3、持有 2', 8.5, lc.C_MUTE,
        'start', maxw=660, tag='us:3')
QX = MX + 740
lc.rect(QX, UY, BXR - QX, 66, '#ffffff', '#94a3b8', rx=7, sw=1.2, dash=True)
lc.text(QX + 16, UY + 22, '哨兵的用处：fake_head / fake_tail 的 block_id = −1', 10, lc.C_TXT, 'start',
        True, maxw=BXR - QX - 30, tag='sn:1')
lc.text(QX + 16, UY + 42, '真实块恒有 prev 和 next——摘谁都不用判边界（消掉边界分支）', 8.5, lc.C_MUTE,
        'start', maxw=BXR - QX - 30, tag='sn:2')
lc.text(QX + 16, UY + 58, 'null 注释原话：needs special care to avoid freeing it', 8.5, lc.C_MUTE,
        'start', maxw=BXR - QX - 30, tag='sn:3')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = UY + 92
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, '本章主角字段（五行）', 8.5, lc.C_TXT, 'start', maxw=170, tag='leg:hot')
lx += 26 + lc.tw('本章主角字段（五行）', 8.5) + 20
lc.rect(lx, LEG_Y - 9, 20, 13, '#f8fafc', '#cbd5e1', rx=3, sw=1.0)
lc.text(lx + 26, LEG_Y + 1, 'ch15 缓存账位（本章恒空）', 8.5, lc.C_TXT, 'start', maxw=190, tag='leg:dim')
lx += 26 + lc.tw('ch15 缓存账位（本章恒空）', 8.5) + 20
lc.rect(lx, LEG_Y - 11, 20, 15, '#fef2f2', lc.C_ABORT, rx=3, sw=1.1, dash=True)
lc.text(lx + 26, LEG_Y + 1, 'null_block（贴封条 · 永不出租）', 8.5, lc.C_TXT, 'start', maxw=210,
        tag='leg:null')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/core/kv_cache_utils.py:L118-L138（KVCacheBlock 七字段 dataclass）· '
        'L184-L234（FreeKVCacheBlockQueue 与哨兵 docstring）· vllm/v1/core/block_pool.py:L162-L191（池构造与 null_block）· '
        'L807-L818（get_usage 分母减 null）', 8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '七字段快照与 usage 读数取自配套精简版 host 实跑（6 块池构造期观测）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 66
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-block-id-card.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
