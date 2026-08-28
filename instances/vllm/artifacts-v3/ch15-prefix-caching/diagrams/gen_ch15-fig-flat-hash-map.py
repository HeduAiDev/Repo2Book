#!/usr/bin/env python3
"""ch15 机制图 2 · 平面哈希表（figure_spec ch15-fig-flat-hash-map，模板 layout）

放大自 L0 KV 账本列（kv_column）缓存区·存储面——「平面哈希表」一格的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：前缀缓存的查找表是一个平面 dict——键是 hash+group_id 打包的 bytes、值是块或
{block_id: block}，没有树节点；同键第二块起合并成内层 dict（故意不去重，保块表 append-only）。

数字全部取自 figure_spec.numbers（pin 源码字面量）。坐标由常量/循环计算；文本全 esc()。
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

# ---------------- 标题区 ----------------
lc.text(MX, 34, '查找表就是一个平面 dict：键 = 哈希 32 字节 + 组号 4 字节——没有树节点',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'hash_block_tokens 产出的块哈希拼上 4 字节组号就是键，一次哈希定位 O(1)、零节点对象分配；同键复本摞在同一格，值退化成内层 dict',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 存储面'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96

# ---------------- 左：键的构造（字节条） ----------------
LX, LW = MX, 420
CELL, CELL_H = 9, 36
STRIP_X, STRIP_Y = LX + 40, LY + 66
lc.rect(LX, LY, LW, 348, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, LY + 22, '键的构造：BlockHashWithGroupId（bytes）', 11.5, lc.C_TXT, 'start', True,
        maxw=LW - 32, tag='lp:t')
for i in range(36):
    xx = STRIP_X + i * CELL
    if i < 32:
        lc.rect(xx, STRIP_Y, CELL, CELL_H, lc.C_KV_F, lc.C_KV_S, rx=0, sw=0.8)
    else:
        lc.rect(xx, STRIP_Y, CELL, CELL_H, '#ffedd5', lc.C_ENG_S, rx=0, sw=0.8)
# 括线 + 标注
by = STRIP_Y + CELL_H + 10
lc.seg(STRIP_X, by, STRIP_X + 32 * CELL - 1, by, lc.C_KV_S, 1.1)
for ex in (STRIP_X, STRIP_X + 32 * CELL - 1):
    lc.seg(ex, by - 4, ex, by + 4, lc.C_KV_S, 1.1)
lc.text(STRIP_X + 16 * CELL, by + 15, '32 字节 · sha256 块哈希', 8.8, lc.C_KV_S, 'middle', True,
        maxw=180, tag='strip:hash')
lc.seg(STRIP_X + 32 * CELL, by, STRIP_X + 36 * CELL - 1, by, lc.C_ENG_S, 1.1)
for ex in (STRIP_X + 32 * CELL, STRIP_X + 36 * CELL - 1):
    lc.seg(ex, by - 4, ex, by + 4, lc.C_ENG_S, 1.1)
lc.text(STRIP_X + 34 * CELL - 8, by + 15, '4 字节 · group_id', 8.8, lc.C_ENG_S, 'middle',
        True, maxw=160, tag='strip:gid')
lc.text(STRIP_X + 34 * CELL - 8, by + 30, '（big-endian · 注意力组号）', 8, lc.C_ENG_S, 'middle',
        maxw=160, tag='strip:gid2')
lc.text(STRIP_X, STRIP_Y - 10, 'make_block_hash_with_group_id：block_hash + group_id.to_bytes(4, "big")', 8.2,
        '#475569', 'start', maxw=LW - 56, tag='strip:mk')
# 产出箭头 → dict
OUT_Y = LY + 250
lc.text(LX + 16, OUT_Y, '把「指纹 + 组号」拍成一个 bytes 当键——', 9, '#334155', 'start',
        maxw=LW - 32, tag='lp:o1')
lc.text(LX + 16, OUT_Y + 17, '不同注意力组的同指纹前缀天然分格（语义隔离）', 9, '#334155', 'start',
        maxw=LW - 32, tag='lp:o2')
lc.rect(LX + 16, OUT_Y + 34, LW - 32, 52, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
lc.text(LX + 30, OUT_Y + 34 + 20, '哈希随请求带来、账本只读不写', 8.6, lc.C_MUTE, 'start',
        maxw=LW - 60, tag='lp:o3')
lc.text(LX + 30, OUT_Y + 34 + 39, '（生成在请求侧、登记在 BlockPool）', 8.6, lc.C_MUTE, 'start',
        maxw=LW - 60, tag='lp:o4')

# ---------------- 中：平面 dict 本体 ----------------
DX, DW = LX + LW + 24, 500
lc.rect(DX, LY, DW, 348, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(DX + 16, LY + 22, '_cache: dict[键 → KVCacheBlock | dict[int, KVCacheBlock]]', 11,
        lc.C_TXT, 'start', True, maxw=DW - 32, tag='dp:t')
KEY_W, VAL_W, VOFF = 118, 296, 10
ROW_Y0, ROW_H, ROW_GAP = LY + 44, 62, 24
ENTRIES = [
    ('键 a', [('block 7', False)], '①'),
    ('键 b', [('block 3', False), ('block 9', False)], '②③'),
    ('键 c', [('block 5', False)], ''),
]
for i, (kname, vals, badge) in enumerate(ENTRIES):
    yy = ROW_Y0 + i * (ROW_H + ROW_GAP)
    lc.rect(DX + VOFF, yy, KEY_W, ROW_H, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.2)
    lc.text(DX + VOFF + KEY_W / 2, yy + ROW_H / 2 + 4, kname, 10, lc.C_KV_S, 'middle', True,
            maxw=KEY_W - 8, tag='k%d' % i)
    if badge:
        lc.text(DX + VOFF + KEY_W - 12, yy + 16, badge, 9, lc.C_ENG_S, 'middle', True,
                maxw=30, tag='kb%d' % i)
    vx = DX + VOFF + KEY_W + 14
    lc.seg(DX + VOFF + KEY_W, yy + ROW_H / 2, vx, yy + ROW_H / 2, lc.C_MUTE, 1.4, 'std')
    vw_one = (VAL_W - (len(vals) - 1) * 8) / len(vals)
    if len(vals) > 1:
        lc.rect(vx - 5, yy - 5, VAL_W + 10, ROW_H + 10, 'none', lc.C_ENG_S, rx=6, sw=1.1,
                dash=True)
        lc.text(vx + VAL_W / 2, yy - 10, '内层 dict：同键复本摞同一格', 8.4, lc.C_ENG_S, 'middle',
                True, maxw=VAL_W, tag='inner%d' % i)
    for j, (v, hl) in enumerate(vals):
        xx = vx + j * (vw_one + 8)
        lc.rect(xx, yy, vw_one, ROW_H, '#ffffff', GRAY, rx=5, sw=1.1)
        lc.text(xx + vw_one / 2, yy + ROW_H / 2 + 4, v, 9.4, lc.C_TXT, 'middle', True,
                maxw=vw_one - 8, tag='v%d%d' % (i, j))
ny = ROW_Y0 + 3 * (ROW_H + ROW_GAP) + 12
lc.text(DX + 16, ny, 'get_one_block（键）→ 命中时任取一块 next(iter(...))——', 8.8, GREEN,
        'start', True, maxw=DW - 32, tag='dp:g1')
lc.text(DX + 16, ny + 17, '重复块之间内容相同，借谁都一样', 8.8, '#334155', 'start', maxw=DW - 32,
        tag='dp:g2')

# 键条 → dict 的连接箭头
lc.seg(LX + LW, LY + 160, DX, LY + 160, lc.C_KV_S, 1.8, 'std')
lc.text((LX + LW + DX) / 2, LY + 150, '键', 9, lc.C_KV_S, 'middle', True, maxw=30, tag='link:k')

# ---------------- 右：对照与取舍 ----------------
RX, RW = DX + DW + 24, BXR - (DX + DW + 24)
RH = 348
lc.rect(RX, LY, RW, RH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, LY + 22, 'radix 树迷思在此澄清', 11.5, lc.C_TXT, 'start', True, maxw=RW - 32,
        tag='rp:t')
ry = LY + 38
lc.rect(RX + 16, ry, RW - 32, 74, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
lc.text(RX + 30, ry + 19, '没有树节点：全仓 v1 core 检索 radix', 9.2, lc.C_TXT, 'start', True,
        maxw=RW - 60, tag='rp:m1')
lc.text(RX + 30, ry + 38, '零命中。前缀查找不靠指针跳转——', 8.6, '#334155', 'start', maxw=RW - 60,
        tag='rp:m2')
lc.text(RX + 30, ry + 56, '就是一次哈希定位，零节点对象分配/GC', 8.6, '#334155', 'start',
        maxw=RW - 60, tag='rp:m3')
ty = ry + 88
lc.text(RX + 16, ty, 'insert 的三段（block_pool.py:L88-L104）', 9.6, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='rp:it')
STEPS = ['① 表无此键 → 值就是单块', '② 同键再来一块 → 旧值+新块并成内层 dict',
         '③ 已是 dict → 直插（不查重）']
for i, s in enumerate(STEPS):
    lc.text(RX + 28, ty + 20 + i * 17, s, 8.6, '#334155', 'start', maxw=RW - 48, tag='rp:s%d' % i)
qy = ty + 20 + 3 * 17 + 8
lc.rect(RX + 16, qy, RW - 32, 76, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.2)
lc.text(RX + 30, qy + 19, '取舍（NOTE #1 白纸黑字）：故意不去重', 9.4, lc.C_KV_S, 'start', True,
        maxw=RW - 60, tag='rp:q1')
lc.text(RX + 30, qy + 38, '换来已发出的 block_id 永不改变、', 8.6, '#334155', 'start',
        maxw=RW - 60, tag='rp:q2')
lc.text(RX + 30, qy + 56, '请求块表只追加不重写（worker 不用回滚）', 8.6, '#334155', 'start',
        maxw=RW - 60, tag='rp:q3')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + 348 + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '平面表的代价与收益同源：键唯一 ⇒ 查找 O(1) 与池大小无关；值允许复本 ⇒ 块表 append-only（NOTE #1）+ union 类型省内层 dict 的 GC（NOTE #2）',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '已缓存的块 = 满块：要么被在跑请求引用（ref_cnt>0），要么在自由队列里当驱逐候选——表与队列共同构成「缓存」的全部状态',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, tcol, name in [
        (lc.C_KV_F, lc.C_KV_S, lc.C_KV_S, '哈希字节（32 字节）'),
        ('#ffedd5', lc.C_ENG_S, lc.C_ENG_S, '组号字节（4 字节）'),
        ('#ffffff', GREEN, GREEN, '读路径（查找/任取）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=180, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '块 3/5/7/9 为例示块 id（insert 语义的演示数据，非实跑账本）；键 a/b/c = 三个不同前缀指纹',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_utils.py:L57-L66（make_block_hash_with_group_id：组号以 4 字节 big-endian 追加进键）· '
        'vllm/v1/core/block_pool.py:L33-L53（类文档 NOTE #1 不去重 / NOTE #2 union 省 GC）· L61-L72（get_one_block 任取）· L88-L104（insert 三段）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '行号基线 vLLM v0.27.1', 8.2, lc.C_FAINT, 'start', maxw=300, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-flat-hash-map.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
