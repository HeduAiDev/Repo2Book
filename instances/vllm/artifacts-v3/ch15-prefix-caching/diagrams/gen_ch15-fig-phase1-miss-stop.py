#!/usr/bin/env python3
"""ch15 机制图 3 · phase 1 逐块查表、miss 即断（figure_spec ch15-fig-phase1-miss-stop，模板 state-table）

放大自 L0 KV 账本列（kv_column）缓存区·命中主循环——「查 → 链上走」两拍的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：phase 1 从头沿满块哈希链逐块查平面 dict、第一个 miss 即断——链式哈希保证
miss 之后必 miss、无需回溯；全命中也退一个 token 拿 logits。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；文本全 esc()。

块号口径（盲审回修后统一）：B 面板探测表行名与命中标注一律用 traces/m4.json 的池块号
（A_block_ids=[1,2,3,4]；命中 hit_block_ids=[1,2]、miss 发生在块 3 的链位）——链位 0..3
只保留在 hash0..hash3 下标里，图例注明「块 i+1 的查表键 = hash_i」，两套编号不再混排。
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
lc.text(MX, 34, '逐块查表、第一个 miss 即断：块 4 连查都不查——链式保证后面必 miss',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'phase 1 沿满块哈希链逐块 get_cached_block，islice 以 max_length//block_size 为预算；全命中也必须退 1 个 token 重算（要 logits），块对齐再回退整块',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 命中主循环「查」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96


def ruler(x0, w, y, total, segs, ticks, h=34, tick_fs=8):
    """token 标尺：segs=[(t0,t1,fill,stroke,label)]，ticks=[边界 token 号]"""
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


# ---------------- 左：场景 B（共享前 32） ----------------
LX, LW = MX, 760
lc.rect(LX, LY, LW, 356, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, LY + 22, 'B：与 A 共享前 32 token · 预算 63 → islice 只许探 63//16 = 3 块', 11.5,
        lc.C_TXT, 'start', True, maxw=LW - 32, tag='lp:t')
CH_W, PR_W, RE_W = 172, 268, 216
GAP1, GAP2 = 34, 34
ROW_Y0, ROW_H, ROW_GAP = LY + 40, 44, 12
PROBES = [
    ('块 1 · token 0-15', 'get_cached_block(hash0)', '✓ 命中', GREEN, False),
    ('块 2 · token 16-31', 'get_cached_block(hash1)', '✓ 命中', GREEN, False),
    ('块 3 · token 32-47', 'get_cached_block(hash2)', '✗ miss → break', lc.C_ABORT, False),
    ('块 4 · token 48-63', '（不探）', '链式保证必 miss', GRAY, True),
]
for i, (blk, probe, res, col, dead) in enumerate(PROBES):
    yy = ROW_Y0 + i * (ROW_H + ROW_GAP)
    px = LX + 20
    lc.rect(px, yy, CH_W, ROW_H, '#ffffff', GRAY if dead else lc.C_MUTE, rx=5, sw=1.2,
            dash=dead)
    lc.text(px + CH_W / 2, yy + ROW_H / 2 + 4, blk, 9.4, lc.C_TXT, 'middle', True, maxw=CH_W - 8,
            tag='pb%d' % i)
    qx = px + CH_W + GAP1
    if not dead:
        lc.seg(px + CH_W, yy + ROW_H / 2, qx, yy + ROW_H / 2, lc.C_MUTE, 1.4, 'std')
        lc.rect(qx, yy, PR_W, ROW_H, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.2)
        lc.text(qx + PR_W / 2, yy + ROW_H / 2 + 4, probe, 9.2, lc.C_KV_S, 'middle', True,
                maxw=PR_W - 8, tag='pr%d' % i)
        rx0 = qx + PR_W + GAP2
        lc.seg(qx + PR_W, yy + ROW_H / 2, rx0, yy + ROW_H / 2, lc.C_MUTE, 1.4, 'std')
        lc.rect(rx0, yy, RE_W, ROW_H, '#ffffff', col, rx=5, sw=1.3)
        lc.text(rx0 + RE_W / 2, yy + ROW_H / 2 + 4, res, 9.4, col, 'middle', True,
                maxw=RE_W - 8, tag='rs%d' % i)
    else:
        rx0 = qx + PR_W + GAP2
        lc.text(rx0 - GAP2 - PR_W - GAP1 + CH_W + 8, yy + ROW_H / 2 + 4,
                '—— break 后不再探：miss 之后必 miss（链式哈希的结构保证），回溯零收益', 8.6, GRAY,
                'start', maxw=LX + LW - 20 - (px + CH_W) - 8, tag='dead%d' % i)
rl_y = ROW_Y0 + 4 * (ROW_H + ROW_GAP) + 8
ruler(LX + 20, LW - 40, rl_y, 64,
      [(0, 32, '#f0fdf4', GREEN, '命中 32 token（块 1、2）', GREEN),
       (32, 64, '#fff7ed', ORANGE, '本步只 prefill 后 32 token', ORANGE)],
      [0, 16, 32, 48, 64])
sl_y = rl_y + 76
lc.text(LX + 20, sl_y, '查表次数 = 命中块数 + 1 = 3 次——与池大小无关（平面 dict O(1)）；B 省下 32/64 = 50% 的 prefill',
        9, '#334155', 'start', maxw=LW - 40, tag='lp:s')

# ---------------- 右上：场景 C（完全一致） ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
lc.rect(RX, LY, RW, 208, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, LY + 22, 'C：与 A 完全一致（64 token）· 预算 3 块全命中（块 1、2、3）', 11.5, lc.C_TXT, 'start',
        True, maxw=RW - 32, tag='cp:t')
rl2_y = LY + 40
ruler(RX + 20, RW - 40, rl2_y, 64,
      [(0, 48, '#f0fdf4', GREEN, '命中 48', GREEN),
       (48, 64, '#ffedd5', ORANGE, '连带 15 + 要 logits 的 1', ORANGE)],
      [0, 16, 32, 48, 64])
_x63 = RX + 20 + (RW - 40) * 63 / 64
lc.seg(_x63, rl2_y + 4, _x63, rl2_y + 30, lc.C_ABORT, 1.6)
for i, ln in enumerate([
        '全命中也退 1 个 token：max_cache_hit_length = num_tokens − 1',
        '（重算最后一个 token 才有 logits——注释原话）',
        'num_computed_tokens 须块对齐：63 砍到 48，白付 15 个 token',
        '重算 16 = 1（要 logits）+ 15（块对齐回退，注释自认的未来优化点）']):
    lc.text(RX + 20, rl2_y + 66 + i * 17, ln, 8.6, '#334155', 'start', maxw=RW - 40,
            tag='cp:l%d' % i)

# ---------------- 右下：场景 D（17 token） ----------------
DY = LY + 208 + 18
DH = 356 - 208 - 18
lc.rect(RX, DY, RW, DH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, DY + 22, 'D：17 token（1 满块 + 1 尾）· 预算 1 块', 11.5, lc.C_TXT, 'start',
        True, maxw=RW - 32, tag='dp:t')
rl3_y = DY + 40
ruler(RX + 20, RW - 40, rl3_y, 17,
      [(0, 16, '#f0fdf4', GREEN, '命中 16', GREEN),
       (16, 17, '#ffedd5', ORANGE, '重算 1', ORANGE)],
      [0, 8, 16, 17])
for i, ln in enumerate([
        '重算恰 1 token——退一 token 的最小损失形态',
        '（下界：再长的全命中也至少重算 1 个）']):
    lc.text(RX + 20, rl3_y + 62 + i * 17, ln, 8.6, '#334155', 'start', maxw=RW - 40,
            tag='dp:l%d' % i)

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + 356 + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不变量：phase 1 至多 max_length//block_size 次查表、第一个 miss 处停下不损失任何命中（Merkle 链保证 miss 后必 miss）',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '下界：全命中时必留至少 1 个 token 重算（max_cache_hit_length = num_tokens − 1）；命中长度被 floor 到块边界——C 例 63 砍到 48',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, tcol, name in [
        ('#f0fdf4', GREEN, GREEN, '命中的 token（免 prefill）'),
        ('#ffedd5', ORANGE, ORANGE, '本步要 prefill / 重算的 token'),
        ('#ffedd5', lc.C_ABORT, lc.C_ABORT, '必须重算的那 1 个（要 logits）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '块 1-4 = A 先跑完留下的 4 个满块（池块号，块 0 = null 块）；hash0..hash3 = 链位 0..3 的指纹——块 i+1 的查表键 = hash_i',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/single_type_kv_cache_manager.py:L682-L777（phase 1 链查 + islice 预算 + break）· '
        'vllm/v1/core/block_pool.py:L198-L213（get_cached_block：重复块取首个）· vllm/v1/core/sched/scheduler.py:L744-L766（准入查询）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（A 64 token 先跑完 free 留表）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=560, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-phase1-miss-stop.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
