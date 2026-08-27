#!/usr/bin/env python3
"""ch12 机制图 4 · cache_blocks 差值转正的前后对照（figure_spec ch12-fig-block-convert，模板 before-after）

放大自 L0 循环框（loop_box）调度器账本位与 kv_column（显存账本列）的交界——即本章
L2 章图 center 拍片 ⑥『真记账 · 占位-1』向 KVCacheManager.cache_blocks 那一次调用的
前后对照。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：交货前 cache_blocks 只敢按差值转正——乐观计数 computed=3 里只有 2 个位置有真 KV，
ph=1 就是那笔还没到账的 1。

数字全部取自 figure_spec.numbers（before: computed=3、ph=2、在飞=3、tws=2、持块=1；
after: ph=1、tws=3、cache_blocks(req-0, 3−1=2)；stale 分支: 抢占后 ph=0、computed=0、
stale=3、stale 送达 [7] 照收但 ph_unchanged=0；e2e 终态: computed=3、ph=0、差值=3 =
2 prompt 位 + 1 decode 位）。位置条斜纹 = 最近 ph 个位置（真 KV 未确认）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 842
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'cache_blocks 的参数为什么是差值：乐观计数 3 里只有 2 个位置有真 KV',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '调度器的 computed 是乐观值——占位也计入其中；KV 缓存的正式身份永远跟着『真算』走，'
        '不跟着乐观计数走', 10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑥ 真记账 · L0：循环框 × 显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左右双态 panel ----------------
PNL_Y, PNL_H = 92, 360
PW = 616
LX, RX = MX, MX + PW + 148
CELL_W, CELL_H, CELL_GAP = 74, 34, 6

PANELS = [
    (LX, 'BEFORE · pop 前（拍 1+2 调度后）', lc.C_ENG_S, lc.C_ENG_F,
     [('computed（乐观）', '3'), ('ph（欠条）', '2'), ('在飞', '3'), ('tws（账本行长）', '2'), ('持块', '1')],
     1, 2,
     ['只敢担保 1 —— 差的那 2 笔还没到账', '（批A 在 GPU 上算、t7 位在 D2H 路上）'],
     None),
    (RX, 'AFTER · pop 批A 交货 [7] 后', lc.C_KV_S, lc.C_KV_F,
     [('computed（乐观）', '3'), ('ph（欠条）', '1'), ('在飞', '1'), ('tws（账本行长）', '3'), ('交货', '[7]')],
     2, 1,
     ['担保数 3−1=2 —— t0,t1 两个 prompt 位转正', '（t7 位仍在 D2H 路上，交货后才轮到它）'],
     'cache_blocks(req-0, 3−1=2)'),
]
for px, title, stroke, fill, ledger, n_sol, n_hat, notes, cbcall in PANELS:
    lc.rect(px, PNL_Y, PW, PNL_H, '#ffffff', stroke, rx=8, sw=1.6)
    lc.rect(px, PNL_Y, PW, 30, fill, stroke, rx=8, sw=1.6)
    lc.rect(px, PNL_Y + 18, PW, 12, fill, 'none', rx=0, sw=0)
    lc.text(px + PW / 2, PNL_Y + 20, title, 11, stroke, 'middle', True, maxw=PW - 20, tag='pt:' + title[:6])
    # 账本五项（两列）
    ly0 = PNL_Y + 56
    for i, (name, val) in enumerate(ledger):
        col, row = i % 2, i // 2
        qx = px + 22 + col * (PW / 2 - 10)
        qy = ly0 + row * 40
        lc.text(qx, qy, name, 8.6, lc.C_MUTE, 'start', maxw=150, tag='lg:' + name)
        lc.text(qx + 158, qy, val, 13, lc.C_TXT, 'start', True, tag='lgv:' + name)
    # 位置条
    sy = ly0 + 3 * 40 + 18
    for k in range(3):
        cx = px + (PW - (3 * CELL_W + 2 * CELL_GAP)) / 2 + k * (CELL_W + CELL_GAP)
        lab = ['位置0 t0', '位置1 t1', '位置2 t7'][k]
        if k < n_sol:
            lc.rect(cx, sy, CELL_W, CELL_H, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.3)
            lc.text(cx + CELL_W / 2, sy + 15, lab.split()[0], 9.5, lc.C_KV_S, 'middle', True,
                    tag='cl' + title[:3] + str(k))
            lc.text(cx + CELL_W / 2, sy + 28, lab.split()[1], 8.6, lc.C_KV_S, 'middle', tag='clt' + title[:3] + str(k))
        elif k < n_sol + n_hat:
            lc.rect(cx, sy, CELL_W, CELL_H, '#fff7ed', lc.C_BEAT_S, rx=4, sw=1.3)
            lc.seg(cx + 6, sy + CELL_H - 6, cx + CELL_W - 6, sy + 6, lc.C_BEAT_S, 1.5)
            lc.text(cx + CELL_W / 2, sy + 15, lab.split()[0], 9.5, lc.C_BEAT_T, 'middle', True,
                    tag='cl' + title[:3] + str(k))
            lc.text(cx + CELL_W / 2, sy + 28, lab.split()[1] + '（在飞）', 7.8, lc.C_BEAT_T, 'middle',
                    maxw=CELL_W - 6, tag='clt' + title[:3] + str(k))
        else:
            lc.rect(cx, sy, CELL_W, CELL_H, '#ffffff', '#cbd5e1', rx=4, sw=1.0, dash=True)
            lc.text(cx + CELL_W / 2, sy + 15, lab.split()[0], 9.5, '#94a3b8', 'middle', maxw=CELL_W - 6,
                    tag='cl' + title[:3] + str(k))
            lc.text(cx + CELL_W / 2, sy + 28, lab.split()[1], 8.6, '#94a3b8', 'middle', maxw=CELL_W - 6,
                    tag='clt' + title[:3] + str(k))
    # 转正调用 / 注记
    ny = sy + CELL_H + 30
    if cbcall:
        lc.rect(px + 20, ny - 14, PW - 40, 30, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.4)
        lc.text(px + PW / 2, ny + 6, cbcall + ' —— 2 个真算位转正', 10, lc.C_KV_S, 'middle', True,
                maxw=PW - 60, tag='cb:' + cbcall)
    else:
        lc.rect(px + 20, ny - 14, PW - 40, 30, '#ffffff', lc.C_ENG_S, rx=6, sw=1.2, dash=True)
        lc.text(px + PW / 2, ny + 6, '不调用 cache_blocks —— 差值 1 不敢全转', 9.5, lc.C_ENG_S,
                'middle', True, maxw=PW - 60, tag='cb:none')
    for i, note in enumerate(notes):
        lc.text(px + PW / 2, ny + 32 + i * 15, note, 8.6, lc.C_MUTE, 'middle', maxw=PW - 36,
                tag='note:%s%d' % (title[:6], i))

# ---------------- 中间转换箭头 ----------------
MID_X = LX + PW + 74
lc.parrow([(LX + PW + 6, PNL_Y + 150), (MID_X, PNL_Y + 150), (MID_X, PNL_Y + 186),
           (RX - 8, PNL_Y + 186)], lc.C_ENG_S, 2.4, 'std')
lc.text(MID_X, PNL_Y + 100, 'pop 批A', 10.5, lc.C_ENG_S, 'middle', True, maxw=130, tag='mid:1')
lc.text(MID_X, PNL_Y + 116, 'update_from_output', 8.6, lc.C_ENG_S, 'middle', maxw=140, tag='mid:2')
lc.text(MID_X, PNL_Y + 132, '交货 [7]', 9.5, lc.C_ENG_S, 'middle', True, maxw=130, tag='mid:3')
lc.text(MID_X, PNL_Y + 236, 'ph 2−1=1', 10, lc.C_ABORT, 'middle', True, maxw=130, tag='mid:4')
lc.text(MID_X, PNL_Y + 252, '销 1 张欠条', 8.4, lc.C_MUTE, 'middle', maxw=130, tag='mid:5')

# ---------------- 底部：stale 分支 + e2e 终态 ----------------
BT_Y = PNL_Y + PNL_H + 16
BT_H = 128
BW1 = 828
lc.rect(MX, BT_Y, BW1, BT_H, '#fef2f2', lc.C_ABORT, rx=7, sw=1.3)
lc.text(MX + 16, BT_Y + 20, '对照 · stale 分支：被抢占的请求另立遗留账（PREEMPTED 整行跳过转正）',
        9.8, lc.C_ABORT, 'start', True, maxw=BW1 - 30, tag='bt1:t')
STALE_ROWS = [
    ('抢占后', 'ph=0、computed=0（stale=3 个旧 token 挂遗留账）'),
    ('stale 送达 [7]', 'token 照收（丢会扰动 spec-decode acceptance）· ph_unchanged=0 不扣'),
    ('若误扣', '0−1=−1 当场 assert 崩——扣了就 underflow（async_scheduler.py:L59-L63 防线）'),
]
for i, (a, b) in enumerate(STALE_ROWS):
    yy = BT_Y + 40 + i * 24
    lc.rect(MX + 18, yy - 11, 118, 18, '#ffffff', lc.C_ABORT, rx=4, sw=1.0)
    lc.text(MX + 77, yy + 2, a, 8.4, lc.C_ABORT, 'middle', True, maxw=110, tag='st:a' + str(i))
    lc.text(MX + 150, yy + 2, b, 8.5, '#334155', 'start', maxw=BW1 - 170, tag='st:b' + str(i))

BX2 = MX + BW1 + 20
BW2 = 1380 - BW1 - 20
lc.rect(BX2, BT_Y, BW2, BT_H, '#ffffff', lc.C_KV_S, rx=7, sw=1.3)
lc.text(BX2 + 14, BT_Y + 20, 'e2e 终态', 9.8, lc.C_KV_S, 'start', True, maxw=BW2 - 28, tag='bt2:t')
lc.text(BX2 + 14, BT_Y + 42, 'computed=3、ph=0', 9.5, lc.C_TXT, 'start', True, maxw=BW2 - 28, tag='bt2:a')
lc.text(BX2 + 14, BT_Y + 60, '差值 3 = 有确认 KV 的位置数', 9, lc.C_KV_S, 'start', True,
        maxw=BW2 - 28, tag='bt2:b')
lc.text(BX2 + 14, BT_Y + 78, '（2 prompt 位 + 1 decode 位）', 8.4, lc.C_MUTE, 'start',
        maxw=BW2 - 28, tag='bt2:c')
lc.text(BX2 + 14, BT_Y + 102, 'cache_blocks 参数 2 → 3 随交货单调涨', 8.6, '#334155', 'start',
        True, maxw=BW2 - 28, tag='bt2:d')

# ---------------- 结论横幅 ----------------
BN_Y = BT_Y + BT_H + 14
lc.rect(MX, BN_Y, 1380, 36, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 690, BN_Y + 22.5, 'KV 块只转正、不回退：cache_blocks 的参数 computed − ph 单调不减（差值 = 真实已算）——这就是显存账本敢给优化买单的依据',
        10, '#155e75', 'middle', True, maxw=1360, tag='banner')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BN_Y + 62
lx = MX
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, '真实已算（有真 KV，转正）', 8.5, lc.C_TXT, 'start', maxw=220, tag='leg:sol')
lx += 28 + lc.tw('真实已算（有真 KV，转正）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 22, 13, '#fff7ed', lc.C_BEAT_S, rx=3, sw=1.2)
lc.seg(lx + 3, LEG_Y + 1, lx + 19, LEG_Y - 7, lc.C_BEAT_S, 1.3)
lc.text(lx + 28, LEG_Y + 1, '占位在飞覆盖（最近 ph 个位置）', 8.5, lc.C_TXT, 'start', maxw=240, tag='leg:hat')
lx += 28 + lc.tw('占位在飞覆盖（最近 ph 个位置）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 22, 13, '#ffffff', '#cbd5e1', rx=3, sw=1.0, dash=True)
lc.text(lx + 28, LEG_Y + 1, '未计入', 8.5, lc.C_TXT, 'start', maxw=100, tag='leg:empty')
lx += 28 + lc.tw('未计入', 8.5) + 18
lc.rect(lx, LEG_Y - 11, 22, 15, '#fef2f2', lc.C_ABORT, rx=3, sw=1.1)
lc.text(lx + 28, LEG_Y + 1, 'stale / 断言防线', 8.5, lc.C_TXT, 'start', maxw=160, tag='leg:stale')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/core/sched/async_scheduler.py:L65-L69（cache_blocks 差值参数）· '
        'L51-L70（_update_request_with_output 交货扣减与 stale 分支）· vllm/v1/core/sched/scheduler.py:L1306（抢占清零）· '
        '账本数字取自配套精简版 host 实跑（交货转换 + stale 送达 + e2e 全程）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-block-convert.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
