#!/usr/bin/env python3
"""ch12 机制图 3 · 占位账本逐拍消长（figure_spec ch12-fig-placeholder-ledger，模板 state-table）

放大自 L0 循环框（loop_box）里调度器账本位——即本章 L2 章图 center 拍片 ①『盲调度 · 占位+1』
与 ⑥『真记账 · 占位-1』共用的账本表展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
图右上角指北小签。

claim：占位账本四拍消长——computed 乐观推进、ph 随完整步 +1 随交货 −1，
computed−ph 恒等于真实已算（拍末口径 1→2→3→3）且 cache_blocks 只按差值转正。

数字全部取自 figure_spec.numbers（四拍账本 (2,1,1)→(3,1,2)→(3,0,3)→终态 (3,0,3)；
cache_blocks 参数序列 2→3；占位步长无 spec=1、chunk 拍不占位 ph=0；
交货扣减 ph 2−1=1、stale 不扣防 underflow）。位置条斜纹 = 最近 ph 个位置
（computed 已计入、真 KV 未确认——差值之前的位置才有确认 KV）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 856
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '占位账本：computed − ph 恒等于真实已算，cache_blocks 只按差值转正',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'ph 是调度器开出的欠条——GPU 上每有一个在飞采样位账上 +1，真 token 到账销一张；'
        'computed 是乐观计数，占位也计入其中', 10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ①⑥ 账本 · L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左 panel：四拍账本主表 ----------------
PW_L, PM_Y, PM_H = 864, 92, 430
lc.rect(MX, PM_Y, PW_L, PM_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(MX + 16, PM_Y + 22, '四拍账本（prompt=2、max_tokens=2 e2e 实拍，拍末口径）',
        10, lc.C_TXT, 'start', True, maxw=PW_L - 30, tag='pmL:t')

# 列几何：拍徽标 | 事件两行 | 位置条 3 格 | computed | ph | 真实 | cache_blocks
BX_BEAT = MX + 26
BX_EVT = MX + 66
BX_STRIP = MX + 336
CELL_W, CELL_H, CELL_GAP = 52, 30, 5
BX_NUM = BX_STRIP + 3 * CELL_W + 2 * CELL_GAP + 24
NUM_COLS = [('computed', 62), ('ph', 40), ('真实已算', 74)]
BX_CB = BX_NUM + sum(w for _, w in NUM_COLS) + 22
CB_W = PW_L - (BX_CB - MX) - 14

HDR_Y = PM_Y + 46
lc.text(BX_BEAT + 15, HDR_Y, '拍', 8.8, lc.C_MUTE, 'middle', True, maxw=30, tag='hd:beat')
lc.text(BX_EVT, HDR_Y, '事件', 8.8, lc.C_MUTE, 'start', True, maxw=240, tag='hd:evt')
lc.text(BX_STRIP + 3 * (CELL_W + CELL_GAP) / 2 - CELL_GAP / 2, HDR_Y, '位置条 t0·t1·t7',
        8.8, lc.C_MUTE, 'middle', True, maxw=170, tag='hd:strip')
_nx = BX_NUM
for name, cw in NUM_COLS:
    lc.text(_nx + cw / 2, HDR_Y, name, 8.8, lc.C_MUTE, 'middle', True, maxw=cw - 2, tag='hd:' + name)
    _nx += cw
lc.text(BX_CB + CB_W / 2, HDR_Y, 'cache_blocks 参数', 8.8, lc.C_MUTE, 'middle', True,
        maxw=CB_W - 4, tag='hd:cb')
lc.seg(MX + 10, HDR_Y + 8, MX + PW_L - 10, HDR_Y + 8, '#e2e8f0', 1.0)

# 行数据：(拍号, 事件行1, 事件行2, solid数, hatch数, computed, ph, real, cache行, hot)
ROWS = [
    ('1', 'schedule 批A（全量 prefill 2）', '非 chunk → ph +1', 1, 1, '2', '1', '1', '—（未交货不转正）', False),
    ('2', 'schedule 批B（盲排 1）→ pop A', '交货 [7]：ph +1 后 −1（销 1 张欠条）', 2, 1, '3', '1', '2', 'cache_blocks(req-0, 2)', True),
    ('3', '剪枝空批C → pop B 交货 [9]', 'LENGTH 终态：ph −1 → 0', 3, 0, '3', '0', '3', 'cache_blocks(req-0, 3)', False),
    ('终', '队列排空（has_work=False）', '欠条清零、差值恒定', 3, 0, '3', '0', '3', '差值 3 = 确认 KV 位置数', False),
]
ROW_H = 88
ROW_Y0 = HDR_Y + 16
for i, (beat, ev1, ev2, n_sol, n_hat, comp, ph, real, cb, hot) in enumerate(ROWS):
    ry = ROW_Y0 + i * ROW_H
    mid = ry + 14
    if hot:
        lc.rect(MX + 8, ry - 4, PW_L - 16, ROW_H - 6, lc.C_BEAT_F, 'none', rx=5, sw=0)
    # 拍徽标
    lc.rect(BX_BEAT, mid - 10, 30, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
    lc.text(BX_BEAT + 15, mid + 3.5, beat, 9.5, lc.C_ENG_S, 'middle', True, tag='bdg' + beat)
    # 事件
    lc.text(BX_EVT, ry + 16, ev1, 8.6, '#334155', 'start', maxw=262, tag='ev1' + beat)
    lc.text(BX_EVT, ry + 31, ev2, 8.2, lc.C_MUTE, 'start', maxw=262, tag='ev2' + beat)
    # 位置条：前 n_sol 实心（真实已算）、接着 n_hat 斜纹（占位在飞覆盖）、其余空
    for k in range(3):
        cx = BX_STRIP + k * (CELL_W + CELL_GAP)
        cy = ry + 12
        lab = ['t0', 't1', 't7'][k]
        if k < n_sol:
            lc.rect(cx, cy, CELL_W, CELL_H, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.2)
            lc.text(cx + CELL_W / 2, cy + 19, lab, 9, lc.C_KV_S, 'middle', True, tag='cl' + beat + str(k))
        elif k < n_sol + n_hat:
            lc.rect(cx, cy, CELL_W, CELL_H, '#fff7ed', lc.C_BEAT_S, rx=4, sw=1.2)
            lc.seg(cx + 5, cy + CELL_H - 5, cx + CELL_W - 5, cy + 5, lc.C_BEAT_S, 1.4)
            lc.text(cx + CELL_W / 2, cy + 19, lab, 9, lc.C_BEAT_T, 'middle', True, tag='cl' + beat + str(k))
        else:
            lc.rect(cx, cy, CELL_W, CELL_H, '#ffffff', '#cbd5e1', rx=4, sw=1.0, dash=True)
            lc.text(cx + CELL_W / 2, cy + 19, lab, 9, '#94a3b8', 'middle', maxw=CELL_W - 6,
                    tag='cl' + beat + str(k))
    # 数值三列
    _nx = BX_NUM
    vals = [(comp, '#334155'), (ph, lc.C_ENG_S), (real, lc.C_KV_S)]
    for (val, col), (name, cw) in zip(vals, NUM_COLS):
        lc.text(_nx + cw / 2, ry + 32, val, 11, col, 'middle', True, tag='nm' + beat + name)
        _nx += cw
    # cache_blocks 列
    lc.text(BX_CB + CB_W / 2, ry + 26, cb, 8.6, lc.C_KV_S if 'cache_blocks' in cb else lc.C_MUTE,
            'middle', True, maxw=CB_W - 6, tag='cb' + beat)
    if i < len(ROWS) - 1:
        lc.seg(MX + 12, ry + ROW_H - 8, MX + PW_L - 12, ry + ROW_H - 8, '#e2e8f0', 1.0)
lc.text(BX_STRIP + 3 * (CELL_W + CELL_GAP) / 2 - CELL_GAP / 2, ROW_Y0 + 4 * ROW_H - 6,
        '斜纹 = 最近 ph 个位置（computed 已计入、真 KV 未确认）——差值之前的位置才有确认 KV',
        8.2, lc.C_MUTE, 'middle', maxw=PW_L - 60, tag='stripnote')

# ---------------- 右 panel：占位步长 + 交货扣减 ----------------
PW_R = BXR - (MX + PW_L) - 22
PX_R = MX + PW_L + 22
RP_H1, RP_H2 = 210, 196
lc.rect(PX_R, PM_Y, PW_R, RP_H1, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(PX_R + 14, PM_Y + 20, '占位步长：无 spec = 1 · prefill-chunk 拍不占位', 9.8, lc.C_TXT,
        'start', True, maxw=PW_R - 26, tag='rp1:t')
STEP_ROWS = [
    ('场景A 全量 prefill（prompt=2）', [('拍1', '非 chunk → ph 0→1'), ('拍2', '盲排 → ph 1→2')]),
    ('场景B chunked（prompt=6、预算 4）', [('拍1', 'chunk → continue，ph=0'), ('拍2', '排完余位 → ph 0→1'), ('拍3', '盲排 → ph 1→2')]),
]
sy = PM_Y + 38
for title, items in STEP_ROWS:
    lc.text(PX_R + 14, sy, title, 8.6, lc.C_ENG_S, 'start', True, maxw=PW_R - 26, tag='sp:' + title[:8])
    for j, (tag, txt) in enumerate(items):
        lc.text(PX_R + 26, sy + 15 + j * 14, tag + '　' + txt, 8.4, '#334155', 'start',
                maxw=PW_R - 40, tag='spi' + tag)
    sy += 15 + len(items) * 14 + 8
lc.text(PX_R + 14, sy + 2, '上界 = 队列深度 2 × 步长 1 = 2（ph 在 {1,2} 振荡，账单有界）',
        8.4, lc.C_MUTE, 'start', maxw=PW_R - 26, tag='rp1:bound')

RP2_Y = PM_Y + RP_H1 + 14
lc.rect(PX_R, RP2_Y, PW_R, RP_H2, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(PX_R + 14, RP2_Y + 20, '交货扣减与 stale 防线', 9.8, lc.C_TXT, 'start', True,
        maxw=PW_R - 26, tag='rp2:t')
lc.text(PX_R + 14, RP2_Y + 40, 'pop 批A 交货 [7]：ph 2−1=1', 8.6, lc.C_KV_S, 'start', True,
        maxw=PW_R - 26, tag='rp2:a1')
lc.text(PX_R + 26, RP2_Y + 55, '扣 len(new_token_ids)=1 —— 担保数 3−1=2', 8.4, '#334155',
        'start', maxw=PW_R - 40, tag='rp2:a2')
lc.text(PX_R + 14, RP2_Y + 78, 'stale 分支：抢占清零后旧账作废', 8.6, lc.C_ABORT, 'start', True,
        maxw=PW_R - 26, tag='rp2:b1')
lc.text(PX_R + 26, RP2_Y + 93, '抢占后 ph=0、computed=0（stale=3 个旧 token）', 8.4, '#334155',
        'start', maxw=PW_R - 40, tag='rp2:b2')
lc.text(PX_R + 26, RP2_Y + 108, 'stale 送达 [7]：token 照收、ph 不扣', 8.4, '#334155',
        'start', maxw=PW_R - 40, tag='rp2:b3')
lc.text(PX_R + 26, RP2_Y + 123, '（扣了就 0−1=−1 当场 assert 崩）', 8.4, lc.C_MUTE,
        'start', maxw=PW_R - 40, tag='rp2:b4')
lc.rect(PX_R + 14, RP2_Y + 138, PW_R - 28, 42, '#fef2f2', lc.C_ABORT, rx=6, sw=1.1)
lc.text(PX_R + PW_R / 2, RP2_Y + 153, 'assert num_output_placeholders ≥ 0 是防线不是装饰——',
        8.4, lc.C_ABORT, 'middle', True, maxw=PW_R - 44, tag='rp2:c1')
lc.text(PX_R + PW_R / 2, RP2_Y + 168, '#42117 / #46066 / #48245 三个 underflow 修复全是这张表被打破的案例',
        8.2, lc.C_ABORT, 'middle', maxw=PW_R - 40, tag='rp2:c2')

# ---------------- 底部结论横幅 ----------------
BN_Y = PM_Y + PM_H + 14
lc.rect(MX, BN_Y, 1380, 38, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(MX + 690, BN_Y + 16.5, '盲调度凭什么敢排、排错了谁兜底——欠条有界（≤ 队列深度 × 步长）、销账有 assert（≥ 0 防线）',
        10, lc.C_BEAT_T, 'middle', True, maxw=1360, tag='banner1')
lc.text(MX + 690, BN_Y + 30.5, 'cache_blocks 只按差值转正 KV 块：担保数随交货单调涨（2 → 3），占位数就是『CPU 世界慢半拍』的记账',
        9, lc.C_BEAT_T, 'middle', True, maxw=1360, tag='banner2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BN_Y + 66
lx = MX
LEG_SW, LEG_SH = 22, 13
# 实心格
lc.rect(lx, LEG_Y - 9, LEG_SW, LEG_SH, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
lc.text(lx + LEG_SW + 6, LEG_Y + 1, '真实已算（敢担保，cache_blocks 只认这些）', 8.5, lc.C_TXT,
        'start', maxw=330, tag='leg:sol')
lx += LEG_SW + 6 + lc.tw('真实已算（敢担保，cache_blocks 只认这些）', 8.5) + 18
# 斜纹格
lc.rect(lx, LEG_Y - 9, LEG_SW, LEG_SH, '#fff7ed', lc.C_BEAT_S, rx=3, sw=1.2)
lc.seg(lx + 3, LEG_Y + 1, lx + LEG_SW - 3, LEG_Y - 7, lc.C_BEAT_S, 1.3)
lc.text(lx + LEG_SW + 6, LEG_Y + 1, '占位在飞覆盖（computed 已计入、未销账）', 8.5, lc.C_TXT,
        'start', maxw=300, tag='leg:hat')
lx += LEG_SW + 6 + lc.tw('占位在飞覆盖（computed 已计入、未销账）', 8.5) + 18
# 空格
lc.rect(lx, LEG_Y - 9, LEG_SW, LEG_SH, '#ffffff', '#cbd5e1', rx=3, sw=1.0, dash=True)
lc.text(lx + LEG_SW + 6, LEG_Y + 1, '未计入（computed 之外）', 8.5, lc.C_TXT, 'start', maxw=200,
        tag='leg:empty')
lx += LEG_SW + 6 + lc.tw('未计入（computed 之外）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, LEG_SW, LEG_SH, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
lc.text(lx + LEG_SW + 6, LEG_Y + 1, '拍2 高亮（交货首次转正）', 8.5, lc.C_TXT, 'start', maxw=200,
        tag='leg:hot')

lc.text(MX, LEG_Y + 30, '逐字锚 vllm/v1/core/sched/async_scheduler.py:L19-L70（占位 +1 / 交货 −1 / cache_blocks 差值转正 / stale 防线）· '
        'vllm/v1/core/sched/scheduler.py:L516-L520（追赶公式的占位项）· 账本数字取自配套精简版 host 实跑'
        '（e2e 四拍 + 占位步长/交货/stale 三场景）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-placeholder-ledger.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
