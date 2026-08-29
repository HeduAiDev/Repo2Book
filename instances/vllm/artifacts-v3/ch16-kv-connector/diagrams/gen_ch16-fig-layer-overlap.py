#!/usr/bin/env python3
"""ch16 机制图 6 · 逐层重叠（figure_spec ch16-fig-layer-overlap，模板 swimlane·双甘特对比）

放大自 L0「GPU 列·层执行格」（本章 l0_zoom）、L2 站 8（worker 一拍·逐层收发——
注意力层前后的两个钩子 maybe_transfer_kv_layer 把传输与计算编织在一起）。

claim：wait_for_layer_load 把『等全部 KV 到齐再开算』拆成『每层只等本层』——传输与
计算在第 i 层重叠，端到端从 Σ传输+Σ计算（sum）降到 ≈max(Σ传输, Σ计算)。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：4 层×传输 2 拍/
计算 3 拍、就绪 2/4/6/8；重叠档期 2-5/5-8/8-11/11-14；朴素 20 vs 重叠 14 省 6；
真实调用序 11 事件）。时长=虚拟拍教学模型（页脚注明），调用序=4 层真跑实测。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 54
BXR = 1446

AX0 = 320            # 时间轴起点
TICK = 48            # px / 虚拟拍
NT = 20              # 总拍数（朴素端到端）
AX1 = AX0 + NT * TICK
BAR_H = 34

C_XFER = lc.C_KV_S       # 传输（青）
C_CALC = lc.C_GPU_S      # 计算（绿）
C_WAITF = '#e2e8f0'      # 等待（灰）


def tx(t):
    return AX0 + t * TICK


def panel(y0, title, sub, transfer_bars, compute_wait, compute_bars, endnote):
    """一个甘特面板：传输泳道 + 计算泳道 + 时间轴。"""
    lc.text(MX, y0, title, 12, lc.C_TXT, 'start', True, maxw=1000, tag=f'p{y0}:t')
    lc.text(MX, y0 + 19, sub, 9, lc.C_MUTE, 'start', maxw=1000, tag=f'p{y0}:s')
    ty = y0 + 40                       # 传输泳道
    cy = ty + BAR_H + 26               # 计算泳道
    # 泳道标签
    lc.text(AX0 - 14, ty + BAR_H / 2 - 4, '传输', 10.5, C_XFER, 'end', True, maxw=120, tag=f'p{y0}:lt')
    lc.text(AX0 - 14, ty + BAR_H / 2 + 12, 'start_load_kv 一次发起', 8, lc.C_MUTE, 'end', maxw=120, tag=f'p{y0}:lt2')
    lc.text(AX0 - 14, cy + BAR_H / 2 - 4, '计算', 10.5, C_CALC, 'end', True, maxw=120, tag=f'p{y0}:lc')
    lc.text(AX0 - 14, cy + BAR_H / 2 + 12, '每层 3 拍', 8, lc.C_MUTE, 'end', maxw=120, tag=f'p{y0}:lc2')
    # 传输条
    for (t0, t1, lab) in transfer_bars:
        lc.rect(tx(t0), ty, (t1 - t0) * TICK, BAR_H, C_XFER, C_XFER, rx=3, sw=0)
        lc.text(tx(t0) + (t1 - t0) * TICK / 2, ty + 21, lab, 9.5, '#ffffff', 'middle', True,
                maxw=(t1 - t0) * TICK - 6, tag=f'p{y0}:x{lab}')
    # 就绪三角标（传输条右端）
    for (t0, t1, lab) in transfer_bars:
        lx, ly = tx(t1), ty + BAR_H / 2
        lc.seg(lx - 7, ly - 6, lx, ly, C_XFER, 2.0)
        lc.seg(lx - 7, ly + 6, lx, ly, C_XFER, 2.0)
    # 计算泳道：等待区 + 计算条
    for (t0, t1, lab) in compute_wait:
        lc.rect(tx(t0), cy, (t1 - t0) * TICK, BAR_H, C_WAITF, lc.C_MUTE, rx=3, sw=1.0, dash=True)
        lc.text(tx(t0) + (t1 - t0) * TICK / 2, cy + 21, lab, 9, lc.C_MUTE, 'middle', True,
                maxw=(t1 - t0) * TICK - 6, tag=f'p{y0}:w{lab}')
    for (t0, t1, lab, zero_wait) in compute_bars:
        lc.rect(tx(t0), cy, (t1 - t0) * TICK, BAR_H, C_CALC, C_CALC, rx=3, sw=0)
        lc.text(tx(t0) + (t1 - t0) * TICK / 2, cy + 21, lab, 9.5, '#ffffff', 'middle', True,
                maxw=(t1 - t0) * TICK - 6, tag=f'p{y0}:c{lab}')
        if zero_wait:
            lc.text(tx(t0) + (t1 - t0) * TICK / 2, cy + BAR_H + 13, '零等待', 8, C_CALC, 'middle',
                    maxw=60, tag=f'p{y0}:z{lab}')
    # 时间轴
    ay = cy + BAR_H + 30
    lc.seg(AX0, ay, AX1, ay, lc.C_MUTE, 1.2)
    for t in range(0, NT + 1, 2):
        lc.seg(tx(t), ay, tx(t), ay + 5, lc.C_MUTE, 1.0)
        lc.text(tx(t), ay + 17, str(t), 8.5, lc.C_FAINT, 'middle', maxw=40, tag=f'p{y0}:tk{t}')
    lc.text(AX1 + 8, ay + 4, '虚拟拍', 8.5, lc.C_FAINT, 'start', maxw=60, tag=f'p{y0}:unit')
    lc.text(MX, ay + 4, endnote, 9, '#334155', 'start', maxw=250, tag=f'p{y0}:en')
    return ay


# ---------------- 标题区 ----------------
lc.text(MX, 36, '逐层重叠：每层只等本层，端到端从 sum 降到 max', 16.5, lc.C_TXT, 'start', True,
        maxw=940, tag='title')
lc.text(MX, 60, 'wait_for_layer_load 把『等全部 KV 到齐再开算』拆开——4 层 ×（传输 2 拍 + 计算 3 拍）教学模型：'
                '层序就绪于第 2/4/6/8 拍（时长=虚拟拍，非 GPU 实测）', 10.5, lc.C_MUTE, 'start', maxw=1100, tag='subtitle')
_ch = '放大自 L2 站 8 worker 一拍·逐层收发 · L0：GPU 列·层执行格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 面板 1：朴素 ----------------
XFER = [(0, 2, 'L0'), (2, 4, 'L1'), (4, 6, 'L2'), (6, 8, 'L3')]
P1_Y = 106
ay1 = panel(
    P1_Y,
    '朴素编排：等第 8 拍全部就绪，再串行算',
    '先 Σ传输 8 拍、后 Σ计算 12 拍，两段完全不重叠',
    XFER,
    [(0, 8, '等 8 拍（全部就绪才开算）')],
    [(8, 11, 'L0', False), (11, 14, 'L1', False), (14, 17, 'L2', False), (17, 20, 'L3', False)],
    '端到端 20 拍 = Σ传输 8 + Σ计算 12')
# 端到端读数
lc.text(AX1 + 26, P1_Y + 130, '端到端 20', 13, lc.C_MUTE, 'start', True, maxw=110, tag='p1:e2e')
lc.text(AX1 + 26, P1_Y + 148, '= sum(8+12)', 9, lc.C_MUTE, 'start', maxw=110, tag='p1:e2e2')

# ---------------- 面板 2：契约重叠 ----------------
P2_Y = ay1 + 52
ay2 = panel(
    P2_Y,
    '契约编排：层前 wait_for_layer_load 只等本层——只有第 0 层付过一次传输等待',
    '层 0 等 2 拍开算，其后三层零等待接力（实测：仅层 0 等过）',
    XFER,
    [(0, 2, '等 2 拍')],
    [(2, 5, 'L0', False), (5, 8, 'L1', True), (8, 11, 'L2', True), (11, 14, 'L3', True)],
    '端到端 14 拍 = max(8,12) + 2')
lc.text(AX1 + 26, P2_Y + 130, '端到端 14', 13, C_CALC, 'start', True, maxw=110, tag='p2:e2e')
lc.text(AX1 + 26, P2_Y + 148, '= max(8,12)+2', 9, C_CALC, 'start', maxw=116, tag='p2:e2e2')

# wait_for_save 栅栏（t=14）
cy2 = P2_Y + 40 + BAR_H + 26
lc.seg(tx(14), cy2 - 12, tx(14), cy2 + BAR_H + 24, lc.C_ABORT, 1.6, 'ab', dash=True)
lc.text(tx(14) - 8, cy2 - 18, 'wait_for_save 栅栏：不出栅栏，paged buffer 可能被下一步覆写', 8.5,
        lc.C_ABORT, 'end', maxw=460, tag='fence')

# 节省徽标（两面板之间右侧）
SV_Y = (ay1 + P2_Y) / 2 - 10
lc.rect(AX1 + 20, SV_Y, BXR - AX1 - 20, 44, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.5)
lc.text((AX1 + 20 + BXR) / 2, SV_Y + 19, '省 6 拍（30%）', 11, lc.C_BEAT_T, 'middle', True,
        maxw=BXR - AX1 - 30, tag='sv')
lc.text((AX1 + 20 + BXR) / 2, SV_Y + 35, 'sum → max 的结构收益', 8, lc.C_BEAT_T, 'middle',
        maxw=BXR - AX1 - 30, tag='sv2')

# ---------------- 真实调用序条 ----------------
SQ_Y = ay2 + 56
lc.rect(MX, SQ_Y, BXR - MX, 84, '#ffffff', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + 18, SQ_Y + 24, '真实调用序（4 层真跑实测 · 共 11 个事件）——档期是教学模型，这条序是真的：', 10,
        lc.C_GPU_S, 'start', True, maxw=1300, tag='sq:t')
SEQ = ['start_load_kv', '(wait_for_layer_load → save_kv_layer) × 4 层 = 8 事件', 'wait_for_save', 'get_finished']
sx = MX + 18
for i, s in enumerate(SEQ):
    cw_ = lc.tw(s, 9.5) + 20
    lc.rect(sx, SQ_Y + 38, cw_, 30, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.1)
    lc.text(sx + cw_ / 2, SQ_Y + 57, s, 9.5, lc.C_TXT, 'middle', maxw=cw_ - 8, tag=f'sq{i}')
    sx += cw_
    if i < len(SEQ) - 1:
        lc.seg(sx + 2, SQ_Y + 53, sx + 14, SQ_Y + 53, lc.C_GPU_S, 1.6, 'std')
        sx += 18

# ---------------- 图例 + 页脚 ----------------
LEG_Y = SQ_Y + 84 + 28
lx = MX
for kind, name in [('xfer', '传输（KV 逐层就绪）'), ('calc', '计算（注意力层）'), ('wait', '等待（等 KV 到齐）')]:
    fill, stroke, dash = {'xfer': (C_XFER, C_XFER, False),
                          'calc': (C_CALC, C_CALC, False),
                          'wait': (C_WAITF, lc.C_MUTE, True)}[kind]
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=200, tag='leg')
    lx += 26 + lc.tw(name, 8.5) + 24
lc.seg(lx, LEG_Y - 3, lx + 26, LEG_Y - 3, lc.C_ABORT, 1.6, 'ab', dash=True)
lc.text(lx + 32, LEG_Y + 2, 'wait_for_save 强制同步栅栏', 8.5, lc.C_TXT, 'start', maxw=200, tag='leg:ab')

FY = LEG_Y + 26
lc.text(MX, FY, '逐字锚 vllm/distributed/kv_transfer/kv_connector/v1/base.py:L323-L335（wait_for_layer_load：等第 i 层）· '
                'L337-L357（save_kv_layer）· L359-L367（wait_for_save docstring：'
                '『This prevents overwrites of paged KV buffer before saving done』）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, 'vllm/model_executor/layers/attention/kv_transfer_utils.py:L15-L43（maybe_transfer_kv_layer 装饰器：层前 wait / 层后 save）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1b')
lc.text(MX, FY + 32, '档期（就绪 2/4/6/8、端到端 20 vs 14、省 6 拍）取自精简版 companion 实测的教学模型 trace'
                '（时长=虚拟拍，真实收益取决于传输/计算比）；调用序 11 事件为 4 层真跑实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 52

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-layer-overlap.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
