#!/usr/bin/env python3
"""ch14 机制图 7 · 精修版水位门（figure_spec ch14-fig-watermark-gate，模板 before-after）

放大自 L0 调度账本列（准入/抢占位）与 KV 账本列的接缝——本章 L2 章图拍片⑦
「水位门 · watermark」的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）。

claim：精修版水位只在『本步已有调度请求且来者是 WAITING/PREEMPTED』时把
watermark_blocks 计入 required——decode-heavy 的抢占抖动循环被 headroom 截断，
RUNNING 涨块与首拍空转不受垫片约束。

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

# ---------------- 标题区 ----------------
lc.text(MX, 34, '精修版水位：垫片只管「新客进门」，不管「在座长个」——截断 decode-heavy 的抢占抖动循环',
        16.5, lc.C_TXT, 'start', True, maxw=990, tag='title')
lc.text(MX, 58, 'watermark_blocks = int(watermark × num_blocks)，只在『本步已有调度请求 且 来者是 WAITING/PREEMPTED』时计入 required——默认 0.0 关闭，交给用户按负载调',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · 调度账本列准入位 · L2 拍片⑦'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

PY0, PH = 92, 316

# ---------------- 左：水位关（默认）—— 抖动循环 ----------------
LX, LW = MX, 680
lc.rect(LX, PY0, LW, PH, '#ffffff', lc.C_ABORT, rx=9, sw=1.4)
lc.text(LX + 16, PY0 + 22, '水位关（默认 0.0）· decode-heavy 抖动循环', 11.5, lc.C_ABORT,
        'start', True, maxw=LW - 32, tag='lp:t')
NW_, NGAP = 200, 20
nx0 = LX + 20
TOP_Y, BOT_Y, NH = PY0 + 44, PY0 + 162, 54
NODES_TOP = [
    ('①', '准入只预留输入长度', '输出未知、不预留'),
    ('②', '短时超收', '门全放行'),
    ('③', '集体增长', 'decode 步步要块'),
]
NODES_BOT = [  # 从右到左摆放：④右、⑤中、⑥左
    ('⑥', '重 prefill', '又吃一大口块'),
    ('⑤', '抢占刚准入者', '→ ch11 抢占环'),
    ('④', '池尽', 'free = 0'),
]
def node(x, y, num, t, s):
    lc.rect(x, y, NW_, NH, '#fef2f2', lc.C_ABORT, rx=6, sw=1.3)
    lc.text(x + 10, y + 20, num + ' ' + t, 9.8, lc.C_ABORT, 'start', True, maxw=NW_ - 20,
            tag='nd:' + num)
    lc.text(x + 10, y + 38, s, 8.4, '#64748b', 'start', maxw=NW_ - 20, tag='nds:' + num)
    return (x, y)
pos = {}
for i, (n_, t, s) in enumerate(NODES_TOP):
    pos[n_] = node(nx0 + i * (NW_ + NGAP), TOP_Y, n_, t, s)
for i, (n_, t, s) in enumerate(NODES_BOT):
    pos[n_] = node(nx0 + i * (NW_ + NGAP), BOT_Y, n_, t, s)
AC = lc.C_ABORT
# ①→②→③（顶行横向）
for a, b in [('①', '②'), ('②', '③')]:
    x1 = pos[a][0] + NW_
    x2 = pos[b][0]
    lc.seg(x1 + 1, TOP_Y + NH / 2, x2 - 3, TOP_Y + NH / 2, AC, 1.6, 'std')
# ③→④（右侧竖向）
x34 = pos['③'][0] + NW_ / 2
lc.seg(x34, TOP_Y + NH + 1, x34, BOT_Y - 3, AC, 1.6, 'std')
# ④→⑤→⑥（底行从右往左）
for a, b in [('④', '⑤'), ('⑤', '⑥')]:
    x1 = pos[a][0]
    x2 = pos[b][0] + NW_
    lc.seg(x1 - 1, BOT_Y + NH / 2, x2 + 3, BOT_Y + NH / 2, AC, 1.6, 'std')
# ⑥→④ 回环（底行下方绕行）
loop_y = BOT_Y + NH + 26
x6c = pos['⑥'][0] + NW_ / 2
x4c = pos['④'][0] + NW_ / 2
lc.parrow([(x6c, BOT_Y + NH + 1), (x6c, loop_y), (x4c, loop_y), (x4c, BOT_Y + NH + 3)],
          AC, 1.6, 'std')
lc.text((x6c + x4c) / 2, loop_y - 7, '再抢占 · 循环抖动', 8.8, AC, 'middle', True, maxw=200,
        tag='loop:lbl')
# 官方复现口径
NY = loop_y + 16
lc.text(LX + 16, NY + 12, '官方复现口径 benchmarks/kv_cache_watermark.sh：并发 200 · input ~300 · output ~4000', 8.4,
        '#64748b', 'start', maxw=LW - 32, tag='lp:n1')
lc.text(LX + 16, NY + 28, '（decode-heavy：输出远大于输入）· KV 池压到均值需求 ~1.5× —— 池一尽就集体抢占重跑', 8.4,
        '#64748b', 'start', maxw=LW - 32, tag='lp:n2')

# ---------------- 右：水位 0.5 —— 三判定 ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
lc.rect(RX, PY0, RW, PH, lc.C_KV_F, lc.C_KV_S, rx=9, sw=1.4)
lc.text(RX + 16, PY0 + 22, '水位 0.5 · 同一个 80-token 请求（5 块）', 11.5, lc.C_KV_S,
        'start', True, maxw=RW - 32, tag='rp:t')
# 公式条
lc.rect(RX + 16, PY0 + 36, RW - 32, 34, '#ffffff', lc.C_KV_S, rx=6, sw=1.1)
lc.text(RX + 28, PY0 + 58, 'watermark_blocks = int(0.5 × 10) = 5 · free = 9 · 池 10 块', 9.6,
        lc.C_KV_S, 'start', True, maxw=RW - 56, tag='rp:f')
# 条件带
lc.text(RX + 16, PY0 + 92, '计入水位的两个条件（都真才加）：', 8.8, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='rp:c')
cond_x = RX + 16
for c in ['本步已有调度请求', '状态 ∈ {WAITING, PREEMPTED}']:
    cw2 = lc.tw(c, 8.6, True) + 14
    lc.rect(cond_x, PY0 + 100, cw2, 20, '#ffffff', lc.C_KV_S, rx=9, sw=1.0)
    lc.text(cond_x + cw2 / 2, PY0 + 114, c, 8.6, lc.C_KV_S, 'middle', True, maxw=cw2 - 6,
            tag='cond:' + c[:6])
    cond_x += cw2 + 10
lc.text(cond_x, PY0 + 114, 'AND', 8.6, lc.C_MUTE, 'start', True, maxw=30, tag='cond:and')
# 判定表
TY0 = PY0 + 134
COLS = [('请求状态 / 条件', 250), ('required 计算', 210), ('判定', 190)]
ROW_H = 30
tw_ = sum(c[1] for c in COLS) + 16
lc.rect(RX + 16, TY0, tw_, ROW_H, lc.C_KV_S, lc.C_KV_S, rx=4, sw=1.0)
cx = RX + 24
for name, cwid in COLS:
    lc.text(cx + cwid / 2, TY0 + 20, name, 9, '#ffffff', 'middle', True, maxw=cwid,
            tag='th:' + name)
    cx += cwid + 8
GATE = [
    ('WAITING · 本步已有调度', '5 + 5 = 10 > free 9', 'None（暂缓：headroom 留给在座）', True),
    ('WAITING · 首拍空转（无调度）', '5 + 0 = 5 ≤ 9', '放行（池全空再保守就饿死）', False),
    ('RUNNING 涨块', '5 + 0 = 5 ≤ 9', '放行（在座长个不受垫片约束）', False),
]
for ri, (st, calc, verdict, blocked) in enumerate(GATE):
    ry = TY0 + ROW_H + 5 + ri * (ROW_H + 4)
    fill = '#fef2f2' if blocked else '#ffffff'
    stroke = lc.C_ABORT if blocked else '#a5f3fc'
    lc.rect(RX + 16, ry, tw_, ROW_H, fill, stroke, rx=3, sw=1.0 if blocked else 0.8)
    cx = RX + 24
    for (name, cwid), val in zip(COLS, [st, calc, verdict]):
        col = lc.C_ABORT if (blocked and name == '判定') else '#334155'
        lc.text(cx + cwid / 2, ry + 20, val, 8.6, col, 'middle', name == '判定',
                maxw=cwid - 6, tag='td%d:%s' % (ri, name))
        cx += cwid + 8

# ---------------- 底部结论条（全宽） ----------------
BY = PY0 + PH + 22
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '水位条件只在『已有 running』时生效，而 running 完成必释放块——水位的头寸始终来自刚被拒的 WAITING 请求，不会锁死系统；', 9.4,
        lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='bd:t')
lc.text(MX + 16, BY + 42, '代价是 headroom 空闲不接客（吞吐换稳定），默认 0.0 关闭交给用户按负载调', 9,
        '#334155', 'start', maxw=BXR - MX - 32, tag='bd:l1')

# ---------------- 页脚 ----------------
FY = BY + 80
lc.text(MX, FY, '逐字锚 vllm/v1/core/kv_cache_manager.py:L168-L171（watermark_blocks 公式）· L463-L470（两条件）· L521-L527（判定与 None）· '
        'vllm/config/scheduler.py:L136-L141（默认 0.0）· benchmarks/kv_cache_watermark.sh:L5-L27（官方复现口径）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '数字取自配套精简版 host 实跑（池 10 块 · free 9 · 80-token 请求 cdiv(80,16)=5 块）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = FY + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-watermark-gate.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)'.replace('wote', 'wrote'))
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
