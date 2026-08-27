#!/usr/bin/env python3
"""ch11 机制图 7 · 水位三限定（figure_spec ch11-fig-watermark-three-limits，模板 state-table）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）——上半 Scheduler 框『调度账本+状态机』位
与下半 KVCacheManager 框的交界：即本章 L2 章图 center ⑦ 恢复准入·水位 watermark 拍片 +
south『KVCacheManager（契约面 · 黑盒）』框 watermark_blocks 的机制展开；非新架构画法，
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：水位是只对『新准入』生效的 headroom：同一池 10 块、水位 5——首拍准入不吃
（8+0≤10）、RUNNING 增长不吃（1+0=1≤2）、WAITING 准入吃（1+5=6>1 拒）；
watermark=0.0（默认）时同一请求放行。

数字全部取自 figure_spec.numbers（三行判定 A-1/A-2/A-3 + 对照 A-3' / 算术 int(0.5×10)=5、
int(0.3×10)=3、int(0.3×100)=30、默认 0.0 / 抖动机理引文），源出配套精简版 host 实跑 trace
（10 块池 × block_size 16，r1=128-token、small=16-token）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 838
MX, BXR = 60, 1440

C_WM_F, C_WM_S = '#fef9c3', '#ca8a04'      # 水位 headroom（黄）

# ---------------- 标题区 ----------------
lc.text(MX, 34, '水位只对『新准入』生效：留 5 块 headroom——首拍不吃、RUNNING 增长不吃、WAITING 准入吃',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'required_blocks = num_blocks_to_allocate + watermark_blocks（kv_cache_manager.py:L521-L527）——wb≥0 单调，只会把『恰好放行』变『拒绝』，绝不反转判定方向',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ + south KV 契约面 · L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 列几何 ----------------
ID_X = 72                       # 身份列
PL_X, CELL, CGAP = 330, 64, 2  # 池标尺
VD_X = 1150                     # 判定列

HDR_Y = 100
lc.text(ID_X, HDR_Y, '拍 · 身份（谁在要块）', 9.5, lc.C_MUTE, 'start', True, maxw=230, tag='hd1')
lc.text(PL_X + 5 * (CELL + CGAP), HDR_Y, '块池标尺（10 格 · 判定时点；need 以块计，block_size=16）', 9.5,
        lc.C_MUTE, 'middle', True, maxw=520, tag='hd2')
lc.text((VD_X + BXR) / 2, HDR_Y, '判定 required ≤ 空闲？', 9.5, lc.C_MUTE, 'middle', True, maxw=240, tag='hd3')

# 行数据：(拍号, 身份, 身份细, 请求动作, used, wb, need, 公式, 判定, 判定细, ok)
ROWS = [
    ('A-1', 'WAITING · running 空', 'has_scheduled_reqs=False', '首拍准入 r1（128-token → 8 块）', 0, 0, 8,
     '8 + 0 = 8 ≤ 10', '✓ 放行', '首拍不吃水位：误计入则 8+5=13>10，空引擎永不起步', True),
    ('A-2', 'RUNNING · decode 增长', '水位不适用（L463-L467）', 'r1 第 129 token 跨入第 9 块（1 块）', 8, 0, 1,
     '1 + 0 = 1 ≤ 2', '✓ 放行', 'RUNNING 增长不吃水位——否则每步 decode 都被压', True),
    ('A-3', 'WAITING · 已有在场者', 'has_scheduled_reqs=True', 'small 准入（16-token → 1 块）', 9, 5, 1,
     '1 + 5 = 6 > 1', '✗ 拒', 'small 留 WAITING；r1 照常 decode（token 进已有块，free 不变）', False),
    ('A-3′', 'WAITING · 在场 · 水位关', 'watermark=0.0（默认）', 'small2 准入（同形对照）', 9, 0, 1,
     '1 + 0 = 1 ≤ 1', '✓ 放行', '同一池同一请求——关掉水位即放行（吞吐换稳定的旋钮）', True),
]
ROW_Y0, ROW_H = 118, 108
POOL_CELLS = 10
for i, (bid, ident, ident2, act, used, wb, need, formula, verdict, vnote, ok) in enumerate(ROWS):
    ry = ROW_Y0 + i * ROW_H
    if i > 0:
        lc.seg(MX, ry - 5, BXR, ry - 5, '#e2e8f0', 1.0)
    # 身份列
    bw = 16 + 11 * len(bid)
    lc.rect(ID_X, ry + 12, bw, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
    lc.text(ID_X + bw / 2, ry + 25.5, bid, 9.5, lc.C_ENG_S, 'middle', True, tag='bdg' + bid)
    lc.text(ID_X, ry + 50, ident, 9.5, lc.C_TXT, 'start', True, maxw=240, tag='id' + bid)
    lc.text(ID_X, ry + 68, ident2, 8.4, lc.C_MUTE, 'start', maxw=240, tag='id2' + bid)
    lc.text(ID_X, ry + 86, act, 8.4, '#334155', 'start', maxw=246, tag='act' + bid)
    # 池标尺
    free = POOL_CELLS - used
    for c in range(POOL_CELLS):
        cx = PL_X + c * (CELL + CGAP)
        if c < used:
            lc.rect(cx, ry + 22, CELL, 36, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.1)
        elif c < used + min(wb, free):
            lc.rect(cx, ry + 22, CELL, 36, C_WM_F, C_WM_S, rx=3, sw=1.1)
        else:
            lc.rect(cx, ry + 22, CELL, 36, '#ffffff', '#cbd5e1', rx=3, sw=1.0)
    lc.text(PL_X, ry + 15, '空闲 ' + str(free), 8.2, lc.C_MUTE, 'start', tag='fr' + bid)
    # 需求条（need 橙 + wb 黄，溢出池界画虚线桩）
    bx0 = PL_X + used * (CELL + CGAP)
    lc.rect(bx0, ry + 64, need * CELL + (need - 1) * CGAP, 12, '#fff7ed', lc.C_ENG_S, rx=3, sw=1.3)
    if wb > 0:
        wx0 = bx0 + need * CELL + (need - 1) * CGAP + CGAP
        room = (POOL_CELLS - used - need) * (CELL + CGAP) - CGAP
        if room > 0:
            lc.rect(wx0, ry + 64, min(wb * (CELL + CGAP) - CGAP, room), 12, C_WM_F, C_WM_S, rx=3, sw=1.3)
        over = wb * (CELL + CGAP) - CGAP - max(room, 0)
        if over > 0:
            # 溢出池界的部分：只画一格虚线桩示意（右缘不越过判定列文字区）
            lc.rect(wx0 + max(room, 0) + CGAP, ry + 64, CELL, 12, '#ffffff', C_WM_S, rx=3, sw=1.2, dash=True)
    lc.text(bx0 - 6, ry + 74, 'need', 8.2, lc.C_ENG_S, 'end', True, tag='nd' + bid)
    # 判定列
    vcol = lc.C_GPU_S if ok else lc.C_ABORT
    lc.text(VD_X, ry + 34, formula, 11, vcol, 'start', True, maxw=280, tag='fm' + bid)
    lc.text(VD_X, ry + 56, verdict, 11.5, vcol, 'start', True, maxw=120, tag='vd' + bid)
    lc.text(VD_X, ry + 78, vnote, 8.4, lc.C_MUTE, 'start', maxw=286, tag='vn' + bid)

# ---------------- 底部：三限定 + 算术 + 抖动机理 ----------------
B1_Y, B1_H = ROW_Y0 + 4 * ROW_H + 16, 118
lc.rect(MX, B1_Y, 660, B1_H, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.4)
lc.text(MX + 16, B1_Y + 22, '三限定——各自防一个死锁/误伤（kv_cache_manager.py:L463-L470）', 10.5, lc.C_KV_S,
        'start', True, maxw=620, tag='b1:t')
for j, ln in enumerate(['① 只对准入侧生效（分配门 L521-L527 计入 required_blocks）',
                        '② 只对 WAITING / PREEMPTED：RUNNING 增长不吃——正常 decode 不被系统性压制',
                        '③ 只在本步已有在场者（has_scheduled_reqs）时计入——空引擎首拍不吃，否则永不起步']):
    lc.text(MX + 16, B1_Y + 44 + j * 18, ln, 8.8, '#334155', 'start', maxw=630, tag='b1:l' + str(j))

B2_X = MX + 684
lc.rect(B2_X, B1_Y, BXR - B2_X, B1_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(B2_X + 16, B1_Y + 22, '算术与配置', 10.5, lc.C_TXT, 'start', True, maxw=400, tag='b2:t')
for j, ln in enumerate(['watermark_blocks = int(watermark × num_blocks)（L170-L171，int 截断）',
                        'int(0.5×10)=5 · int(0.3×10)=3 · int(0.3×100)=30 · 默认 0.0=关',
                        '区间 [0.0, 1.0)（config/scheduler.py:L136-L141）：把取舍留给用户按负载调']):
    lc.text(B2_X + 16, B1_Y + 44 + j * 18, ln, 8.8, '#334155', 'start', maxw=BXR - B2_X - 30, tag='b2:l' + str(j))

QT_Y = B1_Y + B1_H + 14
lc.rect(MX, QT_Y, BXR - MX, 64, '#fff7ed', lc.C_ABORT, rx=8, sw=1.3)
lc.text(MX + 16, QT_Y + 20, '为什么要这道护栏（官方 decode-heavy 复现的抖动机理，benchmarks/kv_cache_watermark.sh:L5-L16）', 10,
        lc.C_ABORT, 'start', True, maxw=900, tag='qt:t')
lc.text(MX + 16, QT_Y + 40, '「admitted based on KV cache they need at admission time … output length is unknown and unreserved」', 8.8,
        '#334155', 'start', maxw=BXR - MX - 32, tag='qt:l1')
lc.text(MX + 16, QT_Y + 56, '→ 短时超收 → 全体增长 → 池尽 → 抢占刚准入者 → 重 prefill → 再抢占：留 headroom 让 decode 有缓冲（与守卫关闸互补：关闸治标，水位治本）', 8.8,
        lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='qt:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = QT_Y + 88
lx = MX
for kind, name in [('used', '已用（不在可用账上）'), ('free', '空闲'), ('wm', '水位 headroom（黄）'),
                   ('need', 'need（本次要分配）'), ('over', '溢出池界（拒）')]:
    if kind == 'used':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.1)
    elif kind == 'free':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', '#cbd5e1', rx=3, sw=1.0)
    elif kind == 'wm':
        lc.rect(lx, LEG_Y - 9, 20, 12, C_WM_F, C_WM_S, rx=3, sw=1.1)
    elif kind == 'need':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#fff7ed', lc.C_ENG_S, rx=3, sw=1.2)
    else:
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', C_WM_S, rx=3, sw=1.1, dash=True)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.8) + 18

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_manager.py:L463-L470（准入门三限定）/ L521-L527（required_blocks）'
        '/ L170-L171（wb 算术）· vllm/config/scheduler.py:L136-L141 · 判定行取自配套精简版 host 实跑（10 块池 × block_size 16）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-watermark-three-limits.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
