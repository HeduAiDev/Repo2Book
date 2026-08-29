#!/usr/bin/env python3
"""ch16 机制图 4 · 已分配未缓存窗口（figure_spec ch16-fig-allocated-not-cached，模板 layout）

放大自 L0「KV 账本列·块账格」（本章 l0_zoom）、L2 站 5-6（护轨分配 delay_cache_blocks /
等待态 WAITING_FOR_REMOTE_KVS）。

claim：异步加载期请求处在『已分配未缓存』窗口：ext_comp 段的块已挂上块表（物理占用、
对外不可见），但缓存账为零、哈希表没登记——传输完成才补缓存（完成回收），失败按第一个
坏块截断（失败回滚）。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：块表挂 2 块
block_ids [1,2]；num_cached_block=0 / 首块哈希 None；num_computed_tokens 先行 32、
本拍零前向；free 61/63）。
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

# ---------------- 标题区 ----------------
lc.text(MX, 36, '『已分配未缓存』：ext_comp 段的账实分离窗口', 16.5, lc.C_TXT, 'start', True,
        maxw=900, tag='title')
lc.text(MX, 60, 'allocate_slots 五段布局 <comp|new_comp|ext_comp|new|lookahead>（kv_cache_manager.py:L390-L446）——'
                'ext_comp 段『not cached by vLLM but cached by the connector』（L370-L371）', 10.5, lc.C_MUTE,
        'start', maxw=1120, tag='subtitle')
_ch = '放大自 L2 站 5-6 护轨分配/等待态 · L0：KV 账本列·块账格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 五段布局条 ----------------
SEG_Y, SEG_H = 100, 108
GAP = 10
SEGS = [
    ('comp', '0', '已算且已缓存', '本地前缀命中的整块', False),
    ('new_comp', '0', '本拍新算已缓存', '算完即登记哈希表', False),
    ('ext_comp', '32', '外部可加载 ★', '块先占位、延迟入缓存', True),
    ('new', '0', '本拍新算', '此拍分配、暂未缓存', False),
    ('lookahead', '0', '预留余量', '调度器看的富余', False),
]
SEG_W = (BXR - MX - GAP * (len(SEGS) - 1)) / len(SEGS)
for i, (name, val, zh, sub, hot) in enumerate(SEGS):
    x = MX + i * (SEG_W + GAP)
    lc.rect(x, SEG_Y, SEG_W, SEG_H, lc.C_BEAT_F if hot else '#ffffff',
            lc.C_BEAT_S if hot else lc.C_MUTE, rx=7, sw=2.0 if hot else 1.3)
    lc.text(x + SEG_W / 2, SEG_Y + 24, name, 12, lc.C_BEAT_T if hot else lc.C_TXT, 'middle', True,
            maxw=SEG_W - 12, tag=f'seg{i}')
    lc.text(x + SEG_W / 2, SEG_Y + 52, val, 17, lc.C_BEAT_T if hot else lc.C_MUTE, 'middle', True,
            maxw=SEG_W - 12, tag=f'seg{i}v')
    lc.text(x + SEG_W / 2, SEG_Y + 74, zh, 9, '#334155', 'middle', maxw=SEG_W - 10, tag=f'seg{i}z')
    lc.text(x + SEG_W / 2, SEG_Y + 92, sub, 8, lc.C_MUTE, 'middle', maxw=SEG_W - 10, tag=f'seg{i}s')

# 本拍读数注（五段条下方）
lc.text(MX, SEG_Y + SEG_H + 24, '本拍实测（64-token 请求 · connector 答 (32, True) 异步）：ext_comp=32 是唯一非零段——'
        '护轨分配 allocate_slots(32, delay_cache_blocks=True)（kv_cache_manager.py:L549-L552）', 9.5, '#334155',
        'start', maxw=BXR - MX, tag='seg:read')

# ---------------- 双账对比 ----------------
LB_Y = 258
PANEL_H = 224
LW_, RW_ = 672, 672
RX0 = MX + LW_ + 20
# 左：块表（物理占用）
lc.rect(MX, LB_Y, LW_, PANEL_H, '#ffffff', lc.C_KV_S, rx=9, sw=1.8)
lc.text(MX + 18, LB_Y + 26, '账① 块表（物理占用）', 12, lc.C_KV_S, 'start', True, maxw=300, tag='l1:t')
lc.text(MX + 18, LB_Y + 48, 'ext 32 token → 挂 2 块（block_size=16）：block_ids [1, 2]，记在 r1 名下', 9.5,
        '#334155', 'start', maxw=LW_ - 36, tag='l1:l1')
# 两个块格
bx0, by0, bs = MX + 18, LB_Y + 62, 76
for i, bid in enumerate([1, 2]):
    x = bx0 + i * (bs + 12)
    lc.rect(x, by0, bs, 58, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.5)
    lc.text(x + bs / 2, by0 + 24, f'块 {bid}', 11, lc.C_KV_S, 'middle', True, maxw=bs - 8, tag=f'blk{bid}')
    lc.text(x + bs / 2, by0 + 44, '16 token', 8.5, lc.C_MUTE, 'middle', maxw=bs - 8, tag=f'blk{bid}s')
lc.text(bx0 + 2 * (bs + 12) + 8, by0 + 24, '挂在 r1 块表上', 9.5, '#334155', 'start', maxw=250, tag='l1:l2')
lc.text(bx0 + 2 * (bs + 12) + 8, by0 + 42, '（物理占用、从池里划走）', 8.5, lc.C_MUTE, 'start', maxw=250, tag='l1:l3')
# 池条：64 格
POOL_Y = LB_Y + 140
lc.text(MX + 18, POOL_Y - 8, '池 64 块 · free 63 → 61（2 块被划走）', 9, lc.C_MUTE, 'start', maxw=400, tag='pool:t')
CELL, CGAP = 8.6, 1.6
px = MX + 18
for b in range(64):
    if b == 0:
        fill, stroke = '#cbd5e1', lc.C_MUTE          # 基线已占
    elif b in (1, 2):
        fill, stroke = lc.C_BEAT_S, lc.C_BEAT_S      # r1 ext_comp 划走
    else:
        fill, stroke = '#ffffff', '#cbd5e1'          # 空闲
    lc.rect(px + b * (CELL + CGAP), POOL_Y, CELL, 22, fill, stroke, rx=1.5, sw=0.8)
lc.text(MX + 18, POOL_Y + 40, 'free 61', 8.5, lc.C_BEAT_T, 'start', True, maxw=80, tag='pool:f')
lc.text(MX + 18 + 3 * (CELL + CGAP), POOL_Y + 40, '空闲 61 块', 8, lc.C_MUTE, 'start', maxw=120, tag='pool:f2')
lc.text(BXR - RW_ - 38, POOL_Y + 40, '灰=基线已占 · 白=空闲 · 琥珀=r1 划走', 8, lc.C_FAINT, 'end',
        maxw=280, tag='pool:leg')

# 右：缓存账（哈希表登记）
lc.rect(RX0, LB_Y, RW_, PANEL_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.6)
lc.text(RX0 + 18, LB_Y + 26, '账② 缓存账（哈希表登记）', 12, lc.C_MUTE, 'start', True, maxw=300, tag='r1:t')
lc.text(RX0 + 18, LB_Y + 54, 'num_cached_block = 0', 15, lc.C_TXT, 'start', True, maxw=420, tag='r1:v1')
lc.text(RX0 + 18, LB_Y + 82, '首块哈希 = None', 15, lc.C_TXT, 'start', True, maxw=420, tag='r1:v2')
lc.text(RX0 + 18, LB_Y + 112, '对外不可见：后续请求查前缀缓存，命不中这 2 块——', 9.5, '#334155', 'start',
        maxw=RW_ - 36, tag='r1:l1')
lc.text(RX0 + 18, LB_Y + 130, '块被占着、账上没有；传输完成前无人能复用它们', 9.5, '#334155', 'start',
        maxw=RW_ - 36, tag='r1:l2')
lc.rect(RX0 + 18, LB_Y + 148, 320, 52, '#f8fafc', '#e2e8f0', rx=6, sw=1.0)
lc.text(RX0 + 178, LB_Y + 168, '哈希表里对应条目：', 8.5, lc.C_MUTE, 'middle', maxw=300, tag='hash:t')
lc.text(RX0 + 178, LB_Y + 188, '—— 空（未登记）——', 9.5, lc.C_MUTE, 'middle', True, maxw=300, tag='hash:v')

# 中缝「账实分离」标记
lc.text(MX + LW_ / 2 + 10, LB_Y + 118, '账①≠账②', 12, lc.C_BEAT_T, 'middle', True, maxw=90, tag='neq')
lc.text(MX + LW_ / 2 + 10, LB_Y + 136, '窗口态', 9, lc.C_MUTE, 'middle', maxw=90, tag='neq2')

# ---------------- 窗口状态行 ----------------
WS_Y = LB_Y + PANEL_H + 20
lc.rect(MX, WS_Y, BXR - MX, 56, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.4)
lc.text(MX + 18, WS_Y + 23, '窗口里的请求：r1 停在 WAITING_FOR_REMOTE_KVS——num_computed_tokens 已先行写成 32，'
        '但本拍零前向（scheduled_tokens=0）：先记账、后交货', 10, lc.C_BEAT_T, 'start', True,
        maxw=BXR - MX - 36, tag='ws:1')
lc.text(MX + 18, WS_Y + 43, '传输完成前，这个 32 没有任何一拍前向消费它——它是对未来 KV 的预记', 9, lc.C_MUTE,
        'start', maxw=BXR - MX - 36, tag='ws:2')

# ---------------- 窗口三端点 ----------------
EP_Y = WS_Y + 76
EP_W = (BXR - MX - 2 * 20) / 3
ENDS = [
    ('窗口开启 · 护轨分配', 'allocate_slots(num_external_computed_tokens, delay_cache_blocks=True)', '块先占位、后登记（kv_cache_manager.py:L549-L552）', lc.C_KV_S),
    ('窗口关闭·成功 · 完成回收', '传输完成的提升拍补缓存：欠的哈希登记此刻补上', '同时查全命中：是则退 1 token 重算（要 logits）', lc.C_GPU_S),
    ('窗口关闭·失败 · 失败回滚', '按第一个坏块截断 num_computed_tokens，重算区补登记清零', '截断起点必是块边界（块对齐断言守护）', lc.C_ABORT),
]
for i, (t, l1, l2, col) in enumerate(ENDS):
    x = MX + i * (EP_W + 20)
    lc.rect(x, EP_Y, EP_W, 92, '#ffffff', col, rx=8, sw=1.5)
    lc.text(x + 14, EP_Y + 22, t, 10.5, col, 'start', True, maxw=EP_W - 28, tag=f'ep{i}t')
    lc.text(x + 14, EP_Y + 44, l1, 8.5, '#334155', 'start', maxw=EP_W - 28, tag=f'ep{i}a')
    lc.text(x + 14, EP_Y + 62, l2, 8.5, '#334155', 'start', maxw=EP_W - 28, tag=f'ep{i}b')

# ---------------- 页脚 ----------------
FY = EP_Y + 92 + 30
lc.text(MX, FY, '逐字锚 vllm/v1/core/kv_cache_manager.py:L390-L446（五段布局注释图）· L370-L371（『not cached by vLLM '
                'but cached by the connector』原文无逗号）· L549-L552（delay_cache_blocks 短路入缓存）· vllm/v1/core/sched/scheduler.py:L1023-L1053（等待态先行设置）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '读数（挂 2 块 [1,2] / 缓存账 0 / 首块哈希 None / 先行 32 / free 61/63）取自精简版 companion host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 36

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-allocated-not-cached.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
