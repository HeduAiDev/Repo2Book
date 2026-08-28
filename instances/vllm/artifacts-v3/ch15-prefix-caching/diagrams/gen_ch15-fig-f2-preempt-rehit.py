#!/usr/bin/env python3
"""ch15 机制图 8 · 抢占哈希保留与重命中（figure_spec ch15-fig-f2-preempt-rehit，模板 state-machine）

放大自 L0 KV 账本列（kv_column）缓存区·留与逐——「抢占哈希保留」到「惰性驱逐」的回环展开
（F2 伏笔在此收口：ch11 埋、ch15 收）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：抢占 free 不清哈希——被打回 waiting 的请求重排回来重走准入查询、重命中自己的前缀、
touch 救回：『重算』变『重载元数据+补算』，只有块被取走复用（惰性驱逐）才退化为全量重 prefill。

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
RED = '#dc2626'
ORANGE = '#ea580c'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '抢占清块不清哈希：重排回来重命中自己的前缀——『重算』变『重载元数据 + 补算』',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, 'F2 收口（ch11 埋）：被抢占的请求回 waiting 队头，重排回来第一站就重走准入查询——只有抢占期间块被取走复用（惰性驱逐摘光哈希），才退化为全量重 prefill',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 留与逐「F2 抢占回环」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96

# ---------------- 主线：三个状态 + 转移 ----------------
SW, SH_ = 360, 118
GAP = 110
SY = LY + 70
SX1 = 70
SX2 = SX1 + SW + GAP
SX3 = SX2 + SW + GAP
STATES = [
    (SX1, '① RUNNING', '已算 64 token · 4 满块在表',
     ['num_computed_tokens = 64', 'map 条目 4（块 1-4 的哈希）', 'block_size = 16 · prompt 64'],
     lc.C_ENG_F, lc.C_ENG_S),
    (SX2, '② PREEMPTED → waiting 队头', 'free 全部块 · 哈希保留',
     ['num_computed_tokens = 0', 'map 条目仍是 4——free 不碰哈希', 'num_preemptions = 1'],
     '#fff7ed', ORANGE),
    (SX3, '③ 重排回来 · 重走准入', '重命中 48 · touch 救回',
     ['重命中 48 token（块 1、2、3）', 'touch 出队救回 + 补算 16', '重算量 16 / 64 = 25%（无缓存则 64）'],
     '#f0fdf4', GREEN),
]
for (sx, nm, sub, lines, fill, stroke) in STATES:
    lc.rect(sx, SY, SW, SH_, fill, stroke, rx=9, sw=1.6)
    lc.text(sx + 16, SY + 24, nm, 11, stroke, 'start', True, maxw=SW - 32, tag='st:' + nm[:6])
    lc.text(sx + 16, SY + 44, sub, 9, lc.C_TXT, 'start', True, maxw=SW - 32, tag='ss:' + nm[:6])
    for i, ln in enumerate(lines):
        lc.text(sx + 16, SY + 66 + i * 17, '· ' + ln, 8.4, '#334155', 'start', maxw=SW - 30,
                tag='sl:%d' % i)

# 转移 ①→②
lc.seg(SX1 + SW + 6, SY + SH_ / 2, SX2 - 6, SY + SH_ / 2, ORANGE, 2.0, 'std')
lc.text((SX1 + SW + SX2) / 2, SY + SH_ / 2 - 30, '池紧触发抢占', 8.8, ORANGE, 'middle', True,
        maxw=GAP - 6, tag='t12:a')
lc.text((SX1 + SW + SX2) / 2, SY + SH_ / 2 + 18, '_preempt_request →', 7.6, GRAY, 'middle',
        maxw=GAP - 4, tag='t12:b')
lc.text((SX1 + SW + SX2) / 2, SY + SH_ / 2 + 32, '_free_request_blocks', 7.6, GRAY, 'middle',
        maxw=GAP - 4, tag='t12:c')
# 转移 ②→③
lc.seg(SX2 + SW + 6, SY + SH_ / 2, SX3 - 6, SY + SH_ / 2, GREEN, 2.0, 'std')
lc.text((SX2 + SW + SX3) / 2, SY + SH_ / 2 - 30, '重排回来（主线）', 8.8, GREEN, 'middle', True,
        maxw=GAP - 6, tag='t23:a')
lc.text((SX2 + SW + SX3) / 2, SY + SH_ / 2 + 18, '块还在表里：', 7.6, GRAY, 'middle',
        maxw=GAP - 4, tag='t23:b')
lc.text((SX2 + SW + SX3) / 2, SY + SH_ / 2 + 32, '重查准入', 7.6, GRAY, 'middle', maxw=GAP - 4,
        tag='t23:c')

# ---------------- 最坏分支：从 ② 下行 ----------------
WY = SY + SH_ + 76
WX, WW = SX2 - 20, 560
lc.parrow([(SX2 + SW / 2, SY + SH_), (SX2 + SW / 2, WY)], RED, 1.8, 'ab', dash=True)
lc.text(SX2 + SW / 2 + 12, (SY + SH_ + WY) / 2 + 4,
        '抢占期间池被抽干（8 块小池）：块被取走复用、惰性驱逐摘光哈希（map → 0）', 8.6, RED,
        'start', maxw=560, tag='t24')
lc.rect(WX, WY, WW, 104, '#fef2f2', RED, rx=9, sw=1.6)
lc.text(WX + 16, WY + 24, '④ 最坏分支：重排回来 · 命中 0', 11, RED, 'start', True, maxw=WW - 32,
        tag='w:t')
lc.text(WX + 16, WY + 44, '前缀失效 → 全量重 prefill', 9, lc.C_TXT, 'start', True, maxw=WW - 32,
        tag='w:s')
lc.text(WX + 16, WY + 66, '· 重算量涨回上界：补 48 token（本例 prompt 48 时即 100%）', 8.4,
        '#334155', 'start', maxw=WW - 30, tag='w:l1')
lc.text(WX + 16, WY + 84, '· 四步链路任何一环被破坏（如某新路径 free 时清了哈希）都落到这里', 8.4,
        '#334155', 'start', maxw=WW - 30, tag='w:l2')

# ---------------- 右下：重算量区间 ----------------
RXX = WX + WW + 24
RWW = BXR - RXX
lc.rect(RXX, WY, RWW, 104, '#f8fafc', GRAY, rx=9, sw=1.2, dash=True)
lc.text(RXX + 16, WY + 24, '重算量区间 [1, P]', 10.5, lc.C_TXT, 'start', True, maxw=RWW - 32,
        tag='rg:t')
lc.text(RXX + 16, WY + 46, 'P = prompt 长。下界 1：全命中仍须重算', 8.4, '#334155', 'start',
        maxw=RWW - 32, tag='rg:l1')
lc.text(RXX + 16, WY + 63, '最后一个 token 拿 logits；上界 P：块全', 8.4, '#334155', 'start',
        maxw=RWW - 32, tag='rg:l2')
lc.text(RXX + 16, WY + 80, '被惰性驱逐时全量重 prefill。', 8.4, '#334155', 'start',
        maxw=RWW - 32, tag='rg:l3')
lc.text(RXX + 16, WY + 97, '主线 16（25%）· 最坏 48/64', 8.4, GREEN, 'start', True,
        maxw=RWW - 32, tag='rg:l4')

# ---------------- 底部不变量条（全宽） ----------------
BY = WY + 104 + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '四步隐式链路：free 不清哈希 → 逆序 + 劈分把带哈希块挂 LRU 尾（给生存窗口续命）→ 重排回 waiting 重走准入 → touch 救回',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '任何一环被破坏即退化为上界——正确性靠链路维持，没有断言兜底；重命中长度 = min(块对齐削减后的缓存链, 仍在表中的最长链)',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, tcol, name in [
        ('#fff7ed', ORANGE, ORANGE, '主线状态（② 被抢占）'),
        ('#f0fdf4', GREEN, GREEN, '主线恢复（③ 重命中）'),
        ('#fef2f2', RED, RED, '最坏分支（④ 全量重算）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.seg(lx + 4, LEG_Y - 3, lx + 34, LEG_Y - 3, RED, 1.8, 'ab', dash=True)
lc.text(lx + 40, LEG_Y + 1, '块被取走复用（惰性驱逐）', 8.8, lc.C_TXT, 'start', maxw=200,
        tag='lg:d')
lx += 40 + lc.tw('块被取走复用（惰性驱逐）', 8.8) + 24
lc.text(lx, LEG_Y + 1, 'map 条目 = 平面哈希表里的键数；touch = 引用计数 +1 且出队救回', 8.8,
        lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L1274-L1315（_preempt_request：free 块、归零、回队头）· '
        'vllm/v1/core/kv_cache_manager.py:L567-L578（free：只动 ref_cnt 与队列，不碰哈希）· vllm/v1/core/block_pool.py:L647-L700（get_new_blocks → 惰性驱逐）· 抢占机制本体归 ch11',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（64 token · 4 满块；最坏分支 8 块小池）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=640, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-f2-preempt-rehit.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
