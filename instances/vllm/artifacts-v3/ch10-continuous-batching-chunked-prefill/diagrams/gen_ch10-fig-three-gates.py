#!/usr/bin/env python3
"""ch10 机制图 6 · 切块三闸（figure_spec ch10-fig-three-gates，模板 flow）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框的「chunked prefill
切块」格——即本章 L2 章图 center ④ 拍片『chunked prefill 切块』的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：同一 70-token prompt 过三道串行闸的三种命运：预算闸切成 [32,32,6]、threshold 闸
切成 [16,16,16,16,6]、chunked 关闸则每拍 break 整体拒收——三闸全是 min 型下钳，只决定
「这一拍切多大」，不改变追赶目标。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：场景 a chunk
[32,32,6]、原始差 70/38/6；场景 b threshold 16 → [16,16,16,16,6]、拍 5 差 6<16 不再钳；
场景 c chunked 关 + 70>32 → 每拍 break、状态 WAITING、已算恒 0；闸序行号 L899-L901 →
L903-L911 → L913+L914）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 806
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '切块三闸：同一块 70-token 的蛋糕，三种切法、或一拍都不收',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'num_new_tokens = num_tokens − num_computed_tokens → ① 刀长 threshold 钳制 → ② 切块开关 → ③ 盘余 min(token_budget)（scheduler.py:L874-L914）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ④ chunked prefill 切块 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 闸门管道（上半） ----------------
PIPE_Y, PIPE_H = 96, 128
ENTRY_X, ENTRY_W = 60, 120
G1_X, G_W, G_GAP = 220, 250, 26
G2_X = G1_X + G_W + G_GAP          # 496
G3_X = G2_X + G_W + G_GAP          # 772
EXIT_X, EXIT_W = 1048, 120

# 入口：原始差数字球
lc.circle(ENTRY_X + ENTRY_W / 2, PIPE_Y + PIPE_H / 2, 34, lc.C_KV_S, 1.8, dash=False)
lc.text(ENTRY_X + ENTRY_W / 2, PIPE_Y + 52, '原始差', 8.5, lc.C_TXT, 'middle', True, maxw=60, tag='en:t')
lc.text(ENTRY_X + ENTRY_W / 2, PIPE_Y + 74, '70', 17, lc.C_KV_S, 'middle', True, maxw=60, tag='en:v')
lc.text(ENTRY_X + ENTRY_W / 2, PIPE_Y + 98, 'num_tokens−已算', 7, lc.C_MUTE, 'middle', maxw=66, tag='en:s')

def gate(x, title, lines, hot=False):
    lc.rect(x, PIPE_Y, G_W, PIPE_H, lc.C_BEAT_F if hot else '#ffffff',
            lc.C_BEAT_S if hot else lc.C_ENG_S, rx=8, sw=1.8 if hot else 1.6)
    lc.text(x + 14, PIPE_Y + 22, title, 10.5, lc.C_BEAT_T if hot else lc.C_TXT, 'start', True,
            maxw=G_W - 28, tag='g:' + title[:8])
    for i, ln in enumerate(lines):
        lc.text(x + 14, PIPE_Y + 44 + i * 17, ln, 8.5, '#334155', 'start', maxw=G_W - 28,
                tag=f'g:{title[:4]}:{i}')

gate(G1_X, '闸 ① 刀长 · threshold', [
    'long_prefill_token_threshold',
    '（L899-L901）',
    '0 = 不钳（默认禁用）；16 → 把 70 钳成 16',
    '用途：钉住单拍插入的计算量，护 TPOT',
])
gate(G2_X, '闸 ② 开关 · chunked', [
    'enable_chunked_prefill',
    '（L903-L911）',
    '关且超预算 → 整拍 break（不偷切半块）',
    'v1 默认开（config/scheduler.py:L74）',
])
gate(G3_X, '闸 ③ 盘余 · min(token_budget)', [
    'num_new_tokens = min(num_new_tokens,',
    'token_budget)（L913）',
    '32 预算把 70 钳成 32；随后 assert > 0（L914）',
    '非空保证：正差经 min 型下钳不会变 0',
], hot=True)

# 出口：本拍 chunk
lc.rect(EXIT_X, PIPE_Y, EXIT_W, PIPE_H, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.8)
lc.text(EXIT_X + EXIT_W / 2, PIPE_Y + 30, '本拍 chunk', 11, lc.C_KV_S, 'middle', True,
        maxw=110, tag='ex:t')
lc.text(EXIT_X + EXIT_W / 2, PIPE_Y + 62, '∈ [1, min(差,', 9.5, lc.C_TXT, 'middle', maxw=110, tag='ex:l1')
lc.text(EXIT_X + EXIT_W / 2, PIPE_Y + 78, 'threshold?, 预算)]', 9.5, lc.C_TXT, 'middle', maxw=110, tag='ex:l2')
lc.text(EXIT_X + EXIT_W / 2, PIPE_Y + 104, '追赶目标不变', 8.5, lc.C_MUTE, 'middle', maxw=110, tag='ex:l3')

# 管道箭头
mid_y = PIPE_Y + PIPE_H / 2
lc.seg(ENTRY_X + ENTRY_W + 2, mid_y, G1_X - 3, mid_y, lc.C_ENG_S, 2.0, 'std')
lc.text((ENTRY_X + ENTRY_W + G1_X) / 2, mid_y - 26, '70 滚入', 8.5, lc.C_ENG_S, 'middle', maxw=70, tag='a:in')
lc.seg(G1_X + G_W + 2, mid_y, G2_X - 3, mid_y, lc.C_ENG_S, 2.0, 'std')
lc.seg(G2_X + G_W + 2, mid_y, G3_X - 3, mid_y, lc.C_ENG_S, 2.0, 'std')
lc.seg(G3_X + G_W + 2, mid_y, EXIT_X - 3, mid_y, lc.C_ENG_S, 2.0, 'std')
# 闸 ② 的 break 自环（关 → 整拍不收 → 回 waiting，下一拍同一颗 70 再滚进来）
BX = G2_X + G_W / 2
BYPASS_Y = 70
lc.parrow([(BX, PIPE_Y), (BX, BYPASS_Y), (BX - 130, BYPASS_Y), (BX - 130, PIPE_Y)],
          lc.C_ABORT, 1.8, 'ab')
lc.text(BX - 65, BYPASS_Y + 14, 'break · 回 waiting', 8.5, lc.C_ABORT, 'middle', True, maxw=120, tag='a:back')
lc.text(BX + 130, 90, '关 且 差 > 预算 → 整拍不收（70-token 在 32 预算下一拍都不进）——下一拍再从入口滚进来',
        8.5, lc.C_ABORT, 'start', True, maxw=560, tag='a:break')

# ---------------- 三条命运走线（下半） ----------------
FATE_Y0 = 262
FATE_H = 128
FATE_GAP = 18
TOKPX = 8.2   # chunk 条 px/token（70 → 574px）

def fate(y, tag, cfg, chunks, note):
    lc.rect(MX, y, BXR - MX, FATE_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
    lc.text(MX + 14, y + 20, tag, 10, lc.C_TXT, 'start', True, maxw=560, tag=f'f:{tag[:8]}')
    lc.text(MX + 14, y + 38, cfg, 8.5, lc.C_MUTE, 'start', maxw=640, tag=f'f:cfg{tag[:6]}')
    lc.text(MX + 14, y + 58, note, 8.5, '#334155', 'start', maxw=760, tag=f'f:nt{tag[:6]}')
    # chunk 序列条（宽 ∝ token 数）
    by = y + 74
    x = MX + 14
    for i, c in enumerate(chunks):
        w = c * TOKPX
        lc.rect(x, by, w, 26, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
        lc.text(x + w / 2, by + 17, str(c), 9, '#ffffff', 'middle', True, maxw=w - 4,
                tag=f'f:ck{i}{tag[:4]}')
        lc.text(x + w / 2, by + 42, f'拍{i + 1}', 7.5, lc.C_MUTE, 'middle', maxw=40,
                tag=f'f:pb{i}{tag[:4]}')
        x += w + 6
    return x

y_a = FATE_Y0
x_end_a = fate(y_a, '命运 a · 预算闸当家（threshold=0，chunked 开，预算 32）',
               '配置：max_num_batched_tokens=32 · long_prefill_token_threshold=0',
               [32, 32, 6],
               '原始差 70/38/6：前两拍被盘余 32 钳、第三拍差 6 < 32 直通——⌈70/32⌉=3 拍收官')
lc.text(x_end_a + 12, y_a + 90, '→ 转出 decode 领 1', 8.5, lc.C_MUTE, 'start', maxw=130, tag='f:a:next')

y_b = FATE_Y0 + FATE_H + FATE_GAP
x_end_b = fate(y_b, '命运 b · 刀长闸当家（threshold=16，chunked 开，预算 2048）',
               '配置：max_num_batched_tokens=2048 · long_prefill_token_threshold=16',
               [16, 16, 16, 16, 6],
               '差 70/54/38/22 前四拍全被 16 钳；拍 5 差 6 < 16 不再钳——⌈70/16⌉=5 拍，TTFT 换 TPOT')
lc.text(x_end_b + 12, y_b + 90, '→ 5 拍后转 decode', 8.5, lc.C_MUTE, 'start', maxw=130, tag='f:b:next')

y_c = FATE_Y0 + 2 * (FATE_H + FATE_GAP)
lc.rect(MX, y_c, BXR - MX, FATE_H, '#ffffff', lc.C_ABORT, rx=7, sw=1.3, dash=True)
lc.text(MX + 14, y_c + 20, '命运 c · 开关关死（chunked 关，预算 32）', 10, lc.C_ABORT, 'start', True,
        maxw=560, tag='f:c:t')
lc.text(MX + 14, y_c + 38, '配置：enable_chunked_prefill=False · max_num_batched_tokens=32', 8.5,
        lc.C_MUTE, 'start', maxw=640, tag='f:c:cfg')
lc.text(MX + 14, y_c + 58, '70 > 32 → 每拍 break：状态一直 WAITING、已算恒 0——「宁可不排」的显式拒绝，不是悄悄切一块',
        8.5, '#334155', 'start', maxw=820, tag='f:c:nt')
lc.text(MX + 14, y_c + 96, '已算 0 · 未进批', 9, lc.C_ABORT, 'start', True, maxw=140, tag='f:c:st')
# 空转循环：waiting → 下一拍再问 → break（回环箭头）
lb_x = MX + 500
lc.parrow([(lb_x, y_c + 76), (lb_x, y_c + 98), (lb_x + 190, y_c + 98), (lb_x + 190, y_c + 80)],
          lc.C_ABORT, 1.5, 'ab')
lc.text(lb_x + 95, y_c + 114, '每拍重问、每拍 break（chunk 恒为空）', 8, lc.C_ABORT,
        'middle', maxw=280, tag='f:c:loop')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = y_c + FATE_H + 26
lx = MX
items = [
    ('chunk', '本拍 chunk 条（宽 ∝ token 数）'),
    ('hot', 'v1 默认在用的闸（threshold=0、chunked 开）'),
]
for kind, name in items:
    if kind == 'chunk':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    else:
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=340, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 22
lc.seg(lx, LEG_Y - 2, lx + 20, LEG_Y - 2, lc.C_ABORT, 1.5, 'ab')
lc.text(lx + 26, LEG_Y + 2, 'break（整拍不收）', 8.5, lc.C_TXT, 'start', maxw=140, tag='leg:brk')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L874-L914（切块三闸）· L899-L901（threshold 钳制）· L903-L911（chunked 关 break）· '
        'L913-L914（min(token_budget) + assert）· 默认开 vllm/config/scheduler.py:L74', 8.5, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '三种命运读数取自精简版 companion host 实测 · threshold 闸在 RUNNING 侧续 chunk 同样生效（L521-L522 同一钳制）· '
        '混相批理论外证 Sarathi-Serve（arXiv:2308.16369，非源码）· 行号基线 vLLM v0.27.1', 8.5, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-three-gates.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
