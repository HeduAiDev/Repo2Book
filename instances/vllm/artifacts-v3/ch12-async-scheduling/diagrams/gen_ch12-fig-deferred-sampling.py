#!/usr/bin/env python3
"""ch12 机制图 7 · structured+async 的推迟采样（figure_spec ch12-fig-deferred-sampling，模板 flow）

放大自 L0 循环框（loop_box）的采样排程分支——即本章 L2 章图 center 拍片 ③『采样排程 ·
立即/推迟』与 ⑥ 后补采支线的时序展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
图右上角指北小签。

claim：structured+async 的每次 decode 拍都推迟采样——上半段见 pending 不采样
（批先入队、前向不浪费），pop+update_from_output 之后才补 bitmask+sample_tokens
重新入队，调用内顺序恒为 update → sample。

数字全部取自 figure_spec.numbers（置位条件 use_structured_output 且 ph>0：拍1 prefill
ph=0 不置位、拍2 起 decode ph=1 恒置位；调用内顺序拍1 [sample]、拍2/拍3 [update, sample]；
token 到账晚一拍：拍2 补采出的 [8] 在拍3 pop 到账、队列长度恒 1 deferred 批重新入队）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 900
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '缺 token 就推迟采样：批先入队，update 之后才补 bitmask + 采样',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'grammar bitmask 要基于本批将采样位置的前文计算——async 下上一拍 token 可能还在 D2H 路上，'
        '立即采样会采出违反语法的 token', 10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ③ 采样排程 · L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 拍2 一次调用内部：上半段/下半段双条 ----------------
GRP_Y = 92
UP_Y, UP_H = 92, 118
DN_Y, DN_H = 234, 118
lc.rect(MX, UP_Y, 1380, UP_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.3)
lc.text(MX + 14, UP_Y + 18, '上半段（core.py:L652-L687）——见 pending 不采样，批先入队',
        9.5, lc.C_BEAT_T, 'start', True, maxw=600, tag='up:t')
lc.rect(MX, DN_Y, 1380, DN_H, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.3)
lc.text(MX + 14, DN_Y + 18, '下半段（core.py:L689-L739）——pop 完上批、token 记齐，才补采重新入队',
        9.5, lc.C_ENG_S, 'start', True, maxw=640, tag='dn:t')

UP_BOXES = [
    ('schedule 批B', ['pending 置位：', 'use_structured_output', '∧ ph=1>0']),
    ('execute_model', ['前向发起（non_block）', '——不浪费']),
    ('见 pending → 不采样', ['bitmask 缺上一拍 token', '立即采会违规']),
    ('appendleft 入队', ['deferred_scheduler_output', '暂存，等下半段']),
]
DN_BOXES = [
    ('pop 批A', ['收最老批结果']),
    ('update_from_output', ['token 记齐（此刻', '账本里前文全了）']),
    ('bitmask + sample_tokens', ['补采（grammar_output', '同样 non_block）']),
    ('appendleft 重新入队', ['复用同一 exec_future', '——不重跑前向']),
]
BW = (1380 - 28 - 3 * 14) / 4
for row_y, boxes, stroke in [(UP_Y, UP_BOXES, lc.C_BEAT_S), (DN_Y, DN_BOXES, lc.C_ENG_S)]:
    for i, (t, lines) in enumerate(boxes):
        bx = MX + 14 + i * (BW + 14)
        lc.rect(bx, row_y + 28, BW, 74, '#ffffff', stroke, rx=6, sw=1.2)
        lc.text(bx + BW / 2, row_y + 46, t, 9.2, stroke, 'middle', True, maxw=BW - 10, tag='bx' + t[:6])
        for j, ln in enumerate(lines):
            lc.text(bx + BW / 2, row_y + 62 + j * 13.5, ln, 8, '#334155', 'middle',
                    maxw=BW - 10, tag='bxl%s%d' % (t[:6], j))
        if i < 3:
            lc.seg(bx + BW + 1, row_y + 65, bx + BW + 13, row_y + 65, stroke, 1.8, 'std')
# 上半段 → 下半段 转段箭头
lc.seg(MX + 1380 - 200, UP_Y + UP_H + 2, MX + 1380 - 200, DN_Y - 4, lc.C_MUTE, 2.0, 'std')
lc.text(MX + 1380 - 192, UP_Y + UP_H + 14, '队满（或无活可调）→ 进下半段', 8.4, lc.C_MUTE,
        'start', maxw=210, tag='trans')
# 顺序徽标（调用内顺序）+ 约束虚线：徽标 → update_from_output 框顶
ORD_X, ORD_Y = MX + 330, UP_Y + UP_H + 13
lc.rect(ORD_X - 108, ORD_Y - 11, 216, 22, '#ffffff', lc.C_ABORT, rx=11, sw=1.3)
lc.text(ORD_X, ORD_Y + 3.5, '调用内顺序恒为 update → sample', 9, lc.C_ABORT, 'middle', True,
        maxw=204, tag='ord')
lc.parrow([(ORD_X + 108, ORD_Y), (MX + 660, ORD_Y), (MX + 660, DN_Y + 27)], lc.C_ABORT, 1.4,
          'ab', dash=True)

# ---------------- 逐拍时序条（拍1/拍2/拍3） ----------------
TB_Y, TB_H = 386, 196
lc.rect(MX, TB_Y, 1380, TB_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(MX + 16, TB_Y + 20, '逐拍时序（prompt=2、max_tokens=8、use_structured_output=True 实拍）',
        9.8, lc.C_TXT, 'start', True, maxw=600, tag='tb:t')
COLW = (1380 - 32 - 2 * 14) / 3
BEATS = [
    ('拍1 prefill', ['ph=0 → 不置位', '调用内顺序 [sample]', '立即采样、正常入队', '（队未满 return None）'], False, None),
    ('拍2 decode', ['ph=1 → pending 置位', '调用内顺序 [update, sample]', 'pop 批A 交货 [7]', '补采的 [8] 晚一拍'], True, '[8]'),
    ('拍3 pop deferred 批', ['又一轮 decode 拍', '调用内顺序 [update, sample]', '拍2 补采的 [8] 到账', '队列长度恒 1（重新入队）'], True, None),
]
for i, (t, lines, hot, late) in enumerate(BEATS):
    bx = MX + 16 + i * (COLW + 14)
    lc.rect(bx, TB_Y + 32, COLW, 126, lc.C_BEAT_F if hot else '#f8fafc',
            lc.C_BEAT_S if hot else lc.C_MUTE, rx=6, sw=1.2)
    lc.text(bx + COLW / 2, TB_Y + 50, t, 9.4, lc.C_BEAT_T if hot else lc.C_TXT, 'middle', True,
            maxw=COLW - 12, tag='bt' + t[:6])
    for j, ln in enumerate(lines):
        lc.text(bx + COLW / 2, TB_Y + 68 + j * 15, ln, 8.5, '#334155', 'middle',
                maxw=COLW - 14, tag='btl%s%d' % (t[:6], j))
    if i < 2:
        lc.seg(bx + COLW + 1, TB_Y + 95, bx + COLW + 13, TB_Y + 95, lc.C_MUTE, 1.8, 'std')
lc.text(MX + 690, TB_Y + 178, '置位条件 = use_structured_output ∧ ph>0 —— decode 稳态 ph 恒 ≥1，'
        '所以每个 structured decode 批都走 deferred', 8.6, lc.C_MUTE, 'middle', maxw=1200, tag='tb:note')

# ---------------- 代价 / 换来 双框 ----------------
COST_Y, COST_H = TB_Y + TB_H + 16, 96
CW_HALF = (1380 - 20) / 2
lc.rect(MX, COST_Y, CW_HALF, COST_H, '#ffffff', lc.C_ABORT, rx=7, sw=1.2, dash=True)
lc.text(MX + 14, COST_Y + 19, '代价（写进 WC3）', 9.5, lc.C_ABORT, 'start', True, maxw=200, tag='cost:t')
lc.text(MX + 14, COST_Y + 40, '每个 structured decode 批在队列里多躺一轮、', 8.5, '#334155',
        'start', maxw=CW_HALF - 28, tag='cost:1')
lc.text(MX + 14, COST_Y + 56, 'token 到账晚 1 拍（拍2 补采的 [8] 拍3 才到）——', 8.5, '#334155',
        'start', maxw=CW_HALF - 28, tag='cost:2')
lc.text(MX + 14, COST_Y + 72, 'TPOT 多一段排队延迟', 8.5, '#334155', 'start', maxw=CW_HALF - 28, tag='cost:3')
GAIN_X = MX + CW_HALF + 20
lc.rect(GAIN_X, COST_Y, CW_HALF, COST_H, '#ffffff', lc.C_GPU_S, rx=7, sw=1.2)
lc.text(GAIN_X + 14, COST_Y + 19, '换来', 9.5, lc.C_GPU_S, 'start', True, maxw=120, tag='gain:t')
lc.text(GAIN_X + 14, COST_Y + 40, '掩码永远基于齐全 token 计算（update → sample 的顺序', 8.5,
        '#334155', 'start', maxw=CW_HALF - 28, tag='gain:1')
lc.text(GAIN_X + 14, COST_Y + 56, '是正确性约束、不是风格选择）——零违规采样；', 8.5,
        '#334155', 'start', maxw=CW_HALF - 28, tag='gain:2')
lc.text(GAIN_X + 14, COST_Y + 72, '前向已发起不浪费：批先排上去，只挪采样这一道工序', 8.5,
        '#334155', 'start', maxw=CW_HALF - 28, tag='gain:3')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = COST_Y + COST_H + 26
lx = MX
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, '上半段 / deferred 拍', 8.5, lc.C_TXT, 'start', maxw=180, tag='leg:up')
lx += 28 + lc.tw('上半段 / deferred 拍', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, '下半段（pop + 补采）', 8.5, lc.C_TXT, 'start', maxw=190, tag='leg:dn')
lx += 28 + lc.tw('下半段（pop + 补采）', 8.5) + 16
lc.seg(lx + 4, LEG_Y - 3, lx + 34, LEG_Y - 3, lc.C_ABORT, 1.4, dash=True)
lc.text(lx + 42, LEG_Y + 1, 'update → sample 顺序约束', 8.5, lc.C_TXT, 'start', maxw=200, tag='leg:ord')
lx += 42 + lc.tw('update → sample 顺序约束', 8.5) + 16
lc.rect(lx, LEG_Y - 11, 22, 15, '#ffffff', lc.C_ABORT, rx=3, sw=1.1, dash=True)
lc.text(lx + 28, LEG_Y + 1, '代价框', 8.5, lc.C_TXT, 'start', maxw=80, tag='leg:cost')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/engine/core.py:L665-L677（上半段分流：无 pending 立即采样 / 有 pending 暂存）· '
        'L719-L737（下半段补采重新入队，复用同一 exec_future）· vllm/v1/core/sched/async_scheduler.py:L31-L33'
        '（pending_structured_output_tokens 置位）· 时序数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-deferred-sampling.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
