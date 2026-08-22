#!/usr/bin/env python3
"""ch08 机制图 4 · FlatLogprobs vs nested（figure_spec ch08-fig-flat-vs-nested，模板 layout）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『上行泳道 logprobs 支路』）的落容器工位——
即本章 L2 章图 center 拍片 ⑨ 『落容器』+ south『why · FlatLogprobs』注的机制展开（上游拍片
⑧ 『U+FFFD 修正』送文本、下游拍片 ⑩ 『出口装车』取容器）。非新架构画法，架构归属回指 L0/L2。

claim：同一份 logprobs 数据两种住法：nested 每位置 1 个 dict + 每条 1 个 Logprob 对象
（L=100 → 实测 301 个受跟踪对象、L=1000 → 3001 线性涨），flat 摊进 6 条平行原生列表 +
区间索引（实测恒 7）——GC 账从 O(L×k) 降到 O(1)，代价是读侧每次现造 dict。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 696
MX, BXR = 60, 1440
MIDX = 750

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'FlatLogprobs：把「每页一本相册」摊成「6 条平行长卷」——GC 对象账 O(L×k) → O(1)',
        16.5, lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, '同一份 L=100 / k=2 数据两种住法：nested 实测 301 个受跟踪对象、10 倍位置线性涨到 3001；'
        'flat 实测恒 7', 10.5, lc.C_MUTE, 'start', maxw=900, tag='subtitle')
_ch = '放大自 L2 拍片 ⑨ 落容器 · L0：上行泳道 logprobs 支路'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')
_up = '← 上游 · L2 拍片 ⑧ U+FFFD 修正：修正后的文本进来'
_uw = lc.tw(_up, 9) + 14
lc.rect(MX, 76, _uw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(MX + _uw / 2, 89.5, _up, 9, lc.C_MUTE, 'middle', maxw=_uw - 4, tag='chip:up')

# ---------------- 左面板：nested 相册 ----------------
LP = (MX, 116, 630, 358)
lc.text(LP[0], 112, 'nested：list[dict[int, Logprob]]（默认）', 11.5, lc.C_API_S, 'start', True,
        maxw=520, tag='pt:L')
lc.rect(*LP[:2], LP[2], LP[3], '#ffffff', lc.C_API_S, rx=10, sw=1.6)

ALB_X, ALB_W, ALB_H = 116, 470, 64
for a in range(3):
    y = 140 + a * 76
    lc.rect(ALB_X, y, ALB_W, ALB_H, lc.C_API_F, lc.C_API_S, rx=7, sw=1.2)
    lc.text(ALB_X + 12, y + 16, f'dict · 位置 {a + 1}', 8, lc.C_API_S, 'start', True, maxw=130,
            tag=f'alb{a}')
    for c in range(2):
        cx = ALB_X + 22 + c * 226
        lc.rect(cx, y + 24, 212, 32, '#ffffff', lc.C_MUTE, rx=5, sw=1.0)
        lc.text(cx + 10, y + 36, 'token_id → Logprob', 7.5, lc.C_TXT, 'start', True, maxw=190,
                tag=f'card{a}{c}')
        lc.text(cx + 10, y + 50, 'logprob · rank · decoded_token', 7, lc.C_MUTE, 'start',
                maxw=190, tag=f'fld{a}{c}')
lc.text(ALB_X + ALB_W / 2, 140 + 3 * 76 - 8, '…… 共 100 本（每位置去重后 2 条）', 8.5,
        lc.C_MUTE, 'middle', maxw=280, tag='ell')
lc.seg(104, 140, 104, 372, lc.C_API_S, 1.3)
lc.seg(104, 140, 110, 140, lc.C_API_S, 1.3)
lc.seg(104, 372, 110, 372, lc.C_API_S, 1.3)
lc.text(150, 390, '外层 list × 1 —— 把 100 本装订成册（书架）', 8.5, lc.C_MUTE, 'start',
        maxw=330, tag='shelf')
lc.text(MX + LP[2] / 2, 428, '100 本 dict + 200 张 Logprob + 1 个 list = 301 个受跟踪对象'
        '（gc 实测 301）', 10, lc.C_TXT, 'middle', True, maxw=LP[2] - 20, tag='cnt:L')
_xw = lc.tw('×10 → L=1000：3001（线性涨）', 9, True) + 16
lc.rect(MX + LP[2] / 2 - _xw / 2, 442, _xw, 19, '#fff7ed', lc.C_ENG_S, rx=9, sw=1.2)
lc.text(MX + LP[2] / 2, 455.5, '×10 → L=1000：3001（线性涨）', 9, lc.C_ENG_S, 'middle', True,
        maxw=_xw - 6, tag='x10:L')

# ---------------- 右面板：flat 长卷 ----------------
RP = (810, 116, 630, 358)
lc.text(RP[0], 112, 'FlatLogprobs（SamplingParams.flat_logprobs=True 可选）', 11.5, lc.C_API_S,
        'start', True, maxw=540, tag='pt:R')
lc.rect(*RP[:2], RP[2], RP[3], '#ffffff', lc.C_API_S, rx=10, sw=1.6)

STRIP_NAMES = ['start_indices', 'end_indices', 'token_ids', 'logprobs', 'ranks',
               'decoded_tokens']
SX, SW = 952, 452                    # 长卷条 x / 宽
for i, nm in enumerate(STRIP_NAMES):
    y = 140 + i * 30
    lc.text(SX - 12, y + 14, nm, 8.5, lc.C_TXT, 'end', maxw=118, tag='sn' + nm)
    if i < 2:                         # 索引卷：标头部实测值
        vals = ['0', '2', '4', '…'] if i == 0 else ['2', '4', '6', '…']
        for j, v in enumerate(vals):
            cx = SX + j * 46
            hot = (j == 1)
            lc.rect(cx, y, 40, 20, lc.C_SAM_F if hot else '#ffffff',
                    lc.C_SAM_S if hot else lc.C_MUTE, rx=4, sw=1.2 if hot else 0.9)
            lc.text(cx + 20, y + 13.5, v, 8.5, lc.C_SAM_S if hot else lc.C_MUTE, 'middle',
                    hot, maxw=30, tag=f'iv{i}{j}')
        lc.text(SX + 4 * 46 + 8, y + 13.5, '（头部）', 7.5, lc.C_FAINT, 'start', maxw=60,
                tag=f'ivh{i}')
    else:                             # 数据卷：原语刻度（不标数值）
        for j in range(8):
            cx = SX + j * 56
            hot = j in (2, 3)
            lc.rect(cx, y, 50, 20, lc.C_SAM_F if hot else '#ffffff',
                    lc.C_SAM_S if hot else lc.C_MUTE, rx=4, sw=1.2 if hot else 0.9)
        if i == 2:
            lc.text(SX + 8 * 56 - 6, 322, '数据卷每条 L×k = 200 条原语', 8, lc.C_FAINT, 'end',
                    maxw=140, tag='prim')
# 区间竖界（穿过数据卷）+ 底部括注
IVX0, IVX1 = SX + 2 * 56 + 2, SX + 4 * 56 - 4
lc.seg(IVX0, 196, IVX0, 332, lc.C_SAM_S, 1.1, dash=True)
lc.seg(IVX1, 196, IVX1, 332, lc.C_SAM_S, 1.1, dash=True)
lc.seg(IVX0, 332, IVX1, 332, lc.C_SAM_S, 1.1)
lc.text(SX + 4 * 56 + 8, 348, '位置 i 的区间 = [start_indices[i], end_indices[i])', 8,
        lc.C_SAM_S, 'start', maxw=252, tag='ivlbl')
BUB = (952, 366, SW, 52)
lc.rect(*BUB[:2], BUB[2], BUB[3], '#ffffff', lc.C_SAM_S, rx=8, sw=1.2, dash=True)
lc.text(BUB[0] + 12, BUB[1] + 20, 'flat[i]：按区间现造一个 dict（O(k)，不缓存）', 9,
        lc.C_TXT, 'start', True, maxw=BUB[2] - 24, tag='bub:1')
lc.text(BUB[0] + 12, BUB[1] + 40, '读侧从直读变现造——顺序消费为主的 logprobs 场景正合适', 8.5,
        lc.C_MUTE, 'start', maxw=BUB[2] - 24, tag='bub:2')
lc.text(RP[0] + RP[2] / 2, 428, '6 条原生 list + 容器实例 = 7（gc 实测 7）', 10, lc.C_TXT,
        'middle', True, maxw=RP[2] - 20, tag='cnt:R')
_xw2 = lc.tw('×10 → L=1000 仍 7（与 L 无关）', 9, True) + 16
lc.rect(RP[0] + RP[2] / 2 - _xw2 / 2, 442, _xw2, 19, '#fff7ed', lc.C_ENG_S, rx=9, sw=1.2)
lc.text(RP[0] + RP[2] / 2, 455.5, '×10 → L=1000 仍 7（与 L 无关）', 9, lc.C_ENG_S, 'middle',
        True, maxw=_xw2 - 6, tag='x10:R')

# ---------------- 中缝：比值徽章 ----------------
lc.circle(MIDX, 295, 46, lc.C_ENG_S, 1.8, dash=False)
lc.text(MIDX, 292, '428.7×', 16, lc.C_ENG_S, 'middle', True, maxw=80, tag='ratio')
lc.text(MIDX, 310, 'L=1000 时', 8, lc.C_MUTE, 'middle', maxw=70, tag='ratio:1')
lc.text(MIDX, 324, 'nested / flat', 8, lc.C_MUTE, 'middle', maxw=70, tag='ratio:2')
lc.text(MIDX, 360, '受跟踪对象数比值', 8, lc.C_MUTE, 'middle', maxw=100, tag='ratio:3')

# ---------------- 底部三注 ----------------
NB_Y, NB_H = 496, 96
notes = [
    ('读侧：e1 == e2 但 e1 is not e2',
     ['flat[42] 两次调用返回相等但非同一对象——', '每次现造 dict（O(k)）；slice 面可重建', '平移版（DELTA 切尾走这条面）']),
    ('开关 SamplingParams.flat_logprobs',
     ['默认 False——兼容第一：MutableSequence 序列面', '使它直接顶替 list[dict] 的位置；提速可选', '（docstring：GC costs of FlatLogprobs is significantly smaller）']),
    ('计数口径（CPython 语义）',
     ['受跟踪 = 容器对象（dict / Logprob 实例 / list）；', 'int / float / str 原语本就不被 GC 跟踪——', 'flat 的元素照常 L×k = 200 条，只是不逐条成对象']),
]
nw = (BXR - MX - 2 * 20) / 3
for i, (t, lines) in enumerate(notes):
    x = MX + i * (nw + 20)
    lc.rect(x, NB_Y, nw, NB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.1)
    lc.text(x + 14, NB_Y + 20, t, 9.5, lc.C_TXT, 'start', True, maxw=nw - 28, tag='nb' + str(i))
    for k, ln in enumerate(lines):
        lc.text(x + 14, NB_Y + 40 + k * 17, ln, 8.5, '#334155', 'start', maxw=nw - 28,
                tag=f'nb{i}l{k}')

# ---------------- 图例 + 读图行 + 下游 chip + 页脚 ----------------
LEG_Y = 616
lx = MX
for kind, name in [('card', 'Logprob 对象（每条一张卡）'), ('cell', '原生 list 长卷（元素为原语）'),
                   ('sam', '位置 i 的区间 [2, 4)')]:
    if kind == 'card':
        lc.rect(lx, LEG_Y - 8, 22, 13, lc.C_API_F, lc.C_API_S, rx=3, sw=1.1)
    elif kind == 'cell':
        lc.rect(lx, LEG_Y - 8, 22, 13, '#ffffff', lc.C_MUTE, rx=3, sw=1.0)
    else:
        lc.rect(lx, LEG_Y - 8, 22, 13, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.2)
    lc.text(lx + 28, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=280, tag='leg' + kind)
    lx += 28 + lc.tw(name, 9) + 20
_dn = '→ 下游 · L2 拍片 ⑩ 出口装车：DELTA 切尾走 flat 的 slice 面'
_dw = lc.tw(_dn, 9) + 14
lc.rect(BXR - _dw, LEG_Y - 11, _dw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(BXR - _dw / 2, LEG_Y + 2, _dn, 9, lc.C_MUTE, 'middle', maxw=_dw - 4, tag='chip:dn')
lc.text(MX, 648, '读图：左相册的「册数×卡数」随长度线性涨——GC 要逐本翻检；右长卷无论多长只有 6 条卷 + 区间索引，'
        '读某位置时按区间现造 dict。', 9.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='readline')
lc.text(MX, 680, 'FlatLogprobs 六列表 + append / append_fast verbatim vllm/logprobs.py:L30-L135 · '
        'gc.get_objects() 建造前后差 host 实测（L=100 / k=2 与 L=1000 探针）· 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch08-fig-flat-vs-nested.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
