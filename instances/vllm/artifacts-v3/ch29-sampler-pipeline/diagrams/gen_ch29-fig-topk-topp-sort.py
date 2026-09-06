#!/usr/bin/env python3
"""ch29 机制图 3 · top-k/top-p 截断的 pytorch sort 路径（figure_spec ch29-fig-topk-topp-sort，模板 tensor-flow）

放大自 L0 采样出口列（L2 章图 center 拍片 ⑦d『Apply top_k and/or top_p』的算法内部
= apply_top_k_top_p_pytorch，批<8 或无 Triton 时每个 decode 步在 GPU 上跑的四步）。
纯算法机制图、不画架构元素。

claim：pytorch sort 截断路径四步走同一根升序轴——升序 sort → top-k 取第 (V-k) 位值做
阈值（严格 < 保并列）→ top-p 用升序 softmax cumsum≤1-p 反向 mask（最末位恒保
at least one）→ scatter 回原位。

数字全部取自本章 explainer 素材（m09 trace：topk_tie_kept / topp_cumsum /
topp_at_least_one / topk_topp_combined）与 pin 源码锚点；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 852
MX = 60
BXR = 1440
GRID = '#e2e8f0'
MASKF = '#e2e8f0'

lc.text(MX, 34, 'top-k 与 top-p 两刀落在同一根升序轴上——最末位恒保，argmax 永不受截断影响',
        16.5, lc.C_TXT, 'start', True, maxw=1100, tag='title')
lc.text(MX, 58, 'apply_top_k_top_p_pytorch 四步（批<8 或无 Triton 时的路径；批≥8 走 Triton pivot 核）：'
               '升序 sort → top-k 取第 (V-k) 位值做阈值（严格 < 保并列）→ top-p 用升序 cumsum≤1-p 反向 mask（at least one）→ scatter 回原位',
        9.6, lc.C_MUTE, 'start', maxw=1300, tag='subtitle')
_ch = '放大自 L0 采样出口列 · L2 章图拍片 ⑦d 的算法内部'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 四个变换站（列头） =================
ST = [(60, 350, '① 升序 sort —— 最小位在左', 'logits.sort(dim=-1, descending=False) · L388'),
      (430, 330, '② top-k 阈值 = 第 (V-k) 位值', 'size(1)-k → gather → 严格 < → masked_fill_(-inf) · L390-L396'),
      (780, 340, '③ top-p = 升序 cumsum ≤ 1-p', 'softmax → cumsum → <=1-p；top_p_mask[:, -1]=False · L398-L405'),
      (1140, 300, '④ scatter 回原位', 'logits.scatter_(dim=-1, index=logits_idx, src=logits_sort) · L408')]
for i, (x, w, t, code) in enumerate(ST):
    lc.rect(x, 100, w, 46, '#f8fafc', GRID, rx=7, sw=1.1)
    lc.text(x + 12, 118, t, 9.6, lc.C_TXT, 'start', True, maxw=w - 20, tag='st' + str(i))
    lc.text(x + 12, 137, code, 7, lc.C_MUTE, 'start', maxw=w - 20, tag='stc' + str(i))
    if i < 3:
        lc.seg(x + w, 123, ST[i + 1][0], 123, lc.C_MUTE, 1.8, 'std')


def cell(x, y, w, h, val, tok, masked=False, hot=False):
    fill = MASKF if masked else (lc.C_BEAT_F if hot else '#ffffff')
    stroke = lc.C_BEAT_S if hot else lc.C_SAM_S
    lc.rect(x, y, w, h, fill, stroke, rx=4, sw=1.1)
    lc.text(x + w / 2, y + 16, '-inf' if masked else val, 8.8 if not masked else 8.2,
            lc.C_MUTE if masked else lc.C_TXT, 'middle', True, maxw=w - 4, tag='c' + val + tok)
    lc.text(x + w / 2, y + 30, tok, 7, lc.C_MUTE if masked else '#334155', 'middle',
            maxw=w - 4, tag='ct' + tok)


def cells_row(x0, y, cw, ch_, gap, vals, toks, masked=None, hot=None):
    masked = masked or []
    hot = hot or []
    for i, (v, t) in enumerate(zip(vals, toks)):
        cell(x0 + i * (cw + gap), y, cw, ch_, v, t, masked=(i in masked), hot=(i in hot))


def skip_box(x, y, w, h, l1, l2):
    lc.rect(x, y, w, h, '#f8fafc', GRID, rx=7, sw=1.0, dash=True)
    lc.text(x + w / 2, y + 24, l1, 9.2, lc.C_MUTE, 'middle', True, maxw=w - 20, tag='sk1')
    lc.text(x + w / 2, y + 43, l2, 7.6, lc.C_MUTE, 'middle', maxw=w - 20, tag='sk2')


# ================= 行 A · top-k 并列例（k=2, V=5） =================
RA_Y, RA_H = 158, 196
lc.rect(MX, RA_Y, 1380, RA_H, '#ffffff', GRID, rx=9, sw=1.1)
lc.text(MX + 14, RA_Y + 19, '行 A · top-k 并列例（k=2, V=5）—— logits=[3.0, 2.0, 0.5, 2.0, 1.0]',
        9.4, lc.C_TXT, 'start', True, maxw=700, tag='ra:lab')

# 站①：原序 → 升序
a_x0, a_y0, a_cw, a_ch, a_gap = 76, RA_Y + 36, 56, 40, 6
cells_row(a_x0, a_y0, a_cw, a_ch, a_gap, ['3.0', '2.0', '0.5', '2.0', '1.0'],
          ['t0', 't1', 't2', 't3', 't4'])
lc.seg(a_x0 + 5 * (a_cw + a_gap) / 2 - a_gap / 2, a_y0 + a_ch, a_x0 + 5 * (a_cw + a_gap) / 2 - a_gap / 2,
       a_y0 + a_ch + 10, lc.C_MUTE, 1.4, 'std')
cells_row(a_x0, a_y0 + 60, a_cw, a_ch, a_gap, ['0.5', '1.0', '2.0', '2.0', '3.0'],
          ['t2', 't4', 't1', 't3', 't0'])
lc.text(a_x0, RA_Y + 148, 'idx=[2, 4, 1, 3, 0]——scatter 时按它放回原位', 7.8, lc.C_MUTE, 'start',
        maxw=330, tag='ra:idx')

# 站②：阈值 + 严格 < 遮罩
b_x0, b_y0, b_cw = 446, RA_Y + 36, 54
cells_row(b_x0, b_y0, b_cw, a_ch, 6, ['0.5', '1.0', '2.0', '2.0', '3.0'],
          ['t2', 't4', 't1', 't3', 't0'], hot=[2, 3])
thr_x = b_x0 + 3 * (b_cw + 6) - 3
lc.seg(thr_x, b_y0 - 8, thr_x, b_y0 + a_ch + 4, lc.C_TXT, 1.4, dash=True)
lc.text(b_x0, b_y0 + a_ch + 18, '阈值 = logits_sort[V-k=3] = 2.0', 7.8, lc.C_TXT, 'start', True,
        maxw=310, tag='ra:thr')
cells_row(b_x0, b_y0 + 66, b_cw, a_ch, 6, ['-inf', '-inf', '2.0', '2.0', '3.0'],
          ['t2', 't4', 't1', 't3', 't0'], masked=[0, 1], hot=[2, 3])
lc.text(b_x0, b_y0 + a_ch + 84, '严格 <：mask=[T,T,F,F,F]——并列 2.0 的 t1/t3 都不 < 2.0 → 全留', 7.6,
        '#334155', 'start', maxw=316, tag='ra:m1')
lc.text(b_x0, b_y0 + a_ch + 99, 'k=2 存活 3 个（无并列时恰好 k 个）· survivors=[0, 1, 3]', 7.6,
        '#334155', 'start', maxw=316, tag='ra:m2')

# 站③：跳过
skip_box(796, RA_Y + 42, 308, 60, '本例 p=None → 本站跳过',
         '（k 单独时 sort 只为取第 k 名阈值，不做 softmax/cumsum）')

# 站④：scatter 回原位
d_x0, d_y0, d_cw = 1152, RA_Y + 52, 52
cells_row(d_x0, d_y0, d_cw, a_ch, 5, ['3.0', '2.0', '-inf', '2.0', '-inf'],
          ['t0', 't1', 't2', 't3', 't4'], masked=[2, 4])
lc.text(d_x0, d_y0 + a_ch + 16, 'survivors=[0, 1, 3]——scatter 纯位置重排、不改值', 7.6,
        '#334155', 'start', maxw=280, tag='ra:d1')
lc.text(d_x0, d_y0 + a_ch + 31, '至少留 1 个且必含最大位 → argmax 不受影响', 7.6, lc.C_SAM_S,
        'start', True, maxw=280, tag='ra:d2')
# 行 A：② 结果绕过 ③ 直达 ④
lc.parrow([(760, b_y0 + 66 + a_ch), (760, RA_Y + RA_H - 16), (1128, RA_Y + RA_H - 16),
           (1128, d_y0 + 20), (1152, d_y0 + 20)], lc.C_MUTE, 1.5, 'std', dash=True)
lc.text(944, RA_Y + RA_H - 22, 'p=None：② 的结果直接进 ④', 7.4, lc.C_MUTE, 'middle',
        maxw=220, tag='ra:skip')

# ================= 行 B · top-p（p=0.9）与 p=0 角案 =================
RB_Y, RB_H = 372, 330
lc.rect(MX, RB_Y, 1380, RB_H, '#ffffff', GRID, rx=9, sw=1.1)
lc.text(MX + 14, RB_Y + 19, '行 B · top-p 例 —— logits=[3.0, 2.0, 1.0, 0.5, 0.1]（p=0.9 主例 + p=0 角案，同一根升序轴）',
        9.4, lc.C_TXT, 'start', True, maxw=760, tag='rb:lab')

# 站①：原序 → 升序 → softmax
cells_row(a_x0, RB_Y + 36, a_cw, a_ch, a_gap, ['3.0', '2.0', '1.0', '0.5', '0.1'],
          ['t0', 't1', 't2', 't3', 't4'])
cells_row(a_x0, RB_Y + 96, a_cw, a_ch, a_gap, ['0.1', '0.5', '1.0', '2.0', '3.0'],
          ['t4', 't3', 't2', 't1', 't0'])
for i, p in enumerate(['0.0335', '0.05', '0.0825', '0.2243', '0.6096']):
    lc.text(a_x0 + i * (a_cw + a_gap) + a_cw / 2, RB_Y + 156, p, 7.4, lc.C_SAM_S, 'middle',
            True, maxw=a_cw, tag='rb:p' + str(i))
lc.text(a_x0, RB_Y + 176, 'softmax 于升序轴 → probs_sort=[0.0335, 0.05, 0.0825, 0.2243, 0.6096]',
        7.6, lc.C_MUTE, 'start', maxw=340, tag='rb:sm')

# 站②：跳过
skip_box(446, RB_Y + 42, 298, 60, '本例 k=None → 本站跳过',
         '（p 单独时同走这根升序轴做 cumsum）')

# 站③：cumsum 柱 + 两条阈值线
BASE_Y, SCALE, BAR_W, BAR_GAP, BAR_X0 = 640, 90, 40, 8, 795
CUM = [('0.0335', 0.0335), ('0.0836', 0.0836), ('0.1661', 0.1661), ('0.3904', 0.3904), ('1.0', 1.0)]
for i, (cs, c) in enumerate(CUM):
    x = BAR_X0 + i * (BAR_W + BAR_GAP)
    hgt = max(c * SCALE, 2.5)
    lc.rect(x, BASE_Y - hgt, BAR_W, hgt, lc.C_MUTE, lc.C_MUTE, rx=2, sw=0)
    lc.text(x + BAR_W / 2, BASE_Y + 13, cs, 7.2, '#334155', 'middle', maxw=BAR_W,
            tag='cum' + str(i))
    lc.text(x + BAR_W / 2, BASE_Y + 25, ['t4', 't3', 't2', 't1', 't0'][i], 7, lc.C_MUTE,
            'middle', maxw=BAR_W, tag='cumt' + str(i))
lc.seg(BAR_X0 - 6, BASE_Y, BAR_X0 + 5 * BAR_W + 4 * BAR_GAP + 6, BASE_Y, '#334155', 1.6)
# 阈值线 1-p=0.1（p=0.9）
y01 = BASE_Y - 0.1 * SCALE
lc.seg(BAR_X0 - 6, y01, BAR_X0 + 5 * BAR_W + 4 * BAR_GAP + 6, y01, lc.C_TXT, 1.3, dash=True)
lc.text(BAR_X0, y01 - 5, '1-p = 0.1（p=0.9）', 7.6, lc.C_TXT, 'start', True, maxw=200, tag='th1')
# 阈值线 1-p=1.0（p=0 角案）
y10 = BASE_Y - 1.0 * SCALE
lc.seg(BAR_X0 - 6, y10, BAR_X0 + 5 * BAR_W + 4 * BAR_GAP + 6, y10, lc.C_ABORT, 1.3, dash=True)
lc.text(BAR_X0, y10 - 5, '1-p = 1.0（p=0 角案）', 7.6, lc.C_ABORT, 'start', True,
        maxw=140, tag='th2')
lc.text(BAR_X0 - 6, RB_Y + 114, 'p=0.9：cumsum≤0.1 的前两位 mask → 核={t0,t1,t2}', 7.6,
        '#334155', 'start', maxw=336, tag='rb:m1')
lc.text(BAR_X0 - 6, RB_Y + 129, '核质量 0.6096+0.2243+0.0825=0.9164 ≥ 0.9（刚过 p）', 7.6,
        '#334155', 'start', maxw=336, tag='rb:m1b')
lc.text(BAR_X0 - 6, RB_Y + 144, 'p=0 角案：cumsum≤1.0 全命中 → 只剩强保的最末位 → survivors=[0]', 7.6,
        '#334155', 'start', maxw=336, tag='rb:m2')

# 站④：两态结果
d2_x0, d2_cw = 1152, 52
lc.text(d2_x0, RB_Y + 46, 'p=0.9 · survivors=[0, 1, 2]', 7.8, lc.C_TXT, 'start', True,
        maxw=280, tag='rb:d1l')
cells_row(d2_x0, RB_Y + 54, d2_cw, a_ch, 5, ['3.0', '2.0', '1.0', '-inf', '-inf'],
          ['t0', 't1', 't2', 't3', 't4'], masked=[3, 4])
lc.text(d2_x0, RB_Y + 128, 'p=0 角案 · survivors=[0]', 7.8, lc.C_TXT, 'start', True,
        maxw=280, tag='rb:d2l')
cells_row(d2_x0, RB_Y + 136, d2_cw, a_ch, 5, ['3.0', '-inf', '-inf', '-inf', '-inf'],
          ['t0', 't1', 't2', 't3', 't4'], masked=[1, 2, 3, 4])
lc.text(d2_x0, RB_Y + 210, '最末位（最大位）被 top_p_mask[:, -1]=False', 7.6, '#334155', 'start',
        maxw=280, tag='rb:d1')
lc.text(d2_x0, RB_Y + 225, '强制保留——「at least one」，argmax 恒不受影响', 7.6, lc.C_SAM_S,
        'start', True, maxw=280, tag='rb:d2')
# 行 B 流：站① probs → 站③ 柱 → 站④
lc.seg(410, RB_Y + 150, 780, RB_Y + 150, lc.C_MUTE, 1.5, 'std', dash=True)
lc.seg(1120, RB_Y + 150, 1140, RB_Y + 150, lc.C_MUTE, 1.5, 'std', dash=True)

# ================= 底部：k+p 联合 + 图例 + 页脚 =================
BT_Y = 720
lc.rect(MX, BT_Y, 1380, 54, '#f8fafc', GRID, rx=8, sw=1.1)
lc.text(MX + 14, BT_Y + 21, 'k+p 联合（k=3, p=0.9）：先 k=3 砍（幸存 [0,1,2]）→ softmax 在幸存者上重归一化 0.6096/0.9164=0.6652 → p=0.9 再砍到 {0,1}',
        8.6, lc.C_TXT, 'start', True, maxw=1340, tag='bt1')
lc.text(MX + 14, BT_Y + 40, '（重归一化后 p=0.9 的核只剩 {0,1}）——两刀都落在同一根升序轴上，先 k 后 p', 8.2,
        lc.C_MUTE, 'start', maxw=1340, tag='bt2')

LY = 806
lx0 = MX
for fill, stroke, name in [('#ffffff', lc.C_SAM_S, '存活位'), (MASKF, lc.C_SAM_S, '-inf（被砍）'),
                           (lc.C_BEAT_F, lc.C_BEAT_S, '与第 k 名并列（仍保留）')]:
    lc.rect(lx0, LY - 9, 16, 11, fill, stroke, rx=3, sw=1.4)
    lc.text(lx0 + 21, LY + 1, name, 9, lc.C_TXT, 'start')
    lx0 += 21 + lc.tw(name, 9) + 18
lc.rect(lx0, LY - 9, 16, 11, lc.C_MUTE, lc.C_MUTE, rx=2, sw=0)
lc.text(lx0 + 21, LY + 1, '升序 cumsum 柱', 9, lc.C_TXT, 'start')
lx0 += 21 + lc.tw('升序 cumsum 柱', 9) + 18
lc.seg(lx0 + 4, LY - 3, lx0 + 30, LY - 3, lc.C_TXT, 1.3, dash=True)
lc.text(lx0 + 36, LY + 1, '阈值线 1-p', 9, lc.C_TXT, 'start')
lx0 += 36 + lc.tw('阈值线 1-p', 9) + 18
lc.text(lx0, LY + 1, 'tN = token 下标（V=5 玩具词表）', 9, lc.C_MUTE, 'start',
        maxw=BXR - lx0, tag='lg:tail')

lc.text(MX, 834, 'vllm/v1/sample/ops/topk_topp_sampler.py:L367-L408 · apply_top_k_top_p_pytorch（sort L388 / top-k L390-L396 / top-p L398-L405 / scatter L408）· 数值＝本章驱动脚本实跑 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft1')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-topk-topp-sort.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
