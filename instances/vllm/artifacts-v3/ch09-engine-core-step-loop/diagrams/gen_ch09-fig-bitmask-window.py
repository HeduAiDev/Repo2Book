#!/usr/bin/env python3
"""ch09 机制图 2 · 语法位掩码的拍内窗口（figure_spec ch09-fig-bitmask-window，模板 before-after）

放大自 L0 循环框（loop_box）内的 ③ 拍窗口——即本章 L2 章图 center 拍片 ③ get_grammar_bitmask
的时序放大：回答「掩码为什么必须卡在 ② 之后 ④ 之前、它对采样行做了什么」。与本章另一张
时间轴图（同一拍的全景）互为缩放关系，非新架构画法，架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
图右上角指北小签。

claim：同一条 logits 行 [0,3,0,0,7,9,0,0]，无掩码贪婪选 5（9.0 最高）；③ 在 ② 发起之后、
④ 采样之前把允许集外的 token 置 -inf（本例允许 {1,4}）→ 改选 4——掩码是唯一变量
（对照实测选 5），且非末块 prefill 请求被 ③ 整体排除（探针零调用）。

数字全部取自 figure_spec.numbers（logits 行 0,3,0,0,7,9,0,0；掩码字 0b00010010 → 允许集 {1,4}；
主场景三拍恒 [4] → LENGTH(3/3)、对照 [5]；窗口时序 execute@0.05 < bitmask@5.806 < apply@5.855
< greedy@6.245；探针预算 2<prompt 3 零调用；③→应用 0.049ms ≪ 前向 5.756ms）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 918
MX, BXR = 60, 1440
SLATE = '#e2e8f0'    # 网格灰（slate 家族，与 ch03 机制图同款）
BAR_F, BAR_S = '#cbd5e1', lc.C_MUTE       # 允许集内柱（中性）
FADE_F, FADE_S = '#f1f5f9', '#cbd5e1'     # 允许集外柱（淡出）
BAN_F = '#fee2e2'                          # favorite 被禁柱（红系淡底）

LOGITS = [0.0, 3.0, 0.0, 0.0, 7.0, 9.0, 0.0, 0.0]
ALLOWED = [1, 4]

# ---------------- 标题区 ----------------
lc.text(MX, 34, '语法位掩码的拍内窗口——为什么可以不等前向算完、采样却非等掩码不可',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '同一条 logits 行 [0,3,0,0,7,9,0,0]（vocab=8）：无掩码贪婪选 5@9.0；'
        '③ 在 ② 发起之后、④ 采样之前把允许集 {1,4} 之外置 -inf → 恒选 4（对照实测选 5——掩码是唯一变量）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ③ get_grammar_bitmask · L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 柱状几何（上下两半共用） ----------------
AXIS_X = 306
SCALE = 120.0 / 9.0     # 9.0 分 → 120px
SLOT = 92
CX0 = 340
BAR_W = 58
GRID_V = [9.0, 7.0, 3.0]


def bars(baseline, upper):
    """画 8 根柱 + 网格 + y 轴刻度 + token 轴。返回各柱 (x0, x1, top, cx)。"""
    geo = []
    for v in GRID_V:
        y = baseline - v * SCALE
        lc.seg(AXIS_X, y, 1006, y, SLATE, 1.0, dash=True)
        lc.text(AXIS_X - 6, y + 3, f'{v:.1f}', 8, lc.C_MUTE, 'end', tag='gy' + str(v))
    lc.seg(AXIS_X, baseline - 138, AXIS_X, baseline + 4, lc.C_MUTE, 1.2)
    lc.text(AXIS_X - 6, baseline + 3, '0', 8, lc.C_MUTE, 'end', tag='gy0')
    for i, v in enumerate(LOGITS):
        cx = CX0 + i * SLOT
        x0, x1 = cx - BAR_W / 2, cx + BAR_W / 2
        top = baseline - max(v * SCALE, 3.0)
        geo.append((x0, x1, top, cx))
        allowed = i in ALLOWED
        if upper:
            fill, stroke, sw = BAR_F, BAR_S, 1.0
        elif allowed:
            if i == 4:
                fill, stroke, sw = lc.C_BEAT_F, lc.C_BEAT_S, 2.0
            else:
                fill, stroke, sw = BAR_F, BAR_S, 1.0
        elif i == 5:
            fill, stroke, sw = BAN_F, lc.C_ABORT, 1.4
        else:
            fill, stroke, sw = FADE_F, FADE_S, 1.0
        lc.rect(x0, top, BAR_W, baseline - top, fill, stroke, rx=2, sw=sw)
        lc.text(cx, baseline + 18, str(i), 9, lc.C_MUTE, 'middle', tag='tk' + str(i))
    return geo


# ---------------- 上半：应用掩码前 ----------------
UP_Y0, UP_Y1 = 118, 318
lc.rect(MX, UP_Y0, 1000, UP_Y1 - UP_Y0, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(76, UP_Y0 + 22, '应用掩码前——贪婪 argmax 盯上 favorite 5（9.0）', 10.5, lc.C_TXT,
        'start', True, maxw=620, tag='up:title')
lc.text(1044, UP_Y0 + 22, 'logits（vocab=8）', 8, lc.C_FAINT, 'end', tag='up:ax')
up_base = 280
up = bars(up_base, upper=True)
VALS = ['0.0', '3.0', '0.0', '0.0', '7.0', '9.0', '0.0', '0.0']
for i, (x0, x1, top, cx) in enumerate(up):
    if LOGITS[i] >= 3.0:
        lab = '9.0 · favorite' if i == 5 else VALS[i]
        lc.text(cx, top + 14, lab, 8.5, lc.C_TXT, 'middle', maxw=86, tag='uv' + str(i))
    else:
        lc.text(cx, top - 6, VALS[i], 8.5, lc.C_MUTE, 'middle', tag='uv' + str(i))
lc.text(76, up_base + 18, 'token id（0-7）', 8, lc.C_FAINT, 'start', tag='up:tok')
# argmax 指针：chip → favorite 5 柱顶
b5 = up[5]
lc.rect(b5[3] - 52, UP_Y0 + 8, 104, 20, lc.C_SAM_F, lc.C_SAM_S, rx=9, sw=1.3)
lc.text(b5[3], UP_Y0 + 22, '贪婪 argmax → 5', 9, lc.C_SAM_S, 'middle', True, maxw=98, tag='am5')
lc.seg(b5[3], UP_Y0 + 28, b5[3], b5[2] - 3, lc.C_MUTE, 1.5, 'std')

# 上半结果卡（品红=采样出口）
lc.rect(1090, UP_Y0, 350, UP_Y1 - UP_Y0, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.4)
lc.text(1106, UP_Y0 + 24, '无掩码 → 采样 [5]', 10.5, lc.C_SAM_S, 'start', True, maxw=320, tag='ur:t')
for j, ln in enumerate([
        '· logits 行：[0.0, 3.0, 0.0, 0.0, 7.0, 9.0, 0.0, 0.0]',
        '  （5@9.0 最高、4@7.0 次高、1@3.0）',
        '· 贪婪 argmax 直取最大 logit：token 5（9.0）',
        '· 对照实测（同行、无结构化标记）采样 [5]']):
    lc.text(1106, UP_Y0 + 46 + j * 17, ln, 8.5, '#334155', 'start', maxw=326, tag='ur:l' + str(j))

# ---------------- 中带：掩码变换（左） + 窗口时序条（右） ----------------
LO_Y0 = 482
lc.seg(170, UP_Y1 + 2, 170, LO_Y0 - 4, lc.C_ABORT, 2.0, 'ab')
lc.text(190, 356, 'apply_grammar_bitmask（位清零 → -inf）', 9, lc.C_TXT, 'start', True,
        maxw=400, tag='tr:t')
lc.text(190, 374, '掩码字 0b00010010（bit 1、bit 4 置位）→ 允许集 {1,4}', 8.5, '#334155',
        'start', maxw=390, tag='tr:l1')
lc.text(190, 390, '允许集外的位被清零 → 采样前该 logit 置 -inf（xgrammar 内核契约）', 8.5,
        '#334155', 'start', maxw=400, tag='tr:l2')

# 时序条
ST_X0, ST_Y0, ST_W, ST_H = 590, 348, 850, 114
lc.rect(ST_X0, ST_Y0, ST_W, ST_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(ST_X0 + 16, ST_Y0 + 20, '③ 的窗口时序——主场景拍 1 实测（刻度 ms，间距按实测比例）',
        9.5, lc.C_TXT, 'start', True, maxw=520, tag='st:t')
lc.text(ST_X0 + 16, ST_Y0 + 36, '③ 只依赖 ① 的批——这拍谁在考、各排几 token；非末块 prefill 直接排除（见左下探针）',
        8.5, lc.C_MUTE, 'start', maxw=560, tag='st:note')
PX0, PX1 = 640.0, 1400.0
RNG = 6.245
PPM = (PX1 - PX0) / RNG


def sx(ms):
    return PX0 + ms * PPM


T_EXE, T_BIT, T_APP, T_GDY = 0.05, 5.806, 5.855, 6.245
LINE_Y = ST_Y0 + 68
lc.seg(630, LINE_Y, 1410, LINE_Y, lc.C_MUTE, 1.2)
lc.rect(sx(T_EXE), LINE_Y - 10, sx(T_BIT) - sx(T_EXE), 20, lc.C_GPU_F, lc.C_GPU_S, rx=2, sw=1.0)
lc.text((sx(T_EXE) + sx(T_BIT)) / 2, LINE_Y + 4, '② 发起的前向窗口 5.756ms（建模的一步前向）',
        8.5, lc.C_GPU_S, 'middle', maxw=560, tag='st:fwd')
for t in (T_EXE, T_BIT, T_APP, T_GDY):
    lc.seg(sx(t), LINE_Y - 16, sx(t), LINE_Y + 16, lc.C_MUTE, 1.3)
lc.text(sx(T_BIT) - 6, LINE_Y - 18, '③ 5.806 → 应用 5.855（间隔 0.049ms）', 8.5, lc.C_TXT,
        'end', maxw=300, tag='st:b3')
lc.text(sx(T_GDY) - 2, LINE_Y - 18, '④ 6.245', 8.5, lc.C_TXT, 'start', maxw=60, tag='st:b4')
lc.text(sx(T_EXE) + 6, LINE_Y + 24, '② 0.05', 8.5, lc.C_MUTE, 'start', maxw=60, tag='st:b2')
lc.text(sx(T_EXE) + 60, LINE_Y + 24,
        '（② execute_model → ③ get_grammar_bitmask → 应用 apply_grammar_bitmask → ④ greedy_sample）',
        7.5, lc.C_FAINT, 'start', maxw=420, tag='st:cap')
lc.text(ST_X0 + ST_W - 16, LINE_Y + 24, '应用必须赶在 ④ 的 argmax 之前——晚一步就采出非法 token',
        8.5, lc.C_ENG_S, 'end', maxw=360, tag='st:right')

# ---------------- 下半：应用掩码后 ----------------
LO_Y1 = 700
lc.rect(MX, LO_Y0, 1000, LO_Y1 - LO_Y0, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(76, LO_Y0 + 22, '应用掩码后——允许集 {1,4} 之外全部置 -inf → argmax 改选 4', 10.5,
        lc.C_TXT, 'start', True, maxw=620, tag='lo:title')
lc.text(1044, LO_Y0 + 22, 'logits（vocab=8）', 8, lc.C_FAINT, 'end', tag='lo:ax')
lo_base = 654
lo = bars(lo_base, upper=False)
for i, (x0, x1, top, cx) in enumerate(lo):
    if i in ALLOWED:
        if i == 4:
            lc.text(cx, top + 14, '7.0', 9, lc.C_BEAT_T, 'middle', True, maxw=60, tag='lv4')
        else:
            lc.text(cx, top + 14, '3.0', 8.5, lc.C_TXT, 'middle', maxw=60, tag='lv1')
    elif i == 5:
        lc.text(cx, top - 6, '-inf（favorite 被禁）', 8.5, lc.C_ABORT, 'middle', maxw=120,
                tag='lv5')
        mx_, my_ = cx, (top + lo_base) / 2
        lc.seg(mx_ - 12, my_ - 12, mx_ + 12, my_ + 12, lc.C_ABORT, 2.4)
        lc.seg(mx_ - 12, my_ + 12, mx_ + 12, my_ - 12, lc.C_ABORT, 2.4)
    else:
        lc.text(cx, top - 6, '-inf', 8, lc.C_ABORT, 'middle', maxw=40, tag='lv' + str(i))
lc.text(76, lo_base + 18, 'token id（0-7）', 8, lc.C_FAINT, 'start', tag='lo:tok')
# argmax 指针：chip → 胜出柱 4
b4 = lo[4]
lc.rect(b4[3] - 52, LO_Y0 + 8, 104, 20, lc.C_BEAT_F, lc.C_BEAT_S, rx=9, sw=1.3)
lc.text(b4[3], LO_Y0 + 22, 'argmax → 4', 9, lc.C_BEAT_T, 'middle', True, maxw=98, tag='am4')
lc.seg(b4[3], LO_Y0 + 28, b4[3], b4[2] - 4, lc.C_MUTE, 1.5, 'std')
# 允许集花括号（柱 1 到柱 4）
b1c, b4c = lo[1][3], lo[4][3]
BRC_Y = lo_base + 28
lc.seg(b1c, BRC_Y, b4c, BRC_Y, lc.C_MUTE, 1.2)
lc.seg(b1c, BRC_Y, b1c, BRC_Y - 4, lc.C_MUTE, 1.2)
lc.seg(b4c, BRC_Y, b4c, BRC_Y - 4, lc.C_MUTE, 1.2)
lc.text((b1c + b4c) / 2, BRC_Y + 14, '允许集 {1,4}', 9, lc.C_TXT, 'middle', True, maxw=120,
        tag='lo:allow')

# 下半结果卡（拍片橙=拍内机制产物）
lc.rect(1090, LO_Y0, 350, LO_Y1 - LO_Y0, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.4)
lc.text(1106, LO_Y0 + 24, '有掩码 → 恒选 [4]', 10.5, lc.C_BEAT_T, 'start', True, maxw=320, tag='lr:t')
for j, ln in enumerate([
        '· 主场景三拍采样 [4]、[4]、[4] → LENGTH(3/3)',
        '· 掩码是唯一变量（对照：同行、无标记 → [5]）',
        '· 9.0 的 favorite 让位 7.0 的 4——',
        '  用『分数』换『合法』：正确性压过 argmax',
        '· 非末块 prefill 请求被 ③ 整体排除（见左下探针）']):
    lc.text(1106, LO_Y0 + 46 + j * 17, ln, 8.5, '#334155', 'start', maxw=326, tag='lr:l' + str(j))

# ---------------- 底部两框：探针 + 窗口收益 ----------------
BT_Y, BT_H = 720, 112
lc.rect(MX, BT_Y, 680, BT_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(76, BT_Y + 22, '排除探针——非末块 prefill 整体排除（③ 对它零调用）', 9.5, lc.C_TXT,
        'start', True, maxw=640, tag='pb:t')
for j, ln in enumerate([
        '· 调度预算 2 token ＜ prompt 3 token → 首拍 is_prefill_chunk=True（2/3 未完）',
        '· ③ 完全跳过（掩码管理器零调用）→ 无采样行、无输出',
        '· 部分前向不出活：跨拍收官的 chunk 深水属第 10 章（预告）']):
    lc.text(76, BT_Y + 44 + j * 17, ln, 8.5, '#334155', 'start', maxw=648, tag='pb:l' + str(j))
lc.rect(770, BT_Y, 670, BT_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(786, BT_Y + 22, '窗口收益 = min(③ 掩码耗时, 前向耗时)', 9.5, lc.C_TXT, 'start', True,
        maxw=630, tag='wb:t')
for j, ln in enumerate([
        '· ③ 算出 → 应用仅 0.049ms ≪ 前向窗口 5.756ms——真实引擎整段藏得进窗口',
        '· 若不藏进窗口、串行排在采样前：每拍白加一整段掩码时长',
        '· 真实引擎前向几十 ms、位掩码每拍按批算一遍——窗口更富余',
        '· 这张位掩码从哪来、怎么算 → 第 30 章回收（预告）']):
    lc.text(786, BT_Y + 44 + j * 17, ln, 8.5, '#334155', 'start', maxw=638, tag='wb:l' + str(j))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 856
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 12, BAR_F, BAR_S, rx=3, sw=1.0)
lc.text(lx + 26, LEG_Y + 1, '允许集内的 logit 柱', 8.5, lc.C_TXT, 'start', maxw=200, tag='lg1')
lx += 26 + lc.tw('允许集内的 logit 柱', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.4)
lc.text(lx + 26, LEG_Y + 1, 'argmax 胜出', 8.5, lc.C_TXT, 'start', maxw=140, tag='lg2')
lx += 26 + lc.tw('argmax 胜出', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 12, BAN_F, lc.C_ABORT, rx=3, sw=1.2)
lc.seg(lx + 6, LEG_Y - 6, lx + 14, LEG_Y + 1, lc.C_ABORT, 1.6)
lc.seg(lx + 6, LEG_Y + 1, lx + 14, LEG_Y - 6, lc.C_ABORT, 1.6)
lc.text(lx + 26, LEG_Y + 1, '被禁 → -inf', 8.5, lc.C_TXT, 'start', maxw=140, tag='lg3')
lx += 26 + lc.tw('被禁 → -inf', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 12, FADE_F, FADE_S, rx=3, sw=1.0)
lc.text(lx + 26, LEG_Y + 1, '允许集外（同样 -inf）', 8.5, lc.C_TXT, 'start', maxw=200, tag='lg4')
lx += 26 + lc.tw('允许集外（同样 -inf）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.0)
lc.text(lx + 26, LEG_Y + 1, 'GPU 前向窗口', 8.5, lc.C_TXT, 'start', maxw=140, tag='lg5')

lc.text(MX, 884, '逐字锚 vllm/v1/engine/core.py:L596-L604（②→③→④ 固定调用顺序）· '
        'vllm/v1/core/sched/scheduler.py:L1646-L1668（get_grammar_bitmask 与 is_prefill_chunk 排除）· '
        'vllm/v1/structured_output/utils.py:L86-L175（apply_grammar_bitmask）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 900, 'logits 行 / 掩码字 / 采样取自配套精简版 host 实测（temperature=0 贪婪 argmax，'
        'vocab=8）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch09-fig-bitmask-window.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
