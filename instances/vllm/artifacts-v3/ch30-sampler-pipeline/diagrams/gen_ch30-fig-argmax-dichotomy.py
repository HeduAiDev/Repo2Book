#!/usr/bin/env python3
"""ch29 机制图 2 · argmax 不变性二分（figure_spec ch29-fig-argmax-dichotomy，模板 flow）

放大自 L0 采样出口列（L2 章图 center 拍片 ⑤『非 argmax 不变列』与 ⑦『sample』两拍 +
south『why·argmax 不变性二分』块）：同一批 logits 在 step5 与 step7c 两个站点分叉生效。
不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：构造期按 is_argmax_invariant() 把处理器分两列——非不变列（MinTokens/LogitBias，
能改第一名）在 step5、greedy 前对全体生效；不变列（MinP，只砍尾不改第一名）在 step7c、
温度后仅随机路径执行。声明是语义承诺不是优化提示，谎报静默破坏 greedy。

数字全部取自本章 explainer 素材（m02 trace：classification / min_tokens_flips_argmax /
min_p_cuts_tail_keeps_argmax / lying_declaration / logit_bias_flips_argmax）与 pin 源码锚点；
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 1000
MX = 60
BXR = 1440
GRID = '#e2e8f0'

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>'
    f'<marker id="mut" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_MUTE}"/></marker>'
    f'<marker id="red" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ABORT}"/></marker>')

lc.text(MX, 34, 'argmax 不变性二分：能改「第一名」的排在 greedy 前，只砍尾的排在温度后',
        16.5, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '构造期按声明 is_argmax_invariant() 分两列——只信申报、无人强校验；声明是语义承诺不是优化提示，'
               '谎报会静默绕过封禁（右下反例）',
        10, lc.C_MUTE, 'start', maxw=1240, tag='subtitle')
_ch = '放大自 L0 采样出口列 · L2 章图拍片 ⑤/⑦ 与「argmax 不变性二分」块'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 左上：构造期分类容器 =================
CT_Y, CT_H = 92, 74
lc.rect(MX, CT_Y, 600, CT_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 14, CT_Y + 21, 'LogitsProcessors（构造期分类 · state.py:L148-L160）', 11.5, lc.C_TXT,
        'start', True, maxw=570, tag='ct:t')
lc.text(MX + 14, CT_Y + 41, '构造时逐个问 processor.is_argmax_invariant()，按声明 append 进两个 list——非 if-else 判值',
        8.3, '#334155', 'start', maxw=570, tag='ct:l1')
lc.text(MX + 14, CT_Y + 58, '声明是抽象方法、无人强校验（interface.py:L87-L94）——正确性靠各处理器自带测试，不靠类型系统',
        8.3, lc.C_MUTE, 'start', maxw=570, tag='ct:l2')

# ================= 左：两列（非不变列在上、不变列在下） =================
PX, PW = 100, 300
NON_Y, NON_H = 196, 184
IX, IY, IH = PX, 396, 224


def decl_chip(x, y, w, name, flag, note, lie=False):
    stroke = lc.C_ABORT if lie else lc.C_SAM_S
    dash = lie
    h = 62 if lie else 48
    lc.rect(x, y, w, h, '#ffffff', stroke, rx=6, sw=1.2, dash=dash)
    lc.text(x + 8, y + 18, name, 8.6, lc.C_TXT if not lie else lc.C_ABORT, 'start', True,
            maxw=w - 60, tag='dc' + name[:6])
    bw = 20 + 8 * len(flag)
    lc.rect(x + w - bw - 8, y + 6, bw, 17, '#fef2f2' if lie else lc.C_SAM_F, stroke, rx=8, sw=1.0)
    lc.text(x + w - bw / 2 - 8, y + 18.5, flag, 8, stroke, 'middle', True, maxw=bw - 4,
            tag='dcb' + name[:4])
    if lie:
        lc.text(x + 8, y + 34, note[0], 7.8, '#334155', 'start', maxw=w - 18, tag='dcn1' + name[:4])
        lc.text(x + 8, y + 49, note[1], 7.8, '#334155', 'start', maxw=w - 18, tag='dcn2' + name[:4])
    else:
        lc.text(x + 8, y + 36, note, 7.8, '#334155', 'start', maxw=w - 18, tag='dcn' + name[:4])


# ---- 非不变列（上）----
lc.rect(PX, NON_Y, PW, NON_H, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.6)
lc.text(PX + 12, NON_Y + 20, 'non_argmax_invariant 列', 10.5, lc.C_SAM_S, 'start', True,
        maxw=PW - 24, tag='non:t')
lc.text(PX + 12, NON_Y + 36, '声明 False · 能改「谁是第一名」', 8.3, lc.C_MUTE, 'start',
        maxw=PW - 24, tag='non:s')
decl_chip(PX + 12, NON_Y + 46, PW - 24, 'MinTokensLogitsProcessor', 'False', '封 stop/EOS：能改 argmax')
decl_chip(PX + 12, NON_Y + 100, PW - 24, 'LogitBiasLogitsProcessor', 'False', '稀疏坐标 +=：能改 argmax')
lc.text(PX + 12, NON_Y + NON_H - 10, '非空才在 step5 逐个 apply', 8.3, lc.C_MUTE, 'start',
        maxw=PW - 24, tag='non:f')

# ---- 不变列（下）----
lc.rect(IX, IY, PW, IH, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.6)
lc.text(IX + 12, IY + 20, 'argmax_invariant 列', 10.5, lc.C_SAM_S, 'start', True,
        maxw=PW - 24, tag='inv:t')
lc.text(IX + 12, IY + 36, '声明 True · 只砍尾部、砍不到第一名', 8.3, lc.C_MUTE, 'start',
        maxw=PW - 24, tag='inv:s')
decl_chip(IX + 12, IY + 46, PW - 24, 'MinPLogitsProcessor', 'True', '阈值=min_p×max_prob，冠军恒过线')
decl_chip(IX + 12, IY + 100, PW - 24, 'MinTokens 谎报 True（D 行）', 'True',
          ['分类容器照单全收、进本列——', '封禁从此挂在随机路径'], lie=True)
lc.text(IX + 12, IY + IH - 10, '只在 step7c 随机路径逐个 apply', 8.3, lc.C_MUTE, 'start',
        maxw=PW - 24, tag='inv:f')

# 分类容器 → 两列 的 fork（左缘竖直通道 x=80 绕到下列）
lc.seg(250, CT_Y + CT_H, 250, NON_Y, lc.C_MUTE, 1.6, 'mut')
lc.parrow([(80, CT_Y + CT_H), (80, IY + 50), (IX, IY + 50)], lc.C_MUTE, 1.6, 'mut')

# ================= 右：9 步主轴（竖直 spine，站号与 L2 章图一致） =================
SX, SW = 720, 290
COMP_H, BIG_H, GAP = 34, 84, 10
y = 92
spine = []  # (y, h)


def spine_comp(sym, name, badge):
    global y
    lc.rect(SX, y, SW, COMP_H, '#ffffff', lc.C_SAM_S, rx=6, sw=1.2)
    lc.text(SX + 12, y + 21, sym + '  ' + name, 9.2, lc.C_TXT, 'start', True, maxw=SW - 76,
            tag='sc' + sym)
    bw = 34
    lc.rect(SX + SW - bw - 8, y + 5, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
    lc.text(SX + SW - bw / 2 - 8, y + 17.5, badge, 8, lc.C_ENG_S, 'middle', True, maxw=bw - 4,
            tag='scb' + sym)
    spine.append((y, COMP_H))
    y += COMP_H + GAP


for sym, name, badge in [('①', 'raw 留底', '站3'), ('②', 'fp32', '站4'),
                         ('③', 'allowed 白名单', '站5'), ('④', 'bad_words', '站6')]:
    spine_comp(sym, name, badge)

# ⑤ 大框（step5 挂点·高亮）
F5_Y = y
lc.rect(SX, F5_Y, SW, BIG_H, lc.C_SAM_F, lc.C_SAM_S, rx=7, sw=2.0)
lc.text(SX + 12, F5_Y + 20, '⑤ 非 argmax 不变列 · step5', 10, lc.C_TXT, 'start', True,
        maxw=SW - 66, tag='f5:t')
bw = 34
lc.rect(SX + SW - bw - 8, F5_Y + 6, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
lc.text(SX + SW - bw / 2 - 8, F5_Y + 18.5, '站7', 8, lc.C_ENG_S, 'middle', True, maxw=bw - 4,
        tag='f5:b')
lc.text(SX + 12, F5_Y + 40, 'greedy 判定前 · 对全体生效（sampler.py:L399-L400）', 8, '#334155',
        'start', maxw=SW - 24, tag='f5:l1')
lc.text(SX + 12, F5_Y + 57, 'MinTokens 封 stop/EOS · LogitBias 稀疏坐标 +=', 8, '#334155',
        'start', maxw=SW - 24, tag='f5:l2')
lc.text(SX + 12, F5_Y + 73, '——它定义了 greedy 的输入', 8, lc.C_SAM_S, 'start', True,
        maxw=SW - 24, tag='f5:l3')
spine.append((F5_Y, BIG_H))
y += BIG_H + GAP

spine_comp('⑥', '惩罚+思考预算', '站8')

# ⑦ 大框：7a-7f 子步
F7_Y, F7_H = y, 216
lc.rect(SX, F7_Y, SW, F7_H, '#ffffff', lc.C_SAM_S, rx=7, sw=2.0)
lc.text(SX + 12, F7_Y + 20, '⑦ sample', 10.5, lc.C_TXT, 'start', True, maxw=SW - 150, tag='f7:t')
lc.text(SX + SW - 12, F7_Y + 20, 'L243-L302', 8, lc.C_FAINT, 'end', tag='f7:w')
bw = 34
lc.rect(SX + SW - bw - 8 - 80, F7_Y + 5, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
lc.text(SX + SW - bw / 2 - 8 - 80, F7_Y + 17.5, '站9', 8, lc.C_ENG_S, 'middle', True,
        maxw=bw - 4, tag='f7:b')
SUBS = [
    ('7a', 'greedy_sample=argmax · all_greedy 在此早退', True),
    ('7b', 'apply_temperature（temp<eps 替 1.0 防除零）', False),
    ('7c', '不变列 apply：MinP 挂点（L282-L283）', True),
    ('7d', 'top-k / top-p 截断', False),
    ('7e', '掷骰（Gumbel / FlashInfer 拒绝采样）', False),
    ('7f', 'torch.where 按逐请求 temp<eps 合并两路', False),
]
sub_ys = {}
for i, (tag7, txt, hot) in enumerate(SUBS):
    yy = F7_Y + 32 + i * 30
    sub_ys[tag7] = yy
    fill = lc.C_ENG_F if hot else '#f8fafc'
    stroke = lc.C_ENG_S if hot else GRID
    lc.rect(SX + 14, yy, SW - 28, 26, fill, stroke, rx=5, sw=1.1 if hot else 0.9)
    lc.text(SX + 24, yy + 17, tag7 + '  ' + txt, 8, lc.C_BEAT_T if hot else '#334155', 'start',
            True, maxw=SW - 50, tag='sub' + tag7)
spine.append((F7_Y, F7_H))
y += F7_H + GAP

E8_Y = y
spine_comp('⑧', 'gather logprobs', '站11')
E9_Y = y - COMP_H - GAP
spine_comp('⑨', 'SamplerOutput 出件', '站12')

# spine 相邻纵向连线（全部从框底到框顶，不穿框）
for (y0, h0), (y1, _) in zip(spine, spine[1:]):
    lc.seg(SX + SW / 2, y0 + h0, SX + SW / 2, y1, lc.C_SAM_S, 1.8, 'sam')

# ================= 两列 → 主轴挂点 =================
# 非不变列（上）→ ⑤：水平直连
lc.parrow([(PX + PW, NON_Y + 92), (SX, F5_Y + 30)], lc.C_SAM_S, 2.0, 'sam')
lc.text((PX + PW + SX) / 2, NON_Y + 84, 'step5 · greedy 判定前 · 对全体生效', 8.8, lc.C_SAM_S,
        'middle', True, maxw=330, tag='a:non')
# 不变列（下）→ 7c：水平 + 微降
lc.parrow([(IX + PW, IY + 104), (705, IY + 104), (705, sub_ys['7c'] + 13), (SX + 14, sub_ys['7c'] + 13)],
          lc.C_SAM_S, 2.0, 'sam')
lc.text((IX + PW + 705) / 2, IY + 96, 'step7c · 温度后 · 仅随机路径', 8.8, lc.C_SAM_S, 'middle',
        True, maxw=280, tag='a:inv')

# ================= all_greedy 早退旁路（7a → ⑧） =================
BP_X = 1076
lc.parrow([(SX + SW, sub_ys['7a'] + 13), (BP_X, sub_ys['7a'] + 13), (BP_X, E8_Y + 17),
           (SX + SW, E8_Y + 17)], lc.C_ENG_S, 1.8, 'up')
lc.text(BP_X + 10, sub_ys['7a'] + 6, 'all_greedy 早退（L261-L271）：', 8.6, lc.C_ENG_S, 'start',
        True, maxw=350, tag='bp1')
lc.text(BP_X + 10, sub_ys['7a'] + 22, '7a argmax 即返回，7b-7e 全部跳过', 8.6, lc.C_ENG_S,
        'start', maxw=350, tag='bp2')
# 7c 挂点旁：谎报被早退绕过的叉标注
lc.text(SX + SW + 8, sub_ys['7c'] + 15, '×', 13, lc.C_ABORT, 'start', True, tag='x7c')
lc.text(BP_X + 10, sub_ys['7c'] + 17, '谎报时：早退在 step7c 之前返回——封禁从未执行', 8.4,
        lc.C_ABORT, 'start', True, maxw=350, tag='x7c:t')

# ================= 谎报反例终点（右下） =================
END_X, END_Y, END_W, END_H = 1090, 700, 350, 84
lc.rect(END_X, END_Y, END_W, END_H, '#fef2f2', lc.C_ABORT, rx=8, sw=1.8)
lc.text(END_X + 14, END_Y + 20, '采样出 2（EOS）——× 正确答案 1（B 行）', 10, lc.C_ABORT,
        'start', True, maxw=END_W - 28, tag='end:t')
lc.text(END_X + 14, END_Y + 40, '声明 True → 进不变列 → all_greedy 批的封禁从未执行', 8.2,
        '#334155', 'start', maxw=END_W - 28, tag='end:l1')
lc.text(END_X + 14, END_Y + 58, 'is_argmax_invariant() 是写给系统的语义承诺，写错没有人报错', 8.2,
        lc.C_MUTE, 'start', maxw=END_W - 28, tag='end:l2')
lc.parrow([(250, IY + 162), (250, END_Y + 42), (END_X, END_Y + 42)], lc.C_ABORT, 1.6, 'red',
          dash=True)
lc.text(258, IY + 174, '谎报路径', 8.4, lc.C_ABORT, 'start', True, maxw=90, tag='lie:path')

# ================= 底部：B/C/E/D 实证卡 =================
EYC, EHC, EWC = 812, 128, 330
CARDS = [
    ('B · MinTokens 封 EOS（step5）',
     ['logits=[1.0, 2.0, 6.0, 0.0]', 'token2: 6.0 → -inf', 'argmax 2 → 1 · greedy 采样出 1（第二名）']),
    ('C · MinP 砍尾（step7c）',
     ['probs=[0.6308, 0.2321, 0.0854, 0.0518]', '阈值=0.3×0.6308=0.1892', 'survivors=[0,1] · argmax 前后都=0（不变）']),
    ('E · LogitBias 稀疏 +=（step5）',
     ['logits=[1.0, 2.0, 3.0, 2.5]', 'token3: 2.5+10.0=12.5', 'argmax 2 → 3 · 非不变列，step5 生效']),
    ('D · 谎报反例',
     ['MinTokens 声明 True → 分进不变列', 'all_greedy 早退在 step7c 前返回', '采样出 2（EOS）——正确答案 1']),
]
for i, (h, lines) in enumerate(CARDS):
    x = MX + i * (EWC + 12)
    dash = i == 3
    lc.rect(x, EYC, EWC, EHC, '#fef2f2' if dash else '#ffffff', lc.C_ABORT if dash else lc.C_MUTE,
            rx=8, sw=1.1, dash=dash)
    lc.text(x + 12, EYC + 19, h, 9.2, lc.C_ABORT if dash else lc.C_TXT, 'start', True,
            maxw=EWC - 24, tag='cd' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x + 12, EYC + 40 + j * 16.5, ln, 7.8, '#334155', 'start', maxw=EWC - 22,
                tag='cdl' + str(i) + str(j))

# ================= 页脚锚点 =================
lc.text(MX, 968, 'vllm/v1/sample/logits_processor/state.py:L148-L160（分类）· interface.py:L87-L94（声明）· sampler.py:L399-L400（step5）/ L282-L283（step7c）/ L261-L271（all_greedy 早退）· builtin.py:L47-L49 · L130-L133 · L183-L186（三条声明的 docstring 论证）',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft1')
lc.text(MX, 988, '数值＝本章驱动脚本实跑（4 元素玩具词表）· 行号基线 vLLM v0.27.1 · 机制图：不另立架构画法，主轴站号与 L2 章图一致',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft2')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-argmax-dichotomy.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
