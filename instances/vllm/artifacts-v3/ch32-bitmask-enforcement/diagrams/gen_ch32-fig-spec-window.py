#!/usr/bin/env python3
"""ch32 机制图 · spec 窗口的逐行走表（figure_spec ch32-fig-spec-window，模板 state-table）

放大自 L0 采样列·结构化输出组『批装配』段串行分支的 spec 窗口展开（L2 站 5-7 的
算法放大——本章算法主体的逐行走表），架构归属回指 L2 章图，不另立第二种架构画法
（FIGURE-SYSTEM §3）。

claim：spec 窗口的每行都是『先试探接受前序草稿、再取当前状态的合法集』：a/b/c 窗口
4 行允许集逐行演化（a 系→b 系→精确 c→EOS），填完 rollback 复位——真推进只留给
update_from_output。

数字全部取自 figure_spec.numbers（行 0..3 允许集 a 系 [64,397,39305] → b 系
[65,15630] → [66] → [50256]；accepts=[[64],[65],[66]]、rollbacks=[3]、复检回 a 系；
哨兵窗口 [65,15630]/整行 -1/bonus [65,15630]、accepts=[[64]]、rollbacks=[1]；
AssertionError 原文；diffusion mask_rows 2<3）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 1000
MX, BXR = 60, 1440
VOCAB = 50256   # 词表末位 token id（= EOS）

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'spec 窗口逐行走表：先试探接受前序草稿、再取当前状态合法集——填完 rollback 复位', 16.5,
        lc.C_TXT, 'start', True, maxw=1150, tag='title')
lc.text(MX, 58, 'root ::= "a" "b" "c"（真 xgrammar matcher · gpt2 词表 50257）· 窗口 = 3 草稿位 + 1 bonus 位 · '
               '词表里有 ab/abc/bc 这种多字符 token——FSM 按 token 接受，行 2 只收精确 c（c 开头的多字 token 会越过串尾）',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列·批装配串行分支 · L2 站 5-7 的算法放大（本章算法主体）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 主走表（左） =================
TX, TY = MX, 108
COLS = [('行', 44), ('草稿位', 128), ('apply_bitmask', 96), ('advance_grammar', 128),
        ('允许集实测（词表 0 … 50256 位条，亮的位=允许）', 380), ('试探后 FSM', 104)]
HDR_H, ROW_H = 40, 92
ROWS = [
    ('行 0', 'a（草稿 1）', 'True', 'True（试探接受 a）', [64, 397, 39305], ["'a'", "'ab'", "'abc'"], '位置 1'),
    ('行 1', 'b（草稿 2）', 'True', 'True（试探接受 b）', [65, 15630], ["'b'", "'bc'"], '位置 2'),
    ('行 2', 'c（草稿 3）', 'True', 'True（试探接受 c）', [66], ["'c'（精确）"], '位置 3（串完整）'),
    ('行 3', 'bonus（采样位，无草稿）', 'True', '—（bonus 行不推进）', [50256], ['EOS'], '位置 3（不动）'),
]
TW = sum(w_ for _, w_ in COLS)
TH = HDR_H + len(ROWS) * ROW_H
lc.rect(TX, TY, TW, TH, '#ffffff', lc.C_SAM_S, rx=9, sw=1.8)

# 表头
cx = TX
for name, w_ in COLS:
    lc.text(cx + w_ / 2, TY + 17, name, 8.6, lc.C_SAM_S, 'middle', True, maxw=w_ - 8,
            tag='th:' + name[:6])
    cx += w_
lc.seg(TX, TY + HDR_H - 4, TX + TW, TY + HDR_H - 4, lc.C_SAM_S, 1.2)

# 列竖线
cx = TX
for _, w_ in COLS[:-1]:
    cx += w_
    lc.seg(cx, TY + HDR_H - 4, cx, TY + TH - 4, '#e2e8f0', 1.0)

# 数据行
STRIP_X0 = None
for i, (r_, draft, apply_, adv, ids, names, st) in enumerate(ROWS):
    ry = TY + HDR_H + i * ROW_H
    if i:
        lc.seg(TX, ry, TX + TW, ry, '#e2e8f0', 1.0)
    cy = ry + ROW_H / 2
    lc.text(TX + COLS[0][1] / 2, cy, r_, 9.6, lc.C_TXT, 'middle', True, maxw=40, tag='r' + str(i))
    lc.text(TX + COLS[0][1] + 10, cy - 8, draft.split('（')[0], 9.4, lc.C_TXT, 'start', True,
            maxw=COLS[1][1] - 16, tag='d' + str(i))
    if '（' in draft:
        lc.text(TX + COLS[0][1] + 10, cy + 10, '（' + draft.split('（')[1], 7.8, lc.C_MUTE,
                'start', maxw=COLS[1][1] - 16, tag='d2:' + str(i))
    # apply / advance 布尔徽标
    bx = TX + COLS[0][1] + COLS[1][1]
    for j, (val, colw) in enumerate([(apply_, COLS[2][1]), (adv, COLS[3][1])]):
        if val == 'True':
            lc.rect(bx + 10, cy - 11, 44, 22, lc.C_SAM_F, lc.C_SAM_S, rx=10, sw=1.2)
            lc.text(bx + 32, cy + 3.5, 'True', 8.6, lc.C_SAM_S, 'middle', True, maxw=40,
                    tag='b' + str(i) + str(j))
        elif val.startswith('—'):
            lc.text(bx + colw / 2, cy + 3.5, val, 8, lc.C_MUTE, 'middle', maxw=colw - 10,
                    tag='b' + str(i) + str(j))
        else:
            lc.rect(bx + 8, cy - 11, 76, 22, '#f8fafc', lc.C_MUTE, rx=10, sw=1.0)
            lc.text(bx + 46, cy + 3.5, val, 7.4, lc.C_MUTE, 'middle', maxw=72,
                    tag='b' + str(i) + str(j))
        bx += colw
    # 允许集位条
    sx0 = TX + COLS[0][1] + COLS[1][1] + COLS[2][1] + COLS[3][1] + 14
    sw = COLS[4][1] - 28
    sy = ry + 34
    lc.rect(sx0, sy, sw, 10, '#f1f5f9', lc.C_MUTE, rx=2, sw=0.8)
    for k, tid in enumerate(ids):
        px = sx0 + tid / VOCAB * sw
        lc.ELEMS.append(((px - 2, sy - 2, px + 2, sy + 12),
                         f'<rect x="{px - 1.6:.1f}" y="{sy - 1.5:.1f}" width="3.2" height="13" '
                         f'fill="{lc.C_SAM_S}" stroke="none"/>'))
        up = (k % 2 == 0)
        ly = sy - 6 if up else sy + 26
        lc.seg(px, sy + (0 if up else 10), px, ly + (3 if up else -9), lc.C_MUTE, 0.9, dash=True)
        lc.text(px, ly, str(tid), 7.6, lc.C_SAM_S, 'middle', True, maxw=60, tag='id' + str(tid))
    # token 名 + 允许计数
    lc.text(sx0, ry + ROW_H - 12, f"{len(ids)} 个 / 50257：{' · '.join(names)}", 7.8, '#334155',
            'start', maxw=sw, tag='cnt' + str(i))
    # 状态
    stx = TX + TW - COLS[5][1] / 2
    lc.text(stx, cy, st.split('（')[0], 9, lc.C_TXT, 'middle', True, maxw=COLS[5][1] - 10,
            tag='st' + str(i))
    if '（' in st:
        lc.text(stx, cy + 16, '（' + st.split('（')[1], 7.4, lc.C_MUTE, 'middle',
                maxw=COLS[5][1] - 8, tag='st2:' + str(i))

# ================= 底部：FSM 链 + rollback 弧 + 试探账 =================
FY0 = TY + TH + 46
NODE_R = 26
NODE_Y = FY0 + 60
NODE_X = [TX + 120, TX + 380, TX + 640, TX + 880]
NODE_LB = ['位置 0', '位置 1', '位置 2', '位置 3']
NODE_SUB = ['起点（串未开吃）', '已吃 a', '已吃 a b', '串完整（a b c）']
for k in range(4):
    lc.circle(NODE_X[k], NODE_Y, NODE_R, lc.C_SAM_S, 1.8, dash=False)
    lc.text(NODE_X[k], NODE_Y - 3, NODE_LB[k], 9, lc.C_TXT, 'middle', True, maxw=56,
            tag='fsm' + str(k))
    lc.text(NODE_X[k], NODE_Y + 12, NODE_SUB[k], 7, lc.C_MUTE, 'middle', maxw=60,
            tag='fsms' + str(k))
EDGE_LB = ['accept_tokens([64]) 试探', 'accept_tokens([65]) 试探', 'accept_tokens([66]) 试探']
for k in range(3):
    lc.seg(NODE_X[k] + NODE_R, NODE_Y, NODE_X[k + 1] - NODE_R, NODE_Y, lc.C_SAM_S, 1.8, 'sam')
    lc.text((NODE_X[k] + NODE_X[k + 1]) / 2, NODE_Y - 32, EDGE_LB[k], 7.8, lc.C_SAM_S, 'middle',
            True, maxw=170, tag='fe' + str(k))
# rollback 弧（状态 3 → 起点）
rb_y = NODE_Y + NODE_R + 40
lc.parrow([(NODE_X[3], NODE_Y + NODE_R), (NODE_X[3], rb_y), (NODE_X[0], rb_y),
           (NODE_X[0], NODE_Y + NODE_R)], lc.C_ABORT, 1.8, 'ab', dash=True)
lc.text((NODE_X[0] + NODE_X[3]) / 2, rb_y + 16,
        '窗口收尾 rollback(3)：3 次试探全部撤销——FSM 精确回到本步开始状态', 8.6, lc.C_ABORT,
        'middle', True, maxw=560, tag='rb:t')
lc.text((NODE_X[0] + NODE_X[3]) / 2, rb_y + 32,
        '复检：再填一行回到 a 系 [64, 397, 39305]（复位是行为实测，不是断言）', 8, '#334155',
        'middle', maxw=560, tag='rb:s')
# 试探账小框（右下，靠近 rollback 弧终点）
LG_X, LG_Y = TX + TW - 330, rb_y + 58
lc.rect(LG_X, LG_Y, 330, 60, '#ffffff', lc.C_MUTE, rx=7, sw=1.1)
lc.text(LG_X + 12, LG_Y + 16, '试探账（实测账本）', 8.6, lc.C_TXT, 'start', True, maxw=300,
        tag='lg:t')
lc.text(LG_X + 12, LG_Y + 33, 'accepts = [[64], [65], [66]] · rollbacks = [3]', 8, '#334155',
        'start', maxw=310, tag='lg:l1')
lc.text(LG_X + 12, LG_Y + 49, '真推进只发生在下一步 update_from_output（accept_tokens）', 7.6,
        lc.C_MUTE, 'start', maxw=310, tag='lg:l2')
lc.text(TX, LG_Y + 16, '真走棋只在赛后：填表用的推进全是试探（棋谱推演），', 8.4, lc.C_MUTE,
        'start', maxw=300, tag='led:n1')
lc.text(TX, LG_Y + 32, '试探与真走严格分离——窗口末尾整体回退恰相同步数', 8.4, lc.C_MUTE,
        'start', maxw=300, tag='led:n2')

# ================= 右栏：哨兵窗口 + 断言 + diffusion =================
RX, RW = 1064, BXR - 1064
# ---- 哨兵窗口小表 ----
SNY, SNH = 108, 262
lc.rect(RX, SNY, RW, SNH, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(RX + 14, SNY + 20, '对照 · -1 哨兵窗口（草稿 [a, -1, c]）', 9.6, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='sn:t')
lc.text(RX + 14, SNY + 36, '-1 = 草稿被 validate 过滤后的补齐位，三层语义见各行', 7.6, lc.C_MUTE,
        'start', maxw=RW - 28, tag='sn:s')
SNROWS = [
    ('行 0 · a', '受约束，试探接受 a', '[64, 397, 39305]'),
    ('行 1 · -1', '旧标志填 + 当行即停', '[65, 15630]（a 之后）'),
    ('行 2 · c', 'apply 已翻 False：整行 -1 放行', '全允许（50257）'),
    ('bonus', '再受约束（should_fill=True）', '[65, 15630]（未再推进）'),
]
for j, (a, b, c) in enumerate(SNROWS):
    yy = SNY + 54 + j * 30
    lc.rect(RX + 10, yy - 12, RW - 20, 26, '#f8fafc' if j != 1 else lc.C_BADGE_F,
            lc.C_MUTE, rx=5, sw=0.9)
    lc.text(RX + 16, yy + 3, a, 8, lc.C_TXT, 'start', True, maxw=70, tag='snr' + str(j))
    lc.text(RX + 92, yy + 3, b, 7.6, '#334155', 'start', maxw=190, tag='snb' + str(j))
    lc.text(RX + RW - 12, yy + 3, c, 7.6, lc.C_SAM_S, 'end', maxw=140, tag='snc' + str(j))
lc.text(RX + 14, SNY + SNH - 22, '三层语义：旧标志填（行 1）· 当行即停 · 次行起放行（行 2）', 7.8,
        lc.C_MUTE, 'start', maxw=RW - 28, tag='sn:f1')
lc.text(RX + 14, SNY + SNH - 8, 'accepts = [[64]] · rollbacks = [1]（只试探 1 次、回退 1 次）', 7.8,
        '#334155', 'start', maxw=RW - 28, tag='sn:f2')

# ---- 断言框（红） ----
ASY = SNY + SNH + 14
ASH = 92
lc.rect(RX, ASY, RW, ASH, '#fef2f2', lc.C_ABORT, rx=8, sw=1.3)
lc.text(RX + 14, ASY + 20, '非法草稿当场断言（防御语义）', 9.2, lc.C_ABORT, 'start', True,
        maxw=RW - 28, tag='as:t')
lc.text(RX + 14, ASY + 38, "绕过 validate 直塞 999 → AssertionError((999, 'r1',", 8, lc.C_ABORT,
        'start', maxw=RW - 28, tag='as:l1')
lc.text(RX + 14, ASY + 52, "{'r1': [999]}))", 8, lc.C_ABORT, 'start', maxw=RW - 28, tag='as:l2')
lc.text(RX + 14, ASY + 72, '守护『进窗口的草稿必已过滤』——调度与语法状态失配是 bug 不是运行态', 7.6,
        '#334155', 'start', maxw=RW - 26, tag='as:l3')

# ---- diffusion 对照 ----
DFY = ASY + ASH + 14
DFH = 64
lc.rect(RX, DFY, RW, DFH, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(RX + 14, DFY + 20, 'diffusion 对照：有草稿位时无 bonus 行', 9.2, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='df:t')
lc.text(RX + 14, DFY + 40, '画布步不采 AR bonus token → mask_rows 2 行 < 常规 3 行', 7.8,
        '#334155', 'start', maxw=RW - 26, tag='df:l1')

# ---- 读法小面板 ----
RDY = DFY + DFH + 14
RDH = 168
lc.rect(RX, RDY, RW, RDH, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(RX + 14, RDY + 20, '读法', 9.6, lc.C_TXT, 'start', True, maxw=RW - 28, tag='rd:t')
for j, ln in enumerate(['· 左表逐行自上而下 = 串 a b c 的前缀推进：',
                        '  行 0 允许 a 开头的一切（含多字 a 系），',
                        '  行 2 只收精确 c——ab/abc 之类多字 token 会越过串尾',
                        '· 位条：整条词表轴上亮的位 = 该行允许集（a 系 3 位）',
                        '· rollback 弧 = 窗口收尾整体回退；真推进在赛后',
                        '· 右上：-1 哨兵行的三层语义（v0.27 新二分的产物）']):
    lc.text(RX + 14, RDY + 40 + j * 20, ln, 7.8, '#334155', 'start', maxw=RW - 26,
            tag='rd:l' + str(j))

# ================= 图例 + 页脚 =================
LY = 956
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.2)
lc.text(lx0 + 21, LY + 1, 'True（本行约束/推进）', 8.8, lc.C_TXT, 'start', maxw=190, tag='lg1')
lx0 += 21 + lc.tw('True（本行约束/推进）', 8.8) + 16
lc.rect(lx0, LY - 11, 22, 15, '#f1f5f9', lc.C_MUTE, rx=2, sw=0.8)
lc.ELEMS.append(((lx0 + 9, LY - 10, lx0 + 12, LY + 3),
                 f'<rect x="{lx0 + 9}" y="{LY - 10}" width="3" height="13" fill="{lc.C_SAM_S}"/>'))
lc.text(lx0 + 27, LY + 1, '位条亮位 = 允许 token（id 标在引线上）', 8.8, lc.C_TXT, 'start',
        maxw=280, tag='lg2')
lx0 += 27 + lc.tw('位条亮位 = 允许 token（id 标在引线上）', 8.8) + 16
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_ABORT, 1.8, 'ab', dash=True)
lc.text(lx0 + 40, LY + 1, 'rollback（试探回退）', 8.8, lc.C_TXT, 'start', maxw=170, tag='lg3')
lx0 += 40 + lc.tw('rollback（试探回退）', 8.8) + 16
lc.circle(lx0 + 8, LY - 3, 8, lc.C_SAM_S, 1.2, dash=False)
lc.text(lx0 + 24, LY + 1, 'FSM 可观测位置', 8.8, lc.C_TXT, 'start', maxw=140, tag='lg4')

lc.text(MX, 982, 'vllm/v1/structured_output/__init__.py:L272-L332（串行填行·spec 窗口：apply_bitmask 与 advance_grammar 二分、accept_tokens 试探、AssertionError）'
                 '· L333-L350（收尾 rollback）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 998 - 8, '四行允许集 / accepts·rollbacks 账本 / 哨兵三层语义 / 断言原文 / diffusion mask_rows 2<3 ＝ 本章驱动脚本实测'
                 '（真 xgrammar 0.2.6，gpt2 词表 50257）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-spec-window.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
