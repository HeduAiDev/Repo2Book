#!/usr/bin/env python3
"""ch32 机制图 · worker 侧重排的行算术走表（figure_spec ch32-fig-reorder-walk，模板 state-table）

放大自 L0 采样列·结构化输出组 worker 侧『重排』段的行算术展开（L2 站 12 的算法放大
——logit_index 前缀和走表），架构归属回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：worker 逐自己批序累计 spec 偏移算 logit_index：B(0+0)=0 占 4 行、C(1+3)=4 不占行、
A(2+3)=5——前缀和区间两两不相交，重排每行恰写一次。

数字全部取自 figure_spec.numbers（批序走表 B[bi0,off0,spec3,li0] / C[bi1,off3,spec0,li4] /
A[bi2,off3,spec0,li5]；out_indices=[5,0,1,2,3]（ids 序 A 先追加）；5<6 → indices 必传；
C 行 full_allow=true 保持初始 -1；全覆盖对照 5==5 → indices=None）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 860
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '重排的行算术：logit_index = batch_index + cumulative_offset——草稿像占座外套', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'worker 批序 [B, C, A]（B 带 3 草稿、C 非语法、A 无草稿）；调度序掩码 ids=[A, B] 共 5 行——'
               'B 的 3 个草稿把后面所有人顶后 3 位，A 从批序直觉位 1 落到 logit_index 5',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列·worker 侧重排 · L2 站 12 的算法放大（前缀和走表）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 走表（左） =================
TX, TY = MX, 108
COLS = [('批序位', 64), ('请求', 120), ('batch_index', 92), ('offset_before', 100),
        ('num_spec', 78), ('logit_index', 88), ('掩码行展开（按 ids 序 [A, B] 追加）', 238)]
HDR_H, ROW_H = 40, 64
ROWS = [
    ('0', 'B（带 3 草稿）', '0', '0', '3', '0', 'B 的 4 行（3 spec + bonus）→ logits 行 0..3'),
    ('1', 'C（非语法）', '1', '3', '0', '4', '无行：sorted 行 4 保持初始 -1（全允许，实测未被动过）'),
    ('2', 'A（无草稿）', '2', '3', '0', '5', 'A 的 1 行 → logits 行 5（批序直觉位 1 被顶到 5）'),
]
WALK_W = sum(w_ for _, w_ in COLS)
WALK_H = HDR_H + len(ROWS) * ROW_H + 80
lc.rect(TX, TY, WALK_W, WALK_H, '#ffffff', lc.C_SAM_S, rx=9, sw=1.8)
cx = TX
for name, w_ in COLS:
    lc.text(cx + w_ / 2, TY + 17, name, 8.4, lc.C_SAM_S, 'middle', True, maxw=w_ - 6,
            tag='th:' + name[:6])
    cx += w_
lc.seg(TX, TY + HDR_H - 4, TX + WALK_W, TY + HDR_H - 4, lc.C_SAM_S, 1.2)
cx = TX
for _, w_ in COLS[:-1]:
    cx += w_
    lc.seg(cx, TY + HDR_H - 4, cx, TY + WALK_H - 4, '#e2e8f0', 1.0)

for i, row in enumerate(ROWS):
    ry = TY + HDR_H + i * ROW_H
    if i:
        lc.seg(TX, ry, TX + WALK_W, ry, '#e2e8f0', 1.0)
    cy = ry + ROW_H / 2
    cx = TX
    for j, (val, (_, w_)) in enumerate(zip(row, COLS)):
        if j in (0, 2, 3, 4):
            lc.text(cx + w_ / 2, cy + 3, val, 10 if j != 0 else 9.4, lc.C_TXT, 'middle',
                    True, maxw=w_ - 8, tag='c' + str(i) + str(j))
        elif j == 5:
            hot = val in ('0', '5')
            lc.rect(cx + (w_ - 46) / 2, cy - 13, 46, 26, lc.C_BADGE_F if hot else '#f8fafc',
                    lc.C_ENG_S if hot else lc.C_MUTE, rx=8, sw=1.1)
            lc.text(cx + w_ / 2, cy + 4, val, 10.5, lc.C_ENG_S if hot else lc.C_MUTE, 'middle',
                    True, maxw=40, tag='li' + str(i))
        else:
            lc.text(cx + 10, cy - 6, val.split('（')[0], 9.4, lc.C_TXT, 'start', True,
                    maxw=w_ - 14, tag='c' + str(i) + str(j))
            if '（' in val:
                lc.text(cx + 10, cy + 12, '（' + val.split('（')[1], 7.6, lc.C_MUTE, 'start',
                        maxw=w_ - 14, tag='c2:' + str(i) + str(j))
        cx += w_

# 收拢行（公式 + out_indices）
RY = TY + HDR_H + len(ROWS) * ROW_H
lc.seg(TX, RY, TX + WALK_W, RY, lc.C_SAM_S, 1.2)
lc.text(TX + 12, RY + 22, '收拢：按 ids 序 [A, B] 追加 → out_indices =', 9.2, lc.C_TXT, 'start',
        True, maxw=330, tag='cl:t')
OX = TX + 360
for j, v_ in enumerate(['5', '0', '1', '2', '3']):
    lc.rect(OX + j * 34, RY + 8, 28, 24, lc.C_SAM_F, lc.C_SAM_S, rx=4, sw=1.1)
    lc.text(OX + j * 34 + 14, RY + 24, v_, 9.4, lc.C_SAM_S, 'middle', True, maxw=24,
            tag='oi' + str(j))
lc.text(OX + 5 * 34 + 10, RY + 25, 'int32 上卡', 8, '#334155', 'start', maxw=90, tag='cl:s')
lc.text(TX + 12, RY + 52, '掩码 5 行 < logits 6 行 → indices 必传（行数对满才免传，差一行都得传）',
        8.2, '#334155', 'start', maxw=WALK_W - 24, tag='cl:f1')
lc.text(TX + 12, RY + 68, 'offset 只加不减（前缀和单调不减）→ 每请求行区间 [li, li+num_spec] 两两不相交、无竞写',
        8.2, lc.C_MUTE, 'start', maxw=WALK_W - 24, tag='cl:f2')

# ================= 右：sorted_bitmask 落位条 =================
BX, BY = 1080, 130
BW, BHH = 220, 40
lc.text(BX + BW / 2, BY - 14, 'sorted_bitmask / logits 落位（6 行）', 9.8, lc.C_GPU_S, 'middle',
        True, maxw=BW + 60, tag='bm:t')
DST = [(0, 'B', 1), (1, 'B', 2), (2, 'B', 3), (3, 'B', 4), (4, 'C', None), (5, 'A', 0)]
for k, who, from_m in DST:
    y = BY + k * (BHH + 8)
    f_, s_ = ('#ffffff', lc.C_GPU_S) if who != 'C' else ('#f1f5f9', lc.C_MUTE)
    lc.rect(BX, y, BW, BHH, f_, s_, rx=6, sw=1.3)
    lc.text(BX + 14, y + BHH / 2 + 3, f'logits 行 {k}', 9, lc.C_TXT, 'start', True, maxw=90,
            tag='bm' + str(k))
    if who == 'C':
        lc.text(BX + BW - 14, y + BHH / 2 + 3, 'C · -1 全允许', 8, lc.C_MUTE, 'end', maxw=110,
                tag='bmc')
    else:
        lc.text(BX + BW - 14, y + BHH / 2 + 3, f'{who} · 允许 token {from_m * 3}', 8,
                lc.C_GPU_S, 'end', maxw=140, tag='bmr' + str(k))
# 走表行 → 落位条的连线
SRC_Y = [TY + HDR_H + i * ROW_H + ROW_H / 2 for i in range(3)]
for i, tgt in enumerate([0, 4, 5]):
    lc.parrow([(TX + WALK_W, SRC_Y[i]), (TX + WALK_W + 26, SRC_Y[i]),
               (TX + WALK_W + 26, BY + tgt * (BHH + 8) + BHH / 2), (BX - 4, BY + tgt * (BHH + 8) + BHH / 2)],
              lc.C_GPU_S, 1.4, 'sam')
lc.text(BX + BW / 2, BY + 6 * (BHH + 8) + 8, 'C 占 logits 位、不占掩码行（不进 ids）——指纹同源：行 i 允许 token i*3',
        7.8, lc.C_MUTE, 'middle', maxw=330, tag='bm:f')

# ================= 底部：skip 快路径对照 =================
NY, NH = 560, 150
lc.rect(MX, NY, 700, NH, '#ffffff', lc.C_SAM_S, rx=9, sw=1.4)
lc.text(MX + 16, NY + 22, 'skip 快路径对照（全覆盖场景）', 10, lc.C_SAM_S, 'start', True,
        maxw=660, tag='sk:t')
for j, ln in enumerate(['· worker 批序 [B, A]（无非语法请求）：logits 5 行 == 掩码 5 行',
                        '→ xgr 调用 indices=None 免传（连 indices 张量都不用上卡，',
                        '  async_tensor_h2d 实测未被调用）',
                        '· 行数对满才免传、差一行都得传——indices 张量是快/慢路径的唯一分界']):
    lc.text(MX + 16, NY + 44 + j * 17, ln, 8.4, '#334155', 'start', maxw=670, tag='sk:l' + str(j))
lc.rect(790, NY, BXR - 790, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(806, NY + 22, '错位为何静默（反事实）', 10, lc.C_MUTE, 'start', True, maxw=600, tag='cf:t')
for j, ln in enumerate(['· 若不按 ids 重排、按批序直填：A 的行 0（允许 token 0）会写进',
                        '  logits 行 0——那是 B 的第一个采样位',
                        '· xgr 只管逐行写 -inf，无从知道行主是谁——约束戴错请求且不报错',
                        '· 生产 vocab=152064 时一行 4752 个 int32：错一行 = 18.6KiB 整行戴错']):
    lc.text(806, NY + 44 + j * 17, ln, 8.4, '#334155', 'start', maxw=BXR - 806 - 14,
            tag='cf:l' + str(j))

# ================= 图例 + 页脚 =================
LY = 736
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, '#ffffff', lc.C_GPU_S, rx=3, sw=1.3)
lc.text(lx0 + 21, LY + 1, 'B 的落位行（绿）', 8.8, lc.C_TXT, 'start', maxw=150, tag='lg1')
lx0 += 21 + lc.tw('B 的落位行（绿）', 8.8) + 14
lc.rect(lx0, LY - 9, 16, 11, '#f1f5f9', lc.C_MUTE, rx=3, sw=1.1)
lc.text(lx0 + 21, LY + 1, 'C 非语法行（-1 全允许）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg2')
lx0 += 21 + lc.tw('C 非语法行（-1 全允许）', 8.8) + 14
lc.rect(lx0, LY - 11, 26, 15, lc.C_BADGE_F, lc.C_ENG_S, rx=6, sw=1.0)
lc.text(lx0 + 13, LY + 1, 'li', 8, lc.C_ENG_S, 'middle', True, maxw=20, tag='lg3b')
lc.text(lx0 + 32, LY + 1, '= logit_index 起点（B/A 高亮）', 8.8, lc.C_TXT, 'start', maxw=230, tag='lg3')

lc.text(MX, 764, 'vllm/v1/structured_output/utils.py:L113-L141（apply_grammar_bitmask 重排循环：逐 req_ids 算 logit_index=batch_index+cumulative_offset、'
                 '按 ids 展开 sorted_bitmask、攒 out_indices）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 782, '走表三行 / out_indices=[5,0,1,2,3] / C 行 full_allow 保持 -1 / 全覆盖 5==5 skip ＝ 本章驱动脚本实测（真 utils + 真 xgr，V=64 指纹词表）'
                 '· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-reorder-walk.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
