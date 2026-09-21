#!/usr/bin/env python3
"""ch34 机制图 · greedy kernel 输出矩阵逐格写出（figure_spec fig_m8_greedy_grid，模板 state-table）

放大自 L0 采样列 spec 块验证期的 greedy 路径——L2 章图第⑧拍内 greedy kernel 分支的
机制小图（步进图层：逐格写出）。

claim：greedy kernel 的 [4, max_spec_len+1] 输出矩阵逐格写出：4 请求 [3,2,1,0] 草稿
各走『对答案』路径——全收补 bonus、首拒写 argmax、-1 占位首位即拒、k=0 请求 bonus
写第 0 位。

数字全部取自 figure_spec.numbers（traces/ch34_m08_greedy_kernel.json：drafts_flat /
target_argmax_flat / bonus_ids / output_matrix / account）。坐标由常量/循环计算。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 930
MX, BXR = 56, 1444
DEFS = lc.DEFS

ACC_F, ACC_S = lc.C_GPU_F, lc.C_GPU_S      # 接受（draft==argmax）
REC_F, REC_S = '#fff7ed', lc.C_ENG_S       # 拒绝位写 argmax（recovered）
BON_F, BON_S = lc.C_SAM_F, lc.C_SAM_S      # bonus
PH_F, PH_S = '#f1f5f9', '#cbd5e1'          # -1 哨兵

# ---------------- 标题区 ----------------
lc.text(MX, 34, '对答案式验证：草稿与标答逐位比，比到第一个不一样就用标答顶上、后面作废', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'greedy kernel（draft == target_argmax 即收）· 4 请求一批 6 个草稿位、buffer [4, max_spec_len+1]=[4,4] 预填 -1 · 标答 = target 的 argmax',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列+spec 验证期（L2 ⑧ 内 greedy kernel 分支·步进图层）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 主走表：每请求一行 =================
TX, TY = MX, 100
COLS = [('请求 · 草稿区间', 250), ('草稿 vs 标答（逐位）', 330), ('输出行（4 列 buffer）', 300), ('本拍产出', 120)]
HDR_H, ROW_H = 34, 118
ROWS = [
    dict(req='req0（k=3）', seg='区间 [0,3)', cmp=[('3', '3', 'eq'), ('5', '5', 'eq'), ('6', '6', 'eq')],
         out=[('3', 'acc'), ('5', 'acc'), ('6', 'acc'), ('90', 'bon')],
         verdict='三位全等 → 全收', prod='4 token（k+1 上限）',
         note='bonus 90 写第 3 位（=k 位）'),
    dict(req='req1（k=2）', seg='区间 [3,5)', cmp=[('1', '1', 'eq'), ('0', '7', 'ne')],
         out=[('1', 'acc'), ('7', 'rec'), ('-1', 'ph'), ('-1', 'ph')],
         verdict='第 1 位草稿 0 ≠ argmax 7', prod='2 token',
         note='写标答 7、早停——后两格保持 -1'),
    dict(req='req2（k=1，pad 占位）', seg='区间 [5,6)', cmp=[('-1', '4', 'ne')],
         out=[('4', 'rec'), ('-1', 'ph'), ('-1', 'ph'), ('-1', 'ph')],
         verdict='-1 ≠ 任何 argmax → 首位即拒', prod='1 token',
         note='占位草稿永远比不中'),
    dict(req='req3（k=0，猜不出）', seg='区间 [6,6)', cmp=[],
         out=[('93', 'bon'), ('-1', 'ph'), ('-1', 'ph'), ('-1', 'ph')],
         verdict='循环 0 次、not rejected', prod='1 token（保底）',
         note='bonus 93 写第 0 位——无草稿也有 1 token'),
]
TW = sum(w_ for _, w_ in COLS)
TH = HDR_H + len(ROWS) * ROW_H
lc.rect(TX, TY, TW, TH, '#ffffff', lc.C_MUTE, rx=9, sw=1.6)
cx = TX
for name, w_ in COLS:
    lc.text(cx + w_ / 2, TY + 20, name, 9.4, lc.C_MUTE, 'middle', True, maxw=w_ - 8, tag='th' + name[:4])
    cx += w_
lc.seg(TX, TY + HDR_H - 4, TX + TW, TY + HDR_H - 4, lc.C_MUTE, 1.2)
cx = TX
for _, w_ in COLS[:-1]:
    cx += w_
    lc.seg(cx, TY + HDR_H - 4, cx, TY + TH - 4, '#e2e8f0', 1.0)

STY = {'acc': (ACC_F, ACC_S), 'rec': (REC_F, REC_S), 'bon': (BON_F, BON_S), 'ph': (PH_F, PH_S)}
for i, row in enumerate(ROWS):
    ry = TY + HDR_H + i * ROW_H
    if i:
        lc.seg(TX, ry, TX + TW, ry, '#e2e8f0', 1.0)
    cy = ry + ROW_H / 2
    # 请求
    lc.text(TX + 12, ry + 24, row['req'], 10, lc.C_TXT, 'start', True, maxw=COLS[0][1] - 20, tag='rq' + str(i))
    lc.text(TX + 12, ry + 40, row['seg'], 8.4, lc.C_FAINT, 'start', maxw=COLS[0][1] - 20, tag='sg' + str(i))
    lc.text(TX + 12, ry + 62, row['verdict'], 8.6, lc.C_MUTE, 'start', maxw=COLS[0][1] - 18, tag='vd' + str(i))
    lc.text(TX + 12, ry + 94, row['note'], 7.8, lc.C_FAINT, 'start', maxw=COLS[0][1] - 18, tag='nt' + str(i))
    # 比较
    bx = TX + COLS[0][1]
    if row['cmp']:
        for j, (d, a, rel) in enumerate(row['cmp']):
            ccx = bx + 12 + j * 102
            eq_ = rel == 'eq'
            lc.rect(ccx, ry + 18, 100, 40, ACC_F if eq_ else REC_F, ACC_S if eq_ else REC_S, rx=5, sw=1.3)
            lc.text(ccx + 48, ry + 36, ('draft ' + d + (' = ' if eq_ else ' ≠ ') + 'argmax ' + a), 8.0,
                    ACC_S if eq_ else REC_S, 'middle', True, maxw=96, tag='cp' + str(i) + str(j))
        lc.text(bx + 18, ry + 82, '逐位比、首拒即停（rejected 单调不回）', 7.8, lc.C_FAINT, 'start',
                maxw=COLS[1][1] - 30, tag='cmpn' + str(i))
    else:
        lc.text(bx + 18, ry + 36, '无草稿位——循环 0 次', 8.6, lc.C_MUTE, 'start', maxw=COLS[1][1] - 30, tag='cp' + str(i) + 'x')
    # 输出行
    ox = TX + COLS[0][1] + COLS[1][1]
    for j, (v, kind) in enumerate(row['out']):
        f, st = STY[kind]
        x = ox + 10 + j * 66
        lc.rect(x, ry + 16, 60, 40, f, st, rx=5, sw=1.4)
        lc.text(x + 30, ry + 41, v, 11.5, st, 'middle', True, maxw=56, tag='oc' + str(i) + str(j))
        if j == 3:
            lc.text(x + 30, ry + 8, 'bonus 槽（第 k 列）', 7.2, BON_S, 'middle', maxw=88, tag='bslot' + str(i))
    if i == 0:
        lc.text(ox + 10, ry + 8, '列 0        列 1        列 2        列 3（=k）', 7.2, lc.C_FAINT, 'start',
                maxw=290, tag='colh')
    # 产出
    px_ = TX + COLS[0][1] + COLS[1][1] + COLS[2][1]
    lc.text(px_ + 60, ry + 40, row['prod'], 9.6, lc.C_TXT, 'middle', True, maxw=COLS[3][1] - 10, tag='pd' + str(i))

# ================= 底部：账 + 图例 =================
BY0 = TY + TH + 18
lc.rect(MX, BY0, BXR - MX, 92, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lx0 = MX + 18
for kind, lab in [('acc', '接受（draft=argmax）'), ('rec', '拒绝位写标答 argmax'), ('bon', 'bonus（外采结果）'), ('ph', '-1 哨兵（未写位）')]:
    f, st = STY[kind]
    lc.rect(lx0, BY0 + 14, 18, 14, f, st, rx=3, sw=1.2)
    lc.text(lx0 + 24, BY0 + 25, lab, 8.8, lc.C_TXT, 'start', maxw=170, tag='lg' + kind)
    lx0 += 24 + lc.tw(lab, 8.8) + 22
lc.text(MX + 18, BY0 + 52, '账：buffer [4,4] 共 16 格，有效 8 格、-1 哨兵 8 格——接受越多白嫖越多、拒绝越多 -1 越多；产出 [4,2,1,1] token（req0 到达一次前向的上限 k+1=4）。',
        9.4, lc.C_TXT, 'start', maxw=BXR - MX - 36, tag='acct')
lc.text(MX + 18, BY0 + 72, '每位写出的 token 恒等于该位 target argmax——greedy 拒绝采样无损（ch33 one-q 特例在 kernel 形态的兑现）。',
        8.6, lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='acct2')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/sample/rejection_sampler.py:L715-L769（rejection_greedy_sample_kernel 逐字）· cu_num_draft_tokens 反推本请求草稿区间 · L755-L761（bonus 写第 num_draft 位）· L807-L809（-1 直接拒）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, 'drafts=[3,5,6,1,0,-1] / argmax=[3,5,6,1,7,4] / bonus=[90,91,92,93] / 输出矩阵 / 16 格·有效 8·哨兵 8 ＝ 本章 Triton kernel 真跑实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m8_greedy_grid.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
