#!/usr/bin/env python3
"""ch26 机制图 · prefill 打分三段流水(figure ch26-fig-prefill-scoring,模板 tensor-flow)

放大自 L0『模型层 indexer 框』——L2 拍片⑤ 打分+选块的 prefill 臂(站 9)展开:
① 分页 IndexCache(物理块碎片)→ ② cp_gather 收成请求序连续 workspace →
③ fp8_fp4_mqa_logits 打分(logits [10,76],因果窗阶梯)→ ④ top_k_per_row_prefill
按 cu_seqlen_ks/ke 因果窗选 top-6 写 buffer 行(不足 -1 哨兵)。

claim:prefill 打分三段流水:cp_gather 把分页 IndexCache 按块表收成请求序连续 workspace
(物理块 2/3 的碎片回到逻辑序)→ fp8_fp4_mqa_logits 对 (q_fp8, weights)×(k_quant, k_scale)
算 I_{t,:}=Σ_j w_j·ReLU(q_j·k_s)(ReLU 与逐头加权在核内,O(L²) 本尊)→ top_k_per_row_prefill
按 cu_seqlen_ks/ke 因果边界选 top-k 写 buffer 行(不足 -1 哨兵)。

logits 矩阵逐格真实:窗内/窗外/被选中三态全部按 traces 实测的 10 行窗与选中位绘制;
数字全部取自 figure spec 的 numbers。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 790
MX = 52
BXR = 1448

# ---- 实测数据(traces m05:10 行的窗与选中,窗内相对位→矩阵绝位列) ----
KS = [0] * 6 + [70] * 4
KE = [65, 66, 67, 68, 69, 70, 73, 74, 75, 76]
ROWS = ['r0·req0 pos64', 'r1·req0 pos65', 'r2·req0 pos66', 'r3·req0 pos67', 'r4·req0 pos68',
        'r5·req0 pos69', 'r6·req1 pos2', 'r7·req1 pos3', 'r8·req1 pos4', 'r9·req1 pos5']
PICKS_REL = [  # buffer 行(窗内相对位)
    [4, 9, 13, 16, 22, 24], [9, 51, 13, 48, 60, 61], [9, 16, 20, 22, 23, 24],
    [36, 61, 7, 34, 44, 58], [30, 1, 23, 50, 47, 61], [11, 12, 13, 15, 21, 22],
    [1, 2, 0], [2, 1, 3, 0], [2, 4, 3, 1, 0], [1, 3, 4, 2, 5, 0],
]
NCOL = 76

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'prefill 打分三段流水:先收卷、再逐题打分、每题只把前 k 名写上榜',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '物理块的碎片经块表收拢回请求序;logits 矩阵只读各自因果窗切片——未来 token 根本不进比较集;窗不足 top-6 时尾部补 -1 哨兵',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0『模型层 indexer 框』· L2 拍片⑤ 打分+选块 prefill 臂(站 9)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- ① 分页 IndexCache:三个物理块 ----------------
lc.text(MX, 136, '① 分页 IndexCache——物理块碎片(每条 132B)', 11, lc.C_TXT, 'start', True,
        maxw=360, tag='a:t')
blocks = [
    (60, '物理块 2', 'slots 128-191', 1.0, 'req0 pos 0-63(64/64)'),
    (172, '物理块 3', 'slots 192-255', 6 / 64, 'req0 pos 64-69(6/64)'),
    (284, '物理块 4', 'slots 256-319', 6 / 64, 'req1 pos 0-5(6/64)'),
]
BH_, BY_ = 118, 152
for bx, t1, t2, frac, note in blocks:
    lc.rect(bx, BY_, 96, BH_, '#ffffff', lc.C_KV_S, rx=7, sw=1.6)
    lc.text(bx + 48, BY_ + 17, t1, 9.5, '#155e75', 'middle', True, tag='b:' + t1)
    lc.text(bx + 48, BY_ + 31, t2, 7.5, lc.C_MUTE, 'middle', tag='b:s' + t1)
    fh = max(14, (BH_ - 44) * frac)
    lc.rect(bx + 8, BY_ + 40, 80, fh, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.0)
    lc.text(bx + 48, BY_ + BH_ - 8, note, 7.5, '#334155', 'middle', maxw=90, tag='b:n' + t1)

# 块表卡(块区右侧)
lc.rect(420, 152, 300, 118, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(570, 172, 'block_table = [[2, 3], [4, 0]]', 9.5, lc.C_TXT, 'middle', True, tag='bt:f')
for j, s in enumerate([
    'req0 → 块 2+3;req1 → 块 4(块 0 未用)',
    '物理块从 2 起:物理位 ≠ 逻辑位',
    '跨块边界:pos 63 在块 2、pos 64 起在块 3',
]):
    lc.text(570, 192 + j * 16, s, 8, '#334155', 'middle', maxw=280, tag='bt:l%d' % j)

# ---------------- ② workspace(收卷后的请求序长条) ----------------
WX0, WY_, CWP = 210, 330, 6.5
for i in range(NCOL):
    x = WX0 + i * CWP
    fill = '#cffafe' if i < 70 else '#67e8f9'
    lc.rect(x, WY_, CWP, 26, fill, 'none', rx=0, sw=0)
lc.rect(WX0, WY_, NCOL * CWP, 26, 'none', lc.C_KV_S, rx=3, sw=1.6)
lc.text(WX0 + 227, WY_ + 17, 'req0 70 条 × 132B', 8.5, '#155e75', 'middle', tag='w:r0')
lc.text(WX0 + 455 + 19, WY_ + 40, 'req1 6 条', 8, '#155e75', 'middle', tag='w:r1')
# 收卷箭头(三块 → workspace)
lc.parrow([(108, BY_ + BH_), (108, 302), (300, 302), (300, WY_)], lc.C_KV_S, 1.6, 'std')
lc.parrow([(220, BY_ + BH_), (220, 310), (630, 310), (630, WY_)], lc.C_KV_S, 1.6, 'std')
lc.parrow([(332, BY_ + BH_), (332, 316), (688, 316), (688, WY_)], lc.C_KV_S, 1.6, 'std')
# 段标签在三条收卷连线之后发射 + 白 halo(盲审 2026-09-14:第三条横线 y=316 从
# 字形中部穿过标签)——halo 让连线在字形处断开;基线 320→322 使 halo 上缘再避开
# 第二条横线(y=310)约 2.4px、下缘距 workspace 条顶(y=330)约 4.4px。
lc.text(457, 322, '② 收卷:请求序连续 workspace [76, 132]——cp_gather_indexer_k_quant_cache 按块表收拢(L456)',
        9.5, lc.C_KV_S, 'middle', True, maxw=520, tag='w:t', halo=True)
# 块界/请求界 tick 与换算注
for cx, lab in [(WX0 + 63 * CWP + CWP / 2, 'pos63→slot191(块2) | pos64→192(块3) | pos65→193'),
                (WX0 + 70 * CWP, 'req1 pos0→slot256(块4)')]:
    lc.seg(cx, WY_ + 26, cx, WY_ + 34, lc.C_MUTE, 1.0)
lc.text(WX0 + 63 * CWP + CWP / 2, WY_ + 48, 'pos63→slot191(块2) | pos64→192(块3) | pos65→193——跨块边界的间接寻址',
        8, lc.C_MUTE, 'middle', maxw=560, tag='w:tick1')
lc.text(WX0 + 70 * CWP, WY_ + 64, 'req1 pos0→slot256(块4)', 8, lc.C_MUTE, 'middle', tag='w:tick2')

# ---------------- ③ 打分核卡(右上) ----------------
KX, KY_, KW_, KH_ = 990, 140, 458, 130
lc.rect(KX, KY_, KW_, KH_, '#ffffff', lc.C_GPU_S, rx=9, sw=1.6)
lc.text(KX + 16, KY_ + 22, '③ 打分核 fp8_fp4_mqa_logits(L500)', 11, lc.C_GPU_S, 'start', True,
        maxw=400, tag='k:t')
lc.text(KX + 16, KY_ + 46, 'I(t,s) = Σ_j  w_j · ReLU(q_j · k_s)', 11.5, lc.C_TXT, 'start',
        True, maxw=KW_ - 30, tag='k:f')
for j, s in enumerate([
    '(q_fp8, weights) × (k_quant, k_scale)——q_scale 已折进 weights,核内只剩点积+ReLU+加权和',
    '逐头点积 + ReLU 门控 + 逐头加权全在核内:O(L²) 本尊',
    'logits [10, 76] fp32(3040 B)',
]):
    lc.text(KX + 16, KY_ + 68 + j * 17, s, 8.5, '#334155', 'start', maxw=KW_ - 30, tag='k:l%d' % j)
# workspace → 打分核
lc.parrow([(WX0 + NCOL * CWP, WY_ + 13), (940, WY_ + 13), (940, 205), (KX, 205)],
          lc.C_KV_S, 1.6, 'std')
lc.text(824, 358, '(k_quant, k_scale) 进核', 8.5, lc.C_KV_S, 'middle', tag='a:kw')

# ---------------- logits 矩阵(因果窗阶梯 + 选中格) ----------------
MY0, ROWH = 486, 17
lc.text(457, 466, 'logits 矩阵 [10, 76](fp32)——行 = 本拍 query token(r0-r9),列 = 历史 token 位(0-75)',
        9.5, lc.C_TXT, 'middle', True, maxw=560, tag='m:t')
C_WIN, C_OUT, C_PICK = '#dcfce7', '#f1f5f9', '#16a34a'
for r in range(10):
    y = MY0 + r * ROWH
    for c in range(NCOL):
        x = WX0 + c * CWP
        in_win = KS[r] <= c < KE[r]
        picked = in_win and ((c - KS[r]) in PICKS_REL[r])
        fill = C_PICK if picked else (C_WIN if in_win else C_OUT)
        lc.rect(x, y, CWP, ROWH, fill, 'none', rx=0, sw=0)
    lc.text(WX0 - 8, y + 12.5, ROWS[r], 7.5, lc.C_TXT if r not in (6,) else lc.C_ENG_S,
            'end', bold=(r in (0, 6, 9)), maxw=150, tag='rl:%d' % r)
lc.rect(WX0, MY0, NCOL * CWP, 10 * ROWH, 'none', lc.C_MUTE, rx=2, sw=1.4)
# r6 窗外框(虚线橙):窗 [70,73) 只 3 条
lc.rect(WX0 + 70 * CWP, MY0 + 6 * ROWH, 3 * CWP, ROWH, 'none', lc.C_ENG_S, rx=2, sw=1.6,
        dash=True)
# 打分核 → logits
lc.parrow([(KX + KW_ / 2, KY_ + KH_), (KX + KW_ / 2, 530), (WX0 + NCOL * CWP, 530)],
          lc.C_GPU_S, 1.8, 'std')
lc.text(1000, 522, 'logits [10, 76] fp32', 8.5, lc.C_GPU_S, 'middle', tag='a:lg')

# 列刻度(矩阵下缘)
for cx, lab, anch in [(WX0, '0', 'middle'), (WX0 + 64 * CWP - 3.25, '63|64 块界', 'middle'),
                      (WX0 + 70 * CWP, '69|70 请求界', 'middle'), (WX0 + NCOL * CWP, '75', 'middle')]:
    lc.seg(cx, MY0 + 10 * ROWH, cx, MY0 + 10 * ROWH + 6, lc.C_MUTE, 1.0)
    lc.text(cx, MY0 + 10 * ROWH + 18, lab, 8, lc.C_MUTE, anch, tag='ct:' + lab)

lc.text(MX, 690, '图例:深绿格 = 被 top-6 选中 · 浅绿格 = 因果窗内 · 灰格 = 窗外(分数不进比较集,结构性排除)',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 708, '因果窗阶梯:req0 窗 [0,65)→[0,70) 逐行 +1;req1 窗 [70,73)→[70,76);r6(虚线框)窗只 3 条 → 选中 3 + 尾部 3 个 -1',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 726, '例:r1 行 top3 分值 57.951 / 50.560 / 45.411(r1 选中位 [9,51,13,48,60,61])',
        9, lc.C_MUTE, 'start', maxw=700, tag='ft:2')

# ---------------- ④ 选块卡(右下) ----------------
DX, DY_, DW_, DH_ = 800, 560, 648, 185
lc.rect(DX, DY_, DW_, DH_, '#ffffff', lc.C_GPU_S, rx=9, sw=1.6)
lc.text(DX + 16, DY_ + 20, '④ 选块:top_k_per_row_prefill(L509)——按 cu_seqlen_ks/ke 因果窗选 top-6 写 buffer 行',
        9.5, lc.C_GPU_S, 'start', True, maxw=DW_ - 30, tag='d:t')
lc.text(DX + 16, DY_ + 38, 'cu_seqlen_ks = [0×6, 70×4] · cu_seqlen_ke = [65,66,67,68,69,70, 73,74,75,76]',
        8.5, '#334155', 'start', maxw=DW_ - 30, tag='d:kske')
lc.text(DX + 16, DY_ + 56, 'buffer 行示例(无效位 -1 哨兵,灰格):', 8.5, lc.C_TXT, 'start', True,
        maxw=300, tag='d:bt')
buf_rows = [
    ('r0(req0 pos64)', [4, 9, 13, 16, 22, 24], 0),
    ('r6(req1 pos2)', [1, 2, 0, -1, -1, -1], 3),
    ('r9(req1 pos5)', [1, 3, 4, 2, 5, 0], 0),
]
for k, (lab, cells, ngray) in enumerate(buf_rows):
    yy = DY_ + 66 + k * 28
    lc.text(DX + 16, yy + 14, lab, 8, lc.C_TXT, 'start', tag='d:r%d' % k)
    for j, v in enumerate(cells):
        gray = j >= 6 - ngray
        lc.rect(DX + 150 + j * 32, yy, 28, 22, '#f1f5f9' if gray else lc.C_GPU_F,
                lc.C_FAINT if gray else lc.C_GPU_S, rx=4, sw=1.0)
        lc.text(DX + 150 + j * 32 + 14, yy + 15, str(v), 9,
                lc.C_MUTE if gray else '#166534', 'middle', True, tag='d:c%d%s' % (k, v))
lc.text(DX + 16, DY_ + DH_ - 10, '全部 10 行与独立暴力参考逐行相等 = true(host 实测) · req1 首行窗长 3 → 尾部 3 个 -1',
        8.5, lc.C_MUTE, 'start', maxw=DW_ - 30, tag='d:f')
# 矩阵 → 选块卡
lc.seg(WX0 + NCOL * CWP, 600, DX, 600, lc.C_GPU_S, 1.8, 'std')

# ---------------- 页脚 ----------------
lc.text(MX, 768, '几何 = host 实测例(seq=[70,6], qry=[6,4], topk=6, block_size=64) · 函数名与行号 = vLLM v0.27.1(sparse_attn_indexer.py)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch26-fig-prefill-scoring.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
