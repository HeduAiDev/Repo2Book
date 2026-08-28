#!/usr/bin/env python3
"""ch14 机制图 6 · SWA 实持块封顶（figure_spec ch14-fig-swa-cap-plateau，模板 state-table）

放大自 L0 调度账本列与 KV 账本列的接缝（准入门）——本章 L2 章图拍片⑥「准入门 ·
full-ISL」的 SWA 侧机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）。

claim：SWA 每请求实持块停在上限 cap = cdiv(window−1+max_in_flight_tokens, block_size)+1
之下：remove_skipped_blocks 在每个 chunk 的预测前先跑，窗外块先回收、实持只涨到
窗口+在途就封顶。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；
文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'SWA 实持块封顶：窗外块每步先回收，实持涨到窗口+在途就到顶——cap=5，序列再长也不涨',
        16.5, lc.C_TXT, 'start', True, maxw=990, tag='title')
lc.text(MX, 58, 'remove_skipped_blocks 在每个 chunk 的分配预测之前先跑；启动期定池大小与运行期准入门用 spec 里同一个方法算这个 cap——单源，漂移即 #39734 死锁',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · 调度×KV 账本接缝 · L2 拍片⑥'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶：cap 公式（左）+ 参数（右） ----------------
FY0 = 92
FW = 660
lc.rect(MX, FY0, FW, 96, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.5)
lc.text(MX + 16, FY0 + 21, '准入上限公式（SWA spec）', 10.5, lc.C_KV_S, 'start', True,
        maxw=FW - 32, tag='f:t')
lc.text(MX + 16, FY0 + 41, 'cap = cdiv(min(window−1+in_flight, max_len), bs) + 1 = cdiv(15,4)+1 = 5',
        9.6, '#334155', 'start', maxw=FW - 32, tag='f:l1')
lc.text(MX + 16, FY0 + 59, '+1 顶着窗口起点不在块首的最坏错位（源码注释例：bs 4 存', 8.8,
        '#475569', 'start', maxw=FW - 32, tag='f:l2')
lc.text(MX + 16, FY0 + 76, '6-token 窗要 [XXCD][EF] 两块）；chunked 窗口从块首开始无 +1：cdiv(8,4)=2',
        8.8, '#475569', 'start', maxw=FW - 32, tag='f:l3')
PX2 = MX + FW + 24
PW2 = BXR - PX2
lc.rect(PX2, FY0, PW2, 96, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
lc.text(PX2 + 16, FY0 + 21, '推进场景', 10.5, lc.C_TXT, 'start', True, maxw=PW2 - 32, tag='p:t')
lc.text(PX2 + 16, FY0 + 41, 'window 8 · block_size 4 · max_in_flight 8 · max_len 64 · 池 16 块',
        9.2, '#334155', 'start', maxw=PW2 - 32, tag='p:l1')
lc.text(PX2 + 16, FY0 + 59, '64-token 请求按 8-token chunk 推进 8 步（chunked prefill）', 9.2,
        '#334155', 'start', maxw=PW2 - 32, tag='p:l2')
lc.text(PX2 + 16, FY0 + 76, '窗外 token = max(0, computed − 8 + 1)，只收整块', 8.8, '#475569',
        'start', maxw=PW2 - 32, tag='p:l3')

# ---------------- 中左：八步推进表 ----------------
TY0 = FY0 + 120
TX, TW_ = MX, 560
COLS = [('步', 44), ('computed 前', 96), ('窗外 token', 90), ('实持 after', 86), ('≤ cap 5', 74),
        ('池 free', 66)]
ROWS = [
    ('1', '0', '0', '2', '✓', '13'),
    ('2', '8', '1', '4', '✓', '11'),
    ('3', '16', '9', '4', '✓', '11'),
    ('4', '24', '17', '4', '✓', '11'),
    ('5', '32', '25', '4', '✓', '11'),
    ('6', '40', '33', '4', '✓', '11'),
    ('7', '48', '41', '4', '✓', '11'),
    ('8', '56', '49', '4', '✓', '11'),
]
ROW_H = 25
lc.rect(TX, TY0, TW_, ROW_H, lc.C_KV_S, lc.C_KV_S, rx=4, sw=1.0)
cx = TX + 10
for name, cwid in COLS:
    lc.text(cx + cwid / 2, TY0 + 17, name, 9, '#ffffff', 'middle', True, maxw=cwid,
            tag='th:' + name)
    cx += cwid + 8
for ri, row in enumerate(ROWS):
    ry = TY0 + ROW_H + 4 + ri * (ROW_H + 3)
    fill = lc.C_KV_F if row[3] == '4' else '#ffffff'
    lc.rect(TX, ry, TW_, ROW_H, fill, '#a5f3fc' if row[3] == '4' else '#e2e8f0', rx=3, sw=0.9)
    cx = TX + 10
    for (name, cwid), val in zip(COLS, row):
        col = lc.C_KV_S if name == '实持 after' else '#334155'
        lc.text(cx + cwid / 2, ry + 17, val, 9, col, 'middle', name == '实持 after',
                maxw=cwid, tag='td%d:%s' % (ri, name))
        cx += cwid + 8
TB0 = TY0 + ROW_H + 4 + 8 * (ROW_H + 3)
lc.text(TX, TB0 + 15, '窗外 1 token（步 2）不够整块不收；稳态每步窗外多 2 块、补 2 块——实持钉在 4，池 free 回升后不再降',
        8.6, lc.C_MUTE, 'start', maxw=TW_ + 140, tag='tb:note')

# ---------------- 中右：plateau 条形图 ----------------
BX0 = TX + TW_ + 40
BW0 = BXR - BX0
BP_H = TB0 - TY0 + 6
lc.rect(BX0, TY0, BW0, BP_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(BX0 + 16, TY0 + 20, '实持块逐 chunk 推进：2 → 4 封顶', 10.5, lc.C_TXT, 'start', True,
        maxw=BW0 - 32, tag='bp:t')
BASE_Y = TY0 + BP_H - 42      # 数值 0 基线（柱底）
UNIT = 26                     # 每 1 块 = 26px（向上）
BARW, BARG = 44, 20
plot_x0 = BX0 + 44
for i in range(6):
    yy = BASE_Y - i * UNIT
    if i:
        lc.seg(BX0 + 28, yy, BX0 + BW0 - 16, yy, '#e2e8f0', 0.8)
    lc.text(BX0 + 22, yy + 3, str(i), 8, lc.C_MUTE, 'end', maxw=16, tag='ax%d' % i)
cap_y = BASE_Y - 5 * UNIT
lc.seg(BX0 + 28, cap_y, BX0 + BW0 - 16, cap_y, lc.C_ABORT, 1.4, dash=True)
lc.text(BX0 + BW0 - 18, cap_y - 6, 'cap = 5（+1 余量）', 8.6, lc.C_ABORT, 'end', maxw=140,
        tag='cap:lbl')
held = [2, 4, 4, 4, 4, 4, 4, 4]
for i, v in enumerate(held):
    bx = plot_x0 + i * (BARW + BARG)
    lc.rect(bx, BASE_Y - v * UNIT, BARW, v * UNIT, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.3)
    lc.text(bx + BARW / 2, BASE_Y - v * UNIT - 7, str(v), 9, lc.C_KV_S, 'middle', True,
            maxw=BARW, tag='bv%d' % i)
    lc.text(bx + BARW / 2, BASE_Y + 14, '步%d' % (i + 1), 8, '#475569', 'middle',
            maxw=BARW + 8, tag='bl%d' % i)
lc.seg(BX0 + 28, BASE_Y, BX0 + BW0 - 16, BASE_Y, '#94a3b8', 1.2)
lc.text(BX0 + BW0 / 2, TY0 + BP_H - 12,
        '第 3 步起稳态：实持 4 ≤ cap 5，序列再长也不涨（窗外早被回收）', 8.8, lc.C_MUTE,
        'middle', maxw=BW0 - 32, tag='bp:note')

# ---------------- 底：混合门对比（全宽） ----------------
HY = TB0 + 38
HH = 150
lc.rect(MX, HY, BXR - MX, HH, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, HY + 21, '混合模型过准入门：full 组按整序列、SWA 组被夹到 cap——4096-token 请求，池 1000 块 · free 999',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='hy:t')
SCALE = 0.62                    # px / 块：999 → 619px
BAR_X0, BAR_H2, BAR_GAP2 = MX + 130, 24, 14
FREE_X = BAR_X0 + int(999 * SCALE)
lc.seg(FREE_X, HY + 34, FREE_X, HY + HH - 16, lc.C_ABORT, 1.4, dash=True)
lc.text(FREE_X - 6, HY + 48, 'free 999', 8.6, lc.C_ABORT, 'end', maxw=80, tag='free:lbl')
BARS = [
    ('夹到 cap', 256, 33, '289 ≤ 999 → 放行', lc.C_KV_S),
    ('不夹（整序列）', 256, 256, '512——白丢一半并发', '#94a3b8'),
]
for bi, (label, full_b, swa_b, verdict, stroke) in enumerate(BARS):
    by = HY + 58 + bi * (BAR_H2 + BAR_GAP2)
    lc.text(BAR_X0 - 10, by + 16, label, 8.8, lc.C_TXT, 'end', maxw=120, tag='hb%d:l' % bi)
    lc.rect(BAR_X0, by, full_b * SCALE, BAR_H2, lc.C_API_F, lc.C_API_S, rx=2, sw=1.1)
    lc.text(BAR_X0 + full_b * SCALE / 2, by + 16, 'full %d' % full_b, 8.2, lc.C_API_S,
            'middle', True, maxw=full_b * SCALE - 8, tag='hb%d:f' % bi)
    lc.rect(BAR_X0 + full_b * SCALE + 2, by, swa_b * SCALE, BAR_H2, lc.C_KV_F, stroke,
            rx=2, sw=1.1)
    end_x = BAR_X0 + full_b * SCALE + 2 + swa_b * SCALE
    if bi == 0:
        lc.text(BAR_X0 + full_b * SCALE + 2 + swa_b * SCALE / 2, by - 5, 'SWA 夹到 33',
                8.2, lc.C_KV_S, 'middle', maxw=90, tag='hb%d:s' % bi)
    lc.text(end_x + 12, by + 16, verdict, 9, stroke if bi == 0 else '#475569', 'start',
            True, maxw=BXR - MX - 16 - end_x - 12, tag='hb%d:v' % bi)
lc.text(MX + 16, HY + HH - 16, '并发换算：1000/289 ≈ 3.4 条 vs 1000/512 ≈ 2.0 条——准入上限把 SWA 混合模型的并发从 ~2 条救回 ~3.4 条',
        8.8, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='hy:note')

# ---------------- 页脚 ----------------
FOOT_Y = HY + HH + 24
lc.text(MX, FOOT_Y, '单源铁律（源码注释原话）：Drift between the two would re-introduce the deadlock from issue #39734 or, worse, mid-prefill OOM——',
        8.4, '#475569', 'start', maxw=BXR - MX, tag='foot0')
lc.text(MX, FOOT_Y + 16, 'cap 由 max_admission_blocks_per_request 算出，启动期池大小器（max_memory_usage_bytes = cap × page）与运行期准入门同方法——预测器 = 分配器同构',
        8.4, '#475569', 'start', maxw=BXR - MX, tag='foot0b')
lc.text(MX, FOOT_Y + 34, '逐字锚 vllm/v1/kv_cache_interface.py:L519-L546 / L587-L618（cap 公式）· '
        'vllm/v1/core/single_type_kv_cache_manager.py:L178-L192（先回收后预测）/ L1861-L1877（准入门注入）· 数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
H = FOOT_Y + 52
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-swa-cap-plateau.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
