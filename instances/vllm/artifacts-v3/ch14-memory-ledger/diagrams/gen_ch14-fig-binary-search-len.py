#!/usr/bin/env python3
"""ch14 机制图 2 · 护栏二分估长（figure_spec ch14-fig-binary-search-len，模板 state-table）

放大自 L0 启动段定账护栏——本章 L2 章图中排拍片②「护栏四道」的二分估长机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：护栏的二分估长：fits(len) 随 len 单调不减（每层 KV 只增不减），upper-bound
二分以 ≤ceil(log2 L) 次循环探针收敛到最大可行长度。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）；区间演化列由探针序列
按标准 upper-bound 二分推导（mid 归一侧并 ±1）。坐标由常量/循环计算；文本全 esc()。
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
lc.text(MX, 34, '护栏二分估长：池只够 100 块时，13 次循环探针在 8192 里折半收敛到 1600',
        16.5, lc.C_TXT, 'start', True, maxw=990, tag='title')
lc.text(MX, 58, 'fits(len) = cdiv(len,16) × 131072 B ≤ 13107200 B（= 100 个长度块）随 len 单调不减——每层 KV 只增不减，这是 upper-bound 二分正确性的根',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · 启动段定账护栏 · L2 拍片②'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：护栏一 + 边界 + 输出去向 ----------------
LX, LW = MX, 400
PANELS = [
    ('g1', '护栏一 check_enough · 至少装下一条 max_model_len', 92,
     ['needed(4096) = 33554432 B  vs  available 33554431 B',
      '差 1 字节也拦 → raise（宁可拒收，不给错账）'],
     'vllm/v1/core/kv_cache_utils.py:L751-L788（raise 现场）'),
    ('bd', '边界就在 1600/1601 之间——返回值恰是最大可行 len', 104,
     ['len 1600 → 100 块 ≤ 100 → 装下（result 终值）',
      'len 1601 → 101 块 > 100 → 不装',
      'cdiv 只增不减：块数随 len 上台阶、不下降'],
     None),
    ('out', '这个数的输出去向', 92,
     ['护栏一拦下的 raise 报错里附提示：',
      '「estimated maximum model length is 1600」',
      '——替用户回答『那我最多能开多长』'],
     None),
]
py = 96
p_pos = {}
for key, title, h, lines, file in PANELS:
    stroke = lc.C_ABORT if key == 'g1' else lc.C_KV_S
    fill = '#fef2f2' if key == 'g1' else lc.C_KV_F
    lc.rect(LX, py, LW, h, fill, stroke, rx=7, sw=1.5)
    lc.text(LX + 14, py + 20, title, 10.2, stroke if key == 'g1' else lc.C_KV_S, 'start', True,
            maxw=LW - 28, tag='p:' + key)
    for i, ln in enumerate(lines):
        lc.text(LX + 14, py + 39 + i * 17, ln, 8.8, '#334155', 'start', maxw=LW - 26,
                tag='pl:%s%d' % (key, i))
    if file:
        lc.text(LX + LW - 10, py + h - 8, file, 8, lc.C_FAINT, 'end', maxw=LW - 24,
                tag='pf:' + key)
    p_pos[key] = (py, h)
    py += h + 16

# ---------------- 左下：为什么必停且必准 ----------------
WY = py
lc.rect(LX, WY, LW, 118, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(LX + 14, WY + 19, '为什么必停、必准（不变量）', 10.2, lc.C_TXT, 'start', True,
        maxw=LW - 28, tag='w:t')
lc.text(LX + 14, WY + 38, '· 必停：每轮探针后区间 [left,right] 严格缩短', 8.8, '#334155',
        'start', maxw=LW - 26, tag='w:l1')
lc.text(LX + 14, WY + 55, '  （mid 归入一侧并 ±1）——至多 ceil(log2 8192) = 13 轮', 8.8,
        '#334155', 'start', maxw=LW - 26, tag='w:l2')
lc.text(LX + 14, WY + 72, '· 必准：fits 单调不减，二分全程维护', 8.8, '#334155', 'start',
        maxw=LW - 26, tag='w:l3')
lc.text(LX + 14, WY + 89, '  『≤ result 全装下、> right 全不装』，收敛即最大可行值', 8.8,
        '#334155', 'start', maxw=LW - 26, tag='w:l4')

# ---------------- 右：14 次探针实录表 ----------------
TX, TW_ = 500, 700
HDR_Y, ROW_H = 96, 24
COLS = [('探针', 50), ('试长度', 76), ('需字节', 96), ('需块', 56), ('判定', 66),
        ('缩后区间', 128), ('result 记账', 130)]
PROBES = [
    (1, '1', '131072', '1', '装下', '[1,8192] 起步', 'result=1（首检查）', 'pre'),
    (2, '4096', '33554432', '256', '不装', '[1,4095]', '1', 'no'),
    (3, '2048', '16777216', '128', '不装', '[1,2047]', '1', 'no'),
    (4, '1024', '8388608', '64', '装下', '[1025,2047]', '1024', 'fit'),
    (5, '1536', '12582912', '96', '装下', '[1537,2047]', '1536', 'fit'),
    (6, '1792', '14680064', '112', '不装', '[1537,1791]', '1536', 'no'),
    (7, '1664', '13631488', '104', '不装', '[1537,1663]', '1536', 'no'),
    (8, '1600', '13107200', '100', '装下', '[1601,1663]', '1600（终值）', 'final'),
    (9, '1632', '13369344', '102', '不装', '[1601,1631]', '1600', 'no'),
    (10, '1616', '13238272', '101', '不装', '[1601,1615]', '1600', 'no'),
    (11, '1608', '13238272', '101', '不装', '[1601,1607]', '1600', 'no'),
    (12, '1604', '13238272', '101', '不装', '[1601,1603]', '1600', 'no'),
    (13, '1602', '13238272', '101', '不装', '[1601,1601]', '1600', 'no'),
    (14, '1601', '13238272', '101', '不装', '[1601,1600] 收敛', '返回 1600', 'end'),
]
FILL = {'pre': '#f1f5f9', 'no': '#ffffff', 'fit': lc.C_KV_F, 'final': lc.C_KV_F,
        'end': lc.C_BEAT_F}
STROKE = {'pre': '#cbd5e1', 'no': '#e2e8f0', 'fit': '#a5f3fc', 'final': lc.C_KV_S,
          'end': lc.C_BEAT_S}
lc.rect(TX, HDR_Y, TW_, ROW_H, lc.C_KV_S, lc.C_KV_S, rx=4, sw=1.0)
cx = TX + 10
for name, cwid in COLS:
    lc.text(cx + cwid / 2, HDR_Y + 16, name, 9, '#ffffff', 'middle', True, maxw=cwid,
            tag='th:' + name)
    cx += cwid + 8
for ri, row in enumerate(PROBES):
    ry = HDR_Y + ROW_H + 5 + ri * (ROW_H + 3)
    kind = row[-1]
    lc.rect(TX, ry, TW_, ROW_H, FILL[kind], STROKE[kind], rx=3,
            sw=1.6 if kind in ('final', 'end') else 0.8)
    cx = TX + 10
    for (name, cwid), val in zip(COLS, row[:-1]):
        if name == '判定':
            col = lc.C_KV_S if val == '装下' else lc.C_ABORT
        elif name == 'result 记账' and ('1600' in val or '1024' in val or '1536' in val):
            col = lc.C_KV_S if kind != 'end' else lc.C_BEAT_T
        elif name == '试长度' and val in ('1600', '1601'):
            col = lc.C_KV_S
        else:
            col = '#334155'
        bold = name in ('判定', 'result 记账') and kind in ('final', 'end')
        lc.text(cx + cwid / 2, ry + 16, val, 8.8, col, 'middle', bold, maxw=cwid,
                tag='td%d:%s' % (ri, name))
        cx += cwid + 8
TB_BOT = HDR_Y + ROW_H + 5 + len(PROBES) * (ROW_H + 3)
lc.text(TX, TB_BOT + 15, '13 次循环探针 = ceil(log2 8192) 恰达上界（加首个 fits(1) 检查共 14 次调用）· '
        'try/finally 恢复 max_model_len = 8192——估长零副作用',
        8.6, lc.C_MUTE, 'start', maxw=TW_ + 240, tag='tb:note')

# ---------------- 表右侧：两段节奏注 ----------------
NX, NW = TX + TW_ + 16, BXR - (TX + TW_ + 16)
lc.rect(NX, HDR_Y, NW, 150, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(NX + 12, HDR_Y + 19, '两段节奏', 10.2, lc.C_TXT, 'start', True, maxw=NW - 24, tag='n:t')
lc.text(NX + 12, HDR_Y + 38, '· 粗定位（探针 2-4）：4096→2048→1024，', 8.6, '#334155',
        'start', maxw=NW - 22, tag='n:l1')
lc.text(NX + 12, HDR_Y + 54, '  每轮区间对半砍', 8.6, '#334155', 'start', maxw=NW - 22, tag='n:l2')
lc.text(NX + 12, HDR_Y + 73, '· 贴边收敛（探针 5-14）：1536→1792→', 8.6, '#334155',
        'start', maxw=NW - 22, tag='n:l3')
lc.text(NX + 12, HDR_Y + 89, '  1664→1600→1601，每轮贴近 1600 一格', 8.6, '#334155',
        'start', maxw=NW - 22, tag='n:l4')
lc.text(NX + 12, HDR_Y + 108, '· 探针 8 的 1600 恰好 100 块 = 池的', 8.6, '#334155',
        'start', maxw=NW - 22, tag='n:l5')
lc.text(NX + 12, HDR_Y + 124, '  全部——此后 6 探针都在证明没有更大的', 8.6, '#334155',
        'start', maxw=NW - 22, tag='n:l6')
lc.text(NX + 12, HDR_Y + 140, '  可行值', 8.6, '#334155', 'start', maxw=NW - 22, tag='n:l7')

# ---------------- 页脚 ----------------
FY = max(TB_BOT + 34, WY + 118 + 20)
lc.text(MX, FY, '逐字锚 vllm/v1/core/kv_cache_utils.py:L751-L788（_check_enough 差 1 字节拦截与 raise 提示）· L800-L851（estimate_max_model_len 二分，区间推导自探针序列）· '
        'L854-L879（check_enough 公开包装）· 数字取自配套精简版 host 实跑（2 层 full · block_size 16 · 每层页 65536 B · 每 16-token 长度块两层共 131072 B）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '行号基线 vLLM v0.27.1', 8.2, lc.C_FAINT, 'start', maxw=400, tag='foot2')

# ---------------- 装配输出 ----------------
H = FY + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-binary-search-len.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
