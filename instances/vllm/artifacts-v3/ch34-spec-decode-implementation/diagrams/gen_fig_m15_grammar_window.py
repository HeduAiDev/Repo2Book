#!/usr/bin/env python3
"""ch34 机制图 · 语法 FSM 的 spec 窗口协议：乐观预推进→rollback→重推（figure_spec fig_m15_grammar_window，模板 state-table）

放大自 L0 主循环 loop_box 调度器臂语法窗口的 spec 变体——L2 章图 south
『spec×结构化输出』组件的展开（站 4 的 validate 过滤 + 站 16 的 rollback 收口；
架构归属回指 L2，不另立第二种架构画法）。

claim：语法 FSM 对 spec 窗口的协议：对 K=3 的窗口逐位预填 k+1=4 个掩码行（3 草稿位
+1 bonus 位）并乐观预推进 state_advancements 步（-1 占位草稿既不填掩码也不推进）、
采样后 grammar.rollback(state_advancements) 撤销全部预推进、本拍真实产出的 token
（accepted+recovered）由 update_from_output 重新前进——净效果=推进实际产出数
（c1：3 草稿收 1、补采 1，净推进 2）。

数字全部取自 figure_spec.numbers（traces/ch34_m02_scheduler_ledger.json c1 +
structured_output/__init__.py:L294-L350 / L228-L234 · backend_xgrammar.py:L119-L123 ·
scheduler.py:L2163-L2166）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 970
MX, BXR = 56, 1444
DEFS = lc.DEFS
ADV_C, RB_C, REAL_C = lc.C_GPU_S, lc.C_ABORT, '#0891b2'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '先乐观后回退：FSM 预推进 K 步填掩码，采样后 rollback、本拍真实产出逐个重推', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '语法窗口协议（c1 例：草稿 [31,32,33]、接受 1 + 补采 77 = 产出 2）· 掩码预算 max_batch*(1+max_spec) · 每请求窗口 k+1=4 行（3 草稿位+1 bonus 位）',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 主循环语法窗口（L2 south spec×结构化输出·站 4+16 展开）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= A. 窗口掩码表（4 行） =================
TX, TY = MX, 100
COLS = [('窗口行', 96), ('token（c1 例）', 130), ('动作', 250), ('状态计数 state_advancements', 200)]
ROWS = [
    ('行 0', '草稿 31', '_fill_bitmasks + accept_tokens(31) 试探推进', '0 → 1'),
    ('行 1', '草稿 32', '_fill_bitmasks + accept_tokens(32) 试探推进', '1 → 2'),
    ('行 2', '草稿 33', '_fill_bitmasks + accept_tokens(33) 试探推进', '2 → 3'),
    ('行 3', 'bonus 位', '末尾再填 1 行（bonus_apply）——不推进', '停在 3'),
]
HDR_H, ROW_H = 30, 40
TW = sum(w_ for _, w_ in COLS)
lc.rect(TX, TY, TW, HDR_H + len(ROWS) * ROW_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.5)
cx = TX
for name, w_ in COLS:
    lc.text(cx + w_ / 2, TY + 19, name, 9.2, lc.C_MUTE, 'middle', True, maxw=w_ - 8, tag='th' + name[:4])
    cx += w_
lc.seg(TX, TY + HDR_H - 4, TX + TW, TY + HDR_H - 4, lc.C_MUTE, 1.1)
for i, row in enumerate(ROWS):
    ry = TY + HDR_H + i * ROW_H
    if i:
        lc.seg(TX, ry, TX + TW, ry, '#e2e8f0', 1.0)
    cy = ry + ROW_H / 2 + 4
    lc.text(TX + COLS[0][1] / 2, cy, ROWS[i][0], 9.2, lc.C_TXT, 'middle', True, maxw=80, tag='r' + str(i))
    lc.text(TX + COLS[0][1] + COLS[1][1] / 2, cy, ROWS[i][1], 9.2, ADV_C if i < 3 else lc.C_SAM_S,
            'middle', True, maxw=110, tag='t' + str(i))
    lc.text(TX + COLS[0][1] + COLS[1][1] + 12, cy, ROWS[i][2], 8.2, '#475569', 'start', maxw=COLS[2][1] - 20, tag='a' + str(i))
    lc.text(TX + COLS[0][1] + COLS[1][1] + COLS[2][1] + COLS[3][1] / 2, cy, ROWS[i][3], 9.4, ADV_C,
            'middle', True, maxw=COLS[3][1] - 10, tag='s' + str(i))
lc.text(TX, TY + HDR_H + len(ROWS) * ROW_H + 18, '-1 占位草稿：既不填掩码也不推进（token == -1 分支直接跳过）', 8.4,
        lc.C_MUTE, 'start', maxw=TW, tag='neg1')

# ================= B. 前置闸（validate_tokens） =================
GX, GY, GW_, GH_ = MX + TW + 24, TY, BXR - (MX + TW + 24), 130
lc.rect(GX, GY, GW_, GH_, '#ffffff', lc.C_ZMQ_S, rx=9, sw=1.6)
lc.text(GX + 14, GY + 22, '前置闸（站 4 · 挂账前）', 10.2, lc.C_ZMQ_S, 'start', True, maxw=GW_ - 28, tag='gt')
for j, ln in enumerate(['grammar.validate_tokens 过滤非法草稿：', '语法不认的草稿直接不排——少了就少排、', '不硬凑（接受率换正确性）',
                        'scheduler.py:L2163-L2166']):
    lc.text(GX + 14, GY + 42 + j * 17, ('· ' if j else '') + ln, 8.2 if j else 8.8,
            '#475569' if j else lc.C_TXT, 'start', maxw=GW_ - 28, tag='gl' + str(j))

# ================= C. FSM 状态计数时间线（乐观→rollback→真实） =================
CY0 = TY + 240
lc.rect(MX, CY0, BXR - MX, 300, '#ffffff', lc.C_MUTE, rx=10, sw=1.4)
lc.text(MX + 16, CY0 + 24, 'FSM 状态计数时间线：乐观预推进 → 采样后 rollback → 真实产出重推（净效果 = 只推进实际产出数）', 11,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='ct')
AXY = CY0 + 250
def sy_(v):
    return AXY - v * 52
# 事件点：(x 比例, 值, 标签, 颜色, 上下)
EVT = [
    (0.05, 0, '窗口开始', lc.C_MUTE, 'down'),
    (0.19, 1, '试推 31', ADV_C, 'up'),
    (0.32, 2, '试推 32', ADV_C, 'up'),
    (0.45, 3, '试推 33', ADV_C, 'up'),
    (0.58, 0, 'rollback(3)\n撤销全部预推进', RB_C, 'down'),
    (0.70, 1, '重推 31（接受）', REAL_C, 'up'),
    (0.82, 2, '重推 77（补采）', REAL_C, 'up'),
    (0.93, 2, '净推进 = 2\n（= 实际产出数）', REAL_C, 'down'),
]
X0, X1 = MX + 130, BXR - 150
def ex_(f):
    return X0 + f * (X1 - X0)
# 轴与刻度
for v in range(4):
    lc.seg(X0 - 8, sy_(v), X1 + 40, sy_(v), '#eef2f7', 1.0)
    lc.text(X0 - 16, sy_(v) + 3, str(v), 9, lc.C_FAINT, 'end', maxw=30, tag='axv' + str(v))
lc.seg(X0 - 8, sy_(0), X0 - 8, sy_(3), '#cbd5e1', 1.2)
# 段（乐观=绿 / rollback=红 / 真实重推=青 / 持平=青无标签）
SEG_META = [(ADV_C, '+1'), (ADV_C, '+1'), (ADV_C, '+1'), (RB_C, '−3'),
            (REAL_C, '+1'), (REAL_C, '+1'), (REAL_C, '')]
for i in range(len(EVT) - 1):
    (f0, v0, *_), (f1, v1, *_rest) = EVT[i], EVT[i + 1]
    col, lab = SEG_META[i]
    lc.seg(ex_(f0), sy_(v0), ex_(f1), sy_(v1), col, 2.6)
    if lab:
        lc.text((ex_(f0) + ex_(f1)) / 2, (sy_(v0) + sy_(v1)) / 2 - 10, lab, 9.4, col, 'middle', True,
                maxw=70, tag='dl' + str(i))
# 点与标签
for f, v, lab, col, pos in EVT:
    lc.circle(ex_(f), sy_(v), 7, col, 2.2, dash=False)
    lines = lab.split('\n')
    dy0 = -18 if pos == 'up' else 24
    for j, ln in enumerate(lines):
        lc.text(ex_(f), sy_(v) + dy0 + j * 13, ln, 8.4, col, 'middle', True, maxw=150, tag='el' + ln[:8])
lc.text(MX + 16, AXY + 44, '时间 →（同一拍内：填掩码在采样前、rollback 在采样后）', 8.4, lc.C_MUTE, 'start',
        maxw=400, tag='tax')
# 真实重推口径注（时间线右上净空区）：update_from_output 吃进本拍全部产出 + c2 全收对照
ANN_X = ex_(0.60)
lc.text(ANN_X, CY0 + 50, '真实重推：update_from_output 落账时 accept_tokens 逐个吃进本拍全部产出 token（31 与 77 各一步）', 8.4,
        REAL_C, 'start', maxw=560, tag='ann1')
lc.text(ANN_X, CY0 + 68, 'c2 全收对照：接受的 41/42/43 + bonus 99 → 产出 4、净推进 4', 8.4,
        lc.C_MUTE, 'start', maxw=560, tag='ann2')

# ================= D. 底部要点 =================
DY0 = CY0 + 316
lc.rect(MX, DY0, BXR - MX, 92, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 18, DY0 + 24, '配套账：max_rollback_tokens = num_speculative_tokens（xgrammar matcher 构造参数——FSM 允许回退的步数就是草稿数）；', 9.4,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='d1')
lc.text(MX + 18, DY0 + 44, '位掩码预算 max_batch*(1+max_spec)（K=3 → 每请求 4 行，一次分配全批用）；-1 占位草稿跳过推进，不污染状态计数。', 8.8,
        lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='d2')
lc.text(MX + 18, DY0 + 68, '口径辨析：站 16 的 token 账回扣数的是拒绝位（c1 拒 2），FSM 净推进数的是产出位（c1 产 2）——两数碰巧相同、口径不同，同拍各记各的账。', 8.8,
        lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='d3')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 46, 'vllm/v1/structured_output/__init__.py:L294-L350（逐位 _fill_bitmasks + accept_tokens 试探 + 末尾 bonus 行 + rollback）· L228-L234（位掩码预算）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 32, 'vllm/v1/structured_output/backend_xgrammar.py:L119-L123（max_rollback_tokens）· vllm/v1/core/sched/scheduler.py:L2163-L2166（validate_tokens 前置过滤）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1b')
lc.text(MX, H - 18, 'K=3 / 4 行掩码 / c1 接受 1（token 账）· 产出 2（FSM 口径）＝ 本章驱动脚本实测（与调度器账本同一例）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m15_grammar_window.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
