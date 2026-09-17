#!/usr/bin/env python3
"""ch32 机制图 · 思考门控三件套（figure_spec ch32-fig-thinking-gates，模板 flow）

放大自 L0 调度列与采样列之间的思考门控决策展开（L2 站 14 的机制放大——⑤ 拍
update_from_output 侧的三问流程），架构归属回指 L2 章图，不另立第二种架构画法
（FIGURE-SYSTEM §3）。

claim：三问一门轴：should_fill（这步填不填）/ should_advance（这步推不推进，v0.27 用
new_token_ids 精确窗口）/ trim（喂之前裁掉思考页）——思考标记本身也是思考内容，
混块直喂会杀死请求。

数字全部取自 figure_spec.numbers（精确窗口 start=7-4=3 覆盖标记 idx 5 → 检出 vs
占位数 start=8-2=6 → delta=[g1] 错过 → False；trim 裁 3 个思考 token 后 accept([g1])
存活 status=None；不 trim 直喂 accept([t1,t2,M,g1]) → FINISHED_ERROR、resumable=False；
prompt 级判定恰 1 次；源码锚 __init__.py:L361-L379 / L381-L439 / L442-L486）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 940
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '思考门控三件套：should_fill / should_advance / trim——思考标记本身也是思考内容', 16.5,
        lc.C_TXT, 'start', True, maxw=1150, tag='title')
lc.text(MX, 58, '同一序列 [p1 p2 p3 t1 t2 M g1]（t 系=思考 token，M=结束标记在 idx 5，g1=语法内容）；'
               'v0.27 拿本步刚追加的 new_token_ids 当精确 delta 窗口——旧占位数推导在 async+spec 草稿被拒时窗口算错',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 调度列↔采样列·思考门控 · L2 站 14 的机制放大（⑤ 拍侧）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 三问链（横排三菱形） =================
QY, QH = 106, 250
GW = 410
GAP = (BXR - MX - 3 * GW) / 2


def gate(x, num, name, sub, lines, anchor_note):
    lc.rect(x, QY, GW, QH, lc.C_SAM_F, lc.C_SAM_S, rx=10, sw=2.0)
    bw = 30
    lc.rect(x + 12, QY + 10, bw + 8, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.0)
    lc.text(x + 16 + bw / 2, QY + 24, num, 8.6, lc.C_ENG_S, 'middle', True, maxw=28, tag='gn' + num)
    lc.text(x + bw + 30, QY + 24, name, 11, lc.C_TXT, 'start', True, maxw=GW - bw - 60,
            tag='gt' + num)
    lc.text(x + 16, QY + 48, sub, 8.2, lc.C_SAM_S, 'start', maxw=GW - 32, tag='gs' + num)
    for j, ln in enumerate(lines):
        lc.text(x + 16, QY + 70 + j * 17, ln, 8.2, '#334155', 'start', maxw=GW - 30,
                tag='gl' + num + str(j))
    lc.seg(x + 16, QY + QH - 34, x + GW - 16, QY + QH - 34, '#e2e8f0', 1.0)
    lc.text(x + 16, QY + QH - 16, anchor_note, 7.8, lc.C_MUTE, 'start', maxw=GW - 32,
            tag='ga' + num)


gate(MX, '①', 'should_fill_bitmask', '这步填不填约束？',
     ['reasoning_ended 未定时，先对 prompt 判一次', '（is_reasoning_end → False：整个 prompt 都在思考内）',
      '并缓存进请求——第二次调用不再判', '（prompt 级判定恰 1 次，两次都返回 False）'],
     '缓存进请求 · 幂等')
gate(MX + GW + GAP, '②', 'should_advance', '这步产出能不能喂 FSM？',
     ['v0.27 新签名：should_advance(req, new_token_ids)', '拿本步刚追加的 token 当精确 delta 窗口',
      '检出结束 → reasoning_ended=True 持久化', '· 边界定位 idx=5（标记本身）'],
     '精确窗口（new_token_ids）')
gate(MX + 2 * (GW + GAP), '③', 'trim_reasoning_for_advance', '喂之前要不要撕掉思考页？',
     ['检出结束 → 逐 token 定位边界（首个触发', 'streaming 判定的 idx 即返回）',
      'trim 裁掉 [first_idx, idx] 闭区间 = 3 个思考', 'token（t1, t2, M 含标记）——只喂 [g1]'],
     '闭区间 [3, 5] · num_reasoning=3')

# 三问之间的箭头
for k in range(2):
    x0 = MX + GW + k * (GW + GAP)
    lc.seg(x0 + 6, QY + QH / 2, x0 + GAP - 6, QY + QH / 2, lc.C_SAM_S, 2.2, 'sam')
lc.text(MX + GW + GAP / 2, QY + QH / 2 - 24, '未结束→不填', 7.8, lc.C_MUTE, 'middle', maxw=100,
        tag='q12')
lc.text(MX + 2 * GW + GAP + GAP / 2, QY + QH / 2 - 24, '检出→定位', 7.8, lc.C_MUTE, 'middle',
        maxw=100, tag='q23')
lc.text(MX + GW + GAP / 2, QY + QH / 2 + 16, 'False 时整行放行', 7.8, lc.C_MUTE, 'middle',
        maxw=110, tag='q12b')
lc.text(MX + 2 * GW + GAP + GAP / 2, QY + QH / 2 + 16, 'idx=5', 7.8, lc.C_MUTE, 'middle',
        maxw=100, tag='q23b')

# ================= 窗口对照条（同一序列、两个 delta 窗口） =================
WY = QY + QH + 30
lc.rect(MX, WY, BXR - MX, 190, '#ffffff', lc.C_MUTE, rx=9, sw=1.3)
lc.text(MX + 16, WY + 22, '窗口对照：同一序列上两个 delta 窗口——精确窗口覆盖标记、占位数推导越过标记', 10,
        lc.C_TXT, 'start', True, maxw=1100, tag='w:t')
SEQ = [('p1', 0), ('p2', 0), ('p3', 0), ('t1', 1), ('t2', 1), ('M', 2), ('g1', 0)]
CELL_W2, CELL_H2, CELL_GAP2 = 92, 34, 8
SX0 = MX + 70
SY0 = WY + 44
for k, (nm, kind) in enumerate(SEQ):
    x = SX0 + k * (CELL_W2 + CELL_GAP2)
    f_, s_, t_ = ('#ffffff', lc.C_MUTE, '#334155')
    if kind == 1:
        f_, s_, t_ = ('#fef9c3', '#ca8a04', '#713f12')
    elif kind == 2:
        f_, s_, t_ = ('#fed7aa', '#ea580c', '#9a3412')
    lc.rect(x, SY0, CELL_W2, CELL_H2, f_, s_, rx=6, sw=1.3)
    lc.text(x + CELL_W2 / 2, SY0 + 15, nm, 9.4, t_, 'middle', True, maxw=CELL_W2 - 6, tag='sq' + nm)
    lc.text(x + CELL_W2 / 2, SY0 + 28, f'idx {k}', 7.2, lc.C_MUTE, 'middle', maxw=CELL_W2,
            tag='sqi' + nm)
# 精确窗口框（start=3，覆盖 t1 t2 M g1）
PW_X = SX0 + 3 * (CELL_W2 + CELL_GAP2) - 5
PW_W = 4 * CELL_W2 + 3 * CELL_GAP2 + 10
lc.rect(PW_X, SY0 - 26, PW_W, CELL_H2 + 52, 'none', lc.C_SAM_S, rx=8, sw=2.0)
lc.text(PW_X + PW_W / 2, SY0 - 34, '精确窗口：start = 7-4 = 3（len(all)-len(new)）', 8.4,
        lc.C_SAM_S, 'middle', True, maxw=380, tag='pw:t')
lc.text(PW_X + PW_W / 2, SY0 + CELL_H2 + 32, 'delta 覆盖 [t1 t2 M g1] → 检出结束（边界 idx 5）', 8,
        lc.C_SAM_S, 'middle', True, maxw=380, tag='pw:s')
# 占位数窗口框（start=6，只含 g1；置于精确窗口标注之下）
FW_X = SX0 + 6 * (CELL_W2 + CELL_GAP2) - 5
FW_W = CELL_W2 + 10
FW_Y = SY0 + CELL_H2 + 44
lc.rect(FW_X, FW_Y, FW_W, 26, 'none', lc.C_ABORT, rx=6, sw=1.6, dash=True)
lc.text(SX0 + 6 * (CELL_W2 + CELL_GAP2) + CELL_W2 / 2, FW_Y + 17, '占位数：start = 8-2 = 6', 7.8,
        lc.C_ABORT, 'middle', True, maxw=200, tag='fw:t')
lc.text(MX + 16, WY + 178, 'delta=[g1] 错过标记 → 返回 False：语法永不生效（#43388 病灶——async+spec 草稿被拒时占位数残留未清零，'
                           '假设 num_output_placeholders==len(new_token_ids) 破裂）',
        8, lc.C_ABORT, 'start', maxw=1340, tag='fw:s')

# ================= 底部：生死对照双支 =================
LY0 = WY + 190 + 26
BH = 150
BW2 = (BXR - MX - 20) / 2
# 存活支（绿）
lc.rect(MX, LY0, BW2, BH, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.8)
lc.text(MX + 16, LY0 + 22, 'trim 后：accept([g1]) → 存活', 10.5, lc.C_GPU_S, 'start', True,
        maxw=BW2 - 32, tag='lv:t')
for j, ln in enumerate(['· 裁掉 3 个思考 token（t1, t2, M——含标记本身）',
                        '· 只喂语法内容 g1：真 xgrammar 接受成功',
                        '· status=None（请求存活）· accept 账 [[31]]']):
    lc.text(MX + 16, LY0 + 46 + j * 18, ln, 8.4, '#334155', 'start', maxw=BW2 - 30,
            tag='lv:l' + str(j))
lc.text(MX + 16, LY0 + BH - 14, 'vllm/v1/structured_output/__init__.py:L442-L486', 7.6,
        lc.C_FAINT, 'start', maxw=BW2 - 30, tag='lv:f')
# 死亡支（红）
DX = MX + BW2 + 20
lc.rect(DX, LY0, BW2, BH, '#fef2f2', lc.C_ABORT, rx=9, sw=1.8)
lc.text(DX + 16, LY0 + 22, '不 trim：accept([t1, t2, M, g1]) → 杀死请求', 10.5, lc.C_ABORT,
        'start', True, maxw=BW2 - 32, tag='die:t')
for j, ln in enumerate(['· 混块直喂：思考内容不合语法 → 真 xgrammar 拒收（False）',
                        '· update_from_output 判 FINISHED_ERROR · resumable=False',
                        '· 一步 4 token 的混块里 3 个是思考——接受率必为零']):
    lc.text(DX + 16, LY0 + 46 + j * 18, ln, 8.4, '#334155', 'start', maxw=BW2 - 30,
            tag='die:l' + str(j))
lc.text(DX + 16, LY0 + BH - 14, '#44006 修复前的真实死法（混块是这套门控存在的全部理由）', 7.6,
        lc.C_FAINT, 'start', maxw=BW2 - 30, tag='die:f')
# 衔接箭头：③ 下边 → 窗口对照框；窗口框底 → 生死双支（避开框内文字）
lc.seg(MX + 2 * (GW + GAP) + GW / 2, QY + QH, MX + 2 * (GW + GAP) + GW / 2, WY - 4,
       lc.C_MUTE, 1.6, 'std')
lc.seg(MX + BW2 / 2, WY + 190, MX + BW2 / 2, LY0 - 4, lc.C_GPU_S, 1.8, 'std')
lc.seg(DX + BW2 / 2, WY + 190, DX + BW2 / 2, LY0 - 4, lc.C_ABORT, 1.8, 'std')

# ================= 图例 + 页脚 =================
LY = LY0 + BH + 26
lx0 = MX
for f_, s_, name in [(lc.C_SAM_F, lc.C_SAM_S, '三问门控（结构化输出侧）'),
                     ('#fef9c3', '#ca8a04', '思考 token'),
                     ('#fed7aa', '#ea580c', '结束标记 M')]:
    lc.rect(lx0, LY - 9, 16, 11, f_, s_, rx=3, sw=1.3)
    lc.text(lx0 + 21, LY + 1, name, 8.8, lc.C_TXT, 'start', maxw=190, tag='lg:' + name[:5])
    lx0 += 21 + lc.tw(name, 8.8) + 14
lc.rect(lx0, LY - 10, 22, 13, 'none', lc.C_SAM_S, rx=3, sw=1.6)
lc.text(lx0 + 27, LY + 1, '精确窗口（v0.27 新）', 8.8, lc.C_TXT, 'start', maxw=190, tag='lg5')
lx0 += 27 + lc.tw('精确窗口（v0.27 新）', 8.8) + 14
lc.rect(lx0, LY - 10, 22, 13, 'none', lc.C_ABORT, rx=3, sw=1.4)
lc.text(lx0 + 27, LY + 1, '占位数推导窗口（旧，坑）', 8.8, lc.C_TXT, 'start', maxw=210, tag='lg6')

lc.text(MX, LY + 28, 'vllm/v1/structured_output/__init__.py:L361-L379（should_fill_bitmask）· L381-L439（should_advance·精确窗口）'
                     '· L442-L486（trim_reasoning_for_advance）——#43388 / #44006 修复物',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LY + 46, '窗口 start 3 vs 6 / 边界 idx 5 / trim 3 个 / 存活 status=None vs FINISHED_ERROR·resumable=False / prompt 级判定恰 1 次'
                     ' ＝ 本章驱动脚本实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-thinking-gates.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
