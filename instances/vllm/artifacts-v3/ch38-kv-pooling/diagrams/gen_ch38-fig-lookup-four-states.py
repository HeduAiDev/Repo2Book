#!/usr/bin/env python3
"""ch38 机制图 m7 · lookup-four-states（figure_spec ch38-fig-lookup-four-states，模板 state-machine）

放大自 L2 站 8（⑥ 查询·四态与收敛）· L0：新请求进 KV 边界时「问外部池」那一步。

claim：单块查询四态（HIT 在池可读 / HIT_PENDING 在池写入在飞 / RETRY 位置未定 / MISS
确定不在）之上，前缀查找叠加两条聚合规则：MISS 即 break（链式早停）、任一 defer 态
（HIT_PENDING/RETRY）让 get_num_new_matched_tokens 整体返回 None『稍后再问』。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · 四态枚举与语义（LookupResult docstring）
  · HIT,HIT,MISS,HIT → hit_count=2（第 4 块在池也不看）
  · HIT_PENDING → None（计数但 defer）；RETRY → None 且穿扫 [0,1,2] 到 MISS
  · SWA 从尾扫连续窗口：尾部命中 end_idx=5、中断 0
  · 多组收敛：full 16 token 被 SWA 收紧到 12
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 60
BXR = W - MX

ST_HIT_F, ST_HIT_S = '#cffafe', '#0e7490'          # HIT：在池可读（青·深）
ST_PEND_F, ST_PEND_S = '#ecfeff', '#0891b2'        # HIT_PENDING：在架不可读（青·浅）
ST_RET_F, ST_RET_S = '#fff7ed', '#ea580c'          # RETRY：未定（橙 = defer 家族）
ST_MIS_F, ST_MIS_S = '#f1f5f9', '#94a3b8'          # MISS：确定不在（灰）


def state_cell(x, y, w, h, name, fill, stroke, sub='', defer=False, dim=False):
    lc.rect(x, y, w, h, fill, stroke, rx=6, sw=1.4, dash=dim)
    fs = 10 if len(name) <= 6 else 8.2
    lc.text(x + w / 2, y + (h / 2 - 2 if sub else h / 2 + 3.5), name, fs,
            lc.C_FAINT if dim else lc.C_TXT, 'middle', True, maxw=w - 6, tag='st:' + name)
    if sub:
        lc.text(x + w / 2, y + h / 2 + 13, sub, 7.8, lc.C_MUTE, 'middle', maxw=w - 4,
                tag='sts:' + name)
    if defer:
        bw = 14 + 7.5 * len('defer')
        lc.rect(x + w - bw - 4, y - 9, bw, 16, ST_RET_F, ST_RET_S, rx=7, sw=1.0)
        lc.text(x + w - bw / 2 - 4, y + 3, 'defer', 7.5, ST_RET_S, 'middle', True,
                maxw=bw - 4, tag='dfr:' + name)


# ---------------- 标题区 ----------------
lc.text(MX, 36, '查池像翻一本四档评价的名册：HIT 记一笔、MISS 合册，defer 态让整趟「稍后再问」', 16.5,
        lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 60, '四态是信息完备度分级；前缀查找叠加两条聚合规则——MISS 即 break（链式早停），任一 defer 态让 get_num_new_matched_tokens 整体返回 None（接 ch16 skipped 队列）',
        10.5, lc.C_MUTE, 'start', maxw=1140, tag='subtitle')
_ch = '放大自 L2 站 8（⑥ 查询·四态与收敛）· L0：新请求进 KV 边界问池'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上：单块查询四态盒 ----------------
FB_Y = 96
lc.text(MX, FB_Y, 'manager.lookup 单块出口：四档', 11, lc.C_TXT, 'start', True,
        maxw=400, tag='four:t')
BOX_Y, BOX_H = FB_Y + 14, 96
BW4 = (BXR - MX - 3 * 20) / 4
four = [
    ('HIT', '在池 · 可读', 'hit_count +1，继续翻', ST_HIT_F, ST_HIT_S, False),
    ('HIT_PENDING', '在池 · 写入在飞', '+1 计数但置 defer 位', ST_PEND_F, ST_PEND_S, True),
    ('RETRY', '位置未定', '不计数、不 break、置 defer', ST_RET_F, ST_RET_S, True),
    ('MISS', '确定不在', 'break 合册（链式：MISS 后必 MISS）', ST_MIS_F, ST_MIS_S, False),
]
for i, (name, sem, eff, fill, stroke, defer) in enumerate(four):
    x = MX + i * (BW4 + 20)
    state_cell(x, BOX_Y, BW4, BOX_H, name, fill, stroke, sem, defer=defer)
    lc.text(x + BW4 / 2, BOX_Y + BOX_H + 16, eff, 8.6, '#334155', 'middle', maxw=BW4 + 16,
            tag='eff%d' % i)

# ---------------- 中：三条前缀扫描场景 ----------------
SC_Y = BOX_Y + BOX_H + 44
lc.text(MX, SC_Y, '前缀扫描三场景（键序列条 = 本图叙事主角）', 11, lc.C_TXT, 'start', True,
        maxw=460, tag='sc:t')
SCB_Y = SC_Y + 14
SC_W = (BXR - MX - 2 * 24) / 3
CELL_W, CELL_H = 66, 44
scenarios = [
    ('场景 A · MISS 早停', ['HIT', 'HIT', 'MISS', 'HIT'], [False, False, False, True],
     'hit_count = 2', ST_HIT_S, '第 4 块在池也不看——MISS 即合册', []),
    ('场景 B · HIT_PENDING', ['HIT', 'HIT_PENDING', 'MISS'], [False, False, False],
     'None（稍后再问）', lc.C_ABORT, '计数继续但 defer 位为真 → 整体返回 None', ['defer']),
    ('场景 C · RETRY 穿扫', ['RETRY', 'HIT', 'MISS'], [False, False, False],
     'None（稍后再问）', lc.C_ABORT, 'RETRY 不 break：穿扫 [0,1,2] 到 MISS 才停', ['defer']),
]
sty = {'HIT': (ST_HIT_F, ST_HIT_S), 'HIT_PENDING': (ST_PEND_F, ST_PEND_S),
       'RETRY': (ST_RET_F, ST_RET_S), 'MISS': (ST_MIS_F, ST_MIS_S)}
for i, (title, states, dims, result, rc, note, flags) in enumerate(scenarios):
    x0 = MX + i * (SC_W + 24)
    h = 152
    lc.rect(x0, SCB_Y, SC_W, h, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(x0 + 12, SCB_Y + 19, title, 9.8, lc.C_TXT, 'start', True, maxw=SC_W - 24,
            tag='sc%d:t' % i)
    cx0 = x0 + 12
    for j, st in enumerate(states):
        fill, stroke = sty[st]
        state_cell(cx0 + j * (CELL_W + 8), SCB_Y + 34, CELL_W, CELL_H, st, fill, stroke,
                   defer=(st in ('HIT_PENDING', 'RETRY')), dim=dims[j])
        lc.text(cx0 + j * (CELL_W + 8) + CELL_W / 2, SCB_Y + 34 + CELL_H + 14, 'k%d' % j, 8,
                lc.C_FAINT, 'middle', tag='k%d_%d' % (i, j))
    # 结果徽标
    rw = lc.tw(result, 9.5, True) + 20
    lc.rect(cx0, SCB_Y + 34 + CELL_H + 28, rw, 24, '#ffffff', rc, rx=9, sw=1.4)
    lc.text(cx0 + rw / 2, SCB_Y + 34 + CELL_H + 44, result, 9.5, rc, 'middle', True,
            maxw=rw - 4, tag='res%d' % i)
    lc.text(x0 + 12, SCB_Y + h - 10, note, 8.4, '#334155', 'start', maxw=SC_W - 22,
            tag='sc%d:n' % i)
SCB_H = 152

# ---------------- 下：SWA 尾扫 / 多组收敛 / None 三来源 ----------------
BT_Y = SCB_Y + SCB_H + 22
BW3 = (BXR - MX - 2 * 24) / 3
# 框 1：SWA 尾扫
x1 = MX
lc.rect(x1, BT_Y, BW3, 150, '#ffffff', lc.C_KV_S, rx=8, sw=1.3)
lc.text(x1 + 14, BT_Y + 20, 'SWA 组：从尾向前数连续窗口', 10, lc.C_KV_S, 'start', True,
        maxw=BW3 - 28, tag='swa:t')
cw6 = (BW3 - 28 - 5 * 4) / 6
for row, (states6, tagv, note) in enumerate([
        (['MISS', 'MISS', 'MISS', 'HIT', 'HIT', 'MISS'], 'a',
         '尾部连续 2 块 HIT → end_idx=5'),
        (['MISS', 'MISS', 'MISS', 'HIT', 'MISS', 'HIT'], 'b',
         '中断（尾 1 块后遇 MISS）→ 0')]):
    ry = BT_Y + 32 + row * 52
    for j, st in enumerate(states6):
        fill, stroke = sty[st]
        hl = (row == 0 and j in (3, 4))
        lc.rect(x1 + 14 + j * (cw6 + 4), ry, cw6, 22,
                '#bbf7d0' if hl else fill, stroke, rx=3, sw=1.2 if hl else 0.9)
    lc.text(x1 + 14, ry + 36, note, 8.3, '#334155', 'start', maxw=BW3 - 28,
            tag='swa:%s' % tagv)
lc.text(x1 + 14, BT_Y + 138, '窗口 = 2 chunk（SWA 注意力只需滑窗内的 KV）', 8, lc.C_MUTE,
        'start', maxw=BW3 - 28, tag='swa:note')
# 框 2：多组收敛
x2 = MX + BW3 + 24
lc.rect(x2, BT_Y, BW3, 150, '#ffffff', lc.C_ENG_S, rx=8, sw=1.3)
lc.text(x2 + 14, BT_Y + 20, '多组收敛：取最保守合并', 10, lc.C_ENG_S, 'start', True,
        maxw=BW3 - 28, tag='cv:t')
cv_lines = [
    '· full 组 4 chunk 全 HIT = 16 token',
    '· SWA 组尾 [MISS, HIT] → 只认 3 chunk',
    '· 收敛结果 = 12 token',
    '· full 组的长命中被 SWA 组收紧',
    '  （多组循环逐组收窄，谁短听谁的）',
]
for j, ln in enumerate(cv_lines):
    lc.text(x2 + 14, BT_Y + 42 + j * 16, ln, 8.6, '#334155', 'start', maxw=BW3 - 26,
            tag='cv:l%d' % j)
# 框 3：None 的三来源
x3 = MX + 2 * (BW3 + 24)
lc.rect(x3, BT_Y, BW3, 150, '#ffffff', lc.C_ABORT, rx=8, sw=1.3)
lc.text(x3 + 14, BT_Y + 20, '整体 None 的三个来源（都接 ch16 skipped）', 10, lc.C_ABORT,
        'start', True, maxw=BW3 - 28, tag='ns:t')
ns_lines = [
    '① defer 位：扫描中出现过 HIT_PENDING 或 RETRY',
    '② 在飞互斥：单请求 load/store 不可并发',
    '   （transfer_jobs={99} → (None, False) 实测）',
    '③ 命中块正被加载（_chunks_being_loaded）',
    '→ 池查询从不阻塞，只许改天再来',
]
for j, ln in enumerate(ns_lines):
    lc.text(x3 + 14, BT_Y + 42 + j * 16, ln, 8.6, '#334155', 'start', maxw=BW3 - 26,
            tag='ns:l%d' % j)

# defer 徽标图例
lc.text(MX, BT_Y - 8, 'defer 徽标 = 出现即整体 None 的态（HIT_PENDING / RETRY）；场景 B/C 结果即 None', 8.5,
        lc.C_MUTE, 'start', maxw=BXR - MX, tag='lg:defer')

# ---------------- 结论 + 页脚 ----------------
CONC_Y = BT_Y + 150 + 24
lc.text(MX, CONC_Y,
        '图注结论：四态是信息完备度分级，None 是 connector 层叠加的「这一趟没翻完」——池查询从不阻塞，只许改天再来。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 vllm/v1/kv_offload/base.py:L107-L113（LookupResult 四态 docstring）· offloading/scheduler.py:L816-L871（前缀查找与聚合规则）· L545-L610（SWA 尾扫）· L631-L802（多组收敛）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 场景状态序列为脚本化回放实测（CPU 单层池真实只产 HIT/HIT_PENDING/MISS；RETRY 真源在分层池）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-lookup-four-states.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
