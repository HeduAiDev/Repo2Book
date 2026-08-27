#!/usr/bin/env python3
"""ch13 机制图 4 · allocate_slots 三段式（figure_spec ch13-fig-allocate-slots-three-stages，模板 flow）

放大自 L0『调度 · 显存账本』列（kv_column）中 KVCacheManager 框的 allocate_slots
本体——即本章 L2 章图中排 ①『入场 allocate_slots』→②『数块』→③『拿块挂账』
三拍片的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：三段式与两个出口：五次调用里 7≤9 过 / 8>2 None / 1≤2 过 / 1≤1 过 /
1>0 None——容量检查在一切记账之前，None 意味着零半截账；RUNNING 侧的 None 就是
ch11 抢占信号的内因。

数字全部取自 figure_spec.numbers（配套精简版 host 实测：五调用判定线、
None 无半截账物证、分到的块 [1..7]/[8]/[9]）。
坐标由常量/循环计算；文本全 esc()。
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
lc.text(MX, 34, 'allocate_slots 三段式：容量检查在一切记账之前——None 即零半截账',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '第一段先算账（预测需块 vs 空闲），不够当场 return None——检查失败则后面的挂命中 / 分新块 / 写回一个都没执行（构造性无部分状态，非事后回滚）',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L2 章图中排 ①→③ 拍片 · L0：显存账本列 KVCacheManager'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：四段管道 ----------------
PX, PW_ = MX, 560
STAGES = [
    ('entry', '入口 · allocate_slots(req, num_new_tokens)', 54,
     'WAITING 入场（scheduler.py:L973-L985）/ RUNNING 长大（L576-L629）', lc.C_KV_S, lc.C_KV_F, False),
    ('s1', '段① 容量检查（get_num_blocks_to_allocate）', 64,
     '预测需块 vs 空闲：不够 → return None（唯一出口 L510-L527）', lc.C_KV_S, lc.C_KV_F, True),
    ('s2', '段② 挂命中块（touch 前缀命中）', 46,
     '本章缓存关：判同短路、命中恒空 → ch15', '#94a3b8', '#f8fafc', False),
    ('s3', '段③ 分新块（popleft_n）', 56,
     '取块 · ref_cnt=1 · 逻辑块表加长 · 记 new_block_ids（清零账）', lc.C_KV_S, lc.C_KV_F, True),
    ('s4', '段④ 写回满块（cache_blocks 哈希登记）', 46,
     'not enable_caching → L551-L552 早退 → ch15', '#94a3b8', '#f8fafc', False),
    ('exit', '出口 · KVCacheBlocks（分到的块）', 42,
     '附在 SchedulerOutput 上过线（block_id 桥另图展开）', lc.C_KV_S, lc.C_KV_F, False),
]
SY, SGAP = 100, 26
ypos = {}
yy = SY
for key, title, h, note, stroke, fill, hot in STAGES:
    sw = 2.0 if hot else 1.3
    lc.rect(PX, yy, PW_, h, fill, stroke, rx=7, sw=sw)
    if not hot:
        lc.rect(PX, yy, PW_, h, 'none', stroke, rx=7, sw=sw, dash=True)
    lc.text(PX + 16, yy + 20, title, 10.5, stroke if hot else '#64748b', 'start', True,
            maxw=PW_ - 32, tag='st:' + key)
    if h > 48:
        lc.text(PX + 16, yy + 39, note, 8.5, '#64748b', 'start', maxw=PW_ - 32, tag='sn:' + key)
    else:
        lc.text(PX + 16, yy + 35, note, 8.5, '#64748b', 'start', maxw=PW_ - 32, tag='sn:' + key)
    ypos[key] = (yy, h)
    if key != 'exit':
        lc.seg(PX + PW_ / 2, yy + h + 2, PX + PW_ / 2, yy + h + SGAP - 3, lc.C_KV_S, 2.0, 'std')
    yy += h + SGAP
PIPE_BOT = yy - SGAP

# 段① 菱形语义：右侧红出口（容量不够）
s1y, s1h = ypos['s1']
NB_X, NB_W = PX + PW_ + 130, BXR - (PX + PW_ + 130)
lc.seg(PX + PW_ + 2, s1y + s1h / 2, NB_X - 3, s1y + s1h / 2, lc.C_ABORT, 2.0, 'ab')
lc.text(PX + PW_ + 10, s1y + s1h / 2 - 10, '不够', 8.5, lc.C_ABORT, 'start', maxw=50, tag='lbl:fail')
lc.text(PX + PW_ / 2 + 12, s1y + s1h + 18, '够 → 往下走', 8.5, lc.C_KV_S, 'start', maxw=90, tag='lbl:pass')
lc.rect(NB_X, s1y - 8, NB_W, 78, '#fef2f2', lc.C_ABORT, rx=7, sw=1.6)
lc.text(NB_X + 14, s1y + 12, 'return None · 零半截账', 11, lc.C_ABORT, 'start', True,
        maxw=NB_W - 28, tag='nb:t')
lc.text(NB_X + 14, s1y + 32, '逻辑块表不添行、空闲计数不动——被拒者完整留在原地等下一拍',
        8.5, '#64748b', 'start', maxw=NB_W - 28, tag='nb:1')
lc.text(NB_X + 14, s1y + 50, '物证：调用 2 后 r2 不在 req_to_blocks、空闲仍 2', 8.5, '#64748b',
        'start', maxw=NB_W - 28, tag='nb:2')
# None 的两个下游（回指 ch10 / ch11——章号均小于本章）
DW_Y = s1y + 88
lc.seg(NB_X + 40, s1y + 70, NB_X + 40, DW_Y - 2, lc.C_ABORT, 1.4, 'std')
for i, (tag, txt) in enumerate([('w', 'WAITING 侧听到 None → break，下一轮再来（ch10 的「拿不到块 break」）'),
                                ('r', 'RUNNING 侧听到 None → while True 抢占环（ch11 抢占信号的内因）')]):
    by = DW_Y + i * 46
    lc.rect(NB_X, by, NB_W, 38, '#ffffff', lc.C_ABORT, rx=6, sw=1.1, dash=True)
    lc.text(NB_X + 12, by + 16, tag.upper() + ' · None', 9, lc.C_ABORT, 'start', True,
            maxw=100, tag='dw%s' % tag)
    lc.text(NB_X + 12, by + 31, txt, 8, '#64748b', 'start', maxw=NB_W - 24, tag='dwt%s' % tag)

# ---------------- 右下：五调用实录表（管道走完后再起，全宽） ----------------
TB_Y = PIPE_BOT + 28
COLS = [('调用', 46), ('侧', 150), ('目标', 46), ('预测', 42), ('空闲', 42), ('判定', 132), ('分到的块', 150), ('拍后空闲', 56)]
TX0 = PX
rowh = 30
tw_ = sum(c[1] for c in COLS) + (len(COLS) - 1) * 8
TB_W = tw_ + 20
lc.rect(TX0, TB_Y, TB_W, rowh, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.2)
cx = TX0 + 10
for name, cwid in COLS:
    lc.text(cx + cwid / 2, TB_Y + 20, name, 9, lc.C_KV_S, 'middle', True, maxw=cwid, tag='th' + name)
    cx += cwid + 8
CALLS = [
    ('1', 'WAITING r1 入场', '100', '7', '9', '7 ≤ 9 → 过', '[1,2,3,4,5,6,7]', '2', False),
    ('2', 'WAITING r2 入场', '128', '8', '2', '8 > 2 → None', '无（零半截账）', '2', True),
    ('3', 'RUNNING r1 长大', '116', '1', '2', '1 ≤ 2 → 过', '[8]', '1', False),
    ('4', 'WAITING r3 入场', '16', '1', '1', '1 ≤ 1 → 过', '[9]', '0', False),
    ('5', 'RUNNING r1 长大', '132', '1', '0', '1 > 0 → None', '无（抢占信号）', '0', True),
]
for ri, row in enumerate(CALLS):
    ry = TB_Y + rowh + 8 + ri * (rowh + 6)
    fillc = '#fef2f2' if row[-1] else ('#ffffff' if ri % 2 == 0 else '#f8fafc')
    lc.rect(TX0, ry, TB_W, rowh, fillc, '#e2e8f0' if not row[-1] else lc.C_ABORT, rx=3, sw=0.9)
    cx = TX0 + 10
    for (name, cwid), val in zip(COLS, row[:-1]):
        col = lc.C_ABORT if (row[-1] and name in ('判定', '分到的块')) else '#334155'
        lc.text(cx + cwid / 2, ry + 20, val, 9, col, 'middle', name in ('判定',), maxw=cwid,
                tag='td%d%s' % (ri, name))
        cx += cwid + 8
TB_BOT = TB_Y + rowh + 8 + 5 * (rowh + 6)
lc.text(TX0, TB_BOT + 16, '「目标」= num_computed_tokens + num_new_tokens 合计（本拍要落位的 token 总数）· '
        '末次 None 发生在 RUNNING 侧——那句 None 就是抢占环的启动信号',
        8.5, lc.C_MUTE, 'start', maxw=BXR - TX0, tag='tb:note')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = TB_BOT + 38
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
lc.text(lx + 26, LEG_Y + 1, '本章活着的段（①③）', 8.5, lc.C_TXT, 'start', maxw=150, tag='lg1')
lx += 26 + lc.tw('本章活着的段（①③）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 13, '#f8fafc', '#94a3b8', rx=3, sw=1.0, dash=True)
lc.text(lx + 26, LEG_Y + 1, '缓存关而早退的段（②④ → ch15）', 8.5, lc.C_TXT, 'start', maxw=210, tag='lg2')
lx += 26 + lc.tw('缓存关而早退的段（②④ → ch15）', 8.5) + 18
lc.rect(lx, LEG_Y - 11, 20, 15, '#fef2f2', lc.C_ABORT, rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, 'None / 被拒调用', 8.5, lc.C_TXT, 'start', maxw=130, tag='lg3')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/core/kv_cache_manager.py:L344-L565（allocate_slots 本体）· '
        'L510-L527（容量检查与 None 唯一出口）· L529-L540（段② 挂命中）· L542-L547（段③ 分新块）· '
        'L551-L552（段④ 写回早退）', 8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, 'vllm/v1/core/sched/scheduler.py:L973-L985（WAITING 侧）/ L576-L629（RUNNING 侧）· '
        '五调用判定与分块数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 66
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-allocate-slots-three-stages.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
