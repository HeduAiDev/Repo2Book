#!/usr/bin/env python3
"""ch15 机制图 · 一拍调度调用全景（figure_spec ch15-fig-call-panorama，模板 swimlane·UML 时序）

放大自 L0 调度×显存账本列接缝——Scheduler 与 KVCacheManager/Coordinator/管家/BlockPool
的一次 tick 里「算·查·挂·写新块·写回」五段编排。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
右上角指北小签。

时序图按 FIGURE-SYSTEM §0 硬规则 2 的 UML 文法：参与者=竖直生命线（顶部名牌）、
共享时间轴（gridline 贯穿全部生命线、按事件 y 对齐）、消息=水平直线（跨中间生命线直穿，
禁一切折线/肘形）、瞬时动作=骑线标记+真实值标注。时间轴为调用序（非墙钟）——图内注明。

claim：算查挂写四操作在一次调度 tick 里的真实编排：查在准入那步（admission_lookup，
仅 num_computed_tokens==0），挂/写新块/写回满块全部发生在同一趟 allocate_slots 内依序执行；
哈希不在 tick 里算——它长在请求身上（构造+每拍 append 增量补算）。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑，17 个调用事件实拍）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印符号免疫

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440

# 五段操作语义色（本图图例口径；消息箭头/相位条/徽章同色）
V_ALG, V_CHK, V_TOUCH, V_NEW, V_WB = '#7c3aed', '#2563eb', '#16a34a', '#ea580c', '#0891b2'
C_RET = '#64748b'
TINT = {V_ALG: '#f5f3ff', V_CHK: '#eff6ff', V_TOUCH: '#f0fdf4', V_NEW: '#fff7ed', V_WB: '#ecfeff'}

# ---------------- 生命线（6 参与者） ----------------
LANES = [
    ('Request', 'vllm/v1/request.py', 310),
    ('Scheduler', 'sched/scheduler.py', 530),
    ('KVCacheManager', 'kv_cache_manager.py', 745),
    ('Coordinator·Unitary', 'kv_cache_coordinator.py', 950),
    ('管家 FullAttentionManager', 'single_type_kv_cache_manager.py', 1160),
    ('BlockPool（块池·平面哈希表）', 'block_pool.py', 1394),
]
X = {name.split('（')[0].split('·')[0]: x for name, _, x in LANES}
XR, XS, XA, XU, XF, XP = (X['Request'], X['Scheduler'], X['KVCacheManager'],
                          X['Coordinator'], X['管家 FullAttentionManager'], X['BlockPool'])

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一拍的全景：算·查·挂·写·写回——谁在何时调用谁', 17, lc.C_TXT, 'start', True,
        maxw=1050, tag='title')
lc.text(MX, 58, '查只在准入步（admission_lookup，num_computed_tokens==0 才查）；挂→写新块→写回恒在'
                '同一趟 allocate_slots 里依序；哈希不在 tick 里算——长在请求身上（构造时+每拍采样后增量补算）',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = 'L0 放大 · 调度 × 显存账本列 · 接缝全景'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# 前置事实（A 留表——B 的查表有东西可命中的前提）
lc.rect(MX, 76, 700, 40, '#f8fafc', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, 92, '前置：A（64 token·4 满块）先跑完已 free——4 条哈希留表（free 不清哈希）',
        8.8, lc.C_TXT, 'start', True, maxw=676, tag='pre:1')
lc.text(MX + 14, 108, 'B（80 token、前 32 与 A 相同）本拍被调度——撞的就是这 4 条',
        8.8, lc.C_MUTE, 'start', maxw=676, tag='pre:2')
lc.text(BXR, 100, '时间轴 = 调用序（自上而下，非墙钟）', 8.8, lc.C_MUTE, 'end', tag='axis:note')

# ---------------- 事件账本（traces 实拍 17 事件，序=日志序） ----------------
# kind: msg(水平消息 src→dst) / mark(骑线瞬时标记) / ret(虚线返回)
ROWS = [
    dict(b='0', ph=V_ALG, kind='mark', lane=XR, out=True,
         l1='算·入场前：B 构造 → update_block_hashes',
         l2='80 token → 5 枚满块哈希（A 当年构造时算 4 枚）',
         an='request.py:L208-L209'),
    dict(b='1', ph=V_CHK, kind='mark', lane=XS,
         l1='拍·admission_lookup（准入步）',
         l2='num_computed_tokens==0 才查；running 拍不查',
         an='scheduler.py:L744-L766'),
    dict(b='2', ph=V_CHK, kind='msg', s=XS, d=XA,
         l1='get_computed_blocks(request=b)',
         l2='预算 max_cache_hit_length=79（=80−1）',
         an='kv_cache_manager.py:L229-L295'),
    dict(b='3', ph=V_CHK, kind='msg', s=XA, d=XU,
         l1='Unitary.find_longest_cache_hit',
         l2='单组：coordinator 直接委托唯一管家',
         an='kv_cache_coordinator.py:L486-L504'),
    dict(b='4', ph=V_CHK, kind='msg', s=XU, d=XF,
         l1='FullAttentionManager.find_longest_cache_hit',
         l2='phase 1 沿链查 · miss 即断',
         an='single_type_kv_cache_manager.py:L682-L739'),
    dict(b='5', ph=V_CHK, kind='msg', s=XF, d=XP,
         l1='get_cached_block(hash0) → 命中块 1', l2='',
         an='block_pool.py:L198-L217'),
    dict(b='6', ph=V_CHK, kind='msg', s=XF, d=XP,
         l1='get_cached_block(hash1) → 命中块 2', l2='', an='block_pool.py:L198-L217'),
    dict(b='7', ph=V_CHK, kind='msg', s=XF, d=XP,
         l1='get_cached_block(hash2) → miss 即断',
         l2='查表 3 次 = 命中 2 + miss 1',
         an='block_pool.py:L198-L217'),
    dict(ph=C_RET, kind='ret', s=XF, d=XS,
         l1='return：hit=32 · 命中块 [1, 2]',
         l2='80−32=48 token 留给本拍分配'),
    dict(b='8', ph=V_TOUCH, kind='msg', s=XS, d=XA,
         l1='拍·schedule → allocate_slots',
         l2='num_new_tokens=48（80−32）',
         an='kv_cache_manager.py:L535-L563'),
    dict(b='9', ph=V_TOUCH, kind='msg', s=XA, d=XU,
         l1='allocate_new_computed_blocks',
         l2='num_local_computed_tokens=32',
         an='kv_cache_manager.py:L535-L540'),
    dict(b='10', ph=V_TOUCH, kind='msg', s=XU, d=XF,
         l1='add_local_computed_blocks',
         l2='new_computed_blocks=[1, 2]',
         an='single_type_kv_cache_manager.py:L232-L289'),
    dict(b='11', ph=V_TOUCH, kind='msg', s=XF, d=XP,
         l1='touch(blocks=[1, 2])',
         l2='ref_cnt 0→1 · O(1) 摘出自由队列',
         an='block_pool.py:L702-L717'),
    dict(b='12', ph=V_NEW, kind='msg', s=XA, d=XU,
         l1='allocate_new_blocks',
         l2='num_tokens=80（总槽位）',
         an='kv_cache_manager.py:L542-L547'),
    dict(b='13', ph=V_NEW, kind='msg', s=XU, d=XF,
         l1='manager.allocate_new_blocks',
         l2='80 token 需槽',
         an='single_type_kv_cache_manager.py:L330-L369'),
    dict(b='14', ph=V_NEW, kind='msg', s=XF, d=XP,
         l1='get_new_blocks(num_blocks=3)',
         l2='→ [5, 6, 7] · popleft 队头 · 惰性摘哈希',
         an='block_pool.py:L647-L661'),
    dict(b='15', ph=V_WB, kind='msg', s=XA, d=XU,
         l1='cache_blocks',
         l2='num_computed_tokens=80',
         an='kv_cache_manager.py:L559-L563'),
    dict(b='16', ph=V_WB, kind='msg', s=XU, d=XF,
         l1='manager.cache_blocks',
         l2='幂等闸：num_cached=2 < num_full=5 → 不短路',
         an='single_type_kv_cache_manager.py:L427-L477'),
    dict(b='17', ph=V_WB, kind='msg', s=XF, d=XP,
         l1='cache_full_blocks',
         l2='登记 [2,5) 3 块 · map 4→7 · 进度账推到 5',
         an='block_pool.py:L225-L342'),
    dict(ph=C_RET, kind='ret', s=XA, d=XS,
         l1='return：allocate_slots 完成',
         l2='B 块表 [1, 2, 5, 6, 7]（5 项）· 进度账=5'),
    dict(ph=V_ALG, kind='msg', s=XS, d=XR, out=True,
         l1='采样后（每拍）append_output_token_ids',
         l2='_update_request_with_output 里 · +1 token',
         an='scheduler.py:L2094-L2111'),
    dict(ph=V_ALG, kind='mark', lane=XR, out=True,
         l1='update_block_hashes：跨满块边界才补 1 枚',
         l2='本拍哈希零成本——80 token 的 5 枚构造时已算',
         an='request.py:L249-L265'),
]

Y0, STEP = 216, 64
for i, r in enumerate(ROWS):
    r['y'] = Y0 + i * STEP
Y_TICK0, Y_TICK1 = 1, 19            # tick 帧（①..ret2）
Y_FR0, Y_FR1 = 9, 19                # allocate_slots 组合片段（⑧..ret2）

# ---------------- 生命线名牌 + 生命线 ----------------
NP_Y, NP_H = 140, 40
for name, file_, x in LANES:
    lc.rect(x - 104, NP_Y, 208, NP_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
    lc.text(x, NP_Y + 17, name, 10.5, lc.C_TXT, 'middle', True, maxw=200, tag='np:' + name[:8])
    lc.text(x, NP_Y + 32, file_, 7.4, lc.C_FAINT, 'middle', maxw=200, tag='npf:' + file_[:10])
    y_bot = ROWS[-1]['y'] + 30
    lc.seg(x, NP_Y + NP_H, x, y_bot, '#cbd5e1', 1.2, dash=True)
# 显存账本侧委托链括注（KVCacheManager → Unitary → 管家）
_br_x0, _br_x1 = XA - 108, XF + 108
lc.seg(_br_x0, 128, _br_x1, 128, lc.C_MUTE, 1.0)
lc.seg(_br_x0, 128, _br_x0, 136, lc.C_MUTE, 1.0)
lc.seg(_br_x1, 128, _br_x1, 136, lc.C_MUTE, 1.0)
lc.text((XA + XF) / 2, 122, '显存账本侧（KVCacheManager → Unitary → 管家，单组一条委托链）',
        8.5, lc.C_MUTE, 'middle', tag='grp:br')

# ---------------- 相位条（左缘色带） ----------------
STRIP_X0, STRIP_X1 = 124, 240


def strip(rows, color, l1, l2, l3=''):
    y0 = ROWS[rows[0]]['y'] - 22
    y1 = ROWS[rows[-1]]['y'] + 22
    lc.rect(STRIP_X0, y0, STRIP_X1 - STRIP_X0, y1 - y0, TINT[color], color, rx=6, sw=1.2)
    cx = (STRIP_X0 + STRIP_X1) / 2
    yc = (y0 + y1) / 2
    lc.text(cx, yc - 8, l1, 13, color, 'middle', True, maxw=STRIP_X1 - STRIP_X0 - 8, tag='st:' + l1)
    if l2:
        lc.text(cx, yc + 8, l2, 8.6, color, 'middle', maxw=STRIP_X1 - STRIP_X0 - 8, tag='st2:' + l2)
    if l3:
        lc.text(cx, yc + 22, l3, 8, lc.C_MUTE, 'middle', maxw=STRIP_X1 - STRIP_X0 - 8, tag='st3:' + l3)


strip([0], V_ALG, '算', '构造时')
strip([1, 2, 3, 4, 5, 6, 7, 8], V_CHK, '查', '①-⑦', '准入步')
strip([9, 10, 11, 12], V_TOUCH, '挂', '⑧-⑪', 'touch')
strip([13, 14, 15], V_NEW, '写新块', '⑫-⑭')
strip([16, 17, 18], V_WB, '写回', '⑮-⑰')
strip([20, 21], V_ALG, '算', '每拍')

# ---------------- tick 帧 + allocate_slots 组合片段 ----------------
FT_X0, FT_X1 = 250, 1446
fy0, fy1 = ROWS[Y_TICK0]['y'] - 30, ROWS[Y_TICK1]['y'] + 22
lc.rect(FT_X0, fy0, FT_X1 - FT_X0, fy1 - fy0, 'none', lc.C_MUTE, rx=10, sw=1.5)
_ft = 'B 的一拍（调度 tick）· 17 个调用事件（①-⑰）'
_ftw = lc.tw(_ft, 10, True)
lc.rect(FT_X0 + 14, fy0 - 10, _ftw + 24, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1)
lc.text(FT_X0 + 26, fy0 + 3.5, _ft, 10, lc.C_TXT, 'start', True, maxw=_ftw + 8,
        tag='frame:tick')
FF_X0, FF_X1 = 298, 1442
ffy0, ffy1 = ROWS[Y_FR0]['y'] - 26, ROWS[Y_FR1]['y'] + 12
lc.rect(FF_X0, ffy0, FF_X1 - FF_X0, ffy1 - ffy0, 'none', C_RET, rx=8, sw=1.1, dash=True)
lc.rect(FF_X0, ffy0 - 9, 172, 18, '#ffffff', C_RET, rx=9, sw=1.1)
lc.text(FF_X0 + 86, ffy0 + 4, 'allocate_slots 同一趟', 8.8, C_RET, 'middle', True,
        maxw=164, tag='frame:frag')

# ---------------- 逐行绘制 ----------------
def mkey(c):
    return 'mk_' + c[1:]


MARKERS = ''.join(
    f'<marker id="mk_{c[1:]}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{c}"/></marker>'
    for c in (V_ALG, V_CHK, V_TOUCH, V_NEW, V_WB, C_RET))


def tick(x, y, color):
    """生命线上的消息触点（白底色边小竖条）——箭头端点落它的边。"""
    lc.rect(x - 2.5, y - 8, 5, 16, '#ffffff', color, rx=1.5, sw=1.0)


for i, r in enumerate(ROWS):
    y = r['y']
    # gridline（贯穿全部生命线；tick 外虚线）
    gx0, gx1 = (306, 1430) if Y_FR0 <= i <= Y_FR1 else (292, 1436)
    lc.seg(gx0, y, gx1, y, '#eef2f7', 1.0, dash=bool(r.get('out')))
    # 徽章
    if r.get('b'):
        lc.rect(258, y - 9, 26, 18, TINT[r['ph']], r['ph'], rx=9, sw=1.1)
        lc.text(271, y + 3.5, r['b'], 9.5, r['ph'], 'middle', True, maxw=20, tag='bg' + r['b'])
    if r['kind'] in ('msg', 'ret'):
        sx, dx = r['s'], r['d']
        color, dash = (C_RET, True) if r['kind'] == 'ret' else (r['ph'], False)
        tick(sx, y, color)
        tick(dx, y, color)
        x1 = sx + (2.5 if dx > sx else -2.5)
        x2 = dx - (2.5 if dx > sx else -2.5)
        lc.seg(x1, y, x2, y, color, 1.8, mkey(color), dash=dash)
        mx = (sx + dx) / 2
        room = min(mx - 300, 1438 - mx) * 2 - 8
        lc.text(mx, y - 17, r['l1'], 9.3, lc.C_TXT, 'middle', True, maxw=room, tag='l1' + r['l1'][:10])
        if r['l2']:
            lc.text(mx, y - 6, r['l2'], 8.4, '#334155', 'middle', maxw=room, tag='l2' + r['l2'][:10])
        if r.get('an'):
            if dx > sx:   # 右行消息：锚收在目标生命线左内侧
                lc.text(dx - 8, y + 13, r['an'], 7.2, lc.C_FAINT, 'end', maxw=260,
                        tag='an' + r.get('b', 'x'))
            else:         # 左行消息：锚放源生命线右侧，避开左缘相位条
                lc.text(sx + 10, y + 13, r['an'], 7.2, lc.C_FAINT, 'start', maxw=260,
                        tag='an' + r.get('b', 'x'))
    else:  # mark：骑线瞬时标记
        lx = r['lane']
        lc.rect(lx - 6, y - 6, 12, 12, r['ph'], r['ph'], rx=2.5, sw=1.0)
        lc.text(lx + 16, y - 6, r['l1'], 9.3, lc.C_TXT, 'start', True, maxw=660, tag='mk1' + r['l1'][:10])
        lc.text(lx + 16, y + 8, r['l2'], 8.4, '#334155', 'start', maxw=660, tag='mk2' + r['l2'][:10])
        if r.get('an'):
            lc.text(lx + 16, y + 22, r['an'], 7.2, lc.C_FAINT, 'start', maxw=300,
                    tag='mkan' + r.get('b', 'x'))

# ---------------- 本拍小结（采样后两行之后，生命线到 ROWS[-1]+24 为止） ----------------
SY = ROWS[-1]['y'] + 30
lc.rect(MX, SY, BXR - MX, 54, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, SY + 21, '本拍账：17 个调用事件 = 查 7（①-⑦）+ 挂 4（⑧-⑪）+ 写新块 3（⑫-⑭）+ 写回 3（⑮-⑰）'
        '——查的调用深度最深（5 层下潜）但每层都 O(1)；写回的新登记量 = 新满块数 3',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='sum:1')
lc.text(MX + 16, SY + 41, '查表 3 次（2 命中 + 1 miss）· hit=32 · touch 2 块 [1,2] · 新块 3 个 [5,6,7]（48 token）'
        '· 登记 [2,5) 3 块、map 4→7 · 哈希零成本（5 枚构造时已算）',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='sum:2')

# ---------------- 图例 ----------------
LY = SY + 78
lx = MX
for c, name in [(V_ALG, '算（哈希·请求侧）'), (V_CHK, '查（命中）'), (V_TOUCH, '挂（touch）'),
                (V_NEW, '写（新块）'), (V_WB, '写回（满块登记）')]:
    lc.rect(lx, LY - 9, 20, 13, TINT[c], c, rx=3, sw=1.2)
    lc.text(lx + 26, LY + 1, name, 8.8, lc.C_TXT, 'start', maxw=170, tag='lg' + name[:6])
    lx += 26 + lc.tw(name, 8.8) + 20
lc.seg(lx + 4, LY - 3, lx + 34, LY - 3, C_RET, 1.6, 'mk_64748b', dash=True)
lc.text(lx + 40, LY + 1, '虚线 = 返回值 / tick 外（构造前·采样后）', 8.8, lc.C_TXT, 'start',
        maxw=280, tag='lg:ret')
lx += 40 + lc.tw('虚线 = 返回值 / tick 外（构造前·采样后）', 8.8) + 20
lc.text(lx, LY + 1, '消息箭头两端的小竖条 = 生命线上的调用触点', 8.8, lc.C_MUTE, 'start',
        maxw=BXR - lx, tag='lg:note')

# ---------------- 页脚 ----------------
FY = LY + 26
lc.text(MX, FY, '读图：⓪ 构造算哈希在 tick 外；①-⑦ 准入查——5 层下潜、平面 dict 查 3 次即断；'
                '⑧-⑰ 同一次 allocate_slots 里挂→写新块→写回依序（写回在最后：新块先到手、才登记满块）；'
                '采样后哈希增量补算（虚线回 Request）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot:read')
lc.text(MX, FY + 16, '逐字锚 vllm/v1/request.py:L208-L209/L249-L265 · v1/core/sched/scheduler.py:L744-L766/L2094-L2111 · '
        'v1/core/kv_cache_manager.py:L229-L295/L535-L563 · v1/core/kv_cache_coordinator.py:L486-L504/L652-L683',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot:anchor1')
lc.text(MX, FY + 32, 'v1/core/single_type_kv_cache_manager.py:L682-L739/L232-L289/L330-L369/L427-L477 · '
        'v1/core/block_pool.py:L198-L217/L647-L661/L702-L717/L225-L342',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot:anchor2')
lc.text(MX, FY + 48, '数字取自配套精简版 host 实跑（B=80 token、前 32 与 A 相同；17 个调用事件实拍；'
        'block_size=hash_block_size=16；观察口径=方法包装器只记录调用与实参、不改行为）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot:prov')

# ---------------- 装配输出 ----------------
H = FY + 64
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>',
       '<defs>' + MARKERS + '</defs>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-call-panorama.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
