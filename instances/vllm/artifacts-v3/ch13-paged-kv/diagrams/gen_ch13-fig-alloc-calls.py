#!/usr/bin/env python3
"""ch13 机制图 · allocate_slots 调用全景（读者反馈驱动新图，非 explainer figure_specs 铺底）

放大自 L0 调度 × 显存账本列接缝（架构归属回指 L0/L2，右上角指北小签）。

claim：一次 allocate_slots = 自上而下五层调用（Scheduler.schedule → KVCacheManager →
KVCacheCoordinator → SingleTypeKVCacheManager → BlockPool）；需块预测在管家里分快慢两路
（记账位在=一句差值钳零 / 不在=六行账），两路算出的数永远一致；数不够 return None 不取块，
够了才依序挂账（touch）与取块（get_new_blocks）。写回满块是第 15 章的地盘——本图用虚线框
预告，并点出它正是 fast-path 开关的武装点（首尾扣环）。

坐标全部由常量/游标计算，零手写魔数；文本全走 lc.text（内部 esc()）。
数字（行号、公式项）逐个取自 pin vllm v0.27.1 与本章 dossier supplement.A。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印符号免疫

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与骨架常量 ----------------
W = 1500
MX, BXR = 60, 1440
CW = BXR - MX                      # 1380
COL_GAP = 60                       # 左右双栏中缝
HALF_W = (CW - COL_GAP) / 2        # 660
LX, RX = MX, MX + HALF_W + COL_GAP  # 60 / 780
LCX, RCX = LX + HALF_W / 2, RX + HALF_W / 2   # 390 / 1110
MIDX = MX + CW / 2                 # 750 全宽框的中轴
N5_W = CW - 300                    # 会合框（右侧留 return None 出口）
N5_CX = MX + N5_W / 2              # 600
NONE_X, NONE_W = MX + N5_W + 30, BXR - (MX + N5_W + 30)   # 1170 / 270

# 语义色（l0_common：调度=橙 C_ENG_S / 显存账本=青 C_KV_S；本图纯用共享常量）
C_EN, C_EN_F = lc.C_ENG_S, lc.C_ENG_F
C_KV, C_KV_F = lc.C_KV_S, lc.C_KV_F
C_RED = lc.C_ABORT
C_GREY = lc.C_MUTE
C_FILL_GREY = '#f8fafc'

EXTRA_DEFS = ('<defs>'
              f'<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV}"/></marker>'
              f'<marker id="gry" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GREY}"/></marker>'
              '</defs>')

MK = {C_EN: 'up', C_KV: 'kv', C_GREY: 'gry', C_RED: 'ab', lc.C_MUTE: 'std'}


# ---------------- 文本换行（CJK 逐字、ASCII 按词；禁手写断行） ----------------
def _sep(a, b):
    if not a:
        return ''
    return '' if (ord(a[-1]) > 0x2E80 or ord(b[0]) > 0x2E80) else ' '


def wrap(s, fs, maxw, bold=False):
    if lc.tw(s, fs, bold) <= maxw:
        return [s]
    lines, cur = [], ''
    for word in s.split(' '):
        cand = cur + _sep(cur, word) + word
        if lc.tw(cand, fs, bold) <= maxw:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        while lc.tw(word, fs, bold) > maxw:            # 单词本身超宽 → 逐字切
            cut = len(word)
            while cut > 1 and lc.tw(word[:cut], fs, bold) > maxw:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        cur = word
    if cur:
        lines.append(cur)
    return lines


def rrow(t, fs=9.2, c=lc.C_TXT, bold=False, ind=0, lead=17, gap=2, tag='', tagc=None):
    """节点内一行（可自动换行）。tag = 右端贴边的行号签。"""
    return dict(t=t, fs=fs, c=c, bold=bold, ind=ind, lead=lead, gap=gap,
                tag=tag, tagc=tagc or lc.C_FAINT)


def nbox(x, y, w, title, tfile, rows, stroke, fill='#ffffff', badge=None,
         dash=False, tfs=11.5, pad=10):
    """调用链节点框：标题（粗）+ 源码路径（灰）+ 内容行（自动换行、行高可调）。
    返回框底 y。"""
    inner = w - 28
    laid = []                                    # [(text, fs, color, bold, ind, lead, tag, tagc)]
    for r in rows:
        body = inner - r['ind'] - (lc.tw(r['tag'], 8.0) + 12 if r['tag'] else 0)
        for i, ln in enumerate(wrap(r['t'], r['fs'], body, r['bold'])):
            laid.append((ln, r['fs'], r['c'], r['bold'], r['ind'],
                         r['lead'], r['tag'] if i == 0 else '', r['tagc']))
    h = (30 if tfile else 22) + sum(r[5] for r in laid) + pad + 4
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.5, dash=dash)
    tx = x + 14
    lc.text(tx, y + 21, title, tfs, lc.C_TXT, 'start', True,
            maxw=inner - (26 + 11 * len(badge) if badge else 0), tag='t:' + title[:14])
    if badge:
        bw = 16 + 11 * len(badge)
        lc.rect(x + w - bw - 10, y + 6, bw, 20, lc.C_BADGE_F, stroke, rx=9, sw=1.1)
        lc.text(x + w - bw / 2 - 10, y + 20, badge, 9.5, stroke, 'middle', True, tag='bg:' + badge)
    if tfile:
        lc.text(tx, y + 37, tfile, 8.6, lc.C_FAINT, 'start', maxw=inner, tag='f:' + tfile[:16])
    cy = y + (30 if tfile else 22)
    for ln, fs, c, bold, ind, lead, tag, tagc in laid:
        cy += lead
        lc.text(tx + ind, cy, ln, fs, c, 'start', bold,
                maxw=inner - ind - (60 if tag else 0), tag='b:' + ln[:18])
        if tag:
            lc.text(x + w - 14, cy, tag, 8.0, tagc, 'end', tag='tag:' + tag)
    return y + h


def vlabel(x, y, s, fs=8.8, c=None, maxw=None, tag='vl'):
    """竖线旁的调用表达式标注（箭头中段）。"""
    lc.text(x + 8, y, s, fs, c or lc.C_TXT, 'start', maxw=maxw or (BXR - x - 8),
            tag=tag + s[:14])


def drop(x, y0, y1, color, sw=2.2, dash=False):
    lc.seg(x, y0, x, y1, color, sw, 'gry' if dash else MK[color], dash=dash)


def fork(x0, y0, targets, color, sw=2.2, bus_gap=26, drop_gap=26):
    """分岔：中轴下潜 → 横向母线 → 各目标竖落。targets=[x…]，返回母线 y。"""
    bus = y0 + bus_gap
    lc.seg(x0, y0, x0, bus, color, sw)
    lc.seg(min(targets + [x0]), bus, max(targets + [x0]), bus, color, sw)
    for t in targets:
        lc.seg(t, bus, t, bus + drop_gap, color, sw, MK[color])
    return bus + drop_gap


def merge(xs, y0s, x_in, y_in, color, sw=2.0, bus_gap=26):
    """汇聚：各源竖落 → 横向母线 → 中轴落到下一框顶。返回母线 y。"""
    bus = max(y0s) + bus_gap
    for x, y in zip(xs, y0s):
        lc.seg(x, y, x, bus, color, sw)
    lc.seg(min(xs + [x_in]), bus, max(xs + [x_in]), bus, color, sw)
    lc.seg(x_in, bus, x_in, y_in, color, sw, MK[color])
    return bus


# ================= 标题区 =================
lc.text(MX, 34, 'allocate_slots 调用全景：从 Scheduler.schedule 到块池，谁调用谁',
        17, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '一拍一次下潜五层：调度器 → 显存账本 → 协调器 → 管家 → 块池——每层各管一件事，'
                '拿到数才往下走；需块预测在管家里分快慢两路，两路算出的数永远一致',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='sub1')
lc.text(MX, 74, '主干粗线：橙 = 调度侧（入口与快路）｜青 = 显存账本（慢路与块账）｜灰虚线 = 写回侧，'
                '第 15 章前缀缓存详解（预告）｜红 = 不够时 return None 的出口',
        9.2, lc.C_MUTE, 'start', maxw=1330, tag='sub2')
_ch = '放大自 L0 · 显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

y = 96
GAP = 46                                    # 节点之间的箭头段高

# ================= ① 入口（橙） =================
n1 = nbox(MX, y, CW,
          'Scheduler.schedule() —— 一拍一次，逐请求调用',
          'vllm/v1/core/sched/scheduler.py:L439',
          [rrow('RUNNING 长大（L578）与 WAITING 入场（L973）各调一次 allocate_slots；'
                '返回 None 就地回退——不排这一步，不留半截账', 9.2, '#334155'),
           rrow('入场首拍才查前缀缓存：if request.num_computed_tokens == 0（L744-L747）'
                '——running 的每一拍都不查（前缀查找的时机归第 15 章，预告）', 8.8, lc.C_MUTE)],
          C_EN, C_EN_F, badge='①')
y_prev = n1
drop(MIDX, y_prev, y_prev + GAP, C_EN)
vlabel(MIDX, y_prev + GAP * 0.62,
       'self.kv_cache_manager.allocate_slots(request, num_new_tokens, …)', 8.8, C_EN)
y = y_prev + GAP

# ================= ② KVCacheManager（青） =================
n2 = nbox(MX, y, CW,
          'KVCacheManager.allocate_slots(request, num_new_tokens, …)',
          'vllm/v1/core/kv_cache_manager.py:L344-L565',
          [rrow('不直接摸块池：把请求按「注意力组」交给协调器（self.coordinator）'
                '——组是什么、为什么分，下一章显存账本（预告）', 9.2, '#334155'),
           rrow('① 预算预测（L510-L519） → ② 对照空闲、不够 return None（L523-L527）'
                ' → ③ 依序派发：先挂命中块（L535）、再取新块（L542）', 9.2, '#334155'),
           rrow('收尾写回满块（L563）——本图在最下方虚线框交代，详解归第 15 章（预告）',
                8.8, lc.C_MUTE)],
          C_KV, C_KV_F, badge='②')
y_prev = n2
drop(MIDX, y_prev, y_prev + GAP, C_KV)
vlabel(MIDX, y_prev + GAP * 0.40,
       'self.coordinator.get_num_blocks_to_allocate(request_id=…, num_tokens=num_tokens_need_slot, …)',
       8.8, C_KV)
vlabel(MIDX, y_prev + GAP * 0.82,
       'num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, '
       'self.max_model_len)（L490-L493）——往下传的是 token 数', 8.0, C_GREY)
y = y_prev + GAP

# ================= ③ 协调器（青） =================
n3 = nbox(MX, y, CW,
          'KVCacheCoordinator.get_num_blocks_to_allocate(…) —— 分发到各组管家',
          'vllm/v1/core/kv_cache_coordinator.py:L130-L190',
          [rrow('逐组问数再求和：num_blocks_to_allocate = 0（L166）→ for i, manager in '
                'enumerate(self.single_type_managers):（L167）→ 每组累加一次 '
                'manager.get_num_blocks_to_allocate(…)（常规组 L181、十字注意力组 L171）→ '
                'return num_blocks_to_allocate（L190）', 8.8, '#334155'),
           rrow('本章单组全注意力 → 只有一位管家，循环走一遍（近似直通：协调器分发的是「谁办」，'
                '不是「办几块」）', 9.2, '#334155'),
           rrow('多组混合时各组独立算、数相加——组的账归下一章显存账本（预告）', 8.8, lc.C_MUTE)],
          C_KV, C_KV_F, badge='③')
y_prev = n3
drop(MIDX, y_prev, y_prev + GAP, C_KV)
vlabel(MIDX, y_prev + GAP * 0.62, 'manager.get_num_blocks_to_allocate(…)——逐组传入', 8.8, C_KV)
y = y_prev + GAP

# ================= ④ 管家（青）· 分岔口 =================
n4 = nbox(MX, y, CW,
          'SingleTypeKVCacheManager.get_num_blocks_to_allocate(…) —— 分岔口',
          'vllm/v1/core/single_type_kv_cache_manager.py:L144-L230',
          [rrow('先算目标表长：num_required_blocks = cdiv(num_tokens, self.block_size)（L178）'
                '——这条两条路都走', 9.2, '#334155'),
           rrow('分岔只认一个开关：if request_id in self.num_cached_block:（L194）'
                '——记账位在 = 本次入场已有整块落袋（挂过命中块或写过满块）', 9.2, '#334155')],
          C_KV, C_KV_F, badge='④')
y_prev = n4
y_br = fork(MIDX, y_prev, [LCX, RCX], C_KV, sw=2.4)

# ================= ⑤L 快路（橙） / ⑤R 慢路（青） =================
fast = nbox(LX, y_br, HALF_W,
            '快路 fast-path（记账位在）',
            '',
            [rrow('assert len(new_computed_blocks) == 0（L196）', 8.8, lc.C_TXT, bold=True),
             rrow('return max(num_required_blocks - num_req_blocks, 0)（L200）',
                  8.8, lc.C_TXT, bold=True),
             rrow('一句差值、负数钳零，走人。max 是给投机解码兜底：草稿 token 被拒后目标回缩，'
                  '需块可能反而小于已持（L197-L199 注释原话）', 8.6, '#334155'),
             rrow('记账位谁写：挂命中块（L282）与写回满块（L477）；请求退场就删（L515）',
                  8.6, lc.C_MUTE),
             rrow('谁没有它：关缓存部署（写回整段被跳过，kv_cache_manager.py:L551-L552）'
                  '与攒不满一个整块的短请求——两条路算出的数永远一样', 8.6, lc.C_MUTE)],
            C_EN, C_EN_F, badge='⑤', tfs=11)

slow = nbox(RX, y_br, HALF_W,
            '慢路（记账位不在）· 完整六行账',
            '',
            [rrow('num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)',
                  8.5, lc.C_TXT, lead=16, tag='L202'),
             rrow('num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks',
                  8.5, lc.C_TXT, lead=16, tag='L203'),
             rrow('num_skipped_blocks = num_skipped_tokens // self.block_size',
                  8.5, lc.C_TXT, lead=16, tag='L206'),
             rrow('num_new_blocks = max(num_required_blocks - max(num_skipped_blocks, '
                  'num_local_computed_blocks), 0)', 8.5, lc.C_TXT, lead=16, tag='L210-L213'),
             rrow('num_evictable_blocks = self._get_num_evictable_blocks(new_computed_blocks[…])',
                  8.5, lc.C_TXT, lead=16, tag='L223-L225'),
             rrow('if self._has_partial_local_hit(…): num_new_blocks += 1',
                  8.5, lc.C_TXT, lead=16, tag='L226-L229'),
             rrow('return num_new_blocks + num_evictable_blocks',
                  8.5, lc.C_TXT, bold=True, lead=16, tag='L230'),
             rrow('返回值 = 本拍将从自由队列摘走多少块：新分配的 + CoW 换的 + 被 touch 摘走的'
                  '可驱逐命中块（ref_cnt==0 的命中块既是复用、也离开自由队列，漏数容量检查就失真）',
                  8.5, '#334155'),
             rrow('两个取整各管一摊：cdiv 管容量（目标表长）、floor 管跳段'
                  '（跨窗块在窗内的一角还得留实体，只有整块全在窗外才换 null 占位）',
                  8.5, lc.C_MUTE)],
            C_KV, C_KV_F, badge='⑤', tfs=11)
y_br_end = max(fast, slow)
bus = merge([LCX, RCX], [fast, slow], N5_CX, y_br_end + GAP, C_GREY, sw=1.8)
lc.text((LCX + RCX) / 2, bus - 7, '两路互斥：同一请求同一拍只走一路（一条 if / return），'
        '不是两个读数相加', 8.6, C_RED, 'middle', True, maxw=520, tag='mutex')

# ================= ⑥ 会合：够不够（青） + return None 出口（红） =================
y = y_br_end + GAP
n5 = nbox(MX, y, N5_W,
          '两路的数在这里会合 —— 够不够？',
          'vllm/v1/core/kv_cache_manager.py:L523-L527',
          [rrow('available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks',
                8.5, lc.C_TXT, lead=16, tag='L523'),
           rrow('required_blocks = num_blocks_to_allocate + watermark_blocks',
                8.5, lc.C_TXT, lead=16, tag='L524'),
           rrow('if required_blocks > available_blocks: return None',
                8.5, lc.C_TXT, bold=True, lead=16, tag='L525-L527'),
           rrow('会合点 = 同一个口径：预测器与分配器数的都是同一个自由队列；'
                '全注意力、无命中、无外部 token 时，skipped=0 使慢路内层 max 恰为 num_req_blocks、'
                '可驱逐为 0 → 严格退化成快路那句 max(required − num_req, 0)', 8.5, '#334155'),
           rrow('None 不是异常，是账本算出来的正常返回值：这一拍不取块、不留半截'
                '（后果见「三段式」图：回指第 10 章拿不到块 break、第 11 章抢占的触发信号）',
                8.5, lc.C_MUTE)],
          C_KV, C_KV_F, badge='⑥')
lc.seg(MX + N5_W + 6, y + 40, NONE_X - 6, y + 40, C_RED, 2.0, 'ab')
nbox(NONE_X, y, NONE_W,
     'return None',
     '',
     [rrow('不够 → 整笔拒绝（回退留给调用方 waiting / preempt）——块表不动、空闲计数不动',
           8.4, '#334155'),
      rrow('回指第 10 章「拿不到块 break」、第 11 章抢占的触发信号', 8.4, lc.C_MUTE)],
     C_RED, '#fef2f2', tfs=11)
y_prev = n5
y_disp = fork(N5_CX, y_prev, [LCX, RCX], C_KV, sw=2.4)
vlabel(N5_CX, y_prev + 15, '够了 → 依序派发：先挂账（②）、后取块（③）', 8.8, C_KV, maxw=N5_W)

# ================= ⑦ 派发：挂账 / 取块（青） =================
disp_l = nbox(LX, y_disp, HALF_W,
              '挂账（先）：复用他人算过的命中块',
              'kv_cache_manager.py:L535 · single_type…:L232-L289',
              [rrow('self.block_pool.touch(new_computed_blocks)（L269）：ref_cnt 0→1，'
                    'ref_cnt==0 的块从自由队列中间摘出（记账即占容量）', 8.6, '#334155'),
               rrow('running 请求这道闸直接 no-op：if any(request_id in manager.num_cached_block '
                    'for manager in self.single_type_managers): → return'
                    '（kv_cache_coordinator.py:L212-L217）', 8.6, '#334155'),
               rrow('只在有命中块 / 外部 token 时才进这一段（kv_cache_manager.py:L529-L532）',
                    8.6, lc.C_MUTE)],
              C_KV, C_KV_F, badge='②', tfs=11)
disp_r = nbox(RX, y_disp, HALF_W,
              '取块（后）：真从自由队列拿',
              'kv_cache_manager.py:L542 · single_type…:L330-L369',
              [rrow('CoW 前奏：if request_id in self._partial_hit_reqs: → get_new_blocks(1) + '
                    '_apply_cow(…)（L347-L357）——本章单组主路径恒不出现', 8.6, '#334155'),
               rrow('num_new_blocks = num_required_blocks - len(req_blocks)（L361）→ '
                    'self.block_pool.get_new_blocks(num_new_blocks)（L365）', 8.6, '#334155'),
               rrow('差值 ≤ 0 时只换不增，只返回 cow_blocks（L362-L363）', 8.6, lc.C_MUTE)],
              C_KV, C_KV_F, badge='③', tfs=11)
y_disp_end = max(disp_l, disp_r)
pool_top = y_disp_end + GAP
merge([LCX, RCX], [disp_l, disp_r], MIDX, pool_top, C_GREY, sw=1.8)
lc.text((LCX + RCX) / 2, pool_top - GAP + 8, '两处下游调用都落在同一个自由队列上（touch 摘一次、'
        'get_new_blocks 摘一段）', 8.6, C_GREY, 'middle', maxw=560, tag='poolink')

# ================= ⑧ 块池（青） =================
n_pool = nbox(MX, pool_top, CW,
              'BlockPool —— 自由队列的出入口（实体块只在这里动）',
              'vllm/v1/core/block_pool.py',
              [rrow('get_new_blocks(num_blocks)（L647-L677）：free_block_queue.popleft_n(num_blocks) '
                    '→ ref_cnt=1；开缓存时逐个 _maybe_evict_cached_block 惰性摘哈希', 8.8, '#334155'),
               rrow('touch(blocks)（L702-L717）：ref_cnt==0 且非 null 的块先从自由队列中间移除，'
                    '再 ref_cnt += 1', 8.8, '#334155'),
               rrow('上面那次容量对照量的就是它：get_num_free_blocks()（kv_cache_manager.py:L523）',
                    8.8, lc.C_MUTE)],
              C_KV, C_KV_F, badge='⑦')
y_prev = n_pool
drop(MIDX, y_prev, y_prev + GAP, C_GREY, sw=2.0, dash=True)
vlabel(MIDX, y_prev + GAP * 0.62, '回到 allocate_slots 收尾：写回满块（L563）', 8.8, C_GREY)
y = y_prev + GAP

# ================= ⑨ 写回侧（灰虚线 · 第 15 章预告） =================
n_wb = nbox(MX, y, CW,
            '写回侧（本图到此为止）—— 满块登记与哈希：第 15 章前缀缓存详解（预告）',
            '',
            [rrow('self.coordinator.cache_blocks(request, num_tokens_to_cache)（kv_cache_manager.py:L563）'
                  ' → manager.cache_blocks(…)（single_type…:L427-L477）'
                  ' → BlockPool.cache_full_blocks(…)（block_pool.py:L225-L342）', 8.6, '#334155'),
             rrow('幂等闸：num_cached_blocks >= num_full_blocks 就 return（L445-L448）', 8.6, '#334155'),
             rrow('它写下的 self.num_cached_block[request.request_id] = num_full_blocks（L477）'
                  '正是上面 fast-path 开关的武装点——本图首尾在这里扣环：这一拍写回，下一拍走快路',
                  8.6, C_KV, bold=True)],
            C_GREY, C_FILL_GREY, dash=True, tfs=11)

# ================= 小结（青底）：本图账 =================
sy = n_wb + 26
sum_rows = [
    '一次 allocate_slots = 五层调用 + 一次分岔：预测在管家里分快慢两路（记账位在=一句差值、'
    '不在=六行账），两路算出的数永远一致；够不够由 allocate_slots 用同一个自由队列的数判，'
    '不够 return None、够了依序挂账取块',
    '数据往哪流：token 数自上往下传（num_tokens_main_model = total_computed_tokens + '
    'num_new_tokens，再 num_tokens_need_slot = min(num_tokens_main_model + '
    'num_lookahead_tokens, self.max_model_len)，L490-L493）、'
    '块数自下往上回（返回值 = 本拍从自由队列摘走的块数）；'
    '块的实体只在块池里动，各层之间只传数',
]
sum_lines = []
for s in sum_rows:
    sum_lines += wrap(s, 9.4, CW - 32)
SH = 20 + len(sum_lines) * 16 + 12
lc.rect(MX, sy, CW, SH, C_KV_F, C_KV, rx=7, sw=1.4)
cy = sy + 20
for i, ln in enumerate(sum_lines):
    lc.text(MX + 16, cy, ln, 9.4, C_KV if i == 0 else '#334155', 'start', i == 0,
            maxw=CW - 32, tag='sum%d' % i)
    cy += 16

# ================= 图例 =================
LY = sy + SH + 28
lx = MX
for c, f, name in [(C_EN, C_EN_F, '调度侧（入口 · 快路）'),
                   (C_KV, C_KV_F, '显存账本（慢路 · 块账 · 块池）'),
                   (C_RED, '#fef2f2', 'return None 出口')]:
    lc.rect(lx, LY - 10, 20, 13, f, c, rx=3, sw=1.4)
    lc.text(lx + 26, LY, name, 8.8, lc.C_TXT, 'start', maxw=220, tag='lg' + name[:6])
    lx += 26 + lc.tw(name, 8.8) + 22
lc.rect(lx, LY - 10, 20, 13, C_FILL_GREY, C_GREY, rx=3, sw=1.4, dash=True)
lc.text(lx + 26, LY, '写回侧（第 15 章预告）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg:wb')
lx += 26 + lc.tw('写回侧（第 15 章预告）', 8.8) + 22
lc.seg(lx, LY - 4, lx + 30, LY - 4, C_KV, 2.2, 'kv')
lc.text(lx + 36, LY, '调用方向（自上而下）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg:ar')
lx += 36 + lc.tw('调用方向（自上而下）', 8.8) + 22
lc.seg(lx, LY - 4, lx + 30, LY - 4, C_GREY, 1.8, 'gry')
lc.text(lx + 36, LY, '会合母线（两路归一）', 8.8, lc.C_TXT, 'start',
        maxw=BXR - lx - 36, tag='lg:mg')

# ================= 图注（给结论）+ 页脚锚 =================
CY0 = LY + 26
lc.text(MX, CY0, '段序与 None 的后果见「allocate_slots 三段式」图；本图只讲谁调用谁、'
                 '两路账从哪来、数往哪流——引用的公式与行号逐字取自 pin 源码',
        8.6, lc.C_MUTE, 'start', maxw=CW, tag='cp0')
lc.text(MX, CY0 + 18, '两路算出的数永远一致、够不够看同一个自由队列：一次 allocate_slots 下潜五层，'
                      '快慢分岔只改「怎么数」，不改「数出来必须与谁摘了块对得上」；'
                      '写回满块则替下一拍武装 fast-path 开关',
        9.5, lc.C_TXT, 'start', True, maxw=CW, tag='cap')
lc.text(MX, CY0 + 40, '逐字锚 vllm/v1/core/sched/scheduler.py:L439/L578/L973/L744-L747 · '
                      'vllm/v1/core/kv_cache_manager.py:L344-L565/L490-L493/L510-L519/L523-L527/L529-L532/'
                      'L535/L542/L551-L552/L563',
        8.2, lc.C_FAINT, 'start', maxw=CW, tag='ft1')
lc.text(MX, CY0 + 54, 'vllm/v1/core/kv_cache_coordinator.py:L130-L190/L210-L217/L238-L271 · '
                      'vllm/v1/core/single_type_kv_cache_manager.py:L144-L230（L178/L192-L200/L202-L213/'
                      'L218-L230）/L232-L289/L269/L282/L330-L369/L347-L357/L427-L477/L515',
        8.2, lc.C_FAINT, 'start', maxw=CW, tag='ft2')
lc.text(MX, CY0 + 68, 'vllm/v1/core/block_pool.py:L647-L677/L702-L717/L225-L342 · '
                      '行号基线 vLLM v0.27.1 · 数据出处：pin 源码逐行核对（块账四问：'
                      'fast-path 开关 / 六行公式 / 五段映射 / CoW 时机）',
        8.2, lc.C_FAINT, 'start', maxw=CW, tag='ft3')

# ================= 装配输出 =================
H = CY0 + 68 + 20
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>',
       lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-alloc-calls.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
