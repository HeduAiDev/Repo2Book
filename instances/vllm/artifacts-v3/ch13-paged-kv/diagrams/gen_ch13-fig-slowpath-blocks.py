#!/usr/bin/env python3
"""ch13 机制图 · 慢路径六行账的块关系（v2 双景重画：E2 全注意力 × E3 滑窗 W=16）

放大自 L0『调度 · 显存账本』列（青）中 KVCacheManager/SingleTypeKVCacheManager 那一格，
本章「先数块」节六行公式的**块空间关系**。

v2 动机（读者反馈）：v1 只画 E2（全注意力），六个量里有一半（跳段 /
local_computed 与跳段的 max、skipped_new_computed）在 E2 恒 0 或恒等于命中数，
读者看不清「各种 block 之间到底是什么关系」。v2 把 E2 与 E3 摆到**同一条块号刻度尺
（0..6）**上，上下两景逐块对齐：同一批量（new_computed_blocks / num_req_blocks /
num_skipped_tokens / num_new_blocks / num_local_computed_blocks /
num_skipped_new_computed_blocks / num_evictable_blocks）两景各就各位，
只在 SWA 才登场的量配一个定制场景（滑窗 W=16 + 外部 32 token）把关系画出来。

claim：E2（全注意力）——块 0-2 命中（touch 摘走、全冷 → 可驱逐 3）、块 3-6 新分配 4，
return = 4 + 3 = 7 是本拍从自由队列摘走的总块数；E3（滑窗 W=16 + 外部 32 token）——
窗外 49 token（floor 49//16 = 3 整块）换 null 占位、命中 2 块落进跳段前缀故不 touch
（可驱逐 0）、跨 48/49 窗边界的块必须留实体，新块 = 7 − max(3, 2) = 4 = 外部段 1 块 +
get_new_blocks 3 块，return = 4。max 取跳段支的病根：ext_comp 抬高跳段基数（total_computed
= 64 → 49）却不进 local_computed（只认命中 2 + 已持 0）——「已算过 ≠ 不用分配」。

数字全部取自本章正文 E2/E3 两例（narrative/chapter.md，E2 表行 + E3 行）与 pin
vllm/v1/core/single_type_kv_cache_manager.py:L178-L230 六行账逐项代入，由常量算出、
脚本内 assert 对账，零即兴数字。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

# ================= 数据：E2（全注意力）与 E3（滑窗 W=16）——唯一真相源 =================
BS = 16                       # block_size：1 块 = 16 token
# ---- E2：prompt 100 token、前缀命中 48 token = 3 整块、冷启动已持 0、命中全冷 ----
E2_PROMPT, E2_HIT_TOK, E2_HELD = 100, 48, 0
E2_HIT_BLOCKS = E2_HIT_TOK // BS                                   # 3
E2_REQUIRED = -(-E2_PROMPT // BS)                                  # 7
E2_SKIPPED_TOK = 0                                                 # 全注意力恒 0（L661）
E2_SKIPPED_BLOCKS = E2_SKIPPED_TOK // BS                           # 0（floor）
E2_LOCAL = E2_HIT_BLOCKS + E2_HELD                                 # 3（L203）
E2_NEW = max(E2_REQUIRED - max(E2_SKIPPED_BLOCKS, E2_LOCAL), 0)    # 4（L210-L213）
E2_SKIPPED_NEW = max(0, E2_SKIPPED_BLOCKS - E2_HELD)               # 0（L218）
E2_EVICTABLE = E2_HIT_BLOCKS                                       # 3（L223-L225 全冷）
E2_RET = E2_NEW + E2_EVICTABLE                                     # 7（L230）
E2_TAIL_SLOTS = E2_REQUIRED * BS - E2_PROMPT                       # 12：尾块空置槽
# ---- E3：滑窗 W=16、本地命中 2 块（32 token）+ connector 外部 32 token、本拍再算 36 ----
SW, E3_HIT_TOK, E3_EXT_TOK, E3_NEW_TOK, E3_HELD = 16, 32, 32, 36, 0
E3_HIT_BLOCKS = E3_HIT_TOK // BS                                   # 2（块 0-1）
E3_PROMPT = E3_HIT_TOK + E3_EXT_TOK + E3_NEW_TOK                   # 100
E3_REQUIRED = -(-E3_PROMPT // BS)                                  # 7
E3_TOTAL_COMPUTED = E3_HIT_TOK + E3_EXT_TOK                        # 64（本地 32 + 外部 32）
E3_SKIPPED_TOK = max(0, E3_TOTAL_COMPUTED - SW + 1)                # 49（L1083 滑窗公式）
E3_SKIPPED_BLOCKS = E3_SKIPPED_TOK // BS                           # 3（L206 floor）
E3_WIN_START, E3_WIN_END = E3_SKIPPED_TOK, E3_TOTAL_COMPUTED       # 窗 = token 49..64
E3_CROSS_BLOCK = E3_SKIPPED_BLOCKS                                 # 3：跨 48/49 边界的块
E3_LOCAL = E3_HIT_BLOCKS + E3_HELD                                 # 2（L203）
E3_NEW = max(E3_REQUIRED - max(E3_SKIPPED_BLOCKS, E3_LOCAL), 0)    # 4（L210-L213）
E3_SKIPPED_NEW_FORMULA = max(0, E3_SKIPPED_BLOCKS - E3_HELD)       # 3（L218）
E3_SKIPPED_NEW = min(E3_HIT_BLOCKS, E3_SKIPPED_NEW_FORMULA)        # 2：命中 2 块全落跳段内
E3_EVICTABLE = 0                                                   # 被剔出 touch → 0
E3_RET = E3_NEW + E3_EVICTABLE                                     # 4（L230）
E3_EXT_ALLOC = 1                                                   # 外部段另发 1 块（跨界的块 3）
E3_POOL_NEW = E3_REQUIRED - E3_SKIPPED_BLOCKS - E3_EXT_ALLOC       # 3：get_new_blocks 摘走
E3_REAL = E3_EXT_ALLOC + E3_POOL_NEW                               # 4：实体块 = 预测 4 ✓
# 逐项对账（对不上就崩，防手滑改数）
assert (E2_HIT_BLOCKS, E2_REQUIRED, E2_SKIPPED_BLOCKS, E2_LOCAL, E2_NEW, E2_SKIPPED_NEW,
        E2_EVICTABLE, E2_RET, E2_TAIL_SLOTS) == (3, 7, 0, 3, 4, 0, 3, 7, 12), 'E2 对不上正文'
assert (E3_HIT_BLOCKS, E3_REQUIRED, E3_TOTAL_COMPUTED, E3_SKIPPED_TOK, E3_SKIPPED_BLOCKS,
        E3_LOCAL, E3_NEW, E3_SKIPPED_NEW_FORMULA, E3_SKIPPED_NEW, E3_EVICTABLE,
        E3_RET) == (2, 7, 64, 49, 3, 2, 4, 3, 2, 0, 4), 'E3 对不上正文'
assert (E3_WIN_START, E3_WIN_END) == (49, 64) and E3_WIN_END - E3_WIN_START + 1 == SW
assert E3_REAL == E3_NEW and E2_NEW + E2_HIT_BLOCKS == E2_REQUIRED

# ================= 配色（l0_common 既有语义色；本图配图例） =================
C_HIT_S, C_HIT_F = lc.C_ENG_S, lc.C_ENG_F        # 命中块（暖）= 已在池中、touch 复用
C_NEW_S, C_NEW_F = lc.C_KV_S, lc.C_KV_F          # 新块（青）= get_new_blocks 真分配
C_NUL_S, C_NUL_F = lc.C_FAINT, '#f1f5f9'         # 跳段 null 占位（灰）
C_EXT_S, C_EXT_F = lc.C_KV_S, 'url(#exth)'       # 外部块（青 + 斜纹）= connector 搬来
C_SLV_F = '#e2e8f0'                              # 跨界块里落在窗外的那一角（1 token）
EXTRA_DEFS = ('<defs><pattern id="exth" width="8" height="8" patternUnits="userSpaceOnUse" '
              'patternTransform="rotate(45)">'
              f'<rect width="8" height="8" fill="{C_NEW_F}"/>'
              f'<line x1="0" y1="0" x2="0" y2="8" stroke="{C_NEW_S}" stroke-width="1.6" '
              'opacity="0.32"/></pattern></defs>')

lc.reset()

W, H = 1500, 1160
MX, BXR = 60, 1440
RAIL_X0, RAIL_X1 = 60, 250          # 左侧竖排图例栏
PX0, PX1 = 266, 1440                # 两个场景面板
BS_X0, BW = 278, 164                # 块条带：7 块 × 164 = 1148 → 278..1426
SH = 86                             # 条带高
PX = BW / BS                        # 10.25 px / token（两景同一条刻度尺）


def bx(i):
    """第 i 块的左缘 x（块号刻度尺唯一入口）。"""
    return BS_X0 + i * BW


def tx(tok):
    """token 位置 → x（与块刻度同尺）。"""
    return BS_X0 + tok * PX


def brk(x0, x1, y, color, sw=1.3, drop=0):
    """方括号：横线 + 两端短竖线；drop 让竖线向上多伸（嵌套外层用）。"""
    lc.seg(x0, y, x1, y, color, sw)
    lc.seg(x0, y - 5 - drop, x0, y + 5, color, sw)
    lc.seg(x1, y - 5 - drop, x1, y + 5, color, sw)


def tint(x, y, w, h, color, op):
    """半透明色带（窗内覆盖区）——压在块上仍看得见块自身颜色与文字。"""
    s = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
         f'fill="{color}" fill-opacity="{op}"/>')
    lc.ELEMS.append(((x - 2, y - 2, x + w + 2, y + h + 2), s))


# ---------------- 标题区 ----------------
lc.text(MX, 34, '同一套块号刻度尺（0..6）上的两个场景：全注意力 E2 命中 3 块 touch 摘走 + 新分配 4 块，'
                '滑窗 E3 跳段 3 块换 null + 实体 4 块',
        16, lc.C_TXT, 'start', True, maxw=1130, tag='title')
lc.text(MX, 58, '七个量（new_computed_blocks / num_req_blocks / num_skipped_tokens / num_new_blocks / '
                'num_local_computed_blocks / num_skipped_new_computed_blocks / num_evictable_blocks）'
                '各长在条带哪一段', 10.5, lc.C_MUTE, 'start', maxw=1130, tag='subtitle')
_ch = '放大自 L0 · 显存账本列 · 先数块节'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左侧竖排图例（块的六种身份） ----------------
RY0, RY1 = 96, 894
lc.rect(RAIL_X0, RY0, RAIL_X1 - RAIL_X0, RY1 - RY0, '#ffffff', lc.C_MUTE, rx=10, sw=1.5)
lc.text((RAIL_X0 + RAIL_X1) / 2, RY0 + 26, '图例 · 块的六种身份', 10.5, lc.C_TXT, 'middle', True,
        maxw=170, tag='rail:t')
lc.text((RAIL_X0 + RAIL_X1) / 2, RY0 + 42, '（同一条刻度尺上）', 8.5, lc.C_MUTE, 'middle',
        maxw=170, tag='rail:s')
LEG = [
    (C_HIT_F, C_HIT_S, True, '命中块（new_comp）',
     ['别的请求算过、内容相同，', 'touch 摘走复用；不新分配，',
      '但全冷时正躺自由队列当驱逐候选'], True),
    (C_NEW_F, C_NEW_S, False, '新块（get_new_blocks）',
     ['本拍真新分配：', 'required 减已就位（含跳段）', '之后的缺口，从自由队列摘走'], False),
    (C_NUL_F, C_NUL_S, True, '跳段 null 占位',
     ['整块落在注意力窗外 → 换占位；', '不落实体、也永不出租，',
      '但表长仍算它一格'], False),
    (C_EXT_F, C_EXT_S, True, '外部块（ext_comp）',
     ['connector 从远端搬来：token 已算过，', '本地却一个实体块都没有，',
      '必须真分配（外部段另发）'], False),
    ('SPLIT', C_NEW_S, False, '跨界保留',
     ['块里还有一角在窗内（floor 只跳整块，', '49 // 16 = 3）→ 必须留实体；',
      '换 null 就丢了窗内那 15 个 token'], False),
    ('#ffffff', C_NUL_S, True, '尾块空置槽',
     ['cdiv 取整的零头（100 token', '用不满 112 槽）——是空置，',
      '不是 partial 命中'], False),
]
LIY0, LIH = RY0 + 62, 112
for i, (fill, stroke, dash, name, lines, warm) in enumerate(LEG):
    iy = LIY0 + i * LIH
    sx, sy, sw_, sh_ = RAIL_X0 + 16, iy - 14, 30, 18
    if fill == 'SPLIT':                       # 跨界块字形：窗外一角 + 窗内实体
        lc.rect(sx, sy, sw_, sh_, C_SLV_F, 'none', rx=0, sw=0)
        lc.rect(sx + sw_ * 1 / 8, sy, sw_ * 7 / 8, sh_, C_EXT_F, C_EXT_S, rx=2, sw=1.2, dash=True)
        lc.seg(sx + sw_ * 1 / 8, sy, sx + sw_ * 1 / 8, sy + sh_, C_NUL_S, 1.1, dash=True)
    else:
        lc.rect(sx, sy, sw_, sh_, fill, stroke, rx=3, sw=1.3, dash=dash)
    lc.text(RAIL_X0 + 56, iy, name, 9.5, stroke, 'start', True,
            maxw=RAIL_X1 - RAIL_X0 - 58, tag='rail:n' + str(i))
    for j, ln in enumerate(lines):
        lc.text(RAIL_X0 + 16, iy + 18 + j * 13, ln, 8.0, lc.C_TXT, 'start',
                maxw=RAIL_X1 - RAIL_X0 - 28, tag=f'rail:d{i}{j}')
# 图例栏底部：两景一眼对照（读者反馈的落点——「各种 block 的关系」）
_cmp_y = LIY0 + len(LEG) * LIH - 22
lc.seg(RAIL_X0 + 16, _cmp_y, RAIL_X1 - 16, _cmp_y, lc.C_FAINT, 1.0)
lc.text((RAIL_X0 + RAIL_X1) / 2, _cmp_y + 18, '两景一眼对照', 9.5, lc.C_TXT, 'middle', True,
        maxw=160, tag='rail:cmp_t')
for j, ln in enumerate(['E2：命中 3 块全冷 → 都可 touch',
                        '（可驱逐 3）——无跳段、无 null',
                        'E3：命中 2 块全落跳段内 → 不 touch',
                        '（可驱逐 0）——3 块换 null，跨界留实体']):
    lc.text(RAIL_X0 + 16, _cmp_y + 34 + j * 13, ln, 8.0, lc.C_TXT, 'start',
            maxw=RAIL_X1 - RAIL_X0 - 28, tag=f'rail:cmp{j}')

# ==================================================================================
# 场景 A（上）：E2 全注意力
# ==================================================================================
PA_Y, PA_H = 96, 428
lc.rect(PX0, PA_Y, PX1 - PX0, PA_H, '#ffffff', C_HIT_S, rx=10, sw=1.8)
lc.text(PX0 + 16, PA_Y + 26, '场景 A · E2 全注意力（主路径六行账走全）', 12.5, C_HIT_S, 'start',
        True, maxw=520, tag='a:t')
lc.text(PX1 - 16, PA_Y + 26, '全注意力没有窗外 token → num_skipped_tokens 恒 0（L661-L672）',
        9, lc.C_MUTE, 'end', maxw=600, tag='a:t2')
lc.text(PX0 + 16, PA_Y + 48, '现场：prompt 100 token、1 块 = 16 token；前缀命中 48 token = 3 整块'
        '（命中不落在块中间 → partial 不触发）、冷启动已持 0 块。', 9.3, lc.C_TXT, 'start',
        maxw=1130, tag='a:d1')
lc.text(PX0 + 16, PA_Y + 66, '块 0-2 命中：touch 复用别人的块（这三块全冷 ref_cnt==0、正躺自由队列）；'
        '块 3-6 新分配：get_new_blocks 真摘——两者都离开自由队列。', 9.3, lc.C_TXT, 'start',
        maxw=1130, tag='a:d2')

a_rl, a_rb = PA_Y + 92, PA_Y + 100          # required 标注行
lc.text((BS_X0 + bx(E2_REQUIRED)) / 2, a_rl,
        f'required = cdiv({E2_PROMPT}, {BS}) = {E2_REQUIRED} 块 = {E2_REQUIRED * BS} 槽'
        f'（目标表长铺满这条尺 · L178）', 9.5, C_HIT_S, 'middle', True, maxw=760, tag='a:req')
brk(BS_X0, bx(E2_REQUIRED), a_rb, C_HIT_S, drop=4)

a_sty = PA_Y + 118
for i in range(E2_REQUIRED):                # 块条带（块 0-2 命中 / 块 3-6 新块）
    hit = i < E2_HIT_BLOCKS
    lc.rect(bx(i), a_sty, BW, SH, C_HIT_F if hit else C_NEW_F,
            C_HIT_S if hit else C_NEW_S, rx=4, sw=1.7, dash=hit)
# 尾块空置槽（token 100-111 未用，cdiv 零头）
lc.rect(tx(E2_PROMPT), a_sty + SH - 22, bx(E2_REQUIRED) - tx(E2_PROMPT), 22, C_NUL_F, 'none',
        rx=0, sw=0)
lc.seg(tx(E2_PROMPT), a_sty + SH - 22, tx(E2_PROMPT), a_sty + SH, C_NUL_S, 1.1, dash=True)
for i in range(E2_REQUIRED):
    hit = i < E2_HIT_BLOCKS
    col = C_HIT_S if hit else C_NEW_S
    lc.text(bx(i) + BW / 2, a_sty + 22, f'块 {i}', 12, col, 'middle', True, maxw=BW - 12,
            tag=f'a:k{i}')
    lc.text(bx(i) + BW / 2, a_sty + 40, f'token {i * BS}-{i * BS + BS - 1}', 8.3, lc.C_MUTE,
            'middle', maxw=BW - 12, tag=f'a:tk{i}')
    lc.text(bx(i) + BW / 2, a_sty + 58, '命中块' if hit else '新块', 9.5, col, 'middle', True,
            maxw=BW - 12, tag=f'a:r{i}')
lc.text((tx(E2_PROMPT) + bx(E2_REQUIRED)) / 2, a_sty + SH - 7, f'空置 {E2_TAIL_SLOTS} 槽', 7.6,
        lc.C_MUTE, 'middle', maxw=BW - 20, tag='a:tail')

# 行 2：条带正下方的两条括注（命中段 / 新块段）
a_r2 = PA_Y + 224
brk(BS_X0, bx(E2_HIT_BLOCKS), a_r2, C_HIT_S)
brk(bx(E2_HIT_BLOCKS), bx(E2_REQUIRED), a_r2, C_NEW_S)
lc.text((BS_X0 + bx(E2_HIT_BLOCKS)) / 2, a_r2 + 16,
        f'命中 {E2_HIT_BLOCKS} 块 · touch 摘走（ref_cnt 0→1、离开自由队列）', 9.5, C_HIT_S,
        'middle', True, maxw=BW * E2_HIT_BLOCKS - 16, tag='a:b1')
lc.text((BS_X0 + bx(E2_HIT_BLOCKS)) / 2, a_r2 + 32,
        f'→ num_evictable_blocks = {E2_EVICTABLE}（三块全冷 ref_cnt==0，touch 就把它们摘走）', 8.5,
        C_HIT_S, 'middle', maxw=BW * E2_HIT_BLOCKS - 8, tag='a:b1b')
lc.text((bx(E2_HIT_BLOCKS) + bx(E2_REQUIRED)) / 2, a_r2 + 16, f'num_new_blocks = {E2_NEW}',
        9.5, C_NEW_S, 'middle', True, maxw=BW * E2_NEW - 16, tag='a:b2')
lc.text((bx(E2_HIT_BLOCKS) + bx(E2_REQUIRED)) / 2, a_r2 + 32,
        f'= {E2_REQUIRED} − max(skipped {E2_SKIPPED_BLOCKS}, local_computed {E2_LOCAL})',
        8.5, lc.C_TXT, 'middle', maxw=BW * E2_NEW - 8, tag='a:b2b')
# 行 3：已持的空括号（冷启动）+ 六行量逐项
a_r3 = PA_Y + 272
lc.seg(BS_X0 - 8, a_r3 - 6, BS_X0 - 8, a_r3 + 6, C_NEW_S, 1.3)
lc.seg(BS_X0 + 8, a_r3 - 6, BS_X0 + 8, a_r3 + 6, C_NEW_S, 1.3)
lc.seg(BS_X0 - 8, a_r3, BS_X0 + 8, a_r3, C_NEW_S, 1.3)
lc.text(BS_X0 + 18, a_r3 + 4, f'num_req_blocks = {E2_HELD}：冷启动，本拍前表上 0 块（L192）——'
        '减数里没有它', 9, C_NEW_S, 'start', True, maxw=520, tag='a:held')

a_chy = PA_Y + 306
A_CH = 56
A_CHIPS = [
    (f'required = {E2_REQUIRED}', f'cdiv({E2_PROMPT}, {BS}) = {E2_REQUIRED} 块', '(L178)',
     lc.C_TXT, '#ffffff'),
    (f'max(skipped {E2_SKIPPED_BLOCKS}, local {E2_LOCAL}) = {E2_LOCAL}',
     f'减数取大的：{E2_SKIPPED_BLOCKS} vs {E2_LOCAL}', '(L203 · L206 · L210-L213)', C_HIT_S, C_HIT_F),
    (f'num_new_blocks = {E2_NEW}', f'{E2_REQUIRED} − {E2_LOCAL} = {E2_NEW}（块 {E2_HIT_BLOCKS}-'
     f'{E2_REQUIRED - 1}）', '(L210-L213)', C_NEW_S, C_NEW_F),
    (f'num_evictable_blocks = {E2_EVICTABLE}', '命中 3 块全冷 ref_cnt==0', '(L223-L225)',
     C_HIT_S, C_HIT_F),
    (f'return = {E2_RET}', f'{E2_NEW} + {E2_EVICTABLE} = {E2_RET}（自由队列净减）', '(L230)',
     C_NEW_S, C_NEW_F),
]
A_CW_A, A_GAP = (PX1 - 16 - (PX0 + 16) - 4 * 42) / 5, 42
for i, (t1, t2, t3, col, fill) in enumerate(A_CHIPS):
    cxx = PX0 + 16 + i * (A_CW_A + A_GAP)
    lc.rect(cxx, a_chy, A_CW_A, A_CH, fill, col, rx=6, sw=1.6)
    lc.text(cxx + A_CW_A / 2, a_chy + 21, t1, 10.5, col, 'middle', True, maxw=A_CW_A - 14,
            tag=f'a:c{i}')
    lc.text(cxx + A_CW_A / 2, a_chy + 39, t2, 9, lc.C_TXT, 'middle', maxw=A_CW_A - 14,
            tag=f'a:c{i}b')
    lc.text(cxx + A_CW_A / 2, a_chy + 52, t3, 7.8, lc.C_MUTE, 'middle', maxw=A_CW_A - 14,
            tag=f'a:c{i}c')
    if i < 4:
        ax0, ax1 = cxx + A_CW_A, cxx + A_CW_A + A_GAP
        lc.seg(ax0, a_chy + A_CH / 2, ax1, a_chy + A_CH / 2, lc.C_MUTE, 1.6, 'std')
for i, op in ((0, '−'), (2, '+')):
    gx = PX0 + 16 + i * (A_CW_A + A_GAP) + A_CW_A + A_GAP / 2
    lc.text(gx, a_chy + 22, op, 14, lc.C_TXT, 'middle', True, maxw=A_GAP - 4, tag=f'a:op{i}')
lc.text(PX0 + 16, a_chy + 74,
        f'六行量逐项（E2 值）：num_skipped_tokens = {E2_SKIPPED_TOK}（全注意力基类无窗外 token · '
        f'L661-L672）· num_skipped_blocks = {E2_SKIPPED_TOK} // {BS} = {E2_SKIPPED_BLOCKS}'
        f'（floor · L206）· num_local_computed_blocks = {E2_HIT_BLOCKS} + {E2_HELD} = {E2_LOCAL}'
        '（命中 + 已持 · L203）', 8.2, lc.C_TXT, 'start', maxw=1130, tag='a:q1')
lc.text(PX0 + 16, a_chy + 88,
        f'num_new_blocks = max({E2_REQUIRED} − {E2_LOCAL}, 0) = {E2_NEW}（L210-L213）· '
        f'num_skipped_new_computed_blocks = max(0, {E2_SKIPPED_BLOCKS} − {E2_HELD}) = '
        f'{E2_SKIPPED_NEW}（E2 无跳段，剔除集为空 · L218）· num_evictable_blocks = {E2_EVICTABLE}'
        f'（L223-L225）· return = {E2_NEW} + {E2_EVICTABLE} = {E2_RET}（L230）', 8.2, lc.C_MUTE,
        'start', maxw=1130, tag='a:q2')

# ==================================================================================
# 场景 B（下）：E3 滑窗 W=16（SWA 定制场景）
# ==================================================================================
PB_Y, PB_H = 540, 496
lc.rect(PX0, PB_Y, PX1 - PX0, PB_H, '#ffffff', C_NEW_S, rx=10, sw=1.8)
lc.text(PX0 + 16, PB_Y + 24, '场景 B · E3 滑窗 W=16（SWA 定制场景：全注意力永远见不到的那几行量上场）',
        12.5, C_NEW_S, 'start', True, maxw=700, tag='b:t')
lc.text(PX1 - 16, PB_Y + 24, '滑窗子类覆写 get_num_skipped_tokens = max(0, total − W + 1)（L1057-L1083）',
        9, lc.C_MUTE, 'end', maxw=620, tag='b:t2')
lc.rect(PX0 + 16, PB_Y + 34, PX1 - PX0 - 32, 58, C_NEW_F, C_NEW_S, rx=6, sw=1.2, dash=True)
lc.text(PX0 + 30, PB_Y + 54, 'SWA 定制现场：同一个 100 token 的请求——前缀命中 2 块（32 token）+ connector '
        '外部缓存 32 token（远端已算）→ 本拍 total_computed = 64、只再算 36 token。', 9, lc.C_TXT,
        'start', maxw=1130, tag='b:card1')
lc.text(PX0 + 30, PB_Y + 74, '滑窗 W = 16 让前 49 个 token 落到窗外：整块换 null 占位、跨 48/49 窗边界的块'
        '必须留实体（floor 救命）；外部 token 本地一个实体块都没有、必须真分配。', 9, lc.C_TXT,
        'start', maxw=1130, tag='b:card2')

b_rl, b_rb = PB_Y + 108, PB_Y + 116          # required（与 A 同尺同值）
lc.text(tx(E3_WIN_END) + 240, b_rl, f'required = cdiv({E3_PROMPT}, {BS}) = {E3_REQUIRED} 块 '
        f'= {E3_REQUIRED * BS} 槽（与 A 同尺同值 · L178）', 9, C_HIT_S, 'middle', True,
        maxw=780, tag='b:req')
brk(BS_X0, bx(E3_REQUIRED), b_rb, C_HIT_S, drop=4)
b_sl1, b_sl2, b_sb = PB_Y + 134, PB_Y + 148, PB_Y + 156   # 跳段（窗外）
lc.text((tx(0) + tx(E3_SKIPPED_TOK)) / 2, b_sl1,
        f'跳段（窗外）：num_skipped_tokens = max(0, {E3_TOTAL_COMPUTED} − {SW} + 1) = '
        f'{E3_SKIPPED_TOK}', 9, C_NUL_S, 'middle', True, maxw=640, tag='b:skip1')
lc.text((tx(0) + tx(E3_SKIPPED_TOK)) / 2, b_sl2,
        f'num_skipped_blocks = {E3_SKIPPED_TOK} // {BS} = {E3_SKIPPED_BLOCKS}'
        f'（floor：只跳整块 · L206）', 9, C_NUL_S, 'middle', True, maxw=640, tag='b:skip2')
brk(tx(0), tx(E3_SKIPPED_TOK), b_sb, C_NUL_S, drop=4)

b_sty = PB_Y + 172
for i in range(E3_REQUIRED):
    if i < E3_SKIPPED_BLOCKS:               # 整块在窗外 → null 占位
        lc.rect(bx(i), b_sty, BW, SH, C_NUL_F, C_NUL_S, rx=4, sw=1.4, dash=True)
    elif i == E3_CROSS_BLOCK:               # 跨界块：外部段另发 1 块的实体块
        lc.rect(bx(i), b_sty, BW, SH, C_EXT_F, C_EXT_S, rx=4, sw=1.6, dash=True)
    else:                                   # get_new_blocks 真新分配
        lc.rect(bx(i), b_sty, BW, SH, C_NEW_F, C_NEW_S, rx=4, sw=1.7)
# 跨界块里落在窗外的那 1 个 token（块 3 左缘 = token 48）——floor 救命的可视化证据
lc.rect(bx(E3_CROSS_BLOCK), b_sty, PX, SH, C_SLV_F, 'none', rx=0, sw=0)
lc.seg(bx(E3_CROSS_BLOCK) + PX, b_sty + 3, bx(E3_CROSS_BLOCK) + PX, b_sty + SH - 3, C_NUL_S,
       1.2, dash=True)
# 窗内色带（token 49..64：W=16）——半透明压在跨块 3 尾与块 4 头上
tint(tx(E3_WIN_START), b_sty, (E3_WIN_END + 1 - E3_WIN_START) * PX, SH, C_NEW_S, 0.14)
# 尾块空置槽（与 A 同）
lc.rect(tx(E3_PROMPT), b_sty + SH - 22, bx(E3_REQUIRED) - tx(E3_PROMPT), 22, C_NUL_F, 'none',
        rx=0, sw=0)
lc.seg(tx(E3_PROMPT), b_sty + SH - 22, tx(E3_PROMPT), b_sty + SH, C_NUL_S, 1.1, dash=True)
B_ROLE = {0: ('命中 → null', C_HIT_S), 1: ('命中 → null', C_HIT_S), 2: ('外部 → null', C_NUL_S),
          E3_CROSS_BLOCK: ('跨界保留 · 外部块', C_EXT_S)}
for i in range(E3_REQUIRED):
    role, col = B_ROLE.get(i, ('新块', C_NEW_S))
    lc.text(bx(i) + BW / 2, b_sty + 22, f'块 {i}', 12, col, 'middle', True, maxw=BW - 12,
            tag=f'b:k{i}')
    lc.text(bx(i) + BW / 2, b_sty + 40, f'token {i * BS}-{i * BS + BS - 1}', 8.3, lc.C_MUTE,
            'middle', maxw=BW - 12, tag=f'b:tk{i}')
    lc.text(bx(i) + BW / 2, b_sty + 58, role, 9.5, col, 'middle', True, maxw=BW - 12,
            tag=f'b:r{i}')
lc.text(bx(E3_CROSS_BLOCK) + BW / 2 + PX / 2, b_sty + 74, '左 1 token（48）在窗外', 8, C_NUL_S,
        'middle', maxw=BW - 12, tag='b:slv')
lc.text((tx(E3_PROMPT) + bx(E3_REQUIRED)) / 2, b_sty + SH - 7, f'空置 {E2_TAIL_SLOTS} 槽', 7.6,
        lc.C_MUTE, 'middle', maxw=BW - 20, tag='b:tail')
# 窗边界（token 49）竖虚线 + 窗内标签
lc.seg(tx(E3_WIN_START), b_sty - 10, tx(E3_WIN_START), b_sty + SH + 6, C_EXT_S, 1.3, dash=True)
lc.text(tx(E3_WIN_START) + 10, b_sl2, f'窗边界 token {E3_WIN_START} → 窗内 {E3_WIN_START}..'
        f'{E3_WIN_END}（W = {SW}，共 {E3_WIN_END - E3_WIN_START + 1} token）', 9, C_EXT_S,
        'start', True, maxw=520, tag='b:win')

# 行 2：条带正下方（命中被剔 / 外部段 / 新摘）
b_r2 = PB_Y + 278
brk(bx(0), bx(E3_HIT_BLOCKS), b_r2, C_HIT_S)
brk(bx(E3_CROSS_BLOCK), bx(E3_CROSS_BLOCK + 1), b_r2, C_EXT_S)
brk(bx(E3_CROSS_BLOCK + 1), bx(E3_REQUIRED), b_r2, C_NEW_S)
lc.text((bx(0) + bx(E3_HIT_BLOCKS)) / 2, b_r2 + 16, f'命中 {E3_HIT_BLOCKS} 块全落跳段前缀里', 9.5,
        C_HIT_S, 'middle', True, maxw=BW * 2 - 16, tag='b:b1')
lc.text((bx(0) + bx(E3_HIT_BLOCKS)) / 2, b_r2 + 32,
        f'→ 不 touch → num_evictable_blocks = {E3_EVICTABLE}', 8.5, C_HIT_S, 'middle',
        maxw=BW * 2 - 8, tag='b:b1b')
lc.text(bx(E3_CROSS_BLOCK) + BW / 2, b_r2 + 16, '外部段另发 1 块', 9.5, C_EXT_S, 'middle', True,
        maxw=BW - 16, tag='b:b2')
lc.text(bx(E3_CROSS_BLOCK) + BW / 2, b_r2 + 32, '（块 3 是实体）', 8.5, lc.C_MUTE, 'middle',
        maxw=BW - 8, tag='b:b2b')
lc.text((bx(E3_CROSS_BLOCK + 1) + bx(E3_REQUIRED)) / 2, b_r2 + 16,
        f'get_new_blocks 摘走 {E3_POOL_NEW} 块', 9.5, C_NEW_S, 'middle', True,
        maxw=BW * 3 - 16, tag='b:b3')
lc.text((bx(E3_CROSS_BLOCK + 1) + bx(E3_REQUIRED)) / 2, b_r2 + 32, '（块 4-6 真新分配）', 8.5,
        lc.C_MUTE, 'middle', maxw=BW * 3 - 8, tag='b:b3b')
# 行 3：外层两条（跳段整块换 null / 预测新块 = 外部 1 + 新摘 3）
b_r3 = PB_Y + 326
brk(bx(0), bx(E3_SKIPPED_BLOCKS), b_r3, C_NUL_S, drop=6)
brk(bx(E3_SKIPPED_BLOCKS), bx(E3_REQUIRED), b_r3, C_NEW_S, drop=6)
lc.text((bx(0) + bx(E3_SKIPPED_BLOCKS)) / 2, b_r3 + 16,
        f'跳段 {E3_SKIPPED_BLOCKS} 整块 → null 占位（表长仍占 3 格）', 9, C_NUL_S, 'middle', True,
        maxw=BW * 3 - 16, tag='b:c1')
lc.text((bx(E3_SKIPPED_BLOCKS) + bx(E3_REQUIRED)) / 2, b_r3 + 16,
        f'num_new_blocks = {E3_REQUIRED} − max({E3_SKIPPED_BLOCKS}, {E3_LOCAL}) = {E3_NEW}'
        f'（外部 {E3_EXT_ALLOC} + 新摘 {E3_POOL_NEW}）', 9, C_NEW_S, 'middle', True,
        maxw=BW * 4 - 16, tag='b:c2')
b_r3n = b_r3 + 32
lc.seg(BS_X0 - 8, b_r3n - 6, BS_X0 - 8, b_r3n + 6, C_NEW_S, 1.3)     # 已持的空括号（与 A 同款）
lc.seg(BS_X0 + 8, b_r3n - 6, BS_X0 + 8, b_r3n + 6, C_NEW_S, 1.3)
lc.seg(BS_X0 - 8, b_r3n, BS_X0 + 8, b_r3n, C_NEW_S, 1.3)
lc.text(BS_X0 + 18, b_r3n + 4,
        f'num_req_blocks = {E3_HELD}（冷启动）· num_local_computed_blocks = {E3_HELD} + '
        f'{E3_HIT_BLOCKS} = {E3_LOCAL}（只认已持 + 命中 · L203）· '
        f'num_skipped_new_computed_blocks = {E3_SKIPPED_NEW}：命中 {E3_HIT_BLOCKS} 块全落进跳段前缀'
        f'（公式 max(0, {E3_SKIPPED_BLOCKS} − {E3_HELD}) = {E3_SKIPPED_NEW_FORMULA} 罩住全部命中、'
        f'切片取空 → 不 touch → 可驱逐 {E3_EVICTABLE} · L218）', 8.0, lc.C_TXT, 'start',
        maxw=1120, tag='b:c3')

b_chy = PB_Y + 376
B_CHIPS = [
    (f'required = {E3_REQUIRED}', f'cdiv({E3_PROMPT}, {BS}) = {E3_REQUIRED} 块', '(L178)',
     lc.C_TXT, '#ffffff'),
    (f'max(skipped {E3_SKIPPED_BLOCKS}, local {E3_LOCAL}) = {E3_SKIPPED_BLOCKS}',
     f'跳段支胜出：{E3_SKIPPED_BLOCKS} > {E3_LOCAL}', '(L203 · L206 · L210-L213)', C_NUL_S, C_NUL_F),
    (f'num_new_blocks = {E3_NEW}', f'{E3_REQUIRED} − {E3_SKIPPED_BLOCKS} = {E3_NEW}（外部 '
     f'{E3_EXT_ALLOC} + 新摘 {E3_POOL_NEW}）', '(L210-L213)', C_NEW_S, C_NEW_F),
    (f'num_evictable_blocks = {E3_EVICTABLE}', f'命中 {E3_HIT_BLOCKS} 块全被剔（不 touch）',
     '(L218 · L223-L225)', C_HIT_S, C_HIT_F),
    (f'return = {E3_RET}', f'{E3_NEW} + {E3_EVICTABLE} = {E3_RET}（自由队列净减）', '(L230)',
     C_NEW_S, C_NEW_F),
]
B_CW_A = (PX1 - 16 - (PX0 + 16) - 4 * A_GAP) / 5
for i, (t1, t2, t3, col, fill) in enumerate(B_CHIPS):
    cxx = PX0 + 16 + i * (B_CW_A + A_GAP)
    lc.rect(cxx, b_chy, B_CW_A, A_CH, fill, col, rx=6, sw=1.6)
    lc.text(cxx + B_CW_A / 2, b_chy + 21, t1, 10.4, col, 'middle', True, maxw=B_CW_A - 14,
            tag=f'b:c{i}')
    lc.text(cxx + B_CW_A / 2, b_chy + 39, t2, 9, lc.C_TXT, 'middle', maxw=B_CW_A - 14,
            tag=f'b:c{i}b')
    lc.text(cxx + B_CW_A / 2, b_chy + 52, t3, 7.8, lc.C_MUTE, 'middle', maxw=B_CW_A - 14,
            tag=f'b:c{i}c')
    if i < 4:
        ax0, ax1 = cxx + B_CW_A, cxx + B_CW_A + A_GAP
        lc.seg(ax0, b_chy + A_CH / 2, ax1, b_chy + A_CH / 2, lc.C_MUTE, 1.6, 'std')
for i, op in ((0, '−'), (2, '+')):
    gx = PX0 + 16 + i * (B_CW_A + A_GAP) + B_CW_A + A_GAP / 2
    lc.text(gx, b_chy + 22, op, 14, lc.C_TXT, 'middle', True, maxw=A_GAP - 4, tag=f'b:op{i}')
lc.text(PX0 + 16, b_chy + 74,
        f'max({E3_SKIPPED_BLOCKS}, {E3_LOCAL}) 取跳段支的病根：ext_comp 的 {E3_EXT_TOK} token 把 '
        f'total_computed 抬到 {E3_TOTAL_COMPUTED}（跳段基数 {E3_SKIPPED_TOK}）却不进 '
        f'local_computed（只认命中 {E3_HIT_BLOCKS} + 已持 {E3_HELD}）——「已算过 ≠ 不用分配」，'
        '外部 token 的块本地一个都没有，必须真分配。', 8.2, lc.C_TXT, 'start', maxw=1130, tag='b:n1')
lc.text(PX0 + 16, b_chy + 88,
        f'分配侧（代码走读推演）：null 占位 {E3_SKIPPED_BLOCKS} 格 + 外部段另发 {E3_EXT_ALLOC} 块 + '
        f'get_new_blocks {E3_POOL_NEW} 块 = 实体 {E3_REAL} 块 = 预测 {E3_NEW} ✓（外部段由 '
        'allocate_external_computed_blocks 单独发，账归 KVConnector 一章）', 8.2, lc.C_MUTE,
        'start', maxw=1130, tag='b:n2')

# ==================================================================================
# 共享刻度尺（token 轴）+ 页脚
# ==================================================================================
AXY = 1050
lc.seg(BS_X0, AXY + 16, bx(E2_REQUIRED), AXY + 16, lc.C_MUTE, 1.5)
for i in range(E2_REQUIRED + 1):
    lc.seg(bx(i), AXY + 16, bx(i), AXY + 10, lc.C_MUTE, 1.3)
for i in range(E2_REQUIRED + 1):
    lc.text(bx(i), AXY + 32, f'{i * BS}', 8.5, lc.C_MUTE, 'middle', maxw=64, tag=f'ax{i}')
lc.text(BS_X0, AXY + 6, '共享刻度尺（token 数）：上下两景的块 0-6 逐块同 x 对齐——1 块 = 16 token',
        8.6, lc.C_TXT, 'start', True, maxw=760, tag='ax:t')
lc.seg(tx(E2_PROMPT), AXY + 16, bx(E2_REQUIRED), AXY + 16, C_NUL_S, 1.2, dash=True)
lc.text((tx(E2_PROMPT) + bx(E2_REQUIRED)) / 2, AXY + 6, f'空置 {E2_TAIL_SLOTS} 槽', 7.8, lc.C_MUTE,
        'middle', maxw=BW - 20, tag='ax:tail')

lc.text(MX, AXY + 56, '逐字锚 vllm/v1/core/single_type_kv_cache_manager.py（vLLM v0.27.1）：L178 cdiv 目标表长 · '
        'L192 num_req_blocks · L202-L206 跳段（floor）· L203 num_local_computed_blocks · '
        'L210-L213 num_new_blocks · L218 num_skipped_new_computed_blocks · L223-L225 num_evictable_blocks · '
        'L230 return', 8.2, lc.C_FAINT, 'start', maxw=1380, tag='foot1')
lc.text(MX, AXY + 72, '滑窗公式取同文件 SlidingWindowManager.get_num_skipped_tokens :L1057-L1083；外部段取同文件 '
        'allocate_external_computed_blocks :L291-L326（发块公式 cdiv(num_total_computed_tokens, block_size) − '
        'len(req_blocks) 在 L323-L325）', 8.2, lc.C_FAINT, 'start', maxw=1380, tag='foot2')
lc.text(MX, AXY + 88, f'六行之外的尾巴：partial 命中（命中停在块中间）时 num_new += 1，为 CoW 私有块预留'
        f'（L226-L229）——E2 命中 {E2_HIT_TOK} = {E2_HIT_BLOCKS}×{BS}、E3 命中 {E3_HIT_TOK} = '
        f'{E3_HIT_BLOCKS}×{BS} 都整块整除，不触发；真实时机见本章「CoW 六拍」一节', 8.2, lc.C_FAINT,
        'start', maxw=1380, tag='foot3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-slowpath-blocks.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
