#!/usr/bin/env python3
"""ch13 机制图 · 慢路径六行账的 block 关系图（figure_id ch13-fig-slowpath-blocks，模板 layout）

放大自 L0『调度 · 显存账本』列（青）中 KVCacheManager/SingleTypeKVCacheManager 那一格——
本章「先数块」节六行公式的**块空间关系**：token 条带与块条带同尺度对齐，公式里
required / skipped / local_computed / new / evictable / skipped_new_computed
六个量各自长在条带的哪一段上。

claim：E2 实例（全注意力 · block_size=16 · prompt 100 token · 前缀命中 48 token = 3 整块
· 冷启动已持 0 · 命中全冷 ref_cnt==0）：required = cdiv(100,16) = 7，减已就位的
max(skipped 0, local_computed 3) 得新块 4（块 3-6，get_new_blocks 摘走），
再加被 touch 摘走的可驱逐命中块 3（块 0-2）→ return 7 = 本拍从自由队列摘走的总块数。

数字全部取自 pin 源码 vllm/v1/core/single_type_kv_cache_manager.py:L192-L230 与
本章 E2 实例（正文 L597-L604 表 + dossier supplement.A2_slow_path_formula.worked_examples[0]），
由常量计算得出、脚本内 assert 逐项对账，零即兴数字。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

# ================= E2 实例（唯一数据源；全部由常量算出） =================
BS = 16                      # block_size（1 块 = 16 token）
PROMPT_TOK = 100             # prompt token 数
HIT_TOK = 48                 # 前缀命中 48 token
HELD_BLOCKS = 0              # 已持块（冷启动）
HIT_BLOCKS = HIT_TOK // BS   # 3：命中天然以块为单位
NEW_TOK = PROMPT_TOK - HIT_TOK          # 52：本拍要算
REQUIRED = -(-PROMPT_TOK // BS)         # 7：cdiv 目标表长（L178）
SKIPPED_TOK = 0                         # 全注意力恒 0（L202）
SKIPPED_BLOCKS = SKIPPED_TOK // BS      # 0（L206，floor）
LOCAL_COMPUTED = HIT_BLOCKS + HELD_BLOCKS           # 3（L203）
NEW_BLOCKS = max(REQUIRED - max(SKIPPED_BLOCKS, LOCAL_COMPUTED), 0)   # 4（L210-L213）
SKIPPED_NEW = max(0, SKIPPED_BLOCKS - HELD_BLOCKS)  # 0（L218）
EVICTABLE = HIT_BLOCKS                              # 3：命中 3 块全冷 ref_cnt==0（L223-L225）
RET = NEW_BLOCKS + EVICTABLE                        # 7（L230）
TAIL_SLOTS = REQUIRED * BS - PROMPT_TOK             # 12：第 7 块空置槽
# 逐项对账（对不上就崩，防手滑改数）
assert (REQUIRED, HIT_BLOCKS, LOCAL_COMPUTED, NEW_BLOCKS, SKIPPED_NEW, EVICTABLE, RET, TAIL_SLOTS) \
       == (7, 3, 3, 4, 0, 3, 7, 12), 'E2 数字对不上 spec'
assert NEW_BLOCKS + HIT_BLOCKS == REQUIRED, '守恒：新块 + 命中块 = 目标表长'

lc.reset()

W, H = 1500, 878
MX, BXR = 60, 1440
CW = BXR - MX

# 尺度：块宽 = 16 token 宽；token 条带与块条带共用，逐块竖直对齐
BW = 128                     # 每块 px
PX_TOK = BW / BS             # 8 px / token
BS_X0 = 250                  # 块条带左缘 = token 条带左缘
BS_X1 = BS_X0 + REQUIRED * BW           # 1146
TOK_X1 = BS_X0 + PROMPT_TOK * PX_TOK    # 1050
HIT_X1 = BS_X0 + HIT_TOK * PX_TOK       # 634（命中段右缘 = 块 2 右缘）

C_HIT_S, C_HIT_F = lc.C_ENG_S, lc.C_ENG_F     # 命中块（暖）= 已在池中、复用它人
C_NEW_S, C_NEW_F = lc.C_KV_S, lc.C_KV_F       # 新块（青）= 显存账本新分配

EXTRA_DEFS = '<defs></defs>'   # 本图复用 lc.DEFS 的箭头 marker；无自有图样

# ---------------- 标题区 ----------------
lc.text(MX, 34, f'慢路径六行账：required {REQUIRED} 减已就位 {LOCAL_COMPUTED} 得新块 {NEW_BLOCKS}，'
                f'再加可驱逐 {EVICTABLE} —— 本拍从自由队列摘走 {RET} 块',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '六个量各长在条带的哪一段：token 条带（五段劈开）与块条带同尺度逐块对齐——'
                '命中块 touch 摘走、新块 get_new_blocks 摘走，两者都离开自由队列',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L0 · 显存账本列 · 先数块节'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= ① token 域 =================
AY, AH = 86, 176
lc.rect(MX, AY, CW, AH, '#ffffff', C_HIT_S, rx=8, sw=1.5)
lc.text(MX + 16, AY + 24, '① token 域：E2 请求的 100 token 五段条带（与下方块条带同尺度，'
                          '每 16 token 一道块界）', 11.5, lc.C_TXT, 'start', True, maxw=900, tag='a:h')
lc.text(BXR - 16, AY + 24, f'本拍入账 num_tokens = {HIT_TOK} + {NEW_TOK} = {PROMPT_TOK} token',
        9.5, C_NEW_S, 'end', maxw=330, tag='a:r')

TB_Y, TB_H = AY + 58, 48                       # 144..192
lc.text(MX + 16, TB_Y + 20, 'token 序列', 9.5, lc.C_TXT, 'start', True, maxw=160, tag='a:rl1')
lc.text(MX + 16, TB_Y + 36, f'{PROMPT_TOK} token', 8.5, lc.C_MUTE, 'start', maxw=160, tag='a:rl2')
# 命中段（new_comp）与要算段（new）
lc.rect(BS_X0, TB_Y, HIT_X1 - BS_X0, TB_H, C_HIT_F, C_HIT_S, rx=3, sw=1.6, dash=True)
lc.rect(HIT_X1, TB_Y, TOK_X1 - HIT_X1, TB_H, C_NEW_F, C_NEW_S, rx=3, sw=1.6)
lc.text((BS_X0 + HIT_X1) / 2, TB_Y + 30, f'new_comp 命中 {HIT_TOK} token = {HIT_BLOCKS} 整块',
        10, C_HIT_S, 'middle', True, maxw=HIT_X1 - BS_X0 - 20, tag='a:s1')
lc.text((HIT_X1 + TOK_X1) / 2, TB_Y + 30, f'new 本拍要算 {NEW_TOK} token',
        10, '#155e75', 'middle', True, maxw=TOK_X1 - HIT_X1 - 20, tag='a:s2')
# 块界刻度（每 16 token 一道）+ token 轴读数
for i in range(1, REQUIRED):
    bx = BS_X0 + i * BS * PX_TOK
    lc.seg(bx, TB_Y + TB_H, bx, TB_Y + TB_H + 7, lc.C_FAINT, 1.1)
for tx, lab in ((BS_X0, '0'), (HIT_X1, f'{HIT_TOK}'), (TOK_X1, f'{PROMPT_TOK}')):
    lc.text(tx, TB_Y + TB_H + 20, lab, 9, lc.C_MUTE, 'middle', maxw=60, tag='a:ax' + lab)
lc.text(MX + 16, AY + AH - 16,
        f'五段实况：comp 已持 {HELD_BLOCKS} ｜ new_comp 命中 {HIT_BLOCKS} 块（{HIT_TOK} token）｜ '
        f'ext_comp 外部 0 ｜ new 本拍要算 {NEW_TOK} token ｜ lookahead 前瞻 0',
        9.5, lc.C_MUTE, 'start', maxw=CW - 32, tag='a:five')

# ================= ② 块域 =================
BY, BH = 274, 196
lc.rect(MX, BY, CW, BH, '#ffffff', C_NEW_S, rx=8, sw=1.5)
lc.text(MX + 16, BY + 24, f'② 块域：同一序列切进 {REQUIRED} 块（1 块 = {BS} token · '
                          f'{REQUIRED} 块 = {REQUIRED * BS} 槽）——块 0-2 命中、块 3-6 新分配',
        11.5, lc.C_TXT, 'start', True, maxw=900, tag='b:h')
lc.text(BXR - 16, BY + 24, '块界与上方 token 刻度逐道对齐', 9.5, lc.C_MUTE, 'end',
        maxw=300, tag='b:r')

BK_Y, BK_H = BY + 66, 66                       # 340..406
lc.text(MX + 16, BK_Y + 20, '块条带', 9.5, lc.C_TXT, 'start', True, maxw=160, tag='b:rl1')
lc.text(MX + 16, BK_Y + 36, f'{REQUIRED} 块 · {REQUIRED * BS} 槽', 8.5, lc.C_MUTE, 'start',
        maxw=160, tag='b:rl2')
for i in range(REQUIRED):
    bx = BS_X0 + i * BW
    hit = i < HIT_BLOCKS
    lc.rect(bx, BK_Y, BW, BK_H, C_HIT_F if hit else C_NEW_F,
            C_HIT_S if hit else C_NEW_S, rx=3, sw=1.6, dash=hit)
# 第 7 块右段的空置槽（token 100..111 未用）——淡灰底 + 虚线分界，块内文字后画保持清晰
lc.rect(TOK_X1, BK_Y, BS_X1 - TOK_X1, BK_H, '#f1f5f9', 'none', rx=3, sw=0)
lc.seg(TOK_X1, BK_Y + 6, TOK_X1, BK_Y + BK_H - 6, lc.C_FAINT, 1.2, dash=True)
for i in range(REQUIRED):                     # 块内文字统一后画，防被底纹盖住
    bx = BS_X0 + i * BW
    hit = i < HIT_BLOCKS
    lc.text(bx + BW / 2, BK_Y + 24, f'块 {i}', 13, C_HIT_S if hit else C_NEW_S, 'middle', True,
            maxw=BW - 12, tag=f'b:k{i}')
    lc.text(bx + BW / 2, BK_Y + 42, f'token {i * BS}-{i * BS + BS - 1}', 8, lc.C_MUTE, 'middle',
            maxw=BW - 12, tag=f'b:t{i}')
lc.text((TOK_X1 + BS_X1) / 2, BK_Y + 58, f'空置 {TAIL_SLOTS} 槽', 8, '#64748b', 'middle',
        maxw=BS_X1 - TOK_X1 - 8, tag='b:tail')
# 尾块台座标注（右侧虚线盒）
NT_X, NT_W = BS_X1 + 20, BXR - 16 - (BS_X1 + 20)
lc.rect(NT_X, BK_Y, NT_W, BK_H, '#ffffff', lc.C_FAINT, rx=6, sw=1.1, dash=True)
lc.text(NT_X + 12, BK_Y + 20, f'块 {REQUIRED - 1} 只用到 token 96-99', 9, lc.C_TXT, 'start',
        maxw=NT_W - 24, tag='b:nt1')
lc.text(NT_X + 12, BK_Y + 38, f'尾 {TAIL_SLOTS} 槽空置（token 100-111）', 9, lc.C_TXT, 'start',
        maxw=NT_W - 24, tag='b:nt2')
lc.text(NT_X + 12, BK_Y + 56, 'cdiv 取整的零头，不是 partial', 8.5, lc.C_MUTE, 'start',
        maxw=NT_W - 24, tag='b:nt3')
# 注释盒 → 空置槽的虚线引线（两端贴边：盒左缘到块 6 右缘）
lc.seg(NT_X, BK_Y + BK_H / 2, BS_X1, BK_Y + BK_H / 2, lc.C_FAINT, 1.2, 'std', dash=True)
# 两条台座括号（无箭头，纯括注）
BR_Y = BK_Y + BK_H + 14                        # 420
for x0, x1, lab, col in ((BS_X0, HIT_X1, f'命中 {HIT_BLOCKS} 块 · touch 摘走（ref_cnt 0→1，离开自由队列）', C_HIT_S),
                         (HIT_X1, BS_X1, f'新块 {NEW_BLOCKS} 块 · get_new_blocks 摘走（真新分配）', C_NEW_S)):
    lc.seg(x0 + 4, BR_Y, x1 - 4, BR_Y, col, 1.2)
    lc.seg(x0 + 4, BR_Y - 4, x0 + 4, BR_Y + 4, col, 1.2)
    lc.seg(x1 - 4, BR_Y - 4, x1 - 4, BR_Y + 4, col, 1.2)
    lc.text((x0 + x1) / 2, BR_Y + 18, lab, 9.5, col, 'middle', maxw=x1 - x0 - 20, tag='b:br')

# ================= ③ 公式流向 =================
CY, CH = 482, 304
lc.rect(MX, CY, CW, CH, '#ffffff', lc.C_MUTE, rx=8, sw=1.5)
lc.text(MX + 16, CY + 24, f'③ 六行公式的流向：新块 {NEW_BLOCKS} = required {REQUIRED} - '
                          f'max(skipped {SKIPPED_BLOCKS}, local_computed {LOCAL_COMPUTED})；'
                          f'return {RET} = 新块 {NEW_BLOCKS} + 可驱逐 {EVICTABLE}',
        11.5, lc.C_TXT, 'start', True, maxw=1160, tag='c:h')

F_Y, F_H, F_W, F_GAP = CY + 56, 72, 232, 47
FX0 = MX + (CW - (5 * F_W + 4 * F_GAP)) / 2
FLOW = [
    (f'required = {REQUIRED}', f'目标表长 cdiv({PROMPT_TOK}, {BS})', '(L178)', lc.C_TXT, '#ffffff',
     f'条带整幅：{REQUIRED} 块'),
    ('max(skipped, local_computed)', f'减数 = max({SKIPPED_BLOCKS}, {LOCAL_COMPUTED}) = {LOCAL_COMPUTED}',
     '(L203 · L206 · L210-L213)', C_HIT_S, C_HIT_F,
     f'已就位 = 命中 {HIT_BLOCKS} + 已持 {HELD_BLOCKS}'),
    (f'num_new = {NEW_BLOCKS}', f'{REQUIRED} - {LOCAL_COMPUTED} = {NEW_BLOCKS}', '(L210-L213)',
     C_NEW_S, C_NEW_F, f'青段：块 {HIT_BLOCKS}-{REQUIRED - 1}'),
    (f'num_evictable = {EVICTABLE}', '加项：命中块 ref_cnt==0', '(L223-L225)', C_HIT_S, C_HIT_F,
     f'暖段：块 0-{HIT_BLOCKS - 1} 全冷'),
    (f'return = {RET}', f'{NEW_BLOCKS} + {EVICTABLE} = {RET}', '(L230)', C_NEW_S, C_NEW_F,
     f'本拍自由队列净减 {RET} 块'),
]
for i, (t1, t2, t3, col, fill, src) in enumerate(FLOW):
    bx = FX0 + i * (F_W + F_GAP)
    lc.rect(bx, F_Y, F_W, F_H, fill, col, rx=6, sw=1.6)
    lc.text(bx + F_W / 2, F_Y + 24, t1, 12, col, 'middle', True, maxw=F_W - 18, tag=f'c:t1{i}')
    lc.text(bx + F_W / 2, F_Y + 44, t2, 9.5, lc.C_TXT, 'middle', maxw=F_W - 18, tag=f'c:t2{i}')
    lc.text(bx + F_W / 2, F_Y + 62, t3, 8, lc.C_MUTE, 'middle', maxw=F_W - 18, tag=f'c:t3{i}')
    lc.text(bx + F_W / 2, F_Y + F_H + 20, src, 9, col, 'middle', maxw=F_W - 6, tag=f'c:src{i}')
    if i < 4:
        ax0, ax1 = bx + F_W, bx + F_W + F_GAP
        lc.seg(ax0, F_Y + F_H / 2, ax1, F_Y + F_H / 2, lc.C_MUTE, 1.6, 'std')
# 运算号（箭头上方，落在框间空隙里）
for i, op in ((0, '-'), (2, '+')):
    gx = FX0 + i * (F_W + F_GAP) + F_W + F_GAP / 2
    lc.text(gx, F_Y + F_H - 46, op, 15, lc.C_TXT, 'middle', True, maxw=F_GAP - 6, tag='c:op')

# 内层三量（逐项代入 E2）
lc.text(MX + 16, CY + 168, '内层量逐项代入（E2 值）：', 9.5, lc.C_MUTE, 'start', maxw=300, tag='c:subh')
P_Y, P_H, P_GAP = CY + 178, 62, 29
P_W = (CW - 32 - 2 * P_GAP) / 3
P_X0 = MX + 16
PARAMS = [
    (f'skipped = {SKIPPED_TOK} // {BS} = {SKIPPED_BLOCKS}', C_HIT_S,
     '全注意力：没有窗外 token，不换 null 占位块', '滑窗/块内局部注意力才有跳段（E3 例）'),
    (f'local_computed = 命中 {HIT_BLOCKS} + 已持 {HELD_BLOCKS} = {LOCAL_COMPUTED}', C_HIT_S,
     '已就位的实体块：早分配过或由 touch 复用', '都不走 get_new_blocks'),
    (f'skipped_new_computed = max(0, {SKIPPED_BLOCKS} - {HELD_BLOCKS}) = {SKIPPED_NEW}', C_HIT_S,
     '命中块落在跳段里的部分：从可驱逐计数剔除', 'E2 无跳段，剔除集为空'),
]
for i, (t1, col, l1, l2) in enumerate(PARAMS):
    px = P_X0 + i * (P_W + P_GAP)
    lc.rect(px, P_Y, P_W, P_H, C_HIT_F, lc.C_FAINT, rx=5, sw=1.1)
    lc.text(px + 12, P_Y + 20, t1, 9.5, col, 'start', True, maxw=P_W - 24, tag=f'c:p{i}a')
    lc.text(px + 12, P_Y + 38, l1, 8.5, lc.C_TXT, 'start', maxw=P_W - 24, tag=f'c:p{i}b')
    lc.text(px + 12, P_Y + 53, l2, 8.5, lc.C_MUTE, 'start', maxw=P_W - 24, tag=f'c:p{i}c')

# CoW 尾巴（六行之外的虚线注）
CW_Y, CW_H = CY + 254, 38
lc.rect(MX + 16, CW_Y, CW - 32, CW_H, '#ffffff', lc.C_ABORT, rx=6, sw=1.2, dash=True)
lc.text(MX + 32, CW_Y + 24,
        f'六行之外的尾巴：partial 命中（命中停在块中间）时 num_new += 1，为 CoW 私有块预留——'
        f'E2 命中 {HIT_TOK} = {HIT_BLOCKS}×{BS}（整块整除），不触发；真实发生时机后文「CoW 六拍」一节单讲（L226-L229）',
        9, lc.C_ABORT, 'start', maxw=CW - 64, tag='c:cow')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = CY + CH + 26
lx = MX
for col, fill, name, dash in ((C_HIT_S, C_HIT_F, '命中块 · touch 摘走（不新分配，但离开自由队列）', True),
                              (C_NEW_S, C_NEW_F, '新块 · get_new_blocks 摘走（真新分配）', False),
                              (lc.C_FAINT, '#f1f5f9', f'尾块空置槽（{TAIL_SLOTS} 槽未用）', True),
                              (lc.C_ABORT, '#ffffff', 'CoW 尾巴（partial 命中，虚线注）', True)):
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, col, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=330, tag='leg' + name[:4])
    lx += 26 + lc.tw(name, 8.5) + 22

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/single_type_kv_cache_manager.py:L192-L230（慢路径六行账）：'
        'L178 cdiv 目标表长 · L202-L206 跳段 · L203 local_computed · L210-L213 新块 · '
        'L218 skipped_new · L223-L225 可驱逐 · L230 返回值', 8.2, lc.C_FAINT, 'start',
        maxw=CW, tag='foot1')
lc.text(MX, LEG_Y + 42, f'E2 实例（全注意力 · block_size={BS} · prompt {PROMPT_TOK} token · '
        f'命中 {HIT_TOK} token = {HIT_BLOCKS} 整块 · 冷启动已持 {HELD_BLOCKS} · 命中全冷 ref_cnt==0）'
        f'→ 预测 {RET} = 新块 {NEW_BLOCKS} + 可驱逐 {EVICTABLE}；分配侧对账：touch {HIT_BLOCKS} 块 + '
        f'get_new_blocks {NEW_BLOCKS} 块 = 自由队列净减 {RET} 块 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=CW, tag='foot2')

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
