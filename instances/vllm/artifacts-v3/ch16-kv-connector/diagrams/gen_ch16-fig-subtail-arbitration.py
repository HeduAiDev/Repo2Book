#!/usr/bin/env python3
"""ch16 机制图 3 · 双命中仲裁·子块尾（figure_spec ch16-fig-subtail-arbitration，模板 before-after）

放大自 L0「KV 账本列·缓存命中格」（本章 l0_zoom）、L2 站 4（双命中仲裁——
ch15 第 9 站留下的『→ ch16』路标在此接上）。

claim：远端严格超过本地整块命中时，truncate_computed_blocks 把本地子块尾砍掉、
让远端加载整块覆盖——免掉对半满共享块的 CoW；不严格更长则保尾、外部加载为零。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：per_group_hits
[40, 32]；场景 A 呈 32/远端 16>8/32+16=48/本拍 8/子块尾块 ref_cnt=0；场景 B 远端
8 不>8/外部 0/回退全组一致边界 32/本拍 24）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 54
BXR = 1446

TOK = 16.0          # px / token：56 token → 896px
RULE_X = 286        # 尺子起点
NTOK = 56
RULE_W = NTOK * TOK

C_LOCAL = lc.C_KV_S          # 本地整块命中（深青）
C_TAILF, C_TAILS = lc.C_KV_F, lc.C_KV_S   # 子块尾（浅青虚线=半满块）
C_REMOTE = lc.C_GPU_S        # 远端加载（绿）
C_NEW = lc.C_BEAT_S          # 本拍新算（橙）
C_NEWT = lc.C_BEAT_T
C_MISSF, C_MISSS = '#e2e8f0', lc.C_MUTE   # 未命中（灰）


def seg_rect(x, y, ntok, fill, stroke, dash=False, sw=1.2):
    lc.rect(x, y, ntok * TOK, 34, fill, stroke, rx=3, sw=sw, dash=dash)


def seg_label(x, y, ntok, num, name, tcol, ncol):
    cx = x + ntok * TOK / 2
    lc.text(cx, y + 15, num, 11, tcol, 'middle', True, maxw=ntok * TOK - 6, tag='sg:n')
    lc.text(cx, y + 28, name, 8, ncol, 'middle', maxw=ntok * TOK - 6, tag='sg:t')


def block_ticks(y, h):
    """块边界刻线（每 16 token 一块）+ 块号标签。"""
    for b in range(NTOK // 16 + 1):
        tx = RULE_X + b * 16 * TOK
        if 0 < b < NTOK // 16:
            lc.seg(tx, y - 4, tx, y + h + 4, '#cbd5e1', 0.9)
    for b in range(NTOK // 16):
        lc.text(RULE_X + b * 16 * TOK + 8 * TOK, y + h + 16, f'块{b}', 8.5, lc.C_FAINT,
                'middle', maxw=13 * TOK, tag=f'blk{b}')


# ---------------- 标题区 ----------------
lc.text(MX, 36, '双命中仲裁：远端整块盖过本地子块尾时，砍尾免 CoW', 16.5, lc.C_TXT, 'start', True,
        maxw=940, tag='title')
lc.text(MX, 60, '本地链式哈希命中 40（整块 32 + 子块尾 8）× 远端外部缓存——仲裁只问一件事：远端严格超过那截子块尾吗？'
                '（scheduler.py:L791-L821）', 10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L2 站 4 双命中仲裁 · L0：KV 账本列·缓存命中格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 仲裁前（两场景共用） ----------------
BZ_Y = 100
lc.text(MX, BZ_Y, '仲裁前（两场景共用 · 56-token probe · block_size=16 · hash_block_size=8）', 11.5,
        lc.C_TXT, 'start', True, maxw=760, tag='bz:t')

R1_Y = BZ_Y + 26
lc.text(RULE_X - 12, R1_Y + 21, 'full 组 命中 40', 10, lc.C_TXT, 'end', True, maxw=118, tag='r1:l')
x = RULE_X
seg_rect(x, R1_Y, 32, C_LOCAL, C_LOCAL)
seg_label(x, R1_Y, 32, '32', '本地整块命中', '#ffffff', '#cffafe')
x += 32 * TOK
seg_rect(x, R1_Y, 8, C_TAILF, C_TAILS, dash=True)
seg_label(x, R1_Y, 8, '8', '子块尾(半满)', C_TAILS, lc.C_MUTE)
x += 8 * TOK
seg_rect(x, R1_Y, 16, C_MISSF, C_MISSS)
seg_label(x, R1_Y, 16, '16', '未命中', lc.C_MUTE, lc.C_MUTE)
block_ticks(R1_Y, 34)

R2_Y = R1_Y + 56
lc.text(RULE_X - 12, R2_Y + 21, 'mamba 组 命中 32', 10, lc.C_TXT, 'end', True, maxw=130, tag='r2:l')
x = RULE_X
seg_rect(x, R2_Y, 32, C_LOCAL, C_LOCAL)
seg_label(x, R2_Y, 32, '32', '本地整块命中', '#ffffff', '#cffafe')
x += 32 * TOK
seg_rect(x, R2_Y, 24, C_MISSF, C_MISSS)
seg_label(x, R2_Y, 24, '24', '未命中', lc.C_MUTE, lc.C_MUTE)
lc.text(RULE_X - 12, R2_Y + 36, '（只到块边界）', 8, lc.C_MUTE, 'end', maxw=118, tag='r2:v2')

# 右侧注记
AN_X, AN_W = 1252, 194
lc.rect(AN_X, BZ_Y - 4, AN_W, 128, 'none', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(AN_X + 12, BZ_Y + 16, '两组命中长度不同（40 vs 32）', 8.5, lc.C_TXT, 'start', True,
        maxw=AN_W - 22, tag='an:1')
lc.text(AN_X + 12, BZ_Y + 32, '→ hit_diverged', 8.5, lc.C_TXT, 'start', True, maxw=AN_W - 22, tag='an:2')
lc.text(AN_X + 12, BZ_Y + 52, '块对齐闸：呈给 connector 的', 8.5, lc.C_MUTE, 'start', maxw=AN_W - 22, tag='an:3')
lc.text(AN_X + 12, BZ_Y + 66, '本地值 = 40 − 8 = 32（砍尾，', 8.5, lc.C_MUTE, 'start', maxw=AN_W - 22, tag='an:4')
lc.text(AN_X + 12, BZ_Y + 80, 'scheduler.py:L773-L781）', 8.5, lc.C_MUTE, 'start', maxw=AN_W - 22, tag='an:5')
lc.text(AN_X + 12, BZ_Y + 100, 'connector 看到的本地命中永远', 8, lc.C_FAINT, 'start', maxw=AN_W - 22, tag='an:6')
lc.text(AN_X + 12, BZ_Y + 113, '隔着一条块对齐边界', 8, lc.C_FAINT, 'start', maxw=AN_W - 22, tag='an:7')

# ---------------- 仲裁 A ----------------
PA_Y = 296
PA_H = 158
lc.rect(MX, PA_Y, BXR - MX, PA_H, '#ffffff', lc.C_KV_S, rx=9, sw=1.8)
lc.text(MX + 18, PA_Y + 26, '仲裁 A · 远端 16 严格更长（16 > 8 → truncate_computed_blocks 砍本地子块尾）', 12,
        lc.C_KV_S, 'start', True, maxw=900, tag='pa:t')
RA_Y = PA_Y + 44
x = RULE_X
seg_rect(x, RA_Y, 32, C_LOCAL, C_LOCAL)
seg_label(x, RA_Y, 32, '32', '本地整块命中（保留）', '#ffffff', '#cffafe')
x += 32 * TOK
seg_rect(x, RA_Y, 16, C_REMOTE, C_REMOTE)
seg_label(x, RA_Y, 16, '16', '远端加载·整块覆盖', '#ffffff', '#dcfce7')
x += 16 * TOK
seg_rect(x, RA_Y, 8, C_NEW, C_NEW)
seg_label(x, RA_Y, 8, '8', '本拍新算', C_NEWT, C_NEWT)
block_ticks(RA_Y, 34)
lc.text(MX + 18, PA_Y + 108, '采用 = 本地 32 + 外部 16 = 48，本拍只算 56 − 48 = 8——56 个 token 里 48 个不用算', 9.5,
        '#334155', 'start', maxw=1340, tag='pa:l1')
lc.text(MX + 18, PA_Y + 128, '被剪的子块尾块：引用数归 0、仍在哈希表——免掉对半满共享块的 CoW（一次整块拷贝带宽 + 一个新块）；远端加载从块边界起整块落位，不与本地半块混写', 9.5,
        '#334155', 'start', maxw=1340, tag='pa:l2')

# ---------------- 仲裁 B ----------------
PB_Y = PA_Y + PA_H + 18
PB_H = 176
lc.rect(MX, PB_Y, BXR - MX, PB_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.6)
lc.text(MX + 18, PB_Y + 26, '仲裁 B · 远端 8 不严格更长（8 不 > 8 → 保尾、什么都不搬）', 12,
        lc.C_MUTE, 'start', True, maxw=900, tag='pb:t')
RB_Y = PB_Y + 44
x = RULE_X
seg_rect(x, RB_Y, 32, C_LOCAL, C_LOCAL)
seg_label(x, RB_Y, 32, '32', '回退全组一致边界', '#ffffff', '#cffafe')
x += 32 * TOK
seg_rect(x, RB_Y, 24, C_NEW, C_NEW)
seg_label(x, RB_Y, 24, '24', '本拍新算', C_NEWT, C_NEWT)
block_ticks(RB_Y, 34)
lc.text(MX + 18, PB_Y + 108, '外部采用 0——不剪尾、不加载：本地缓存原样保留（full 组的 40 含子块尾仍登记在哈希表）', 9.5,
        '#334155', 'start', maxw=1340, tag='pb:l1')
lc.text(MX + 18, PB_Y + 128, '发散深命中（full 40 vs mamba 32）没有外部 KV 撑腰 → 回退两组都认的 32——恢复边界处必须有合法的 Mamba 状态，宁可少用 8 token', 9.5,
        '#334155', 'start', maxw=1340, tag='pb:l2')
lc.text(MX + 18, PB_Y + 148, '本拍算 56 − 32 = 24 · update_state_after_alloc 收到 0', 9.5,
        '#334155', 'start', maxw=1340, tag='pb:l3')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = PB_Y + PB_H + 28
lx = MX
for fill, stroke, dash, name, tcol in [
        (C_LOCAL, C_LOCAL, False, '本地整块命中', '#ffffff'),
        (C_TAILF, C_TAILS, True, '子块尾（半满块）', C_TAILS),
        (C_REMOTE, C_REMOTE, False, '远端加载（整块覆盖）', '#ffffff'),
        (C_NEW, C_NEW, False, '本拍新算', C_NEWT),
        (C_MISSF, C_MISSS, False, '未命中', lc.C_MUTE)]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=170, tag='leg')
    lx += 26 + lc.tw(name, 8.5) + 24

FY = LEG_Y + 26
lc.text(MX, FY, '逐字锚 vllm/v1/core/sched/scheduler.py:L791-L821（双命中仲裁：partial_tail=本地命中%block_size · 远端严格更长 → '
                'truncate『no CoW needed, and let the load cover it』）· vllm/v1/core/kv_cache_manager.py:L777-L794（truncate 断言 num_computed_tokens % block_size == 0）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '两组命中 [40, 32] 与 A/B 读数（48/8、ref_cnt=0、回退 32/24）取自精简版 companion host 实测（partial-hit 粒度配置承 ch15）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 36

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-subtail-arbitration.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
