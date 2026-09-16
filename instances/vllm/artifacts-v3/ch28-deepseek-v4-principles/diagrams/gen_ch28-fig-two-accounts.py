#!/usr/bin/env python3
"""ch28 机制图 · 两本账：1M 上下文的 FLOPs 账与 KV 账（ch28-fig-two-accounts）

claim：三个数字三个分母——27% FLOPs / 10% KV 的分母是 V3.2，『约 2% KV』的分母是
BF16 GQA8（head_dim=128）基线；而这笔省出来的账里滑窗是**加账**：
43 层 × 128 条 × 576B 的常数窗永远留着。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  压缩账 89.540698 B/token/层；计滑窗总账 92.711002 B/token
  分母一 4096 B/token（2×8×128×2）→ 2.1861%；分母二 656 B/token（V3.2）→ 13.6495%
  逐层摊销：CSA 179.0 B（含 IndexCache 132B/4）、不计则 146.0 B；HCA 4.5625 B
  层型 swaonly 2 / c4a 21 / c128a 20（43 层）
  滑窗常数窗 3170304 B = 43×128×576.0；1M 下摊薄 3.1703 B/token
  口径敏感性：不计 IndexCache 73.424419 B/token → 1.7926% / 11.1927%
  FLOPs 三档：纯滑窗 128 / CSA 512（top-k 封顶）/ HCA 7812（每 query 实看条目数）

配色走 book/cartography/l0_common.py 的角色常量（KV 青、GPU 绿、引擎橙）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX, BXR = 40, 1460

# ---------------- 语义配色（角色色 + 同源深浅） ----------------
C_KV_S, C_KV_F = lc.C_KV_S, lc.C_KV_F
C_KV_DEEP = '#155e75'
C_SWA_F, C_SWA_S = '#cffafe', '#0e7490'      # 滑窗（同色浅调）
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_REF_S = '#94a3b8'                          # 参考线/分母
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT

EXTRA_DEFS = ('<defs>'
              '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
              f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
              '</defs>')

# ---------------- 布局常量 ----------------
LPX0, LPX1 = MX, 726            # 左栏（FLOPs 账）
RPX0, RPX1 = 774, BXR           # 右栏（KV 账）
PAN_Y0, PAN_Y1 = 128, 620       # 两栏面板竖直范围
SWAY0, SWAY1 = 642, 800         # 滑窗常数带

AX_L, AX_R = LPX0 + 76, LPX1 - 24        # 左栏对数轴
AX2_L, AX2_R = RPX0 + 76, RPX1 - 24      # 右栏对数轴
AXT = PAN_Y1 - 96                        # 轴底线（刻度标签留在面板内）
FLOG_LO, FLOG_HI = 1.8, 4.4              # FLOPs 条数轴：10^1.8 … 10^4.4
KLOG_LO, KLOG_HI = 0.4, 3.8              # KV 字节轴：10^0.4 … 10^3.8
LOG_TICKS = (2, 3, 4)


def lx(v, lo=FLOG_LO, hi=FLOG_HI, x0=AX_L, x1=AX_R):
    import math
    t = (math.log10(v) - lo) / (hi - lo)
    return x0 + t * (x1 - x0)


# ---------------- 标题区 ----------------
lc.text(MX, 32, '两本账：1M 上下文的 FLOPs 账与 KV 账', 17, C_TXT, 'start', True,
        maxw=620, tag='title')
lc.text(MX, 56, '三个数字三个分母——27% FLOPs / 10% KV 对着 V3.2，『约 2% KV』对着 BF16 GQA8；'
                '而滑窗是两本账里唯一的加账项', 10.5, C_MUTE, 'start', maxw=1000, tag='sub')
lc.text(BXR, 32, '本图的位置', 9.5, C_MUTE, 'end', True, maxw=80, tag='l0:t')
lc.text(BXR, 50, '全景架构图（L0）里『调度 · 显存账本』的 KV 池块', 9, C_MUTE, 'end',
        maxw=420, tag='l0:a')
lc.text(BXR, 66, '＋『GPU 执行臂 · 模型层』的注意力计算块——这张图打开这两块的账', 9, C_MUTE,
        'end', maxw=470, tag='l0:b')

# ---------------- 图例（两种以上语义色） ----------------
LEG = [('kv', 'KV 账（青）：每 token 要囤多少', C_KV_F, C_KV_S),
       ('swa', '滑窗加账（浅青）：常数窗，不随上下文增长', C_SWA_F, C_SWA_S),
       ('gpu', 'FLOPs 账（绿）：每 query 要看多少条', C_GPU_F, C_GPU_S),
       ('ref', '分母/参考线（灰）：对照口径', '#f1f5f9', C_REF_S)]
lgx = MX
for k, lab, f, s in LEG:
    lc.rect(lgx, 76, 17, 12, f, s, rx=2, sw=1.1)
    lc.text(lgx + 23, 86, lab, 8.6, '#334155', 'start', maxw=232, tag='lg:%s' % k)
    lgx += 17 + 23 + lc.tw(lab, 8.6) + 16

# ================= 左栏：FLOPs 账 =================
lc.rect(LPX0, PAN_Y0, LPX1 - LPX0, PAN_Y1 - PAN_Y0, '#ffffff', C_GPU_S, rx=10, sw=1.5)
lc.text(LPX0 + 16, PAN_Y0 + 24, 'FLOPs 账：每 query 实际看多少条', 12.5, C_TXT, 'start', True,
        maxw=340, tag='lf:t')
lc.text(LPX0 + 16, PAN_Y0 + 44, '分母是 V3.2（同住 1M）：条数少了，算的量就少了——27% 是这么来的',
        9, C_MUTE, 'start', maxw=470, tag='lf:s')

# 对数轴刻度（横向：条数 → 条长）
FLOP_AX = AXT + 8
for exp in LOG_TICKS:
    x = lx(10 ** exp)
    lc.seg(x, FLOP_AX, x, AXT, '#e2e8f0', 1.2, dash=True)
    lc.text(x, AXT + 30, '10^%d' % exp, 8.5, C_MUTE, 'middle', tag='lf:tk%d' % exp)
lc.seg(AX_L, FLOP_AX, AX_R, FLOP_AX, C_REF_S, 1.4)
lc.text(AX_R, FLOP_AX + 22, '条数（对数刻度）', 8.5, C_MUTE, 'end', maxw=140, tag='lf:ax')

# 三档条
FLOP = [('纯滑窗层', '牌子 0', 128, '常数窗：与上下文长度无关', C_GPU_F, C_GPU_S, 0),
        ('CSA 层', '牌子 4', 512, 'top-k 512 封顶（候选 25 万只挑 512）', C_KV_F, C_KV_S, 1),
        ('HCA 层', '牌子 128', 7812, '一条不挑：7812 = 1000000 // 128 全看', C_KV_F, C_KV_S, 2)]
BAR_H = 24
for t1, t2, v, note, f, s, i in FLOP:
    cy = 286 + i * 74
    x1 = lx(v)
    lc.text(AX_L - 10, cy + 5, t1, 10, C_TXT, 'end', True, tag='lf:t1%d' % i)
    lc.text(AX_L - 10, cy + 20, t2, 8.4, s, 'end', tag='lf:t2%d' % i)
    lc.rect(AX_L, cy - BAR_H / 2, max(4.0, x1 - AX_L), BAR_H, f, s, rx=4, sw=1.3)
    lc.text(x1 + 10, cy + 1, '%d 条' % v, 12, s, 'start', True, tag='lf:v%d' % i)
    lc.text(AX_R, cy + 17, note, 8.4, C_MUTE, 'end', maxw=330, tag='lf:n%d' % i)
lc.text(LPX0 + 16, 236, '每 query 的实看条目数：压缩砍一刀、top-k 再砍一刀（三档对三种牌子）',
        9, C_MUTE, 'start', maxw=560, tag='lf:a0')
lc.text(LPX0 + 16, PAN_Y1 - 16, '实看越多 → 主注意力越贵；HCA 层用 7812 条换掉了整个索引器。',
        8.8, '#334155', 'start', maxw=560, tag='lf:a1')

# ================= 右栏：KV 账 =================
lc.rect(RPX0, PAN_Y0, RPX1 - RPX0, PAN_Y1 - PAN_Y0, '#ffffff', C_KV_S, rx=10, sw=1.5)
lc.text(RPX0 + 16, PAN_Y0 + 24, 'KV 账：每 token 每层要囤多少字节', 12.5, C_TXT, 'start', True,
        maxw=360, tag='rk:t')
lc.text(RPX0 + 16, PAN_Y0 + 44, '一条压缩条目的账：448 fp8 NoPE + 128 bf16 RoPE + 8 fp8 scale '
                                '= 584 B，除以压缩率就是每 token 的摊销', 9, C_MUTE, 'start',
        maxw=620, tag='rk:s')

# 上子面板：逐层摊销（对数轴）
SUB1_Y0, SUB1_Y1 = PAN_Y0 + 64, PAN_Y0 + 250
lc.text(RPX0 + 16, SUB1_Y0 + 12, '逐层摊销（B/token/层，对数刻度）', 10, C_TXT, 'start', True,
        maxw=300, tag='rk1:t')
AXT1 = SUB1_Y1 - 30
for i, (lab, v, f, s, note) in enumerate([
        ('CSA 层', 179.0, C_KV_F, C_KV_S, '584/4 + 132/4（含索引器缓存）'),
        ('HCA 层', 4.5625, C_KV_F, C_KV_S, '584/128（没有索引器对象）'),
        ('纯滑窗层', 0.0, C_SWA_F, C_SWA_S, '没有压缩机 → 压缩账为 0')]):
    cy = SUB1_Y0 + 36 + i * 30
    x = lx(v, KLOG_LO, KLOG_HI, AX2_L, AX2_R) if v > 0 else AX2_L
    lc.text(AX2_L - 8, cy + 4, lab, 9.5, C_TXT, 'end', True, tag='rk1:l%d' % i)
    lc.rect(AX2_L, cy - 8, AX2_R - AX2_L, 16, '#f8fafc', '#e2e8f0', rx=3, sw=1)
    lc.rect(AX2_L, cy - 8, max(3.0, x - AX2_L), 16, f, s, rx=3, sw=1.2)
    lc.text(x + 8, cy + 4, '%.4f B' % v if v else '0.0000 B', 10, s, 'start', True, tag='rk1:v%d' % i)
    lc.text(RPX1 - 16, cy + 4, note, 8.5, C_MUTE, 'end', maxw=280, tag='rk1:n%d' % i)
for exp in (1, 2):
    x = lx(10 ** exp, KLOG_LO, KLOG_HI, AX2_L, AX2_R)
    lc.seg(x, SUB1_Y0 + 30, x, AXT1, '#e2e8f0', 1.2, dash=True)
    lc.text(x, AXT1 + 14, '10^%d' % exp, 8.5, C_MUTE, 'middle', tag='rk1:tk%d' % exp)
lc.text(AX2_L, AXT1 + 14, '0', 8.5, C_MUTE, 'middle', tag='rk1:0')
lc.text(RPX0 + 16, SUB1_Y1 - 4, '整机加权（43 层）= (2×0 + 21×179.0 + 20×4.5625) / 43 '
                                '= 89.540698 B/token/层', 9.5, C_KV_DEEP, 'start', True,
        maxw=560, tag='rk1:agg')

# 下子面板：分母对照
SUB2_Y0, SUB2_Y1 = PAN_Y0 + 264, PAN_Y1 - 42
lc.text(RPX0 + 16, SUB2_Y0 + 12, '同一笔账，三个分母（B/token，线性刻度）', 10, C_TXT, 'start',
        True, maxw=320, tag='rk2:t')
AXT2 = SUB2_Y1 - 14
L2X0, L2X1 = RPX0 + 130, RPX1 - 96
DEN = [('本模型 89.540698', 89.540698, C_KV_F, C_KV_S, '计滑窗 92.711002'),
       ('V3.2 656', 656, C_REF_S, C_REF_S, '摘要 10% 的分母'),
       ('BF16 GQA8 4096', 4096, C_REF_S, C_REF_S, '论文『约 2%』的分母')]
for i, (lab, v, f, s, note) in enumerate(DEN):
    cy = SUB2_Y0 + 34 + i * 34
    wpx = (v / 4096.0) * (L2X1 - L2X0)
    lc.text(L2X0 - 8, cy + 4, lab, 9, C_TXT, 'end', tag='rk2:l%d' % i)
    lc.rect(L2X0, cy - 9, max(2.0, wpx), 18, f, s, rx=3, sw=1.2)
    lc.text(L2X0 + max(wpx, 2) + 8, cy + 4, note, 8.5, C_MUTE, 'start', tag='rk2:n%d' % i)
lc.text(L2X0, AXT2, '0 B/token', 8.5, C_MUTE, 'middle', tag='rk2:z')
lc.text(RPX1 - 20, SUB2_Y1 - 24, '比值：89.540698 / 4096 = 2.1861 %；89.540698 / 656 = 13.6495 %',
        9.2, C_KV_DEEP, 'end', True, maxw=620, tag='rk2:r')
lc.text(RPX1 - 20, SUB2_Y1 - 8, '两个百分比的差别全在分母——同一个分子 89.540698',
        8.6, C_MUTE, 'end', maxw=620, tag='rk2:r2')

# ================= 底部：滑窗常数带（加账） =================
lc.rect(MX, SWAY0, BXR - MX, SWAY1 - SWAY0, C_SWA_F, C_SWA_S, rx=10, sw=1.6)
lc.text(MX + 18, SWAY0 + 26, '滑窗是唯一的加账：43 层 × 128 条 × 576 B = 3170304 B 的常数窗',
        13, '#0e7490', 'start', True, maxw=520, tag='sw:t')
lc.text(MX + 18, SWAY0 + 48, '窗口长度 = min(pos+1, 128)：与压缩率 m 无关、与上下文长度无关——'
                             '它不随 token 数增长，所以长上下文里被摊薄、短上下文里反而最贵', 9.5,
        '#334155', 'start', maxw=760, tag='sw:s')
SWV = [('1M 上下文', '3170304 / 1000000', '3.1703 B/token', '占压缩账 0.035406'),
       ('1K 上下文', '3170304 / 1000', '3170.304 B/token', '同一常数窗按 1/context 放大')]
for i, (a, b, c, d) in enumerate(SWV):
    cx = MX + 60 + i * 430
    lc.text(cx, SWAY0 + 84, a, 10.5, C_TXT, 'start', True, maxw=120, tag='sw:a%d' % i)
    lc.text(cx + 104, SWAY0 + 84, b, 10, C_MUTE, 'start', tag='sw:b%d' % i)
    lc.text(cx, SWAY0 + 104, c, 11, '#0e7490', 'start', True, tag='sw:c%d' % i)
    lc.text(cx + 176, SWAY0 + 104, d, 8.5, C_MUTE, 'start', maxw=230, tag='sw:d%d' % i)
lc.text(MX + 18, SWAY0 + 130, '口径敏感性：不计索引器缓存 → 73.424419 B/token（对 GQA8 1.7926%、'
                              '对 V3.2 11.1927%）——三个比例都在个位数百分比内漂移', 9, '#155e75',
        'start', maxw=900, tag='sw:e')

# ---------------- 页脚 ----------------
lc.text(MX, 818, '数字口径：1M 上下文、43 层主干（前 2 层纯滑窗、中段 4/128 交替：21 层 c4a + 20 层 c128a）；'
                 '一条压缩条目 584 B、索引器缓存 132 B/条、滑窗行宽 576 B。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 836, '依据 DeepSeek-V4 技术报告 §2.3.4 与摘要（arXiv:2606.19348）的对照口径；'
                 '全部数值由本章驱动脚本纯算术实跑（纯乘除/整除）。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-two-accounts.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
