#!/usr/bin/env python3
"""ch20 论文精髓图 ③ · paper-fig-3(arXiv:2307.08691 Fig.3 忠实重绘)

writer figure-requests.json add:FA-2 前向里 warp 间工作划分的对比——左 (a) FA 的 split-K:
K 维切给 4 个 warp 各算一段 QK^T,部分结果写 shared memory 同步相加;右 (b) FA-2 的
split-Q:切 Q,每个 warp 独立算完自己那几行的完整输出,warp 间零通信。

原图真相源(arXiv e-print 2307.08691 源码 figs/flash_partitioning.png +
figs/flash2_partitioning.png,亲眼看图核对布局):
- (a):Q 竖长条(全体 warp 共享);K^T 横条切成 4 色(warp 0-3)、V 横条切成 4 色;
  各 warp 的 QK^T 切片汇入 "shared memory" 框(4 色部分和竖叠,写→同步→相加),
  再出 O_i 写入右侧 O 竖条;warp 标签标在色块下。
- (b):Q 竖长条切成 4 色(各 warp 认领一段行块);K^T、V 整条不切(全体可访);
  各 warp 独立通路 Q_i → Q_i K^T → ·V → O_i 直写右侧 O 竖条(4 色行块);无跨 warp 箭头。
- §3.3 原话(split-K):"all warps need to write their intermediate results out to shared
  memory, synchronize, then add up the intermediate results … slow down the forward pass";
  (split-Q):"split Q across 4 warps while keeping K and V accessible by all warps …
  There is no need for communication between warps"。

布局与信息结构对齐原图;配色/字体套本书视觉语言;文字译中。provenance=原论文图本身。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 740
MX = 42
BXR = 1458
WARP = [('#0d9488', '#ccfbf1'),    # warp 0 = teal
        ('#d97706', '#fef3c7'),    # warp 1 = amber
        ('#dc2626', '#fee2e2'),    # warp 2 = red
        ('#7c3aed', '#ede9fe')]    # warp 3 = violet
C_Q = lc.C_API_S                     # Q 相关 = 蓝
C_RED = '#dc2626'
EXTRA_DEFS = ''.join(
    f'<marker id="w{i}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{c}"/></marker>'
    for i, (c, _) in enumerate(WARP))

# ---------------- 标题区 ----------------
lc.text(MX, 32, 'warp 分工:FA 切 K(split-K,要共享内存同步)vs FA-2 切 Q(split-Q,零通信)',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 56, '重绘自 arXiv:2307.08691 Fig.3(前向传播):同一个线程块里 4 个 warp 怎么切活——切 K 维各算一段、部分和经 shared memory 相加;还是切 Q 各管一段行、各自直通输出',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = 'primer · 论文精髓图重绘'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

PY, PH = 96, 470                      # 两面板带(96-566)


def warp_tag(x, y, i, anchor='middle'):
    cst, _ = WARP[i]
    lc.text(x, y, 'warp %d' % i, 8, cst, anchor, True, maxw=54, tag='warp%d' % i)


# ================= (a) FlashAttention:split-K =================
AX_, AW = MX, 690
lc.rect(AX_, PY, AW, PH, '#ffffff', C_RED, rx=10, sw=1.6)
lc.text(AX_ + 16, PY + 22, '(a) FlashAttention:split-K——切 K、V,部分和相加要同步',
        11.5, C_RED, 'start', True, maxw=AW - 32, tag='a:t')
lc.text(AX_ + 16, PY + 40, '每个 warp 拿 K、V 的一段,Q 全体共享;各算一段 QK^T,乘各自 V 段得部分输出',
        8.5, lc.C_MUTE, 'start', maxw=AW - 28, tag='a:sub')

# Q 竖条(全体共享,不切)
AQX, AQW = AX_ + 30, 32
AQY, AQH = PY + 76, 300              # y 172-472
for r in range(6):
    lc.rect(AQX, AQY + r * (AQH / 6), AQW, AQH / 6 - 3, '#dbeafe', '#cbd5e1', rx=2, sw=0.8)
lc.text(AQX + AQW / 2, AQY + AQH + 18, '(全体共享)', 8, lc.C_MUTE, 'middle', maxw=70, tag='a:qs')
lc.text(AQX + AQW / 2, AQY - 14, 'Q', 12, C_Q, 'middle', True, maxw=30, tag='a:q')

# K^T 横条:切 4 段(warp 0-3)
AKX, AKY = AX_ + 140, PY + 78        # x 182-602, y 174-204
AKW, AKH = 420, 30
seg_w = AKW / 4
kc = []
for i, (cst, cfl) in enumerate(WARP):
    lc.rect(AKX + i * seg_w, AKY, seg_w - 4, AKH, cfl, cst, rx=3, sw=1.4)
    kc.append(AKX + i * seg_w + seg_w / 2)
    warp_tag(kc[i], AKY - 8, i)
lc.text(AKX - 8, AKY + AKH / 2 + 3, 'K^T', 10.5, lc.C_TXT, 'end', True, maxw=44, tag='a:k')
lc.text(AKX, AKY - 22, 'K^T(按 K 切 4 段)', 8, lc.C_MUTE, 'start', maxw=150, tag='a:kcap')

# QK^T 切片行(各 warp 算一段):x 182-602, y 240-272
QK_Y = PY + 144
lc.text(AKX, QK_Y - 8, 'QK^T(按 K 切,各 warp 一段)', 8, lc.C_MUTE, 'start', maxw=190,
        tag='a:qkcap')
for i, (cst, cfl) in enumerate(WARP):
    lc.rect(AKX + i * seg_w, QK_Y, seg_w - 4, 32, cfl, cst, rx=3, sw=1.2)
    lc.text(AKX + i * seg_w + seg_w / 2, QK_Y + 20, 'QK^T', 8.5, cst, 'middle', True,
            maxw=90, tag='a:qk%d' % i)
# Q → QK^T(全体共享同一条 Q)
lc.parrow([(AQX + AQW + 2, QK_Y + 16), (AKX - 4, QK_Y + 16)], C_Q, 2.0, 'std')

# shared memory 框:4 色部分和 + 同步相加
SMX, SMY, SMW, SMH = AX_ + 438, PY + 204, 160, 130   # x 480-640, y 300-430
lc.rect(SMX, SMY, SMW, SMH, '#fff7ed', '#9a3412', rx=6, sw=1.8)
lc.text(SMX - 10, SMY + 40, 'shared memory', 9, '#9a3412', 'end', True, maxw=90,
        tag='a:smt')
lc.text(SMX - 10, SMY + 56, '(共享内存)', 8.5, '#9a3412', 'end', maxw=90, tag='a:smt2')
pb_c = []
for i, (cst, cfl) in enumerate(WARP):
    bx_ = SMX + 14 + i * 34
    lc.rect(bx_, SMY + 16, 28, 46, cfl, cst, rx=3, sw=1.2)
    pb_c.append(bx_ + 14)
    # QK^T 切片 i → 部分和 i
    lc.parrow([(kc[i], QK_Y + 34), (kc[i], SMY - 20), (pb_c[i], SMY - 20), (pb_c[i], SMY - 2)],
              cst, 1.6, 'w%d' % i)
lc.text(SMX + SMW / 2, SMY + SMH - 24, '写中间结果 → 同步 → 相加', 8.5, '#9a3412',
        'middle', True, maxw=SMW - 12, tag='a:sync')

# V 横条:切 4 段 + 细箭头汇入 SM(各 warp 乘自己的 V 段)
AVX, AVY = AX_ + 140, PY + 386       # y 482-512
for i, (cst, cfl) in enumerate(WARP):
    lc.rect(AVX + i * seg_w, AVY, seg_w - 4, AKH, cfl, cst, rx=3, sw=1.4)
    warp_tag(AVX + i * seg_w + seg_w / 2, AVY + AKH + 14, i)
    lc.parrow([(AVX + i * seg_w + seg_w / 2, AVY - 2), (AVX + i * seg_w + seg_w / 2, SMY + SMH + 18),
               (pb_c[i], SMY + SMH + 18), (pb_c[i], SMY + SMH + 2)], cst, 1.3, 'w%d' % i)
lc.text(AVX - 8, AVY + AKH / 2 + 3, 'V', 10.5, lc.C_TXT, 'end', True, maxw=30, tag='a:v')

# SM → O_i(Σ 相加得整行)
AOX = AX_ + 618                       # x 660-692
AOY = PY + 76
for r in range(6):
    lc.rect(AOX, AOY + r * (AQH / 6), AQW, AQH / 6 - 3, '#f1f5f9', '#cbd5e1', rx=2, sw=0.8)
lc.rect(AOX, AOY + 2.5 * (AQH / 6), AQW, AQH / 6 - 3, '#bbf7d0', lc.C_GPU_S, rx=2, sw=1.6)
lc.parrow([(SMX + SMW + 2, SMY + SMH / 2 - 45), (AOX - 4, SMY + SMH / 2 - 45)], '#9a3412', 2.0, 'w0')
lc.text(AOX + AQW / 2, AOY - 30, '(Σ 后整行)', 8, lc.C_MUTE, 'middle', maxw=70, tag='a:os')
lc.text(AOX + AQW / 2, AOY - 14, 'O', 12, lc.C_GPU_S, 'middle', True, maxw=30, tag='a:o')
# 代价注(红)
lc.text(AX_ + 16, PY + 452, 'shared memory 读/写拖慢前向(§3.3:write out, synchronize, then add up)',
        8.5, C_RED, 'start', True, maxw=560, tag='a:cost')

# ================= (b) FlashAttention-2:split-Q =================
BX_, BW = 780, 678
lc.rect(BX_, PY, BW, PH, '#ffffff', lc.C_GPU_S, rx=10, sw=1.6)
lc.text(BX_ + 16, PY + 22, '(b) FlashAttention-2:split-Q——切 Q,各 warp 一条龙直通输出',
        11.5, lc.C_GPU_S, 'start', True, maxw=BW - 32, tag='b:t')
lc.text(BX_ + 16, PY + 40, 'K、V 整条全体共享;每个 warp 认领一段 Q 行块,独立算完 QK^T → softmax → ·V → 自己那几行 O',
        8.5, lc.C_MUTE, 'start', maxw=BW - 28, tag='b:sub')

# Q 竖条:切 4 段(行块)
BQX, BQY = BX_ + 60, PY + 96
BQW, BQH = 44, 300
for i, (cst, cfl) in enumerate(WARP):
    lc.rect(BQX, BQY + i * (BQH / 4), BQW, BQH / 4 - 4, cfl, cst, rx=3, sw=1.4)
    warp_tag(BQX - 10, BQY + i * (BQH / 4) + BQH / 8 + 8, i, 'end')
lc.text(BQX + BQW / 2, BQY - 12, 'Q', 12, C_Q, 'middle', True, maxw=30, tag='b:q')
lc.text(BQX + BQW / 2, BQY - 28, '(按行切 4 段)', 8, lc.C_MUTE, 'middle', maxw=90, tag='b:qs')

# K^T / V 整条(不切,全体共享)
BKY, BKH = PY + 66, 26
lc.rect(BX_ + 200, BKY, 380, BKH, '#f1f5f9', '#94a3b8', rx=3, sw=1.2)
lc.text(BX_ + 200 + 190, BKY + BKH / 2 + 3, 'K^T(整条,全体 warp 共享)', 8.5, lc.C_MUTE,
        'middle', True, maxw=360, tag='b:k')
BVY = PY + 384
lc.rect(BX_ + 200, BVY, 380, BKH, '#f1f5f9', '#94a3b8', rx=3, sw=1.2)
lc.text(BX_ + 200 + 190, BVY + BKH / 2 + 3, 'V(整条,全体 warp 共享)', 8.5, lc.C_MUTE,
        'middle', True, maxw=340, tag='b:v')

# 4 条独立通路:Q_i → QK^T → ·V → O_i(各 warp 一行)
for i, (cst, cfl) in enumerate(WARP):
    ly = PY + 130 + i * 66
    # 通路小盒链
    lc.rect(BX_ + 200, ly, 96, 34, cfl, cst, rx=4, sw=1.2)
    lc.text(BX_ + 200 + 48, ly + 21, 'Q_i K^T', 9, cst, 'middle', True, maxw=88,
            tag='b:p%d' % i)
    lc.rect(BX_ + 340, ly, 86, 34, cfl, cst, rx=4, sw=1.2)
    lc.text(BX_ + 340 + 43, ly + 21, 'softmax', 9, cst, 'middle', True, maxw=78,
            tag='b:s%d' % i)
    lc.rect(BX_ + 470, ly, 86, 34, cfl, cst, rx=4, sw=1.2)
    lc.text(BX_ + 470 + 43, ly + 21, '· V', 9, cst, 'middle', True, maxw=78, tag='b:v%d' % i)
    # Q_i → 链头
    lc.parrow([(BQX + BQW + 2, BQY + i * (BQH / 4) + BQH / 8), (BX_ + 160, BQY + i * (BQH / 4) + BQH / 8), (BX_ + 160, ly + 17), (BX_ + 198, ly + 17)], cst, 1.6, 'w%d' % i)
    # 链内箭头
    lc.parrow([(BX_ + 296, ly + 17), (BX_ + 338, ly + 17)], cst, 1.6, 'w%d' % i)
    lc.parrow([(BX_ + 426, ly + 17), (BX_ + 468, ly + 17)], cst, 1.6, 'w%d' % i)
# O 竖条(右):4 色行块
BOX, BOY = BX_ + 600, PY + 96
BOW, BOH = 44, 300
for i, (cst, cfl) in enumerate(WARP):
    lc.rect(BOX, BOY + i * (BOH / 4), BOW, BOH / 4 - 4, cfl, cst, rx=3, sw=1.4)
    ly = PY + 130 + i * 66
    lc.parrow([(BX_ + 556, ly + 17), (BOX - 8, ly + 17), (BOX - 8, BOY + i * (BOH / 4) + (BOH / 4 - 4) / 2 + 3), (BOX - 2, BOY + i * (BOH / 4) + (BOH / 4 - 4) / 2 + 3)], cst, 1.6, 'w%d' % i)
lc.text(BOX + BOW / 2, BOY - 12, 'O', 12, lc.C_GPU_S, 'middle', True, maxw=30, tag='b:o')
lc.text(BOX + BOW / 2, BOY - 28, '(O_i 各行各写)', 8, lc.C_MUTE, 'middle', maxw=90,
        tag='b:os')
# 零通信注(绿)
lc.text(BX_ + 200, PY + 430, 'warp 间零通信(§3.3:There is no need for communication between warps)——省掉全部 shared-memory 往返',
        8.5, '#166534', 'start', True, maxw=560, tag='b:zero')

# ---------------- 中缝对比标 ----------------
lc.parrow([(AX_ + AW + 3, PY + PH / 2), (BX_ - 5, PY + PH / 2)], lc.C_MUTE, 2.4, dash=True)
lc.text((AX_ + AW + BX_) / 2, PY + PH / 2 - 34, '切法', 9.5, lc.C_TXT, 'middle', True,
        maxw=40, tag='mid:t')
lc.text((AX_ + AW + BX_) / 2, PY + PH / 2 - 18, '对调', 9.5, lc.C_TXT, 'middle', True,
        maxw=40, tag='mid:t2')

# ---------------- 页脚:图例 + 出处 ----------------
LY = 580
lc.text(MX, LY, '图例:四色 = warp 0-3 各自认领的块/通路 · 蓝竖条 = Q · 灰横条 = K^T / V(不切)· 橙底框 = shared memory(仅 (a) 有)· 绿竖条 = O',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, LY + 18, '(a) split-K:all warps need to write their intermediate results out to shared memory, synchronize, then add up——shared memory 读写拖慢前向 · (b) split-Q:split Q across 4 warps while keeping K and V accessible by all warps——no need for communication(§3.3 原话)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, LY + 36, '重绘自 arXiv:2307.08691 Fig.3:Work partitioning between different warps in the forward pass · 与 ch20-fig-fa2-loop-order(线程块级循环序)互补:那张讲 block 间、这张讲 block 内 warp 间 · provenance = 论文原图',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-3.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
