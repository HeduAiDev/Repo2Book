#!/usr/bin/env python3
"""ch22 机制图 3 · PAD 哨兵三处分工与两拍残留（figure_spec ch22-fig-pad-sentries，模板 state-table）

放大自 L0『GPU 执行臂』（gpu_column 绿列）与 KV 池接缝——即本章 L2 章图 center 拍片
⑤『kernel 内景 · 恒等式+PAD』的 PAD 半边 + 拍片 ⑥『双口径装配 · slot/块表尾行』的
机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：CUDA graph 回放要求形状恒为 max、地址恒不变，持久缓冲尾部会残留上一拍真数据
——PAD 程序每拍重填 [num_tokens,max) 为 -1、块表尾行填 0，每个哨兵各配消费端的
跳过逻辑。

数字全部取自 figure_spec.numbers（128/120/100/10/-1/0/74/999/6，精简版 host 实跑：
两拍残留实验 + FULL 图档一拍四件套）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 742
MX = 60
BXR = 1440

C_STALE_F, C_STALE_S = '#f1f5f9', '#cbd5e1'   # 上一拍残留（沿 ch19 padding-four 既有色语）

# ---------------- 标题区 ----------------
lc.text(MX, 34, '持久缓冲的宿命：尾部躺着上一拍真数据，PAD 每拍重填哨兵',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, 'CUDA graph 录的是 max 形状、地址永不变——拍间真 slot 还印在尾部：PAD 程序每拍重填 [num_tokens,128) 为 -1、块表尾行填 0（NULL_BLOCK_ID）、query_start_loc 尾填非递减、positions 尾清零',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L2 拍片 ⑤ 恒等式+PAD 半边 + ⑥ 双口径装配 · L0：GPU 执行臂 × KV 接缝'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 前提横幅 ----------------
BY, BH_ = 78, 44
lc.rect(MX, BY, BXR - MX, BH_, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + 16, BY + 19, '前提（ch18 固定地址 · ch19 捕获形状，均已立）：CUDA graph 回放两恒——形状恒为捕获形状 128 · 地址恒不变（持久缓冲）',
        9.3, lc.C_GPU_S, 'start', True, maxw=BXR - MX - 32, tag='ban:1')
lc.text(MX + 16, BY + 36, '两恒之间：真 token 每拍在变（本例 100 → 10；FULL 拍 120 → 128）——差额段不重写，上一拍的真数据就活着',
        9.3, '#334155', 'start', maxw=BXR - MX - 32, tag='ban:2')

# ---------------- A · 两拍残留实验 ----------------
lc.text(MX, 148, 'A · 两拍残留实验：同一个持久 slot_mapping 缓冲 [0,128)，地址不变、内容每拍全换',
        10.5, lc.C_TXT, 'start', True, maxw=900, tag='sa:t')
BX0, BX1 = 250, 1350
PER = (BX1 - BX0) / 128.0
BAR_H = 28

def slot_x(k):
    return BX0 + k * PER

def bar(y, tag):
    """三拍状态条的分段绘制"""
    lc.rect(slot_x(0), y, slot_x(128) - slot_x(0), BAR_H, '#ffffff', '#e2e8f0', rx=3, sw=1.0)

def rlabel(y, l1, l2):
    lc.text(238, y + 12, l1, 8.6, lc.C_TXT, 'end', True, maxw=176, tag='rl' + l1[:6])
    lc.text(238, y + 24, l2, 7.4, lc.C_MUTE, 'end', maxw=176, tag='rl2' + l1[:6])

# A1 拍1 写完
A1 = 162
rlabel(A1, '拍1 写完', '100 token')
bar(A1, 'a1')
lc.rect(slot_x(0), A1, slot_x(100) - slot_x(0), BAR_H, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.4)
lc.text(slot_x(50), A1 + 18, '拍1 的 100 个真 slot（slot[9]=73、slot[10]=74 …）', 8.4,
        lc.C_GPU_S, 'middle', True, maxw=slot_x(100) - slot_x(0) - 10, tag='a1:t')
lc.rect(slot_x(100), A1, slot_x(128) - slot_x(100), BAR_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.4)
lc.text(slot_x(114), A1 + 18, 'PAD 程序写 [100,128) = -1', 7.4, lc.C_BEAT_T, 'middle', True,
        maxw=slot_x(128) - slot_x(100) - 4, tag='a1:p')

# A2 拍2 开拍前
A2 = 204
rlabel(A2, '拍2 开拍前', '缓冲原样')
bar(A2, 'a2')
lc.rect(slot_x(0), A2, slot_x(10) - slot_x(0), BAR_H, '#ffffff', lc.C_GPU_S, rx=3, sw=1.2, dash=True)
lc.text(slot_x(5), A2 + 18, '拍2 将写 10', 7.0, lc.C_GPU_S, 'middle', True,
        maxw=slot_x(10) - slot_x(0) - 2, tag='a2:n')
lc.rect(slot_x(10), A2, slot_x(128) - slot_x(10), BAR_H, C_STALE_F, C_STALE_S, rx=3, sw=1.2)
lc.text(slot_x(69), A2 + 18, '拍1 残留：[10,20) 还是真 slot，[100,128) 是上一拍的 -1', 8.4,
        C_STALE_S, 'middle', True, maxw=slot_x(128) - slot_x(10) - 10, tag='a2:t')

# 放大条（开拍前 [10,20) 的真值 → 拍后全 -1）
ZY = 246
lc.text(BX0, ZY + 11, '开拍前放大 [10,20)：', 8.4, lc.C_TXT, 'start', True, maxw=140, tag='z:t')
STALE = [74, 75, 76, 77, 78, 79, 144, 145, 146, 147]
zx = BX0 + 148
for i, v in enumerate(STALE):
    lc.rect(zx + i * 34, ZY - 6, 30, 24, C_STALE_F, C_STALE_S, rx=4, sw=1.0)
    lc.text(zx + i * 34 + 15, ZY + 10, str(v), 8.6, '#64748b', 'middle', True, maxw=26,
            tag=f'z{i}')
lc.parrow([(slot_x(15), A2 + BAR_H), (zx + 15, ZY - 6)], '#94a3b8', 1.3, 'std', dash=True)
ax = zx + len(STALE) * 34 + 14
lc.text(ax, ZY + 11, '→ 拍2 写完：', 8.4, lc.C_TXT, 'start', True, maxw=90, tag='z:a')
for i in range(3):
    lc.rect(ax + 88 + i * 34, ZY - 6, 30, 24, lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.2)
    lc.text(ax + 88 + i * 34 + 15, ZY + 10, '-1', 8.6, lc.C_BEAT_T, 'middle', True, maxw=26,
            tag=f'zp{i}')
lc.text(ax + 88 + 3 * 34 + 8, ZY + 11, '……[10,128) 全部 -1，真 slot 寿命 ≤ 一拍', 8.2,
        lc.C_BEAT_T, 'start', True, maxw=300, tag='z:n')

# A3 拍2 写完
A3 = 286
rlabel(A3, '拍2 写完', '10 token')
bar(A3, 'a3')
lc.rect(slot_x(0), A3, slot_x(10) - slot_x(0), BAR_H, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.4)
lc.text(slot_x(5), A3 + 18, '拍2 真 slot', 7.0, lc.C_GPU_S, 'middle', True,
        maxw=slot_x(10) - slot_x(0) - 2, tag='a3:n')
lc.rect(slot_x(10), A3, slot_x(128) - slot_x(10), BAR_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.4)
lc.text(slot_x(69), A3 + 18, 'PAD 程序每拍重填 [10,128) = -1——上一拍残留活不过本拍', 8.4,
        lc.C_BEAT_T, 'middle', True, maxw=slot_x(128) - slot_x(10) - 10, tag='a3:t')

# ---------------- B · FULL 图档一拍的 PAD 税 ----------------
lc.text(MX, 348, 'B · FULL 图档一拍的 PAD 税（120 真 token → padded 128 · 2 请求 → 8 行）：三处哨兵 + 尾清零，各配消费端',
        10.5, lc.C_TXT, 'start', True, maxw=1100, tag='sb:t')
R_Y0, R_PITCH, NAME_W = 366, 58, 176
CELL_W, CELL_H = 30, 26

ROWS = [
    dict(name='slot_mapping 尾', span='[120,128) 填 -1', src='装配：_get_slot_mappings 尾段 fill_(-1)',
         pre=('真 120 个', 120), cells=['-1'] * 8, ccap='尾 8 项',
         cons=('写腿 kernel：slot<0 直接 return', '——PAD 不写 KV 池（cache_kernels.cu）', lc.C_GPU_S)),
    dict(name='block_table 尾行', span='[2,8) 共 6 行填 0', src='装配：_get_block_table 尾行 NULL_BLOCK_ID=0',
         pre=('活跃 2 行', 54), cells=['0'] * 6, ccap='每行 8 项全 0',
         cons=('读腿：padded 请求查块 0——全零保留块', '「Block 0 is reserved for padding」', lc.C_KV_S)),
    dict(name='query_start_loc 尾', span='尾 6 项全填 120', src='四件套之三：非递减填到最后真边界',
         pre=('真 CU 偏移 …120', 100), cells=['120'] * 6, ccap='值 = 最后真边界',
         cons=('FlashAttention 类 kernel 要求非递减', '——pad 段区间长为 0，不派发工作', lc.C_MUTE)),
    dict(name='positions 尾', span='[120,128) 残留 999 → 清 0', src='四件套之四：拍前预埋 999，拍后全 0',
         pre=None, cells=['999'] * 4 + ['→'] + ['0'] * 4, ccap='拍前 → 拍后',
         cons=('位置型 kernel 对 pad 行算垃圾；输出只收', '活跃请求——清零即不携带上一拍真位置', lc.C_MUTE)),
]

CX0 = 250
for ri, r in enumerate(ROWS):
    y = R_Y0 + ri * R_PITCH
    # 名牌
    lc.rect(MX, y, NAME_W, 46, '#ffffff', C_STALE_S, rx=6, sw=1.1)
    lc.text(MX + 10, y + 16, r['name'], 9.2, lc.C_TXT, 'start', True, maxw=NAME_W - 16,
            tag=f'nm{ri}')
    lc.text(MX + 10, y + 31, r['span'], 7.8, lc.C_BEAT_T, 'start', True, maxw=NAME_W - 16,
            tag=f'sp{ri}')
    lc.text(MX + 10, y + 42, r['src'], 6.6, C_STALE_S, 'start', maxw=NAME_W - 14, tag=f'sr{ri}')
    # 值格
    x = CX0
    if r['pre']:
        lab, w = r['pre']
        lc.rect(x, y + 8, w, 30, lc.C_GPU_F, lc.C_GPU_S, rx=5, sw=1.2)
        lc.text(x + w / 2, y + 27, lab, 8, lc.C_GPU_S, 'middle', True, maxw=w - 6,
                tag=f'pr{ri}')
        x += w + 14
    for ci, v in enumerate(r['cells']):
        if v == '→':
            lc.text(x + ci * (CELL_W + 4) + CELL_W / 2, y + 27, '→', 11, lc.C_MUTE, 'middle',
                    True, maxw=CELL_W, tag=f'ar{ri}')
            continue
        stale = (v == '999')
        fill, st = (C_STALE_F, C_STALE_S) if stale else (lc.C_BEAT_F, lc.C_BEAT_S)
        tcol = '#64748b' if stale else lc.C_BEAT_T
        lc.rect(x + ci * (CELL_W + 4), y + 8, CELL_W, CELL_H, fill, st, rx=4, sw=1.2)
        lc.text(x + ci * (CELL_W + 4) + CELL_W / 2, y + 25, v, 8.6, tcol, 'middle', True,
                maxw=CELL_W - 4, tag=f'c{ri}{ci}')
    ncell = len(r['cells'])
    lc.text(x + ncell * (CELL_W + 4) + 6, y + 27, r['ccap'], 7.4, lc.C_MUTE, 'start',
            maxw=120, tag=f'cc{ri}')
    # 消费端
    lc.rect(1020, y, 420, 46, '#ffffff', r['cons'][2], rx=6, sw=1.2)
    lc.text(1032, y + 19, r['cons'][0], 8.2, r['cons'][2], 'start', True, maxw=400,
            tag=f'co{ri}')
    lc.text(1032, y + 35, r['cons'][1], 7.8, '#334155', 'start', maxw=400, tag=f'co2{ri}')
    lc.seg(1012, y + 23, 1020, y + 23, r['cons'][2], 1.6, 'std')

# ---------------- 底部注记 ----------------
NY = R_Y0 + 4 * R_PITCH + 12
lc.rect(MX, NY, BXR - MX, 40, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 16, NY + 17, '值域论证：合法 slot = 块号 × 16 + 偏移 ≥ 0，-1 不在值域内、判别无歧义；物理块 0 被 null_block 全局保留且清零（ch13 第 1 站）——三个哨兵不可能与真值撞车',
        8.3, '#334155', 'start', maxw=BXR - MX - 32, tag='nt:1')
lc.text(MX + 16, NY + 32, 'FULL 拍换来的是 128 形状捕获图直接回放（ch19 的形状纪律）——PAD 税 = slot 尾 8 项 + 块表尾 6 行 + qsl 尾 6 项 + positions 尾 8 项清零',
        8.3, '#334155', 'start', maxw=BXR - MX - 32, tag='nt:2')

# ---------------- 页脚 ----------------
lc.text(MX, 712, '图例：绿 = 本拍真数据 · 橙 = PAD 哨兵（-1 / 0 / 清零后的 0） · 灰 = 上一拍残留 · 青框 = 读腿消费端',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 728, '逐字锚 vllm/v1/attention/backends/utils.py:L45-L46（PAD_SLOT_ID=-1 · NULL_BLOCK_ID=0）· vllm/v1/worker/block_table.py:L399-L408（PAD 程序）· gpu_model_runner.py:L4082-L4154 · L2325-L2341（尾段装配）· cache_kernels.cu:L326-L333（slot<0 return）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch22-fig-pad-sentries.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
