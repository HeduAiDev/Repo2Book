#!/usr/bin/env python3
"""ch13 机制图 6 · 槽位恒等式（figure_spec ch13-fig-slot-identity，模板 tiling）

放大自 L0 GPU 列（绿）里 worker 页表/槽位层——即本章 L2 章图中排 ⑥『槽位换算 ·
恒等式』拍片的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：一条恒等式接通两层：块表行 [3,1,7] 把 48 个连续的 token 位置映射到三段
物理槽位 48..63 / 16..31 / 112..127（位置递增而物理槽位中段反而更低——间接寻址
脱钩的直接可视化），尾部 [20,64) 全填 -1。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑：三段映射、PAD 尾 [20,64)、
双请求行 [2]/[5]、逆运算 112→块7 / 96→块6 / 80→块5）。
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
SEG_COL = {0: lc.C_API_S, 1: lc.C_ENG_S, 2: lc.C_SAM_S}      # 三段类别色（图例兜底）
SEG_FILL = {0: lc.C_API_F, 1: lc.C_ENG_F, 2: lc.C_SAM_F}

# ---------------- 标题区 ----------------
lc.text(MX, 34, '槽位恒等式：块表行 [3,1,7] 把 48 个连续位置接到三段不相邻的物理槽位',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'slot = block_table[req][pos // 16] × 16 + pos % 16——写腿存进 slot、读腿翻块表，同一条算术两条腿共用；换算在 GPU 的 Triton kernel 里做，不落 CPU',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L2 章图中排 ⑥ 拍片「槽位换算 · 恒等式」· L0：GPU 列 worker 页表/槽位层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上带：逻辑位置 48 格 ----------------
UP_Y, UP_H = 118, 40
UP_X, UP_CW = MX + 130, 22
UP_W = 48 * UP_CW                                     # 1056
lc.text(MX, UP_Y + 16, '逻辑位置 pos', 10, lc.C_TXT, 'start', True, maxw=110, tag='up:t')
lc.text(MX, UP_Y + 34, '（连续 0..47）', 8, lc.C_MUTE, 'start', maxw=110, tag='up:s')
for i in range(48):
    x = UP_X + i * UP_CW
    seg = i // 16
    lc.rect(x, UP_Y, UP_CW - 2, UP_H, SEG_FILL[seg], SEG_COL[seg], rx=2, sw=1.0)
    if i % 16 in (0, 15) or i in (0, 15, 16, 31, 32, 47):
        lc.text(x + (UP_CW - 2) / 2, UP_Y + 25, str(i), 8, SEG_COL[seg], 'middle',
                maxw=UP_CW - 4, tag='up%d' % i)
lc.text(UP_X + 8 * UP_CW, UP_Y - 8, '段 A：pos 0-15', 8.5, SEG_COL[0], 'middle', maxw=100, tag='seg0')
lc.text(UP_X + 24 * UP_CW, UP_Y - 8, '段 B：pos 16-31', 8.5, SEG_COL[1], 'middle', maxw=100, tag='seg1')
lc.text(UP_X + 40 * UP_CW, UP_Y - 8, '段 C：pos 32-47', 8.5, SEG_COL[2], 'middle', maxw=100, tag='seg2')

# ---------------- 中层：块表行三张卡 ----------------
MID_Y = UP_Y + UP_H + 58
CARD_W, CARD_H, CARD_GAP = 150, 58, 120
CARDS = [('块表[0] = 3', '服务段 A（pos 0-15）', 0), ('块表[1] = 1', '服务段 B（pos 16-31）', 1),
         ('块表[2] = 7', '服务段 C（pos 32-47）', 2)]
card_pos = []
MID_X0 = UP_X + 48 * UP_CW / 2 - (3 * CARD_W + 2 * CARD_GAP) / 2
for i, (t, s, seg) in enumerate(CARDS):
    x = MID_X0 + i * (CARD_W + CARD_GAP)
    lc.rect(x, MID_Y, CARD_W, CARD_H, '#ffffff', SEG_COL[seg], rx=6, sw=1.8)
    lc.rect(x, MID_Y, CARD_W, 20, SEG_FILL[seg], SEG_COL[seg], rx=6, sw=1.8)
    lc.rect(x, MID_Y + 8, CARD_W, 12, SEG_FILL[seg], 'none', rx=0, sw=0)
    lc.text(x + CARD_W / 2, MID_Y + 14, t, 9.5, SEG_COL[seg], 'middle', True, maxw=CARD_W - 10,
            tag='mc%d' % i)
    lc.text(x + CARD_W / 2, MID_Y + 36, s, 8, '#64748b', 'middle', maxw=CARD_W - 10, tag='ms%d' % i)
    lc.text(x + CARD_W / 2, MID_Y + 48, str((3, 1, 7)[i]), 13, SEG_COL[seg], 'middle',
            True, maxw=60, tag='mn%d' % i)
    card_pos.append(x + CARD_W / 2)
    # 上带段 → 块表卡（肘形）
    seg_cx = UP_X + (16 * seg + 8) * UP_CW
    lc.parrow([(seg_cx, UP_Y + UP_H + 2), (seg_cx, MID_Y - 30), (card_pos[i], MID_Y - 30),
               (card_pos[i], MID_Y - 3)], SEG_COL[seg], 1.8, 'std')
    lc.text((seg_cx + card_pos[i]) / 2, MID_Y - 38, 'pos // 16 = %d' % seg, 8, SEG_COL[seg], 'middle',
            maxw=110, tag='dv%d' % i)

# ---------------- 下带：物理槽位 128 格 ----------------
DN_Y = MID_Y + CARD_H + 52
DN_H = 40
DN_X, DN_CW = UP_X, UP_W / 128.0
lc.text(MX, DN_Y + 16, '物理槽位 slot', 10, lc.C_TXT, 'start', True, maxw=120, tag='dn:t')
lc.text(MX, DN_Y + 34, '（0..127，按块号落位）', 7.5, lc.C_MUTE, 'start', maxw=120, tag='dn:s')
SEG2SLOT = [(3, 48, 63, 0), (1, 16, 31, 1), (7, 112, 127, 2)]     # (块号, 起, 止, 段)
seg_of_slot = {}
for blk, a, b, seg in SEG2SLOT:
    for s in range(a, b + 1):
        seg_of_slot[s] = seg
for s in range(128):
    x = DN_X + s * DN_CW
    seg = seg_of_slot.get(s)
    if seg is None:
        lc.rect(x, DN_Y, DN_CW - 1, DN_H, '#f1f5f9', '#e2e8f0', rx=1, sw=0.6)
    else:
        lc.rect(x, DN_Y, DN_CW - 1, DN_H, SEG_FILL[seg], SEG_COL[seg], rx=1, sw=0.9)
# 槽位刻度（下方；段端点值入段题注）
for s in (0, 16, 32, 48, 64, 80, 112, 127):
    x = DN_X + s * DN_CW
    anchor = 'start' if s == 0 else ('end' if s == 127 else 'middle')
    lc.text(x + (0 if s in (0, 127) else DN_CW / 2), DN_Y + DN_H + 16, str(s), 8,
            SEG_COL[seg_of_slot[s]] if s in seg_of_slot else '#94a3b8', anchor, maxw=34,
            tag='tk%d' % s)
# 块表卡 → 物理段：色对应 + 段题注（不画会共线相绞的肘形箭头）
for (blk, a, b, seg) in SEG2SLOT:
    tx = DN_X + (a + b + 1) / 2 * DN_CW
    lc.text(tx, DN_Y - 12, '块 %d ← 段 %s · 槽 %d..%d' % (blk, 'ABC'[seg], a, b), 8.5,
            SEG_COL[seg], 'middle', True, maxw=170, tag='sc%d' % blk)
# 物理与逻辑脱钩注记
NOTE_Y = DN_Y + DN_H + 40
lc.text(DN_X + 300, NOTE_Y, '物理 ≠ 逻辑：段 B（pos 16-31）的槽位 16..31 反而低于段 A 的 48..63——位置越走越高、槽位先降后升，这就是「翻页」',
        9, '#475569', 'start', True, maxw=820, tag='decouple')

# ---------------- 底部两小图：PAD 尾 + 双请求 + 逆运算 ----------------
BY = NOTE_Y + 24
# PAD 尾
PD_X, PD_W = MX, 660
lc.rect(PD_X, BY, PD_W, 120, '#ffffff', '#94a3b8', rx=7, sw=1.2)
lc.text(PD_X + 14, BY + 20, 'PAD 尾：另一场景（行 [1,2]、20 真 token / max 64）', 9.5, lc.C_TXT,
        'start', True, maxw=PD_W - 28, tag='pd:t')
PD_CW = (PD_W - 60) / 64.0
PD_Y, PD_H = BY + 34, 26
for i in range(64):
    x = PD_X + 24 + i * PD_CW
    if i < 20:
        lc.rect(x, PD_Y, PD_CW - 1, PD_H, '#ecfeff', lc.C_KV_S, rx=1, sw=0.7)
    else:
        lc.rect(x, PD_Y, PD_CW - 1, PD_H, '#f1f5f9', '#cbd5e1', rx=1, sw=0.6)
        if 24 <= i < 60 and (i - 24) % 8 == 0:
            lc.text(x + PD_CW / 2, PD_Y + 17, '-1', 7, '#94a3b8', 'middle', maxw=24, tag='pd%d' % i)
lc.text(PD_X + 24, PD_Y + PD_H + 16, '前 20 格实（槽 16..35）', 8, lc.C_KV_S, 'start', maxw=170,
        tag='pd:real')
lc.text(PD_X + 24 + 20 * PD_CW + 8, PD_Y + PD_H + 16, '尾部 [20,64) 每拍重填 -1（PAD_SLOT_ID）——CUDA graph 捕获 max 形状',
        8, '#94a3b8', 'start', maxw=340, tag='pd:pad')
lc.text(PD_X + 14, BY + 106, '最后一个 program 专职 PAD 尾；合法 slot ≥ 16 > 0 > −1，值域不相交',
        8, lc.C_MUTE, 'start', maxw=PD_W - 28, tag='pd:n')
# 双请求 + 逆运算
DX2 = MX + 690
DW2 = BXR - DX2
lc.rect(DX2, BY, DW2, 120, '#ffffff', '#94a3b8', rx=7, sw=1.2)
lc.text(DX2 + 14, BY + 20, '双请求（decode）：positions 各自从 0 起', 9.5, lc.C_TXT, 'start', True,
        maxw=DW2 - 28, tag='d2:t')
for i, (row, blk, a, b) in enumerate([('行 [2]', 2, 32, 47), ('行 [5]', 5, 80, 95)]):
    yy = BY + 42 + i * 26
    lc.text(DX2 + 14, yy + 3, row, 9, '#334155', 'start', True, maxw=50, tag='d2r%d' % i)
    lc.seg(DX2 + 66, yy - 2, DX2 + 96, yy - 2, '#94a3b8', 1.4, 'std')
    lc.text(DX2 + 104, yy + 3, '16 token → 块 %d → 槽 %d..%d' % (blk, a, b), 8.5, '#334155',
            'start', maxw=210, tag='d2v%d' % i)
lc.text(DX2 + 330, BY + 48, '读侧逆运算（同一恒等式）：', 9, lc.C_TXT, 'start', True, maxw=190,
        tag='inv:t')
for i, (slot, blk) in enumerate([(112, 7), (96, 6), (80, 5)]):
    lc.text(DX2 + 330, BY + 66 + i * 16, 'slot %d → 块 %d · 偏移 0' % (slot, blk), 8.5, '#334155',
            'start', maxw=190, tag='inv%d' % i)
lc.text(DX2 + 330, BY + 113, '块号 = slot // 16 · 偏移 = slot % 16', 8, lc.C_MUTE, 'start',
        maxw=190, tag='inv:f')
BOT2 = BY + 120

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BOT2 + 22
lx = MX
for seg, name in [(0, '段 A（块 3）'), (1, '段 B（块 1）'), (2, '段 C（块 7）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, SEG_FILL[seg], SEG_COL[seg], rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=110, tag='lg%d' % seg)
    lx += 26 + lc.tw(name, 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 13, '#f1f5f9', '#e2e8f0', rx=3, sw=0.8)
lc.text(lx + 26, LEG_Y + 1, '其他块的槽位（本请求不碰）', 8.5, lc.C_TXT, 'start', maxw=200,
        tag='lg:oth')
lx += 26 + lc.tw('其他块的槽位（本请求不碰）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 13, '#ecfeff', lc.C_KV_S, rx=3, sw=0.9)
lc.text(lx + 26, LEG_Y + 1, 'PAD 尾实格', 8.5, lc.C_TXT, 'start', maxw=100, tag='lg:pd')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/worker/block_table.py:L380-L442（_compute_slot_mapping_kernel 恒等式本体）· '
        'L182-L211（compute_slot_mapping 派发）· vllm/v1/worker/gpu_model_runner.py:L2188-L2201（positions = num_computed_tokens[req_indices_gpu] + query_pos.gpu，GPU 张量进、换算不落 CPU）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '三段映射 / PAD 尾 / 双请求 / 逆运算数字取自配套精简版 host 实跑（kernel 的逐行 CPU 镜像，CUDA 分支容器内真跑、数值无差）· '
        'CP 分片与 PAD 三值语义 → ch22 · 行号基线 vLLM v0.27.1', 8.2, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 66
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-slot-identity.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
