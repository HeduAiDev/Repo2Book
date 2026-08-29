#!/usr/bin/env python3
"""ch22 机制图 8 · CP 分片：I-token 条带交错 + 本地偏移重排 + 非本秩 PAD
（figure_spec ch22-fig-cp-interleave，模板 tiling）

放大自 L0『GPU 执行臂』（gpu_column 绿列）——即本章 L2 章图 center 拍片 ⑤
『kernel 内景 · 恒等式+PAD』CP 分片半边的机制展开（多 rank 部署形态归分布式
Part，本章只看 kernel 内的分片数学）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）。

claim：CP 分片不是整块切分而是 I-token 交错：32-token 虚拟块切成 2-token 小条
交替归属两个 rank，每 rank 把自己的条带紧凑重排成本地 [0,16) 偏移，非本秩
token 的 slot 打 PAD。

数字全部取自 figure_spec.numbers（32/2/16/35/321/322/320/-1，精简版 host 实跑：
CP 三件逐字镜像 + 覆盖核对 + 单卡退化对照）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 700
MX = 60
BXR = 1440
R0_F, R0_S = lc.C_GPU_F, lc.C_GPU_S     # rank 0 = 绿
R1_F, R1_S = lc.C_ZMQ_F, lc.C_ZMQ_S     # rank 1 = 紫

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'CP 分片：32-token 虚拟块切成 2-token 条带交替归属——每 rank 紧凑重排到本地 [0,16)',
        16.5, lc.C_TXT, 'start', True, maxw=1100, tag='title')
lc.text(MX, 58, '不是前 16/后 16 的整刀切：条带 0,2,4,… 归 rank0、1,3,5,… 归 rank1（is_local = (voff//2)%2 == R）；非本秩 token 的 slot 打 PAD',
        10.5, lc.C_MUTE, 'start', maxw=1090, tag='subtitle')
_ch = '放大自 L2 拍片 ⑤ kernel 内景 · CP 分片半边 · L0：GPU 执行臂 × KV 接缝'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- ① 虚拟块条带（32 cell = 16 条带 × 2 token） ----------------
lc.text(MX, 106, '① 一个 32-token 虚拟块（vbi=1 · 块表行 [10, 20] → 行[1]=20 · pos 32..63）', 10,
        lc.C_TXT, 'start', True, maxw=640, tag='s:t')
SX, SY, CW_, CH_ = 180, 150, 30, 44

def cell_x(j):
    return SX + j * CW_ + (j // 2) * 2      # 条带间 2px 分隔

for j in range(32):
    x = cell_x(j)
    voff = j
    pos = 32 + j
    owner = (voff // 2) % 2
    fill, st, tc = (R0_F, R0_S, R0_S) if owner == 0 else (R1_F, R1_S, R1_S)
    hero = voff in (3, 4)
    lc.rect(x, SY, CW_ - 0.8, CH_, fill, st, rx=2, sw=2.4 if hero else 1.0)
    lc.text(x + (CW_ - 0.8) / 2, SY + 27, str(voff), 9.5, tc, 'middle', True, maxw=CW_ - 6,
            tag=f'v{j}')
    lc.text(x + (CW_ - 0.8) / 2, 144, str(pos), 6.5, '#94a3b8', 'middle', maxw=CW_ - 4,
            tag=f'p{j}')
for k in range(16):
    x = cell_x(2 * k)
    lc.text(x + CW_ - 0.4, SY + CH_ + 12, f'条带{k}', 6.4,
            R0_S if k % 2 == 0 else R1_S, 'middle', maxw=CW_ * 2, tag=f'st{k}')
lc.text(cell_x(3) + (CW_ - 0.8) / 2, 118, 'pos=35', 8.2, R1_S, 'middle', True, maxw=60,
        tag='hero35')
lc.text(cell_x(4) + (CW_ - 0.8) / 2, 132, 'pos=36', 8.2, R0_S, 'middle', True, maxw=60,
        tag='hero36')
lc.text(MX, SY + 27, 'voff →', 8, lc.C_MUTE, 'end', maxw=110, tag='vofflab')

# ---------------- ② 每 rank 的本地偏移重排 [0,16) ----------------
lc.text(MX, 232, '② 紧凑重排：各 rank 把零散条带压成本地连续 [0,16) 偏移（lbo）', 10, lc.C_TXT,
        'start', True, maxw=640, tag='rr:t')
LCW = 60
RY0, RY1 = 246, 306
for ri, (ry, rf, rs, name, base) in enumerate(((RY0, R0_F, R0_S, 'rank 0', 32),
                                               (RY1, R1_F, R1_S, 'rank 1', 34))):
    lc.text(MX, ry + 18, name, 10, rs, 'start', True, maxw=80, tag=f'rn{ri}')
    lc.text(MX, ry + 34, '本地 [0,16)', 7.2, lc.C_MUTE, 'start', maxw=80, tag=f'rn2{ri}')
    lc.text(MX, ry + 50, 'lbo →', 7.2, lc.C_MUTE, 'start', maxw=80, tag=f'lb{ri}')
    for lbo in range(16):
        x = SX + lbo * LCW
        pos = base + 4 * (lbo // 2) + lbo % 2
        lc.rect(x, ry, LCW - 2, 46, rf, rs, rx=4, sw=1.2)
        lc.text(x + (LCW - 2) / 2, ry + 18, str(lbo), 10.5, rs, 'middle', True, maxw=LCW - 10,
                tag=f'l{ri}{lbo}')
        lc.text(x + (LCW - 2) / 2, ry + 36, f'pos {pos}', 6.8, '#334155', 'middle',
                maxw=LCW - 8, tag=f'lp{ri}{lbo}')

# 公式条
FY = 368
lc.rect(SX, FY, 990, 30, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
lc.text(SX + 12, FY + 19, '条带号 = voff // I（I=2）· 归属 rank = 条带号 % W（W=2）· is_local = (voff // I) % W == R · lbo = (voff // (W×I)) × I + voff % I',
        8.2, '#334155', 'start', maxw=966, tag='f:t')

# ---------------- ③ 例证卡：pos=32 / 35 / 36 ----------------
lc.text(MX, 424, '③ 三个 token 的归属裁决（行[1]=20 → slot = 20×16 + lbo）', 10, lc.C_TXT,
        'start', True, maxw=560, tag='ex:t')
EX_Y, EX_H = 438, 64
for cx, title, rank, lbo, slot in (
        (180, 'pos=32（voff=0 · 条带 0）', 0, 0, 320),
        (560, 'pos=35（voff=3 · 条带 1）', 1, 1, 321),
        (940, 'pos=36（voff=4 · 条带 2）', 0, 2, 322)):
    rf, rs = (R0_F, R0_S) if rank == 0 else (R1_F, R1_S)
    lc.rect(cx, EX_Y, 340, EX_H, rf, rs, rx=8, sw=1.5)
    lc.text(cx + 14, EX_Y + 18, title, 8.8, rs, 'start', True, maxw=320, tag='ex:t')
    lc.text(cx + 14, EX_Y + 36, f'→ rank{rank}：lbo={lbo} → slot = 20×16+{lbo} = {slot} ✓', 8.6,
            lc.C_TXT, 'start', True, maxw=320, tag='ex:a')
    lc.text(cx + 14, EX_Y + 53, f'→ rank{1 - rank}：slot = -1（PAD，非本秩）', 8.2, lc.C_BEAT_T,
            'start', maxw=320, tag='ex:b')

# ---------------- ④ 覆盖账 + 不是整刀切 ----------------
CV_Y = 522
lc.rect(180, CV_Y, 560, 84, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(194, CV_Y + 18, '覆盖账（64 token 全查）', 9.4, lc.C_TXT, 'start', True, maxw=520,
        tag='cv:t')
for i, ln in enumerate([
        '· 每 rank 恰 32 真 slot + 32 PAD；并集覆盖全部 64 token（无重无漏）',
        '· 每 rank 每 vblock 恰认领 16 token，本地偏移连续铺满 [0,16)',
        '· 每 rank 的 KV 池只存一半 KV——slot 恒等式外层逐字不变']):
    lc.text(194, CV_Y + 36 + i * 16, ln, 7.8, '#334155', 'start', maxw=532, tag=f'cv{i}')

CT_Y = 522
lc.rect(770, CT_Y, 670, 84, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(784, CT_Y + 18, '对照：不是整刀切', 9.4, lc.C_TXT, 'start', True, maxw=440,
        tag='ct:t')
mw = 420 / 32
# 整刀切（✗）
for j in range(32):
    lc.rect(784 + j * mw, CT_Y + 28, mw - 1, 12, R0_F if j < 16 else R1_F,
            R0_S if j < 16 else R1_S, rx=1, sw=0.5)
lc.seg(784 + 16 * mw, CT_Y + 24, 784 + 16 * mw, CT_Y + 44, lc.C_ABORT, 1.6)
lc.text(1214, CT_Y + 37, '✗ 前 16 / 后 16 整刀切（不是这样）', 7.6, lc.C_ABORT, 'start',
        True, maxw=230, tag='ct:x')
# 实际交错（✓）
for j in range(32):
    owner = (j // 2) % 2
    lc.rect(784 + j * mw, CT_Y + 48, mw - 1, 12, R0_F if owner == 0 else R1_F,
            R0_S if owner == 0 else R1_S, rx=1, sw=0.5)
lc.text(1214, CT_Y + 57, '✓ 实际：2-token 条带交替', 7.6, lc.C_GPU_S, 'start', True,
        maxw=230, tag='ct:y')

# ---------------- 单卡退化 ----------------
DG_Y = 622
lc.rect(180, DG_Y, 1060, 26, '#ffffff', lc.C_MUTE, rx=6, sw=1.1, dash=True)
lc.text(192, DG_Y + 17, '单卡退化（W=1、I=1）：virtual_block_size=16、is_local 恒真、lbo = pos%16——CP 三件全部退化为恒等：slot_35 = 行[2]×16+3 = 179，服从普通恒等式',
        7.8, '#334155', 'start', maxw=1036, tag='dg:t')

# ---------------- 页脚 ----------------
lc.text(MX, 674, '图例：绿 = rank 0 认领 · 紫 = rank 1 认领 · 橙 = PAD（-1，非本秩）',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 690, '逐字锚 vllm/v1/worker/block_table.py:L413-L428（CP 三件：virtual_block_size / is_local / local_block_offsets）· L441（tl.where(is_local, slot_ids, PAD_ID)）· W=dcp_world_size、I=cp_kv_cache_interleave_size（多 rank 部署归分布式 Part，本章只看 kernel 内分片数学）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch22-fig-cp-interleave.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
