#!/usr/bin/env python3
"""ch22 机制图 1 · 槽位恒等式与逆分解（figure_spec ch22-fig-slot-identity，模板 layout）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）与 KV 池的接缝——即本章 L2 章图 center
拍片 ⑤『kernel 内景 · 恒等式+PAD』（站 6-7）恒等式半边的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：一个 token 的物理槽位 = 块表行条目 × block_size + 块内偏移——pos=99 查行得
物理块 9，slot=9×16+3=147；写侧 kernel 再用 slot//16、slot%16 逆着做回同一格
（正逆同一恒等式，100 token round-trip 全闭合）。

数字全部取自 figure_spec.numbers（16/99/6/3/9/147/128，精简版 host 实跑逐 token 记账；
块表行 [3,8,2,7,1,5,9] 为同一 worked example 的演示数据）。坐标由常量/循环计算；
文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 648
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'slot = 行[pos//16] × 16 + pos%16：一张行表把乱序物理块对 token 位置透明化',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '寄存柜取件——pos=99 → 逻辑块 6、块内偏移 3 → 行第 6 项是物理块 9 → slot = 9×16+3 = 147；写侧 kernel 拿 147//16、147%16 原路返回同一格',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L2 拍片 ⑤ kernel 内景 · 恒等式+PAD · L0：GPU 执行臂 × KV 接缝'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- ① token 位置条（0..99，7 个逻辑块分段） ----------------
lc.text(MX, 104, '① token 位置 pos（请求内，0..99）', 10.5, lc.C_TXT, 'start', True,
        maxw=330, tag='s1:t')
SX0, STRIP_Y, STRIP_H = 220, 112, 34
TOK_W = 11.6                      # 100 token × 11.6 = 1160 → 220..1380
NTOK = 100
BS = 16
ROW = [3, 8, 2, 7, 1, 5, 9]

def seg_x(k):                     # 逻辑块 k 的分段 x 起点/宽
    x0 = SX0 + k * BS * TOK_W
    w = (BS if k < 6 else NTOK - 6 * BS) * TOK_W
    return x0, w

# pos 标记（两处主线/副线例）
lc.text(SX0 + 16 * TOK_W, 105, 'pos=16', 8.2, lc.C_GPU_S, 'middle', True, maxw=60, tag='m16')
lc.text(SX0 + NTOK * TOK_W, 105, 'pos=99', 8.2, lc.C_GPU_S, 'end', True, maxw=60, tag='m99')

for k in range(7):
    x0, w = seg_x(k)
    lc.rect(x0, STRIP_Y, w, STRIP_H, '#f8fafc', '#cbd5e1', rx=3, sw=1.0)
    lo, hi = k * BS, min((k + 1) * BS, NTOK) - 1
    lab = f'pos {lo}–{hi}' if k < 6 else f'{lo}–{hi}'
    lc.text(x0 + w / 2, STRIP_Y + 21, lab, 8.5, '#334155', 'middle',
            maxw=w - 6, tag=f'seg{k}')
# 高亮子格：pos=16（逻辑块 1 首格）与 pos=99（末格）
for pos, lab in ((16, ''), (99, '')):
    x0 = SX0 + pos * TOK_W
    lc.rect(x0, STRIP_Y, TOK_W, STRIP_H, lc.C_GPU_S, lc.C_GPU_S, rx=2, sw=0)
# 刻度
for pos in (0, 16, 32, 48, 64, 80, 96):
    lc.text(SX0 + pos * TOK_W, 160, str(pos), 7.2, '#94a3b8', 'middle', maxw=34,
            tag='tick' + str(pos))
lc.text(SX0 + NTOK * TOK_W, 160, '99', 7.2, '#94a3b8', 'end', maxw=34, tag='tick99')

# ---------------- ② 块表行（与①分段同 x 对位） ----------------
lc.text(MX, 196, '② 块表行 block_table[req]', 10.5, lc.C_TXT, 'start', True, maxw=200,
        tag='s2:t')
ROW_Y, ROW_H = 208, 44
for k, v in enumerate(ROW):
    x0, w = seg_x(k)
    hot = k in (1, 6)
    lc.rect(x0, ROW_Y, w, ROW_H, lc.C_KV_F if hot else '#ffffff', lc.C_KV_S, rx=4,
            sw=1.8 if hot else 1.2)
    lc.text(x0 + w / 2, ROW_Y + 14, f'逻辑块 {k}', 7.2, lc.C_MUTE, 'middle', maxw=w - 4,
            tag=f'idx{k}')
    lc.text(x0 + w / 2, ROW_Y + 35, str(v), 13.5, lc.C_KV_S, 'middle', True, maxw=w - 4,
            tag=f'val{k}')

# ①→② 查行箭头（kernel 的带余除法）
for k, lab in ((6, '99//16 = 6'), (1, '16//16 = 1')):
    x0, w = seg_x(k)
    cx = x0 + w / 2
    lc.seg(cx, STRIP_Y + STRIP_H, cx, ROW_Y - 2, lc.C_GPU_S, 2.0, 'std')
    lc.text(cx + 8, (STRIP_Y + STRIP_H + ROW_Y) / 2 + 3, lab, 8.2, lc.C_GPU_S, 'start',
            True, maxw=110, tag='div' + str(k))
lc.text(MX, (STRIP_Y + STRIP_H + ROW_Y) / 2 + 3, '带余除法选格', 7.8, lc.C_GPU_S, 'start',
        True, maxw=160, tag='divnote')

# ---------------- ③ KV 池物理块（16 行货栈） ----------------
lc.text(MX, 300, '③ KV 池物理块（每块 16 行）', 10.5, lc.C_TXT, 'start', True, maxw=300,
        tag='s3:t')
STACK_W, ROW_PITCH = 118, 8.4
STACK_Y0 = 318
BLOCKS = [(3, 300, 'full'), (8, 640, 'hot0'), (9, 980, 'hot3')]   # (块号, x, 强调模式)
for bid, bx, mode in BLOCKS:
    lc.text(bx + STACK_W / 2, STACK_Y0 - 6, f'块 {bid}', 10, lc.C_KV_S, 'middle', True,
            maxw=80, tag=f'blk{bid}')
    for r in range(16):
        y = STACK_Y0 + r * ROW_PITCH
        if mode == 'hot3' and r == 3:
            lc.rect(bx, y, STACK_W, ROW_PITCH - 1, lc.C_GPU_S, lc.C_GPU_S, rx=1.5, sw=0)
            lc.text(bx + STACK_W / 2, y + 6, f'slot={bid * 16 + r}  ← pos=99', 7.2,
                    '#ffffff', 'middle', True, maxw=STACK_W - 4, tag=f'hot{bid}')
        elif mode == 'hot0' and r == 0:
            lc.rect(bx, y, STACK_W, ROW_PITCH - 1, lc.C_GPU_S, lc.C_GPU_S, rx=1.5, sw=0)
            lc.text(bx + STACK_W / 2, y + 6, f'slot={bid * 16 + r}  ← pos=16', 7.2,
                    '#ffffff', 'middle', True, maxw=STACK_W - 4, tag=f'hot{bid}')
        else:
            fill = lc.C_KV_F if mode == 'full' else '#f8fafc'
            lc.rect(bx, y, STACK_W, ROW_PITCH - 1, fill, '#cbd5e1', rx=1.5, sw=0.7)
        if r % 4 == 0:
            lc.text(bx - 6, y + 6, str(r), 6.4, '#94a3b8', 'end', maxw=24,
                    tag=f'rn{bid}{r}')
lc.text(300 + STACK_W / 2, STACK_Y0 + 16 * ROW_PITCH + 14, 'pos 0–15 全落此块', 8,
        lc.C_KV_S, 'middle', maxw=140, tag='b3note')

# ②→③ 查行得块（肘形）
ELB = 288
for k, (bx, _) in ((0, (300, 3)), (1, (640, 8)), (6, (980, 9))):
    x0, w = seg_x(k)
    cx = x0 + w / 2
    tx = bx + STACK_W / 2
    lc.parrow([(cx, ROW_Y + ROW_H), (cx, ELB), (tx, ELB), (tx, STACK_Y0 - 2)],
              lc.C_KV_S, 1.6 if k else 1.1, 'std')
    lc.text(tx - 6, ELB - 5, f'行[{k}]={ROW[k]}', 7.6, lc.C_KV_S, 'end', True, maxw=90,
            tag=f'row{k}lab')

# 右侧洞见盒
IB_X, IB_W = 1160, 280
lc.rect(IB_X, 300, IB_W, 158, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(IB_X + 14, 322, '为什么必须查行表', 10, lc.C_TXT, 'start', True, maxw=IB_W - 28,
        tag='ib:t')
for i, ln in enumerate([
        '本例 7 个逻辑块落在物理块',
        '{3,8,2,7,1,5,9}——毫无顺序，',
        '分配器说了算；token 只认 pos，',
        '查表把任意排列变得透明。',
        '第 2 个逻辑块落在物理块 8',
        '（slot 从 128 起）——与逻辑序',
        '毫无关系，第 7 个反而是 9。']):
    lc.text(IB_X + 14, 342 + i * 16, ln, 8.2, '#334155', 'start', maxw=IB_W - 28,
            tag=f'ib{i}')

# ---------------- ④ 恒等式正逆闭合 ----------------
FY = 500
lc.rect(220, FY, 500, 64, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.5)
lc.text(236, FY + 19, '正向换算（_compute_slot_mapping_kernel，每 token O(1) 查表）', 9,
        lc.C_GPU_S, 'start', True, maxw=468, tag='fw:t')
lc.text(236, FY + 39, 'slot = 行[pos//16] × 16 + pos%16', 10.5, lc.C_TXT, 'start', True,
        maxw=468, tag='fw:f')
lc.text(236, FY + 56, 'pos=99：行[6]=9 → 9×16+3 = 147', 8.6, '#334155', 'start', maxw=468,
        tag='fw:e')
lc.rect(800, FY, 500, 64, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.5)
lc.text(816, FY + 19, '逆向分解（reshape_and_cache_flash 反用同一恒等式）', 9, lc.C_GPU_S,
        'start', True, maxw=468, tag='bw:t')
lc.text(816, FY + 39, '块 = slot//16 → 147//16 = 9；行 = slot%16 → 147%16 = 3', 10.5,
        lc.C_TXT, 'start', True, maxw=468, tag='bw:f')
lc.text(816, FY + 56, '(块 9, 行 3) 原路返回 ✓ 带余除法唯一性保证闭合', 8.6, '#334155',
        'start', maxw=468, tag='bw:e')
lc.seg(720, FY + 32, 800, FY + 32, lc.C_GPU_S, 2.0, 'std')
lc.text(220, FY + 82, '本例 100 个 token 逐个记账：恒等式 100/100 round-trip 闭合（写侧落点核验 (块,行) 全中）',
        8.8, lc.C_MUTE, 'start', maxw=1000, tag='rt')

# ---------------- 页脚 ----------------
lc.text(MX, 616, '图例：青 = 块表行 / KV 池 · 绿 = kernel 寻址换算（正逆同一恒等式） · 灰 = token 位置轴 · 深绿格 = 演示落点',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 632, '逐字锚 vllm/v1/worker/block_table.py:L410-L442（恒等式主体 block_table_ptr 查行 + slot_ids 计算）· csrc/libtorch_stable/cache_kernels.cu:L326-L333（写侧 slot//block_size、slot%block_size 逆分解）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch22-fig-slot-identity.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
