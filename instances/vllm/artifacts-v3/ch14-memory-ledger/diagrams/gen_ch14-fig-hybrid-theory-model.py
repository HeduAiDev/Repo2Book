#!/usr/bin/env python3
"""ch14 机制图 · 混合缓存理论模型（figure_spec ch14-fig-hybrid-theory-model，模板 layout）

站 5（混合组化）之前的纯理论铺垫（一等约束：零 vLLM 标识符）——把「为什么混合
形状必须特殊处理」抽象成形状代数：层形状 =（每 token 字节 × 每 block token 数）
→ 页宽；池子只认等宽块号。三种形状关系阶梯：全同 / 同型异宽且整除 / 跨型不整除
（三条出路全死）→ 唯一无损解 = 重叠时间复用。与 ch14-fig-packed-slicing 成对：
这张讲理论必然性，那张讲代码实现。

数字全部取自 figure_spec.numbers（2 层玩具示教算术：A 层 2B×16=32、B 层 6B×16=96、
跨型例 32 与 80、块长 16→48（×3）→ 页 32→96）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与语义色 ----------------
W, MX, BXR = 1500, 60, 1440
CA, CA_F = lc.C_API_S, lc.C_API_F            # A 层形状 = 蓝
CB, CB_F = lc.C_KV_S, lc.C_KV_F              # B / C 层形状 = 青
C_OK = lc.C_GPU_S                            # 可行解 = 绿
C_DEAD = lc.C_ABORT                          # 死路 = 红
AMBER = '#b45309'                            # 唯一解 / 指针 = 琥珀

BPT_A, BPT_B, TOKS = 2, 6, 16                # 玩具：A 层 2 B/token、B 层 6 B/token、16 token/块
PAGE_A, PAGE_B, PAGE_C = 32, 96, 80          # 页宽：A 32、B 96、跨型例 C 80

# ---------------- 标题区 ----------------
lc.text(MX, 34, '混合缓存的三种形状关系：页宽代数与「重叠时间复用」的必然', 16.5, lc.C_TXT,
        'start', True, maxw=1050, tag='title')
lc.text(MX, 58, '层形状 = 每 token 字节 × 每块 token 数 → 页宽；池子只认等宽块号——玩具例：'
                'A 层 2 B/token × 16 token/块 = 32 B/块，B 层 6 B/token × 16 = 96 B/块',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='sub1')
lc.text(MX, 76, '读图：上排两条公理（形状从哪来、池子认什么）；下排三种形状关系阶梯——梯③ 三条出路全死，'
                '唯一无损解是重叠时间复用（一个块号同一时刻只归一组）',
        9.0, lc.C_MUTE, 'start', maxw=1330, tag='sub2')
_ch = '理论铺垫 · 纯形状代数（零代码）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 公理一：一个块的形状（含字节格） =================
P1X, P1W, PY0, PH = MX, 820, 94, 204
lc.rect(P1X, PY0, P1W, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(P1X + 16, PY0 + 22, '公理一 · 一个块的形状：页宽 = 每 token 字节 × 每块 token 数', 11.5,
        lc.C_TXT, 'start', True, maxw=P1W - 32, tag='p1:t')
CS = 5.2                                     # 每字节的像素宽
BAR_X, BAR_H = P1X + 90, 20
rows = [('A 层', BPT_A, CA, CA_F, PAGE_A),
        ('B 层', BPT_B, CB, CB_F, PAGE_B)]
for i, (name, bpt, cs_, cf_, page) in enumerate(rows):
    ry = PY0 + 46 + i * 50
    lc.text(P1X + 16, ry + 14, name, 10, cs_, 'start', True, maxw=60, tag='p1:' + name)
    # 字节格：每格 1 B；token 边界 = 每 bpt 格一条深色刻度
    for b in range(page):
        lx = BAR_X + b * CS
        lc.rect(lx, ry, CS - 0.7, BAR_H, cf_, cs_, rx=0.8, sw=0.5)
    for t in range(TOKS + 1):                # token 刻度线（bar 下方小刻度）
        tx = BAR_X + t * bpt * CS
        lc.seg(tx, ry + BAR_H + 1, tx, ry + BAR_H + 5, cs_, 0.9)
    lab = f'页宽 {page} B（{TOKS} token × {bpt} B/token）'
    lc.text(BAR_X + page * CS + 12, ry + 14, lab, 9.4, cs_, 'start', True,
            maxw=P1X + P1W - 16 - (BAR_X + page * CS + 12), tag='p1:lab' + name)
lc.text(P1X + 16, PY0 + 46 + 2 * 50 + 6, '同是 16 token/块，每 token 字节不同 → 页宽不同——这就是「混合」的根源',
        8.8, '#334155', 'start', maxw=P1W - 32, tag='p1:n')

# ================= 公理二：池子只认块号 =================
P2X, P2W = 916, BXR - 916
lc.rect(P2X, PY0, P2W, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(P2X + 16, PY0 + 22, '公理二 · 池子只认块号', 11.5, lc.C_TXT, 'start', True,
        maxw=P2W - 32, tag='p2:t')
SL_W, SL_H, SL_GAP = 74, 40, 10
sx0 = P2X + 20
for j in range(5):
    lc.rect(sx0 + j * (SL_W + SL_GAP), PY0 + 46, SL_W, SL_H, '#ffffff', lc.C_MUTE, rx=5, sw=1.2)
    lc.text(sx0 + j * (SL_W + SL_GAP) + SL_W / 2, PY0 + 70, f'块 {j}', 9.4, lc.C_TXT, 'middle', True,
            maxw=SL_W - 6, tag='p2:s%d' % j)
lc.text(sx0 + 5 * (SL_W + SL_GAP) + 6, PY0 + 70, '… 共 N 块', 9.4, lc.C_MUTE, 'start',
        maxw=P2X + P2W - 16 - (sx0 + 5 * (SL_W + SL_GAP) + 6), tag='p2:ell')
for j, note in enumerate([
        '块号是池子的唯一词汇：每块等宽，容量 = 块数 × 块宽',
        '块号只数数、不认识层——不同页宽的层要进同一个池，必须先把「宽」归一']):
    lc.text(P2X + 16, PY0 + 118 + j * 20, note, 8.8, '#334155' if j == 0 else lc.C_MUTE,
            'start', maxw=P2W - 32, tag='p2:n%d' % j)

# ================= 三梯级阶梯 =================
LY0 = PY0 + PH + 24
COL_W, COL_GAP = 424, 24
CX = [MX + i * (COL_W + COL_GAP) for i in range(3)]
SC = 3.75                                    # 阶梯区的每字节像素宽


def bar(x, y, page, color, fill, h=22, label=None, lab_fs=8.6):
    lc.rect(x, y, page * SC, h, fill, color, rx=2, sw=1.2)
    if label:
        lc.text(x + page * SC / 2, y + h / 2 + 3, label, lab_fs, color, 'middle', True,
                maxw=page * SC - 4, tag='bar:' + label[:10])


def col_header(cx, title, verdict, vc):
    lc.rect(cx, LY0, COL_W, 28, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
    lc.text(cx + 12, LY0 + 19, title, 10.8, lc.C_TXT, 'start', True, maxw=COL_W - 150, tag='ch:' + title[:8])
    vw = lc.tw(verdict, 8.8, True) + 14
    lc.rect(cx + COL_W - vw - 8, LY0 + 5, vw, 19, '#ffffff', vc, rx=9, sw=1.2)
    lc.text(cx + COL_W - vw / 2 - 8, LY0 + 18.5, verdict, 8.8, vc, 'middle', True, maxw=vw - 4,
            tag='vd:' + verdict[:8])


# ---- 梯① 全同 ----
c0 = CX[0]
col_header(c0, '梯① 全同（32 / 32）', '✓ 无需归一', C_OK)
vy = LY0 + 42
slot_w, slot_h = PAGE_A * SC + 24, 96
lc.rect(c0 + 14, vy, slot_w, slot_h, '#ffffff', lc.C_MUTE, rx=5, sw=1.2)
lc.text(c0 + 14 + slot_w / 2, vy + slot_h + 16, '块 0', 8.6, lc.C_MUTE, 'middle', tag='t1:slot')
bar(c0 + 26, vy + 16, PAGE_A, CA, CA_F, label='层 A₁ 32 B')
bar(c0 + 26, vy + 54, PAGE_A, CA, '#dbeafe', label='层 A₂ 32 B')
lc.text(c0 + 14, vy + slot_h + 44, '两层页宽全同 → 直接同池，零调整', 9.0, lc.C_TXT, 'start', True,
        maxw=COL_W - 28, tag='t1:l1')
lc.text(c0 + 14, vy + slot_h + 62, '多数模型正是这种：所有层一种形状', 8.6, lc.C_MUTE, 'start',
        maxw=COL_W - 28, tag='t1:l2')

# ---- 梯② 同型异宽·整除 ----
c1 = CX[1]
col_header(c1, '梯② 同型异宽 · 整除（32 / 96）', '✓ 调大块长·无损', C_OK)
vy2 = LY0 + 42
# before：两层宽不齐
bar(c1 + 14, vy2, PAGE_A, CA, CA_F, 20, 'A 32 B')
bar(c1 + 14, vy2 + 26, PAGE_B, CB, CB_F, 20, 'B 96 B')
lc.text(c1 + 14 + PAGE_A * SC + 10, vy2 + 16, '等宽的块装不下两种宽', 8.4, C_DEAD, 'start', True,
        maxw=COL_W - 24 - PAGE_A * SC - 10, tag='t2:mis')
lc.seg(c1 + 14 + 40, vy2 + 56, c1 + 14 + 40, vy2 + 76, lc.C_MUTE, 1.6, 'std')
lc.text(c1 + 14 + 52, vy2 + 70, '小层块长 16→48 token（×3）→ 页宽 32→96', 8.8, AMBER, 'start', True,
        maxw=COL_W - 70, tag='t2:xfm')
vy3 = vy2 + 82
slot2_w = PAGE_B * SC + 16
lc.rect(c1 + 14, vy3, slot2_w, 76, '#ffffff', lc.C_MUTE, rx=5, sw=1.2)
bar(c1 + 22, vy3 + 12, PAGE_B, CA, CA_F, 20, 'A 96 B（48 token × 2 B）')
bar(c1 + 22, vy3 + 44, PAGE_B, CB, CB_F, 20, 'B 96 B')
lc.text(c1 + 14, vy3 + 92, '96 % 32 = 0（整除）→ 调大块长即同宽', 9.0, lc.C_TXT, 'start', True,
        maxw=COL_W - 28, tag='t2:l1')
lc.text(c1 + 14, vy3 + 110, '每 token 字节不变——每 token 成本不变，无损对齐', 8.6, lc.C_MUTE,
        'start', maxw=COL_W - 28, tag='t2:l2')

# ---- 梯③ 跨型·不整除 ----
c2 = CX[2]
col_header(c2, '梯③ 跨型 · 不整除（32 / 80）', '△ 唯一解=时间复用', AMBER)
vy4 = LY0 + 42
slot3_w = PAGE_C * SC + 16
lc.rect(c2 + 14, vy4, slot3_w, 76, '#ffffff', lc.C_MUTE, rx=5, sw=1.2)
bar(c2 + 22, vy4 + 12, PAGE_A, CA, CA_F, 20, 'A 32 B')
bar(c2 + 22, vy4 + 44, PAGE_C, CB, CB_F, 20, 'C 80 B')
# 重叠区红虚线框：offset 0 起的 32 B 同时是 A 与 C 的条带
ov_x, ov_w = c2 + 22, PAGE_A * SC
lc.rect(ov_x - 3, vy4 + 9, ov_w + 6, 68, 'none', C_DEAD, rx=3, sw=1.4, dash=True)
lc.text(c2 + 14 + slot3_w + 10, vy4 + 24, '同一块号', 8.4, C_DEAD, 'start', True,
        maxw=CX[2] + COL_W - 14 - (c2 + 14 + slot3_w + 10), tag='t3:ov')
lc.text(c2 + 14 + slot3_w + 10, vy4 + 40, '物理重叠', 8.4, C_DEAD, 'start', True,
        maxw=CX[2] + COL_W - 14 - (c2 + 14 + slot3_w + 10), tag='t3:ov2')
lc.text(c2 + 14 + slot3_w + 10, vy4 + 60, 't₁ 归 A 组', 8.2, lc.C_MUTE, 'start',
        maxw=CX[2] + COL_W - 14 - (c2 + 14 + slot3_w + 10), tag='t3:t1')
lc.text(c2 + 14 + slot3_w + 10, vy4 + 74, 't₂ 归 C 组', 8.2, lc.C_MUTE, 'start',
        maxw=CX[2] + COL_W - 14 - (c2 + 14 + slot3_w + 10), tag='t3:t2')
# 三条死路 chips
dy = vy4 + 96
dead = ['调块长 ✗ 不整除', '垫 pad ✗ 浪费', '拆池 ✗ 浪费']
dx = c2 + 14
for dchip in dead:
    dw = lc.tw(dchip, 8.4, True) + 12
    lc.rect(dx, dy, dw, 19, '#fef2f2', C_DEAD, rx=9, sw=1.0)
    lc.text(dx + dw / 2, dy + 13.5, dchip, 8.4, C_DEAD, 'middle', True, maxw=dw - 4, tag='dd:' + dchip[:6])
    dx += dw + 8
lc.text(c2 + 14, dy + 40, '80 % 32 = 16 ≠ 0（不整除）——三条出路全死', 9.0, lc.C_TXT, 'start', True,
        maxw=COL_W - 28, tag='t3:l1')
lc.text(c2 + 14, dy + 58, '唯一无损解：共用块号、每组块内各占各的偏移，', 8.6, AMBER, 'start', True,
        maxw=COL_W - 28, tag='t3:l2')
lc.text(c2 + 14, dy + 74, '一个块号同一时刻只归一组（重叠时间复用）', 8.6, AMBER, 'start', True,
        maxw=COL_W - 28, tag='t3:l3')

# ================= 底部：判定链 + 指针 =================
SY = LY0 + 44 + 96 + 78                      # 对齐三列最深正文底
SY = max(SY, vy3 + 126, dy + 90) + 16
lc.rect(MX, SY, BXR - MX, 108, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(MX + 16, SY + 22, '判定链：两种层进同一个池，先问页宽关系', 10.2, lc.C_TXT, 'start', True,
        maxw=560, tag='sy:t')
# 根节点 → 三分支
RN_W, RN_H = 150, 34
rn_x, rn_y = MX + 26, SY + 44
lc.rect(rn_x, rn_y, RN_W, RN_H, '#ffffff', lc.C_MUTE, rx=6, sw=1.3)
lc.text(rn_x + RN_W / 2, rn_y + 21, '读两层页宽', 9.4, lc.C_TXT, 'middle', True, maxw=RN_W - 8, tag='rn')
BR = [('全同 → 同池直装', C_OK, '多数模型'),
      ('同型且整除 → 调大块长', C_OK, '每 token 成本不变'),
      ('都不满足 → 重叠时间复用', AMBER, '唯一无损解（梯③）')]
bx0 = rn_x + RN_W + 40
bw_tot = BXR - 24 - bx0
bw = (bw_tot - 2 * 16) / 3
for k, (t, vc, sub) in enumerate(BR):
    bxx = bx0 + k * (bw + 16)
    lc.rect(bxx, rn_y, bw, RN_H, '#ffffff', vc, rx=6, sw=1.3)
    lc.text(bxx + bw / 2, rn_y + 15, t, 9.2, vc, 'middle', True, maxw=bw - 8, tag='br%d' % k)
    lc.text(bxx + bw / 2, rn_y + 29, sub, 8.0, lc.C_MUTE, 'middle', maxw=bw - 8, tag='brs%d' % k)
    lc.parrow([(rn_x + RN_W, rn_y + RN_H / 2), (rn_x + RN_W + 20, rn_y + RN_H / 2),
               (bxx, rn_y + RN_H / 2)], lc.C_MUTE, 1.3, 'std')
lc.text(MX + 16, SY + 98, '梯③ 的解就是 V4 packed 布局的理论必然性——代码形态见后文 packed 切片图；纯理论到此为止',
        9.4, AMBER, 'start', True, maxw=BXR - MX - 32, tag='ptr')

# ================= 图例 + 页脚 =================
GY = SY + 122
lx = MX
for cs_, name in [(CA, 'A 层（2 B/token）'), (CB, 'B / C 层（B 层 6 B/token；C 层页宽 80）')]:
    lc.rect(lx, GY - 9, 18, 12, '#ffffff', cs_, rx=3, sw=1.5)
    lc.text(lx + 24, GY, name, 8.8, lc.C_TXT, 'start', maxw=320, tag='gl' + name[:6])
    lx += 24 + lc.tw(name, 8.8) + 20
lc.rect(lx, GY - 10, 34, 15, '#ffffff', C_OK, rx=7, sw=1.2)
lc.text(lx + 17, GY, '✓ 可行', 8.4, C_OK, 'middle', True, maxw=30, tag='gl:ok')
lx += 42
lc.rect(lx, GY - 10, 34, 15, '#fef2f2', C_DEAD, rx=7, sw=1.2)
lc.text(lx + 17, GY, '✗ 死路', 8.4, C_DEAD, 'middle', True, maxw=30, tag='gl:dead')
lx += 42
lc.text(lx, GY, '红虚线框 = 跨组物理重叠区', 8.8, lc.C_TXT, 'start', maxw=BXR - lx, tag='gl:ov')
lc.text(MX, GY + 20, '玩具例为 2 层示教算术（页宽 32 / 96 / 80，块长 16→48）——数字与代码侧页宽换算同源；'
                     '本图零代码标识符，对应代码形态见后文 packed 切片图',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ================= 装配输出 =================
H = int(GY + 40)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-hybrid-theory-model.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
