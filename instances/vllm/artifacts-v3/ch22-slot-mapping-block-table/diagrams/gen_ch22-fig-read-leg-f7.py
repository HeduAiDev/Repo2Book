#!/usr/bin/env python3
"""ch22 机制图 6 · 读腿穿表 vs 写腿直寻址——同一张页表的两副面孔（F7 回收）
（figure_spec ch22-fig-read-leg-f7，模板 tensor-flow）

放大自 L0『GPU 执行臂』（gpu_column 绿列）——即本章 L2 章图 south『读腿 ·
flash_attn 穿表（F7）』（站 14）的机制展开。架构归属回指 L0/L2
（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：读腿与写腿寻址方式对偶（F7 回收）：写按 slot_mapping 每 token 直寻址
（slot 96/97 直落块 6），读穿 block_table 逻辑块→物理块间接寻址
（逻辑 0→3、1→8、2→6）——同一张页表的两副面孔。

数字全部取自 figure_spec.numbers（3/8/6/96/97/34/16/2/5.96e-08，精简版 host 实跑：
读腿逐逻辑块 gather 与稠密参照逐元素对拍）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 762
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '写直寻址、读穿表——同一张页表的两副面孔（F7 回收）',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '写腿拿 slot_mapping 每 token 直落格（96/97 直进块 6，不查表）；读腿 flash_attn_varlen_func(block_table=) 逐逻辑块现场查物理块（逻辑 0→3、1→8、2→6）拼回 34-token 序列',
        10.5, lc.C_MUTE, 'start', maxw=1090, tag='subtitle')
_ch = '放大自 L2 南行 读腿 · flash_attn 穿表（站 14）· L0：GPU 执行臂 × KV 接缝'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- ① 写腿入口（左） ----------------
WP_X, WP_Y, WP_W, WP_H = 60, 150, 270, 152
lc.rect(WP_X, WP_Y, WP_W, WP_H, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.6)
lc.text(WP_X + 14, WP_Y + 20, '① 写腿 · 本拍 2 个新 token', 10, lc.C_GPU_S, 'start', True,
        maxw=WP_W - 28, tag='wp:t')
lc.text(WP_X + 14, WP_Y + 40, 'slot_mapping（token 级）：', 8, '#334155', 'start',
        maxw=WP_W - 150, tag='wp:l')
SLOT_CX, SLOT_CW, SLOT_CH = WP_X + WP_W - 58, 58, 26
for i, v in enumerate((96, 97)):
    y = WP_Y + 30 + i * 40
    lc.rect(SLOT_CX, y, SLOT_CW, SLOT_CH, '#ffffff', lc.C_GPU_S, rx=5, sw=1.6)
    lc.text(SLOT_CX + SLOT_CW / 2, y + 17.5, str(v), 11, lc.C_GPU_S, 'middle', True,
            maxw=SLOT_CW - 8, tag=f'wp:v{i}')
lc.text(WP_X + 14, WP_Y + 118, 'slot = 6×16 + {0,1} = 96 / 97', 8.4, lc.C_GPU_S, 'start',
        True, maxw=WP_W - 28, tag='wp:f')
lc.text(WP_X + 14, WP_Y + 136, 'slot 已含块号——kernel 除一下模一下就落格', 7.6, '#334155',
        'start', maxw=WP_W - 28, tag='wp:n')

# ---------------- ② 块表行（中） ----------------
TT_X, TT_W = 430, 210
lc.text(TT_X, 148, '② 块表行（读腿要穿的页表）· [3, 8, 6]', 9.5, lc.C_KV_S, 'start', True,
        maxw=330, tag='tt:t')
TABLE = [(0, 3, 182), (1, 8, 242), (2, 6, 302)]
for k, v, cy in TABLE:
    lc.rect(TT_X, cy - 22, TT_W, 44, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.5)
    lc.text(TT_X + 14, cy + 4, f'逻辑块 {k}', 9, '#334155', 'start', maxw=80, tag=f'tt:k{k}')
    lc.text(TT_X + 105, cy + 4, '→', 10, lc.C_KV_S, 'middle', True, maxw=20, tag=f'tt:a{k}')
    lc.text(TT_X + 196, cy + 4, f'物理块 {v}', 11, lc.C_KV_S, 'end', True, maxw=100,
            tag=f'tt:v{k}')

# ---------------- ③ KV 池物理块（右列三栈） ----------------
STK_X, STK_W, RP = 950, 220, 8.5
STACKS = [(3, 140, 'hist', '16 行（pos 0–15）'),
          (8, 300, 'hist', '16 行（pos 16–31）'),
          (6, 460, 'fresh', '2 行刚写（pos 32–33）· 容量 16')]
for bid, sy, mode, cap in STACKS:
    lc.text(STK_X + STK_W / 2, sy - 8, f'③ 块 {bid}', 10, lc.C_KV_S, 'middle', True, maxw=120,
            tag=f'st:{bid}')
    for r in range(16):
        y = sy + r * RP
        if mode == 'fresh' and r < 2:
            lc.rect(STK_X, y, STK_W, RP - 1, lc.C_GPU_S, lc.C_GPU_S, rx=1.5, sw=0)
            lc.text(STK_X + STK_W / 2, y + 6, f'slot={bid * 16 + r}（刚写）', 6.8, '#ffffff',
                    'middle', True, maxw=STK_W - 6, tag=f'fresh{r}')
        else:
            lc.rect(STK_X, y, STK_W, RP - 1, lc.C_KV_F if mode == 'hist' else '#f8fafc',
                    '#cbd5e1', rx=1.5, sw=0.7)
    lc.text(STK_X, sy + 16 * RP + 15, cap, 7.5, lc.C_MUTE, 'start', maxw=240,
            tag=f'cap:{bid}')
STK_MID = {3: 140 + 8 * RP, 8: 300 + 8 * RP, 6: 460 + 8 * RP}

# 读腿肘形箭头：②→③（逐逻辑块现场查行）
for (k, v, cy), chx in zip(TABLE, (795, 810, 825)):
    lc.parrow([(TT_X + TT_W, cy), (chx, cy), (chx, STK_MID[v]), (STK_X - 2, STK_MID[v])],
              lc.C_KV_S, 1.6, 'std')
lc.text(836, 430, '间接寻址：', 8.2, lc.C_KV_S, 'start', True, maxw=110, tag='rl:t')
lc.text(836, 444, '逐逻辑块', 8.2, lc.C_KV_S, 'start', maxw=110, tag='rl:2')
lc.text(836, 458, '现场查行', 8.2, lc.C_KV_S, 'start', maxw=110, tag='rl:3')

# 写腿大弧：左面板 → 顶部 → 右侧通道 → 块 6 右缘（不碰表）
ROW6 = {0: 460 + 0 * RP + (RP - 1) / 2, 1: 460 + 1 * RP + (RP - 1) / 2}
lc.parrow([(WP_X + WP_W, WP_Y + 43), (370, WP_Y + 43), (370, 100), (1240, 100),
           (1240, ROW6[0]), (STK_X + STK_W + 2, ROW6[0])], lc.C_GPU_S, 2.0, 'std')
lc.parrow([(WP_X + WP_W, WP_Y + 83), (378, WP_Y + 83), (378, 108), (1248, 108),
           (1248, ROW6[1]), (STK_X + STK_W + 2, ROW6[1])], lc.C_GPU_S, 2.0, 'std')
lc.text(560, 92, '直寻址：slot 现成——6×16+{0,1} = 96 / 97 直接落块 6，不经过任何表', 8.4,
        lc.C_GPU_S, 'start', True, maxw=560, tag='wl:lab')

# ---------------- ④ 对拍证据盒（底） ----------------
EV_Y = 626
lc.rect(TT_X, EV_Y, 900, 66, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(TT_X + 14, EV_Y + 18, '④ 读回来的对账（F7 回收的收尾）：拼回 34-token 逻辑序列算 causal attention', 9.2,
        lc.C_TXT, 'start', True, maxw=872, tag='ev:t')
lc.text(TT_X + 14, EV_Y + 37, '块 3 的 16 行（pos 0–15）+ 块 8 的 16 行（pos 16–31）+ 块 6 的 2 行（pos 32–33）= 34 token——逻辑连续、物理乱放', 8,
        '#334155', 'start', maxw=872, tag='ev:1')
lc.text(TT_X + 14, EV_Y + 55, '与「按块表行拼回逻辑序列」的稠密参照逐元素对拍：max|diff| = 5.96e-08（浮点噪声）——写直、读间，闭合', 8,
        lc.C_KV_S, 'start', True, maxw=872, tag='ev:2')

# ---------------- 页脚 ----------------
lc.text(MX, 716, '图例：绿 = 写腿（直寻址，不过表） · 青 = 页表与 KV 池（读腿穿表间接寻址） · 绿格 = 本拍刚写的行',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 732, '逐字锚 vllm/v1/attention/backends/flash_attn.py:L1041-L1066（读腿 flash_attn_varlen_func(block_table=)）· L934（从 metadata 取 block_table）· cache_kernels.cu:L326-L333（写腿落格）· vllm/v1/worker/block_table.py:L410-L442（同一张表的装配侧）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 748, '读数（块表行 [3,8,6]、slot 96/97、34 token、各块行数、对拍 5.96e-08）取自精简版 host 实跑（读腿逐逻辑块 gather vs 稠密参照对拍）',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch22-fig-read-leg-f7.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
