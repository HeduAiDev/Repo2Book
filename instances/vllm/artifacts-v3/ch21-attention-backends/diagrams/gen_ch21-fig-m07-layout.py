#!/usr/bin/env python3
"""ch21 机制图 ③ · KV 张量布局:逻辑形与 stride 置换(figure_spec ch21-fig-m07-layout,模板 layout)

放大自 L0 GPU 执行臂(绿)× 显存账本列(青)交界:KV 张量定形(站 7)——裸显存如何被
「看成」后端要的形状。架构归属回指 L0(FIGURE-SYSTEM §3.3),不另立第二种架构画法。

claim:后端自报 KV 布局两件套——get_kv_cache_shape 给逻辑形 (B,H,N,2D)(K/V 打包进
content 维),get_kv_cache_stride_order 给逻辑→物理的置换(NHD=(0,2,1,3) 同 token 的
heads 内存连续 / HND=(0,1,2,3) 恒等)——裸显存 as_strided 一次定形。

数字全部取自 figure_spec.numbers:逻辑形 (256,2,64,128)、NHD 步长 (16384,128,256,1)、
HND 步长 (16384,8192,128,1)、stride_order NHD=(0,2,1,3)/HND=(0,1,2,3)、哨兵 _S=1234567
→ 块维 0——本章精简版 host 实测 + pin 源码逐字。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 776
MX = 60
BXR = 1440
C_KV_K, C_KV_V = '#ecfeff', '#cffafe'      # K 半格 / V 半格(同族青系两档)

# ---------------- 标题区 ----------------
lc.text(MX, 34, '同一块裸显存、两种「看法」:逻辑形说四维语义,stride_order 定物理码放',
        16.5, lc.C_TXT, 'start', True, maxw=1050, tag='title')
lc.text(MX, 58, '逻辑形 (B,H,N,2D) 把 K/V 打包进最后一维;NHD 让同一 token 的头在内存里相邻、HND 让同一头的 token 相邻——as_strided 零拷贝切换,块维位置由哨兵自动探测',
        10.5, lc.C_MUTE, 'start', maxw=1100, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(绿)×显存账本列(青)·KV 定形(站 7)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左:逻辑形 ----------------
LX = MX
lc.text(LX, 106, '① 逻辑形 get_kv_cache_shape —— 这块显存怎么被「看成」', 10.5, lc.C_GPU_S,
        'start', True, maxw=560, tag='l:t')
lc.text(LX, 124, '(num_blocks, block_size, num_kv_heads, head_size) → (256, 2, 64, 128)',
        9, lc.C_MUTE, 'start', maxw=560, tag='l:sub')

AX_Y, AX_H = 136, 30
segs = [('B = 256 块', 128, lc.C_GPU_F, lc.C_GPU_S),
        ('H = 2 头', 92, lc.C_GPU_F, lc.C_GPU_S),
        ('N = 64 token', 140, lc.C_GPU_F, lc.C_GPU_S),
        ('2D = 128(K 64 + V 64)', 240, '#ffffff', lc.C_KV_S)]
ax_x = LX + 40
seg_edges = []
for lab, sw_, fill, stroke in segs:
    lc.rect(ax_x, AX_Y, sw_, AX_H, fill, stroke, rx=4, sw=1.3)
    lc.text(ax_x + sw_ / 2, AX_Y + 19, lab, 8.2, (lc.C_KV_S if stroke == lc.C_KV_S else lc.C_GPU_S),
            'middle', maxw=sw_ - 6, tag='ax:' + lab[:6])
    seg_edges.append((ax_x, ax_x + sw_))
    ax_x += sw_ + 6
kv_seg = seg_edges[-1]
lc.seg(kv_seg[0] + 120, AX_Y, kv_seg[0] + 120, AX_Y + AX_H, lc.C_KV_S, 1.2)   # K|V 分界

# 放大镜虚线:轴条 → 块 0 网格
G_X0, G_X1 = 108, 690
G_TOP = 214
lc.seg(seg_edges[0][0], AX_Y + AX_H, G_X0, G_TOP - 14, lc.C_FAINT, 1.1, None, dash=True)
lc.seg(seg_edges[-1][1], AX_Y + AX_H, G_X1, G_TOP - 14, lc.C_FAINT, 1.1, None, dash=True)
lc.text((G_X0 + G_X1) / 2, G_TOP - 20, '放大块 0(只画前 4 个 token,N=64;B=256 块同构)', 8,
        lc.C_FAINT, 'middle', maxw=560, tag='l:zoom')

CELL_W, CELL_H, CGAP = 138, 84, 6
COLS = 4
for r in range(2):
    gy = G_TOP + 16 + r * (CELL_H + CGAP)
    lc.text(G_X0 - 10, gy + CELL_H / 2 + 3, f'头 {r}', 8.5, lc.C_TXT, 'end', True, maxw=40,
            tag=f'l:r{r}')
    for c in range(COLS):
        gx = G_X0 + c * (CELL_W + CGAP)
        lc.rect(gx, gy, CELL_W / 2, CELL_H, C_KV_K, lc.C_KV_S, rx=4, sw=1.2)
        lc.rect(gx + CELL_W / 2, gy, CELL_W / 2, CELL_H, C_KV_V, lc.C_KV_S, rx=4, sw=1.2)
        if r == 0 and c == 0:
            lc.text(gx + CELL_W / 4, gy + CELL_H / 2 + 3, 'K·64', 8.5, lc.C_KV_S, 'middle',
                    True, maxw=CELL_W / 2 - 6, tag='l:k')
            lc.text(gx + 3 * CELL_W / 4, gy + CELL_H / 2 + 3, 'V·64', 8.5, '#0e7490',
                    'middle', True, maxw=CELL_W / 2 - 6, tag='l:v')
        if r == 0:
            lc.text(gx + CELL_W / 2, gy - 6, f'token {c}', 8, lc.C_MUTE, 'middle', maxw=CELL_W,
                    tag=f'l:c{c}')
    lc.text(G_X0 + COLS * (CELL_W + CGAP) + 8, gy + CELL_H / 2 + 3, '⋮', 12, lc.C_FAINT,
            'start', tag=f'l:dots{r}')
lc.text(LX, G_TOP + 16 + 2 * (CELL_H + CGAP) + 18,
        '同一格 = 一个 (token, 头) 的全部 K/V:K 与 V 打包进最后一维 content dim(2*head_size = 2×64 = 128)',
        8.5, lc.C_MUTE, 'start', maxw=680, tag='l:note')

# ---------------- 右:NHD/HND 物理码放 ----------------
RX = 770
lc.text(RX, 106, '② NHD 物理码放 —— 同一块显存怎么「放下」', 10.5, lc.C_KV_S, 'start', True,
        maxw=560, tag='r:t')
lc.text(RX, 124, '物理步长 strides = (16384, 128, 256, 1) → 物理内存序 (B, N, H, 2D)', 9,
        lc.C_MUTE, 'start', maxw=560, tag='r:sub')
stride_lines = [
    ('沿 B 走一步 = 16384 元素', ' = 整块(64 token × 2 头 × 128)'),
    ('沿 N 走一步 = 256 元素', ' = 跳过该 token 的全部头(2 × 128)'),
    ('沿 H 走一步 = 128 元素', ' = 一个 content 维'),
    ('沿 2D 走一步 = 1 元素', ' = 最内层连续'),
]
for k, (a, b) in enumerate(stride_lines):
    lc.text(RX, 148 + k * 17, a, 8.5, lc.C_KV_S, 'start', True, maxw=190, tag=f'r:s{k}a')
    lc.text(RX + 192, 148 + k * 17, b, 8.5, '#334155', 'start', maxw=380, tag=f'r:s{k}b')

MB_Y = 226
lc.text(RX, MB_Y - 6, '块 0 的物理内存条(NHD,只画前 4 个 token;每格 128 元素)', 8.5,
        lc.C_KV_S, 'start', True, maxw=560, tag='r:mb')
CW, CH, CG = 66, 40, 4
for t in range(4):
    px = RX + t * (2 * CW + CG + 14)
    lc.rect(px, MB_Y, 2 * CW + CG, CH, 'none', lc.C_KV_S, rx=5, sw=1.4)
    for h_ in range(2):
        cx = px + h_ * (CW + CG)
        fill = C_KV_K if h_ == 0 else C_KV_V
        lc.rect(cx, MB_Y, CW, CH, fill, lc.C_KV_S, rx=3, sw=1.1)
        lc.text(cx + CW / 2, MB_Y + 24, f't{t}·h{h_}', 7.5, lc.C_KV_S, 'middle', maxw=CW - 4,
                tag=f'r:c{t}{h_}')
    lc.text(px + CW, MB_Y + CH + 14, f'token {t}', 8, lc.C_MUTE, 'middle', maxw=2 * CW,
            tag=f'r:pt{t}')
lc.text(RX, MB_Y + CH + 34, '同一 token 的全部头内存相邻——取一个 token 的头一次拿完(喂 TMA 友好);竖框 = 头 0|头 1 两半格',
        8.5, lc.C_KV_S, 'start', maxw=660, tag='r:mbn')

HB_Y = 348
lc.text(RX, HB_Y - 6, '若是 HND(stride_order=(0,1,2,3) 恒等):同一头的所有 token 连续', 8.5,
        lc.C_MUTE, 'start', True, maxw=560, tag='r:hb')
hnd_cells = ['h0·t0', 'h0·t1', 'h0·t2', 'h0·t3']
hx = RX
for k, cell in enumerate(hnd_cells):
    lc.rect(hx, HB_Y, CW, CH, C_KV_K, lc.C_KV_S, rx=3, sw=1.1)
    lc.text(hx + CW / 2, HB_Y + 24, cell, 7.5, lc.C_KV_S, 'middle', maxw=CW - 4, tag=f'r:h{k}')
    hx += CW + CG
lc.text(hx + 4, HB_Y + 24, '⋮ 到 t63', 8, lc.C_FAINT, 'start', maxw=60, tag='r:hdots')
hx += 72
lc.seg(hx - 6, HB_Y - 2, hx - 6, HB_Y + CH + 2, lc.C_MUTE, 1.4)      # 头分界
for k, cell in enumerate(['h1·t0', 'h1·t1']):
    lc.rect(hx, HB_Y, CW, CH, C_KV_V, lc.C_KV_S, rx=3, sw=1.1)
    lc.text(hx + CW / 2, HB_Y + 24, cell, 7.5, '#0e7490', 'middle', maxw=CW - 4, tag=f'r:h1{k}')
    hx += CW + CG
lc.text(hx + 4, HB_Y + 24, '⋮', 10, lc.C_FAINT, 'start', tag='r:hdots2')
lc.text(RX, HB_Y + CH + 14, '头的分界两侧是不同的 token——两种码放各有取舍,后端按 kernel 的取数模式自选',
        8.5, lc.C_MUTE, 'start', maxw=660, tag='r:hbn')
lc.text(RX, HB_Y + CH + 32, 'HND 步长(按逻辑维序 B,H,N,2D)=(16384, 8192, 128, 1)——头维一步 8192 = 64×128(同头的全部 token)',
        8.5, lc.C_MUTE, 'start', maxw=660, tag='r:hbs')

# ---------------- 底部三块 ----------------
BB_Y, BB_H = 470, 132
# (a) 置换可视化
lc.rect(MX, BB_Y, 500, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 14, BB_Y + 20, '③ stride_order = 逻辑形 → 物理内存序的置换', 9.5, lc.C_TXT,
        'start', True, maxw=460, tag='b1:t')
chip_w, chip_h, chip_g = 46, 22, 10
row0 = ['B', 'H', 'N', '2D']
row1 = ['B', 'N', 'H', '2D']
cy0, cy1 = BB_Y + 34, BB_Y + 82
lx0 = MX + 150
for k in range(4):
    x = lx0 + k * (chip_w + chip_g)
    lc.rect(x, cy0, chip_w, chip_h, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.2)
    lc.text(x + chip_w / 2, cy0 + 15, row0[k], 9, lc.C_GPU_S, 'middle', True, tag=f'b1:a{k}')
    lc.rect(x, cy1, chip_w, chip_h, C_KV_K, lc.C_KV_S, rx=4, sw=1.2)
    lc.text(x + chip_w / 2, cy1 + 15, row1[k], 9, lc.C_KV_S, 'middle', True, tag=f'b1:b{k}')
lc.text(MX + 22, cy0 + 15, '逻辑维序', 8.5, lc.C_GPU_S, 'start', maxw=80, tag='b1:l0')
lc.text(MX + 22, cy1 + 15, 'NHD 物理序', 8.5, lc.C_KV_S, 'start', maxw=80, tag='b1:l1')
xh0 = lx0 + 1 * (chip_w + chip_g) + chip_w / 2          # row0 的 H 中心
xn0 = lx0 + 2 * (chip_w + chip_g) + chip_w / 2          # row0 的 N 中心
xh1 = lx0 + 2 * (chip_w + chip_g) + chip_w / 2          # row1 的 H 中心
xn1 = lx0 + 1 * (chip_w + chip_g) + chip_w / 2          # row1 的 N 中心
midy = (cy0 + chip_h + cy1) / 2
lc.parrow([(xh0, cy0 + chip_h), (xh0, midy), (xn1, midy), (xn1, cy1)], lc.C_KV_S, 1.4, 'std')
lc.parrow([(xn0, cy0 + chip_h), (xn0, midy), (xh1, midy), (xh1, cy1)], lc.C_KV_S, 1.4, 'std')
lc.text(MX + 14, BB_Y + 122, 'NHD=(0,2,1,3):N 提到 H 前;HND=(0,1,2,3):恒等(物理序 = 逻辑序)',
        8, lc.C_MUTE, 'start', maxw=470, tag='b1:n')

# (b) as_strided
b2x = MX + 520
lc.rect(b2x, BB_Y, 380, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(b2x + 14, BB_Y + 20, 'as_strided 一次定形', 9.5, lc.C_TXT, 'start', True,
        maxw=340, tag='b2:t')
for k, ln in enumerate([
        '裸显存 as_strided(shape, strides) 一次切好',
        '只改「怎么看」,不改「怎么放」——零拷贝',
        '混布组块维分歧(K/V-first vs blocks-first)时',
        '再 as_strided_ 统一成 blocks-first(本章末节)']):
    lc.text(b2x + 14, BB_Y + 42 + k * 18, ln, 8.5, '#334155', 'start', maxw=350,
            tag=f'b2:l{k}')

# (c) 哨兵探测
b3x = b2x + 400
b3w = BXR - b3x
lc.rect(b3x, BB_Y, b3w, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(b3x + 14, BB_Y + 20, '块维在第几维?哨兵探测,不逐家手写', 9.5, lc.C_TXT, 'start',
        True, maxw=b3w - 28, tag='b3:t')
for k, ln in enumerate([
        'get_kv_cache_block_dim:把 _S = 1234567 当 num_blocks 代入 shape,',
        '再 shape.index(_S) 反查块维 → 0(blocks-first)',
        '不同后端把块维放在不同位置,哨兵一探即知——',
        '后端与归一逻辑都不必手写维号']):
    lc.text(b3x + 14, BB_Y + 42 + k * 18, ln, 8.5, '#334155', 'start', maxw=b3w - 26,
            tag=f'b3:l{k}')

# ---------------- 页脚 ----------------
FY = BB_Y + BB_H + 24
lc.text(MX, FY, '图例:绿 = 逻辑形(后端声明的四维语义) · 青 = 物理显存码放(K/V 两档同族色) · 灰 = 语义与工具注 · ⋮ = 截断(实际 N=64)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/v1/attention/backends/flash_attn.py:L133-L144(逻辑形)· L146-L168(stride_order NHD/HND)· vllm/v1/attention/backend.py:L100-L117(哨兵探测)· 形状/步长/块维=0 取自本章精简版 host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch21-fig-m07-layout.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
