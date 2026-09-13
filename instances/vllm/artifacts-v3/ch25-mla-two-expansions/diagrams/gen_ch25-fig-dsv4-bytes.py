#!/usr/bin/env python3
"""ch25 机制图 · DSV4 第三代:压柜子→合租格(figure ch25-fig-dsv4-bytes,模板 before-after)

放大自 L0『模型层 MLA 框』在 DSV4 的第三代演化——cache 布局与压缩比升级
(L2 站 5/6 的 spec 自报在 DSV4 侧的新面)。三列储物柜:DSV3 bf16 / DSV3.2 fp8_ds_mla /
DSV4 fp8_ds_mla(每格覆盖 4 token);格内三段着色标字节;底部每 token 有效字节比例条
1152/656/146 与节省比 1.76/7.9;附 compress_ratios 逐层色带 [1,4,128,4,1]。

claim:DSV4 把每存储格压到 584B(448B fp8 NoPE + 128B bf16 RoPE 不量化 + 8B scale)且每格
覆盖 4 个 token——每 token 有效 146B,对照 DSV3 bf16 1152B 压 7.9 倍;RoPE 段三代始终不量化。

数字全部取自 figure spec 的 numbers(三代 1152/656/584 构成 · 节省比 1.76/7.9 ·
storage_block_size 256//4=64、real_page_size 64×584=37376 B、对齐 37440、语义 head_size 512 ·
compress_ratios [1,4,128,4,1] 与护栏 · 584B 构成 kv_cache_interface.py:L409-L413)。
坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 750
MX = 52
BXR = 1448
C_NOPE = lc.C_GPU_S          # NoPE 段 = 绿
C_NOPE_F = '#dcfce7'
C_ROPE = lc.C_API_S          # RoPE 段 = 蓝(bf16 不量化)
C_ROPE_F = '#dbeafe'
C_SC = lc.C_SAM_S            # scale 段 = 品红
C_SC_F = '#fdf2f8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '从『压柜子』到『四人合租一格』:DSV4 把每 token 有效字节压到 146B',
        16, lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '第一级换格式(bf16→fp8:1152→656B,NoPE 量化、scale 补账);第二级换粒度(compress_ratio=4:每 4 token 挤一格 584B)',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· DSV4 侧的新面(L2 站 5/6)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 三列储物柜 ----------------
CY0, CW_, CH_ = 112, 420, 240
cols = [
    ('DSV3 · bf16', '每 token 一格:1152 B', [('512 bf16 NoPE', 1024, C_NOPE, C_NOPE_F),
                                              ('64 bf16 RoPE', 128, C_ROPE, C_ROPE_F)], 1,
     'block 64 · page 73728 B', '576 元素 × 2B'),
    ('DSV3.2 · fp8_ds_mla', '每 token 一格:656 B', [('512B fp8 NoPE', 512, C_NOPE, C_NOPE_F),
                                                     ('16B scale(4×fp32)', 16, C_SC, C_SC_F),
                                                     ('128B bf16 RoPE', 128, C_ROPE, C_ROPE_F)], 1,
     'block 256 · page 167936 B', '省 1.76 倍'),
    ('DSV4 · fp8_ds_mla', '每存储格 584 B(覆盖 4 token)', [('448B fp8 NoPE', 448, C_NOPE, C_NOPE_F),
                                                          ('8B scale(7×ue8m0+1B pad)', 8, C_SC, C_SC_F),
                                                          ('128B bf16 RoPE', 128, C_ROPE, C_ROPE_F)], 4,
     'storage 256//4=64 · page 37376 B(对齐 37440)', '每 token 有效 146B——省 7.9 倍'),
]
gx0, gap = 64, 38
for ci, (nm, sub, segs, cov, foot, ratio) in enumerate(cols):
    x = gx0 + ci * (CW_ + gap)
    lc.rect(x, CY0, CW_, CH_, '#ffffff', lc.C_MUTE, rx=9, sw=1.6)
    lc.text(x + CW_ / 2, CY0 + 24, nm, 12, lc.C_TXT, 'middle', True, tag='c%d:t' % ci)
    lc.text(x + CW_ / 2, CY0 + 44, sub, 9, '#334155', 'middle', True, maxw=CW_ - 20, tag='c%d:s' % ci)
    # 存储格:格内按字节比例分段(宽度∝字节)
    cell_w = CW_ - 60
    total_b = sum(s[1] for s in segs)
    cy = CY0 + 66
    lc.rect(x + 30, cy, cell_w, 46, '#f8fafc', lc.C_MUTE, rx=5, sw=1.2)
    sx = x + 30
    for nm2, byt, stk, fl in segs:
        w_ = cell_w * byt / total_b
        lc.rect(sx, cy, w_, 46, fl, stk, rx=0, sw=1.0)
        sx += w_
    # token → 格 箭头(紧贴格下,先于文字标注,避免穿字)
    ty = cy + 66
    if cov == 1:
        lc.text(x + CW_ / 2, ty + 4, '1 token → 1 格', 8, lc.C_MUTE, 'middle', tag='c%d:cov' % ci)
    else:
        for k in range(4):
            tx = x + CW_ / 2 - 120 + k * 66
            lc.text(tx, ty, 't%d' % (k + 1), 7.5, lc.C_MUTE, 'middle', tag='c%d:tk%d' % (ci, k))
            lc.seg(tx, ty + 6, x + CW_ / 2 - 9 + k * 6, cy + 46, lc.C_MUTE, 1.0, 'std')
        lc.text(x + CW_ / 2 + 108, ty + 4, '4 token → 1 格', 8, lc.C_MUTE, 'middle', tag='c%d:cov' % ci)
    # 段标注(两行,位于 token 行之下)
    lc.text(x + CW_ / 2, cy + 96, ' + '.join('%s' % s[0] for s in segs), 7.5, '#334155', 'middle',
            maxw=CW_ - 24, tag='c%d:seg' % ci)
    lc.text(x + CW_ / 2, cy + 114, '= %d B' % total_b, 9, lc.C_TXT, 'middle', True, tag='c%d:tot' % ci)
    lc.text(x + CW_ / 2, CY0 + CH_ - 34, foot, 7.5, lc.C_MUTE, 'middle', maxw=CW_ - 20, tag='c%d:f' % ci)
    lc.text(x + CW_ / 2, CY0 + CH_ - 16, ratio, 9, lc.C_KV_S if ci else lc.C_MUTE, 'middle', True,
            maxw=CW_ - 20, tag='c%d:r' % ci)

# ---------------- 底部:每 token 有效字节比例条 ----------------
BY = 400
lc.rect(64, BY, 1380, 118, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.5)
lc.text(84, BY + 22, '每 token 有效字节(严格比例条):RoPE 段 128B 三代始终 bf16 不量化——量化收益最差、损伤最险的段', 10,
        lc.C_KV_S, 'start', True, maxw=1100, tag='ba:t')
bars = [('DSV3 1152 B', 1152, lc.C_MUTE, '#e2e8f0'), ('DSV3.2 656 B', 656, C_NOPE, C_NOPE_F),
        ('DSV4 146 B(584÷4)', 146, C_ROPE, C_ROPE_F)]
bx0, bw_max, bh = 84, 640, 16
for i, (nm, val, stk, fl) in enumerate(bars):
    y = BY + 36 + i * 26
    w_ = bw_max * val / 1152
    lc.rect(bx0, y, max(w_, 3), bh, fl, stk, rx=2, sw=1.0)
    lc.text(bx0 + max(w_, 3) + 10, y + 12, nm, 8.5, lc.C_TXT, 'start', maxw=240, tag='ba:l%d' % i)
lc.text(84, BY + 118 - 10, '节省比:DSV3.2 / DSV3 = 1.76;DSV4 / DSV3 = 7.9(146 = 584 ÷ compress_ratio 4)', 9,
        lc.C_KV_S, 'start', True, maxw=900, tag='ba:r')

# ---------------- 右下:compress_ratios 逐层色带 + 记法 ----------------
SB = (64, 542, 1380, 108)
lc.rect(*SB, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(84, 564, 'compress_ratios 逐层可以不同:5 层迷你模型 config [1, 4, 128, 4, 1] 原样解析(128 档也合法)', 9.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='sb:t')
band_y, band_h = 578, 26
bx1 = 84
ratios = [1, 4, 128, 4, 1]
band_colors = ['#e2e8f0', '#bbf7d0', '#86efac', '#bbf7d0', '#e2e8f0']
for i, r in enumerate(ratios):
    w_ = 100
    lc.rect(bx1 + i * (w_ + 8), band_y, w_, band_h, band_colors[i], lc.C_MUTE, rx=4, sw=1.1)
    lc.text(bx1 + i * (w_ + 8) + w_ / 2, band_y + 11, 'layer %d' % i, 8, '#334155', 'middle', True,
            tag='sb:cy%d' % i)
    lc.text(bx1 + i * (w_ + 8) + w_ / 2, band_y + 22, 'ratio %d' % r, 8, '#334155', 'middle', True,
            tag='sb:cr%d' % i)
lc.text(700, band_y + 6, '· MTP 层(layer_id ≥ num_hidden_layers)恒 1', 8.5, '#334155', 'start',
        maxw=330, tag='sb:l1')
lc.text(700, band_y + 22, '· max(1, ·) 护栏把 config 值 0 钳成 1(除数恒 ≥1)', 8.5, '#334155',
        'start', maxw=330, tag='sb:l2')
lc.text(700, band_y + 38, '· 语义 head_size 保持 512;字节在 real_page_size_bytes 特算', 8.5,
        '#334155', 'start', maxw=330, tag='sb:l3')
lc.text(1080, band_y + 6, '· 584B 布局:448B fp8 NoPE + 128B bf16', 8.5, '#334155', 'start',
        maxw=340, tag='sb:l4')
lc.text(1080, band_y + 22, '  RoPE(不量化保精度)+ 8B scale', 8.5, '#334155', 'start', maxw=340,
        tag='sb:l5')
lc.text(1080, band_y + 38, '· 两本账分开记:语义 head_size 512 ≠ 物理字节', 8.5, '#334155',
        'start', maxw=340, tag='sb:l6')
lc.text(84, SB[1] + SB[3] - 10, '进出 compress_ratio>1 的层都要过 DeepseekCompressor(compress→norm→RoPE→store 融合核)——读写代价与命中语义随之改变(indexer 只对压缩后的候选选 top-k,归 ch26 预告)',
        8, lc.C_MUTE, 'start', maxw=1340, tag='sb:n')

# ---------------- 页脚 ----------------
lc.text(MX, 700, '图例:绿段 = NoPE(可量化) · 蓝段 = RoPE(三代恒 bf16 不量化) · 品红段 = scale · 格内分段宽度∝字节',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 720, '布局/字节数 = 真实例化 host 实测(vLLM v0.27.1, 6e448d0ea;kv_cache_interface.py:L403-L426 特账与 L409-L413 注释) · 压缩比解析 = deepseek_v4/attention.py:L210-L213',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-dsv4-bytes.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
