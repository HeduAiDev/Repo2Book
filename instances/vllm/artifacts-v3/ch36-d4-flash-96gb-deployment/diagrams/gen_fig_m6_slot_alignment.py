#!/usr/bin/env python3
"""ch36 机制图 · 584B 槽位与 576 对齐税（figure_spec fig_m6_slot_alignment，模板 layout）

放大自 L2 章图拍片⑤（五类 spec 自报）里『页怎么算出来』——L0 kv_column 的字节级放大。

claim：584B/token 槽位 = 448B NoPE + 128B RoPE + 8B fp8 scale（head_size 保持语义值
512 不进算式）；每页再补齐到 576 的倍数——对齐税对页越小越重：C128 主账 1,168→1,728
（垫 47.95%）而三本大页账只垫 0.17-0.20%。

数字全部取自 figure_spec.numbers（kv_cache_interface.py:L411-L413 / L403-L405 /
L353-L359 / pages 表）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1440, 742
MX, BXR = 56, 1384
DEFS = lc.DEFS

C_RAW_S, C_RAW_F = lc.C_KV_S, lc.C_KV_F    # 载荷字节 = 青
C_PAD_S, C_PAD_F = '#d97706', '#fde68a'    # 垫字节 = 琥珀
C_NOPE_S, C_NOPE_F = lc.C_API_S, lc.C_API_F        # NoPE 段 = 蓝
C_ROPE_S, C_ROPE_F = lc.C_ZMQ_S, lc.C_ZMQ_F        # RoPE 段 = 紫
C_SCL_S, C_SCL_F = lc.C_SAM_S, lc.C_SAM_F          # scale 段 = 品红

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一个 token 的行李 584B，行李箱必须按 576 的倍数定制——对齐税累退', 16.5,
        lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '槽位 = 448B NoPE + 128B RoPE + 8B fp8 scale（head_size 保持语义值 512，不进算式）· 每页 round_up 到 576 的倍数，补齐值在 spec 构造期一次钉死',
        10.5, lc.C_MUTE, 'start', maxw=1260, tag='subtitle')
_ch = '放大自 L2 拍片⑤ · L0 显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：槽位解剖 ----------------
LP_W = 596
lc.rect(MX, 100, LP_W, 240, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(MX + 18, 126, '一个存储 token 的槽位（fp8_ds_mla · UE8M0 块缩放 fp8 打包 uint8）', 11.5,
        lc.C_TXT, 'start', True, maxw=LP_W - 36, tag='lp:t')

BAR_Y, BAR_H = 176, 42
BX0 = MX + 24
SCALE = 0.87                       # px/B：584B → 508px
segs = [(448, '448B NoPE', '无位置编码', C_NOPE_S, C_NOPE_F),
        (128, '128B RoPE', '旋转位置编码', C_ROPE_S, C_ROPE_F),
        (8, '8B', 'fp8 scale', C_SCL_S, C_SCL_F)]
bx = BX0
for nbytes, lab, sub, s, f in segs:
    w_ = nbytes * SCALE                    # 严格按字节比例（8B 段 = 7px，窄但真实）
    lc.rect(bx, BAR_Y, w_, BAR_H, f, s, rx=2, sw=1.4)
    if nbytes >= 64:
        lc.text(bx + w_ / 2, BAR_Y + 18, lab, 9.8, s, 'middle', True, maxw=w_ - 6, tag='seg:' + lab)
        lc.text(bx + w_ / 2, BAR_Y + 34, sub, 8.2, s, 'middle', maxw=w_ - 6, tag='seg:s' + lab)
    bx += w_
# 8B 段太窄（7px）：标签放段外侧，画一根细引线指着它
lc.text(bx + 12, BAR_Y + 18, '8B', 9, C_SCL_S, 'start', True, maxw=30, tag='seg:8b')
lc.text(bx + 12, BAR_Y + 34, 'fp8 scale', 7.8, C_SCL_S, 'start', maxw=60, tag='seg:8bs')
lc.seg(bx + 3, BAR_Y + BAR_H / 2 - 6, bx + 10, BAR_Y + BAR_H / 2 - 6, C_SCL_S, 1.0)

TOT_X = BX0 + 584 * SCALE
lc.seg(BX0 - 8, BAR_Y - 8, BX0 - 8, BAR_Y + BAR_H + 8, '#cbd5e1', 1.2)
lc.seg(TOT_X, BAR_Y - 8, TOT_X, BAR_Y + BAR_H + 8, '#cbd5e1', 1.2)
lc.seg(BX0 - 8, BAR_Y - 14, TOT_X, BAR_Y - 14, '#cbd5e1', 1.2)
lc.text((BX0 + TOT_X) / 2, BAR_Y - 20, '584B / 存储 token（一个槽）', 10, lc.C_TXT, 'middle', True,
        maxw=280, tag='slot:tot')

lc.text(MX + 18, 268, '· storage_block_size = block_size // compress_ratio（窗口/主账共用此槽）', 9.2,
        '#334155', 'start', maxw=LP_W - 36, tag='lp:l1')
lc.text(MX + 18, 286, '· 页 = 槽数 × 584B，再 round_up(page, 576) —— kv_cache_interface.py:L403-L405 · L353-L359',
        9.2, '#334155', 'start', maxw=LP_W - 36, tag='lp:l2')
lc.text(MX + 18, 304, '· 584 = 448 + 128 + 8 的出处：kv_cache_interface.py:L411-L413（同 sparse_mla.py:L104）',
        9.2, '#334155', 'start', maxw=LP_W - 36, tag='lp:l3')

# 图例（左：三段字节；全图另有 载荷/垫 对）
LGY = 316
lx0 = MX + 18
for s, f, name in [(C_NOPE_S, C_NOPE_F, 'NoPE'), (C_ROPE_S, C_ROPE_F, 'RoPE'), (C_SCL_S, C_SCL_F, 'scale')]:
    lc.rect(lx0, LGY - 9, 16, 11, f, s, rx=3, sw=1.4)
    lc.text(lx0 + 21, LGY + 1, name, 9, lc.C_TXT, 'start', maxw=70, tag='lg:' + name)
    lx0 += 21 + lc.tw(name, 9) + 20

# ---------------- 右：五种页的对齐账 ----------------
RP_X, RP_W = MX + LP_W + 22, BXR - (MX + LP_W + 22)
lc.rect(RP_X, 100, RP_W, 268, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(RP_X + 18, 126, '五种页的 576 对齐账（青 = 载荷 · 琥珀 = 垫字节）', 11.5,
        lc.C_TXT, 'start', True, maxw=RP_W - 36, tag='rp:t')

PAGES = [
    ('窗口账（block 64）', '64 槽 × 584B = 37,376 → 37,440', 37376, 37440),
    ('C4A 压缩主账（ratio 4）', '256÷4 = 64 槽 × 584B = 37,376 → 37,440', 37376, 37440),
    ('C128 压缩主账（ratio 128）', '256÷128 = 2 槽 × 584B = 1,168 → 1,728', 1168, 1728),
    ('C4I indexer（fp4 档）', '64 槽 × 68B = 4,352 → 4,608', 4352, 4608),
    ('C4 状态账（fp32 恒定）', '4 × 2048 × 4B = 32,768 → 32,832', 32768, 32832),
]
ROW_Y0, ROW_H = 142, 42
LBL_X, LBL_W = RP_X + 18, 258
BAR_X0, BAR_W = RP_X + 286, 312
for i, (name, formula, raw, padded) in enumerate(PAGES):
    ry = ROW_Y0 + i * ROW_H
    pad = padded - raw
    pct = pad / raw * 100        # 垫% = 垫/原始页（r1 评审口径统一：与正文 m6 表/图注/底注同分母）
    lc.text(LBL_X, ry + 14, name, 9.6, lc.C_TXT, 'start', True, maxw=LBL_W, tag=f'p{i}n')
    lc.text(LBL_X, ry + 28, formula, 8.2, '#475569', 'start', maxw=LBL_W + 20, tag=f'p{i}f')
    raw_w = raw / padded * BAR_W
    lc.rect(BAR_X0, ry + 4, raw_w, 16, C_RAW_F, C_RAW_S, rx=2, sw=1.2)
    lc.rect(BAR_X0 + raw_w, ry + 4, BAR_W - raw_w, 16, C_PAD_F, C_PAD_S, rx=2, sw=1.2)
    lc.text(BAR_X0 + BAR_W + 10, ry + 16, f'+{pad}B（{pct:.2f}%）'.replace('.00%', '%'),
            8.6, C_PAD_S if pct > 5 else lc.C_MUTE, 'start', maxw=96, tag=f'p{i}p')

lc.text(RP_X + 18, ROW_Y0 + 5 * ROW_H + 8, '每行满宽 = 对齐后的页；垫% = 垫字节 ÷ 原始页（条标与底注同口径）；pad 段宽度按同尺度放大可见', 8.4,
        lc.C_FAINT, 'start', maxw=RP_W - 36, tag='rp:n')

# ---------------- 底部结论带 ----------------
BB_Y = 396
lc.rect(MX, BB_Y, BXR - MX, 92, lc.C_KV_F, lc.C_KV_S, rx=10, sw=1.6)
lc.text(MX + 18, BB_Y + 26, '对齐税累退：页越小、付的比例越重，付的绝对值越小', 12.5, lc.C_KV_S,
        'start', True, maxw=700, tag='bb:t')
lc.text(MX + 18, BB_Y + 48, '· 最小的 C128 主账页被垫掉近一半（1,168 → 1,728 · 47.95%），但摊到每 token 只有 6.75B；三本大页账合计垫税 < 0.2%',
        9.4, '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l1')
lc.text(MX + 18, BB_Y + 68, '· 补齐在 spec 构造期一次写入 page_size_padded（frozen dataclass）——分组、定账、寻址全程读到同一补齐值，不会二次漂移',
        9.4, '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l2')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 44, 'vllm/v1/kv_cache_interface.py:L411-L413（584=448+128+8）· L403-L405（storage_block_size = block//ratio）· L353-L359（_apply_alignment_padding · round_up(page,576)）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 29, '页字节 / 垫字节 / 垫% ＝ 本章账本算术脚本实算（构造 pin 真码 spec 类后读 page_size_bytes）· C128 摊销 6.75B/token（页 ÷ 256 token）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 14, '行号基线 vLLM v0.27.1（6e448d0ea）', 8.5, lc.C_FAINT, 'start', maxw=400, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m6_slot_alignment.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
