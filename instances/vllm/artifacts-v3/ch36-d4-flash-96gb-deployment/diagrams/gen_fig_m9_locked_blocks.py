#!/usr/bin/env python3
"""ch36 机制图 · 张量共享锁死块大小（figure_spec fig_m9_locked_blocks，模板 before-after）

放大自 L2 章图 south『why · 五本账一个池』注里『块大小不是自由旋钮』的展开——
L0 kv_column 页形状一格的字节级放大。

claim：块大小被物理张量共享锁死：SWA block 64 由 C4A 压缩块 [256//4, 584]=[64, ·] 定形
（sparse_swa.py:L77-L82 注释原话 'Block size is constrained by tensor sharing'）；状态账
block 4/8 由 [4, 2·512·2·4] / [8, 512·2·4] 定形（compressor.py:L177-L189 同款注释）——
两处 TODO 自认技术债，代价换零拷贝共享。

数字全部取自 figure_spec.numbers。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1440, 768
MX, BXR = 56, 1384
DEFS = lc.DEFS

C_TEN_S, C_TEN_F = lc.C_KV_S, lc.C_KV_F        # 物理张量 = 青
C_RES_S, C_RES_F = lc.C_GPU_S, lc.C_GPU_F      # 住户 = 绿
C_QUOTE = '#7c2d12'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '谁定了谁的块：块大小不是 config 旋钮，是物理张量共享的下游结果', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '同一张物理张量必须同一页形状——后住的跟着先住的定形；两处 TODO 自认技术债，锁死换来零拷贝共享',
        10.5, lc.C_MUTE, 'start', maxw=1240, tag='subtitle')
_ch = '放大自 L2 south『why · 五本账一个池』· L0 显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 面板 A：KV 块合租 =================
PA_W = 664
PY, PH = 100, 380
lc.rect(MX, PY, PA_W, PH, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(MX + 18, PY + 26, 'A · KV 块合租：C4A 压缩块定 64 行', 12.5, lc.C_TXT, 'start', True,
        maxw=PA_W - 36, tag='pa:t')

# 物理张量（8×8 网格代表 64 行）
TG_X, TG_Y, TG_W, TG_H = MX + 232, PY + 62, 200, 150
lc.rect(TG_X, TG_Y, TG_W, TG_H, C_TEN_F, C_TEN_S, rx=6, sw=2.0)
cell_n = 8
cw_, chh_ = TG_W / cell_n, TG_H / cell_n
for r in range(cell_n):
    for c in range(cell_n):
        lc.rect(TG_X + 6 + c * (cw_ - 12 / cell_n) + 1.5, TG_Y + 6 + r * (chh_ - 12 / cell_n) + 1.5,
                cw_ - 12 / cell_n - 3, chh_ - 12 / cell_n - 3, '#ffffff', C_TEN_S, rx=1.5, sw=0.7)
lc.text(TG_X + TG_W / 2, PY + 52, '同一张物理张量', 10, C_TEN_S, 'middle', True, maxw=TG_W, tag='tg:t')
lc.text(TG_X + TG_W / 2, TG_Y + TG_H + 18, 'C4A 压缩 KV 块 [256//4, 584]', 9.6, lc.C_TXT, 'middle', True,
        maxw=TG_W + 40, tag='tg:s1')
lc.text(TG_X + TG_W / 2, TG_Y + TG_H + 34, '= [64 行, 584B/行] → 64 槽', 9.2, '#475569', 'middle',
        maxw=TG_W + 40, tag='tg:s2')

# 两个住户
res_l = (MX + 18, 'C4A 压缩主账', '256÷4 = 64 槽 × 584B', C_RES_S)
res_r_x = MX + PA_W - 18 - 180
res_r = (res_r_x, 'SWA 窗口账', 'block 只能 64（跟着定形）', C_RES_S)
for rx_, name, sub, col in (res_l, res_r):
    lc.rect(rx_, PY + 120, 180, 64, C_RES_F, col, rx=8, sw=1.5)
    lc.text(rx_ + 90, PY + 144, name, 10, col, 'middle', True, maxw=170, tag='rs:' + name)
    lc.text(rx_ + 90, PY + 164, sub, 8.6, '#475569', 'middle', maxw=170, tag='rs:s' + name[:4])
lc.seg(MX + 18 + 180, PY + 152, TG_X - 2, PY + 152, C_RES_S, 1.8, 'std')
lc.seg(res_r_x, PY + 152, TG_X + TG_W + 2, PY + 152, C_RES_S, 1.8, 'std')
lc.text((MX + 198 + TG_X) / 2, PY + 144, '共住', 8.6, C_RES_S, 'middle', True, maxw=40, tag='sh1')
lc.text((res_r_x + TG_X + TG_W) / 2, PY + 144, '共住', 8.6, C_RES_S, 'middle', True, maxw=40, tag='sh2')

# 注释原话
lc.rect(MX + 18, PY + 262, PA_W - 36, 76, '#fff7ed', '#fdba74', rx=8, sw=1.2)
lc.text(MX + 32, PY + 282, '注释原话：Block size is constrained by tensor sharing', 9.4, C_QUOTE,
        'start', True, maxw=PA_W - 60, tag='q1')
lc.text(MX + 32, PY + 300, '· SWA 与 C4A 的 KV 块共享物理张量 → 必须同一页形状 → SWA 只能 64 token/块',
        8.8, '#7c2d12', 'start', maxw=PA_W - 60, tag='q2')
lc.text(MX + 32, PY + 318, '· vllm/v1/attention/backends/mla/sparse_swa.py:L77-L82 · TODO(yifan) L81：make block size automatically determined',
        8.2, '#9a3412', 'start', maxw=PA_W - 60, tag='q3')
lc.text(MX + 18, PY + 356, '对齐取 576（fp8_ds_mla）/ 512——窗口账与压缩主账页同为 37,440B，不是巧合',
        8.8, lc.C_MUTE, 'start', maxw=PA_W - 36, tag='pa:n')

# ================= 面板 B：状态块定形 =================
PB_X = MX + PA_W + 22
PB_W = BXR - PB_X
lc.rect(PB_X, PY, PB_W, PH, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(PB_X + 18, PY + 26, 'B · 状态块定形：压缩器状态与 KV 块共住', 12.5, lc.C_TXT, 'start', True,
        maxw=PB_W - 36, tag='pb:t')

SB = [
    ('C4 状态块', '[4, 2*512*2*4]', '→ block_size = 4', '（window 8 = 2×4）'),
    ('C128 状态块', '[8, 512*2*4]', '→ block_size = 8', '（window 128）'),
]
sy = PY + 56
for name, shape, blk, win in SB:
    lc.rect(PB_X + 18, sy, PB_W - 36, 58, C_TEN_F, C_TEN_S, rx=8, sw=1.5)
    lc.text(PB_X + 32, sy + 22, name, 10, C_TEN_S, 'start', True, maxw=120, tag='sb:' + name[:4])
    lc.text(PB_X + 150, sy + 22, shape, 9.6, lc.C_TXT, 'start', True, maxw=150, tag='sb:h' + name[:4])
    lc.text(PB_X + 305, sy + 22, blk, 9.6, C_TEN_S, 'start', True, maxw=140, tag='sb:b' + name[:4])
    lc.text(PB_X + 32, sy + 44, win + ' · 状态页补齐后 32,832B', 8.8, '#475569', 'start', maxw=PB_W - 60,
            tag='sb:w' + name[:4])
    sy += 70

lc.rect(PB_X + 18, PY + 200, PB_W - 36, 64, '#fff7ed', '#fdba74', rx=8, sw=1.2)
lc.text(PB_X + 32, PY + 220, '同款注释：compressor states share the same physical tensor', 9.4, C_QUOTE,
        'start', True, maxw=PB_W - 60, tag='q4')
lc.text(PB_X + 32, PY + 238, '· 两档状态页补齐后恰好等大：32,832B（32,768 + 64）——能共享物理张量的前提',
        8.8, '#7c2d12', 'start', maxw=PB_W - 60, tag='q5')
lc.text(PB_X + 32, PY + 254, '· vllm/models/deepseek_v4/compressor.py:L177-L189 · TODO(yifan) L183（同款债）',
        8.2, '#9a3412', 'start', maxw=PB_W - 60, tag='q6')
lc.text(PB_X + 18, PY + 290, '· indexer 的 twin 压缩器（head 128 → state 512）页 8,640B，同桶合租',
        8.8, '#475569', 'start', maxw=PB_W - 36, tag='pb:n1')
lc.text(PB_X + 18, PY + 308, '· C4 状态账 dtype 恒 fp32——五本账里唯一不量化的账（量化它伤收敛）',
        8.8, '#475569', 'start', maxw=PB_W - 36, tag='pb:n2')

# ================= 底部结论带 =================
BB_Y = PY + PH + 20
lc.rect(MX, BB_Y, BXR - MX, 84, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.6)
lc.text(MX + 18, BB_Y + 24, '锁死换来什么：零拷贝合租', 12.5, lc.C_GPU_S, 'start', True, maxw=500, tag='bb:t')
lc.text(MX + 18, BB_Y + 46, '· 窗口账与压缩账零拷贝共住一张张量、两档状态页恰好等大合租——块大小一旦写死，五本账的页形状全部联动',
        9.4, '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l1')
lc.text(MX + 18, BB_Y + 66, '· 块大小不是 config 旋钮，是张量共享的下游结果——想改就得搬家（页形状对不上就没法共享）',
        9.4, '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l2')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 44, 'vllm/v1/attention/backends/mla/sparse_swa.py:L77-L82（注释原话 · block 64 · TODO）· vllm/models/deepseek_v4/compressor.py:L177-L189（状态块形状 · 同款注释 · TODO）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 29, '两档状态页等大（32,832B）· twin 状态页 8,640B ＝ 本章账本算术脚本实算（pin spec 类）· fp32 恒定出处 compressor.py:L173（assert）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 14, '行号基线 vLLM v0.27.1（6e448d0ea）', 8.5, lc.C_FAINT, 'start', maxw=400, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m9_locked_blocks.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
