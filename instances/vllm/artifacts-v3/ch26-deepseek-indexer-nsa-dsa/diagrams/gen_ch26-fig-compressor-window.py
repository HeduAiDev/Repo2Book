#!/usr/bin/env python3
"""ch26 机制图 · V4 压缩窗状态表(figure ch26-fig-compressor-window,模板 state-table)

放大自 L0『模型层 indexer 框』——L2 下排『V4 压缩索引 K^IComp』(站 13 的 compressor 臂)
展开:8 行窗位状态表(头切换/score/softmax 权重比例条)+ 输出链四步(Σ→RMSNorm→RoPE→
FP8)+ 右缘 token→压缩块 坐标换算轴。

claim:DeepseekCompressor 在 (pos+1)%4==0 的边界 token 上把 8-token 窗(前半读状态头0、
后半读头1——overlap)softmax 门控压成 1 个块:Σ kv·score → RMSNorm → GPT-J RoPE
(位置=压缩位 (pos//4)*4)→ FP8 132B 一步融合写进压缩坐标系的 IndexCache——indexer
活在 //4 坐标系里(max_model_len//4、seq_lens//4、slot=(pos+1)%4==0)。

数字全部取自 figure spec 的 numbers(权重 8 项/和 1.0000 / 头切换 t<4 头0、t≥4 头1 /
rrms 19.878185·RoPE 压缩位 4·scale 0.03125·slot 7 / seq_lens [16,8]→[4,2]·slot 15→3、
7→17·top-k [0..3] / skip_k_cache_insert=True)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 740
MX = 52
BXR = 1448

# ---- 实测数据(traces m11:8 窗位) ----
WIN = [
    (0, 0, 0, 'a0(one-hot 维 0)', 1, 0.0259),
    (1, 1, 0, 'a1(one-hot 维 1)', 2, 0.0704),
    (2, 2, 0, 'a2(one-hot 维 2)', 3, 0.1913),
    (3, 3, 0, 'a3(one-hot 维 3)', 4, 0.5200),
    (4, 4, 1, 'b4(one-hot 维 12)', 1, 0.0259),
    (5, 5, 1, 'b5(one-hot 维 13)', 1, 0.0259),
    (6, 6, 1, 'b6(one-hot 维 14)', 2, 0.0704),
    (7, 7, 1, 'b7(one-hot 维 15)', 2, 0.0704),
]
WMAX = 0.5200

# ---------------- 标题区 ----------------
lc.text(MX, 34, '压缩窗状态表:8-token 窗 softmax 门控压成 1 块——indexer 活在 //4 坐标系里',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '只有边界 token (pos+1)%4==0 触发一次压缩;窗比 ratio 宽一倍(overlap):前半读状态头0、后半读头1;权重和恒 1——压缩向量是窗内 kv 的凸组合',
        10.5, lc.C_MUTE, 'start', maxw=1130, tag='subtitle')
_ch = '放大自 L0『模型层 indexer 框』· L2 下排 V4 压缩索引 K^IComp(站 13)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主表:8 行 × 5 列 ----------------
TX, TY, RH = MX, 128, 29
COLS = [('窗位 t', 52, 58), ('pos', 112, 52), ('读哪头(状态切片)', 168, 232),
        ('门控 score', 404, 96), ('softmax 权重(比例条)', 504, 396)]
lc.rect(TX, TY - 24, 852, 24 + 8 * RH + 12, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
for name, cx, cwd in COLS:
    lc.text(cx + cwd / 2 if name != '窗位 t' else cx + 20, TY - 8, name, 9, lc.C_TXT,
            'middle', True, maxw=cwd, tag='th:' + name[:6])
for i, (t, pos, head, vec, score, wgt) in enumerate(WIN):
    y = TY + i * RH
    hot = t == 3
    if hot:
        lc.rect(TX + 6, y, 840, RH, '#fff7ed', 'none', rx=4, sw=0)
    lc.text(cx_t := 72, y + 19, 't=%d' % t, 9, lc.C_TXT, 'middle',
            bold=hot, tag='r%d:t' % i)
    lc.text(138, y + 19, str(pos), 9, lc.C_TXT, 'middle', tag='r%d:p' % i)
    lc.text(168, y + 19, ('头%d · ' % head) + vec, 9,
            '#0e7490' if head == 0 else '#155e75', 'start', maxw=228, tag='r%d:v' % i)
    lc.text(452, y + 19, str(score), 9, lc.C_TXT, 'middle', tag='r%d:s' % i)
    bw = 286 * wgt / WMAX
    lc.rect(504, y + 7, bw, 15, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.0)
    lc.text(504 + bw + 8, y + 19, '%.4f' % wgt, 8.5,
            lc.C_ENG_S if hot else '#334155', 'start', bold=hot, tag='r%d:w' % i)
# 头切换分隔线(前半/后半)
lc.seg(TX + 6, TY + 4 * RH, TX + 846, TY + 4 * RH, lc.C_ENG_S, 1.2, dash=True)
lc.text(TX + 6, TY + 8 * RH + 10, 'score [1,2,3,4,1,1,2,2] 的 softmax,权重和 1.0000;score=4 的 token 独占 52%(高亮行 t=3)',
        8.5, lc.C_MUTE, 'start', maxw=840, tag='tbl:f')
lc.text(TX + 6, TY + 8 * RH + 28, '虚线 = 头切换:前半窗(t0-t3)读状态头 0(pos 0-3)、后半窗(t4-t7)读头 1(pos 4-7)——overlap 使窗(8)比 ratio(4)宽一倍',
        8.5, lc.C_MUTE, 'start', maxw=840, tag='tbl:f2')

# ---------------- 输出链:四步 + 132B 条 ----------------
CY_ = 420
chain = [
    ('① Σ kv · 门控', ['compressed[0:8] =', '[0.0259, 0.0704,', '0.1913, 0.5200, …]']),
    ('② RMSNorm', ['rrms = 19.878185', '(eps=1e-6, weight=1)', 'normed 首维 0.514586']),
    ('③ GPT-J RoPE', ['只打末 64 维(rope_dim)', '位置 = 压缩位 4', '= (7//4)*4']),
    ('④ FP8 写缓存', ['scale = 0.03125(2 的幂)', '值首维 16.000000', '写 slot 7(132B/条)']),
]
cbw, cgap = 196, 22
for k, (t, lines) in enumerate(chain):
    x = MX + k * (cbw + cgap)
    lc.rect(x, CY_, cbw, 96, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
    lc.text(x + cbw / 2, CY_ + 20, t, 10, lc.C_GPU_S, 'middle', True, maxw=cbw - 12,
            tag='ch:%d' % k)
    for j, s in enumerate(lines):
        lc.text(x + cbw / 2, CY_ + 38 + j * 15, s, 8, '#334155', 'middle', maxw=cbw - 10,
                tag='ch:%d:%d' % (k, j))
    if k < 3:
        lc.seg(x + cbw, CY_ + 48, x + cbw + cgap, CY_ + 48, lc.C_GPU_S, 1.8, 'std')
# ④ → 132B 字节条
lc.seg(MX + 3 * (cbw + cgap) + cbw / 2, CY_ + 96, MX + 3 * (cbw + cgap) + cbw / 2, CY_ + 122,
       lc.C_KV_S, 1.6, 'std')
BBX = MX + 3 * (cbw + cgap) + cbw / 2 - 130
lc.rect(BBX, CY_ + 126, 252, 20, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
lc.rect(BBX + 244, CY_ + 126, 8, 20, '#ffedd5', lc.C_ENG_S, rx=2, sw=1.0)
lc.text(BBX + 122, CY_ + 140, 'IndexCache 条目 132B = 128B fp8 值 + 4B scale', 8, '#155e75',
        'middle', True, maxw=240, tag='bb:t')
lc.text(MX, CY_ - 12, '输出链(一步融合:quantize 与缓存插入不再分家):', 9.5, lc.C_TXT,
        'start', True, maxw=400, tag='chain:t')

# ---------------- 左下:skip_k_cache_insert 注 ----------------
NY_ = 580
lc.rect(MX, NY_, 852, 76, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 14, NY_ + 20, 'skip_k_cache_insert=True:K^IComp 的插入归 compressor 管', 10,
        lc.C_TXT, 'start', True, maxw=600, tag='sk:t')
for j, s in enumerate([
    '· indexer_op 不插 K——打分与建缓存分工(attention.py:L820)',
    '· 省打分不省建缓存:短上下文全选快路径下,缓存仍照建(正文『全选即最优』节)',
]):
    lc.text(MX + 14, NY_ + 38 + j * 16, s, 8.5, '#334155', 'start', maxw=820,
            tag='sk:l%d' % j)

# ---------------- 右栏:坐标换算轴 ----------------
RX, RW_ = 940, BXR - 940
lc.rect(RX, 104, RW_, 436, '#ffffff', lc.C_KV_S, rx=9, sw=1.5)
lc.text(RX + 14, 104 + 22, '坐标换算:indexer 活在 //4 坐标系', 11, lc.C_KV_S, 'start', True,
        maxw=RW_ - 28, tag='cc:t')
# token 轴:16 格分 4 组
ax, ay, acw, ach = RX + 22, 104 + 44, 22, 24
SHADES = ['#ecfeff', '#cffafe', '#a5f3fc', '#67e8f9']
for p in range(16):
    blk = p // 4
    lc.rect(ax + p * acw, ay, acw - 2, ach, SHADES[blk], lc.C_KV_S, rx=2, sw=0.9)
    lc.text(ax + p * acw + (acw - 2) / 2, ay + 16, str(p), 7, '#155e75', 'middle', tag='tk:%d' % p)
for blk in range(4):
    bx = ax + blk * 4 * acw
    lc.text(bx + 2 * acw, ay + ach + 16, '块 %d' % blk, 8.5, '#155e75', 'middle', True,
            tag='bk:%d' % blk)
    lc.seg(bx, ay + ach + 22, bx + 4 * acw - 2, ay + ach + 22, lc.C_KV_S, 1.2)
    lc.seg(bx + 2 * acw, ay + ach + 22, bx + 2 * acw, ay + ach + 34, lc.C_KV_S, 1.2)
# 压缩块行
cy2 = ay + ach + 44
for blk in range(4):
    bx = ax + blk * 4 * acw
    lc.rect(bx, cy2, 4 * acw - 2, 24, SHADES[blk], lc.C_KV_S, rx=3, sw=1.2)
    lc.text(bx + (4 * acw - 2) / 2, cy2 + 16, '压缩块 %d' % blk, 8, '#0c4a6e', 'middle', True,
            tag='cb:%d' % blk)
lc.text(ax + 16 * acw + 14, ay + 12, 'token 位 0-15', 8.5, lc.C_MUTE, 'start', tag='ax:l1')
lc.text(ax + 16 * acw + 14, cy2 + 16, 'top-k 在 [0..3] 选', 8.5, lc.C_MUTE, 'start',
        tag='ax:l2')
# 换算事实
for j, s in enumerate([
    '· seq_lens // compress_ratio:[16, 8] → [4, 2]',
    '· max_model_len //4:实尺 163840 → 40960 个候选块',
    '· 压缩 slot 映射例:pos 15→3、pos 7→17',
    '   (storage_block=16 = block//4 分页;',
    '   仅 (pos+1)%4==0 的边界 token 落 slot)',
    '· 每块恰写一次:整数除法 pos//4 划分互不相交,',
    '   边界条件在每块内恰对最大 pos 成立一次',
]):
    lc.text(RX + 14, cy2 + 52 + j * 17, s, 8.5, '#334155', 'start', maxw=RW_ - 26,
            tag='cc:l%d' % j)

# ---------------- 页脚 ----------------
lc.text(MX, 700, '图例:青系 = KV/压缩缓存角色(深浅=块归属) · 橙高亮行 = 最大门控权重(score=4 独占 52%) · 橙段 = fp32 scale 字节',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 720, '窗值/rrms/RoPE/scale/slot = host 实测(one-hot 例:头0 列 a_p=维 p,头1 列 b_p=维 p+8) · 行号 = vLLM v0.27.1(deepseek_v4/compressor.py · attention.py)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch26-fig-compressor-window.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
