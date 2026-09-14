#!/usr/bin/env python3
"""ch26 机制图 · V4 三类层 compress_ratios 逐层表(figure ch26-fig-v4-layer-types,模板 tiling)

放大自 L0『模型层 indexer 框』——L2 下排『V4 三类层 compress_ratios 表』(站 13 装配面)
的展开:一条 61 格层带被 compress_ratios 表(checkpoint 发布)染成三色,只有 C4A 建
indexer;MTP 层钉在带尾恒为 1;三类层各有独立 planner。

claim:compress_ratios 逐层表把 V4 的每一层定型为三类之一:SWAonly(1) 只滑窗、
C4A(4) 带 indexer 做 4 压 1 的块级 top-k、C128A(128) 压缩后 1280 个候选在 metadata 期
无条件全选(不比较 topk)——只有 C4A 建 indexer,MTP 层恒回 1。

数字全部取自 figure spec 的 numbers(ratios [1,4,128]→indexer [False,True,False] /
『Only C4A uses sparse attention and hence has indexer』L277-L278 / 163840//128=1280 候选
metadata 期无条件全选、这条路径不比较 topk / 163840//4=40960 候选块远超官方 top-k
512/1024(论文 §4.2.1)→真打分 / C4A 全选边界=上下文 4×topk(官方两档 2048/4096) /
MTP layer_id 5≥num_hidden_layers 3→1 /
SWA 块 64 与 C4A 块 [256//4] 同页宽、FlashMLASchedMeta 不共享)。着色=示意,
逐层 Pattern 随 checkpoint 发布。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 660
MX = 52
BXR = 1448

SWA_F, SWA_S = '#ecfeff', '#67e8f9'      # swaonly:最浅青
C4A_F, C4A_S = '#a5f3fc', '#0891b2'      # c4a:中青
C128_F, C128_S = '#67e8f9', '#155e75'    # c128a:深青

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'V4 的层是一条逐层定型的带子:compress_ratios 表染色,只有 C4A 建 indexer',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, 'SWAonly(1) 只滑窗 · C4A(4) 带 indexer、4 token 压 1 块后块级 top-k · C128A(128) 压缩后 1280 候选 metadata 期无条件全选（不比较 topk）——MTP 层钉在带尾恒为 1',
        10.5, lc.C_MUTE, 'start', maxw=1130, tag='subtitle')
_ch = '放大自 L0『模型层 indexer 框』· L2 下排 V4 三类层(站 13 装配面)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 层带:61 格(示意着色)+ MTP 尾格 ----------------
SY_, SH_, CW_ = 128, 52, 18.4
N = 61
X0 = 60
# 示意着色(逐层 Pattern 随 checkpoint 发布,不杜撰具体排布)
pattern = (['swa'] * 2 + ['c4a'] * 4 + ['swa'] * 2 + ['c4a'] * 6 + ['c128'] * 6
           + ['c4a'] * 8 + ['swa'] * 4 + ['c4a'] * 8 + ['c128'] * 6 + ['c4a'] * 8
           + ['swa'] * 7)
pattern = pattern[:N]
while len(pattern) < N:
    pattern.append('swa')
STYLE = {'swa': (SWA_F, SWA_S), 'c4a': (C4A_F, C4A_S), 'c128': (C128_F, C128_S)}
prev = None
for i, kind in enumerate(pattern):
    x = X0 + i * CW_
    f, s = STYLE[kind]
    lc.rect(x, SY_, CW_ - 1.6, SH_, f, s, rx=2, sw=1.0)
    if kind == 'c4a':
        lc.text(x + (CW_ - 1.6) / 2, SY_ + 31, '4→1', 6, '#0e7490', 'middle', True,
                tag='c4a:i%d' % i)
    # 类型分界 tick
    if prev is not None and kind != prev:
        lc.seg(x - 0.8, SY_ + SH_, x - 0.8, SY_ + SH_ + 7, lc.C_MUTE, 1.0)
    prev = kind
lc.text(X0 - 8, SY_ + SH_ / 2 + 3, '层 0', 8, lc.C_MUTE, 'end', tag='lbd:0')
lc.text(X0 + N * CW_ + 4, SY_ + SH_ / 2 + 3, '层 60', 8, lc.C_MUTE, 'start', tag='lbd:60')
# MTP 尾格(隔开)
MGX = X0 + N * CW_ + 40
lc.rect(MGX, SY_, 74, SH_, SWA_F, lc.C_ENG_S, rx=4, sw=1.8, dash=True)
lc.text(MGX + 37, SY_ + 22, 'MTP 层', 8.5, lc.C_ENG_S, 'middle', True, tag='mtp:t')
lc.text(MGX + 37, SY_ + 40, '恒 compress_ratio=1', 7, lc.C_ENG_S, 'middle', tag='mtp:s')
lc.seg(X0 + N * CW_ + 4, SY_ + SH_ / 2, MGX, SY_ + SH_ / 2, lc.C_FAINT, 1.0, dash=True)

lc.text(X0, SY_ + SH_ + 24, 'compress_ratios 逐层表随 checkpoint 发布(vLLM 不设默认值)——本带 61 格、着色与分界均为示意,不指任何具体模型',
        8.5, lc.C_MUTE, 'start', maxw=1100, tag='strip:note')
# 图例(带右上)
lx = MGX + 90
for lab, f, s in [('SWAonly', SWA_F, SWA_S), ('C4A', C4A_F, C4A_S), ('C128A', C128_F, C128_S)]:
    lc.rect(lx, SY_ - 26, 14, 14, f, s, rx=3, sw=1.2)
    lc.text(lx + 19, SY_ - 15, lab, 8.5, lc.C_TXT, 'start', tag='lg:' + lab)
    lx += 19 + lc.tw(lab, 8.5) + 16

# ---------------- 三类层卡 ----------------
CY_ = 240
cw3, gap3 = 440, 22
cards = [
    (MX, 'SWAonly · compress_ratio=1', SWA_S, [
        '只滑窗(SWA 滑窗缓存),无 indexer',
        'ratios=1 → indexer 建 = False',
        '类型常量 sparse_swa.py:L38-L53',
    ]),
    (MX + cw3 + gap3, 'C4A · compress_ratio=4', C4A_S, [
        '带 indexer:4 token 压 1 块后块级 top-k',
        '163840//4 = 40960 候选块远超 top-k（官方 512/1024）→ 真打分',
        '上下文 ≤ 4×topk 时整批判全选（官方两档 2048/4096）',
        'skip_k_cache_insert=True:K 插入归 compressor',
    ]),
    (MX + 2 * (cw3 + gap3), 'C128A · compress_ratio=128', C128_S, [
        '无 indexer:压缩后 163840//128 = 1280',
        '1280 候选无条件全选（不比较 topk）',
        'metadata 期直算全选表(_build_c128a_metadata)',
        'ratios=128 → indexer 建 = False',
    ]),
]
for x, t, stk, lines in cards:
    lc.rect(x, CY_, cw3, 118, '#ffffff', stk, rx=9, sw=1.6)
    lc.text(x + 14, CY_ + 20, t, 10.5, stk, 'start', True, maxw=cw3 - 28, tag='c:' + t[:8])
    for j, s in enumerate(lines):
        lc.text(x + 14, CY_ + 40 + j * 17, '· ' + s, 8.5, '#334155', 'start', maxw=cw3 - 26,
                tag='c:l%s%d' % (t[:4], j))
# 实例化证据条(三卡上方连接到带)
lc.text(MX, CY_ - 12, '三实例实测:ratios [1, 4, 128] → indexer 建 [False, True, False];类型映射 1→swaonly · 4→c4a · 128→c128a',
        9, lc.C_TXT, 'start', True, maxw=1000, tag='ev:t')
lc.text(BXR, CY_ - 12, '『Only C4A uses sparse attention and hence has indexer.』(attention.py:L277-L278 原话)',
        8.5, lc.C_GPU_S, 'end', maxw=560, tag='ev:q')

# ---------------- 底部两注:MTP 护栏 + 共享页/独立 planner ----------------
NY_ = 396
lc.rect(MX, NY_, 660, 118, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 14, NY_ + 20, 'MTP 护栏:不在表内的层恒回 1', 10.5, lc.C_TXT, 'start', True,
        maxw=400, tag='n1:t')
for j, s in enumerate([
    'layer_id ≥ num_hidden_layers(实例 5 ≥ 3)→ compress_ratio=1',
    '逐字 L207-L213:『MTP layer is not included in the',
    'compress ratio list』——MTP 层钉死滑窗,不进压缩坐标系',
]):
    lc.text(MX + 14, NY_ + 40 + j * 17, '· ' + s, 8.5, '#334155', 'start', maxw=630,
            tag='n1:l%d' % j)
lc.text(MX + 14, NY_ + 106, 'vllm/models/deepseek_v4/attention.py:L207-L213', 8, lc.C_FAINT,
        'start', tag='n1:f')

lc.rect(BXR - 660, NY_, 660, 118, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(BXR - 660 + 14, NY_ + 20, '同页不同账:共享张量页,不共享 planner', 10.5, lc.C_TXT,
        'start', True, maxw=420, tag='n2:t')
for j, s in enumerate([
    'SWA 滑窗块宽 64 与 C4A 压缩 KV 块 [256//4, head_dim] 同页宽',
    '——两类层共享物理张量页(sparse_swa.py:L81-L83)',
    '三类层各有独立的 FlashMLASchedMeta,planner 不共享',
    '(flashmla.py:L210-L226)',
]):
    lc.text(BXR - 660 + 14, NY_ + 40 + j * 17, '· ' + s, 8.5, '#334155', 'start', maxw=630,
            tag='n2:l%d' % j)

# ---------------- 页脚 ----------------
lc.text(MX, 560, '图例:三色青系 = 三类层(KV 青族深浅,同源配色) · 格内 4→1 = 压缩小图标 · 竖 tick = 类型分界(示意)',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 580, '分类与实例化 = 真实装配路径 host 实测 · 引语与行号 = vLLM v0.27.1(deepseek_v4/attention.py · sparse_swa.py · flashmla.py)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 600, '消费侧:C4A 的块级 top-k 如何与 SWA 滑窗在同一核里合算——见『一核双源』图(站 14)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch26-fig-v4-layer-types.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
