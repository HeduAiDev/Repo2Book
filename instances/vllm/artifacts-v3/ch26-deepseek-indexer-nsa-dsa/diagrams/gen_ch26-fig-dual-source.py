#!/usr/bin/env python3
"""ch26 机制图 · V4 消费一核双源(figure ch26-fig-dual-source,模板 flow)

放大自 L0『模型层 indexer 框』——L2 拍片⑥ 消费·稀疏 MLA 的 V4 臂(站 14)展开:
中央 flash_mla_with_kvcache 核,左上 SWA 滑窗缓存源(支路③)、左下压缩 KV 池源
(支路①②,indexer 选),C4A 换算小框,右侧 union 账 + attn_out,右下 NSA 谱系卡,
底部 prefill 并集与 C128A 直选注。

claim:V4 decode 消费是一次 flash_mla_with_kvcache 调用吃两本 KV:k_cache=SWA 滑窗缓存
(indices=滑窗位)+ extra_k_cache=压缩 KV 池(extra_indices_in_kvcache=indexer 选的
top-k 位)——输出严格等于 union 上的 softmax(trace max diff 0.000000),NSA 的
滑窗/压缩/选择三支路在这一次调用里结构性回归。

数字全部取自 figure spec 的 numbers(pos15·滑窗8·SWA位[8..15]·union 10 vs 16·
max diff 0.000000 / [3,1,-1]→[323,321]·块5:5×64+3/5×64+1 / token0 len=2+4=6·
token2 len=1+4=5 / C128A 1280 候选无条件全选、不比较 topk)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 810
MX = 52
BXR = 1448
SWA_F, SWA_S = '#ecfeff', '#67e8f9'      # SWA 滑窗缓存:浅青
CMP_F, CMP_S = '#a5f3fc', '#155e75'      # 压缩 KV 池:深青

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一核双源:一次 flash_mla_with_kvcache 同时吃 SWA 滑窗与压缩 top-k',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '两本 KV 在同一次 softmax 里合算——输出严格等于 union 上的 softmax;NSA 的滑窗/压缩/选择三支路在这一次调用里结构性回归',
        10.5, lc.C_MUTE, 'start', maxw=1130, tag='subtitle')
_ch = '放大自 L0『模型层 indexer 框』· L2 拍片⑥ 消费·稀疏 MLA 的 V4 臂(站 14)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 中央核 ----------------
CORE = (560, 280, 390, 150)
lc.rect(*CORE, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.4)
lc.text(CORE[0] + CORE[2] / 2, CORE[1] + 26, 'flash_mla_with_kvcache', 13.5, '#166534',
        'middle', True, tag='core:t')
lc.text(CORE[0] + CORE[2] / 2, CORE[1] + 46, '一次调用吃两本 KV(decode 消费面)', 9.5,
        lc.C_TXT, 'middle', tag='core:s')
for j, s in enumerate([
    'k_cache = SWA 滑窗缓存 · extra_k_cache = 压缩 KV 池',
    '两路在同一次 softmax 里合算——结构性合算,非两次算完拼接',
]):
    lc.text(CORE[0] + CORE[2] / 2, CORE[1] + 70 + j * 17, s, 8.5, '#334155', 'middle',
            maxw=CORE[2] - 24, tag='core:l%d' % j)
lc.text(CORE[0] + CORE[2] / 2, CORE[1] + CORE[3] - 12,
        'vllm/models/deepseek_v4/nvidia/flashmla.py:L228-L244', 8, lc.C_FAINT, 'middle',
        tag='core:f')

# ---------------- 左列:两个源 + C4A 换算 ----------------
SWA = (60, 150, 360, 118)
lc.rect(*SWA, SWA_F, SWA_S, rx=9, sw=1.8)
lc.text(SWA[0] + 14, SWA[1] + 20, '支路③ 滑窗:k_cache = SWA 滑窗缓存', 10.5, '#0c4a6e',
        'start', True, maxw=330, tag='swa:t')
for j, s in enumerate([
    'indices = 滑窗位 [8..15] 共 8 条(窗 8)',
    '最近 8 个 token 无条件全看——例 pos 15',
]):
    lc.text(SWA[0] + 14, SWA[1] + 42 + j * 17, '· ' + s, 8.5, '#155e75', 'start', maxw=330,
            tag='swa:l%d' % j)
lc.text(SWA[0] + SWA[2] - 12, SWA[1] + SWA[3] - 10, 'sparse_swa.py 滑窗缓存', 8, lc.C_MUTE,
        'end', tag='swa:f')

CMP = (60, 330, 360, 130)
lc.rect(*CMP, CMP_F, CMP_S, rx=9, sw=1.8)
lc.text(CMP[0] + 14, CMP[1] + 20, '支路①+②:extra_k_cache = 压缩 KV 池', 10.5, '#0c4a6e',
        'start', True, maxw=330, tag='cmp:t')
for j, s in enumerate([
    'extra_indices_in_kvcache = indexer 选的 top-k',
    'buffer 行 [3, 1, -1](请求内压缩坐标,-1 无效)',
    '① 压缩 = 池本身 · ② 选择 = indexer 挑位',
]):
    lc.text(CMP[0] + 14, CMP[1] + 42 + j * 17, '· ' + s, 8.5, '#155e75', 'start', maxw=330,
            tag='cmp:l%d' % j)

CVT = (60, 480, 360, 96)
lc.rect(*CVT, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(CVT[0] + 14, CVT[1] + 18, 'C4A 换算:compute_global_topk_indices_and_lens', 8.5,
        lc.C_TXT, 'start', True, maxw=340, tag='cvt:t')
lc.text(CVT[0] + 14, CVT[1] + 40, '[3, 1, -1] → 物理 slot [323, 321]', 10, lc.C_TXT,
        'start', True, maxw=340, tag='cvt:f')
lc.text(CVT[0] + 14, CVT[1] + 60, '块表块 5:5×64+3 / 5×64+1(局部压缩位→全局物理位)', 8.5,
        '#334155', 'start', maxw=340, tag='cvt:l')
lc.seg(240, CVT[1], 240, CMP[1] + CMP[3], lc.C_MUTE, 1.6, 'std')

# 源 → 核 箭头
lc.seg(SWA[0] + SWA[2], 209, CORE[0], 322, SWA_S, 2.2, 'std')
lc.text(492, 242, '支路③', 9, '#0e7490', 'middle', True, tag='a:3')
lc.seg(CMP[0] + CMP[2], 395, CORE[0], 372, CMP_S, 2.2, 'std')
lc.text(492, 352, '支路①②', 9, '#0e7490', 'middle', True, tag='a:12')

# ---------------- 右列:union 账 → attn_out ----------------
UNI = (990, 280, 200, 150)
lc.rect(*UNI, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(UNI[0] + 100, UNI[1] + 20, 'union = 8 + 2 = 10 条', 10, lc.C_TXT, 'middle', True,
        tag='uni:t')
for k in range(10):
    x = UNI[0] + 14 + k * 17
    lc.rect(x, UNI[1] + 34, 15, 15, SWA_F if k < 8 else CMP_F, SWA_S if k < 8 else CMP_S,
            rx=2, sw=1.0)
lc.text(UNI[0] + 100, UNI[1] + 66, 'vs 全上下文 16 条', 9, lc.C_MUTE, 'middle', tag='uni:v')
lc.text(UNI[0] + 100, UNI[1] + 88, '滑窗 8 + top-k 有效 2', 8.5, '#334155', 'middle',
        tag='uni:l1')
lc.text(UNI[0] + 100, UNI[1] + 106, '(topk 行 [3,1,-1] → 有效 2)', 8, lc.C_MUTE, 'middle',
        tag='uni:l2')
lc.text(UNI[0] + 100, UNI[1] + 128, 'max diff 0.000000', 9, lc.C_ENG_S, 'middle', True,
        tag='uni:d')
lc.text(UNI[0] + 100, UNI[1] + 144, '(host 实测对账)', 7.5, lc.C_MUTE, 'middle', tag='uni:n')

OUT = (1230, 300, 218, 110)
lc.rect(*OUT, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.8)
lc.text(OUT[0] + 109, OUT[1] + 26, 'attn_out → o_proj', 11, '#166534', 'middle', True,
        tag='out:t')
lc.text(OUT[0] + 109, OUT[1] + 50, '输出 == union 上的 softmax', 9.5, lc.C_TXT, 'middle',
        True, tag='out:s')
for j, s in enumerate(['只对 union 10 条真算', '不读完全部 16 条历史']):
    lc.text(OUT[0] + 109, OUT[1] + 72 + j * 16, s, 8.5, '#334155', 'middle', tag='out:l%d' % j)
lc.seg(CORE[0] + CORE[2], 350, UNI[0], 350, lc.C_GPU_S, 2.2, 'std')
lc.seg(UNI[0] + UNI[2], 350, OUT[0], 350, lc.C_GPU_S, 2.2, 'std')

# ---------------- 右下:NSA 谱系卡 ----------------
NSA = (990, 460, 458, 175)
lc.rect(*NSA, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(NSA[0] + 14, NSA[1] + 20, 'NSA 三支路 → V4 回归(谱系闭环)', 10.5, lc.C_TXT,
        'start', True, maxw=400, tag='nsa:t')
for j, s in enumerate([
    'NSA(arXiv:2502.11089)三支路:压缩 / 选择 / 滑窗',
    '③ 滑窗 = k_cache + indices(SWA 位)',
    '① 压缩 = extra_k_cache(压缩 KV 池)',
    '② 选择 = extra_indices_in_kvcache(indexer 选)',
    'DSA(V3.2)砍成一支:token 级 top-k(本章主线)',
    'V4:三支在这一次调用里结构性回归',
]):
    lc.text(NSA[0] + 14, NSA[1] + 40 + j * 17, '· ' + s, 8.5, '#334155', 'start',
            maxw=NSA[2] - 28, tag='nsa:l%d' % j)
for j, s in enumerate(['vLLM v0.27.1 无 NSA 独立落', '地,只有这条谱系映射']):
    lc.text(NSA[0] + NSA[2] - 14, NSA[1] + 128 + j * 15, s, 8, lc.C_MUTE, 'end',
            tag='nsa:n%d' % j)

# ---------------- 底部:prefill 并集 + C128A 直选 ----------------
PY_ = 660
lc.rect(MX, PY_, 508, 100, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(MX + 14, PY_ + 20, 'prefill 侧:并集拼接', 10.5, lc.C_TXT, 'start', True, maxw=300,
        tag='pf:t')
for j, s in enumerate([
    'token0:len = 2 + 4 = 6(top-k 段 [3,-1] + SWA 段 4)',
    'token2:len = 1 + 4 = 5(top-k 段 1 条 + SWA 段 4)',
    'combine_topk_swa_indices 段拼接;gather 后与连续 KV',
    '并集成一次注意力(flashmla.py:L331-L363)',
]):
    lc.text(MX + 14, PY_ + 38 + j * 16, '· ' + s, 8.5, '#334155', 'start', maxw=478,
            tag='pf:l%d' % j)

lc.rect(580, PY_, 370, 100, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(594, PY_ + 20, 'C128A:无 indexer 的同款消费面', 10.5, lc.C_TXT, 'start', True,
        maxw=340, tag='c1:t')
for j, s in enumerate([
    '压缩后 1280 候选无条件全选（不比较 topk）',
    'metadata 期直算全选表(不作选择)',
    '同一个核吃直选表(flashmla.py:L185-L187)',
]):
    lc.text(594, PY_ + 38 + j * 16, '· ' + s, 8.5, '#334155', 'start', maxw=340,
            tag='c1:l%d' % j)

# ---------------- 页脚 ----------------
lc.text(MX, 776, '图例:浅青 = SWA 滑窗缓存 · 深青 = 压缩 KV 池 · 绿 = GPU 执行臂角色 · 虚线框 = 谱系/背景注',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 794, 'union 账与 max diff = host 实测(decode 例 pos 15、滑窗 8) · 函数名与行号 = vLLM v0.27.1(deepseek_v4/nvidia/flashmla.py)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch26-fig-dual-source.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
