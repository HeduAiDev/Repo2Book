#!/usr/bin/env python3
"""ch25 机制图 · 模型层自报 spec → KV 池组化(figure ch25-fig-spec-grouping,模板 flow)

放大自 L0『模型层 MLA 框』自报 spec 之后与 KV 池的组化面——L2 站 5-6
(get_kv_cache_spec → get_kv_cache_groups);ch14『一组一管理器共享一池』在 MLA 侧的兑现。
左→右:层堆(每层小卡自报 spec,DSV4 三种着色) → 收集漏斗 → 四级分流菱形链 → 组箱×N
共享底部一个 BlockPool 长条。

claim:模型层逐层自报什么 spec 直接决定这层落进哪个组、跟谁共享池:runner 遍历收 spec →
get_kv_cache_groups 四级分流(全同 spec→一组 / 全同 token 槽数→一组 / DSV4 grouped→多组 /
页归一→多组),DSV4 一个模型三类子缓存(压缩层/SWA/indexer cache)同池分组。

数字全部取自 figure spec 的 numbers(MLAAttentionSpec(num_kv_heads=1, head_size=576) ·
四级分流链 kv_cache_utils.py:L1781-L1852 · merge 断言四字段 kv_cache_interface.py:L429-L468 ·
DSV4 三类子缓存 m12 实测 · 页协商 SWA block 64 = 256//4)。坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 730
MX = 52
BXR = 1448
C_K = lc.C_KV_S              # KV/组化 = 青
C_K_T = '#155e75'
C_G = lc.C_GPU_S             # 模型层自报 = 绿
C_G_T = '#166534'
C_SWA = lc.C_API_S           # SWA 子缓存 = 蓝
C_SWA_T = '#1e40af'
C_IX = lc.C_SAM_S            # indexer 子缓存 = 品红

# ---------------- 标题区 ----------------
lc.text(MX, 34, '模型层说什么,KV 池就长什么样:逐层自报 spec,四级分流定分组',
        16, lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '每组一个管理器共享一个 BlockPool——ch14 的账在 MLA 侧兑现;自报错一个字段,这层就进错组',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 5-6 的机制展开(ch14 回指)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左:层堆(DSV3 全同 + DSV4 三色) ----------------
lc.text(70, 96, '① 每层自报(DSV3/DSV4 两种例)', 10, C_G_T, 'start', True, maxw=300, tag='lh:t')
layers = [
    ('DSV3:MLA 层', 'MLAAttentionSpec(num_kv_heads=1, head_size=576)', C_G, C_G_T, '#f0fdf4'),
    ('DSV4:压缩层', 'MLAAttentionSpec(compress_ratio=4, alignment 576)', C_G, C_G_T, '#f0fdf4'),
    ('DSV4:SWA 层', 'SlidingWindowMLASpec(window 4096 / block 64)', C_SWA, C_SWA_T, '#eff6ff'),
    ('DSV4:indexer', 'indexer cache(候选选择,归 ch26 预告)', C_IX, '#831843', '#fdf2f8'),
]
LY0, LH, LGAP = 112, 58, 12
for i, (nm, spec, stk, tc, fl) in enumerate(layers):
    y = LY0 + i * (LH + LGAP)
    lc.rect(70, y, 330, LH, fl, stk, rx=6, sw=1.4)
    lc.text(240, y + 22, nm, 9.5, tc, 'middle', True, maxw=318, tag='ly:t' + nm)
    lc.text(240, y + 42, spec, 7, '#334155', 'middle', maxw=318, tag='ly:s' + nm)
LY_END = LY0 + 4 * (LH + LGAP) - LGAP
lc.text(240, LY_END + 16, '潜向量不拆 K/V、单『头』——cache 形状的唯一真相源是自报',
        7.5, lc.C_MUTE, 'middle', maxw=330, tag='ly:n')

# ---------------- 中左:收集漏斗 ----------------
FY = 236
lc.rect(452, FY, 150, 90, '#ffffff', C_G, rx=8, sw=1.6)
lc.text(527, FY + 26, '② runner 收集', 10, C_G_T, 'middle', True, maxw=140, tag='fn:t')
lc.text(527, FY + 46, 'get_kv_cache_spec', 8.5, C_G_T, 'middle', True, maxw=140, tag='fn:m')
lc.text(527, FY + 62, '遍历 static_forward_context', 7, '#334155', 'middle', maxw=140, tag='fn:l1')
lc.text(527, FY + 76, '逐层收 spec', 7, '#334155', 'middle', tag='fn:l2')
for (sy, ty) in [(141, 250), (211, 268), (281, 288), (351, 306)]:
    lc.parrow([(400, sy), (452, ty)], C_G, 1.6, 'std')

# ---------------- 中右:四级分流链 ----------------
lc.text(648, 96, '③ 四级分流 get_kv_cache_groups(命中即停)', 10, C_K_T, 'start', True, maxw=400,
        tag='dv:t')
levels = [
    ('一级:is_kv_cache_spec_uniform', '全同 spec → 一组(DSV3 常态)'),
    ('二级:UniformType', '全同 token 槽数 → 一组'),
    ('三级:group_and_unify', 'DSV4 grouped → 多组'),
    ('四级:unify_page_size', '页大小归一 → 多组'),
]
VY0, VH, VGAP = 112, 56, 14
for i, (t, d) in enumerate(levels):
    y = VY0 + i * (VH + VGAP)
    # 菱形近似:窄顶宽底的六边形感 → 用圆角矩形+左右尖(简化为矩形)
    lc.rect(648, y, 330, VH, '#ffffff', C_K, rx=6, sw=1.5)
    lc.text(813, y + 23, t, 9, C_K_T, 'middle', True, maxw=318, tag='dv%d:t' % i)
    lc.text(813, y + 42, d, 7.5, '#334155', 'middle', maxw=318, tag='dv%d:d' % i)
    if i < 3:
        lc.seg(813, y + VH, 813, y + VH + VGAP, C_K, 1.5, 'std')
        lc.text(821, y + VH + 11, '未命中', 6.5, lc.C_MUTE, 'start', tag='dv%d:n' % i)
VY_END = VY0 + 4 * (VH + VGAP) - VGAP
lc.text(813, VY_END + 16, 'vllm/v1/core/kv_cache_utils.py:L1781-L1852', 7.5, lc.C_FAINT, 'middle', tag='dv:src')
lc.parrow([(602, FY + 45), (625, FY + 45), (625, 140), (648, 140)], C_K, 1.8, 'std')

# ---------------- 右:组箱×3 + 共享 BlockPool ----------------
GX = 1050
lc.text(GX + 130, 96, '④ 组 ×N(每组一管理器)', 10, C_K_T, 'start', True, maxw=260, tag='gb:t')
groups = [
    ('组 1:压缩层 MLA', C_G, '#f0fdf4'),
    ('组 2:SWA 子缓存', C_SWA, '#eff6ff'),
    ('组 3:indexer cache', C_IX, '#fdf2f8'),
]
for i, (nm, stk, fl) in enumerate(groups):
    y = 112 + i * 76
    lc.rect(GX, y, 260, 62, fl, stk, rx=7, sw=1.5)
    lc.text(GX + 130, y + 22, nm, 9.5, '#0f172a', 'middle', True, maxw=248, tag='gb%d:t' % i)
    lc.rect(GX + 60, y + 34, 140, 20, lc.C_BADGE_F, C_K, rx=9, sw=1.1)
    lc.text(GX + 130, y + 48, 'KVCacheManager ×1', 7.5, C_K_T, 'middle', True, maxw=132,
            tag='gb%d:m' % i)
lc.parrow([(978, 350), (GX, 143)], C_K, 1.8, 'std')
lc.parrow([(978, 350), (GX, 219)], C_K, 1.8, 'std')
lc.parrow([(978, 350), (GX, 295)], C_K, 1.8, 'std')

# 共享 BlockPool 长条
PB = (GX, 342, 260, 40)
lc.rect(*PB, '#ecfeff', C_K, rx=6, sw=1.8)
lc.text(GX + 130, 360, '共享 BlockPool(一池)', 9.5, C_K_T, 'middle', True, maxw=248, tag='pb:t')
lc.text(GX + 130, 374, '组间仍共享物理块(ch14 已立,回指)', 7, '#334155', 'middle', maxw=248,
        tag='pb:s')
for i in range(3):
    y = 112 + i * 76
    lc.seg(GX + 130, y + 62, GX + 130, 342, C_K, 1.2, 'std')

# ---------------- 底部:merge 判据 + 页协商 + DSV4 实测注 ----------------
BB = (70, 430, 1364, 178)
lc.rect(*BB, lc.C_KV_F, C_K, rx=9, sw=1.6)
lc.text(90, 454, '组的等价类判据:MLAAttentionSpec.merge 断言四字段全同(vllm/v1/kv_cache_interface.py:L429-L468)', 10,
        C_K_T, 'start', True, maxw=1100, tag='bb:t')
fields = ['cache_dtype_str', 'compress_ratio', 'model_version', 'block_stride(indexes_kv_by_block_stride)']
for i, f in enumerate(fields):
    x = 90 + i * 330
    lc.rect(x, 470, 310, 30, '#ffffff', C_K, rx=6, sw=1.2)
    lc.text(x + 155, 489, f, 8, C_K_T, 'middle', True, maxw=300, tag='bb:f%d' % i)
lc.text(90, 526, '四字段任一不同 → 进不了同一组;组的合并产物 = 一份等价 spec + 一组层名', 8.5,
        '#334155', 'start', maxw=900, tag='bb:l1')
lc.text(90, 550, '页协商实证(DSV4 实测):SWA block_size 64 与压缩层 storage 256//4=64 共享同一物理张量粒度——', 8.5,
        C_K_T, 'start', True, maxw=1000, tag='bb:l2')
lc.text(90, 568, 'SlidingWindowMLASpec:window 4096 / block 64 / page 65536 B;压缩层 compress_ratio=4、alignment 576', 8,
        '#334155', 'start', maxw=1000, tag='bb:l3')
lc.text(90, 590, '三级组化是 DSV4 的常态:三类子缓存同池分组,每组一个管理器——「一组一管理器共享一池」的 MLA 兑现', 8.5,
        C_K_T, 'start', True, maxw=1200, tag='bb:l4')

# ---------------- 页脚 ----------------
lc.text(MX, 656, '图例:绿 = 模型层自报/常规 MLA 组 · 蓝 = SWA 子缓存 · 品红 = indexer cache · 青 = 组化与池',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 676, 'spec 自报 = 源码原文(vLLM v0.27.1, 6e448d0ea) · DSV4 实例数值 = host 实测真实例化 · indexer 归 ch26(预告)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 696, '站 5(get_kv_cache_spec 自报)→ 站 6(get_kv_cache_groups 组化)——启动期一次,不再逐拍改', 8.5,
        lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-spec-grouping.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
