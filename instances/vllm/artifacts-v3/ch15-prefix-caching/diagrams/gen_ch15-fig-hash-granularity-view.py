#!/usr/bin/env python3
"""ch15 机制图 9 · 哈希粒度与粗块视图（figure_spec ch15-fig-hash-granularity-view，模板 state-table）

放大自 L0 KV 账本列（kv_column）缓存区·存储面——「哈希粒度 hash_block_size」一格的展开
（与平面哈希表相邻）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：细粒度哈希零成本重串成粗块视图——粗块哈希就是块内最后一个细粒度哈希（链尾即前缀
指纹），细粒度查找则保留原始列表供块内探测。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
GREEN = '#16a34a'
GRAY = '#64748b'   # 对比度回修：原 #94a3b8 在常规阅读缩放下视图行近不可见，加深一档

# ---------------- 标题区 ----------------
lc.text(MX, 34, '换粗块不重算哈希：粗块指纹 = 块内最后一个细粒度哈希——重串 0 次哈希',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '64 token 在 16 粒度下 4 个哈希；32 粒度视图直接取第 2、4 个、64 粒度取第 4 个——视图只是索引选择，不调用哈希函数',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 存储面「哈希粒度」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96
TOTAL = 64
RX0, RW_ = MX + 240, 700     # 标尺区 x/宽（token 0..64）
LBL_X = MX + 20              # 行标签 x

# ---------------- token 标尺 ----------------
RY = LY + 16
for i in range(4):
    t0, t1 = i * 16, (i + 1) * 16
    xx = RX0 + RW_ * t0 / TOTAL
    ww = RW_ * 16 / TOTAL
    lc.rect(xx, RY, ww, 26, '#ffffff', lc.C_MUTE, rx=0, sw=1.2)
    lc.text(xx + ww / 2, RY + 17, 'token %d-%d' % (t0, t1 - 1), 8.4, GRAY, 'middle', maxw=ww - 4,
            tag='rul%d' % i)
for t in (0, 16, 32, 48, 64):
    xx = RX0 + RW_ * t / TOTAL
    lc.text(xx, RY + 40, str(t), 8, lc.C_MUTE, 'middle', maxw=30, tag='tk%d' % t)
lc.text(LBL_X, RY + 17, '细粒度边界（每 16 token）：', 9, lc.C_TXT, 'start', True, maxw=200,
        tag='rul:lbl')

# ---------------- 三个视图行 ----------------
VY0 = RY + 62
VH, VGAP = 66, 26
VIEWS = [
    ('原始 · hash_block_size=16', '4 块 · 每块覆盖 16 token',
     [(0, 16, 'h0', '@16'), (16, 32, 'h1', '@32'), (32, 48, 'h2', '@48'), (48, 64, 'h3', '@64')],
     '原始列表 0..3'),
    ('粗视图 · target=32（m=2）', '2 块 · 取原始索引 1、3',
     [(0, 32, 'h1', '@32 · 0-31'), (32, 64, 'h3', '@64 · 32-63')],
     'len = 2'),
    ('粗视图 · target=64（m=4）', '1 块 · 取原始索引 3',
     [(0, 64, 'h3', '@64 · 0-63')],
     'len = 1'),
]
for vi, (nm, sub, cells, tail) in enumerate(VIEWS):
    vy = VY0 + vi * (VH + VGAP)
    lc.text(LBL_X, vy + 20, nm, 9.6, lc.C_TXT, 'start', True, maxw=210, tag='v%d:n' % vi)
    lc.text(LBL_X, vy + 38, sub, 8.2, GRAY, 'start', maxw=210, tag='v%d:s' % vi)
    for (t0, t1, hname, ann) in cells:
        xx = RX0 + RW_ * t0 / TOTAL
        ww = RW_ * (t1 - t0) / TOTAL
        lc.rect(xx, vy, ww, VH, '#f0fdf4' if vi == 0 else lc.C_KV_F, GREEN if vi == 0 else lc.C_KV_S,
                rx=5, sw=1.5)
        lc.text(xx + ww / 2, vy + 27, hname, 11, GREEN if vi == 0 else lc.C_KV_S, 'middle', True,
                maxw=ww - 6, tag='v%d:%s' % (vi, hname))
        lc.text(xx + ww / 2, vy + 48, ann, 8.2, '#334155', 'middle', maxw=ww - 6,
                tag='v%d:%s:a' % (vi, hname))
    # 视图取用指示
    lc.text(RX0 + RW_ + 12, vy + 27, tail, 8.4, lc.C_MUTE, 'start', maxw=70, tag='v%d:t' % vi)
    lc.text(RX0 + RW_ + 12, vy + 45, '直接复用链尾' if vi > 0 else '', 8.2, GREEN, 'start',
            maxw=70, tag='v%d:u' % vi)
# 视图间投影线（@32/@64 边界竖线贯穿三行）
for t in (32, 64):
    xx = RX0 + RW_ * t / TOTAL
    lc.seg(xx, RY + 26, xx, VY0 + 3 * VH + 2 * VGAP - 8, GRAY, 1.2, dash=True)

# ---------------- 右：细粒度查找 + 恒等式 ----------------
PXX = BXR - 350
PWW = 350
PHT = VY0 + 3 * VH + 2 * VGAP + 4 - (RY - 6)
lc.rect(PXX, RY - 6, PWW, PHT, '#f8fafc', GRAY, rx=9, sw=1.2, dash=True)
lc.text(PXX + 16, RY + 12, '细粒度查找：原始列表原样保留', 10, lc.C_TXT, 'start', True,
        maxw=PWW - 32, tag='p:t')
for i, ln in enumerate([
        'supports_fine_grained_hash_lookup 且',
        'alignment(16) < block_size 时——',
        '保留原始 4 个哈希供 phase 2 块内',
        '探测（块内边界自高向低试）。',
        '',
        '粗视图则按 target/block 缩放因子 m',
        '取每块最后一个细哈希：',
        '粗块 i 的哈希 = raw[(i+1)·m−1]',
        '——它已经链住了粗块结尾之前的',
        '全部内容，与按粗粒度重算逐块相等。']):
    if ln:
        lc.text(PXX + 16, RY + 34 + i * 19, ln, 8.5, '#334155', 'start', maxw=PWW - 32,
                tag='p:l%d' % i)
lc.text(PXX + 16, RY + 34 + 10 * 19 + 8, '重算 0 次 · 新哈希 0 次', 9, GREEN, 'start', True,
        maxw=PWW - 32, tag='p:z')

# ---------------- 底部不变量条（全宽） ----------------
BY = VY0 + 3 * VH + 2 * VGAP + 14
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '混合模型里最粗 64-token 块与最细 16 粒度共用同一串请求侧哈希：请求侧每 16 token 付 1 次 sha256，两种块尺寸都够用',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '这就是 hash_block_size 与 block_size 能解耦、prefix_match_unit 能设得比物理块还细的全部机制基础——「只控匹配粒度，不控存储频率」',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, tcol, name in [
        ('#f0fdf4', GREEN, GREEN, '原始细粒度哈希（请求侧逐块算出）'),
        (lc.C_KV_F, lc.C_KV_S, lc.C_KV_S, '粗视图直接取用的链尾哈希')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=250, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '@N = 该哈希盖住的 token 上界（链式边界）；竖虚线 = 32/64 粗块边界与细粒度边界的重合处',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_utils.py:L2245-L2320（BlockHashListWithBlockSize）· '
        'L2321-L2380（resolve_block_hashes：粗视图取链尾 / 细粒度保留原始列表）· vllm/config/cache.py:L56-L67（prefix_match_unit 只控匹配粒度）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（64 token · 4 个细哈希 @16/@32/@48/@64 · 重串 0 次）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=680, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-hash-granularity-view.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
