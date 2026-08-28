#!/usr/bin/env python3
"""ch15 机制图 12 · Marconi junction 三件套（figure_spec ch15-fig-marconi-junction，模板 flow）

放大自 L0 KV 账本列（kv_column）缓存区·从「多组不动点」到「写回」——junction 三件套
（产出→写回→特赦+停点）的全链展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：junction 三件套把『稀疏组掉队的共享前缀』钉进缓存：不动点产出
uncached=longest−reconciled → 写回 Request.shared_prefix_boundary → reachable_boundaries
特赦 + chunk 停点让该边界状态真被算出来。

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
ORANGE = '#ea580c'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'junction 三件套：把『稀疏组掉队的共享前缀』钉进缓存——不停点就白谈',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '不动点产出差值 → 写回 Request.shared_prefix_boundary → mask 特赦 + chunk 停点让该边界状态真被算出来——三件缺一即落空',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 多组不动点 → 写回'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96

# ---------------- ① 产出 ----------------
C1X, C1W = MX, 400
C1H = 420
lc.rect(C1X, LY, C1W, C1H, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(C1X + 16, LY + 22, '① 不动点产出差值', 11.5, lc.C_TXT, 'start', True, maxw=C1W - 32,
        tag='c1:t')
for i, (ln, col, bold) in enumerate([
        ('full(64)+mamba(64) 混合 · A 48 token', '#334155', False),
        ('full 组认 48（longest=48）', '#334155', False),
        ('mamba 组条目被摘（没缓）→ 它交 0', '#334155', False),
        ('调和 hit = 0——但前缀是真的', lc.C_TXT, True),
        ('uncached = longest 48 − reconciled 0', ORANGE, True),
        ('= 48：各组都认、稀疏组还没缓', ORANGE, False),
        ('boundary = hit + uncached = 48', GREEN, True),
        ('（写回值 = 最长单组命中——注释原话）', GRAY, False)]):
    lc.text(C1X + 18, LY + 48 + i * 24, ln, 8.8, col, 'start', bold, maxw=C1W - 34,
            tag='c1:l%d' % i)
# 对照：两组都缓时
cy = LY + 48 + 8 * 24 + 10
lc.rect(C1X + 16, cy, C1W - 32, 60, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
lc.text(C1X + 30, cy + 20, '对照：两组都缓 @48 时', 8.8, lc.C_TXT, 'start', True, maxw=C1W - 60,
        tag='c1:c1')
lc.text(C1X + 30, cy + 40, 'hit=48、uncached=0 → boundary 归零，', 8.4, '#334155', 'start',
        maxw=C1W - 60, tag='c1:c2')
lc.text(C1X + 30, cy + 54, '特赦与停点都不触发、无副作用', 8.4, '#334155', 'start',
        maxw=C1W - 60, tag='c1:c3')

# ---------------- ② 写回 ----------------
C2X, C2W = C1X + C1W + 44, 330
lc.rect(C2X, LY, C2W, C1H, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(C2X + 16, LY + 22, '② 写回 Request', 11.5, lc.C_TXT, 'start', True, maxw=C2W - 32,
        tag='c2:t')
wb = LY + 40
lc.rect(C2X + 16, wb, C2W - 32, 52, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.3)
lc.text(C2X + 30, wb + 21, 'shared_prefix_boundary = 48', 9.6, lc.C_KV_S, 'start', True,
        maxw=C2W - 60, tag='c2:w')
lc.text(C2X + 30, wb + 41, '调度器在准入查询处写入', 8.2, '#334155', 'start', maxw=C2W - 60,
        tag='c2:w2')
for i, ln in enumerate([
        '谁读它：',
        '· cache_blocks → reachable_boundaries',
        '  （特赦的输入之一）',
        '· _mamba_block_aligned_split',
        '  （chunk 停点的依据）',
        '',
        '一个字段、两路配合——',
        '跨模块协议，不是局部状态。']):
    if ln:
        lc.text(C2X + 18, wb + 78 + i * 19, ln, 8.6, '#334155', 'start', maxw=C2W - 34,
                tag='c2:l%d' % i)

# ---------------- ③ 特赦 + 停点 ----------------
C3X = C2X + C2W + 44
C3W = BXR - C3X
# ③a 特赦
ah = 196
lc.rect(C3X, LY, C3W, ah, '#ffffff', ORANGE, rx=9, sw=1.4)
lc.text(C3X + 16, LY + 22, '③a mask 特赦：稀疏驻留不掉复用点', 11.5, ORANGE, 'start', True,
        maxw=C3W - 32, tag='c3a:t')
for i, ln in enumerate([
        'retention=0（只留锚点）下 8 块只真 2 块：',
        '· replay 边界 159 → 128：位置 1 留一块',
        '· junction 边界 112 → 64：位置 0 留一块',
        '（块内子边界按对齐下取整归属块）',
        '不特赦 ⇒ 稀疏驻留把复用点也筛掉——',
        '后续同类请求 longest 掉队、junction 永远到不了缓存，',
        '稀疏省下的显存被重算吃回去。']):
    lc.text(C3X + 18, LY + 46 + i * 20, ln, 8.6, '#334155', 'start', maxw=C3W - 34,
            tag='c3a:l%d' % i)
# ③b 停点
bh = C1H - ah - 20
by2 = LY + ah + 20
lc.rect(C3X, by2, C3W, bh, '#ffffff', GREEN, rx=9, sw=1.4)
lc.text(C3X + 16, by2 + 22, '③b chunk 停点：让边界状态真被算出来', 11.5, GREEN, 'start', True,
        maxw=C3W - 32, tag='c3b:t')
# chunk 可视
cv_w = C3W - 36
cvs = [(0, 64, '停在 64', '#f0fdf4', GREEN), (64, 100, '36', '#ffffff', GRAY)]
cvs_y = by2 + 40
for (t0, t1, label, fill, stroke) in cvs:
    xx = C3X + 18 + cv_w * t0 / 100
    ww = cv_w * (t1 - t0) / 100
    lc.rect(xx, cvs_y, ww, 30, fill, stroke, rx=3, sw=1.2)
    lc.text(xx + ww / 2, cvs_y + 20, label, 8.4, stroke, 'middle', True, maxw=ww - 4,
            tag='cv:' + label)
lc.text(C3X + 18 + cv_w + 6, cvs_y + 20, 'chunk [0,100)', 8, GRAY, 'start', maxw=90,
        tag='c3b:cvn')
lc.seg(C3X + 18 + cv_w * 0.64, cvs_y - 4, C3X + 18 + cv_w * 0.64, cvs_y, ORANGE, 1.6)
for i, ln in enumerate([
        'junction 64 落在 chunk [0,100) 内 → 停在 64（块对齐下取整）',
        '无 junction 不截：100 原样放行',
        'prompt 210 的 partial-tail：首 chunk 停 192、次 chunk 收 208',
        '——边界状态真被算出来才可缓存；不停点，钉住就落了空']):
    lc.text(C3X + 18, cvs_y + 48 + i * 18, ln, 8.6, '#334155', 'start', maxw=C3W - 34,
            tag='c3b:l%d' % i)

# 流程箭头 ①→②→③
ay2 = LY + 150
lc.seg(C1X + C1W + 4, ay2, C2X - 4, ay2, lc.C_MUTE, 2.0, 'std')
lc.text((C1X + C1W + C2X) / 2, ay2 - 10, 'uncached > 0', 8.2, lc.C_MUTE, 'middle', True,
        maxw=120, tag='fa1')
lc.seg(C2X + C2W + 4, ay2, C3X - 4, ay2, lc.C_MUTE, 2.0, 'std')
lc.text((C2X + C2W + C3X) / 2, ay2 - 10, 'boundary 写回', 8.2, lc.C_MUTE, 'middle', True,
        maxw=120, tag='fa2')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + C1H + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不变量：junction 只在『确有稀疏组掉队』时非零（uncached = longest − reconciled ≥ 0），=0 时 boundary 归零、无副作用',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '三件套缺一即落空：不写回 → 后两路没输入；不特赦 → 稀疏驻留筛掉复用点；不停点 → 边界状态根本不会被算出来、钉住落空',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for stroke, tcol, name in [
        (ORANGE, ORANGE, 'mask 特赦侧（稀疏驻留）'),
        (GREEN, GREEN, 'chunk 停点侧（调度器）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=190, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '数字对（159→128 / 112→64）= 边界 token 数下取整到块边界；两组场景 = full(64)+mamba(64)、hash_block_size=16',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_coordinator.py:L685-L817（uncached=longest−reconciled；get_computed_blocks 写回）· '
        'vllm/v1/core/sched/scheduler.py:L744-L766（准入处写 shared_prefix_boundary）· L424-L437（_mamba_block_aligned_split 停点，块对齐下取整）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（full+mamba 混合 · A 48 / B 80 · mask 例 8 块 · 停点演示 chunk 100 / prompt 210）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-marconi-junction.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
