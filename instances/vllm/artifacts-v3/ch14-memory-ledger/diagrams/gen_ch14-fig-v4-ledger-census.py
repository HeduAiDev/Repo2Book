#!/usr/bin/env python3
"""ch14 机制图 · V4 账本普查（figure_spec ch14-fig-v4-ledger-census，模板 tiling）

ch14-fig-hybrid-groups 的边界补图：那两块是等页路径的正例（Gemma3/gpt-oss），
本图画路径外的 DeepSeek V4——六种形态自报四种页宽，等页组化接不住，
vLLM 为它单开第三条分组路径（group_and_unify_kv_cache_specs，注释原话
Currently, this is only used for DeepseekV4）。

claim：六种形态自报四族 spec、四种页宽（37440/8640/1728/32832），等页组化的
六条假设第 3 条接不住——vLLM 为它单开第三条分组路径：MLA 全家打包成
[C4I,C4A,C128] 元组组、滑窗族按 (block_size, window) 分桶组，共 5 组。

数字全部取自 figure_spec.numbers（provenance = traces 直跑 census +
kv_cache_utils.py:L1592-L1754 分组走查转写）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
MLA_S, MLA_F = lc.C_API_S, lc.C_API_F      # MLAAttentionSpec = 蓝（full 管家，与全书 full attention 同色）
SWA_S, SWA_F = lc.C_KV_S, lc.C_KV_F        # SlidingWindowMLASpec = 青（滑窗管家，与全书 SWA 同色）
HL_S, HL_F = lc.C_ENG_S, lc.C_ENG_F        # 本图主角路径（四路分发里的 ③）= 橙（本章高亮色）
AMBER = '#b45309'                          # 同页 37440 的设计证据标注（虚线连线）

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'DeepSeek V4 的账本户口：六种形态自报四种页宽——第三路径「MLA 元组打包 + 滑窗分桶」收编成 5 组',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '等页组化六条假设的第 3 条（每 token 每层字节相同）当场破——vLLM 为 V4 单开第三条分组路径：'
                'MLA 全家打包成 [C4I,C4A,C128] 元组、滑窗族按 (block_size, window) 分桶',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = 'L0 放大 · KV 账本列（池内组化）· 边界补图'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

PY0, PH = 92, 424
LX, LW = MX, 640
RX, RW = LX + LW + 20, BXR - (MX + 640 + 20)

# ---------------- 左面板：普查 · 六类缓存的页宽 ----------------
lc.rect(LX, PY0, LW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, PY0 + 22, '① 普查 · 六种形态自报的页宽（spec 直跑 census）', 11.5, lc.C_TXT,
        'start', True, maxw=LW - 32, tag='lp:t')
lc.text(LX + 16, PY0 + 40, '条长 ∝ 对齐后页字节 · 蓝 = MLAAttentionSpec（full 管家·全历史）· 青 = SlidingWindowMLASpec（滑窗管家）',
        9.2, lc.C_MUTE, 'start', maxw=LW - 32, tag='lp:s')

# 行数据：(名称, 家族, 对齐页 B, spec 类, 块/压缩比, 槽×每槽字节=原始页, 管家)
ROWS = [
    ('滑窗缓存 · 每层一挂', SWA_S, 37440, 'SlidingWindowMLASpec', 'block 64 · 压缩比 1',
     '64 槽 × 584 B = 37376', '滑窗管家 · 窗 2048（示教代入）'),
    ('C4A 主 KV', MLA_S, 37440, 'MLAAttentionSpec', 'block 256 · 压缩比 4',
     '64 槽 × 584 B = 37376', 'full 管家 · 全历史'),
    ('C4I 索引器 KV', MLA_S, 8640, 'MLAAttentionSpec', 'block 256 · 压缩比 4',
     '64 槽 × 132 B = 8448', 'full 管家 · 全历史'),
    ('C128A 主 KV', MLA_S, 1728, 'MLAAttentionSpec', 'block 256 · 压缩比 128',
     '2 槽 × 584 B = 1168', 'full 管家 · 全历史'),
    ('C4 压缩器状态', SWA_S, 32832, 'SlidingWindowMLASpec', 'block 4 · 压缩比 1',
     '4 槽 × 8192 B = 32768', '滑窗管家 · 窗 8（状态滚动缓冲）'),
    ('C128 压缩器状态', SWA_S, 32832, 'SlidingWindowMLASpec', 'block 8 · 压缩比 1',
     '8 槽 × 4096 B = 32768', '滑窗管家 · 窗 128'),
]
PAGE_MAX = 37440
BAR_X0, BAR_MAXW, BAR_H = LX + 158, 320, 15
R0, PITCH, PAIR_EXTRA = PY0 + 72, 46, 16      # 同页对（行 0/1）之间加高放标注
row_y = [R0]
for i in range(1, len(ROWS)):
    row_y.append(row_y[-1] + PITCH + (PAIR_EXTRA if i == 1 else 0))
for (name, fam, page, cls, br, slotmath, mgr), ry in zip(ROWS, row_y):
    lc.text(LX + 16, ry + 16, name, 9.4, fam, 'start', True, maxw=138, tag='row:' + name[:10])
    bw = BAR_MAXW * page / PAGE_MAX
    lc.rect(BAR_X0, ry + 4, bw, BAR_H, fam, fam, rx=2, sw=0.8)
    lc.text(BAR_X0 + bw + 6, ry + 16, f'{page} B', 9.2, fam, 'start', True, maxw=52, tag='pg:%d' % page)
    lc.text(BAR_X0, ry + 33, f'{cls} · {br} · {slotmath} · {mgr}', 7.8, lc.C_MUTE, 'start',
            maxw=LX + LW - 16 - BAR_X0, tag='dt:%d' % page)
# 同页 37440 设计证据：滑窗缓存与 C4A 主 KV 条等长 → 右端虚线相连 + 标注
b_end = BAR_X0 + BAR_MAXW
lc.seg(b_end, row_y[0] + 4 + BAR_H, b_end, row_y[1] + 4, AMBER, 1.6, dash=True)
SP_MAXW = LX + LW - 16 - (b_end + 58)
lc.text(b_end + 58, row_y[0] + 44, '同页 37440 B', 8.8, AMBER, 'start', True, maxw=SP_MAXW, tag='sp1')
lc.text(b_end + 58, row_y[0] + 58, '同一张物理张量', 8.8, AMBER, 'start', maxw=SP_MAXW, tag='sp2')
# 底注三行
for j, note in enumerate([
    '页宽集 {37440, 8640, 1728, 32832}——共 4 种：六条假设第 3 条「每 token 每层字节相同」当场破',
    '32832 与 37440 互不整除、压缩器状态又不属 stride 索引层——硬走等页路径抛 NotImplementedError：'
    '第三路径是唯一可行解，不是优化',
    '同页的设计因果：C4A 块形 [256/4, 584] 反定滑窗块 64 token（sparse_swa.py:L77-L82）'
    '——两类块共享同一张物理张量，故必须同页',
]):
    lc.text(LX + 16, row_y[-1] + 33 + 17 + j * 15, note, 8.4, '#334155', 'start', maxw=LW - 32,
            tag='lp:n%d' % j)

# ---------------- 右面板：打包 · 第三路径分组 ----------------
lc.rect(RX, PY0, RW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, PY0 + 22, '② 打包 · 第三路径分组：MLA 元组 + 滑窗分桶', 11.5, lc.C_TXT,
        'start', True, maxw=RW - 32, tag='rp:t')
lc.text(RX + 16, PY0 + 40, '示例 census（源码注释自带）：11 个 C4 层 + 10 个 C128 层，共 21 层各挂滑窗缓存'
        ' → 按 spec 分 4 桶再组化，共 5 组（元组组 1 + 桶组 4）',
        9.2, lc.C_MUTE, 'start', maxw=RW - 32, tag='rp:s')

# 桶列（4 桶）
BX, BW_, BH, BPITCH = RX + 16, 240, 52, 62
BY0 = PY0 + 70
BUCKETS = [
    ('MLA 全家桶 · 32 个 spec', 'MLAAttentionSpec：21 主 KV + 11 索引器', MLA_S),
    ('滑窗桶 (64, 2048) · 21 个 spec', 'SlidingWindowMLASpec：每层一挂 swa_cache', SWA_S),
    ('C4 状态桶 (4, 8) · 11 个 spec', 'SlidingWindowMLASpec：压缩器 state_cache', SWA_S),
    ('C128 状态桶 (8, 128) · 10 个 spec', 'SlidingWindowMLASpec：压缩器 state_cache', SWA_S),
]
for i, (t, s, fam) in enumerate(BUCKETS):
    by = BY0 + i * BPITCH
    lc.rect(BX, by, BW_, BH, '#ffffff', fam, rx=6, sw=1.5)
    lc.text(BX + 12, by + 21, t, 9.6, fam, 'start', True, maxw=BW_ - 24, tag='bk:%d' % i)
    lc.text(BX + 12, by + 38, s, 8.0, lc.C_MUTE, 'start', maxw=BW_ - 24, tag='bks:%d' % i)

# 组列（5 组：G1 元组组高，G2-G5 桶组）
GX, GW = RX + 408, 296
G1_Y, G1_H = PY0 + 70, 96
lc.rect(GX, G1_Y, GW, G1_H, MLA_F, MLA_S, rx=6, sw=1.6)
lc.text(GX + 12, G1_Y + 17, '组 1 · MLA 元组组（32 个 spec → 11 个元组）', 9.5, MLA_S,
        'start', True, maxw=GW - 24, tag='g1:t')
CHW, CHH, CHGAP = 84, 24, 8
for k, cl in enumerate(['C4I · 8640', 'C4A · 37440', 'C128 · 1728']):
    cx = GX + 14 + k * (CHW + CHGAP)
    lc.rect(cx, G1_Y + 26, CHW, CHH, '#ffffff', MLA_S, rx=4, sw=1.1)
    lc.text(cx + CHW / 2, G1_Y + 42, cl, 8.2, MLA_S, 'middle', True, maxw=CHW - 6, tag='g1:c%d' % k)
lc.text(GX + 12, G1_Y + 64, '× 11 个元组（approx_gcd：C4 恰装满 · C128A 补 1 位）', 8.0,
        '#334155', 'start', maxw=GW - 24, tag='g1:x11')
lc.text(GX + 12, G1_Y + 82, '一个块号 = 元组里每层各一页 · 组页 = 各层页之和', 8.0,
        lc.C_MUTE, 'start', maxw=GW - 24, tag='g1:blk')
G_Y0, G_H, G_PITCH = G1_Y + G1_H + 12, 42, 54
GROUPS = [
    ('组 2 · 滑窗桶切组 a（11 层）', '每层一挂 swa_cache · 桶 (64, 2048)', SWA_S),
    ('组 3 · 滑窗桶切组 b（10 层）', '每层一挂 swa_cache · 桶 (64, 2048)', SWA_S),
    ('组 4 · C4 状态桶组（11 层）', '压缩器 state_cache · 桶 (4, 8)', SWA_S),
    ('组 5 · C128 状态桶组（10 层）', '压缩器 state_cache · 桶 (8, 128)', SWA_S),
]
g_cy = []
for i, (t, s, fam) in enumerate(GROUPS):
    gy = G_Y0 + i * G_PITCH
    lc.rect(GX, gy, GW, G_H, SWA_F, fam, rx=6, sw=1.5)
    lc.text(GX + 12, gy + 17, t, 9.4, fam, 'start', True, maxw=GW - 24, tag='g%d:t' % (i + 2))
    lc.text(GX + 12, gy + 33, s, 8.0, lc.C_MUTE, 'start', maxw=GW - 24, tag='g%d:s' % (i + 2))
    g_cy.append(gy + G_H / 2)
# 桶 → 组连线（tiling：B1 一对二）
LINKS = [(0, G1_Y + G1_H / 2), (0 + 1, g_cy[0]), (1, g_cy[1]), (2, g_cy[2]), (3, g_cy[3])]
for bi, tcy in LINKS:
    fam = BUCKETS[bi][2]
    sy = BY0 + bi * BPITCH + BH / 2
    lc.seg(BX + BW_, sy, GX, tcy, fam, 1.6, marker='std')
# 面板底注两行
for j, note in enumerate([
    '组间页宽不必相等——块号靠重叠打包布局各有落点：一个块 id 同一时刻只归一组',
    '滑窗桶 21 层 → 切 11 + 10 两组；桶组各自成组——「每组层数相同」的约束在第三路径里依然成立',
]):
    lc.text(RX + 16, PY0 + 396 + j * 14, note, 8.4, '#334155', 'start', maxw=RW - 32, tag='rp:n%d' % j)

# ---------------- 页脚 ③：四路分发定位 + 深讲边界 ----------------
FY = PY0 + PH + 16
lc.text(MX, FY + 13, '③ 定位 · get_kv_cache_groups 的四路分发（按序判走）', 10.5, lc.C_TXT,
        'start', True, maxw=700, tag='ft:t')
lc.text(BXR, FY + 13, 'vllm/v1/core/kv_cache_utils.py:L1781-L1852', 8.6, lc.C_FAINT, 'end',
        maxw=420, tag='ft:anchor')
LB_Y, LB_H, LB_GAP = FY + 22, 64, 12
LB_W = (BXR - MX - 3 * LB_GAP) / 4
LADDER = [
    ('① 全层同 spec → 单组', 'uniform 模型全层一组（多数模型）', '_get_kv_cache_groups_uniform_spec', False),
    ('② 等槽位同型 → 单组', '全 full 或同窗 SWA、槽位数一致', 'UniformTypeKVCacheSpecs.from_specs', False),
    ('③ 判据命中 → 元组打包（本图 · V4 专属）', '含 SlidingWindowMLASpec 且页宽多于一种',
     'Currently, this is only used for DeepseekV4', True),
    ('④ 其余混合 → 页宽统一 + 分桶等量', 'Gemma3 / gpt-oss 走的路（前图）', '_get_kv_cache_groups_uniform_page_size', False),
]
for i, (t, s1, s2, hot) in enumerate(LADDER):
    bx = MX + i * (LB_W + LB_GAP)
    stroke, fill, tc = (HL_S, HL_F, HL_S) if hot else (lc.C_FAINT, '#ffffff', lc.C_MUTE)
    lc.rect(bx, LB_Y, LB_W, LB_H, fill, stroke, rx=6, sw=2.0 if hot else 1.2)
    lc.text(bx + 12, LB_Y + 19, t, 9.5, tc, 'start', True, maxw=LB_W - 24, tag='ld%d:t' % i)
    lc.text(bx + 12, LB_Y + 37, s1, 8.2, '#9a3412' if hot else lc.C_MUTE, 'start', maxw=LB_W - 24,
            tag='ld%d:s1' % i)
    lc.text(bx + 12, LB_Y + 53, s2, 7.6, '#9a3412' if hot else lc.C_FAINT, 'start',
            maxw=LB_W - 24, tag='ld%d:s2' % i)
BB_Y = LB_Y + LB_H + 10
lc.rect(MX, BB_Y, BXR - MX, 52, '#ffffff', lc.C_MUTE, rx=6, sw=1.1, dash=True)
lc.text(MX + 16, BB_Y + 17, '深讲边界（均为预告）：584 B 槽特形与 compress_ratio 数学 → 第 24 章（原理章）· '
        '索引器/压缩器/滑窗缓存三件套与打包张量布局 → 第 25/26 章 · MTP 草稿层混进滑窗组（is_eagle_group）→ 第 28 章',
        8.8, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='bb:1')
lc.text(MX + 16, BB_Y + 36, '等页组化（④ 两位前辈 Gemma3 / gpt-oss 走的路）到 V4 为止——元组打包的正片归后续专章展开',
        8.4, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='bb:2')
lc.text(MX, BB_Y + 72, '逐字锚 vllm/v1/kv_cache_interface.py（六种形态的 spec 直跑 census）· '
        'vllm/v1/core/kv_cache_utils.py:L1592-L1754（分组走查·host 转写非直跑）· '
        'vllm/v1/attention/backends/mla/sparse_swa.py:L77-L82（同页注释）· 窗 2048 为示教代入 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
H = int(BB_Y + 72 + 20)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-v4-ledger-census.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
