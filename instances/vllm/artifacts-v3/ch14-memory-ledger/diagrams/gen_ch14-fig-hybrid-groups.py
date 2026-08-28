#!/usr/bin/env python3
"""ch14 机制图 3 · 混合组化（figure_spec ch14-fig-hybrid-groups，模板 tiling）

放大自 L0 KV 账本列（池内）的组化层——本章 L2 章图中排拍片①「混合组化
get_kv_cache_groups」的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）。

claim：混合模型按 spec 分桶、组数等量化：12 SW + 13 full 用 1.5 启发式取组大小
13 补成 13/13（而非 12/24），层按桶内交错分派——一池共享的代价是每组层数必须相同。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；
文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
SWA_S, SWA_F = lc.C_KV_S, lc.C_KV_F          # SWA 层 = 青
FUL_S, FUL_F = lc.C_API_S, lc.C_API_F        # full attention 层 = 蓝
PAD_S, PAD_F = '#94a3b8', '#f8fafc'          # padding 层 = 灰虚线

# ---------------- 标题区 ----------------
lc.text(MX, 34, '混合组化：分桶切组、每组层数必须相同——1.5 启发式取 13 补成 13/13，交错入组保 PP 均衡',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'uniform 模型全层一组（32 full → 1 组 × 32 层，多数模型、无 padding）；混合模型按注意力类型分桶——每组层数相同 = 一池共享等大块的硬约束',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列（池内组化）· L2 拍片①'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

PY0, PH = 92, 322

# ---------------- 左面板：Gemma3 式 10 SWA + 2 full ----------------
LX, LW = MX, 680
lc.rect(LX, PY0, LW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, PY0 + 22, '场景一 · Gemma3 式 10 SWA + 2 full', 11.5, lc.C_TXT, 'start', True,
        maxw=LW - 32, tag='lp:t')
lc.text(LX + 16, PY0 + 40, '组大小 2（= min 桶层数）→ 6 组 × 2 层 · 无 padding', 9.2,
        lc.C_MUTE, 'start', maxw=LW - 32, tag='lp:s')
# 顶行 12 个层片
CHW, CHH, CGAP = 44, 26, 6
cx0 = LX + 24
CHIP_Y = PY0 + 62
for i in range(12):
    s, f = (SWA_S, SWA_F) if i < 10 else (FUL_S, FUL_F)
    lc.rect(cx0 + i * (CHW + CGAP), CHIP_Y, CHW, CHH, f, s, rx=4, sw=1.3)
    lc.text(cx0 + i * (CHW + CGAP) + CHW / 2, CHIP_Y + 17, str(i), 9.5, s, 'middle', True,
            maxw=CHW - 4, tag='top%d' % i)
lc.text(LX + 16, CHIP_Y + 17, '层', 9, lc.C_MUTE, 'end', maxw=20, tag='toplbl')
# 6 个组框
GW, GGAP = 100, 8
gx0 = LX + 24
GY, GH = PY0 + 148, 78
GROUPS = [([0, 5], SWA_S, SWA_F), ([1, 6], SWA_S, SWA_F), ([2, 7], SWA_S, SWA_F),
          ([3, 8], SWA_S, SWA_F), ([4, 9], SWA_S, SWA_F), ([10, 11], FUL_S, FUL_F)]
for j, (layers, s, f) in enumerate(GROUPS):
    gxx = gx0 + j * (GW + GGAP)
    lc.rect(gxx, GY, GW, GH, '#ffffff', s, rx=6, sw=1.5)
    lc.text(gxx + GW / 2, GY + 16, '组%d' % (j + 1), 9.5, s, 'middle', True, maxw=GW - 8,
            tag='g%d:t' % j)
    for k, lay in enumerate(layers):
        ccx = gxx + 12 + k * 40
        lc.rect(ccx, GY + 26, 36, 22, f, s, rx=4, sw=1.1)
        lc.text(ccx + 18, GY + 41, str(lay), 9, s, 'middle', True, maxw=32, tag='g%d:c%d' % (j, k))
    # 连线：顶行层片 → 组框顶边
    for lay in layers:
        sx = cx0 + lay * (CHW + CGAP) + CHW / 2
        lc.seg(sx, CHIP_Y + CHH, gxx + GW / 2, GY - 2, s, 1.0)
# 面板底注
NY = GY + GH + 20
lc.text(LX + 16, NY + 12, '桶内交错切片：SWA 桶 10 层按 layers[i::5] 进 5 个组（0,5 / 1,6 / …）', 8.8,
        '#334155', 'start', maxw=LW - 32, tag='lp:n1')
lc.text(LX + 16, NY + 29, '——PP 时每个 stage 都分得到组，不出空组；full 桶 2 层单成 1 组', 8.8,
        '#334155', 'start', maxw=LW - 32, tag='lp:n2')
lc.text(LX + 16, NY + 46, '组大小 = min(10, 2) = 2；10 % 2 = 0 恰好整除 → 0 padding', 8.8,
        '#334155', 'start', maxw=LW - 32, tag='lp:n3')

# ---------------- 右面板：gpt-oss 式 12 SW + 13 full ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
lc.rect(RX, PY0, RW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, PY0 + 22, '场景二 · gpt-oss 式 12 SW + 13 full', 11.5, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='rp:t')
lc.text(RX + 16, PY0 + 40, '1.5 启发式：max(13) < 12 × 1.5 = 18 → 组大小取 13（而非 min=12）', 9.2,
        lc.C_MUTE, 'start', maxw=RW - 32, tag='rp:s')
# 两个组行（13 片横排）
RW_CH, RW_CHH, RW_CG = 22, 24, 3
ROW1_Y, ROW2_Y = PY0 + 66, PY0 + 118
lab_x = RX + 16
chip_x0 = RX + 96
lc.text(lab_x, ROW1_Y + 16, '组1 · 13 层', 9.5, SWA_S, 'start', True, maxw=76, tag='r1:lab')
lc.text(lab_x, ROW1_Y + 30, 'SWA 桶 + pad', 8, lc.C_MUTE, 'start', maxw=76, tag='r1:sub')
for i in range(13):
    xx = chip_x0 + i * (RW_CH + RW_CG)
    if i < 12:
        lc.rect(xx, ROW1_Y, RW_CH, RW_CHH, SWA_F, SWA_S, rx=3, sw=1.1)
        lc.text(xx + RW_CH / 2, ROW1_Y + 16, str(i), 8, SWA_S, 'middle', True, maxw=20,
                tag='r1c%d' % i)
    else:
        lc.rect(xx, ROW1_Y, RW_CH, RW_CHH, PAD_F, PAD_S, rx=3, sw=1.1, dash=True)
        lc.text(xx + RW_CH / 2, ROW1_Y + 16, 'P', 8, PAD_S, 'middle', True, maxw=20, tag='r1pad')
lc.text(chip_x0 + 13 * (RW_CH + RW_CG) + 6, ROW1_Y + 16, '= SWA 12 层 + 1 padding', 8.4,
        '#334155', 'start', maxw=RW - 96 - 13 * (RW_CH + RW_CG) - 12, tag='r1:note')
lc.text(lab_x, ROW2_Y + 16, '组2 · 13 层', 9.5, FUL_S, 'start', True, maxw=76, tag='r2:lab')
lc.text(lab_x, ROW2_Y + 30, 'full 桶', 8, lc.C_MUTE, 'start', maxw=76, tag='r2:sub')
for i in range(13):
    xx = chip_x0 + i * (RW_CH + RW_CG)
    lc.rect(xx, ROW2_Y, RW_CH, RW_CHH, FUL_F, FUL_S, rx=3, sw=1.1)
    lc.text(xx + RW_CH / 2, ROW2_Y + 16, str(12 + i), 8, FUL_S, 'middle', True, maxw=20,
            tag='r2c%d' % i)
# warning 与对照
WY2 = ROW2_Y + 44
lc.rect(RX + 16, WY2, RW - 32, 46, '#fffbeb', '#d97706', rx=6, sw=1.2)
lc.text(RX + 28, WY2 + 18, 'warning 原话：Add 1 padding layers, may waste at most 8.33% KV cache memory', 8.8,
        '#92400e', 'start', True, maxw=RW - 56, tag='rp:w')
lc.text(RX + 28, WY2 + 35, 'padding 层不存 KV、只为凑每组层数相同——上界 = padding 1 层 ÷ 桶内真实 12 层 = 8.33%', 8.4,
        '#92400e', 'start', maxw=RW - 56, tag='rp:w2')
CY2 = WY2 + 58
lc.text(RX + 16, CY2 + 12, '若按 min=12 切：full 桶 13 层要补 11 层到 24（组 ×2）', 8.8, '#334155',
        'start', maxw=RW - 32, tag='rp:c1')
lc.text(RX + 16, CY2 + 29, '——padding 11 层 vs 1 层，这就是 1.5 启发式存在的理由', 8.8, '#334155',
        'start', maxw=RW - 32, tag='rp:c2')

# ---------------- 底部：硬约束条（全宽） ----------------
BY = PY0 + PH + 22
lc.rect(MX, BY, BXR - MX, 64, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '硬约束：每组层数必须相同——一池共享「每块物理字节相等」等大块的前提（页大小统一后组内等价）', 10.2,
        lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='bd:t')
lc.text(MX + 16, BY + 42, '运行时哨兵：get_kv_cache_configs 里 assert Σlen(group.layer_names) == len(spec)——漏一层/重一层当场炸，不带错账进池', 9,
        '#334155', 'start', maxw=BXR - MX - 32, tag='bd:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 88
lx = MX
for s, f, dash, name in [(SWA_S, SWA_F, False, 'SWA / 滑窗层'), (FUL_S, FUL_F, False, 'full attention 层'),
                         (PAD_S, PAD_F, True, 'padding 层（不存 KV，凑数）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, f, s, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=170, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '组框内数字 = 层号（模型里的真实层序）', 8.8, lc.C_MUTE, 'start',
        maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_utils.py:L1781-L1852（分桶与合并）· L1205-L1279（组数等量化）· '
        'L1248-L1265（padding 警告原文）· 数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
H = LEG_Y + 44
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-hybrid-groups.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
