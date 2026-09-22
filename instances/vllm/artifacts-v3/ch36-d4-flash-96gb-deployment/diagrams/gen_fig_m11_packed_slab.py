#!/usr/bin/env python3
"""ch36 机制图 · packed 定账：一个池五个租户（figure_spec fig_m11_packed_slab，模板 layout）

放大自 L2 章图拍片⑧（packed 定账）——L0 kv_column『池』格的字节布局展开。

claim：packed 定账：一个池而不是五个池——五组从偏移 0 各自叠放层页，块宽取最重组
（full_mla 917,568B）、其余组叠放其上（块 id 同一时刻只属一组，组间可重叠）；
num_blocks = available // 917,568 一次除法，落地时全部张量别名同一 torch.zeros。

数字全部取自 figure_spec.numbers（final_groups stack_bytes / block_stride / offsets['0']=5 /
scenarios[2].num_blocks=9361 / 误读 4,092,480B=4.46×）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 742
MX, BXR = 56, 1444
DEFS = lc.DEFS

C_MLA_S, C_MLA_F = lc.C_ZMQ_S, lc.C_ZMQ_F      # full_mla = 紫
C_SWA_S, C_SWA_F = lc.C_API_S, lc.C_API_F      # SWA = 蓝
C_ST4_S, C_ST4_F = lc.C_ENG_S, lc.C_ENG_F      # C4 状态 = 橙
C_ST128_S, C_ST128_F = lc.C_SAM_S, lc.C_SAM_F  # C128 状态 = 品红
C_ERR_S, C_ERR_F = lc.C_ABORT, '#fee2e2'       # 误读 = 红

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一个池五个租户：块宽按最宽租户修（917,568B），窄租户叠在其上', 16.5,
        lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '每组层页在自己块内从偏移 0 叠起 · 一个块 id 同一时刻只租给一个组 → 组间可重叠、组内互斥 · num_blocks = available // 917,568 一次整除',
        10.5, lc.C_MUTE, 'start', maxw=1300, tag='subtitle')
_ch = '放大自 L2 拍片⑧ · L0 显存账本列 · 池格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 五条泳道（同一尺度：917,568B → 780px） ----------------
LANE_X0 = MX + 206                       # 偏移 0 的 x
PX_PER_B = 780 / 917568
STRIDE_X = LANE_X0 + 917568 * PX_PER_B   # 1042

LANES = [
    (C_MLA_S, C_MLA_F, 'full_mla', '62 层', 917568,
     [('21 × 37,440B（C4A 主）', 786240), ('21 × 4,608B（C4I）', 96768), ('20 × 1,728B（C128 主）', 34560)]),
    (C_SWA_S, C_SWA_F, 'SWA-1', '22 层', 823680, [('22 × 37,440B（窗口）', 823680)]),
    (C_SWA_S, C_SWA_F, 'SWA-2', '22 层 · 标 eagle', 823680, [('22 × 37,440B（窗口 · 含 MTP 末层）', 823680)]),
    (C_ST4_S, C_ST4_F, '状态 (4,8)', '42 层 = 21 对双生', 870912,
     [('21 × 32,832B（attn 态）', 689472), ('21 × 8,640B（indexer 态）', 181440)]),
    (C_ST128_S, C_ST128_F, '状态 (8,128)', '20 层', 656640, [('20 × 32,832B', 656640)]),
]
LY0, LANE_H, LANE_GAP = 148, 56, 13
for i, (s, f, name, layers, total, segs) in enumerate(LANES):
    ly = LY0 + i * (LANE_H + LANE_GAP)
    lc.text(MX + 14, ly + 24, name, 10.2, s, 'start', True, maxw=110, tag=f'ln{i}')
    lc.text(MX + 14, ly + 42, layers, 8.2, '#475569', 'start', maxw=150, tag=f'll{i}')
    defines = (name == 'full_mla')
    bx = LANE_X0
    out_row = 0                            # 窄段外置标签的堆叠行号（防同 y 相撞）
    for j, (lab, nbytes) in enumerate(segs):
        w_ = nbytes * PX_PER_B
        lc.rect(bx, ly + 8, w_, 36, f if j % 2 == 0 else '#ffffff', s, rx=3, sw=1.3)
        if w_ >= 96:
            lc.text(bx + w_ / 2, ly + 30, lab, 8.6, s, 'middle', True, maxw=w_ - 8, tag=f'sg{i}.{j}')
        else:                              # 窄段：标签挂到条外上方，逐段换行防叠
            lc.text(bx + w_ / 2, ly - 1 - out_row * 12, lab, 7.6, s, 'middle', maxw=130, tag=f'sg{i}.{j}o')
            out_row += 1
        bx += w_
    lc.text(bx + 8, ly + 30, f'Σ {total:,}', 8.6, s if defines else lc.C_MUTE, 'start',
            bold=defines, maxw=90, tag=f'st{i}')

# 偏移 0 线 + 标注
TOP_Y, BOT_Y = LY0 - 26, LY0 + 5 * (LANE_H + LANE_GAP) - LANE_GAP + 8 + 10
lc.seg(LANE_X0, TOP_Y, LANE_X0, BOT_Y, lc.C_MUTE, 1.4, dash=True)
lc.text(LANE_X0 - 6, LY0 + 5 * (LANE_H + LANE_GAP) + 22, '偏移 0（五组共享——组间可重叠的直接证据：offsets[0] = 5 组）',
        8.8, lc.C_MUTE, 'end', maxw=390, tag='off0')
# stride 线 + 标注
lc.seg(STRIDE_X, TOP_Y, STRIDE_X, BOT_Y, lc.C_TXT, 1.6, dash=True)
lc.text(STRIDE_X, TOP_Y - 8, 'block_stride = max(各组 Σ) = 917,568B', 10, lc.C_TXT, 'middle', True,
        maxw=300, tag='stride')

# 右列：定账 + 误读对照
RB_X, RB_W = 1168, BXR - 1168             # 276
lc.rect(RB_X, LY0 - 6, RB_W, 156, lc.C_KV_F, lc.C_KV_S, rx=9, sw=1.6)
lc.text(RB_X + 14, LY0 + 16, '定账：一次整除', 11, lc.C_KV_S, 'start', True, maxw=RB_W - 28, tag='db:t')
for i, ln in enumerate(['num_blocks = available // 917,568',
                        '8GiB 假设池 → 9,361 块',
                        '落地：一次 torch.zeros、五组全部',
                        '张量别名同一背衬（一次 CUDA 分配）']):
    lc.text(RB_X + 14, LY0 + 38 + i * 17, ln, 8.6, '#334155', 'start',
            maxw=RB_W - 24, tag=f'db:{i}')

EB_Y = LY0 + 172
lc.rect(RB_X, EB_Y, RB_W, 176, C_ERR_F, C_ERR_S, rx=9, sw=1.6)
lc.text(RB_X + 14, EB_Y + 20, '误读对照（直觉陷阱）', 10.5, C_ERR_S, 'start', True, maxw=RB_W - 28, tag='eb:t')
lc.text(RB_X + 14, EB_Y + 40, '按『一块 = 全模型每层一页』：', 8.6, '#334155', 'start', maxw=RB_W - 24, tag='eb:l1')
lc.text(RB_X + 14, EB_Y + 56, 'Σ 全五组 = 4,092,480B（4.46×）', 8.8, C_ERR_S, 'start', True, maxw=RB_W - 24, tag='eb:l2')
# 两根对照条（同尺度：917,568B → 46px），标签在条上方
cmp_x = RB_X + 14
lc.text(cmp_x, EB_Y + 68, '实际 max = 917,568', 7.8, lc.C_TXT, 'start', maxw=150, tag='eb:b1')
lc.rect(cmp_x, EB_Y + 72, 46, 13, '#ffffff', lc.C_TXT, rx=2, sw=1.1)
lc.text(cmp_x, EB_Y + 98, '误读 Σ 全五组 = 4,092,480', 7.8, C_ERR_S, 'start', maxw=170, tag='eb:b2')
lc.rect(cmp_x, EB_Y + 102, 46 * 4.46, 13, C_ERR_F, C_ERR_S, rx=2, sw=1.1, dash=True)
lc.text(RB_X + 14, EB_Y + 138, '块数直接缩到不足 1/4.46（同池少装 3/4 以上）', 8.6, '#334155', 'start', maxw=RB_W - 24, tag='eb:l3')

# ---------------- 底部结论带 ----------------
BB_Y = 560
lc.rect(MX, BB_Y, BXR - MX, 96, '#f8fafc', lc.C_MUTE, rx=10, sw=1.5)
lc.text(MX + 18, BB_Y + 24, 'docstring 原话：A block ID is owned by one cache group at a time, so layouts from different groups may overlap', 10.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='bb:t')
lc.text(MX + 18, BB_Y + 48, '· 组内层页互斥不重叠（一条条排开）；组间从偏移 0 叠放、块宽取 max —— 任何时刻一块只租一组，窄租户的层页压在宽租户不用的地方',
        9.2, '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l1')
lc.text(MX + 18, BB_Y + 68, '· 一个池 = 零池间碎片：五本账不再各划各界，页结构一变两组同时移位（全模型耦合的代价）',
        9.2, '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l2')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 46, 'vllm/v1/core/kv_cache_utils.py:L1283-L1305（_get_packed_kv_cache_layout · 逐组偏移 0 叠放 · max）· L1342（num_blocks = available // block_stride）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 31, 'vllm/v1/worker/gpu_model_runner.py:L7327-L7342（单背衬 torch.zeros · 全部 packed 张量别名）· 五组叠放 / 块数 / 误读对照 ＝ 本章账本算术脚本实算（pin 函数逐字复刻）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 16, '行号基线 vLLM v0.27.1（6e448d0ea）· SWA-2 组含 MTP 末层 → 标 eagle（回指分组步④）', 8.5,
        lc.C_FAINT, 'start', maxw=700, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m11_packed_slab.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
