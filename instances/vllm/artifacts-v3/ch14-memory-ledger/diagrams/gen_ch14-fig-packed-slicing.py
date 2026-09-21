#!/usr/bin/env python3
"""ch14 机制图 · packed 布局与视图切片（figure_spec ch14-fig-packed-slicing，模板 tensor-flow/layout）

站 5 packed 布局的 worker 落地形态，与 ch14-fig-hybrid-theory-model 成对：那张讲
理论必然性（形状代数），这张讲代码实现——上：一条 slab 按 block_stride 切 N 块、
各组在固定 offset 密排（红框 = 跨组重叠）；下：某层 view(-1,stride)[:,off:off+page]
的条带视图（红框 = 碎片化切片：层空间不是连续一整块，而是散在各块内固定 offset
处的小条带）。角注真实 61 层 block_stride=1,435,968。

数字全部取自 figure_spec.numbers（简化例 = traces 场景 E：74880 / 41472 / offset 桶
{0:[a0,b0],8640:[b1],37440:[a1]} / 1000000//74880=13 余 26560 / 切片视图；61 层行 =
block_stride 1435968、202 offset、offset 0 压 5 层、10GiB→7477 块）。坐标由常量/循环
计算；文本全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与语义色 ----------------
W, MX, BXR = 1500, 60, 1440
CA, CA_F1, CA_F2 = lc.C_API_S, '#dbeafe', '#bfdbfe'   # A 组条带 = 蓝两档
CB, CB_F1, CB_F2 = lc.C_KV_S, '#cffafe', '#a5f3fc'    # B 组条带 = 青两档
C_RED = lc.C_ABORT                                    # 红框语义（重叠 / 碎片化）
AMBER = '#b45309'

# 玩具数据（traces toy_two_group_packed）
STRIDE = 74880
A0, A1, B0, B1 = 37440, 37440, 8640, 32832
B_DENSE, NUM_BLOCKS, SLAB = 41472, 13, 973440
REAL_STRIDE, REAL_OFFSETS, REAL_OFF0, REAL_BLOCKS = 1435968, 202, 5, 7477

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'packed 布局与视图切片：一条 slab、固定 offset、按块格周期抽条带', 16.5,
        lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '全部组密排进同一条 slab——block_stride = 最宽组组页和（简化例 74880 = max(2×37440, 8640+32832)）；'
                '每层在固定 offset、以块格为周期抽条带：组内不重叠，跨组同 offset 物理重叠而互不串写',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='sub1')
lc.text(MX, 76, '读图：上 = 分配后的物理形态（红框 = 跨组重叠）；下 = 某层取回自己条带的四步切片'
                '（红框 = 碎片化条带）——层空间不是连续一整块，而是散在各块内固定 offset 处的小条带',
        9.0, lc.C_MUTE, 'start', maxw=1330, tag='sub2')
_ch = 'L0 放大 · GPU 列 · worker 落地'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 上半：slab 物理形态 =================
TY0, TH = 96, 316
lc.rect(MX, TY0, BXR - MX, TH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(MX + 16, TY0 + 22, '上 · 一条 packed_backing slab 按 block_stride 切 13 块——A / B 两组各在固定 offset 密排',
        11.5, lc.C_TXT, 'start', True, maxw=980, tag='tp:t')
lc.text(BXR - 16, TY0 + 22, 'vllm/v1/core/kv_cache_utils.py:L1283-L1358', 8.4, lc.C_FAINT, 'end',
        maxw=340, tag='tp:a')

# ---- 左侧信息列 ----
IX, IW = MX + 16, 236
iy = TY0 + 36
for title, lines, stroke, fill in [
    ('A 组（2 层）', [f'a0 页 {A0} + a1 页 {A1}', f'组宽 {STRIDE} → 最宽，定格宽'], CA, CA_F1),
    ('B 组（2 层）', [f'b0 页 {B0} + b1 页 {B1}', f'组宽 {B_DENSE}'], CB, CB_F1)]:
    h = 24 + len(lines) * 14
    lc.rect(IX, iy, IW, h, fill, stroke, rx=6, sw=1.3)
    lc.text(IX + 12, iy + 17, title, 9.8, stroke, 'start', True, maxw=IW - 24, tag='ix:' + title[:4])
    for k, s in enumerate(lines):
        lc.text(IX + 12, iy + 33 + k * 14, s, 8.6, '#334155', 'start', maxw=IW - 20, tag='ix:' + s[:10])
    iy += h + 10
lc.rect(IX, iy, IW, 38, '#ffffff', AMBER, rx=6, sw=1.3)
lc.text(IX + 12, iy + 16, 'block_stride = max(两组组宽)', 8.8, AMBER, 'start', True, maxw=IW - 24, tag='bs:t')
lc.text(IX + 12, iy + 31, '= max(74880, 41472) = 74880', 8.8, AMBER, 'start', True, maxw=IW - 20, tag='bs:v')
iy += 48
lc.rect(IX, iy, IW, 84, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
lc.text(IX + 12, iy + 16, 'offset 桶（3 个 distinct offset）', 8.8, lc.C_TXT, 'start', True, maxw=IW - 24,
        tag='ob:t')
for k, s in enumerate(['0 → [a0, b0]（跨组重叠）', '8640 → [b1]', '37440 → [a1]',
                       '每桶发一张张量声明，全部别名整条 slab']):
    lc.text(IX + 12, iy + 31 + k * 13.5, s, 8.0, '#334155' if k < 3 else lc.C_MUTE, 'start',
            maxw=IW - 20, tag='ob:%d' % k)

# ---- 主 slab 绘区：块 0..2 + … + 块 12 ----
LANE_LX = 348                                   # 走道标签列
BLK_X0, BLK_W, BLK_GAP = 410, 180, 14
SC = BLK_W / STRIDE                             # 每字节的像素宽
A_TOP = TY0 + 52
A_H, B_H, LANE_GAP = 30, 30, 6
B_TOP = A_TOP + A_H + LANE_GAP
lab_x = [BLK_X0 + i * (BLK_W + BLK_GAP) for i in range(4)]
last_x = BLK_X0 + 4 * (BLK_W + BLK_GAP) + 30    # 省略号后挪一块位


def draw_block(bx, detailed):
    """一个块：外框 + A 道（a0|a1 铺满）+ B 道（b0|b1|尾空）。"""
    lc.rect(bx, A_TOP - 4, BLK_W, A_H + LANE_GAP + B_H + 8, 'none', lc.C_MUTE, rx=4, sw=1.1)
    lc.rect(bx + 1, A_TOP, A0 * SC, A_H, CA_F1, CA, rx=1.5, sw=0.8)
    lc.rect(bx + 1 + A0 * SC, A_TOP, A1 * SC, A_H, CA_F2, CA, rx=1.5, sw=0.8)
    lc.rect(bx + 1, B_TOP, B0 * SC, B_H, CB_F1, CB, rx=1.5, sw=0.8)
    lc.rect(bx + 1 + B0 * SC, B_TOP, B1 * SC, B_H, CB_F2, CB, rx=1.5, sw=0.8)
    if detailed:                                # 仅块 0 标层名
        for t, cx in [('a0', bx + A0 * SC / 2), ('a1', bx + A0 * SC + A1 * SC / 2),
                      ('b0', bx + B0 * SC / 2), ('b1', bx + B0 * SC + B1 * SC / 2)]:
            lc.text(cx, (A_TOP if t[0] == 'a' else B_TOP) + (A_H if t[0] == 'a' else B_H) / 2 + 3,
                    t, 8.2, CA if t[0] == 'a' else CB, 'middle', True, tag='bk:' + t)
        lc.text(bx + B0 * SC + B1 * SC + (BLK_W - B_DENSE * SC) / 2, B_TOP + B_H / 2 + 3,
                '（B 组无条带）', 7.4, lc.C_FAINT, 'middle', maxw=BLK_W - B_DENSE * SC - 6, tag='bk:tail')


for i, bx in enumerate(lab_x):
    draw_block(bx, i == 0)
draw_block(last_x, False)
lc.text((lab_x[3] + BLK_W + last_x) / 2, (A_TOP + B_TOP + B_H) / 2 + 4, '…', 13, lc.C_MUTE, 'middle',
        tag='blk:ell')
# 走道标签（右对齐贴块 0 左缘，让开红虚线框）
lc.text(LANE_LX + 52, A_TOP + A_H / 2 + 3, 'A 组条带', 9.0, CA, 'end', True, maxw=110, tag='lane:A')
lc.text(LANE_LX + 52, B_TOP + B_H / 2 + 3, 'B 组条带', 9.0, CB, 'end', True, maxw=110, tag='lane:B')

# 块 0 的字节刻度：A 道上方（0/37440/74880）、B 道下方（0/8640/41472/74880）
AR = A_TOP - 8
for v, px, anc in [(0, 0, 'middle'), (A0, A0 * SC, 'middle'), (STRIDE, BLK_W, 'middle')]:
    lc.seg(BLK_X0 + px, A_TOP - 1, BLK_X0 + px, AR + 3, lc.C_MUTE, 0.8)
    lc.text(BLK_X0 + px, AR, str(v), 7.6, lc.C_MUTE, anc, maxw=60, tag='ar:%d' % v)
BR_ = B_TOP + B_H + 12
for v, px, anc in [(0, 0, 'middle'), (B0, B0 * SC, 'start'), (B_DENSE, B_DENSE * SC, 'middle'),
                   (STRIDE, BLK_W, 'middle')]:
    lc.seg(BLK_X0 + px, B_TOP + B_H + 1, BLK_X0 + px, BR_ - 3, lc.C_MUTE, 0.8)
    lc.text(BLK_X0 + px, BR_, str(v), 7.6, lc.C_MUTE, anc, maxw=60, tag='br:%d' % v)
# 红虚线框：块 0 的 B 组足迹 [0, 41472) 跨两道 = 跨组重叠
ov_w = B_DENSE * SC
lc.rect(BLK_X0 - 3, A_TOP - 3, ov_w + 6, A_H + LANE_GAP + B_H + 6, 'none', C_RED, rx=3, sw=1.5, dash=True)
# 块号
for i, bx in enumerate(lab_x):
    lc.text(bx + BLK_W / 2, BR_ + 16, f'块 {i}', 8.2, lc.C_MUTE, 'middle', tag='bidx:%d' % i)
lc.text(last_x + BLK_W / 2, BR_ + 16, '块 12', 8.2, lc.C_MUTE, 'middle', tag='bidx:12')
# 红框语义 + 块数算术
ny = BR_ + 36
lc.text(BLK_X0, ny, '红框 = 跨组重叠：B 组条带全程叠在 A 组条带上——安全，因为一个块 id 同一时刻只归一组，互不串写',
        9.0, C_RED, 'start', True, maxw=BXR - 16 - BLK_X0, tag='tp:red')
lc.text(BLK_X0, ny + 20, f'示教显存 1000000 B // 74880 = 13 块（floor 余 26560 B 闲置）· slab 总宽 = 74880 × 13 = {SLAB} B',
        8.8, '#334155', 'start', maxw=BXR - 16 - BLK_X0, tag='tp:math')

# ================= 下半：四步切片与条带视图 =================
BY0 = TY0 + TH + 18
lc.seg(MX + 30, TY0 + TH, MX + 30, BY0, lc.C_MUTE, 1.6, 'std')       # 上下两半的接力箭头
BH = 356
lc.rect(MX, BY0, BXR - MX, BH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(MX + 16, BY0 + 22, '下 · 某层取回自己条带：四步切片链（以 B 组 b1 为例）', 11.5, lc.C_TXT,
        'start', True, maxw=800, tag='bp:t')
lc.text(BXR - 16, BY0 + 22, 'vllm/v1/worker/gpu/attn_utils.py:L225-L233（packing 分支）', 8.4,
        lc.C_FAINT, 'end', maxw=400, tag='bp:a')

# ---- 左侧四步 ----
STEPS = [('view(-1, 74880)', '整条 slab 折成 [13, 74880] 块矩阵（每行 = 一块）'),
         ('[:, 8640:41472]', '每块只抽 b1 那条 32832 B 竖带（右图红框）'),
         ('.view(float32)', '32832 B ÷ 4 B/元素 = 8208 元素/块'),
         ('.view(shape)', '成形 [13, 8208] float32——层直接读写自己的视图')]
sx, sw, sh, sgap = MX + 16, 388, 62, 12
sy = BY0 + 38
for k, (expr, desc) in enumerate(STEPS):
    lc.rect(sx, sy, sw, sh, CB_F1 if k == 1 else '#ffffff', CB, rx=7, sw=1.4)
    lc.rect(sx + 8, sy + 8, 22, 17, lc.C_BADGE_F, CB, rx=8, sw=1.0)
    lc.text(sx + 19, sy + 20, '①②③④'[k], 9.5, CB, 'middle', True, tag='st:n%d' % k)
    lc.text(sx + 38, sy + 20, expr, 10.2, lc.C_TXT, 'start', True, maxw=sw - 48, tag='st:e%d' % k)
    lc.text(sx + 14, sy + 44, desc, 8.6, '#334155', 'start', maxw=sw - 26, tag='st:d%d' % k)
    if k < 3:
        lc.seg(sx + sw / 2, sy + sh, sx + sw / 2, sy + sh + sgap, CB, 1.5, 'std')
    sy += sh + sgap

# ---- 右侧：块矩阵条带视图 ----
MX0 = sx + sw + 36                           # 矩阵区左缘（让开行标签与步骤列）
MRW = BXR - 20 - MX0                         # 矩阵行宽
sc_m = MRW / STRIDE                          # 矩阵区每字节像素
row_names = ['块 0', '块 1', '块 2', None, '块 11', '块 12']
ROW_H, ROW_GAP = 26, 12
ry0 = BY0 + 44
lc.text(MX0, ry0 - 10, 'view(-1, 74880) 之后：每行一块、宽 74880——红框 = [:, 8640:41472] 抽出的 b1 条带',
        9.4, lc.C_TXT, 'start', True, maxw=MRW, tag='mt:t')
rows_y = []
for k, name in enumerate(row_names):
    ry = ry0 + 14 + k * (ROW_H + ROW_GAP)
    rows_y.append(ry)
    if name is None:
        lc.text(MX0 + 30, ry + 14, '…', 12, lc.C_MUTE, 'middle', tag='mt:ell')
        continue
    lc.text(MX0 - 8, ry + ROW_H / 2 + 3, name, 8.2, lc.C_MUTE, 'end', tag='mt:' + name)
    lc.rect(MX0, ry, MRW, ROW_H, '#e2e8f0', lc.C_MUTE, rx=2, sw=0.8)          # 块行（别层字节）
    lc.rect(MX0 + B0 * sc_m, ry, B1 * sc_m, ROW_H, CB_F2, C_RED, rx=1.5, sw=1.6)   # b1 条带
    lc.text(MX0 + B0 * sc_m + B1 * sc_m / 2, ry + ROW_H / 2 + 3, 'b1 条带 32832 B', 8.0, CB,
            'middle', True, maxw=B1 * sc_m - 6, tag='mt:b1')
# 红虚线周期导线：条带起止贯穿全部行——碎片化的「隔着块格周期重复」
gx1, gx2 = MX0 + B0 * sc_m, MX0 + (B0 + B1) * sc_m
y_top, y_bot = rows_y[0] - 4, rows_y[-1] + ROW_H + 4
lc.seg(gx1, y_top, gx1, y_bot, C_RED, 1.1, dash=True)
lc.seg(gx2, y_top, gx2, y_bot, C_RED, 1.1, dash=True)
lc.text(gx2 + 8, y_bot + 14, '条带在物理内存里按块格周期（74880）重复、不连续——碎片化',
        8.8, C_RED, 'start', True, maxw=BXR - 20 - (gx2 + 8), tag='mt:frag')
lc.text(MX0, y_bot + 32, '别的层同式切：a0 [:,0:37440]→[13,37440] uint8 · b0 [:,0:8640]→[13,8640] uint8 · '
                         'a1 [:,37440:74880]→[13,37440] uint8',
        8.4, lc.C_MUTE, 'start', maxw=BXR - 20 - MX0, tag='mt:others')

# ================= 角注：真实 61 层 =================
CY0 = BY0 + BH + 16
lc.rect(MX, CY0, BXR - MX, 58, '#fffbeb', AMBER, rx=9, sw=1.2, dash=True)
lc.text(MX + 16, CY0 + 22, '真实 61 层配置：block_stride = 1,435,968 B（MLA 组 91 层最宽：31×1728 + 30×8640 + 30×37440）',
        9.6, AMBER, 'start', True, maxw=BXR - MX - 32, tag='cy:t')
lc.text(MX + 16, CY0 + 42, f'243 条 spec 的层位分布在 {REAL_OFFSETS} 个 offset 上，offset 0 压 {REAL_OFF0} 层'
                           f'（五组都从 0 号偏移起步，重叠最密）；10 GiB 显存 → {REAL_BLOCKS} 块',
        9.0, '#334155', 'start', maxw=BXR - MX - 32, tag='cy:s')

# ================= 图例 + 页脚 =================
GY = CY0 + 74
lx = MX
for f, s_, name in [(CA_F1, CA, 'A 组条带'), (CB_F2, CB, 'B 组条带'), ('#e2e8f0', lc.C_MUTE, '块内其他层字节')]:
    lc.rect(lx, GY - 9, 18, 12, f, s_, rx=2, sw=1.2)
    lc.text(lx + 24, GY, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='lg:' + name[:6])
    lx += 24 + lc.tw(name, 8.8) + 20
lc.rect(lx, GY - 9, 18, 12, 'none', C_RED, rx=2, sw=1.4, dash=True)
lc.text(lx + 24, GY, '红框：上=跨组重叠 · 下=碎片化条带', 8.8, lc.C_TXT, 'start', maxw=320, tag='lg:red')
lx += 24 + lc.tw('红框：上=跨组重叠 · 下=碎片化条带', 8.8) + 20
lc.rect(lx, GY - 9, 18, 12, '#fffbeb', AMBER, rx=2, sw=1.2, dash=True)
lc.text(lx + 24, GY, '角注=真实 61 层数字', 8.8, lc.C_TXT, 'start', maxw=BXR - lx - 24, tag='lg:amber')
lc.text(MX, GY + 20, '布局 vllm/v1/core/kv_cache_utils.py:L1283-L1358 · 切片 vllm/v1/worker/gpu/attn_utils.py:L225-L233 · '
                     '简化例与 61 层数字 = pin 算法实跑复算（层分布为社区实读口径）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ================= 装配输出 =================
H = int(GY + 40)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-packed-slicing.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
