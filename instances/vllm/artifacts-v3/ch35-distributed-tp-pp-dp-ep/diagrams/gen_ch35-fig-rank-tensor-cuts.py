#!/usr/bin/env python3
"""ch35 机制图 · m2 建组四刀：5 维 rank 张量的机械撕法（figure_spec ch35-fig-rank-tensor-cuts）

放大自 L0 多实例视角 ① 建组·5 维 rank 张量（L2 站 1-3）。

claim：全部 rank 先排进 (ExternalDP, DP, PP, PCP, TP) 五维张量（8 GPU 例
shape=(1,2,2,1,2)），任一维度的组都由『transpose 到末维→reshape→unbind』机械导出——
TP 相邻切 [[0,1],[2,3],[4,5],[6,7]]、PP [[0,2],[1,3],[4,6],[5,7]]、
DP [[0,4],[2,6],[1,5],[3,7]]、EP(MoE) [[0,1,4,5],[2,3,6,7]]。

数字全部取自 explainer figure_spec.numbers（traces/m02 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common（EP 用 MUTE 灰描边，
不用 KV 青——避免与显存账本角色撞色）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1460, 1004
MX = 64
BXR = 1396

CUT = {  # 四刀角色色（TP 绿 / PP 橙 / DP 蓝 / EP 灰描边）
    'TP': lc.C_GPU_S, 'PP': lc.C_ENG_S, 'DP': lc.C_API_S, 'EP': lc.C_MUTE,
}


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


# ---------------- 标题区 ----------------
lc.text(MX, 36, '同一张床位表切四刀：组不是名单，是 5 维 rank 张量的不同撕法', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 62, '8 个 rank 先排进 (外DP, DP, PP, PCP, TP) = (1, 2, 2, 1, 2) 五维张量——想按哪个维度分组，'
        '就把那一维转到最后一列、抻平、按列撕开（transpose → reshape → unbind）', 10.5,
        lc.C_MUTE, 'start', maxw=1200, tag='subtitle')
chip(BXR, 12, '放大自 L2 站 1-3 · L0：多实例视角', lc.C_GPU_S)

# ---------------- 左上：物理床位网格 ----------------
GX = MX + 84
GY = 128
CW, CH_, CG = 52, 42, 6            # 格宽/高/缝
BGAP = 46                          # 两引擎块间距
# all_ranks[0][dp][pp][0][tp] → 格值
GRID = {0: [[0, 1], [2, 3]], 1: [[4, 5], [6, 7]]}   # dp → pp 行 → [tp 两格]
for dp, rows in GRID.items():
    bx = GX + dp * (2 * CW + CG + BGAP)
    lc.text(bx + CW + CG / 2, GY - 8, '引擎 %d（dp=%d）' % (dp, dp), 10.5, lc.C_TXT,
            'middle', True, tag='blk%d:t' % dp)
    for pp, ranks in enumerate(rows):
        ry = GY + 16 + pp * (CH_ + CG)
        lc.text(bx - 12, ry + CH_ / 2 + 4, 'pp=%d' % pp, 9, lc.C_MUTE, 'end', tag='pp%d' % pp)
        for tp, r in enumerate(ranks):
            cx = bx + tp * (CW + CG)
            lc.rect(cx, ry, CW, CH_, '#ffffff', lc.C_GPU_S, rx=5, sw=1.4)
            lc.text(cx + CW / 2, ry + CH_ / 2 + 5, str(r), 13, lc.C_TXT, 'middle', True,
                    tag='cell%d' % r)
# TP 内层括号（两块共用一行括号线）
tb_y = GY + 16 + 2 * (CH_ + CG) + 6
for dp in (0, 1):
    bx = GX + dp * (2 * CW + CG + BGAP)
    lc.seg(bx, tb_y, bx + 2 * CW + CG, tb_y, lc.C_GPU_S, 1.6)
    lc.seg(bx, tb_y - 5, bx, tb_y, lc.C_GPU_S, 1.6)
    lc.seg(bx + 2 * CW + CG, tb_y - 5, bx + 2 * CW + CG, tb_y, lc.C_GPU_S, 1.6)
lc.text(GX + CW + CG / 2, tb_y + 16, 'TP 维（最内层·相邻）', 9, lc.C_GPU_S, 'middle',
        tag='tpbr')
lc.text(GX + 2 * CW + CG + BGAP / 2, tb_y + 16, 'TP 维', 9, lc.C_GPU_S, 'middle', tag='tpbr2')
lc.text(MX, GY + 16 + CH_ + CG / 2 - 14, 'PP 维（行）', 9, lc.C_MUTE, 'end', tag='pplab')
lc.text(MX, GY - 8, 'DP 维（块）', 9, lc.C_MUTE, 'end', tag='dplab')
# 右侧收起维注记
NX = GX + 2 * (2 * CW + CG) + BGAP + 30
lc.rect(NX, GY - 4, BXR - NX, 118, '#f8fafc', lc.C_FAINT, rx=6, sw=1.0)
lc.text(NX + 12, GY + 16, 'all_ranks：shape (1, 2, 2, 1, 2)', 9.5, lc.C_TXT, 'start', True,
        maxw=BXR - NX - 24, tag='note:t')
lc.text(NX + 12, GY + 36, '内容 [[[[[0,1]],[[2,3]]],[[[4,5]],[[6,7]]]]]', 9, '#334155',
        'start', maxw=BXR - NX - 24, tag='note:l1')
lc.text(NX + 12, GY + 56, '外DP=1、PCP=1 两维收起（画不出一格）——', 9, lc.C_MUTE, 'start',
        maxw=BXR - NX - 24, tag='note:l2')
lc.text(NX + 12, GY + 74, '五维全展开即左图；抻平后 0..7 顺序排尽', 9, lc.C_MUTE, 'start',
        maxw=BXR - NX - 24, tag='note:l3')
lc.text(NX + 12, GY + 94, 'TP 放最内层 = 同组 rank 物理相邻（同 DGX box，带宽最高）', 9,
        lc.C_GPU_S, 'start', maxw=BXR - NX - 24, tag='note:l4')

# ---------------- 中部：四刀撕法行 ----------------
CY0 = 328
RH = 108                          # 每刀行高
SX = MX + 236                     # 撕出的组条带起点
cuts = [
    ('TP', 'all_ranks.view(-1, 2)', '.unbind(0) —— 相邻 4 组', '相邻切·4 组×2 rank',
     [[0, 1], [2, 3], [4, 5], [6, 7]],
     '同组 rank 物理相邻——TP 睡相邻铺（NVLink 岛），独享 SHM mq_broadcaster'),
    ('PP', 'all_ranks.transpose(2, 4)', '→ reshape(-1, 2) → unbind(0)', '跨段 stride 2·4 组×2 rank',
     [[0, 2], [1, 3], [4, 6], [5, 7]],
     '每引擎各自的 tp0/tp1 两条流水 lane——PP 是引擎内的段序，不跨引擎'),
    ('DP', 'all_ranks.transpose(1, 4)', '→ reshape(-1, 2) → unbind(0)', '跨引擎 stride 4·4 组×2 rank',
     [[0, 4], [2, 6], [1, 5], [3, 7]],
     '同 (tp,pp,pcp) 槽位跨引擎相望——[0,4] 是「跨引擎」的那组：各引擎里干同一份活的人'),
    ('EP', 'all_ranks.transpose(1, 2)', '→ reshape(-1, 4) → unbind(0)', '四格合并·2 组×4 rank',
     [[0, 1, 4, 5], [2, 3, 6, 7]],
     '同一 PP 段内 DP×PCP×TP 全合并；独享 use_all2all（dense 模型这一刀不切）'),
]
gcw, ggap, igap = 44, 16, 5       # 组内格宽 / 组间距 / 组内缝
for i, (dim, f1, f2, cnt, groups, meaning) in enumerate(cuts):
    y = CY0 + i * RH
    color = CUT[dim]
    # 维度徽标
    lc.rect(MX, y + 4, 44, 26, '#ffffff', color, rx=6, sw=1.8)
    lc.text(MX + 22, y + 22, dim + ' 刀', 11, color, 'middle', True, tag='cut%d:d' % i)
    lc.text(MX, y + 48, f1, 8.5, '#334155', 'start', maxw=224, tag='cut%d:f' % i)
    lc.text(MX, y + 66, f2, 8.5, '#334155', 'start', maxw=224, tag='cut%d:f2' % i)
    lc.text(MX, y + 88, cnt, 9, lc.C_MUTE, 'start', maxw=224, tag='cut%d:c' % i)
    # 撕出的组条带（transpose→reshape 后的抻平序，组=行）
    x = SX
    for g, grp in enumerate(groups):
        gw = len(grp) * gcw + (len(grp) - 1) * igap
        lc.rect(x, y + 8, gw, 46, '#ffffff', color, rx=6, sw=1.8)
        for k, r in enumerate(grp):
            lc.text(x + gcw / 2 + k * (gcw + igap), y + 37, str(r), 13, lc.C_TXT, 'middle',
                    True, tag='cut%d:g%dr%d' % (i, g, r))
        # 跨引擎标记（DP 组 [0,4]）：两格不同引擎
        if dim == 'DP' and g == 0:
            lc.text(x + gw / 2, y + 68, '▲跨引擎', 8, lc.C_API_S, 'middle', tag='dp-xeng')
        x += gw + ggap
    # 物理含义（条带右侧）
    lc.text(SX + 8, y + 86, meaning, 9.5, '#334155', 'start', maxw=BXR - SX - 12,
            tag='cut%d:m' % i)

# ---------------- 底部两注记框 ----------------
BY = CY0 + 4 * RH + 18
BW1 = 660
lc.rect(MX, BY, BW1, 128, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, BY + 22, 'dense 对照与特权件（实测）', 10, lc.C_TXT, 'start', True,
        tag='b1:t')
lc.text(MX + 16, BY + 42, '· dense 模型不建 EP 组：『Don\'t create EP group for dense models』'
        '（parallel_state.py:L1920-L1921）', 9, '#334155', 'start', maxw=BW1 - 30, tag='b1:l1')
lc.text(MX + 16, BY + 60, '· factory 调用 5 次(MoE) → 4 次(dense)；TP/PP/DP 三刀两组同式', 9,
        '#334155', 'start', maxw=BW1 - 30, tag='b1:l2')
lc.text(MX + 16, BY + 78, '· 特权件各归一刀：mq_broadcaster 仅 TP 组（recorded 仅 tp:mq）、'
        'use_all2all 仅 EP 组（dp>1）', 9, '#334155', 'start', maxw=BW1 - 30, tag='b1:l3')
lc.text(MX + 16, BY + 96, '· 每维组数：tp=4 · pcp=8 · pp=4 · dp=4 · ep=2 → 22 组 × 2 双群组 '
        '= 44 个 ProcessGroup', 9, '#334155', 'start', maxw=BW1 - 30, tag='b1:l4')
lc.text(MX + 16, BY + 114, '· 每一刀都互斥全覆盖：任一 rank 在每一维恰属一个组（划分不变量）', 9,
        lc.C_MUTE, 'start', maxw=BW1 - 30, tag='b1:l5')

BX2 = MX + BW1 + 20
BW2 = BXR - BX2
lc.rect(BX2, BY, BW2, 128, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(BX2 + 16, BY + 22, 'worker 入全局 world（建 DP/EP 组的前提）', 10, lc.C_TXT, 'start',
        True, maxw=BW2 - 30, tag='b2:t')
lc.text(BX2 + 16, BY + 42, '· global = dp_rank × 4 + engine_rank：rank5 = 1×4 + 1'
        '（实测 global_rank=5 · dp_rank=1 · engine_rank=1）', 9, '#334155', 'start',
        maxw=BW2 - 30, tag='b2:l1')
lc.text(BX2 + 16, BY + 60, '· 各引擎 worker 同处一个全局 world——跨引擎的 DP/EP 组才建得起来', 9,
        '#334155', 'start', maxw=BW2 - 30, tag='b2:l2')
lc.text(BX2 + 16, BY + 78, '· 建组只走一次（启动期）；运行期 get_*_group() 只是取单例', 9,
        '#334155', 'start', maxw=BW2 - 30, tag='b2:l3')
lc.text(BX2 + 16, BY + 96, '· 组列表实测与推导逐组一致：TP/PP/DP/EP 四刀把 0..7 恰好各出现一次', 9,
        '#334155', 'start', maxw=BW2 - 30, tag='b2:l4')
lc.text(BX2 + 16, BY + 114, '· 想按任何一维分组不用另做名单——机械撕法保证组自己掉出来', 9,
        lc.C_MUTE, 'start', maxw=BW2 - 30, tag='b2:l5')

# ---------------- 图例 + 页脚 ----------------
LY = BY + 128 + 26
lx0 = MX
for dim in ('TP', 'PP', 'DP', 'EP'):
    color = CUT[dim]
    lc.rect(lx0, LY - 9, 16, 11, '#ffffff', color, rx=3, sw=1.6)
    lc.text(lx0 + 21, LY + 1, dim + ' 刀撕出的组', 9.5, lc.C_TXT, 'start', tag='leg:' + dim)
    lx0 += 21 + lc.tw(dim + ' 刀撕出的组', 9.5) + 22
lc.text(lx0 + 4, LY + 1, '格内数字 = rank 号 · 同色框 = 同一组', 9.5, lc.C_MUTE, 'start',
        maxw=420, tag='leg:note')
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· 建组为真跑实测（8 进程 gloo，init_model_parallel_group '
        '打桩只记录 group_ranks）· EP 用灰描边非角色色', 9, lc.C_MUTE, 'start',
        maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch35-fig-rank-tensor-cuts.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
