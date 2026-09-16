#!/usr/bin/env python3
"""ch35 机制图 · m22 EP 全网重排：两次集合通信凑 all-to-all（figure_spec ch35-fig-ep-dispatch-combine）

放大自 L0 多实例视角 ⑤ EP·专家全网重排（L2 站 14）的数据面展开。

claim：EP 每个 MoE 层用两次集合通信凑 all-to-all：dispatch=all_gatherv 让每 rank
拿到全网 token（2×3→每 rank 6 行，多付不归管的行），各 rank 只算命中本地专家的
token（6 行 vs 4 行），combine=reduce_scatterv 按原 sizes 归位求和。

数字全部取自 explainer figure_spec.numbers（traces/m22 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common（GPU 绿系主、灰 MUTE；
奇/偶专家归属用小色点+图例表达）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 64
BXR = 1436

DEFS = lc.DEFS.replace('</defs>',
                       '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="7" markerHeight="5" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
                       '</defs>')

# 2 rank × 3 token × topk 2（traces/m22 同一组数据）
TOPK = [[0, 1], [2, 0], [1, 2]]           # t0/t1/t2 的 topk_ids
RANK0_LOCAL, RANK1_LOCAL = [0, 2], [1, 3]  # 专家 e 归 rank e%2
C_EVEN, C_ODD = lc.C_GPU_S, lc.C_ENG_S


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


def parity_dots(x, y, ids):
    """topk 专家归属小色点（偶=绿 rank0 · 奇=橙 rank1）。"""
    for k, e in enumerate(ids):
        c = C_EVEN if e % 2 == 0 else C_ODD
        lc.ELEMS.append(((x - 4, y - 4, x + 4, y + 4),
                         f'<circle cx="{x}" cy="{y}" r="4" fill="{c}"/>'))
        x += 11


def row_card(x, y, w, h, t, local_parity, side, lit=True):
    """行卡：token + 归属点 + topk + 右侧小注。local_parity=None → 左列无侧注。"""
    stroke = C_EVEN if local_parity == 0 else (C_ODD if local_parity == 1 else lc.C_MUTE)
    fill = '#ffffff' if lit else '#f1f5f9'
    if not lit:
        stroke = lc.C_MUTE
    lc.rect(x, y, w, h, fill, stroke if local_parity is not None else lc.C_FAINT,
            rx=5, sw=1.3)
    parity_dots(x + 14, y + h / 2, TOPK[t])
    lc.text(x + 46, y + h / 2 + 3.5, 't%d · [%d,%d]' % (t, TOPK[t][0], TOPK[t][1]),
            9.5, '#334155' if lit else lc.C_MUTE, 'start', tag='rc%d%d' % (t, y))
    if side:
        lc.text(x + w - 10, y + h / 2 + 3.5, side, 8.5,
                lc.C_MUTE if not lit else stroke, 'end', maxw=w - 120,
                tag='rs%d%d' % (t, y))


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'EP 全网重排：两次集合通信凑一个 all-to-all——dispatch 全网互见，combine 按位归位', 16.5,
        lc.C_TXT, 'start', True, maxw=1140, tag='title')
lc.text(MX, 62, '2 rank × 3 token × topk 2（每 token 两名专家——本例 t0/t2 跨 rank、t1 双偶造出白收白传行）'
        '· 权重 0.5+0.5 · 专家 e 归 rank e%2：rank0 本地 [0,2]、rank1 本地 [1,3]', 10.5, lc.C_MUTE,
        'start', maxw=1300, tag='subtitle')
chip(BXR, 12, '放大自 L2 站 14 · L0：多实例视角', lc.C_GPU_S)

# ---------------- 三列舞台 ----------------
CA_X, CA_W = 64, 292              # 列 A：各持 3 行
CB_X, CB_W = 488, 292             # 列 B：dispatch 后的两桌
CC_X, CC_W = 1012, 292            # 列 C：combine 归位
R0_Y, R1_Y = 156, 388             # rank0 / rank1 两个横带的 y
CARD_H, PITCH = 32, 36

lc.text(CA_X + CA_W / 2, 132, '① 各持 3 行（本地 token）', 11.5, lc.C_TXT, 'middle',
        True, tag='hA')
lc.text(CB_X + CB_W / 2, 118, '② dispatch = all_gatherv(sizes=[3,3]) 沿 dim 0 拼接——每 rank 得 6 行',
        10.5, lc.C_GPU_S, 'middle', True, maxw=620, tag='hB1')
lc.text(CB_X + CB_W / 2, 134, '③ 本地计算：只算命中本地专家的行', 10.5, lc.C_TXT, 'middle',
        True, tag='hB2')
lc.text(CC_X + CC_W / 2, 118, '④ combine = reduce_scatterv：SUM 后按 sizes 切回原主', 10.5,
        lc.C_GPU_S, 'middle', True, maxw=560, tag='hC1')
lc.text(CC_X + CC_W / 2, 134, '每 rank 收回 3 行 = 原值', 10.5, lc.C_TXT, 'middle', True,
        tag='hC2')

# —— 列 A：两叠本地行 ——
for band, (ry, rank, local) in enumerate([(R0_Y, 0, RANK0_LOCAL), (R1_Y, 1, RANK1_LOCAL)]):
    lc.text(CA_X + 2, ry - 8, 'rank%d（本地专家 %s）' % (rank, str(local).replace(' ', '')),
            9.5, C_EVEN if rank == 0 else C_ODD, 'start', True, tag='la%d' % rank)
    for t in range(3):
        row_card(CA_X, ry + t * PITCH, CA_W, CARD_H, t, None, '')

# —— 列 B：dispatch 后的两桌（6 行，gathered 顺序 t0,t1,t2,t0,t1,t2）——
GATHERED = [0, 1, 2, 0, 1, 2]
tables = [
    (R0_Y, 0, 'rank0 桌：6 行全亮（命中 6 · 槽位 8）', True),
    (R1_Y, 1, 'rank1 桌：4 亮 2 灰（命中 4 · 槽位 4）', False),
]
TB_H, TB_PITCH = 26, 29
for ry, rank, header, all_lit in tables:
    lc.text(CB_X + 2, ry - 8, header, 9.5, C_EVEN if rank == 0 else C_ODD, 'start', True,
            maxw=CB_W, tag='tbh%d' % rank)
    for i, t in enumerate(GATHERED):
        hit = any(e % 2 == rank for e in TOPK[t])
        n_slot = sum(1 for e in TOPK[t] if e % 2 == rank)
        side = ('贡献 %d 槽' % n_slot) if hit else '白收白传（贡献 0）'
        row_card(CB_X, ry + i * TB_PITCH, CB_W, TB_H, t, rank, side, lit=hit)

# —— 列 C：combine 归位（每 rank 3 行）——
for ry, rank in [(R0_Y, 0), (R1_Y, 1)]:
    lc.text(CC_X + 2, ry - 8, 'rank%d 收回 3 行（实测 back_equals_h）' % rank, 9.5,
            C_EVEN if rank == 0 else C_ODD, 'start', True, maxw=CC_W, tag='och%d' % rank)
    for t in range(3):
        row_card(CC_X, ry + t * PITCH, CC_W, CARD_H, t, rank, '原值 h[t]')
    lc.text(CC_X + CC_W / 2, ry + 3 * PITCH + 2, '每行总贡献 = 0.5 + 0.5 = 1.0 倍 h[t]',
            8.5, lc.C_MUTE, 'middle', maxw=CC_W, tag='ocf%d' % rank)

# —— 列间大箭头（A→B dispatch、B→C combine） ——
for ry_a, ry_b in [(R0_Y + 3 * PITCH / 2, R0_Y + 6 * TB_PITCH / 2),
                   (R1_Y + 3 * PITCH / 2, R1_Y + 6 * TB_PITCH / 2)]:
    lc.seg(CA_X + CA_W + 4, ry_a, CB_X - 4, ry_b, lc.C_GPU_S, 2.4, 'gn')
    lc.seg(CB_X + CB_W + 4, ry_b, CC_X - 4, ry_a, lc.C_GPU_S, 2.4, 'gn')
lc.text((CA_X + CA_W + CB_X) / 2, 262, '全网互见', 9, lc.C_GPU_S, 'middle', True,
        tag='arr1')
lc.text((CB_X + CB_W + CC_X) / 2, 262, '归位求和', 9, lc.C_GPU_S, 'middle', True,
        tag='arr2')

# ---------------- 底部对账条 ----------------
BY = 620
lc.rect(MX, BY, BXR - MX, 122, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, BY + 22, '投递量对账：AgRs 12 行（每行发给全部 rank）vs 真 A2A 10 行'
        '（rank0 需 6 + rank1 需 4）——多付 2 行、比真 A2A 多 20%', 10.5, lc.C_TXT,
        'start', True, maxw=BXR - MX - 32, tag='acct:l1')
lc.text(MX + 16, BY + 44, 'rank1 有 2 行完全不命中（t1 两专家皆偶——白收白传）；槽位 8 vs 4：'
        '两 rank 计算量也不均——专家分布不均时更甚（EPLB 话题的门牌）', 9.5, '#334155',
        'start', maxw=BXR - MX - 32, tag='acct:l2')
lc.text(MX + 16, BY + 64, 'cluster 越大税越重：AgRs 投递量 ∝ dp_size × tokens，真 A2A ∝ Σ命中行'
        '——这是 DeepEP HT/LL、MoRI、nixl、flashinfer NVLink 等真 A2A 后端存在的理由', 9.5,
        '#334155', 'start', maxw=BXR - MX - 32, tag='acct:l3')
lc.text(MX + 16, BY + 84, '替换面：真 A2A 后端替换的是同一对 dispatch / combine 接口——'
        'all2all_backend 旋钮（all2all.py:L44-L150 · config/parallel.py:L188-L199）', 9.5,
        '#334155', 'start', maxw=BXR - MX - 32, tag='acct:l4')
lc.text(MX + 16, BY + 106, '消费现场：MoE prepare 尾段调 get_ep_group().dispatch（量化完成后、专家 kernel 前）、'
        'finalize 调 combine 覆写输出（naive_dp_ep.py:L158-L164 / L187-L209）', 9, lc.C_MUTE,
        'start', maxw=BXR - MX - 32, tag='acct:l5')

# ---------------- 图例 + 页脚 ----------------
LY = BY + 122 + 26
lx0 = MX
parity_dots(lx0 + 8, LY - 3, [0])
lc.text(lx0 + 26, LY + 1, '偶专家 → rank0 本地', 9.5, lc.C_TXT, 'start', tag='leg:e')
lx0 += 26 + lc.tw('偶专家 → rank0 本地', 9.5) + 22
parity_dots(lx0 + 8, LY - 3, [1])
lc.text(lx0 + 26, LY + 1, '奇专家 → rank1 本地', 9.5, lc.C_TXT, 'start', tag='leg:o')
lx0 += 26 + lc.tw('奇专家 → rank1 本地', 9.5) + 22
lc.rect(lx0 + 4, LY - 9, 16, 11, '#f1f5f9', lc.C_MUTE, rx=3, sw=1.3)
lc.text(lx0 + 26, LY + 1, '灰行 = 不归我管（白收白传）', 9.5, lc.C_TXT, 'start', tag='leg:g')
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· 实测 2 进程 gloo（裸 dispatch/combine 与 MoE '
        'prepare/finalize 消费现场两段都真跑；恒等专家核验证全链 moe_out_equals_a1）', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch35-fig-ep-dispatch-combine.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
