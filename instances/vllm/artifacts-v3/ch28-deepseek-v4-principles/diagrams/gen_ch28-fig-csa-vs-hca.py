#!/usr/bin/env python3
"""ch28 机制图 · 同一套压缩机制的两档：CSA 与 HCA（ch28-fig-csa-vs-hca）

claim：同一套压缩机制的两档：CSA（两组 C/Z、窗宽 2m、有重叠、带索引器只挑 512）
与 HCA（一组 C/Z、窗宽 m′、无重叠、一条不挑）；压到 1/128 后全看比养一个打分器
更划算（每头口径 8.561644 倍）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  同一条目两档产出：i=1 CSA [0.548559, 0.297268] vs HCA [0.642875, 0.777963]；
                    i=2 CSA [0.779006, 0.400221] vs HCA [0.890307, 0.723258]
  i=0 两档退化重合 [0.576481, 0.469171]（逐位相同 = True）
  候选 250000 vs 7812；实看 512 vs 7812；倍数 15.257812
  索引器打分 32000000（每索引头）/ 2048000000（64 头）vs HCA 多看 3737600（每主头）→ 8.561644 倍
  逐层摊销 CSA (584+132)/4 = 179.000000 B vs HCA 584/128 = 4.562500 B
  HCA 的 topk 缓冲＝全填：pos=1000 → 7 条 → [0, 1, 2, 3, 4, 5, 6, −1]

配色走 book/cartography/l0_common.py 的角色常量（KV 青、GPU 绿、橙=分数/选择器）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_CUR_F, C_CUR_S = '#e0f2fe', '#0284c7'
C_PRE_F, C_PRE_S = '#f1f5f9', '#94a3b8'
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'

EXTRA_DEFS = ('<defs>'
              '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '两档压缩机：CSA 挑 512 条，HCA 一条不挑', 17, C_TXT, 'start', True, maxw=620,
        tag='title')
lc.text(MX, 56, '同一台机器换档位：窗宽从 2m = 8 拉到 m′ = 128、去掉重叠、也去掉选择器'
                '——压到 1/128 之后，全看比再养一个打分器更划算', 10, C_MUTE, 'start',
        maxw=1050, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』的两条注意力支路', 9, C_MUTE,
        'end', maxw=430, tag='l0:a')
lc.text(BXR, 64, '——就是 43 层电梯上交替挂着的那两块牌子', 9, C_MUTE, 'end', maxw=430,
        tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_CUR_F, C_CUR_S, '当前窗（a 半区）'), (C_PRE_F, C_PRE_S, '前一个窗（b 半区）'),
                  (C_SCOR_F, C_SCOR_S, '选择器 / 分数'), (C_KV_F, C_KV_S, '压缩条目（KV 青）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ================= 左：CSA =================
LX0, LX1 = MX, 726
lc.rect(LX0, 118, LX1 - LX0, 316, '#ffffff', C_KV_S, rx=10, sw=1.6)
lc.text(LX0 + 16, 144, 'CSA 层（牌子 4）', 13, C_KV_DEEP, 'start', True, maxw=220, tag='cs:t')
lc.text(LX0 + 180, 144, '两组 C/Z · 窗宽 2m = 8 · 有重叠 · 带索引器', 9.5, C_MUTE, 'start',
        maxw=440, tag='cs:s')
# 8 格窗
CBX, CCW = LX0 + 30, 74
for i in range(8):
    f, s = (C_PRE_F, C_PRE_S) if i < 4 else (C_CUR_F, C_CUR_S)
    lc.rect(CBX + i * CCW + 2, 176, CCW - 4, 34, f, s, rx=3, sw=1.2)
lc.text(CBX + 2 * CCW, 168, '前一个窗 4', 8.8, '#64748b', 'middle', tag='cs:w0')
lc.text(CBX + 6 * CCW, 168, '当前窗 4', 8.8, C_CUR_S, 'middle', tag='cs:w1')
lc.text(CBX, 232, '每条条目吃 2m = 8 个位置（相邻两条共享一半）', 9, C_MUTE, 'start', maxw=560,
        tag='cs:w2')
# 选择器漏斗
lc.parrow([(CBX, 244), (CBX, 268)], C_SCOR_S, 2.0, marker=None)
lc.rect(LX0 + 30, 268, 620, 58, C_SCOR_F, C_SCOR_S, rx=8, sw=1.6)
lc.text(LX0 + 46, 292, '索引器：给每个压缩块打分，只挑 top-k = 512', 11, '#7c2d12', 'start',
        True, maxw=420, tag='cs:f0')
lc.text(LX0 + 46, 312, '候选 250000 块 → 实看 min(512, 250000) = 512 条', 9, '#334155', 'start',
        maxw=460, tag='cs:f1')
lc.text(LX0 + 470, 312, '（1M 上下文）', 9, C_MUTE, 'start', tag='cs:f2')
# 核心注意力
lc.parrow([(LX0 + 340, 326), (LX0 + 340, 350)], C_KV_DEEP, 2.2, marker='dn')
lc.rect(LX0 + 30, 350, 620, 44, C_KV_F, C_KV_S, rx=8, sw=1.6)
lc.text(LX0 + 46, 377, '核心注意力：512 条压缩条目 + 128 条滑窗 = KV 轴 640 条', 10.5,
        C_KV_DEEP, 'start', True, maxw=580, tag='cs:c0')

# ================= 右：HCA =================
RX0, RX1 = 774, BXR
lc.rect(RX0, 118, RX1 - RX0, 316, '#ffffff', C_GPU_S, rx=10, sw=1.6)
lc.text(RX0 + 16, 144, 'HCA 层（牌子 128）', 13, C_TXT, 'start', True, maxw=240, tag='hc:t')
lc.text(RX0 + 200, 144, '一组 C/Z · 窗宽 m′ = 128 · 无重叠 · 无选择器', 9.5, C_MUTE, 'start',
        maxw=460, tag='hc:s')
HBX, HCW = RX0 + 30, 150
for i in range(4):
    lc.rect(HBX + i * HCW + 2, 176, HCW - 4, 34, C_CUR_F, C_CUR_S, rx=3, sw=1.2)
lc.text(HBX + 2 * HCW, 168, '一条窗 = m′ = 128 个位置（示例画 4 格）', 8.8, C_CUR_S, 'middle',
        tag='hc:w0')
lc.text(HBX, 232, '每条条目吃 m′ = 128 个位置，与相邻条目不共享', 9, C_MUTE, 'start', maxw=580,
        tag='hc:w1')
# 无选择器（划掉的打分框）
lc.rect(RX0 + 30, 268, 620, 58, '#f8fafc', C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(RX0 + 46, 292, '没有选择器：不建索引缓存、不打分、不排序', 11, '#334155', 'start', True,
        maxw=420, tag='hc:f0')
lc.text(RX0 + 46, 312, '候选 7812 条 → 一次 seq 全填，全部进注意力', 9, '#334155', 'start',
        maxw=460, tag='hc:f1')
scx, scy = RX0 + 468, 300
lc.seg(scx - 44, scy - 12, scx + 44, scy + 12, C_SCOR_S, 2.4)
lc.seg(scx - 44, scy + 12, scx + 44, scy - 12, C_SCOR_S, 2.4)
lc.text(scx, scy - 22, '打分', 8.6, C_SCOR_S, 'middle', tag='hc:x')
lc.parrow([(RX0 + 340, 326), (RX0 + 340, 350)], C_KV_DEEP, 2.2, marker='dn')
lc.rect(RX0 + 30, 350, 620, 44, C_KV_F, C_KV_S, rx=8, sw=1.6)
lc.text(RX0 + 46, 377, '核心注意力：7812 条压缩条目 + 128 条滑窗 = KV 轴 7940 条', 10.5,
        C_KV_DEEP, 'start', True, maxw=580, tag='hc:c0')

# 中缝：m vs m′
lc.seg(750, 130, 750, 424, '#cbd5e1', 1.4, dash=True)
lc.text(750, 462, 'm = 4', 10.5, C_KV_DEEP, 'middle', True, tag='mid:a')
lc.text(750, 480, '对 m′ = 128', 9.5, C_MUTE, 'middle', tag='mid:b')

# ================= 对账表 =================
TY0 = 500
lc.rect(MX, TY0, BXR - MX, 148, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, TY0 + 24, '两档的对账（1M 上下文）：HCA 多看的那部分，比 CSA 省掉的打分小一个量级',
        11.5, C_TXT, 'start', True, maxw=560, tag='tb:t')
ROWS = [['', 'CSA（牌子 4）', 'HCA（牌子 128）', '比值'],
        ['候选压缩块', '250000', '7812', 'HCA 少一个量级'],
        ['每 query 实看条目', '512（top-k 封顶）', '7812（全看）', '15.257812 倍'],
        ['选择器成本（每头/每 query）', '打分 32000000 次乘加', 'HCA 多看 3737600 次乘加', '8.561644 倍'],
        ['逐层摊销', '179.000000 B/token', '4.562500 B/token', 'CSA 含 132 B/条索引缓存']]
COLX = [MX + 16, MX + 240, MX + 560, MX + 900]
for r, row in enumerate(ROWS):
    ty = TY0 + 46 + r * 19
    for c, cell in enumerate(row):
        if not cell:
            continue
        bold = (r == 0)
        lc.text(COLX[c], ty, cell, 9.2, C_TXT if bold else '#334155', 'start', bold,
                maxw=300, tag='tb:%d%d' % (r, c))
    if r == 0:
        lc.seg(COLX[0], ty + 7, BXR - 30, ty + 7, C_MUTE, 1.2)
lc.text(MX + 16, TY0 + 140, '口径：所有比值都是『每头』并排（索引器有 64 个头、主注意力头数本包未取到）'
                            '——把 64 个头合计起来，差距还要再乘 64。', 8.8, '#155e75', 'start',
        maxw=BXR - MX - 40, tag='tb:n')

# ================= 底左：i=0 退化 + i≥1 真差别 =================
BY0, BY1 = 668, 800
lc.rect(MX, BY0, 726 - MX, BY1 - BY0, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, BY0 + 24, '对账时别拿第一条：i = 0 两档结构性地重合', 11, C_TXT, 'start', True,
        maxw=460, tag='bA:t')
lc.text(MX + 16, BY0 + 46, 'i=0 时 CSA 的 b 半区被 −inf 打掉，归一范围 8 缩回 4——'
                          '两档在这一格上退化成同一件事。', 9, C_MUTE, 'start', maxw=640,
        tag='bA:s')
CP = [('条目 i=0', '[0.576481, 0.469171]', '[0.576481, 0.469171]', '逐位相同 = True（重合）'),
      ('条目 i=1', '[0.548559, 0.297268]', '[0.642875, 0.777963]', '逐位相同 = False'),
      ('条目 i=2', '[0.779006, 0.400221]', '[0.890307, 0.723258]', '逐位相同 = False')]
for i, (a, b, c, d) in enumerate(CP):
    ty = BY0 + 72 + i * 20
    lc.text(MX + 16, ty, a, 9.2, C_TXT, 'start', True, tag='bA:a%d' % i)
    lc.text(MX + 90, ty, 'CSA ' + b, 9.2, C_KV_DEEP, 'start', tag='bA:b%d' % i)
    lc.text(MX + 290, ty, 'HCA ' + c, 9.2, '#334155', 'start', tag='bA:c%d' % i)
    lc.text(MX + 500, ty, d, 9.2, C_MUTE, 'start', maxw=180, tag='bA:d%d' % i)

# ================= 底右：HCA 的『选择』是全填 =================
lc.rect(774, BY0, BXR - 774, BY1 - BY0, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(774 + 16, BY0 + 24, 'HCA 的『选择』写在实现里：把已完成的块按序全填', 11, C_TXT, 'start',
        True, maxw=520, tag='bB:t')
lc.text(774 + 16, BY0 + 50, 'pos = 1000 → (1000 + 1) // 128 = 7 条已完成，缓冲按序全填：', 9.2,
        C_TXT, 'start', maxw=560, tag='bB:a1')
for i, v in enumerate(['0', '1', '2', '3', '4', '5', '6', '−1']):
    cxx = 774 + 16 + i * 42
    hit = (i < 7)
    lc.rect(cxx, BY0 + 60, 36, 22, C_KV_F if hit else '#ffffff', C_KV_S if hit else '#cbd5e1',
            rx=3, sw=1.1, dash=not hit)
    lc.text(cxx + 18, BY0 + 75, v, 9.5, C_KV_DEEP if hit else C_MUTE, 'middle', True,
            tag='bB:v%d' % i)
lc.text(774 + 380, BY0 + 75, '缓冲 = [0, 1, 2, 3, 4, 5, 6, −1]', 9.2, C_KV_DEEP, 'start', True,
        maxw=280, tag='bB:n')
lc.text(774 + 16, BY0 + 100, '省下的不是注意力算力，而是整个选择器（一份 132 B/条的索引缓存、'
                             '一次 25 万分的排序、以及选择带来的盲区）。', 8.8, '#155e75',
        'start', True, maxw=620, tag='bB:z0')
lc.text(774 + 16, BY0 + 120, '代价也诚实：HCA 层每 query 实看条目数确实是 CSA 的 15.257812 倍'
                             '——这就是它只占一半层的原因。', 8.8, '#334155', 'start', maxw=620,
        tag='bB:z1')

# ---------------- 页脚 ----------------
lc.text(MX, 828, '数字口径：1M 上下文、m = 4、m′ = 128、top-k = 512、c = 512、c^I = 128、索引器 64 头。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 846, '依据 DeepSeek-V4 技术报告 §2.3.2 的 Eq.(20)-(26) 与 §2.3.1（arXiv:2606.19348）；'
                 '两档的同构对照与乘加数由本章驱动脚本实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-csa-vs-hca.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
