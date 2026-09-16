#!/usr/bin/env python3
"""ch28 机制图 · MTP：顺序多 token 预测的一档与损失（ch28-fig-mtp-flow）

claim：MTP 一档＝『上一档隐状态 + 第 i+k 个 token 的 embedding』拼接 → 因果 TRM →
共享输出头；输入错位 k 格、目标错位 k+1 格，因此严格不偷看；V4 原样沿用（D = 1、λ = 0.3）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  mtp_alignment(T=5, k=1)：隐藏位置 [0, 1, 2, 3]、目标 [(0, 2), (1, 3), (2, 4)]（隐藏比目标多 1 个）
  i=0 的拼接 = [1.26491, 0.632455, 1.414212, 0.0]；h'^1_0 = [0.900367, 0.881912]
  i=0 的预测分布 = [0.072536, 0.069367, 0.856462, 0.001635]（目标 token = 2）
  L^1_MTP = 3.144246 → L_MTP = (0.3/1) × 3.144246 = 0.943274
  扰动 Emb(t_4)：受影响 h'^1 行 = [3]，逐行变化量 [0.0, 0.0, 0.0, 0.468328]
  顺序 vs 对照：逐位相同 = False；扰动 h^1_0 后顺序版变化 [0.001348, 0.001532, 0.000124,
  0.000455, 0.000112, 0.000203]、对照版全 0

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂，品红 = 出口/共享件）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 840
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_SAM_S, C_SAM_F = lc.C_SAM_S, lc.C_SAM_F

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="mg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SAM_S}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '顺手多预测一步：一档 MTP 的数据流', 17, C_TXT, 'start', True, maxw=680,
        tag='title')
lc.text(MX, 56, '上一档的隐状态与第 i+k 个 token 的 embedding 各自归一后拼接，过一个保持完整'
                '因果链的块，再用「与主模型共享」的输出头吐一个分布', 10, C_MUTE, 'start',
        maxw=1100, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』尾部的分叉', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——与出口动线接在一起（先拷给 MTP、再过 hc_head）', 9, C_MUTE, 'end',
        maxw=430, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '主干残差 / 因果块'),
                  (C_KV_F, C_KV_S, '拼接与投影'),
                  (C_SAM_F, C_SAM_S, '与主模型共享的输出头')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ---------------- ① 数据流链 ----------------
CY, CH = 132, 74
BOXW = 220
CHAIN = [('主干尾部残差', 'hc_head 之前的展平态', '（4 条流的信息都在）', C_GPU_F, C_GPU_S),
         ('拼接（Eq.21）', 'RMS(h^0_i) ‖ RMS(Emb(t_{i+1}))', '每个位置 2d → 4 维（玩具）',
          C_KV_F, C_KV_S),
         ('M_1 投影', '2d → d 的拼接投影', 'embedding 与输出头都与主模型共享', C_KV_F, C_KV_S),
         ('TRM_1（Eq.22）', '因果 Transformer 块', '保持完整因果链：位置 i 看不到 j > i',
          C_GPU_F, C_GPU_S),
         ('共享输出头（Eq.23）', 'OutHead（与主模型共享）', '给出对 t_{i+2} 的 logits',
          C_SAM_F, C_SAM_S),
         ('预测 t_{i+2}', '输入错位 1 格、目标错位 2 格', 'i=0 的分布最大值落在 token 2 上',
          C_SAM_F, C_SAM_S)]
for i, (t1, t2, t3, f, s) in enumerate(CHAIN):
    x = MX + i * (BOXW + 20)
    lc.rect(x, CY, BOXW, CH, f, s, rx=8, sw=1.5)
    lc.text(x + 12, CY + 24, t1, 10.5, C_TXT, 'start', True, maxw=BOXW - 22, tag='ch:t%d' % i)
    lc.text(x + 12, CY + 44, t2, 8.6, '#334155', 'start', maxw=BOXW - 22, tag='ch:s%d' % i)
    lc.text(x + 12, CY + 62, t3, 8.4, C_MUTE, 'start', maxw=BOXW - 22, tag='ch:u%d' % i)
    if i < 5:
        lc.seg(x + BOXW, CY + CH / 2, x + BOXW + 18, CY + CH / 2, C_SAM_S if i >= 3 else '#64748b',
               2.2, marker='mg' if i >= 3 else 'gy')

# ---------------- ② 错位标尺 ----------------
RY, RH = 236, 30
lc.text(MX, RY - 10, '错位口径：输入错位 k = 1 格，目标错位 k + 1 = 2 格', 10.5, C_TXT, 'start',
        True, maxw=420, tag='rl:t')
CELL = 86
RX0 = MX + 150
for i in range(5):
    lc.rect(RX0 + i * CELL + 2, RY, CELL - 4, RH, C_GPU_F, C_GPU_S, rx=3, sw=1.2)
    lc.text(RX0 + (i + 0.5) * CELL, RY + 20, 't_%d' % i, 10, C_TXT, 'middle', tag='rl:c%d' % i)
lc.text(RX0 - 8, RY + 20, 'token 序列', 9.5, C_TXT, 'end', True, tag='rl:l0')
for i in range(4):
    lc.rect(RX0 + i * CELL + 2, RY + 44, CELL - 4, RH, C_KV_F, C_KV_S, rx=3, sw=1.2)
    lc.text(RX0 + (i + 0.5) * CELL, RY + 64, 'h_%d' % i, 10, C_TXT, 'middle', tag='rl:h%d' % i)
lc.text(RX0 - 8, RY + 64, '隐藏位置（4 个）', 9.5, C_KV_DEEP, 'end', True, tag='rl:l1')
for i in range(3):
    lc.rect(RX0 + (i + 2) * CELL + 2, RY + 88, CELL - 4, RH, C_SAM_F, C_SAM_S, rx=3, sw=1.2)
    lc.text(RX0 + (i + 2.5) * CELL, RY + 108, 't_%d' % (i + 2), 10, C_TXT, 'middle', tag='rl:g%d' % i)
lc.text(RX0 - 8, RY + 108, '有目标的位置（3 个）', 9.5, C_SAM_S, 'end', True, tag='rl:l2')
lc.text(RX0 + 5 * CELL + 14, RY + 20, '位置 i=0 吃 (h_0, Emb(t_1)) → 预测 t_2', 9.2, '#334155',
        'start', maxw=300, tag='rl:n0')
lc.text(RX0 + 5 * CELL + 14, RY + 38, '隐藏位置比目标位置多 1 个：最后一个 h 没有目标', 8.8,
        C_MUTE, 'start', maxw=320, tag='rl:n1')
lc.text(RX0 + 5 * CELL + 14, RY + 56, '（Eq.24 的区间比 Eq.22 短一格）', 8.8, C_MUTE, 'start',
        maxw=320, tag='rl:n2')

# ---------------- ③ 一档走完的读数 ----------------
NY = 386
lc.rect(MX, NY, 900, 168, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, NY + 26, '一档走完（T = 5、d = 2、V = 4、D = 1、λ = 0.3）', 11.5, C_TXT,
        'start', True, maxw=460, tag='num:t')
NUMS = [('Eq.21 拼接', 'i=0 的拼接 = [1.26491, 0.632455, 1.414212, 0.0]（前半 RMS(h)、后半 RMS(Emb)）'),
        ('Eq.21 投影', 'h\'^1_0 = [0.900367, 0.881912]'),
        ('Eq.23 输出', 'i=0 的预测分布 = [0.072536, 0.069367, 0.856462, 0.001635]'
                       '（目标 token = 2，最大值恰好落在它上面）'),
        ('Eq.24 损失', 'L^1_MTP = 3.144246（只对 3 个有目标的位置取平均）'),
        ('Eq.25 加权', 'L_MTP = (λ/D) × 3.144246 = (0.3/1) × 3.144246 = 0.943274')]
for i, (a, b) in enumerate(NUMS):
    ty = NY + 54 + i * 22
    lc.text(MX + 16, ty, a, 9.2, C_TXT, 'start', True, tag='num:a%d' % i)
    lc.text(MX + 120, ty, b, 9.2, '#334155', 'start', maxw=760, tag='num:b%d' % i)

lc.rect(960, NY, BXR - 960, 168, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(960 + 16, NY + 26, '推理期两条路', 11.5, C_KV_DEEP, 'start', True, maxw=340, tag='inf:t')
INF = [('直接丢掉', '官方参考实现有 MTP 块，但推理脚本不用它'),
       ('留着当起草器', 'pin 把它当投机解码的起草器：接的是主干 hc_head 之前的展平残差，'
                    '把单流 unsqueeze 成 hc_mult 条流')]
for i, (a, b) in enumerate(INF):
    ty = NY + 60 + i * 52
    lc.text(960 + 16, ty, a, 10, C_TXT, 'start', True, tag='inf:a%d' % i)
    lc.text(960 + 16, ty + 20, b, 8.8, '#334155', 'start', maxw=450, tag='inf:b%d' % i)
lc.text(960 + 16, NY + 152, 'config 口径：num_nextn_predict_layers = 1（与论文 D = 1 一致）。', 8.6,
        C_MUTE, 'start', maxw=460, tag='inf:c')

# ---------------- ④ 因果证据 + 顺序 vs 并行 ----------------
BY0 = 570
lc.rect(MX, BY0, 700, 190, '#ffffff', C_GPU_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 26, '两条独立的因果证据（都在实测里）', 11.5, C_TXT, 'start', True,
        maxw=440, tag='ev:t')
EV = [('① 输入侧错位', '扰动 Emb(t_4)（最后一格）+1.0',
       '受影响的 h\'^1 行 = [3]；逐行最大变化量 = [0.0, 0.0, 0.0, 0.468328]'),
      ('② TRM 内部因果', '扰动 h\'^1 的最后一行 +1.0',
       '逐位最大变化量 = [0.0, 0.0, 0.0, 1.166934]（位置 0..2 纹丝不动）')]
for i, (a, b, c) in enumerate(EV):
    ty = BY0 + 58 + i * 58
    lc.text(MX + 16, ty, a, 10, C_TXT, 'start', True, tag='ev:a%d' % i)
    lc.text(MX + 150, ty, b, 9.2, '#334155', 'start', maxw=520, tag='ev:b%d' % i)
    lc.text(MX + 16, ty + 20, c, 9.2, C_KV_DEEP, 'start', maxw=660, tag='ev:c%d' % i)
lc.text(MX + 16, BY0 + 168, '两句相乘 → 位置 i 的预测只依赖 ≤ i + k 的 token，'
                            '而目标是 i + k + 1 → 严格不偷看。', 8.8, '#334155', 'start',
        maxw=660, tag='ev:n')

lc.rect(770, BY0, BXR - 770, 190, C_SAM_F, C_SAM_S, rx=10, sw=1.4)
lc.text(770 + 16, BY0 + 26, '『顺序』不是修辞：第 2 档真的吃第 1 档的输出', 11.5, C_SAM_S,
        'start', True, maxw=520, tag='seq:t')
SEQ = [('顺序版', '第 2 档输入 = h^1（第 1 档的输出）', ''),
       ('对照版', '第 2 档输入 = h\'^1（不接第 1 档的 TRM 输出）', '')]
for i, (a, b, _c) in enumerate(SEQ):
    ty = BY0 + 58 + i * 30
    lc.text(770 + 16, ty, a, 10, C_TXT, 'start', True, tag='seq:a%d' % i)
    lc.text(770 + 90, ty, b, 9.2, '#334155', 'start', maxw=570, tag='seq:b%d' % i)
lc.text(770 + 16, BY0 + 128, '两种第 2 档输出逐位相同 = False（不接上一档会改变结果）。', 9.2,
        '#334155', 'start', True, maxw=660, tag='seq:c0')
lc.text(770 + 16, BY0 + 148, '扰动 h^1_0 后：顺序版变化 [0.001348, 0.001532, 0.000124, '
                             '0.000455, 0.000112, 0.000203]（非 0）', 9, C_SAM_S, 'start',
        maxw=660, tag='seq:c1')
lc.text(770 + 16, BY0 + 166, '对照版恒 0——它压根不读第 1 档。D = 1 时这条链只有一格。', 9,
        C_MUTE, 'start', maxw=660, tag='seq:c2')

# ---------------- 页脚 ----------------
lc.text(MX, 782, '数字口径：玩具 T = 5、d = 2、V = 4、D = 1、λ = 0.3、k = 1；'
                 'token 序列 = [0, 1, 2, 1, 3]。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 800, '依据 DeepSeek-V3 技术报告 §2.2 的 Eq.(21)-(25)（arXiv:2412.19437）'
                 '与 V4 技术报告 §2.1 的 without modification（arXiv:2606.19348）；'
                 '全部数值由本章驱动脚本实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-mtp-flow.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
