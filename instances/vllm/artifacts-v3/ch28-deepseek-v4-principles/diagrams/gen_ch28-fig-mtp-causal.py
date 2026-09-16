#!/usr/bin/env python3
"""ch28 机制图 · MTP 的错位与因果（ch28-fig-mtp-causal）

claim：错位两格：输入喂 Emb(t_{i+1})、目标要 t_{i+2}——所以第 i 个位置预测时最远只能看到
自己喂进去的那个 token（扰动 Emb(t_4) 只有第 3 行会动，逐行变化 [0.0, 0.0, 0.0, 0.468328]），
而因果 TRM 保证改后面的输入动不了前面的输出。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  T = 5、k = 1：隐藏位置 [0, 1, 2, 3]（4 个）、有目标的位置 [(0, 2), (1, 3), (2, 4)]（3 个）
  扰动 Emb(t_4)（+= 1.0）⇒ 受影响的 h'^1 行 = [3]，逐行最大变化量 = [0.0, 0.0, 0.0, 0.468328]
  再扰动 TRM 输入的最后一行：逐位最大变化量 = [0.0, 0.0, 0.0, 1.166934]；
        反向对照（扰动第 0 行）= [1.006723, 0.003312, 0.002701, 0.000735]
  损失对齐：目标 token = [2, 1, 3]（取自序列位置 [2, 3, 4]）、L^1_MTP = 3.144246、
        λ/D = 0.3/1 ⇒ L_MTP = 0.943274

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 780
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_NONE_F = '#f1f5f9'

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '多预测一步的错位：喂 t_{i+1}、要 t_{i+2}', 17, C_TXT, 'start', True, maxw=520,
        tag='title')
lc.text(MX, 56, '输入错位 1 格、目标错位 2 格——于是第 i 个位置预测时，能看到的最远 token '
                '就是它自己喂进去的那个', 10, C_MUTE, 'start', maxw=1040, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』多预测一步那一档的对齐表', 9,
        C_MUTE, 'end', maxw=470, tag='l0:a')
lc.text(BXR, 64, '——骨架在另一张图上，这一张把它的对齐与因果摊开', 9, C_MUTE, 'end',
        maxw=430, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '位置 / 喂进去的 embedding'),
                  (C_KV_F, C_KV_S, '要预测的目标 token'),
                  ('#fff7ed', C_ACC_S, '扰动实验与损失')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=280, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= token 序列条 =================
TOK = ['0', '1', '2', '1', '3']
TCW, TGAP = 140, 20
TX0 = (W - (5 * TCW + 4 * TGAP)) / 2
lc.text(MX, 112, 'token 序列（T = 5）：[0, 1, 2, 1, 3]', 9.6, C_MUTE, 'start', maxw=400,
        tag='tk:l')
for k, v in enumerate(TOK):
    x = TX0 + k * (TCW + TGAP)
    lc.rect(x, 122, TCW, 44, C_GPU_F, C_GPU_S, rx=6, sw=1.3)
    lc.text(x + TCW / 2, 142, 't_%d' % k, 9.0, C_MUTE, 'middle', maxw=TCW, tag='tk:n')
    lc.text(x + TCW / 2, 160, '= %s' % v, 11, '#166534', 'middle', True, maxw=TCW, tag='tk:v')

# 错位尺
def tok_center(k):
    return TX0 + k * (TCW + TGAP) + TCW / 2

for (k1, k2, y, lab, col) in [(1, 2, 182, '错位 1 格：位置 i 喂的是 t_{i+1}', '#166534'),
                              (0, 2, 206, '错位 2 格：位置 i 要预测的是 t_{i+2}', C_KV_S)]:
    lc.parrow([(tok_center(k1), y), (tok_center(k2), y)], col, 1.6, marker=None)
    lc.seg(tok_center(k1), y - 6, tok_center(k1), y + 6, col, 1.6)
    lc.seg(tok_center(k2), y - 6, tok_center(k2), y + 6, col, 1.6)
    lc.text(tok_center(k2) + 12, y + 4, lab, 8.8, col, 'start', maxw=440, tag='tk:r')

# ================= 对齐表 =================
TH_Y, TH_H = 238, 28
RY0, RH, RGAP = 266, 56, 6
COLX = [(MX, 100, '位置 i'), (150, 360, '喂进去的 embedding（t_{i+1}，错位 1 格）'),
        (530, 360, '要预测的目标（t_{i+2}，错位 2 格）'), (910, 340, '算出来的 h^1_i'),
        (1270, BXR - 1270, '这一位有目标吗')]
lc.rect(MX, TH_Y, BXR - MX, TH_H, '#e2e8f0', '#94a3b8', rx=4, sw=1.1)
for x, w, lab in COLX:
    lc.text(x + 10, TH_Y + 19, lab, 9.0, '#334155', 'start', True, maxw=w + 40, tag='th')

ROWS = [(0, 'Emb(t_1)', 't_2 = 序列里的 2', 'h^1_0', '有'),
        (1, 'Emb(t_2)', 't_3 = 序列里的 1', 'h^1_1', '有'),
        (2, 'Emb(t_3)', 't_4 = 序列里的 3', 'h^1_2', '有'),
        (3, 'Emb(t_4)', '没有目标（记号的边界）', 'h^1_3', '没有')]
for i, (idx, emb, tgt, h, has) in enumerate(ROWS):
    ry = RY0 + i * (RH + RGAP)
    fill = '#ffffff' if has == '有' else C_NONE_F
    lc.rect(MX, ry, BXR - MX, RH, fill, C_SOFT_S, rx=5, sw=1.1)
    lc.text(MX + 12, ry + 34, '%d' % idx, 12, C_TXT, 'start', True, tag='rw:i%d' % i)
    lc.text(160, ry + 34, emb, 10.5, '#166534', 'start', True, maxw=340, tag='rw:e%d' % i)
    lc.text(540, ry + 34, tgt, 10.5, C_KV_DEEP if has == '有' else C_MUTE, 'start',
            maxw=340, tag='rw:t%d' % i)
    lc.text(920, ry + 34, h, 10.5, '#334155', 'start', True, maxw=320, tag='rw:h%d' % i)
    lc.text(1280, ry + 34, has, 10, C_KV_DEEP if has == '有' else C_MUTE, 'start',
            maxw=170, tag='rw:y%d' % i)
lc.text(MX, RY0 + 4 * (RH + RGAP) + 22, '隐藏位置比有目标的位置多 1 个：h^1_3 算出来了但没有'
                                        '目标——这是记号的边界，不是实现缺陷。', 8.8, C_MUTE,
        'start', maxw=1200, tag='rw:n')

# ================= 底：损失 + 扰动证据 =================
BY0 = 542
lc.rect(MX, BY0, 660 - MX, 176, '#fff7ed', C_ACC_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 28, '损失怎么对齐（同一个错位口径）', 11.5, C_ACC_S, 'start', True,
        maxw=440, tag='b:t')
BP = [('目标 token', '[2, 1, 3]（取自序列位置 [2, 3, 4]）'),
      ('第 1 档交叉熵', 'L^1_MTP = 3.144246（只对 3 个有目标的位置取平均）'),
      ('加权', '(λ/D)·Σ = (0.3/1) × 3.144246 = 0.943274'),
      ('超参', 'D = 1（只堆一档）、λ = 0.3')]
for i, (a, b) in enumerate(BP):
    ty = BY0 + 60 + i * 30
    lc.text(MX + 16, ty, a, 9.4, C_TXT, 'start', True, maxw=120, tag='b:a%d' % i)
    lc.text(MX + 148, ty, b, 9.4, '#334155', 'start', maxw=460, tag='b:b%d' % i)

lc.rect(680, BY0, BXR - 680, 176, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(696, BY0 + 28, '因果性的两组扰动证据', 11.5, C_KV_DEEP, 'start', True, maxw=420,
        tag='c:t')
CE = [('① 扰动最后一个 embedding', '受影响的 h′^1 行 = [3]',
       '逐行最大变化量 = [0.0, 0.0, 0.0, 0.468328]'),
      ('② 扰动 TRM 输入的最后一行', '位置 0..2 纹丝不动',
       '逐位最大变化量 = [0.0, 0.0, 0.0, 1.166934]'),
      ('③ 反向对照（扰动第 0 行）', '后面全都会变（因果是只看前面，不是互不影响）',
       '逐位最大变化量 = [1.006723, 0.003312, 0.002701, 0.000735]')]
for i, (a, b, c) in enumerate(CE):
    ty = BY0 + 58 + i * 38
    lc.text(696, ty, a, 9.6, C_KV_DEEP, 'start', True, maxw=240, tag='c:a%d' % i)
    lc.text(696, ty + 17, b, 9.0, C_MUTE, 'start', maxw=330, tag='c:b%d' % i)
    lc.text(1040, ty, c, 9.2, '#334155', 'start', maxw=420, tag='c:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 742, '数字口径：玩具 T = 5、k = 1、d = 2、词表 4；config 口径只堆 1 档、λ = 0.3。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 760, '依据 DeepSeek-V3 技术报告 §2.2 的 Eq.(21)-(25)（arXiv:2412.19437）；'
                 '对齐表、损失与三组扰动数值由本章算例实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-mtp-causal.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
