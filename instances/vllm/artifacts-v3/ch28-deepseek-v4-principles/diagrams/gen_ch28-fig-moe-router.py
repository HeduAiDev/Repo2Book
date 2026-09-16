#!/usr/bin/env python3
"""ch28 机制图 · 路由三步：换激活 / bias 只进选择 / 权重从无偏分 gather（ch28-fig-moe-router）

claim：路由三步各管一件事：Sqrt(Softplus) 换掉 Sigmoid（大 logits 处 2.828486 vs 0.999665，
不再压平到 1）、bias 只进选择（选中集从 [2, 1] 变成 [2, 3]）、权重仍从**无偏**分数 gather
再归一（[0.66152, 0.33848]；拿带偏分数当权重会压平成 [0.594004, 0.405996]）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  gate logits = [-2.0, 0.5, 3.0, 0.2]、bias = [0.0, -0.1, 0.0, 0.3]
  V3 口径 Sigmoid = [0.119203, 0.622459, 0.952574, 0.549834]
  V4 口径 Sqrt(Softplus) = [0.35627, 0.986953, 1.74602, 0.893386]
  logit = 8.0 时：Sigmoid = 0.999665 vs Sqrt(Softplus) = 2.828486
  不带 bias 的选择 top-2 = 专家 [2, 1]；加 bias 后 top-2 = 专家 [2, 3]；
        权重仍从无偏分数 gather = [0.66152, 0.33848]
  对照（拿带偏分数当权重）= [0.594004, 0.405996]；两集合不同 = True

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 730
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_ON_F, C_ON_S = '#dcfce7', C_GPU_S
C_OFF_F, C_OFF_S = '#f1f5f9', '#cbd5e1'

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '路由三步：换激活、bias 只进选择、权重仍从无偏分取', 17, C_TXT, 'start', True,
        maxw=620, tag='title')
lc.text(MX, 56, 'V4 相对 V3 换了亲和分与均衡机制；三步各管一件事，谁也不越界', 10, C_MUTE,
        'start', maxw=900, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』专家池的路由那一格', 9, C_MUTE,
        'end', maxw=450, tag='l0:a')
lc.text(BXR, 64, '——专家池怎么切在另一张图上，这一张讲池子里怎么挑', 9, C_MUTE, 'end',
        maxw=430, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_ON_F, C_ON_S, '被选中的专家'), (C_OFF_F, C_OFF_S, '没被选中的专家'),
                  ('#fff7ed', C_ACC_S, '换掉 / 对照的那一笔')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=260, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 三拍 =================
PW, PGAP = 452, 18
PX = [MX + i * (PW + PGAP) for i in range(3)]
PY0, PY1 = 116, 480

STEPS = [
    ('① 亲和分：换掉 Sigmoid',
     [('t', '4 个专家的 gate logits = [−2.0, 0.5, 3.0, 0.2]', C_TXT, True, 0),
      ('t', 'V3 口径 Sigmoid =', '#334155', True, 0),
      ('v', '[0.119203, 0.622459, 0.952574, 0.549834]', '#334155', False, 0),
      ('t', 'V4 口径 Sqrt(Softplus) =', C_GPU_S, True, 0),
      ('v', '[0.35627, 0.986953, 1.74602, 0.893386]', '#166534', True, 0),
      ('gap', '', '', False, 0),
      ('t', '为什么换（一条数字对比）：', C_ACC_S, True, 0),
      ('t', 'logit = 8.0 时 Sigmoid = 0.999665', '#334155', False, 0),
      ('t', '——已经压平到 1 附近，区分度被吃光；', C_MUTE, False, 0),
      ('t', 'Sqrt(Softplus) = 2.828486（不饱和上界，', '#334155', False, 0),
      ('t', '同样的 logits 仍在继续增长）。', '#334155', False, 0)]),
    ('② 选择：bias 只进选择',
     [('t', 'bias = [0.0, −0.1, 0.0, 0.3]（每个专家一个）', '#334155', True, 0),
      ('t', '不带 bias 的 top-2 = 专家 [2, 1]', '#334155', True, 0),
      ('gap', '', '', False, 0),
      ('t', 'scores + bias =', C_ACC_S, True, 0),
      ('v', '[0.35627, 0.886953, 1.74602, 1.193386]', '#7c2d12', False, 0),
      ('t', '带 bias 的 top-2 = 专家 [2, 3]', C_ACC_S, True, 0),
      ('t', '——被换进来的是专家 3：bias 真的改动了', C_MUTE, False, 0),
      ('t', '选择（排序用的是加了偏置的分数）。', C_MUTE, False, 0),
      ('gap', '', '', False, 0),
      ('t', '非平凡性核对', C_GPU_S, True, 0),
      ('t', '无偏选中的是 {1, 2}、带偏选中的是 {2, 3}', '#334155', False, 0),
      ('t', '两个集合不同 = True（本例确实落在', '#334155', False, 0),
      ('t', '「bias 改变选择」的那条分支上）。', '#334155', False, 0)]),
    ('③ 权重：仍从无偏分数取',
     [('t', '选中 [2, 3] 之后，权重怎么来？', '#334155', True, 0),
      ('t', '从**无偏** scores gather 再归一：', C_GPU_S, True, 0),
      ('v', '[0.66152, 0.33848]', '#166534', True, 0),
      ('gap', '', '', False, 0),
      ('t', '错误做法（顺手拿带偏分数当权重）：', C_ACC_S, True, 0),
      ('v', '[0.594004, 0.405996]', '#7c2d12', False, 0),
      ('t', '→ 更平：最大权重从 0.66152 掉到 0.594004。', C_ACC_S, False, 0),
      ('gap', '', '', False, 0),
      ('t', '为什么必须分开：偏置被调大时（config 口径', C_MUTE, False, 0),
      ('t', 'bias ≈ 8.08 接近均匀），带偏分数会趋同；', C_MUTE, False, 0),
      ('t', '若拿它当权重，归一后就逼近均匀分布——', C_MUTE, False, 0),
      ('t', '权重不再携带「这个专家多合适」的信息。', C_MUTE, False, 0)]),
]
for px, (title, rows) in zip(PX, STEPS):
    lc.rect(px, PY0, PW, PY1 - PY0, '#ffffff', C_GPU_S, rx=10, sw=1.5)
    lc.text(px + 16, PY0 + 30, title, 12.5, '#166534', 'start', True, maxw=PW - 32, tag='st:t')
    ry = PY0 + 62
    for kind, s, col, bold, _ in rows:
        if kind == 'gap':
            ry += 12
            continue
        fs = 10.6 if kind == 'v' else 9.4
        lc.text(px + 16, ry, s.replace('**', ''), fs, col, 'start', bold, maxw=PW - 32,
                tag='st:%s' % s[:8])
        ry += 14 if kind == 'v' else 21

# ================= 底：三步的一句话总结 =================
BY0 = 500
lc.rect(MX, BY0, BXR - MX, 148, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
lc.text(MX + 16, BY0 + 28, '三步各管一件事——把它们混起来就会出问题', 11.5, C_TXT, 'start',
        True, maxw=520, tag='b:t')
B = [('① 分数形状', 'Sqrt(Softplus) 严格单调且没有上界（Sigmoid 的上界是 1）',
      '让大 logits 处的专家之间仍然分得出高下'),
     ('② 谁来排序', '选择用 scores + bias；这一步只改「谁被选中」',
      '均衡的目标是「别让某些专家饿死」，那就只该动选择'),
     ('③ 谁来定量', '权重从无偏 scores 取；bias 完全不进权重',
      '否则均衡偏置会把「谁更合适」这件事从权重里抹掉')]
for i, (a, b, c) in enumerate(B):
    ty = BY0 + 60 + i * 28
    lc.text(MX + 16, ty, a, 9.6, C_GPU_S, 'start', True, maxw=120, tag='b:a%d' % i)
    lc.text(MX + 140, ty, b, 9.6, '#334155', 'start', maxw=620, tag='b:b%d' % i)
    lc.text(MX + 780, ty, c, 8.8, C_MUTE, 'start', maxw=620, tag='b:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 676, '数字口径：玩具 4 个专家、topk = 2、bias = [0.0, −0.1, 0.0, 0.3]；'
                 'config 口径 n_routed_experts = 256、num_experts_per_tok = 6、'
                 'scoring_func = sqrtsoftplus。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 694, '依据 DeepSeek-V4 技术报告 §2.1（arXiv:2606.19348）；'
                 '三步的逐项数值由本章算例实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')
lc.text(MX, 712, '读法：三拍从左到右读，每一步只看它自己那一列数字——'
                 '②的选中集与③的权重不是同一件事。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-moe-router.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
