#!/usr/bin/env python3
"""ch28 机制图 · hash 路由：按 token id 查表派单（ch28-fig-hash-route）

claim：hash 路由＝按 token id 查一张 [词表, 每 token 选几个] 的整数表：选择与
hidden state 和 gate 分数完全无关（也不需要均衡损失），但权重仍从当层无偏亲合分 gather。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  玩具表 tid2eid = [[3, 1], [0, 2], [2, 3], [1, 0], [3, 0], [1, 2]]（vocab 6 × topk 2）
  token 4 → [3, 0]（权重 [0.818182, 0.181818]）；token 1 → [0, 2]（[0.65, 0.35]）；
  token 5 → [1, 2]（[0.25, 0.75]）
  同一批 token 走 gate 打分的 top-2 = [[1, 3], [0, 2], [2, 3]]（与查表不同）
  config：num_hash_layers = 3、vocab_size = 129280 → 表 [129280, 6]

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂内的 FFN 段）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 600
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="wb" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '最前 3 层不挑：按 token id 直接查表派单', 17, C_TXT, 'start', True, maxw=680,
        tag='title')
lc.text(MX, 56, '表是一张 [词表, 每 token 选几个] 的整数表，随权重一起装载，运行时只做一次 gather'
                '——没有路由参数、也没有均衡损失', 10, C_MUTE, 'start', maxw=1100, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』最前三层的 FFN 派单方式', 9,
        C_MUTE, 'end', maxw=460, tag='l0:a')
lc.text(BXR, 64, '——与上一张图的『挑选』并排看，是同一层 FFN 的两种派单', 9, C_MUTE, 'end',
        maxw=470, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '被选中的专家（表说了算）'),
                  (C_KV_F, C_KV_S, '查表 / 权重（亲合分现算）'),
                  ('#f8fafc', '#cbd5e1', '被划掉的门控打分')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ---------------- ① token id 列 ----------------
TX, TY = MX + 10, 176
lc.text(TX, TY - 14, '输入 token（只用到 id）', 10, C_TXT, 'start', True, maxw=240, tag='tk:t')
IDS = [4, 1, 5]
for i, tid in enumerate(IDS):
    cy = TY + i * 52
    lc.rect(TX, cy, 130, 38, C_GPU_F, C_GPU_S, rx=6, sw=1.5)
    lc.text(TX + 65, cy + 24, 'token id = %d' % tid, 10.5, C_TXT, 'middle', True, maxw=124,
            tag='tk:%d' % i)
# ---------------- ② 表 ----------------
BX, BY = 260, 140
lc.text(BX, BY - 14, '整数表 tid2eid（玩具：6 行 × 2 列）', 10, C_TXT, 'start', True, maxw=300,
        tag='tb:t')
TAB = [[3, 1], [0, 2], [2, 3], [1, 0], [3, 0], [1, 2]]
HIT = {4, 1, 5}
for r in range(6):
    for c in range(2):
        cxx = BX + c * 60
        cyy = BY + r * 34
        hit = r in HIT
        lc.rect(cxx, cyy, 56, 28, C_KV_F if hit else '#ffffff', C_KV_S if hit else '#cbd5e1',
                rx=3, sw=1.3)
        lc.text(cxx + 28, cyy + 19, str(TAB[r][c]), 10.5, C_KV_DEEP if hit else C_MUTE, 'middle',
                True, tag='tb:%d%d' % (r, c))
    lc.text(BX + 124, BY + r * 34 + 19, 'row %d' % r, 9, C_KV_DEEP if r in HIT else C_MUTE,
            'start', r in HIT, tag='tb:r%d' % r)
lc.text(BX - 4, BY + 6 * 34 + 12, '高亮的行 = 本次三个 token 命中的行', 8.8, C_KV_DEEP, 'start',
        maxw=340, tag='tb:n')

# ---------------- ③ 派单结果 ----------------
RX = 520
lc.text(RX, TY - 14, '派单结果（专家号 + 现算的权重）', 10, C_TXT, 'start', True, maxw=320,
        tag='rs:t')
RES = [(4, 'row 4 = [3, 0]', '[3, 0]', '[0.818182, 0.181818]'),
       (1, 'row 1 = [0, 2]', '[0, 2]', '[0.65, 0.35]'),
       (5, 'row 5 = [1, 2]', '[1, 2]', '[0.25, 0.75]')]
for i, (tid, a, b, c) in enumerate(RES):
    cy = TY + i * 52
    lc.text(RX, cy + 14, a, 9.2, C_KV_DEEP, 'start', tag='rs:a%d' % i)
    lc.rect(RX, cy + 20, 150, 34, C_GPU_F, C_GPU_S, rx=6, sw=1.4)
    lc.text(RX + 75, cy + 43, '专家 ' + b, 10.5, C_TXT, 'middle', True, maxw=144, tag='rs:b%d' % i)
    lc.text(RX + 160, cy + 28, '权重 ' + c, 9.8, '#334155', 'start', maxw=200, tag='rs:c%d' % i)
    lc.text(RX + 160, cy + 46, '（从这层的无偏亲合分 gather）', 8.6, C_MUTE, 'start', maxw=200,
            tag='rs:d%d' % i)

# 两条流程箭头：token → 表 → 派单结果
lc.parrow([(TX + 130, TY + 78), (BX - 6, BY + 2 * 34 + 14)], C_KV_S, 2.2, marker='wb')
lc.text((TX + 130 + BX) / 2 - 20, TY + 62, '查表：行号 = id', 9, C_KV_DEEP, 'middle', True,
        maxw=170, tag='fl:a')
lc.parrow([(BX + 200, BY + 4 * 34 + 14), (RX - 6, TY + 52 + 36)], C_GPU_S, 2.2, marker='gn')
lc.text((BX + 200 + RX) / 2 + 10, BY + 4 * 34 + 4, '一次 gather', 9, C_GPU_S, 'middle', True,
        maxw=170, tag='fl:b')

# ---------------- ④ 对照：划掉的门控打分 ----------------
GX, GY = 900, 148
lc.rect(GX, GY, BXR - GX, 168, '#f8fafc', '#cbd5e1', rx=10, sw=1.3, dash=True)
lc.text(GX + 16, GY + 26, '这一层不走的另一条路（对照）', 11, C_TXT, 'start', True, maxw=380,
        tag='gt:t')
lc.text(GX + 16, GY + 50, '若走 gate 打分取 top-2，同一批 token 抽到的是：', 9.2, '#334155',
        'start', maxw=520, tag='gt:s')
lc.text(GX + 16, GY + 70, '[[1, 3], [0, 2], [2, 3]]', 10.5, '#334155', 'start', True, maxw=520,
        tag='gt:s2')
lc.text(GX + 16, GY + 90, '与查表结果 [[3, 0], [0, 2], [1, 2]] 不同 → hash 的『选谁』'
                          '与 gate 分数完全无关。', 9.2, C_KV_DEEP, 'start', True, maxw=520,
        tag='gt:s3')
lc.seg(GX + 30, GY + 112, GX + 250, GY + 150, lc.C_ABORT, 2.6)
lc.seg(GX + 30, GY + 150, GX + 250, GY + 112, lc.C_ABORT, 2.6)
lc.text(GX + 264, GY + 138, '分数不参与选择', 10, '#334155', 'start', True, maxw=200, tag='gt:x')
lc.text(GX + 264, GY + 156, '（但仍决定权重）', 8.8, C_MUTE, 'start', maxw=200, tag='gt:x2')

# ---------------- ⑤ 权限边界说明 ----------------
MY0 = 372
lc.rect(MX, MY0, 700, 160, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, MY0 + 26, '表只决定『选谁』，给多少权重仍现算', 11.5, C_KV_DEEP, 'start',
        True, maxw=420, tag='pw:t')
lc.text(MX + 16, MY0 + 52, '权重从这一层的无偏亲合分 gather 再归一：token 4 的 '
                          '[0.818182, 0.181818] 来自 scores 两个分量之比。', 9.2, '#334155',
        'start', maxw=640, tag='pw:s')
lc.text(MX + 16, MY0 + 78, '所以这几层并不是『没有 gate』——gate 的分数不参与选择，'
                          '但依然决定每条被选中的专家拿多少份额。', 9.2, C_KV_DEEP, 'start', True,
        maxw=640, tag='pw:s2')
lc.text(MX + 16, MY0 + 108, '均衡性来自表的构造方式（Hash Layers：均衡指派与随机指派表现相近，'
                           '都不需要在目标函数里加均衡项），不是来自运行时机制——', 9, C_MUTE,
        'start', maxw=660, tag='pw:s3')
lc.text(MX + 16, MY0 + 128, '这也是它比路由方案少一整项损失的算术原因。', 9, C_MUTE, 'start',
        maxw=660, tag='pw:s4')
lc.text(MX + 16, MY0 + 150, '对照上一张图的 bias 口径：bias 改的是排序（仍由分数决定），'
                           'hash 把这个排序整个换掉。', 8.8, lc.C_FAINT, 'start', maxw=660,
        tag='pw:s5')

# ---------------- ⑥ config 口径 ----------------
lc.rect(774, MY0, BXR - 774, 160, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(774 + 16, MY0 + 26, 'config 口径：真实规模长什么样', 11.5, C_TXT, 'start', True,
        maxw=420, tag='cf:t')
CFG = [('生效范围', 'num_hash_layers = 3（最前 3 层）'),
       ('表的形状', 'vocab_size = 129280 × 每 token 挑 6 → [129280, 6]'),
       ('运行时的动作', '每个 token 一次 6 元素 gather（表随权重一起装载）'),
       ('论文没说的', '为什么只有前 3 层——本书不替它编理由')]
for i, (a, b) in enumerate(CFG):
    ty = MY0 + 58 + i * 28
    lc.text(774 + 16, ty, a, 9.5, C_TXT, 'start', True, tag='cf:a%d' % i)
    lc.text(774 + 140, ty, b, 9.5, C_KV_DEEP, 'start', True, maxw=530, tag='cf:b%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 556, '数字口径：玩具 vocab = 6、topk = 2（真实 topk = 6）；三个 token 的派单与'
                 '权重由本章驱动脚本实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 574, '依据 Hash Layers 论文（arXiv:2106.04426）与 DeepSeek-V4 技术报告 §2.1 '
                 '（arXiv:2606.19348）的 hash 路由句。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-hash-route.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
