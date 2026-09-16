#!/usr/bin/env python3
"""ch28 机制图 · attention sink 只进 softmax 的分母（ch28-fig-attn-sink）

claim：sink 只进 softmax 的**分母**：logits [2.0, 1.0] 加 sink=1.5 后两个权重同比例缩小、
总质量掉到 0.692804（sink 拿走 0.307196），而分子里没有对应的 value 项——所以每个头能把
自己的总注意力调成不等于 1、甚至接近 0。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  sink = 1.5：权重 = [0.50648, 0.186324]、总质量 = 0.692804、sink 拿走 = 0.307196
  sink = None(−inf)：权重 = [0.731059, 0.268941]、总质量 = 1.0；sink = 0.0：总质量 = 0.909969
  sink = 10.0：总质量 = 0.00045866（权重 0.00033531、0.00012335）；sink = 20.0：总质量 = 0.00000002
  分母 = Σ_j exp(z_j − m) + exp(sink − m)：分子里没有 sink 对应的 value 项 ⇒ 权重和 < 1

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 700
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_TAKEN_F, C_TAKEN_S = '#fef2f2', '#fca5a5'

# ---------------- 标题区 ----------------
lc.text(MX, 32, 'sink 只加在分母上：每个头可以少花注意力', 17, C_TXT, 'start', True, maxw=560,
        tag='title')
lc.text(MX, 56, '同一组 logits [2.0, 1.0]，只改 sink：两个权重的比例一点没变，'
                '但加起来（总质量）可以掉到接近 0', 10, C_MUTE, 'start', maxw=1040, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』注意力归一化那一格', 9, C_MUTE,
        'end', maxw=450, tag='l0:a')
lc.text(BXR, 64, '——同一张注意力图上多出来的那一项：分母里有个 exp(sink)', 9, C_MUTE, 'end',
        maxw=450, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_KV_F, C_KV_S, '总质量（这个头实际花掉的注意力）'),
                  (C_TAKEN_F, C_TAKEN_S, 'sink 拿走的那一份（分子里没有它）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=300, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18
lc.text(lgx, 87, 'logits 固定为 [2.0, 1.0]', 8.6, C_MUTE, 'start', maxw=200, tag='lg:fix')

# ================= 表 =================
CX = [(MX, 160, 'sink 取值'), (200, 150, '权重 1'), (350, 150, '权重 2'),
      (520, 520, '总质量（条长 = 权重和，满格 = 1.0）'), (1080, 200, '总质量数值'),
      (1290, 170, 'sink 拿走')]
HDR_Y = 112
lc.rect(MX, HDR_Y, BXR - MX, 26, '#e2e8f0', '#94a3b8', rx=4, sw=1.1)
for x, w, lab in CX:
    lc.text(x + 10, HDR_Y + 18, lab, 9.0, '#334155', 'start', True, maxw=w + 60, tag='h')

ROWS = [('−∞（无 sink）', '0.731059', '0.268941', 1.0, '1.00000000', '0.0'),
        ('0.0', '0.665241', '0.244728', 0.909969, '0.909969', '0.090031'),
        ('1.5', '0.50648', '0.186324', 0.692804, '0.692804', '0.307196'),
        ('10.0', '0.00033531', '0.00012335', 0.00045866, '0.00045866', '0.999541'),
        ('20.0', '0.00000002', '0.00000001', 0.00000002, '0.00000002', '0.99999998')]
RY0, RH, RGAP = 142, 44, 8
TRACK_X, TRACK_W = 530, 520
for i, (sink, w1, w2, mass, mass_s, taken) in enumerate(ROWS):
    ry = RY0 + i * (RH + RGAP)
    lc.rect(MX, ry, BXR - MX, RH, '#fff7ed' if i == 2 else '#ffffff', C_SOFT_S, rx=5, sw=1.0)
    lc.text(MX + 12, ry + 28, sink, 10.5, C_ACC_S if i == 2 else C_TXT, 'start', True,
            maxw=140, tag='r:s%d' % i)
    lc.text(210, ry + 28, w1, 9.6, '#334155', 'start', maxw=130, tag='r:w1%d' % i)
    lc.text(360, ry + 28, w2, 9.6, '#334155', 'start', maxw=130, tag='r:w2%d' % i)
    # 满格轨道 + 总质量填充 + sink 拿走
    lc.rect(TRACK_X, ry + 11, TRACK_W, 22, '#ffffff', C_SOFT_S, rx=3, sw=1.0)
    fw = max(mass * TRACK_W, 1.2 if mass > 0 else 0.0)
    lc.rect(TRACK_X, ry + 11, fw, 22, C_KV_F, C_KV_S, rx=3, sw=1.1)
    if mass < 1.0:
        lc.rect(TRACK_X + fw, ry + 11, TRACK_W - fw, 22, C_TAKEN_F, C_TAKEN_S, rx=3, sw=1.0,
                dash=True)
    lc.text(1090, ry + 28, mass_s, 9.6, C_KV_DEEP, 'start', True, maxw=180, tag='r:m%d' % i)
    lc.text(1300, ry + 28, taken, 9.6, C_MUTE if i < 2 else C_ACC_S, 'start', maxw=160,
            tag='r:t%d' % i)

# ================= 底：两句话 =================
BY0 = 420
lc.rect(MX, BY0, 900 - MX, 172, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 28, '机制上只有一句话：这一项只加在分母上', 11.5, C_KV_DEEP, 'start',
        True, maxw=520, tag='b:t')
lc.text(MX + 16, BY0 + 58, '分母 = Σ_j exp(z_j − m) + exp(sink − m)', 11, '#334155', 'start',
        True, maxw=820, tag='b:f')
lc.text(MX + 16, BY0 + 84, '分子里没有 sink 对应的 value 项 ⇒ 权重和 < 1（比例不变、'
                           '幅度被总质量缩放）。', 9.4, C_ACC_S, 'start', maxw=820, tag='b:n1')
lc.text(MX + 16, BY0 + 112, '相对比例完全不受 sink 影响：sink = 1.5 与无 sink 的比例都是 '
                            '[0.731059, 0.268941]；变的只是「加起来是多少」。', 9.4, '#334155',
        'start', maxw=820, tag='b:n2')
lc.text(MX + 16, BY0 + 140, '方向记牢（最容易记反的一格）：sink 越大 ⇒ 总质量越小；'
                            'sink → −∞ 才退回普通 softmax（总质量回到 1.0）。', 9.4, '#334155',
        'start', maxw=820, tag='b:n3')

lc.rect(920, BY0, BXR - 920, 172, '#fff7ed', C_ACC_S, rx=10, sw=1.4)
lc.text(936, BY0 + 28, '读表时抓住这一列', 11.5, C_ACC_S, 'start', True, maxw=400, tag='c:t')
CP = [('sink = −∞', '总质量 1.00000000 —— 退化成普通 softmax'),
      ('sink = 1.5', '总质量 0.692804，被拿走 0.307196'),
      ('sink = 10.0', '总质量 0.00045866 —— 这个头几乎不看任何东西'),
      ('sink = 20.0', '总质量 0.00000002 —— 趋向 0 的极限')]
for i, (a, b) in enumerate(CP):
    ty = BY0 + 58 + i * 30
    lc.text(936, ty, a, 9.4, C_ACC_S, 'start', True, maxw=120, tag='c:a%d' % i)
    lc.text(1076, ty, b, 9.4, '#334155', 'start', maxw=360, tag='c:b%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 626, '数字口径：玩具 logits = [2.0, 1.0]；sink 是每层一个可学习标量，'
                 '有没有真值取决于检查点权重（不能断言「只有某一档层有 sink」）。', 8.5,
        C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 644, '依据 DeepSeek-V4 技术报告 §2.3.3 的 Exp(z′_h) 进分母句（arXiv:2606.19348）；'
                 '逐档权重与总质量由本章算例实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')
lc.text(MX, 662, '读法：横着看每一行——前两列的比例始终没变，变的只有第三列那根条的长度。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-attn-sink.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
