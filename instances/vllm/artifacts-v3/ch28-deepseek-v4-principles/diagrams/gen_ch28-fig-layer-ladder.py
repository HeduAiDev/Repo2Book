#!/usr/bin/env python3
"""ch28 机制图 · 三类层与 4/128 交替排布（ch28-fig-layer-ladder）

claim：一张 compress_ratios 表决定每层付哪笔账：0/4/128 三类、中段严格交替
（21 层 c4a 与 20 层 c128a），而每档的『每 query 实看条目数』（128 / 512 / 7812）
就是 FLOPs 账的分档来源。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  44 项 compress_ratios（前两项 0、中段 4/128 交替、末项 0）→ 主干 43 层分类
  {swaonly 2, c4a 21, c128a 20}
  每档每 query 实看条目数：纯滑窗 128（常数窗）、c4a 512（top-k）、c128a 7812（=1000000//128）
  逐层摊销：CSA 179.0 B / HCA 4.5625 B / 纯滑窗 0 B
  短上下文倒挂：1K 上下文时 c4a 候选 250 < topk 512 → 全选（250 条），c128a 只有 7 条

配色走 book/cartography/l0_common.py 的角色常量（GPU 执行臂绿为主、KV 青标缓存账）。
坐标全部由常量与循环计算；文本全 esc()。层号只标首尾与三类切换处（防文字过密）。
"""
import sys
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 900
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F = lc.C_KV_S, lc.C_KV_F
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
# 三种牌子：同一绿色系内做深浅（0 最浅 = 无压缩机、4 中、128 最深）
PLATE = {'0': ('#dcfce7', '#4ade80', '#166534'),
         '4': (lc.C_GPU_F, C_GPU_S, '#14532d'),
         '128': ('#166534', '#166534', '#ffffff')}

EXTRA_DEFS = ('<defs>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
              '</defs>')

# ---------------- 电梯几何（43 层，横放一行一层） ----------------
LDY0, RH, STEP = 160, 11.0, 12.8
LAD_X, LAD_W = 96, 116
MID_X = 262                       # 电梯右侧的档位标注列
NR = 43

RATIOS = [0, 0] + ([4, 128] * 21)[:41] + [0]      # 44 项（末项 0 = 多 token 预测槽）
assert len(RATIOS) == 44
MAIN = RATIOS[:43]                                  # 主干 43 层
KIND = ['0' if r == 0 else ('4' if r == 4 else '128') for r in MAIN]
assert {k: KIND.count(k) for k in ('0', '4', '128')} == {'0': 2, '4': 21, '128': 20}
LAD_END = LDY0 + (NR - 1) * STEP + RH

# ---------------- 标题区 ----------------
lc.text(MX, 32, '一张 compress_ratios 表，决定 43 层各付哪笔账', 17, C_TXT, 'start', True,
        maxw=620, tag='title')
lc.text(MX, 56, '每层门口挂一块牌子：0（没有压缩机，只看最近 128 个 token）、'
                '4（每 4 个 token 压 1 条、再看 512 条）、128（每 128 个 token 才留 1 条、一条不挑全看）',
        10, C_MUTE, 'start', maxw=1120, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层 forward』这一块', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——本章其余每一张机制图都是这梯子上某一层的内部放大', 9, C_MUTE, 'end',
        maxw=430, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for k in ('0', '4', '128'):
    f, s, t = PLATE[k]
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, '牌子 %s' % k, 8.6, '#334155', 'start', tag='lg:%s' % k)
    lgx += 86
for f, s, lab in ((C_KV_F, C_KV_S, 'KV 账（青）'), ('#f1f5f9', '#94a3b8', 'FLOPs 账/参考刻度（灰）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 左：43 层电梯 =================
lc.text(MX, LDY0 - 34, '43 层主干（自上而下 = 第 1 层到第 43 层）', 11, C_TXT, 'start', True,
        maxw=300, tag='ld:t')
lc.text(MX, LDY0 - 18, '牌子就是逐层压缩率表的那一项', 8.5, C_MUTE, 'start', maxw=280, tag='ld:s')

for i, k in enumerate(KIND):
    y = LDY0 + i * STEP
    f, s, t = PLATE[k]
    lc.rect(LAD_X, y, LAD_W, RH, f, s, rx=2.5, sw=1.0)
    lc.text(LAD_X + LAD_W / 2, y + RH - 2.4, k, 8.0, t, 'middle', True, maxw=LAD_W - 10,
            tag='ld:p%d' % i)
for i in (0, 1, 2, 3, 4, 5, 41, 42):
    y = LDY0 + i * STEP
    lc.text(LAD_X - 10, y + RH - 2.4, '第 %d 层' % (i + 1), 8.0, C_TXT, 'end', tag='ld:n%d' % i)
lc.text(LAD_X - 10, LDY0 + 22 * STEP + RH - 3.0, '…', 10, C_MUTE, 'end', tag='ld:dots')

# 电梯右侧：两段的档位读数（括号 + 账）
RUNS = [('第 1–2 层 · 牌子 0', 0, 2, '纯滑窗：压缩账 0；每 query 实看 128 条常数窗', '#22c55e'),
        ('第 3–43 层 · 4 与 128 严格交替', 2, 43,
         '21 层 c4a（每 query 实看 512 条）＋ 20 层 c128a（实看 7812 条）', C_GPU_S)]
for lab, i0, i1, note, col in RUNS:
    y0 = LDY0 + i0 * STEP - 1
    y1 = LDY0 + (i1 - 1) * STEP + RH + 1
    lc.seg(MID_X, y0, MID_X, y0 + 5, col, 1.6)
    lc.seg(MID_X, y1 - 5, MID_X, y1, col, 1.6)
    lc.seg(MID_X, y0 + 2, MID_X, y1 - 2, col, 1.6, dash=(i1 - i0) > 3)
    lc.text(MID_X + 12, (y0 + y1) / 2 - 3, lab, 9.5, C_TXT, 'start', True, tag='run:t%s' % i0)
    lc.text(MID_X + 12, (y0 + y1) / 2 + 12, note, 8.4, C_MUTE, 'start', maxw=330,
            tag='run:n%s' % i0)

# ================= 右：刻度盘（两轴并置） =================
PDX0, PDX1 = 700, BXR
TOP_Y0, TOP_Y1 = 150, 414
lc.rect(PDX0, TOP_Y0, PDX1 - PDX0, TOP_Y1 - TOP_Y0, '#ffffff', C_GPU_S, rx=10, sw=1.4)
lc.text(PDX0 + 16, TOP_Y0 + 24, '同一张表，两种读数（都按对数刻度）', 12, C_TXT, 'start', True,
        maxw=330, tag='ax:t')

AXL, AXR = PDX0 + 92, PDX1 - 30
A1Y, A2Y = TOP_Y0 + 96, TOP_Y0 + 200


def lx(v, lo, hi, y):
    return AXL + (math.log10(v) - lo) / (hi - lo) * (AXR - AXL)


# 上轴：每 query 实看条目数
lc.text(PDX0 + 16, A1Y + 4, '实看条目数（条/query）', 9.5, C_TXT, 'start', True, maxw=210,
        tag='a1:t')
lc.seg(AXL, A1Y, AXR, A1Y, '#94a3b8', 1.4)
for exp in (2, 3, 4):
    x = lx(10 ** exp, 1.8, 4.2, A1Y)
    lc.seg(x, A1Y - 5, x, A1Y + 5, '#cbd5e1', 1.2)
    lc.text(x, A1Y + 18, '10^%d' % exp, 8.2, C_MUTE, 'middle', tag='a1:tk%d' % exp)
for i, (lab, v, col) in enumerate([('纯滑窗 128', 128, '#22c55e'),
                                   ('CSA 512（top-k 封顶）', 512, C_KV_S),
                                   ('HCA 7812（全看）', 7812, C_KV_S)]):
    x = lx(v, 1.8, 4.2, A1Y)
    lc.seg(x, A1Y - 14, x, A1Y, col, 2.2)
    lc.rect(x - 4, A1Y - 20, 8, 8, C_KV_F if col != '#22c55e' else '#dcfce7', col, rx=1.5, sw=1.2)
    ly = A1Y - 32 - (i % 2) * 20
    an = 'end' if i == 2 else 'middle'
    tx = x - 6 if i == 2 else x
    lc.text(tx, ly, lab, 9, '#334155', an, (i == 2), maxw=210, tag='a1:m%d' % i)
    lc.seg(tx if i == 2 else x, ly + 5, x, A1Y - 22, col, 1.0, dash=True)

# 下轴：每 token 摊销字节
lc.text(PDX0 + 16, A2Y + 4, '每 token 摊销（B/token/层）', 9.5, C_TXT, 'start', True, maxw=210,
        tag='a2:t')
lc.seg(AXL, A2Y, AXR, A2Y, '#94a3b8', 1.4)
for exp in (1, 2, 3):
    x = lx(10 ** exp, -0.2, 3.4, A2Y)
    lc.seg(x, A2Y - 5, x, A2Y + 5, '#cbd5e1', 1.2)
    lc.text(x, A2Y + 18, '10^%d' % exp, 8.2, C_MUTE, 'middle', tag='a2:tk%d' % exp)
for i, (lab, v, col) in enumerate([('HCA 层 4.5625 B', 4.5625, C_KV_S),
                                   ('CSA 层 179.0 B', 179.0, C_KV_S),
                                   ('整机加权 89.540698 B', 89.540698, '#155e75')]):
    x = lx(v, -0.2, 3.4, A2Y)
    lc.seg(x, A2Y - 14, x, A2Y, col, 2.2)
    lc.rect(x - 4, A2Y - 20, 8, 8, C_KV_F, col, rx=1.5, sw=1.2)
    ly = A2Y - 32 - (i % 2) * 20
    lc.text(x, ly, lab, 9, '#334155', 'middle', True, maxw=210, tag='a2:m%d' % i)
    lc.seg(x, ly + 5, x, A2Y - 22, col, 1.0, dash=True)
lc.text(PDX0 + 16, TOP_Y1 - 30, '两轴一起读：实看条目数越多（右）的档，每 token 摊销反而越小'
                                '（左）——省算力与省显存，在这张表上是两个方向。', 8.6, '#334155',
        'start', maxw=620, tag='ax:n0')
lc.text(PDX0 + 16, TOP_Y1 - 12, 'CSA 与 HCA 是同一套压缩机制的两档：牌子 4 的层挑 512 条，'
                                '牌子 128 的层一条不挑。', 8.6, C_MUTE, 'start', maxw=620, tag='ax:n1')

# ================= 右下：短上下文倒挂 =================
BOT_Y0, BOT_Y1 = 430, 700
lc.rect(PDX0, BOT_Y0, PDX1 - PDX0, BOT_Y1 - BOT_Y0, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(PDX0 + 16, BOT_Y0 + 24, '短上下文会倒挂：候选墙还没长起来', 12, C_TXT, 'start', True,
        maxw=420, tag='bt:t')
lc.text(PDX0 + 16, BOT_Y0 + 44, '候选数 = 上下文长度 // 压缩率；一旦候选 ≤ 512，排序就没有意义——'
                                '实现直接全选，索引器打分成了纯开销。', 8.8, C_MUTE, 'start',
        maxw=460, tag='bt:s')
TAB = [['上下文', 'c4a 候选', '实看', 'c128a 候选', '实看'],
       ['1M', '250000', '512（封顶）', '7812', '7812（全看）'],
       ['1K', '250', '250（全选）', '7', '7（全看）']]
TW = [86, 118, 128, 122, 118]
tx0 = PDX0 + 20
for r, row in enumerate(TAB):
    ty = BOT_Y0 + 78 + r * 26
    cx = tx0
    for c, cell in enumerate(row):
        bold = (r == 0)
        lc.text(cx, ty, cell, 9.2 if bold else 9.5, C_TXT if bold else '#334155',
                'start', bold, tag='bt:%d%d' % (r, c))
        cx += TW[c]
    if r == 0:
        lc.seg(tx0, ty + 8, tx0 + sum(TW), ty + 8, C_MUTE, 1.2)
lc.text(PDX0 + 20, BOT_Y1 - 58, '实现里的快路径：候选 ≤ topk 就直接全选（8//4 = 2 ≤ 512 → 真；'
                                '2048//4 = 512 ≤ 512 → 真；4096//4 = 1024 → 假）', 8.6, '#155e75',
        'start', True, maxw=620, tag='bt:f')
lc.text(PDX0 + 20, BOT_Y1 - 38, '1K 上下文里 c4a 的候选只有 250 条：压缩把候选降 4 倍、'
                                '稀疏再把实看封在 512——两笔账在短上下文里都还没生效。', 8.6, C_MUTE,
        'start', maxw=620, tag='bt:g')
lc.text(PDX0 + 20, BOT_Y1 - 18, '（论文只说这些层交错排布，没有给出为什么交替——'
                                '本图把两档的读数列在一起，不做因果断言。）', 8.4, lc.C_FAINT,
        'start', maxw=620, tag='bt:h')

# ================= 左下：表外三处口径 =================
NBY0, NBY1 = 724, 854
lc.rect(MX, NBY0, 640, NBY1 - NBY0, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, NBY0 + 24, '表外三处口径（读表前先对齐）', 11, C_TXT, 'start', True, maxw=380,
        tag='nb:t')
for i, s in enumerate([
        '① 全长 44 项、主干 43 层：末项 0 是留给多 token 预测那一档的槽，不在主干里。',
        '② 配置里的 0 在分类时归一成 1（max(1,·)）：44 项全算也归三类 {3, 21, 20}。',
        '③ HCA 层没有索引器对象：它不建索引缓存，账里只有那 584 B 的主条目。']):
    lc.text(MX + 16, NBY0 + 46 + i * 19, s, 8.8, '#334155', 'start', maxw=610, tag='nb:%d' % i)
lc.text(MX + 16, NBY1 - 12, '牌子 4 的层同时也建一份索引器缓存（132 B/条）——'
                            '所以它每 token 的摊销是 179.0 B 而不是 146.0 B。', 8.6, '#155e75',
        'start', maxw=610, tag='nb:x')

# ---------------- 页脚 ----------------
lc.text(MX, 876, '数字口径：1M 上下文；实看条目数 = min(top-k 512, 该层候选数)；'
                 '7812 = 1000000 // 128。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 894, '压缩率取值 = config 的 V4-Flash 口径；层型分类与逐层摊销由本章驱动脚本'
                 '纯算术实跑（12 行读数全部可复算）。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-layer-ladder.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems, ladder ends {LAD_END:.1f})')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
