#!/usr/bin/env python3
"""ch28 机制图 · CSA 压缩机的跨拍续窗（ch28-fig-csa-state-scroll）

claim：跨拍续窗：第二拍的第 1 条条目与一次性 prefill 的第 4 条条目逐位相同
（max|差| = 0.00000000），而接缝那一半来自上一拍留在状态里的 b 投影——去掉它输出就变
（差 [0.023024, 0.01599]）；官方实现的 state 是两个半区（行 0..3 重叠窗 / 行 4..7 当前窗），
块尾那一拍把两半区拼成 2m 行再滚动。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  两拍（12 + 4）合计 4 条；第二拍第 1 条 = [0.599393, 0.241721]，与一次性第 4 条逐位相同 = True
  对照（第二拍不给状态）：[0.576368, 0.257711]，与有状态差 [0.023024, 0.01599]
  第 4 条两半区权重逐列和：b 半区 [0.498234, 0.720135]、a 半区 [0.501766, 0.279865]
  官方状态布局：state 形状 (8, 2)、行 0..3 = 重叠窗、行 4..7 = 当前窗；
        块尾 pos=7 的 2m 输入 = [1.0, 2.0, 3.0, 4.0] + [5.0, 6.0, 7.0, 8.0]，
        权重 [0.321957, 0.118441, 0.043572, 0.016029] × 2
  14 个 token 喂完：产出 3 条条目、buffer 里剩 2 行（= 14 % 4）

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩缓存侧）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 800
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_KV_LIGHT = '#ecfeff'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_ACC_S = '#c2410c'
C_ROW_A, C_ROW_B = '#e0f2fe', '#cffafe'

EXTRA_DEFS = ('<defs>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_S}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '压缩机不是无状态的：每拍把这一拍的窗留给下一拍', 17, C_TXT, 'start', True,
        maxw=600, tag='title')
lc.text(MX, 56, '第二拍只来 4 个 token，却产出了与一次性 prefill 逐位相同的第 4 条条目——'
                '接缝那一半来自上一拍留在状态里的投影', 10, C_MUTE, 'start', maxw=1080, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）青列『KV 池』块旁的压缩机状态', 9, C_MUTE, 'end', maxw=430,
        tag='l0:a')
lc.text(BXR, 64, '——它把那张讲压缩坐标系图里的「未满的窗」展开成一次完整的状态滚动', 9,
        C_MUTE, 'end', maxw=500, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_ROW_A, C_KV_S, '重叠窗（行 0..3，上一拍留下）'),
                  (C_ROW_B, C_KV_S, '当前窗（行 4..7，这一拍）'),
                  ('#fff7ed', C_ACC_S, '对照与输出读数')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 上：三拍时间线 =================
BEATS = [
    (MX, 420, '第一拍：token 0..11', ['产出 3 条条目（12 // 4 = 3）',
                                      '末窗 token [8, 9, 10, 11] 的投影',
                                      '留在状态里，等下一拍'],
     C_KV_F, C_KV_S),
    (520, 420, '第二拍：token 12..15', ['产出 1 条条目（4 // 4 = 1）',
                                        '块尾那一拍：2m 行的前一半正是',
                                        '上一拍留在状态里的那半区'],
     C_KV_F, C_KV_S),
    (1000, BXR - 1000, '两拍合计', ['3 + 1 = 4 条，与一次性 16 个 token',
                                    '的 4 条条目完全一致——',
                                    '状态就是那条「接缝」'],
     C_SOFT_F, C_SOFT_S),
]
for bx, bw, t1, rows, fill, stroke in BEATS:
    lc.rect(bx, 112, bw, 108, fill, stroke, rx=10, sw=1.5)
    lc.text(bx + 16, 140, t1, 12, C_KV_DEEP, 'start', True, maxw=bw - 32, tag='bt:t')
    for i, s in enumerate(rows):
        lc.text(bx + 16, 164 + i * 19, s, 9.0, '#334155', 'start', maxw=bw - 32, tag='bt:r')
lc.parrow([(MX + 420, 166), (516, 166)], C_KV_S, 2.2, marker='kv')
lc.parrow([(520 + 420, 166), (996, 166)], C_KV_S, 2.2, marker='kv')

# ================= 左：状态表 =================
TX0, TX1 = MX, 620
TITLE_Y = 252
lc.text(TX0, TITLE_Y, '状态表：8 行、两个半区（官方实现的 decode 布局）', 11.5, C_KV_DEEP,
        'start', True, maxw=560, tag='st:t')
lc.text(TX0, TITLE_Y + 18, 'state 形状 (8, 2)；下表列第 0 列的值与它对应的 softmax 权重',
        8.8, C_MUTE, 'start', maxw=560, tag='st:s')

HDR_Y, HDR_H = 284, 26
ROWS_Y, ROW_H = 310, 27
COL = [(TX0, 86, '行'), (TX0 + 86, 250, '值（第 0 列）'), (TX0 + 336, TX1 - TX0 - 336, 'softmax 权重')]
lc.rect(TX0, HDR_Y, TX1 - TX0, HDR_H, '#e2e8f0', '#94a3b8', rx=4, sw=1.1)
for x, w, lab in COL:
    lc.text(x + 10, HDR_Y + 18, lab, 9.2, '#334155', 'start', True, tag='st:h%s' % lab[:2])

ROWS_DATA = [('0', '1.0', '0.321957'), ('1', '2.0', '0.118441'), ('2', '3.0', '0.043572'),
             ('3', '4.0', '0.016029'), ('4', '5.0', '0.321957'), ('5', '6.0', '0.118441'),
             ('6', '7.0', '0.043572'), ('7', '8.0', '0.016029')]
for i, (a, b, c) in enumerate(ROWS_DATA):
    ry = ROWS_Y + i * ROW_H
    lc.rect(TX0, ry, TX1 - TX0, ROW_H, C_ROW_A if i < 4 else C_ROW_B, C_KV_S, rx=3, sw=1.0)
    lc.text(TX0 + 10, ry + 18, a, 9.2, C_KV_DEEP, 'start', True, tag='st:r%d' % i)
    lc.text(TX0 + 96, ry + 18, b, 9.2, '#334155', 'start', tag='st:v%d' % i)
    lc.text(TX0 + 346, ry + 18, c, 9.2, C_KV_DEEP, 'start', tag='st:w%d' % i)
lc.text(TX0, ROWS_Y + 8 * ROW_H + 20, '块尾 pos = 7 的那一拍：2m 行 = [1.0, 2.0, 3.0, 4.0]（上半区，'
                                      '上一拍留下）+ [5.0, 6.0, 7.0, 8.0]（下半区，当前窗）',
        8.8, '#155e75', 'start', maxw=580, tag='st:n1')
lc.text(TX0, ROWS_Y + 8 * ROW_H + 38, '8 个权重加起来 = 1.0：两半区合起来才是那一份完整权重。',
        8.8, C_MUTE, 'start', maxw=580, tag='st:n2')

# ================= 右：滚动动线 + 对照 =================
RX0 = 660
lc.rect(RX0, 240, BXR - RX0, 190, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
lc.text(RX0 + 16, 268, '块尾那一拍：拼两半 → softmax → 一条条目 → 滚动', 11.5, C_TXT,
        'start', True, maxw=560, tag='fl:t')
FLOW = [(288, '① 把 state 的两半区拼成 2m 行', '上半区 [1.0, 2.0, 3.0, 4.0] ＋ '
         '下半区 [5.0, 6.0, 7.0, 8.0]'),
        (316, '② 跨这 8 个元素做一次 softmax', '权重 [0.321957, 0.118441, 0.043572, '
         '0.016029] × 2，和 = 1.0'),
        (344, '③ 加权求和 → 1 条压缩条目', '这一条就是全局的第 4 条'),
        (372, '④ 滚动：当前窗 → 重叠窗', '算完把行 4..7 复制到行 0..3，'
         '作为下一拍的「前一个窗」')]
for y, a, b in FLOW:
    lc.text(RX0 + 16, y, a, 9.8, C_KV_DEEP, 'start', True, tag='fl:a%d' % y)
    lc.text(RX0 + 268, y, b, 9.0, '#334155', 'start', maxw=500, tag='fl:b%d' % y)

CY0 = 444
lc.rect(RX0, CY0, BXR - RX0, 146, '#fff7ed', C_ACC_S, rx=10, sw=1.4)
lc.text(RX0 + 16, CY0 + 26, '对照：跨拍续窗 vs 一次性 prefill vs 不给状态', 11.5, C_ACC_S,
        'start', True, maxw=520, tag='cp:t')
CP = [('第二拍第 1 条', '[0.599393, 0.241721]'),
      ('一次性 prefill 的第 4 条', '[0.599393, 0.241721]'),
      ('逐位相同 = True｜max|差|', '0.00000000'),
      ('对照（第二拍不给状态）', '[0.576368, 0.257711]，与有状态差 [0.023024, 0.01599]')]
for i, (a, b) in enumerate(CP):
    ty = CY0 + 54 + i * 26
    lc.text(RX0 + 16, ty, a, 9.4, C_TXT, 'start', True, tag='cp:a%d' % i)
    lc.text(RX0 + 224, ty, b, 9.4, C_ACC_S if i < 3 else '#334155', 'start', maxw=560,
            tag='cp:b%d' % i)

# ================= 底：三条补充 =================
BY0 = 604
lc.rect(MX, BY0, BXR - MX, 130, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 28, '读这张图要带走的三件事', 11.5, C_KV_DEEP, 'start', True, maxw=340,
        tag='bt2:t')
B2 = [('① 两半区都在贡献',
       '第 4 条的两半区权重逐列和：b 半区 [0.498234, 0.720135]、a 半区 [0.501766, 0.279865]',
       '不是「随便看一眼前窗」——接缝那一半真的参与了加权'),
      ('② 去掉状态就变',
       '第二拍第 1 条从 [0.599393, 0.241721] 变成 [0.576368, 0.257711]',
       '状态不是可选项，它就是那条接缝的一半'),
      ('③ 没攒满的窗留在状态里',
       '14 个 token 喂完：产出 3 条条目、缓冲里剩 2 行（= 14 % 4）',
       '最新的 2 个 token 这一拍没有压缩条目可看，只有滑窗看得见它们的原样 KV')]
for i, (a, b, c) in enumerate(B2):
    ty = BY0 + 56 + i * 34
    lc.text(MX + 16, ty, a, 9.6, C_KV_DEEP, 'start', True, maxw=220, tag='b2:a%d' % i)
    lc.text(MX + 218, ty, b, 9.4, '#334155', 'start', maxw=740, tag='b2:b%d' % i)
    lc.text(MX + 980, ty, c, 8.8, C_MUTE, 'start', maxw=460, tag='b2:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 754, '数字口径：玩具 m = 4、c = 2、d = 3；两拍 = 前 12 个 token + 后 4 个 token。'
                 '状态布局逐行转写自官方参考实现（DeepSeek-V4-Pro inference/model.py:L303-L304、'
                 'L347-L354）的 decode 分支，在本机 NumPy 上跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 772, '依据 DeepSeek-V4 技术报告 §2.3.1（arXiv:2606.19348）的 1/m 与产出单位；'
                 '跨拍续窗与对照数值由本章算例实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-csa-state-scroll.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
