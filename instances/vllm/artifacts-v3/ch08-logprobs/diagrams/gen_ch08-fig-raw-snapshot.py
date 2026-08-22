#!/usr/bin/env python3
"""ch08 机制图 1 · raw 留底（figure_spec ch08-fig-raw-snapshot，模板 before-after）

放大自 L0 绿色 GPU 带采样列（sample_column · 本章 l0_zoom『上行泳道 logprobs 支路』）里
Sampler 的第 1 步——即本章 L2 章图 center 拍片 ② 『raw 留底』的机制展开；
上游回指拍片 ① 『批登记』（max_num_logprobs 进来）、下游前指拍片 ③ 『gather 三件套』
（raw_logprobs 出去）。非新架构画法，架构归属回指 L0/L2。

claim：raw 留底在一切采样变换之前对原始 logits 做 log_softmax（token0 -0.8386），
惩罚之后的分布给出完全不同的 top-k（top1 换成 token1 -0.4705、token0 掉到 -2.3705），
v1 默认报前者、V0 报后者——同一被采样 token1 两个视角差 0.4681。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 972
MX, BXR = 60, 1440
LCTR, RCTR, TLX = 390, 1110, 750          # 左/右栏中心、时间轴 x
PEN_F = '#fef2f2'                          # 惩罚改写底色（图例声明）

# ---------------- 标题区 ----------------
lc.text(MX, 34, '「惩罚不扭曲模型意见」：raw 留底抢在一切采样变换之前',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '留底段 verbatim vllm/v1/sample/sampler.py:L79-L96（NOTE(woosuk)）：compute_logprobs = log_softmax 在 fp32 转换、'
        '惩罚、温度之前执行；后续采样管线全程原地改写 logits，碰不到已物化的独立张量',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L2 拍片 ② raw 留底 · L0：上行泳道 logprobs 支路'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')
_up = '← 上游 · L2 拍片 ① 批登记：max_num_logprobs = k = 2 进来'
_uw = lc.tw(_up, 9, ) + 14
lc.rect(MX, 76, _uw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(MX + _uw / 2, 89.5, _up, 9, lc.C_MUTE, 'middle', maxw=_uw - 4, tag='chip:up')

# ---------------- 两栏列头 + 顶栏 logits 行（同一行画两遍） ----------------
lc.text(LCTR, 124, 'before · v1 默认（raw_logprobs 模式 · 镜头 A）', 11.5, lc.C_GPU_S,
        'middle', True, maxw=560, tag='colhdr:L')
lc.text(RCTR, 124, 'after · V0 语义（用采样时的分布算 · 镜头 B）', 11.5, lc.C_ENG_S,
        'middle', True, maxw=560, tag='colhdr:R')

LOGITS = ['2.0', '1.9', '0.5', '0.0', '-1.0']
CELL_W, CELL_H, CELL_G, CELL_Y = 72, 34, 6, 156
x0_L = LCTR - (5 * CELL_W + 4 * CELL_G) / 2
x0_R = RCTR - (5 * CELL_W + 4 * CELL_G) / 2
lc.text(LCTR, 148, '模型前向输出 · 原始 logits（V=5）', 9, lc.C_MUTE, 'middle', maxw=400,
        tag='rowlbl:L')
lc.text(RCTR, 148, '同一行 logits 送进采样管线（与左栏同一份模型输出）', 9, lc.C_MUTE,
        'middle', maxw=440, tag='rowlbl:R')
for base, tagp in ((x0_L, 'L'), (x0_R, 'R')):
    for i, v in enumerate(LOGITS):
        cx = base + i * (CELL_W + CELL_G)
        lc.rect(cx, CELL_Y, CELL_W, CELL_H, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.3)
        lc.text(cx + CELL_W / 2, CELL_Y + 22, v, 11.5, lc.C_TXT, 'middle', True,
                maxw=CELL_W - 8, tag=f'lg{tagp}{i}')
        lc.text(cx + CELL_W / 2, CELL_Y + CELL_H + 14, f'token{i}', 8, lc.C_MUTE,
                'middle', maxw=52, tag=f'tk{tagp}{i}')

# ---------------- 左栏：log_softmax 立即留底 → 概率条 ----------------
lc.seg(LCTR, CELL_Y + CELL_H + 19, LCTR, 220, lc.C_GPU_S, 2.0, 'dn')
BOX_L = (110, 222, 560, 46)
lc.rect(*BOX_L[:2], BOX_L[2], BOX_L[3], '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(BOX_L[0] + 16, BOX_L[1] + 18, 'log_softmax 立即留底——此刻 fp32 转换 / 处理器 / 采样都还没碰 logits',
        10.5, lc.C_TXT, 'start', True, maxw=BOX_L[2] - 28, tag='lbox:t')
lc.text(BOX_L[0] + 16, BOX_L[1] + 36, 'compute_logprobs(logits) = logits.log_softmax(dim=-1)（非原地 → 独立张量 raw_logprobs）',
        9, lc.C_MUTE, 'start', maxw=BOX_L[2] - 28, tag='lbox:s')

# 概率条（条长 ∝ e^logprob，取自 spec 数字行；条上标 logprob 值）
RAW = [('token0', '-0.8386', 0.4325, True, '← top1（模型的）'),
       ('token1', '-0.9386', 0.3911, False, ''),
       ('token2', '-2.3386', 0.0964, False, ''),
       ('token3', '-2.8386', 0.0585, False, ''),
       ('token4', '-3.8386', 0.0215, False, '')]
PROC = [('token0', '-2.3705', 0.0936, False, '← 模型的 top1 被压到这里'),
        ('token1', '-0.4705', 0.6247, True, '← top1（换人）'),
        ('token2', '-1.8705', 0.1540, False, ''),
        ('token3', '-2.3705', 0.0936, False, ''),
        ('token4', '-3.3705', 0.0344, False, '')]
PMAX = 0.6247
BAR_XL, BAR_XR, BAR_MAXW, BAR_H, BAR_PITCH = 178, 968, 390, 17, 27


def bars(rows, y0, tagp, bar_x):
    for i, (tk, val, p, hot, note) in enumerate(rows):
        y = y0 + i * BAR_PITCH
        bl = p / PMAX * BAR_MAXW
        lc.text(bar_x - 10, y + 13, tk, 9.5, lc.C_TXT, 'end', maxw=56, tag=f'{tagp}lb{i}')
        lc.rect(bar_x, y, bl, BAR_H, lc.C_SAM_F if hot else lc.C_GPU_F,
                lc.C_SAM_S if hot else lc.C_GPU_S, rx=3, sw=1.3 if hot else 1.0)
        lc.text(bar_x + bl + 8, y + 13, val, 10, lc.C_SAM_S if hot else '#334155',
                'start', hot, maxw=70, tag=f'{tagp}v{i}')
        if note:
            lc.text(bar_x + bl + 8 + 62, y + 13, note, 8.5,
                    lc.C_SAM_S if hot else lc.C_ABORT, 'start', maxw=190, tag=f'{tagp}n{i}')


bars(RAW, 286, 'raw', BAR_XL)
lc.text(LCTR, 286 + 5 * BAR_PITCH + 12, 'raw_logprobs 行 = log_softmax(原始 logits)：top1 = token0（-0.8386）',
        9.5, lc.C_TXT, 'middle', maxw=520, tag='raw:cap')

# 报告盒（左）
REP_L = (110, 452, 560, 62)
lc.rect(*REP_L[:2], REP_L[2], REP_L[3], lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.6)
lc.text(REP_L[0] + 16, REP_L[1] + 20, '被采样 token1 在这一行的报告值', 9.5, lc.C_MUTE,
        'start', maxw=300, tag='repl:t')
lc.text(REP_L[0] + 16, REP_L[1] + 48, '-0.9386', 20, lc.C_TXT, 'start', True, maxw=130,
        tag='repl:v')
lc.text(REP_L[0] + REP_L[2] - 16, REP_L[1] + 46, '模型本来的意见——RL 训练 / 评分要的这份',
        9, lc.C_MUTE, 'end', maxw=300, tag='repl:n')

# 贪心对照盒（左）
GC_L = (110, 532, 560, 84)
lc.rect(*GC_L[:2], GC_L[2], GC_L[3], '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(GC_L[0] + 16, GC_L[1] + 20, '贪心对照（镜头 A 的 argmax）', 9.5, lc.C_TXT,
        'start', True, maxw=300, tag='gcl:t')
lc.text(GC_L[0] + 16, GC_L[1] + 40, '该输入的 argmax = token0；真实引擎这一步先施加惩罚再采样——实际采到 token1。',
        9, lc.C_MUTE, 'start', maxw=GC_L[2] - 30, tag='gcl:l1')
lc.text(GC_L[0] + 16, GC_L[1] + 58, '左栏对 token1 的取值 -0.9386，即 raw 视角对「实际被采到者」的报告。',
        9, lc.C_MUTE, 'start', maxw=GC_L[2] - 30, tag='gcl:l2')

# ---------------- 右栏：改写器 → 惩罚后行 → 概率条 ----------------
lc.seg(RCTR, CELL_Y + CELL_H + 22, RCTR, 246, lc.C_ABORT, 2.0, 'ab')
RW = (830, 248, 560, 54)
lc.rect(*RW[:2], RW[2], RW[3], PEN_F, lc.C_ABORT, rx=8, sw=1.8)
lc.text(RW[0] + 16, RW[1] + 20, '采样管线原地改写 logits（此后才算 V0 的 logprobs）', 10.5,
        lc.C_ABORT, 'start', True, maxw=RW[2] - 28, tag='rw:t')
lc.text(RW[0] + 16, RW[1] + 40, 'token0 已生成 + presence_penalty = 2.0 → logits[0] -= 2.0（惩罚公式 utils.py:L88）',
        9, lc.C_MUTE, 'start', maxw=RW[2] - 28, tag='rw:s')
lc.text(RCTR, 328, '惩罚后 logits（token0 被压到 0.0）', 9, lc.C_MUTE, 'middle', maxw=400,
        tag='pen:lbl')
PEN_LOGITS = ['0.0', '1.9', '0.5', '0.0', '-1.0']
PEN_Y = 336
for i, v in enumerate(PEN_LOGITS):
    cx = x0_R + i * (CELL_W + CELL_G)
    hot = (i == 0)
    lc.rect(cx, PEN_Y, CELL_W, CELL_H, '#fee2e2' if hot else lc.C_GPU_F,
            lc.C_ABORT if hot else lc.C_GPU_S, rx=6, sw=1.6 if hot else 1.3)
    lc.text(cx + CELL_W / 2, PEN_Y + 22, v, 11.5, lc.C_ABORT if hot else lc.C_TXT,
            'middle', hot, maxw=CELL_W - 8, tag=f'pl{i}')
    lc.text(cx + CELL_W / 2, PEN_Y + CELL_H + 14, f'token{i}', 8, lc.C_MUTE, 'middle',
            maxw=52, tag=f'ptk{i}')
lc.seg(RCTR, PEN_Y + CELL_H + 22, RCTR, 414, lc.C_ENG_S, 2.0, 'up')
V0B = (830, 416, 560, 38)
lc.rect(*V0B[:2], V0B[2], V0B[3], '#ffffff', lc.C_ENG_S, rx=8, sw=1.5)
lc.text(V0B[0] + 16, V0B[1] + 24, 'V0 视角：log_softmax(惩罚后 logits)', 10.5, lc.C_ENG_S,
        'start', True, maxw=V0B[2] - 28, tag='v0b')
bars(PROC, 478, 'prc', BAR_XR)
lc.text(RCTR, 478 + 5 * BAR_PITCH + 12, '惩罚后分布：top1 换成 token1（-0.4705）——被惩罚扭曲的报告',
        9.5, lc.C_TXT, 'middle', maxw=520, tag='prc:cap')

# 报告盒（右）
REP_R = (830, 646, 560, 62)
lc.rect(*REP_R[:2], REP_R[2], REP_R[3], PEN_F, lc.C_ABORT, rx=8, sw=1.6)
lc.text(REP_R[0] + 16, REP_R[1] + 20, '被采样 token1 在这一行的报告值', 9.5, lc.C_MUTE,
        'start', maxw=300, tag='repr:t')
lc.text(REP_R[0] + 16, REP_R[1] + 48, '-0.4705', 20, lc.C_TXT, 'start', True, maxw=130,
        tag='repr:v')
lc.text(REP_R[0] + REP_R[2] - 16, REP_R[1] + 46, '干预后分布的意见（V0 报这份）', 9,
        lc.C_MUTE, 'end', maxw=300, tag='repr:n')

# ---------------- 中缝时间轴：留底时点在改写之前 ----------------
lc.seg(TLX, 150, TLX, 640, lc.C_FAINT, 1.4, dash=True)
TICKS = [
    (172, '① logits 出生', None),
    (240, '② raw 留底·左栏', (TLX - 8, 240, BOX_L[0] + BOX_L[2], 240)),
    (278, '③ 惩罚改写·右栏', (TLX + 8, 278, RW[0], 278)),
    (600, '④ 温度 / top-k / 采样', None),
]
for ty, lbl, leader in TICKS:
    lc.seg(TLX - 6, ty, TLX + 6, ty, lc.C_FAINT, 1.2)
    lc.text(TLX, ty + 15, lbl, 8.5, lc.C_MUTE, 'middle', maxw=118, tag='tick:' + lbl[:6])
    if leader:
        lc.seg(leader[0], leader[1], leader[2], leader[3], lc.C_FAINT, 1.1, dash=True)
lc.text(TLX, 660, '时间轴（早 → 晚）', 8.5, lc.C_FAINT, 'middle', maxw=118, tag='tl:lbl')

# ---------------- 底部：同一 token 的两个数字 ----------------
CMP = (MX, 700, BXR - MX, 92)
lc.rect(*CMP[:2], CMP[2], CMP[3], '#ffffff', lc.C_TXT, rx=10, sw=1.5)
lc.text(750, 722, '同一被采样 token1 的两个数字（镜头 D）', 11, lc.C_TXT, 'middle', True,
        maxw=560, tag='cmp:t')
lc.rect(150, 736, 330, 40, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.4)
lc.text(315, 761, 'raw 视角：-0.9386（v1 默认报这）', 12, lc.C_TXT, 'middle', True,
        maxw=314, tag='cmp:L')
lc.rect(630, 736, 240, 40, '#fff7ed', lc.C_ENG_S, rx=8, sw=1.8)
lc.text(750, 761, '差 0.4681', 15, lc.C_ENG_S, 'middle', True, maxw=200, tag='cmp:delta')
lc.rect(1020, 736, 330, 40, PEN_F, lc.C_ABORT, rx=8, sw=1.4)
lc.text(1185, 761, 'processed 视角：-0.4705（V0 报这）', 12, lc.C_TXT, 'middle', True,
        maxw=314, tag='cmp:R')
lc.text(750, 788, '惩罚是采样干预，不是模型意见——两个视角的分道之处（NOTE(woosuk) 与 V0 分道正在此）',
        9.5, lc.C_MUTE, 'middle', maxw=1100, tag='cmp:sub')

# ---------------- 四态面 ----------------
ST = (MX, 812, BXR - MX, 74)
lc.rect(*ST[:2], ST[2], ST[3], '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(ST[0] + 16, ST[1] + 20, '四态开关 logprobs_mode（v0.27 引擎级，config/model.py:L99-L105）',
        9.5, lc.C_TXT, 'start', True, maxw=700, tag='st:t')
lc.text(ST[0] + 16, ST[1] + 40, 'raw_logprobs＝v1 默认（左栏） ｜ processed_logprobs＝贪心路径物化并覆写留底'
        '（数值与右栏逐位相同，equals_v0_lens=true）', 9, lc.C_MUTE, 'start',
        maxw=ST[2] - 30, tag='st:l1')
lc.text(ST[0] + 16, ST[1] + 58, 'raw_logits＝留底即原始 logits、连 log_softmax 都不做（k+1 列 = [2.0, 2.0, 1.9]） ｜ '
        'processed_logits＝干预后原始分', 9, lc.C_MUTE, 'start', maxw=ST[2] - 30, tag='st:l2')

# ---------------- 图例 + 下游 chip + 页脚 ----------------
LEG_Y = 916
lx = MX
for kind, name in [('gpu', '概率条（条长 ∝ e^logprob，条上数字即 logprob 值）'),
                   ('sam', '各视角 top1'), ('pen', '被惩罚改写（改写器 / token0 格）')]:
    if kind == 'gpu':
        lc.rect(lx, LEG_Y - 8, 22, 13, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.0)
    elif kind == 'sam':
        lc.rect(lx, LEG_Y - 8, 22, 13, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.2)
    else:
        lc.rect(lx, LEG_Y - 8, 22, 13, PEN_F, lc.C_ABORT, rx=3, sw=1.2)
    lc.text(lx + 28, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=330, tag='leg' + kind)
    lx += 28 + lc.tw(name, 9) + 22
_dn = '→ 下游 · L2 拍片 ③ gather 三件套：raw_logprobs [B, V] 出去（只把 k+1 列带回来）'
_dw = lc.tw(_dn, 9) + 14
lc.rect(BXR - _dw, LEG_Y - 11, _dw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(BXR - _dw / 2, LEG_Y + 2, _dn, 9, lc.C_MUTE, 'middle', maxw=_dw - 4, tag='chip:dn')
lc.text(MX, 952, '留底段与覆写行 verbatim vllm/v1/sample/sampler.py:L79-L104 · compute_logprobs = L304-L306 · '
        '惩罚公式 vllm/model_executor/layers/utils.py:L88 · 镜头 A/B/C/D 数值 host 实测（V=5 玩具词表、k=2、'
        '惩罚后张量按 L88 公式构造）· 行号基线 vLLM v0.27.1', 9, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch08-fig-raw-snapshot.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
