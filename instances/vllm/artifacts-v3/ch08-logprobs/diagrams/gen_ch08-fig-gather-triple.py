#!/usr/bin/env python3
"""ch08 机制图 2 · gather 三件套（figure_spec ch08-fig-gather-triple，模板 tensor-flow）

放大自 L0 绿色 GPU 带采样列（sample_column · 本章 l0_zoom『上行泳道 logprobs 支路』）里
Sampler 的第 8 步——即本章 L2 章图 center 拍片 ③ 『gather 三件套』的机制展开；
上游回指拍片 ② 『raw 留底』（raw_logprobs [num_tok, V] 进）、下游前指拍片 ④ 『D2H』
（[num_tok, k+1] 出）。非新架构画法，架构归属回指 L0/L2。

claim：gather 三件套把 [num_tok, V] 的 raw_logprobs 缩成 [num_tok, k+1]：topk 取领奖台、
gather 取被采样者成绩、(x>=v).sum(-1) 不排序数出 rank，cat 把被采样者钉在列 0——
V=6/k=2 实测三行 [0,0,1]/[3,0,1]/[2,0,1]，落榜者（rank 4）与并列上界（rank 3）都有交代。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 676
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'gather 三件套：把 [num_tok, V] 缩成 [num_tok, k+1]，被采样者恒钉在列 0',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, 'topk 摘领奖台 · gather 按下标取被采样者成绩 · (x>=v).sum(-1) 不排序、整行数出 1-based 词表 rank'
        '——O(V) 计数免 O(V log V) 排序', 10.5, lc.C_MUTE, 'start', maxw=980, tag='subtitle')
_ch = '放大自 L2 拍片 ③ gather 三件套 · L0：上行泳道 logprobs 支路'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')
_up = '← 上游 · L2 拍片 ② raw 留底：raw_logprobs [num_tok, V] 进（本例 V=6、同一行复用 3 次）'
_uw = lc.tw(_up, 9) + 14
lc.rect(MX, 76, _uw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(MX + _uw / 2, 89.5, _up, 9, lc.C_MUTE, 'middle', maxw=_uw - 4, tag='chip:up')

# ---------------- 左：输入 raw_logprobs 一行（V=6） ----------------
IN_X0, IN_CW, IN_CG, IN_CY, IN_CH = 90, 74, 4, 172, 32
IN_VALS = ['-0.9084', '-1.4084', '-1.4084', '-2.9084', '-3.4084', '-3.9084']
lc.text(IN_X0, 150, 'raw_logprobs [num_tok, V]——例：1 行 × V=6', 10, lc.C_TXT, 'start', True,
        maxw=440, tag='in:t')
for i, v in enumerate(IN_VALS):
    cx = IN_X0 + i * (IN_CW + IN_CG)
    hot_top1 = (i == 0)
    tie = i in (1, 2)
    lc.rect(cx, IN_CY, IN_CW, IN_CH,
            lc.C_SAM_F if hot_top1 else lc.C_GPU_F,
            lc.C_SAM_S if hot_top1 else (lc.C_ENG_S if tie else lc.C_GPU_S),
            rx=5, sw=1.5 if hot_top1 else 1.2)
    lc.text(cx + IN_CW / 2, IN_CY + 21, v, 10.5,
            lc.C_SAM_S if hot_top1 else lc.C_TXT, 'middle', hot_top1, maxw=IN_CW - 8,
            tag=f'in{i}')
    lc.text(cx + IN_CW / 2, IN_CY - 8, f'id{i}', 8.5, lc.C_MUTE, 'middle', maxw=40,
            tag=f'inid{i}')
# 并列括注（id1/id2）
tb_x0 = IN_X0 + 1 * (IN_CW + IN_CG)
tb_x1 = IN_X0 + 2 * (IN_CW + IN_CG) + IN_CW
lc.seg(tb_x0, IN_CY + IN_CH + 6, tb_x1, IN_CY + IN_CH + 6, lc.C_ENG_S, 1.3)
lc.seg(tb_x0, IN_CY + IN_CH + 6, tb_x0, IN_CY + IN_CH + 2, lc.C_ENG_S, 1.3)
lc.seg(tb_x1, IN_CY + IN_CH + 6, tb_x1, IN_CY + IN_CH + 2, lc.C_ENG_S, 1.3)
lc.text((tb_x0 + tb_x1) / 2, IN_CY + IN_CH + 20, 'id1 / id2 并列（值相同）', 8.5, lc.C_ENG_S,
        'middle', maxw=170, tag='tie:note')
for i, mk in ((0, '↑ 行 0 采样'), (2, '↑ 行 2 采样'), (3, '↑ 行 1 采样')):
    cx = IN_X0 + i * (IN_CW + IN_CG) + IN_CW / 2
    lc.text(cx, IN_CY + IN_CH + 38, mk, 8, lc.C_MUTE, 'middle', maxw=76, tag=f'mark{i}')
lc.text(IN_X0, IN_CY + IN_CH + 62, '由 logits [3.0, 2.5, 2.5, 1.0, 0.5, 0.0] 经 log_softmax 而来',
        8.5, lc.C_MUTE, 'start', maxw=440, tag='in:src')

# ---------------- 中：三个算子框 ----------------
OPX, OPW = 620, 280


def opbox(y, h, title, lines):
    lc.rect(OPX, y, OPW, h, '#ffffff', lc.C_GPU_S, rx=7, sw=1.5)
    lc.text(OPX + 14, y + 19, title, 10, lc.C_TXT, 'start', True, maxw=OPW - 26, tag='op:' + title[:10])
    for i, ln in enumerate(lines):
        lc.text(OPX + 14, y + 38 + i * 15, ln, 8.5, '#334155', 'start', maxw=OPW - 26,
                tag='opl:' + ln[:10])
    return y


OP1 = opbox(170, 92, '① torch.topk(logprobs, k=2)',
            ['摘走领奖台 k=2 格（值 + 下标）', '平手按下标排：id1 进榜、id2 落榜',
             '→ topk_indices = [0, 1]', '   topk_values = [-0.9084, -1.4084]'])
OP2 = opbox(292, 77, '② logprobs.gather(-1, token_ids)',
            ['按被采样 token 的下标取它自己的成绩', '三行被采样 = 0 / 3 / 2',
             '→ -0.9084 / -2.9084 / -1.4084'])
OP3 = opbox(399, 107, '③ batched_count_greater_than',
            ['(x >= v).sum(-1)：整行数一遍（含自身）', '不排序——O(V) 计数免 O(V log V) 排序',
             '→ 1-based 词表 rank（并列取上界）'])
# op3 迷你整行扫描图元：6 小格 + 波浪线 + sum
gy = OP3 + 96
gx = OPX + 24
for i in range(6):
    lc.rect(gx + i * 23, gy - 16, 20, 13, lc.C_GPU_F, lc.C_GPU_S, rx=2, sw=0.8)
wave = f'M{gx - 2},{gy - 22} ' + ' '.join(
    f'q5.75,-7 11.5,0 q5.75,7 11.5,0' for _ in range(6))
lc.ELEMS.append(((gx - 6, gy - 32, gx + 6 * 23 + 6, gy - 10),
                 f'<path d="{wave}" fill="none" stroke="{lc.C_GPU_S}" stroke-width="1.2"/>'))
lc.text(gx + 6 * 23 + 14, gy - 8, '→ sum', 8.5, lc.C_GPU_S, 'start', True, maxw=50, tag='sum')

# 输入行 → 三算子的总线箭头（端点全部贴框边）
BUS_X = 587
IN_R = IN_X0 + 6 * IN_CW + 5 * IN_CG          # 输入行右缘 = 554
for ty, oy_mid in ((OP1 + 46, IN_CY + IN_CH / 2), (OP2 + 38, IN_CY + IN_CH / 2),
                   (OP3 + 53, IN_CY + IN_CH / 2)):
    lc.parrow([(IN_R, oy_mid), (BUS_X, oy_mid), (BUS_X, ty), (OPX, ty)], lc.C_GPU_S, 1.8, 'dn')

# ---------------- 右：cat + 产出表 ----------------
CAT = (966, 150, BXR - 966, 48)
lc.rect(*CAT[:2], CAT[2], CAT[3], '#ffffff', lc.C_SAM_S, rx=8, sw=1.8)
lc.text(CAT[0] + CAT[2] / 2, CAT[1] + 19, 'torch.cat((token_ids, topk_indices), dim=1)', 10,
        lc.C_SAM_S, 'middle', True, maxw=CAT[2] - 20, tag='cat:t')
lc.text(CAT[0] + CAT[2] / 2, CAT[1] + 37, '把被采样者钉在列 0——列序是下游一切「第 0 个 = 被采样」不变式的物理起点',
        8.5, lc.C_MUTE, 'middle', maxw=CAT[2] - 20, tag='cat:s')

OUT = (966, 210, BXR - 966, 176)
lc.rect(*OUT[:2], OUT[2], OUT[3], '#f8fafc', lc.C_GPU_S, rx=8, sw=1.2)
lc.text(OUT[0] + OUT[2] / 2, OUT[1] + 18, '产出三件对齐同一张表：indices [3, 3] · logprobs [3, 3] · ranks [3]',
        9, lc.C_TXT, 'middle', True, maxw=OUT[2] - 20, tag='out:t')

LBL_X, C0X, CCW, CCG, RANK_X, RANK_W = 1092, 1108, 52, 4, 1358, 58
COLHDRS = ['列 0=被采样', 'top1', 'top2', 'rank']
COLX = [C0X + j * (CCW + CCG) for j in range(3)] + [RANK_X]
HDR_W = [58, CCW, CCW, RANK_W]
HDR_DX = [-3, 0, 0, 0]      # col0 表头略加宽、保持居中
for j, hd in enumerate(COLHDRS):
    hot = (j == 0)
    lc.rect(COLX[j] + HDR_DX[j], 234, HDR_W[j], 15,
            lc.C_SAM_F if hot else '#ffffff', lc.C_SAM_S if hot else lc.C_MUTE, rx=4, sw=1.0)
    lc.text(COLX[j] + HDR_DX[j] + HDR_W[j] / 2, 245.5, hd, 8,
            lc.C_SAM_S if hot else lc.C_MUTE, 'middle', hot, maxw=HDR_W[j] - 4,
            tag='ch' + hd)
ROWS = [
    ('行 0 · top1 本尊', [('0', '-0.9084'), ('0', '-0.9084'), ('1', '-1.4084')], 'rank 1'),
    ('行 1 · 落榜者', [('3', '-2.9084'), ('0', '-0.9084'), ('1', '-1.4084')], 'rank 4'),
    ('行 2 · 并列平手', [('2', '-1.4084'), ('0', '-0.9084'), ('1', '-1.4084')], 'rank 3'),
]
for i, (lbl, cells, rk) in enumerate(ROWS):
    y = 258 + i * 40
    lc.text(LBL_X, y + 21, lbl, 8.5, lc.C_TXT, 'end', maxw=130, tag='rl' + lbl[:4])
    for j, (tid, lp) in enumerate(cells):
        hot = (j == 0)
        lc.rect(COLX[j], y, CCW, 34, lc.C_SAM_F if hot else '#ffffff',
                lc.C_SAM_S if hot else lc.C_GPU_S, rx=4, sw=1.4 if hot else 1.0)
        lc.text(COLX[j] + CCW / 2, y + 14, tid, 11, lc.C_TXT, 'middle', hot, maxw=30,
                tag=f'c{i}{j}')
        lc.text(COLX[j] + CCW / 2, y + 28, lp, 7.5, lc.C_MUTE, 'middle', maxw=48,
                tag=f'cl{i}{j}')
    lc.rect(RANK_X, y + 5, RANK_W, 24, '#fff7ed', lc.C_ENG_S, rx=12, sw=1.2)
    lc.text(RANK_X + RANK_W / 2, y + 21, rk, 9, lc.C_ENG_S, 'middle', True, maxw=50, tag=rk)

# 三算子 → 右侧：topk/gather 进 cat、count 绕过 cat 直供 rank 列
lc.parrow([(OPX + OPW, OP1 + 30), (933, OP1 + 30), (933, CAT[1] + 14), (CAT[0], CAT[1] + 14)],
          lc.C_GPU_S, 1.8, 'dn')
lc.parrow([(OPX + OPW, OP2 + 24), (948, OP2 + 24), (948, CAT[1] + 34), (CAT[0], CAT[1] + 34)],
          lc.C_GPU_S, 1.8, 'dn')
lc.parrow([(OPX + OPW, OP3 + 53), (948, OP3 + 53), (948, OUT[1] + 58), (OUT[0], OUT[1] + 58)],
          lc.C_ENG_S, 1.8, 'up')
lc.alabel(902, OP1 + 26, '下标 + 值', 8, lc.C_GPU_S, 'start')
lc.alabel(902, OP2 + 14, '被采样者成绩', 8, lc.C_GPU_S, 'start')
lc.alabel(902, OP3 + 43, '词表 rank', 8, lc.C_ENG_S, 'start')
lc.seg(CAT[0] + CAT[2] / 2, CAT[1] + CAT[3], CAT[0] + CAT[2] / 2, OUT[1], lc.C_GPU_S, 1.8, 'dn')

# ---------------- 底部：两条气泡注 ----------------
BB_Y, BB_H = 514, 82
BUB = [
    ('行 1 · 落榜者也要交代',
     ['被采样 token3 排名 4、不在领奖台：k=2 只带 2 个候选，但它必须给——',
      '列 0 恒被采样、张量恒 k+1=3 列，-2.9084 照样给']),
    ('行 2 · 并列取上界',
     ['id1 / id2 同 2.5：计数把并列都算上 → rank 3（上界，不是排序名次 2）；',
      'topk 平手按下标排——id1 进榜、id2 只剩列 0']),
]
bw = (BXR - MX - 20) / 2
for i, (t, lines) in enumerate(BUB):
    x = MX + i * (bw + 20)
    lc.rect(x, BB_Y, bw, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
    lc.text(x + 14, BB_Y + 20, t, 9.5, lc.C_TXT, 'start', True, maxw=bw - 28, tag='bb' + t[:6])
    for k, ln in enumerate(lines):
        lc.text(x + 14, BB_Y + 40 + k * 17, ln, 8.5, '#334155', 'start', maxw=bw - 28,
                tag='bbl' + t[:6] + str(k))

# ---------------- 图例 + 下游 chip + 页脚 ----------------
LEG_Y = 626
lx = MX
for kind, name in [('sam', '列 0 = 被采样（恒钉在首位）'), ('gpu', 'topk 候选格 / logprob 格'),
                   ('eng', 'id1 / id2 并列（同值）'), ('rank', 'count 出的 1-based 词表 rank')]:
    if kind == 'rank':
        lc.rect(lx, LEG_Y - 10, 46, 16, '#fff7ed', lc.C_ENG_S, rx=8, sw=1.1)
        lx += 26
    else:
        lc.rect(lx, LEG_Y - 8, 22, 13,
                lc.C_SAM_F if kind == 'sam' else (lc.C_GPU_F if kind == 'gpu' else '#fff7ed'),
                lc.C_SAM_S if kind == 'sam' else (lc.C_GPU_S if kind == 'gpu' else lc.C_ENG_S),
                rx=3, sw=1.1)
    lc.text(lx + 28, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=240, tag='leg' + kind)
    lx += 28 + lc.tw(name, 9) + 20
_dn = '→ 下游 · L2 拍片 ④ D2H：[num_tok, k+1] 随采样 token 同一次搬运出 GPU'
_dw = lc.tw(_dn, 9) + 14
lc.rect(BXR - _dw, LEG_Y - 11, _dw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(BXR - _dw / 2, LEG_Y + 2, _dn, 9, lc.C_MUTE, 'middle', maxw=_dw - 4, tag='chip:dn')
lc.text(MX, 660, 'gather_logprobs + batched_count_greater_than verbatim '
        'vllm/v1/sample/sampler.py:L308-L356 / vllm/v1/sample/ops/logprobs.py:L11-L27 · '
        'V=6 / k=2、三行被采样 0 / 3 / 2 host 实测 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch08-fig-gather-triple.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
