#!/usr/bin/env python3
"""ch20 机制图 ⑤ · FA-2 循环序对调与三处榨取(figure_spec ch20-fig-fa2-loop-order,模板 before-after)

放大自 L0 中列『GPU 执行臂』(绿色列)第三块『模型层 forward + 编译』内 attention kernel 的
调度形态——今天 vLLM 真正在跑的 kernel(FA2/FA3/FA4)即 FA-2 变体。primer 推导链第 ⑥ 环。
架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:FA-2 三处工程榨取同一份数学——(1) 循环序对调,外层 Q 行块跨线程块并行、行块间零通信;
(2) O 中间不除 ℓ、收尾只除一次(避开贵 16 倍的非矩阵乘 FLOP);(3) 只存 logsumexp L=m+log(ℓ);
外加因果掩码整块跳过(1.6-1.78 倍工作量红利)。

数字全部取自 figure_spec.numbers(312 vs 19.5 TFLOPs/s;跳块计数 16→10 / 64→36(1.6/1.7778×);
论文区间 1.7-1.8× 趋近 2×;A100 230 TFLOPs/s = 73% 峰值;return_softmax_lse)。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 682
MX = 60
BXR = 1440
C_RED = '#dc2626'
EXTRA_DEFS = ('<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
              '<marker id="org" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#ea580c"/></marker>')

CELL, CGAP, NG = 28, 3, 8          # 格边长/格距/每边块数
GRID = NG * CELL + (NG - 1) * CGAP  # 245

# ---------------- 标题区 ----------------
lc.text(MX, 34, '同一份数学的三处工程榨取:FA-2 让 Q 行块各自承包,行块间零通信',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '循环序对调(外层 Q 行块跨线程块并行)· O 中间不除 ℓ、收尾只除一次 · 只存 logsumexp L=m+log(ℓ);因果掩码再送整块跳过(arXiv:2307.08691 §3.1-3.3)',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '推导链 ⑥ · 放大自 L0 GPU 执行臂内 kernel 调度形态——vLLM 在跑的 FA2/FA3/FA4 即此变体'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')


def cell_xy(gx, gy, r, c):
    return gx + c * (CELL + CGAP), gy + r * (CELL + CGAP)


# ---------------- 左:FA(FA-1)外层 KV 列块 ----------------
lc.text(MX, 96, 'FA(FA-1):外层 KV 列块,Q 小组排队上门', 12, lc.C_TXT, 'start', True,
        maxw=520, tag='lp:t')
LGX, LGY = 130, 130
# 64 块全访问:0-1 列已完成、第 2 列进行中、其余未到
for r in range(NG):
    for c in range(NG):
        x, y = cell_xy(LGX, LGY, r, c)
        if c <= 1:
            fill, stroke, dash = '#dcfcc7', lc.C_GPU_S, False
        elif c == 2:
            fill, stroke, dash = '#86efac', lc.C_GPU_S, False
        else:
            fill, stroke, dash = '#f1f5f9', '#cbd5e1', True
        lc.rect(x, y, CELL, CELL, fill, stroke, rx=3, sw=1.0, dash=dash)
# 外层 j 箭头(顶部,横向) + 内层 i 箭头(第 2 列,纵向)
lc.seg(LGX + 4, LGY - 14, LGX + GRID - 4, LGY - 14, lc.C_ENG_S, 2.2, 'org')
lc.text(LGX + GRID / 2, LGY - 22, '外层 j:KV 列块(K/V 搬一遍,逐列推进)', 8.5, lc.C_ENG_S,
        'middle', True, maxw=GRID + 120, tag='lp:outer')
lc.seg(LGX + 2 * (CELL + CGAP) + CELL / 2, LGY + 4,
       LGX + 2 * (CELL + CGAP) + CELL / 2, LGY + GRID - 4, lc.C_GPU_S, 2.2, 'grn')
lc.text(LGX + 2 * (CELL + CGAP) + CELL / 2 + 8, LGY + GRID / 2, '内层 i:', 8.5, lc.C_GPU_S,
        'start', True, maxw=70, tag='lp:inner1')
lc.text(LGX + 2 * (CELL + CGAP) + CELL / 2 + 8, LGY + GRID / 2 + 12, 'Q 行块', 8.5, lc.C_GPU_S,
        'start', maxw=70, tag='lp:inner2')
lc.text(LGX - 8, LGY - 8, '64 块全访问(因果也不跳)', 8.5, lc.C_MUTE, 'end', maxw=120,
        tag='lp:n64')
lc.text(LGX, LGY + GRID + 22, '每个 (i,j) 块算完都要 O ← O/ℓ 一次(除法散在循环里)', 9,
        lc.C_BEAT_T, 'start', True, maxw=560, tag='lp:div')
lc.text(LGX, LGY + GRID + 40, 'Q 小组轮流上门,KV 工位固定——长序列小 batch 时工位开不满', 9,
        lc.C_MUTE, 'start', maxw=560, tag='lp:note')

# 中间对调箭头
lc.parrow([(408, 253), (748, 253)], lc.C_MUTE, 2.2, 'std', dash=True)
lc.text(578, 243, '循环序对调(外层↔内层)', 9.5, lc.C_TXT, 'middle', True, maxw=200, tag='swap')

# ---------------- 右:FA-2 外层 Q 行块 ----------------
lc.text(760, 96, 'FA-2:外层 Q 行块,各自承包 + 因果整块跳过', 12, lc.C_TXT, 'start', True,
        maxw=640, tag='rp:t')
RGX, RGY = 800, 130
for r in range(NG):
    for c in range(NG):
        x, y = cell_xy(RGX, RGY, r, c)
        if c > r:                                   # 严格右上三角:整块跳过
            lc.rect(x, y, CELL, CELL, '#fef2f2', '#fca5a5', rx=3, sw=1.0)
            m = 6
            lc.seg(x + m, y + m, x + CELL - m, y + CELL - m, C_RED, 1.2)
            lc.seg(x + CELL - m, y + m, x + m, y + CELL - m, C_RED, 1.2)
        elif c == r:                                # 对角块:块内半掩码
            lc.rect(x, y, CELL, CELL, '#bbf7d0', lc.C_GPU_S, rx=3, sw=1.2)
        else:
            lc.rect(x, y, CELL, CELL, '#dcfcc7', lc.C_GPU_S, rx=3, sw=1.0)
# 外层 i 箭头(左缘,纵向)
lc.seg(RGX - 16, RGY + 4, RGX - 16, RGY + GRID - 4, lc.C_ENG_S, 2.2, 'org')
lc.text(RGX - 24, RGY + GRID / 2, '外层 i:Q 行块', 8.5, lc.C_ENG_S, 'end', True, maxw=90,
        tag='rp:outer')
# 内层 j 箭头(第 0/3/7 行已访问跨度内,横向)
for r in (0, 3, 7):
    span_end = RGX + r * (CELL + CGAP) + CELL - 4 if r >= 0 else RGX
    yrow = RGY + r * (CELL + CGAP) + CELL + 2
    lc.seg(RGX + 4, yrow, RGX + (r + 1) * (CELL + CGAP) - CGAP - 4, yrow, lc.C_GPU_S, 1.6,
           'grn')
lc.text(RGX + GRID + 8, RGY + 12, '内层 j:KV 列块', 8.5, lc.C_GPU_S, 'start', True, maxw=110,
        tag='rp:inner')
lc.text(RGX + GRID + 8, RGY + 26, '(沿行扫到对角)', 8, lc.C_MUTE, 'start', maxw=110,
        tag='rp:inner2')
lc.text(RGX + GRID + 8, RGY + 60, '64 → 36 块', 10, lc.C_BEAT_T, 'start', True, maxw=110,
        tag='rp:cnt')
lc.text(RGX + GRID + 8, RGY + 74, '28 块整块跳过', 8.5, '#334155', 'start', maxw=110,
        tag='rp:cnt2')
lc.text(RGX + GRID + 8, RGY + 88, '= 1.7778×', 9, lc.C_BEAT_T, 'start', True, maxw=110,
        tag='rp:cnt3')
# 行块独立标注(两行之间无箭头)
lc.text(RGX, RGY + GRID + 22, '行块各自承包从头干到尾,行块间零通信(embarrassingly parallel)', 9,
        lc.C_GPU_S, 'start', True, maxw=640, tag='rp:ind')
lc.text(RGX, RGY + GRID + 40, '收尾只除一次 ℓ(Tweak 1);只存 L = m + log(ℓ)(Tweak 2)', 9,
        lc.C_GPU_S, 'start', True, maxw=640, tag='rp:t12')

# ---------------- 底部三签 ----------------
SB_Y, SB_H = 480, 118
SBW = (BXR - MX - 2 * 20) / 3
chips = [
    ('① 非矩阵乘差价 16×(Tweak 1 的动机)', lc.C_ENG_S, [
        'A100 FP16/BF16 matmul:312 TFLOPs/s',
        '非 matmul FP32:19.5 TFLOPs/s',
        '——每个非矩阵乘 FLOP 贵 16 倍',
        'Tweak 1:O 中间不除 ℓ、收尾只除一次,',
        '省掉的是每块一次的除法',
    ]),
    ('② 只存一个标量 logsumexp(Tweak 2)', lc.C_GPU_S, [
        '不必同时存 max 与 sum:',
        'L^(j) = m^(j) + log(ℓ^(j))',
        'vLLM flash_attn_varlen_func 的',
        'return_softmax_lse 返回的正是它——',
        'cascade 拆段与一切合并的通用货币',
    ]),
    ('③ 因果整块跳过 + 综合实测', C_RED, [
        '列块全在行块之上的块直接不算:',
        'N=64、B=8:64→36(1.7778×)',
        'N=8、B=2:16→10(1.6×)',
        '论文区间 1.7-1.8×,N→∞ 趋近 2×',
        'FA-2 ≈ 2× FA:A100 230 TFLOPs/s(73% 峰值)',
    ]),
]
for si, (title, color, lines) in enumerate(chips):
    x0 = MX + si * (SBW + 20)
    lc.rect(x0, SB_Y, SBW, SB_H, '#ffffff', color, rx=8, sw=1.4)
    lc.text(x0 + 12, SB_Y + 20, title, 10, color, 'start', True, maxw=SBW - 24,
            tag=f'chip{si}:t')
    for li, ln in enumerate(lines):
        lc.text(x0 + 12, SB_Y + 38 + li * 16, ln, 8.3, '#334155', 'start', maxw=SBW - 20,
                tag=f'chip{si}:l{li}')

# ---------------- 页脚:图例 + 出处 ----------------
LY = SB_Y + SB_H + 22
lc.text(MX, LY, '图例:绿块 = 访问 · 浅绿 = 已完成 · 灰虚线 = 未到 · 红✗ = 因果整块跳过 · 中绿对角块 = 块内半掩码 · 橙箭头 = 外层循环 · 绿箭头 = 内层循环',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, LY + 20, 'warp 分工:split-K → split-Q,免 shared-memory 通信(§3.3);FA-2 Alg.1 line 10 的缩放方向按论文不变式修正(旧账乘 e^(m^(j−1)−m^(j)) 折算,原文印刷有误)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, LY + 38, '出处 arXiv:2307.08691 §3.1-§3.3 · 跳块计数取自论文忠实参考实现实跑(host,两版输出对标准注意力 allclose)· vllm/vllm_flash_attn/flash_attn_interface.py:L264-L268(return_softmax_lse)· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-fa2-loop-order.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
