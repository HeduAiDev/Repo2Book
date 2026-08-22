#!/usr/bin/env python3
"""ch07 机制图 4 · Slow 路径双 offset 滑窗（figure_spec ch07-fig-double-offset-window，模板 state-table）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的去 token 化
工位慢线——即本章 L2 章图 south『detokenizer 三路工厂』组件的 Slow 展开与 center 拍片 ④
的 decode_next 支线。架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：Slow 路径的双 offset 滑窗把每步 decode 限制在 [prefix:] 尾窗（实测首步触达 6、
稳态恒 2——上下文 1 token + 新 1 token），而朴素全量重解是整个序列（本例 12）——增量由
decode[窗口] 与 decode[窗口+新] 相减得出，窗口只为给 cleanup 算法相邻上下文。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 862
MX = 60
BXR = 1440
P_S, R_S = lc.C_API_S, lc.C_ENG_S     # prefix 游标（蓝）/ read 游标（橙，图例声明）


def dot(cx, cy, r, fill):
    lc.ELEMS.append(((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2),
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"/>'))


# ---------------- 标题区 ----------------
lc.text(MX, 34, '慢线的滑动玻璃窗：每步只重读尾窗，稳态触达恒 2 个 token（上下文 1 + 新 1）',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, 'decode[窗内旧段] 与 decode[窗内旧段 + 新 token] 相减得增量——窗口只为给 cleanup 算法相邻上下文',
        10.5, lc.C_MUTE, 'start', maxw=880, tag='subtitle')
_ch = '放大自 L2 south『detokenizer 三路工厂』Slow 展开 · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- token 序列条 ----------------
BAR_X, CELL_W, CELL_H = 140, 52, 44
BAR_Y = 176
prompt = list('abcdefghij')          # id 97-106
output = list('klmno')               # id 107-111
cells = prompt + output
for i, ch_ in enumerate(cells):
    x = BAR_X + i * CELL_W
    if i < 3:                        # 头 3 个不进转换表
        lc.rect(x, BAR_Y, CELL_W - 4, CELL_H, '#f1f5f9', lc.C_FAINT, rx=5, sw=1.0, dash=True)
    elif i < 10:                     # 其余 prompt
        lc.rect(x, BAR_Y, CELL_W - 4, CELL_H, lc.C_API_F, lc.C_MUTE, rx=5, sw=1.0)
    else:                            # 输出
        lc.rect(x, BAR_Y, CELL_W - 4, CELL_H, '#ffffff', lc.C_API_S, rx=5, sw=1.2)
    lc.text(x + (CELL_W - 4) / 2, BAR_Y + 18, ch_, 10.5, lc.C_TXT, 'middle', True, maxw=CELL_W - 8,
            tag='cell' + ch_)
    lc.text(x + (CELL_W - 4) / 2, BAR_Y + 34, str(97 + i), 7.5, lc.C_FAINT, 'middle',
            maxw=CELL_W - 8, tag='cid' + str(i))
lc.text(BAR_X - 8, BAR_Y + 24, 'prompt', 8.5, lc.C_MUTE, 'end', tag='bar:pl')
lc.text(BAR_X - 8, BAR_Y + 38, '10 个', 8, lc.C_MUTE, 'end', tag='bar:pl2')
lc.text(790, 164, '输出（每轮 1 个）', 8.5, lc.C_API_S, 'middle', True, maxw=150,
        tag='bar:ol')


def xc(c):        # 转换坐标 c（0 = "d"）→ 画布 x（格边界）
    return BAR_X + (3 + c) * CELL_W


# 转换坐标轴（d 起算 0..12）
for c in range(13):
    lc.seg(xc(c), BAR_Y + CELL_H, xc(c), BAR_Y + CELL_H + 6, lc.C_FAINT, 1.0)
    lc.text(xc(c), BAR_Y + CELL_H + 18, str(c), 7.5, lc.C_FAINT, 'middle', maxw=20,
            tag='ax' + str(c))
lc.seg(xc(0), BAR_Y + CELL_H + 22, xc(12), BAR_Y + CELL_H + 22, lc.C_FAINT, 1.0)
lc.text(xc(6), BAR_Y + CELL_H + 36, 'offset 计数轴（转换表内 token 序号，"d" = 0）', 8,
        lc.C_FAINT, 'middle', maxw=380, tag='ax:lbl')
lc.text(BAR_X + 76, BAR_Y + CELL_H + 36, '头 3 个不进转换表', 8, lc.C_FAINT, 'middle',
        maxw=150, tag='ax:head')

# ---------------- 游标梯（bar 上方，6 行） ----------------
LADDER = [('初始', 2, 7), ('轮 1 +k', 7, 8), ('轮 2 +l', 8, 9), ('轮 3 +m', 9, 10),
          ('轮 4 +n', 10, 11), ('轮 5 +o', 11, 12)]
row_y0, row_dy = 164, 17            # 初始行最贴近 bar，向上逐轮
for k, (lbl, p, r) in enumerate(LADDER):
    y = row_y0 - k * row_dy
    lc.text(BAR_X - 8, y - 2, lbl, 8.5, lc.C_TXT, 'end', maxw=80, tag='ld' + lbl)
    lc.seg(xc(p), y, xc(p), y - 10, P_S, 2.2)
    lc.seg(xc(r), y, xc(r), y - 10, R_S, 2.2)
    lc.text(xc(p) - 4, y - 12, str(p), 7.5, P_S, 'end', maxw=18, tag='tp' + str(p) + str(k))
    lc.text(xc(r) + 4, y - 12, str(r), 7.5, R_S, 'start', maxw=18, tag='tr' + str(r) + str(k))
# 梯左空区（初始行 p 游标在 x≈400，故 x<380 全空）：两条游标注
lc.text(BAR_X, 72, '初始后每轮：新 prefix = 旧 read', 8.5, P_S, 'start', True, maxw=230,
        tag='ld:rule')
lc.text(BAR_X, 90, '——从上次读完的地方重读一小段', 8.5, P_S, 'start', maxw=230, tag='ld:rule2')
lc.text(BAR_X, 112, '蓝竖标 = prefix 游标（重读上下文起点）', 8.5, lc.C_MUTE, 'start', maxw=230,
        tag='ld:leg')
lc.text(BAR_X, 130, '橙竖标 = read 游标（已确认读完处）', 8.5, lc.C_MUTE, 'start', maxw=230,
        tag='ld:leg2')

# ---------------- 轮次账表 ----------------
TBL_Y = 288
COLS = [('轮次', 150, 234), ('新 token', 234, 344), ('读前 p/r', 344, 452),
        ('窗口切片 [p:r]', 452, 622), ('decode 触达', 622, 722), ('增量', 722, 800),
        ('读后 p/r', 800, 920)]
for name, x0, x1 in COLS:
    lc.text((x0 + x1) / 2, TBL_Y, name, 8.5, lc.C_TXT, 'middle', True, maxw=x1 - x0 - 8,
            tag='th:' + name)
lc.seg(140, TBL_Y + 8, 920, TBL_Y + 8, lc.C_MUTE, 1.2)
ROWS = [
    ('初始', '—（构造时）', '2 / 7', '"fghij"（5 个）', '首步 6', '—', '（首步前）'),
    ('轮 1', "107（'k'）", '2 / 7', '"fghij"', '6（盖初始窗）', '"k"', '7 / 8'),
    ('轮 2', "108（'l'）", '7 / 8', '"k"', '2', '"l"', '8 / 9'),
    ('轮 3', "109（'m'）", '8 / 9', '"l"', '2', '"m"', '9 / 10'),
    ('轮 4', "110（'n'）", '9 / 10', '"m"', '2', '"n"', '10 / 11'),
    ('轮 5', "111（'o'）", '10 / 11', '"n"', '2', '"o"', '11 / 12'),
]
for ri, row in enumerate(ROWS):
    y = TBL_Y + 30 + ri * 23
    if ri % 2 == 1:
        lc.rect(140, y - 15, 780, 21, '#f8fafc', '#f8fafc', rx=3, sw=0.8)
    for (name, x0, x1), v in zip(COLS, row):
        lc.text((x0 + x1) / 2, y, v, 8.8, '#334155', 'middle', maxw=x1 - x0 - 6,
                tag='td' + str(ri) + name)
lc.text(140, TBL_Y + 30 + 6 * 23 + 4, '触达 = decode 实际触到的 token 数：首步 6 盖初始窗，此后恒 2（上下文 1 + 新 1）'
        '——与序列总长无关', 8.5, lc.C_MUTE, 'start', maxw=760, tag='tbl:n')

# ---------------- 右栏：触达账折线 ----------------
RP_X = 960
lc.text(RP_X + 16, 106, '触达账：窗口 vs 朴素全量重解', 10.5, lc.C_TXT, 'start', True,
        maxw=400, tag='ch:t')
PX0, PX1 = 1020, 1380
PY0, PY1 = 320, 140          # y(值)：PY0=0 线，值 12 → PY1


def gy(v):
    return PY0 - (PY0 - PY1) * v / 12


def gx(r):
    return PX0 + (PX1 - PX0) * (r - 1) / 4


lc.rect(PX0 - 40, PY1 - 18, PX1 - PX0 + 90, PY0 - PY1 + 40, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
for v in (0, 2, 6, 12):
    lc.seg(PX0 - 30, gy(v), PX1 + 10, gy(v), '#e2e8f0', 1.0)
    lc.text(PX0 - 44, gy(v) + 3, str(v), 8, lc.C_MUTE, 'end', maxw=20, tag='gy' + str(v))
for r in range(1, 6):
    lc.text(gx(r), PY0 + 16, '轮' + str(r), 8, lc.C_MUTE, 'middle', maxw=36, tag='gx' + str(r))
win_spans = [6, 2, 2, 2, 2]
naive_spans = [8, 9, 10, 11, 12]
lc.parrow([(gx(i + 1), gy(v)) for i, v in enumerate(win_spans)], P_S, 1.8, marker=None)
lc.parrow([(gx(i + 1), gy(v)) for i, v in enumerate(naive_spans)], '#94a3b8', 1.8, marker=None)
for i, v in enumerate(win_spans):
    dot(gx(i + 1), gy(v), 3.5, P_S)
for i, v in enumerate(naive_spans):
    dot(gx(i + 1), gy(v), 3.5, '#94a3b8')
lc.text(gx(1) + 6, gy(6) - 8, '6', 8.5, P_S, 'start', True, tag='sp:w1')
lc.text(gx(5) + 6, gy(12) - 2, '12', 8.5, '#94a3b8', 'start', True, tag='sp:n5')
lc.text(gx(2), gy(2) + 16, '稳态恒 2', 8.5, P_S, 'middle', True, maxw=90, tag='sp:w2')
lc.text(PX0 + 40, PY1 - 4, '蓝折线 = 窗口触达 · 灰折线 = 全量重解（= 序列全长）', 8.5, lc.C_MUTE,
        'middle', maxw=400, tag='ch:leg')
lc.text(RP_X + 16, 356, '序列长到 12：窗口 max 触达 6、稳态恒 2；朴素全量重解每步都在涨（末步 12）',
        8.8, '#334155', 'start', maxw=460, tag='ch:c')

# ---------------- 右栏：why + 初始窗 + Fast 注 ----------------
WY = 388
lc.rect(RP_X, WY, BXR - RP_X, 86, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(RP_X + 14, WY + 20, '窗口存在的唯一理由（docstring 原话：defeat cleanup algorithms）', 9.5,
        lc.C_TXT, 'start', True, maxw=450, tag='why:t')
lc.text(RP_X + 14, WY + 40, 'cleanup 算法看相邻 token 决定加不加空格——不给上下文它就在边界翻脸；'
        '但窗口只有一两块玻璃宽，绝不重读全文', 8.5, '#334155', 'start', maxw=450, tag='why:l')
lc.text(RP_X + 14, WY + 62, 'detokenizer_utils.py:L201-L203', 8, lc.C_FAINT, 'start', tag='why:f')
IY = WY + 100
lc.rect(RP_X, IY, BXR - RP_X, 86, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
lc.text(RP_X + 14, IY + 20, '初始窗（convert_prompt_ids_to_tokens）', 9.5, lc.C_TXT, 'start', True,
        maxw=450, tag='iw:t')
lc.text(RP_X + 14, IY + 40, 'prompt 10 个只转尾部 7 个（OFFSET 5+2）——转换表即 "defghij"；'
        'read = 7，prefix 再退 5 → 2', 8.5, '#334155', 'start', maxw=450, tag='iw:l')
lc.text(RP_X + 14, IY + 62, 'detokenizer_utils.py:L119-L140（注释：5 是对一切 tokenizer 都够用的保守值）',
        8, lc.C_FAINT, 'start', maxw=450, tag='iw:f')
FY = IY + 100
lc.rect(RP_X, FY, BXR - RP_X, 60, lc.C_API_F, lc.C_API_S, rx=7, sw=1.2)
lc.text(RP_X + 14, FY + 22, 'Fast 路径把这整套窗口下沉进 Rust DecodeStream', 9.5, lc.C_TXT,
        'start', True, maxw=450, tag='fn:t')
lc.text(RP_X + 14, FY + 42, 'Python 侧每 token 只剩一次 stream.step（见三岔分派图）', 8.5,
        lc.C_MUTE, 'start', maxw=450, tag='fn:l')

# ---------------- 底部小结 + 图例 + 页脚 ----------------
SM_Y = 660
lc.rect(MX, SM_Y, BXR - MX, 54, lc.C_API_F, lc.C_API_S, rx=8, sw=1.4)
lc.text(MX + 16, SM_Y + 22, '五轮增量 "k" / "l" / "m" / "n" / "o" 拼接 = "klmno"（增量无损：Σ 增量 == 全量重解 − prompt 部分）',
        9.5, lc.C_TXT, 'start', True, maxw=1100, tag='sm:t')
lc.text(MX + 16, SM_Y + 42, 'prompt 不计入输出：num_output_tokens = 5 · output_text = "klmno"（Slow 覆写计数减 prompt_len）',
        8.8, lc.C_MUTE, 'start', maxw=1100, tag='sm:s')
LEG_Y = SM_Y + 84
lx = MX
items = [('pcell', 'prompt token'), ('ocell', '输出 token'), ('grey', '头 3 个（不进转换表）'),
         ('ptick', 'prefix 游标'), ('rtick', 'read 游标')]
for kind, name in items:
    if kind == 'pcell':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_API_F, lc.C_MUTE, rx=4, sw=1.0)
    elif kind == 'ocell':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', lc.C_API_S, rx=4, sw=1.1)
    elif kind == 'grey':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#f1f5f9', lc.C_FAINT, rx=4, sw=1.0, dash=True)
    elif kind == 'ptick':
        lc.seg(lx + 3, LEG_Y - 9, lx + 3, LEG_Y + 4, P_S, 2.2)
        lx += 4
    else:
        lc.seg(lx + 3, LEG_Y - 9, lx + 3, LEG_Y + 4, R_S, 2.2)
        lx += 4
    lc.text(lx + 26, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=200, tag='leg' + name)
    lx += 26 + lc.tw(name, 9) + 20
lc.text(MX, LEG_Y + 28, '双 offset 推进 verbatim vllm/v1/engine/detokenizer.py:L292-L307 → vllm/tokenizers/detokenizer_utils.py:L176-L268 · '
        '初始窗 L119-L140 · 游标数值与触达计数 host 实测 · 行号基线 vLLM v0.27.1', 9,
        lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-double-offset-window.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
