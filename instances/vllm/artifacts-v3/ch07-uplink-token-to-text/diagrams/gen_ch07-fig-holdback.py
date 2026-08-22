#!/usr/bin/env python3
"""ch07 机制图 5 · 取文本 holdback（figure_spec ch07-fig-holdback，模板 before-after）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的取文本工位——
即本章 L2 章图 center 拍片 ⑤ 『取文本 holdback』的机制展开。架构归属回指 L2/L0。

claim：排他停止串下，get_next_output_text 每轮把尾部 max(len)-1=2 个字符扣在门内
（实测文本长 3/4/5 时只放行 1/1/1，前缀 "EN" 始终未流出），停止串凑齐的同一拍 update
截断回 "ABC"、终读恰为空串——消费者拼接 "ABC" 与截断后全文全等，零字节泄漏；
include=True 或无 stop 时扣留为 0。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 922
MX = 60
BXR = 1440
HELD_F, HELD_S = lc.C_ENG_F, lc.C_ENG_S      # 扣留段（暖色，图例声明）
CUT = lc.C_ABORT

# ---------------- 标题区 ----------------
lc.text(MX, 34, '门口的保安：排他停止串下，每轮扣住尾部 2 个字符——凑齐那拍同剪，零泄漏',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, 'get_next_output_text 的 delta 切片：未完成时只放行 len(output_text) − stop_buffer_length',
        10.5, lc.C_MUTE, 'start', maxw=880, tag='subtitle')
_ch = '放大自 L2 拍片 ⑤ 取文本 holdback · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 列头 ----------------
lc.text(460, 130, '门内 · output_text（update 后全量，截断前）', 9.5, lc.C_TXT, 'middle', True,
        maxw=420, tag='col:in')
lc.text(1180, 130, '门外 · 已流出（消费者拼接的流）', 9.5, lc.C_TXT, 'middle', True, maxw=380,
        tag='col:out')
lc.seg(60, 138, 1440, 138, '#e2e8f0', 1.2)

CW_, CH_, GAP = 46, 44, 6
IN_X = 200          # 门内格子起点
OUT_X = 920         # 门外格子起点
GATE_X = 835        # 闸口中轴


def cell(x, y, txt, kind):
    if kind == 'held':
        lc.rect(x, y, CW_, CH_, HELD_F, HELD_S, rx=6, sw=1.4)
    elif kind == 'cut':
        lc.rect(x, y, CW_, CH_, '#f8fafc', lc.C_FAINT, rx=6, sw=1.1, dash=True)
    else:
        lc.rect(x, y, CW_, CH_, lc.C_API_F, lc.C_API_S, rx=6, sw=1.3)
    lc.text(x + CW_ / 2, y + 28, txt, 11, lc.C_TXT, 'middle', True, maxw=CW_ - 6, tag='hc:' + txt)


ROUNDS = [
    dict(lbl='轮 1', inp='输入 [65,66,67]', inside=[('A', 'n'), ('B', 'h'), ('C', 'h')],
         held_lbl='扣留 2（= max(len(stop)) − 1）', note='3 字符，放行 1',
         delta='"A"', gate='未完成：只切到 len − 2（本轮 1 字符）', out=[('A', 'n')], out_note='累计 "A"'),
    dict(lbl='轮 2', inp='输入 [69]', inside=[('A', 'n'), ('B', 'n'), ('C', 'h'), ('E', 'h')],
         held_lbl='扣留 2', note='4 字符，放行 1', delta='"B"', gate='未完成：只切到 len − 2（本轮 1 字符）',
         out=[('A', 'n'), ('B', 'n')], out_note='累计 "AB"'),
    dict(lbl='轮 3', inp='输入 [78]', inside=[('A', 'n'), ('B', 'n'), ('C', 'n'), ('E', 'h'), ('N', 'h')],
         held_lbl='扣留 2 —— 停止串前缀 "EN" 押在门内', note='5 字符，放行 1',
         delta='"C"', gate='未完成：只切到 len − 2（本轮 1 字符）',
         out=[('A', 'n'), ('B', 'n'), ('C', 'n')], out_note='累计 "ABC"'),
    dict(lbl='轮 4', inp='输入 [68] → 命中 "END"',
         inside=[('A', 'n'), ('B', 'n'), ('C', 'n'), ('E', 'c'), ('N', 'c'), ('D', 'c')],
         held_lbl='截断：E N D 同拍剪掉（扣留 0）', note='截回 "ABC"（返回 stop_string "END"）',
         delta='""', gate='finished：放行全部 → 终读 ""',
         out=[('A', 'n'), ('B', 'n'), ('C', 'n')], out_note='累计 "ABC" == 截断后全文（零泄漏）'),
]
ry0, pitch = 162, 148
for k, r in enumerate(ROUNDS):
    ry = ry0 + k * pitch
    lc.text(180, ry + 30, r['lbl'], 10.5, lc.C_TXT, 'end', True, maxw=60, tag='rl' + r['lbl'])
    lc.text(180, ry + 47, r['inp'], 8, lc.C_MUTE, 'end', maxw=115, tag='ri' + r['lbl'])
    # 门内格子 + 扣留括注
    x = IN_X
    held_x0 = held_x1 = None
    cut_x = None
    for i, (t, kind) in enumerate(r['inside']):
        if kind == 'h' and held_x0 is None:
            held_x0 = x
        if kind == 'h':
            held_x1 = x + CW_
        if kind == 'c' and cut_x is None:
            cut_x = x - GAP / 2
        cell(x, ry, t, kind)
        x += CW_ + GAP
    if held_x0 is not None and r['inside'][-1][1] == 'h':
        lc.seg(held_x0, ry - 10, held_x1, ry - 10, HELD_S, 1.4)
        lc.seg(held_x0, ry - 10, held_x0, ry - 5, HELD_S, 1.4)
        lc.seg(held_x1, ry - 10, held_x1, ry - 5, HELD_S, 1.4)
    elif cut_x is not None:      # 轮 4：截断括注
        lc.seg(IN_X, ry - 10, cut_x, ry - 10, CUT, 1.4)
        lc.seg(IN_X, ry - 10, IN_X, ry - 5, CUT, 1.4)
        lc.seg(cut_x, ry - 10, cut_x, ry - 5, CUT, 1.4)
        lc.seg(cut_x, ry - 8, cut_x, ry + CH_ + 8, CUT, 1.6, dash=True)
        lc.text(cut_x - 4, ry - 12, '✂', 11, CUT, 'end', tag='sc' + r['lbl'])
    if held_x0 is not None or cut_x is not None:
        bx = held_x0 if held_x0 is not None and r['inside'][-1][1] == 'h' else IN_X
        lc.text(bx, ry - 16, r['held_lbl'], 8.5, HELD_S if held_x0 is not None else CUT,
                'start', True, maxw=400, tag='hl' + r['lbl'])
    lc.text(x + 12, ry + 28, r['note'], 8.5, lc.C_MUTE, 'start', maxw=240, tag='nt' + r['lbl'])
    # 闸口 + delta chip（箭头自 chip 底缘出发，送入门外格）
    lc.parrow([(GATE_X - 44, ry + 2), (GATE_X - 44, ry + CH_ / 2), (OUT_X - 2, ry + CH_ / 2)],
              lc.C_API_S, 2.0, 'dn')
    lc.rect(GATE_X - 44, ry - 22, 88, 24, '#ffffff', lc.C_API_S, rx=12, sw=1.3)
    lc.text(GATE_X, ry - 6, 'delta ' + r['delta'], 9, lc.C_API_S, 'middle', True, maxw=80,
            tag='dl' + r['lbl'])
    lc.text(GATE_X, ry + CH_ + 20, r['gate'], 8, lc.C_MUTE, 'middle', maxw=170,
            tag='gt' + r['lbl'])
    # 门外格子
    x = OUT_X
    for t, kind in r['out']:
        cell(x, ry, t, kind)
        x += CW_ + GAP
    lc.text(x + 12, ry + 28, r['out_note'], 8.5, lc.C_MUTE, 'start', maxw=300,
            tag='on' + r['lbl'])

# ---------------- 底部三盒：反事实 / include / 无 stop ----------------
BB_Y, BB_H = 756, 92
boxes = [
    ('反事实（对照推演）：若扣留为 0', '轮 3 门外将是 "ABCEN"——"CEN" 流出，"EN" 泄漏进用户流',
     '（停止串前缀本可躲在文末 2 字符里）', CUT),
    ('include_stop_str_in_output=True：扣留 0', '四轮流出 "ABC" / "E" / "N" / "D"，累计 "ABCEND"'
     '（停止串是交付物——不扣不截）', '', lc.C_API_S),
    ('无 stop：扣留 0', '两轮 "AB" / "C" 全量流出——没有排他停止串就不付这笔延迟', '',
     lc.C_API_S),
]
bw = (BXR - MX - 2 * 20) / 3
for i, (t, l1, l2, cc) in enumerate(boxes):
    x = MX + i * (bw + 20)
    lc.rect(x, BB_Y, bw, BB_H, '#ffffff', cc, rx=8, sw=1.3, dash=(i == 0))
    lc.text(x + 14, BB_Y + 22, t, 9.5, cc, 'start', True, maxw=bw - 28, tag='bb' + str(i))
    lc.text(x + 14, BB_Y + 44, l1, 8.5, '#334155', 'start', maxw=bw - 28, tag='bbl' + str(i))
    if l2:
        lc.text(x + 14, BB_Y + 64, l2, 8.5, lc.C_MUTE, 'start', maxw=bw - 28, tag='bbl2' + str(i))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BB_Y + BB_H + 34
lx = MX
items = [('cell', '门内保留 / 已流出'), ('held', '扣留中（未放行）'), ('cutcell', '同拍剪掉的字符'),
         ('cutline', '截断位'), ('chipbox', 'delta 返回值')]
for kind, name in items:
    if kind == 'cell':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_API_F, lc.C_API_S, rx=4, sw=1.2)
    elif kind == 'held':
        lc.rect(lx, LEG_Y - 8, 20, 13, HELD_F, HELD_S, rx=4, sw=1.3)
    elif kind == 'cutcell':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#f8fafc', lc.C_FAINT, rx=4, sw=1.0, dash=True)
    elif kind == 'cutline':
        lc.seg(lx, LEG_Y - 6, lx + 20, LEG_Y - 6, CUT, 1.5, dash=True)
    else:
        lc.rect(lx, LEG_Y - 11, 24, 15, '#ffffff', lc.C_API_S, rx=7, sw=1.1)
        lx += 4
    lc.text(lx + 28, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=200, tag='leg' + name)
    lx += 28 + lc.tw(name, 9) + 18
lc.text(MX, LEG_Y + 28, 'get_next_output_text 与扣留计算 verbatim vllm/v1/engine/detokenizer.py:L149-L165 / L85-L91 · '
        '四轮账、反事实、B/C 场景 host 实测（stop=["END"]，byte 级 tokenizer 为 seam）· 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-holdback.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
