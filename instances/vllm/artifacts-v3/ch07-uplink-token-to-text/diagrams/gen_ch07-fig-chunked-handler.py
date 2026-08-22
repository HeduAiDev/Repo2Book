#!/usr/bin/env python3
"""ch07 机制图 1 · output_handler 拉批分块（figure_spec ch07-fig-chunked-handler，模板 state-table）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的事件循环
时间线剖面——即本章 L2 章图 center 拍片 ② 『output_handler 拉批分块』+ south『why · 分块
让出事件循环』注的机制展开。架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：一批 300 条 EngineCoreOutput 被唯一的常驻 output_handler 任务按 128 切成 3 片
（128+128+44）逐片 process_outputs，片间 await asyncio.sleep(0) 恰好让出事件循环 2 次
——期间其他任务（SSE 写出/accept/add_request）得以插队。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点 + 外部基准 #12287）；
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 830
MX = 60
BXR = 1440
C_BODY = '#334155'
WARM_S = lc.C_ENG_S          # 让出缝 / 心跳拍（暖色点缀，图例声明）


def dot(cx, cy, r, fill):
    lc.ELEMS.append(((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2),
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"/>'))


# ---------------- 标题区 ----------------
lc.text(MX, 34, '柜员的呼吸节奏：一批 300 条按 128 切 3 片，片间让出事件循环',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'output_handler 单任务拉批分块——吞吐没变，变的是等待的分布：一次长停顿摊成 3 次短停顿',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 拍片 ② output_handler 拉批分块 · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 面板 1：事件循环时间线 ----------------
P1_Y, P1_H = 76, 300
lc.rect(MX, P1_Y, BXR - MX, P1_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, P1_Y + 26, '事件循环时间线：唯一的 output_handler 任务处理一批 300 条 EngineCoreOutputs', 11.5,
        lc.C_TXT, 'start', True, maxw=900, tag='p1:t')
lc.text(BXR - 16, P1_Y + 26, 'async_llm.py:L687-L703', 9, lc.C_FAINT, 'end', tag='p1:file')

# 心跳车道（时间带上方）：2 拍对齐 2 个让出缝
BAND_Y, BAND_H = 190, 60
# 片宽按条数比例：300 条铺 150..1400，缝各 36px
BX0, BX1 = 150, 1400
GAPW = 36
slice_px = (BX1 - BX0 - 2 * GAPW) * [128, 128, 44][0] // 300  # 128 条的像素宽
s1 = [BX0, BX0 + slice_px]
g1 = [s1[1], s1[1] + GAPW]
s2 = [g1[1], g1[1] + slice_px]
g2 = [s2[1], s2[1] + GAPW]
s3 = [g2[1], BX1]
gap_cx = [(g1[0] + g1[1]) / 2, (g2[0] + g2[1]) / 2]

lc.text(MX + 16, 140, '心跳任务（其他任务之一）批处理期间恰跑 2 拍', 9.5, lc.C_MUTE, 'start',
        maxw=330, tag='hb:t')
for i, cx in enumerate(gap_cx):
    dot(cx, 138, 5, WARM_S)
    lc.seg(cx, 145, cx, BAND_Y - 2, WARM_S, 1.2, dash=True)
lc.text(BXR - 16, 128, '拍点只能落在 sleep(0) 让出点——证明片间确实交还了事件循环', 9, lc.C_MUTE,
        'end', maxw=460, tag='hb:why')
lc.text(BXR - 16, 146, '（心跳 2 拍 = 让出点 2，host 单次实测）', 8.5, lc.C_FAINT, 'end',
        maxw=460, tag='hb:sub')

# 时间带：三片 + 两缝
SLICES = [(s1, '片 1 · 128 条 process_outputs', '128'),
          (s2, '片 2 · 128 条 process_outputs', '128'),
          (s3, '片 3 · 44 条 process_outputs', '44')]
for (sx, ex), t, n in SLICES:
    lc.rect(sx, BAND_Y, ex - sx, BAND_H, lc.C_API_S, lc.C_API_S, rx=6, sw=1.2)
    if ex - sx >= 300:
        lc.text((sx + ex) / 2, BAND_Y + 26, t, 11, '#ffffff', 'middle', True, maxw=ex - sx - 16,
                tag='sl:' + n)
        lc.text((sx + ex) / 2, BAND_Y + 46, '（不让出）', 8.5, '#dbeafe', 'middle',
                maxw=ex - sx - 16, tag='sl:s' + n)
    else:   # 末片窄：短标题 + 方法名下沉
        lc.text((sx + ex) / 2, BAND_Y + 24, '片 3 · 44 条', 11, '#ffffff', 'middle', True,
                maxw=ex - sx - 12, tag='sl:' + n)
        lc.text((sx + ex) / 2, BAND_Y + 43, 'process_outputs', 8.5, '#dbeafe', 'middle',
                maxw=ex - sx - 12, tag='sl:m' + n)
        lc.text((sx + ex) / 2, BAND_Y + 56, '（不让出）', 8, '#dbeafe', 'middle',
                maxw=ex - sx - 12, tag='sl:s' + n)
for gx, gy in [(g1, BAND_Y), (g2, BAND_Y)]:
    for x in gx:
        lc.seg(x, gy, x, gy + BAND_H, WARM_S, 1.3, dash=True)
    lc.seg(gx[0], gy + BAND_H / 2, gx[1], gy + BAND_H / 2, WARM_S, 1.0, dash=True)
for i, cx in enumerate(gap_cx):
    lc.text(cx, BAND_Y + BAND_H + 18, 'await asyncio.sleep(0)', 8.5, WARM_S, 'middle', True,
            maxw=130, tag='gap' + str(i))
    lw = 258
    lc.rect(cx - lw / 2, BAND_Y + BAND_H + 30, lw, 22, lc.C_ENG_F, WARM_S, rx=5, sw=1.0, dash=True)
    lc.text(cx, BAND_Y + BAND_H + 45, '让出点：SSE 写出 / accept / add_request 插队', 8.5,
            lc.C_TXT, 'middle', maxw=lw - 8, tag='gapc' + str(i))
lc.text(BXR - 16, BAND_Y + BAND_H + 18, '片宽按条数比例 · 缝隙不代表时长', 9, lc.C_FAINT, 'end',
        maxw=300, tag='axis:n')

# ---------------- 面板 2：分块 vs 不分块对照 ----------------
P2_Y, P2_H = 396, 178
lc.rect(MX, P2_Y, BXR - MX, P2_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, P2_Y + 24, '对照：若不分块——其他任务一起停 vs 分块后得以插队', 11, lc.C_TXT,
        'start', True, maxw=640, tag='p2:t')
# 不分块灰条
gb_y, gb_h = P2_Y + 38, 40
lc.rect(BX0, gb_y, BX1 - BX0, gb_h, lc.C_MUTE, lc.C_MUTE, rx=6, sw=1.0)
lc.text((BX0 + BX1) / 2, gb_y + 25, '整批 300 一口气 · 让出点 0 —— 全部连接的 SSE 写出 / accept / add_request 一起停',
        10, '#ffffff', 'middle', True, maxw=BX1 - BX0 - 20, tag='p2:bar')
lc.text(140, gb_y + 25, '若不分块', 9.5, lc.C_MUTE, 'end', maxw=80, tag='p2:l1')
# 分块后迷你条
mb_y, mb_h = gb_y + gb_h + 22, 30
for (sx, ex), t, n in SLICES:
    lc.rect(sx, mb_y, ex - sx, mb_h, lc.C_API_F, lc.C_API_S, rx=4, sw=1.1)
for gx in (g1, g2):
    for x in gx:
        lc.seg(x, mb_y, x, mb_y + mb_h, WARM_S, 1.2, dash=True)
    dot((gx[0] + gx[1]) / 2, mb_y + mb_h / 2, 4, WARM_S)
lc.text(140, mb_y + 20, '分块后', 9.5, lc.C_API_S, 'end', maxw=80, tag='p2:l2')
lc.text((BX0 + BX1) / 2, mb_y + mb_h + 16, '让出点 2 —— 一次长停顿摊成 3 次短停顿，等待被摊开',
        9.5, lc.C_MUTE, 'middle', maxw=640, tag='p2:cap')

# ---------------- 切片账 chips ----------------
CH_Y, CH_H = 596, 46
chips = [
    ('切片账：[128, 128, 44] · 3 片 = ceil(300/128)', 'host 实测切片账'),
    ('chunk 默认 128（VLLM_V1_OUTPUT_PROC_CHUNK_SIZE）', 'envs.py:L160'),
    ('三片返回列表全空——输出全走 collector', 'async_llm.py:L698-L699'),
    ('心跳拍 2 = 让出点 2（host 单次实测）', '让出判定 async_llm.py:L702-L703'),
]
cw = (BXR - MX - 3 * 14) / 4
for i, (t, s) in enumerate(chips):
    x = MX + i * (cw + 14)
    lc.rect(x, CH_Y, cw, CH_H, '#ffffff', lc.C_API_S, rx=7, sw=1.3)
    lc.text(x + cw / 2, CH_Y + 19, t, 9.5, lc.C_TXT, 'middle', True, maxw=cw - 14, tag='ch' + str(i))
    lc.text(x + cw / 2, CH_Y + 36, s, 8, lc.C_MUTE, 'middle', maxw=cw - 14, tag='chs' + str(i))

# ---------------- 外部基准条 ----------------
BM_Y, BM_H = 662, 64
lc.rect(MX, BM_Y, BXR - MX, BM_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 16, BM_Y + 22, '外部基准 · #12287（A100 · Llama-3.2-1B · 6000 请求）——分块让出的收益',
        10, lc.C_ENG_S, 'start', True, maxw=760, tag='bm:t')
lc.text(MX + 16, BM_Y + 46, 'mean TTFT 229.23→197.34 ms（−14%） · p99 TPOT 68.90→47.38 ms（−31%） · req/s 63.62→67.72（+6.4%）',
        9.5, lc.C_TXT, 'start', maxw=1150, tag='bm:d')
lc.text(BXR - 16, BM_Y + 22, '外部', 9, lc.C_MUTE, 'end', tag='bm:tag')
lc.text(BXR - 16, BM_Y + 46, '引自 issue 实测，非本章环境', 8, lc.C_FAINT, 'end', maxw=280,
        tag='bm:tag2')

# ---------------- 边界注 + 图例 + 页脚 ----------------
lc.text(MX, 748, '边界：sleep(0) 只让出同级就绪任务——detokenize 仍是单核上限：分块摊薄延迟、不扩吞吐',
        9.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='bound')
LEG_Y = 778
lx = MX
items = [('block', None, '深蓝块 = process_outputs 片（不让出）'),
         ('dashline', None, '虚线缝 = sleep(0) 让出点'),
         ('dot', None, '● = 心跳拍'),
         ('grey', None, '灰条 = 不分块对照')]
for kind, _, name in items:
    if kind == 'block':
        lc.rect(lx, LEG_Y - 8, 22, 12, lc.C_API_S, lc.C_API_S, rx=3, sw=1.0)
    elif kind == 'dashline':
        lc.seg(lx, LEG_Y - 2, lx + 24, LEG_Y - 2, WARM_S, 1.4, dash=True)
    elif kind == 'dot':
        dot(lx + 5, LEG_Y - 2, 4, WARM_S)
        lx += 2
    else:
        lc.rect(lx, LEG_Y - 8, 22, 12, lc.C_MUTE, lc.C_MUTE, rx=3, sw=1.0)
    lc.text(lx + 30, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=250, tag='leg' + name)
    lx += 30 + lc.tw(name, 9) + 24
lc.text(MX, 806, '切片账 / 心跳计数 host 单次实测（心跳 2 为本次运行值，正文引切片数学与布尔判定） · 分块编排 verbatim async_llm.py:L691-L703 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-chunked-handler.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
