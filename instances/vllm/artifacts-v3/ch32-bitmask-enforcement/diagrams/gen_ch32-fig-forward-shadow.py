#!/usr/bin/env python3
"""ch32 机制图 · CPU 活藏进 GPU 前向的影子（figure_spec ch32-fig-forward-shadow，模板 swimlane）

放大自 L0 循环框②③两拍的时序展开——调度列（CPU 算表）与 GPU 列（前向）的并行窗口
（L2 站 1 的机制放大），架构归属回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。
时序图画法守 FIGURE-SYSTEM §0 规约 9（UML 文法）：参与者=竖直生命线（顶部名牌）、
共享时间轴（gridlines 贯穿、元素按时间戳 y 对齐、活动条高=时长×比例尺）、消息=水平
直线、瞬时动作=骑线时刻标记+真实值标注；两段比例尺不同，各自标注并声明压缩比。

claim：execute_model(non_block=True) 发车即返后，CPU 在 GPU 前向的执行窗口内完成
整批掩码装配——future.result() 之前掩码活已干完，一步的等待里不含填表时间。

数字全部取自 figure_spec.numbers（时间线 ms：dispatch 0.0 / 掩码 0.797 起、3.309ms
完成（4.106 止）/ future.result 51.138 / sample_tokens 51.138 / update_from_output
51.165 / overlap_verified=true / 行 0 允许集 / 64 行）与编排面锚 core.py:L596-L604。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1580, 1190
MX, BXR = 60, 1540
L_CPU, L_GPU = 470.0, 1130.0        # 两条生命线 x
RULER_X = 190.0
GRID_X0, GRID_X1 = 196.0, 1420.0
TL_TOP, TL_END = 200.0, 966.0

# 两段比例尺：A 段 1ms=80px（0-5ms），B 段 1ms=6.8px（5-52ms）→ 压缩比 ≈ 11.8:1
BRK_T, BRK_Y0, BRK_Y1 = 5.0, 600.0, 640.0
SC_A, SC_B = 80.0, 6.8


def ty(t):
    """时间戳 → y（两段比例尺；BRK_Y0..BRK_Y1 为断轴带）。"""
    if t <= BRK_T:
        return TL_TOP + t * SC_A
    return BRK_Y1 + (t - BRK_T) * SC_B


def brk_marks(x):
    """断轴双斜线。"""
    for dy in (0, 10):
        lc.seg(x - 7, BRK_Y0 + 20 + dy - 10, x + 7, BRK_Y0 + 20 + dy - 2, lc.C_MUTE, 1.6)


DEFS = lc.DEFS + (
    f'<marker id="gpu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>'
    f'<pattern id="sim" width="8" height="8" patternUnits="userSpaceOnUse" '
    f'patternTransform="rotate(45)"><rect width="8" height="8" fill="{lc.C_GPU_F}"/>'
    f'<line x1="0" y1="0" x2="0" y2="8" stroke="{lc.C_GPU_S}" stroke-width="1.4"/></pattern>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'CPU 活藏进 GPU 前向的影子：发车即返之后、future.result() 之前，掩码 3.309ms 全部干完',
        16.5, lc.C_TXT, 'start', True, maxw=1200, tag='title')
lc.text(MX, 58, '两段式 API 形态买的正是这段重叠——若串行排在采样前，这 3.309ms 就是每步白加的 CPU 时间；'
               '时间轴 0-5ms 段 1ms=80px、5-52ms 段 1ms=6.8px（压缩比 ≈12:1，断轴双斜线处切换）',
        10.5, lc.C_MUTE, 'start', maxw=1370, tag='subtitle')
_ch = '放大自 L0 循环框②③两拍时序 · L2 站 1 的机制放大'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道头 + 生命线 ----------------
HDR_Y, HDR_H = 88, 36
LANES = [
    (L_CPU, 'CPU · EngineCore（调度进程）', 'step() 编排面：算表在 L597', lc.C_ENG_F, lc.C_ENG_S),
    (L_GPU, 'GPU · worker 进程', '前向执行（活动条=时长×比例尺）', lc.C_GPU_F, lc.C_GPU_S),
]
for cx, nm, sub, f_, s_ in LANES:
    lc.rect(cx - 150, HDR_Y, 300, HDR_H, f_, s_, rx=7, sw=1.8)
    lc.text(cx, HDR_Y + 15, nm, 10.5, s_, 'middle', True, maxw=290, tag='lane:' + nm[:6])
    lc.text(cx, HDR_Y + 29, sub, 7.8, lc.C_MUTE, 'middle', maxw=290, tag='lane:s' + nm[:6])
    lc.seg(cx, HDR_Y + HDR_H + 2, cx, 994, lc.C_FAINT, 1.0, dash=True)

# ---------------- 共享时间轴（两段比例尺 + 断轴） ----------------
lc.seg(RULER_X, TL_TOP - 16, RULER_X, TL_END, lc.C_MUTE, 1.2)
for t in [0, 1, 2, 3, 4, 5]:
    y = ty(t)
    lc.seg(RULER_X - 4, y, RULER_X + 4, y, lc.C_MUTE, 1.0)
    lc.text(RULER_X - 8, y + 3, f'{t}', 8.2, lc.C_MUTE, 'end', maxw=40, tag='tick' + str(t))
    lc.seg(GRID_X0, y, GRID_X1, y, '#e2e8f0', 0.9, dash=True)
for t in [10, 20, 30, 40, 50]:
    y = ty(t)
    lc.seg(RULER_X - 4, y, RULER_X + 4, y, lc.C_MUTE, 1.0)
    lc.text(RULER_X - 8, y + 3, f'{t}', 8.2, lc.C_MUTE, 'end', maxw=40, tag='tick' + str(t))
    lc.seg(GRID_X0, y, GRID_X1, y, '#e2e8f0', 0.9, dash=True)
lc.text(RULER_X - 8, TL_TOP - 26, 'ms', 8.2, lc.C_MUTE, 'end', maxw=40, tag='ruler:unit')
lc.text(RULER_X - 8, TL_TOP - 40, '时间 ↓', 8.2, lc.C_MUTE, 'end', maxw=52, tag='ruler:cap')
brk_marks(RULER_X)
lc.text(RULER_X - 8, (BRK_Y0 + BRK_Y1) / 2 + 3, '断轴 ≈12:1', 7.6, lc.C_MUTE, 'end',
        maxw=70, tag='ruler:brk')

# 事件时刻小刻（品红短刻；具体 ms 值就近标在活动条/消息上，避免与整秒刻度相撞）
for t in [0.797, 4.106, 51.138, 51.165]:
    y = ty(t)
    lc.seg(RULER_X - 3, y, RULER_X + 3, y, lc.C_SAM_S, 1.4)

# ---------------- GPU 前向活动条（模拟窗口，斜线纹理） ----------------
FW_Y0, FW_Y1 = ty(0.0), ty(50.0)
lc.ELEMS.append(((L_GPU - 12, FW_Y0 - 2, L_GPU + 12, FW_Y1 + 2),
                 f'<rect x="{L_GPU - 10:.1f}" y="{FW_Y0:.1f}" width="20" height="{FW_Y1 - FW_Y0:.1f}" '
                 f'fill="url(#sim)" stroke="{lc.C_GPU_S}" stroke-width="1.4" rx="3"/>'))
lc.text(L_GPU + 18, FW_Y0 + 18, '前向执行窗口 50ms（模拟）', 9.2, lc.C_GPU_S, 'start', True,
        maxw=280, tag='fw:t')
lc.text(L_GPU + 18, FW_Y0 + 33, 'spy 后台线程模拟——真实前向 host 未测、量级几十毫秒；', 7.8,
        lc.C_MUTE, 'start', maxw=300, tag='fw:l1')
lc.text(L_GPU + 18, FW_Y0 + 46, '掩码时长为真 xgrammar 实测（取证环境差异，图注须挑明）', 7.8,
        lc.C_MUTE, 'start', maxw=300, tag='fw:l2')

# ---------------- CPU 掩码装配活动条（真实测，品红） ----------------
BM_Y0, BM_Y1 = ty(0.797), ty(4.106)
lc.ELEMS.append(((L_CPU - 12, BM_Y0 - 2, L_CPU + 12, BM_Y1 + 2),
                 f'<rect x="{L_CPU - 9:.1f}" y="{BM_Y0:.1f}" width="18" height="{BM_Y1 - BM_Y0:.1f}" '
                 f'fill="{lc.C_SAM_S}" stroke="{lc.C_SAM_S}" stroke-width="1.2" rx="3"/>'))
lc.text(L_CPU + 16, BM_Y0 + 16, 'get_grammar_bitmask：64 行掩码装配', 9.2, lc.C_SAM_S, 'start', True,
        maxw=400, tag='bm:t')
lc.text(L_CPU + 16, BM_Y0 + 32, '0.797ms 起 → 3.309ms 干完（4.106 止）——真 xgrammar 实测', 8,
        '#334155', 'start', maxw=400, tag='bm:l1')
lc.text(L_CPU + 16, BM_Y0 + 47, '行 0 允许集 [77, 88, 3919, 5948, 8505]（choice yes|no 前缀闭包，64 行同构）',
        8, lc.C_MUTE, 'start', maxw=420, tag='bm:l2')

# CPU 空闲段（表已填完、等 result）
lc.text(L_CPU + 16, (BM_Y1 + ty(51.138)) / 2, 'CPU 空闲——表已填完，等 future.result()', 8.2,
        lc.C_MUTE, 'start', maxw=320, tag='idle')

# ---------------- 消息（水平直线） ----------------
# t=0 dispatch
Y0 = ty(0.0)
lc.seg(L_CPU, Y0, L_GPU, Y0, lc.C_ENG_S, 2.2, 'dn')
lc.text((L_CPU + L_GPU) / 2, Y0 - 22, 'execute_model(non_block=True)　发车即返 Future', 9.2,
        lc.C_ENG_S, 'middle', True, maxw=420, tag='msg:dispatch')
lc.text((L_CPU + L_GPU) / 2, Y0 - 8, 'core.py:L596', 7.6, lc.C_FAINT, 'middle', maxw=120, tag='msg:dispatch:a')

# t=51.138 future.result（GPU→CPU 上行）
YR = ty(51.138) - 30
lc.seg(L_GPU, YR, L_CPU, YR, lc.C_ENG_S, 2.2, 'up')
lc.text((L_CPU + L_GPU) / 2, YR - 22, 'future.result() → None（51.138）', 9.2, lc.C_ENG_S,
        'middle', True, maxw=380, tag='msg:result')
lc.text((L_CPU + L_GPU) / 2, YR - 8, 'core.py:L602', 7.6, lc.C_FAINT, 'middle', maxw=120, tag='msg:result:a')

# t=51.138 sample_tokens（CPU→GPU）
YS = ty(51.138) + 6
lc.seg(L_CPU, YS, L_GPU, YS, lc.C_ENG_S, 2.2, 'dn')
lc.text((L_CPU + L_GPU) / 2, YS + 14, 'sample_tokens(grammar_output)（51.138）· core.py:L603-L604', 9.2,
        lc.C_ENG_S, 'middle', True, maxw=460, tag='msg:sample')

# t=51.165 update_from_output：骑线时刻标记
YU = ty(51.165) + 28
lc.ELEMS.append(((L_CPU - 6, YU - 6, L_CPU + 6, YU + 6),
                 f'<circle cx="{L_CPU}" cy="{YU}" r="4.5" fill="{lc.C_ENG_S}"/>'))
lc.text(L_CPU + 14, YU + 3, 'update_from_output（51.165）——瞬时动作，骑线时刻标记', 8.2,
        lc.C_ENG_S, 'start', maxw=400, tag='msg:update')

# ---------------- 重叠证据（4.106 处横向证据线） ----------------
YE = ty(4.106)
lc.seg(GRID_X0 + 6, YE, L_GPU - 16, YE, lc.C_SAM_S, 1.4, dash=True)
lc.seg(L_GPU + 16, YE, GRID_X1 - 40, YE, lc.C_SAM_S, 1.4, dash=True)
lc.text(GRID_X1 - 46, YE + 3, '掩码 4.106 返回 ⊂ 前向窗口 50ms', 8.2, lc.C_SAM_S, 'end', True,
        maxw=250, tag='ov:t')
lc.text(GRID_X1 - 46, YE + 17, 'overlap_verified=true——一步的等待里不含填表时间', 8,
        '#334155', 'end', maxw=270, tag='ov:s')

# ---------------- 底部两注 ----------------
NY, NH = 996, 118
lc.rect(MX, NY, 720, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 16, NY + 20, '这段重叠是两段式 API 形态买来的', 9.5, lc.C_TXT, 'start', True,
        maxw=680, tag='n1:t')
for j, ln in enumerate(['· 编排面四行（core.py:L593-L614 实录）：L596 发车（non_block）→ L597 趁 GPU 在算',
                        '  立刻 get_grammar_bitmask → L602 future.result() → L603-L604 返回 None 则 sample_tokens',
                        '· 若串行排在采样前：3.309ms 就是每步白加的 CPU 时间——重叠把它藏进前向影子里']):
    lc.text(MX + 16, NY + 40 + j * 16, ln, 8.2, '#334155', 'start', maxw=690, tag='n1:l' + str(j))
lc.rect(800, NY, BXR - 800, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(816, NY + 20, '取证口径（读者须知）', 9.5, lc.C_TXT, 'start', True, maxw=690, tag='n2:t')
for j, ln in enumerate(['· 时间线为 host 单次运行实测：掩码装配 64 行=真 xgrammar；前向窗口 50ms 为',
                        '  spy 后台线程模拟（真实前向量级几十毫秒，host 未测）',
                        '· 计时只作数量级证据，不引为生产毫秒数；gpt2 词表 50257、xgrammar 0.2.6']):
    lc.text(816, NY + 40 + j * 16, ln, 8.2, '#334155', 'start', maxw=BXR - 816 - 12, tag='n2:l' + str(j))

# ---------------- 图例 + 页脚 ----------------
LY = 1140
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, 'CPU · EngineCore 生命线', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg1')
lx0 += 21 + lc.tw('CPU · EngineCore 生命线', 8.8) + 16
lc.rect(lx0, LY - 9, 16, 11, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, 'GPU · worker 生命线', 8.8, lc.C_TXT, 'start', maxw=180, tag='lg2')
lx0 += 21 + lc.tw('GPU · worker 生命线', 8.8) + 16
lc.rect(lx0, LY - 9, 16, 11, lc.C_SAM_S, lc.C_SAM_S, rx=3, sw=1.2)
lc.text(lx0 + 21, LY + 1, '掩码装配活动（真实测）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg3')
lx0 += 21 + lc.tw('掩码装配活动（真实测）', 8.8) + 16
lc.rect(lx0, LY - 11, 22, 15, 'url(#sim)', lc.C_GPU_S, rx=3, sw=1.2)
lc.text(lx0 + 27, LY + 1, '前向窗口（斜线纹理=模拟）', 8.8, lc.C_TXT, 'start', maxw=220, tag='lg4')
lx0 += 27 + lc.tw('前向窗口（斜线纹理=模拟）', 8.8) + 16
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_ENG_S, 2.0, 'dn')
lc.text(lx0 + 40, LY + 1, '消息（水平直线）', 8.8, lc.C_TXT, 'start', maxw=150, tag='lg5')

lc.text(MX, 1168, 'vllm/v1/engine/core.py:L593-L614（step() 编排面四段：L596 发车 → L597 算表 → L602 result → L603-L604 sample_tokens）'
                 '· 调度列算表吃 structured_output/__init__.py（跨 L2 站 2-9）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 1186, '时间线 ms / 64 行 / 行 0 允许集 / overlap_verified=true ＝ 本章驱动脚本实测（掩码=真 xgrammar，前向窗口=模拟）'
                 '· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-forward-shadow.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
