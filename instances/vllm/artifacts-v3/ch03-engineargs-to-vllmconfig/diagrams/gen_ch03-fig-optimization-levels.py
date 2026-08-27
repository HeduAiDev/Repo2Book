#!/usr/bin/env python3
"""ch03 机制图 2 · O0-O3 优化级预设落地（figure_spec ch03-fig-optimization-levels，模板 state-table）

放大自 L0 启动视角（boot）第 12 站——即本章 L2 章图 center 节拍 ⑥ 『O0-O3 落地』的机制展开。
架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）：图右上角指北小签。编译与 CUDA Graph 机制本身
是 ch19 的门牌，本图只画『旋钮档位如何落到字段终值』。

claim：O0-O3 预设经『递归只填 None』落地：O0 全关纯 eager 立即启动、O2 默认 VLLM_COMPILE +
FULL_AND_PIECEWISE + autotune，用户显式值与 enforce_eager 永远压过预设，且预设值可为按
整份 config 求值的谓词(同一档不同结果)。

数字全部取自 figure_spec.numbers 与 explainer.quantified（五场景 host 实测 trace + pin 锚点）；
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 748
MX = 60
BXR = 1440
GRID = '#e2e8f0'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '模式转盘 -O0..-O3：预设递归只填 None——你拧过的旋钮，它永远不碰',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'O0 全关立即启动，O2 默认 VLLM_COMPILE + FULL_AND_PIECEWISE + autotune——用户显式与 enforce_eager 永远压过预设；同一档还能按整份 config 现算（谓词）',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 节拍 ⑥ O0-O3 落地 · L0：启动视角（boot）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 转盘（四档 chip） ----------------
DIAL_Y = 106
lc.text(MX, 96, '转盘 optimization_level——「启动时间 vs 运行性能」的总旋钮（vllm/config/vllm.py:L104-L116）',
        9.5, lc.C_MUTE, 'start', maxw=780, tag='dial:lbl')
DIALS = [
    ('O0 · 掏出来就拍', ['不编译（mode=NONE）', '不捕 CUDA Graph（NONE）'], '启动：立即（eager 直跑）', False),
    ('O1 · 快速档', ['Dynamo+Inductor 编译', 'PIECEWISE 捕图'], '启动：要付编译', False),
    ('O2 · 默认档', ['FULL_AND_PIECEWISE 捕图', '融合 pass + autotune'], '启动 usually 5~20 s（官方自述）', True),
    ('O3 · 试验档', ['当前等同 O2', '（docstring 自述 same as -O2）'], '——', False),
]
DW, DGAP = 185, 12
for i, (head, lines, cost, badge) in enumerate(DIALS):
    x = MX + i * (DW + DGAP)
    lc.rect(x, DIAL_Y, DW, 96, '#ffffff', lc.C_BEAT_S, rx=7, sw=1.5 if badge else 1.2)
    hw = lc.tw(head, 10.5, True)
    lc.text(x + 12, DIAL_Y + 20, head, 10.5, lc.C_TXT, 'start', True, maxw=DW - 24, tag='dial:h' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x + 12, DIAL_Y + 40 + j * 16, ln, 8.5, '#334155', 'start', maxw=DW - 24,
                tag='dial:l' + str(i) + str(j))
    lc.text(x + 12, DIAL_Y + 86, cost, 8, lc.C_MUTE, 'start', maxw=DW - 24, tag='dial:c' + str(i))
    if badge:
        bl = '出厂默认'
        bw = lc.tw(bl, 8, True) + 10
        lc.rect(x + DW - bw - 6, DIAL_Y + 6, bw, 15, lc.C_BADGE_F, lc.C_ENG_S, rx=7, sw=1.0)
        lc.text(x + DW - bw / 2 - 6, DIAL_Y + 17, bl, 8, lc.C_ENG_S, 'middle', True, maxw=bw - 4,
                tag='dial:b' + str(i))

# ---------------- 优先级链 ----------------
PR_X, PR_W = 900, 540
PR_Y, PR_H = 88, 130
lc.rect(PR_X, PR_Y, PR_W, PR_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(PR_X + 14, PR_Y + 18, '优先级链——同一个键，谁说了算', 10.5, lc.C_TXT, 'start', True,
        maxw=PR_W - 28, tag='pr:t')
CHAIN = [
    ('① 用户显式设置（进场非 None）——预设永不覆盖', 'user'),
    ('② enforce_eager / 环境变量——覆盖链先行改写', 'chain'),
    ('③ 优化级预设——递归只填 None（本图主角）', 'preset'),
]
ry = PR_Y + 30
for txt_, kind in CHAIN:
    if kind == 'user':
        lc.rect(PR_X + 14, ry, PR_W - 28, 17, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
    elif kind == 'chain':
        lc.rect(PR_X + 14, ry, PR_W - 28, 17, '#ffffff', lc.C_MUTE, rx=4, sw=1.2, dash=True)
    else:
        lc.rect(PR_X + 14, ry, PR_W - 28, 17, '#ffffff', lc.C_MUTE, rx=4, sw=1.0)
    lc.text(PR_X + 24, ry + 12, txt_, 8.8, '#334155', 'start', maxw=PR_W - 48, tag='pr:' + kind)
    if kind != 'preset':
        lc.text(PR_X + PR_W / 2, ry + 29, '＞', 10, lc.C_MUTE, 'middle', True, maxw=20, tag='pr:gt')
    ry += 25
lc.text(PR_X + 14, PR_Y + 108, '落笔唯一入口 _set_config_default：if getattr(config_obj, key) is None', 8,
        lc.C_FAINT, 'start', maxw=PR_W - 28, tag='pr:mech')
lc.text(PR_X + 14, PR_Y + 122, '——两行代码的不动点（vllm.py:L811-L853 · 应用点 L1299）', 8,
        lc.C_FAINT, 'start', maxw=PR_W - 28, tag='pr:mech2')

# ---------------- 主表：五场景 × 关键开关终值 ----------------
COLS = [
    ('场景（同一模型）', '只拧图上这些旋钮', 60, 232),
    ('mode', '按优化级推导 L1274-L1279', 292, 170),
    ('cudagraph_mode', '进 15 键预设字典', 462, 210),
    ('autotune', 'kernel 旗标 · 预设填', 672, 150),
    ('fuse_allreduce_rms', '预设值=谓词函数 L155-L175', 822, 180),
    ('custom_ops', '随 mode 派生 L1288-L1296', 1002, 140),
    ('值的来源账', '进场非 None 键数 / 15 键预设', 1142, 298),
]
HDR_Y, HDR_H = 232, 44
for title, sub, x, w in COLS:
    lc.rect(x, HDR_Y, w - 4, HDR_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.2)
    lc.text(x + (w - 4) / 2, HDR_Y + 17, title, 9.5, lc.C_BEAT_T, 'middle', True, maxw=w - 12,
            tag='th:' + title)
    lc.text(x + (w - 4) / 2, HDR_Y + 33, sub, 7.5, lc.C_MUTE, 'middle', maxw=w - 12, tag='ts:' + sub[:10])

ROWS = [
    dict(label='1 · O0 默认', sub='TP=1 · 离线 LLM()',
         cells=[('NONE', None, 'flat'), ('NONE', None, 'flat'), ('False', None, 'flat'),
                ('False', None, 'flat'), ('all', None, 'flat')],
         ev='0 / 15 · 全部由 O0 预设填'),
    dict(label='2 · O2 默认', sub='TP=1（出厂路径）',
         cells=[('VLLM_COMPILE', None, 'flat'), ('FULL_AND_PIECEWISE', None, 'flat'), ('True', None, 'flat'),
                ('False', None, 'flat'), ('none', None, 'flat')],
         ev='0 / 15 · 全部由 O2 预设填'),
    dict(label='3 · O2 · TP=2', sub='谓词按整份 config 求值',
         cells=[('VLLM_COMPILE', None, 'flat'), ('FULL_AND_PIECEWISE', None, 'flat'), ('True', None, 'flat'),
                ('True', '谓词：TP>1 才翻', 'flat'), ('none', None, 'flat')],
         ev='0 / 15 · 谓词以整份 config 为根求值'),
    dict(label='4 · O2 + 用户显式', sub='进场手设 cudagraph_mode',
         cells=[('VLLM_COMPILE', None, 'flat'), ('PIECEWISE', '用户显式存活', 'user'), ('True', None, 'flat'),
                ('False', None, 'flat'), ('none', None, 'flat')],
         ev='1 / 15 · 用户键保留，预设只填其余 14'),
    dict(label='5 · O2 + enforce_eager', sub='覆盖链先行改写',
         cells=[('NONE', None, 'chain'), ('NONE', 'max_cudagraph_capture_size=0', 'chain'),
                ('True', '预设照填——eager 不碰这条', 'flat'),
                ('False', None, 'flat'), ('all', None, 'flat')],
         ev='0 / 15 · mode/cudagraph 先被覆盖链改写'),
]
ROW_Y0, ROW_H, ROW_STEP = 284, 48, 52
for i, row in enumerate(ROWS):
    y = ROW_Y0 + i * ROW_STEP
    # 行标签列
    x0, w0 = COLS[0][2], COLS[0][3]
    lc.rect(x0, y, w0 - 4, ROW_H, '#ffffff', lc.C_MUTE, rx=4, sw=1.1)
    lc.text(x0 + 12, y + 19, row['label'], 9.5, lc.C_TXT, 'start', True, maxw=w0 - 24,
            tag='rl' + str(i))
    lc.text(x0 + 12, y + 36, row['sub'], 8, lc.C_MUTE, 'start', maxw=w0 - 24, tag='rs' + str(i))
    # 值列
    for j, (val, tag_, kind) in enumerate(row['cells']):
        x, w = COLS[j + 1][2], COLS[j + 1][3]
        if kind == 'user':
            lc.rect(x, y, w - 4, ROW_H, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.6)
            lc.text(x + (w - 4) / 2, y + 21, val, 9.5, lc.C_ENG_S, 'middle', True, maxw=w - 12,
                    tag='cv' + str(i) + str(j))
        elif kind == 'chain':
            lc.rect(x, y, w - 4, ROW_H, '#ffffff', lc.C_MUTE, rx=4, sw=1.3, dash=True)
            lc.text(x + (w - 4) / 2, y + 21, val, 9.5, lc.C_TXT, 'middle', True, maxw=w - 12,
                    tag='cv' + str(i) + str(j))
        else:
            lc.rect(x, y, w - 4, ROW_H, '#ffffff', GRID, rx=4, sw=1.0)
            lc.text(x + (w - 4) / 2, y + 21, val, 9.5, '#334155', 'middle', maxw=w - 12,
                    tag='cv' + str(i) + str(j))
        if tag_:
            lc.text(x + (w - 4) / 2, y + 38, tag_, 7.5, lc.C_MUTE, 'middle', maxw=w - 10,
                    tag='ct' + str(i) + str(j))
    # 来源账列
    x, w = COLS[6][2], COLS[6][3]
    lc.rect(x, y, w - 4, ROW_H, '#ffffff', GRID, rx=4, sw=1.0)
    lc.text(x + 12, y + ROW_H / 2 + 3, row['ev'], 8.5, '#334155', 'start', maxw=w - 24, tag='ev' + str(i))

# ---------------- 底部两注 ----------------
NT_Y, NT_H = 560, 92
lc.rect(MX, NT_Y, 680, NT_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(MX + 14, NT_Y + 18, '15 个叶子键 = O0 与 O2 预设同构', 9.5, lc.C_TXT, 'start', True,
        maxw=650, tag='nt1:t')
lc.text(MX + 14, NT_Y + 36, 'pass_config 12 个融合旗标 + cudagraph_mode + use_inductor_graph_partition',
        8.5, '#334155', 'start', maxw=650, tag='nt1:l1')
lc.text(MX + 14, NT_Y + 52, '+ kernel_config.enable_flashinfer_autotune（vllm.py:L229-L327）——应用成本 = O(15) 次递归属性检查，',
        8.5, '#334155', 'start', maxw=650, tag='nt1:l2')
lc.text(MX + 14, NT_Y + 68, '与模型规模无关', 8.5, '#334155', 'start', maxw=650, tag='nt1:l3')

lc.rect(780, NT_Y, 660, NT_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(794, NT_Y + 18, '谓词默认值与 host seam 注', 9.5, lc.C_TXT, 'start', True, maxw=630,
        tag='nt2:t')
lc.text(794, NT_Y + 36, 'O2 的 fuse_allreduce_rms 预设值是函数 enable_allreduce_rms_fusion(cfg)——以整份',
        8.5, '#334155', 'start', maxw=630, tag='nt2:l1')
lc.text(794, NT_Y + 52, 'config 为根求值：TP>1 才翻 True（场景 2 vs 3 同档不同值）；真机谓词还门控',
        8.5, '#334155', 'start', maxw=630, tag='nt2:l2')
lc.text(794, NT_Y + 68, 'Hopper/Blackwell + flashinfer 探测（L160-L174）——本表取 host seam 值（只保留 TP>1 前置）',
        8.5, '#334155', 'start', maxw=630, tag='nt2:l3')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = NT_Y + NT_H + 26
lx = MX
items = [
    ('user', '用户显式存活（进场非 None——预设不碰）'),
    ('chain', '覆盖链改写（enforce_eager / 环境变量）'),
    ('flat', '档位机制落定（预设字典 15 键；mode / custom_ops 随档推导）'),
]
for kind, name in items:
    if kind == 'user':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
    elif kind == 'chain':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', lc.C_MUTE, rx=4, sw=1.1, dash=True)
    else:
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', GRID, rx=4, sw=1.0)
    lc.text(lx + 26, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=320, tag='leg' + kind)
    lx += 26 + lc.tw(name, 9) + 22

lc.text(MX, LEG_Y + 26, 'verbatim vllm/config/vllm.py:L104-L116（档位枚举）· L229-L327（预设字典）· L811-L853（递归只填 None）· '
        'L1193-L1197 / L1424-L1430（eager 覆盖链）· 场景终值取自精简版 companion host 实测 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch03-fig-optimization-levels.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
