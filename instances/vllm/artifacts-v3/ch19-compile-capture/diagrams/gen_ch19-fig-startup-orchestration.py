#!/usr/bin/env python3
"""ch19 机制图 10 · 启动编排一晚流水线（figure_spec ch19-fig-startup-orchestration，模板 swimlane）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ④ 拍片
『启动编排』（站 6，compile_or_warm_up_model）与 center ⑥ 拍片『捕获·从大到小』
（站 9，capture_model）的编排展开。架构归属回指 L0/L2：右上角指北小签。

claim：编译/捕获/warmup/防退化全部前移启动期：warmup 逐尺寸 dummy → kernel 调优
→ capture 从大到小（每形状先 eager 热身再进捕获窗口）→ sampler 预热（刻意在
capture 后）→ inductor lazy init → 三纠察（JIT monitor / freeze GC / sync check），
运行期零惊喜。

数字/引语全部取自 figure_spec.numbers（编排顺序 gpu_worker.py:L679-L853 ·
从大到小理由原文 L6829-L6831 · 每形状先热身再捕 L6920-L6966 · 5~20 秒 L6912 ·
86 key 默认档 · sampler 刻意靠后 NOTE L796-L816 · 捕完关窗 L6895-L6900+monitor）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 736
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '开业前的一晚：warmup → 调优 → 捕获 → 预热 → 三纠察——运行期零惊喜',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'compile_or_warm_up_model（gpu_worker.py:L679-L853）：编译/捕获/warmup/防退化全部前移到第一个请求之前——明早开门起，后厨不允许出现任何『第一次』',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ④⇢⑥ 启动编排 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主时间线：8 工位 ----------------
STY, STH = 100, 112
SW_, SGAP = 155, 12
STATIONS = [
    ('①', 'warmup 逐尺寸', ['warmup_sizes 逐个', '_dummy_run（首个', '触发编译切图）'], False),
    ('②', 'kernel 调优', ['kernel_warmup：', 'capture 前调优', '要用的 kernel'], False),
    ('③', 'capture 从大到小', ['大图先捕、小图', '复用图池显存', '（放大见下 ▼）'], True),
    ('④', 'sampler 预热', ['末 PP 段', '_dummy_sampler_run', '（刻意在 capture 后）'], False),
    ('⑤', 'inductor lazy init', ['trigger_inductor_', 'lazy_init：免运行期', 'compile cache-miss'], False),
    ('⑥', 'JIT monitor', ['activate_jit_monitor：', '意外编译=', '延迟尖峰警报'], False),
    ('⑦', 'freeze GC', ['freeze_gc_heap：', 'GC 不再扫描', '静态对象'], False),
    ('⑧', 'sync check', ['enable_gpu_sync_', 'check：此后每拍', 'enforce 体检'], False),
]
sx0 = MX
centers = []
for i, (no, t, lines, hot) in enumerate(STATIONS):
    x = sx0 + i * (SW_ + SGAP)
    centers.append(x + SW_ / 2)
    lc.rect(x, STY, SW_, STH, lc.C_GPU_F if hot else '#ffffff',
            lc.C_GPU_S if hot else lc.C_MUTE, rx=8, sw=2.0 if hot else 1.2)
    lc.text(x + 14, STY + 20, no, 10.5, lc.C_GPU_S, 'start', True, maxw=20, tag='stno' + no)
    lc.text(x + 34, STY + 20, t, 8.8, lc.C_GPU_S if hot else lc.C_TXT, 'start', True,
            maxw=SW_ - 42, tag='stt' + no)
    for j, ln in enumerate(lines):
        lc.text(x + 14, STY + 42 + j * 16, ln, 7.4, '#334155', 'start', maxw=SW_ - 24,
                tag='stl%s%d' % (no, j))
    if i < 7:
        lc.seg(x + SW_, STY + STH / 2, x + SW_ + SGAP, STY + STH / 2, lc.C_MUTE, 1.4,
               marker='std')
lc.text(MX, STY - 10, '启动序（左 → 右，一次走完；伴读版 tests 断言同序）', 8.5, lc.C_MUTE,
        'start', maxw=400, tag='tl:t')

# NOTE 便签（③→④ 之间）
NX = centers[2] - 60
NY_ = STY + STH + 14
lc.rect(NX, NY_, 560, 46, '#ffffff', lc.C_ENG_S, rx=7, sw=1.2, dash=True)
lc.text(NX + 12, NY_ + 18, 'NOTE（L796-L816 原话节选）：This is called after capture_model on purpose to prevent', 7.8,
        '#334155', 'start', maxw=536, tag='n1')
lc.text(NX + 12, NY_ + 36, 'memory buffers from being cleared by torch.accelerator.empty_cache.——sampler 预热刻意排在捕获之后', 7.8,
        '#334155', 'start', maxw=536, tag='n2')
lc.parrow([(centers[2] + (centers[3] - centers[2]) / 2, STY + STH),
           (centers[2] + (centers[3] - centers[2]) / 2, NY_)], lc.C_ENG_S, 1.2, dash=True)

# ---------------- ③ capture_model 展开面板 ----------------
PY, PH_ = 278, 268
lc.rect(MX, PY, BXR - MX, PH_, '#ffffff', lc.C_GPU_S, rx=10, sw=1.6)
lc.text(MX + 18, PY + 24, '③ capture_model 展开（站 9）：形状从大到小，每形状『先 eager 热身、再进捕获窗口捕一次』',
        10, lc.C_GPU_S, 'start', True, maxw=1000, tag='cp:t')
lc.text(MX + 18, PY + 44, 'Capture the large shapes first so that the smaller shapes can reuse the memory pool allocated for the large shapes.（gpu_model_runner.py:L6829-L6831 注释原话）',
        7.8, '#334155', 'start', maxw=BXR - MX - 36, tag='cp:q')

SHAPES = [('形状 A（最大）', '先捕', True), ('形状 B', '', False), ('形状 C（更小）', '', False)]
COLW, CGAP = 300, 50
cx0 = MX + 30
for si, (name, tag, first) in enumerate(SHAPES):
    x = cx0 + si * (COLW + CGAP)
    lc.text(x + COLW / 2, PY + 74, name, 8.8, lc.C_GPU_S, 'middle', True, maxw=COLW,
            tag='sh%d' % si)
    steps = [('热身：num_warmups 次 _dummy_run（mode=NONE）', 'eager 照跑、不进图'),
             ('torch.accelerator.synchronize()——热身流收干', ''),
             ('捕一次：_dummy_run(is_graph_capturing=True)', 'wrapper 首遇 key 即捕')]
    for j, (t1, t2) in enumerate(steps):
        y = PY + 86 + j * 52
        lc.rect(x, y, COLW, 44, '#ffffff', lc.C_GPU_S, rx=6, sw=1.1)
        lc.text(x + 12, y + 19, t1, 8.0, '#334155', 'start', maxw=COLW - 24, tag='s%d%d' % (si, j))
        if t2:
            lc.text(x + 12, y + 35, t2, 7.2, lc.C_MUTE, 'start', maxw=COLW - 24,
                    tag='s2%d%d' % (si, j))
        if j < 2:
            lc.seg(x + COLW / 2, y + 44, x + COLW / 2, y + 52, lc.C_GPU_S, 1.2, marker='std')
    if si < 2:
        ax = x + COLW
        ay = PY + 86 + 52
        lc.parrow([(ax + 6, ay + 20), (ax + CGAP - 6, ay + 20)], lc.C_GPU_S, 1.4)
        lc.text(ax + CGAP / 2, ay + 12, '图池复用', 7.2, lc.C_GPU_S, 'middle', maxw=CGAP - 12,
                tag='pool%d' % si)
# 关窗框（第 4 槽）
WX = cx0 + 3 * COLW + 3 * CGAP
lc.rect(WX, PY + 86, BXR - MX - 30 - WX, 140, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.5)
lc.text(WX + 14, PY + 108, '捕完关窗 ⚡', 10, lc.C_BEAT_T, 'start', True, maxw=150, tag='wz:t')
WZ = ['set_cudagraph_capturing_enabled(False)',
      '——此后任何意外 cudagraph 捕获',
      '直接 RuntimeError（tripwire，',
      'monitor.py:L90-L99）。',
      '每形状收尾 torch.accelerator.',
      'synchronize() 收干捕获流。']
for i, ln in enumerate(WZ):
    lc.text(WX + 14, PY + 130 + i * 16, ln, 7.8, '#334155', 'start',
            maxw=BXR - MX - 44 - WX, tag='wz:l' + str(i))

# ---------------- 底部账本卡 ----------------
BY2 = PY + PH_ + 20
lc.rect(MX, BY2, BXR - MX, 74, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + 16, BY2 + 22, '这一晚捕完的账（默认刻度，51 捕获档）：51 PIECEWISE + 35 FULL = 86 个查表 key——FULL 档每 key 一张全图、PIECEWISE 档每个编译片各一张（片数按 2L+1 增长）',
        8.8, lc.C_GPU_S, 'start', True, maxw=BXR - MX - 32, tag='acct:l1')
lc.text(MX + 16, BY2 + 44, '官方自述：This usually takes 5~20 seconds.（capture 段，L6912-L6913）——这笔启动预算买断的是运行期的零惊喜：每拍只做查表+喂图+回放，零编译零捕获零分配。',
        8.6, '#334155', 'start', maxw=BXR - MX - 32, tag='acct:l2')
lc.text(MX + 16, BY2 + 62, '三纠察（⑥⑦⑧）把『运行期不许出现第一次』从注释升级为 tripwire：意外 JIT / GC 扫静态堆 / 漏 sync 全部当场上报。', 8.6,
        '#334155', 'start', maxw=BXR - MX - 32, tag='acct:l3')

# ---------------- 页脚 ----------------
lc.text(MX, BY2 + 102, '逐字锚 vllm/v1/worker/gpu_worker.py:L679-L853（compile_or_warm_up_model 编排）· L796-L816（sampler 刻意靠后 NOTE）· vllm/v1/worker/gpu_model_runner.py:L6829-L6913（从大到小+5~20 秒）· L6920-L6966（先热身再捕）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, BY2 + 120, '86 key 默认刻度取自精简版 companion host 实跑（51+35）；编排顺序与伴读版测试断言同序',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-startup-orchestration.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
