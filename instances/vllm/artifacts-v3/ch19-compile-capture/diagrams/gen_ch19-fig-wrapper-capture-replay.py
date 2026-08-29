#!/usr/bin/env python3
"""ch19 机制图 9 · CUDAGraphWrapper 捕获/回放状态机（figure_spec ch19-fig-wrapper-capture-replay，模板 state-machine）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ⑧ 拍片
『喂图·算子与回放』（站 15，CUDAGraphWrapper 捕获/回放）的机制展开。
架构归属回指 L0/L2：右上角指北小签。

claim：CUDAGraphWrapper 是一台盲信的录放机：无 forward context/模式不匹配直通
runnable，key 首遇即捕（捕获窗口校验+共享图池+弱引用输出），命中即回放
（DEBUG 逐 data_ptr 断言）；它不存缓冲不拷输入——固定地址是 runner 的职责。

数字/引语全部取自 figure_spec.numbers（docstring 四步工作流 L146-L168 ·
地址体检 assert L346-L355 · 边界声明 L161-L167 · tripwire L285-L287+monitor L90-L99 ·
两层挂载 runner L5467-L5469 + backends L246-L254 · 捕获细节 L303-L336）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 762
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'CUDAGraphWrapper：一台盲信的录放机——判责在 dispatcher、执行在 wrapper',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '从 forward context 收 mode + descriptor、盲信照做；key 首遇即捕（窗口校验+共享图池+弱引用输出），命中即回放（DEBUG 逐 data_ptr 断言）（cuda_graph.py:L233-L361）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑧ 喂图·算子与回放 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主判定链（stadium 判定框） ----------------
DY, DH_ = 128, 76
lc.rect(MX, DY + 10, 140, 56, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 70, DY + 32, '__call__(args)', 9, lc.C_TXT, 'middle', True, maxw=130, tag='en')
lc.text(MX + 70, DY + 50, 'wrapper 入口', 7.4, lc.C_MUTE, 'middle', maxw=130, tag='ens')

def decision(x, w, l1, l2):
    lc.rect(x, DY, w, DH_, '#ffffff', lc.C_ENG_S, rx=DH_ / 2, sw=1.8)
    lc.text(x + w / 2, DY + 32, l1, 9, lc.C_TXT, 'middle', True, maxw=w - 20, tag='d' + l1[:6])
    lc.text(x + w / 2, DY + 52, l2, 9, lc.C_ENG_S, 'middle', True, maxw=w - 20, tag='d' + l2[:6])

D1X, D2X, D3X, DDW = 250, 480, 710, 190
decision(D1X, DDW, 'forward context', '可用？')
decision(D2X, DDW, 'mode 匹配', '本 wrapper？')
decision(D3X, 200, 'key 已在', 'entries 表？')

lc.parrow([(MX + 140, DY + 38), (D1X, DY + 38)], lc.C_MUTE, 1.6)
for x0, x1 in ((D1X + DDW, D2X), (D2X + DDW, D3X)):
    lc.parrow([(x0, DY + 38), (x1, DY + 38)], lc.C_MUTE, 1.6)
    lc.text((x0 + x1) / 2, DY + 30, '是', 8, lc.C_MUTE, 'middle', True, maxw=30, tag='yes')

# 直通框（两否分支汇入）
PBX, PBY, PBW, PBH = 1100, 72, 320, 84
lc.rect(PBX, PBY, PBW, PBH, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(PBX + 14, PBY + 20, '直通 runnable（eager 照跑）', 9.3, lc.C_TXT, 'start', True,
        maxw=PBW - 28, tag='pb:t')
lc.text(PBX + 14, PBY + 40, '· 无 context：常规推理路径之外（如视觉编码器前向）', 7.8,
        '#334155', 'start', maxw=PBW - 28, tag='pb:l1')
lc.text(PBX + 14, PBY + 58, '· mode 不匹配 / NONE：warmup、profile、或嵌套 wrapper 各认领', 7.8,
        '#334155', 'start', maxw=PBW - 28, tag='pb:l2')
lc.text(PBX + 14, PBY + 74, '  自己的档（嵌套多 wrapper 分派语义，L246-L254）', 7.8,
        '#334155', 'start', maxw=PBW - 28, tag='pb:l3')
lc.parrow([(D1X + DDW / 2, DY), (D1X + DDW / 2, 48), (1240, 48), (1240, PBY)], lc.C_MUTE, 1.5)
lc.text(D1X + DDW / 2 + 16, 42, '否——无 context（视觉编码器等）', 7.8, lc.C_MUTE, 'start',
        maxw=280, tag='nb1')
lc.parrow([(D2X + DDW / 2, DY), (D2X + DDW / 2, 108), (PBX, 108)], lc.C_MUTE, 1.5)
lc.text(D2X + DDW / 2 + 8, 100, '否——mode 不匹配 / NONE', 7.8, lc.C_MUTE, 'start', maxw=180,
        tag='nb2')

# ---------------- 捕获支（否）与回放支（是） ----------------
BR_Y, BR_H = 300, 252
# 捕获支
CX0, CW0 = 560, 470
lc.rect(CX0, BR_Y, CW0, BR_H, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.8)
lc.text(CX0 + 16, BR_Y + 24, '首遇即捕（一次性，key 首次出现）', 10, lc.C_GPU_S, 'start',
        True, maxw=CW0 - 32, tag='cap:t')
CAP = [
    '① validate_cudagraph_capturing_enabled()——捕获窗口校验：',
    '   启动编排捕完关窗后，意外捕获直接 RuntimeError（tripwire）',
    '② 记录 input_addresses = [x.data_ptr() for x in args]——回放断言的对台本',
    '③ 共享图池 set_graph_pool_id（小图复用大图的池）；gc_disable 补丁挡住',
    '   跨层反复 gc.collect（否则捕获极慢）',
    '④ with torch.cuda.graph(cudagraph, pool=…)：跑一遍 runnable；',
    '   输出 weak_ref_tensors(output) 省显存（只有末片安全）',
    '⑤ 建表项 concrete_cudagraph_entries[key] 缓存；返回 output 本体',
    '   （而非弱引用——pytorch 靠它管捕获期内存）',
]
for i, ln in enumerate(CAP):
    lc.text(CX0 + 16, BR_Y + 46 + i * 22, ln, 8.2, '#334155', 'start', maxw=CW0 - 32,
            tag='cap:l' + str(i))
lc.parrow([(D3X + 100, DY + DH_), (D3X + 100, BR_Y)], lc.C_GPU_S, 1.8)
lc.text(D3X + 108, (DY + DH_ + BR_Y) / 2 + 3, '否——key 首遇', 8, lc.C_GPU_S, 'start', True,
        maxw=110, tag='nb3')

# 回放支
RX0, RW0 = 1070, BXR - 1070
lc.rect(RX0, BR_Y, RW0, BR_H, '#eff6ff', lc.C_API_S, rx=9, sw=1.8)
lc.text(RX0 + 16, BR_Y + 24, '命中即回放（每拍）', 10, lc.C_API_S, 'start', True,
        maxw=RW0 - 32, tag='rep:t')
REP = [
    '① is_debugging_mode：逐 data_ptr 体检——',
    '   assert new_input_addresses == entry.input_addresses',
    '   错误信息原话：Input addresses for cudagraphs are',
    '   different during replay. Expected {...}, got {...}',
    '   （仅 DEBUG 模式，L346-L355）',
    '② entry.cudagraph.replay()——一次 replay 顶整段',
    '   kernel launch',
    '③ return entry.output（弱引用，省显存）',
]
for i, ln in enumerate(REP):
    lc.text(RX0 + 16, BR_Y + 46 + i * 22, ln, 8.2, '#334155', 'start', maxw=RW0 - 32,
            tag='rep:l' + str(i))
lc.parrow([(D3X + 200, DY + 38), (1240, DY + 38), (1240, BR_Y)], lc.C_API_S, 1.8)
lc.text(1000, DY + 30, '是——key 命中', 8, lc.C_API_S, 'middle', True, maxw=110, tag='yb3')

# ---------------- 左侧两注卡 ----------------
A_Y = 300
lc.rect(MX, A_Y, 470, 128, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, A_Y + 20, '两层挂载，各认领自己的 mode', 9.3, lc.C_TXT, 'start', True,
        maxw=440, tag='a:t')
AL = ['· 模型外：load_model 尾部 CUDAGraphWrapper(self.model,',
      '  runtime_mode=FULL)（gpu_model_runner.py:L5467-L5469）',
      '· 编译器内：每片包 CUDAGraphWrapper(PIECEWISE)',
      '  （backends.py:L633-L684）——嵌套多 wrapper 按 mode 各自认领']
for i, ln in enumerate(AL):
    lc.text(MX + 14, A_Y + 40 + i * 21, ln, 8.0, '#334155', 'start', maxw=440,
            tag='a:l' + str(i))
B_Y = 444
lc.rect(MX, B_Y, 470, 180, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 14, B_Y + 20, '边界声明（docstring 原话）', 9.3, lc.C_TXT, 'start', True,
        maxw=440, tag='b:t')
BL = ['CUDAGraphWrapper does not store persistent buffers or',
      'copy any runtime inputs into that buffers for replay. We',
      'assume implementing them is done outside of the wrapper.',
      '（L161-L167）',
      '——wrapper 两手空空：形状全等由 dispatcher 的 padding 供给、',
      '地址不变由 ch18 固定缓冲供给；DEBUG 的逐 data_ptr 断言是',
      '第二条的运行期体检。']
for i, ln in enumerate(BL):
    lc.text(MX + 14, B_Y + 40 + i * 19, ln, 8.0, '#334155', 'start', maxw=440,
            tag='b:l' + str(i))

# ---------------- 底部盲信引语 ----------------
QY = 634
lc.rect(MX, QY, BXR - MX, 56, '#ffffff', lc.C_GPU_S, rx=7, sw=1.1, dash=True)
lc.text(MX + 16, QY + 22, '盲信原文（四步工作流第 2 步，L146-L168）：the wrapper receives a runtime_mode and a batch_descriptor(key) from the forward context and blindly trust them for cudagraph dispatching.',
        8.3, '#334155', 'start', maxw=BXR - MX - 32, tag='q1')
lc.text(MX + 16, QY + 42, '——判责在 dispatcher（决定放哪张图）、执行在 wrapper（照放、从不问为什么）。', 8.3,
        lc.C_GPU_S, 'start', maxw=BXR - MX - 32, tag='q2')

# ---------------- 页脚 ----------------
lc.text(MX, 716, '逐字锚 vllm/compilation/cuda_graph.py:L146-L168（docstring 四步工作流+边界声明）· L233-L361（__call__ 两分支）· L346-L355（DEBUG data_ptr 断言）· vllm/compilation/monitor.py:L90-L99（tripwire）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 734, '全图 wrapper 挂载点 vllm/v1/worker/gpu_model_runner.py:L5467-L5469 · 逐片挂载 vllm/compilation/backends.py:L633-L684 · 本图为源码锚点机制图（捕获/回放的 CUDA 段 host 不可实跑）',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-wrapper-capture-replay.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
