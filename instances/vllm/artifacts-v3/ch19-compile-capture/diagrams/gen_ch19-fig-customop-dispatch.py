#!/usr/bin/env python3
"""ch19 机制图 1 · CustomOp 构造期排班（figure_spec ch19-fig-customop-dispatch，模板 before-after）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ② 拍片
『构造期算子层』（站 3-4）内 CustomOp 框的机制展开。架构归属回指 L0/L2
（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：每个算子在构造期一次性选定平台身体并冻结为 _forward_method，运行期
forward 只剩一次属性转发——把『按平台 if-else』与『每拍 dispatch』都消灭在启动期。

数字/引语全部取自 figure_spec.numbers（源码逐字锚：custom_op.py:L122-L136 构造期
一次绑定+运行期单行转发；L174-L207 dispatch_forward 六分支序 + NOTE(woosuk) 自白；
layernorm.py:L36-L122 RMSNorm 三实现）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 808
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '「一个算子多份身体」：构造期排一次班，运行期只剩一次转发',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'CustomOp 在 __init__ 里 dispatch_forward 按平台把 _forward_method 绑死为具名方法——运行期 forward 是一行属性转发，零分支零查表（custom_op.py:L103-L207）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ② 构造期算子层 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左右两大栏 ----------------
LPX, LPW = MX, 640          # 左栏：构造期
RPX, RPW = 760, 680         # 右栏：运行期
PY, PH = 96, 600

lc.rect(LPX, PY, LPW, PH, '#ffffff', lc.C_BEAT_S, rx=10, sw=2.0)
lc.text(LPX + 18, PY + 26, '构造期 · __init__（每实例一次，开业排班）', 12.5, lc.C_BEAT_T,
        'start', True, maxw=LPW - 36, tag='lp:t')
lc.rect(RPX, PY, RPW, PH, '#ffffff', lc.C_GPU_S, rx=10, sw=2.0)
lc.text(RPX + 18, PY + 26, '运行期 · forward（每层每拍，营业中）', 12.5, lc.C_GPU_S,
        'start', True, maxw=RPW - 36, tag='rp:t')

# ================= 左栏：构造期 =================
# RMSNorm 卡（一个算子多份身体的实例）
RX, RY, RW, RH = LPX + 24, PY + 44, 300, 140
lc.rect(RX, RY, RW, RH, '#ffffff', lc.C_BEAT_S, rx=7, sw=1.5)
lc.text(RX + 14, RY + 22, 'RMSNorm', 12, lc.C_TXT, 'start', True, maxw=180, tag='rms:t')
lc.text(RX + 14, RY + 40, '@CustomOp.register("rms_norm")', 8.5, lc.C_BEAT_T, 'start',
        maxw=RW - 28, tag='rms:reg')
lc.text(RX + 14, RY + 55, 'layernorm.py:L36-L122', 8, lc.C_FAINT, 'start', maxw=RW - 28,
        tag='rms:file')
BODY = [('forward_native', 'Inductor 可融合'), ('forward_cuda', '手工 fused kernel'),
        ('forward_xpu', '')]
BCW, BGAP = 92, 8
bx0 = RX + 14
for i, (nm, sub) in enumerate(BODY):
    bx = bx0 + i * (BCW + BGAP)
    hot = (nm == 'forward_cuda')
    lc.rect(bx, RY + 66, BCW, 52, lc.C_GPU_F if hot else '#ffffff',
            lc.C_GPU_S if hot else lc.C_MUTE, rx=5, sw=1.5 if hot else 1.0)
    lc.text(bx + BCW / 2, RY + 86, nm, 8.5, lc.C_GPU_S if hot else lc.C_TXT, 'middle', True,
            maxw=BCW - 6, tag='body' + nm)
    if sub:
        lc.text(bx + BCW / 2, RY + 102, sub, 7.2, lc.C_MUTE, 'middle', maxw=BCW - 4,
                tag='bodys' + nm)
lc.text(RX + 14, RY + RH - 8, '同一份菜谱的三份身体（RMSNorm 实有实现）', 8, lc.C_FAINT,
        'start', maxw=RW - 28, tag='rms:note')

# dispatch_forward 框
DX, DY, DW, DH = LPX + 348, PY + 44, 268, 132
lc.rect(DX, DY, DW, DH, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.5)
lc.text(DX + 12, DY + 20, 'dispatch_forward(compile_native=…)', 9, lc.C_BEAT_T, 'start', True,
        maxw=DW - 24, tag='dsp:t')
lc.text(DX + 12, DY + 36, '平台分支 6 支，全返回具名方法：', 8.2, '#334155', 'start',
        maxw=DW - 24, tag='dsp:h')
BRANCH = ['is_rocm → forward_hip · is_cpu → forward_cpu',
          'is_tpu → forward_tpu · is_xpu → forward_xpu',
          'is_out_of_tree → forward_oot · else → forward_cuda']
for i, ln in enumerate(BRANCH):
    lc.text(DX + 12, DY + 52 + i * 15, ln, 7.8, '#334155', 'start', maxw=DW - 24,
            tag='dsp:b' + str(i))
lc.text(DX + 12, DY + DH - 10, 'disabled → maybe_compile(forward_native)', 7.8, lc.C_BEAT_T,
        'start', maxw=DW - 24, tag='dsp:dis')

# 选中说明：forward_cuda 卡加粗绿框即 dispatch 的选择（槽位框写明去向，箭头见下方写入槽位）
fc_cx = bx0 + 1 * (BCW + BGAP) + BCW / 2

# _forward_method 槽位
SX, SY, SW, SH = LPX + 24, PY + 208, 300, 76
lc.rect(SX, SY, SW, SH, lc.C_BADGE_F, lc.C_BEAT_S, rx=7, sw=1.6)
lc.text(SX + 14, SY + 22, '_forward_method = forward_cuda', 10, lc.C_BEAT_T, 'start', True,
        maxw=SW - 28, tag='slot:t')
lc.text(SX + 14, SY + 42, '构造期一次绑定、冻结在实例上', 8.5, '#334155', 'start',
        maxw=SW - 28, tag='slot:l1')
lc.text(SX + 14, SY + 58, '想换身体？重启进程重新开业再说', 8.5, '#334155', 'start',
        maxw=SW - 28, tag='slot:l2')
lc.parrow([(fc_cx, RY + RH), (fc_cx, SY)], lc.C_GPU_S, 1.8)
lc.text(fc_cx + 6, (RY + RH + SY) / 2 + 3, '写入槽位', 8, lc.C_GPU_S, 'start', maxw=60,
        tag='wr:t')

# enabled()/default_on() 开关框
GX, GY, GW, GH = LPX + 348, PY + 208, 268, 76
lc.rect(GX, GY, GW, GH, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
lc.text(GX + 12, GY + 20, 'enabled() · default_on() 开关', 9, lc.C_TXT, 'start', True,
        maxw=GW - 24, tag='gt:t')
lc.text(GX + 12, GY + 38, 'Inductor 时基础档 = \'none\'（让编译器融合）', 7.8, '#334155',
        'start', maxw=GW - 24, tag='gt:l1')
lc.text(GX + 12, GY + 53, '否则 \'all\'——融合 or 手工 kernel 是配置不是分支', 7.8,
        '#334155', 'start', maxw=GW - 24, tag='gt:l2')
lc.text(GX + 12, GY + 68, '（custom_op.py:L209-L311）', 7.5, lc.C_FAINT, 'start',
        maxw=GW - 24, tag='gt:f')

# NOTE 引语条
QY = PY + 308
lc.rect(LPX + 24, QY, LPW - 48, 66, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
QT = ['NOTE(woosuk)：Here we assume that vLLM was built for only one specific',
      'backend. Currently, we do not support dynamic dispatching.',
      '——构造期冻结的官方自白（custom_op.py:L175-L177）']
for i, ln in enumerate(QT):
    lc.text(LPX + 40, QY + 20 + i * 16, ln, 8.7, '#334155', 'start', maxw=LPW - 84,
            tag='q' + str(i))

# 底部：全部身体槽位清单
lc.text(LPX + 24, QY + 96, '基类身体槽位（CustomOp 全集）：', 9, lc.C_TXT, 'start', True,
        maxw=LPW - 48, tag='slots:t')
SLOTS = ['forward_native', 'forward_cuda', 'forward_hip', 'forward_cpu', 'forward_xpu',
         'forward_tpu', 'forward_oot']
sx = LPX + 24
for i, nm in enumerate(SLOTS):
    w = lc.tw(nm, 7.6) + 12
    lc.rect(sx, QY + 106, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=0.9)
    lc.text(sx + w / 2, QY + 120, nm, 7.6, '#334155', 'middle', maxw=w - 4, tag='sl' + nm)
    sx += w + 7
lc.text(LPX + 24, QY + 152, 'op 内调用的算子对模型级 torch.compile 不可见——disabled 时单独编译（maybe_compile）', 8.3,
        lc.C_MUTE, 'start', maxw=LPW - 48, tag='lp:foot')

# ================= 右栏：运行期 =================
FA, FB, FC = (RPX + 24, 150, 232, 68), (RPX + 316, 150, 168, 68), (RPX + 548, 150, 152, 68)
lc.rect(*FA, '#ffffff', lc.C_GPU_S, rx=7, sw=1.6)
lc.text(FA[0] + 12, FA[1] + 24, 'forward(self, *args, **kwargs)', 8.8, lc.C_TXT, 'start', True,
        maxw=FA[2] - 24, tag='fa:t')
lc.text(FA[0] + 12, FA[1] + 44, 'return self._forward_method(*args, **kwargs)', 7.9,
        '#334155', 'start', maxw=FA[2] - 24, tag='fa:l')
lc.text(FA[0] + 12, FA[1] + 60, '（custom_op.py:L135-L136 运行期仅此两行）', 7.4, lc.C_FAINT,
        'start', maxw=FA[2] - 24, tag='fa:f')
lc.parrow([(FA[0] + FA[2], FA[1] + 34), (FB[0], FB[1] + 34)], lc.C_GPU_S, 1.8)
lc.text((FA[0] + FA[2] + FB[0]) / 2, FA[1] + 26, '属性转发', 8, lc.C_GPU_S, 'middle', True,
        maxw=70, tag='ar1')
lc.rect(*FB, lc.C_BADGE_F, lc.C_GPU_S, rx=7, sw=1.4)
lc.text(FB[0] + 12, FB[1] + 24, '_forward_method', 9, lc.C_GPU_S, 'start', True,
        maxw=FB[2] - 24, tag='fb:t')
lc.text(FB[0] + 12, FB[1] + 44, '构造期已冻结', 8.5, '#334155', 'start', maxw=FB[2] - 24,
        tag='fb:l')
lc.text(FB[0] + 12, FB[1] + 58, '= forward_cuda', 8, lc.C_BEAT_T, 'start', True,
        maxw=FB[2] - 24, tag='fb:v')
lc.parrow([(FB[0] + FB[2], FB[1] + 34), (FC[0], FC[1] + 34)], lc.C_GPU_S, 1.8)
lc.text((FB[0] + FB[2] + FC[0]) / 2, FB[1] + 26, '直调', 8, lc.C_GPU_S, 'middle', True,
        maxw=60, tag='ar2')
lc.rect(*FC, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.6)
lc.text(FC[0] + 12, FC[1] + 24, 'forward_cuda(x)', 9.5, lc.C_GPU_S, 'start', True,
        maxw=FC[2] - 24, tag='fc:t')
lc.text(FC[0] + 12, FC[1] + 44, '手工 fused kernel', 8.5, '#334155', 'start',
        maxw=FC[2] - 24, tag='fc:l')
lc.text(FC[0] + 12, FC[1] + 58, 'kernel 已在启动期就位', 7.4, lc.C_FAINT, 'start',
        maxw=FC[2] - 24, tag='fc:f')

lc.text(RPX + RPW / 2, PY + 262, '零分支 · 零查表 · 一次属性转发', 13, lc.C_GPU_S, 'middle',
        True, maxw=RPW - 48, tag='rp:claim')

# 对照：被消灭的形态（虚影 + 划掉）
GY2 = PY + 292
lc.rect(RPX + 24, GY2, RPW - 48, 112, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(RPX + 40, GY2 + 20, '被消灭的形态（对照）', 9, lc.C_MUTE, 'start', True, maxw=200,
        tag='gh:t')
gh_boxes = ['forward(x)', '查平台 if-else', '选身体', '执行']
gx = RPX + 40
gh_w = [86, 118, 74, 74]
for i, (nm, w) in enumerate(zip(gh_boxes, gh_w)):
    lc.rect(gx, GY2 + 34, w, 34, '#ffffff', lc.C_FAINT, rx=5, sw=1.0, dash=True)
    lc.text(gx + w / 2, GY2 + 55, nm, 8.2, lc.C_FAINT, 'middle', maxw=w - 6, tag='gh' + nm)
    if i < 3:
        lc.seg(gx + w, GY2 + 51, gx + w + 14, GY2 + 51, lc.C_FAINT, 1.2, marker='std')
    gx += w + 14
lc.text(RPX + 40, GY2 + 92, '每拍一次平台判定：CPU 开销 + 平台分支挡在融合路上', 8.3, lc.C_MUTE,
        'start', maxw=RPW - 96, tag='gh:l')
# 红色划掉线
lc.seg(RPX + 30, GY2 + 6, RPX + RPW - 30, GY2 + 106, lc.C_ABORT, 2.0)
lc.seg(RPX + 30, GY2 + 106, RPX + RPW - 30, GY2 + 6, lc.C_ABORT, 2.0)

# 收益注记
RY2 = PY + 436
lc.rect(RPX + 24, RY2, RPW - 48, 100, '#ffffff', lc.C_GPU_S, rx=7, sw=1.2)
GAIN = ['· 『Inductor 融合（forward_native）』还是『手工 fused kernel（forward_cuda）』，',
        '  二选一变成配置（custom_ops 档位），不再是代码分支；',
        '· 每层每拍的 forward 不带任何平台判定——RMSNorm / LinearBP 等实例全走同一套排班。',
        '· 代价：dispatch 烤死在构造时，运行期不可切换（vLLM 为单一后端构建）。']
for i, ln in enumerate(GAIN):
    lc.text(RPX + 40, RY2 + 22 + i * 19, ln, 8.6, '#334155', 'start', maxw=RPW - 96,
            tag='gain' + str(i))

# ---------------- 页脚 ----------------
lc.text(MX, 740, '逐字锚 vllm/model_executor/custom_op.py:L122-L136（构造期绑定 + 运行期单行转发）· L174-L207（dispatch_forward 分支序 + NOTE）· vllm/model_executor/layers/layernorm.py:L36-L122（RMSNorm 三实现）',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 758, '运行期转发原文：def forward(self, *args, **kwargs): return self._forward_method(*args, **kwargs) · 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-customop-dispatch.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
