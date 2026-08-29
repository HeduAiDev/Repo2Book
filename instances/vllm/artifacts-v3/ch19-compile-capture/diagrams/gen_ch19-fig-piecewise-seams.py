#!/usr/bin/env python3
"""ch19 机制图 6 · piecewise 运行形态与接缝成本（figure_spec ch19-fig-piecewise-seams，模板 swimlane）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ⑤ 拍片
『切图与逐片编译』（站 8，逐片编译与接缝）与 center ⑧ 拍片（运行期拼跑）的
运行形态展开。架构归属回指 L0/L2：右上角指北小签。

claim：piecewise 的运行形态=片内编译产物、片间接缝 eager：PiecewiseCompileInterpreter
按序拼跑，非切分子图送 Inductor 并各自包 CUDAGraphWrapper(PIECEWISE)；接缝处连
view/slice 都是可感知的 CPU 开销。

数字/引语全部取自 figure_spec.numbers（每片包装三选项 backends.py:L633-L684 ·
接缝成本 NOTE flash_attn.py:L880-L888 原话 · 片数公式实跑标定 · PIECEWISE 恒定注 ·
拼跑器 L730-L776）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 660
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '两种优化砍两种开销、接缝两不沾：片内编译图、片间 eager',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'Inductor 砍算子粒度（融合小算子、消中间显存往返）、CUDA graph 砍提交粒度（一次 replay 顶整段 launch）——被切出去的接缝留在 eager Python 里跑',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑤⇢⑧ · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 双泳道 ----------------
LANE_LBL_W = 128
TLX = MX + LANE_LBL_W          # 时间线起点
L1Y, L2Y, LANE_H = 100, 244, 104
# 泳道底
lc.rect(TLX, L1Y, BXR - TLX, LANE_H, '#fbfdff', lc.C_GPU_S, rx=9, sw=1.2)
lc.rect(TLX, L2Y, BXR - TLX, LANE_H, '#fffaf5', lc.C_BEAT_S, rx=9, sw=1.2)
# 泳道标签
lc.rect(MX, L1Y, LANE_LBL_W - 12, LANE_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + (LANE_LBL_W - 12) / 2, L1Y + 34, '编译片泳道', 10.5, lc.C_GPU_S, 'middle', True,
        maxw=LANE_LBL_W - 24, tag='l1:t')
lc.text(MX + (LANE_LBL_W - 12) / 2, L1Y + 56, 'Inductor 产物', 8, '#334155', 'middle',
        maxw=LANE_LBL_W - 24, tag='l1:s1')
lc.text(MX + (LANE_LBL_W - 12) / 2, L1Y + 74, '+ CUDAGraphWrapper', 7.6, '#334155', 'middle',
        maxw=LANE_LBL_W - 24, tag='l1:s2')
lc.text(MX + (LANE_LBL_W - 12) / 2, L1Y + 90, '(PIECEWISE)', 7.6, '#334155', 'middle',
        maxw=LANE_LBL_W - 24, tag='l1:s3')
lc.rect(MX, L2Y, LANE_LBL_W - 12, LANE_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.4)
lc.text(MX + (LANE_LBL_W - 12) / 2, L2Y + 34, 'eager 接缝泳道', 10.5, lc.C_BEAT_T, 'middle',
        True, maxw=LANE_LBL_W - 24, tag='l2:t')
lc.text(MX + (LANE_LBL_W - 12) / 2, L2Y + 56, 'attention 后端', 8, '#334155', 'middle',
        maxw=LANE_LBL_W - 24, tag='l2:s1')
lc.text(MX + (LANE_LBL_W - 12) / 2, L2Y + 74, 'Python 代码', 8, '#334155', 'middle',
        maxw=LANE_LBL_W - 24, tag='l2:s2')
lc.text(MX + (LANE_LBL_W - 12) / 2, L2Y + 90, '（KV 写 + attention）', 7.4, '#334155',
        'middle', maxw=LANE_LBL_W - 24, tag='l2:s3')

# 五块（执行序交错）
BLOCKS = [
    ('submod_0', '8 节点', 'first', TLX + 24, 236, 1),
    ('submod_1 · 接缝', 'kv_update → attention', 'seam', 0, 196, 2),
    ('submod_2', '10 节点', 'mid', 0, 252, 1),
    ('submod_3 · 接缝', 'kv_update → attention', 'seam', 0, 186, 2),
    ('submod_4', '2 节点', 'last', 0, 152, 1),
]
# 顺序排 x
cx = TLX + 24
GAPX = 22
pos = []
for name, sub, kind, _x, w, lane in BLOCKS:
    pos.append((cx, w, lane, name, sub, kind))
    cx += w + GAPX
for x, w, lane, name, sub, kind in pos:
    y = L1Y if lane == 1 else L2Y
    bh = 74
    by = y + (LANE_H - bh) / 2
    seam = (kind == 'seam')
    lc.rect(x, by, w, bh, '#ffffff', lc.C_BEAT_S if seam else lc.C_GPU_S, rx=7, sw=1.6)
    col = lc.C_BEAT_T if seam else lc.C_GPU_S
    lc.text(x + w / 2, by + 17, name, 9.5, col, 'middle', True, maxw=w - 10, tag='b:' + name)
    lc.text(x + w / 2, by + 33, sub if seam else sub + ' · Inductor 编译', 7.6, '#334155',
            'middle', maxw=w - 10, tag='bs:' + name)
    if not seam:
        lc.text(x + w / 2, by + 49, 'CUDAGraphWrapper(PIECEWISE)', 7.2, lc.C_MUTE, 'middle',
                maxw=w - 10, tag='bw:' + name)
        # 每片包装三选项标注（首/中/末片各对应一项，全表见泳道上方选项行）
        opt = {'first': '〔首片：debug_log_enable=True〕',
               'mid': '〔非首片：gc_disable=True〕',
               'last': '〔末片：weak_ref_output=True〕'}[kind]
        lc.text(x + w / 2, by + 65, opt, 7.0, lc.C_GPU_S, 'middle', True, maxw=w - 10,
                tag='bdg:' + kind)
    else:
        lc.text(x + w / 2, by + 49, '2 个 eager 算子调用/拍', 7.2, lc.C_MUTE, 'middle',
                maxw=w - 10, tag='bw:' + name)
        lc.text(x + w / 2, by + 65, '〔每层每拍 2 次 op dispatch〕', 7.0, lc.C_BEAT_T, 'middle',
                True, maxw=w - 10, tag='bdg:seam')

# 执行序箭头（交错跨泳道，端点贴框边）
for i in range(len(pos) - 1):
    x0, w0, lane0 = pos[i][0], pos[i][1], pos[i][2]
    x1, lane1 = pos[i + 1][0], pos[i + 1][2]
    if lane0 == 1 and lane1 == 2:
        ya, yb = L1Y + LANE_H / 2, L2Y + LANE_H / 2
    else:
        ya, yb = L2Y + LANE_H / 2, L1Y + LANE_H / 2
    lc.parrow([(x0 + w0, ya), (x1, yb)], lc.C_MUTE, 1.6)
lc.text(TLX + 10, L2Y + LANE_H + 18, '执行序：片 → 缝 → 片 → 缝 → 片（PiecewiseCompileInterpreter 按序拼跑，一片一缝交错成一条时间线）',
        8.5, lc.C_MUTE, 'start', maxw=BXR - TLX - 20, tag='order')

# 角标说明行
lc.text(TLX + 10, L1Y - 14, '每片包装三选项（wrap_with_cudagraph_if_needed，backends.py:L633-L684）：首片 debug_log_enable=True · 非首片 gc_disable=True（跨层反复 gc.collect 让捕获极慢）· 末片 weak_ref_output=True（仅末片安全）',
        7.8, lc.C_MUTE, 'start', maxw=BXR - TLX - 20, tag='opt:l')

# ---------------- NOTE 引语条 ----------------
QY = 388
lc.rect(MX, QY, BXR - MX, 84, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
QT = ['接缝成本官方自白（flash_attn.py:L880-L888 原话）：NOTE(woosuk)：With piece-wise CUDA graphs, this method is executed in',
      'eager-mode PyTorch. Thus, we need to be careful about any CPU overhead in this method. For example, `view` and `slice`',
      '(or `[:n]`) operations are surprisingly slow even in the case they do not invoke any GPU ops. … Whenever making a',
      'change in this method, please benchmark the performance to make sure it does not introduce any overhead.']
for i, ln in enumerate(QT):
    lc.text(MX + 16, QY + 20 + i * 16, ln, 8.3, '#334155', 'start', maxw=BXR - MX - 32,
            tag='q' + str(i))

# ---------------- 底部三注 ----------------
CY, CH2 = 496, 96
cards = [
    (MX, 424, '片数公式（实跑标定 + 推论）', lc.C_GPU_S,
     ['L=2 层 → 2L+1=5 片（3 编译 + 2 接缝）；一般 L 层 → 2L+1 片、',
      'L 道接缝、每拍 2L 次 eager 算子调用。', '接缝是 piecewise 的固定成本入口。']),
    (MX + 448, 448, '拼跑器：PiecewiseCompileInterpreter', lc.C_BEAT_S,
     ['call_module 命中编译名单的子图 → 建 PiecewiseBackend（Inductor 编译，',
      'L755-L764）挂回 module.__dict__[target]；其余（切点算子）eager 直调',
      '（backends.py:L730-L776）。']),
    (MX + 920, BXR - MX - 920, '逐片 wrapper 恒 PIECEWISE', lc.C_MUTE,
     ['无论挂在全图还是片上，逐片 wrapper 恒 PIECEWISE 运行模式——',
      'to distinguish it from the FULL cudagraph runtime mode',
      '（backends.py:L669-L674 注释原话）。']),
]
for cx, cwd, t, st, lines in cards:
    lc.rect(cx, CY, cwd, CH2, '#ffffff', st, rx=8, sw=1.3)
    lc.text(cx + 14, CY + 20, t, 9.3, st, 'start', True, maxw=cwd - 28, tag='c:' + t[:8])
    for i, ln in enumerate(lines):
        lc.text(cx + 14, CY + 40 + i * 18, ln, 8.0, '#334155', 'start', maxw=cwd - 28,
                tag='cl:' + t[:6] + str(i))

# ---------------- 页脚 ----------------
lc.text(MX, 622, '逐字锚 vllm/compilation/backends.py:L633-L684（wrap_with_cudagraph_if_needed 三选项+PIECEWISE 恒定注）· L687-L776（PiecewiseCompileInterpreter）· vllm/v1/attention/backends/flash_attn.py:L880-L888（NOTE 原话）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-piecewise-seams.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
