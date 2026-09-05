#!/usr/bin/env python3
"""ch20 机制图 ① · 带宽墙:慢在搬运,不在计算(figure_spec ch20-fig-bandwidth-wall,模板 flow)

放大自 L0 中列『GPU 执行臂』(绿色列)第三块『模型层 forward + 编译』内 Attention 插座里的
attention kernel——ch19 捕进 CUDA graph 的那个不透明算子节点的内部。primer 推导链第 ① 环。
架构归属回指 L0(FIGURE-SYSTEM §3.3):图右上角指北小签。

claim:标准注意力 Alg.0 三步把 S=QK^T、P=softmax(S) 两张 N×N 写出 HBM 再读回,
wall-clock 被 HBM 访存支配(SRAM 比 HBM 快 9.5-12.67 倍);vLLM 主路径一次
flash_attn_varlen_func 调用零张 N×N 物化。

数字全部取自 figure_spec.numbers(A100 层级逐字论文原文 arXiv:2205.14135 §2.1;
表尺寸元素级实算 8K 上下文/GPT-2 两组;vLLM 对照 flash_attn.py:L1041-L1066)。
坐标由常量/循环计算;文本全 esc();配色 l0_common(GPU 绿=C_GPU_S 恒为角色色)。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 60
BXR = 1440
EXTRA_DEFS = ('<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
              '<marker id="cyn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0891b2"/></marker>'
              '<marker id="red" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
              'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>')

C_RED = '#dc2626'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '慢在搬运,不在计算——标准注意力把 S、P 两张 N×N 表在显存里搬进搬出',
        16.5, lc.C_TXT, 'start', True, maxw=960, tag='title')
lc.text(MX, 58, 'A100 上 SRAM 比 HBM 快 9.5-12.67 倍、容量却小三个数量级(合计 20736KB vs 40-80GB);三步实现的 wall-clock 被整表访存卡死(arXiv:2205.14135 §2.1-2.2)',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '推导链 ① · 放大自 L0 GPU 执行臂(绿列)模型层 forward 内 attention kernel'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左栏:GPU 两级存储(竖剖) ----------------
lc.text(MX, 96, '先看舞台:同一块 GPU 的两级存储(A100)', 12, lc.C_TXT, 'start', True,
        maxw=420, tag='lp:t')

# GPU 芯片框(片上 SRAM)
CHIP_X, CHIP_Y, CHIP_W, CHIP_H = MX, 116, 420, 308
lc.rect(CHIP_X, CHIP_Y, CHIP_W, CHIP_H, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(CHIP_X + 14, CHIP_Y + 22, '片上 SRAM(GPU 芯片内,每 SM 一块)', 11.5, lc.C_GPU_S,
        'start', True, maxw=CHIP_W - 28, tag='sram:t')
# 108 格 SRAM(12 列 × 9 行 = 108 个 SM,格数与 SM 数严格一致)
CELL, GAP = 22, 3.5
GRID_W = 12 * CELL + 11 * GAP
GX0 = CHIP_X + (CHIP_W - GRID_W) / 2
GY0 = CHIP_Y + 40
for r in range(9):
    for c in range(12):
        lc.rect(GX0 + c * (CELL + GAP), GY0 + r * (CELL + GAP), CELL, CELL,
                '#bbf7d0', lc.C_GPU_S, rx=2, sw=0.8)
GRID_H = 9 * CELL + 8 * GAP
lc.text(CHIP_X + CHIP_W / 2, GY0 + GRID_H + 20, '108 个 SM × 192KB/SM(每格一个 SM 的 SRAM)',
        9.5, '#334155', 'middle', maxw=CHIP_W - 20, tag='sram:n')
lc.text(CHIP_X + CHIP_W / 2, GY0 + GRID_H + 38,
        '合计 20736KB ≈ 20MB · 带宽估算 ~19TB/s', 10.5, lc.C_GPU_S, 'middle', True,
        maxw=CHIP_W - 20, tag='sram:bw')

# load/store 双箭头(芯片 ↔ HBM)
BRIDGE_Y0, BRIDGE_Y1 = CHIP_Y + CHIP_H, 500
lc.seg(170, BRIDGE_Y0, 170, BRIDGE_Y1, lc.C_KV_S, 2.0, 'cyn')
lc.seg(270, BRIDGE_Y1, 270, BRIDGE_Y0, lc.C_GPU_S, 2.0, 'grn')
lc.text(170, BRIDGE_Y1 - 10, 'store 写下去', 8.5, lc.C_KV_S, 'middle', maxw=90, tag='store')
lc.text(270, BRIDGE_Y1 - 10, 'load 搬上来', 8.5, lc.C_GPU_S, 'middle', maxw=90, tag='load')
lc.rect(316, 440, 164, 50, '#ffffff', lc.C_MUTE, rx=8, sw=1.1)
lc.text(398, 458, '带宽比 9.5-12.67 倍', 9.5, lc.C_BEAT_T, 'middle', True, maxw=156, tag='ratio:t')
lc.text(398, 476, 'SRAM ~19 vs HBM 1.5-2.0TB/s', 8, lc.C_MUTE, 'middle', maxw=156, tag='ratio:s')

# HBM 大水池
HBM_Y, HBM_H = 500, 104
lc.rect(MX, HBM_Y, CHIP_W, HBM_H, lc.C_KV_F, lc.C_KV_S, rx=10, sw=2.0)
lc.text(MX + 14, HBM_Y + 24, 'HBM 高带宽显存(片外大水池)', 11.5, lc.C_KV_S, 'start', True,
        maxw=CHIP_W - 28, tag='hbm:t')
lc.text(MX + 14, HBM_Y + 48, '容量 40-80GB', 10.5, '#334155', 'start', True, maxw=190, tag='hbm:cap')
lc.text(MX + 220, HBM_Y + 48, '带宽 1.5-2.0TB/s', 10.5, '#334155', 'start', True, maxw=190, tag='hbm:bw')
lc.text(MX + 14, HBM_Y + 72, '容量比 ≈1/2000-1/4000,小三个数量级(20736KB vs 40-80GB)', 9, lc.C_MUTE,
        'start', maxw=CHIP_W - 28, tag='hbm:cmp')

# 左栏直觉小结
lc.rect(MX, 624, CHIP_W, 66, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(MX + 14, 644, '直觉:softmax 要把一整行分数全部加起来,像图书馆', 9.5, lc.C_TXT,
        'start', True, maxw=CHIP_W - 28, tag='note:t')
lc.text(MX + 14, 661, '必须把整页借齐才能点数——标准实现干脆把整页复印两', 9, '#334155',
        'start', maxw=CHIP_W - 28, tag='note:l1')
lc.text(MX + 14, 677, '遍(S、P 两张 N×N)摊在桌上(HBM)再拿起来算——时间都花在搬纸上,不在点数上', 9,
        '#334155', 'start', maxw=CHIP_W - 28, tag='note:l2')

# ---------------- 右栏:Alg.0 三步流水 ----------------
lc.text(524, 96, '标准实现 Alg.0:三步流水,同一张表的 4 次整表搬运(以 8K 上下文一个 head 计)',
        12, lc.C_TXT, 'start', True, maxw=916, tag='rp:t')
FX, FW = 524, 420
FCX = FX + FW / 2
LX = 954          # 右侧标注列 x
LW = BXR - LX     # 右侧标注列宽


def flowbox(y, h, line1, line2=None, kind='comp'):
    fill, stroke = ('#ffffff', lc.C_GPU_S) if kind == 'comp' else (lc.C_KV_F, lc.C_KV_S)
    lc.rect(FX, y, FW, h, fill, stroke, rx=8, sw=1.8)
    if line2:
        lc.text(FCX, y + 17, line1, 10.5, lc.C_TXT, 'middle', True, maxw=FW - 16, tag='fb:' + line1[:10])
        lc.text(FCX, y + 33, line2, 9, lc.C_MUTE, 'middle', maxw=FW - 16, tag='fb2:' + line2[:10])
    else:
        lc.text(FCX, y + h / 2 + 4, line1, 10.5, lc.C_TXT, 'middle', True, maxw=FW - 16,
                tag='fb:' + line1[:10])


def hbm_band(y, name):
    lc.rect(FX, y, FW, 46, lc.C_KV_F, lc.C_KV_S, rx=8, sw=2.2)
    lc.text(FCX, y + 19, f'HBM 里的 {name} 表(N×N)', 10.5, lc.C_KV_S, 'middle', True,
            maxw=FW - 16, tag='hb:' + name)
    lc.text(FCX, y + 36, '8192 × 8192 = 67108864 元素(fp16 下 ≈ 134.2MB)', 9, '#334155',
            'middle', maxw=FW - 16, tag='hb2:' + name)


flowbox(116, 38, 'Q, K, V ∈ R^(N×d) · 住 HBM')
lc.seg(FCX, 154, FCX, 174, lc.C_GPU_S, 2.0, 'grn')
lc.text(LX, 168, '读 Q,K(输入,轻)', 9.5, lc.C_GPU_S, 'start', maxw=LW, tag='a:in')
flowbox(178, 38, '① S = QK^T(片上 GEMM)')
lc.seg(FCX, 216, FCX, 244, C_RED, 3.2, 'red')
lc.text(LX, 228, '写 S 出 HBM', 9.5, C_RED, 'start', True, maxw=LW, tag='a:ws')
lc.text(LX, 242, '67108864 元素 ≈ 134.2MB(fp16)', 9, '#334155', 'start', maxw=LW, tag='a:ws2')
hbm_band(248, 'S')
lc.seg(FCX, 294, FCX, 322, C_RED, 3.2, 'red')
lc.text(LX, 302, '读 S 回来', 9.5, C_RED, 'start', True, maxw=LW, tag='a:rs')
lc.text(LX, 316, '又 67108864 元素', 9, '#334155', 'start', maxw=LW, tag='a:rs2')
flowbox(326, 38, '② P = softmax(S)(片上)')
lc.seg(FCX, 364, FCX, 392, C_RED, 3.2, 'red')
lc.text(LX, 372, '写 P 出 HBM', 9.5, C_RED, 'start', True, maxw=LW, tag='a:wp')
lc.text(LX, 386, '67108864 元素', 9, '#334155', 'start', maxw=LW, tag='a:wp2')
hbm_band(396, 'P')
lc.seg(FCX, 442, FCX, 470, C_RED, 3.2, 'red')
lc.text(LX, 450, '读 P 回来(+ 读 V,轻)', 9.5, C_RED, 'start', True, maxw=LW, tag='a:rp')
lc.text(LX, 464, '67108864 元素', 9, '#334155', 'start', maxw=LW, tag='a:rp2')
flowbox(474, 38, '③ O = P·V(片上)')
lc.seg(FCX, 512, FCX, 532, lc.C_GPU_S, 2.0, 'grn')
lc.text(LX, 526, '写 O(输出,轻)', 9.5, lc.C_GPU_S, 'start', maxw=LW, tag='a:out')
flowbox(536, 36, 'O ∈ R^(N×d) · HBM')

lc.text(LX, 566, '写S·读S·写P·读P:同一张量级的整表搬运 ×4 次', 9.5, C_RED, 'start', True,
        maxw=LW, tag='sum:4x')
lc.text(LX, 586, 'S、P 两张合计 134217728 元素 / 268435456 字节(fp16)', 9, '#334155',
        'start', maxw=LW, tag='sum:bytes')
lc.text(LX, 610, 'GPT-2 尺寸对照(N=1024、d=64):两张表共 2097152 元素、4194304 字节(fp16)',
        9, lc.C_MUTE, 'start', maxw=LW, tag='gpt2')

# vLLM 对照(虚线框)
CB_Y, CB_H = 630, 60
lc.rect(LX, CB_Y, LW, CB_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.4, dash=True)
lc.text(LX + 12, CB_Y + 19, 'vLLM 主路径对照(本章主角):flash_attn_varlen_func 一次调用 = 一个融合 kernel', 9.5,
        lc.C_GPU_S, 'start', True, maxw=LW - 24, tag='cb:t')
lc.text(LX + 12, CB_Y + 37, 'S/P 只在 SRAM 里以块存在——物化 N×N 表 0 张;这正是本章要讲清的 kernel 内部', 9,
        '#334155', 'start', maxw=LW - 24, tag='cb:l1')
lc.text(LX + 12, CB_Y + 53, 'vllm/v1/attention/backends/flash_attn.py:L1041-L1066', 8.5,
        lc.C_FAINT, 'start', maxw=LW - 24, tag='cb:file')

# ---------------- 页脚:图例 + 出处 ----------------
lc.text(MX, 738, '图例:红粗箭头 = N×N 整表搬运(痛点) · 绿箭头 = 轻量输入/输出 · 青框 = HBM 里的存储 · 绿框 = 片上计算',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 758, 'softmax 本尊(elementwise/reduction)是 memory-bound,kernel fusion 是常规武器;但朴素融合救不了注意力——中间结果 S、P 太大必须落 HBM,除非换算法不物化(arXiv:2205.14135 §2.1)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 776, 'A100 内存层级数字逐字取自论文原文(arXiv:2205.14135 §2.1)· 表尺寸为元素级实算 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-bandwidth-wall.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
