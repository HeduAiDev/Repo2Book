#!/usr/bin/env python3
"""ch22 机制图 4 · commit 先行拷贝与 CPU/GPU 双镜像（figure_spec ch22-fig-commit-overlap，模板 swimlane）

放大自 L0『GPU 执行臂』（gpu_column 绿列）——即本章 L2 章图 center 拍片 ②
『commit_block_table · 先行拷贝』（站 3）的机制展开；基座 = south 区
『CpuGpuBuffer · 双镜像基座』（ch18 已立）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）。

claim：块表是 CPU/GPU 双镜像页表：CPU 侧行长账增量写行，每拍第一句 commit 只拷
活跃行先发车——H2D 拷贝与后续 CPU 活重叠。

数字全部取自 figure_spec.numbers（2/8/64/4/5/88，精简版 host 实跑：行长账两拍接龙 +
commit 只拷活跃行 + 迟到写不上镜像）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 656
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '每拍第一句先发车：commit 只拷 2/8 活跃行，PCIe 拷贝与后续 CPU 活重叠',
        16.5, lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '块表是 CPU/GPU 双镜像页表（CpuGpuBuffer 基座，ch18 已立）：CPU 侧 numpy 行长账增量追加；copy_to_gpu(num_reqs) 只送活跃行前缀——64B 在路上，CPU 同时干别的活',
        10.5, lc.C_MUTE, 'start', maxw=1090, tag='subtitle')
_ch = '放大自 L2 拍片 ② commit_block_table · 先行拷贝 · L0：GPU 执行臂 × KV 接缝'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 双镜像账本（CPU / GPU 两个 8×8 格盘 + PCIe 传送带） ----------------
GX_CPU, GX_GPU, GY = 250, 770, 128
CW_, CH_ = 30, 20
NROW, NCOL = 8, 8
ROW0 = [3, 8, 2, 7, 1, 0, 0, 0]          # 两拍 append 接龙 → 行长账 5
ROW1 = [9, 4, 6, 8, 0, 0, 0, 0]          # add_row 整行重写 → 行长 4

lc.text(GX_CPU, 112, '① CPU 侧 · numpy 行长账', 10, lc.C_KV_S, 'start', True, maxw=220,
        tag='lab:cpu')
lc.text(630, 112, '② PCIe · H2D', 10, lc.C_ENG_S, 'middle', True, maxw=200, tag='lab:pcie')
lc.text(GX_GPU, 112, '③ GPU 镜像 · 固定地址', 10, lc.C_GPU_S, 'start', True, maxw=230,
        tag='lab:gpu')

def draw_grid(gx, row0, row1, gpu):
    """8×8 账本格盘：r0=r0 两拍增量（拍1 青/拍2 橙）、r1=add_row、其余空行"""
    for r in range(NROW):
        y = GY + r * CH_
        lc.text(gx - 8, y + 13, f'r{r}', 6.8, '#94a3b8', 'end', maxw=22, tag=f'rl{r}')
        for c in range(NCOL):
            x = gx + c * CW_
            if r == 0:
                v = ROW0[c]
                if c < 3:
                    fill, st, tc = lc.C_KV_F, lc.C_KV_S, lc.C_KV_S      # 拍1 append
                elif c < 5:
                    fill, st, tc = lc.C_BEAT_F, lc.C_BEAT_S, lc.C_BEAT_T  # 拍2 append（增量）
                else:
                    fill, st, tc = '#f8fafc', '#e2e8f0', '#94a3b8'
            elif r == 1:
                v = ROW1[c]
                if c < 4:
                    fill, st, tc = lc.C_KV_F, lc.C_KV_S, lc.C_KV_S
                else:
                    fill, st, tc = '#f8fafc', '#e2e8f0', '#94a3b8'
            else:
                v = 0
                fill, st, tc = '#f8fafc', '#e2e8f0', '#94a3b8'
            if gpu and v == 0 and r < 2:
                fill, st, tc = '#f0fdf4', '#bbf7d0', '#166534'          # 拷过去的整行（含尾零）
            lc.rect(x, y, CW_ - 1.5, CH_ - 1.5, fill, st, rx=3, sw=1.0)
            lc.text(x + CW_ / 2, y + 13, str(v), 8.5, tc, 'middle', True, maxw=CW_ - 4,
                    tag=f'g{r}{c}')

draw_grid(GX_CPU, ROW0, ROW1, gpu=False)
draw_grid(GX_GPU, ROW0, ROW1, gpu=True)
GRID_R = GY + NROW * CH_                                          # 288

# PCIe 传送带
BTX0, BTX1, BTY0, BTY1 = 520, 740, 176, 240
lc.rect(BTX0, BTY0, BTX1 - BTX0, BTY1 - BTY0, '#fff7ed', lc.C_ENG_S, rx=9, sw=1.6)
lc.text(630, BTY0 + 18, 'copy_to_gpu(num_reqs=2)', 9.5, lc.C_ENG_S, 'middle', True, maxw=200,
        tag='belt:t')
chip_w = 96
for ci, base in enumerate((BTX0 + 12, BTX0 + 12 + chip_w + 12)):
    row = ROW0 if ci == 0 else ROW1
    for c in range(8):
        lc.rect(base + c * 12, BTY0 + 26, 11, 11, '#ffffff' if row[c] else '#f1f5f9',
                lc.C_ENG_S, rx=2, sw=0.7)
        if row[c]:
            lc.rect(base + c * 12, BTY0 + 26, 11, 11, lc.C_ENG_S, lc.C_ENG_S, rx=2, sw=0)
lc.text(630, BTY0 + 52, '只带 2/8 行', 8.2, lc.C_ENG_S, 'middle', True, maxw=120, tag='belt:n')
lc.text(630, GRID_R + 2, '2 行 × 8 项 × 4B(int32) = 64B · non_blocking=True', 8.4, lc.C_ENG_S,
        'middle', True, maxw=276, tag='belt:b')
lc.text(630, GRID_R + 18, '发车时机 = _prepare_inputs 第一句', 8.2, lc.C_MUTE, 'middle',
        maxw=276, tag='belt:t2')

# ①→②→③ 肘形箭头（行 0 高度）
MIDY = GY + CH_ / 2
lc.parrow([(GX_CPU + NCOL * CW_, MIDY), (500, MIDY), (500, 206), (BTX0, 206)],
          lc.C_ENG_S, 1.8, 'std')
lc.parrow([(BTX1, 206), (760, 206), (760, MIDY), (GX_GPU, MIDY)],
          lc.C_ENG_S, 1.8, 'std')

# 账本注记（格盘下方）
lc.text(GX_CPU, GRID_R + 22, '拍1 append_row([3,8,2]) + 拍2 append_row([7,1])', 7.8, '#334155',
        'start', maxw=240, tag='ca:1')
lc.text(GX_CPU, GRID_R + 37, '→ 行长账 num_blocks_per_row[0] = 5，新块接着写', 7.8, '#334155',
        'start', maxw=240, tag='ca:2')
lc.text(GX_CPU, GRID_R + 52, 'r1 = 新增请求：add_row 整行重写（行长 4）', 7.8, '#334155',
        'start', maxw=240, tag='ca:3')
lc.text(GX_GPU, GRID_R + 22, 'slot_mapping kernel 与 attention 查的都是这份镜像', 7.8,
        lc.C_GPU_S, 'start', True, maxw=240, tag='ga:1')
lc.text(GX_GPU, GRID_R + 37, '（CpuGpuBuffer 双镜像基座 · 地址固定，ch18 已立）', 7.8,
        lc.C_MUTE, 'start', maxw=240, tag='ga:2')

# 证据卡：迟到写不上镜像
EVX, EVY, EVW, EVH = 1040, 128, 400, 160
lc.rect(EVX, EVY, EVW, EVH, '#ffffff', lc.C_MUTE, rx=9, sw=1.3, dash=True)
lc.text(EVX + 14, EVY + 19, '只拷前缀是设计——迟到写不上镜像', 9.6, lc.C_TXT, 'start', True,
        maxw=EVW - 28, tag='ev:t')
lc.text(EVX + 14, EVY + 38, 'CPU 行 2（非活跃）拍后被写 88：', 8, '#334155', 'start',
        maxw=EVW - 28, tag='ev:l1')
for c in range(6):
    x = EVX + 14 + c * 34
    if c == 0:
        lc.rect(x, EVY + 46, 30, 20, '#fef2f2', lc.C_ABORT, rx=4, sw=1.3)
        lc.text(x + 15, EVY + 60, '88', 8.6, lc.C_ABORT, 'middle', True, maxw=26, tag='ev:88')
    else:
        lc.rect(x, EVY + 46, 30, 20, '#f8fafc', '#e2e8f0', rx=4, sw=0.8)
        lc.text(x + 15, EVY + 60, '0', 8.6, '#94a3b8', 'middle', maxw=26, tag=f'ev:c{c}')
lc.text(EVX + 14, EVY + 82, '× 本拍 commit 已过，不上镜像', 8, lc.C_ABORT, 'start', True,
        maxw=EVW - 28, tag='ev:x')
lc.text(EVX + 14, EVY + 100, 'GPU 镜像行 2 纹丝不动：', 8, '#334155', 'start', maxw=EVW - 28,
        tag='ev:l2')
for c in range(6):
    x = EVX + 14 + c * 34
    lc.rect(x, EVY + 108, 30, 20, '#f0fdf4', '#bbf7d0', rx=4, sw=0.8)
    lc.text(x + 15, EVY + 122, '0', 8.6, '#166534', 'middle', maxw=26, tag=f'ev:g{c}')
lc.text(EVX + 14, EVY + 144, '迟到写只进 CPU 账本，下一拍 commit 才会带上（stale 不上 GPU）',
        7.6, lc.C_MUTE, 'start', maxw=EVW - 28, tag='ev:f')

# ---------------- 时序面板：先行拷贝为什么省时间 ----------------
PY0, PY1 = 356, 584
lc.rect(MX, PY0, BXR - MX, PY1 - PY0, '#ffffff', lc.C_MUTE, rx=9, sw=1.3)
lc.text(MX + 16, PY0 + 21, '④ 先行拷贝为什么省时间——_prepare_inputs 开头的时序（左：实际；右：对照）',
        9.8, lc.C_TXT, 'start', True, maxw=800, tag='tl:t')
TX0, TX1 = 320, 1020
lc.text(300, 424, 'CPU', 8.6, lc.C_KV_S, 'end', True, maxw=60, tag='tr:cpu')
lc.text(300, 456, 'PCIe', 8.6, lc.C_ENG_S, 'end', True, maxw=60, tag='tr:pcie')
# CPU 轨：commit 发车（短橙）+ 后续 CPU 活（长青）
lc.rect(TX0, 414, 34, 18, lc.C_ENG_S, lc.C_ENG_S, rx=3, sw=0)
lc.text(TX0 + 17, 427, 'commit', 6.8, '#ffffff', 'middle', True, maxw=32, tag='tl:c0')
lc.rect(TX0 + 34, 414, TX1 - TX0 - 34, 18, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
lc.text((TX0 + 34 + TX1) / 2, 427, 'np.repeat · index_select · positions 在 GPU 端组装 · …后续 CPU 活',
        7.6, lc.C_KV_S, 'middle', True, maxw=TX1 - TX0 - 44, tag='tl:c1')
# PCIe 轨：块表 64B 在路上
PCIE_END = TX0 + 270
lc.rect(TX0, 446, PCIE_END - TX0, 18, '#fdba74', lc.C_ENG_S, rx=3, sw=1.2)
lc.text((TX0 + PCIE_END) / 2, 459, '块表 64B H2D 在路上（non_blocking，不堵 CPU）', 7.6,
        '#9a3412', 'middle', True, maxw=PCIE_END - TX0 - 8, tag='tl:p1')
# 重叠带
lc.rect(TX0, 406, PCIE_END - TX0, 66, 'none', lc.C_GPU_S, rx=4, sw=1.2, dash=True)
lc.text((TX0 + PCIE_END) / 2, 401, '重叠：拷贝在路上，CPU 同时干活', 8, lc.C_GPU_S, 'middle',
        True, maxw=PCIE_END - TX0, tag='tl:ov')
# 时刻标记
lc.seg(TX0, 400, TX0, 476, '#94a3b8', 1.0)
lc.text(TX0, 490, 't0 开拍', 7.4, '#94a3b8', 'middle', maxw=70, tag='tl:t0')
lc.seg(PCIE_END, 464, PCIE_END, 476, '#94a3b8', 1.0)
lc.text(PCIE_END, 490, 't1 GPU 拿到 2 行', 7.4, lc.C_ENG_S, 'middle', True, maxw=110,
        tag='tl:t1')
# 对照（串行）
DIV = 1060
lc.seg(DIV, 400, DIV, 476, '#e2e8f0', 1.2)
lc.text(DIV + 14, 416, '对照：commit 排到 CPU 活之后（串行）', 8.2, '#64748b', 'start', True,
        maxw=340, tag='gh:t')
lc.rect(DIV + 14, 426, 170, 14, '#f1f5f9', '#cbd5e1', rx=3, sw=0.8)
lc.text(DIV + 99, 436, '后续 CPU 活', 7, '#64748b', 'middle', True, maxw=160, tag='gh:c')
lc.rect(DIV + 184, 426, 120, 14, 'none', '#cbd5e1', rx=3, sw=0.9, dash=True)
lc.text(DIV + 244, 436, '再拷 64B', 7, '#94a3b8', 'middle', True, maxw=110, tag='gh:p')
lc.text(DIV + 308, 436, '→', 9, '#94a3b8', 'middle', maxw=16, tag='gh:a')
lc.text(DIV + 14, 456, 'GPU 更晚拿到块表——先行发车省的就是这段', 7.4, '#64748b', 'start',
        maxw=340, tag='gh:n')
# 注释原话
QY = 510
lc.text(MX + 16, QY, '注释原话（gpu_model_runner.py:L1977-L1979）：', 8.4, lc.C_TXT, 'start',
        True, maxw=500, tag='q:t')
lc.text(MX + 16, QY + 17, '# OPTIMIZATION: Start copying the block table first.', 8.4,
        '#334155', 'start', maxw=700, tag='q:1')
lc.text(MX + 16, QY + 34, '# This way, we can overlap the copy with the following CPU operations.',
        8.4, '#334155', 'start', maxw=760, tag='q:2')
lc.text(MX + 16, QY + 54, 'commit_block_table(num_reqs) = block_table.copy_to_gpu(num_reqs)（block_table.py:L213-L214）；append_row / add_row / 行长账 = block_table.py:L138-L158',
        7.8, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='q:3')

# ---------------- 页脚 ----------------
lc.text(MX, 614, '图例：青 = CPU 侧行长账 / 后续 CPU 活 · 橙 = PCIe H2D（先行发车） · 绿 = GPU 镜像 · 红 = 迟到写（不上镜像的证据）',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 630, '行长账/拷贝量/迟到写读数取自精简版 host 实跑（CpuGpuBuffer 逐字 vllm/v1/utils.py:L110-L149）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch22-fig-commit-overlap.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
