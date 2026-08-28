#!/usr/bin/env python3
"""ch18 机制图 5 · CpuGpuBuffer 三视图固定缓冲（figure_spec ch18-fig-cpugpubuffer，模板 layout）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）『执行臂中层』GPUModelRunner 框内的缓冲
地基——即本章 L2 章图 south『CpuGpuBuffer · 地址永不变』块本体的机制展开（该块方法行
1-3 的三视图/前缀拷贝/一次分配）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角
指北小签。

claim：CpuGpuBuffer 把每个 per-step 张量做成『cpu(pinned)+gpu+np 三视图一体』的固定
形状双端缓冲：地址一次分配永不再变，copy_to_gpu(n) 只把 [0,n) 活跃前缀送过 PCIe——
带宽按需、地址恒定，两个要求同时满足。

数字全部取自 figure_spec.numbers（traces/ch18_m02_reconcile.json config/beats（五拍
total 与 data_ptr）+ traces/ch18_m06_gather.json（query_start_loc 尾 pad）+
gpu_model_runner.py:L763-L810 + v1/utils.py:L139-L142）。坐标由常量/循环计算；文本全
esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 622
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '形状按 max 一次浇铸，每拍只把活跃前缀送过 PCIe——地址从头到尾没动过',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'CpuGpuBuffer（utils.py:L110-L149）：cpu(pinned)+gpu+np 三视图一体；copy_to_gpu(n) 即 gpu[:n].copy_(cpu[:n], non_blocking=True)——input_ids 缓冲按 64 分配，五拍只拷 5/2/3/1/4',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 南行 CpuGpuBuffer 地址永不变 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 三层：cpu(pinned) / PCIe / gpu ----------------
BUF_X, BUF_W = MX, 880
N_CELL = 64
CELL_PITCH = BUF_W / N_CELL                    # 13.75
CPU_Y, BAR_H = 116, 40
PCIE_Y = CPU_Y + BAR_H + 44                    # 200
GPU_Y = PCIE_Y + BAR_H + 44                    # 284

def buffer_bar(y, tag):
    for i in range(N_CELL):
        lc.rect(BUF_X + i * CELL_PITCH, y, CELL_PITCH - 1.2, BAR_H, '#f8fafc', '#e2e8f0',
                rx=1.5, sw=0.6)

# ---- 上层：cpu(pinned) —— 本拍（拍 5）只写活跃前缀 total=4 ----
ACTIVE = 4
buffer_bar(CPU_Y, 'cpu')
for i in range(ACTIVE):
    lc.rect(BUF_X + i * CELL_PITCH, CPU_Y, CELL_PITCH - 1.2, BAR_H, lc.C_KV_S, lc.C_KV_S,
            rx=1.5, sw=0)
lc.text(MX, CPU_Y - 22, 'cpu 视图 · torch.zeros(64, dtype=int32, pin_memory=True)', 10.5,
        lc.C_TXT, 'start', True, maxw=520, tag='cpu:t')
lc.text(BXR, CPU_Y - 22, 'pinned → DMA 快速路径', 8.6, lc.C_MUTE, 'end', maxw=220, tag='cpu:p')
lc.text(BUF_X + ACTIVE * CELL_PITCH + 10, CPU_Y + BAR_H / 2 + 3,
        '…其余 60 格：形状占位，本拍不写', 8.2, '#94a3b8', 'start', maxw=280, tag='cpu:rest')
lc.text(BUF_X + 2, CPU_Y - 8, '本拍活跃前缀 total=4 ↓', 8.2, lc.C_KV_S, 'start',
        True, maxw=150, tag='cpu:act')

# ---- 中层：PCIe 传送带 ----
lc.text(BUF_X + 8, PCIE_Y + BAR_H / 2 + 3.5, 'copy_to_gpu(n=4)', 11, lc.C_ENG_S, 'start',
        True, maxw=170, tag='pcie:call')
lc.text(BUF_X + 8, PCIE_Y + BAR_H / 2 + 18, 'non_blocking=True——只盖住染色的 [0,n) 段', 8.2,
        lc.C_MUTE, 'start', maxw=310, tag='pcie:nb')
# 传送带箭头：4 段小箭头盖住活跃段
for i in range(ACTIVE):
    x0 = BUF_X + i * CELL_PITCH + 1
    lc.seg(x0, PCIE_Y + 6, x0 + CELL_PITCH - 4, PCIE_Y + 6, lc.C_ENG_S, 2.2, 'std')
    lc.seg(x0, PCIE_Y + 14, x0 + CELL_PITCH - 4, PCIE_Y + 14, lc.C_ENG_S, 2.2, 'std')
# n=None 全量的对照（虚影）
lc.text(BUF_X + ACTIVE * CELL_PITCH + 16, PCIE_Y + 12, 'n=None 才全量 64 格（本例不用）', 8,
        '#94a3b8', 'start', maxw=250, tag='pcie:full')

# ---- 下层：gpu —— 同 64 格，地址锚点跨拍不变 ----
buffer_bar(GPU_Y, 'gpu')
for i in range(ACTIVE):
    lc.rect(BUF_X + i * CELL_PITCH, GPU_Y, CELL_PITCH - 1.2, BAR_H, lc.C_GPU_S, lc.C_GPU_S,
            rx=1.5, sw=0)
lc.text(MX, GPU_Y - 22, 'gpu 视图 · torch.zeros_like(cpu, device=cuda)', 10.5, lc.C_TXT,
        'start', True, maxw=520, tag='gpu:t')
lc.text(BXR, GPU_Y - 22, '接收地址 = 捕获时的地址', 8.6, lc.C_GPU_S, 'end', True, maxw=220,
        tag='gpu:a')
# 地址锚点锁形符号（左端，跨拍不变）
for ay in (CPU_Y, GPU_Y):
    lc.text(BUF_X - 14, ay + BAR_H / 2 + 4.5, '🔒', 13, lc.C_GPU_S, 'end', tag='lock' + str(ay))
lc.text(BUF_X - 14, CPU_Y - 8, 'data_ptr', 7.8, lc.C_GPU_S, 'end', maxw=60, tag='dptr')

# 五拍 data_ptr 全等注（gpu 条下方）
lc.text(BUF_X, GPU_Y + BAR_H + 18,
        '五拍 data_ptr 逐一全等（input_ids cpu/gpu · positions · query_start_loc(cpu) · seq_lens · token_ids_cpu_tensor）——稳如浇铸',
        8.6, lc.C_GPU_S, 'start', True, maxw=BUF_W, tag='gpu:stable')
lc.text(BUF_X, GPU_Y + BAR_H + 34,
        '批形状五拍四变（请求 2→2→2→1→2、token 账 5→2→3→1→4），地址一个都没动——CUDA graph 回放的直接供给。',
        8.4, lc.C_MUTE, 'start', maxw=BUF_W, tag='gpu:note')

# np 视图虚挂（cpu 条左下角注）
lc.text(BUF_X, CPU_Y + BAR_H + 16, 'np 视图 = cpu.numpy() 零拷贝（Python 侧直接改格子）', 8.2,
        lc.C_MUTE, 'start', maxw=430, tag='np:v')

# ---------------- 右上：三视图小卡 + qsl 尾 pad 实例 ----------------
RV_X, RV_Y, RV_W, RV_H = 990, 116, 450, 190
lc.rect(RV_X, RV_Y, RV_W, RV_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(RV_X + 14, RV_Y + 20, '三视图一体（utils.py:L110-L137）', 10.5, lc.C_TXT, 'start', True,
        maxw=RV_W - 28, tag='rv:t')
rv_rows = [
    ('self.cpu', 'torch CPU 张量（pinned）', lc.C_KV_S),
    ('self.gpu', 'torch GPU 张量（同形状）', lc.C_GPU_S),
    ('self.np', 'numpy 视图（零拷贝共享内存）', lc.C_ZMQ_S),
]
for i, (n, d, c) in enumerate(rv_rows):
    y = RV_Y + 44 + i * 30
    lc.rect(RV_X + 14, y - 12, 96, 22, '#ffffff', c, rx=5, sw=1.3)
    lc.text(RV_X + 62, y + 3, n, 9, c, 'middle', True, maxw=90, tag='rv:n' + str(i))
    lc.text(RV_X + 124, y + 3, d, 8.8, '#334155', 'start', maxw=RV_W - 140, tag='rv:d' + str(i))
lc.text(RV_X + 14, RV_Y + 44 + 3 * 30 + 8, 'copy_to_gpu(n) 语义原文（utils.py:L139-L142）：',
        8.4, lc.C_TXT, 'start', True, maxw=RV_W - 28, tag='rv:q:t')
lc.text(RV_X + 14, RV_Y + 44 + 3 * 30 + 24, 'gpu[:n].copy_(cpu[:n], non_blocking=True)',
        8.4, '#334155', 'start', maxw=RV_W - 28, tag='rv:q1')
lc.text(RV_X + 14, RV_Y + 44 + 3 * 30 + 40, '——n 省略（None）时才整条 copy_(cpu)。',
        8.2, lc.C_MUTE, 'start', maxw=RV_W - 28, tag='rv:q2')

# qsl 尾 pad 实例（右中）
QS_Y = RV_Y + RV_H + 18
lc.rect(RV_X, QS_Y, RV_W, 148, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(RV_X + 14, QS_Y + 20, 'query_start_loc：按 max_num_reqs+1=5 分配', 10.5, lc.C_TXT,
        'start', True, maxw=RV_W - 28, tag='qs:t')
lc.text(RV_X + 14, QS_Y + 38, '值域 [0,2,7,10,10]——尾部 pad 到非递减，FlashAttention 类 kernel 直接消费（L2073-L2078）',
        8.2, lc.C_MUTE, 'start', maxw=RV_W - 28, tag='qs:s')
qsl_vals = [0, 2, 7, 10, 10]
cw_ = (RV_W - 28 - 4 * 8) / 5
for i, v in enumerate(qsl_vals):
    x = RV_X + 14 + i * (cw_ + 8)
    tail = i == 4                              # 3 请求：前 4 格真 CU 偏移，末 1 格 pad
    lc.rect(x, QS_Y + 52, cw_, 40, '#ffffff', lc.C_ZMQ_S, rx=6,
            sw=1.5, dash=tail)
    lc.text(x + cw_ / 2, QS_Y + 78, str(v), 13, lc.C_ZMQ_S, 'middle', True, maxw=cw_ - 6,
            tag=f'qs{i}')
lc.text(RV_X + 14, QS_Y + 108, '3 请求：前 4 格 = 真 CU 偏移，末 1 格 pad（虚线格）；', 8.2,
        '#334155', 'start', maxw=RV_W - 28, tag='qs:p1')
lc.text(RV_X + 14, QS_Y + 124, '固定形状因此可被 CUDA graph 捕获。', 8.2, '#334155', 'start',
        maxw=RV_W - 28, tag='qs:p2')

# ---------------- 左下：runner 持久缓冲清单（三列） ----------------
PB_X, PB_Y, PB_W, PB_H = MX, 380, 880, 140
lc.rect(PB_X, PB_Y, PB_W, PB_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(PB_X + 14, PB_Y + 19, '「Persistent buffers for CUDA graphs」——runner __init__ 一次分配清单（L763-L810）',
        9.6, lc.C_TXT, 'start', True, maxw=PB_W - 28, tag='pb:t')
NAMES = ['input_ids', 'positions', 'query_start_loc', 'seq_lens',
         'optimistic_seq_lens_cpu', 'num_computed_tokens', 'prev_num_draft_tokens',
         'req_indices', 'prev_positions', 'num_scheduled_tokens', 'is_token_ids',
         'discard_request_mask', 'num_decode_draft_tokens', 'num_accepted_tokens']
col_w = (PB_W - 28) / 3 - 8
for i, n in enumerate(NAMES):
    col = i // 5
    row = i % 5
    x = PB_X + 14 + col * (col_w + 12)
    y = PB_Y + 44 + row * 17
    lc.text(x, y, '· ' + n, 8, '#334155', 'start', maxw=col_w, tag='pb:' + n)
lc.text(PB_X + 14, PB_Y + PB_H - 10, '地址一次分配、永不再变——回放命中的前提（证据见本章「捕获 vs 回放」图）',
        8, lc.C_MUTE, 'start', maxw=PB_W - 28, tag='pb:f')

# ---------------- 右下：五拍拷贝量 vs 缓冲恒 64 ----------------
ST_Y = PB_Y + PB_H + 40
lc.text(MX, ST_Y, '五拍并排：每拍 copy_to_gpu 的 n 与缓冲全长', 9.5, lc.C_TXT, 'start', True,
        maxw=420, tag='st:t')
tot = [5, 2, 3, 1, 4]
bx = MX + 240
for i, v in enumerate(tot):
    x = bx + i * 116
    lc.rect(x, ST_Y - 11, 104, 26, '#ffffff', lc.C_MUTE, rx=5, sw=1.1)
    # n 段染色
    seg_w = 104 * v / N_CELL
    if seg_w > 1:
        lc.rect(x, ST_Y - 11, seg_w, 26, lc.C_ENG_S, lc.C_ENG_S, rx=5, sw=0)
    lc.text(x + 52, ST_Y + 6, f'拍{i + 1} n={v}', 8.4, lc.C_TXT, 'middle', True, maxw=100,
            tag=f'st{i}')
lc.text(bx + 5 * 116 + 14, ST_Y + 6, '缓冲全长恒 64——拷的少，不代表格子少', 8.6, lc.C_MUTE,
        'start', maxw=BXR - (bx + 5 * 116 + 14), tag='st:c')

lc.text(MX, ST_Y + 30, '逐字锚 vllm/v1/utils.py:L110-L149（CpuGpuBuffer 全文）· L139-L142（copy_to_gpu 语义）· vllm/v1/worker/gpu_model_runner.py:L763-L810（Persistent buffers 清单）· L2073-L2078（qsl 尾 pad 注释）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, ST_Y + 46, '五拍 total 与 data_ptr 读数取自精简版 companion host 实测（六缓冲×五拍 data_ptr 全等）· qsl 尾 pad 读数取自 3 请求单拍实录 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch18-fig-cpugpubuffer.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
