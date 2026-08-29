#!/usr/bin/env python3
"""ch22 机制图 5 · 写腿：ForwardContext 取表 → reshape_and_cache_flash 散写
（figure_spec ch22-fig-write-leg，模板 flow）

放大自 L0『GPU 执行臂』（gpu_column 绿列）——即本章 L2 章图 south『写腿 ·
unified_kv_cache_update』（站 12-13）的机制展开。架构归属回指 L0/L2
（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：写腿每 token 一次直寻址：前向内按 layer_name 从 ForwardContext 取
slot_mapping，reshape_and_cache_flash 逐 token 散写——slot=50 落块 3 行 2，
slot<0 的 token 直接 return。

数字全部取自 figure_spec.numbers（50/-1/3/2/51/52/16/0，精简版 host 实跑：
op 链真身 + CUDA kernel 的逐行同构镜像）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 636
MX = 60
BXR = 1440
LX, LW = 60, 580          # 左列：调用链
RX = 700                  # 右列：slot 条 + 落点

# ---------------- 标题区 ----------------
lc.text(MX, 34, '写腿：每 token 一次直寻址散写——slot<0 的 token 直接 return',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '前向内按 layer_name 从 ForwardContext 取本层 slot_mapping（与装配时同一张 GPU 张量）→ reshape_and_cache_flash 逐 token 拆格：slot=50 落块 3 行 2，PAD token 一个不写',
        10.5, lc.C_MUTE, 'start', maxw=1090, tag='subtitle')
_ch = '放大自 L2 南行 写腿 · unified_kv_cache_update（站 12-13）· L0：GPU 执行臂 × KV 接缝'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左列：调用链 flow ----------------
NODES = [
    (96, 44, '模型前向 · Attention 层',
     ['每个 DecoderLayer 的 Attention 各持 layer_name（ch19 已立的算子化纪律）']),
    (152, 58, 'unified_kv_cache_update(key, value, layer_name)',
     ['自定义算子入口 · vllm/model_executor/layers/attention/attention.py:L775-L798']),
    (222, 66, 'get_attention_context(layer_name)',
     ['从 ForwardContext 取 slot_mapping[layer_name]（forward_context.py:L136）',
      '——与第 11 站装配的是同一张 GPU 张量（data_ptr 不变）']),
    (300, 66, 'impl.do_kv_cache_update → reshape_and_cache_flash',
     ['flash_attn.py:L1098-L1132 · CUDA kernel：cache_kernels.cu:L326-L333',
      '逐 token：blockIdx.x = token_idx，一 token 一线程']),
]
CX = LX + LW / 2
for i, (y, h, title, subs) in enumerate(NODES):
    lc.rect(LX, y, LW, h, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.5)
    lc.text(LX + 14, y + 20, title, 10.2, lc.C_GPU_S, 'start', True, maxw=LW - 28,
            tag=f'n{i}')
    for j, s in enumerate(subs):
        lc.text(LX + 14, y + 37 + j * 15, s, 7.8, '#334155', 'start', maxw=LW - 26,
                tag=f'n{i}s{j}')
    if i < len(NODES) - 1:
        ny = NODES[i + 1][0]
        lc.seg(CX, y + h, CX, ny - 2, lc.C_GPU_S, 1.8, 'std')

# kernel 本体（逐字）
KY, KH = 380, 164
lc.rect(LX, KY, LW, KH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(LX + 14, KY + 19, 'kernel 逐 token 本体（cache_kernels.cu:L326-L333，逐字）', 9,
        lc.C_GPU_S, 'start', True, maxw=LW - 28, tag='k:t')
CODE = [
    'const int64_t slot_idx = slot_mapping[token_idx];',
    '// NOTE: slot_idx can be -1 if the token is padded',
    'if (slot_idx < 0) {',
    '    return;',
    '}',
    'const int64_t block_idx = slot_idx / block_size;',
    'const int64_t block_offset = slot_idx % block_size;',
]
for i, ln in enumerate(CODE):
    lc.text(LX + 22, KY + 37 + i * 14.5, ln, 8.2, '#334155', 'start', maxw=LW - 40,
            tag=f'k{i}')
ANN = [
    '—— token0：slot=50 → 块 50//16=3、行 50%16=2；token1：slot=-1 → return',
    '—— key_dst = key_cache + block_idx*block_stride + offset*page_stride（直寻址）',
]
for i, ln in enumerate(ANN):
    lc.text(LX + 14, KY + 37 + len(CODE) * 14.5 + 6 + i * 14, ln, 7.6, lc.C_MUTE,
            'start', maxw=LW - 26, tag=f'ka{i}')

# ---------------- 右列：slot 条 → 落点 ----------------
lc.text(RX, 96, 'slot_mapping[layer_name]（本拍 4 token）', 9.5, lc.C_TXT, 'start', True,
        maxw=300, tag='s:t')
CELL_W2, CELL_H2, PITCH2, STRIP_Y = 64, 32, 72, 122
SLOTS = [50, -1, 51, 52]
for i, v in enumerate(SLOTS):
    x = RX + i * PITCH2
    pad = (v == -1)
    lc.text(x + CELL_W2 / 2, 114, f'token {i}', 7.4, '#94a3b8', 'middle', maxw=60,
            tag=f'stk{i}')
    lc.rect(x, STRIP_Y, CELL_W2, CELL_H2, lc.C_BEAT_F if pad else lc.C_GPU_F,
            lc.C_BEAT_S if pad else lc.C_GPU_S, rx=5, sw=1.5)
    lc.text(x + CELL_W2 / 2, STRIP_Y + 21, str(v), 12,
            lc.C_BEAT_T if pad else lc.C_GPU_S, 'middle', True, maxw=CELL_W2 - 6,
            tag=f'sv{i}')

# PAD 跳写盒
PB_X, PB_Y, PB_W, PB_H = 1010, 152, 430, 62
lc.rect(PB_X, PB_Y, PB_W, PB_H, '#fef2f2', lc.C_ABORT, rx=7, sw=1.3, dash=True)
lc.text(PB_X + 14, PB_Y + 19, 'slot=-1 → 直接 return（不写任何块）', 9.2, lc.C_ABORT,
        'start', True, maxw=PB_W - 28, tag='pb:t')
lc.text(PB_X + 14, PB_Y + 36, 'kernel 注释原话：slot_idx can be -1 if the token is padded', 7.8,
        '#334155', 'start', maxw=PB_W - 28, tag='pb:c')
lc.text(PB_X + 14, PB_Y + 52, 'PAD 哨兵的消费端——合法 slot 恒 ≥ 0，判别无歧义', 7.8,
        '#334155', 'start', maxw=PB_W - 28, tag='pb:n')
lc.parrow([(RX + PITCH2 + CELL_W2 / 2, STRIP_Y + CELL_H2), (RX + PITCH2 + CELL_W2 / 2, 166),
           (990, 166), (990, 183), (PB_X, 183)], lc.C_ABORT, 1.4, 'std', dash=True)

# 块 3 货栈（16 行）
SX, SY, SW_, RP = 720, 206, 220, 8.4
lc.text(SX, 196, '块 3（= 50//16）· 每行一个 token 的 K/V（head_dim=8）', 8.6, lc.C_KV_S,
        'start', True, maxw=330, tag='b:t')
HIT = {2: ('slot=50', 'token 0'), 3: ('slot=51', 'token 2'), 4: ('slot=52', 'token 3')}
for r in range(16):
    y = SY + r * RP
    if r in HIT:
        sl, tk = HIT[r]
        lc.rect(SX, y, SW_, RP - 1, lc.C_GPU_S, lc.C_GPU_S, rx=1.5, sw=0)
        lc.text(SX + SW_ / 2, y + 6, f'{sl} ← {tk}', 6.8, '#ffffff', 'middle', True,
                maxw=SW_ - 6, tag=f'h{r}')
    else:
        lc.rect(SX, y, SW_, RP - 1, lc.C_KV_F, '#cbd5e1', rx=1.5, sw=0.7)
    if r % 4 == 0:
        lc.text(SX - 6, y + 6, str(r), 6.2, '#94a3b8', 'end', maxw=22, tag=f'rn{r}')
lc.text(SX, SY + 16 * RP + 16, '三个真 token 全落块 3 的行 2/3/4——散写：每 token 一次直寻址落格',
        8, lc.C_KV_S, 'start', True, maxw=430, tag='b:n')
# 散写肘形箭头（t0/t2/t3 → 行 2/3/4；channel 670..690 下探，进栈左缘）
for ci, chx, row, hy in ((0, 686, 2, 170), (2, 678, 3, 176), (3, 670, 4, 182)):
    cx = RX + ci * PITCH2 + CELL_W2 / 2
    ry = SY + row * RP + (RP - 1) / 2
    lc.parrow([(cx, STRIP_Y + CELL_H2), (cx, hy), (chx, hy), (chx, ry), (SX, ry)],
              lc.C_GPU_S, 1.5, 'std')

# ---------------- 右下列：算术卡 + dummy 返回 ----------------
AC_Y, AC_H = 380, 74
lc.rect(RX, AC_Y, BXR - RX, AC_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(RX + 14, AC_Y + 18, '直寻址算术（恒等式逆用——与装配侧同一式）', 9.2, lc.C_TXT,
        'start', True, maxw=400, tag='ac:t')
lc.text(RX + 14, AC_Y + 36, 'block = slot // 16 → 50//16 = 3；offset = slot % 16 → 50%16 = 2', 8.4,
        '#334155', 'start', maxw=BXR - RX - 28, tag='ac:1')
lc.text(RX + 14, AC_Y + 54, '验算：3 × 16 + 2 = 50 ✓（每 token O(1)：一次除、一次模、一次写）',
        8, lc.C_MUTE, 'start', maxw=BXR - RX - 28, tag='ac:2')
DB_Y = 470
lc.rect(RX, DB_Y, BXR - RX, 74, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(RX + 14, DB_Y + 18, '返回 key.new_empty(0)：空张量作 dummy 数据依赖', 9.2, lc.C_TXT,
        'start', True, maxw=460, tag='db:t')
lc.text(RX + 14, DB_Y + 36, 'numel=0 的空张量把写腿挂进 torch.compile 图——保「先写 KV、后读 attention」的顺序（ch19 已立）',
        8, '#334155', 'start', maxw=BXR - RX - 28, tag='db:1')
lc.text(RX + 14, DB_Y + 54, '模型 forward 签名零污染：slot_mapping 不透传，按 layer_name 就地取',
        8, lc.C_MUTE, 'start', maxw=BXR - RX - 28, tag='db:2')

# ---------------- 页脚 ----------------
lc.text(MX, 584, '图例：绿 = GPU 执行臂（写腿链与落格） · 青 = KV 池页张量 · 橙 = PAD 哨兵 · 红虚线 = 跳写路径',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 600, '逐字锚 attention.py:L775-L798（算子入口）· flash_attn.py:L1098-L1132（do_kv_cache_update）· cache_kernels.cu:L326-L333（kernel 本体与 NOTE 注释）· forward_context.py:L136（slot_mapping dict）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 616, 'token 级读数（slot=50/-1/51/52、落块 3 行 2/3/4、PAD 不写、空张量返回）取自精简版 host 实跑（CUDA kernel 的逐行同构镜像）',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch22-fig-write-leg.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
