#!/usr/bin/env python3
"""ch13 机制图 7 · 一页多大：GPU 物理页布局（figure_spec ch13-fig-page-brick，模板 layout）

放大自 L0 GPU 列（绿）最底层的 KV 物理显存——即本章 L2 章图北行『GPU 物理页
布局 · 每层一张』框的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
图右上角指北小签。

claim：一块页的物理形状：2×16×8×128×2 = 65536 B——K 半页 + V 半页，每半
16 token × 8 kv_head × 128 head_dim × fp16；每层张量 reshape 成 [10, 2, 16, 8, 128]，
num_blocks = 655360 // 65536。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑：页公式 65536 B、
Llama-2-7B 每 token 每层 16384 B / 全模型 0.5 MB / 4096 序列 2 GiB、
worker 换算 10 块与视图、DEFAULT_BLOCK_SIZE=16）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
C_K, C_V = lc.C_API_S, lc.C_ENG_S        # K 半页蓝 / V 半页橙（图例兜底）
F_K, F_V = lc.C_API_F, lc.C_ENG_F

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一页多大：2 × 16 × 8 × 128 × 2 B = 65536 B——K 半页 ＋ V 半页',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'real_page_size_bytes = 2(K,V) × block_size × num_kv_heads × head_dim × dtype 字节——DEFAULT_BLOCK_SIZE = 16（vllm/config/cache.py:L47），分配 / 哈希 / 寻址三处共用的最小粒度',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L2 章图北行「GPU 物理页布局 · 每层一张」框 · L0：GPU 列最底层显存'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：放大的砖 ----------------
BR_X, BR_Y = MX + 40, 100
BR_W = 420
HALF_H = 200
ROW_N, COL_N = 16, 8                      # 行 = token 位，列 = kv_head
def half(x, y, w, h, col, fill, label, rows):
    lc.rect(x, y, w, h, fill, col, rx=5, sw=1.6)
    gw, gh = (w - 96) / COL_N, (h - 26) / ROW_N
    for r in range(ROW_N):
        for c in range(COL_N):
            gx, gy = x + 66 + c * gw, y + 13 + r * gh
            lc.rect(gx, gy, gw - 1.5, gh - 1.5, '#ffffff', col, rx=1, sw=0.5)
    lc.text(x + 14, y + h / 2 - 16, label, 12.5, col, 'middle', True, maxw=44, tag='hl' + label)
    lc.text(x + 14, y + h / 2 + 4, '半页', 9, col, 'middle', maxw=44, tag='hs' + label)
    for r in (0, 8, 15):
        lc.text(x + 54, y + 13 + r * gh + gh / 2 + 3, str(r), 7, '#64748b', 'end', maxw=24,
                tag='hr%s%d' % (label, r))
half(BR_X, BR_Y, BR_W, HALF_H, C_K, F_K, 'K', ROW_N)
half(BR_X, BR_Y + HALF_H + 8, BR_W, HALF_H, C_V, F_V, 'V', ROW_N)
BR_BOT = BR_Y + 2 * HALF_H + 8
# 行/列轴注
lc.text(BR_X + 66 + 4 * ((BR_W - 96) / COL_N), BR_Y - 10, '列 = kv_head（0..7）', 8, '#64748b',
        'middle', maxw=150, tag='ax:c')
lc.text(BR_X - 4, BR_Y + HALF_H - 8, '行 = token 位（0..15）', 8, '#64748b', 'end', maxw=120,
        tag='ax:r')
# 一格放大：head_dim × dtype（放砖右侧、刻度阶梯左侧的空档）
gw, gh = (BR_W - 96) / COL_N, (HALF_H - 26) / ROW_N
ZC_X, ZC_Y, ZC_W = BR_X + BR_W + 36, BR_Y + 70, 136
lc.parrow([(BR_X + BR_W - 4, BR_Y + 13 + 3 * gh + gh / 2), (ZC_X - 4, ZC_Y + 18)],
          C_K, 1.3, None, dash=True)
lc.rect(ZC_X, ZC_Y, ZC_W, 74, '#ffffff', C_K, rx=6, sw=1.3)
lc.text(ZC_X + ZC_W / 2, ZC_Y + 16, '一格放大', 8.5, C_K, 'middle', True, maxw=ZC_W - 10, tag='zc:t')
lc.text(ZC_X + ZC_W / 2, ZC_Y + 33, 'head_dim = 128 维', 8, '#334155', 'middle', maxw=ZC_W - 10,
        tag='zc:1')
lc.text(ZC_X + ZC_W / 2, ZC_Y + 48, '× 2 B（fp16）', 8, '#334155', 'middle', maxw=ZC_W - 10,
        tag='zc:2')
lc.text(ZC_X + ZC_W / 2, ZC_Y + 64, '= 256 B / 格', 8, '#334155', 'middle', maxw=ZC_W - 10,
        tag='zc:3')
# 砖底标注
lc.text(BR_X + BR_W / 2, BR_BOT + 20, '一块砖 = 2 × (16 × 8 × 128 × 2 B) = 65536 B（无量化 padding 时 real = page_size_bytes）',
        9.5, '#155e75', 'middle', True, maxw=620, tag='br:n')

# ---------------- 右上：刻度阶梯（Llama-2-7B 计算例）----------------
SC_X, SC_W = 700, BXR - 700
lc.rect(SC_X, 100, SC_W, 236, '#ffffff', lc.C_KV_S, rx=8, sw=1.5)
lc.text(SC_X + 16, 124, '放到 Llama-2-7B FP16（计算例）', 11, lc.C_KV_S, 'start', True,
        maxw=SC_W - 32, tag='sc:t')
LADDER = [
    ('每 token 每层', '2 × 32(kv_head) × 128 × 2 B = 16384 B'),
    ('每 token 全模型', '16384 × 32 层 = 524288 B = 0.5 MB'),
    ('4096-token 序列', '524288 × 4096 = 2 GiB'),
]
for i, (a, b) in enumerate(LADDER):
    yy = 152 + i * 44
    lc.rect(SC_X + 16, yy - 13, 150, 26, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.0)
    lc.text(SC_X + 91, yy + 3, a, 9, lc.C_KV_S, 'middle', True, maxw=140, tag='ld%d' % i)
    lc.text(SC_X + 182, yy + 3, b, 9.5, '#334155', 'start', maxw=SC_W - 200, tag='lv%d' % i)
    if i < 2:
        lc.seg(SC_X + 91, yy + 15, SC_X + 91, yy + 29, lc.C_KV_S, 1.4, 'std')
lc.text(SC_X + 16, 300, '「权重之外的全部显存都是 KV 的粮仓」——显存为什么是主角，这条公式就是答案',
        8.5, lc.C_MUTE, 'start', maxw=SC_W - 32, tag='sc:n')
lc.text(SC_X + 16, 320, '（Llama-2-7B 32 kv_head × 32 层口径的算术例，非源码断言）', 8, lc.C_FAINT,
        'start', maxw=SC_W - 32, tag='sc:f')

# ---------------- 右中：worker 换算 ----------------
CV_Y = 356
lc.rect(SC_X, CV_Y, SC_W, 96, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(SC_X + 16, CV_Y + 22, 'worker 侧换算（每层一张张量）', 11, lc.C_GPU_S, 'start', True,
        maxw=SC_W - 32, tag='cv:t')
lc.text(SC_X + 16, CV_Y + 46, 'num_blocks = numel // page_size_bytes = 655360 B ÷ 65536 B = 10 块', 9.5,
        '#334155', 'start', maxw=SC_W - 32, tag='cv:1')
lc.text(SC_X + 16, CV_Y + 66, '视图 reshape：[10, 2, 16, 8, 128]（num_blocks, K/V, token 位, kv_head, head_dim）', 9.5,
        '#334155', 'start', maxw=SC_W - 32, tag='cv:2')
lc.text(SC_X + 16, CV_Y + 84, '整除由构造保证（assert）；worker 不做独立决策，只复原 config 里的数', 8.5,
        lc.C_MUTE, 'start', maxw=SC_W - 32, tag='cv:3')

# ---------------- 右下：砖墙 + 调度器账本（单一事实源）----------------
WL_Y = 560
lc.rect(MX, WL_Y, 640, 236, '#ffffff', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + 16, WL_Y + 22, '每层一张张量 = 10 块砖叠成的砖墙', 11, lc.C_GPU_S, 'start', True,
        maxw=560, tag='wl:t')
BW_, BH_, BG_ = 150, 26, 8
for r in range(4):
    for c in range(3):
        bx = MX + 18 + c * (BW_ + BG_)
        by = WL_Y + 40 + r * (BH_ + BG_)
        idx = r * 3 + c
        if idx >= 10:
            break
        lc.rect(bx, by, BW_, BH_, '#ffffff', '#94a3b8', rx=3, sw=0.9)
        lc.rect(bx, by, BW_, BH_ / 2 - 1, F_K, C_K, rx=3, sw=0.7)
        lc.rect(bx, by + BH_ / 2 + 1, BW_, BH_ / 2 - 1, F_V, C_V, rx=3, sw=0.7)
        lc.text(bx + BW_ / 2, by + BH_ / 2 + 3, str(idx), 7.5, '#475569', 'middle', maxw=30,
                tag='wb%d' % idx)
lc.text(MX + 18, WL_Y + 216, '每块上半 K / 下半 V · 共 10 块（0..9）', 8.5, lc.C_MUTE, 'start',
        maxw=420, tag='wl:n')
# 调度器小账本
LB_X, LB_Y = MX + 500, WL_Y + 60
lc.rect(LB_X, LB_Y, 130, 116, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.4)
lc.text(LB_X + 65, LB_Y + 20, '调度器账本', 9.5, lc.C_KV_S, 'middle', True, maxw=120, tag='lb:t')
lc.text(LB_X + 65, LB_Y + 44, 'num_blocks = 10', 10, lc.C_KV_S, 'middle', True, maxw=120, tag='lb:v')
lc.text(LB_X + 65, LB_Y + 66, '「同一个数」', 8.5, '#334155', 'middle', maxw=120, tag='lb:n1')
lc.text(LB_X + 65, LB_Y + 84, '单一事实源 = 启动期', 8, lc.C_MUTE, 'middle', maxw=120, tag='lb:n2')
lc.text(LB_X + 65, LB_Y + 100, 'KVCacheConfig 一次下发', 8, lc.C_MUTE, 'middle', maxw=120, tag='lb:n3')
lc.parrow([(LB_X - 3, LB_Y + 58), (551, LB_Y + 58), (551, WL_Y + 87), (547, WL_Y + 87)],
          lc.C_KV_S, 1.4, 'std', dash=True)
# 右下角 why 注
WY_X = 730
lc.rect(WY_X, WL_Y + 20, BXR - WY_X, 196, '#ffffff', '#94a3b8', rx=8, sw=1.2, dash=True)
lc.text(WY_X + 16, WL_Y + 42, '边界与去向', 10, lc.C_TXT, 'start', True, maxw=BXR - WY_X - 32,
        tag='wy:t')
WNOTES = [
    '· 块 0 是 null 块（占位语义 → ch14），首租从 1 号起',
    '· 真实 GPU 布局由注意力 backend 仲裁（get_kv_cache_shape → ch21）——页字节数与 num_blocks 不变',
    '· packed 跨层别名布局（多组共享一块物理分配）→ ch14',
    '· 池多大、显存怎么 profile 出来 → ch14 显存账本',
]
for i, nt in enumerate(WNOTES):
    lc.text(WY_X + 16, WL_Y + 66 + i * 34, nt, 8.5, '#64748b', 'start', maxw=BXR - WY_X - 32,
            tag='wn%d' % i)

# ---------------- 图例 + 页脚 ----------------
LEG_Y = WL_Y + 262
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, F_K, C_K, rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, 'K 半页（蓝）', 8.5, lc.C_TXT, 'start', maxw=110, tag='lgk')
lx += 26 + lc.tw('K 半页（蓝）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 13, F_V, C_V, rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, 'V 半页（橙）', 8.5, lc.C_TXT, 'start', maxw=110, tag='lgv')
lx += 26 + lc.tw('V 半页（橙）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', '#94a3b8', rx=3, sw=0.9)
lc.text(lx + 26, LEG_Y + 1, '砖 = 一个块（16 token 位 × 8 kv_head）', 8.5, lc.C_TXT, 'start',
        maxw=290, tag='lgb')
lx += 26 + lc.tw('砖 = 一个块（16 token 位 × 8 kv_head）', 8.5) + 18
lc.parrow([(lx + 2, LEG_Y - 3), (lx + 26, LEG_Y - 3)], lc.C_KV_S, 1.3, 'std', dash=True)
lc.text(lx + 32, LEG_Y + 1, '同一个 num_blocks（单一事实源）', 8.5, lc.C_TXT, 'start', maxw=230,
        tag='lgl')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/kv_cache_interface.py:L220-L226（real_page_size_bytes 公式）· '
        'vllm/v1/worker/gpu_model_runner.py:L7400-L7413（num_blocks = numel // page_size_bytes）· '
        'L7433-L7439（[num_blocks, 2, block_size, kv_heads, head_dim] 布局切面）', 8.2, lc.C_FAINT,
        'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, 'vllm/config/cache.py:L47（DEFAULT_BLOCK_SIZE = 16）· 页公式与 worker 换算数字取自配套精简版 host 实跑 · '
        'Llama-2-7B 刻度为计算例 · 行号基线 vLLM v0.27.1', 8.2, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 66
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-page-brick.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
