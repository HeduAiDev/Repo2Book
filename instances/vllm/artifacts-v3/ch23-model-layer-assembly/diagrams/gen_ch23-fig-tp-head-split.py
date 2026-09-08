#!/usr/bin/env python3
"""ch23 机制图 2 · TP 切头分片（figure_spec ch23-fig-tp-head-split，模板 tiling）

放大自本章 L2 章图拍片 ③ 拼装积木·④ 权重流入（站 4-5）× south 权重装载三型格 ·
L0：GPU 执行臂 × 模型层框。

claim: 8q/2kv/tp=2 时全局 qkv 权重 [64|16|16] 行被切成两半——q 段按 tp_rank 切、
k/v 段按 tp_rank // replicas 切（replicas=1 时同样取下半），每 rank 拼成
[32|8|8]=48 行的融合权重；replicate 分支（8q/1kv → replicas=2）同一段 k 权重复制进两卡。

数字全部取自 figure_spec.numbers（8 头 / 2 KV 头 / 全局 64+16+16 行 / 每 rank 32+8+8=48 行 /
replicas=2 / 实尺 768 行，均出自本章精简版实跑 trace）。坐标由常量/循环计算；文本全 esc()。
行号基线 vLLM v0.27.1（linear.py）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1560, 812
MX = 56
BXR = 1504
R0_F, R0_S = '#dcfcc7', '#15803d'      # rank0 段：绿系（模型层角色色家族）
R1_F, R1_S = '#fef3c7', '#b45309'      # rank1 段：琥珀（中性分类色，见 ch20 tiling 先例）

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'TP 切头：q 段按 tp_rank 切、k/v 段按 tp_rank // replicas 切——每 rank 拼成 [32|8|8] = 48 行融合权重',
        15.5, lc.C_TXT, 'start', True, maxw=1130, tag='title')
lc.text(MX, 58, 'GQA 8q/2kv、head_dim=8、hidden=64、tp=2 的 partition 案例（构造期切头数学 = 纯索引算术，rank=1 视角与 rank=0 对称）',
        10, lc.C_MUTE, 'start', maxw=980, tag='subtitle')
_ch = '放大自 L2 拍片 ③④ 拼装积木·权重流入（站 4-5）× south 权重装载三型 · L0：GPU 执行臂 × 模型层框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# 图例（rank 段色 + 刀口虚线）
LY = 104
lx = MX
for fill, stroke, lab in ((R0_F, R0_S, 'rank0 拿走的段'), (R1_F, R1_S, 'rank1 拿走的段')):
    lc.rect(lx, LY - 9, 18, 12, fill, stroke, rx=3, sw=1.4)
    lc.text(lx + 24, LY + 1, lab, 9, '#334155', 'start', tag='leg:' + lab[:6])
    lx += 24 + lc.tw(lab, 9) + 26
lc.seg(lx + 2, LY - 3, lx + 30, LY - 3, lc.C_MUTE, 1.4, dash=True)
lc.text(lx + 36, LY + 1, '刀口（切割线，段内中线）', 9, '#334155', 'start', tag='leg:cut')

# ---------------- 带 ① 全局三段（上行） ----------------
S = 9                                  # px / 行
X0 = 140
GAP = 18
GY0, GH = 150, 52                      # 全局条 y / 高
SEG = [('q', 64, 'q 段 64 行 = 8 头 × 8'),
       ('k', 16, 'k 16 行'), ('v', 16, 'v 16 行')]
gx = [X0]
for _, rows, _lab in SEG:
    gx.append(gx[-1] + rows * S + GAP)
lc.text(X0, 132, '① 全局视角：qkv 融合权重三段 [64|16|16] 行 × 64 列（全部头都在——output_sizes_total）',
        10.5, lc.C_TXT, 'start', True, maxw=880, tag='b1')
for i, (name, rows, lab) in enumerate(SEG):
    x, w = gx[i], rows * S
    half = w / 2
    lc.rect(x, GY0, half, GH, R0_F, R0_S, rx=0, sw=1.4)
    lc.rect(x + half, GY0, half, GH, R1_F, R1_S, rx=0, sw=1.4)
    lc.rect(x, GY0, w, GH, 'none', lc.C_MUTE, rx=0, sw=1.2)          # 段外框
    lc.seg(x + half, GY0 - 8, x + half, GY0 + GH + 8, lc.C_MUTE, 1.4, dash=True)  # 刀口
    if name == 'q':
        lc.text(x + half / 2, GY0 + 22, f'{name}[0:{rows // 2})', 9.5, R0_S, 'middle', True, tag='gq0')
        lc.text(x + half / 2, GY0 + 40, '→ rank0', 8, R0_S, 'middle', tag='gq0b')
        lc.text(x + 3 * half / 2, GY0 + 22, f'{name}[{rows // 2}:{rows})', 9.5, R1_S, 'middle', True, tag='gq1')
        lc.text(x + 3 * half / 2, GY0 + 40, '→ rank1', 8, R1_S, 'middle', tag='gq1b')
    else:
        lc.text(x + half / 2, GY0 + 24, f'{name}[0:{rows // 2})', 8.2, R0_S, 'middle', True, tag='g' + name + '0')
        lc.text(x + 3 * half / 2, GY0 + 24, f'{name}[{rows // 2}:{rows})', 8.2, R1_S, 'middle', True, tag='g' + name + '1')
    lc.text(x + w / 2, GY0 - 6, lab, 8.2, lc.C_MUTE, 'middle', tag='glab' + name)
# 段间连接（同一张融合权重的三段）
for i in range(len(SEG) - 1):
    x_join = gx[i] + SEG[i][1] * S
    lc.seg(x_join, GY0 + GH / 2, x_join + GAP, GY0 + GH / 2, lc.C_MUTE, 1.2, dash=True)

# ---------------- 带 ② 各取一半（中行）----------------
SY0, SH = 300, 46
PICE_GAP = 14
# rank0 条：q[0:32) + k[0:8) + v[0:8)
r0x = 240
r0_pieces = [('q[0:32)', 32, R0_F, R0_S), ('k[0:8)', 8, R0_F, R0_S), ('v[0:8)', 8, R0_F, R0_S)]
r0_c = []
px = r0x
for lab, rows, f, s_ in r0_pieces:
    w = rows * S
    lc.rect(px, SY0, w, SH, f, s_, rx=4, sw=1.5)
    lc.text(px + w / 2, SY0 + SH / 2 + 3.5, lab, 9 if rows > 8 else 7.8, R0_S, 'middle', True,
            maxw=w - 6, tag='p0' + lab[:3])
    r0_c.append(px + w / 2)
    px += w + PICE_GAP
lc.text(r0x - 10, SY0 + SH / 2 + 3.5, 'rank0', 10, R0_S, 'end', True, tag='r0lab')
# rank1 条：q[32:64) + k[8:16) + v[8:16)
r1x = 720
r1_pieces = [('q[32:64)', 32, R1_F, R1_S), ('k[8:16)', 8, R1_F, R1_S), ('v[8:16)', 8, R1_F, R1_S)]
r1_c = []
px = r1x
for lab, rows, f, s_ in r1_pieces:
    w = rows * S
    lc.rect(px, SY0, w, SH, f, s_, rx=4, sw=1.5)
    lc.text(px + w / 2, SY0 + SH / 2 + 3.5, lab, 9 if rows > 8 else 7.8, R1_S, 'middle', True,
            maxw=w - 6, tag='p1' + lab[:3])
    r1_c.append(px + w / 2)
    px += w + PICE_GAP
lc.text(r1x - 10, SY0 + SH / 2 + 3.5, 'rank1', 10, R1_S, 'end', True, tag='r1lab')

# 带 ② 左侧说明（左缘空白带，避开箭头走廊）
lc.text(MX + 6, SY0 - 2, '② 各取一半', 9.5, lc.C_TXT, 'start', True, tag='b2t')
lc.text(MX + 6, SY0 + 14, 'q 的刀口与 k/v', 8.3, lc.C_MUTE, 'start', maxw=180, tag='b2a')
lc.text(MX + 6, SY0 + 27, '的刀口独立', 8.3, lc.C_MUTE, 'start', maxw=180, tag='b2b')
lc.text(MX + 6, SY0 + 42, '（公式不同）', 8.3, lc.C_MUTE, 'start', maxw=180, tag='b2c')

# 箭头：全局各半 → 对应段（绿 3 支 + 琥珀 3 支）
g_half_c = []
for i, (name, rows, _lab) in enumerate(SEG):
    w = rows * S
    g_half_c.append((gx[i] + w / 4, gx[i] + 3 * w / 4))
for j in range(3):
    lc.seg(g_half_c[j][0], GY0 + GH, r0_c[j], SY0, R0_S, 1.6, 'std')
for j in range(3):
    lc.seg(g_half_c[j][1], GY0 + GH, r1_c[j], SY0, R1_S, 1.6, 'std')

# ---------------- 带 ③ 拼装融合权重（下行） ----------------
FY0, FH = 420, 74
f_off = [0, 32, 40, 48]
for rx0, c_list, stroke, fill, rlab in (
        (240, r0_c, R0_S, R0_F, 'rank0'), (720, r1_c, R1_S, R1_F, 'rank1')):
    fw = 48 * S
    seg_bounds = [rx0 + off * S for off in f_off]
    for k in range(3):
        x0_, x1_ = seg_bounds[k], seg_bounds[k + 1]
        lc.rect(x0_, FY0, x1_ - x0_, FH, fill, stroke, rx=0, sw=1.5)
        lc.text((x0_ + x1_) / 2, FY0 + FH / 2 + 3.5,
                ['q', 'k', 'v'][k], 9.5, stroke, 'middle', True, tag='fw' + rlab + str(k))
    lc.rect(rx0, FY0, fw, FH, 'none', lc.C_MUTE, rx=0, sw=1.2)
    # strip → fused 箭头
    f_seg_c = [(seg_bounds[k] + seg_bounds[k + 1]) / 2 for k in range(3)]
    for j in range(3):
        lc.seg(c_list[j], SY0 + SH, f_seg_c[j], FY0, stroke, 1.6, 'std')
    # offset 刻度
    for k, off in enumerate(f_off):
        lc.text(seg_bounds[k], FY0 + FH + 14, str(off), 7.8, lc.C_MUTE,
                'middle' if 0 < k < 3 else 'start', tag='off' + rlab + str(off))
lc.text(240, FY0 + FH + 32, 'rank0 融合权重 qkv_proj.weight：[32|8|8] = 48 行 × 64 列——三段 offset 线性递推（0/32/40），无重不漏铺满',
        8.5, R0_S, 'start', maxw=640, tag='fwr0')
lc.text(720, FY0 + FH + 32, 'rank1 同款 48 行；forward 一次 GEMM 吃完，', 8.5, R1_S, 'start',
        maxw=560, tag='fwr1a')
lc.text(720, FY0 + FH + 47, '再按同一刀口 split([32, 8, 8]) 拆回三份——装填与 split 互为镜像',
        8.5, R1_S, 'start', maxw=560, tag='fwr1b')
lc.text(MX + 6, FY0 + 18, '③ 拼装', 9.5, lc.C_TXT, 'start', True, tag='b3t')
lc.text(MX + 6, FY0 + 34, 'offset 递推', 8.3, lc.C_MUTE, 'start', maxw=180, tag='b3a')
lc.text(MX + 6, FY0 + 47, '铺满本 rank', 8.3, lc.C_MUTE, 'start', maxw=180, tag='b3b')

# ---------------- 右栏：刀口公式 ----------------
PX, PW = 1200, 304
P0Y, P_H = 150, 352
lc.rect(PX, P0Y, PW, P_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(PX + 14, P0Y + 22, '刀口公式（构造期算术）', 10.5, lc.C_GPU_S, 'start', True,
        maxw=PW - 28, tag='p:t')
_plines = [
    ('q 段：shard_rank = tp_rank', True),
    ('start = tp_rank × q_size', False),
    ('rank1：1 × 32 = 32 → q[32:64)', False),
    ('', False),
    ('k/v 段：shard_rank =', True),
    ('tp_rank // num_kv_head_replicas', False),
    ('replicas = tp // total_kv = 2 // 2 = 1', False),
    ('（2 KV 头 ≥ 2 卡 → partition 分支）', False),
    ('rank1：(1 // 1) × 8 = 8 → k[8:16)、v[8:16)', False),
    ('', False),
    ('两把刀口独立：q 与 k/v 的', True),
    ('shard_rank 公式不同，可各切各的', False),
    ('', False),
    ('整除双保险：divide() 不整除即', False),
    ('ValueError + llama.py:L142-L152 断言', False),
    ('构造期快速失败，不留到装载期', False),
]
py = P0Y + 44
for ln, bold in _plines:
    if ln:
        lc.text(PX + 14, py, ln, 8.3 if not bold else 8.8, lc.C_TXT if bold else '#334155',
                'start', bold, maxw=PW - 26, tag='p:l' + ln[:8])
    py += 18.5
lc.text(PX + 14, P0Y + P_H - 10, 'vllm/model_executor/layers/linear.py', 8, lc.C_FAINT, 'start',
        maxw=PW - 26, tag='p:file')

# ---------------- 底部左：replicate 分支小图 ----------------
RX, RW = 240, 500
RY, R_H = 552, 168
lc.rect(RX, RY, RW, R_H, '#ffffff', R1_S, rx=8, sw=1.4)
lc.text(RX + 14, RY + 20, 'replicate 分支：KV 头数 < 卡数（8q / 1kv、tp=2）', 10, R1_S, 'start',
        True, maxw=RW - 28, tag='rp:t')
# 全局 k 一小条（1 头 × 8 行 = 72px）
kg_x, kg_y, kg_w = RX + 24, RY + 38, 72
lc.rect(kg_x, kg_y, kg_w, 26, '#e2e8f0', lc.C_MUTE, rx=3, sw=1.2)
lc.text(kg_x + kg_w / 2, kg_y + 17, '全局 k = 8 行', 7.5, lc.C_TXT, 'middle', True, maxw=kg_w - 4,
        tag='rp:gk')
# 两个 rank 的 k 段
rk_y = RY + 108
lc.rect(kg_x - 18, rk_y, kg_w, 26, R0_F, R0_S, rx=3, sw=1.4)
lc.text(kg_x - 18 + kg_w / 2, rk_y + 17, 'rank0 k 段', 7.5, R0_S, 'middle', True, maxw=kg_w - 4,
        tag='rp:k0')
lc.rect(kg_x + 66, rk_y, kg_w, 26, R1_F, R1_S, rx=3, sw=1.4)
lc.text(kg_x + 66 + kg_w / 2, rk_y + 17, 'rank1 k 段', 7.5, R1_S, 'middle', True, maxw=kg_w - 4,
        tag='rp:k1')
lc.seg(kg_x + kg_w / 2 - 14, kg_y + 26, kg_x - 18 + kg_w / 2, rk_y, R0_S, 1.5, 'std')
lc.seg(kg_x + kg_w / 2 + 14, kg_y + 26, kg_x + 66 + kg_w / 2, rk_y, R1_S, 1.5, 'std')
lc.text(kg_x + 108, RY + 46, 'replicas = tp // total_kv = 2 // 1 = 2', 8.3, lc.C_TXT, 'start',
        True, maxw=RW - 130, tag='rp:f')
lc.text(kg_x + 108, RY + 64, 'k/v 段复制进 2 个 rank：两个 rank 的', 8.2, '#334155', 'start',
        maxw=RW - 130, tag='rp:l1')
lc.text(kg_x + 108, RY + 79, 'k 段都指向同一段全局权重（无刀口：整段复制）', 8.2, '#334155', 'start',
        maxw=RW - 130, tag='rp:l2')
lc.text(kg_x + 108, RY + 97, '——GQA 装载语义：每卡的 q 头组需要', 8.2, '#334155', 'start',
        maxw=RW - 130, tag='rp:l3')
lc.text(kg_x + 108, RY + 112, '一份完整 KV 头权重；复制的显存代价', 8.2, '#334155', 'start',
        maxw=RW - 130, tag='rp:l4')
lc.text(kg_x + 108, RY + 130, '直接可见（两卡各背同一份 k/v 段）', 8.2, '#334155', 'start',
        maxw=RW - 130, tag='rp:l5')
lc.text(RX + 14, RY + R_H - 10, '分支判定在构造期：total_num_kv_heads 与 tp_size 比大小（llama.py:L145-L152 断言）', 7.5,
        lc.C_FAINT, 'start', maxw=RW - 28, tag='rp:ft')

# ---------------- 底部右：实尺同形 ----------------
EX, EW = 770, 734
EY, E_H = 552, 168
lc.rect(EX, EY, EW, E_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(EX + 14, EY + 20, '实尺同形（head_dim=128、hidden=4096、8q/2kv、tp=2）', 10, lc.C_TXT,
        'start', True, maxw=EW - 28, tag='es:t')
lc.text(EX + 14, EY + 42, '同一副图形按 head_dim 等比放大：每 rank 融合权重 [512|128|128] = 768 行 × 4096 列',
        8.5, '#334155', 'start', maxw=EW - 28, tag='es:l1')
lc.text(EX + 14, EY + 60, 'fp16 恰 6291456 B ≈ 6 MB/rank——tp=2 下每卡只存一半注意力投影权重',
        8.5, '#334155', 'start', maxw=EW - 28, tag='es:l2')
lc.text(EX + 14, EY + 78, '若换 1-KV-头模型（replicas=2）：两张卡各再背一份相同的 k/v 段——replicate 的显存代价随实尺放大',
        8.5, '#334155', 'start', maxw=EW - 28, tag='es:l3')
lc.text(EX + 14, EY + 100, '切头整除断言：total_num_heads=4、tp=3 → 构造期即 AssertionError（快速失败，不静默错位）',
        8.3, lc.C_MUTE, 'start', maxw=EW - 28, tag='es:l4')
lc.text(EX + 14, EY + R_H - 10, 'vllm/model_executor/layers/linear.py · vllm/model_executor/models/llama.py:L142',
        8, lc.C_FAINT, 'start', maxw=EW - 28, tag='es:ft')

# ---------------- 页脚 ----------------
lc.text(MX, 762, '读图：上行全局三段各按中线切开 → 中行 rank0 取三段上半、rank1 取下半（q 与 k/v 刀口公式不同、各自独立）→ 下行每 rank 拼成 [32|8|8]=48 行融合权重；左下 replicate = KV 头不足时整段复制。',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 780, '行号基线 vLLM v0.27.1 · 数值取自本章精简版实跑核验（host CPU：切头/分片装载是纯索引算术，tp=2 即真实代码路径）',
        8, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch23-fig-tp-head-split.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
