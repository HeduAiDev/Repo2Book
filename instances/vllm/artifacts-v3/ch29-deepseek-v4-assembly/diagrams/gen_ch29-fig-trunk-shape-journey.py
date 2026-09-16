#!/usr/bin/env python3
"""ch28 机制图 · 全干形状主线(ch28-fig-trunk-shape-journey, 模板 tensor-flow)

= L0『GPU 执行臂·模型层 forward + 编译』块的整条主干(L2 章图 ⑥⑦ 的机制版主图——
L2 给装配全景, 本图给形状旅程单线索; 与单层两半图互为表里: 那张讲一层的门控,
本图讲全干的形状旅程)。

claim: DeepseekV4Model.forward 的形状主线——embed 出 2D (T,H)(不再显式 repeat,
首层核内广播展开) → 逐层 4 元组穿针(hidden_states 恒 2D、多流活在 residual) →
EAGLE3 命中层中途 mhc_post 重建 aux 流(mean(dim=1)) → 层尾塌回 (T,hc_mult,H) →
先 copy_ 进 _mtp_hidden_buffer 再 hc_head 压回 (T,H) → norm; PP 非末 rank 以
多流形状直接交 IntermediateTensors; mega_moe 下 input_ids 先 int64 化。

锚点 = nvidia/model.py:L908-L943/L1111-L1129/L1131-L1215 · tilelang.py:L720-L748。
坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 700
MX = 56
BXR = 1608
C_STREAM_A, C_STREAM_B = '#16a34a', '#86efac'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '全干形状主线：2D 进、2D 出，多流活在层间', 16, lc.C_TXT, 'start', True,
        maxw=900, tag='title')
lc.text(MX, 58, 'embed 的 2D 在首层被核内广播展开成 hc_mult 条流——子层永远只见 2D，多流的「身份」活在 residual 里穿完全程，层尾塌回 3D 先喂 MTP 再压回单流',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
_ch = '放大自 L0『模型层 forward + 编译』块整条主干 · L2 站 10 的机制版主图'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')


def box(x, y, w, h, title, lines, file='', stroke=lc.C_GPU_S, fill='#ffffff', dash=False,
        lfs=8.5, tc=lc.C_TXT):
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=dash)
    lc.text(x + 11, y + 19, title, 10, tc, 'start', True, maxw=w - 22, tag='t:' + title)
    yy = y + 37
    for s in lines:
        lc.text(x + 11, yy, s, lfs, '#334155', 'start', maxw=w - 20, tag='l:' + s[:12])
        yy += 15.5
    if file:
        lc.text(x + 11, y + h - 9, file, 7.5, lc.C_FAINT, 'start', maxw=w - 18, tag='f:' + title)


# ---------------- 主轴站位 ----------------
AX_Y0, AX_Y1 = 160, 250
lc.rect(MX, AX_Y0, 120, 80, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 60, AX_Y0 + 26, 'input_ids', 10, lc.C_TXT, 'middle', True, tag='in:t')
lc.text(MX + 60, AX_Y0 + 46, '(T,)', 9, lc.C_MUTE, 'middle', tag='in:s')
lc.text(MX, 262, 'mega：.to(int64)（hash 查表要）', 8, '#334155', 'start', maxw=140, tag='in:n1')
lc.text(MX, 277, 'SP：sp_shard 入口', 8, '#334155', 'start', maxw=140, tag='in:n2')

box(196, AX_Y0, 120, 80, 'embed_tokens', ['(T, H) —— 2D', '不再显式 repeat'], 'model.py:L1138-L1142')
lc.seg(176, 205, 196, 205, lc.C_GPU_S, 3.2, 'std')

# 首层核
box(336, AX_Y0, 130, 102, '首层核 mhc_pre', ['（broadcast）', 'x.dim()==2 特判', '核内展开 hc_mult 流',
    '用预折 fn_broadcast'], 'model.py:L910-L927', fill=lc.C_GPU_F)
lc.seg(316, 205, 336, 205, lc.C_GPU_S, 3.2, 'std')

# ---- 4 元组线束: 首层核右缘 → 层尾核左缘 ----
BX0, BX1 = 466, 1030
# 容器(画在最底层): DecoderLayer ×N
lc.rect(560, 130, 450, 170, '#f8fafc', lc.C_GPU_S, rx=10, sw=1.6, dash=True)
lc.text(785, 152, 'DecoderLayer ×num_hidden_layers（每层一对半层核）', 10, '#14532d', 'middle',
        True, maxw=430, tag='ct:t')
# 线束四轨
lc.seg(BX0, 205, BX1, 205, lc.C_GPU_S, 3.2, 'std')            # x 2D 粗线
for k in range(4):                                             # residual 多流带
    lc.rect(BX0, 211 + k * 3.5, BX1 - BX0, 3.5,
            C_STREAM_A if k % 2 == 0 else C_STREAM_B, 'none', rx=0, sw=0)
lc.seg(BX0, 233, BX1, 233, '#475569', 1.2)                     # post_mix
lc.seg(BX0, 241, BX1, 241, '#94a3b8', 1.2)                     # res_mix
# 线束标签(首段空隙 466..560)
lc.text(513, 196, 'x (T,H) 恒 2D', 7.5, '#14532d', 'middle', True, maxw=92, tag='bd:x')
lc.text(513, 262, 'residual 多流带', 7.5, '#166534', 'middle', maxw=92, tag='bd:r')
lc.text(513, 274, 'post_mix · res_mix', 7.5, lc.C_MUTE, 'middle', maxw=92, tag='bd:m')
# 两个半层核芯片(画在线束之上=线从核中穿过)
box(600, 176, 130, 82, 'attn 半：融合核', ['post+pre+RMSNorm', '一核完成'], '', fill=lc.C_GPU_F)
box(840, 176, 130, 82, 'ffn 半：融合核', ['同款 mhc_fused_', 'post_pre_tilelang'], '', fill=lc.C_GPU_F)
lc.text(785, 196, '子层接口', 7.5, lc.C_MUTE, 'middle', tag='gap:lab')
lc.text(785, 292, '‖ 下一层 ×(num_hidden_layers−1)：同一对半层核重复（4 元组进 4 元组出）',
        8.5, lc.C_MUTE, 'middle', maxw=440, tag='ct:loop')

# PP rank 边界（竖虚线, 穿线束区）
PPX = 545
lc.seg(PPX, 110, PPX, 430, '#7c3aed', 1.6, dash=True)
lc.text(PPX, 102, 'PP rank 边界（非末 rank）', 8.5, '#7c3aed', 'middle', True, maxw=200, tag='pp:t')
lc.text(PPX, 442, 'IntermediateTensors: (T, hc_mult, hidden)——多流 3D 载荷交接', 8.5, '#7c3aed',
        'middle', maxw=420, tag='pp:b')

# aux 重建分叉（EAGLE3）
lc.parrow([(990, 207), (990, 132)], '#0369a1', 1.6, 'std', dash=True)
box(865, 88, 250, 46, 'aux 流重建（EAGLE3 draft 用）', [
    '命中层 idx+1 ∈ aux_hidden_state_layers：', 'mhc_post + mean(dim=1)'],
    '', stroke='#0369a1', dash=True, lfs=8)

# 层尾核
box(1030, AX_Y0, 130, 102, '层尾核 mhc_post', ['塌回多流残差', '(T, hc_mult, H)',
    '（末层若是 aux 层则复用，', '避免重复 mhc_post）'], 'model.py:L1185-L1192', fill=lc.C_GPU_F)
# 3D 载荷宽带
for k in range(4):
    lc.rect(1160, 196 + k * 6, 20, 6, C_STREAM_A if k % 2 == 0 else C_STREAM_B, 'none', rx=0, sw=0)

# hc_head + norm
box(1180, AX_Y0, 130, 90, 'hc_head 融合核', ['(T,hc_mult,H)→(T,H)', 'bf16 · 空批早退'],
    'tilelang.py:L720-L748', fill=lc.C_GPU_F)
lc.rect(1340, 176, 220, 58, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(1450, 198, 'norm → hidden (T,H)', 9.5, lc.C_TXT, 'middle', True, maxw=200, tag='out:t')
lc.text(1450, 218, '→ compute_logits（采样位，ch29）', 8.5, lc.C_MUTE, 'middle', maxw=200,
        tag='out:s')
lc.seg(1310, 205, 1340, 205, lc.C_GPU_S, 3.2, 'std')

# MTP 旁路
lc.rect(1180, 316, 380, 96, '#ffffff', lc.C_ENG_S, rx=8, sw=1.4, dash=True)
lc.text(1192, 336, '_mtp_hidden_buffer.copy_(flatten(1))', 9.5, '#9a3412', 'start', True,
        maxw=360, tag='buf:t')
for i, s in enumerate(['shape (max_num_batched_tokens, hc_mult×H)', 'pre-hc_head 残差 → MTP draft 原料',
                       '（先暂存后定型：copy_ 先于 hc_head）']):
    lc.text(1192, 356 + i * 16, s, 8.5, '#334155', 'start', maxw=360, tag='buf:l%d' % i)
lc.parrow([(1170, 228), (1170, 364), (1180, 364)], lc.C_ENG_S, 1.6, 'std', dash=True)
lc.text(1162, 300, 'copy_', 8, lc.C_ENG_S, 'end', True, tag='bypass:lab')

# ---------------- 底部三注 ----------------
NY, NH = 476, 150
lc.rect(MX, NY, 500, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 14, NY + 20, 'PP 中间张量的注释原话（model.py:L1111-L1129）', 10.5, lc.C_TXT,
        'start', True, maxw=460, tag='n1:t')
for i, s in enumerate([
        '· \'V4 expands the token embedding to hc_mult',
        '   streams before the first decoder layer and',
        '   keeps that shape until hc_head() collapses it\'',
        '· 严格说：穿层的 hidden_states 是 2D，多流活在',
        '   residual——PP 载荷才是 3D (T, hc_mult, hidden)',
        '· make_empty_intermediate_tensors 直接按 3D 建零张量']):
    lc.text(MX + 14, NY + 42 + i * 17, s, 8.5, '#334155', 'start', maxw=470, tag='n1:l%d' % i)

lc.rect(590, NY, 480, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(604, NY + 20, '入口的两笔小账（model.py:L1146-L1158）', 10.5, lc.C_TXT, 'start', True,
        maxw=440, tag='n2:t')
for i, s in enumerate([
        '· mega 下 input_ids 先 .to(int64)：hash 路由',
        '   tid2eid 查表要整数索引（hash 层=前',
        '   num_hash_layers 层）',
        '· 序列并行（SP）时 sp_shard 入口把 hidden 与',
        '   input_ids 切片；attn 半层被 sp_all_gather /',
        '   sp_reduce_scatter 包裹（原理归分布式章）']):
    lc.text(604, NY + 42 + i * 17, s, 8.5, '#334155', 'start', maxw=450, tag='n2:l%d' % i)

lc.rect(1110, NY, BXR - 1110, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(1124, NY + 20, '两个中途分叉的账', 10.5, lc.C_TXT, 'start', True, maxw=440, tag='n3:t')
for i, s in enumerate([
        '· aux 重建：idx+1 ∈ aux_hidden_state_layers 时',
        '   mhc_post + mean(dim=1)（model.py:L1163-L1188）',
        '   ——draft 模型（EAGLE3）要单流 aux hidden',
        '· hc_head 融合核：(T,hc_mult,H)→(T,H) bf16，',
        '   空批早退（tilelang.py:L720-L748）',
        '· 与单层两半图互为表里：那张讲一层门控，本图讲全程']):
    lc.text(1124, NY + 42 + i * 17, s, 8.5, '#334155', 'start', maxw=460, tag='n3:l%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 656, '图例：粗绿线 = x (T,H) 恒 2D（子层接口）· 绿深浅相间带 = residual 多流带（hc_mult 条流）/ 3D 多流载荷 · 细灰线 = post_mix / res_mix · 绿框 = hc 核',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 676, '锚点 = vllm/models/deepseek_v4/nvidia/model.py:L908-L943/L1111-L1129/L1131-L1215 · vllm/model_executor/kernels/mhc/tilelang.py:L720-L748',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-trunk-shape-journey.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
