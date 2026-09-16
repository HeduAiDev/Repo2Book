#!/usr/bin/env python3
"""ch29 机制图 · mHC 多流残差的主视图(ch29-fig-mhc-residual-journey, 模板 tensor-flow)

放大自 L0『GPU 执行臂·模型层 forward + 编译』块里的 DecoderLayer 残差通道——
L2 拍片⑦(单层两半)的机制版下钻: ch23 的 LlamaDecoderLayer 在同一位置画的是
RMSNorm→attn→add;本图把那一条道展开成 hc_mult 条流的穿针图。

claim: 一个 token 的 hidden 在 DSV4 主干里是「2D 只在子层接口存在、多流活在层间残差」
的旅程: embed 2D → 首层核内 broadcast 展开 hc_mult 流 → 4 元组逐半层穿针(每半层一个
融合核 post+pre+RMSNorm) → 层尾 mhc_post 塌回 3D, 先分叉给 MTP(pre-hc_head 残差)
再 hc_head 压回 2D。

数值 = 本章实跑(hc_mult=2、H=2、T=1 玩具例, bfloat16 存储后), 与正文数值表同源;
坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 985
MX = 56
BXR = 1608

# ---------------- 标题区 ----------------
lc.text(MX, 34, '多流残差的主视图：hidden 在子层接口是 2D，「身份」活在 residual 的 hc_mult 条流里',
        16, lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '一个 token 的旅程：embed 2D → 首层核内展开 hc_mult 流 → 4 元组逐半层穿针（每半层一个融合核）→ 层尾塌回 3D，先分叉给 MTP 再 hc_head 压回 2D',
        10.5, lc.C_MUTE, 'start', maxw=1160, tag='subtitle')
_ch = '放大自 L0『GPU 执行臂·模型层 forward + 编译』块 · L2 拍片⑦'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主旅程几何常量 ----------------
X_TOP, X_H = 140, 104          # x 轨道框顶/高
K_TOP, K_BOT = 130, 560        # 融合核纵向跨度
T_Y0 = 366                     # 三件套首行顶
ARROW_Y = X_TOP + X_H / 2 - 8  # x 轨道箭头 y(线中心)

EMB_X, EMB_W = 56, 120
K1_X, K1_W = 196, 140
ATTN_X, ATTN_W = 356, 190
K2_X, K2_W = 566, 160
FFN_X, FFN_W = 746, 190
K3_X, K3_W = 956, 130

# 容器: 一个 DecoderLayer 的两半
lc.rect(186, 104, 920, 486, 'none', lc.C_FAINT, rx=10, sw=1.2, dash=True)
lc.text(700, 126, '一个 DecoderLayer 的两半（hc 门控骨架：子层只见 2D，多流活在三件套）',
        10, lc.C_MUTE, 'middle', True, maxw=880, tag='container')


def xflow_box(x, w, title, lines, dashed=False, fill='#ffffff', stroke=lc.C_MUTE):
    h = X_H
    lc.rect(x, X_TOP, w, h, fill, stroke, rx=8, sw=1.4, dash=dashed)
    lc.text(x + 12, X_TOP + 20, title, 10.5, lc.C_TXT, 'start', True, maxw=w - 24, tag='t:' + title)
    for i, s in enumerate(lines):
        lc.text(x + 12, X_TOP + 40 + i * 17, s, 8.5, '#334155', 'start', maxw=w - 22,
                tag='l:' + title + str(i))
    return x, x + w


def kernel_box(x, w, title, sub, lines, file):
    h = K_BOT - K_TOP
    lc.rect(x, K_TOP, w, h, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.8)
    lc.text(x + w / 2, K_TOP + 22, title, 10.5, '#14532d', 'middle', True, maxw=w - 14, tag='k:' + title)
    lc.text(x + w / 2, K_TOP + 38, sub, 8.5, lc.C_GPU_S, 'middle', maxw=w - 12, tag='ks:' + title)
    yy = K_TOP + 62
    for s in lines:
        lc.text(x + w / 2, yy, s, 8.5, '#166534', 'middle', maxw=w - 12, tag='kl:' + s[:10])
        yy += 17
    lc.text(x + w / 2, K_BOT - 12, file, 8, lc.C_FAINT, 'middle', maxw=w - 10, tag='kf:' + title)


def trio_group(x, w, tag, residual_rows, post_row, mix_cells, extra=None):
    """三件套竖排: residual / post_mix / res_mix(2x2 小方阵)。返回各行 y 中心。"""
    rh, ph, mh = 76, 40, 74
    y0 = T_Y0
    lc.rect(x, y0, w, rh, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
    lc.text(x + 10, y0 + 15, 'residual (1,2,2)', 8.5, lc.C_TXT, 'start', True, maxw=w - 20, tag=tag + ':r')
    lc.text(x + 10, y0 + 31, residual_rows[0], 8.5, '#334155', 'start', maxw=w - 18, tag=tag + ':r0')
    lc.text(x + 10, y0 + 46, residual_rows[1], 8.5, '#334155', 'start', maxw=w - 18, tag=tag + ':r1')
    lc.text(x + 10, y0 + 65, extra, 7.5, lc.C_MUTE, 'start', maxw=w - 18, tag=tag + ':rx')
    y1 = y0 + rh + 8
    lc.rect(x, y1, w, ph, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
    lc.text(x + 10, y1 + 15, 'post_mix (1,2,1)', 8.5, lc.C_TXT, 'start', True, maxw=w - 20, tag=tag + ':p')
    lc.text(x + 10, y1 + 31, post_row, 8.5, '#334155', 'start', maxw=w - 18, tag=tag + ':pv')
    y2 = y1 + ph + 8
    lc.rect(x, y2, w, mh, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
    lc.text(x + 10, y2 + 15, 'res_mix (1,2,2)', 8.5, lc.C_TXT, 'start', True, maxw=w - 20, tag=tag + ':m')
    cw_, chh = 52, 19
    gx, gy = x + 12, y2 + 22
    for i in range(2):
        for j in range(2):
            lc.rect(gx + j * (cw_ + 4), gy + i * (chh + 3), cw_, chh, lc.C_GPU_F, lc.C_GPU_S,
                    rx=3, sw=1.0)
            lc.text(gx + j * (cw_ + 4) + cw_ / 2, gy + i * (chh + 3) + 13,
                    mix_cells[i][j], 7.5, '#166534', 'middle', maxw=cw_ - 2, tag=tag + ':m%d%d' % (i, j))
    return (y0 + rh / 2, y1 + ph / 2, y2 + mh / 2)


# ---- x 轨道: embed → 首层核 → 注意力(占位) → 融合核 → FFN(占位) → 层尾核 ----
xflow_box(EMB_X, EMB_W, 'embed', ['x0 (1,2)', '[1.0, 2.0]', '2D（不显式 repeat）'])
kernel_box(K1_X, K1_W, '首层核 mhc_pre', '（broadcast）',
           ['2D 核内按流展开', '+ pre 门控出三件套', 'x.dim()==2 特判', '用预折的', 'fn_broadcast'],
           'model.py:L910-L927')
xflow_box(ATTN_X, ATTN_W, '注意力半层（占位）',
          ['in: layer_input=[1.2422, 2.4844]', 'out: x_attn=[0.6016, -0.3008]',
           '形状恒 (1,2)', '本体归 ch25/26（插座+核内展开）'], dashed=True)
kernel_box(K2_X, K2_W, '融合核（每半层一个）', 'mhc_fused_post_pre',
           ['post+pre+RMSNorm', '融成一核', 'norm 非独立层：', 'weight/eps 传进核'],
           'model.py:L944-L989')
xflow_box(FFN_X, FFN_W, 'FFN / MoE 半层（占位）',
          ['in: layer_input=[2.2812, 2.0]', 'out: x_ffn=[0.4004, 0.8984]',
           '形状恒 (1,2)', 'MoE 装配见本章双后端图'], dashed=True)
kernel_box(K3_X, K3_W, '层尾核 mhc_post', '（模型级，非层内）',
           ['塌回多流残差', 'out (1,2,2)'], 'model.py:L1190-L1192')

AY = X_TOP + X_H / 2 - 8
for a, b in [(EMB_X + EMB_W, K1_X), (K1_X + K1_W, ATTN_X), (ATTN_X + ATTN_W, K2_X),
             (K2_X + K2_W, FFN_X), (FFN_X + FFN_W, K3_X)]:
    lc.seg(a, AY, b, AY, lc.C_MUTE, 1.8, 'std')

# ---- 三件套轨道: 两组站间值 ----
g1_ys = trio_group(356, 190, 'g1',
                   ['流0 [1.0, 2.0]', '流1 [1.0, 2.0]'],
                   '流0→1.3667 流1→1.3345',
                   [['0.5056', '0.4944'], ['0.4944', '0.5056']],
                   extra='（首层广播：两流相同）')
g2_ys = trio_group(746, 190, 'g2',
                   ['流0 [1.8203, 1.5859]', '流1 [1.8047, 1.6016]'],
                   '流0→1.3331 流1→1.2995',
                   [['0.4932', '0.5068'], ['0.5068', '0.4932']],
                   extra='（层间穿针：三件套形状不变）')
# 三件套↔核 的三股平行针线
for ys, x_a, x_b in [(g1_ys, K1_X + K1_W, 356), (g1_ys, 356 + 190, K2_X),
                     (g2_ys, K2_X + K2_W, 746), (g2_ys, 746 + 190, K3_X)]:
    for y in ys:
        lc.seg(x_a, y, x_b, y, lc.C_GPU_S, 1.3, 'std')

# ---------------- 层尾三连: 先暂存（MTP 旁路）后定型 ----------------
lc.text(1250, 340, '层尾三连：先暂存（MTP 旁路）、后定型（hc_head）· model.py:L1200-L1212',
        9.5, lc.C_MUTE, 'start', maxw=360, tag='fork:note')
# 盲审修复第 2 轮（2026-09-14）：右列三框 260→172 收窄，给右侧 BUF 让出整条无重叠
# 走廊（原 BUF 与 residual_out/hc_head 重叠，白色填充盖掉 hc_head 整条右边框）。
RO_X, RO_W = 1250, 172
lc.rect(RO_X, 356, RO_W, 104, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(RO_X + 12, 374, 'residual_out (1,2,2)', 10, lc.C_TXT, 'start', True, maxw=RO_W - 24, tag='ro:t')
lc.text(RO_X + 12, 394, '[[2.3438, 2.7969],', 9, '#334155', 'start', tag='ro:v0')
lc.text(RO_X + 12, 410, '  [2.3281, 2.7656]]', 9, '#334155', 'start', tag='ro:v1')
lc.text(RO_X + 12, 432, '（pre-hc_head 残差）', 8, lc.C_MUTE, 'start', tag='ro:v2')
# 层尾核 → residual_out
lc.seg(K3_X + K3_W, 408, RO_X, 408, lc.C_GPU_S, 2.0, 'std')

HC_X, HC_W = 1250, 172
lc.rect(HC_X, 505, HC_W, 76, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(HC_X + 12, 525, 'hc_head：对流维加权求和', 10, '#14532d', 'start', True, maxw=HC_W - 24, tag='hc:t')
lc.text(HC_X + 12, 543, '(1,2,2)→(1,2) 唯一收敛算子', 8.5, '#166534', 'start', maxw=HC_W - 22, tag='hc:l')
lc.text(HC_X + 12, 560, 'hc_head_fused_kernel_tilelang', 8, lc.C_FAINT, 'start', maxw=HC_W - 22, tag='hc:f')
lc.seg(RO_X + 60, 460, HC_X + 60, 505, lc.C_GPU_S, 2.0, 'std')          # 主路
lc.text(HC_X + 70, 489, '主路', 8.5, lc.C_GPU_S, 'start', tag='main:lab')

OUT_Y = 613
lc.rect(HC_X, OUT_Y, HC_W, 100, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(HC_X + 12, OUT_Y + 20, 'norm → hidden_out (1,2)', 10, lc.C_TXT, 'start', True,
        maxw=HC_W - 24, tag='out:t')
lc.text(HC_X + 12, OUT_Y + 40, '[2.7188, 3.25]', 9, '#334155', 'start', tag='out:v')
lc.text(HC_X + 12, OUT_Y + 60, '→ 下一层 / compute_logits', 8.5, lc.C_MUTE, 'start',
        maxw=HC_W - 22, tag='out:l')
lc.text(HC_X + 12, OUT_Y + 80, '（采样位，2D 回到 2D）', 8, lc.C_MUTE, 'start', tag='out:l2')
lc.seg(HC_X + 60, 581, HC_X + 60, OUT_Y, lc.C_GPU_S, 2.0, 'std')

# 旁路: residual_out → _mtp_hidden_buffer（虚线）
# 盲审修复第 2 轮（2026-09-14，箭头悬空+压字判 FAIL）：BUF 右移到 x1478，与右列
# （x1250-1422）/OUT 全部脱开，hc_head 右边框不再被白色填充遮盖；旁路箭头两端贴
# 框边（RO 右缘 1422 → BUF 左缘 1478），y=448 对准框内首行 .copy_(flatten(1))
# （baseline 452），与标题带（baseline 432，band ~425-434）净空 ~13px 不压字顶；
# 标签居中骑箭头上方（baseline 438，两侧距 RO/BUF 框边各 ~8px）。
BUF_X, BUF_W, BUF_Y, BUF_H = 1478, 130, 414, 176
lc.rect(BUF_X, BUF_Y, BUF_W, BUF_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.4, dash=True)
lc.text(BUF_X + 10, BUF_Y + 18, '_mtp_hidden_buffer', 9.5, '#9a3412', 'start', True,
        maxw=BUF_W - 20, tag='buf:t')
for i, s in enumerate([
        '.copy_(flatten(1))', 'flat (1,4)', '[2.3438, 2.7969,', '  2.3281, 2.7656]',
        '形状 (max_num_batched_', 'tokens, hc_mult×H)', 'pre-hc_head 残差',
        '→ MTP draft 原料', '（runner 采样后取走）']):
    lc.text(BUF_X + 10, BUF_Y + 38 + i * 15.5, s, 8, '#334155', 'start', maxw=BUF_W - 16,
            tag='buf:l%d' % i)
lc.seg(RO_X + RO_W, 448, BUF_X, 448, lc.C_ENG_S, 1.6, 'std', dash=True)
lc.text((RO_X + RO_W + BUF_X) / 2, 438, '旁路 copy_', 8, lc.C_ENG_S, 'middle', tag='bypass:lab')

# ---------------- 底部三注 ----------------
NY, NH = 758, 148
lc.rect(MX, NY, 500, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 14, NY + 20, 'Sinkhorn：搬运矩阵不创造、不销毁流量', 10, lc.C_TXT, 'start', True,
        maxw=460, tag='na:t')
for i, s in enumerate([
        '· res_mix 经 Sinkhorn 归一逼近双随机矩阵（行列和≈1）',
        '· 本例行和偏差：repeat=1 → 0.000247；repeat≥2 → 1e-06',
        '· 取值对齐树内测试 sinkhorn_repeat=20、eps=1e-6；',
        '   真实模型读 config.hc_sinkhorn_iters / hc_eps']):
    lc.text(MX + 14, NY + 42 + i * 17, s, 8.5, '#334155', 'start', maxw=470, tag='na:l%d' % i)
lc.text(MX + 14, NY + NH - 12, 'vllm/model_executor/kernels/mhc/tilelang.py（Sinkhorn 在核内）',
        8, lc.C_FAINT, 'start', maxw=470, tag='na:f')

lc.rect(576, NY, 560, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(590, NY + 20, 'hc 参数账（「学习式混合」的价签）', 10, lc.C_TXT, 'start', True,
        maxw=500, tag='nb:t')
for i, s in enumerate([
        '· hc_attn_fn / hc_ffn_fn 各 (mix_hc, hc_dim)',
        '   = ((2+hc_mult)·hc_mult, hc_mult·H)：本例 (8,4)；',
        '   hc_mult=4 测试基线 → (24, 4H) 同构放大',
        '· hc_head_fn (hc_mult, hc_dim)=(2,4)；hc_post_alpha=2.0',
        '   （硬编码真值，测试用 1.0，model.py:L852）',
        '· 数学真源 = 生产 tilelang 核的对拍基准（atol 5e-2）']):
    lc.text(590, NY + 40 + i * 17, s, 8.5, '#334155', 'start', maxw=530, tag='nb:l%d' % i)

lc.rect(1156, NY, 452, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(1170, NY + 20, '首层 broadcast 预折 + 数值口径', 10, lc.C_TXT, 'start', True,
        maxw=400, tag='nc:t')
for i, s in enumerate([
        '· 预折等价：fn 沿流维求和后直接算 ≡ 先展开再乘',
        '   （allclose=true，最大差 1.2e-07）',
        '· finalize_mhc_broadcast_weights：',
        '   fn.view(-1, hc_mult, H).sum(1)（model.py:L1371-L1380）',
        '· 数值 = bfloat16 存储后的实跑值（T=1 玩具例）',
        '· 2D 输入各流相同，故可沿流维预折']):
    lc.text(1170, NY + 40 + i * 17, s, 8.5, '#334155', 'start', maxw=424, tag='nc:l%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 942, '图例：绿 = hc 核（GPU 执行臂角色）· 灰虚线框 = 占位子层（注意力/FFN 本体不在本图）· 橙虚线 = pre-hc_head 旁路（MTP）· 三股平行线 = 三件套（residual/post_mix/res_mix）',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 962, '形状与数值 = 本章实跑（与正文数值表同源）· 锚点 = vllm/models/deepseek_v4/nvidia/model.py:L816-L897/L1095-L1106/L1200-L1212 · vllm/model_executor/kernels/mhc/tilelang.py:L720-L748',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-mhc-residual-journey.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
