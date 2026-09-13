#!/usr/bin/env python3
"""ch25 机制图 · 权重吸收重排:同一份权重,三种排布常驻(figure ch25-fig-weight-reorder,模板 before-after)

放大自 L0『模型层 MLA 框』装配期收尾的权重重排——L2 站 4(process_weights_after_loading)。
左=kv_b_proj 原权重大块;中=转置→view→split 三步(矩阵画格示意按头切开);
右=两份 bmm 就绪副本(吸收腿/输出上投影),原形状旁标『prefill 腿仍用它』;
底部=「三处常驻」同一色系三种纹理 + 注释原话。

claim:process_weights_after_loading 把 kv_b_proj 同一份权重拆成两份 bmm 排布副本常驻:
权重转置后按头拆 W_UK/W_UV,再重排成 W_UV(N,L,V) 与 W_UK_T(N,P,L)——prefill 腿用原形状、
decode 腿用副本,同一份权重两种形状存两份。

数字全部取自 figure spec 的 numbers(view/split 现场 L1033-L1035 · DSV3 [32768,512]→
W_UK_T [128,128,512]+W_UV [128,512,128] · MINI [128,512]→[4,16,512]/[4,512,16] ·
注释原话 L995-L998)。坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 820
MX = 52
BXR = 1448
C_W = lc.C_GPU_S              # 权重 = 绿系(模型层权重),三种纹理区分排布
C_W_T = '#166534'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '权重跟着腿走:一生一次的重排——同一份 kv_b_proj,装配后三种排布常驻',
        16, lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, '加载完成后把权重转置、view 成 [L=512, N=128 头, P+V=256]、split 出每头 K 半段与 V 半段,再重排成 bmm 就绪形状',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 4 的机制展开'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左:原权重大块 ----------------
OB_ = (60, 150, 240, 190)
lc.rect(*OB_, '#ffffff', C_W, rx=8, sw=2.0)
lc.text(OB_[0] + OB_[2] / 2, 176, 'kv_b_proj(原形状)', 11, C_W_T, 'middle', True, tag='o:t')
lc.text(OB_[0] + OB_[2] / 2, 196, 'DSV3 [32768, 512]', 9.5, C_W, 'middle', True, tag='o:s')
lc.text(OB_[0] + OB_[2] / 2, 214, '(MINI [128, 512])', 8, lc.C_MUTE, 'middle', tag='o:n')
# 画格示意:8×4 小格矩阵
gx0, gy0, gc = OB_[0] + 40, 236, 20
for r in range(4):
    for c in range(8):
        lc.rect(gx0 + c * gc, gy0 + r * gc, gc - 1.5, gc - 1.5, '#dcfce7', 'none', rx=1, sw=0)
lc.text(OB_[0] + OB_[2] / 2, gy0 + 4 * gc + 16, '输出宽 32768 = 128 头 × (128+128)',
        7.5, lc.C_MUTE, 'middle', maxw=OB_[2] - 16, tag='o:g')

# ---------------- 中:三步小图(转置 → view → split) ----------------
SX0 = 380
steps = [
    ('① 转置 .T', '[512, 32768]', '行=潜维 L,列=输出'),
    ('② view', '[512, 128, 256]', 'L × N 头 × (P+V)'),
    ('③ split(dim=-1)', 'W_UK (512,128,128)\nW_UV (512,128,128)', 'K 半段 | V 半段'),
]
sw_, sh_, sgap = 140, 128, 36
for i, (t, s1, s2) in enumerate(steps):
    x = SX0 + i * (sw_ + sgap)
    lc.rect(x, 150, sw_, sh_, '#ffffff', C_W, rx=8, sw=1.5)
    lc.text(x + sw_ / 2, 172, t, 10, C_W_T, 'middle', True, maxw=sw_ - 12, tag='s%d:t' % i)
    lines = s1.split('\n')
    for j, ln in enumerate(lines):
        lc.text(x + sw_ / 2, 196 + j * 16, ln, 8.5, C_W, 'middle', True, maxw=sw_ - 12,
                tag='s%d:l%d' % (i, j))
    lc.text(x + sw_ / 2, 252, s2, 7.5, lc.C_MUTE, 'middle', maxw=sw_ - 12, tag='s%d:n' % i)
    if i == 2:
        # split 示意:每头一列切成 K|V 两半(2 头示意)
        hy = 268
        for h_ in range(2):
            hx = x + 24 + h_ * 56
            lc.rect(hx, hy, 24, 26, '#dcfce7', C_W, rx=2, sw=1.0)
            lc.text(hx + 12, hy + 17, 'K', 8, '#166534', 'middle', True, tag='sp:k%d' % h_)
            lc.rect(hx + 26, hy, 24, 26, '#bbf7d0', C_W, rx=2, sw=1.0)
            lc.text(hx + 38, hy + 17, 'V', 8, '#166534', 'middle', True, tag='sp:v%d' % h_)
    if i < 2:
        lc.seg(x + sw_, 214, x + sw_ + sgap, 214, C_W, 1.8, 'std')
lc.seg(OB_[0] + OB_[2], 245, SX0, 214, C_W, 1.8, 'std')
lc.text(SX0 + (3 * sw_ + 2 * sgap) / 2 - sw_ / 2, 300, 'vllm/model_executor/layers/attention/mla_attention.py:L994-L1100 · 拆分现场 L1033-L1035',
        8, lc.C_FAINT, 'middle', maxw=560, tag='mid:anc')

# ---------------- 右:两份 bmm 副本 + 消费腿 ----------------
RB0 = 916
# W_UK_T 副本
UKB = (RB0, 150, 250, 120)
lc.rect(*UKB, '#f0fdf4', C_W, rx=8, sw=2.0)
lc.text(UKB[0] + UKB[2] / 2, 174, 'W_UK_T [128, 128, 512]', 10.5, C_W_T, 'middle', True, tag='uk:t')
lc.text(UKB[0] + UKB[2] / 2, 194, '(N, P, L)——bmm 就绪', 8.5, C_W, 'middle', tag='uk:s')
lc.text(UKB[0] + UKB[2] / 2, 216, '吸收腿乘它:(N,B,P)×(N,P,L)', 8, '#334155', 'middle', maxw=UKB[2] - 14,
        tag='uk:c')
lc.text(UKB[0] + UKB[2] / 2, 234, 'decode 腿消费', 9, C_W_T, 'middle', True, tag='uk:d')
# 斜纹示意(SVG pattern 不可用 → 画斜线组)
for k in range(9):
    lc.seg(UKB[0] + 8 + k * 26, UKB[1] + 96, UKB[0] + 22 + k * 26, UKB[1] + 112, '#86efac', 2.2,
           marker=None)
# W_UV 副本
UVB = (RB0, 290, 250, 120)
lc.rect(*UVB, '#f0fdf4', C_W, rx=8, sw=2.0)
lc.text(UVB[0] + UVB[2] / 2, 314, 'W_UV [128, 512, 128]', 10.5, C_W_T, 'middle', True, tag='uv:t')
lc.text(UVB[0] + UVB[2] / 2, 334, '(N, L, V)——bmm 就绪', 8.5, C_W, 'middle', tag='uv:s')
lc.text(UVB[0] + UVB[2] / 2, 356, '输出上投影乘它:(N,B,L)×(N,L,V)', 8, '#334155', 'middle',
        maxw=UVB[2] - 14, tag='uv:c')
lc.text(UVB[0] + UVB[2] / 2, 374, 'decode 腿消费', 9, C_W_T, 'middle', True, tag='uv:d')
for k in range(9):
    lc.seg(UVB[0] + 8 + k * 26, UVB[1] + 96, UVB[0] + 22 + k * 26, UVB[1] + 112, '#86efac', 2.2,
           marker=None)
# 原形状消费注(右列顶)
PF = (RB0 + 262, 150, 270, 120)
lc.rect(*PF, '#ffffff', C_W, rx=8, sw=1.5)
lc.text(PF[0] + PF[2] / 2, 174, '原形状 [32768, 512]', 10, C_W_T, 'middle', True, tag='pf:t')
lc.text(PF[0] + PF[2] / 2, 194, '不转置、不拆分', 8.5, lc.C_MUTE, 'middle', tag='pf:s')
lc.text(PF[0] + PF[2] / 2, 216, 'prefill 腿上投影仍用它', 9, C_W_T, 'middle', True, maxw=PF[2] - 14,
        tag='pf:c')
lc.text(PF[0] + PF[2] / 2, 240, '(站 12 的 kv_b_proj GEMM)', 8, lc.C_MUTE, 'middle', tag='pf:n')
# 中部 → 右部箭头
s2r = SX0 + 2 * (sw_ + sgap) + sw_
lc.parrow([(s2r, 214), (RB0, 200)], C_W, 1.8, 'std')
lc.parrow([(s2r, 240), (RB0, 340)], C_W, 1.8, 'std')
lc.parrow([(OB_[0] + OB_[2] / 2, OB_[1] + OB_[3]), (OB_[0] + OB_[2] / 2, 420), (1180, 420),
           (1180, PF[1] + PF[3])], C_W, 1.6, 'std', dash=True)
lc.text(1188, 414, '原权重不动', 7.5, lc.C_MUTE, 'start', tag='pf:lbl')

# ---------------- 底部:三处常驻横带 + 注释原话 ----------------
BB = (60, 450, 1382, 108)
lc.rect(*BB, lc.C_GPU_F, C_W, rx=9, sw=1.6)
lc.text(80, 474, '装配收尾后的常驻状态:同一份权重,三种排布(显存换速度的明码交易)', 11, C_W_T,
        'start', True, maxw=900, tag='bb:t')
cells = [
    (80, 500, 400, 40, '#ffffff', '原形状 kv_b_proj [32768, 512]', 'prefill 腿(白底)'),
    (500, 500, 400, 40, '#f0fdf4', 'W_UK_T [128,128,512] + W_UV [128,512,128]', 'decode 腿(浅绿底+斜纹)'),
    (920, 500, 500, 40, '#dcfce7', 'replace_parameter 换成 bmm 副本常驻', '一生一次,加载后不再变'),
]
for x, y, w_, h_, fl, t1, t2 in cells:
    lc.rect(x, y, w_, h_, fl, C_W, rx=6, sw=1.2)
    lc.text(x + w_ / 2, y + 17, t1, 8.5, C_W_T, 'middle', True, maxw=w_ - 12, tag='bb:c1')
    lc.text(x + w_ / 2, y + 33, t2, 7.5, lc.C_MUTE, 'middle', tag='bb:c2')

QB = (60, 580, 1382, 76)
lc.rect(*QB, '#ffffff', lc.C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(QB[0] + QB[2] / 2, 602, '注释原话(mla_attention.py:L995-L998):暂无量化 bmm,就存 fp16/bf16 副本做 16-bit bmm,额外显存开销 fairly low',
        9, lc.C_MUTE, 'middle', maxw=1360, tag='q:1')
lc.text(QB[0] + QB[2] / 2, 624, 'MINI 同款:原权重 [128, 512] 与 W_UK_T [4,16,512] / W_UV [4,512,16] 同驻——host 实测同一实例两份并存',
        8.5, lc.C_MUTE, 'middle', maxw=1360, tag='q:2')
lc.text(QB[0] + QB[2] / 2, 644, '重排只动排布不动数值——这是 MQA 吸收腿等价性(下一张图)的前提之二', 8.5,
        lc.C_MUTE, 'middle', tag='q:3')

# ---------------- 页脚 ----------------
lc.text(MX, 700, '图例:绿 = 模型层权重(GPU 执行臂角色) · 白/浅绿/网格 = 同一份权重的三种排布纹理 · 绿虚线 = 原权重不动',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 720, '形状 = host 实测例化(DSV3 实尺 + MINI 双列) · 行号基线 vLLM v0.27.1(6e448d0ea)', 8.5,
        lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-weight-reorder.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
