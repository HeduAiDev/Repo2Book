#!/usr/bin/env python3
"""ch24 机制图 ① · MHA 接线:四拍与 llama.py forward 五行同构(figure_spec ch24-fig-mha-wiring,模板 tensor-flow)

放大自 L0 中列『GPU 执行臂』(绿)的『模型层 forward + 编译』块——ch23 拼装 LlamaDecoderLayer
的那块注意力积木,本图打开积木内部的形状账。primer 推导链第 ① 环(基线):后面四环全是改
这张图的形状。架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:MHA 四拍(三投影→逐头切分→逐头 softmax(QK^T/√d_h)V→W^O 拼回)在 T=3/H=2/d_h=4
玩具上每步形状可心算,且与 llama.py forward 五行逐行同构——split 尺寸表里 kv_size 出现
两次就是『K 一份 V 一份』的 2 因子,scaling=0.5 就是式(7) 的 1/√d_h。

数字全部取自 figure_spec.numbers(形状链/因果 softmax 两行/scale=0.5/缓存 16 元素/五行同构:
论文忠实 NumPy 参考实现实跑 trace;vLLM 锚点 llama.py:L157-L159、L221-L231)。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 762
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'MHA 的接线:四拍走完一个注意力层——三投影 → 切头 → 逐头因果 softmax → 拼回',
        16.5, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '玩具 T=3 · H=2 头 · d_h=4 · d=6,每步形状可心算;下排 vLLM 的 llama.py forward 五行与四拍逐行同构(arXiv:2405.04434 §2.1.1 Eq.(1)-(8))',
        10.5, lc.C_MUTE, 'start', maxw=1140, tag='subtitle')
_ch = '推导链 ① · 放大自 L0『GPU 执行臂』模型层 forward 块内的注意力积木'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 四拍区 ----------------
Z1, Z2, Z3, Z4 = (60, 355), (370, 640), (655, 1120), (1135, 1440)
BEAT_Y = 108
for (zx0, zx1), bt in zip((Z1, Z2, Z3, Z4),
                          ('① 三投影 Eq.(1)-(3)', '② 逐头切分 Eq.(4)-(6)',
                           '③ 逐头注意力 Eq.(7)', '④ 输出拼回 Eq.(8)')):
    lc.text((zx0 + zx1) / 2, BEAT_Y, bt, 11, lc.C_TXT, 'middle', True,
            maxw=zx1 - zx0, tag='beat:' + bt[:2])

# ----- ① 三投影 -----
HB = (66, 150, 92, 110)          # h 框
lc.rect(*HB, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(HB[0] + HB[2] / 2, 166, 'h', 11, lc.C_TXT, 'middle', True, tag='h:t')
for i in range(3):               # T=3 个 token 行
    lc.rect(HB[0] + 8, 176 + i * 24, HB[2] - 16, 18, '#f1f5f9', '#cbd5e1', rx=3, sw=1.0)
lc.text(HB[0] + HB[2] / 2, 254, '(3, 6)', 9.5, lc.C_MUTE, 'middle', tag='h:shape')
lc.seg(HB[0] + HB[2], 205, 196, 205, lc.C_MUTE, 1.8, marker='std')

PB = (196, 142, 130, 126)        # 三投影框
lc.rect(*PB, '#ffffff', lc.C_GPU_S, rx=7, sw=1.5)
lc.text(PB[0] + PB[2] / 2, 160, '三投影', 9.5, lc.C_GPU_S, 'middle', True, tag='pb:t')
for i, wn in enumerate(('W^Q (8×6)', 'W^K (8×6)', 'W^V (8×6)')):
    lc.rect(PB[0] + 8, 170 + i * 30, PB[2] - 16, 24, lc.C_GPU_F, 'none', rx=4, sw=0)
    lc.text(PB[0] + PB[2] / 2, 186 + i * 30, wn, 9, lc.C_TXT, 'middle', tag='pb:' + wn[:3])
lc.text(PB[0] + PB[2] / 2, 285, '每个投影:8 行 × 6 列', 8, lc.C_MUTE, 'middle', tag='pb:note')

QKV_Y = (148, 202, 256)          # q/k/v 三框顶
QKV_N = ('q', 'k', 'v')
qkv_boxes = []
for (qy, qn) in zip(QKV_Y, QKV_N):
    lc.seg(PB[0] + PB[2], qy + 20, 350, qy + 20, lc.C_GPU_S, 1.8, marker='std')
    lc.rect(350, qy, 88, 40, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.5)
    lc.text(394, qy + 24, f'{qn} (3, 8)', 10.5, lc.C_TXT, 'middle', True, tag='qkv:' + qn)
    qkv_boxes.append((350, qy, 88, 40))

# ----- ② 逐头切分 -----
CELL_W, CELL_H = 16, 24
STRIP_X = 466
HEAD_F = ('#dcfce7', '#bbf7d0')  # head0 深 / head1 浅
for si, (qy, qn) in enumerate(zip(QKV_Y, QKV_N)):
    cy = qy + 8                  # 条带与 q/k/v 框同中心
    lc.seg(438, qy + 20, 462, qy + 20, lc.C_GPU_S, 1.6, marker='std')
    for ci in range(8):
        lc.rect(STRIP_X + ci * CELL_W, cy, CELL_W, CELL_H, HEAD_F[ci // 4],
                lc.C_GPU_S, rx=2, sw=1.0)
    lc.seg(STRIP_X + 8 * CELL_W, qy + 20, Z3[0], qy + 20, lc.C_GPU_S, 1.6, marker='std')
lc.text(STRIP_X + 2 * CELL_W, 150, 'head0 (4 维)', 7.5, lc.C_MUTE, 'middle', tag='hd0')
lc.text(STRIP_X + 6 * CELL_W, 150, 'head1 (4 维)', 7.5, lc.C_MUTE, 'middle', tag='hd1')
lc.text((Z2[0] + Z2[1]) / 2, 320, '(3, 8) → (3, 2, 4):每头分到 4 维', 9, lc.C_TXT,
        'middle', True, tag='z2:shape')
lc.text((Z2[0] + Z2[1]) / 2, 340, '每 token 缓存 2×2×4 = 16 元素/层', 9, lc.C_KV_S,
        'middle', True, tag='z2:cache')

# ----- ③ 逐头注意力面板 -----
P3 = (Z3[0], 122, Z3[1] - Z3[0], 350)
lc.rect(*P3, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(P3[0] + P3[2] / 2, 144, 'head0 的分数矩阵 (2, 3, 3) 里的一张 (3×3)', 10.5,
        lc.C_TXT, 'middle', True, maxw=P3[2] - 20, tag='p3:t')

MG = (712, 178)                  # 矩阵原点(左上)
MC = 62                          # 格宽
for j in range(3):
    lc.text(MG[0] + j * MC + MC / 2, 172, f'j={j}', 8.5, lc.C_MUTE, 'middle', tag=f'mg:h{j}')
for i in range(3):
    lc.text(MG[0] - 8, MG[1] + i * MC + MC / 2 + 3, f't={i}', 8.5, lc.C_MUTE, 'end', tag=f'mg:r{i}')
SCORES = {(0, 0): '10.0789', (2, 0): '-0.1727', (2, 1): '7.6051', (2, 2): '6.0289'}
for i in range(3):
    for j in range(3):
        x, y = MG[0] + j * MC, MG[1] + i * MC
        if j > i:                # 上三角 = 因果掩码
            lc.rect(x, y, MC, MC, '#e2e8f0', '#cbd5e1', rx=3, sw=1.0)
            lc.text(x + MC / 2, y + MC / 2 + 3, '-inf', 9, lc.C_FAINT, 'middle', tag=f'mg:{i}{j}')
        else:
            hot = (i == 2)
            lc.rect(x, y, MC, MC, lc.C_GPU_F if hot else '#ffffff', '#cbd5e1', rx=3, sw=1.0)
            v = SCORES.get((i, j), '·')
            lc.text(x + MC / 2, y + MC / 2 + 3, v, 9 if v != '·' else 10,
                    lc.C_TXT if v != '·' else lc.C_FAINT, 'middle', hot, tag=f'mg:{i}{j}v')
lc.text(MG[0] + 3 * MC / 2, MG[1] + 3 * MC + 18,
        '灰格 = 因果掩码 -inf(Eq.(7) 求和上限 j ≤ t)', 8.5, lc.C_MUTE, 'middle', tag='p3:mask')
lc.text(MG[0] + 3 * MC / 2, MG[1] + 3 * MC + 34,
        '· = 因果可见、未标值(本图只印实跑核验的 t=0 / t=2 两行)', 8.5, lc.C_FAINT,
        'middle', maxw=270, tag='p3:dot')

# 面板右列:t=2 行因果 softmax
RX = 918
lc.text(RX, 178, '缩放:分数 = q·k / √d_h', 9, lc.C_TXT, 'start', True, tag='sc:t')
lc.text(RX, 193, '分母 √4 → scale = 0.5', 8.5, lc.C_MUTE, 'start', tag='sc:l')
lc.text(RX, 207, '(代码 self.scaling = self.head_dim**-0.5)', 7.8, lc.C_FAINT,
        'start', maxw=196, tag='sc:l2')
lc.text(RX + 27, 224, '分数(t=2 行)', 8, lc.C_MUTE, 'middle', tag='sc:lbl')
for k, sv in enumerate(('-0.1727', '7.6051', '6.0289')):
    lc.rect(RX + k * 62, 230, 58, 22, '#ffffff', '#cbd5e1', rx=4, sw=1.0)
    lc.text(RX + k * 62 + 29, 245, sv, 8.5, lc.C_TXT, 'middle', tag='sc:v' + str(k))
lc.seg(RX + 87, 254, RX + 87, 272, lc.C_GPU_S, 1.6, marker='std')
lc.text(RX + 97, 266, '因果 softmax', 8, lc.C_GPU_S, 'start', tag='sc:op')
for k, pv in enumerate(('0.0003', '0.8284', '0.1713')):
    lc.rect(RX + k * 62, 276, 58, 22, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.0)
    lc.text(RX + k * 62 + 29, 291, pv, 8.5, lc.C_TXT, 'middle', True, tag='pr:v' + str(k))
lc.text(RX + 87, 312, '和 = 1.0', 8.5, lc.C_MUTE, 'middle', tag='pr:sum')
lc.text(RX, 336, '因果核验:t=0 行 = [10.0789, -inf, -inf]', 8.5, lc.C_TXT, 'start', tag='ck:1')
lc.text(RX, 352, '→ 概率 [1, 0, 0]:未来恰为 0(精确,非近似)', 8.5, lc.C_TXT,
        'start', maxw=196, tag='ck:2')

HD = (671, 404, 433, 52)         # 逐头独立注
lc.rect(*HD, '#f8fafc', '#cbd5e1', rx=6, sw=1.1)
lc.text(HD[0] + HD[2] / 2, 422, '逐头独立:同一 token 的 q,head0 ≠ head1——各算各的分数 (2, 3, 3)',
        8.5, lc.C_TXT, 'middle', maxw=HD[2] - 16, tag='hd:t')
lc.text(HD[0] + HD[2] / 2, 440, 'token0:head0 = [1.657, -8.118, -2.781, 8.368] / head1 = [1.165, -1.727, -0.011, -3.407]',
        7.8, lc.C_MUTE, 'middle', maxw=HD[2] - 12, tag='hd:v')

# ----- ④ 输出拼回 -----
lc.seg(Z3[1], 166, 1138, 166, lc.C_MUTE, 1.6, marker='std')
lc.seg(Z3[1], 206, 1138, 206, lc.C_MUTE, 1.6, marker='std')
lc.rect(1138, 150, 90, 32, HEAD_F[0], lc.C_GPU_S, rx=6, sw=1.2)
lc.text(1183, 170, 'o head0 (3,4)', 8.5, lc.C_TXT, 'middle', tag='o:a')
lc.rect(1138, 190, 90, 32, HEAD_F[1], lc.C_GPU_S, rx=6, sw=1.2)
lc.text(1183, 210, 'o head1 (3,4)', 8.5, lc.C_TXT, 'middle', tag='o:b')
lc.seg(1228, 166, 1252, 166, lc.C_MUTE, 1.6, marker='std')
lc.seg(1228, 206, 1252, 206, lc.C_MUTE, 1.6, marker='std')
lc.rect(1252, 150, 86, 72, '#ffffff', lc.C_MUTE, rx=6, sw=1.3)
lc.text(1295, 178, '拼回全部头', 9, lc.C_TXT, 'middle', True, tag='cc:t')
lc.text(1295, 200, '(3, 2, 4) → (3, 8)', 8.5, lc.C_MUTE, 'middle', tag='cc:s')
lc.seg(1338, 186, 1356, 186, lc.C_MUTE, 1.6, marker='std')
lc.rect(1356, 162, 72, 48, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.3)
lc.text(1392, 182, 'W^O', 10, lc.C_TXT, 'middle', True, tag='wo:t')
lc.text(1392, 198, '(8×8)', 8.5, lc.C_MUTE, 'middle', tag='wo:s')
lc.seg(1392, 210, 1392, 232, lc.C_MUTE, 1.6, marker='std')
lc.rect(1356, 232, 72, 40, '#ffffff', lc.C_MUTE, rx=6, sw=1.3)
lc.text(1392, 256, 'u (3, 8)', 9.5, lc.C_TXT, 'middle', True, tag='u:t')
lc.text(1138, 276, 'o(t=2, head0) = [-1.4976, -0.3665,', 8, lc.C_MUTE, 'start', tag='o:v1')
lc.text(1138, 290, '  4.1907, 0.4120]', 8, lc.C_MUTE, 'start', tag='o:v2')
lc.text(1290, 330, '进出宽度被矩阵形状定死:8 进 8 出', 8.5, lc.C_MUTE, 'middle',
        maxw=290, tag='z4:cap')

# ---------------- vLLM 同构带 ----------------
VB = (MX, 495, BXR - MX, 155)
lc.rect(*VB, '#ffffff', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + 15, 517, '同一四拍,vLLM 的长相——LlamaAttention.forward 五行,逐行同构',
        11, lc.C_GPU_S, 'start', True, maxw=800, tag='vb:t')
CELLS = [
    ('qkv_proj(hidden)', '一把投出 24 = 8+8+8', 'Eq.(1)-(3) 三投影合一'),
    ('split([q_size, kv_size, kv_size])', 'q, k, v = qkv.split(...)', 'Eq.(4)-(6) 切头'),
    ('rotary_emb(positions, q, k)', '位置旋转作用于 q / k', '(进入 Eq.(7) 前的位置编码)'),
    ('attn(q, k, v) 插座', 'kernel 数学回指 ch20', '后端家族回指 ch21'),
    ('o_proj(attn_output)', '投回隐维度 d = 6', 'Eq.(8) 输出投影'),
]
cw_, gap_ = 249, 26
cx0 = MX + 15
for i, (t1, t2, t3) in enumerate(CELLS):
    x = cx0 + i * (cw_ + gap_)
    lc.rect(x, 531, cw_, 58, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.1)
    lc.text(x + cw_ / 2, 548, t1, 8.5 if len(t1) > 22 else 9.5, lc.C_TXT, 'middle', True,
            maxw=cw_ - 12, tag=f'vc{i}:t')
    lc.text(x + cw_ / 2, 566, t2, 8.5, '#334155', 'middle', maxw=cw_ - 10, tag=f'vc{i}:m')
    lc.text(x + cw_ / 2, 582, t3, 8, lc.C_MUTE, 'middle', maxw=cw_ - 10, tag=f'vc{i}:b')
    if i < len(CELLS) - 1:
        lc.seg(x + cw_ + 3, 560, x + cw_ + gap_ - 4, 560, lc.C_GPU_S, 1.6, marker='std')
# split 尺寸表(kv_size 两次高亮)
SZ = [('q_size', '8', False), ('kv_size', '8', True), ('kv_size', '8', True)]
sx = cx0 + (cw_ + gap_)
for i, (nm, val, hot) in enumerate(SZ):
    lc.rect(sx + i * 80, 601, 76, 22, lc.C_KV_F if hot else '#f1f5f9',
            lc.C_KV_S if hot else '#cbd5e1', rx=4, sw=1.2 if hot else 1.0)
    lc.text(sx + i * 80 + 38, 616, f'{nm} = {val}', 8, lc.C_KV_S if hot else lc.C_MUTE,
            'middle', hot, tag=f'sz{i}')
lc.text(sx + 3 * 80 + 12, 616, 'kv_size 出现两次 = 『K 一份 V 一份』的 2 因子', 8.5,
        lc.C_KV_S, 'start', True, maxw=560, tag='sz:note')
lc.text(MX + 15, 641, '五行出自 vllm/model_executor/models/llama.py:L221-L231 · q_size / kv_size / scaling 三行 = L157-L159(MHA 时 q_size = kv_size = 8)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX - 30, tag='vb:anchor')

# ---------------- 页脚 ----------------
lc.text(MX, 676, '图例:绿 = 投影/执行流 · 青 = KV 缓存账 · 灰格 = 因果掩码(-inf)· 条带深浅 = head0 / head1',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 698, '本章所有变体(GQA / MQA / MLA)改的都是这张图的形状——头数、份数、缓存宽度;四拍结构不动。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='ft:claim')
lc.text(MX, 720, '公式出处 arXiv:2405.04434 §2.1.1 Eq.(1)-(8) · 数值取自论文忠实 NumPy 参考实现实跑(host,float64)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 736, 'vLLM 行号基线 v0.27.1(6e448d0ea)', 8.5, lc.C_FAINT, 'start', tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch24-fig-mha-wiring.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
