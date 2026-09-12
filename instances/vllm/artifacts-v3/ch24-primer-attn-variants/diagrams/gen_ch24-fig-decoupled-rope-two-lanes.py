#!/usr/bin/env python3
"""ch24 机制图 ④ · 解耦 RoPE 双流:C 流可吸收,R 流走旁路(figure_spec ch24-fig-decoupled-rope-two-lanes,模板 swimlane)

放大自 L0 中列『GPU 执行臂』(绿)『模型层 forward + 编译』块内 MLA 前向的 rotary 段——
mla.py:L200-L203 只把 q 的后 64 维(q[..., qk_nope_head_dim:])与 k_pe 送进 rotary_emb,
前 128 维内容段不过旋转。primer 推导链第 ④ 环(双流):内容流可压缩可吸收,位置流旁路共享。

claim:位置信息走旁路:C 流(可压缩、可吸收、缓存为 c^KV)与 R 流(承载 RoPE、所有头
共享单份 64 维 k^R、不参与吸收)在拼接 [C;R] 处会合、分母 √(128+64)=√192——若把 RoPE
塞进 C 流,折叠矩阵 F(m,t) 随位置对变化(实测 F(0,0)≠F(0,1)),吸收失效。

数字全部取自 figure_spec.numbers(AB≠BA 直感/F00 vs F01/+100 解耦核验/维度账与分母/
YaRN 落点:论文忠实 NumPy 参考实现实跑 trace)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 892
MX = 60
BXR = 1440
C_BLUE, C_BLUE_T = lc.C_API_S, '#1e40af'   # 论文蓝框=缓存项

# ---------------- 标题区 ----------------
lc.text(MX, 34, '解耦 RoPE:位置信息走旁路——内容卷宗永远可折叠,时间戳盖在共享小纸条上',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'RoPE 的旋转矩阵逐位置不同、矩阵乘法不可交换:塞进内容流,离线吸收矩阵就得逐位置对重算;解法 = 一根全头共享的 64 维 k^R(arXiv:2405.04434 §2.1.3 Eq.(14)-(19))',
        10.5, lc.C_MUTE, 'start', maxw=1150, tag='subtitle')
_ch = '推导链 ④ · 放大自 L0『GPU 执行臂』MLA 前向的 rotary 段'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')


def flowbox(x, y, w, h, t1, t2, stroke, fill='#ffffff', t2c=None, sw=1.3):
    lc.rect(x, y, w, h, fill, stroke, rx=6, sw=sw)
    lc.text(x + w / 2, y + (15 if t2 else 22), t1, 8.5, lc.C_TXT, 'middle', True,
            maxw=w - 8, tag='fb:' + t1[:10])
    if t2:
        lc.text(x + w / 2, y + 29, t2, 7.3, t2c or lc.C_MUTE, 'middle', maxw=w - 8,
                tag='fb2:' + t1[:10])


# ---------------- C 流泳道 ----------------
LANE_C = (MX, 92, 1060, 172)
lc.rect(*LANE_C, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.6)
lc.text(76, 110, 'C 流 · 内容(绿)', 9.5, lc.C_GPU_S, 'start', True, tag='lc:t')
# q 行
flowbox(90, 126, 84, 36, 'c^Q', '(1536)', lc.C_MUTE)
lc.seg(174, 144, 206, 144, lc.C_GPU_S, 1.5, marker='std')
flowbox(206, 126, 100, 36, 'W^UQ', '(16384×1536)', lc.C_GPU_S)
lc.seg(306, 144, 334, 144, lc.C_GPU_S, 1.5, marker='std')
flowbox(334, 126, 100, 36, 'q^C_i', '(128/头)', lc.C_GPU_S)
# k 行(右移错位,给分叉线让路)
flowbox(206, 180, 84, 36, 'c^KV', '(512)', C_BLUE, '#eff6ff', C_BLUE_T, sw=2.2)
lc.seg(290, 198, 316, 198, lc.C_GPU_S, 1.5, marker='std')
flowbox(316, 180, 100, 36, 'W^UK', '(16384×512)', lc.C_GPU_S)
lc.seg(416, 198, 438, 198, lc.C_GPU_S, 1.5, marker='std')
flowbox(438, 180, 100, 36, 'k^C_i', '(128/头)', lc.C_GPU_S)
# 吸收注
AB_ = (560, 122, 535, 90)
lc.rect(*AB_, '#ffffff', lc.C_GPU_S, rx=6, sw=1.1, dash=True)
lc.text(AB_[0] + 14, 142, '可吸收(结合律,离线一次,App C):移动的只是括号——', 9,
        lc.C_GPU_S, 'start', True, maxw=AB_[2] - 24, tag='ab:t')
lc.text(AB_[0] + 14, 162, '分数 = (W^UQ·c^Q)ᵀ(W^UK·c^KV) = ((W^UK)ᵀW^UQ·c^Q)ᵀ·c^KV', 8.5,
        '#334155', 'start', tag='ab:l1')
lc.text(AB_[0] + 14, 180, 'B = (W^UK)ᵀ·W^UQ 离线折好;潜向量直接当 K 用,K/V 甚至不必算出来', 8.5,
        '#334155', 'start', maxw=AB_[2] - 24, tag='ab:l2')
lc.text(AB_[0] + 14, 198, 'Eq.(42)(43)', 7.5, lc.C_FAINT, 'start', tag='ab:eq')
# 泳道脚注
lc.text(590, 248, '本流无 RoPE 算子——位置平移不改变本流任何向量(实测 diff = 0.0)', 8.5,
        lc.C_GPU_S, 'middle', True, maxw=520, tag='lc:ft')

# 分叉线 c^Q → R 流
lc.seg(132, 162, 132, 314, lc.C_GPU_S, 1.3, dash=True)
lc.text(142, 240, 'c^Q 分叉', 7.5, lc.C_GPU_S, 'start', tag='fork')

# ---------------- R 流泳道 ----------------
LANE_R = (MX, 280, 1060, 172)
lc.rect(*LANE_R, '#ecfeff', lc.C_KV_S, rx=9, sw=1.6)
lc.text(76, 298, 'R 流 · 位置(青)', 9.5, lc.C_KV_S, 'start', True, tag='lr:t')
# q 行
flowbox(90, 314, 84, 36, 'c^Q', '(自 C 流)', lc.C_MUTE)
lc.seg(174, 332, 206, 332, lc.C_KV_S, 1.5, marker='std')
flowbox(206, 314, 100, 36, 'W^QR', '(8192×1536)', lc.C_KV_S)
lc.seg(306, 332, 334, 332, lc.C_KV_S, 1.5, marker='std')
flowbox(334, 314, 90, 36, 'RoPE', '旋转 Eq.(14)', lc.C_KV_S)
lc.seg(424, 332, 448, 332, lc.C_KV_S, 1.5, marker='std')
flowbox(448, 314, 110, 36, 'q^R_i', '(64/头,逐头)', lc.C_KV_S)
# k 行
flowbox(90, 368, 84, 36, 'h_t', '(5120)', lc.C_MUTE)
lc.seg(174, 386, 206, 386, lc.C_KV_S, 1.5, marker='std')
flowbox(206, 368, 100, 36, 'W^KR', '(64×5120)', lc.C_KV_S)
lc.seg(306, 386, 334, 386, lc.C_KV_S, 1.5, marker='std')
flowbox(334, 368, 90, 36, 'RoPE', '旋转 Eq.(15)', lc.C_KV_S)
lc.seg(424, 386, 448, 386, lc.C_KV_S, 1.5, marker='std')
flowbox(448, 368, 110, 36, 'k^R', '(64,单份·蓝框)', C_BLUE, '#eff6ff', C_BLUE_T, sw=2.2)
# 不参与吸收注
NR = (590, 310, 500, 100)
lc.rect(*NR, '#ffffff', lc.C_KV_S, rx=6, sw=1.1, dash=True)
lc.text(NR[0] + 14, 330, '不参与吸收:RoPE 矩阵随位置对变化——', 9, lc.C_KV_S, 'start', True,
        maxw=NR[2] - 24, tag='nr:t')
lc.text(NR[0] + 14, 350, 'R(m)ᵀ·R(t) = R(t−m),仅在 m = t 时化为 I', 8.5, '#334155',
        'start', tag='nr:l1')
lc.text(NR[0] + 14, 368, 'k^R 单份共享:不逐头、不压入潜向量', 8.5, '#334155', 'start', tag='nr:l2')
lc.text(NR[0] + 14, 386, '缓存代价:+64,总账 512+64 = 576——『买回吸收权』的价格', 8.5,
        '#334155', 'start', maxw=NR[2] - 24, tag='nr:l3')
lc.text(NR[0] + 14, 402, 'YaRN 扩上下文也只作用 k^R(§3.1.4 原句)', 7.8, lc.C_FAINT,
        'start', tag='nr:y')

# ---------------- 会合节点 ----------------
NODE = (1160, 150, 280, 250)
lc.seg(LANE_C[0] + LANE_C[2], 200, NODE[0], 200, lc.C_GPU_S, 1.8, marker='std')
lc.seg(LANE_R[0] + LANE_R[2], 366, NODE[0], 366, lc.C_KV_S, 1.8, marker='std')
lc.rect(*NODE, '#ffffff', lc.C_TXT, rx=8, sw=1.8)
lc.text(NODE[0] + NODE[2] / 2, 176, '会合:拼接 [C ; R]  Eq.(16)(17)', 10, lc.C_TXT,
        'middle', True, maxw=NODE[2] - 16, tag='nd:t')
lc.text(NODE[0] + NODE[2] / 2, 204, 'q_i = [q^C_i ; q^R_i]', 9.5, lc.C_GPU_S, 'middle',
        True, tag='nd:q')
lc.text(NODE[0] + NODE[2] / 2, 222, '(128 + 64 = 192 维/头)', 8, lc.C_MUTE, 'middle', tag='nd:qn')
lc.text(NODE[0] + NODE[2] / 2, 246, 'k_i = [k^C_i ; k^R]', 9.5, lc.C_KV_S, 'middle', True,
        tag='nd:k')
lc.seg(NODE[0] + 40, 262, NODE[0] + NODE[2] - 40, 262, '#e2e8f0', 1.2)
lc.text(NODE[0] + NODE[2] / 2, 288, '分数 = q_i·k_i / √(d_h+d_h^R)', 9.5, lc.C_TXT,
        'middle', True, maxw=NODE[2] - 16, tag='nd:sc')
lc.text(NODE[0] + NODE[2] / 2, 310, '= √(128+64) = √192', 9, lc.C_TXT, 'middle', tag='nd:sc2')
lc.text(NODE[0] + NODE[2] / 2, 332, 'scale = 0.0722', 10.5, lc.C_TXT, 'middle', True, tag='nd:sc3')
lc.text(NODE[0] + NODE[2] / 2, 352, '(玩具 1/√6 = 0.4082)', 8, lc.C_MUTE, 'middle', tag='nd:sc4')
lc.text(NODE[0] + NODE[2] / 2, 376, 'Eq.(18) 分母', 7.5, lc.C_FAINT, 'middle', tag='nd:eq')

# ---------------- 灾难版风暴框 ----------------
ST = (MX, 470, 1060, 168)
lc.rect(*ST, '#ffffff', lc.C_ABORT, rx=8, sw=1.5, dash=True)
lc.text(MX + 530, 492, '若把 RoPE 塞进 C 流(灾难版):折叠矩阵随位置对变化,吸收失效(§2.1.3 原论证)', 10,
        lc.C_ABORT, 'middle', True, maxw=1030, tag='st:t')
L1 = [
    '① 乘法不可交换(2×2 手算):',
    'W^UQ = [[1,2],[3,4]],W^UK = [[5,6],[7,8]]',
    'AB = [[19,22],[43,50]]  ≠  BA = [[23,34],[31,46]]',
    '差的最大元 12',
    'R(1) = [[0.5403,-0.8415],[0.8415,0.5403]]',
]
for i, ln in enumerate(L1):
    lc.text(90, 516 + i * 22, ln, 8.5, lc.C_TXT if i != 3 else lc.C_ABORT, 'start', i == 3,
            maxw=470, tag=f'st:l{i}')
L2 = [
    '② 折叠矩阵 F(m,t) = (W^UK)ᵀR(m)ᵀR(t)W^UQ:',
    'F(0,0) = [[26,38],[30,44]](= 固定 B,R(0)=I 的退化基例)',
    'F(0,1) = [[7.3161,15.4827],[7.7944,17.0415]]',
    'max 差 26.9585 → 位置对一变折叠矩阵就变',
    '每 token 要为全部前缀重算 keys——红利归零',
]
for i, ln in enumerate(L2):
    lc.text(590, 516 + i * 22, ln, 8.5, lc.C_TXT if i != 3 else lc.C_ABORT, 'start', i == 3,
            maxw=500, tag=f'st:r{i}')

# ---------------- 底部实测条 ----------------
BT = (MX, 656, 1380, 84)
lc.rect(*BT, '#f8fafc', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 690, 678, '位置平移核验:positions 整体 +100 重跑整段前向——位置只走 R 段', 10,
        lc.C_TXT, 'middle', True, maxw=1360, tag='bt:t')
lc.text(MX + 350, 706, 'C 段:q^C / k^C / c^KV 缓存 max diff = 0.0(纹丝不动)', 9,
        lc.C_GPU_S, 'middle', True, maxw=640, tag='bt:c')
lc.text(MX + 1030, 706, 'R 段:q^R diff = 2.0764 · k^R diff = 1.9473(被旋)', 9,
        lc.C_KV_S, 'middle', True, maxw=640, tag='bt:r')
lc.text(MX + 690, 728, '这就是 rotary 只吃 q 后 64 维的原因——位置维度独立成流', 8,
        lc.C_MUTE, 'middle', tag='bt:n')

# ---------------- vLLM 落点 ----------------
VY = 756
lc.rect(MX, VY, BXR - MX, 52, '#ffffff', lc.C_GPU_S, rx=8, sw=1.3, dash=True)
lc.text(MX + 14, VY + 20, 'vLLM 落点:vllm/model_executor/layers/mla.py:L200-L203——rotary 只吃 q[..., qk_nope_head_dim:](后 64 维)与 k_pe;L192 k_pe.unsqueeze(1) 单份广播',
        8.5, lc.C_GPU_S, 'start', True, maxw=BXR - MX - 28, tag='vl:t')
lc.text(MX + 14, VY + 38, '分母 = self.qk_head_dim**-0.5(vllm/model_executor/models/deepseek_v2.py:L1003,含 yarn mscale 修正分支在同函数后段)',
        8, '#334155', 'start', maxw=BXR - MX - 28, tag='vl:l')

# ---------------- 页脚 ----------------
lc.text(MX, 836, '图例:绿 = C 流(内容,可吸收)· 青 = R 流(位置,旁路)· 蓝 = 缓存项(c^KV / k^R)· 红 = 失败对照',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 856, '公式出处 arXiv:2405.04434 §2.1.3 Eq.(14)-(19) · 数值取自论文忠实 NumPy 参考实现实跑(host,float64)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 872, 'vLLM 行号基线 v0.27.1(6e448d0ea)', 8.5, lc.C_FAINT, 'start', tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch24-fig-decoupled-rope-two-lanes.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
