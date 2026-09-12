#!/usr/bin/env python3
"""ch24 机制图 ③ · MLA 低秩联合压缩:只缓存 512 维潜向量(figure_spec ch24-fig-mla-latent-compression,模板 tensor-flow)

放大自 L0 中列『GPU 执行臂』(绿)『模型层 forward + 编译』块内 DeepseekV2MLAAttention 的
投影骨架(fused_qkv_a_proj 的 kv 段 = W^DKV、kv_b_proj = [W^UK;W^UV]),写入侧连到 L0
『调度 · 显存账本』带的 KV 块池(青)——每 token 写进块池的就是蓝框那 576 个数。
primer 推导链第 ③ 环:重绘 V2 Fig.3 MLA 面板的蓝框语义。

claim:MLA 把 K/V 联合压进单个 512 维潜向量 c^KV(缓存蓝框),128 个头各用自己的
W^UK_i/W^UV_i 从同一潜向量恢复 k^C/v^C——共享的是生成基底而非现成向量(GQA 是离散的
头选择、MLA 是连续的低秩子空间),所以 576 个元素等效 2.25 组 GQA 却不受砍头的质量罚。

数字全部取自 figure_spec.numbers(压缩链形状/头间实测 4.944/缓存账 576 vs 32768/
kv_b_proj 32768 巧合/vLLM 投影骨架:论文忠实 NumPy 参考实现实跑 trace)。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 734
MX = 60
BXR = 1440
C_BLUE = lc.C_API_S            # 论文『蓝框=缓存项』语义(区别于角色色,图例声明)
C_BLUE_T = '#1e40af'

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'MLA 低秩联合压缩:不砍头、改降维——只存一份 512 维底稿,各头现场『放大复印』',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '128 份逐头 K/V 不再各自存档:W^DKV 把 h_t 压进单个潜向量 c^KV(缓存蓝框),每个头用自己的 W^UK/W^UV 从同一底稿恢复(arXiv:2405.04434 §2.1.2 Eq.(9)-(11))',
        10.5, lc.C_MUTE, 'start', maxw=1150, tag='subtitle')
_ch = '推导链 ③ · 放大自 L0『GPU 执行臂』DeepseekV2MLAAttention 投影骨架,写入侧连 KV 块池'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左列:纵向压缩链(DSV3 真值) ----------------
HB = (100, 122, 300, 58)       # h_t
lc.rect(*HB, '#ffffff', lc.C_GPU_S, rx=7, sw=1.5)
lc.text(HB[0] + HB[2] / 2, 146, 'h_t —— 每 token 5120 维', 10.5, lc.C_TXT, 'middle', True,
        tag='h:t')
lc.text(HB[0] + HB[2] / 2, 166, '(全部 128 头的信息都从它出发)', 8, lc.C_MUTE, 'middle', tag='h:s')

lc.seg(250, 180, 250, 202, lc.C_GPU_S, 1.8, marker='std')
WB = (185, 202, 130, 40)       # 细腰 W^DKV
lc.rect(*WB, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.5)
lc.text(WB[0] + WB[2] / 2, 220, 'W^DKV 下投影', 9, lc.C_GPU_S, 'middle', True, tag='wdkv:t')
lc.text(WB[0] + WB[2] / 2, 235, '(512 × 5120)  Eq.(9)', 7.8, lc.C_MUTE, 'middle', tag='wdkv:s')
lc.seg(250, 242, 250, 262, lc.C_GPU_S, 1.8, marker='std')

CB = (140, 262, 220, 60)       # 蓝框① c^KV
lc.rect(*CB, '#eff6ff', C_BLUE, rx=7, sw=2.6)
lc.text(CB[0] + CB[2] / 2, 284, 'c^KV 潜向量 · 512 维', 10.5, C_BLUE, 'middle', True, tag='ckv:t')
lc.text(CB[0] + CB[2] / 2, 304, '缓存蓝框①:所有头的 K/V 都能从它恢复', 8, C_BLUE_T,
        'middle', maxw=CB[2] - 10, tag='ckv:s')

KR = (430, 122, 190, 52)       # 蓝框② k^R(旁路)
lc.rect(*KR, '#eff6ff', C_BLUE, rx=7, sw=2.0)
lc.text(KR[0] + KR[2] / 2, 142, 'k^R · 64 维(蓝框②)', 9, C_BLUE, 'middle', True, tag='kr:t')
lc.text(KR[0] + KR[2] / 2, 160, '全头共享单份,只承载位置', 7.5, C_BLUE_T, 'middle', tag='kr:s')
lc.text(KR[0] + KR[2] / 2, 192, 'RoPE(W^KR · h_t)——不逐头、不压入潜向量(位置流旁路,见双流图)',
        7.5, lc.C_FAINT, 'middle', maxw=280, tag='kr:src')

# 扇出:128 头上投影
lc.text(555, 240, '上投影 Eq.(10)(11):每个头 = 同一潜向量的不同线性投影', 9, lc.C_GPU_S,
        'middle', True, maxw=290, tag='up:t')
HEADS = [(262, '0'), (318, '1'), (374, '127')]
for hy, hl in HEADS:
    lc.seg(360, 280 if hy == 262 else (300 if hy == 318 else 312), 430, hy + 20,
           lc.C_GPU_S, 1.3, marker='std')
    lc.rect(430, hy, 150, 40, '#ffffff', lc.C_GPU_S, rx=6, sw=1.2)
    lc.text(505, hy + 17, f'W^UK_{hl} | W^UV_{hl}', 8.5, lc.C_TXT, 'middle', maxw=140,
            tag=f'hd{hl}:w')
    lc.text(505, hy + 32, '(各 128 × 512)', 7.5, lc.C_MUTE, 'middle', tag=f'hd{hl}:s')
    for sy, sn in ((hy + 1, 'k^C (128)'), (hy + 21, 'v^C (128)')):
        lc.seg(580, sy + 9, 606, sy + 9, lc.C_GPU_S, 1.3, marker='std')
        lc.rect(606, sy, 58, 19, '#dcfce7', lc.C_GPU_S, rx=3, sw=1.0)
        lc.text(635, sy + 13, sn, 7.2, lc.C_TXT, 'middle', tag=f'hd{hl}:{sn[0]}')
lc.text(505, 367, '⋮(共 128 头)', 8.5, lc.C_MUTE, 'middle', True, tag='vdots')

# ---------------- 左下:KV 块池(青,576 格) ----------------
POOL = (MX, 450, 640, 190)
lc.rect(*POOL, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.5)
lc.text(MX + 320, 472, 'L0『调度 · 显存账本』BlockPool(青)——每 token 写进块池的,就是蓝框那两个向量',
        9.5, lc.C_KV_S, 'middle', True, maxw=620, tag='pool:t')
gx0, gy0, gc_ = 100, 490, 5.0
for r in range(16):            # c^KV:512 格 = 32×16
    for c in range(32):
        lc.rect(gx0 + c * gc_, gy0 + r * gc_, gc_ - 0.8, gc_ - 0.8, '#67e8f9', 'none',
                rx=0.5, sw=0)
kx0, ky0 = gx0 + 32 * gc_ + 36, gy0 + 20
for r in range(8):             # k^R:64 格 = 8×8
    for c in range(8):
        lc.rect(kx0 + c * gc_, ky0 + r * gc_, gc_ - 0.8, gc_ - 0.8, '#a5f3fc', 'none',
                rx=0.5, sw=0)
lc.text(gx0 + 80, gy0 + 16 * gc_ + 16, 'c^KV:512 格', 8.5, lc.C_KV_S, 'middle', True, tag='pool:c1')
lc.text(kx0 + 20, ky0 + 8 * gc_ + 16, 'k^R:64 格', 8.5, lc.C_KV_S, 'middle', True, tag='pool:c2')
lc.text(MX + 320, POOL[1] + POOL[3] - 44, '合计 576 格 /(token · 层)—— MHA 若 32768 格的 1.75%,等效 2.25 组 GQA',
        9.5, lc.C_KV_S, 'middle', True, maxw=620, tag='pool:sum')
lc.text(MX + 320, POOL[1] + POOL[3] - 26, '(格数 = 元素数,逐格对应;1.75% = 576/32768,2.25 = (512+64)/(2×128))',
        7.8, lc.C_MUTE, 'middle', maxw=620, tag='pool:note')
# 蓝框 → 块池 引线
lc.seg(250, 322, 250, POOL[1], C_BLUE, 1.6, marker='std', dash=True)
lc.parrow([(620, 148), (688, 148), (688, POOL[1])], C_BLUE, 1.6, marker='std', dash=True)
lc.text(465, 430, '每 token 写入块池', 8, C_BLUE, 'middle', tag='pool:w')

# ---------------- 右列:toy 实测 + GQA 对照 + 缓存账 + vLLM 骨架 ----------------
RX0, RW = 730, 710
TY = 122
lc.rect(RX0, TY, RW, 178, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(RX0 + 355, TY + 22, '玩具实测(d=8, d_c=6, 2 头):同一 token 的潜向量唯一,头间却逐位不同',
        9.5, lc.C_TXT, 'middle', True, maxw=RW - 24, tag='toy:t')
lc.text(RX0 + 30, TY + 48, 'token0 的 c^KV(唯一底稿):', 8.5, lc.C_MUTE, 'start', tag='toy:l1')
lc.rect(RX0 + 190, TY + 36, 250, 22, '#eff6ff', C_BLUE, rx=4, sw=1.0)
lc.text(RX0 + 315, TY + 51, '[1.426, -0.392, -1.07, 0.542, -1.5, -0.355]', 8, C_BLUE_T,
        'middle', tag='toy:v0')
lc.text(RX0 + 30, TY + 86, 'head0 的 k(自己的放大镜):', 8.5, lc.C_MUTE, 'start', tag='toy:l2')
lc.rect(RX0 + 190, TY + 74, 250, 22, '#dcfce7', lc.C_GPU_S, rx=4, sw=1.0)
lc.text(RX0 + 315, TY + 89, '[-0.102, -0.651, -1.364, -0.821]', 8, lc.C_TXT, 'middle', tag='toy:v1')
lc.text(RX0 + 30, TY + 126, 'head1 的 k(另一只放大镜):', 8.5, lc.C_MUTE, 'start', tag='toy:l3')
lc.rect(RX0 + 190, TY + 114, 250, 22, '#dcfce7', lc.C_GPU_S, rx=4, sw=1.0)
lc.text(RX0 + 315, TY + 129, '[0.787, 0.057, 3.58, 2.17]', 8, lc.C_TXT, 'middle', tag='toy:v2')
lc.text(RX0 + 470, TY + 106, '头间 max diff = 4.944', 9.5, C_BLUE, 'start', True, tag='toy:diff')
lc.text(RX0 + 470, TY + 126, '(逐位不同——同底稿、不同投影)', 8, lc.C_MUTE, 'start',
        maxw=220, tag='toy:diffn')
lc.text(RX0 + 355, TY + 162, '『潜向量够用』不是代数保证而是训练事实:App D.2 里 MLA 质量反超 MHA(cache 仅 14% / 4%)',
        8, lc.C_MUTE, 'middle', maxw=RW - 24, tag='toy:fact')

GY = 316
lc.rect(RX0, GY, RW, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(RX0 + 355, GY + 20, '与 GQA 的分水岭:复制现成向量 vs 共享生成基底', 9.5, lc.C_TXT,
        'middle', True, tag='cmp:t')
lc.rect(RX0 + 20, GY + 34, 320, 46, '#f8fafc', '#cbd5e1', rx=6, sw=1.0)
lc.text(RX0 + 180, GY + 52, 'GQA:同组头 K 逐位相同(max diff = 0.0)', 8.5, lc.C_MUTE,
        'middle', tag='cmp:g1')
lc.text(RX0 + 180, GY + 70, '离散的头选择——存的是成品复印件', 8, lc.C_MUTE, 'middle', tag='cmp:g2')
lc.rect(RX0 + 370, GY + 34, 320, 46, '#ecfeff', lc.C_KV_S, rx=6, sw=1.1)
lc.text(RX0 + 530, GY + 52, 'MLA:头间逐位不同(max diff = 4.944)', 8.5, lc.C_KV_S,
        'middle', True, tag='cmp:m1')
lc.text(RX0 + 530, GY + 70, '连续的低秩子空间——存的是底稿本身', 8.5, lc.C_KV_S, 'middle', tag='cmp:m2')

AY = 426
lc.rect(RX0, AY, RW, 100, '#ffffff', lc.C_KV_S, rx=8, sw=1.4)
lc.text(RX0 + 355, AY + 22, '缓存账:576(512 + 64) vs MHA 32768 → 1.75%;等效 2.25 组 GQA',
        9.5, lc.C_KV_S, 'middle', True, maxw=RW - 24, tag='acc:t')
lc.text(RX0 + 355, AY + 46, 'd_c = 512 ≪ d_h·n_h = 16384(32 倍压缩)——『低秩』的字面义:全部头的 K/V 落在同一个至多 512 维的线性子空间',
        8.5, lc.C_TXT, 'middle', maxw=RW - 24, tag='acc:l1')
lc.text(RX0 + 355, AY + 66, '巧合提示:kv_b_proj 输出宽 32768 恰等于 MHA 每 token 缓存数——但方向相反,',
        8, lc.C_MUTE, 'middle', maxw=RW - 24, tag='acc:l2')
lc.text(RX0 + 355, AY + 82, '那是把潜向量升回逐头空间的参数乘法宽度,不进缓存', 8, lc.C_MUTE,
        'middle', tag='acc:l3')

VY = 538
lc.rect(RX0, VY, RW, 72, '#ffffff', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(RX0 + 355, VY + 20, 'vLLM 投影骨架:fused_qkv_a_proj 输出 [1536, 576] · kv_b_proj 512→32768([W^UK;W^UV] 按头拼接)',
        9, lc.C_GPU_S, 'middle', True, maxw=RW - 24, tag='vsk:t')
lc.text(RX0 + 355, VY + 40, '缓存的是 kv_a_layernorm 之后的形态(kv_c_normed)· 吸收形态与 prefill/decode 双路 = ch25(预告)',
        8, '#334155', 'middle', maxw=RW - 24, tag='vsk:l1')
lc.text(RX0 + 355, VY + 58, 'vllm/model_executor/models/deepseek_v2.py:L1010-L1066 · vllm/model_executor/layers/mla.py:L150-L226',
        7.5, lc.C_FAINT, 'middle', tag='vsk:a')

# ---------------- 页脚 ----------------
lc.text(MX, 672, '图例:蓝框 = 缓存项(论文 Fig.3 蓝框语义,仅 c^KV 与 k^R 两项)· 青 = L0 显存账本/块池 · 绿 = 投影/执行流',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 692, '公式出处 arXiv:2405.04434 §2.1.2 Eq.(9)-(11) / App C Eq.(41)(43)(44) · 数值取自论文忠实 NumPy 参考实现实跑(host,float64)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 708, 'vLLM 行号基线 v0.27.1(6e448d0ea)', 8.5, lc.C_FAINT, 'start', tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch24-fig-mla-latent-compression.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
