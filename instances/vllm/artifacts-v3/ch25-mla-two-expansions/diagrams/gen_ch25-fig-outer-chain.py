#!/usr/bin/env python3
"""ch25 机制图 · 外层前向:一次融合 GEMM + 两刀解耦 split + RoPE 只打 rope 段
(figure ch25-fig-outer-chain,模板 tensor-flow)

放大自 L0『模型层 MLA 框』一拍前向的外层展开——L2 站 9(Wrapper.forward 低秩链+解耦 RoPE)。
主带走 DSV3 数字(hidden 7168 → 融合输出 2112 → q 潜 1536 / KV 576),MINI 括注;
RoPE 作用域画成虚线框只罩 rope 切片,nope 段原样直通(实测逐位相等)。

claim:一个 token 流过外层只做三件事:一次融合 GEMM 同时压 q(1536)与 KV(576=512 潜+64 位置);
两刀 split 把位置段解耦;RoPE 只旋转 q 与 k 各自尾部的 64 维 rope 段——nope 段 128 维
原样不动,这是后面吸收推导的数学前提。

数字全部取自 figure spec 的 numbers(fused_out_width 640/2112 · KV 段 576=512+64 ·
cache 账 576 vs 40960=71.1 倍 · RoPE 实证 pos0 0.0000/pos1 0.9517 · kv_a_layernorm
只打 kv_c diff 0.0000 · rotary_emb 调用点 mla.py:L201-L203)。坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 800
MX = 52
BXR = 1448
C_LAT = lc.C_API_S           # 蓝 = 潜向量/cache 条目(延续 ch24「蓝框=缓存项」语义,图例声明)
C_LAT_T = '#1e40af'
C_ROPE = lc.C_BEAT_T         # 深橙 = RoPE 作用域/位置段(图例声明)

# ---------------- 标题区 ----------------
lc.text(MX, 34, '外层前向三件事:融合 GEMM 一次压完 · 两刀 split 解耦位置 · RoPE 只打尾部 64 维',
        16, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, 'nope 段 128 维原样直通(实测与未旋转手工链逐位相等)——旋转对角阵与 W_UK 不可交换,缠了位置就没法再被吸收',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 9 的机制展开'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主带:hidden → 融合 GEMM → [2112] 分段条 ----------------
HB = (60, 330, 130, 66)
lc.rect(*HB, '#ffffff', lc.C_GPU_S, rx=7, sw=1.6)
lc.text(HB[0] + HB[2] / 2, 354, 'hidden_states', 10, lc.C_TXT, 'middle', True, tag='hb:t')
lc.text(HB[0] + HB[2] / 2, 372, '[S, 7168]', 9, lc.C_GPU_S, 'middle', True, tag='hb:s')
lc.text(HB[0] + HB[2] / 2, 388, '每 token 一行', 7.5, lc.C_MUTE, 'middle', tag='hb:n')

GB = (236, 318, 168, 90)     # 融合 GEMM 框(下投影:宽进窄出,标签注明)
lc.rect(*GB, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.6)
lc.text(GB[0] + GB[2] / 2, 348, 'fused_qkv_a_proj', 9.5, '#166534', 'middle', True, maxw=GB[2] - 10, tag='g:t')
lc.text(GB[0] + GB[2] / 2, 366, '[2112, 7168]', 8.5, lc.C_GPU_S, 'middle', True, tag='g:s')
lc.text(GB[0] + GB[2] / 2, 384, '一次 GEMM 压完 q 与 KV', 7.5, lc.C_MUTE, 'middle', maxw=GB[2] - 10, tag='g:n')
lc.seg(190, 363, 236, 363, lc.C_GPU_S, 1.8, 'std')

# 输出分段条(比例宽:q 潜 1536 | kv_c 512 | k_pe 64 → 总 2112)
OB = (452, 318, 500, 90)
wq = 500 * 1536 / 2112       # ≈364
wc = 500 * 512 / 2112        # ≈121
wp = 500 * 64 / 2112         # ≈15
lc.seg(GB[0] + GB[2], 363, OB[0], 363, lc.C_GPU_S, 1.8, 'std')
lc.rect(OB[0], OB[1], wq, OB[3], '#dcfce7', lc.C_GPU_S, rx=0, sw=1.2)
lc.text(OB[0] + wq / 2, 344, 'q 潜段 1536', 9.5, '#166534', 'middle', True, tag='ob:q')
lc.text(OB[0] + wq / 2, 362, '= W_DQ 输出', 7.5, '#166534', 'middle', tag='ob:qn')
lc.rect(OB[0] + wq, OB[1], wc, OB[3], '#cffafe', lc.C_KV_S, rx=0, sw=1.2)
lc.text(OB[0] + wq + wc / 2, 344, 'kv_c', 9.5, '#155e75', 'middle', True, tag='ob:c')
lc.text(OB[0] + wq + wc / 2, 362, '512', 8.5, '#155e75', 'middle', True, tag='ob:cn')
lc.rect(OB[0] + wq + wc, OB[1], wp, OB[3], '#ffedd5', C_ROPE, rx=0, sw=1.2)
lc.text(OB[0] + OB[2] / 2 + 120, 309, '融合输出 [S, 2112](MINI 640)', 9.5, lc.C_TXT, 'middle', True,
        maxw=260, tag='ob:t')
lc.text(OB[0] + OB[2] / 2, 424, '第二刀 split:kv_lora 段 576 切 kv_c | k_pe', 8.5, lc.C_KV_S,
        'middle', tag='ob:cut')
lc.text(OB[0] + wq + wc + wp / 2 + 4, 440, 'k_pe 64', 8, C_ROPE, 'start', tag='ob:p')

# ---------------- 上泳道:q 低秩链 ----------------
QY = 122
lc.text(310, QY - 10, 'q 低秩链(第一刀 split 出 q 潜段:MINI 64 / DSV3 1536)', 10, '#166534',
        'start', True, maxw=440, tag='q:t')
lc.rect(452, QY, 148, 56, '#ffffff', lc.C_GPU_S, rx=7, sw=1.5)
lc.text(526, QY + 24, 'q_a_layernorm', 10, lc.C_TXT, 'middle', True, maxw=138, tag='q:ln')
lc.text(526, QY + 43, '[1536]', 8.5, lc.C_GPU_S, 'middle', True, tag='qs:ln')
lc.rect(646, QY, 148, 56, '#ffffff', lc.C_GPU_S, rx=7, sw=1.5)
lc.text(720, QY + 24, 'q_b_proj', 10, lc.C_TXT, 'middle', True, maxw=138, tag='q:bp')
lc.text(720, QY + 43, '[24576, 1536]', 8.5, lc.C_GPU_S, 'middle', True, tag='qs:bp')
lc.seg(600, QY + 28, 646, QY + 28, lc.C_GPU_S, 1.8, 'std')
# 融合输出 q 段顶 → q_a_layernorm 底
lc.parrow([(OB[0] + wq / 2, OB[1]), (OB[0] + wq / 2, 240), (526, 240), (526, QY + 56)],
          lc.C_GPU_S, 1.8, 'std')

# view 出口:nope | rope 分段条(192 = 128 + 64,比例)
VB = (846, QY, 176, 56)
wn = 176 * 128 / 192
lc.rect(*VB, '#ffffff', lc.C_GPU_S, rx=0, sw=1.2)
lc.rect(VB[0], VB[1], wn, VB[3], '#dcfce7', 'none', rx=0, sw=0)
lc.text(VB[0] + wn / 2, QY + 22, 'nope 128', 8.5, '#166534', 'middle', True, tag='vb:n')
lc.text(VB[0] + wn / 2, QY + 40, '不旋转', 7.5, '#166534', 'middle', tag='vb:nn')
lc.rect(VB[0] + wn, VB[1], 176 - wn, VB[3], '#ffedd5', 'none', rx=0, sw=0)
lc.text(VB[0] + wn + (176 - wn) / 2, QY + 22, 'rope 64', 8, C_ROPE, 'middle', True, tag='vb:r')
lc.text(VB[0] + wn + (176 - wn) / 2, QY + 40, '旋转', 7.5, C_ROPE, 'middle', tag='vb:rn')
lc.seg(794, QY + 28, 846, QY + 28, lc.C_GPU_S, 1.8, 'std')
lc.text(VB[0] + VB[2] / 2, QY - 8, 'view [S, 128, 192](MINI [5, 4, 80])', 8.5, lc.C_MUTE, 'middle', tag='vb:t')

# nope 直通(绕过 RoPE 框,从上方)
NOPE_Y = 88
lc.parrow([(VB[0] + wn / 2, QY), (VB[0] + wn / 2, NOPE_Y), (1344, NOPE_Y), (1344, 216)],
          lc.C_GPU_S, 1.8, 'std')
lc.text(1124, NOPE_Y - 7, 'nope 段原样直通——实测与未旋转手工链逐位相等', 8.5, '#166534',
        'middle', maxw=430, tag='byp:t')
# rope 段进 RoPE 作用域(view 条 rope 段底 → RoPE 框顶)
lc.parrow([(VB[0] + wn + (176 - wn) / 2, QY + VB[3]), (VB[0] + wn + 29, 216)], C_ROPE, 1.6, 'std')

# ---------------- 下泳道:KV 链 ----------------
KY = 480
lc.text(310, KY - 10, 'KV 压缩链(归一化只打 kv_c)', 10, '#155e75', 'start', True, maxw=400, tag='kv:t')
lc.rect(452, KY, 148, 56, '#ffffff', lc.C_KV_S, rx=7, sw=1.5)
lc.text(526, KY + 22, 'kv_a_layernorm', 9.5, lc.C_TXT, 'middle', True, maxw=138, tag='kv:ln')
lc.text(526, KY + 39, '只打 kv_c(k_pe 不归一)', 7.8, '#155e75', 'middle', maxw=140, tag='kv:lns')
lc.text(526, KY + 51, 'max diff 0.0000', 7.5, lc.C_MUTE, 'middle', tag='kv:lnn')
# 分段条 kv_c 段底 → kv_a_layernorm 顶;k_pe 段底 → RoPE 框底
lc.parrow([(OB[0] + wq + wc / 2, OB[1] + OB[3]), (OB[0] + wq + wc / 2, 466), (526, 466), (526, KY)],
          lc.C_KV_S, 1.8, 'std')
lc.parrow([(OB[0] + wq + wc + wp / 2, OB[1] + OB[3]), (OB[0] + wq + wc + wp / 2, 452),
           (1052, 452), (1052, 316)], C_ROPE, 1.8, 'std')
lc.text(770, 446, 'k_pe 单独走:不归一、不压缩、只旋转', 8.5, C_ROPE, 'middle', tag='kv:pe')
lc.text(1058, 430, 'k_pe [S, 1, 64]', 8, C_ROPE, 'start', tag='rp:kin')

# kv_c_normed 条(蓝)
KB2 = (652, KY, 142, 56)
lc.rect(*KB2, '#eff6ff', C_LAT, rx=7, sw=2.0)
lc.text(KB2[0] + KB2[2] / 2, KY + 22, 'kv_c_normed', 9.5, C_LAT_T, 'middle', True, tag='kvc:t')
lc.text(KB2[0] + KB2[2] / 2, KY + 41, '[S, 512]', 8.5, C_LAT_T, 'middle', tag='kvc:s')
lc.seg(600, KY + 28, 652, KY + 28, lc.C_KV_S, 1.8, 'std')

# ---------------- RoPE 作用域(虚线框) ----------------
RB = (900, 216, 280, 100)
lc.rect(*RB, 'none', C_ROPE, rx=12, sw=1.8, dash=True)
lc.text(RB[0] + RB[2] / 2, 236, 'RoPE 作用域——只旋转 rope 段', 10, C_ROPE, 'middle', True, tag='rp:t')
lc.text(RB[0] + RB[2] / 2, 254, 'rotary_emb(positions, q[..., P:], k_pe)', 8.5, C_ROPE, 'middle',
        tag='rp:call')
lc.text(RB[0] + RB[2] / 2, 274, 'vllm/model_executor/layers/mla.py:L201-L203', 7.5, lc.C_FAINT,
        'middle', tag='rp:f')
lc.text(RB[0] + RB[2] / 2, 294, '实证:pos=0 差 0.0000(恒等) · pos=1 差 0.9517(旋转生效)', 8,
        lc.C_MUTE, 'middle', maxw=RB[2] - 12, tag='rp:ev')
lc.text(962, 208, 'q[..., 128:]', 8, C_ROPE, 'start', tag='rp:qin')
# RoPE 出口 → 内层左边
lc.seg(RB[0] + RB[2], 266, 1240, 266, C_ROPE, 1.8, 'std')

# ---------------- 内层插座 + cache 写腿 ----------------
IB = (1240, 216, 208, 200)
lc.rect(*IB, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(IB[0] + IB[2] / 2, 240, '内层 mla_attn', 11, '#166534', 'middle', True, tag='in:t')
lc.text(IB[0] + IB[2] / 2, 258, 'mla_attn(q, kv_c_normed, k_pe)', 8, '#334155', 'middle',
        maxw=IB[2] - 12, tag='in:call')
lc.text(IB[0] + IB[2] / 2, 282, 'q [S, 128, 192]', 8, lc.C_GPU_S, 'middle', tag='in:q')
lc.text(IB[0] + IB[2] / 2, 298, 'kv_c_normed [S, 512]', 8, C_LAT_T, 'middle', tag='in:c')
lc.text(IB[0] + IB[2] / 2, 314, 'k_pe [S, 1, 64]', 8, C_ROPE, 'middle', tag='in:p')
lc.rect(IB[0] + 14, 330, IB[2] - 28, 40, '#eff6ff', C_LAT, rx=6, sw=1.8)
lc.text(IB[0] + IB[2] / 2, 346, 'cache 写腿输入', 8.5, C_LAT_T, 'middle', True, tag='cw:t')
lc.text(IB[0] + IB[2] / 2, 362, '潜向量拼接 576 = kv_c_normed ⊕ k_pe', 7.8, C_LAT_T, 'middle',
        maxw=IB[2] - 20, tag='cw:s')
lc.text(IB[0] + IB[2] / 2, 392, 'cache 里只有潜向量', 8, C_LAT_T, 'middle', True, tag='cw:n')
lc.text(IB[0] + IB[2] / 2, 406, '无任何完整 K/V', 8, C_LAT_T, 'middle', tag='cw:n2')
# kv_c_normed → 内层底
lc.parrow([(KB2[0] + KB2[2], KY + 28), (1360, KY + 28), (1360, IB[1] + IB[3])], C_LAT, 1.8, 'std')

# ---------------- 右下:cache 行账 ----------------
AB = (452, 566, 700, 96)
lc.rect(*AB, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.5)
lc.text(AB[0] + AB[2] / 2, 586, '每 token 每层 cache 行账:潜向量 576 元素 = 512 + 64', 10, lc.C_KV_S,
        'middle', True, tag='ac:t')
bar_y, bar_h = 602, 13
bx0, bw_max = AB[0] + 24, 400
w_lat = max(bw_max * 576 / 40960, 3)     # 潜向量条(严格比例)
lc.rect(bx0, bar_y, bw_max, bar_h, '#fda4af', lc.C_ABORT, rx=2, sw=1.0)
lc.text(bx0 + bw_max + 10, bar_y + 10, 'MHA 等价 40960 = 128×(192+128) 元素', 8.5, lc.C_ABORT,
        'start', tag='ac:mha')
lc.rect(bx0, bar_y + 21, w_lat, bar_h, '#bfdbfe', C_LAT, rx=2, sw=1.0)
lc.text(bx0 + w_lat + 10, bar_y + 31, '潜向量 576——71.1 倍;对照 GQA(Llama-70B)2048 元素约 3.6 倍节省',
        8.5, C_LAT_T, 'start', maxw=340, tag='ac:lat')
lc.text(AB[0] + AB[2] / 2, AB[1] + AB[3] - 10, '(q 侧低秩 1536 是算力/参数账,不进 cache——cache 只省 KV 侧)', 8,
        lc.C_MUTE, 'middle', tag='ac:n')

# ---------------- 页脚 ----------------
lc.text(MX, 700, '图例:绿 = 投影/执行流(GPU 执行臂角色) · 青 = KV 压缩链 · 蓝 = 潜向量/cache 条目(延续 ch24 蓝框语义) · 深橙 = RoPE 作用域/位置段',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 720, '数值 = host 实测(vLLM v0.27.1 精简版实跑,float32;MINI 档保留 576/64 实尺) · 行号基线 v0.27.1(6e448d0ea)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 740, 'DSV3 实尺:7168 / 1536 / 128 头 / 128 / 64 / 512 / 128——正文数字取 DSV3 行,MINI 仅括注', 8.5,
        lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-outer-chain.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
