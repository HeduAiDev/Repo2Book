#!/usr/bin/env python3
"""ch25 机制图 · MHA 展开腿:潜向量一次性上投影成完整 K/V(figure ch25-fig-mha-leg,模板 tensor-flow)

放大自 L0『模型层 MLA 框』一拍前向的 MHA 腿展开——L2 站 12(forward_mha 上投影)。
单向带:潜向量窄条 → kv_b_proj → [140,4,32] 宽条 → 劈 k_nope/v → k_pe 广播拼尾 → 标准 MHA;
底部元素账条直观呈现 576 → 40960 的放大(DSV3 口径)。

claim:prefill 腿把 576 维潜向量经 kv_b_proj 一次性上投影成 128 头完整 K/V(DSV3 每 token
40960 元素、71 倍于潜向量),拼上共享的 rope 段后交给标准 MHA kernel——算力支出换
『通用后端都能算』的形状。

数字全部取自 figure spec 的 numbers(形状链 [140,512]→[140,4,32]→[140,4,16]×2→[140,4,80] ·
DSV3 每 token 40960 vs 576(71 倍)与 kv_b_proj [32768,512] · max diff 0.000000 ·
M=140 > 阈值 128)。坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 760
MX = 52
BXR = 1448
C_LAT = lc.C_API_S
C_LAT_T = '#1e40af'
C_ROPE = lc.C_BEAT_T

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'MHA 展开腿:576 维潜向量一次性上投影成 128 头完整 K/V,再按标准 MHA 算',
        16, lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, '算力换形状:DSV3 每 token 上投影 40960 元素、71 倍于潜向量——prefill 的 Sq/Skv≈1 让每份 K/V 被整段 query 摊薄,换任何通用后端都能算',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 12 的机制展开'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 门槛盒(左上) ----------------
GB = (60, 100, 330, 96)
lc.rect(*GB, '#fffbeb', '#d97706', rx=8, sw=1.5)
lc.text(GB[0] + GB[2] / 2, 122, '走这条腿的门:M=140 > 128', 10.5, '#92400e', 'middle', True, tag='gate:t')
lc.text(GB[0] + GB[2] / 2, 142, '128 = FlashMLA 自报的 reorder_batch_threshold', 8, '#92400e',
        'middle', maxw=GB[2] - 16, tag='gate:l1')
lc.text(GB[0] + GB[2] / 2, 158, '注释原话:process small prefills with decode pathway', 8, lc.C_MUTE,
        'middle', maxw=GB[2] - 16, tag='gate:l2')
lc.text(GB[0] + GB[2] / 2, 176, '≤128 的小段划进 MQA(下一张图)', 8.5, '#92400e', 'middle', True,
        tag='gate:l3')
lc.text(GB[0] + GB[2] / 2, 190, '本例 fresh prefill:num_decode_tokens=0 整批走 MHA', 7.5, lc.C_MUTE,
        'middle', maxw=GB[2] - 16, tag='gate:l4')

# ---------------- 主带 ----------------
MY = 300     # 主带中线
# 潜向量窄条(蓝)+ k_pe 小条
LB = (60, 258, 46, 84)
lc.rect(*LB, '#eff6ff', C_LAT, rx=5, sw=2.2)
lc.text(LB[0] + LB[2] / 2, 348, 'kv_c_normed', 9, C_LAT_T, 'middle', True, tag='lat:t')
lc.text(LB[0] + LB[2] / 2, 363, '[140, 512]', 8.5, C_LAT_T, 'middle', True, tag='lat:s')
lc.text(LB[0] + LB[2] / 2, 378, '潜向量', 8, lc.C_MUTE, 'middle', tag='lat:n')
PB = (60, 430, 46, 30)
lc.rect(*PB, '#ffedd5', C_ROPE, rx=5, sw=1.8)
lc.text(PB[0] + PB[2] / 2, 449, 'k_pe', 7.5, '#92400e', 'middle', True, tag='pe:t')
lc.text(PB[0] + PB[2] / 2, 424, 'k_pe [140, 1, 64]', 8.5, C_ROPE, 'middle', tag='pe:s')

lc.seg(106, 300, 152, 300, C_LAT, 1.8, 'std')

# kv_b_proj 框(上投影:窄进宽出)
UB = (152, 258, 170, 84)
lc.rect(*UB, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.8)
lc.text(UB[0] + UB[2] / 2, 288, 'kv_b_proj', 11, '#166534', 'middle', True, tag='up:t')
lc.text(UB[0] + UB[2] / 2, 306, 'DSV3 [32768, 512]', 8.5, lc.C_GPU_S, 'middle', True, tag='up:s')
lc.text(UB[0] + UB[2] / 2, 324, '(MINI [128, 512])', 7.5, lc.C_MUTE, 'middle', tag='up:n')
lc.seg(UB[0] + UB[2], 300, 366, 300, lc.C_GPU_S, 2.0, 'std')

# 上投影宽条 [140,4,32](32 = nope 16 + v 16,双色)
WB_ = (366, 258, 250, 84)
wn = 125      # nope 半段宽
lc.rect(*WB_, '#ffffff', lc.C_GPU_S, rx=0, sw=1.4)
lc.rect(WB_[0], WB_[1], wn, WB_[3], '#dcfce7', 'none', rx=0, sw=0)
lc.text(WB_[0] + wn / 2, 288, 'K 半段 16', 8.5, '#166534', 'middle', True, tag='w:n')
lc.text(WB_[0] + wn / 2, 306, '(k_nope)', 7.5, '#166534', 'middle', tag='w:nn')
lc.rect(WB_[0] + wn, WB_[1], 250 - wn, WB_[3], '#f0fdf4', 'none', rx=0, sw=0)
lc.text(WB_[0] + wn + (250 - wn) / 2, 288, 'V 半段 16', 8.5, '#166534', 'middle', True, tag='w:v')
lc.text(WB_[0] + wn + (250 - wn) / 2, 306, '(v)', 7.5, '#166534', 'middle', tag='w:vn')
lc.text(WB_[0] + WB_[2] / 2, 246, '上投影 kv_nope [140, 4, 32]——同一 GEMM 出 K/V 两半', 9, lc.C_TXT,
        'middle', True, maxw=340, tag='w:t')
lc.text(WB_[0] + WB_[2] / 2, 356, 'view 成每头两半后 split(dim=-1)', 8, lc.C_MUTE, 'middle', tag='w:cut')

# split 两片
SPB = [(406, 396, 170, 46, 'k_nope [140, 4, 16]', '#dcfce7'), (596, 396, 170, 46, 'v [140, 4, 16]', '#f0fdf4')]
for x, y, w_, h_, nm, fl in SPB:
    lc.rect(x, y, w_, h_, fl, lc.C_GPU_S, rx=6, sw=1.3)
    lc.text(x + w_ / 2, y + 28, nm, 9, '#166534', 'middle', True, tag='sp:' + nm[:6])
lc.parrow([(491, WB_[1] + WB_[3]), (491, 396)], lc.C_GPU_S, 1.8, 'std')
lc.parrow([(681, WB_[1] + WB_[3]), (681, 396)], lc.C_GPU_S, 1.8, 'std')

# k_pe 广播拼尾:k_pe → concat
CB = (820, 396, 150, 46)
lc.rect(*CB, '#ffedd5', C_ROPE, rx=6, sw=1.3)
lc.text(CB[0] + CB[2] / 2, 418, '_concat_k_nope_k_pe', 8.5, C_ROPE, 'middle', True, maxw=CB[2] - 10,
        tag='cc:t')
lc.text(CB[0] + CB[2] / 2, 434, 'k_pe expand 到头维', 7.5, '#92400e', 'middle', tag='cc:s')
lc.parrow([(491, 442), (491, 480), (895, 480), (895, 442)], lc.C_GPU_S, 1.8, 'std')   # k_nope 底出绕行进 concat
lc.parrow([(83, 460), (83, 528), (880, 528), (880, 442)],
          C_ROPE, 1.6, 'std', dash=True)                              # k_pe 广播绕行进 concat 底
lc.text(520, 516, 'k_pe 全头共用同一段位置键(broadcast)', 8, C_ROPE, 'middle', tag='cc:bc')

# 标准注意力块
AB = (1020, 258, 210, 130)
lc.rect(*AB, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(AB[0] + AB[2] / 2, 282, '标准 MHA(无特形)', 11, '#166534', 'middle', True, tag='at:t')
lc.text(AB[0] + AB[2] / 2, 302, 'prefill_backend', 8.5, '#334155', 'middle', tag='at:b')
lc.text(AB[0] + AB[2] / 2, 316, '.run_prefill_new_tokens', 8.5, '#334155', 'middle', tag='at:b2')
lc.text(AB[0] + AB[2] / 2, 338, 'q [140, 4, 80] ← 外层送入', 8.5, lc.C_GPU_S, 'middle', tag='at:q')
lc.text(AB[0] + AB[2] / 2, 354, 'k [140, 4, 80]', 8.5, lc.C_GPU_S, 'middle', tag='at:k')
lc.text(AB[0] + AB[2] / 2, 370, 'v [140, 4, 16]', 8.5, lc.C_GPU_S, 'middle', tag='at:v')
lc.parrow([(CB[0] + CB[2], 419), (980, 419), (980, 340), (1020, 340)], lc.C_GPU_S, 1.8, 'std')
lc.text(1000, 388, 'k [140, 4, 80]', 8, '#166534', 'end', tag='at:kin')
lc.parrow([(766, 419 - 23), (766, 236), (1125, 236), (1125, AB[1])], lc.C_GPU_S, 1.8, 'std')
lc.text(950, 230, 'v 直接送标准后端', 8, '#166534', 'middle', tag='at:vin')

# 出口 → (与 MQA 腿殊途同归注)
EB = (1020, 430, 210, 58)
lc.rect(*EB, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(EB[0] + EB[2] / 2, 452, 'output [140, 128]', 9, lc.C_TXT, 'middle', True, tag='out:t')
lc.text(EB[0] + EB[2] / 2, 470, '回 hidden 维,交 o_proj', 8, lc.C_MUTE, 'middle', tag='out:s')
lc.seg(AB[0] + AB[2] / 2, AB[1] + AB[3], AB[0] + AB[2] / 2, EB[1], lc.C_GPU_S, 1.8, 'std')

# 验证注(右上)
VB = (1020, 100, 380, 76)
lc.rect(*VB, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(VB[0] + VB[2] / 2, 122, '与上投影 MHA 参照逐元素对照:max diff 0.000000', 9.5, '#166534',
        'middle', True, maxw=VB[2] - 16, tag='v:t')
lc.text(VB[0] + VB[2] / 2, 142, '参照 = 文件头 Compute Friendly 伪码(mla_attention.py:L44-L118)逐式', 8,
        lc.C_MUTE, 'middle', maxw=VB[2] - 16, tag='v:l1')
lc.text(VB[0] + VB[2] / 2, 158, '上投影不增不减:只是 kv_c → K/V 的线性重建', 8, lc.C_MUTE, 'middle',
        tag='v:l2')

# ---------------- 底部:元素账(比例条) ----------------
AC = (60, 566, 1380, 116)
lc.rect(*AC, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.5)
lc.text(AC[0] + 20, 588, '元素账(DSV3 口径):每 token 上投影 K+V = 128×(128+64)+128×128 = 40960 元素,潜向量 576——71 倍',
        10, lc.C_KV_S, 'start', True, maxw=1100, tag='ac:t')
bar_y, bar_h = 606, 15
bx0 = AC[0] + 20
bw_max = 900
w_lat = max(bw_max * 576 / 40960, 3)
lc.rect(bx0, bar_y, bw_max, bar_h, '#bbf7d0', lc.C_GPU_S, rx=2, sw=1.0)
lc.text(bx0 + bw_max + 10, bar_y + 12, '上投影 K+V 40960 元素 / token(kv_b_proj [32768, 512] 每层常驻)',
        8.5, '#166534', 'start', maxw=440, tag='ac:up')
lc.rect(bx0, bar_y + 23, w_lat, bar_h, '#bfdbfe', C_LAT, rx=2, sw=1.0)
lc.text(bx0 + w_lat + 10, bar_y + 35, '潜向量 576 元素 / token(条宽按 71:1 严格比例)', 8.5, C_LAT_T,
        'start', maxw=400, tag='ac:lat')
lc.text(AC[0] + 20, AC[1] + AC[3] - 12, 'MINI 缩维档 384 < 576 是伪象(N/P/V 缩了、L 未缩)——大小对比一律取本行 DSV3 账;每份上投影 K/V 被 Sq=140 个 query 复用摊薄',
        8, lc.C_MUTE, 'start', maxw=1340, tag='ac:n')

# ---------------- 页脚 ----------------
lc.text(MX, 716, '图例:绿 = 执行流(GPU 执行臂角色) · 蓝 = 潜向量(延续 ch24 蓝框语义) · 深橙 = RoPE 位置段 · 青框 = 元素账',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 736, '形状/数值 = host 实测(vLLM v0.27.1 精简版实跑,float32) · 阈值 128 = vllm/v1/attention/backends/mla/flashmla.py:L121 · 行号基线 v0.27.1(6e448d0ea)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-mha-leg.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
