#!/usr/bin/env python3
"""ch25 机制图 · 分块上投影 + LSE 精确合并(figure ch25-fig-chunked-merge,模板 flow)

放大自 L0『模型层 MLA 框』MHA 腿内部的历史上下文处理——L2 站 12(_compute_prefill_context
分块循环 + merge_attn_states)。双层:上层=手算档三轮 LSE 合并(6 key 分 3 块,逐位==整块 softmax);
下层=实跑档 workspace 分块(context 150 → 64/64/22,现场上投影算完即弃)+ 字节对比条。

claim:上投影大张量永不整段物化:历史上下文按 workspace 容量切片(本例 64 token 一块,
150 分成 64/64/22),每块 gather 潜向量→现场上投影→算一块→LSE 记账,块间与 suffix 全部用
merge_attn_states 精确合并——分块只省显存、不损一位精度。

数字全部取自 figure spec 的 numbers(手算档 24.6212/2.3133、0.3348:0.6652→21.5470/3.4076、
0.3444:0.6556→11.9588/4.4735==整块 11.9588/4.4735 差 0.000001/0.000000 · 实跑档
workspace 64、3 块 64/64/22、3 次 merge、diff 0.000000 · 字节账 75497472 B vs 3221225472 B、
注释 144mb 实为 72 MiB · 定容公式 L1803-L1831)。坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 780
MX = 52
BXR = 1448
C_BLK = lc.C_GPU_S          # 块 = 绿(执行)
C_MRG = lc.C_BEAT_T         # 合并节点 = 深橙(记账/合并)
C_MRG_F = lc.C_BEAT_F
C_LAT = lc.C_API_S          # 潜向量/参照 = 蓝

# ---------------- 标题区 ----------------
lc.text(MX, 34, '灶台小也能炖整锅汤:workspace 分块上投影,LSE 记账精确合并——恒等式,非近似',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '历史潜向量按 workspace 容量切片,每块现场上投影、算完即弃;块间按 e^LSE 归一权重合并,与整块 softmax 逐位一致',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 12 内部(_compute_prefill_context)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 上层:手算档三轮合并 =================
lc.text(MX, 96, '上层 · 手算档:单头单 query,6 个 key 分 3 块(2+1+3),每步可心算', 10.5,
        lc.C_TXT, 'start', True, maxw=700, tag='lay1:t')
lc.text(1360, 96, 'V=1 · scale=1', 8.5, lc.C_MUTE, 'end', tag='lay1:p')

blocks = [
    ('块1', '[1.0, 2.0]', '[0.2689, 0.7311]', '24.6212 / 2.3133'),
    ('块2', '[3.0]', '[1.0]', '20.0000 / 3.0000'),
    ('块3', '[0.5, 0.2, 4.0]', '[0.0287, 0.0213, 0.9501]', '6.9214 / 4.0512'),
]
BY0, BW_, BH = 112, 250, 86
bx = [70, 350, 630]
for (nm, sc, pr, out), x in zip(blocks, bx):
    lc.rect(x, BY0, BW_, BH, '#ffffff', C_BLK, rx=7, sw=1.5)
    lc.text(x + BW_ / 2, BY0 + 19, nm + ':scores → probs', 9, C_BLK and '#166534', 'middle', True,
            maxw=BW_ - 12, tag='b:t' + nm)
    lc.text(x + BW_ / 2, BY0 + 38, sc + ' → ' + pr, 7.5, '#334155', 'middle', maxw=BW_ - 10,
            tag='b:s' + nm)
    lc.text(x + BW_ / 2, BY0 + 62, '块输出 / LSE:%s' % out, 8.5, lc.C_TXT, 'middle', True,
            maxw=BW_ - 10, tag='b:o' + nm)

# 合并节点1(块1+块2)
M1 = (180, 246, 210, 64)
lc.rect(*M1, C_MRG_F, C_MRG, rx=9, sw=1.8)
lc.text(M1[0] + M1[2] / 2, 266, 'merge ①:权重 0.3348 : 0.6652', 8.5, '#7c2d12', 'middle', True,
        maxw=M1[2] - 10, tag='m1:t')
lc.text(M1[0] + M1[2] / 2, 284, '→ 21.5470 / 3.4076', 9, '#7c2d12', 'middle', True, tag='m1:v')
lc.text(M1[0] + M1[2] / 2, 300, '(e^LSE 归一的热度权重)', 7.5, lc.C_MUTE, 'middle', tag='m1:n')
lc.parrow([(195, BY0 + BH), (195, 246)], C_MRG, 1.6, 'std')
lc.parrow([(475, BY0 + BH), (475, 262), (390, 262)], C_MRG, 1.6, 'std')

# 合并节点2(+块3)
M2 = (520, 246, 210, 64)
lc.rect(*M2, C_MRG_F, C_MRG, rx=9, sw=1.8)
lc.text(M2[0] + M2[2] / 2, 266, 'merge ②:权重 0.3444 : 0.6556', 8.5, '#7c2d12', 'middle', True,
        maxw=M2[2] - 10, tag='m2:t')
lc.text(M2[0] + M2[2] / 2, 284, '→ 11.9588 / 4.4735', 9, '#7c2d12', 'middle', True, tag='m2:v')
lc.text(M2[0] + M2[2] / 2, 300, '(块12 合块3)', 7.5, lc.C_MUTE, 'middle', tag='m2:n')
lc.parrow([(M1[0] + M1[2], 278), (520, 278)], C_MRG, 1.8, 'std')
lc.parrow([(755, BY0 + BH), (755, 262), (730, 262)], C_MRG, 1.6, 'std')

# 整块参照(右)
REF = (1030, 112, 400, 86)
lc.rect(*REF, '#eff6ff', C_LAT, rx=8, sw=2.0)
lc.text(REF[0] + REF[2] / 2, 134, '整块参照:6 个 key 放同一个 softmax', 9.5, C_LAT and '#1e40af',
        'middle', True, maxw=REF[2] - 12, tag='ref:t')
lc.text(REF[0] + REF[2] / 2, 156, '输出 / LSE = 11.9588 / 4.4735', 9.5, '#1e40af', 'middle', True,
        tag='ref:v')
lc.text(REF[0] + REF[2] / 2, 180, '与三轮合并逐位一致:输出差 0.000001 / LSE 差 0.000000', 8,
        '#1e40af', 'middle', maxw=REF[2] - 12, tag='ref:d')
# merge② → 参照 的等号
lc.rect(934, 252, 52, 52, '#eff6ff', C_LAT, rx=26, sw=2.0)
lc.text(960, 286, '=', 20, C_LAT and '#1e40af', 'middle', True, tag='eq')
lc.parrow([(M2[0] + M2[2], 278), (934, 278)], C_MRG, 1.8, 'std')
lc.parrow([(960, 252), (960, 155), (1030, 155)], C_MRG, 1.8, 'std')
lc.text(960, 316, 'softmax 分块合并是恒等式', 8.5, lc.C_MUTE, 'middle', tag='eq:n')
lc.text(960, 330, '全局分母 = Σ e^(LSE_i),不是近似', 8, lc.C_MUTE, 'middle', maxw=220, tag='eq:n2')

# ================= 下层:实跑档 workspace 分块 =================
lc.text(MX, 372, '下层 · 实跑档:context 150 个历史潜向量,workspace 64 token 的锅,分 64/64/22 三批现切现算', 10.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='lay2:t')
lc.text(1360, 372, '新 token M=140(suffix)', 8.5, lc.C_MUTE, 'end', tag='lay2:p')

# workspace 锅形框(圆角大框)
WK = (70, 392, 700, 150)
lc.rect(*WK, '#ffffff', C_BLK, rx=18, sw=2.0)
lc.text(WK[0] + 16, 414, 'workspace(64 token 定容:min(max(8×max_model_len, 4×max_num_seqs×block), 64×1024))', 8.5,
        '#166534', 'start', True, maxw=670, tag='wk:t')
chunks = [('块 ①:64 key', 190), ('块 ②:64 key', 415), ('块 ③:22 key', 640)]
for nm, cx in chunks:
    lc.rect(cx - 80, 432, 160, 62, '#f0fdf4', C_BLK, rx=6, sw=1.3)
    lc.text(cx, 452, nm, 9, '#166534', 'middle', True, tag='wk:' + nm[:3])
    lc.text(cx, 470, 'gather 潜向量', 7.5, '#334155', 'middle', tag='wk:g' + nm[:3])
    lc.text(cx, 486, '→ 现场上投影 → 算一块', 7.5, '#334155', 'middle', tag='wk:u' + nm[:3])
lc.seg(270, 463, 335, 463, C_BLK, 1.6, 'std')
lc.seg(495, 463, 560, 463, C_BLK, 1.6, 'std')
lc.text(WK[0] + WK[2] / 2, WK[1] + WK[3] - 12, '每块算完即弃——上投影大张量永不整段物化', 8.5,
        lc.C_MUTE, 'middle', tag='wk:n')

# 循环回环(块③ → 块①)
lc.parrow([(640, 494), (640, 516), (190, 516), (190, 494)], C_BLK, 1.4, 'std', dash=True)
lc.text(415, 510, '分块循环 iters=ceil(150/64)=3', 8, lc.C_MUTE, 'middle', tag='wk:loop')

# suffix 终合并
SG = (830, 412, 240, 110)
lc.rect(*SG, '#ffffff', C_BLK, rx=8, sw=1.6)
lc.text(SG[0] + SG[2] / 2, 434, 'suffix:140 新 token', 9.5, '#166534', 'middle', True, tag='sg:t')
lc.text(SG[0] + SG[2] / 2, 454, '本拍要算的新 query 段', 8, '#334155', 'middle', tag='sg:s')
lc.text(SG[0] + SG[2] / 2, 476, '块间 merge 共 2 次', 8, lc.C_MUTE, 'middle', tag='sg:l1')
lc.text(SG[0] + SG[2] / 2, 492, '+ context⊕suffix 终合并 1 次', 8, lc.C_MUTE, 'middle', tag='sg:l2')
# 终合并节点
FM = (1110, 412, 200, 110)
lc.rect(*FM, C_MRG_F, C_MRG, rx=9, sw=1.8)
lc.text(FM[0] + FM[2] / 2, 436, '末次块间 merge 的 LSE', 8.5, '#7c2d12', 'middle', True,
        maxw=FM[2] - 12, tag='fm:t')
lc.text(FM[0] + FM[2] / 2, 458, 'prefix 5.040', 9, '#7c2d12', 'middle', True, tag='fm:v1')
lc.text(FM[0] + FM[2] / 2, 476, 'suffix 3.394', 9, '#7c2d12', 'middle', True, tag='fm:v2')
lc.text(FM[0] + FM[2] / 2, 500, 'merge_attn_states ×3', 8.5, '#7c2d12', 'middle', True, tag='fm:v3')
lc.parrow([(WK[0] + WK[2], 463), (SG[0], 452)], C_BLK, 1.8, 'std')
lc.parrow([(SG[0] + SG[2], 467), (FM[0], 467)], C_BLK, 1.8, 'std')
lc.text(1032, 540, '输出与全量上投影参照 max diff 0.000000', 9, '#166534', 'middle', True, maxw=300,
        tag='fm:ok')

# ---------------- 字节账条(底) ----------------
BA = (70, 572, 1380, 108)
lc.rect(*BA, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.5)
lc.text(90, 594, '字节账(64k 生产档,fp16/bf16 口径):workspace 只装潜向量,上投影在 64-token 块内现场做', 10,
        lc.C_KV_S, 'start', True, maxw=900, tag='ba:t')
bar_y, bar_h = 610, 14
bx0, bw_max = 90, 300
lc.rect(bx0, bar_y, bw_max, bar_h, '#bfdbfe', C_LAT, rx=2, sw=1.0)
lc.text(bx0 + bw_max + 10, bar_y + 11, 'workspace 75497472 B(= 72 MiB,只装潜向量)', 8.5, '#1e40af',
        'start', maxw=380, tag='ba:ws')
w_up = 1000     # 全量上投影条(截断:真比例 42.7 倍画不下,右侧标注)
lc.rect(bx0, bar_y + 22, w_up, bar_h, '#fda4af', lc.C_ABORT, rx=2, sw=1.0)
lc.text(bx0 + w_up + 10, bar_y + 33, '全量上投影仅 K 就要 3221225472 B(3 GB)——条长截断示意,真比例远长于画布', 8.5,
        lc.C_ABORT, 'start', maxw=350, tag='ba:up')
lc.text(bx0 + 16, bar_y + 33, '截断示意', 7.5, lc.C_ABORT, 'start', tag='ba:cut')
lc.text(90, BA[1] + BA[3] - 12, '源码注释自称 144mb,按 2B×576×65536 精确积为 75497472 B(72 MiB)——口径恰差 2 倍,引用注释须带此勘误',
        8, lc.C_MUTE, 'start', maxw=1340, tag='ba:err')

# ---------------- 页脚 ----------------
lc.text(MX, 724, '图例:绿 = 块计算(执行) · 深橙 = LSE 合并/记账节点 · 蓝 = 参照/潜向量口径 · 青 = 字节账 · 上层=手算档,下层=实跑档',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 744, '数值 = host 实测(vLLM v0.27.1 精简版实跑,float32) · merge_attn_states = vllm/v1/attention/ops/merge_attn_states.py · 定容与循环 mla_attention.py:L1803-L1831 / L2301-L2422',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-chunked-merge.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
