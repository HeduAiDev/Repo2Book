#!/usr/bin/env python3
"""ch26 机制图 · indexer 独立小头的解剖现场(figure ch26-fig-indexer-head,模板 layout)

放大自 L0『模型层 indexer 框』——L2 拍片② 装配·小头+IndexCache(站 3)的类型展开:
左=主 MLA(128 头×576 维,ch25 主角)与右=indexer 小头(64 头×128 维)并排,
头表/权重/归一化/RoPE/缓存五件事全部独立;底部=三标量折进 weights + FP8 wk 融合加载。

claim:indexer 是一套与主注意力完全独立的打分头:头表全来自 config.index_*(64 头×128 维,
与主 128 头无关)、wq_b 从 1536 维潜向量上投且复制不切 TP、wk_weights_proj 一枪 GEMM 出
128 维 key 与 64 个逐头权重、key 经 LayerNorm(eps=1e-6) 与专属 RoPE(默认 NeoX 半分式) 后以 132B
量化条目进 IndexCache——主 MLA 一个头都没动用。

数字全部取自 figure spec 的 numbers(wq_b [8192,1536] / wk_weights_proj [192,7168]
output_sizes [128,64] / 64 vs 128 头 / eps=1e-6·softmax_scale 0.0884·n_head_scale 0.125 /
k_cache 132B·workspace 6553600 / FP8 wk+BF16 weights_proj 融合加载——traces 实测 + 源码)。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 900
MX = 52
BXR = 1448

# ---------------- 标题区 ----------------
lc.text(MX, 34, '『独立小头』的解剖现场:头表、权重、归一化、RoPE、缓存——五件事与主 MLA 互不相干',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '右边的 indexer 打分头(64 头×128 维)全件装配一次成型:q_c 经 wq_b 上投、hidden 一枪 GEMM 出 k 与逐头权重——主 MLA(128 头×576 维)一个头都没动用',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0『模型层 indexer 框』· L2 拍片② 装配·小头+IndexCache(站 3)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 共享输入(两个芯片,箭头进两框) ----------------
IN_Y, IN_H = 84, 40
QC = (520, IN_Y, 300, IN_H)      # q_c 芯片
HD = (900, IN_Y, 320, IN_H)      # hidden 芯片
for (x, y, w, h), t1, t2 in [
    (QC, 'q_c [num_tokens, 1536]', 'q 低秩瓶颈——与主 query 共享(ch25 站 9)'),
    (HD, 'hidden_states [num_tokens, 7168]', '本层输入 hidden 流'),
]:
    lc.rect(x, y, w, h, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
    lc.text(x + w / 2, y + 17, t1, 10, lc.C_TXT, 'middle', True, maxw=w - 12, tag='in:' + t1[:8])
    lc.text(x + w / 2, y + 33, t2, 8.5, lc.C_MUTE, 'middle', maxw=w - 12, tag='in:s' + t1[:8])

# ---------------- 左右两大框 + 中缝五维对照 ----------------
LB = (MX, 168, 540, 470)          # 主 MLA 框
RB = (BXR - 540, 168, 540, 470)   # indexer 框
MIDX = (LB[0] + LB[2] + RB[0]) / 2   # 中缝中线

lc.rect(*LB, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.8)
lc.text(LB[0] + 16, LB[1] + 24, '主 MLA(ch25 的主角)', 12, lc.C_GPU_S, 'start', True, tag='lb:t')
lc.text(LB[0] + LB[2] - 14, LB[1] + 24, '128 头 × 576 维', 9.5, lc.C_MUTE, 'end', tag='lb:h')

lc.rect(*RB, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.2)
_bw = 16 + 11 * len('FP8 · 头数减半')
lc.rect(RB[0] + RB[2] - _bw - 8, RB[1] + 6, _bw, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
lc.text(RB[0] + RB[2] - _bw / 2 - 8, RB[1] + 20, 'FP8 · 头数减半', 9.5, lc.C_ENG_S, 'middle', True)
lc.text(RB[0] + 16, RB[1] + 24, 'indexer 打分小头(本章)', 12, lc.C_GPU_S, 'start', True, tag='rb:t')
lc.text(RB[0] + RB[2] - 16 - _bw - 12, RB[1] + 24, '64 头 × 128 维', 9.5, lc.C_MUTE, 'end', tag='rb:h')

# 五维行(左右同 y,中缝标签)——行内容
rows = [
    ('头表', ['128 头 × 每头 576 维', '(512 nope + 64 rope)'],
     ['64 头 × 每头 128 维(rope 段 64)', '全来自 config.index_* 前缀——与主头表无关']),
    ('权重', ['q_b_proj / kv_b_proj 等六积木', '(ch25 站 3 的装配现场)'],
     ['wq_b [8192, 1536] ReplicatedLinear', 'wk_weights_proj [192, 7168] 一枪 GEMM',
      '『no tensor parallel, just replicated』——复制不切 TP']),
    ('归一化', ['q_a / kv_a RMSNorm(字段名 layernorm 名不副实)', '(q 低秩链 + KV 压缩链各一)'],
     ['k_norm = LayerNorm(128, eps=1e-6)', '只打 indexer 自己的 k']),
    ('RoPE', ['GPT-J 交错式·打 q/k 的 rope 段(末 64 维)', '(is_neox_style=False,相邻维两两成对)'],
     ['专属 RoPE·默认 NeoX 半分式(is_neox 取反)', '只打前 64 维——与主 RoPE 相反']),
    ('缓存', ['576 元素潜向量 bf16', '= 1152 B/token/layer'],
     ['IndexCache 132 B/条(每 token 每层)', '128B fp8 值 + 4B fp32 scale(ue8m0)',
      '量化+缓存插入一步融合(indexer_k_quant_and_cache)']),
]
RY0, RH, RGAP = 210, 78, 8
for i, (dim, ll, rr) in enumerate(rows):
    ry = RY0 + i * (RH + RGAP)
    # 左行块
    lc.rect(LB[0] + 14, ry, LB[2] - 28, RH, '#ffffff', lc.C_GPU_S, rx=7, sw=1.1)
    for j, s in enumerate(ll):
        lc.text(LB[0] + 28, ry + 22 + j * 17, s, 9.5 if j == 0 else 8.5,
                lc.C_TXT if j == 0 else '#475569', 'start',
                bold=(j == 0), maxw=LB[2] - 56, tag='l%d:%d' % (i, j))
    # 右行块
    lc.rect(RB[0] + 14, ry, RB[2] - 28, RH, '#ffffff', lc.C_GPU_S, rx=7, sw=1.1)
    for j, s in enumerate(rr):
        lc.text(RB[0] + 28, ry + 22 + j * 17, s, 9.5 if j == 0 else 8.5,
                lc.C_TXT if j == 0 else '#475569', 'start',
                bold=(j == 0), maxw=RB[2] - 56, tag='r%d:%d' % (i, j))
    # 中缝维度标签
    tw_ = lc.tw(dim, 10, True) + 22
    lc.rect(MIDX - tw_ / 2, ry + RH / 2 - 13, tw_, 26, '#ffffff', lc.C_MUTE, rx=12, sw=1.1)
    lc.text(MIDX, ry + RH / 2 + 4, dim, 10, lc.C_TXT, 'middle', True, maxw=tw_ - 6, tag='dim:' + dim)
    lc.text(MIDX, ry + RH / 2 - 20, '≠', 11, lc.C_ABORT, 'middle', True, tag='neq')

lc.text(MIDX, 192, '五件事全部独立', 10.5, lc.C_MUTE, 'middle', True, tag='gutter:t')

# 字节条(缓存行内的视觉对比:宽 vs 窄)
cy = RY0 + 4 * (RH + RGAP) + RH - 14
lc.rect(LB[0] + 28, cy, 484, 9, lc.C_KV_F, lc.C_KV_S, rx=2, sw=1.0)
lc.text(LB[0] + 28, cy - 3, '1152B', 8, lc.C_KV_S, 'start', tag='bb:main')
_nw = int(484 * 132 / 1152)
lc.rect(RB[0] + 28, cy, _nw, 9, lc.C_KV_F, lc.C_KV_S, rx=2, sw=1.0)
lc.text(RB[0] + 28 + _nw + 6, cy + 7, '132B = 11.46%', 8, lc.C_KV_S, 'start', tag='bb:indexer')

# ---------------- 输入箭头(芯片 → 两框顶边) ----------------
# 先画全部连线、后发标注:两条 y=150 横线(实线 q_c 支路 / 灰虚线 hidden 支路)会从
# 字形中部穿过三处标注(盲审 2026-09-14 点名『wq_b(qr) 上投』『fused_qkv_a_proj』,
# 同族还有实线横穿『q_b_proj』)——被穿标注启用白 halo:白描边同字先铺、彩字后绘,
# 连线在字形处断开;标注必须在连线之后发射才有遮盖序(SVG 文档序=绘制序)。
LT, RT = LB[1], RB[1]
# q_c → 左框(q_b_proj 消费) / 右框(wq_b 消费)
lc.parrow([(QC[0] + 60, QC[1] + QC[3]), (QC[0] + 60, 150), (LB[0] + 260, 150), (LB[0] + 260, LT)],
          lc.C_MUTE, 1.6, 'std')
lc.seg(QC[0] + 240, QC[1] + QC[3], QC[0] + 240, RT, lc.C_GPU_S, 1.8, 'std')
# hidden → 左框 / 右框
lc.parrow([(HD[0] + 40, HD[1] + HD[3]), (HD[0] + 40, 150), (LB[0] + 420, 150), (LB[0] + 420, LT)],
          lc.C_MUTE, 1.6, 'std', dash=True)
lc.seg(HD[0] + 260, HD[1] + HD[3], HD[0] + 260, RT, lc.C_GPU_S, 1.8, 'std')
lc.text(QC[0] + 248, QC[1] + QC[3] + 30, 'wq_b(qr) 上投 64×128', 8.5, lc.C_GPU_S, 'start',
        tag='a:qcb', halo=True)
lc.text(HD[0] + 268, HD[1] + HD[3] + 30, '一枪 GEMM:k(128)+w(64)', 8.5, lc.C_GPU_S, 'start', tag='a:hdb')
lc.text(QC[0] - 8, 150, 'q_b_proj', 8, lc.C_MUTE, 'end', tag='a:qca', halo=True)
lc.text(HD[0] + 32, 150, 'fused_qkv_a_proj(ch25)', 8, lc.C_MUTE, 'start', tag='a:hda', halo=True)

# ---------------- 底部两卡:标量折叠(右,接 indexer 框)+ FP8 wk 融合加载(左) ----------------
BY = 668
B1 = (BXR - 660, BY, 660, 108)
B2 = (MX, BY, 660, 108)
lc.rect(*B1, '#ffffff', lc.C_GPU_S, rx=9, sw=1.6)
lc.text(B1[0] + 16, BY + 22, '进打分核之前:三个标量全部折进 weights', 11, lc.C_GPU_S, 'start', True,
        maxw=520, tag='b1:t')
lc.text(B1[0] + 16, BY + 44, 'weights = raw_w · q_scale · softmax_scale(0.0884) · n_head_scale(0.125)',
        9.5, lc.C_TXT, 'start', True, maxw=B1[2] - 30, tag='b1:f')
lc.text(B1[0] + 16, BY + 63, 'q per-token-group FP8 量化(128 一组、ue8m0)——q_scale 是 2 的幂,折叠不改变打分值',
        9, '#334155', 'start', maxw=B1[2] - 30, tag='b1:l1')
lc.text(B1[0] + 16, BY + 81, '打分核里只剩『点积 + ReLU + 加权和』(deepseek_v2.py:L817)',
        9, '#334155', 'start', maxw=B1[2] - 30, tag='b1:l2')

lc.rect(*B2, '#ffffff', lc.C_MUTE, rx=9, sw=1.6)
lc.text(B2[0] + 16, BY + 22, '加载侧痕迹:wk 与 weights_proj 分家的 checkpoint', 11, lc.C_TXT, 'start', True,
        maxw=520, tag='b2:t')
lc.text(B2[0] + 16, BY + 44, 'FP8 wk 权重 + BF16 weights_proj —— 两段缓冲、到齐才反量化 bf16 融合进 wk_weights_proj',
        9.5, lc.C_TXT, 'start', maxw=B2[2] - 30, tag='b2:f')
lc.text(B2[0] + 16, BY + 63, '(deepseek_v2.py:L822-L871,融合要在加载期维持:FP8 wk upcast BF16)',
        9, '#334155', 'start', maxw=B2[2] - 30, tag='b2:l1')
lc.text(B2[0] + 16, BY + 81, '单独训练的 indexer 在 checkpoint 里留下的独立形态',
        9, '#334155', 'start', maxw=B2[2] - 30, tag='b2:l2')

# indexer 框 → 标量折叠卡(q_fp8/k/weights 交给打分核)
lc.seg(RB[0] + RB[2] / 2, LB[1] + LB[3], RB[0] + RB[2] / 2, BY, lc.C_GPU_S, 1.8, 'std')
lc.text(RB[0] + RB[2] / 2 - 10, BY - 10, '(q_fp8, k, weights) 交给打分核——归一化已搬出核外',
        8.5, lc.C_GPU_S, 'end', maxw=500, tag='a:op')

# ---------------- 页脚 ----------------
lc.text(MX, 830, '图例:绿 = 模型层 / GPU 执行臂角色(L0 中列) · 青 = KV / 显存账本角色 · 虚线箭头 = 共享上游输入 · 形状按 [out, in] 标注',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 850, '形状与标量 = DSV3.2 实尺(traces 实测) · 头表独立证据 = 字段前缀 index_*(deepseek_v2.py:L655-L660) · softmax_scale=128^-0.5 · n_head_scale=64^-0.5',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 870, 'k_cache workspace 6553600 条(=163840×40) · 两套头表互不相干:测试例 2 头 vs 主 8 头,实尺 64 头 vs 128 头 · 每对 MAC 账(8192 vs 73728,11.11%)归正文算力节',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch26-fig-indexer-head.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
