#!/usr/bin/env python3
"""ch24 机制图 ② · 头数谱系:MHA → GQA → MQA 的整除映射(figure_spec ch24-fig-gqa-spectrum,模板 layout)

放大自 L0 中列『GPU 执行臂』(绿)『模型层 forward + 编译』块内 QKVParallelLinear 的
total_num_kv_heads 一根参数轴——MHA/GQA/MQA 在 vLLM 代码里只是这一个整数
(None 默认 = MHA,= 1 即 MQA)。primer 推导链第 ② 环(头数谱系):重绘 GQA 论文
Fig.2 三架构对照的语义,组件画法回指第 ① 环 MHA 接线图。

claim:『分组』只是 query 头→KV 头的整除映射 idx//(H/G):H=8、G=2 时头 0-3 吃 KV0、
头 4-7 吃 KV1;端点 G=1 全共享(MQA)、G=H 恒等(MHA),每 token 元素账 2·G·d_h
随 G 线性——Llama-3-70B 取 G=8:cache = MHA 的 1/8。

数字全部取自 figure_spec.numbers(三面板映射/端点合拢/Llama-3-70B 账/Fig.6 组数取舍/
mean-pool 玩具/linear.py 落点:论文忠实 NumPy 参考实现实跑 trace)。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 772
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '头数谱系:MHA → GQA → MQA 只是 query 头 → KV 头的整除映射 idx // (H/G)',
        16.5, lc.C_TXT, 'start', True, maxw=930, tag='title')
lc.text(MX, 58, 'G = H 恒等(= MHA)、G = 1 全共享(= MQA)——端点是同一映射的两个取值;每 token 元素账 2·G·d_h 随 G 线性(arXiv:2305.13245 §2.2)',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '推导链 ② · 放大自 L0『GPU 执行臂』QKVParallelLinear 的 total_num_kv_heads 参数轴'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 三面板 ----------------
PANELS = [
    (60, 'MHA(G = H = 8)', '每个 query 头配自己的 K/V', '[0, 1, 2, 3, 4, 5, 6, 7]',
     '端点:G = H → 恒等映射(每头自己的 KV)'),
    (440, 'GQA-2(G = 2,组宽 4)', '每 4 个 query 头共享 1 份 K/V', '[0, 0, 0, 0, 1, 1, 1, 1]',
     '中间点:组宽 4——离散的头选择'),
    (820, 'MQA(G = 1)', '全部 8 个 query 头共享 1 份', '[0, 0, 0, 0, 0, 0, 0, 0]',
     '端点:G = 1 → 常值映射(全员一份)'),
]
PW, PY, PH = 360, 90, 380
DOT_CY, DOT_R = 196, 7
KB_Y, KB_H = 250, 26
ACCT = [('2·G·d_h = 2×8×128 = 2048', 'MHA 2·n_h·d_h 同式'),
        ('2·G·d_h = 2×2×128 = 512', 'G2 的账'),
        ('2·G·d_h = 2×1×128 = 256', 'MQA 2·d_h 同式')]

for pi, (px, ttl, sub, mapping, endpoint) in enumerate(PANELS):
    lc.rect(px, PY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
    lc.text(px + PW / 2, 114, ttl, 11.5, lc.C_TXT, 'middle', True, maxw=PW - 16, tag=f'p{pi}:t')
    lc.text(px + PW / 2, 132, sub, 8.5, lc.C_MUTE, 'middle', maxw=PW - 16, tag=f'p{pi}:s')
    lc.text(px + PW / 2, 152, f'映射:{mapping}', 8, lc.C_MUTE, 'middle', tag=f'p{pi}:m')
    # query 头圆点列(绿)
    dot_x = [px + 25 + i * 40 for i in range(8)]     # 25..305(+px),最后 +25+280=305 < 360
    for i, dx in enumerate(dot_x):
        lc.circle(dx, DOT_CY, DOT_R, lc.C_GPU_S, sw=1.4, dash=False)
        lc.text(dx - 12, DOT_CY + 3, f'q{i}', 7, lc.C_MUTE, 'end', tag=f'p{pi}:q{i}')
    # KV 头方框(青)+ 映射连线
    if pi == 0:      # MHA:8 框,一一对齐
        for i, dx in enumerate(dot_x):
            bx = dx - 16
            lc.seg(dx, DOT_CY + DOT_R, dx, KB_Y, lc.C_GPU_S, 1.1)
            lc.rect(bx, KB_Y, 32, KB_H, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.2)
            lc.text(dx, KB_Y + 17, f'K{i}V{i}', 6.8, lc.C_KV_S, 'middle', tag=f'p{pi}:kb{i}')
    elif pi == 1:    # GQA-2:组内扇入
        for gi, (bx0, bw) in enumerate(((px + 20, 124), (px + 216, 124))):
            lc.rect(bx0, KB_Y, bw, KB_H, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.2)
            lc.text(bx0 + bw / 2, KB_Y + 17, f'KV{gi}(组{gi})', 7.5, lc.C_KV_S, 'middle',
                    tag=f'p{pi}:kb{gi}')
            for j in range(4):
                i = gi * 4 + j
                tx = bx0 + 18 + j * 30
                lc.seg(dot_x[i], DOT_CY + DOT_R, tx, KB_Y, lc.C_GPU_S, 1.1)
    else:            # MQA:全员扇入单框
        bx0, bw = px + 80, 200
        lc.rect(bx0, KB_Y, bw, KB_H, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.2)
        lc.text(bx0 + bw / 2, KB_Y + 17, 'KV0(全员共享)', 7.5, lc.C_KV_S, 'middle', tag='p2:kb0')
        for i in range(8):
            tx = bx0 + 18 + i * 24
            lc.seg(dot_x[i], DOT_CY + DOT_R, tx, KB_Y, lc.C_GPU_S, 1.1)
    # 元素账 + 端点注
    lc.text(px + PW / 2, 312, '每 token 每层元素账(H = 8,d_h = 128)', 8.5, lc.C_MUTE,
            'middle', tag=f'p{pi}:al')
    lc.text(px + PW / 2, 334, ACCT[pi][0], 10.5, lc.C_KV_S, 'middle', True, tag=f'p{pi}:av')
    lc.text(px + PW / 2, 356, '头 0-3 吃 KV0、头 4-7 吃 KV1(idx // 组宽)' if pi == 1
            else ('同一式:2·n_h·d_h' if pi == 0 else '同一式:2·d_h'), 7.8, lc.C_FAINT,
            'middle', tag=f'p{pi}:a2')
    lc.text(px + PW / 2, 388, endpoint, 8.5, lc.C_TXT, 'middle', maxw=PW - 20, tag=f'p{pi}:e')
    lc.text(px + PW / 2, 410, '= MHA 端点' if pi == 2 else ('组数取舍见右栏' if pi == 1 else '= GQA-H 端点'),
            7.8, lc.C_FAINT, 'middle', tag=f'p{pi}:e2')

lc.text(760, 481, '两个端点是 GQA 的特例:GQA-1 ≡ MQA、GQA-H ≡ MHA——同一根映射轴上的插值',
        9.5, lc.C_TXT, 'middle', True, maxw=760, tag='endpoint:claim')

# ---------------- 右列:组数取舍 + uptraining ----------------
RX0, RW = 1200, 240
lc.rect(RX0, PY, RW, 205, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(RX0 + RW / 2, 112, '组数怎么选(§3.3 Fig.6)', 10, lc.C_TXT, 'middle', True, tag='rt:t')
gx = [RX0 + 30 + i * 25.7 for i in range(8)]        # G=1,2,4,8,16,32,64,H(log 间距示意)
gy = [192, 190, 188, 184, 170, 150, 135, 126]
lc.parrow(list(zip(gx, gy)), lc.C_ENG_S, 1.8, marker=None)
for i, lbl in ((0, '1(MQA)'), (3, '8'), (7, 'H(MHA)')):
    lc.text(gx[i], 236, lbl, 7.5, lc.C_MUTE, 'middle', tag='rt:x' + str(i))
lc.seg(gx[3], 184, gx[3], 205, lc.C_ENG_S, 1.4, dash=True)
lc.circle(gx[3], 184, 4, lc.C_ENG_S, sw=1.6, dash=False)
lc.text(gx[3] + 10, 180, '选 8', 8.5, lc.C_ENG_S, 'start', True, tag='rt:pick')
lc.text(RX0 + 6, 132, '开销', 7.5, lc.C_MUTE, 'start', tag='rt:y')
lc.seg(gx[0] - 8, 226, gx[7] + 8, 226, '#cbd5e1', 1.0)
lc.text(RX0 + RW / 2, 256, '1 → 8 组只温和变慢,', 8, '#334155', 'middle', tag='rt:l1')
lc.text(RX0 + RW / 2, 270, '越靠近 MHA 代价递增(定性)', 8, '#334155', 'middle', tag='rt:l2')
lc.text(RX0 + RW / 2, 288, 'LLaMA-2 / 3 的 8 KV 头出处即此', 8, lc.C_MUTE, 'middle',
        maxw=RW - 12, tag='rt:l3')

lc.rect(RX0, 315, RW, 155, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(RX0 + RW / 2, 337, 'uptraining:老模型不必重训', 10, lc.C_TXT, 'middle', True, tag='up:t')
lc.text(RX0 + RW / 2, 359, '组内 K/V 头 mean-pool 并成一本', 8.5, '#334155', 'middle', tag='up:l1')
lc.text(RX0 + RW / 2, 377, '(优于选单头 / 随机初始化)', 7.8, lc.C_MUTE, 'middle', tag='up:l2')
lc.text(RX0 + RW / 2, 399, '再按原配方补 α = 0.05 的步数', 8.5, '#334155', 'middle', tag='up:l3')
lc.text(RX0 + RW / 2, 417, '(约 600 TPUv3 chip-days)', 7.8, lc.C_MUTE, 'middle', tag='up:l4')
lc.text(RX0 + RW / 2, 439, 'MHA checkpoint 直接迁过来', 8.5, lc.C_GPU_S, 'middle', True, tag='up:l5')

# ---------------- 底部三盒 ----------------
BY, BH = 486, 196
# 左:mean-pool 玩具
lc.rect(MX, BY, 470, BH, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 235, BY + 22, 'mean-pool 玩具:两头并一本(先投再平均 == 先平均再投)', 9.5,
        lc.C_TXT, 'middle', True, maxw=446, tag='mp:t')
lc.rect(MX + 36, BY + 42, 78, 24, '#ffffff', '#cbd5e1', rx=4, sw=1.0)
lc.text(MX + 75, BY + 58, '[1, 2]', 9, lc.C_TXT, 'middle', tag='mp:a')
lc.text(MX + 128, BY + 58, '(组0 头0 的 K)', 7.5, lc.C_MUTE, 'start', tag='mp:an')
lc.rect(MX + 36, BY + 74, 78, 24, '#ffffff', '#cbd5e1', rx=4, sw=1.0)
lc.text(MX + 75, BY + 90, '[3, 6]', 9, lc.C_TXT, 'middle', tag='mp:b')
lc.text(MX + 128, BY + 90, '(组0 头1 的 K)', 7.5, lc.C_MUTE, 'start', tag='mp:bn')
lc.seg(MX + 114, BY + 54, MX + 218, BY + 70, lc.C_GPU_S, 1.4, marker='std')
lc.seg(MX + 114, BY + 86, MX + 218, BY + 70, lc.C_GPU_S, 1.4, marker='std')
lc.text(MX + 166, BY + 44, '平均', 7.5, lc.C_MUTE, 'middle', tag='mp:avg')
lc.rect(MX + 224, BY + 58, 78, 24, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.3)
lc.text(MX + 263, BY + 74, '[2, 4]', 9, lc.C_GPU_S, 'middle', True, tag='mp:r')
lc.text(MX + 310, BY + 74, '(组0 池化头)', 7.5, lc.C_MUTE, 'start', tag='mp:rn')
lc.text(MX + 235, BY + 122, 'max diff = 0.0(平均是线性组合,分配律保证可交换)', 8.5,
        lc.C_MUTE, 'middle', maxw=446, tag='mp:c1')
lc.text(MX + 235, BY + 144, '玩具:W^K(组0) 行 = [2,0,0] / [0,2,0],h = [1, 2, 3]', 8,
        lc.C_FAINT, 'middle', maxw=446, tag='mp:c2')
lc.text(MX + 235, BY + 166, '§2.2 原句:mean-pooling all the original heads within that group',
        7.5, lc.C_FAINT, 'middle', maxw=446, tag='mp:c3')

# 中:Llama-3-70B 账
lc.rect(550, BY, 380, BH, '#ffffff', lc.C_KV_S, rx=8, sw=1.4)
lc.text(740, BY + 22, 'Llama-3-70B:G = 8 的真实账', 9.5, lc.C_TXT, 'middle', True, tag='l3:t')
lc.text(740, BY + 46, '64 query 头 / 8 KV 头 → 组宽 8', 9, '#334155', 'middle', tag='l3:l1')
lc.text(740, BY + 68, 'cache = 2×8×128 = 2048 元素/层', 9.5, lc.C_KV_S, 'middle', True, tag='l3:l2')
BAR_X, BAR_Y, BAR_W, BAR_H = 580, BY + 86, 320, 18
lc.rect(BAR_X, BAR_Y, BAR_W, BAR_H, '#e2e8f0', '#cbd5e1', rx=3, sw=1.0)
lc.rect(BAR_X, BAR_Y, BAR_W / 8, BAR_H, lc.C_KV_S, 'none', rx=3, sw=0)
lc.text(BAR_X + BAR_W / 16, BAR_Y + 32, 'GQA-8:2048', 7.5, lc.C_KV_S, 'middle', True, tag='l3:b1')
lc.text(BAR_X + BAR_W / 2 + 40, BAR_Y + 32, '若 MHA:16384(2×64×128)', 7.5, lc.C_MUTE,
        'middle', tag='l3:b2')
lc.text(740, BY + 146, 'cache = MHA 的 1/8(0.125)', 9, lc.C_TXT, 'middle', True, tag='l3:l3')
lc.text(740, BY + 168, '端点合拢(H=4 核验):G=4 映射 [0,1,2,3] / G=1 [0,0,0,0],账 16/8/4 随 G 线性',
        7.5, lc.C_FAINT, 'middle', maxw=356, tag='l3:l4')

# 右:vLLM 落点
lc.rect(950, BY, 490, BH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(1195, BY + 22, 'vLLM 落点:三张图只是一个整数参数', 9.5, lc.C_GPU_S, 'middle',
        True, maxw=466, tag='vl:t')
CODE = ('QKVParallelLinear(',
        '    total_num_heads = 64,',
        '    total_num_kv_heads = 8,   # ← G(= 组数)',
        ')')
for i, ln in enumerate(CODE):
    lc.text(975, BY + 46 + i * 19, ln, 9, lc.C_TXT if i != 2 else lc.C_GPU_S, 'start',
            i == 2, maxw=440, tag=f'vl:c{i}')
lc.text(1195, BY + 132, 'total_num_kv_heads = None → 默认 MHA(= total_num_heads)', 8.5,
        '#334155', 'middle', maxw=466, tag='vl:n')
lc.text(1195, BY + 150, 'output_sizes 的 k / v 段按 num_kv_heads 计', 8.5, '#334155',
        'middle', tag='vl:o')
lc.text(1195, BY + 168, 'tp=8 时 MQA:num_kv_head_replicas = 8(每 rank 复制同一 KV 头)', 8.5,
        '#334155', 'middle', maxw=466, tag='vl:rep')
lc.text(1195, BY + 186, 'vllm/model_executor/layers/linear.py:L1070-L1088', 7.5, lc.C_FAINT,
        'middle', tag='vl:anchor')

# ---------------- 页脚 ----------------
lc.text(MX, 706, '图例:绿点 = query 头 · 青框 = KV 头(K/V 各一份)· 绿线 = idx // 组宽 映射 · 橙线 = 开销曲线(定性)',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 728, '谱系画法重绘自 arXiv:2305.13245 Fig.2 语义(组件画法回指第 ① 环 MHA 接线图)· 数值取自论文忠实 NumPy 参考实现实跑(host,float64)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 744, 'vLLM 行号基线 v0.27.1(6e448d0ea)', 8.5, lc.C_FAINT, 'start', tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch24-fig-gqa-spectrum.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
