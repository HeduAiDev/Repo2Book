#!/usr/bin/env python3
"""ch20 机制图 ⑥ · LSE 合并六步(figure_spec ch20-fig-lse-merge,模板 flow)

放大自 L0 中列『GPU 执行臂』(绿色列)『模型层 forward + 编译』块内 attention kernel 之后的
合并 kernel——数学从论文走进 vLLM 源码的落点(vllm/v1/attention/ops/triton_merge_attn_states.py)。
primer 推导链第 ⑦ 环:⊕ 在 (lse,output) 表示上的作用。架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:两段部分注意力 (O_a,lse_a)、(O_b,lse_b) 经六步合并——max_lse 稳定化 → e^(lse−max) →
out_se=Σ → 权重=占比 → 加权 O → 合并 lse=log(out_se)+max_lse——与对拼接 KV 一次性 softmax
逐位相等;vLLM Triton kernel 逐 (token,head) 即此六步,变量名一一对应。

数字全部取自 figure_spec.numbers(四行合并账/反超行 2.0064<2.4076→0.599/合并与一次性全等:
host NumPy 参考实现实跑;Triton 六步原文与空段护栏的行号锚点)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 778
MX = 60
BXR = 1440
C_RED = '#dc2626'
EXTRA_DEFS = ('<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'LSE 合并:两段注意力各带一个标量,按占比缝成一个——与一口气算逐位相等',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, '每段只多带 lse = log(softmax 分母);示例 = 请求 A 行 1(两段 lse 不相等 → 权重非 50/50,且后缀反超)——vLLM Triton merge kernel 逐 (token,head) 变量名一一对应',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '推导链 ⑦ · 放大自 L0 GPU 执行臂内 attention kernel 之后的合并 kernel'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左:两段入口 ----------------
EA_X, EA_W = 60, 244
lc.text(EA_X, 96, '两段各交一份摘要', 11, lc.C_TXT, 'start', True, maxw=240, tag='ea:t')
lc.rect(EA_X, 108, EA_W, 78, '#ffffff', lc.C_KV_S, rx=8, sw=1.6)
lc.text(EA_X + 12, 128, '前缀段(共享 KV 一次算)', 10, lc.C_KV_S, 'start', True, maxw=EA_W - 24,
        tag='ea:a:t')
lc.text(EA_X + 12, 147, '(O_a, lse_a = 2.0064)', 10, lc.C_TXT, 'start', True, maxw=EA_W - 24,
        tag='ea:a:v')
lc.text(EA_X + 12, 166, 'block_table[:1] · causal=False', 8.5, lc.C_MUTE, 'start', maxw=EA_W - 24,
        tag='ea:a:f')
lc.rect(EA_X, 216, EA_W, 78, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(EA_X + 12, 236, '后缀段(各请求私有 KV)', 10, lc.C_GPU_S, 'start', True, maxw=EA_W - 24,
        tag='ea:b:t')
lc.text(EA_X + 12, 255, '(O_b, lse_b = 2.4076)', 10, lc.C_TXT, 'start', True, maxw=EA_W - 24,
        tag='ea:b:v')
lc.text(EA_X + 12, 274, 'block_table[:, num_common_kv_blocks:]', 8.5, lc.C_MUTE, 'start',
        maxw=EA_W - 24, tag='ea:b:f')
lc.text(EA_X, 316, 'lse_b > lse_a:后缀反超', 9, lc.C_BEAT_T, 'start', True, maxw=240,
        tag='ea:rev')
lc.text(EA_X, 333, '(查询见 4+3 个键,质量在后段)', 8.5, lc.C_MUTE, 'start', maxw=240,
        tag='ea:rev2')

# ---------------- 中:六步合并 ----------------
ST_X, ST_W = 356, 640
lc.text(ST_X, 96, '六步合并 = vLLM Triton merge kernel 的变量名(逐 token,head)', 11, lc.C_TXT,
        'start', True, maxw=640, tag='st:t')
STEPS = [
    ('①', 'max_lse = maximum(lse_a, lse_b) = 2.4076',
     '稳定化:后续两个指数都 ≤ 1,不溢出(safe softmax 减 max 同一招)'),
    ('②', 'p_se = e^(lse_a−max) = 0.6695   s_se = e^(lse_b−max) = 1.0', None),
    ('③', 'out_se = p_se + s_se = 1.6695', None),
    ('④', 'p_scale = p_se/out_se = 0.401   s_scale = s_se/out_se = 0.599',
     '权重 = 各段指数质量占总盘子的比例(和恒 1)'),
    ('⑤', 'out = p_out × p_scale + s_out × s_scale',
     'NOTE(woosuk):先算 scale 再乘 output——数值稳定纪律'),
    ('⑥', 'out_lse = log(out_se) + max_lse = 2.9201',
     '合并 lse 可继续并第三段(⊕ 结合律)'),
]
SY, SH, SG = 110, 50, 8
for i, (num, main, sub) in enumerate(STEPS):
    y = SY + i * (SH + SG)
    lc.rect(ST_X, y, ST_W, SH, '#ffffff', lc.C_GPU_S, rx=7, sw=1.2)
    lc.text(ST_X + 12, y + 20, num, 10, lc.C_GPU_S, 'start', True, maxw=20, tag='st:n' + num)
    lc.text(ST_X + 34, y + 20, main, 10, lc.C_TXT, 'start', True, maxw=ST_W - 46,
            tag='st:m' + num)
    if sub:
        lc.text(ST_X + 34, y + 38, sub, 8.3, lc.C_MUTE, 'start', maxw=ST_W - 46,
                tag='st:s' + num)
    if i < len(STEPS) - 1:
        lc.seg(ST_X + ST_W / 2, y + SH + 1, ST_X + ST_W / 2, y + SH + SG - 1, lc.C_GPU_S,
               1.8, 'grn')
ST_END = SY + len(STEPS) * (SH + SG) - SG

# 入口 → 步骤列箭头
lc.parrow([(EA_X + EA_W, 147), (ST_X - 16, 147), (ST_X - 16, SY + 22), (ST_X, SY + 22)],
          lc.C_KV_S, 2.0, 'grn')
lc.parrow([(EA_X + EA_W, 255), (ST_X - 6, 255), (ST_X - 6, SY + SH + 2)],
          lc.C_GPU_S, 2.0, 'grn')

# ---------------- 右:出口与对照 ----------------
OX, OW = 1060, 380
lc.text(OX, 96, '缝回来的结果', 11, lc.C_TXT, 'start', True, maxw=200, tag='ox:t')
lc.rect(OX, 110, OW, 84, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(OX + 14, 132, '合并输出(两段缝合)', 10.5, lc.C_GPU_S, 'start', True, maxw=OW - 28,
        tag='ox:t2')
lc.text(OX + 14, 154, 'O = [2.2957, 2.5654]', 11, lc.C_TXT, 'start', True, maxw=OW - 28,
        tag='ox:o')
lc.text(OX + 14, 176, 'out_lse = 2.9201', 10, lc.C_TXT, 'start', maxw=OW - 28, tag='ox:lse')
# 连线:⑥ → 出口
lc.seg(ST_X + ST_W + 1, SY + 5 * (SH + SG) + SH / 2, OX - 1, SY + 5 * (SH + SG) + SH / 2,
       lc.C_GPU_S, 2.2, 'grn')
# 一次性对照(虚线)
lc.rect(OX, 250, OW, 84, '#ffffff', lc.C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(OX + 14, 272, '对照:一次性算(拼接 KV 全长)', 10, lc.C_MUTE, 'start', True, maxw=OW - 28,
        tag='ox:c:t')
lc.text(OX + 14, 294, 'O = [2.2957, 2.5654]', 10.5, lc.C_TXT, 'start', maxw=OW - 28, tag='ox:c:o')
lc.text(OX + 14, 316, 'lse = 2.9201', 9.5, lc.C_TXT, 'start', maxw=OW - 28, tag='ox:c:lse')
# 相等徽标
lc.rect(OX + OW / 2 - 92, 202, 184, 36, lc.C_GPU_F, lc.C_GPU_S, rx=18, sw=1.6)
lc.text(OX + OW / 2, 225, '逐位相等:差 0.0', 10.5, lc.C_GPU_S, 'middle', True, maxw=176,
        tag='ox:eq')
lc.seg(OX + OW / 2, 194 + 8, OX + OW / 2, 202, lc.C_GPU_S, 1.4, dash=True)
lc.seg(OX + OW / 2, 238, OX + OW / 2, 250, lc.C_GPU_S, 1.4, dash=True)

# ---------------- 四行合并账表 ----------------
TB_Y = 470
lc.text(MX, TB_Y, '四行合并账(实跑)——权重和恒 1;合并 lse 与一次性 lse 全等,O 差 0.0',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='tb:t')
COLS2 = [('请求行', 92), ('前缀 lse', 100), ('后缀 lse', 100), ('max_lse', 92), ('p_se', 82),
         ('s_se', 82), ('out_se', 88), ('前缀权重', 100), ('后缀权重', 100),
         ('合并 lse = 一次性', 150)]
TX0 = (W - sum(w for _, w in COLS2)) / 2
xs2 = []
cx = TX0
for _, w in COLS2:
    xs2.append((cx, w))
    cx += w
TB_W = cx - TX0
HDR2_Y, RH2 = TB_Y + 14, 27
lc.rect(TX0, HDR2_Y, TB_W, RH2 * 5, '#ffffff', lc.C_MUTE, rx=6, sw=1.3)
lc.rect(TX0, HDR2_Y, TB_W, RH2, '#f1f5f9', lc.C_MUTE, rx=6, sw=1.1)
for i in range(1, len(COLS2)):
    lc.seg(xs2[i][0], HDR2_Y, xs2[i][0], HDR2_Y + RH2 * 5, '#e2e8f0', 1.0)
for (x, w), (name, _) in zip(xs2, COLS2):
    lc.text(x + w / 2, HDR2_Y + 18, name, 9, lc.C_MUTE, 'middle', True, maxw=w - 8,
            tag='tbh:' + name)
TROWS = [
    ('A 行0', '3.0064', '1.6931', '3.0064', '1.0', '0.2689', '1.2689', '0.7881', '0.2119',
     '3.2446', False),
    ('A 行1', '2.0064', '2.4076', '2.4076', '0.6695', '1.0', '1.6695', '0.401', '0.599',
     '2.9201', True),
    ('B 行0', '4.2539', '2.0', '4.2539', '1.0', '0.105', '1.105', '0.905', '0.095',
     '4.3537', False),
    ('B 行1', '3.0064', '2.6931', '3.0064', '1.0', '0.7311', '1.7311', '0.5777', '0.4223',
     '3.5551', False),
]
for ri, row in enumerate(TROWS):
    ry = HDR2_Y + RH2 * (ri + 1)
    hot = row[-1]
    if hot:
        lc.rect(TX0 + 1, ry + 1, TB_W - 2, RH2 - 2, lc.C_BEAT_F, 'none', rx=0, sw=0)
    if ri > 0:
        lc.seg(TX0, ry, TX0 + TB_W, ry, '#e2e8f0', 1.0)
    for ci, v in enumerate(row[:-1]):
        x, w = xs2[ci]
        bold = ci in (7, 8, 9)
        fill = lc.C_GPU_S if ci == 9 else lc.C_TXT
        lc.text(x + w / 2, ry + 18, v, 9, fill, 'middle', bold, maxw=w - 8,
                tag=f'tr{ri}c{ci}')

# ---------------- 反超注 + 空段护栏 ----------------
NY = HDR2_Y + RH2 * 5 + 18
lc.text(MX, NY, 'A 行 1(橙底):后缀 lse 反超(2.4076 > 2.0064)→ 后缀权重 0.599 更大——权重跟着归一化质量走,不认『前缀』名分;两段权重之和恒为 1',
        9, lc.C_BEAT_T, 'start', True, maxw=BXR - MX, tag='ny:rev')
GY = NY + 16
lc.rect(MX, GY, BXR - MX, 50, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 14, GY + 19, '空段与双空的工程护栏(kernel 内建):FA2 对空序列返回 inf、FA3 返回 −inf——kernel 先做 inf→−inf 归一(L270-L276);双空(max_lse=−inf)输出置 0、合并 lse 保持 −inf(L319-L322)',
        9, lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='gy:t')
lc.text(MX + 14, GY + 37, '空段 lse=−inf → 权重自然为 0,合并退化为另一段(⊕ 的空元)——cascade 拆段与 split-KV 的正确性都在这一页',
        8.7, '#334155', 'start', maxw=BXR - MX - 28, tag='gy:l')

# ---------------- 页脚 ----------------
lc.text(MX, GY + 72, '合并方法出处:merge_attn_states docstring 自引 arXiv:2501.01005 §2.2(split-KV 合并;数学同 ⊕ 结合律)· 数值取自 NumPy 参考实现实跑(host,float64)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, GY + 90, 'vllm/v1/attention/ops/triton_merge_attn_states.py:L259-L322 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-lse-merge.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
