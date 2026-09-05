#!/usr/bin/env python3
"""ch20 机制图 ④ · IO 账:数趟数不数算力(figure_spec ch20-fig-io-accounting,模板 before-after)

放大自 L0 中列『GPU 执行臂』(绿色列)第三块『模型层 forward + 编译』内 attention kernel 的
访存账。primer 推导链第 ⑤ 环:同一段 HBM↔SRAM 通道上,两种搬法的趟数差。
架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:N=1024、d=64 的精确元素级记账——标准 Alg.0 三步 4456448 次访存+物化 2097152 元素;
FA 一趟 KV + T_c 趟轻统计量,B_c=128 时 1736704 次(标准:FA=2.566)、256 时 933888 次(4.7719)
——块越大趟数越少,但受 SRAM 容量与算力双重封顶。

数字全部取自 figure_spec.numbers(host 元素级实算:standard_attention 三步账/flash_sweep
三行/单调断言/内存脚印;论文 Thm.2+Prop.3)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 720
MX = 60
BXR = 1440
C_RED = '#dc2626'
EXTRA_DEFS = ('<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')

# 条形刻度:380px = 4456448 元素(标准总计)
STD_TOTAL = 4456448
SCALE = 380.0 / STD_TOTAL


def px(elems):
    return max(3.0, elems * SCALE)


# ---------------- 标题区 ----------------
lc.text(MX, 34, '为什么快:数的是趟数,不是算力——N=1024、d=64(GPT-2 头维)的元素级访存账',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '标准实现被两张 N×N 物化顶在 Ω(N²) 下不来;FA 只把 K/V 各搬一遍,再把轻的 Q/O/ℓ/m 来回搬 ⌈N/B_c⌉ 趟——块越大趟数越少,直到 SRAM 装不下(arXiv:2205.14135 §3.2 Thm.2)',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '推导链 ⑤ · 放大自 L0 GPU 执行臂内 attention kernel 的访存账'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左:before = 标准 Alg.0 ----------------
lc.text(MX, 96, '标准 Alg.0:三步,每步整表过手', 12, lc.C_TXT, 'start', True, maxw=430, tag='lp:t')
lc.text(680, 96, 'Θ(Nd + N²)', 11, C_RED, 'end', True, maxw=120, tag='lp:theta')
lc.text(76, 122, '步骤', 9, lc.C_MUTE, 'start', True, maxw=100, tag='lp:hd1')
lc.text(260, 122, 'HBM 访存(元素,条长∝数量)', 9, lc.C_MUTE, 'start', True, maxw=300, tag='lp:hd2')
STD_ROWS = [
    ('① 读 Q,K → 写 S', 1179648, False),
    ('② 读 S → 写 P', 2097152, True),
    ('③ 读 P,V → 写 O', 1179648, False),
]
ry = 134
for label, elems, hot in STD_ROWS:
    lc.text(76, ry + 16, label, 9.5, '#334155', 'start', maxw=180, tag='lr:' + label[:6])
    bw = px(elems)
    lc.rect(260, ry, bw, 22, C_RED, C_RED, rx=3, sw=0)
    lc.text(260 + bw + 8, ry + 16, f'{elems}', 10, C_RED, 'start', hot, maxw=120,
            tag='lr:n' + label[:4])
    ry += 46
# 总计
lc.text(76, ry + 18, '三步总计', 10.5, lc.C_TXT, 'start', True, maxw=120, tag='lp:tot')
lc.rect(260, ry + 2, px(STD_TOTAL), 24, C_RED, C_RED, rx=3, sw=0)
lc.text(260 + px(STD_TOTAL) + 8, ry + 20, f'{STD_TOTAL}', 11.5, C_RED, 'start', True, maxw=140,
        tag='lp:totn')
TOT_Y = ry + 30
# 物化注 + 两张 N×N 小方块
lc.text(76, TOT_Y + 26, '其中物化 S、P 两张 N×N 共 2097152 元素——步② 的读与写全是它们:',
        9.5, '#334155', 'start', maxw=470, tag='lp:mat')
for i, nm in enumerate(['S', 'P']):
    lc.rect(560 + i * 46, TOT_Y + 12, 34, 34, '#fee2e2', C_RED, rx=4, sw=1.4)
    lc.text(577 + i * 46, TOT_Y + 34, nm, 12, C_RED, 'middle', True, maxw=30, tag='lp:sq' + nm)
lc.text(76, TOT_Y + 52, 'softmax 行归一化要整行分数——朴素实现无处安放,只能物化两张 N×N、',
        9, lc.C_MUTE, 'start', maxw=560, tag='lp:lb1')
lc.text(76, TOT_Y + 68, '读写至少各一遍 → Ω(N²) 把下限顶死', 9, lc.C_MUTE, 'start', maxw=560,
        tag='lp:lb2')

# ---------------- 右:after = FlashAttention ----------------
lc.text(740, 96, 'FlashAttention:K/V 各搬一遍 + 轻统计量多趟', 12, lc.C_TXT, 'start', True,
        maxw=480, tag='rp:t')
lc.text(1440, 96, 'Θ(N²d²/M)', 11, lc.C_GPU_S, 'end', True, maxw=140, tag='rp:theta')
lc.text(756, 122, '构成', 9, lc.C_MUTE, 'start', True, maxw=100, tag='rp:hd1')
lc.text(990, 122, 'HBM 访存(元素,同一刻度)', 9, lc.C_MUTE, 'start', True, maxw=280, tag='rp:hd2')
# K/V 一遍
lc.text(756, 150, 'K,V 各搬一遍(外层恰一遍)', 9.5, '#334155', 'start', maxw=220, tag='rp:kv')
lc.rect(990, 136, px(131072), 22, lc.C_GPU_S, lc.C_GPU_S, rx=3, sw=0)
lc.text(990 + px(131072) + 8, 152, '131072', 10, lc.C_GPU_S, 'start', maxw=100, tag='rp:kvn')
# 每趟轻统计量 + 循环小箭头
lc.text(756, 192, '每趟 Q,O,ℓ,m = 3Nd+4N', 9.5, '#334155', 'start', maxw=220, tag='rp:pp')
lc.text(756, 208, '= 200704 元素 × T_c 趟', 9.5, '#334155', 'start', maxw=220, tag='rp:pp2')
lc.rect(990, 178, px(200704), 22, lc.C_GPU_S, lc.C_GPU_S, rx=3, sw=0)
lc.text(990 + px(200704) + 8, 194, '200704', 10, lc.C_GPU_S, 'start', maxw=100, tag='rp:ppn')
lc.parrow([(1230, 184), (1248, 184), (1248, 194), (1230, 194), (1230, 187)], lc.C_GPU_S, 1.4,
          'grn')
lc.text(1256, 194, '× T_c 趟', 8.5, lc.C_GPU_S, 'start', True, maxw=80, tag='rp:loop')
# 对照刻度(标准虚线框)
lc.text(756, 244, '对照刻度:标准 4456448', 8.5, C_RED, 'start', maxw=200, tag='rp:ref')
lc.rect(990, 230, px(STD_TOTAL), 22, 'none', C_RED, rx=3, sw=1.4, dash=True)
lc.text(990 + px(STD_TOTAL) + 8, 246, '4456448', 9, C_RED, 'start', maxw=100, tag='rp:refn')
# 扫描三行
SWEEP = [
    ('B_c=64 → 16 趟', 16, 3342336, '1.3333×'),
    ('B_c=128 → 8 趟', 8, 1736704, '2.566×'),
    ('B_c=256 → 4 趟', 4, 933888, '4.7719×'),
]
sy = 266
for label, passes, total, ratio in SWEEP:
    lc.text(756, sy + 16, label, 9.5, '#334155', 'start', True, maxw=200, tag='sw:' + label[:6])
    lc.rect(990, sy, px(total), 22, lc.C_GPU_S, lc.C_GPU_S, rx=3, sw=0)
    lc.text(990 + px(total) + 8, sy + 16, f'{total}', 10, lc.C_GPU_S, 'start', True, maxw=110,
            tag='sw:n' + label[:4])
    lc.text(1440, sy + 16, f'标准:FA = {ratio}', 9, lc.C_BEAT_T, 'end', True, maxw=160,
            tag='sw:r' + label[:4])
    sy += 44
lc.text(756, sy + 6, 'B_c 增大 → 趟数减半 → 访存严格减少(16→8→4,实跑单调断言);0 张 N×N 物化',
        9, lc.C_MUTE, 'start', maxw=660, tag='rp:mono')

# ---------------- 底部两块 + Prop.3 条 ----------------
BY = 468
lc.rect(MX, BY, 620, 108, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 20, '内存脚印对照(Theorem 1 的 O(N) 额外内存)', 10, lc.C_TXT, 'start',
        True, maxw=592, tag='b1:t')
lc.text(MX + 14, BY + 40, '标准:物化两张 N×N = 2097152 元素(fp16 下 4194304 字节)', 9,
        '#334155', 'start', maxw=592, tag='b1:l1')
lc.text(MX + 14, BY + 58, 'FA:额外统计量 (m,ℓ) 各 N 个 = 2048 元素(4096 字节)', 9,
        '#334155', 'start', maxw=592, tag='b1:l2')
lc.text(MX + 14, BY + 78, '约 1024 倍差——省下的显存随 N 线性、不再是平方', 9, lc.C_GPU_S,
        'start', True, maxw=592, tag='b1:l3')
lc.rect(740, BY, 700, 108, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(754, BY + 20, 'B_c 怎么定、为何封顶', 10, lc.C_TXT, 'start', True, maxw=672, tag='b2:t')
lc.text(754, BY + 40, '论文规则:Bc = ⌈M/4d⌉、Br = min(⌈M/4d⌉,d)(Alg.1 line 1)——M≈100KB 按 fp16 元素计 51200 → Bc=200',
        9, '#334155', 'start', maxw=672, tag='b2:l1')
lc.text(754, BY + 58, '工程上 FA-2 按 d 与 SMEM 取 {64,128}(arXiv:2307.08691 §3.2)', 9,
        '#334155', 'start', maxw=672, tag='b2:l2')
lc.text(754, BY + 78, 'B_c>256 收益封顶:块再大放不进 SRAM、算力成瓶颈(论文 Fig.2 中图实测)', 9,
        lc.C_BEAT_T, 'start', True, maxw=672, tag='b2:l3')

PY = BY + 124
lc.rect(MX, PY, BXR - MX, 54, '#ffffff', lc.C_KV_S, rx=8, sw=1.3, dash=True)
lc.text(MX + 14, PY + 20, 'Prop.3(下界):对 M∈[d,Nd] 全域,不存在 o(N²d²/M) 的精确注意力算法——FA 的 Θ(N²d²/M) 渐进最优,没有更聪明的精确算法',
        9.5, lc.C_KV_S, 'start', True, maxw=BXR - MX - 28, tag='prop:t')
lc.text(MX + 14, PY + 38, 'Thm.2 对照:标准 Θ(Nd+N²) vs FlashAttention Θ(N²d²M^(−1))——对典型 d 与 M,前者多倍访存(本例实测 1.33-4.77 倍)',
        9, '#334155', 'start', maxw=BXR - MX - 28, tag='prop:l1')

# ---------------- 页脚 ----------------
lc.text(MX, PY + 84, '图例:红条 = 标准 Alg.0 的整表访存 · 绿条 = FlashAttention 的访存 · 红虚线框 = 对照刻度(标准总计 4456448)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, PY + 102, '计数口径:元素级精确计数(非 Θ 记号)——Alg.0 三步各读/写、FA 侧外层 K/V 各恰一遍 + 每遍重过 Q,O,ℓ,m(host 实算)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, PY + 120, '出处 arXiv:2205.14135 §3.2 Theorem 2 + Proposition 3 · 工程化身:FA3 host 侧 get_scheduler_metadata 即 tile/split 计账落地(vllm/vllm_flash_attn/flash_attn_interface.py:L122-L173)· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-io-accounting.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
