#!/usr/bin/env python3
"""ch19 机制图 4 · 切点账本（figure_spec ch19-fig-splitting-ops-account，模板 layout）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ① 拍片
『档位表与切点账本』（站 2，set_splitting_ops_for_v1）的账本内容展开。
架构归属回指 L0/L2：右上角指北小签。

claim：『在哪切』是一份显式账本：13 个注意力算子 + 2 个 KV 写算子共 15 条；
KV 写入账不是因为数学不可融合，而是其字符串参数阻止 Inductor 复用 piecewise
图（issue #33267）。

数字/引语全部取自 figure_spec.numbers（账本 15 条=13+2 实跑；13 算子清单头尾
compilation.py:L762-L778；追加注释原话 L1177-L1180；对照实跑 kv_update 落编译片；
full cudagraph 保留 piecewise 结构 PR #20059）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '「在哪切」是一份 15 条的显式账本——KV 写入账的理由是图复用、不是数学',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'set_splitting_ops_for_v1：13 个注意力算子 + 2 条 KV 写算子——切点显式记账，不由编译器自己猜',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ① 档位表与切点账本 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 右上：full cudagraph 注记卡 ----------------
NX, NY, NW, NH = 600, 96, BXR - 600, 92
lc.rect(NX, NY, NW, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(NX + 14, NY + 22, '另注：full cudagraph 也保留 piecewise 结构', 9.5, lc.C_TXT,
        'start', True, maxw=NW - 28, tag='n1:t')
lc.text(NX + 14, NY + 42, '开 full cudagraph 时不设空列表进图内捕——保留 piecewise fx 图结构、在全图外捕（PR #20059，', 8.3, '#334155',
        'start', maxw=NW - 28, tag='n1:l1')
lc.text(NX + 14, NY + 60, 'compilation.py:L1149-L1156 注释），减少 runtime batch 不在捕获表时的 CPU 开销。', 8.3, '#334155',
        'start', maxw=NW - 28, tag='n1:l2')
lc.text(NX + 14, NY + 78, '——即便『整图一张』，15 条账本也先切再捕。', 8.3, lc.C_BEAT_T,
        'start', maxw=NW - 28, tag='n1:l3')

# ---------------- 左：账本卡 ----------------
LX, LY, LW, LH = MX, 96, 512, 452
lc.rect(LX, LY, LW, LH, '#ffffff', lc.C_BEAT_S, rx=9, sw=1.8)
lc.text(LX + 16, LY + 26, '切点账本 splitting_ops（15 条）', 11.5, lc.C_BEAT_T, 'start', True,
        maxw=LW - 32, tag='led:t')
lc.text(LX + LW - 16, LY + 26, 'namespace::name 格式', 7.8, lc.C_FAINT, 'end', maxw=170,
        tag='led:fmt')
lc.text(LX + 16, LY + 44, 'set_splitting_ops_for_v1（vllm/config/compilation.py:L1133-L1184）',
        8, lc.C_FAINT, 'start', maxw=LW - 32, tag='led:f')

ROWS = [
    ('1', 'vllm::unified_attention_with_output', 'attention'),
    ('2', 'vllm::unified_mla_attention_with_output', 'attention'),
    ('…', '⋯ 中略 10 行（mamba / short_conv / linear_attention / GDN 系 /', 'ellipsis'),
    ('', 'sparse_attn / deepseek_v4_attention 等）⋯', 'ellipsis2'),
    ('13', 'vllm::hpc_rope_norm_forward', 'attention'),
    ('SEP', '', ''),
    ('14', 'vllm::unified_kv_cache_update', 'kvwrite'),
    ('15', 'vllm::unified_mla_kv_cache_update', 'kvwrite'),
]
ry = LY + 62
ROWH = 27
for num, name, kind in ROWS:
    if kind == 'SEP':
        lc.seg(LX + 16, ry + 4, LX + LW - 16, ry + 4, lc.C_MUTE, 1.0, dash=True)
        lc.text(LX + LW / 2, ry + 16, '── 账本尾部分隔线：KV 写两算子另起 ──', 7.4,
                lc.C_FAINT, 'middle', maxw=LW - 32, tag='sep')
        ry += ROWH
        continue
    hot = (kind == 'kvwrite')
    faint = kind.startswith('ellipsis')
    if hot:
        lc.rect(LX + 13, ry - 18, LW - 26, 25, lc.C_BEAT_F, lc.C_BEAT_S, rx=5, sw=1.2)
    lc.rect(LX + 16, ry - 15, 26, 19, '#ffffff', lc.C_MUTE, rx=4, sw=0.8)
    lc.text(LX + 29, ry - 2, num, 8, lc.C_TXT, 'middle', True, maxw=22, tag='rn' + num)
    lc.text(LX + 52, ry - 2, name, 8.6 if not faint else 7.8,
            lc.C_BEAT_T if hot else ('#334155' if not faint else lc.C_MUTE), 'start',
            maxw=LW - 70, tag='rw' + num)
    ry += ROWH
# 右侧括注
lc.text(LX + LW - 16, LY + 62 + 2 * ROWH - 8, '← 13 条注意力算子', 8, lc.C_BEAT_T, 'end',
        True, maxw=150, tag='br1')
lc.text(LX + LW - 16, LY + 62 + 6 * ROWH + 2 * ROWH - 6, '← 2 条 KV 写算子', 8, lc.C_BEAT_T,
        'end', True, maxw=150, tag='br2')

# ---------------- 右下：#33267 便签 ----------------
SX, SY, SW_, SH_ = 600, 204, BXR - 600, 128
lc.rect(SX, SY, SW_, SH_, '#ffffff', lc.C_ABORT, rx=8, sw=1.3, dash=True)
lc.text(SX + 14, SY + 22, '追加两算子的入账理由（注释原话 · issue #33267）', 9.5, lc.C_ABORT,
        'start', True, maxw=SW_ - 28, tag='stk:t')
STK = ['unified_kv_cache_update has a string param that prevents Inductor from reusing',
       'piecewise graphs. Remove it from the compiled graph.（L1177-L1180）',
       '——不是『数学不可融合』，是字符串参数让 piecewise 图复用不了；移出编译图',
       '的副作用是 KV cache 不进 cudagraph，实测不影响性能。']
for i, ln in enumerate(STK):
    lc.text(SX + 14, SY + 42 + i * 19, ln, 8.3, '#334155', 'start', maxw=SW_ - 28,
            tag='stk:l' + str(i))

# 小提示：账本空着 = 一处也不切（对照见下）
lc.text(SX + 14, SY + SH_ + 22, '名单空着 → 一处也不切（整图 1 片 24 节点全编译）——见下方对照 C。', 8.5,
        lc.C_MUTE, 'start', maxw=SW_ - 28, tag='empty')

# ---------------- 底部：对照两格 ----------------
CX0, CY0, CW0, CH0 = MX, 572, BXR - MX, 156
lc.rect(CX0, CY0, CW0, CH0, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(CX0 + 16, CY0 + 24, '对照：同一条 24 节点两层玩具流（精简版 companion host 实跑，真实 split_graph）',
        9.5, lc.C_TXT, 'start', True, maxw=CW0 - 32, tag='cmp:t')

def strip(px, py, pw, title, counts, kinds, rings=()):
    """5 格条：counts=节点数, kinds='c'编译/'s'接缝; rings=要红圈的格序号"""
    lc.text(px, py - 10, title, 8.6, lc.C_TXT, 'start', True, maxw=pw, tag='st:' + title[:10])
    n = len(counts)
    gap = 6
    cw = (pw - (n - 1) * gap) / n
    for i, (cnt, k) in enumerate(zip(counts, kinds)):
        x = px + i * (cw + gap)
        seam = (k == 's')
        lc.rect(x, py, cw, 46, lc.C_BEAT_F if seam else lc.C_GPU_F,
                lc.C_BEAT_S if seam else lc.C_GPU_S, rx=6, sw=1.5)
        col = lc.C_BEAT_T if seam else lc.C_GPU_S
        if seam:
            lc.text(x + cw / 2, py + 20, 'kv+attn', 7.8, col, 'middle', True, maxw=cw - 4,
                    tag='sc%d' % i)
            lc.text(x + cw / 2, py + 36, '2 节点·缝', 7, col, 'middle', maxw=cw - 4,
                    tag='scn%d' % i)
        else:
            lc.text(x + cw / 2, py + 21, str(cnt), 12, col, 'middle', True, maxw=cw - 4,
                    tag='cc%d' % i)
            lc.text(x + cw / 2, py + 37, '节点·编译', 7, col, 'middle', maxw=cw - 4,
                    tag='ccn%d' % i)
        if i in rings:
            lc.circle(x + cw / 2, py + 23, 21, lc.C_ABORT, 1.6, dash=True)
            lc.text(x + cw / 2, py + 60, 'kv_update 在内', 7.2, lc.C_ABORT, 'middle', True,
                    maxw=cw, tag='ring%d' % i)

PW = (CW0 - 64) / 2
strip(CX0 + 24, CY0 + 58, PW, 'A · 账本 15 条：kv_update+attention 落接缝（5 片 = 3 编译 + 2 缝）',
      [8, 2, 10, 2, 2], ['c', 's', 'c', 's', 'c'])
strip(CX0 + 40 + PW, CY0 + 58, PW, 'B · 只留 13 条：kv_update 落进编译片 submod_0（9）/ submod_2（11）',
      [9, 1, 11, 1, 2], ['c', 's', 'c', 's', 'c'], rings=(0, 2))
lc.text(CX0 + 16, CY0 + CH0 - 12, 'C · 空账本：整图 1 片、24 节点全编译——attention 的外部副作用进不了图，此路不通（旧死结）。',
        8.3, lc.C_MUTE, 'start', maxw=CW0 - 32, tag='cmp:c')

# 图例
lx = CX0 + CW0 - 330
lc.rect(lx, CY0 + 14, 18, 12, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.2)
lc.text(lx + 24, CY0 + 24, '编译片', 8, lc.C_TXT, 'start', maxw=60, tag='lg1')
lc.rect(lx + 78, CY0 + 14, 18, 12, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
lc.text(lx + 102, CY0 + 24, 'eager 接缝', 8, lc.C_TXT, 'start', maxw=80, tag='lg2')
lc.circle(lx + 196, CY0 + 20, 8, lc.C_ABORT, 1.2, dash=True)
lc.text(lx + 210, CY0 + 24, '危险', 8, lc.C_ABORT, 'start', maxw=40, tag='lg3')

# ---------------- 页脚 ----------------
lc.text(MX, 764, '逐字锚 vllm/config/compilation.py:L762-L778（_attention_ops 13 条清单）· L1149-L1156（full cudagraph 注）· L1177-L1180（KV 写入账注释原话 · issue #33267）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 782, '账本 15 条与两格对照数字取自精简版 companion host 实跑（15=13+2、片 8/2/10/2/2 vs 9/1/11/1/2、kv_update 落 submod_0/submod_2）',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-splitting-ops-account.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
