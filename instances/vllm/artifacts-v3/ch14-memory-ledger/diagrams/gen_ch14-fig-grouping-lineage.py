#!/usr/bin/env python3
"""ch14 机制图 · KVCache 分组三形态前史(figure_spec ch14-fig-grouping-lineage,模板 flow)

站 5(混合组化)的前史带:回答「这些组形态从哪来」。
时间线三段演进:全同 spec 组 → 类型统一组 UniformTypeKVCacheSpecs → 混合 spec 组 HMA;
再往前 V0 手工补丁 → V1 统一分配器;本章 V4 第三路 = 类型统一组的 DeepSeek 特化。

数字(年份 / PR 号 / 比例)全部照 figure_spec.numbers 抄录——社区史料
(provenance = 研究包条目,一手出处),非 pin 实跑;不自行新增任何年份或 PR 号。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
HL_S, HL_F = lc.C_ENG_S, lc.C_ENG_F        # V4 落点(本章主角)= 橙
DOT_C = {'plain': lc.C_MUTE, 'hot': HL_S}

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'KVCache 分组三形态:被模型潮推着长出来', 16.5, lc.C_TXT, 'start', True,
        maxw=820, tag='title')
lc.text(MX, 58, '全同 spec 组(多数模型)→ 类型统一组(同型不同宽)→ 混合 spec 组 HMA(跨类型共享物理 buffer)'
                '——形态是被需求推着长的,不是一次设计出来的',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = 'L0 放大 · KV 账本列 · 站 5 前史'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 三栏演进卡片 ----------------
CY0, CH_ = 84, 236
CW = (BXR - MX - 2 * 18) / 3
CARDS = [
    ('形态 ① 全同 spec 组', 'identical KVCacheSpec', [
        '所有层页形状一模一样——多数模型',
        '无需任何混合机制,全层一组',
        '代表:Qwen2.5、DeepSeek V3.1',
    ], lc.C_MUTE),
    ('形态 ② 类型统一组', 'UniformTypeKVCacheSpecs(PR #25101)', [
        '同是注意力缓存但页宽不同,打包成',
        '一个分配单位。需求来源:MTP 投机',
        '层 + MiniCPM 4.1 小页 indexer',
        '(V3.2 indexer 同款问题)',
    ], lc.C_API_S),
    ('形态 ③ 混合 spec 组', 'HMA(混合内存分配器)', [
        '跨类型层共享同一物理 buffer。代表:',
        'Gemma3、Qwen3-Next(full:linear = 1:3,',
        'linear 层每请求 N=1 块——朴素分配',
        '放大浪费)。多块池方案被否:前缀缓存',
        '让 linear 层峰值块需求不可预测——',
        '「无法预知运行时会有多少个 block 被命中」',
    ], lc.C_KV_S),
]
for i, (t, sub, lines, cs) in enumerate(CARDS):
    cx = MX + i * (CW + 18)
    lc.rect(cx, CY0, CW, CH_, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
    lc.rect(cx, CY0, CW, 6, cs, cs, rx=3, sw=0)
    lc.text(cx + 16, CY0 + 28, t, 11.5, cs, 'start', True, maxw=CW - 32, tag='cd%d:t' % i)
    lc.text(cx + 16, CY0 + 46, sub, 8.8, lc.C_MUTE, 'start', maxw=CW - 32, tag='cd%d:s' % i)
    for j, ln in enumerate(lines):
        lc.text(cx + 16, CY0 + 70 + j * 16.5, ln, 8.8, '#334155', 'start', maxw=CW - 32,
                tag='cd%d:l%d' % (i, j))
    if i < 2:
        lc.seg(cx + CW + 2, CY0 + CH_ / 2, cx + CW + 16, CY0 + CH_ / 2, lc.C_MUTE, 2.0,
               marker='std')

# ---------------- 时间线 ----------------
TLY = CY0 + CH_ + 46
lc.text(MX, TLY - 14, '时间线(社区史料,一手出处——日期与 PR 号照研究包条目录入,非 pin 实跑)', 10.5,
        lc.C_TXT, 'start', True, maxw=760, tag='tl:t')
AXIS_Y = TLY + 24
lc.seg(MX + 10, AXIS_Y, BXR - 10, AXIS_Y, lc.C_MUTE, 1.6)
NODES = [
    ('2024-11', 'Marconi 论文', ['诊断混合前缀', '缓存难题'], 'plain'),
    ('2025-09-09', 'PR #24486', ['kernel 块大小与', 'KV 页大小解耦'], 'plain'),
    ('2025-09-17', 'PR #25101', ['UniformTypeKVCacheSpecs', 'MTP + MiniCPM 4.1'], 'plain'),
    ('2025-11-05', '混合模型一等公民', ['V0 hack 退役、', 'V1 统一分配器'], 'plain'),
    ('2026-04-24', 'DSV4 双落地', ['压缩器状态伪装滑窗 spec', '复用混合管理全家桶'], 'hot'),
]
NW = (BXR - MX - 40) / (len(NODES) - 1)
for i, (date, name, desc_lines, kind) in enumerate(NODES):
    nx = MX + 20 + i * NW
    cs = DOT_C[kind]
    lc.circle(nx, AXIS_Y, 5, cs, sw=1.8, dash=False)
    if kind == 'hot':
        lc.circle(nx, AXIS_Y, 9, cs, sw=1.0, dash=True)
    lc.text(nx, AXIS_Y - 16, date, 9.4, cs, 'middle', True, maxw=NW - 8, tag='nd%d:d' % i)
    lc.text(nx, AXIS_Y + 24, name, 9.2, lc.C_TXT, 'middle', True, maxw=NW - 8, tag='nd%d:n' % i)
    for j, chunk in enumerate(desc_lines):
        lc.text(nx, AXIS_Y + 40 + j * 13, chunk, 7.8, lc.C_MUTE, 'middle', maxw=NW + 4,
                tag='nd%d:c%d' % (i, j))

# ---------------- 前史条 ----------------
FY = AXIS_Y + 78
lc.rect(MX, FY, BXR - MX, 74, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 16, FY + 19, '前史 · V0 手工补丁:状态存独立张量、按 max_num_seqs 猜并发——猜大 OOM、猜小排队',
        9.4, lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='fh:t')
lc.text(MX + 16, FY + 38, '→ V1 统一分配器落地,对混合模型一次解锁三件事:前缀缓存 / KV 传输 / PD 分离',
        9.0, '#334155', 'start', maxw=BXR - MX - 32, tag='fh:s')
lc.text(MX + 16, FY + 56, '(「旧世界只有一张表一种类型」的实证出处:PyTorch 官方博客 2025-11-05——V0 hack 的具体形态)',
        8.0, lc.C_FAINT, 'start', maxw=BXR - MX - 32, tag='fh:f')

# ---------------- V4 落点条 ----------------
VY = FY + 88
lc.rect(MX, VY, BXR - MX, 64, HL_F, HL_S, rx=7, sw=1.6)
lc.text(MX + 16, VY + 21, '本章落点 · V4 的第三条分组路径:形态 ② 的打包容器 × 形态 ③ 的跨类型共池', 10.8, HL_S,
        'start', True, maxw=900, tag='v4:t')
lc.text(MX + 16, VY + 42, '四种页宽装进多个 UniformType 组——后文 V4 节的 5 组正是它;压缩器状态伪装滑窗 spec,'
        '前缀缓存 / CUDA graphs / MTP 全部免费复用同一抽象',
        8.8, '#9a3412', 'start', maxw=BXR - MX - 32, tag='v4:s')

# ---------------- 页脚 ----------------
PY = VY + 78
lc.text(MX, PY + 12, '日期 / PR 号 / 模型比例(Qwen3-Next full:linear = 1:3、每请求 N=1 块)全部出自社区研究包一手条目'
        '(博客 / GitHub PR / 论文)· 分组三形态口径:社区深读 + PyTorch 官方博客 · 分组代码以 pin vLLM v0.27.1 为准',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
H = int(PY + 12 + 18)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-grouping-lineage.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
