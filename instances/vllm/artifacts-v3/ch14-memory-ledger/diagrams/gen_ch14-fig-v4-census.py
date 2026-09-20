#!/usr/bin/env python3
"""ch14 机制图 · V4 census:7 形态收进 4 页宽(figure_spec ch14-fig-v4-census,模板 tiling)

替换 ch14-fig-v4-ledger-census(旧图随现稿同错:主张六形态、漏 indexer_state 与 c1 层)。
按 7 形态 + 同页配对重画:7 种缓存形态只占 4 种页宽,同页是模型侧显存复用的设计。
配 ch14-fig-v4-layer-anatomy:那张拆一层,这张看全模型页宽账。

数字全部取自 figure_spec.numbers(provenance = traces/v4_cache_groups.json
census_toy.page_census_detail + groups_11_10.page_counts + sparse_swa.py:L77-L82)。
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
MLA_S = lc.C_API_S                                  # MLAAttentionSpec 族 = 蓝
SWA_S = lc.C_KV_S                                   # SlidingWindowMLASpec 族 = 青
AMBER = '#b45309'                                   # 刻意同页的设计证据
HL_S, HL_F = lc.C_ENG_S, lc.C_ENG_F                 # 主角行(37440 刻意同页)= 橙高亮

# ---------------- 标题区 ----------------
lc.text(MX, 34, '7 种缓存形态只占 4 种页宽——同页是显存复用的设计,不是待统一的麻烦',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'DeepSeek V4 全模型 census(站 5 组化的 census 侧,配上一张的层解剖):同页配对把页宽种数从 7 压到 4',
        10.5, lc.C_MUTE, 'start', maxw=1000, tag='subtitle')
_ch = 'L0 放大 · KV 账本列 · 站 5 census'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶部 7 → 4 徽标带 ----------------
TB_Y, TB_H = 80, 34
lc.rect(MX, TB_Y, 250, TB_H, '#ffffff', MLA_S, rx=8, sw=1.4)
lc.text(MX + 125, TB_Y + 22, '7 种缓存形态', 12, MLA_S, 'middle', True, tag='tb:l')
lc.text(MX + 285, TB_Y + 22, '同页配对', 10.5, AMBER, 'middle', True, tag='tb:m')
lc.seg(MX + 258, TB_Y + TB_H / 2, MX + 322, TB_Y + TB_H / 2, AMBER, 1.8, marker='std')
lc.rect(MX + 340, TB_Y, 250, TB_H, HL_F, HL_S, rx=8, sw=1.4)
lc.text(MX + 465, TB_Y + 22, '4 种页宽', 12, HL_S, 'middle', True, tag='tb:r')
lc.text(BXR, TB_Y + 22, '蓝 = MLAAttentionSpec 族(3 形态)· 青 = SlidingWindowMLASpec 族(4 形态)',
        9.2, lc.C_MUTE, 'end', maxw=560, tag='tb:leg')

# ---------------- 主面板:四行配对表 ----------------
PY0 = TB_Y + TB_H + 14
PH = 496
LX, LW = MX, BXR - MX
lc.rect(LX, PY0, LW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, PY0 + 22, '每行一档页宽:左列住户(形态)→ 汇聚成同一种页宽 → 右列官方算例页宽计数',
        10.5, lc.C_TXT, 'start', True, maxw=LW - 32, tag='mp:t')
lc.text(LX + 16, PY0 + 40, '计数取官方算例形状 11 c4 + 10 c128(kv_cache_utils.py:L1693-L1697,共 85 条 spec 的页宽分布)',
        8.8, lc.C_MUTE, 'start', maxw=LW - 32, tag='mp:s')

# 行数据:(住户[(名, 家族色)], 页宽, 行注, 计数, 计数细分, 是否刻意同页高亮)
ROWS = [
    ([('滑窗缓存 swa', SWA_S), ('主 KV main(r=4)', MLA_S)], 37440,
     '刻意同页:同一张物理张量——C4A 块形 [256//4, head_dim] 反定滑窗块 64 token(sparse_swa.py:L77-L82)',
     '×32(11 主 KV c4 + 21 滑窗)', True),
    ([('indexer k_cache', MLA_S), ('indexer 压缩器状态 indexer_state(r=4)', SWA_S)], 8640,
     'indexer 与其内嵌的压缩器状态同页',
     '×22(11 indexer + 11 状态)', False),
    ([('注意力压缩器状态 attn_state(r=4)', SWA_S), ('attn_state(r=128)', SWA_S)], 32832,
     '两族注意力压缩器状态同页',
     '×21(11 + 10 两族注意力状态)', False),
    ([('主 KV main(r=128)', MLA_S)], 1728,
     '独门独户——c128 主 KV 独占一档',
     '×10', False),
]
R0, PITCH = PY0 + 62, 106
CELL_W, CELL_H = 252, 40
PG_X, PG_W = LX + 320, 130                          # 页宽徽标列
CT_X = LX + 1050                                    # 计数列
for i, (tenants, page, note, count, hot) in enumerate(ROWS):
    ry = R0 + i * PITCH
    if hot:
        lc.rect(LX + 10, ry - 4, LW - 20, PITCH - 8, HL_F, 'none', rx=6, sw=0)
    n = len(tenants)
    for j, (name, fam) in enumerate(tenants):
        cy = ry + 26 + (j - (n - 1) / 2) * (CELL_H + 8) - CELL_H / 2
        lc.rect(LX + 26, cy, CELL_W, CELL_H, '#ffffff', fam, rx=6, sw=1.4)
        lc.text(LX + 26 + CELL_W / 2, cy + CELL_H / 2 + 4, name, 9.2, fam, 'middle', True,
                maxw=CELL_W - 14, tag='tn:%d-%d' % (i, j))
        # 住户 → 页宽徽标(汇聚箭头)
        lc.seg(LX + 26 + CELL_W, cy + CELL_H / 2, PG_X - 3, ry + 26, fam, 1.5, marker='std')
    # 页宽徽标
    lc.rect(PG_X, ry + 6, PG_W, 40, '#ffffff', AMBER if hot else lc.C_MUTE, rx=6,
            sw=2.0 if hot else 1.4)
    lc.text(PG_X + PG_W / 2, ry + 31, f'{page} B', 12.5, AMBER if hot else lc.C_TXT, 'middle',
            True, tag='pg:%d' % page)
    # 行注 + 计数
    lc.text(PG_X + PG_W + 16, ry + 22, note, 8.6, AMBER if hot else lc.C_MUTE, 'start',
            maxw=CT_X - (PG_X + PG_W + 16) - 16, tag='nt:%d' % i)
    lc.rect(CT_X, ry + 6, 118, 40, '#ffffff', lc.C_MUTE, rx=6, sw=1.2)
    lc.text(CT_X + 59, ry + 31, count.split('(')[0].strip(), 12, lc.C_TXT, 'middle', True,
            maxw=110, tag='ct:%d' % i)
    if '(' in count:
        lc.text(CT_X + 128, ry + 31, count[count.index('('):], 8.2, lc.C_MUTE, 'start',
                maxw=BXR - 24 - (CT_X + 128), tag='cts:%d' % i)

# ---------------- 底注 ----------------
FY = PY0 + PH + 12
lc.rect(MX, FY, BXR - MX, 58, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 16, FY + 19, '页宽多样性是 tensor sharing 的刻意设计——等页 unify 的三条出路(调大较小层 bs / pad 到公共页 / 拒收)一条都接不住它;'
        '7 形态若各占一页宽,装包要面对 7 种页宽,同页设计把它压到 4',
        8.8, '#334155', 'start', maxw=BXR - MX - 32, tag='bn:0')
lc.text(MX + 16, FY + 38, '页宽与计数出自 pin 真码 host 桩跑的 census(5 层玩具表 + 官方算例 11 c4 + 10 c128)· '
        '同页依据逐字锚 vllm/v1/attention/backends/mla/sparse_swa.py:L77-L82 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX - 32, tag='bn:1')

# ---------------- 装配输出 ----------------
H = int(FY + 58 + 14)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-v4-census.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
