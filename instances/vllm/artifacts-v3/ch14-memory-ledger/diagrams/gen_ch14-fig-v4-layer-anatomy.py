#!/usr/bin/env python3
"""ch14 机制图 · V4 层解剖:一个 CSA 层向账本报的缓存户口(figure_spec ch14-fig-v4-layer-anatomy,模板 layout)

站 5(混合组化)的放大:把一个 c4 层拆开看它向账本报的缓存户口。
左:CSA 层三股流(滑窗 token 流 / 压缩主 KV 流 / indexer 检索流);
右:五挂缓存拓扑(两种刻意同页 37440 / 8640 + 32832);
底:户口数由 compress_ratio 唯一决定(c1/c4/c128 = 1/5/3 类)。

数字全部取自 figure_spec.numbers(provenance = traces/v4_cache_groups.json
census_toy + attention.py:L655-L659)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
MLA_S, MLA_F = lc.C_API_S, lc.C_API_F      # MLAAttentionSpec 族 = 蓝(full 管家,与全书同色)
SWA_S, SWA_F = lc.C_KV_S, lc.C_KV_F        # SlidingWindowMLASpec 族 = 青(滑窗管家,与全书同色)
AMBER = '#b45309'                          # 刻意同页的设计证据标注(虚线连线)

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'CSA 层(c4)向账本报 5 类缓存——户口数由 compress_ratio 唯一决定',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '滑窗流、压缩主 KV 流、indexer 检索流三股各挂各的账,外加注意力压缩器与 indexer 压缩器两份滚动状态;'
                'c128(HCA)去掉 indexer 两挂剩 3 类、纯滑窗层(c1,含 MTP)只报 1 类',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = 'L0 放大 · KV 账本列 · 站 5 混合组化'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

PY0, PH = 92, 430
LX, LW = MX, 620
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)

# ---------------- 左面板:三股流 ----------------
lc.rect(LX, PY0, LW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, PY0 + 22, '① 数据视角 · CSA 层的三股流', 11.5, lc.C_TXT, 'start', True,
        maxw=LW - 32, tag='lp:t')
lc.text(LX + 16, PY0 + 40, 'c4 = 每 4 个 token 压一份(vLLM 工程名 c4a;CSA = 压缩稀疏注意力,DeepSeek 模型卡)',
        9.0, lc.C_MUTE, 'start', maxw=LW - 32, tag='lp:s')

# 输入 token 流
TK_Y, TK_X0, TK_N, TK_W, TK_H = PY0 + 62, LX + 52, 16, 28, 20
lc.text(LX + 16, TK_Y - 2, '输入 token 流', 8.8, lc.C_MUTE, 'start', maxw=120, tag='tk:lbl')
for i in range(TK_N):
    near = i >= TK_N - 6                      # 尾部 6 格 = 近窗原文
    fill = lc.C_KV_F if near else '#ffffff'
    lc.rect(TK_X0 + i * TK_W, TK_Y, TK_W - 3, TK_H, fill, lc.C_MUTE, rx=2, sw=0.8)
lc.text(TK_X0 + TK_N * TK_W + 6, TK_Y + 13, '…', 10, lc.C_MUTE, 'start', tag='tk:ell')
lc.text(TK_X0 + (TK_N - 6) * TK_W + (6 * TK_W - 3) / 2, TK_Y - 6, '近窗', 8.2, SWA_S, 'middle', True,
        maxw=60, tag='tk:near')

# 三股流框(左边留走线槽,分发干线不穿框)
TRUNK_X = LX + 30                               # 分发干线 x(流框左侧外)
FL_Y0, FL_H, FL_GAP = TK_Y + TK_H + 34, 74, 14
FLOWS = [
    ('① 滑窗流:近窗原文直存', '窗内 token 不压缩,原样进滑窗缓存——粗粒度读近处的便宜路径',
     SWA_S, SWA_F),
    ('② 压缩主 KV 流:每 r=4 个 token 压一份', '压缩后 KV 入主 KV 缓存——远处的细粒度记忆全靠它',
     MLA_S, MLA_F),
    ('③ indexer 检索流:在压缩键上打分', '给压缩 KV 建索引、挑 top 条目——读远处的检索入口',
     MLA_S, MLA_F),
]
fl_cy = []
for i, (t, s, cs, cf) in enumerate(FLOWS):
    fy = FL_Y0 + i * (FL_H + FL_GAP)
    lc.rect(LX + 52, fy, LW - 72, FL_H, cf, cs, rx=7, sw=1.4)
    lc.text(LX + 66, fy + 24, t, 10.2, cs, 'start', True, maxw=LW - 96, tag='fl%d:t' % i)
    lc.text(LX + 66, fy + 44, s, 8.6, lc.C_MUTE, 'start', maxw=LW - 96, tag='fl%d:s' % i)
    fl_cy.append(fy + FL_H / 2)
# token 流 → 三股:左侧走线槽的树状分发(干线 + 三条支线,支线贴流框左边)
lc.parrow([(TK_X0 + 6 * TK_W, TK_Y + TK_H), (TK_X0 + 6 * TK_W, TK_Y + TK_H + 12),
           (TRUNK_X, TK_Y + TK_H + 12), (TRUNK_X, fl_cy[2])], lc.C_MUTE, 1.3, marker=None)
for i in range(3):
    lc.seg(TRUNK_X, fl_cy[i], LX + 50, fl_cy[i], FLOWS[i][2], 1.4, marker='std')
# 面板底注
lc.text(LX + 16, FL_Y0 + 3 * (FL_H + FL_GAP) + 8,
        '三股流各挂各的账:一股滑窗 + 两股压缩(indexer 本身还带滚动状态)——一层之内就已是混合缓存',
        8.6, '#334155', 'start', maxw=LW - 32, tag='lp:n0')

# ---------------- 右面板:五挂缓存户口 ----------------
lc.rect(RX, PY0, RW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, PY0 + 22, '② 账本视角 · 一个 c4 层自报的五挂缓存', 11.5, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='rp:t')
lc.text(RX + 16, PY0 + 40, '每挂 = 一条 KVCacheSpec · 页宽 = storage_block_size × 每槽字节 → 576 对齐 · 蓝 = MLAAttentionSpec 族 · 青 = SlidingWindowMLASpec 族',
        8.8, lc.C_MUTE, 'start', maxw=RW - 32, tag='rp:s')

# 行数据:(名称, 家族色, 页宽, spec 注)
ROWS = [
    ('滑窗缓存 swa', SWA_S, 37440, 'SlidingWindowMLASpec · 64-token 块'),
    ('主 KV(压缩后)', MLA_S, 37440, 'MLAAttentionSpec · 管理块 256 / 物理 64 行'),
    ('indexer k_cache', MLA_S, 8640, 'MLAAttentionSpec · 索引器检索 KV'),
    ('indexer 压缩器状态', SWA_S, 8640, 'SlidingWindowMLASpec · indexer 内嵌滚动状态'),
    ('注意力压缩器状态', SWA_S, 32832, 'SlidingWindowMLASpec · 两族(r=4 / r=128)同页'),
]
PG_X0, PG_W, PG_H = RX + 178, 86, 17
R0, PITCH, PAIR_EXTRA = PY0 + 66, 47, 15      # 同页对(行 0/1、行 2/3)之间加高放标注
row_y = [R0]
for i in range(1, len(ROWS)):
    row_y.append(row_y[-1] + PITCH + (PAIR_EXTRA if i in (1, 3) else 0))
for (name, fam, page, note), ry in zip(ROWS, row_y):
    lc.text(RX + 16, ry + 15, name, 9.4, fam, 'start', True, maxw=158, tag='row:' + name[:8])
    lc.rect(PG_X0, ry + 3, PG_W, PG_H, '#ffffff', fam, rx=3, sw=1.1)
    lc.text(PG_X0 + PG_W / 2, ry + 15.5, f'页 {page} B', 8.8, fam, 'middle', True, maxw=PG_W - 6,
            tag='pg:%d' % page)
    lc.text(PG_X0 + PG_W + 10, ry + 15, note, 8.2, lc.C_MUTE, 'start',
            maxw=RX + RW - 16 - (PG_X0 + PG_W + 10), tag='nt:%d' % page)
# 两处刻意同页连线(37440 对、8640 对)
PE = PG_X0 + PG_W
for (a, b, lab) in [(0, 1, '刻意同页 37440 · 同一张物理张量'), (2, 3, '同页 8640 · indexer 与其压缩器状态')]:
    lc.seg(PE + 4, row_y[a] + 3 + PG_H, PE + 4, row_y[b] + 3, AMBER, 1.5, dash=True)
    lc.text(PE + 12, (row_y[a] + row_y[b]) / 2 + 14, lab, 8.4, AMBER, 'start', True,
            maxw=RX + RW - 16 - (PE + 12), tag='pair:%d' % a)
# 底注两行
for j, note in enumerate([
    '同页是模型侧显存复用的设计:C4A 块形 [256//4, head_dim] 反定滑窗块 64 token(sparse_swa.py:L77-L82)',
    '全模型 census 视角(7 形态 → 4 页宽)见下一图;压缩块的记账口径再下一图',
]):
    lc.text(RX + 16, row_y[-1] + 46 + j * 15, note, 8.4, '#334155', 'start', maxw=RW - 32,
            tag='rp:n%d' % j)

# ---------------- 底部:户口数由 compress_ratio 唯一决定 ----------------
BY = PY0 + PH + 14
lc.text(MX, BY + 13, '③ 户口数由 compress_ratio 唯一决定(合法压缩比仅 {1, 4, 128})', 10.5, lc.C_TXT,
        'start', True, maxw=760, tag='bt:t')
lc.text(BXR, BY + 13, '逐层 census = compress_ratio 表的确定函数', 8.6, lc.C_FAINT, 'end',
        maxw=380, tag='bt:r')
CB_Y, CB_H, CB_GAP = BY + 22, 104, 14
CB_W = (BXR - MX - 2 * CB_GAP) / 3
CARDS = [
    ('c≤1 层(含 MTP)· 1 类', ['get_kv_cache_spec 返回 None——主 KV 不报账,',
                             '唯一缓存面是 SWA:只剩滑窗一挂(attention.py:L655-L659)'], SWA_S, SWA_F),
    ('c4 层(CSA)· 5 类', ['左表五挂:swa + 主 KV(同页 37440)+ indexer +',
                          'indexer 压缩器状态(同页 8640)+ 注意力压缩器状态(32832)'], MLA_S, MLA_F),
    ('c128 层(HCA)· 3 类', ['同构去 indexer 两挂:swa + 主 KV(页 1728 独占)',
                            '+ 注意力压缩器状态(与 c4 状态同页 32832)'], SWA_S, SWA_F),
]
for i, (t, lines, cs, cf) in enumerate(CARDS):
    bx = MX + i * (CB_W + CB_GAP)
    lc.rect(bx, CB_Y, CB_W, CB_H, cf, cs, rx=7, sw=1.5)
    lc.text(bx + 14, CB_Y + 22, t, 10.0, cs, 'start', True, maxw=CB_W - 28, tag='cb%d:t' % i)
    for j, s in enumerate(lines):
        lc.text(bx + 14, CB_Y + 42 + j * 15, s, 8.4, lc.C_MUTE, 'start', maxw=CB_W - 28,
                tag='cb%d:s%d' % (i, j))

# ---------------- 页脚 ----------------
FY = CB_Y + CB_H + 12
lc.text(MX, FY + 12, '数字出自 pin 真码 host 桩跑的逐层 census(官方回归测试的 5 层玩具表)· '
        '逐字锚 vllm/models/deepseek_v4/attention.py:L655-L659(c≤1 返回 None)· '
        'vllm/v1/attention/backends/mla/sparse_swa.py:L77-L82(同页 37440)',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot0')
lc.text(MX, FY + 28, 'CSA/HCA/mHC = 压缩稀疏注意力 / 重压缩注意力 / 流形约束超连接(DeepSeek 模型卡;'
        'vLLM 工程名 c4a/c128a,两套名字指同一东西)· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
H = int(FY + 28 + 20)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-v4-layer-anatomy.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
