#!/usr/bin/env python3
"""ch28 机制图 · 一层四本 KV 账(ch28-fig-four-kv-ledgers, 模板 layout)

放大自 L0『显存账本列（青）×GPU 执行臂交界』——ch14/ch25 立的『KV 账本按 spec 分组』
机制在 V4 推到一层×4 的极致形态(L2 下排『四本 KV 账』注的机制版下钻)。

claim: 一个 V4 注意力层最多同时开四本 KV 账——主 MLA(584B/token 潜向量)、IndexerCache
(68B/条 FP4 打分缓存)、CompressorStateCache(fp32 部分和)、SWACache(短窗副本)——后三本
的块大小都被『与 KV 块共享物理张量』的页对齐约束锁死。

数字锚点 = vllm/models/deepseek_v4/attention.py:L207-L213/L655-L674/L782-L810 ·
compressor.py:L155-L203 · v1/attention/backends/mla/sparse_swa.py:L56-L96 ·
v1/kv_cache_interface.py:L409-L424。坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 955
MX = 56
BXR = 1608

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一层四本账：主 MLA 584B/token 之外，还有三本各管一摊的 KV 账', 16, lc.C_TXT,
        'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '每本各有 prefix、各自注册 static_forward_context、各有 builder——账本怎么分组全由模型层 spec 自报（ch14/ch25 站 6 机制在旗舰上的极致形态）',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
_ch = '放大自 L0『显存账本列 × GPU 执行臂交界』· L2 下排注'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左 panel: 一个注意力层的四条账本条 ----------------
LX, LW = MX, 1010
LY0, LH0 = 96, 700
lc.rect(LX, LY0, LW, LH0, '#f8fafc', lc.C_KV_S, rx=10, sw=1.8)
lc.text(LX + 16, LY0 + 24, '一个 V4 注意力层（每层的四本账）', 11.5, '#155e75', 'start', True,
        maxw=600, tag='frame:t')
lc.text(LX + LW - 16, LY0 + 24, '条宽按每条字节数真实比例（584 : 132 : 68）', 9, lc.C_MUTE, 'end',
        maxw=360, tag='frame:s')

BAR_X, BAR_FULL, BAR_H = LX + 150, 520, 30   # 584B → 520px
SCALE = BAR_FULL / 584.0
C_NOPE, C_ROPE, C_SCFILL = '#a5f3fc', '#cffafe', '#ffedd5'
C_NOPE_S, C_SCALES = '#0e7490', '#c2410c'

RY = LY0 + 52
LH_LINE = 156


def ledger(y, idx, name, spec, file, facts, draw_bar, f_dy=62):
    lc.text(LX + 16, y + 16, f'账本 {idx}', 9, lc.C_KV_S, 'start', True, tag='l%d:n' % idx)
    lc.text(LX + 16, y + 34, name, 10.5, lc.C_TXT, 'start', True, maxw=130, tag='l%d:t' % idx)
    lc.text(LX + 16, y + 52, spec, 7.5, lc.C_MUTE, 'start', maxw=132, tag='l%d:s' % idx)
    draw_bar(y)
    for j, s in enumerate(facts):
        lc.text(BAR_X, y + f_dy + j * 15.5, s, 8.5, '#334155', 'start', maxw=LW - 170,
                tag='l%d:f%d' % (idx, j))
    lc.text(BAR_X, y + LH_LINE - 8, file, 8, lc.C_FAINT, 'start', maxw=700, tag='l%d:file' % idx)


# ---- 账本 1: 主 MLA ----
def bar_main(y):
    w_nope, w_rope, w_sc = 448 * SCALE, 128 * SCALE, 8 * SCALE
    x = BAR_X
    for w, fill, stroke, lab in [(w_nope, C_NOPE, C_NOPE_S, '448 B NoPE'),
                                 (w_rope, C_ROPE, C_NOPE_S, '128 B RoPE')]:
        lc.rect(x, y, w, BAR_H, fill, stroke, rx=3, sw=1.3)
        lc.text(x + w / 2, y + 20, lab, 9, '#155e75', 'middle', True, maxw=w - 4, tag='bm:' + lab)
        x += w
    lc.rect(x, y, w_sc, BAR_H, C_SCFILL, C_SCALES, rx=2, sw=1.2)
    lc.text(x + w_sc + 8, y + 20, '8 B fp8 scale', 9, C_SCALES, 'start', True, tag='bm:sc')
    lc.text(BAR_X + BAR_FULL + 10, y - 8, '584 B/token（潜向量一条）', 9.5, '#155e75', 'end',
            True, tag='bm:tot')


ledger(RY, 1, '主 MLA 账', 'MLAAttentionSpec', 'attention.py:L655-L674 · kv_cache_interface.py:L409-L424', [
    'fp8_ds_mla uint8 · alignment 576（plain 布局 512）· storage_block = block_size // compress_ratio',
    '部署真值（DeepSeek-V4-Flash eval）：block-size 256 + kv-cache-dtype fp8 → storage_block = 64',
    'real_page_size_bytes = storage_block × 584 · num_kv_heads=1（潜向量一条，无 K+V 之分）'], bar_main)

# ---- 账本 2: IndexerCache ----


def bar_indexer(y):
    w_val, w_sc = 64 * SCALE, 4 * SCALE
    lc.rect(BAR_X, y, w_val, BAR_H, C_NOPE, C_NOPE_S, rx=3, sw=1.3)
    lc.text(BAR_X + w_val / 2, y + 20, '64 B', 9, '#155e75', 'middle', True, maxw=w_val - 4,
            tag='bi:val')
    lc.text(BAR_X + w_val / 2, y + BAR_H + 11, '↑FP4 打包值（两值一字节）', 7.5, lc.C_MUTE, 'middle',
            maxw=200, tag='bi:valsub')
    lc.rect(BAR_X + w_val, y, w_sc, BAR_H, C_SCFILL, C_SCALES, rx=2, sw=1.2)
    lc.text(BAR_X + w_val + w_sc + 8, y + 20, '4 B UE8M0 尺度', 9, C_SCALES, 'start', True, tag='bi:sc')
    lc.text(BAR_X + BAR_FULL + 12, y - 6, '68 B/条', 10, '#155e75', 'end', True, tag='bi:tot')
    # FP8 对照条（同比例尺）；盲审修复 2026-09-14（第三轮）：下注与 64B 格底边净空仅 0-1px
    # ——下注下移 4px、对照条下移 6px，让下注两侧各留出净空（重渲后像素实测复核）。
    y2 = y + BAR_H + 18
    w8_val, w8_sc = 128 * SCALE, 4 * SCALE
    lc.rect(BAR_X, y2, w8_val, 18, '#e0f2fe', '#0369a1', rx=2, sw=1.1)
    lc.text(BAR_X + w8_val / 2, y2 + 13, '128 B fp8 值', 8, '#075985', 'middle', True, maxw=w8_val - 4,
            tag='bi8:val')
    lc.rect(BAR_X + w8_val, y2, w8_sc, 18, C_SCFILL, C_SCALES, rx=2, sw=1.1)
    lc.text(BAR_X + w8_val + w8_sc + 8, y2 + 13, '4 B fp32 尺度', 8, C_SCALES, 'start', tag='bi8:sc')
    lc.text(LX + LW - 16, y2 + 13, 'FP8 布局对照：132 B/条', 8.5, lc.C_MUTE, 'end',
            tag='bi8:tot')


ledger(RY + LH_LINE, 2, 'IndexerCache', 'MLAAttentionSpec', 'attention.py:L782-L810', [
    'FP4 条目 = head_dim/2=64 打包值 + head_dim/32=4 UE8M0（MXFP4_BLOCK_SIZE=32）',
    'FP8 条目 = 128 fp8 + 4 fp32 尺度（与 V3.2 同布局）· alignment 576（FlashMLA）/512（FlashInfer #44577）',
    '仅 C4A 层建（compress_ratio==4 才建 indexer）· 缓存的是打分原料，不是分数'], bar_indexer,
    f_dy=76)

# ---- 账本 3: CompressorStateCache ----


def bar_compressor(y):
    for k, (n, bs, lab) in enumerate([('C4', 4, '块 4'), ('C128', 8, '块 8')]):
        x = BAR_X + k * 250
        for i in range(bs):
            lc.rect(x + i * 18, y, 16, BAR_H - 8, '#bae6fd', '#0369a1', rx=2, sw=1.1)
        lc.text(x + bs * 18 + 10, y + 12, lab + '（compress_ratio=%s）' % ('4' if bs == 4 else '128'),
                9, '#075985', 'start', True, maxw=160, tag='bc:%s' % n)
        lc.text(x + bs * 18 + 10, y + 26, 'fp32 状态向量', 8, lc.C_MUTE, 'start', tag='bc:%s:s' % n)
    lc.text(BAR_X + 560, y + 12, 'SlidingWindowMLASpec', 9, '#155e75', 'start', True, tag='bc:spec')
    lc.text(BAR_X + 560, y + 26, '（只有一根状态向量）', 8, lc.C_MUTE, 'start', tag='bc:spec:s')


ledger(RY + 2 * LH_LINE, 3, 'Compressor 账', 'CompressorStateCache', 'compressor.py:L155-L203', [
    'assert dtype==float32 · compress_ratio∈{4,128} · C4 块 4 / C128 块 8',
    '注释原话：与 KV 块共享同一物理张量、必须同页大小——KV 块形状 [256//4, head_dim]=[64, 584] 定死',
    'C4 块形 [4, 2·512·2·4] / C128 块形 [8, 512·2·4]（记法出自同一条注释；第一维=块大小 4/8，被页对齐锁死）'],
    bar_compressor)

# ---- 账本 4: SWACache ----


def bar_swa(y):
    x = BAR_X
    # 盲审修复 2026-09-14（第三轮）：账本 3 的条按「格数=块大小」画（4 格=块 4、8 格=块 8），
    # 本条画 12 格却标『块 64』易被误读为 12 token/块——保持 12 格但在尾部加省略号、
    # 标签明写『格数示意』，与账本 3 的严格口径显式区分（64 格画不下）。
    for i in range(12):
        lc.rect(x + i * 22, y, 20, BAR_H - 8, '#bae6fd', '#0369a1', rx=2, sw=1.1)
    lc.text(x + 12 * 22 + 2, y + 15, '…', 11, lc.C_MUTE, 'start', tag='bs:ell')
    lc.text(x + 12 * 22 + 26, y + 12, '块 64（64 token/块·格数示意）', 9, '#075985', 'start',
            True, maxw=185, tag='bs:lab')
    lc.text(x + 12 * 22 + 26, y + 26, '滑窗短窗副本', 8, lc.C_MUTE, 'start', tag='bs:s')
    lc.text(x + 12 * 22 + 254, y + 12, 'SlidingWindowMLASpec', 9, '#155e75', 'start', True, tag='bs:spec')


ledger(RY + 3 * LH_LINE, 4, 'SWACache', 'SlidingWindowMLASpec', 'v1/attention/backends/mla/sparse_swa.py:L56-L96', [
    '同一页对齐约束：与 C4A KV 块共享物理张量 → 块 64 由 [64, head_dim] 的 KV 块形状定',
    'dtype 三布局：uint8（fp8_ds_mla）/ bfloat16 / float8_e4m3fn · SWA 段短窗副本，单独分配'], bar_swa)

# ---------------- 右 panel: 三类层开哪几本 ----------------
RX, RW_ = 1100, BXR - 1100
TY, TH = 96, 330
lc.rect(RX, TY, RW_, TH, '#ffffff', lc.C_KV_S, rx=9, sw=1.6)
lc.text(RX + 16, TY + 24, '三类层各开哪几本（compress_ratio 逐层）', 11, '#155e75', 'start', True,
        maxw=RW_ - 130, tag='tbl:t')
lc.text(RX + RW_ - 14, TY + 24, 'attention.py:L207-L213', 8, lc.C_FAINT, 'end', maxw=170, tag='tbl:f')
col_x = [RX + 16, RX + 130, RX + 214, RX + 306, RX + 394, RX + 470]
hdrs = ['层类型', '主 MLA', 'Indexer', 'Compr.', 'SWA', '合计']
# 盲审修复 2026-09-14（第四轮，出框定点修）：末列（合计）标签比另两行长一截
# （「四本全开」4 字 vs「三本」「一本」2 字），左对齐时墨迹 x=1590→1625 越过面板右边框
# x=1608——青边框从「本」「全」之间横穿该行、末两字整字出框（@2x 实测出框 34px，
# 且该行浅蓝底卡止于 @2x x=3196，末两字同时脱卡；另两行标签止于 @2x x≈3215-3217，0px 净空贴线）。
# 修法（改动面最小）：末列改右对齐、右缘钉死在 TOT_RIGHT = 面板右缘内 18 单位
# （@2x = 36px 净空 ≥ 盲审要求的 10px；行底卡右缘 = RX+RW_-10，字缘再内收 8 单位）。
# 列内文字/行数/字号零改动，其余列仍左对齐。
TOT_RIGHT = RX + RW_ - 18


def col_tx(j):
    """列内文字发射 x：末列（j==5，右对齐）给右缘锚点，其余给左对齐起点。"""
    return TOT_RIGHT if j == 5 else col_x[j] + (0 if j == 0 else 20)


for j, htxt in enumerate(hdrs):
    lc.text(col_tx(j), TY + 48, htxt, 8.5, lc.C_MUTE, 'end' if j == 5 else 'start', True,
            tag='tbl:h%d' % j)
# 盲审修复 2026-09-14（第三轮）：在册标记 ✓(U+2713) 在本机字体栈渲染成空心豆腐
# （.notdef）——换 GB2312 字集内的 √（本章 fp4 图同款、已过盲审）。
rows = [('C4A（cr=4）', '√', '√', '√', '√', '四本全开'),
        ('C128（cr=128）', '√', '—', '√', '√', '三本'),
        ('C1（cr≤1）', 'None', '—', '—', '√', '一本')]
for i, row in enumerate(rows):
    yy = TY + 72 + i * 26
    if i % 2 == 0:
        lc.rect(RX + 10, yy - 14, RW_ - 20, 24, '#f0f9ff', 'none', rx=4, sw=0.5)
    for j, cell in enumerate(row):
        color = '#334155' if j == 0 else (lc.C_KV_S if cell == '√' else lc.C_FAINT)
        lc.text(col_tx(j), yy, cell, 9, color, 'end' if j == 5 else 'start',
                bold=(j in (0, 5)), tag='tbl:r%d c%d' % (i, j))
lc.text(RX + 16, TY + 168, '· cr≤1 的层主账返回 None（滑窗账单独分配为', 8.5, '#334155', 'start',
        maxw=RW_ - 32, tag='tbl:n0')
lc.text(RX + 16, TY + 184, '   DeepseekV4SWACache，构造期恒建）', 8.5, '#334155', 'start',
        maxw=RW_ - 32, tag='tbl:n1')
lc.text(RX + 16, TY + 208, '· indexer 只在 compress_ratio==4 的层建', 8.5, '#334155', 'start',
        maxw=RW_ - 32, tag='tbl:n2')
lc.text(RX + 16, TY + 228, '   （复用 aux_stream_list[2]）', 8.5, '#334155', 'start',
        maxw=RW_ - 32, tag='tbl:n3')
lc.text(RX + 16, TY + 252, '· compressor 在 compress_ratio>1 的层建', 8.5, '#334155', 'start',
        maxw=RW_ - 32, tag='tbl:n4')
lc.text(RX + 16, TY + 288, '部署真值：DeepSeek-V4-Flash eval', 9, C_SCALES, 'start', True,
        maxw=RW_ - 32, tag='tbl:d0')
lc.text(RX + 16, TY + 306, 'block-size 256 + fp8 + use_fp4_indexer_cache=True', 8.5, '#334155',
        'start', maxw=RW_ - 32, tag='tbl:d1')

# 右下: 机制注（框高 222：盲审修复 2026-09-14——原 210 的底边虚线(y=654)横穿末行橙字
# 「后三本的页大小不是自己选的」(baseline 652)的字底笔画；加高后底边 666，净空 ~10px）
MY = TY + TH + 18
lc.rect(RX, MY, RW_, 222, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(RX + 16, MY + 22, 'spec 自报 → 账本分组', 11, lc.C_TXT, 'start', True, maxw=300, tag='mech:t')
for i, s in enumerate([
        '· 四本各有 prefix、各自注册', '   static_forward_context（重复名即 raise）',
        '· 四本各有 builder（各自后端：', '   FlashMLA/DeepseekV4IndexerBackend/',
        '   CompressorBackend/SWA）',
        '· 账本怎么分组，全由模型层 spec', '   自报——不是调度器规定的',
        '· 分组机制归 ch14/ch25 站 6：', '   每组同 spec 才能共享页池']):
    lc.text(RX + 16, MY + 44 + i * 17, s, 8.5, '#334155', 'start', maxw=RW_ - 32, tag='mech:l%d' % i)
lc.text(RX + 16, MY + 208, '后三本的页大小不是自己选的', 9, C_SCALES, 'start', True, maxw=RW_ - 32,
        tag='mech:hot')

# ---------------- 页脚 ----------------
lc.text(MX, 826, '图例：青系 = KV / 显存账本角色 · 橙段 = 尺度字节（fp8 scale / UE8M0 / fp32）· 白格 = 状态/短窗格 · 主条与 Indexer 条宽按真实字节比例',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 846, '锚点 = vllm/models/deepseek_v4/attention.py:L207-L213/L655-L674/L782-L810 · compressor.py:L155-L203 · v1/attention/backends/mla/sparse_swa.py:L56-L96 · v1/kv_cache_interface.py:L409-L424',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')
lc.text(MX, 866, '部署参数出自树内 DeepSeek-V4-Flash eval 配置（GSM8K · mega-moe）· compress_ratio∈{1,4,128} 的断言与逐层表 = attention.py:L207-L213/L656-L659',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-four-kv-ledgers.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
