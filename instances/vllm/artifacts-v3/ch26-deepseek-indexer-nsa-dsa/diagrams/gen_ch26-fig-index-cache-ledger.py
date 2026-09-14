#!/usr/bin/env python3
"""ch26 机制图 · IndexCache 第二本账(figure ch26-fig-index-cache-ledger,模板 layout)

放大自 L0『模型层 indexer 框』——L2 拍片② 的 IndexCache 池(站 4)展开:与主 KV cache
并排的两本账——条目字节条按真实比例(1152B vs 132B = 11.46%)+ 132B 布局放大图 +
spec 自报卡 + workspace 魔数账卡 + 满长单层对比条。

claim:IndexCache 每条 132 字节 = 128B fp8 值 + 4B fp32 scale(ue8m0 幂次),spec 自报
num_kv_heads=1 的 MLAAttentionSpec——『只有一根向量、无 K+V 之分』,与主 KV cache
(bf16 576 元素 = 1152B/token)分开分配、分开分组的第二本账:缓存的是打分原料(量化
索引键),不是分数。

数字全部取自 figure spec 的 numbers(132=128+4 / spec(1,132,uint8)·块 64·256 /
40×163840=6553600×132B=865075200B=825MB·(576×2//132)×5=40 / 132B vs 1152B=11.46%·
满长 21626880 vs 188743680)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 800
MX = 52
BXR = 1448

# ---------------- 标题区 ----------------
lc.text(MX, 34, '两本账,分开分配、分开分组:主 KV cache 记潜向量,IndexCache 记量化索引键',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '同是『每 token 一条』,spec 却不同:主 MLA bf16·576 元素 vs indexer uint8·132B 单向量——IndexCache 缓存的是打分原料,不是分数(分数每拍重算)',
        10.5, lc.C_MUTE, 'start', maxw=1100, tag='subtitle')
_ch = '放大自 L0『模型层 indexer 框』· L2 拍片② IndexCache 池(站 4)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左panel:两本账字节条(真实比例) ----------------
PX, PW_ = MX, 700           # 条形区原点与满宽(=1152B 比例尺)
BAR_Y1, BAR_H = 148, 34
lc.text(PX, 122, '① 主 KV cache(MLA 潜向量)——ch25 的产物', 11, lc.C_TXT, 'start', True,
        maxw=520, tag='l1:t')
lc.text(PX + PW_, 122, 'bf16 · 576 元素/token/layer', 9.5, lc.C_MUTE, 'end', tag='l1:s')
lc.rect(PX, BAR_Y1, PW_, BAR_H, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.8)
lc.text(PX + PW_ / 2, BAR_Y1 + 22, '1152 B/token/layer(576 元素 × bf16 2B)', 10, '#155e75',
        'middle', True, tag='l1:bar')

BAR_Y2 = BAR_Y1 + 74
NW = int(PW_ * 132 / 1152)   # 132B 比例宽 = 80
lc.text(PX, BAR_Y2 - 12, '② IndexCache(indexer 专属 K 池)——本章的新账本', 11, lc.C_TXT,
        'start', True, maxw=520, tag='l2:t')
lc.text(PX + PW_, BAR_Y2 - 12, 'uint8 · 132 B/token/layer', 9.5, lc.C_MUTE, 'end', tag='l2:s')
lc.rect(PX, BAR_Y2, NW, BAR_H, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.8)
lc.text(PX + NW + 10, BAR_Y2 + 22, '132 B/token/layer', 10, '#155e75', 'start', True, tag='l2:bar')
# 11.46% 标注:双端卡尺只跨条②条宽(PX..PX+NW = 80px,同一比例尺下 132B 的宽度)。
# 盲审 2026-09-14 两修:①原量规线走在两右缘之间(y≈206),横穿『② IndexCache…』
# 『uint8 · 132 B/token/layer』两处标签的字形——上移到条①底边(182)与标签行
# 字顶(≈202)之间的净空带 y=193;②原刻度跨度(620px)恰是 11.46% 的补数(88.6%),
# 读者易把『刻度长』读成占比——改跨条②实宽,刻度长即占比本身。
gy = BAR_Y1 + BAR_H + 11
lc.seg(PX, gy, PX + NW, gy, lc.C_ABORT, 1.4)
lc.seg(PX, gy - 5, PX, gy + 5, lc.C_ABORT, 1.4)
lc.seg(PX + NW, gy - 5, PX + NW, gy + 5, lc.C_ABORT, 1.4)
lc.text(PX + NW + 8, gy + 3.5, '11.46% (132B/1152B, 即条②宽)', 9.5, lc.C_ABORT, 'start', True,
        tag='pct')

# 132B 布局放大镜(虚线引线从窄条引出)
ZY = BAR_Y2 + BAR_H + 26
ZW, ZH = 640, 116
lc.rect(PX, ZY, ZW, ZH, '#ffffff', lc.C_MUTE, rx=9, sw=1.4, dash=True)
lc.text(PX + 14, ZY + 20, '132B 条目布局放大(uint8 视角)', 10, lc.C_TXT, 'start', True,
        maxw=300, tag='z:t')
_bw = 16 + 11 * len('一步融合 indexer_k_quant_and_cache')
lc.rect(PX + ZW - _bw - 10, ZY + 8, _bw, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
lc.text(PX + ZW - _bw / 2 - 10, ZY + 22, '一步融合 indexer_k_quant_and_cache', 9, lc.C_ENG_S,
        'middle', True, maxw=_bw - 6, tag='z:badge')
# 内部两段:128B 值 + 4B scale(比例 128:4;scale 段过窄,标签置段下方)
seg_w = ZW - 28
w_val = seg_w * 128 / 132
lc.rect(PX + 14, ZY + 34, w_val, 30, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.6)
lc.rect(PX + 14 + w_val + 2, ZY + 34, seg_w - w_val - 2, 30, '#ffedd5', lc.C_ENG_S, rx=4, sw=1.6)
lc.text(PX + 20, ZY + 80, '← 128 B fp8 值(per-token-group 128 一组)', 8.5, '#155e75', 'start',
        tag='z:val')
lc.text(PX + 14 + seg_w, ZY + 80, '4 B fp32 scale ↑', 8.5, '#9a3412', 'end', tag='z:sc')
lc.text(PX + 14, ZY + 98, 'scale 是 ue8m0 幂次(2 的幂,恒正)——dequant 回环与 ue8m0 参考逐位一致(host 实测)',
        8.5, lc.C_MUTE, 'start', maxw=ZW - 28, tag='z:f')
# 窄条 → 放大镜 引线
lc.seg(PX + NW / 2, BAR_Y2 + BAR_H, PX + 60, ZY, lc.C_FAINT, 1.2, dash=True)

# ---------------- 右panel:spec 卡 + workspace 卡 ----------------
RX, RW_ = 800, BXR - 800
SY, SH = 122, 200
lc.rect(RX, SY, RW_, SH, '#ffffff', lc.C_KV_S, rx=9, sw=1.6)
lc.text(RX + 16, SY + 24, 'spec 自报:MLAAttentionSpec(单向量账本)', 11, lc.C_KV_S, 'start',
        True, maxw=430, tag='sp:t')
for j, s in enumerate([
    'num_kv_heads=1 · head_size=132 · dtype=uint8',
    '『Only has one vector instead of K + V』',
    '   (deepseek_v2.py:L638-L641 注释原话)',
    '后端块大小:V3.2 块 64 / V4 块 256',
    'stride_order 恒等排列 = 不支持跨层 KV 布局',
    'K-cache 形状(10 块例):V3.2 [10,64,132] / V4 [10,256,132]',
]):
    lc.text(RX + 16, SY + 46 + j * 17, s, 9, lc.C_TXT if j < 2 else '#334155', 'start',
            bold=(j == 0), maxw=RW_ - 30, tag='sp:l%d' % j)
lc.text(RX + 16, SY + SH - 12, '注册 static_forward_context——metadata 按层名隐式契约取(ch19/ch25 已立)',
        8.5, lc.C_MUTE, 'start', maxw=RW_ - 30, tag='sp:f')

WY = SY + SH + 18
lc.rect(RX, WY, RW_, 158, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(RX + 16, WY + 24, 'workspace 魔数账:40 从哪来', 11, lc.C_TXT, 'start', True,
        maxw=430, tag='ws:t')
lc.text(RX + 16, WY + 48, '(576 × 2 // 132) × 5 = 40', 11, lc.C_TXT, 'start', True, tag='ws:f0')
for j, s in enumerate([
    'flashmla_sparse 的 workspace = 5×max_model_len 条,每条 576×2 字节;',
    'indexer 每条 132B——同预算能塞下的条数放大到 (576×2//132)×5=40 倍',
    '40 × 163840 = 6553600 条 × 132B = 865075200 B = 825 MB',
    '(indexer.py:L442-L452 注释:对齐稀疏后端 workspace,最大化利用)',
]):
    lc.text(RX + 16, WY + 70 + j * 17, s, 8.5, '#334155', 'start', maxw=RW_ - 30,
            tag='ws:l%d' % j)

# ---------------- 底部:满长单层对比 + 记账纪律 ----------------
FY = 520
lc.text(MX, FY, '满长 163840 单层对比(同一比例尺):', 10.5, lc.C_TXT, 'start', True,
        maxw=300, tag='fl:t')
FLW = 1160                   # =188743680 B 满宽
fl2 = int(FLW * 21626880 / 188743680)
lc.rect(MX, FY + 12, FLW, 18, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
lc.text(MX + 10, FY + 25, '主 KV bf16:188743680 B/层', 9, '#155e75', 'start', True, tag='fl:1')
lc.rect(MX, FY + 36, fl2, 18, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
lc.text(MX + fl2 + 10, FY + 49, 'IndexCache:21626880 B/层(11.46%)', 9, '#155e75', 'start', True,
        tag='fl:2')

NY = FY + 78
lc.rect(MX, NY, BXR - MX, 92, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 16, NY + 22, '第二本账的记账纪律', 10.5, lc.C_TXT, 'start', True, maxw=300,
        tag='nb:t')
for j, s in enumerate([
    '· 缓存的是打分原料(量化索引键),不是分数——分数每拍重算,原料跨拍只读',
    '· 『每 token 只算一次』:历史 token 的索引键进池后,decode 打分直接对它 paged 读(sparse_attn_indexer.py:L530-L688)',
    '· 与主 KV cache 分开分配、分开分组——两本账各自的 spec、各自的后端(KV 账本分组归 ch14/ch25 站 6)',
]):
    lc.text(MX + 16, NY + 42 + j * 17, s, 9, '#334155', 'start', maxw=BXR - MX - 32,
            tag='nb:l%d' % j)

# ---------------- 页脚 ----------------
lc.text(MX, 756, '图例:青 = KV / 显存账本角色 · 橙段 = fp32 scale 字节 · 条宽按真实字节数比例(1152 : 132 = 700 : 80 px)',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 774, '字节与 spec = DSV3.2 实尺(host 实测回环) · 注释原文 = vLLM v0.27.1 源码 · 满长= max_model_len 163840',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch26-fig-index-cache-ledger.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
