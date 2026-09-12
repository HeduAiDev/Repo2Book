#!/usr/bin/env python3
"""ch24 机制图 ⑤ · KV cache 总账 Table 1:四机制对决(figure_spec ch24-fig-table1-account,模板 before-after)

放大自 L0 左带『调度 · 显存账本』(青)的『BlockPool + 前缀缓存』块——每 token 写进块池的
宽度由本章四行公式决定,页字节公式(real_page_size_bytes 带/不带 2 因子)就是 ch14 显存账本
定 num_blocks 的输入。primer 推导链第 ⑤ 环(收束):前四环的形状账在此对总。

claim:四机制每 token 每层元素账:MHA 32768 → GQA-8 2048 → MQA 256 → MLA 576——MLA 等效
2.25 组 GQA(576/(2·128)=2.25)、是自身 MHA 配置的 1.75%;页字节口径 GQA-8 层 65536 B vs
MLA 层 18432 B(去 2 因子),Table 1(表体为包内重建)四行在引擎账本以两份 page_size 公式落地。

数字全部取自 figure_spec.numbers(四行账/相对 MHA 比/2.25 组换算/页字节镜像/消融口径/
表体重建标注:论文忠实 NumPy 参考实现实跑 trace)。坐标由常量/循环计算;文本全 esc()。
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX = 60
BXR = 1440
SLATE_F, SLATE_S = '#cbd5e1', '#94a3b8'   # 砍头路(改头数)
CYAN = lc.C_KV_S

LOG0 = math.log10(200.0)   # 条长基线:10^2.3 ≈ 200


def loglen(v, k):
    return (math.log10(v) - LOG0) * k


# ---------------- 标题区 ----------------
lc.text(MX, 34, '四种注意力,一杆秤:每 token 每层付多少元素——砍头路 vs 降维路',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'DSV3 超参 n_h=128 · d_h=128 · d_c=512 · d_h^R=64:MHA 32768 / GQA-8 2048 / MQA 256 / MLA 576;MLA 等效 2.25 组 GQA、是自身 MHA 配置的 1.75%(arXiv:2405.04434 §2.1.4 Table 1)',
        10.5, lc.C_MUTE, 'start', maxw=1150, tag='subtitle')
_ch = '推导链 ⑤ · 放大自 L0『调度 · 显存账本』的 BlockPool 块池'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, CYAN, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左面板:砍头路 ----------------
P1 = (MX, 92, 640, 320)
lc.rect(*P1, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(P1[0] + P1[2] / 2, 116, '砍头路:改头数——元素账 2·G·d_h 随组数 G 线性', 11.5,
        lc.C_TXT, 'middle', True, maxw=P1[2] - 20, tag='p1:t')
CUT = [
    ('MHA·G=128', '2·128·128', 32768, '32768 · 1.0(基线)'),
    ('GQA-8·G=8', '2·8·128', 2048, '2048 · 1/16(0.0625)'),
    ('MQA·G=1', '2·1·128', 256, '256 · 1/128(0.0078125)'),
]
BX0, BK = 160, 180.0
for i, (nm, fml, v, lab) in enumerate(CUT):
    y = 142 + i * 46
    lc.text(148, y + 10, nm, 10, lc.C_TXT, 'end', True, tag=f'cut{i}:n')
    lc.text(148, y + 24, fml, 7.5, lc.C_MUTE, 'end', tag=f'cut{i}:f')
    bl = loglen(v, BK)
    lc.rect(BX0, y, bl, 26, SLATE_F, SLATE_S, rx=4, sw=1.1)
    if bl > 70:
        lc.text(BX0 + bl - 8, y + 17, lab, 8.5, '#334155', 'end', maxw=bl - 12, tag=f'cut{i}:v')
    else:
        lc.text(BX0 + bl + 8, y + 17, lab, 8.5, '#334155', 'start', tag=f'cut{i}:v')
D1 = (80, 300, 600, 46)
lc.rect(*D1, '#fef2f2', lc.C_ABORT, rx=6, sw=1.1)
lc.text(D1[0] + D1[2] / 2, 318, '质量代价(D.1,7B dense):MHA 显著优于 GQA / MQA', 8.8,
        lc.C_ABORT, 'middle', True, tag='d1:t')
lc.text(D1[0] + D1[2] / 2, 336, '砍 KV 头确实掉质量——分组路线的代价上限,逼出两条『别砍这么狠』的路', 7.8,
        '#7f1d1d', 'middle', maxw=D1[2] - 14, tag='d1:l')

# ---------------- 右面板:降维路 ----------------
P2 = (720, 92, 720, 320)
lc.rect(*P2, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(P2[0] + P2[2] / 2, 116, '降维路:MLA 换了一根轴——头数全保留,压宽度', 11.5,
        lc.C_TXT, 'middle', True, maxw=P2[2] - 20, tag='p2:t')
lc.text(P2[0] + P2[2] / 2, 136, 'GQA 组数轴(G = 1..128 扫描,d_h = 128,对数)——把 MLA 放回头数谱系定位',
        8.5, lc.C_MUTE, 'middle', maxw=P2[2] - 20, tag='p2:s')
AX0, AX1, AY = 760, 1380, 300
AK = (AX1 - AX0) / (math.log10(32768) - LOG0)


def axx(v):
    return AX0 + (math.log10(v) - LOG0) * AK


lc.seg(AX0 - 10, AY, AX1 + 10, AY, '#cbd5e1', 1.2)
SWEEP = [(256, 'G1', '256'), (512, 'G2', '512'), (1024, 'G4', '1024'),
         (2048, 'G8', '2048'), (32768, 'G128(MHA)', '32768')]
for v, gl, el in SWEEP:
    x = axx(v)
    lc.seg(x, AY, x, AY + 6, '#94a3b8', 1.2)
    anchor = 'end' if v == 32768 else 'middle'
    lc.text(x, AY + 19, gl, 8.5, lc.C_TXT, anchor, True, tag='sw:g' + gl)
    lc.text(x, AY + 33, el, 8, lc.C_MUTE, anchor, tag='sw:e' + gl)
    lc.rect(x - 4, AY - 4, 8, 8, SLATE_F, SLATE_S, rx=1.5, sw=1.0)
# MLA 定位线:576 插在 G2 与 G4 之间
mx_ = axx(576)
lc.seg(mx_, AY, mx_, 252, CYAN, 1.8, dash=True)
lc.rect(mx_ - 4, AY - 4, 8, 8, lc.C_KV_F, CYAN, rx=1.5, sw=1.4)
lc.text(mx_, 246, 'MLA 576(512+64)', 9.5, CYAN, 'middle', True, tag='mla:flag')
# 2.25 组标注
lc.seg(axx(512), 216, axx(1024), 216, CYAN, 1.3)
lc.seg(axx(512), 212, axx(512), 220, CYAN, 1.3)
lc.seg(axx(1024), 212, axx(1024), 220, CYAN, 1.3)
lc.text(axx(1024) + 14, 220, '等效 2.25 组 = (d_c+d_h^R)/(2·d_h) = (4·128+64)/(2·128)', 9,
        CYAN, 'start', True, maxw=390, tag='mla:225')
lc.text(axx(1024) + 14, 236, '—— 576 落在 G2=512 与 G4=1024 之间', 8, lc.C_MUTE, 'start',
        tag='mla:225n')
D2 = (740, 346, 680, 52)
lc.rect(*D2, '#ecfeff', CYAN, rx=6, sw=1.1)
lc.text(D2[0] + D2[2] / 2, 364, '质量反超(D.2,MoE 对齐实验):MLA 优于 MHA,cache 仅 14% / 4%', 8.8,
        CYAN, 'middle', True, tag='d2:t')
lc.text(D2[0] + D2[2] / 2, 382, 'MLA 不是谱系上又一个插值点——是换了轴(降维代替砍头),不受砍头的质量罚', 7.8,
        '#155e75', 'middle', maxw=D2[2] - 14, tag='d2:l')

# ---------------- 中部:同一杆秤(对数) ----------------
STRIP = (MX, 432, BXR - MX, 128)
lc.rect(*STRIP, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 690, 452, '同一杆秤(对数刻度):四机制每 token 每层元素,同一基线 10^2.3 起条', 10,
        lc.C_TXT, 'middle', True, maxw=1340, tag='st:t')
FOUR = [
    ('MHA', 32768, '32768 · 1.0(基线)', False),
    ('GQA-8', 2048, '2048 · 0.0625(1/16)', False),
    ('MLA', 576, '576 · 0.017578125(1.75%,1/56.89)', True),
    ('MQA', 256, '256 · 0.0078125(1/128)', False),
]
SX0, SK = 100, 550.0
for i, (nm, v, lab, is_mla) in enumerate(FOUR):
    y = 470 + i * 22
    lc.text(92, y + 13, nm, 9, CYAN if is_mla else lc.C_TXT, 'end', True, tag=f'four{i}:n')
    bl = loglen(v, SK)
    lc.rect(SX0, y, bl, 16, lc.C_KV_F if is_mla else SLATE_F,
            CYAN if is_mla else SLATE_S, rx=3, sw=1.1)
    lc.text(SX0 + bl + 8, y + 12, lab, 8.3, CYAN if is_mla else '#334155', 'start',
            is_mla, tag=f'four{i}:v')
for e in (3, 4):
    x = SX0 + (e - LOG0) * SK
    lc.seg(x, 558, x, 564, '#94a3b8', 1.0)
    lc.text(x, 574, f'10^{e}', 7.5, lc.C_FAINT, 'middle', tag=f'stick{e}')

# ---------------- 底部左:引擎账本口径 ----------------
PG = (MX, 580, 700, 228)
lc.rect(*PG, '#ffffff', CYAN, rx=8, sw=1.4)
lc.text(410, 602, '引擎账本口径:block=16,fp16 —— 页字节就是 ch14 显存账本的输入(回指 ch14)', 9.5,
        CYAN, 'middle', True, maxw=676, tag='pg:t')
g1w, g2w = 300.0, 300.0 * 18432 / 65536
lc.rect(86, 616, g1w, 58, '#a5f3fc', CYAN, rx=4, sw=1.2)
lc.text(400, 634, 'GQA-8 层:65536 B/页(每 token 4096 B)', 9, lc.C_TXT, 'start', True,
        tag='pg:g1')
lc.text(400, 652, '= 2 × 16 × 8 × 128 × 2B(带 2 因子:K、V 各一份)', 8, lc.C_MUTE,
        'start', tag='pg:g1f')
lc.rect(86, 696, g2w, 58, '#a5f3fc', CYAN, rx=4, sw=1.2)
lc.text(400, 714, 'MLA 层:18432 B/页(每 token 1152 B)', 9, lc.C_TXT, 'start', True, tag='pg:g2')
lc.text(400, 732, '= 16 × 1 × 576 × 2B(去 2 因子:单潜向量无 K/V 之分)', 8, lc.C_MUTE,
        'start', tag='pg:g2f')
lc.text(410, 764, '普通层 real_page_size_bytes = 2×block×num_kv_heads×head_dim×dtype ↔ MLA 层 = storage_block_size×num_kv_heads×head_dim×dtype',
        7.4, '#334155', 'middle', maxw=676, tag='pg:fml')
lc.text(410, 782, 'vllm/v1/kv_cache_interface.py:L212-L226 vs L388-L426(同式镜像算术)· ch14:num_blocks = available_memory // page_size // num_layers',
        7.4, lc.C_FAINT, 'middle', maxw=676, tag='pg:anchor')
lc.text(253, 770, '页宽 ∝ 字节数', 7.5, '#155e75', 'middle', tag='pg:ratio')

# ---------------- 底部右:口径警示 + 诚实标注 ----------------
WR = (780, 580, 660, 228)
lc.rect(*WR, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(1110, 602, '口径警示:三个百分比比较基不同,互不可换算', 9.5, lc.C_TXT, 'middle',
        True, maxw=636, tag='wr:t')
PCT = [
    ('93.3%', 'Abstract:相对 DeepSeek 67B(不同头配置)的 KV cache 削减'),
    ('1.75%', 'Table 1 算术:576/32768(本章主用口径)'),
    ('14% / 4%', 'App D.2:MoE 对齐实验实测(小 / 大两档)'),
]
for i, (pv, pd) in enumerate(PCT):
    y = 628 + i * 22
    lc.text(810, y, pv, 9.5, lc.C_ABORT, 'start', True, tag=f'wr:p{i}')
    lc.text(910, y, pd, 8.5, '#334155', 'start', maxw=510, tag=f'wr:d{i}')
lc.seg(800, 694, 1420, 694, '#e2e8f0', 1.0)
lc.text(1110, 716, '诚实标注:Table 1 表体为论文包内重建——抓取只留 caption;', 8.5,
        lc.C_TXT, 'middle', True, maxw=636, tag='wr:h1')
lc.text(1110, 734, 'MHA / MLA 行出自原句、GQA / MQA 行由 factor-H 直推(引用保留标注)', 8,
        lc.C_MUTE, 'middle', maxw=636, tag='wr:h2')
lc.text(1110, 760, 'l=60 全模型每 token 元素:MHA 1966080 / GQA-8 122880 / MLA 34560 / MQA 15360',
        8.5, '#334155', 'middle', maxw=636, tag='wr:l60')
lc.text(1110, 784, 'DSV3 头形、4096-token 上下文的感受数:若 MHA 需 ≈15.0 GiB 的 KV cache,MLA 需 ≈270.0 MiB(差 56.89 倍)',
        7.8, lc.C_FAINT, 'middle', maxw=636, tag='wr:feel')

# ---------------- 页脚 ----------------
lc.text(MX, 832, '图例:灰条 = 砍头路(改头数)· 青条 = 降维路(MLA,改维度)· 方点 = GQA 组数扫描 · 青框页 = 引擎页字节(宽 ∝ 字节)',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 852, '出处 arXiv:2405.04434 §2.1.4 Table 1 / §3.1.2 / App D.1-D.2 · 数值取自论文忠实 NumPy 参考实现实跑(host,float64)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 868, 'vLLM 行号基线 v0.27.1(6e448d0ea)', 8.5, lc.C_FAINT, 'start', tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch24-fig-table1-account.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
