#!/usr/bin/env python3
"""ch06 机制图 3 · mm 特征展平与缓存命中（explainer figure_spec ch06-fig-mm-flatten，模板 before-after）

放大自 L0 蓝色 API 进程带（api_band）中 InputProcessor 组装段的多模态支路——即本章
L2 章图 center 拍片 ⑦ mm 展平 + south『mm 特征 · 缓存命中省 IPC』注的机制展开。
架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：argsort_mm_positions 把按品类分组的 dict-of-list 按 offset 升序摊平成 prompt
出现序的 list[MultiModalFeatureSpec]（品类序 ≠ 出现序时必然重排），缓存命中的 item
data=None——张量留前端、只有哈希与位置过线。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；
文本全 esc()。

R2（盲审回修，两处）：①首根括弧标签『image @3-4 · len 2』原 baseline=468，与「展平后」
节标题（baseline 478、字带顶 468.2）共享同一文字带——蓝标签右半压在标题「offsets
升序 [3,」字形顶部上（行级文字带交叠，linter 照不到）。整排括弧上移 6px（BRK_Y
448→442）+ 标签贴近括弧（offset +20→+17 → baseline 459、字带 [451.8,461.1]）：与
标题字带净距 7.1px、与括弧横线净距 3.8px、与格下标字带净距 8.1px——标签行与标题行
两条文字带完全分离。②展平后卡片 CARD_W 300→280：卡2 [684,984] 与卡3 [978,1278]
横向交叠 6px（cx 间距 294 < 300；rect-rect 盲区，盲审与 linter 均未照到、本轮自查
像素核出）——280 后三卡 [300,580]/[694,974]/[988,1268]，卡2-卡3 净距 14px；卡片
最长行 mm_position 行估宽 158 << 卡宽-28=252，无缩字。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 980
MX = 60
BXR = 1440
C_BODY = '#334155'
C_IMG_S, C_IMG_F = lc.C_API_S, lc.C_API_F      # image 品类辅助色（蓝）
C_AUD_S, C_AUD_F = lc.C_ENG_S, lc.C_ENG_F      # audio 品类辅助色（橙）

# ---------------- 标题区 ----------------
lc.text(MX, 34, '从分箱到上菜线：mm 特征按 offset 重排，缓存命中只过哈希不过张量', 16.5,
        lc.C_TXT, 'start', True, maxw=940, tag='title')
lc.text(MX, 58, '品类分组的 dict-of-list ≠ prompt 出现序——argsort_mm_positions 按 offset 升序摊平成 '
        'list[MultiModalFeatureSpec]；缓存命中的 item data=None，张量留前端', 10.5,
        lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ mm 展平 · L0：API 进程下行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 展平前：两个品类箱 ----------------
lc.text(MX + 24, 112, '展平前：dict-of-list（品类分组）——品类序 ≠ prompt 出现序', 11.5,
        lc.C_TXT, 'start', True, maxw=600, tag='before:h')

IMG_BIN = (84, 128, 300, 190)
lc.rect(*IMG_BIN, C_IMG_F, C_IMG_S, rx=8, sw=1.8)
lc.text(IMG_BIN[0] + 14, IMG_BIN[1] + 20, 'image 箱（2 件）', 10.5, C_IMG_S, 'start',
        True, maxw=IMG_BIN[2] - 28, tag='bin:i')
for k, (name, off, ln) in enumerate([('IMG-B', 3, 2), ('IMG-A', 11, 2)]):
    y = IMG_BIN[1] + 36 + k * 72
    lc.rect(IMG_BIN[0] + 16, y, IMG_BIN[2] - 32, 60, '#ffffff', C_IMG_S, rx=6, sw=1.3)
    lc.text(IMG_BIN[0] + 30, y + 24, f'data {name}', 9.5, lc.C_TXT, 'start', True,
            maxw=IMG_BIN[2] - 60, tag='bin:i' + name)
    lc.text(IMG_BIN[0] + 30, y + 44, f'(offset {off}, length {ln})', 9, C_BODY, 'start',
            maxw=IMG_BIN[2] - 60, tag='bin:ip' + name)

AUD_BIN = (408, 128, 280, 118)
lc.rect(*AUD_BIN, C_AUD_F, C_AUD_S, rx=8, sw=1.8)
lc.text(AUD_BIN[0] + 14, AUD_BIN[1] + 20, 'audio 箱（1 件）', 10.5, C_AUD_S, 'start',
        True, maxw=AUD_BIN[2] - 28, tag='bin:a')
lc.rect(AUD_BIN[0] + 16, AUD_BIN[1] + 36, AUD_BIN[2] - 32, 60, '#ffffff', C_AUD_S, rx=6, sw=1.3)
lc.text(AUD_BIN[0] + 30, AUD_BIN[1] + 60, 'data AUD-A', 9.5, lc.C_TXT, 'start', True,
        maxw=AUD_BIN[2] - 60, tag='bin:a1')
lc.text(AUD_BIN[0] + 30, AUD_BIN[1] + 80, '(offset 7, length 3)', 9, C_BODY, 'start',
        maxw=AUD_BIN[2] - 60, tag='bin:ap1')
lc.text(408, 272, '箱内序只是品类内序；直接按箱序消费就把顺序吃错了', 8.5, lc.C_MUTE,
        'start', maxw=340, tag='before:n')

# ---------------- 变换盒：argsort_mm_positions ----------------
TRF = (760, 128, 680, 190)
lc.rect(*TRF, '#ffffff', lc.C_API_S, rx=8, sw=1.8)
lc.text(TRF[0] + 16, TRF[1] + 22, 'argsort_mm_positions（multimodal/utils.py:L145-L165）', 10.5,
        lc.C_TXT, 'start', True, maxw=TRF[2] - 32, tag='trf:t')
lc.text(TRF[0] + 16, TRF[1] + 42, '· 先摊平 dict 成 (modality, idx, item) 生成器', 9, C_BODY,
        'start', maxw=TRF[2] - 32, tag='trf:l1')
lc.text(TRF[0] + 16, TRF[1] + 59, '· 再 sorted(key=lambda x: x[2].offset)——整数键全序、双射（每件恰出现一次）',
        9, C_BODY, 'start', maxw=TRF[2] - 32, tag='trf:l2')
# dict 序 vs 排序后（重排列）
ROW_A, ROW_B = TRF[1] + 96, TRF[1] + 132
lc.text(TRF[0] + 16, ROW_A, 'dict 序', 8.5, lc.C_MUTE, 'start', True, maxw=60, tag='trf:ra')
lc.text(TRF[0] + 16, ROW_B, 'offset 序', 8.5, lc.C_API_S, 'start', True, maxw=60, tag='trf:rb')
seq_a = [('image@3', C_IMG_S), ('image@11', C_IMG_S), ('audio@7', C_AUD_S)]
seq_b = [('image@3', C_IMG_S), ('audio@7', C_AUD_S), ('image@11', C_IMG_S)]
for row_y, seq in [(ROW_A, seq_a), (ROW_B, seq_b)]:
    x = TRF[0] + 84
    for i, (s, c) in enumerate(seq):
        w = lc.tw(s, 8.5, True) + 16
        lc.rect(x, row_y - 13, w, 20, '#ffffff', c, rx=9, sw=1.2)
        lc.text(x + w / 2, row_y + 1, s, 8.5, c, 'middle', True, maxw=w - 4,
                tag='sq' + s)
        if i < 2:
            lc.seg(x + w + 2, row_y - 3, x + w + 14, row_y - 3, c, 1.2, 'std')
        x += w + 18
lc.text(TRF[0] + 16, TRF[1] + 168, '输入 input_processor.py:L341-L377 → list[MultiModalFeatureSpec]'
        '（条目数守恒 2+1=3）', 8.5, lc.C_MUTE, 'start', maxw=TRF[2] - 32, tag='trf:l3')
# 箱区 → 变换盒 的大箭头（品类箱整体 → argsort）
lc.seg(690, 223, 758, 223, lc.C_API_S, 2.4, 'dn')

# ---------------- prompt_token_ids 标尺 ----------------
lc.text(MX + 24, 356, 'prompt_token_ids 标尺：13 个 token，三段占位区间互不交叠（offset 3 / 7 / 11 · '
        'length 2 / 3 / 2）', 11.5, lc.C_TXT, 'start', True, maxw=900, tag='rul:h')
IDS = [3, 4, 5, 40, 40, 6, 7, 45, 45, 45, 8, 40, 40]
CELL_W, CELL_H, CELL_Y = 86, 46, 372
RX0 = 96
SEG = [(3, 2, C_IMG_S, C_IMG_F, 'image'), (7, 3, C_AUD_S, C_AUD_F, 'audio'),
       (11, 2, C_IMG_S, C_IMG_F, 'image')]
seg_of = {}
for off, ln, s, f, name in SEG:
    for i in range(off, off + ln):
        seg_of[i] = (s, f)
for i, tid in enumerate(IDS):
    x = RX0 + i * CELL_W
    if i in seg_of:
        s, f = seg_of[i]
    else:
        s, f = '#cbd5e1', '#ffffff'
    lc.rect(x, CELL_Y, CELL_W - 4, CELL_H, f, s, rx=5, sw=1.3)
    lc.text(x + (CELL_W - 4) / 2, CELL_Y + 29, str(tid), 10, lc.C_TXT, 'middle', True,
            maxw=CELL_W - 10, tag='cell' + str(i))
    lc.text(x + (CELL_W - 4) / 2, CELL_Y + CELL_H + 14, str(i), 7.5, lc.C_FAINT, 'middle',
            maxw=40, tag='idx' + str(i))
# 区间括弧（cells 下方）：段中心与卡片对齐
# R2：BRK_Y 448→442、标签 offset +20→+17——原 baseline 468 与「展平后」标题字带
# （顶 468.2）交叠一行；上移后标签字带 [451.8,461.1] 与标题带净距 7.1px。
BRK_Y = CELL_Y + CELL_H + 24
BRK = [(3, 440, 'image @3-4 · len 2', C_IMG_S), (7, 834, 'audio @7-9 · len 3', C_AUD_S),
       (11, 1128, 'image @11-12 · len 2', C_IMG_S)]
for off, cx, lab, c in BRK:
    x0 = RX0 + off * CELL_W + 2
    x1 = RX0 + (off + (3 if c == C_AUD_S else 2)) * CELL_W - 8
    lc.seg(x0, BRK_Y, x0, BRK_Y + 6, c, 1.4)
    lc.seg(x0, BRK_Y + 6, x1, BRK_Y + 6, c, 1.4)
    lc.seg(x1, BRK_Y, x1, BRK_Y + 6, c, 1.4)
    lc.text(cx, BRK_Y + 17, lab, 8.5, c, 'middle', True, maxw=170, tag='brk' + lab)

# ---------------- 展平后：上菜线卡片 ----------------
# R2：CARD_W 300→280——卡2/卡3 cx 间距 294 < 300，原两卡横向交叠 6px；280 后净距 14px。
CARD_Y, CARD_H, CARD_W = 490, 54, 280
lc.text(MX + 24, 478, '展平后：list[MultiModalFeatureSpec]（上菜线）——offsets 升序 [3, 7, 11] = '
        'prompt 出现序（实测判定 true）', 11.5, lc.C_TXT, 'start', True, maxw=900,
        tag='after:h')
CARDS = [
    (440, 'image · data IMG-B', 'mm_position offset 3 · length 2', 'mm_hash 080948a1ad26abb6', C_IMG_S),
    (834, 'audio · data AUD-A', 'mm_position offset 7 · length 3', 'mm_hash 34675dd13045ffde', C_AUD_S),
    (1128, 'image · data IMG-A', 'mm_position offset 11 · length 2', 'mm_hash 14d5d6ed4fb50d69', C_IMG_S),
]
for cx, t, l1, l2, c in CARDS:
    lc.rect(cx - CARD_W / 2, CARD_Y, CARD_W, CARD_H, '#ffffff', c, rx=7, sw=1.6)
    lc.text(cx - CARD_W / 2 + 14, CARD_Y + 21, t, 9.5, lc.C_TXT, 'start', True,
            maxw=CARD_W - 28, tag='cd:t' + t)
    lc.text(cx - CARD_W / 2 + 14, CARD_Y + 38, l1, 8.5, C_BODY, 'start', maxw=CARD_W - 28,
            tag='cd:l1' + t)
    lc.text(cx - CARD_W / 2 + 14, CARD_Y + 50, l2, 8, lc.C_MUTE, 'start', maxw=CARD_W - 28,
            tag='cd:l2' + t)
# 区间括弧 → 卡片的虚线位置对应
for _, cx, _, _ in BRK:
    lc.seg(cx, BRK_Y + 26, cx, CARD_Y - 2, lc.C_MUTE, 1.2, dash=True)

# ---------------- 底部：缓存命中 + identifier 双键 ----------------
BP_Y, BP_H = 596, 210
CH_P = (MX, BP_Y, 700, BP_H)
lc.rect(*CH_P, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(CH_P[0] + 16, CH_P[1] + 22, '缓存命中（同批 item 重发）——张量不过线', 11, lc.C_TXT,
        'start', True, maxw=CH_P[2] - 32, tag='chp:t')
EMPTY = ['080948a1ad26abb6', '34675dd13045ffde', '14d5d6ed4fb50d69']
for i, h_ in enumerate(EMPTY):
    x = CH_P[0] + 20 + i * 222
    lc.rect(x, CH_P[1] + 40, 200, 44, '#ffffff', lc.C_MUTE, rx=6, sw=1.2, dash=True)
    lc.text(x + 100, CH_P[1] + 58, 'data=None', 9.5, lc.C_MUTE, 'middle', True, maxw=180,
            tag='em' + str(i))
    lc.text(x + 100, CH_P[1] + 76, '哈希 ' + h_, 8, lc.C_MUTE, 'middle', maxw=190,
            tag='emh' + str(i))
chp_lines = [
    '· sender cache 命中返回 (None, updates)（multimodal/cache.py:L410-L416）：hits +3 · data=None×3',
    "· 'Can be `None` if the item is cached, to skip IPC between API server and",
    "   engine core processes.'（inputs.py:L331-L337）——MB 级多模态张量留在前端",
    '· 哈希与位置（mm_hash / identifier / mm_position）不变、仍过线，引擎侧按哈希取回',
]
for i, ln in enumerate(chp_lines):
    lc.text(CH_P[0] + 16, CH_P[1] + 108 + i * 18, ln, 8.5, C_BODY, 'start',
            maxw=CH_P[2] - 30, tag='chp:l' + str(i))

ID_P = (780, BP_Y, 660, BP_H)
lc.rect(*ID_P, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(ID_P[0] + 16, ID_P[1] + 22, '编码器缓存键 identifier：双键防串（input_processor.py:L174-L190）',
        11, lc.C_TXT, 'start', True, maxw=ID_P[2] - 32, tag='idp:t')
idp_lines = [
    '· tower-connector LoRA：identifier = style:080948a1ad26abb6',
    '   （lora_name: 前缀——mm 嵌入随 LoRA 变，加前缀防跨 LoRA 错误命中）',
    '· 普通 / 无 LoRA：identifier = 裸哈希 080948a1ad26abb6',
    '· 双键分工：mm_hash = processor 输出缓存键（无前缀）；',
    '   identifier = 编码器输出缓存键（可带 LoRA 前缀）',
    '· 占位符本身就是书签：PlaceholderRange(offset, length)——源码自带 AAAA BBBB 教学例',
    '   （inputs.py:L122-L159）；单件超编码器缓存预算则出门前拒收（详见正文）',
]
for i, ln in enumerate(idp_lines):
    lc.text(ID_P[0] + 16, ID_P[1] + 46 + i * 18, ln, 8.5, C_BODY, 'start',
            maxw=ID_P[2] - 30, tag='idp:l' + str(i))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BP_Y + BP_H + 34
lx = MX
items = [('swatch', C_IMG_S, C_IMG_F, 'image 占位段 / 卡片'), ('swatch', C_AUD_S, C_AUD_F, 'audio'),
         ('dashline', None, None, '位置对应（offset → 出现序）'),
         ('dashbox', None, None, '空盒 = data=None（张量不过线）')]
for kind, s, f, name in items:
    if kind == 'swatch':
        lc.rect(lx, LEG_Y - 9, 20, 13, f, s, rx=3, sw=1.4)
    elif kind == 'dashline':
        lc.seg(lx, LEG_Y, lx + 28, LEG_Y, lc.C_MUTE, 1.3, dash=True)
    else:
        lc.rect(lx, LEG_Y - 9, 22, 14, '#ffffff', lc.C_MUTE, rx=4, sw=1.1, dash=True)
    lc.text(lx + (34 if kind != 'swatch' else 26), LEG_Y + 3, name, 9.5, lc.C_TXT, 'start',
            maxw=280, tag='leg' + name)
    lx += (34 if kind != 'swatch' else 26) + lc.tw(name, 9.5) + 24
lc.text(MX, LEG_Y + 26, '占位长 image 2 / audio 3、占位 id 40 / 45 为 seam 示意值（真实 = 编码器特征尺寸，'
        '如 image 576）；哈希为运行样本——排序 / 展平 / 缓存命中语义为真代码路径',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '展平 verbatim：vllm/multimodal/utils.py:L145-L165 · 组装：vllm/v1/engine/'
        'input_processor.py:L341-L377 · 行号基线 vLLM v0.27.1', 9, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch06-fig-mm-flatten.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
