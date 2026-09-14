#!/usr/bin/env python3
"""ch26 机制图 · 共享 topk_indices_buffer 的泳道协议(figure ch26-fig-buffer-protocol,模板 swimlane)

放大自 L0『模型层 indexer 框』——L2 拍片①(站 1 分配)/④(站 6 接线)/⑤(站 11 落账)/
⑥(站 12 消费)的写读协议展开。UML 时序图法(FIGURE-SYSTEM §0):参与者=竖直生命线、
共享时间轴(拍=网格带)、消息=水平直线(禁折线)。

claim:一块 [max_num_batched_tokens, index_topk] int32 裸 buffer 是 indexer 与稀疏 MLA
的唯一接口:每层的 indexer 先于 mla_attn 被调用、纯副作用写本拍 query token 的行(返回值
无人接收),稀疏后端取前 num_actual_toks 行读——无所有权封装的共享让 skip 层『不写只读
旧值』的跨层复用成为可能。

数字全部取自 figure spec 的 numbers([512,2048] int32=4194304B/8192 档 64MiB /
两层同一对象 / -1 哨兵两例 rowEnd=1·[0,-1]·空上下文整行 -1 / 拍1[7,1]→拍2[2,5] /
mla.py:L205-L206 接线)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 840
MX = 52
BXR = 1448

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一块裸 buffer 的泳道协议:indexer 写本拍行、稀疏 MLA 读走——skip 层不写只读',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '无所有权封装的共享:[max_num_batched_tokens, index_topk] int32 全模型一块——所有层的 indexer 写、稀疏 MLA 读;行=本拍 query token,历史行不残留',
        10.5, lc.C_MUTE, 'start', maxw=1100, tag='subtitle')
_ch = '放大自 L0『模型层 indexer 框』· L2 拍片①④⑤⑥(站 1·6·11·12)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 生命线与时间轴 ----------------
PLATE_Y, PLATE_H = 96, 44
LIFE_TOP, LIFE_BOT = PLATE_Y + PLATE_H, 700
IX, BX, MX_, SX, CX = 150, 350, 550, 750, 950
BUF_HW = 32            # buffer 半宽
ACT_HW = 6             # 激活条半宽

# 拍带(共享时间轴)先画(底层浅底色,后画的线压在其上)
BANDS = [(196, 448, '拍 1'), (468, LIFE_BOT - 6, '拍 2')]
for by0, by1, lab in BANDS:
    lc.rect(MX, by0, 1000 - MX, by1 - by0, '#f8fafc', 'none', rx=6, sw=0)
    lc.text(62, by0 + 18, lab, 10, lc.C_MUTE, 'start', True, tag='band:' + lab)
    if by0 > 200:
        lc.seg(MX, by0, 1000, by0, lc.C_FAINT, 1.0, dash=True)
lc.text(MX, 184, '共享时间轴:1 拍 = 引擎一次 step(全部层各过一遍);两拍等比例,拍内消息按发生先后',
        8.5, lc.C_MUTE, 'start', maxw=940, tag='axis')

parts = [
    (IX, '层 i 的 indexer', '写者(custom op)', lc.C_GPU_S, False),
    (BX, 'topk_indices_buffer', '[512, 2048] int32', lc.C_GPU_S, True),
    (MX_, '层 i 的稀疏 MLA', '读者(同层随即读)', lc.C_GPU_S, False),
    (SX, 'skip 层(无 indexer)', '不写,复用别层写的行', lc.C_MUTE, False),
    (CX, 'V3.2 消费后端', 'FlashMLASparseImpl', lc.C_GPU_S, False),
]
for x, t1, t2, stk, is_buf in parts:
    pw = 196
    lc.rect(x - pw / 2, PLATE_Y, pw, PLATE_H, '#ffffff' if not is_buf else lc.C_GPU_F,
            stk, rx=8, sw=1.8 if is_buf else 1.4)
    lc.text(x, PLATE_Y + 19, t1, 9.5, lc.C_TXT, 'middle', True, maxw=pw - 10, tag='pl:' + t1[:8])
    lc.text(x, PLATE_Y + 35, t2, 8, lc.C_MUTE, 'middle', maxw=pw - 10, tag='pl:s' + t1[:8])
    if not is_buf:
        lc.seg(x, LIFE_TOP, x, LIFE_BOT, lc.C_FAINT, 1.1, dash=True)

# buffer 参与者画成竖直内存条(GPU 绿),行刻度=buffer 行
lc.rect(BX - BUF_HW, LIFE_TOP, BUF_HW * 2, LIFE_BOT - LIFE_TOP, lc.C_GPU_F, lc.C_GPU_S,
        rx=6, sw=2.0)
slot_h = 26
n_slots = int((LIFE_BOT - LIFE_TOP - 16) // slot_h)
for i in range(n_slots):
    sy = LIFE_TOP + 8 + (i + 1) * slot_h
    lc.seg(BX - BUF_HW + 8, sy, BX + BUF_HW - 8, sy, lc.C_GPU_S, 0.7)

# 激活条(UML:消息端点落条边)——每个非 buffer 参与者、每拍一条(范围覆盖该拍全部消息 y)
ACT_SPANS = [(240, 432), (504, 690)]
for x in (IX, MX_, SX, CX):
    for ay0, ay1 in ACT_SPANS:
        lc.rect(x - ACT_HW, ay0, ACT_HW * 2, ay1 - ay0, '#e2e8f0', lc.C_MUTE, rx=3, sw=1.0)

# ---------------- 消息(水平直线,A@y → B@y) ----------------
def msg(y, x1, x2, color, dash, l1, l2=''):
    lc.seg(x1, y, x2, y, color, 1.8, 'std', dash=dash)
    mx = (x1 + x2) / 2
    lc.text(mx, y - 13 if l2 else y - 7, l1, 8.5, color, 'middle', True,
            maxw=abs(x2 - x1) - 10, tag='m:' + l1[:10])
    if l2:
        lc.text(mx, y - 3, l2, 7.5, lc.C_MUTE, 'middle', maxw=abs(x2 - x1) - 10,
                tag='m2:' + l2[:10])


HOT = lc.C_ENG_S
# 拍 1(激活条区 y 240..432)
msg(262, IX + ACT_HW, BX - BUF_HW, lc.C_GPU_S, False, '① 写本拍 token 行', '-1 预清 → top-k 覆写')
msg(330, BX + BUF_HW, MX_ - ACT_HW, lc.C_GPU_S, False, '② 同层随即读走', '选块交给稀疏 MLA')
msg(384, BX + BUF_HW, SX - ACT_HW, lc.C_MUTE, True, 'skip 层:不写', '直接读别层写的同一行')
msg(424, BX + BUF_HW, CX - ACT_HW, lc.C_GPU_S, False, '③ 取前 num_actual_toks 行', '')
# 拍 2(激活条区 y 512..694)
msg(530, IX + ACT_HW, BX - BUF_HW, HOT, False, '④ 整块重写本拍行', '[7,1]→[2,5] 旧值不残留')
msg(594, BX + BUF_HW, MX_ - ACT_HW, lc.C_GPU_S, False, '读走', '')
msg(648, BX + BUF_HW, SX - ACT_HW, lc.C_MUTE, True, 'skip:读,不写', '')
msg(688, BX + BUF_HW, CX - ACT_HW, lc.C_GPU_S, False, '读前 num_actual_toks 行', '')

# ---------------- 右侧:一行的放大 + 分配账 ----------------
ZX, ZW_ = 1060, BXR - 1060
lc.rect(ZX, PLATE_Y, ZW_, 286, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(ZX + 14, PLATE_Y + 20, '一行的放大:协议的字面形态', 10.5, lc.C_TXT, 'start', True,
        maxw=ZW_ - 28, tag='z:t')


def row_cells(x, y, cells, cell_w=34, cell_h=24, gray_from=None):
    for k, v in enumerate(cells):
        gray = gray_from is not None and k >= gray_from
        lc.rect(x + k * (cell_w + 4), y, cell_w, cell_h,
                '#f1f5f9' if gray else lc.C_GPU_F, lc.C_FAINT if gray else lc.C_GPU_S,
                rx=4, sw=1.1)
        lc.text(x + k * (cell_w + 4) + cell_w / 2, y + 16, str(v), 9.5,
                lc.C_MUTE if gray else '#166534', 'middle', True, tag='cell:%s' % v)


zy = PLATE_Y + 34
lc.text(ZX + 14, zy + 10, '-1 哨兵 · 例一:decode 因果自界', 9, lc.C_TXT, 'start', True,
        maxw=ZW_ - 28, tag='z:a1')
row_cells(ZX + 14, zy + 18, [0, -1], gray_from=1)
lc.text(ZX + 14 + 2 * 38 + 10, zy + 34, 'rowEnd=1:logits=[9,9,1,9]', 8, '#334155', 'start',
        tag='z:a2')
lc.text(ZX + 14 + 2 * 38 + 10, zy + 48, '窗外高分 9 不得入选', 8, '#334155', 'start',
        tag='z:a3')
zy2 = zy + 62
lc.text(ZX + 14, zy2 + 10, '-1 哨兵 · 例二:空上下文 prefill', 9, lc.C_TXT, 'start', True,
        maxw=ZW_ - 28, tag='z:b1')
row_cells(ZX + 14, zy2 + 18, [-1, -1, -1, -1], gray_from=0)
lc.text(ZX + 14 + 4 * 38 + 10, zy2 + 34, 'op 预清 -1', 8, '#334155', 'start', tag='z:b2')
lc.text(ZX + 14 + 4 * 38 + 10, zy2 + 48, '无历史 → 整行 -1 直落', 8, '#334155', 'start',
        tag='z:b3')
zy3 = zy2 + 62
lc.text(ZX + 14, zy3 + 10, '跨拍整块重写:行=本拍 query token', 9, lc.C_TXT, 'start', True,
        maxw=ZW_ - 28, tag='z:c1')
row_cells(ZX + 14, zy3 + 18, [7, 1])
lc.text(ZX + 14, zy3 + 52, '拍 1 行(top 分值 1.519 / 0.438)', 7.5, lc.C_MUTE, 'start',
        tag='z:c2')
row_cells(ZX + 14 + 150, zy3 + 18, [2, 5])
lc.text(ZX + 14 + 150, zy3 + 52, '拍 2 整块重写(1.121 / 0.531)', 7.5, lc.C_MUTE, 'start',
        tag='z:c3')
lc.seg(ZX + 14 + 76, zy3 + 30, ZX + 14 + 150, zy3 + 30, HOT, 1.4, 'std')

# 分配账卡
AY = PLATE_Y + 294
lc.rect(ZX, AY, ZW_, 168, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(ZX + 14, AY + 20, '分配账(装配期,一生一次)', 10.5, lc.C_TXT, 'start', True,
        maxw=ZW_ - 28, tag='al:t')
for j, s in enumerate([
    '形状 [512, 2048] int32 = 4194304 B(4 MiB)',
    '8192 token 档 = 67108864 B(64 MiB)',
    'torch.empty 分配、不初始化',
    '两层共用同一对象:layer0/layer1 的',
    '   buffer 是同一 tensor(实测核验)',
    '非 DSA 模型(无 index_topk)→ buffer=None',
]):
    lc.text(ZX + 14, AY + 42 + j * 17, s, 8.5, '#334155', 'start', maxw=ZW_ - 28,
            tag='al:l%d' % j)
lc.text(ZX + 14, AY + 152, 'deepseek_v2.py:L1376-L1389', 8, lc.C_FAINT, 'start', tag='al:f')

# ---------------- 底部:接线位原文 ----------------
QY = LIFE_BOT + 16
lc.rect(MX, QY, BXR - MX, 74, '#ffffff', lc.C_GPU_S, rx=9, sw=1.4)
lc.text(MX + 16, QY + 20, '接线位(每层每拍,mla.py:L205-L206):', 9.5, lc.C_GPU_S, 'start',
        True, maxw=400, tag='q:t')
lc.text(MX + 16, QY + 40, 'if indexer and is_sparse and not skip_topk:  indexer(hidden_states, q_c, positions, indexer_rope_emb)',
        9.5, lc.C_TXT, 'start', True, maxw=BXR - MX - 30, tag='q:code')
lc.text(MX + 16, QY + 58, '返回值无人接收——纯写 buffer 的副作用;custom op 声明 mutates_args=[\'topk_indices_buffer\'](写者身份的自我描述)',
        9, '#334155', 'start', maxw=BXR - MX - 30, tag='q:note')

# ---------------- 页脚 ----------------
lc.text(MX, 812, '图例:绿 = GPU 侧角色(indexer/buffer/MLA/后端) · 实线 = 写/读 · 虚线 = skip 层只读不写 · 灰格 = -1 哨兵位 · 底纹带 = 拍',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 830, '行值/哨兵例 = host 实测(traces m03) · 接线与分配 = vLLM v0.27.1 源码(deepseek_v2.py / mla.py)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch26-fig-buffer-protocol.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
