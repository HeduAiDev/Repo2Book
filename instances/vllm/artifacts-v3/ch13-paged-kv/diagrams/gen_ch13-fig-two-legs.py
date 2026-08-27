#!/usr/bin/env python3
"""ch13 机制图 8 · 前向读写两条腿（figure_spec ch13-fig-two-legs，模板 flow）

放大自 L0 GPU 列（绿）中注意力消费 KV 的那一格——即本章 L2 章图中排 ⑦『前向
读写 · 两条腿』拍片的机制展开（F7 伏笔埋点）。架构归属回指 L0/L2
（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：同一池子两条腿：写腿走 slot_mapping（每 token 直寻址——行 [3,1,7] 的
48 个 token 各得一个门牌号），读腿走块表张量（间接寻址交 attention metadata，
pad 行填 NULL_BLOCK_ID=0）——间接寻址的代价是 ch22 的账单（F7）。

数字全部取自 figure_spec.numbers（写腿槽位承 m9 实测：48..63 / 16..31 /
112..127；读腿 pad 张量 [4,8]、未用行全 0；positions GPU 组装；F7 planted=13 /
paid=22）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
SEG_COL = {0: lc.C_API_S, 1: lc.C_ENG_S, 2: lc.C_SAM_S}
SEG_FILL = {0: lc.C_API_F, 1: lc.C_ENG_F, 2: lc.C_SAM_F}

# ---------------- 标题区 ----------------
lc.text(MX, 34, '前向读写两条腿：写腿直寻址（门牌号），读腿间接寻址（翻楼层图）',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '同一池子、两种寻址——这一拍算出的新 token 按 slot_mapping 直塞；下一拍注意力读全部历史时拿的是块表张量，kernel 得自己翻页（两条腿无先后因果）',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L2 章图中排 ⑦ 拍片「前向读写 · 两条腿」· L0：GPU 列注意力消费 KV 格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 中：KV 池（砖墙，接 m10 画法） ----------------
POOL_X, POOL_W = 560, 400
POOL_Y, POOL_H = 300, 300
lc.rect(POOL_X, POOL_Y, POOL_W, POOL_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(POOL_X + POOL_W / 2, POOL_Y + 22, 'KV 池（每层一张张量）', 11, lc.C_GPU_S, 'middle', True,
        maxw=POOL_W - 20, tag='pool:t')
BW_, BH_ = 108, 56
BLK_COL = {1: SEG_COL[1], 3: SEG_COL[0], 7: SEG_COL[2]}
BLK_FILL = {1: SEG_FILL[1], 3: SEG_FILL[0], 7: SEG_FILL[2]}
BLK_SEG = {1: 1, 3: 0, 7: 2}
for i, blk in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
    r, c = divmod(i, 4)
    bx = POOL_X + 22 + c * (BW_ + 10)
    by = POOL_Y + 38 + r * (BH_ + 12)
    hot = blk in BLK_COL
    lc.rect(bx, by, BW_, BH_, BLK_FILL.get(blk, '#f8fafc'), BLK_COL.get(blk, '#cbd5e1'),
            rx=4, sw=1.4 if hot else 0.9)
    if blk == 0:
        lc.seg(bx + 8, by + 6, bx + BW_ - 8, by + BH_ - 6, '#94a3b8', 1.2)
        lc.seg(bx + 8, by + BH_ - 6, bx + BW_ - 8, by + 6, '#94a3b8', 1.2)
    lc.text(bx + BW_ / 2, by + 18, '块 %d' % blk, 9.5, BLK_COL.get(blk, '#94a3b8'), 'middle',
            True, maxw=BW_ - 8, tag='bk%d' % blk)
    if hot:
        lc.text(bx + BW_ / 2, by + 36, '槽 %s' % {3: '48..63', 1: '16..31', 7: '112..127'}[blk],
                7.5, BLK_COL[blk], 'middle', maxw=BW_ - 8, tag='bks%d' % blk)
    else:
        lc.text(bx + BW_ / 2, by + 36, {0: 'null 封条', 2: '别的请求', 4: '别的请求',
                                        5: '别的请求', 6: '别的请求'}[blk], 7.5, '#94a3b8',
                'middle', maxw=BW_ - 8, tag='bks%d' % blk)
lc.text(POOL_X + POOL_W / 2, POOL_Y + POOL_H - 10, '三个热块 = 块表行 [3,1,7] 指到的页', 8.5,
        lc.C_MUTE, 'middle', maxw=POOL_W - 20, tag='pool:n')

# ---------------- 左流：写腿 ----------------
WL_X, WL_W = MX, 420
lc.rect(WL_X, 108, WL_W, 560, '#ffffff', SEG_COL[0], rx=8, sw=1.6)
lc.text(WL_X + WL_W / 2, 132, '写腿 · slot_mapping（直寻址）', 11, SEG_COL[0], 'middle', True,
        maxw=WL_W - 20, tag='wl:t')
# 48 个 token 点
DOT_Y = 158
for i in range(48):
    dx = WL_X + 18 + i * ((WL_W - 36) / 48.0)
    seg = i // 16
    lc.rect(dx, DOT_Y, 6.4, 6.4, SEG_FILL[seg], SEG_COL[seg], rx=1, sw=0.7)
for s, lab in [(0, '段 A 16 个'), (1, '段 B 16 个'), (2, '段 C 16 个')]:
    cx = WL_X + 18 + (s * 16 + 8) * ((WL_W - 36) / 48.0)
    lc.text(cx, DOT_Y + 22, lab, 7.5, SEG_COL[s], 'middle', maxw=90, tag='wd%d' % s)
lc.text(WL_X + WL_W / 2, 148, '本拍新算 48 个 token', 9, '#334155', 'middle', True, maxw=WL_W - 20,
        tag='wl:nt')
# 恒等式盒
lc.rect(WL_X + 30, 214, WL_W - 60, 66, SEG_FILL[0], SEG_COL[0], rx=6, sw=1.3)
lc.text(WL_X + WL_W / 2, 236, '槽位恒等式（GPU Triton kernel）', 9, SEG_COL[0], 'middle', True,
        maxw=WL_W - 76, tag='idb:t')
lc.text(WL_X + WL_W / 2, 254, 'slot = 块号 × 16 + pos % 16', 10, '#334155', 'middle', True,
        maxw=WL_W - 76, tag='idb:f')
lc.text(WL_X + WL_W / 2, 270, '行 [3,1,7] → 48 个门牌号', 8, '#64748b', 'middle',
        maxw=WL_W - 76, tag='idb:n')
lc.parrow([(WL_X + WL_W / 2, DOT_Y + 30), (WL_X + WL_W / 2, 210)], SEG_COL[0], 1.8, 'std')
# 门牌号条
LC_Y = 310
lc.text(WL_X + WL_W / 2, LC_Y - 10, 'slot_mapping（每 token 一个门牌号）', 9, '#334155', 'middle',
        True, maxw=WL_W - 30, tag='lc:t')
CHUNKS = [('48..63', 16, 0), ('16..31', 16, 1), ('112..127', 16, 2)]
cx = WL_X + 24
for lab, n, seg in CHUNKS:
    w_ = (WL_W - 48) * n / 48.0
    lc.rect(cx, LC_Y, w_, 30, SEG_FILL[seg], SEG_COL[seg], rx=3, sw=1.1)
    lc.text(cx + w_ / 2, LC_Y + 19, lab, 9, SEG_COL[seg], 'middle', True, maxw=w_ - 6,
            tag='lc%d' % seg)
    cx += w_
# 直箭头 → 池
lc.parrow([(WL_X + WL_W, 325), (POOL_X - 4, 325)], SEG_COL[0], 2.2, 'std')
lc.text((WL_X + WL_W + POOL_X) / 2, 312, '直塞：一 token 一格', 8.5, SEG_COL[0], 'middle', True,
        maxw=110, tag='wl:direct')
# positions 注
lc.text(WL_X + 18, 372, 'positions 本身是 GPU 张量：', 8.5, lc.C_MUTE, 'start', maxw=WL_W - 36,
        tag='wl:p1')
lc.text(WL_X + 18, 390, 'num_computed_tokens[req_indices_gpu]', 8, lc.C_MUTE, 'start',
        maxw=WL_W - 36, tag='wl:p2')
lc.text(WL_X + 18, 407, '＋ query_pos.gpu——换算全程不落 CPU', 8, lc.C_MUTE, 'start',
        maxw=WL_W - 36, tag='wl:p3')
# 写腿小结
lc.rect(WL_X + 18, 428, WL_W - 36, 26, '#ffffff', SEG_COL[0], rx=5, sw=1.1)
lc.text(WL_X + WL_W / 2, 445, '写完：本拍 48 格各就各位', 8.5, SEG_COL[0], 'middle', True,
        maxw=WL_W - 48, tag='wl:done')
lc.text(WL_X + 18, 486, '「门牌号」= 物理槽位号：', 8.5, lc.C_MUTE, 'start', maxw=WL_W - 36,
        tag='wl:note1')
lc.text(WL_X + 18, 504, '算好直接塞、不用查任何表', 8.5, lc.C_MUTE, 'start', maxw=WL_W - 36,
        tag='wl:note2')
lc.text(WL_X + 18, 540, '形状：[max_num_batched_tokens]', 8, lc.C_FAINT, 'start', maxw=WL_W - 36,
        tag='wl:note3')
lc.text(WL_X + 18, 557, 'int64，逐 token 一个数', 8, lc.C_FAINT, 'start', maxw=WL_W - 36,
        tag='wl:note4')

# ---------------- 右流：读腿 ----------------
RL_X, RL_W = 1020, BXR - 1020
lc.rect(RL_X, 108, RL_W, 560, '#ffffff', SEG_COL[2], rx=8, sw=1.6)
lc.text(RL_X + RL_W / 2, 132, '读腿 · 块表张量（间接寻址）', 11, SEG_COL[2], 'middle', True,
        maxw=RL_W - 20, tag='rl:t')
# 块表张量 [4,8]
TT_X, TT_Y = RL_X + 60, 158
TT_CW, TT_CH = 42, 30
lc.text(RL_X + 18, TT_Y + 14, '块表张量 [4, 8]', 9, '#334155', 'start', True, maxw=140,
        tag='tt:t')
ROW0 = [3, 1, 7, 0, 0, 0, 0, 0]
for r in range(4):
    for c in range(8):
        x, y = TT_X + c * TT_CW, TT_Y + 26 + r * TT_CH
        v = ROW0[c] if r == 0 else 0
        if r == 0 and c < 3:
            seg = BLK_SEG[v]
            lc.rect(x, y, TT_CW - 3, TT_CH - 4, SEG_FILL[seg], SEG_COL[seg], rx=3, sw=1.2)
            lc.text(x + (TT_CW - 3) / 2, y + 18, str(v), 9.5, SEG_COL[seg], 'middle', True,
                    maxw=TT_CW - 8, tag='tt%d%d' % (r, c))
        else:
            lc.rect(x, y, TT_CW - 3, TT_CH - 4, '#f8fafc', '#cbd5e1', rx=3, sw=0.8)
            lc.text(x + (TT_CW - 3) / 2, y + 18, str(v), 9, '#94a3b8', 'middle', maxw=TT_CW - 8,
                    tag='tt%d%d' % (r, c))
lc.text(RL_X + 90, TT_Y + 160, '行 0 = [3,1,7,0,...]（pad 行全 0）', 8.5, SEG_COL[2], 'start',
        maxw=RL_W - 100, tag='tt:r0')
lc.text(RL_X + 90, TT_Y + 178, '0 = NULL_BLOCK_ID：块 0 是 null 块，读它永远安全', 8,
        lc.C_MUTE, 'start', maxw=RL_W - 100, tag='tt:null')
lc.text(RL_X + 90, TT_Y + 194, '（CUDA graph padding 语义；get_device_tensor', 8, lc.C_MUTE,
        'start', maxw=RL_W - 100, tag='tt:n1')
lc.text(RL_X + 90, TT_Y + 210, '交 attention metadata builder）', 8, lc.C_MUTE, 'start',
        maxw=RL_W - 100, tag='tt:n2')
# 弯箭头（逐块跳着读）→ 池
HOP_Y = 420
lc.parrow([(RL_X - 2, HOP_Y), (POOL_X + POOL_W + 4, HOP_Y)], SEG_COL[2], 2.2, 'std')
lc.text((RL_X + POOL_X + POOL_W) / 2, HOP_Y - 14, '翻页：逐块跳着读', 8.5, SEG_COL[2], 'middle',
        True, maxw=130, tag='rl:hop')
# attention kernel 图标
AK_Y = 460
lc.rect(RL_X + 40, AK_Y, RL_W - 80, 70, SEG_FILL[2], SEG_COL[2], rx=8, sw=1.4)
lc.text(RL_X + RL_W / 2, AK_Y + 24, '注意力 kernel', 11, SEG_COL[2], 'middle', True,
        maxw=RL_W - 96, tag='ak:t')
lc.text(RL_X + RL_W / 2, AK_Y + 44, '读全部历史 KV——必须学会穿块表', 8.5, '#64748b', 'middle',
        maxw=RL_W - 96, tag='ak:n')
lc.parrow([(RL_X + 60, TT_Y + 26 + 4 * TT_CH + 4), (RL_X + 60, AK_Y - 4)], SEG_COL[2], 1.8, 'std')
lc.text(RL_X + 66, (TT_Y + 26 + 4 * TT_CH + AK_Y) / 2, '查表', 8, SEG_COL[2], 'start', maxw=50,
        tag='rl:lookup')
lc.parrow([(RL_X + RL_W - 60, AK_Y - 4), (RL_X + RL_W - 60, HOP_Y + 4)], SEG_COL[2], 1.8, 'std')
lc.text(RL_X + RL_W - 54, (AK_Y + HOP_Y) / 2 + 8, '取数', 8, SEG_COL[2], 'end', maxw=50,
        tag='rl:fetch')

# ---------------- 汇聚注记 + F7 伏笔（预告 ch22）----------------
CV_Y = POOL_Y + POOL_H + 24
lc.text(POOL_X + POOL_W / 2, CV_Y, '两条腿无先后因果：写腿存本拍 · 读腿读历史——同一池子两种寻址', 8.5,
        lc.C_MUTE, 'middle', maxw=460, tag='cv')
FB_Y = CV_Y + 34
lc.rect(MX, FB_Y, BXR - MX, 40, '#fff7ed', lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(MX + (BXR - MX) / 2, FB_Y + 25, '伏笔 F7（预告 → ch22 结账）：翻页的代价——PagedAttention / FlashAttention 变体 kernel 必须学会间接寻址，复杂度记在 F7 账上（数学 ch20 · 后端选择 ch21 为沿途消费站）',
        9.5, lc.C_BEAT_T, 'middle', True, maxw=BXR - MX - 30, tag='f7')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = FB_Y + 64
lx = MX
for seg, name in [(0, '段 A（块 3）'), (1, '段 B（块 1）'), (2, '段 C（块 7）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, SEG_FILL[seg], SEG_COL[seg], rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=110, tag='lg%d' % seg)
    lx += 26 + lc.tw(name, 8.5) + 16
lc.rect(lx, LEG_Y - 9, 20, 13, '#f8fafc', '#cbd5e1', rx=3, sw=0.8)
lc.text(lx + 26, LEG_Y + 1, '别的块 / pad（读它永远安全）', 8.5, lc.C_TXT, 'start', maxw=210,
        tag='lg:oth')
lx += 26 + lc.tw('别的块 / pad（读它永远安全）', 8.5) + 16
lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, SEG_COL[0], 2.0)
lc.text(lx + 38, LEG_Y + 1, '直寻址（写腿）', 8.5, lc.C_TXT, 'start', maxw=130, tag='lg:w')
lx += 38 + lc.tw('直寻址（写腿）', 8.5) + 14
lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, SEG_COL[2], 2.0)
lc.text(lx + 38, LEG_Y + 1, '间接寻址（读腿 · 翻页）', 8.5, lc.C_TXT, 'start', maxw=180, tag='lg:r')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/worker/gpu_model_runner.py:L2325-L2341（写腿 slot_mapping / 读腿 _get_block_table → get_device_tensor 交 attention metadata builder）· '
        'L2188-L2201（positions GPU 组装）', 8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '写腿槽位分段承前图（块表行 [3,1,7]）；读腿 pad 张量 [4,8] 与未用行全 0 取自配套精简版 host 实跑 · '
        'F7：block_table 间接寻址 → ch22 回收（planted=13 / paid=22）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 66
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-two-legs.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
