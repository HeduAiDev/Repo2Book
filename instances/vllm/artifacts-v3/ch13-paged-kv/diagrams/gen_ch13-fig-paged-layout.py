#!/usr/bin/env python3
"""ch13 机制图 1 · 分页总布局（figure_spec ch13-fig-paged-layout，模板 layout）

放大自 L0『调度 · 显存账本』列（kv_column 青色列）下半的 KVCacheManager/BlockPool
整块 + 跨进程到 GPU 列的 block_id 桥——即本章 L2 章图整幅的机制总览：
north 账本三件套 + center 一个请求 KV 的一生。架构归属回指 L0/L2（FIGURE-SYSTEM
§3.3）：图右上角指北小签。

claim：等大块池 + 每请求逻辑块表：130 token 的两条请求装进 9 块（合计尾部浪费
14 < 2×16），r1 还块后 r3 复用 [7,6,5]——逻辑连续、物理不相邻，这就是
「blocks as pages」的落地。

数字全部取自 figure_spec.numbers（配套精简版 host 实测：130 token / 9 块 144 槽 /
浪费 14；r3 [7,6,5] token 0/16/32 → 槽 112/96/80；旧设计 2048 预留 1948 白买
95.12%；论文 20.4%-38.2%、2-4×）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 928
MX, BXR = 60, 1440
C_R1, C_R2, C_R3 = lc.C_API_S, lc.C_ENG_S, lc.C_SAM_S   # 请求类别色（图例兜底）
F_R1, F_R2, F_R3 = lc.C_API_F, lc.C_ENG_F, lc.C_SAM_F

# ---------------- 标题区 ----------------
lc.text(MX, 34, '分页总布局：130 token 的两条请求装进 9 块，r1 还块后 r3 复用 [7,6,5]',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '整块 KV 显存切成 16 token 一页的等大块池（BlockPool），每个请求拿一张逻辑块表（req_to_blocks）——'
                '论文类比「blocks as pages, tokens as bytes, requests as processes」',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L0 显存账本列（kv_column）· 本章 L2 章图整幅总览'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上栏：旧设计 ----------------
OY, OH = 92, 234
lc.rect(MX, OY, BXR - MX, OH, '#ffffff', '#94a3b8', rx=8, sw=1.4)
lc.text(MX + 16, OY + 22, '旧设计 · 按 max_len 连续预分配（论文举例 max_len=2048）', 11.5, '#475569',
        'start', True, maxw=560, tag='old:t')
lc.text(BXR - 16, OY + 22, '预留即占用：货没来齐，仓位也空着', 9, lc.C_MUTE, 'end', maxw=300, tag='old:s')

BAR_X, BAR_W, BAR_H = MX + 190, 950, 42
PPS = BAR_W / 2048.0            # px / 槽
BARS = [
    (OY + 52, 'r1 预留 2048 槽', 100, C_R1, 'r1 实用 100 槽（终长 100）', '白买 1948 槽 · 从未写入（95.12% 内部碎片）'),
    (OY + 138, 'r2 预留 2048 槽', 30, C_R2, 'r2 实用 30 槽', '白买 2018 槽'),
]
GAP_Y0, GAP_Y1 = OY + 52 + BAR_H, OY + 138      # 两根预留条之间的空档
for by, lab, used, col, ulab, wlab in BARS:
    lc.text(BAR_X - 12, by + BAR_H / 2 + 4, lab, 9.5, '#475569', 'end', maxw=176, tag='old:' + lab[:6])
    lc.rect(BAR_X, by, BAR_W, BAR_H, '#f1f5f9', '#94a3b8', rx=3, sw=1.2)          # 预留整段
    lc.rect(BAR_X, by, used * PPS, BAR_H, col, col, rx=3, sw=1.2)                  # 实用头部
    lc.text(BAR_X + used * PPS + 8, by - 8, ulab, 8.5, col, 'start', maxw=240, tag='old:u' + lab[:4])
    # 灰网格纹（白买段的空置感）——先画线，标签区间留白不让线穿字
    wl_half = lc.tw(wlab, 9.5) / 2 + 10
    wl_cx = BAR_X + (BAR_W + used * PPS) / 2
    gx = BAR_X + used * PPS + 40
    while gx < BAR_X + BAR_W - 14:
        if not (wl_cx - wl_half < gx < wl_cx + wl_half):
            lc.seg(gx, by + 6, gx, by + BAR_H - 6, '#cbd5e1', 0.8)
        gx += 40
    lc.text(wl_cx, by + BAR_H / 2 + 4, wlab, 9.5, '#64748b', 'middle',
            maxw=BAR_W - used * PPS - 30, tag='old:w' + lab[:4])
# 外部碎片注记：两根预留条之间的空档（连续性要求留下的空洞）
DIV_X = BAR_X + 560
lc.seg(DIV_X, GAP_Y0 + 2, DIV_X, GAP_Y1 - 2, '#94a3b8', 1.4, dash=True)
lc.text(DIV_X + 12, (GAP_Y0 + GAP_Y1) / 2 + 4, '段间空洞 = 外部碎片（连续性要求，插不进新请求）',
        8.5, '#64748b', 'start', maxw=330, tag='old:frag')
lc.text(MX + 16, OY + OH - 14, '论文三源浪费：内部碎片（白买）＋ 外部碎片（连续性空洞）＋ 共享前缀重复复制（解药 = 引用计数 + 前缀缓存 → ch15）',
        9, '#475569', 'start', maxw=900, tag='old:3src')

# 右侧论文读数盒
PB_X, PB_W = 1168, 258
lc.rect(PB_X, OY + 44, PB_W, 150, '#fef2f2', lc.C_ABORT, rx=7, sw=1.2)
lc.text(PB_X + 14, OY + 66, '论文实测（arXiv:2309.06180）', 9.5, lc.C_ABORT, 'start', True,
        maxw=PB_W - 28, tag='pb:t')
lc.text(PB_X + 14, OY + 90, '现有系统 KV 有效利用率', 9, '#334155', 'start', maxw=PB_W - 28, tag='pb:1')
lc.text(PB_X + 14, OY + 112, '20.4% - 38.2%', 15, lc.C_ABORT, 'start', True, maxw=PB_W - 28, tag='pb:2')
lc.text(PB_X + 14, OY + 136, '分页后：同延迟下吞吐 2-4×', 9.5, '#334155', 'start', True,
        maxw=PB_W - 28, tag='pb:3')
lc.text(PB_X + 14, OY + 158, '（§5.2 实测 · 是 38.2 不是 38.3）', 8, lc.C_MUTE, 'start',
        maxw=PB_W - 28, tag='pb:4')

# ---------------- 中间对比横幅 ----------------
BN_Y = OY + OH + 14
lc.rect(MX, BN_Y, BXR - MX, 36, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + (BXR - MX) / 2, BN_Y + 23,
        '同一段负载 130 token：旧设计浪费 1948 + 2018 = 3966 槽（两条预留 4096 只用 130，96.83%） vs 分页合计尾部浪费 14 个 token 位（每请求恒 < 1 块）',
        10.5, '#155e75', 'middle', True, maxw=BXR - MX - 30, tag='banner')

# ---------------- 下栏：分页 ----------------
PY, PH = BN_Y + 50, 434
lc.rect(MX, PY, BXR - MX, PH, '#ffffff', lc.C_KV_S, rx=8, sw=1.8)
lc.text(MX + 16, PY + 24, '分页 · 等大块池（16 token/块）＋ 每请求逻辑块表（req_to_blocks）', 11.5,
        lc.C_KV_S, 'start', True, maxw=620, tag='pg:t')
lc.text(BXR - 16, PY + 24, '谁用谁拿、用完还箱——尾块先还（驱逐序）', 9, lc.C_MUTE, 'end',
        maxw=300, tag='pg:s')

# ---- 左：块池（2×5 格） ----
lc.text(MX + 24, PY + 44, '块池 num_gpu_blocks=10 · 可用 9（0 号被 null 占）· 终态：空闲 4（[4,3,2,1]）＋ r3 ×3 ＋ r2 ×2',
        8.5, lc.C_MUTE, 'start', maxw=560, tag='pool:n')
CW, CH, CG = 100, 78, 12
GX0, GY0 = MX + 24, PY + 56
POOL = []          # (cx, cy, bid)
for i in range(10):
    r, c = divmod(i, 5)
    x, y = GX0 + c * (CW + CG), GY0 + r * (CH + CG)
    bid = i
    if bid == 0:
        lc.rect(x, y, CW, CH, '#e2e8f0', '#64748b', rx=5, sw=1.4)
        lc.seg(x + 6, y + CH - 10, x + CW - 6, y + 10, '#64748b', 1.6)
        lc.seg(x + 6, y + 10, x + CW - 6, y + CH - 10, '#64748b', 1.6)
        lc.text(x + CW / 2, y + 22, '0', 13, '#475569', 'middle', True, tag='p0')
        lc.text(x + CW / 2, y + 42, 'null_block', 7.5, '#475569', 'middle', maxw=CW - 8, tag='p0n')
        lc.text(x + CW / 2, y + 58, '封条 · 永不出租', 7.5, '#475569', 'middle', maxw=CW - 8, tag='p0s')
    else:
        held = 3 if bid in (5, 6, 7) else (2 if bid in (8, 9) else 0)
        fill, stroke, sub = {3: (F_R3, C_R3, 'r3 在住'), 2: (F_R2, C_R2, 'r2 在住'),
                             0: ('#ffffff', '#94a3b8', '空闲')}[held]
        lc.rect(x, y, CW, CH, fill, stroke, rx=5, sw=1.4)
        lc.text(x + CW / 2, y + 30, str(bid), 15, stroke if held else '#64748b', 'middle', True, tag='pb%d' % bid)
        lc.text(x + CW / 2, y + 56, sub, 8, stroke if held else '#94a3b8', 'middle', maxw=CW - 8, tag='ps%d' % bid)
    POOL.append((x, y, bid))

# ---- 中：三条逻辑块表票据 ----
TX, TW_ = GX0 + 5 * CW + 4 * CG + 34, 400
STW, STH = 46, 40
TICKETS = [
    (PY + 56, 'r1 · 100 token（已完成离场）', list(range(1, 8)), '#94a3b8', '#f1f5f9',
     '7 块回池：驱逐序 [7,6,5,4,3,2,1]——尾块最先处于被驱逐位', True),
    (PY + 176, 'r2 · 30 token（在住）', [8, 9], C_R2, F_R2,
     'cdiv(30,16)=2 块 · 32 槽 · 尾部浪费 2', False),
    (PY + 296, 'r3 · 35 token（新入住 · 复用 r1 还的块）', [7, 6, 5], C_R3, F_R3,
     'cdiv(35,16)=3 块 · 48 槽 · 尾部浪费 13——提货单连续，堆场里不相邻', False),
]
for ty, tt, ids, col, fill, note, struck in TICKETS:
    lc.text(TX, ty + 4, tt, 9.5, col if col != '#94a3b8' else '#475569', 'start', True, maxw=TW_, tag='tk' + tt[:6])
    sx = TX
    for j, bid in enumerate(ids):
        lc.rect(sx, ty + 14, STW, STH, fill, col, rx=4, sw=1.2)
        lc.text(sx + STW / 2, ty + 40, str(bid), 12, col if col != '#94a3b8' else '#64748b', 'middle', True,
                tag='sk%d%d' % (ids[0], j))
        if struck:
            lc.seg(sx + 5, ty + 14 + STH - 6, sx + STW - 5, ty + 14 + 6, '#64748b', 1.2)
        if j < len(ids) - 1:
            lc.seg(sx + STW + 2, ty + 14 + STH / 2, sx + STW + 8, ty + 14 + STH / 2, '#94a3b8', 1.2)
        sx += STW + 10
    lc.text(TX, ty + 14 + STH + 18, note, 8.5, lc.C_MUTE, 'start', maxw=TW_ + 40, tag='tn' + tt[:6])

# r3 票据 → 池中块 7/6/5 的虚线（复用与不相邻）：
# 干线自票据左缘绕行（票据无容器框，起 点=首 stub 左缘），走廊 y=PY+272 在 r2 票据注记
# （~PY+256 底）与 r3 标题（~PY+289 顶）之间；三支竖线自格底下方上插格底边
pool_by_id = {bid: (x, y) for x, y, bid in POOL}
r3_ty = PY + 296
RUN_Y = PY + 272
lc.parrow([(TX, r3_ty + 30), (648, r3_ty + 30), (648, RUN_Y), (pool_by_id[5][0] + CW - 24, RUN_Y)],
          C_R3, 1.3, None, dash=True)
for bid in (7, 6, 5):
    px, py_ = pool_by_id[bid]
    vx = px + CW - 24                                # 目标格内竖巷
    lc.parrow([(vx, RUN_Y), (vx, py_ + CH + 1)], C_R3, 1.3, 'std', dash=True)

# ---- 右：r3 的 token→槽 小页表 ----
MP_X, MP_W = TX + TW_ + 60, 268
lc.rect(MP_X, PY + 56, MP_W, 200, '#ffffff', lc.C_KV_S, rx=7, sw=1.3)
lc.text(MP_X + MP_W / 2, PY + 78, 'r3 的 token → 槽（页表预览）', 10, lc.C_KV_S, 'middle', True,
        maxw=MP_W - 16, tag='mp:t')
MAPROWS = [('token 0', '块 7 · 槽 112'), ('token 16', '块 6 · 槽 96'), ('token 32', '块 5 · 槽 80')]
for i, (a, b) in enumerate(MAPROWS):
    yy = PY + 106 + i * 34
    lc.rect(MP_X + 18, yy - 14, 88, 24, F_R3, C_R3, rx=4, sw=1.1)
    lc.text(MP_X + 62, yy + 2, a, 9, C_R3, 'middle', True, maxw=80, tag='mp:a%d' % i)
    lc.seg(MP_X + 110, yy - 2, MP_X + 138, yy - 2, C_R3, 1.6, 'std')
    lc.rect(MP_X + 142, yy - 14, 108, 24, '#ffffff', '#cbd5e1', rx=4, sw=1.1)
    lc.text(MP_X + 196, yy + 2, b, 9, '#334155', 'middle', True, maxw=100, tag='mp:b%d' % i)
lc.text(MP_X + MP_W / 2, PY + 224, '逻辑连续 · 物理不相邻', 8.5, lc.C_MUTE, 'middle', maxw=MP_W - 16, tag='mp:n1')
lc.text(MP_X + MP_W / 2, PY + 242, '槽位恒等式后文展开', 8, lc.C_MUTE, 'middle', maxw=MP_W - 16, tag='mp:n2')

# ---- 带底小结 ----
lc.text(MX + (BXR - MX) / 2, PY + PH - 16,
        '130 token 装进 9 块 · 144 槽 · 合计尾部浪费 14（12+2+13）< 2×16——分页把单请求浪费钉死在一块以内',
        10, '#155e75', 'middle', True, maxw=BXR - MX - 30, tag='pg:sum')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = PY + PH + 26
lx = MX
for col, fill, name in [(C_R1, F_R1, 'r1（蓝）'), (C_R2, F_R2, 'r2（橙）'), (C_R3, F_R3, 'r3（品红）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, col, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=110, tag='leg' + name[:2])
    lx += 26 + lc.tw(name, 8.5) + 20
lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', '#94a3b8', rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, '空闲块', 8.5, lc.C_TXT, 'start', maxw=80, tag='leg:free')
lx += 26 + lc.tw('空闲块', 8.5) + 20
lc.rect(lx, LEG_Y - 9, 20, 13, '#e2e8f0', '#64748b', rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, 'null 封条（永不出租）', 8.5, lc.C_TXT, 'start', maxw=170, tag='leg:null')
lx += 26 + lc.tw('null 封条（永不出租）', 8.5) + 20
lc.rect(lx, LEG_Y - 9, 20, 13, '#f1f5f9', '#94a3b8', rx=3, sw=1.2)
for k in range(3):
    lc.seg(lx + 3 + k * 6, LEG_Y - 8, lx + 3 + k * 6, LEG_Y + 3, '#cbd5e1', 0.9)
lc.text(lx + 26, LEG_Y + 1, '白买 / 空置（旧设计）', 8.5, lc.C_TXT, 'start', maxw=170, tag='leg:waste')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/core/block_pool.py:L175-L181（等大块池构造）· '
        'vllm/v1/core/single_type_kv_cache_manager.py:L94-L97（req_to_blocks 逻辑块表）· '
        'vllm/v1/worker/block_table.py:L105-L112（slot_mapping int64 缓冲）', 8.2, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '账面数字取自配套精简版 host 实跑（r1/r2/r3 三事件与池终态）· '
        '旧设计对照为论文口径算术例（arXiv:2309.06180 §2.2 e.g., 2048 tokens）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-paged-layout.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
