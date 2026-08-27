#!/usr/bin/env python3
"""ch11 机制图 6 · 前缀重命中：free 不清哈希（figure_spec ch11-fig-prefix-rehit，模板 before-after）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）——上半 Scheduler 框『调度账本+状态机』位
的 ch11 放大区：即本章 L2 章图 center ⑥ 双队列遍历·前缀重命中拍片 + south『KVCacheManager
（契约面 · 黑盒）』框 free/get_computed_blocks 两条虚线的机制展开；非新架构画法，
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：free 归还块但不清哈希——被抢者恢复时重命中自己的前缀：65 token 只补 1；同 prompt 的
新请求命中 48/64（cap=num_tokens-1 使第 4 块整块重算）。

数字全部取自 figure_spec.numbers（A-2→A-3：4 块归还 空闲 0→4 / cached 哈希 4 留表 / 恢复调度
{r1:1} 命中 64+补 1 / 无缓存=65 全量 / held [-1,-1,-1,-1,4]；B：rA 终点 free 哈希 4 留表 /
rB 命中 3 块=48 第 4 块 16 重算 / cap=63 / NOTE 原文），源出配套精简版 host 实跑 trace。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX, BXR = 60, 1440

EXTRA_DEFS = ('<defs>'
              '<marker id="kvm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'free 归还块、从不擦哈希表——被抢者恢复时重命中自己的前缀：65 token 只补 1',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, '抢占敢扔 KV，是因为扔了还能捡回来（block_pool.py:L719-L742 的 free 只动 ref_cnt 与自由队列）；『重算』从来是『重载元数据 + 补算未命中尾段』',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 拍片 ⑥ + south KV 契约面 · L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

BLK, BLK_GAP = 52, 8                # 块格

def blkcell(x, y, label, sub, fill, stroke, dash=False, sw=1.6):
    lc.rect(x, y, BLK, 44, fill, stroke, rx=5, sw=sw, dash=dash)
    lc.text(x + BLK / 2, y + 19, label, 10, lc.C_TXT, 'middle', True, maxw=BLK - 6, tag='bk' + label + sub)
    lc.text(x + BLK / 2, y + 35, sub, 8, stroke, 'middle', maxw=BLK - 6, tag='bks' + label + sub)

def hashchip(x, y, label, on=True):
    lc.rect(x, y, 34, 26, lc.C_KV_F if on else '#ffffff', lc.C_KV_S, rx=13, sw=1.4)
    lc.text(x + 17, y + 17, label, 9, lc.C_KV_S, 'middle', True, tag='hc' + label)

# ---------------- 左联：A-2 被抢拍 ----------------
LP_X, LP_W = 60, 660
lc.rect(LP_X, 96, LP_W, 360, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.rect(LP_X, 96, LP_W, 26, '#fff7ed', lc.C_ABORT, rx=9, sw=1.2)
lc.text(LP_X + 14, 114, '左联 · A-2 被抢拍：4 块归还池，哈希全留表', 11, lc.C_ABORT, 'start', True,
        maxw=LP_W - 28, tag='lp:t')

# 请求账上的 4 块（归还前虚化）
ry = 160
lc.text(LP_X + 16, ry - 8, 'r1 账上的块（64-token prompt 恰 4 块，第 5 块要不到 → 自我被抢）', 9,
        '#334155', 'start', maxw=LP_W - 32, tag='lp:l0')
bx = LP_X + 40
for i in range(4):
    blkcell(bx + i * (BLK + BLK_GAP), ry, f'块{i}', '', '#ffffff', lc.C_ABORT, dash=True)
    # 归还箭头：每块 → 池
    lc.seg(bx + i * (BLK + BLK_GAP) + BLK / 2, ry + 44, bx + i * (BLK + BLK_GAP) + BLK / 2, ry + 78,
           lc.C_ABORT, 1.6, 'ab')
lc.text(LP_X + LP_W - 16, ry + 62, 'free 逆序归还 · 空闲 0 → 4', 9, lc.C_ABORT, 'end', True,
        maxw=250, tag='lp:free')

# 池
py = ry + 82
lc.rect(bx - 10, py, 4 * BLK + 3 * BLK_GAP + 20, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.6)
lc.text(bx - 10 + (4 * BLK + 3 * BLK_GAP + 20) / 2, py + 17, '块池（4 格全空）', 9.5, lc.C_KV_S, 'middle', True,
        tag='pool:t')
for i in range(4):
    lc.rect(bx + i * (BLK + BLK_GAP), py + 26, BLK, 24, '#ffffff', lc.C_KV_S, rx=4, sw=1.0, dash=True)
    lc.text(bx + i * (BLK + BLK_GAP) + BLK / 2, py + 42, '空', 8, lc.C_KV_S, 'middle', tag='pe' + str(i))

# 哈希表
hy = py + 96
lc.text(LP_X + 16, hy - 8, 'cached 哈希表（block_pool 的 LRU 缓存）', 9, '#334155', 'start',
        maxw=LP_W - 32, tag='lp:h0')
lc.rect(bx - 10, hy, 4 * BLK + 3 * BLK_GAP + 20, 44, '#ffffff', lc.C_KV_S, rx=7, sw=1.4)
for i in range(4):
    hashchip(bx + i * (BLK + BLK_GAP) + 9, hy + 9, f'H{i}')
lc.text(bx - 10 + (4 * BLK + 3 * BLK_GAP + 20) / 2, hy + 60, '4 条指纹全亮着 —— free 不清哈希（恢复时重命中的伏笔，→ ch15）',
        8.8, lc.C_KV_S, 'middle', True, maxw=LP_W - 40, tag='lp:stamp')
lc.text(LP_X + 16, hy + 84, 'block_pool.py:L719-L742：free 只把 ref_cnt 减 1、块挂回自由队列——', 8.6,
        lc.C_MUTE, 'start', maxw=LP_W - 32, tag='lp:h1')
lc.text(LP_X + 16, hy + 100, '哈希条目原样留表（满块指纹覆盖前 0..i 块全部内容的链式哈希）', 8.6,
        lc.C_MUTE, 'start', maxw=LP_W - 32, tag='lp:h2')

# ---------------- 右联：A-3 恢复拍 ----------------
RP_X, RP_W = 760, 680
lc.rect(RP_X, 96, RP_W, 360, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.rect(RP_X, 96, RP_W, 26, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.2)
lc.text(RP_X + 14, 114, '右联 · A-3 恢复拍：get_computed_blocks 沿指纹重命中，65 只补 1', 11, lc.C_GPU_S,
        'start', True, maxw=RP_W - 28, tag='rp:t')

# 指纹 → 命中块
ry2 = 160
lc.text(RP_X + 16, ry2 - 8, 'r1 恢复后的块表（调度 {r1:1}，resumed）', 9, '#334155', 'start',
        maxw=RP_W - 32, tag='rp:l0')
bx2 = RP_X + 40
for i in range(4):
    # 指纹行
    hashchip(bx2 + i * (BLK + BLK_GAP) + 9, ry2, f'H{i}')
    # 命中箭头（指纹 → 块）
    lc.seg(bx2 + i * (BLK + BLK_GAP) + 26, ry2 + 26, bx2 + i * (BLK + BLK_GAP) + 26, ry2 + 48,
           lc.C_KV_S, 1.6, 'kvm')
    blkcell(bx2 + i * (BLK + BLK_GAP), ry2 + 50, f'块{i}', '-1 命中', lc.C_KV_F, lc.C_KV_S)
# 第 5 块：新块
blkcell(bx2 + 4 * (BLK + BLK_GAP), ry2 + 50, '块4', '新块', '#f0fdf4', lc.C_GPU_S)
lc.text(bx2 + 4 * (BLK + BLK_GAP) + BLK / 2, ry2 + 20, '补 1 token', 8.4, lc.C_GPU_S, 'middle', True,
        maxw=70, tag='rp:plus1')
lc.text(RP_X + 16, ry2 + 118, 'held_blocks = [-1, -1, -1, -1, 4]（-1=命中占位；真实为引用计数+1 复用，→ ch13/15）', 8.6,
        lc.C_MUTE, 'start', maxw=RP_W - 32, tag='rp:held')

# 账单对比
by = ry2 + 144
lc.rect(RP_X + 24, by, RP_W - 48, 104, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(RP_X + 40, by + 22, '恢复拍的账单', 10, lc.C_TXT, 'start', True, maxw=RP_W - 80, tag='rp:b0')
lc.rect(RP_X + 40, by + 34, 280, 26, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.4)
lc.text(RP_X + 180, by + 51, '命中 64 + 补算 1（实跑）', 10, lc.C_KV_S, 'middle', True, maxw=270, tag='rp:b1')
lc.rect(RP_X + 340, by + 34, 280, 26, '#ffffff', lc.C_MUTE, rx=5, sw=1.2, dash=True)
lc.text(RP_X + 480, by + 51, '无前缀缓存 = 65 全量重算', 10, lc.C_MUTE, 'middle', maxw=270, tag='rp:b2')
lc.text(RP_X + 40, by + 84, '重算降到约 1/65——v1 recompute-only 赌注的承重墙：期望代价=未命中尾段，不是全长', 8.8,
        '#334155', 'start', maxw=RP_W - 80, tag='rp:b3')
lc.text(RP_X + 16, by + 122, '最坏情况无界：被抢期间块被 LRU 逐出 → 全量 O(prompt+output)——那半边归 ch15', 8.6,
        lc.C_MUTE, 'start', maxw=RP_W - 32, tag='rp:wc')

# 中联箭头：A-2 → A-3（下一拍）
lc.parrow([(LP_X + LP_W / 2, 320), (RP_X, 320)], lc.C_ENG_S, 2.4, 'up')
lc.text((LP_X + LP_W + RP_X) / 2, 306, '下一拍', 9.5, lc.C_ENG_S, 'middle', True, tag='mid1')
lc.text((LP_X + LP_W + RP_X) / 2, 334, 'WAITING 准入', 9, lc.C_ENG_S, 'middle', tag='mid2')

# ---------------- 底部窄条：B 场景 ----------------
BQ_Y, BQ_H = 480, 250
lc.rect(MX, BQ_Y, BXR - MX, BQ_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(MX + 16, BQ_Y + 24, '底部 · B 场景：终点的哈希是下一个请求的礼物（同 prompt rA 跑完 free 后，rB 准入）', 11,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='bq:t')
# rA 行
lc.text(MX + 30, BQ_Y + 52, 'rA 跑完（max_tokens=1 长度封顶）→ 终点 free：4 块归池，哈希 4 条留表', 9,
        '#334155', 'start', maxw=700, tag='bq:a0')
# rB 的 4 块：3 命中 + 1 重算
lc.text(MX + 30, BQ_Y + 84, 'rB（同 64-token prompt）的块表：', 9, '#334155', 'start', maxw=240, tag='bq:b0')
bx3 = MX + 250
labels = [('块0', '命中', lc.C_KV_F, lc.C_KV_S), ('块1', '命中', lc.C_KV_F, lc.C_KV_S),
          ('块2', '命中', lc.C_KV_F, lc.C_KV_S), ('块3', '整块重算', '#fff7ed', lc.C_ABORT)]
for i, (lab, sub, fill, stroke) in enumerate(labels):
    blkcell(bx3 + i * (BLK + BLK_GAP), BQ_Y + 70, lab, sub, fill, stroke)
# cap 注释
cx3 = bx3 + 4 * (BLK + BLK_GAP) + 20
lc.text(cx3, BQ_Y + 84, 'cap = num_tokens − 1 = 63 → 命中按块对齐向下取 3 块 = 48 token', 9, '#334155',
        'start', maxw=560, tag='bq:cap1')
lc.text(cx3, BQ_Y + 104, '第 4 块的哈希虽在表里也被 cap 挡下：整块 16 token 重算', 9, lc.C_ABORT,
        'start', True, maxw=560, tag='bq:cap2')
# NOTE 引文框
nq_y = BQ_Y + 130
lc.rect(MX + 30, nq_y, 660, 100, '#ffffff', lc.C_KV_S, rx=8, sw=1.3, dash=True)
lc.text(MX + 46, nq_y + 22, '为什么必须重算最后一个 token（kv_cache_manager.py:L253-L259 NOTE 原文）', 9.5,
        lc.C_KV_S, 'start', True, maxw=620, tag='nq:t')
for j, ln in enumerate(['「When all tokens hit the cache, we must recompute the last token to obtain',
                        'logits. Thus, set max_cache_hit_length to prompt_length - 1. This can',
                        'trigger recomputation of an entire block … because allocate_slots()',
                        'requires num_computed_tokens to be block-size aligned.」']):
    lc.text(MX + 46, nq_y + 42 + j * 15, ln, 8.4, '#334155', 'start', maxw=630, tag='nq:l' + str(j))
# 右侧：命中统计
hs_x = MX + 720
lc.rect(hs_x, nq_y, BXR - hs_x, 100, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.4)
lc.text(hs_x + 16, nq_y + 22, 'rB 的命中账（实跑 hit_accounting）', 9.5, lc.C_KV_S, 'start', True,
        maxw=400, tag='hs:t')
for j, ln in enumerate(['表里哈希 4 条 · 最大命中 3 块 = 48 token（48/64=75%）',
                        '重算块：第 4 块整块 16 token（哈希在表、被 cap 挡下）',
                        '对照 A 场景：自我恢复 cap=64 → 命中 64 + 补 1']):
    lc.text(hs_x + 16, nq_y + 42 + j * 16, ln, 8.6, '#334155', 'start', maxw=BXR - hs_x - 30,
            tag='hs:l' + str(j))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BQ_Y + BQ_H + 28
lx = MX
for kind, name in [('hit', '前缀命中（青）'), ('new', '补算/新块（绿）'), ('re', '整块重算（红）'),
                   ('hash', '留表的哈希指纹'), ('dash', '归还/对照（虚线）')]:
    if kind == 'hit':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.3)
    elif kind == 'new':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#f0fdf4', lc.C_GPU_S, rx=3, sw=1.3)
    elif kind == 're':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#fff7ed', lc.C_ABORT, rx=3, sw=1.3)
    elif kind == 'hash':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=9, sw=1.2)
    else:
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', lc.C_MUTE, rx=3, sw=1.1, dash=True)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.8) + 20

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_manager.py:L229-L259（get_computed_blocks + cap NOTE）/ L719-L742 于'
        'vllm/v1/core/block_pool.py（free 不清哈希）/ scheduler.py:L744-L766（恢复查缓存）· A/B 两场景账目取自配套精简版 host 实跑'
        '（A：4 块池 64-token；B：8 块池同 prompt×2）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-prefix-rehit.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
