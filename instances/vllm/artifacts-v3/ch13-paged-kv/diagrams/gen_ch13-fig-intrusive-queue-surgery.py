#!/usr/bin/env python3
"""ch13 机制图 3 · 自由队列的指针手术（figure_spec ch13-fig-intrusive-queue-surgery，模板 state-table）

放大自 L0『调度 · 显存账本』列（kv_column）中 BlockPool 框内的自由队列层——即本章
L2 章图北行『FreeKVCacheBlockQueue · 自实现链表』框的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：指针长在块上：五步手术的队列全景 [0,1,2,3,4]→[1,2,3,4]→[1,3,4]→[1,3,4,0]→
[3,4,0]→[1,3,4,0]，每步只改两三个块上字段、零对象分配——remove 的 O(1) 中间摘是
ch15 touch 救回命中块的原语前提。

数字全部取自 figure_spec.numbers（配套精简版 host 实测：六行队列状态与 num_free
5→4→3→4→3→4；remove(块 2) 的 1.next→3 / 3.prev→1；零分配 7 对象；哨兵 −1）。
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

# ---------------- 标题区 ----------------
lc.text(MX, 34, '自由队列的指针手术：链表不存在容器里，长在每个块的字段上',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'FreeKVCacheBlockQueue——侵入式双向链表 + 双哨兵（block_id=−1）：popleft_n 取块 / remove O(1) 中间摘 / '
                'append_n 挂尾 / prepend_n 挂头，每步只重接两三个指针',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L2 章图北行「FreeKVCacheBlockQueue · 自实现链表」框 · L0：显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 行参数 ----------------
CW_, CH_, CGAP = 54, 44, 26          # 卡片宽高 / 链间距
HXW = 44                              # 六边形哨兵宽
CHX0 = 460                            # 链区起点（首卡片左缘）
LIFT = 56                             # 被摘卡片抬升高度
ROW0, PITCH = 118, 118

ROWS = [
    # (op 标题, 新链, 摘出块 / 其在新链左邻（None=队头缝）, 挂回块 / 'head', 备注[])
    ('步 0 · 初始：相邻互串 + 双哨兵', [0, 1, 2, 3, 4], None, None,
     ['fake_head(−1) ↔ 0 ↔ 1 ↔ 2 ↔ 3 ↔ 4 ↔ fake_tail(−1)', '每个真实块都有 prev 和 next（哨兵消掉边界分支）']),
    ('步 1 · popleft_n(1)：队头取块', [1, 2, 3, 4], (0, None), None,
     ['块 0 指针置 None；fake_head 直连块 1', '分配取块走这里——null_block 的出生就是它']),
    ('步 2 · remove(块 2)：O(1) 中间摘', [1, 3, 4], (2, 1), None,
     ['块 1.next 直指块 3 · 块 3.prev 直指块 1（链保持）', '块 2 指针清 None——ch15 touch 救回命中块的原语前提']),
    ('步 3 · append_n([0])：归还挂尾', [1, 3, 4, 0], None, (0, 3),
     ['块 0 接到原尾块 4 之后、next 指向 fake_tail', '归还缝入（绿）——队尾 = 最近归还']),
    ('步 4 · popleft()：单取队头', [3, 4, 0], (1, None), None,
     ['拿到块 1；fake_head 直连块 3', '']),
    ('步 5 · prepend_n([1])：挂回队头', [1, 3, 4, 0], None, (1, 'head'),
     ['劈分挂点：ch15 的 LRU 双不变量用（caching 关不触发）', '本章验原语语义——四个原语各自保持链完整']),
]


def hexagon(cx, cy, fill, stroke, label):
    hw, hh = HXW / 2, 15
    pts = ' '.join(f'{cx + dx:.1f},{cy + dy:.1f}' for dx, dy in
                   [(-hw, 0), (-hw + 9, hh), (hw - 9, hh), (hw, 0), (hw - 9, -hh), (-hw + 9, -hh)])
    lc.ELEMS.append(((cx - hw - 2, cy - hh - 2, cx + hw + 2, cy + hh + 2),
                     f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'))
    lc.text(cx, cy + 3, label, 8, '#475569', 'middle', maxw=HXW - 6, tag='hx' + label)


def card(x, y, bid, stroke, fill, dash=False, txtcol=None):
    lc.rect(x, y, CW_, CH_, fill, stroke, rx=6, sw=1.4, dash=dash)
    lc.text(x + CW_ / 2, y + 22, str(bid), 12, txtcol or stroke, 'middle', True, tag='c%s' % bid)
    lc.text(x + CW_ / 2, y + 37, '块', 7.5, txtcol or '#64748b', 'middle', maxw=30, tag='cl%s' % bid)


def link(x1, x2, cy, color='#94a3b8'):
    """双向链（两端箭头）"""
    s = (f'<line x1="{x1 + 2:.1f}" y1="{cy:.1f}" x2="{x2 - 2:.1f}" y2="{cy:.1f}" stroke="{color}" '
         f'stroke-width="1.2" marker-start="url(#std)" marker-end="url(#std)"/>')
    lc.ELEMS.append(((x1, cy - 6, x2, cy + 6), s))


def curve(x1, y1, x2, y2, ctrl_y, color, marker='std'):
    """越过被摘卡片 / 缝入点的弧形箭头"""
    mx = (x1 + x2) / 2
    d = f'M{x1:.1f},{y1:.1f} Q{mx:.1f},{ctrl_y:.1f} {x2:.1f},{y2:.1f}'
    s = f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.0" marker-end="url(#{marker})"/>'
    lc.ELEMS.append(((min(x1, x2) - 8, ctrl_y - 8, max(x1, x2) + 8, max(y1, y2) + 8), s))


for ri, (op, q, pull, push, notes) in enumerate(ROWS):
    ry = ROW0 + ri * PITCH
    cy = ry + CH_ // 2 + 8                      # 链中线
    # 左：步标题 + num_free 徽标
    lc.text(MX, ry + 14, op, 10, lc.C_TXT, 'start', True, maxw=340, tag='op%d' % ri)
    lc.rect(MX, ry + 24, 118, 19, lc.C_BADGE_F, lc.C_KV_S, rx=9, sw=1.1)
    lc.text(MX + 59, ry + 37, 'num_free = %d' % (len(q)), 8.5, lc.C_KV_S, 'middle', True,
            maxw=112, tag='nf%d' % ri)
    # 中：链（哨兵 + 卡片）
    hexagon(CHX0 - 14 - HXW / 2, cy, '#f1f5f9', '#64748b', '−1')
    link(CHX0 - 14 - HXW + 2, CHX0 - 2, cy)
    for i, bid in enumerate(q):
        x = CHX0 + i * (CW_ + CGAP)
        if push is not None and bid == push[0]:
            card(x, ry + 8, bid, lc.C_GPU_S, lc.C_GPU_F)
        else:
            card(x, ry + 8, bid, '#64748b', '#ffffff')
        if i < len(q) - 1:
            link(x + CW_, x + CW_ + CGAP, cy)
    tail_x = CHX0 + len(q) * (CW_ + CGAP) - CGAP
    link(tail_x + 2, tail_x + 14, cy)
    hexagon(tail_x + 14 + HXW / 2, cy, '#f1f5f9', '#64748b', '−1')
    # 被摘出的块：抬升摆在原位上方（队头缝 / 旧邻之间），mid 加弧形新接箭头
    if pull is not None:
        pulled, left_nb = pull
        if left_nb is None:
            px = CHX0 - 7 - CW_ / 2               # 队头缝中心
        else:
            idx = q.index(left_nb)
            px = CHX0 + idx * (CW_ + CGAP) + CW_ + CGAP / 2 - CW_ / 2
            xL = CHX0 + idx * (CW_ + CGAP) + CW_
            curve(xL - CW_ / 2, ry + 8, xL + CGAP + CW_ / 2 - 6, ry + 8, ry - 100, lc.C_ABORT)
        card(px, ry + 8 - LIFT, pulled, lc.C_ABORT, '#fef2f2', dash=True)
        lc.text(px + CW_ + 8, ry + 8 - LIFT + 26, '摘走（指针清 None）', 8, lc.C_ABORT, 'start',
                maxw=150, tag='pull%d' % ri)
    # 挂回的块：绿色缝入弧
    if push is not None:
        pushed, where = push
        if where == 'head':
            curve(CHX0 - 14 - HXW + 4, ry + 14, CHX0 + 10, ry + 8, ry - 18, lc.C_GPU_S)
            lc.text(CHX0 + 34, ry - 18, '挂头', 8, lc.C_GPU_S, 'middle', maxw=50, tag='pushh%d' % ri)
        else:
            xi = q.index(pushed)
            x = CHX0 + xi * (CW_ + CGAP)
            curve(x - CGAP - 8, ry + 8, x + 12, ry + 8, ry - 30, lc.C_GPU_S)
            lc.text(x + CW_ / 2 + 6, ry - 26, '缝入', 8, lc.C_GPU_S, 'middle', maxw=50,
                    tag='push%d' % ri)
    # 右：本步注记
    for ni, nt in enumerate(notes):
        if nt:
            lc.text(980, ry + 16 + ni * 16, '· ' + nt, 8.5, '#64748b', 'start', maxw=BXR - 980,
                    tag='nt%d_%d' % (ri, ni))
BOT = ROW0 + 5 * PITCH + CH_ + 16

# ---------------- 底部：零分配物证 + 为什么不用 deque ----------------
ZY = BOT + 16
lc.rect(MX, ZY, 660, 92, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.4)
lc.text(MX + 16, ZY + 22, '零对象分配物证：全程 7 个对象（5 真实块 + 2 哨兵）', 10, '#166534', 'start',
        True, maxw=620, tag='z:1')
lc.text(MX + 16, ZY + 42, 'id() 集合前后不变——手术只写块上字段，不新建任何 Python 对象', 8.5, '#64748b',
        'start', maxw=620, tag='z:2')
lc.text(MX + 16, ZY + 58, '类 docstring 原话：does not allocate any Python objects', 8.5, lc.C_MUTE,
        'start', maxw=620, tag='z:3')
lc.text(MX + 16, ZY + 78, '配套纪律：slots + 预构空块——调度循环是 CPU 主战场，喂不起 GC', 8.5,
        '#64748b', 'start', maxw=620, tag='z:4')
DX = MX + 690
lc.rect(DX, ZY, BXR - DX, 92, '#ffffff', '#94a3b8', rx=7, sw=1.2, dash=True)
lc.text(DX + 16, ZY + 22, '为什么不用 Python 自带 deque（类 docstring 两条）', 10, lc.C_TXT, 'start', True,
        maxw=BXR - DX - 30, tag='d:1')
lc.text(DX + 16, ZY + 42, '· 要 O(1) 从队伍中间摘人——deque 中间删除 O(n)，ch15 前缀命中天天要摘', 8.5,
        '#64748b', 'start', maxw=BXR - DX - 30, tag='d:2')
lc.text(DX + 16, ZY + 58, '· 逼近 C++ deque 的性能，只能零对象分配', 8.5, '#64748b', 'start',
        maxw=BXR - DX - 30, tag='d:3')
lc.text(DX + 16, ZY + 78, '· 队头 = 最旧空闲（取），队尾 = 最近归还（还）——LRU 次序语义 ch15 展开', 8.5,
        '#64748b', 'start', maxw=BXR - DX - 30, tag='d:4')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = ZY + 116
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', '#64748b', rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, '链上空闲块', 8.5, lc.C_TXT, 'start', maxw=100, tag='lg1')
lx += 26 + lc.tw('链上空闲块', 8.5) + 18
lc.rect(lx, LEG_Y - 11, 20, 15, '#fef2f2', lc.C_ABORT, rx=3, sw=1.2, dash=True)
lc.text(lx + 26, LEG_Y + 1, '被摘出（指针已清 None）', 8.5, lc.C_TXT, 'start', maxw=190, tag='lg2')
lx += 26 + lc.tw('被摘出（指针已清 None）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.2)
lc.text(lx + 26, LEG_Y + 1, '挂回的块（归还 / 挂头）', 8.5, lc.C_TXT, 'start', maxw=180, tag='lg3')
lx += 26 + lc.tw('挂回的块（归还 / 挂头）', 8.5) + 18
hexagon(lx + 10, LEG_Y - 4, '#f1f5f9', '#64748b', '−1')
lc.text(lx + 26, LEG_Y + 1, '哨兵 fake_head / fake_tail（block_id=−1）', 8.5, lc.C_TXT, 'start',
        maxw=280, tag='lg4')
lx += 26 + lc.tw('哨兵 fake_head / fake_tail（block_id=−1）', 8.5) + 22
lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, lc.C_ABORT, 2.0)
lc.text(lx + 38, LEG_Y + 1, '弧形 = 指针新接', 8.5, lc.C_TXT, 'start', maxw=110, tag='lg5')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/core/kv_cache_utils.py:L184-L234（FreeKVCacheBlockQueue 构造与哨兵）· '
        'L273-L304（popleft_n）· L306-L324（remove O(1) 中间摘）· 六行队列状态与 num_free 取自配套精简版 host 实跑',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '队列次序规则的完整语义（LRU 双不变量）属 ch15 驱逐策略——本章只用「队头取、队尾还」的分配语义 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 66
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-intrusive-queue-surgery.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
