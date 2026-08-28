#!/usr/bin/env python3
"""ch15 机制图 7 · free_blocks 的 LRU 劈分（figure_spec ch15-fig-split-free，模板 before-after）

放大自 L0 KV 账本列（kv_column）缓存区·留与逐——「劈分」一格的展开（与逆序 free 成对：
逆序定链序、劈分定两类块的相对序）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：free_blocks 把归零块劈两半——无哈希块 prepend 到队头先驱逐（never match APC），
有哈希块 append 到 LRU 尾；缓存关闭时跳过劈分全部 append 保 GPU 局部性。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
GREEN = '#16a34a'
ORANGE = '#ea580c'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '归零块劈两半：无哈希块插队队头先走，带哈希块沉 LRU 尾——差 6 个身位',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '无哈希块从未入表、永不可能命中（never match APC）——先还不损失任何命中率；缓存关闭时跳过劈分、全部 append：刚用过的块沉队尾待复用',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 留与逐「劈分」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96
PW = (BXR - MX - 24) / 2
PH = 400

QC, QG = 40, 8   # 队列格宽/距


def queue_strip(x, y, cells, tag):
    """cells = [(块号, kind)]，kind ∈ no(无哈希)/yes(有哈希)/old(原有空闲)"""
    qx = x
    for j, (b, kind) in enumerate(cells):
        if kind == 'no':
            fill, stroke, tcol = '#fff7ed', ORANGE, ORANGE
        elif kind == 'yes':
            fill, stroke, tcol = lc.C_KV_F, lc.C_KV_S, lc.C_KV_S
        else:
            fill, stroke, tcol = '#ffffff', GRAY, GRAY
        lc.rect(qx, y, QC, 32, fill, stroke, rx=4, sw=1.2)
        lc.text(qx + QC / 2, y + 21, str(b), 10, tcol, 'middle', True, maxw=QC - 4,
                tag='%s:q%d' % (tag, j))
        qx += QC + QG
    return qx


# ---------------- 左：缓存开（劈分生效） ----------------
AX = MX
lc.rect(AX, LY, PW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(AX + 16, LY + 22, '缓存开（enable_caching=True）：劈分生效', 11.5, lc.C_TXT, 'start',
        True, maxw=PW - 32, tag='a:t')
lc.text(AX + 16, LY + 44, '一次 free 归还 3 块：1、3（无哈希）+ 2（有哈希）· 池 8 块（0 号 = null）',
        8.8, '#334155', 'start', maxw=PW - 32, tag='a:s')
# 之前队列
qy0 = LY + 66
lc.text(AX + 16, qy0, '劈分前的自由队列（头→尾）：', 9, lc.C_TXT, 'start', True, maxw=200,
        tag='a:b')
queue_strip(AX + 16, qy0 + 8, [(4, 'old'), (5, 'old'), (6, 'old'), (7, 'old')], 'a:bq')
# 劈分动作
sy = qy0 + 62
lc.text(AX + 16, sy, 'free_blocks 内部劈两半：', 9, lc.C_TXT, 'start', True, maxw=200, tag='a:sp')
lc.rect(AX + 16, sy + 8, (PW - 32 - 12) / 2, 44, '#fff7ed', ORANGE, rx=5, sw=1.2)
lc.text(AX + 16 + 12, sy + 26, '无哈希 1、3 → prepend_n 到队头', 8.8, ORANGE, 'start', True,
        maxw=(PW - 32 - 12) / 2 - 24, tag='a:sp1')
lc.rect(AX + 16 + (PW - 32 - 12) / 2 + 12, sy + 8, (PW - 32 - 12) / 2, 44, lc.C_KV_F, lc.C_KV_S,
        rx=5, sw=1.2)
lc.text(AX + 16 + (PW - 32 - 12) / 2 + 24, sy + 26, '有哈希 2 → append_n 到 LRU 尾', 8.8,
        lc.C_KV_S, 'start', True, maxw=(PW - 32 - 12) / 2 - 24, tag='a:sp2')
# 之后队列
qy1 = sy + 74
lc.text(AX + 16, qy1, '劈分后的自由队列（头→尾）：', 9, lc.C_TXT, 'start', True, maxw=200,
        tag='a:af')
endx = queue_strip(AX + 16, qy1 + 8,
                   [(1, 'no'), (3, 'no'), (4, 'old'), (5, 'old'), (6, 'old'), (7, 'old'),
                    (2, 'yes')], 'a:aq')
lc.text(AX + 16, qy1 + 58, '队头 = 驱逐端：无哈希 1、3 插队先走（永不可能命中，先还容量）',
        8.4, ORANGE, 'start', True, maxw=PW - 32, tag='a:n1')
lc.text(AX + 16, qy1 + 76, '队尾 = 最可复用端：带哈希块 2 沉底——驱逐优先级差 6 个身位',
        8.4, lc.C_KV_S, 'start', True, maxw=PW - 32, tag='a:n2')

# ---------------- 右：缓存关（跳过劈分） ----------------
BX = MX + PW + 24
lc.rect(BX, LY, PW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(BX + 16, LY + 22, '缓存关（enable_caching=False）：跳过劈分', 11.5, lc.C_TXT, 'start',
        True, maxw=PW - 32, tag='b:t')
lc.text(BX + 16, LY + 44, '命中恒 0、劈分失去意义——全部走 append 分', 8.8, '#334155', 'start',
        maxw=PW - 32, tag='b:s')
qy0b = LY + 66
lc.text(BX + 16, qy0b, '同三块、同一条 free_blocks（传入 [3,2,1]），全 append：', 9, lc.C_TXT,
        'start', True, maxw=300, tag='b:b')
queue_strip(BX + 16, qy0b + 8, [(4, 'old'), (5, 'old'), (6, 'old'), (7, 'old')], 'b:bq')
qy1b = qy0b + 62
lc.text(BX + 16, qy1b, '归还后的自由队列（头→尾）：', 9, lc.C_TXT, 'start', True, maxw=200,
        tag='b:af')
queue_strip(BX + 16, qy1b + 8,
            [(4, 'old'), (5, 'old'), (6, 'old'), (7, 'old'), (3, 'no'), (2, 'yes'), (1, 'no')],
            'b:aq')
lc.text(BX + 16, qy1b + 58, '尾段 3、2、1：刚用过的块追加回队尾（逆序传入后 3、2、1 依次 append）',
        8.4, '#334155', 'start', maxw=PW - 32, tag='b:n1')
lc.text(BX + 16, qy1b + 76, '下次分配从队头拿不到它们 → 大概率复用同一物理块，GPU 显存局部性更好',
        8.4, '#334155', 'start', maxw=PW - 32, tag='b:n2')
# 对照小结
cy = qy1b + 100
lc.rect(BX + 16, cy, PW - 32, 52, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
lc.text(BX + 30, cy + 20, '同一条 free 路径、两种模式各取所需：', 8.8, lc.C_TXT, 'start', True,
        maxw=PW - 60, tag='b:c1')
lc.text(BX + 30, cy + 39, '开缓存保命中率（无哈希先走）；关缓存保局部性（刚用过的沉尾）', 8.6,
        '#334155', 'start', maxw=PW - 60, tag='b:c2')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + PH + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不变量：无哈希块先于一切缓存块被驱逐，且命中率严格不降（它从未入表 ⇒ 永不命中 ⇒ 存活对命中概率零贡献，有哈希块全部后移）',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '若不劈分：这 2 个无哈希块就混在 LRU 端白占 2/8 = 25%；随每次 free 累积，8 块池滞留 3 块即 3/8 = 37.5%（这就是 #42656 修的「白占容量」坑）',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, tcol, name in [
        ('#fff7ed', ORANGE, ORANGE, '无哈希块（从未入表 · never match APC）'),
        (lc.C_KV_F, lc.C_KV_S, lc.C_KV_S, '有哈希块（驱逐候选 · LRU 尾端）'),
        ('#ffffff', GRAY, GRAY, '原有空闲块')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=250, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '队列方向：左 = 队头（先驱逐）→ 右 = 队尾（最可复用）；null（0 号）不参与自由队列',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/block_pool.py:L719-L742（free_blocks：劈分 + prepend_n/append_n；'
        '注释明言缓存关时 append 保 GPU cache locality）· 改进出处 #42656（无哈希块白占容量）/ #48017（关缓存跳过劈分）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（池 8 块 · 一次 free 归还 1、3、2）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=620, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-split-free.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
