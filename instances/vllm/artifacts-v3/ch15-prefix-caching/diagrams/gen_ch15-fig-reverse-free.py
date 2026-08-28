#!/usr/bin/env python3
"""ch15 机制图 6 · 逆序 free 的驱逐序（figure_spec ch15-fig-reverse-free，模板 before-after）

放大自 L0 KV 账本列（kv_column）缓存区·留与逐——「逆序 free」一格的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：free 必须逆序传入——尾块（复用条件最苛刻）排最靠驱逐端、链头（人人可用）沉到
最可复用端；正序 free 会让池紧时先砍前缀的头。

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
RED = '#dc2626'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '还书要倒着还：逆序 free 把链尾排到驱逐端——池紧先丢「最难被借走」的块',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '同一条 3 块链、同一条 free_blocks 原语，只差传入顺序：逆序取走一块后 32-token 前缀照常命中；正序同样取走一块、命中直接归零',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 留与逐「逆序 free」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96
PW = (BXR - MX - 24) / 2
PH = 392

PANELS = [
    dict(tag='a', x=MX, title='真实路径：manager.free 逆序传入', real=True,
         order='reversed([1, 2, 3]) = [3, 2, 1]',
         queue=[3, 2, 1], take='取走链尾 3（覆盖 0-47 的整链块）',
         result='来者命中 32 token（链头两块 1、2 仍可命中）——48-token 链保住 67%',
         col=GREEN, rfill='#f0fdf4'),
    dict(tag='b', x=MX + PW + 24, title='反事实：正序传入（违反约定）', real=False,
         order='[1, 2, 3]（正序，违反约定）',
         queue=[1, 2, 3], take='取走链头 1（任何 ≥16 token 的共享前缀都用得上它）',
         result='命中 0——链头一断、整条前缀报废（同样只取走 1 块）',
         col=RED, rfill='#fef2f2'),
]
CHIPS = [('块 1', 'token 0-15', '链头 · 前缀 ≥16 即可复用'),
         ('块 2', 'token 16-31', '中段 · 前缀 ≥32 才可复用'),
         ('块 3', 'token 32-47', '链尾 · 前缀 ≥48 才可复用（最苛刻）')]
for P in PANELS:
    px, real = P['x'], P['real']
    lc.rect(px, LY, PW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
    lc.text(px + 16, LY + 22, P['title'], 11.5, P['col'], 'start', True, maxw=PW - 32,
            tag="p%s:t" % P['tag'])
    # 链
    cy = LY + 42
    CW3, CH3, CG3 = 186, 52, 14
    cx = px + 18
    for i, (nm, tok, cond) in enumerate(CHIPS):
        lc.rect(cx, cy, CW3, CH3, '#ffffff', lc.C_MUTE, rx=5, sw=1.2)
        lc.text(cx + CW3 / 2, cy + 21, '%s · %s' % (nm, tok), 9.6, lc.C_TXT, 'middle', True,
                maxw=CW3 - 8, tag="p%s:c%d" % (P['tag'], i))
        lc.text(cx + CW3 / 2, cy + 40, cond, 7.8, GRAY, 'middle', maxw=CW3 - 8,
                tag="p%s:d%d" % (P['tag'], i))
        if i < 2:
            lc.seg(cx + CW3, cy + CH3 / 2, cx + CW3 + CG3, cy + CH3 / 2, lc.C_MUTE, 1.4, 'std')
        cx += CW3 + CG3
    lc.text(px + 18, cy + CH3 + 18, '48-token 请求的 3 块链（分配序 = 链序）· free 传入：%s' % P['order'],
            8.8, '#334155', 'start', maxw=PW - 36, tag="p%s:fo" % P['tag'])
    # 自由队列条
    qy = cy + CH3 + 36
    lc.text(px + 18, qy, '自由队列尾段（头→尾）：', 9, lc.C_TXT, 'start', True, maxw=190,
            tag="p%s:ql" % P['tag'])
    QC, QG = 34, 8
    qx = px + 18 + 150
    for j, b in enumerate(P['queue']):
        first = (j == 0)
        lc.rect(qx, qy - 16, QC, 28, P['rfill'], P['col'], rx=4, sw=1.3)
        lc.text(qx + QC / 2, qy + 3, str(b), 10, P['col'], 'middle', True, maxw=QC - 4,
                tag="p%s:q%d" % (P['tag'], j))
        qx += QC + QG
    lc.text(px + 18 + 150 - 8, qy + 3, '…', 10, GRAY, 'end', maxw=20, tag="p%s:qe" % P['tag'])
    lc.text(qx, qy + 3, '← 驱逐先来；队尾 = 最可复用端', 8.2, GRAY, 'start', maxw=PW - 18 - (qx - px),
            tag="p%s:qn" % P['tag'])
    if real:
        lc.text(px + 18, qy + 22, '链尾 3 最靠驱逐端：复用条件最苛刻、潜在复用者最少', 8.2,
                '#475569', 'start', maxw=PW - 36, tag="p%s:qn2" % P['tag'])
    else:
        lc.text(px + 18, qy + 22, '链头 1 最靠驱逐端：人人都可能借它——却最先被丢', 8.2,
                '#475569', 'start', maxw=PW - 36, tag="p%s:qn2" % P['tag'])
    # 池紧取 1 块
    ty = qy + 46
    lc.rect(px + 18, ty, PW - 36, 44, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
    lc.text(px + 32, ty + 18, '池 4 块（null + 这 3 块）· 池紧取 1 块：', 8.8, lc.C_TXT, 'start',
            True, maxw=PW - 68, tag="p%s:t1" % P['tag'])
    lc.text(px + 32, ty + 35, P['take'], 8.6, '#334155', 'start', maxw=PW - 68,
            tag="p%s:t2" % P['tag'])
    # 结果
    ry = ty + 58
    lc.rect(px + 18, ry, PW - 36, 62, P['rfill'], P['col'], rx=6, sw=1.3)
    lc.text(px + 32, ry + 24, P['result'], 9.6, P['col'], 'start', True, maxw=PW - 68,
            tag="p%s:r1" % P['tag'])
    lc.text(px + 32, ry + 46, ('逆序约定：丢 j 块仍剩 (3−j)×16 token 可命中' if real
                               else '正序约定最坏：丢 1 块即全灭（0% vs 67%）'), 8.4, '#334155',
            'start', maxw=PW - 68, tag="p%s:r2" % P['tag'])

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + PH + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '顺序即策略：链上第 i 块可被复用 ⟺ 来者前缀 ≥ (i+1)·16——i 越大条件越苛刻，且潜在复用者集合随 i 嵌套递减',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '这条规则写在 FreeKVCacheBlockQueue 类文档里、由类外的调用约定维持（"This operation is outside of this class"）——没有断言兜底，传错顺序不会报错、只会默默掉命中率',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, tcol, name in [
        ('#f0fdf4', GREEN, GREEN, '真实路径（逆序）'),
        ('#fef2f2', RED, RED, '反事实（正序）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=160, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, '队列条只画本请求相关的尾段（…= 队列更早内容）；「取 1 块」= 池紧时 get_new_blocks 从队头拿块',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_utils.py:L184-L207（FreeKVCacheBlockQueue 类文档：驱逐序规则与「reversing the block order … outside of this class」）· '
        'vllm/v1/core/single_type_kv_cache_manager.py:L519-L527（manager.free 逆序传入）· vllm/v1/core/block_pool.py:L719-L742（free_blocks append_n）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（48 token 3 块链 · 池 4 块小池对照）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=620, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-reverse-free.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
