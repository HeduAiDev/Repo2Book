#!/usr/bin/env python3
"""ch19 机制图 5 · 切图五片（figure_spec ch19-fig-split-graph-pieces，模板 flow）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ⑤ 拍片
『切图与逐片编译』（站 7，split_graph）的算法展开。架构归属回指 L0/L2：右上角指北小签。

claim：15 条账本切 24 节点的两层玩具 → 5 个子图：3 片编译（8/10/2 节点）+
2 道接缝（每道恰 kv_update+attention 2 节点、连续切点合并）；拼跑与原图数值
等价（allclose=true、max_abs_diff=0）。

数字全部取自 figure_spec.numbers（五片节点数、连续切点合并、对照 B/C、等价核验、
keep_original_order 重排禁令注释原话——精简版 companion host 实跑 + 源码逐字锚）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 706
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '15 条账本切 24 节点：5 片 = 3 编译 + 2 接缝，拼跑与原图分毫不差',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'split_graph 一次线性游走（backends.py:L553-L627）：遍历 FX 节点、命中账本算子换段子图 id、连续切点合并成一道缝——每个节点恰入一片、有限步终止',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑤ 切图与逐片编译 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 24 节点条 ----------------
NODES = [
    ('l0_in_proj', 0), ('getitem', 0), ('view', 0), ('getitem_1', 0), ('view_1', 0),
    ('getitem_2', 0), ('view_2', 0), ('empty_like', 0),
    ('unified_kv_cache_update', 1), ('unified_attention_with_output', 1),
    ('view_3', 0), ('l0_o_proj', 0), ('l1_in_proj', 0), ('getitem_3', 0), ('view_4', 0),
    ('getitem_4', 0), ('view_5', 0), ('getitem_5', 0), ('view_6', 0), ('empty_like_1', 0),
    ('unified_kv_cache_update_1', 1), ('unified_attention_with_output_1', 1),
    ('view_7', 0), ('l1_o_proj', 0),
]
CUT_AFTER = [7, 9, 19, 21]      # 切刀落点（节点序号之后）
NW_N, NW_OP, NGAP, NH = 48, 74, 5, 40
NY = 108
x = MX
node_x = []
WRAP = {
    'unified_kv_cache_update': ('unified_kv_cache', '_update'),
    'unified_kv_cache_update_1': ('unified_kv_cache', '_update_1'),
    'unified_attention_with_output': ('unified_attention', '_with_output'),
    'unified_attention_with_output_1': ('unified_attention', '_with_output_1'),
}
for nm, is_op in NODES:
    w = NW_OP if is_op else NW_N
    node_x.append((x, w))
    lc.rect(x, NY, w, NH, lc.C_BEAT_F if is_op else '#ffffff',
            lc.C_BEAT_S if is_op else lc.C_MUTE, rx=4, sw=1.2 if is_op else 0.9)
    if is_op:
        a, b = WRAP[nm]
        lc.text(x + w / 2, NY + 17, a, 6.3, lc.C_BEAT_T, 'middle', maxw=w - 4, tag='n' + nm)
        lc.text(x + w / 2, NY + 30, b, 6.3, lc.C_BEAT_T, 'middle', maxw=w - 4, tag='n2' + nm)
    else:
        lc.text(x + w / 2, NY + 24, nm, 6.3, '#334155', 'middle', maxw=w - 4, tag='n' + nm)
    x += w + NGAP
STRIP_R = x - NGAP
lc.text(MX, NY - 8, 'FX 图 24 个计算节点（真实节点名单）', 8.5, lc.C_MUTE, 'start', True,
        maxw=300, tag='strip:t')

# 切刀虚线
for i in CUT_AFTER:
    cx = node_x[i][0] + node_x[i][1] + NGAP / 2
    lc.seg(cx, NY - 26, cx, NY + NH + 22, lc.C_ABORT, 1.2, dash=True)

# ---------------- 五个子图框（对齐各自节点跨度） ----------------
PY0 = NY + NH + 22
pieces = [
    ('submod_0', 0, 7, '编译', '8 节点'),
    ('submod_1', 8, 9, '缝', '2 节点'),
    ('submod_2', 10, 19, '编译', '10 节点'),
    ('submod_3', 20, 21, '缝', '2 节点'),
    ('submod_4', 22, 23, '编译', '2 节点'),
]
PH_ = 58
for name, i0, i1, kind, cnt in pieces:
    x0 = node_x[i0][0]
    x1 = node_x[i1][0] + node_x[i1][1]
    seam = (kind == '缝')
    lc.rect(x0, PY0, x1 - x0, PH_, lc.C_BEAT_F if seam else lc.C_GPU_F,
            lc.C_BEAT_S if seam else lc.C_GPU_S, rx=6, sw=1.6)
    col = lc.C_BEAT_T if seam else lc.C_GPU_S
    cxm = (x0 + x1) / 2
    lc.text(cxm, PY0 + 22, name, 9.5, col, 'middle', True, maxw=x1 - x0 - 6, tag='p' + name)
    lab = ('eager 接缝 · ' + cnt) if seam else ('送 Inductor · ' + cnt)
    lc.text(cxm, PY0 + 40, lab, 7.4, col, 'middle', maxw=x1 - x0 - 6, tag='pl' + name)
    # 切刀到框的连接（刀线已到框顶）
lc.text(MX, PY0 + PH_ + 18, '每节点恰入一片 · 子图间连线即原数据流边——切图是纯重排不重写',
        8.3, lc.C_MUTE, 'start', maxw=700, tag='walk:l')

# ---------------- 接缝特写 + 等价核验/重排禁令 ----------------
IY = PY0 + PH_ + 36
# 接缝特写（左）
lc.rect(MX, IY, 640, 108, '#ffffff', lc.C_BEAT_S, rx=8, sw=1.5)
lc.text(MX + 14, IY + 20, '接缝特写：连续切点合并', 9.5, lc.C_BEAT_T, 'start', True,
        maxw=300, tag='ins:t')
lc.rect(MX + 24, IY + 34, 232, 34, lc.C_BEAT_F, lc.C_BEAT_S, rx=5, sw=1.2)
lc.text(MX + 24 + 116, IY + 55, 'unified_kv_cache_update', 8, lc.C_BEAT_T, 'middle', True,
        maxw=224, tag='ins:a')
lc.parrow([(MX + 256, IY + 51), (MX + 284, IY + 51)], lc.C_BEAT_S, 1.6)
lc.rect(MX + 284, IY + 34, 264, 34, lc.C_BEAT_F, lc.C_BEAT_S, rx=5, sw=1.2)
lc.text(MX + 284 + 132, IY + 55, 'unified_attention_with_output', 8, lc.C_BEAT_T, 'middle',
        True, maxw=256, tag='ins:b')
lc.rect(MX + 566, IY + 38, 60, 26, lc.C_BADGE_F, lc.C_BEAT_S, rx=12, sw=1.0)
lc.text(MX + 596, IY + 55, '同一缝', 8, lc.C_BEAT_T, 'middle', True, maxw=54, tag='ins:bdg')
lc.text(MX + 14, IY + 94, '两切点连续 → 先减后增立即抵消、取得同一 subgraph id：L=2 层得 2L+1=5 片；不合并则 3L+1=7 片、每缝 1 节点。',
        8.0, '#334155', 'start', maxw=612, tag='ins:l')

# 等价核验 + 重排禁令（右）
EX, EY, EW, EH = 724, IY, BXR - 724, 108
lc.rect(EX, EY, EW, EH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(EX + 14, EY + 20, '数值等价核验（实跑）', 9.5, lc.C_GPU_S, 'start', True, maxw=240,
        tag='eq:t')
lc.text(EX + 14, EY + 40, '5 片按序拼跑 split_gm(x) 对比原图 gm(x)：allclose=true · max_abs_diff=0.0；', 8.2,
        '#334155', 'start', maxw=EW - 28, tag='eq:l1')
lc.text(EX + 14, EY + 58, '每层 impl 各收到 2 次 kv_update + 2 次 forward（两次运行各一次），顺序保持。', 8.2,
        '#334155', 'start', maxw=EW - 28, tag='eq:l2')
lc.text(EX + 14, EY + 82, 'keep_original_order is important! otherwise pytorch might reorder the nodes and', 7.8,
        '#334155', 'start', maxw=EW - 28, tag='eq:q1')
lc.text(EX + 14, EY + 96, 'the semantics of the graph will change when we have mutations（L595-L608 原话）', 7.8,
        '#334155', 'start', maxw=EW - 28, tag='eq:q2')

# ---------------- 底部两条对照条 ----------------
BY = IY + EH + 22
lc.text(MX, BY, '对照 B（账本只留 13 注意力算子）：', 8.8, lc.C_TXT, 'start', True, maxw=280,
        tag='b:t')
BC = [('9', True), ('1', False), ('11', True), ('1', False), ('2', True)]
bx = MX + 270
for cnt, danger in BC:
    w = 56 if len(cnt) < 3 else 62
    seam = (cnt == '1')
    lc.rect(bx, BY - 16, w, 32, lc.C_BEAT_F if seam else lc.C_GPU_F,
            lc.C_BEAT_S if seam else lc.C_GPU_S, rx=5, sw=1.2)
    lc.text(bx + w / 2, BY + 4, cnt, 10, lc.C_BEAT_T if seam else lc.C_GPU_S, 'middle', True,
            maxw=w - 4, tag='bc' + cnt)
    if danger:
        lc.circle(bx + w / 2, BY, 17, lc.C_ABORT, 1.3, dash=True)
    bx += w + 8
lc.text(bx + 4, BY + 4, 'kv_update 落进编译片 submod_0 / submod_2——片数仍 5', 8.3,
        lc.C_ABORT, 'start', maxw=460, tag='b:l')

CY = BY + 34
lc.text(MX, CY, '对照 C（空账本）：', 8.8, lc.C_TXT, 'start', True, maxw=200, tag='c:t')
lc.rect(MX + 270, CY - 16, 130, 32, lc.C_GPU_F, lc.C_GPU_S, rx=5, sw=1.2)
lc.text(MX + 270 + 65, CY + 4, '24', 10, lc.C_GPU_S, 'middle', True, maxw=120, tag='cc')
lc.text(MX + 412, CY + 4, '整图 1 片全编译——attention 的外部副作用进不了图：旧死结，此路不通。', 8.3,
        lc.C_MUTE, 'start', maxw=700, tag='c:l')

# ---------------- 页脚 ----------------
lc.text(MX, 668, '逐字锚 vllm/compilation/backends.py:L553-L627（split_graph 游走+连续切点合并+keep_original_order）· vllm/compilation/partition_rules.py:L14-L38（should_split）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 686, '五片节点数 8/2/10/2/2、接缝内容、对照 B/C、等价核验（allclose=true · max_abs_diff=0.0 · 每层 2 次 kv_update+2 次 forward）取自精简版 companion host 实跑',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-split-graph-pieces.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
