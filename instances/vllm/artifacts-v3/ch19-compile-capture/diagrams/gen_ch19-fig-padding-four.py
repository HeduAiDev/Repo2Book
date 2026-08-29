#!/usr/bin/env python3
"""ch19 机制图 8 · padding 四件套（figure_spec ch19-fig-padding-four，模板 before-after）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ⑦ 拍片
『一拍裁决·查表·padding』（站 12，padding 四件套）的机制展开。
架构归属回指 L0/L2：右上角指北小签。

claim：同一拍（5 活跃→pad 8、命中 FULL key (8,8,True)、白算 3 行）四个持久缓冲
的 pad 段各写一个专属哨兵：qsl 尾=5（非递减）、block_table 尾行=0（保留块）、
slot_mapping 尾=-1（跳写）、positions 尾=0（垃圾无害）——活跃前缀逐字节不动。

数字全部取自 figure_spec.numbers（裁决 ruling / 四件套 before/after 数组——
精简版 companion host 实跑，裁决与 slot_mapping 走真实方法全调用）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 856
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一拍 5→8：四个持久缓冲的 pad 段各写一个专属哨兵，活跃前缀逐字节不动',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '把每拍变形的 batch 塞进捕获形状：『活跃前缀原样 + pad 段专属哨兵』——每个哨兵对着一个具体 kernel 的安全条件（gpu_model_runner.py 四段 pinned span）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ 一拍裁决·查表·padding · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 裁决横幅 ----------------
BY, BH_ = 92, 54
lc.rect(MX, BY, BXR - MX, BH_, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.5)
lc.text(MX + 16, BY + 22, '裁决（真实 _determine_batch_execution_and_padding）：num_tokens=5 · num_reqs=5 · uniform decode → 构 key (8,8,True) 命中 FULL',
        9.3, lc.C_GPU_S, 'start', True, maxw=BXR - MX - 500, tag='ban:l1')
lc.text(MX + 16, BY + 42, '→ num_tokens_padded=8、num_reqs_padded=8——白算 3 行 decode forward', 9.3,
        '#334155', 'start', maxw=700, tag='ban:l2')
lc.rect(BXR - 250, BY + 12, 232, 30, '#ffffff', lc.C_BEAT_S, rx=8, sw=1.3)
lc.text(BXR - 134, BY + 32, 'CUDAGraphStat.num_paddings = 3', 9, lc.C_BEAT_T, 'middle', True,
        maxw=222, tag='ban:b')

# ---------------- 四行 before/after ----------------
LX, LW = MX, 196          # 左标签
BFX = 270                 # before 数组起点
ARX0, ARX1 = 760, 836     # 中箭头区
AFX = 846                 # after 数组起点
WHYX = 1180               # 右侧安全说明

CELL_H, CELL_FS = 34, 9.5

def cell(x, y, w, s, kind):
    if kind == 'act':
        lc.rect(x, y, w, CELL_H, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.0)
        lc.text(x + w / 2, y + CELL_H / 2 + 3.5, s, CELL_FS, lc.C_GPU_S, 'middle', True,
                maxw=w - 4, tag='c' + s + kind)
    elif kind == 'stale':
        lc.rect(x, y, w, CELL_H, '#f1f5f9', '#cbd5e1', rx=4, sw=0.9)
        lc.text(x + w / 2, y + CELL_H / 2 + 3.5, s, CELL_FS, lc.C_MUTE, 'middle',
                maxw=w - 4, tag='c' + s + kind)
    else:  # pad 哨兵
        lc.rect(x, y, w, CELL_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
        lc.text(x + w / 2, y + CELL_H / 2 + 3.5, s, CELL_FS, lc.C_BEAT_T, 'middle', True,
                maxw=w - 4, tag='c' + s + kind)

ROWS = [
    dict(name='query_start_loc', role='每请求 token 偏移（CU 前缀和）',
         anchor='gpu_model_runner.py:L2073-L2078', buf=9,
         before=[('0', 'stale')] * 9,
         after=[('0', 'act'), ('1', 'act'), ('2', 'act'), ('3', 'act'), ('4', 'act'),
                ('5', 'act'), ('5', 'pad'), ('5', 'pad'), ('5', 'pad')],
         span='写尾段 [6:9)',
         why=['尾 3 项 = cu 末值 5：cu_seqlen 保持非递减——',
              'kernels like FlashAttention requires that（注释原话）；',
              'pad 段区间长 cu[-1]−cu[-1]=0，不派发工作。']),
    dict(name='block_table', role='请求 → 块页表',
         anchor='gpu_model_runner.py:L2338-L2341', buf=8,
         before=[('7,9', 'act'), ('12,15', 'act'), ('3,6', 'act'), ('11,2', 'act'),
                 ('8,10', 'act'), ('13,4', 'stale'), ('9,14', 'stale'), ('5,6', 'stale')],
         after=[('7,9', 'act'), ('12,15', 'act'), ('3,6', 'act'), ('11,2', 'act'),
                ('8,10', 'act'), ('0,0', 'pad'), ('0,0', 'pad'), ('0,0', 'pad')],
         span='写尾段 [5:8)',
         why=['尾行 NULL_BLOCK_ID=0——Block 0 is reserved',
              'for padding（注释原话）：pad 行读到空页、',
              '算出垃圾不外泄。']),
    dict(name='slot_mapping', role='token → KV 物理槽位',
         anchor='gpu_model_runner.py:L4128-L4130', buf=8,
         before=[('10', 'act'), ('11', 'act'), ('12', 'act'), ('13', 'act'), ('14', 'act'),
                 ('99', 'stale'), ('98', 'stale'), ('97', 'stale')],
         after=[('10', 'act'), ('11', 'act'), ('12', 'act'), ('13', 'act'), ('14', 'act'),
                ('-1', 'pad'), ('-1', 'pad'), ('-1', 'pad')],
         span='写尾段 [5:8)',
         why=['尾 -1：reshape_and_cache 逐槽判 -1 跳过写——',
              'KV cache 不被 pad token 污染（唯一写副作用',
              '消费者，哨兵必须让写路径跳过）。']),
    dict(name='positions', role='token 位置（RoPE 输入）',
         anchor='gpu_model_runner.py:L3663-L3664', buf=8,
         before=[('7', 'act'), ('100', 'act'), ('3', 'act'), ('42', 'act'), ('55', 'act'),
                 ('5', 'stale'), ('6', 'stale'), ('7', 'stale')],
         after=[('7', 'act'), ('100', 'act'), ('3', 'act'), ('42', 'act'), ('55', 'act'),
                ('0', 'pad'), ('0', 'pad'), ('0', 'pad')],
         span='写尾段 [5:8)',
         why=['尾清零：RoPE 对 pad 行算垃圾；输出只收集',
              '活跃请求的末 token（ch17 站 10 已立），',
              '垃圾不被读出。']),
]

RY0, RH_ = 168, 138
for ri, r in enumerate(ROWS):
    ry = RY0 + ri * RH_
    if ri > 0:
        lc.seg(MX, ry - 4, BXR, ry - 4, '#e2e8f0', 1.0)
    # 左标签
    lc.rect(LX, ry + 12, 30, 30, lc.C_BADGE_F, lc.C_BEAT_S, rx=6, sw=1.1)
    lc.text(LX + 15, ry + 32, str(ri + 1), 11, lc.C_BEAT_T, 'middle', True, maxw=26,
            tag='rn%d' % ri)
    lc.text(LX + 40, ry + 26, r['name'], 10, lc.C_TXT, 'start', True, maxw=LW - 40,
            tag='nm%d' % ri)
    lc.text(LX + 40, ry + 43, r['role'], 7.8, '#334155', 'start', maxw=LW - 36,
            tag='ro%d' % ri)
    lc.text(LX + 40, ry + 58, r['anchor'], 7.2, lc.C_FAINT, 'start', maxw=LW - 36,
            tag='ra%d' % ri)
    # before / after 数组
    n = r['buf']
    gap = 5
    cw_b = min(48, (ARX0 - 20 - BFX - (n - 1) * gap) / n)
    cw_a = min(48, (WHYX - 24 - AFX - (n - 1) * gap) / n)
    ay = ry + 46
    lc.text(BFX, ry + 32, 'before（上一拍陈旧尾）', 7.4, lc.C_MUTE, 'start', True, maxw=200,
            tag='bl%d' % ri)
    lc.text(AFX, ry + 32, 'after（本拍写完）', 7.4, lc.C_MUTE, 'start', True, maxw=160,
            tag='al%d' % ri)
    for i, (s, k) in enumerate(r['before']):
        cell(BFX + i * (cw_b + gap), ay, cw_b, s, k)
    for i, (s, k) in enumerate(r['after']):
        cell(AFX + i * (cw_a + gap), ay, cw_a, s, k)
    # 中箭头
    midy = ay + CELL_H / 2
    b_end = BFX + n * cw_b + (n - 1) * gap
    lc.parrow([(b_end + 8, midy), (ARX1, midy)], lc.C_BEAT_S, 2.0)
    lc.text((b_end + 8 + ARX1) / 2, midy - 10, r['span'], 7.6, lc.C_BEAT_T, 'middle', True,
            maxw=ARX1 - b_end - 16, tag='sp%d' % ri)
    # 右侧安全说明
    for i, ln in enumerate(r['why']):
        lc.text(WHYX, ry + 40 + i * 17, ln, 8.0, '#334155', 'start', maxw=BXR - WHYX,
                tag='wh%d%d' % (ri, i))

# ---------------- 底部注记 ----------------
NY = RY0 + 4 * RH_ + 8
lc.rect(MX, NY, BXR - MX, 66, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
NOTES = ['四段全写在 runner 的固定缓冲尾段（地址不变——ch18 固定地址的同一批缓冲）：每拍共 4 次定长尾段写、O(1) 次 CPU 写、零分配',
         '（本例每件 3 项、最坏 = 段长−1 = 7）。默认刻度对照：bs=9 pad 到 16、白算 7 行——decode 一行一个 token 的 forward 便宜可接受。']
for i, ln in enumerate(NOTES):
    lc.text(MX + 16, NY + 22 + i * 19, ln, 8.3, '#334155', 'start', maxw=BXR - MX - 32,
            tag='nt' + str(i))

# 图例
lx = MX
LEG = [('act', '活跃前缀（逐字节不动）'), ('stale', '陈旧尾（覆盖前）'), ('pad', 'pad 哨兵（本拍写入）')]
for kind, lab in LEG:
    if kind == 'act':
        lc.rect(lx, NY + 44, 18, 11, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.1)
    elif kind == 'stale':
        lc.rect(lx, NY + 44, 18, 11, '#f1f5f9', '#cbd5e1', rx=3, sw=0.9)
    else:
        lc.rect(lx, NY + 44, 18, 11, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
    lc.text(lx + 24, NY + 54, lab, 8, lc.C_TXT, 'start', maxw=200, tag='lg' + kind)
    lx += 24 + lc.tw(lab, 8) + 26

# ---------------- 页脚 ----------------
lc.text(MX, NY + 96, '逐字锚 vllm/v1/worker/gpu_model_runner.py:L2073-L2078 · L2338-L2341 · L4128-L4130 · L3663-L3664（四段 pinned span）· L3932-L4044（裁决）· L4265-L4278（call site）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, NY + 114, '裁决与四件套 before/after 数组取自精简版 companion host 实跑（裁决 _determine_batch_execution_and_padding 与 slot_mapping _get_slot_mappings 走真实方法全调用）',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-padding-four.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
