#!/usr/bin/env python3
"""ch38 机制图 m3 · tail-block-collection（figure_spec ch38-fig-tail-block-collection，模板 tiling）

放大自 L2 站 4（③ 满块采集·每步增量）· L0：GPU 池与外部池之间「换出去」第一步。

claim：满块采集按 chunk 切片、每 chunk 以尾块做准入判据：GPU 块 1 5 6 7 2 4 9 3 8 在
blocks_per_chunk=3 下切 3 箱，尾块 6 4 8 非 0 → 3 箱全采，搬运时整箱逐块走
（src=全部 9 块）；下步 decode +3 块只采新箱。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · GPU 块序 1 5 6 7 2 4 9 3 8 → 每 chunk 取最后一个：6 4 8（源码注释原例）
  · 搬运覆盖整 chunk：src=全部 9 块、group_sizes=[9]、CPU 槽 [0,1,2]、游标→3
  · 增量步：新块 [11,12,13]、起块 index 9、游标 3→4
  · block_id=0：[1,0,3,4] → 跳过 chunk1、src=[1,3,4]
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 60
BXR = W - MX

# ---------------- 标题区 ----------------
lc.text(MX, 36, '满块采集按「箱」不按「件」：准入看尾块、搬运走整箱、游标保增量', 16.5,
        lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 60, 'blocks_per_chunk 个 GPU 块并成一个 offload 块；每箱只看「箱底那一件」在不在——尾块非 0 就整箱搬；游标记到哪箱，下一步只搬新箱',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L2 站 4（③ 满块采集·每步增量）· L0：GPU 池→外部池'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')


def chunk_box(x, y, box_w, box_h, ids, bpc, stored, label):
    """一个 chunk 箱：bpc 个块并排；标签在箱上方；stored=True 画成已寄存（浅色+对勾）。"""
    pad = 8
    cell = (box_w - 2 * pad - (bpc - 1) * 6) / bpc
    if stored:
        lc.rect(x, y, box_w, box_h, '#f8fafc', lc.C_FAINT, rx=6, sw=1.1, dash=True)
    else:
        lc.rect(x, y, box_w, box_h, '#ffffff', lc.C_GPU_S, rx=6, sw=1.5)
    for i, num in enumerate(ids):
        cx = x + pad + i * (cell + 6)
        if stored:
            lc.rect(cx, y + pad, cell, box_h - 2 * pad, '#f1f5f9', lc.C_FAINT, rx=4, sw=1.0)
            lc.text(cx + cell / 2, y + pad + (box_h - 2 * pad) / 2 + 4, str(num), 10,
                    lc.C_FAINT, 'middle', maxw=cell - 4, tag='sblk%d' % num)
        else:
            tail = (i == bpc - 1)
            fill = '#bbf7d0' if tail else lc.C_GPU_F
            stroke = '#15803d' if tail else lc.C_GPU_S
            lc.rect(cx, y + pad, cell, box_h - 2 * pad, fill, stroke, rx=4,
                    sw=1.6 if tail else 1.1)
            lc.text(cx + cell / 2, y + pad + (box_h - 2 * pad) / 2 + 4, str(num),
                    11 if tail else 10, lc.C_TXT if tail else '#334155', 'middle',
                    bold=tail, maxw=cell - 4, tag='blk%d' % num)
    lc.text(x + box_w / 2, y - 8, label, 8.5, lc.C_MUTE, 'middle', maxw=box_w + 40,
            tag='clab:' + label)


# ---------------- 面板 A：步 1 · 注释原例 ----------------
PA_X, PA_W = MX, 660
PA_Y, PA_H = 106, 380
lc.rect(PA_X, PA_Y, PA_W, PA_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(PA_X + 16, PA_Y + 22, '步 1 · 注释原例：36 token = 9 GPU 块，blocks_per_chunk=3 → 3 箱', 10.5,
        lc.C_GPU_S, 'start', True, maxw=PA_W - 32, tag='pa:t')
step1 = [[1, 5, 6], [7, 2, 4], [9, 3, 8]]
BOX_W, BOX_H, BOX_GAP = 186, 60, 22
BX0 = PA_X + (PA_W - (3 * BOX_W + 2 * BOX_GAP)) / 2
BY0 = PA_Y + 58
for i, ids in enumerate(step1):
    chunk_box(BX0 + i * (BOX_W + BOX_GAP), BY0, BOX_W, BOX_H, ids, 3, False,
              'chunk %d · 尾块 %d' % (i, ids[-1]))
# CPU 槽行（箱底 → 槽顶，搬运箭头走廊无文字）
SLOT_Y = BY0 + BOX_H + 40
for i in range(3):
    sx = PA_X + 20 + i * 96
    lc.rect(sx, SLOT_Y, 84, 30, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.3)
    lc.text(sx + 42, SLOT_Y + 19, 'CPU 槽 %d' % i, 9, lc.C_KV_S, 'middle', True, maxw=80,
            tag='slot%d' % i)
    cbx = BX0 + i * (BOX_W + BOX_GAP) + BOX_W / 2
    lc.seg(cbx, BY0 + BOX_H, sx + 42, SLOT_Y, lc.C_KV_S, 1.4, 'std', dash=True)
lc.text(PA_X + 20 + 3 * 96 + 8, SLOT_Y + 19, '落 3 个槽', 9, lc.C_MUTE, 'start',
        maxw=PA_W - 20 - 3 * 96 - 24, tag='pa:slots')
ADM_Y = SLOT_Y + 30 + 22
lc.text(PA_X + 16, ADM_Y, '准入判据：每 chunk 取最后一个 GPU 块 → 尾块切片 [6, 4, 8] 全非 0 → 3 箱全准入', 9.5,
        lc.C_TXT, 'start', True, maxw=PA_W - 32, tag='pa:adm')
lc.text(PA_X + 16, ADM_Y + 20, '· 搬运整箱：src = 全部 9 块 · group_sizes=[9] · 9 × 512B = 4608B', 9,
        '#334155', 'start', maxw=PA_W - 32, tag='pa:tr1')
lc.text(PA_X + 16, ADM_Y + 38, '· 已寄存的不重寄（右）· 占位 0 的跳过（下条）', 9, '#334155',
        'start', maxw=PA_W - 32, tag='pa:tr2')
# 游标尺
CUR_Y = PA_Y + PA_H - 22
R0, R1 = PA_X + 70, PA_X + PA_W - 210
lc.seg(R0, CUR_Y, R1, CUR_Y, lc.C_MUTE, 1.4)
for i in range(4):
    lx = R0 + i * (R1 - R0) / 3
    lc.seg(lx, CUR_Y - 4, lx, CUR_Y + 4, lc.C_MUTE, 1.1)
    lc.text(lx, CUR_Y + 15, str(i), 8.5, lc.C_MUTE, 'middle', tag='cur%d' % i)
lc.parrow([(R1 - 12, CUR_Y), (R1, CUR_Y)], lc.C_GPU_S, 1.4, 'std')
lc.text(R1 + 10, CUR_Y + 3, '游标 next_stored_chunk_idx：0 → 3', 8.5, lc.C_GPU_S, 'start',
        maxw=PA_W - (R1 - PA_X) - 30, tag='pa:cur')

# ---------------- 面板 B：步 2 · 增量游标 ----------------
PB_X = PA_X + PA_W + 24
PB_W = BXR - PB_X
PB_Y, PB_H = PA_Y, PA_H
lc.rect(PB_X, PB_Y, PB_W, PB_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.5)
lc.text(PB_X + 16, PB_Y + 22, '步 2 · 增量：decode +12 token → 只新增一箱', 10.5,
        lc.C_ENG_S, 'start', True, maxw=PB_W - 32, tag='pb:t')
step2 = [[1, 5, 6], [7, 2, 4], [9, 3, 8], [11, 12, 13]]
B2_W, B2_H, B2_GAP = 158, 60, 14
BX1 = PB_X + (PB_W - (4 * B2_W + 3 * B2_GAP)) / 2
BY1 = PB_Y + 58
labels2 = ['✓ chunk 0（已寄存）', '✓ chunk 1（已寄存）', '✓ chunk 2（已寄存）', '新 chunk 3 · 尾块 13']
for i, ids in enumerate(step2):
    chunk_box(BX1 + i * (B2_W + B2_GAP), BY1, B2_W, B2_H, ids, 3, i < 3, labels2[i])
lc.text(BX1 + 3 * (B2_W + B2_GAP) + B2_W / 2, BY1 + B2_H + 18,
        '↑ 只有这一箱进采集区间', 8.5, lc.C_ENG_S, 'middle', True, maxw=B2_W + 40, tag='pb:only')
lc.text(PB_X + 16, BY1 + B2_H + 44, '采集区间恒为 [next_stored_chunk_idx, storable_chunks)——左端只增不减：', 9,
        lc.C_TXT, 'start', True, maxw=PB_W - 32, tag='pb:inv')
pb_lines = [
    '· 只采新箱：src=[11,12,13] · group_sizes=[3] · 起块 index 9',
    '· 3 × 512B = 1536B（旧箱零重发）',
    '· 推进取 max(游标, num_chunks)——注释原话 the index must not move backwards',
    '· 拒收不丢游标：同一 chunk 下一步仍在区间内（best-effort 容忍重试）',
]
for j, ln in enumerate(pb_lines):
    lc.text(PB_X + 16, BY1 + B2_H + 64 + j * 16, ln, 8.8, '#334155', 'start',
            maxw=PB_W - 30, tag='pb:l%d' % j)
CUR2_Y = PB_Y + PB_H - 22
S0, S1 = PB_X + 70, PB_X + PB_W - 130
lc.seg(S0, CUR2_Y, S1, CUR2_Y, lc.C_MUTE, 1.4)
for i in range(5):
    lx = S0 + i * (S1 - S0) / 4
    lc.seg(lx, CUR2_Y - 4, lx, CUR2_Y + 4, lc.C_MUTE, 1.1)
    lc.text(lx, CUR2_Y + 15, str(i), 8.5, lc.C_MUTE, 'middle', tag='cur2_%d' % i)
lc.parrow([(S1 - 12, CUR2_Y), (S1, CUR2_Y)], lc.C_ENG_S, 1.4, 'std')
lc.text(S1 + 10, CUR2_Y + 3, '游标 3 → 4', 8.5, lc.C_ENG_S, 'start',
        maxw=PB_W - (S1 - PB_X) - 30, tag='pb:cur')

# ---------------- 底条：block_id=0 跳过 ----------------
ZS_Y = PA_Y + PA_H + 22
lc.rect(MX, ZS_Y, BXR - MX, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 16, ZS_Y + 20, '占位跳过：block_id=0（null / 陈旧 stale）不入池——「采了也命中不了」的同类减法', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='zs:t')
zx = MX + 16
for num in [1, 0, 3, 4]:
    if num == 0:
        lc.rect(zx, ZS_Y + 34, 34, 34, '#f1f5f9', lc.C_ABORT, rx=4, sw=1.4, dash=True)
    else:
        lc.rect(zx, ZS_Y + 34, 34, 34, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.1)
    lc.text(zx + 17, ZS_Y + 55, str(num), 10, lc.C_ABORT if num == 0 else '#334155',
            'middle', maxw=30, tag='z%d' % num)
    zx += 42
lc.text(zx + 12, ZS_Y + 48, 'GPU 块 [1, 0, 3, 4]（blocks_per_chunk=1）→ 尾块 0 的 chunk 跳过', 9,
        '#334155', 'start', maxw=520, tag='zs:l1')
lc.text(zx + 12, ZS_Y + 66, '搬运 src=[1, 3, 4]——跳过 1 个 chunk、零字节搬运', 9,
        '#334155', 'start', maxw=520, tag='zs:l2')
lc.text(BXR - 16, ZS_Y + 48, '同类减法：SWA 对齐段不可达 chunk · eagle 尾块易变剔除', 8.5,
        lc.C_MUTE, 'end', maxw=420, tag='zs:l3')
lc.text(BXR - 16, ZS_Y + 66, '（正文散文交代，不在本图展开）', 8.5, lc.C_FAINT, 'end',
        maxw=420, tag='zs:l4')

# ---------------- 结论 + 页脚 ----------------
CONC_Y = ZS_Y + 96 + 24
lc.text(MX, CONC_Y,
        '图注结论：准入判据看尾块（0=占位跳过）、搬运走整箱、游标保证增量不重寄——每个 chunk 至多被采集进池一次。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L1023-L1099（『每 chunk 取最后一个 GPU 块』注释原例 + 游标推进）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 块内数字 = GPU 块 id（实测序列）；512B = 主例每 worker 每块字节数',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-tail-block-collection.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
