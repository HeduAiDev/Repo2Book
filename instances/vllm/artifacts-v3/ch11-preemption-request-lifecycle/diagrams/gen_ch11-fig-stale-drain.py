#!/usr/bin/env python3
"""ch11 机制图 4 · stale 在途输出协议：照常送达 + 锁步冲销（figure_spec ch11-fig-stale-drain，模板 swimlane）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框——即本章 L2 章图
center ④ 六件事拍片的 stale 标注 + south『stale · async 交叉面』注框的机制展开；
非新架构画法，架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：被抢请求的在途输出走一条『照常送达+锁步冲销』的平行账：每来一批冲销一份（stale
2→1→0），排空前恢复被推迟，排空后下一拍 resumed——drop-mode 则整段丢弃。

数字全部取自 figure_spec.numbers（stale 2→1→0 / 送达 [42][43][44] / P3 落 skipped_waiting /
P6 emitted=0 / P7 同步自中和 stale 恒 0），源出配套精简版 host 实跑 trace（async 深度 2 人工模拟）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX, BXR = 60, 1440

EXTRA_DEFS = ('<defs>'
              '<marker id="kvm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker>'
              '<marker id="okg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '在途输出的平行账：照常送达 + 锁步冲销——排空前恢复推迟，排空后下一拍 resumed',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '抢占不清在途输出，只给它换一本账（request.py:L150-L162）：stale=in_flight（assign），每个在途步回账冲销其份额——丢掉会扰动 spec acceptance，而计数器已清零不能再扣',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 拍片 ④ + south stale 注框 · L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 列几何（P1..P5 主线 + P6/P7 对照） ----------------
COL_W, COL_GAP = 168, 14
N_COLS = 7
COLS_X0 = 104
centers = [COLS_X0 + i * (COL_W + COL_GAP) + COL_W / 2 for i in range(N_COLS)]
L1_Y0, L1_H = 130, 204                 # 上泳道：调度器拍
WB_Y0, WB_H = 356, 118                 # stale 水位带
L2_Y0, L2_H = 496, 150                 # 下泳道：输出管线

def lane(y0, h, title, sub, dash=False):
    lc.rect(MX + 40, y0, BXR - MX - 40, h, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=dash)
    lc.text(MX + 54, y0 + 22, title, 11, lc.C_TXT, 'start', True, maxw=500, tag='lane:' + title[:6])
    lc.text(BXR - 12, y0 + 22, sub, 8.8, lc.C_MUTE, 'end', maxw=460, tag='lanes:' + title[:6])

lane(L1_Y0, L1_H, '上泳道 · 调度器拍（schedule / update_from_output）', 'P1..P5 = 主线故事')
lane(L2_Y0, L2_H, '下泳道 · 输出管线（在途步到达 → 送达）', '每个在途步回来：token 照常送达，账上锁步冲销')

# P1..P5 与 P6/P7 之间的虚线分隔
lc.seg((centers[4] + centers[5]) / 2, L1_Y0 + 6, (centers[4] + centers[5]) / 2, L1_Y0 + L1_H - 6,
       lc.C_FAINT, 1.2, dash=True)
lc.seg((centers[4] + centers[5]) / 2, L2_Y0 + 6, (centers[4] + centers[5]) / 2, L2_Y0 + L2_H - 6,
       lc.C_FAINT, 1.2, dash=True)
lc.text((centers[4] + centers[5]) / 2 + 6, L1_Y0 + 38, '对照', 8.5, lc.C_FAINT, 'start', tag='div')

# ---------------- 列数据 ----------------
PH = [
    ('P1', ['调度后、输出回来前', '被抢（async 模拟深度 2）'], 'PREEMPTED · 回 waiting 队头', 'stale ← in_flight（assign）', lc.C_KV_S),
    ('P2', ['第 1 个在途输出到达', '42 仍送达（不丢弃）'], 'PREEMPTED', 'stale 2→1 · in_flight 2→1', lc.C_KV_S),
    ('P3', ['下一拍 schedule：', 'stale=1>0 且非 drop'], '推迟恢复 → 落 skipped_waiting', '本拍不调度它（{}）', lc.C_ENG_S),
    ('P4', ['第 2 个在途输出到达', '（同一 out 二次回账模拟）'], 'PREEMPTED · 已排空', 'stale 1→0 · in_flight 1→0', lc.C_KV_S),
    ('P5', ['stale=0，下一拍恢复', '重命中 16 + 补 3'], 'RUNNING · resumed', '首输出 44 照常送达', lc.C_GPU_S),
    ('P6', ['drop-mode 抢占', '（同拍抢占+恢复形态）'], '整段丢弃：42 不外送', '也不入账（emitted=0）', lc.C_ABORT),
    ('P7', ['同步版自中和（对照）', '抢占发生在上拍输出'], '已回账之后：in_flight=0', '→ stale 恒 0', lc.C_MUTE),
]
for i, (pid, acts, status, acct, col) in enumerate(PH):
    cx = centers[i]
    x0 = cx - COL_W / 2
    # 拍徽标
    bw = 34
    lc.rect(x0 + 2, L1_Y0 + 34, bw, 20, lc.C_BADGE_F, col, rx=9, sw=1.1)
    lc.text(x0 + 2 + bw / 2, L1_Y0 + 47.5, pid, 9.5, col, 'middle', True, tag='bdg' + pid)
    # 动作两行
    for j, ln in enumerate(acts):
        lc.text(cx, L1_Y0 + 72 + j * 16, ln, 8.8, '#334155', 'middle', maxw=COL_W, tag='ac' + pid + str(j))
    # 状态
    lc.text(cx, L1_Y0 + 116, status, 8.8, col, 'middle', True, maxw=COL_W, tag='st' + pid)
    # 记账
    lc.text(cx, L1_Y0 + 136, acct, 8.4, lc.C_MUTE, 'middle', maxw=COL_W, tag='ac2' + pid)

# 列间竖线（时间推进）
for i in range(N_COLS - 1):
    if i == 4:
        continue
    lc.seg(centers[i] + COL_W / 2 + 2, L1_Y0 + 44, centers[i] + COL_W / 2 + 2, L1_Y0 + 60, lc.C_MUTE, 1.4, 'std')

# ---------------- stale 水位带 ----------------
lc.text(MX + 54, WB_Y0 + 20, 'stale 水位（在途冲销账）', 10.5, lc.C_TXT, 'start', True, maxw=260, tag='wb:t')
lc.text(MX + 54, WB_Y0 + 38, 'P1 置 2 = in_flight（assign 不累加）', 8.6, lc.C_MUTE, 'start', maxw=260, tag='wb:s')
LVL_Y = {2: WB_Y0 + 58, 1: WB_Y0 + 86, 0: WB_Y0 + 114}       # 水位 → y
levels = [2, 1, 1, 0, 0, 0, 0]                                 # P1..P7 各拍后水位
AX_X = 96
# 轴刻度（0/1/2，最左侧）
for v, yy in LVL_Y.items():
    lc.text(AX_X - 8, yy + 3.5, str(v), 9, lc.C_MUTE, 'end', tag='ax' + str(v))
lc.seg(AX_X, LVL_Y[2], AX_X, LVL_Y[0] + 6, '#cbd5e1', 1.2)
# 每拍的平台段（P6/P7 虚线=对照）
for i in range(N_COLS):
    x0 = centers[i] - COL_W / 2 + (10 if i == 0 else 8)
    x1 = centers[i] + COL_W / 2 - 8
    dashed = i >= 5
    lc.seg(x0, LVL_Y[levels[i]], x1, LVL_Y[levels[i]],
           lc.C_KV_S if not dashed else lc.C_FAINT, 3.0, dash=dashed)
# 相邻拍之间的落差连接与 −1 标注
for i in range(N_COLS - 1):
    if levels[i] != levels[i + 1]:
        cxn = centers[i] + COL_W / 2 + 4
        lc.seg(cxn, LVL_Y[levels[i]], cxn, LVL_Y[levels[i + 1]], lc.C_KV_S, 3.0)
        lc.text(cxn + 6, (LVL_Y[levels[i]] + LVL_Y[levels[i + 1]]) / 2 + 3, '−1', 9.5, lc.C_KV_S,
                'start', True, tag='drop' + str(i))
# P1 平台标注（右半段上方）
lc.text(centers[0] + 58, LVL_Y[2] - 8, '置 2', 9, lc.C_KV_S, 'start', True, tag='set2')
# P3 停滞标注
lc.text(centers[2], LVL_Y[1] - 10, 'P3 停在 1：推迟恢复', 8.6, lc.C_ENG_S, 'middle', True, maxw=COL_W, tag='p3hold')
# P5 恢复标注（level 0 线下方，锚 end 避开 P4/P5 的到达箭头竖线）
lc.text(centers[4] - 18, LVL_Y[0] + 20, 'P5：stale=0 → 放行恢复', 8.6, lc.C_GPU_S, 'end', True, maxw=COL_W + 20,
        tag='p5go')

# ---------------- 下泳道：token 到达与送达 ----------------
TOK = [
    ('P1', None, '—（刚被抢，尚无在途步回来）'),
    ('P2', ('42', True), ''),
    ('P3', None, '—（本拍无到达）'),
    ('P4', ('43', True), ''),
    ('P5', ('44', True), '（resumed 后首输出）'),
    ('P6', ('42', False), ''),
    ('P7', None, '—（同步：无在途步）'),
]
for i, (pid, tok, extra) in enumerate(TOK):
    cx = centers[i]
    if tok is None:
        lc.text(cx, L2_Y0 + 92, extra, 8.6, lc.C_MUTE, 'middle', maxw=COL_W, tag='tkn' + pid)
        continue
    val, delivered = tok
    ty = L2_Y0 + 52
    lc.rect(cx - 24, ty, 48, 26, lc.C_GPU_F if delivered else '#ffffff', lc.C_GPU_S if delivered else lc.C_ABORT,
            rx=5, sw=1.4, dash=not delivered)
    lc.text(cx, ty + 17, '[' + val + ']', 10, lc.C_GPU_S if delivered else lc.C_ABORT, 'middle', True, tag='tk' + pid + val)
    # 到达箭头：token → 上方泳道底边（触发锁步冲销）
    lc.seg(cx, ty - 4, cx, L1_Y0 + L1_H + 2, lc.C_KV_S, 1.8, 'kvm')
    if delivered:
        lc.text(cx, ty + 42, '外送 [' + val + ']', 8.8, lc.C_GPU_S, 'middle', True, maxw=COL_W, tag='dl' + pid)
        if extra:
            lc.text(cx, ty + 58, extra, 8.2, lc.C_MUTE, 'middle', maxw=COL_W, tag='dlx' + pid)
    else:
        lc.text(cx, ty + 42, '丢弃：不外送不入账', 8.8, lc.C_ABORT, 'middle', True, maxw=COL_W, tag='dl' + pid)
        lc.text(cx, ty + 58, 'emitted_count=0', 8.2, lc.C_ABORT, 'middle', tag='dlx' + pid)

# P6/P7 到达箭头对齐：P6 的箭头也画（触发 drop 分支）——已由上面循环处理

# ---------------- 底部注记 ----------------
BN_Y = L2_Y0 + L2_H + 22
lc.rect(MX, BN_Y, 880, 78, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.4)
lc.text(MX + 16, BN_Y + 20, '为什么仍要送达、为什么不得改计数器', 10.5, lc.C_BEAT_T, 'start', True, maxw=840, tag='bn:t')
lc.text(MX + 16, BN_Y + 40, '· 丢掉会扰动 spec-decode acceptance 统计（源码注释原话）；num_computed_tokens 已清零、', 9,
        '#334155', 'start', maxw=848, tag='bn:l1')
lc.text(MX + 16, BN_Y + 58, '  num_output_placeholders 已置 0，再扣就是 underflow——协议只在 async（服务默认）与 PP 下咬合', 9,
        '#334155', 'start', maxw=848, tag='bn:l2')

NT_X = MX + 904
lc.rect(NT_X, BN_Y, BXR - NT_X, 78, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(NT_X + 14, BN_Y + 20, '窗口 ≤ pipeline 深度', 10.5, lc.C_TXT, 'start', True, maxw=400, tag='nt:t')
lc.text(NT_X + 14, BN_Y + 40, '全部在途步的调度数之和恰为 in_flight → 排空有限，', 8.8, '#334155', 'start',
        maxw=BXR - NT_X - 26, tag='nt:l1')
lc.text(NT_X + 14, BN_Y + 58, '本例深度 2：P2/P4 各冲销 1，第三拍（P5）恢复', 8.8, lc.C_MUTE, 'start',
        maxw=BXR - NT_X - 26, tag='nt:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BN_Y + 104
lx = MX
for kind, name in [('kv', 'stale 水位（青=主线）'), ('drop', '水位下降 −1（锁步冲销）'), ('tk', '送达的 token（绿）'),
                   ('bad', 'drop-mode 丢弃（红虚线）'), ('faint', '对照段（虚线）')]:
    if kind == 'kv':
        lc.seg(lx, LEG_Y - 3, lx + 18, LEG_Y - 3, lc.C_KV_S, 3.0)
    elif kind == 'drop':
        lc.text(lx, LEG_Y + 1, '−1', 9.5, lc.C_KV_S, 'start', True)
    elif kind == 'tk':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.2)
    elif kind == 'bad':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', lc.C_ABORT, rx=3, sw=1.2, dash=True)
    else:
        lc.seg(lx, LEG_Y - 3, lx + 18, LEG_Y - 3, lc.C_FAINT, 3.0, dash=True)
    off = 24 if kind == 'drop' else 26
    lc.text(lx + off, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=220, tag='leg' + kind)
    lx += off + lc.tw(name, 8.8) + 18

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/request.py:L150-L162（stale/in_flight/drop 字段）/ scheduler.py:L1297-L1308（assign）/ L1737-L1743（锁步 drain）'
        '/ L713-L722（推迟恢复）/ L1757-L1759（drop-mode）· P1 的 in_flight=2 为人工置位模拟 async 深度 2、P4 同一 scheduler_output 二次回账模拟第 2 在途步 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-stale-drain.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
