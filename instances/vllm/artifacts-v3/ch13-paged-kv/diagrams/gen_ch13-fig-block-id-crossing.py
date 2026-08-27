#!/usr/bin/env python3
"""ch13 机制图 5 · block_id 跨进程契约（figure_spec ch13-fig-block-id-crossing，模板 swimlane）

放大自 L0 上 EngineCore（橙）与 worker/GPU（绿）两列之间过线的那条 block_id
桥——即本章 L2 章图 frame『EngineCore 进程（调度器侧 KV 账本）+ worker 进程
（GPU 侧页表与槽位）· block_id = 唯一共享键』中 center ④ 过线打包 → ⑤ worker
镜像 两拍片的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：block_id 是调度器与 worker 之间唯一共享键：新请求发全量块表、在跑请求只发
增量 new_block_ids（空则 None）、被抢占恢复者整表替换——三拍实录 [全量 [1,2,3] /
增量 [4] / 同帧 r2 全量 [5]+r1 增量 None]，worker 侧三种镜像动作与之一一对应。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑：拍 1/1.5/2/3 过线载荷、
worker 镜像 [1,2,3]→[1,2,3,4]、num_blocks_per_row=4、resumed [1]→[2,3]、
旁路通道 [1,2,3,4,5,6]）。
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
lc.text(MX, 34, '一个 block_id 过两次江：首帧寄整箱档案，之后只发电报，没新货连电报都省',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '调度器进程独占块账本（谁用哪块 / ref_cnt / 自由队列），worker 进程独占 GPU 张量——两边唯一都认得的键是 block_id（块表经 IPC 序列化天然各持一份）',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L2 章图中排 ④→⑤ 拍片与 frame 进程边界 · L0：EngineCore×GPU 之间的 block_id 桥'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道框架 ----------------
SCH_Y, SCH_H = 92, 196
RIV_Y, RIV_H = 312, 292
WRK_Y, WRK_H = 628, 232
for y, h, stroke, fill, name, note in [
        (SCH_Y, SCH_H, lc.C_ENG_S, lc.C_ENG_F, '调度器进程（EngineCore）', '独占元数据：req_to_blocks · ref_cnt · 自由队列'),
        (RIV_Y, RIV_H, lc.C_ZMQ_S, '#faf5ff', 'IPC 江面 · SchedulerOutput 过线', 'ZMQ + msgpack 序列化——空增量 None 不占带宽'),
        (WRK_Y, WRK_H, lc.C_GPU_S, lc.C_GPU_F, 'worker 进程', '独占 GPU 张量：CachedRequestState + BlockTable（CPU 页表行）')]:
    lc.rect(MX, y, BXR - MX, h, fill, stroke, rx=10, sw=1.8)
    lc.text(MX + 16, y + 22, name, 12, stroke, 'start', True, maxw=430, tag='lane:' + name[:6])
    lc.text(BXR - 16, y + 22, note, 8.5, lc.C_MUTE, 'end', maxw=560, tag='ln:' + name[:6])

# ---------------- 调度器泳道内容 ----------------
LED_X, LED_W = 90, 470
lc.rect(LED_X, SCH_Y + 40, LED_W, 132, '#ffffff', lc.C_ENG_S, rx=7, sw=1.3)
lc.text(LED_X + 14, SCH_Y + 60, 'KVCacheManager 台账 · req_to_blocks', 10, lc.C_ENG_S, 'start', True,
        maxw=LED_W - 28, tag='led:t')
ROWS = [('拍 1 时 r1（33 token）', '[1, 2, 3]'), ('拍 2 后 r1（长到 49 token）', '[1, 2, 3, 4]'),
        ('拍 3 时 r2（16 token）', '[5]')]
for i, (a, b) in enumerate(ROWS):
    yy = SCH_Y + 84 + i * 28
    lc.text(LED_X + 14, yy, a, 8.5, '#334155', 'start', maxw=200, tag='lr%d' % i)
    lc.seg(LED_X + 220, yy - 3, LED_X + 244, yy - 3, '#94a3b8', 1.2, 'std')
    lc.rect(LED_X + 252, yy - 12, 118, 20, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.1)
    lc.text(LED_X + 311, yy + 2, b, 9, lc.C_KV_S, 'middle', True, maxw=110, tag='lv%d' % i)
lc.text(LED_X + 14, SCH_Y + 162, 'get_block_ids(allow_none=True)：全组空 → None（kv_cache_manager.py:L89-L91）',
        7.8, lc.C_MUTE, 'start', maxw=LED_W - 28, tag='led:n')

ZRO_X, ZRO_W = 1120, BXR - 1120
lc.rect(ZRO_X, SCH_Y + 40, ZRO_W, 132, '#ffffff', lc.C_ENG_S, rx=7, sw=1.2, dash=True)
lc.text(ZRO_X + 14, SCH_Y + 60, '清零账（旁路通道）', 10, lc.C_ENG_S, 'start', True, maxw=ZRO_W - 28,
        tag='z:t')
lc.text(ZRO_X + 14, SCH_Y + 82, 'take_new_block_ids 每步排干', 8.5, '#334155', 'start',
        maxw=ZRO_W - 28, tag='z:1')
lc.text(ZRO_X + 14, SCH_Y + 100, '→ new_block_ids_to_zero 随 SchedulerOutput 过线', 8.5, '#334155',
        'start', maxw=ZRO_W - 28, tag='z:2')
lc.text(ZRO_X + 14, SCH_Y + 122, 'uniform 单组 → None（清零通道关）', 8.5, lc.C_MUTE, 'start',
        maxw=ZRO_W - 28, tag='z:3')
lc.text(ZRO_X + 14, SCH_Y + 140, '混合精度两组 → [1,2,3,4,5,6]', 8.5, lc.C_MUTE, 'start',
        maxw=ZRO_W - 28, tag='z:4')
lc.text(ZRO_X + 14, SCH_Y + 160, '（注释：does not grow unbounded）', 7.8, lc.C_FAINT, 'start',
        maxw=ZRO_W - 28, tag='z:5')

# ---------------- 江面：四拍包裹 + 右端恢复者小图 ----------------
def parcel(x, y, w, h, title, lines, kind):
    stroke = {'box': lc.C_ZMQ_S, 'tele': lc.C_ZMQ_S, 'stamp': '#94a3b8', 'resume': lc.C_ABORT}[kind]
    fill = {'box': '#ffffff', 'tele': '#ffffff', 'stamp': '#f8fafc', 'resume': '#fef2f2'}[kind]
    dash = kind in ('stamp', 'resume')
    lc.rect(x, y, w, h, fill, stroke, rx=6, sw=1.4, dash=dash)
    lc.text(x + w / 2, y + 18, title, 9.5, stroke if kind != 'stamp' else '#64748b', 'middle', True,
            maxw=w - 12, tag='pt:' + title[:6])
    for i, ln in enumerate(lines):
        lc.text(x + w / 2, y + 38 + i * 16, ln, 8.5, '#334155', 'middle', maxw=w - 12,
                tag='pl:%s%d' % (title[:4], i))

P1_X, P1_W = 100, 300
parcel(P1_X, RIV_Y + 60, P1_W, 118, '拍 1 · 厚货箱（新请求 r1）',
       ['全量块表 ([1,2,3],)', '＋ 33 个 prompt token ＋ computed', '——首帧整箱装备'], 'box')
P2_X, P2_W = 436, 190
parcel(P2_X, RIV_Y + 60, P2_W, 118, '拍 1.5 · 免电报戳',
       ['本步无新块', 'new_block_ids = None', '——空增量不占带宽'], 'stamp')
P3_X, P3_W = 656, 210
parcel(P3_X, RIV_Y + 60, P3_W, 118, '拍 2 · 窄电报',
       ['增量 ([4],)', '——只发本步新块', '（r1 长到 49 token）'], 'tele')
P4_X, P4_W = 896, 200
parcel(P4_X, RIV_Y + 60, P4_W, 118, '拍 3 · 双包裹',
       ['r2 全量 ([5],)', 'r1 增量 None（免电报）', '——两种包裹同帧过江'], 'tele')
RS_X, RS_W = 1120, BXR - 1120
parcel(RS_X, RIV_Y + 60, RS_W, 118, '拍 4 · 恢复者整箱重寄',
       ['抢占前 [1] → 恢复后 [2,3]', '整表替换非追加', 'assert req_index is None'], 'resume')

# ---------------- worker 泳道内容 ----------------
CS_X, CS_W = 90, 540
lc.rect(CS_X, WRK_Y + 40, CS_W, 130, '#ffffff', lc.C_GPU_S, rx=7, sw=1.3)
lc.text(CS_X + 14, WRK_Y + 60, 'CachedRequestState.block_ids（worker 侧镜像）', 10, lc.C_GPU_S,
        'start', True, maxw=CS_W - 28, tag='cs:t')
lc.text(CS_X + 14, WRK_Y + 84, '新请求建档：block_ids = 全量 + add_row 整行写', 8.5, '#334155',
        'start', maxw=CS_W - 28, tag='cs:1')
lc.text(CS_X + 14, WRK_Y + 104, '在跑请求：block_ids.extend(new) 差量追加', 8.5, '#334155', 'start',
        maxw=CS_W - 28, tag='cs:2')
for i, (lab, val) in enumerate([('r1', '[1,2,3] → [1,2,3,4]'), ('r2', '[5]')]):
    yy = WRK_Y + 128 + i * 24
    lc.rect(CS_X + 20, yy - 12, 40, 18, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.0)
    lc.text(CS_X + 40, yy + 1, lab, 8.5, lc.C_GPU_S, 'middle', True, maxw=36, tag='csr%d' % i)
    lc.text(CS_X + 74, yy + 1, val + ('（extend [4]）' if i == 0 else ''), 9, '#334155', 'start',
            maxw=280, tag='csv%d' % i)
BT_X, BT_W = 660, 430
lc.rect(BT_X, WRK_Y + 40, BT_W, 130, '#ffffff', lc.C_GPU_S, rx=7, sw=1.3)
lc.text(BT_X + 14, WRK_Y + 60, 'BlockTable · CPU 页表行（append_row 差量）', 10, lc.C_GPU_S, 'start',
        True, maxw=BT_W - 28, tag='bt:t')
for i in range(4):
    cx = BT_X + 20 + i * 62
    lc.rect(cx, WRK_Y + 80, 54, 26, lc.C_KV_F if i < 3 else lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.1)
    lc.text(cx + 27, WRK_Y + 97, str([1, 2, 3, 4][i]), 10, lc.C_KV_S, 'middle', True, tag='bt%d' % i)
    if i == 3:
        lc.text(cx + 27, WRK_Y + 120, '← append_row', 7.5, lc.C_GPU_S, 'middle', maxw=70, tag='bt:ar')
lc.text(BT_X + 14, WRK_Y + 140, '行 0 = [1,2,3,4] · num_blocks_per_row = 4（行内偏移记账）', 8.5,
        '#334155', 'start', maxw=BT_W - 28, tag='bt:n')
lc.text(BT_X + 14, WRK_Y + 158, 'CpuGpuBuffer 双镜像：CPU 写行、commit 拷活跃行（→ ch18）', 7.8,
        lc.C_MUTE, 'start', maxw=BT_W - 28, tag='bt:n2')
ZW_X, ZW_W = 1120, BXR - 1120
lc.rect(ZW_X, WRK_Y + 40, ZW_W, 130, '#ffffff', lc.C_GPU_S, rx=7, sw=1.3)
lc.text(ZW_X + 14, WRK_Y + 60, '保洁 · KVBlockZeroer', 10, lc.C_GPU_S, 'start', True, maxw=ZW_W - 28,
        tag='zw:t')
lc.text(ZW_X + 14, WRK_Y + 82, '清零刚到手的块：上一任主人', 8.5, '#334155', 'start', maxw=ZW_W - 28,
        tag='zw:1')
lc.text(ZW_X + 14, WRK_Y + 100, '的字节还躺在显存里', 8.5, '#334155', 'start', maxw=ZW_W - 28,
        tag='zw:2')
lc.text(ZW_X + 14, WRK_Y + 122, '两组混合精度时清 [1,2,3,4,5,6]', 8.5, '#334155', 'start',
        maxw=ZW_W - 28, tag='zw:3')
lc.text(ZW_X + 14, WRK_Y + 142, '（防止陈旧 NaN 污染注意力）', 7.8, lc.C_MUTE, 'start',
        maxw=ZW_W - 28, tag='zw:4')

# ---------------- 过江箭头 ----------------
def vdrop(cx, y1, y2, color, marker='std', dash=False, sw=2.0):
    lc.seg(cx, y1, cx, y2, color, sw, marker, dash)

# 台账 → 拍1/拍2/拍3 包裹顶
vdrop(P1_X + P1_W / 2, SCH_Y + SCH_H + 2, RIV_Y + 58, lc.C_ENG_S, 'dn')
vdrop(P2_X + P2_W / 2, SCH_Y + SCH_H + 2, RIV_Y + 58, '#94a3b8', 'std', dash=True, sw=1.4)
vdrop(P3_X + P3_W / 2, SCH_Y + SCH_H + 2, RIV_Y + 58, lc.C_ENG_S, 'dn')
vdrop(P4_X + P4_W / 2, SCH_Y + SCH_H + 2, RIV_Y + 58, lc.C_ENG_S, 'dn')
# 包裹底 → worker 顶
vdrop(P1_X + P1_W / 2, RIV_Y + 60 + 118 + 2, WRK_Y - 3, lc.C_GPU_S, 'dn')
vdrop(P3_X + P3_W / 2, RIV_Y + 60 + 118 + 2, WRK_Y - 3, lc.C_GPU_S, 'dn')
vdrop(P4_X + P4_W / 2, RIV_Y + 60 + 118 + 2, WRK_Y - 3, lc.C_GPU_S, 'dn')
vdrop(RS_X + RS_W / 2, RIV_Y + 60 + 118 + 2, WRK_Y - 3, lc.C_GPU_S, 'dn')
# 拍1.5 免电报：不下投（无动作）
lc.text(P2_X + P2_W / 2, RIV_Y + 60 + 118 + 24, '× worker 无动作', 8, '#94a3b8', 'middle', maxw=150,
        tag='noact')
# 清零账旁路：自清零账盒左缘，走 拍4/恢复者小图 之间的竖巷，下投保洁盒
lc.parrow([(ZRO_X - 1, SCH_Y + 106), (1108, SCH_Y + 106), (1108, WRK_Y + 92), (ZW_X - 3, WRK_Y + 92)],
          lc.C_ABORT, 1.4, 'ab', dash=True)
lc.text(1100, 540, '旁路：new_block_ids_to_zero', 8, lc.C_ABORT, 'end', maxw=170, tag='bp:lbl')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = WRK_Y + WRK_H + 24
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', lc.C_ZMQ_S, rx=3, sw=1.3)
lc.text(lx + 26, LEG_Y + 1, '过线包裹（全量 / 增量）', 8.5, lc.C_TXT, 'start', maxw=180, tag='lg1')
lx += 26 + lc.tw('过线包裹（全量 / 增量）', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 20, 13, '#f8fafc', '#94a3b8', rx=3, sw=1.1, dash=True)
lc.text(lx + 26, LEG_Y + 1, '免电报（None）', 8.5, lc.C_TXT, 'start', maxw=120, tag='lg2')
lx += 26 + lc.tw('免电报（None）', 8.5) + 16
lc.rect(lx, LEG_Y - 11, 20, 15, '#fef2f2', lc.C_ABORT, rx=3, sw=1.2, dash=True)
lc.text(lx + 26, LEG_Y + 1, '恢复者整箱重寄 / 旁路通道', 8.5, lc.C_TXT, 'start', maxw=210, tag='lg3')
lx += 26 + lc.tw('恢复者整箱重寄 / 旁路通道', 8.5) + 16
lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, lc.C_ABORT, 1.4, dash=True)
lc.text(lx + 38, LEG_Y + 1, '清零旁路（红虚线）', 8.5, lc.C_TXT, 'start', maxw=150, tag='lg4')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/core/sched/scheduler.py:L1144-L1149（新请求全量）· L1451-L1453（增量 allow_none=True，空则 None）· '
        'L1260-L1272（清零账排干）· vllm/v1/core/kv_cache_manager.py:L89-L91（空增量语义）', 8.2, lc.C_FAINT,
        'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, 'vllm/v1/worker/gpu_model_runner.py:L1442-L1474（worker 镜像三态：建档 / extend / 整表替换）· '
        'vllm/v1/worker/block_table.py:L138-L154（append_row 差量）· 三拍过线与镜像数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 66
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-block-id-crossing.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
