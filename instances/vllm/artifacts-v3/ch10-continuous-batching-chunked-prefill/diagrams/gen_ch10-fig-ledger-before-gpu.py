#!/usr/bin/env python3
"""ch10 机制图 8 · 账本先记、GPU 后算（figure_spec ch10-fig-ledger-before-gpu，模板 before-after）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框的「账本」本体——
即本章 L2 章图 center ⑦ 拍片『GPU 还没算先记账 · is_prefill_chunk 标记』的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：schedule() 返回的那一刻，账本已推进而 GPU 未动：40-token prompt 三拍间已算
0→16→32→40 全部发生在 schedule() 内部（in_flight 同步累计 16/32/40），第三拍
is_prefill_chunk 翻 False、移出 _inflight_prefills；抢占拍的 preempted 集合「换新不
clear」——输出持旧集 {a2}，调度器侧已是新空集。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：三拍已算
0→16→32→40、chunk [16,16,8]；in_flight 16/32/40；is_prefill_chunk True/True/False、
第三拍移出；集合换新 same_object=True、输出侧 {a2}、调度器侧 []；注释三条理由
L1317-L1327 原文）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 828
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '账本先记、GPU 后算——schedule() 返回那一刻，已算读数已经到位',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '_update_after_schedule（scheduler.py:L1317-L1343）：三拍 chunk [16,16,8] 的已算 0→16→32→40 全部发生在 schedule() 内部——此刻没有任何执行调用',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ GPU 还没算先记账 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左面板（实拍）：schedule() 返回时刻 ----------------
PX, PY, PW, PH = MX, 92, 880, 330
lc.rect(PX, PY, PW, PH, '#ffffff', lc.C_KV_S, rx=9, sw=2.0)
lc.text(PX + 16, PY + 24, '实拍 · schedule() 返回时刻（40-token prompt，预算 16）', 11.5,
        lc.C_KV_S, 'start', True, maxw=PW - 32, tag='p:t')
lc.text(PX + 16, PY + 42, 'GPU 列全程灰暗——「未动」；账本列每拍已 +n', 8.5, lc.C_MUTE,
        'start', maxw=PW - 32, tag='p:s')

# 列头
LED_X, LED_W = PX + 40, 320
GPU_X, GPU_W = PX + 560, 280
lc.text(LED_X + LED_W / 2, PY + 62, '账本（Scheduler 侧）', 9.5, lc.C_TXT, 'middle', True,
        maxw=200, tag='p:lh')
lc.text(GPU_X + GPU_W / 2, PY + 62, 'GPU', 9.5, lc.C_MUTE, 'middle', True, maxw=100, tag='p:gh')

BEATS = [
    (1, 16, 16, 16, True, True),
    (2, 16, 32, 32, True, True),
    (3, 8, 40, 40, False, False),
]
ROW_Y0, ROW_H = PY + 74, 78
BAR_MAXW = 300.0 / 40   # px per token（进度条：40 token = 300px）
for bi, (beat, chunk, computed, inflight, ispc, inflight_set) in enumerate(BEATS):
    ry = ROW_Y0 + bi * ROW_H
    mid = ry + 30
    # 拍号徽标
    lc.rect(PX + 14, mid - 12, 40, 24, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.1)
    lc.text(PX + 34, mid + 3.5, f'拍 {beat}', 9, lc.C_ENG_S, 'middle', True, maxw=36,
            tag=f'bdg{beat}')
    # 账本进度条：0..40（40-token 片长），已算染深、本拍新记加深标注
    bx0, bw_total = LED_X, 300
    lc.rect(bx0, mid - 10, bw_total, 20, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.1)
    fw = computed * BAR_MAXW
    if fw > 0:
        lc.rect(bx0, mid - 10, fw, 20, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    cw_ = chunk * BAR_MAXW
    lc.rect(bx0 + fw - cw_, mid - 10, cw_, 20, '#0e7490', '#0e7490', rx=2, sw=0)
    # 读数行
    lc.text(bx0 + 316, mid - 2, f'chunk +{chunk}', 9, '#0e7490', 'start', True, maxw=90,
            tag=f'ck{beat}')
    lc.text(bx0 + 316, mid + 15, f'已算 {computed}/40', 9, lc.C_KV_S, 'start', True, maxw=90,
            tag=f'cp{beat}')
    lc.text(bx0 + 316, mid + 32, f'in_flight {inflight}', 8, lc.C_MUTE, 'start', maxw=90,
            tag=f'fl{beat}')
    # is_prefill_chunk 标记
    tag_txt = ('is_prefill_chunk=True · 在 _inflight_prefills' if ispc
               else 'is_prefill_chunk=False · 移出 _inflight_prefills')
    lc.text(PX + 66, mid + 44 if not ispc else mid + 44, tag_txt, 8,
            lc.C_BEAT_T if not ispc else lc.C_MUTE, 'start', True, maxw=430,
            tag=f'pc{beat}')
    # GPU 列：灰暗未动
    lc.rect(GPU_X, mid - 12, GPU_W, 24, '#f1f5f9', lc.C_FAINT, rx=4, sw=1.0, dash=True)
    lc.text(GPU_X + GPU_W / 2, mid + 3.5, '未动（没有任何执行调用）', 8, lc.C_FAINT, 'middle',
            maxw=GPU_W - 10, tag=f'gpu{beat}')

# ---------------- 右面板（对照，虚化）：⑤ 拍之后 ----------------
QX, QY, QW, QH = 972, 92, 468, 330
lc.rect(QX, QY, QW, QH, '#ffffff', lc.C_MUTE, rx=9, sw=1.3, dash=True)
lc.text(QX + 16, QY + 24, '对照 · ⑤ 拍之后（ch9 已立 / ch11 展开）', 11, lc.C_TXT, 'start',
        True, maxw=QW - 32, tag='q:t')
lc.text(QX + 16, QY + 42, '本图不讲它，只为摆正记账方向——数字不标', 8.5, lc.C_MUTE,
        'start', maxw=QW - 32, tag='q:s')
# GPU 点亮 + in_flight 归零方向（示意，无数字）
GY = QY + 80
lc.rect(QX + 30, GY, 180, 28, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.4)
lc.text(QX + 120, GY + 18, 'GPU 点亮 · 前向 + 采样', 8.5, lc.C_GPU_S, 'middle', True,
        maxw=170, tag='q:gpu')
lc.rect(QX + 250, GY, 190, 28, '#ffffff', lc.C_MUTE, rx=4, sw=1.2)
lc.text(QX + 345, GY + 18, 'update_from_output 记账回冲', 8.5, lc.C_MUTE, 'middle',
        maxw=180, tag='q:upd')
lc.parrow([(QX + 215, GY + 14), (QX + 245, GY + 14)], lc.C_GPU_S, 1.8, 'std')
# 注释三条理由（L1317-L1327 原文要点）
RY = QY + 132
lc.text(QX + 16, RY, '为什么敢先记（源码注释三条理由，L1317-L1327）：', 9, lc.C_TXT,
        'start', True, maxw=QW - 32, tag='q:whyt')
for i, (num, txt) in enumerate([
        ('①', '本拍 output 要用原始数定输入——先记账会污染输入'),
        ('②', '让 prefill 请求下一拍立即再可调度——chunked prefill 连拍的关键'),
        ('③', '被拒时（如 spec token 被拒）在 update_from_output 回调冲回')]):
    lc.text(QX + 26, RY + 20 + i * 18, num, 9, lc.C_BEAT_T, 'start', True, maxw=14,
            tag=f'q:r{i}')
    lc.text(QX + 42, RY + 20 + i * 18, txt, 8.3, '#334155', 'start', maxw=QW - 60,
            tag=f'q:rt{i}')
lc.text(QX + 16, RY + 84, '本精简版无 ⑤ 拍：in_flight 只增不降（trace 已标注）——', 8,
        lc.C_MUTE, 'start', maxw=QW - 32, tag='q:c1')
lc.text(QX + 16, RY + 100, '验证的是 _update_after_schedule 的记账方向，非全生命周期', 8,
        lc.C_MUTE, 'start', maxw=QW - 32, tag='q:c2')

# ---------------- 底部：集合换新不 clear ----------------
SW_Y = PY + PH + 28
lc.rect(MX, SW_Y, BXR - MX, 148, '#ffffff', lc.C_MUTE, rx=9, sw=1.3)
lc.text(MX + 16, SW_Y + 24, '抢占拍的集合「换新不 clear」（B 段拍 2 实测：块池 2，a2 被 FCFS 抢占）', 11,
        lc.C_TXT, 'start', True, maxw=900, tag='sw:t')
# 旧集气泡（连 SchedulerOutput）
B1_X, B1_Y, B1_W, B1_H = MX + 40, SW_Y + 44, 380, 60
lc.rect(B1_X, B1_Y, B1_W, B1_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=18, sw=1.5)
lc.text(B1_X + B1_W / 2, B1_Y + 24, 'SchedulerOutput 持旧集', 9.5, lc.C_BEAT_T, 'middle',
        True, maxw=B1_W - 20, tag='sw:b1t')
lc.text(B1_X + B1_W / 2, B1_Y + 44, '{a2} 原样保留（随批下发抢占通知）', 9.5, lc.C_BEAT_T,
        'middle', True, maxw=B1_W - 20, tag='sw:b1v')
# 新空集气泡（连 scheduler）
B2_X, B2_Y, B2_W, B2_H = MX + 900, SW_Y + 44, 380, 60
lc.rect(B2_X, B2_Y, B2_W, B2_H, lc.C_KV_F, lc.C_KV_S, rx=18, sw=1.5)
lc.text(B2_X + B2_W / 2, B2_Y + 24, '调度器侧已换新空集', 9.5, lc.C_KV_S, 'middle', True,
        maxw=B2_W - 20, tag='sw:b2t')
lc.text(B2_X + B2_W / 2, B2_Y + 44, 'reset_preempted_req_ids = set()', 9.5,
        lc.C_KV_S, 'middle', True, maxw=B2_W - 20, tag='sw:b2v')
# 中缝换新标注（连线 + 线上标签）
SC_CX = (B1_X + B1_W + B2_X) / 2
lc.seg(B1_X + B1_W + 4, B1_Y + 30, B2_X - 4, B2_Y + 30, lc.C_MUTE, 1.4)
lc.text(SC_CX, B1_Y + 22, '换新（新对象）而非 clear()', 9.5, lc.C_ABORT, 'middle', True,
        maxw=200, tag='sw:cut')
# 实测注 + 注释原文
lc.text(MX + 16, SW_Y + 124, '实测：output_set_is_scheduler_set_object = True——输出侧仍 {a2}、调度器侧 []；就地 clear() 会让已发出的 SchedulerOutput 跟着变空，worker 就收不到抢占通知', 8.2,
        '#334155', 'start', maxw=BXR - MX - 32, tag='sw:ev')
lc.text(MX + 16, SW_Y + 140, '「We shouldn\'t just clear() here because it will also affect the scheduler output.」——L1361-L1365 注释原文', 8.2,
        lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='sw:q')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = SW_Y + 172
lx = MX
items = [
    ('old', '已算（此前拍累计）'),
    ('new', '本拍新记 = chunk'),
]
for kind, name in items:
    if kind == 'old':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    else:
        lc.rect(lx, LEG_Y - 8, 20, 12, '#0e7490', '#0e7490', rx=3, sw=0)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=220, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 22
lc.rect(lx, LEG_Y - 9, 22, 14, '#f1f5f9', lc.C_FAINT, rx=3, sw=1.0, dash=True)
lc.text(lx + 28, LEG_Y + 2, 'GPU 未动（虚化=不在本图）', 8.5, lc.C_TXT, 'start', maxw=220,
        tag='leg:gpu')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L1317-L1343（_update_after_schedule：computed += n · in_flight += n · is_prefill_chunk 标记 · 移出 _inflight_prefills）· L1361-L1365（换新不 clear）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, 'A/B 两段读数取自精简版 companion host 实测（A：预算 16、40-token prompt；B：块池 2、拍 2 抢占 a2）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-ledger-before-gpu.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
