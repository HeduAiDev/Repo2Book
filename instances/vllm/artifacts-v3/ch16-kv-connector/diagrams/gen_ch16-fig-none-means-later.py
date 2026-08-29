#!/usr/bin/env python3
"""ch16 机制图 2 · None ≠ 0（figure_spec ch16-fig-none-means-later，模板 state-machine）

放大自 L0「KV 账本列·调度查询格」（本章 l0_zoom）、L2 站 3（双查·外部=第二个前缀缓存）。
架构归属回指 L0/L2：图右上角指北小签。

claim：None ≠ 0：『还没查到』和『查到为零』是两种答案——前者把请求移入 skipped
退避队列稍后再问（不堵 waiting 队头），后者立即按零外部命中排入计算；有数字则
外部 token 与本地命中同一拍合成 num_computed_tokens。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：步 1 None →
skipped_waiting=true / blocks_held=0 / 零调度；步 2 (32, False) → 本地 0+外部 32、
本拍算 64−32=32、RUNNING；查询入参 = block_aligned_local）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 54
BXR = 1446

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'None ≠ 0：『还没查到』与『查到为零』是两种答案', 16.5, lc.C_TXT, 'start', True,
        maxw=880, tag='title')
lc.text(MX, 60, 'get_num_new_matched_tokens 把外部缓存当第二个前缀缓存查——三态答复；None 走 skipped 退避队列稍后再问，'
                '不堵 waiting 队头（scheduler.py:L773-L789）', 10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L2 站 3 双查 · L0：KV 账本列·调度查询格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 查询节点（左） ----------------
Q_X, Q_Y, Q_W, Q_H = 90, 210, 400, 150
lc.rect(Q_X, Q_Y, Q_W, Q_H, lc.C_KV_F, lc.C_KV_S, rx=22, sw=2.0)
lc.text(Q_X + Q_W / 2, Q_Y + 32, '① 查外部缓存', 13, lc.C_KV_S, 'middle', True, maxw=Q_W - 30, tag='q:t')
lc.text(Q_X + Q_W / 2, Q_Y + 56, 'get_num_new_matched_tokens(', 10, lc.C_TXT, 'middle', maxw=Q_W - 24, tag='q:c1')
lc.text(Q_X + Q_W / 2, Q_Y + 72, '  request, block_aligned_local)', 10, lc.C_TXT, 'middle', maxw=Q_W - 24, tag='q:c2')
lc.text(Q_X + Q_W / 2, Q_Y + 96, '入参 = 本地命中砍尾后的块对齐值', 9, '#334155', 'middle', maxw=Q_W - 24, tag='q:s1')
lc.text(Q_X + Q_W / 2, Q_Y + 112, '（scheduler.py:L773-L781，本例值 0）', 9, '#334155', 'middle', maxw=Q_W - 24, tag='q:s2')
lc.text(Q_X + Q_W / 2, Q_Y + 134, '外部缓存 = 第二个前缀缓存 · 查询 side-effect free', 9, lc.C_MUTE, 'middle',
        maxw=Q_W - 24, tag='q:s3')

# ---------------- 三态结果节点（右） ----------------
R_X, R_W = 660, 560
# 有数字（命中）
H_Y, H_H = 120, 118
lc.rect(R_X, H_Y, R_W, H_H, lc.C_KV_F, lc.C_KV_S, rx=10, sw=1.8)
lc.text(R_X + 20, H_Y + 26, '返回数字 (32, False) —— 命中', 12, lc.C_KV_S, 'start', True,
        maxw=R_W - 40, tag='h:t')
lc.text(R_X + 20, H_Y + 50, '外部 token 与本地命中同一拍合成已算：本地 0 + 外部 32 = 32', 9.5, '#334155',
        'start', maxw=R_W - 40, tag='h:l1')
lc.text(R_X + 20, H_Y + 68, '本拍只算 64 − 32 = 32（一半 prefill 被外部缓存吸收）→ 转 RUNNING', 9.5, '#334155',
        'start', maxw=R_W - 40, tag='h:l2')
lc.text(R_X + 20, H_Y + 90, 'update_state_after_alloc 收到 32 · 计划随 SchedulerOutput 过线', 9, lc.C_MUTE,
        'start', maxw=R_W - 40, tag='h:l3')
# 返回 0
Z_Y, Z_H = 296, 96
lc.rect(R_X, Z_Y, R_W, Z_H, '#f1f5f9', lc.C_MUTE, rx=10, sw=1.6)
lc.text(R_X + 20, Z_Y + 26, '返回 0 (0, False) —— 查到为零', 12, lc.C_MUTE, 'start', True,
        maxw=R_W - 40, tag='z:t')
lc.text(R_X + 20, Z_Y + 50, '未命中是确定的答案：立即按零外部命中排入计算', 9.5, '#334155', 'start',
        maxw=R_W - 40, tag='z:l1')
lc.text(R_X + 20, Z_Y + 68, '本拍全量算（无外部抵扣）——与「没有 connector」的行为一致', 9.5, '#334155',
        'start', maxw=R_W - 40, tag='z:l2')
# 返回 None（主角）
N_Y, N_H = 452, 150
lc.rect(R_X, N_Y, R_W, N_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=10, sw=2.2)
lc.text(R_X + 20, N_Y + 28, '返回 None —— 还没查到（≠ 0！）', 12.5, lc.C_BEAT_T, 'start', True,
        maxw=R_W - 40, tag='n:t')
lc.text(R_X + 20, N_Y + 52, '请求移入 skipped_waiting 退避队列：状态仍 WAITING · 零占块 · 本拍零调度', 9.5,
        '#334155', 'start', maxw=R_W - 40, tag='n:l1')
lc.text(R_X + 20, N_Y + 70, '不堵 waiting 队头——后面的请求照常调度（scheduler.py:L783-L789）', 9.5, '#334155',
        'start', maxw=R_W - 40, tag='n:l2')
lc.text(R_X + 20, N_Y + 92, '下一拍必被重新查询：FCFS 下 skipped 队列先于 waiting 被选', 9, lc.C_MUTE,
        'start', maxw=R_W - 40, tag='n:l3')
lc.text(R_X + 20, N_Y + 110, '（scheduler.py:L2065-L2066）· 查询无副作用——重试免费，', 9, lc.C_MUTE,
        'start', maxw=R_W - 40, tag='n:l4')
lc.text(R_X + 20, N_Y + 128, '答案从 None 推进到数字的那一刻立即被采纳', 9, lc.C_MUTE,
        'start', maxw=R_W - 40, tag='n:l5')

# ---------------- 查询 → 三态的分叉箭头（贴框边） ----------------
lc.seg(Q_X + Q_W, Q_Y + 36, R_X - 4, H_Y + H_H / 2, lc.C_KV_S, 2.0, 'std')
lc.text((Q_X + Q_W + R_X) / 2, H_Y + H_H / 2 - 26, '有数字', 9.5, lc.C_KV_S, 'middle', True, maxw=90, tag='a:h')
lc.seg(Q_X + Q_W, Q_Y + Q_H / 2, R_X - 4, Z_Y + Z_H / 2, lc.C_MUTE, 2.0, 'std')
lc.text((Q_X + Q_W + R_X) / 2, Z_Y + Z_H / 2 - 26, '返回 0', 9.5, lc.C_MUTE, 'middle', True, maxw=90, tag='a:z')
lc.seg(Q_X + Q_W, Q_Y + Q_H - 36, R_X - 4, N_Y + N_H / 2, lc.C_BEAT_S, 2.2, 'std')
lc.text((Q_X + Q_W + R_X) / 2, N_Y + N_H / 2 - 26, '返回 None', 9.5, lc.C_BEAT_T, 'middle', True, maxw=90, tag='a:n')

# ---------------- skipped 重问回环（None → 查询） ----------------
LB_Y = N_Y + N_H + 46
lc.parrow([(R_X + 120, N_Y + N_H), (R_X + 120, LB_Y), (Q_X + Q_W / 2, LB_Y), (Q_X + Q_W / 2, Q_Y + Q_H)],
          lc.C_BEAT_S, 1.8, 'std')
lc.text((R_X + Q_X + Q_W / 2) / 2, LB_Y + 16, '② 下一拍重问：每步都有一拍重试、不损失活性', 9, lc.C_BEAT_T,
        'middle', True, maxw=560, tag='loop')

# ---------------- 实测证据条（页脚上方） ----------------
EV_Y = LB_Y + 40
lc.rect(MX, EV_Y, BXR - MX, 74, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 18, EV_Y + 22, '实测两拍（64-token prompt · kv_consumer）', 10, lc.C_TXT, 'start', True,
        maxw=520, tag='ev:t')
lc.text(MX + 18, EV_Y + 44, '步 1 答 None：r1 进 skipped_waiting（=true）、持有块 0、本拍零调度——waiting 队头不被堵住',
        9, '#334155', 'start', maxw=1300, tag='ev:l1')
lc.text(MX + 18, EV_Y + 61, '步 2 答 (32, False)：本地 0 + 外部 32、本拍算 64−32=32、r1 转 RUNNING——外部缓存抵掉一半 prefill',
        9, '#334155', 'start', maxw=1300, tag='ev:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = EV_Y + 100
lx = MX
for fill, stroke, name in [(lc.C_KV_F, lc.C_KV_S, '有数字 = 外部命中（并入已算）'),
                           ('#f1f5f9', lc.C_MUTE, '0 = 确定未命中（照常计算）'),
                           (lc.C_BEAT_F, lc.C_BEAT_S, 'None = 还没查到（退避重问）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=4, sw=1.4)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=260, tag='leg')
    lx += 26 + lc.tw(name, 8.5) + 26

FY = LEG_Y + 26
lc.text(MX, FY, '逐字锚 vllm/v1/core/sched/scheduler.py:L773-L781（block_aligned_local 入参）· L783-L789（None → '
                'step_skipped_waiting）· L2065-L2066（FCFS 先选 skipped 队列）· vllm/distributed/kv_transfer/kv_connector/v1/base.py:L465-L498（docstring：可多次调用、side-effect free）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '两拍读数取自精简版 companion host 实测 · 行号基线 vLLM v0.27.1', 8.5, lc.C_FAINT,
        'start', maxw=BXR - MX, tag='foot2')

H = FY + 36

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-none-means-later.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
