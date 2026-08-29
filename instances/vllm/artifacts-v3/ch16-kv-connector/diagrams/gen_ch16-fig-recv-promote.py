#!/usr/bin/env python3
"""ch16 机制图 7 · 传输完成的提升（figure_spec ch16-fig-recv-promote，模板 state-machine）

放大自 L0「KV 账本列·请求状态格」（本章 l0_zoom）、L2 站 9（完成回收·退一 token）——
WAITING_FOR_REMOTE_KVS 状态的闭环（入口在护轨分配、出口在提升拍）。

claim：传输完成的提升不是直接放行：_update_waiting_for_remote_kv 先补缓存（『已分配
未缓存』窗口的块此刻才入哈希表），全命中还要退一个 token 重算（要 logits），再按
num_preemptions 分流回 WAITING/PREEMPTED 重入调度。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：部分命中先行 48 →
块账 4（3 补+1 新算）、同拍续算 16；全命中 64→63、本拍补算 1；状态闭环）。
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
lc.text(MX, 36, '传输完成的提升：先补账、再退一、然后才放行', 16.5, lc.C_TXT, 'start', True,
        maxw=900, tag='title')
lc.text(MX, 60, 'get_finished 报 finished_recving 只是入场券——下一拍 schedule 的 _try_promote 才真正结算：'
                '补缓存 → 全命中查 → 按 num_preemptions 分流', 10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L2 站 9 完成回收·退一 token · L0：KV 账本列·请求状态格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 状态链 ----------------
ST_Y, ST_H = 104, 64
S1 = (70, 220)
S2 = (330, 300)
S3 = (700, 330)
sx1, sx2, sx3 = S1[0], S2[0], S3[0]
lc.rect(sx1, ST_Y, S1[1], ST_H, '#ffffff', lc.C_MUTE, rx=12, sw=1.6)
lc.text(sx1 + S1[1] / 2, ST_Y + 39, 'WAITING', 13, lc.C_TXT, 'middle', True, maxw=S1[1] - 16, tag='st1')
lc.rect(sx2, ST_Y, S2[1], ST_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=12, sw=2.0)
lc.text(sx2 + S2[1] / 2, ST_Y + 27, 'WAITING_FOR_REMOTE_KVS', 11.5, lc.C_BEAT_T, 'middle', True, maxw=S2[1] - 16, tag='st2')
lc.text(sx2 + S2[1] / 2, ST_Y + 47, '已分配未缓存窗口（占块、零前向）', 8.5, lc.C_MUTE, 'middle', maxw=S2[1] - 16, tag='st2s')
lc.rect(sx3, ST_Y, S3[1], ST_H, '#ffffff', lc.C_KV_S, rx=12, sw=2.0)
lc.text(sx3 + S3[1] / 2, ST_Y + 27, '提升拍 · _try_promote', 11.5, lc.C_KV_S, 'middle', True, maxw=S3[1] - 16, tag='st3')
lc.text(sx3 + S3[1] / 2, ST_Y + 47, '（下一拍 schedule 里结算，scheduler.py:L2678-L2693）', 8.5, lc.C_MUTE,
        'middle', maxw=S3[1] - 16, tag='st3s')

# 分流双态
F_Y0, F_H, F_X, F_W = 84, 48, 1130, 230
lc.rect(F_X, F_Y0, F_W, F_H, '#ffffff', lc.C_MUTE, rx=12, sw=1.6)
lc.text(F_X + F_W / 2, F_Y0 + 30, 'WAITING（回队）', 11.5, lc.C_TXT, 'middle', True, maxw=F_W - 16, tag='f1')
lc.rect(F_X, F_Y0 + 62, F_W, F_H, '#f1f5f9', lc.C_MUTE, rx=12, sw=1.4)
lc.text(F_X + F_W / 2, F_Y0 + 92, 'PREEMPTED（回队）', 11.5, lc.C_MUTE, 'middle', True, maxw=F_W - 16, tag='f2')

# 链箭头
lc.seg(sx1 + S1[1], ST_Y + ST_H / 2, sx2 - 3, ST_Y + ST_H / 2, lc.C_BEAT_S, 2.0, 'std')
lc.text((sx1 + S1[1] + sx2) / 2, ST_Y - 8, 'connector 答 (N, True) 异步命中', 9, lc.C_BEAT_T, 'middle', True,
        maxw=230, tag='a:1')
lc.seg(sx2 + S2[1], ST_Y + ST_H / 2, sx3 - 3, ST_Y + ST_H / 2, lc.C_KV_S, 2.0, 'std')
lc.text((sx2 + S2[1] + sx3) / 2, ST_Y - 8, 'get_finished 报 finished_recving', 9, lc.C_KV_S, 'middle', True,
        maxw=250, tag='a:2')
lc.parrow([(sx3 + S3[1], ST_Y + 26), (F_X - 3, F_Y0 + F_H / 2)], lc.C_MUTE, 1.8, 'std')
lc.text((sx3 + S3[1] + F_X) / 2, F_Y0 + F_H / 2 - 34, 'num_preemptions = 0', 8.5, lc.C_MUTE, 'middle',
        maxw=160, tag='a:3')
lc.parrow([(sx3 + S3[1], ST_Y + ST_H - 26), (F_X - 3, F_Y0 + 62 + F_H / 2)], lc.C_MUTE, 1.8, 'std')
lc.text((sx3 + S3[1] + F_X) / 2, F_Y0 + 62 + F_H / 2 + 24, '被抢占过 > 0', 8.5, lc.C_MUTE, 'middle',
        maxw=160, tag='a:4')
lc.text(F_X + F_W / 2, F_Y0 + 62 + F_H + 20, '重入正常调度', 8.5, lc.C_MUTE, 'middle', maxw=F_W, tag='a:5')

# ---------------- 提升站内部展开 ----------------
EX_Y = 226
lc.rect(MX, EX_Y, BXR - MX, 330, lc.C_KV_F, lc.C_KV_S, rx=10, sw=1.8)
lc.text(MX + 18, EX_Y + 26, '提升站内部 · _update_waiting_for_remote_kv（scheduler.py:L2635-L2676）——三步结算，不许跳步',
        11.5, lc.C_KV_S, 'start', True, maxw=1300, tag='ex:t')
# 展开连线（提升拍框底 → 展开框顶）
lc.parrow([(sx3 + S3[1] / 2, ST_Y + ST_H), (sx3 + S3[1] / 2, EX_Y)], lc.C_KV_S, 1.8, 'std')
lc.text(sx3 + S3[1] / 2 + 8, (ST_Y + ST_H + EX_Y) / 2, '展开', 8.5, lc.C_KV_S, 'start', maxw=60, tag='ex:a')

STP_Y, STP_H = EX_Y + 44, 58
STP_W = (BXR - MX - 36 - 2 * 24) / 3
steps = [
    ('① 补缓存 cache_blocks', '『已分配未缓存』窗口的块此刻入哈希表——欠的登记补上'),
    ('② 全命中查', '外部缓存覆盖整个 prompt？覆盖则退一 token 重算'),
    ('③ 分流回队', '按 num_preemptions 回 WAITING / PREEMPTED 重入调度'),
]
for i, (t, s) in enumerate(steps):
    x = MX + 18 + i * (STP_W + 24)
    lc.rect(x, STP_Y, STP_W, STP_H, '#ffffff', lc.C_KV_S, rx=8, sw=1.4)
    lc.text(x + STP_W / 2, STP_Y + 23, t, 10.5, lc.C_KV_S, 'middle', True, maxw=STP_W - 16, tag=f'stp{i}t')
    lc.text(x + STP_W / 2, STP_Y + 42, s, 8.5, '#334155', 'middle', maxw=STP_W - 16, tag=f'stp{i}s')
    if i < 2:
        lc.seg(x + STP_W, STP_Y + STP_H / 2, x + STP_W + 24 - 3, STP_Y + STP_H / 2, lc.C_KV_S, 1.8, 'std')

# 两个案例卡
C_Y = STP_Y + STP_H + 24
C_H = 148
C_W = (BXR - MX - 36 - 24) / 2
# 案例卡标题行
lc.text(MX + 18, C_Y - 8, '② 的两个出口（实测）：', 9.5, lc.C_MUTE, 'start', maxw=300, tag='c:t')
# case1
cx1 = MX + 18
lc.rect(cx1, C_Y, C_W, C_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(cx1 + 16, C_Y + 24, '部分命中（ext 48 · prompt 64）', 11, lc.C_TXT, 'start', True, maxw=C_W - 32, tag='c1:t')
lc.text(cx1 + 16, C_Y + 48, 'num_computed_tokens 先行 48 → 提升后仍 48（promote_log 48→48）', 9, '#334155',
        'start', maxw=C_W - 32, tag='c1:l1')
lc.text(cx1 + 16, C_Y + 66, '提升拍块账 4 = 3 块补缓存 + 1 块新算（窗口欠账此刻结清）', 9, '#334155',
        'start', maxw=C_W - 32, tag='c1:l2')
lc.text(cx1 + 16, C_Y + 84, '同拍续算 16 token → 转 RUNNING', 9, '#334155', 'start', maxw=C_W - 32, tag='c1:l3')
lc.text(cx1 + 16, C_Y + 108, '缓存的块从这一拍起可被后续请求命中', 8.5, lc.C_MUTE, 'start', maxw=C_W - 32, tag='c1:l4')
# case2
cx2 = cx1 + C_W + 24
lc.rect(cx2, C_Y, C_W, C_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.6)
lc.text(cx2 + 16, C_Y + 24, '全命中（ext 64 = prompt 64）', 11, lc.C_BEAT_T, 'start', True, maxw=C_W - 32, tag='c2:t')
lc.text(cx2 + 16, C_Y + 48, 'num_computed_tokens 64 → 63（退一 token，promote_log 64→63）', 9, '#334155',
        'start', maxw=C_W - 32, tag='c2:l1')
lc.text(cx2 + 16, C_Y + 66, '本拍补算最后 1 个 token——要 logits 必须亲手算最后一个 token', 9, '#334155',
        'start', maxw=C_W - 32, tag='c2:l2')
lc.text(cx2 + 16, C_Y + 84, '与本地前缀缓存同一条契约（远端缓存不豁免采样必要性）', 9, '#334155',
        'start', maxw=C_W - 32, tag='c2:l3')
lc.text(cx2 + 16, C_Y + 108, '若不退一：最后一个 token 的 logits 无从采出', 8.5, lc.C_MUTE, 'start',
        maxw=C_W - 32, tag='c2:l4')
# 步② → 两案例的箭头（步②框底 → 两卡顶边）
s2x = MX + 18 + STP_W + 24 + STP_W / 2
lc.parrow([(s2x, STP_Y + STP_H), (s2x, C_Y - 16), (cx1 + C_W / 2, C_Y - 16), (cx1 + C_W / 2, C_Y)],
          lc.C_MUTE, 1.5, 'std')
lc.parrow([(s2x, STP_Y + STP_H), (s2x, C_Y - 16), (cx2 + C_W / 2, C_Y - 16), (cx2 + C_W / 2, C_Y)],
          lc.C_MUTE, 1.5, 'std')

# ---------------- 状态闭环注 ----------------
LP_Y = EX_Y + 330 + 26
lc.text(MX, LP_Y, '状态闭环：WAITING --异步命中--> WAITING_FOR_REMOTE_KVS --finished_recving--> 提升补缓存 --> '
        'WAITING（或 PREEMPTED）——一环走完，请求回到正常调度世界', 10, '#334155', 'start', True,
        maxw=BXR - MX, tag='loop')

# ---------------- 页脚 ----------------
FY = LP_Y + 30
lc.text(MX, FY, '逐字锚 vllm/v1/core/sched/scheduler.py:L2635-L2676（_update_waiting_for_remote_kv：cache_blocks 补缓存 + '
                '全命中退一 + 失败分支）· L2678-L2693（_try_promote_blocked_waiting_request）· L2714-L2741（finished_recving 入集）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '两案例读数（48→48/块账 4/续算 16；64→63/补算 1）取自精简版 companion host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 36

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-recv-promote.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
