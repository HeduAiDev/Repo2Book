#!/usr/bin/env python3
"""ch10 机制图 1 · 批是 token 账单（figure_spec ch10-fig-batch-is-token-ledger，模板 state-table）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半的 Scheduler 调度账本框——
即本章 L2 章图 center ⑦ 拍片『断言 · 组装 · 乐观推进』产出的 num_scheduled_tokens
账本的机制展开（产出面）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：同一个 running 批，五拍里批组成从「3 请求 / 24 token」到「4 请求 / 32 token」
再到「4 请求 / 4 token」——schedule() 产出的批以 {req_id: num_tokens} 记账、以 token
合计受预算约束，请求数从来不是约束。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：五拍批形状
3/24→4/32→4/32→4/9→4/4；r4 一生 29/29/6/1；预算 32，拍 2/3 打满、拍 4 余 23）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 848
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '批不是「一桌人」，是「一张账单」——schedule() 只认 token 数、不认请求数',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '每拍交出 {req_id: num_tokens}：同一批里 decode 请求各 1 份、新 prompt 首 chunk 29 份、续 chunk 又 29 份、尾 chunk 6 份——'
        '账单合计被预算 32 钉死（拍 2/3 恰好打满）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ 账单产出 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主表：五拍账单 ----------------
HDR_Y = 96
ROW_Y0, ROW_H = 114, 86
REQS = ['r1', 'r2', 'r3', 'r4']
REQ_X0, REQ_CW = 124, 140          # r1/r2/r3 列；r4 列更宽
R4_X, R4_CW = 544, 296             # 29 token × 10px = 290
TOKPX = 10.0

BEATS = [
    (1, {'r1': 8, 'r2': 8, 'r3': 8}, 3, 24, 8,
     ['三个新 prompt 同拍全量进批', '（拍 1 running 为空，WAITING 收新）']),
    (2, {'r1': 1, 'r2': 1, 'r3': 1, 'r4': 29}, 4, 32, 0,
     ['3 个 decode 各 1 + r4 首 chunk 29', '需求 64 被余额 29 截——恰好打满']),
    (3, {'r1': 1, 'r2': 1, 'r3': 1, 'r4': 29}, 4, 32, 0,
     ['r4 续 chunk 又领 29', '（差 35，预算又只剩 29）']),
    (4, {'r1': 1, 'r2': 1, 'r3': 1, 'r4': 6}, 4, 9, 23,
     ['r4 尾 chunk 6，prefill 收官', '批不再打满：余 23']),
    (5, {'r1': 1, 'r2': 1, 'r3': 1, 'r4': 1}, 4, 4, 28,
     ['全 decode 稳态：每人恰 1', '「份数」从此恒定']),
]

# 列头
for i, rid in enumerate(REQS):
    if rid == 'r4':
        cx, w = R4_X + R4_CW / 2, R4_CW
        lab = 'r4 · 64-token prompt（拍 2 到达）'
    else:
        cx, w = REQ_X0 + i * REQ_CW + REQ_CW / 2, REQ_CW
        lab = f'{rid} · 8-token prompt'
    lc.text(cx, HDR_Y, lab, 9.5, lc.C_MUTE, 'middle', True, maxw=w - 8, tag='hd' + rid)
CNT_X, TOT_X = 876, 986
GAUGE_X, GAUGE_W = 1058, 24
OBS_X = 1136
lc.text(CNT_X, HDR_Y, '批内请求数', 9.5, lc.C_MUTE, 'middle', True, maxw=100, tag='hd:cnt')
lc.text(TOT_X, HDR_Y, 'token 合计', 9.5, lc.C_MUTE, 'middle', True, maxw=100, tag='hd:tot')
lc.text(GAUGE_X + GAUGE_W / 2, HDR_Y, '预算 32 水位', 9.5, lc.C_MUTE, 'middle', True, maxw=110, tag='hd:gage')
lc.text(OBS_X, HDR_Y, '关键观察', 9.5, lc.C_MUTE, 'start', True, maxw=100, tag='hd:obs')

for bi, (beat, sched, nreq, total, left, obs) in enumerate(BEATS):
    ry = ROW_Y0 + bi * ROW_H
    # 行分隔线
    if bi > 0:
        lc.seg(MX, ry - 2, 1110, ry - 2, '#e2e8f0', 1.0)
    # 拍号徽标
    lc.rect(MX, ry + 26, 46, 34, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.2)
    lc.text(MX + 23, ry + 47, f'拍 {beat}', 10.5, lc.C_ENG_S, 'middle', True, maxw=42, tag=f'bdg{beat}')
    # 批组成票据（宽度 ∝ num_tokens）
    for rid in REQS:
        if rid == 'r4':
            bx, bw = R4_X, R4_CW
        else:
            bx, bw = REQ_X0 + REQS.index(rid) * REQ_CW, REQ_CW
        n = sched.get(rid)
        if n is None:
            lc.text(bx + bw / 2, ry + 46, '—（未到）', 8.5, lc.C_FAINT, 'middle', maxw=bw - 6,
                    tag=f'abs{beat}{rid}')
            continue
        w = max(10.0, n * TOKPX)
        ty = ry + 24
        if n == 1:
            lc.rect(bx, ty, w, 24, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.3)
        else:
            lc.rect(bx, ty, w, 24, lc.C_KV_S, lc.C_KV_S, rx=3, sw=1.0)
        lc.text(bx + w / 2, ry + 72, f'{rid}:{n}', 9, lc.C_TXT if n > 1 else lc.C_MUTE,
                'middle', True, maxw=70, tag=f'tk{beat}{rid}')
    # 请求数 / token 合计
    lc.text(CNT_X, ry + 50, str(nreq), 20, lc.C_TXT, 'middle', True, maxw=90, tag=f'cnt{beat}')
    hot = (total == 32)
    lc.text(TOT_X, ry + 50, str(total), 20, lc.C_BEAT_T if hot else lc.C_TXT, 'middle', True,
            maxw=90, tag=f'tot{beat}')
    # 预算水位竖条（0..32，拍末余额留白）
    gy0, gh = ry + 8, 66
    lc.rect(GAUGE_X, gy0, GAUGE_W, gh, '#ffffff', lc.C_MUTE, rx=3, sw=1.1)
    fh = gh * total / 32
    if fh > 0.5:
        lc.rect(GAUGE_X, gy0 + gh - fh, GAUGE_W, fh, lc.C_BEAT_S, lc.C_BEAT_S, rx=2, sw=0)
    lc.text(GAUGE_X + GAUGE_W + 8, ry + 44, f'余 {left}', 8.5, lc.C_MUTE, 'start', maxw=52,
            tag=f'lft{beat}')
    # 关键观察
    for li, ln in enumerate(obs):
        lc.text(OBS_X, ry + 38 + li * 19, ln, 8.7, '#334155', 'start', maxw=BXR - OBS_X,
                tag=f'obs{beat}:{li}')

# ---------------- r4 一生小注 ----------------
lc.text(R4_X, ROW_Y0 + 5 * ROW_H + 16, 'r4 一生领到的份数：首 chunk 29（被余 29 截）→ 续 29 → 尾 6 → decode 1',
        8.7, lc.C_KV_S, 'start', True, maxw=640, tag='r4life')

# ---------------- 底部读数条：恒定的人数 vs 波动的份数 ----------------
STRIP_T = ROW_Y0 + 5 * ROW_H + 42
lc.text(MX, STRIP_T, '五拍并排读数：请求数恒 3→4（自拍 2 纹丝不动），token 合计 24→32→32→9→4（随「份数」波动）——约束是份数合计，不是人头数',
        10.5, lc.C_TXT, 'start', True, maxw=1080, tag='strip:t')
CH_Y0, CH_H = STRIP_T + 14, 64

def spark(x0, x1, vals, vmax, color, title, budget_line=None):
    w = x1 - x0
    lc.text(x0, CH_Y0 + 4, title, 8.5, lc.C_MUTE, 'start', True, maxw=w, tag='sp:' + title[:8])
    base, ph = CH_Y0 + CH_H, CH_H - 22
    pts = []
    for i, v in enumerate(vals):
        px = x0 + 26 + i * (w - 52) / (len(vals) - 1)
        py = base - 4 - v * ph / vmax
        pts.append((px, py))
        lc.rect(px - 3.5, py - 3.5, 7, 7, color, color, rx=1.5, sw=0)
        lc.text(px, base + 12, f'拍{i + 1}', 7.5, lc.C_FAINT, 'middle', maxw=44, tag=f'bx{i}' + title[:4])
        lc.text(px, py - 8, str(v), 8.5, color, 'middle', True, maxw=34, tag=f'bv{i}' + title[:4])
    for i in range(len(pts) - 1):
        lc.seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color, 1.4)
    if budget_line is not None:
        by = base - 4 - budget_line * ph / vmax
        lc.seg(x0 + 8, by, x1 - 8, by, lc.C_BEAT_S, 1.3, dash=True)
        lc.text(x1 - 10, by - 5, f'预算 {budget_line}', 8, lc.C_BEAT_T, 'end', True, maxw=80,
                tag='bl' + title[:6])

spark(MX, 640, [3, 4, 4, 4, 4], 6, lc.C_MUTE, '批内请求数（人）——恒定')
spark(700, BXR, [24, 32, 32, 9, 4], 40, lc.C_KV_S, 'token 合计（份）——波动', budget_line=32)

# ---------------- 契约原文引文 ----------------
Q_Y = CH_Y0 + CH_H + 26
lc.rect(MX, Q_Y, BXR - MX, 50, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 16, Q_Y + 20, '契约原文（vllm/v1/core/sched/interface.py:L54-L67）：num_tokens can be as large as the number of prompt tokens for new requests, or it can be 1 … one by one.',
        8.7, '#334155', 'start', maxw=BXR - MX - 32, tag='q:l1')
lc.text(MX + 16, Q_Y + 38, 'Otherwise, it can be somewhere in between in case of chunked prefills, prefix caching, speculative decoding, etc.——份数可为整段 prompt、可为 1、可居中。',
        8.7, '#334155', 'start', maxw=BXR - MX - 32, tag='q:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = Q_Y + 72
lx = MX
items = [
    ('dec', 'decode 份（num_tokens = 1）'),
    ('pre', 'prefill / chunk 份（num_tokens > 1）'),
    ('gag', '预算水位（本拍 token 合计 / 32）'),
]
for kind, name in items:
    if kind == 'dec':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
    elif kind == 'pre':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    else:
        lc.rect(lx, LEG_Y - 8, 14, 14, lc.C_BEAT_S, lc.C_BEAT_S, rx=3, sw=0)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 22
lc.seg(lx, LEG_Y - 2, lx + 22, LEG_Y - 2, lc.C_BEAT_S, 1.3, dash=True)
lc.text(lx + 28, LEG_Y + 2, '预算上限线', 8.5, lc.C_TXT, 'start', maxw=100, tag='leg:bud')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L439（schedule）· L636-L637（num_scheduled_tokens[req_id] = n · token_budget −= n）· 守恒断言 L1108-L1111',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '五拍读数取自精简版 companion host 实测（32 为示教预算；真实默认 2048——config/scheduler.py:L42 自认测试便利，服务端 8192/16384——arg_utils.py:L2541-L2563）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-batch-is-token-ledger.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
