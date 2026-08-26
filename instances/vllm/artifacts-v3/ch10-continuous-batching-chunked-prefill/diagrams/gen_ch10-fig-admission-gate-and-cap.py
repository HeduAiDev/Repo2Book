#!/usr/bin/env python3
"""ch10 机制图 5 · 收新的两道闸（figure_spec ch10-fig-admission-gate-and-cap，模板 state-machine）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框的「RUNNING 先行/抢占」
格到「WAITING 收新」格之间的闸门——即本章 L2 章图 center ②→③ 拍片『preempted_reqs 空？』
判定与守卫行的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：两道门决定收不收新：① 本拍抢占过（preempted_reqs 非空）→ 整拍关闸——实测拍 2 里
r3 连一次 allocate 询问都没有，尽管空闲 1 块、预算剩 2047；② len(running) ≥ max_num_seqs
→ break——实测 cap=2 下 r3 等到有人退场才进。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：a·拍2 r3 零 allocate、
空闲 1 块、预算余 2047；拍 3 waiting [r2,r3]、r2 需 3 块 > 1 → break；b 场景 cap=2 下
num_running 2/2、r1 退场后拍 3 收 r3 16 全量；cap 默认 128、服务端 256/1024）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 724
MX = 60
BXR = 1440


def diamond(cx, cy, hw, hh, fill, stroke, sw=1.6):
    d = f'M{cx:.1f},{cy - hh:.1f} L{cx + hw:.1f},{cy:.1f} L{cx:.1f},{cy + hh:.1f} L{cx - hw:.1f},{cy:.1f} Z'
    s = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    lc.ELEMS.append(((cx - hw - 2, cy - hh - 2, cx + hw + 2, cy + hh + 2), s))


def dtext(cx, cy, lines, fs=9.5, w=150):
    n = len(lines)
    for i, ln in enumerate(lines):
        lc.text(cx, cy - (n - 1) * 8 + i * 16 - 3, ln, fs, lc.C_TXT, 'middle', True,
                maxw=w, tag='d:' + ln[:10])


# ---------------- 标题区 ----------------
lc.text(MX, 34, '收新的两道闸：刚抢过人就不检票，在座满了就不放行',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'if not preempted_reqs and UNPAUSED（L684）· num_running ≥ max_num_running_reqs → break（L690-L692）——两道闸都把延迟压力推向 TTFT 一侧',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ③ WAITING 收新 · 守卫 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 状态机主体 ----------------
# 左态：阶段一 RUNNING（含抢占环）
LX, LY, LW, LH = MX, 108, 240, 108
lc.rect(LX, LY, LW, LH, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.6)
lc.text(LX + 14, LY + 22, '阶段一 RUNNING 先行', 11, lc.C_TXT, 'start', True, maxw=LW - 28, tag='st1:t')
lc.text(LX + 14, LY + 42, '· 在途请求按追赶公式领 token', 8.5, '#334155', 'start', maxw=LW - 26, tag='st1:l1')
lc.text(LX + 14, LY + 58, '· allocate_slots 返 None → FCFS 抢队尾', 8.5, '#334155', 'start', maxw=LW - 26, tag='st1:l2')
lc.text(LX + 14, LY + 74, '· 抢过的名字记进 preempted_reqs', 8.5, '#334155', 'start', maxw=LW - 26, tag='st1:l3')
lc.text(LX + 14, LY + 94, '（抢占环细节 → ch11）', 8, lc.C_FAINT, 'start', maxw=LW - 26, tag='st1:l4')

# 菱形 1：preempted_reqs 空？
D1_CX, D1_CY, D1_HW, D1_HH = 460, 162, 118, 46
diamond(D1_CX, D1_CY, D1_HW, D1_HH, '#ffffff', lc.C_ENG_S, 1.8)
dtext(D1_CX, D1_CY, ['preempted_reqs 空？', '（且 UNPAUSED）'], 9.5, 190)

# 菱形 2：num_running < max？
D2_CX, D2_CY, D2_HW, D2_HH = 460, 330, 118, 46
diamond(D2_CX, D2_CY, D2_HW, D2_HH, '#ffffff', lc.C_ENG_S, 1.8)
dtext(D2_CX, D2_CY, ['num_running <', 'max_num_running_reqs？'], 9.5, 200)

# 三态
GATE_CLOSED = (900, 118, 460, 118)     # 整拍不收新（闸关）
CAP_FULL = (900, 268, 460, 104)        # break 限容满
GATE_OPEN = (900, 418, 460, 118)       # 收新（闸开）

# 态 1：整拍不收新（暖）
gx, gy, gw, gh = GATE_CLOSED
lc.rect(gx, gy, gw, gh, '#fff7ed', lc.C_ABORT, rx=8, sw=1.9)
lc.text(gx + 14, gy + 22, '闸 ① 关——整拍不收新', 11, lc.C_ABORT, 'start', True, maxw=gw - 150, tag='gc:t')
lc.text(gx + 14, gy + 42, '「刚抢过人」本身就是内存紧张信号：再放新人', 8.5, '#334155', 'start', maxw=gw - 28, tag='gc:l1')
lc.text(gx + 14, gy + 58, '进来等于制造下一轮让座——守卫作用域恰一拍，', 8.5, '#334155', 'start', maxw=gw - 28, tag='gc:l2')
lc.text(gx + 14, gy + 74, '下一拍 preempted_reqs 重新初始化为空、恢复检票', 8.5, '#334155', 'start', maxw=gw - 28, tag='gc:l3')
lc.text(gx + 14, gy + 96, '实测 a·拍2：r3 连一次 allocate 询问都没有——', 8.5, lc.C_ABORT, 'start', True, maxw=gw - 28, tag='gc:l4')
lc.text(gx + 14, gy + 112 - 5, '尽管空闲 1 块、预算剩 2047（实调序列里只有 r1 的两条记录）', 8, lc.C_MUTE, 'start', maxw=gw - 28, tag='gc:l5')

# 态 2：break 限容满（暖）
bx, by, bw, bh = CAP_FULL
lc.rect(bx, by, bw, bh, '#ffffff', lc.C_ABORT, rx=8, sw=1.6, dash=True)
lc.text(bx + 14, by + 22, '闸 ② 满——break（限容）', 11, lc.C_ABORT, 'start', True, maxw=bw - 28, tag='cf:t')
lc.text(bx + 14, by + 42, '在座人数到顶：len(running) ≥ max_num_running_reqs', 8.5, '#334155', 'start', maxw=bw - 28, tag='cf:l1')
lc.text(bx + 14, by + 58, '门口队伍再长也不放——腾位才收（断言 L1113 兜底）', 8.5, '#334155', 'start', maxw=bw - 28, tag='cf:l2')
lc.text(bx + 14, by + 82, '实测 b·拍1-2：cap=2 下 num_running 2/2，r3 等两拍；', 8.5, lc.C_ABORT, 'start', True, maxw=bw - 28, tag='cf:l3')
lc.text(bx + 14, by + 96, '拍 3 r1 退场（⑤ 拍路径）→ 1 < 2 → r3 进批 16 全量', 8, lc.C_MUTE, 'start', maxw=bw - 28, tag='cf:l4')

# 态 3：收新（闸开，青）
ox, oy, ow, oh = GATE_OPEN
lc.rect(ox, oy, ow, oh, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.8)
lc.text(ox + 14, oy + 22, '收新——闸开（队头先问）', 11, lc.C_KV_S, 'start', True, maxw=ow - 28, tag='go:t')
lc.text(ox + 14, oy + 42, 'pop 队头 → 切块三闸 → allocate_slots 准入 → 入 running', 8.5, '#334155', 'start', maxw=ow - 28, tag='go:l1')
lc.text(ox + 14, oy + 58, '被抢者 prepend 回队头：老请求先恢复、压住后到者', 8.5, '#334155', 'start', maxw=ow - 28, tag='go:l2')
lc.text(ox + 14, oy + 80, '实测 a·拍5：r2 以 resumed 重入 33（队头优先）', 8.5, lc.C_KV_S, 'start', True, maxw=ow - 28, tag='go:l3')
lc.text(ox + 14, oy + 94, '拍 3-4 waiting 序 [r2, r3]：r2 需 3 块 > 空闲 1 → break，r3 陪等', 8, lc.C_MUTE, 'start', maxw=ow - 28, tag='go:l4')

# ---------------- 边 ----------------
# 阶段一 → 菱形1
lc.seg(LX + LW, LY + LH / 2, D1_CX - D1_HW, D1_CY, lc.C_KV_S, 1.8, 'std')
lc.text(LX + LW + 8, LY + LH / 2 - 8, '阶段二入口', 8.5, lc.C_KV_S, 'start', maxw=80, tag='e:in')
# 菱形1 否 → 闸关
lc.parrow([(D1_CX + D1_HW, D1_CY), (gx - 4, D1_CY), (gx - 4, gy + gh / 2), (gx, gy + gh / 2)],
          lc.C_ABORT, 1.8, 'ab')
lc.text((D1_CX + D1_HW + gx) / 2, D1_CY - 8, '否（本拍抢过）', 8.5, lc.C_ABORT, 'middle', True, maxw=100, tag='e:no1')
# 菱形1 是 → 菱形2
lc.seg(D1_CX, D1_CY + D1_HH, D2_CX, D2_CY - D2_HH, lc.C_ENG_S, 1.8, 'std')
lc.text(D1_CX + 8, (D1_CY + D1_HH + D2_CY - D2_HH) / 2, '是（没抢过）', 8.5, lc.C_ENG_S, 'start', maxw=90, tag='e:yes1')
# 菱形2 否 → 限容 break
lc.parrow([(D2_CX + D2_HW, D2_CY), (bx - 4, D2_CY), (bx - 4, by + bh / 2), (bx, by + bh / 2)],
          lc.C_ABORT, 1.8, 'ab')
lc.text((D2_CX + D2_HW + bx) / 2, D2_CY - 8, '否（在座满）', 8.5, lc.C_ABORT, 'middle', True, maxw=100, tag='e:no2')
# 菱形2 是 → 闸开（单折线：下 → 右 → 落闸开框顶）
lc.parrow([(D2_CX, D2_CY + D2_HH), (D2_CX, oy - 4), (ox + 60, oy - 4), (ox + 60, oy)],
          lc.C_KV_S, 1.8, 'std')
lc.text(D2_CX + 8, D2_CY + D2_HH + 14, '是（还有座）', 8.5, lc.C_KV_S, 'start', maxw=90, tag='e:yes2')

# PAUSED_ALL 小注（总闸亲戚）
PZ = (900, 556, 460, 44)
px, py, pw, ph = PZ
lc.rect(px, py, pw, ph, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(px + 14, py + 18, '第三态亲戚：PAUSED_ALL → token_budget = 0（L460-L462）', 8.5, lc.C_MUTE,
        'start', maxw=pw - 28, tag='pz:l1')
lc.text(px + 14, py + 34, '两个入环条件同时失效，一拍空转返回（弹性场景 → ch39）', 8, lc.C_MUTE,
        'start', maxw=pw - 28, tag='pz:l2')

# ---------------- cap 地形小注（左下） ----------------
CAP_Y = 560
lc.rect(MX, CAP_Y, 240, 40, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 12, CAP_Y + 17, 'cap 默认 128（DEFAULT_MAX_NUM_SEQS，', 8, lc.C_MUTE, 'start', maxw=220, tag='cap:l1')
lc.text(MX + 12, CAP_Y + 32, 'config/scheduler.py:L44）· 服务端 256/1024', 8, lc.C_MUTE, 'start', maxw=220, tag='cap:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 640
lx = MX
items = [
    ('dec', '判定菱形（两道闸的问句）'),
    ('cls', '闸关 / 闸满（收新被挡）'),
    ('opn', '闸开（收新）'),
]
for kind, name in items:
    if kind == 'dec':
        diamond(lx + 11, LEG_Y - 3, 11, 8, '#ffffff', lc.C_ENG_S, 1.4)
        lx += 4
    elif kind == 'cls':
        lc.rect(lx, LEG_Y - 10, 20, 13, '#fff7ed', lc.C_ABORT, rx=4, sw=1.4)
    else:
        lc.rect(lx, LEG_Y - 10, 20, 13, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.4)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=280, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 22

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/core/sched/scheduler.py:L683-L692（守卫+限容 break）· L194（preempted_reqs 每拍重置）· L1113（len(running) ≤ cap 断言）· '
        'vllm/config/scheduler.py:L44（DEFAULT_MAX_NUM_SEQS = 128）', 8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '两场景读数取自精简版 companion host 实测（r1 退场为驱动侧手工模拟 ⑤ 拍完成路径）· 服务端 256/1024 见 arg_utils.py:L2541-L2563 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-admission-gate-and-cap.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
