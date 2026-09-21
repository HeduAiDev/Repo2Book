#!/usr/bin/env python3
"""ch34 机制图 · 草稿的所有权环：sync/async 双轨时序（figure_spec fig_m1_spec_cycle，模板 swimlane）

放大自 L0 主循环 loop_box + 采样列 spec 块的跨进程展开——L2 章图 center 拍片 ①-⑨ 的
『所有权泳道』重画：不重复九拍流水（L2 已画），专画 sync/async 双轨下草稿归谁持有。
时序图按 UML 文法（FIGURE-SYSTEM §0 硬规则 2）：参与者=竖直生命线（顶部名牌）、
共享时间轴（横向 gridlines）、消息=水平直线，禁折线/肘形。

claim：一个投机解码周期里草稿的所有权环：worker 产 k 个草稿 → 同步模式经
take_draft_token_ids 回 EngineCore 挂账、再随下一拍 SchedulerOutput 回流；异步模式
草稿全程留在 worker 进程（只在语法/惩罚/bad_words 需要时才拷回）→ target 一次前向
验证 → 记账回扣，环回下一轮。

数字全部取自 figure_spec.numbers（K=3；c1 产 2 token、c2 产 4 token；async 跨进程
往返 0 次；站 1→16＝traces/ch34_m02_scheduler_ledger.json + l2-specs/ch34.json 站账本）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 1032
MX, BXR = 56, 1444
DEFS = lc.DEFS

C_W, C_E = lc.C_GPU_S, lc.C_ENG_S          # worker=绿 / EngineCore=橙（角色色即身份）
LINE_W = 1.6

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一拍之内草稿的旅程：同步调度要过两次河，异步调度草稿根本不出 worker 进程', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '所有权泳道（K=3，eagle）· 跨进程往返：sync 2 次 / async 0 次 · 站号 = 一个投机周期的流经顺序（1-6 挂账排批 · 7-14 摊平+验证 · 15-16 回扣）',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 主循环+采样列 spec 块（L2 拍片①-⑨ 的所有权重画）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 双轨两个时序面板 =================
PANW = (BXR - MX - 36) / 2
PX = [MX, MX + PANW + 36]
PTY, PLH = 92, 824          # 面板顶 / 生命线高
LFX_OFF = PANW * 0.30       # worker 生命线 x 偏移
LFE_OFF = PANW * 0.78       # EngineCore 生命线 x 偏移

# 每轨事件（y 为拍位：t1..t8 等距）。消息=水平直线；自消息=骑线小框。
# (y_idx, kind, side, badge, head, tail)  side: 'w'=worker 线上自动作 'e'=EngineCore 线上自动作
#             msg: (from,to) 'w'/'e' 水平消息线 + label
TICKS = [PTY + 68 + i * 84 for i in range(8)]   # 8 个时间拍位
GRID_Y0, GRID_Y1 = PTY + 48, PTY + PLH - 16


def panel(px, title, sub):
    lc.rect(px, PTY, PANW, PLH, '#ffffff', lc.C_MUTE, rx=10, sw=1.2)
    lc.rect(px, PTY, PANW, 40, '#f8fafc', lc.C_MUTE, rx=10, sw=1.2)
    lc.seg(px, PTY + 40, px + PANW, PTY + 40, lc.C_MUTE, 1.2)
    lc.text(px + 14, PTY + 26, title, 12.5, lc.C_TXT, 'start', True, maxw=PANW - 28, tag='pt:' + title[:8])
    lc.text(px + 14, PTY + 36 - 2, sub, 8.8, lc.C_MUTE, 'start', maxw=PANW - 28, tag='ps:' + title[:8])


def lifelines(px):
    xs = (px + LFX_OFF, px + LFE_OFF)
    for x, name, col in ((xs[0], 'worker 进程（gpu_model_runner）', C_W),
                         (xs[1], 'EngineCore 进程（scheduler）', C_E)):
        lw_ = lc.tw(name, 10, True) + 22
        lc.rect(x - lw_ / 2, PTY + 46, lw_, 24, '#ffffff', col, rx=11, sw=1.5)
        lc.text(x, PTY + 62, name, 10, col, 'middle', True, maxw=lw_ - 8, tag='lf:' + name[:8])
        for gy in TICKS[1:]:      # 共享时间轴 gridlines（贯穿全部生命线）
            lc.seg(px + 12, gy, px + PANW - 12, gy, '#eef2f7', 1.0)
        lc.seg(x, PTY + 70, x, GRID_Y1, '#cbd5e1', 1.2)
    return xs


def selfbox(px, x, y, badge, head, lines, col, w=330):
    """骑线时刻标记：动作小框挂在生命线上（左侧或右侧）"""
    h = 30 + 14.5 * len(lines)
    bx = x - w / 2
    lc.rect(bx, y - 16, w, h, '#ffffff', col, rx=6, sw=1.2)
    if badge:
        bw = 13 + 8.2 * len(badge)
        lc.rect(bx + w - bw - 5, y - 13, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
        lc.text(bx + w - bw / 2 - 5, y, badge, 9, lc.C_ENG_S, 'middle', True, maxw=bw - 4, tag='bg' + badge)
    lc.text(bx + 8, y + 3, head, 9.8, lc.C_TXT, 'start', True, maxw=w - 16 - (46 if badge else 0), tag='sb' + head[:8])
    for i, ln in enumerate(lines):
        lc.text(bx + 8, y + 17.5 + i * 14.5, ln, 8.4, '#475569', 'start', maxw=w - 16, tag='sl' + head[:6] + str(i))
    return bx, y - 16, w, h


def msg(xs, yi, arrow_from, label, val, col, badge='', dash=False):
    """水平消息线：from 'w'→'e' 或 'e'→'w'，label 骑线"""
    y = TICKS[yi]
    x0, x1 = (xs[0], xs[1]) if arrow_from == 'w' else (xs[1], xs[0])
    mk = 'std'
    lc.seg(x0, y, x1, y, col, 2.0, mk, dash)
    mid = (x0 + x1) / 2
    gap = abs(x1 - x0) - 8
    lc.text(mid, y - 20, label, 9.2, col, 'middle', True, maxw=gap, tag='ml' + label[:10])
    if val:
        lc.text(mid, y - 8, val, 8.2, lc.C_MUTE, 'middle', maxw=gap, tag='mv' + label[:8])


# ---------------- 左轨：sync 同步调度 ----------------
panel(PX[0], 'sync 同步调度：草稿要过两次河', '跨进程往返 2 次（worker→EngineCore 挂账 → 随下一拍 SchedulerOutput 回 worker）')
xs0 = lifelines(PX[0])
# t0: 站1 worker 产草稿（自动作）
selfbox(PX[0], xs0[0], TICKS[0], '站 1', 'drafter.propose → 产 K=3 草稿',
        ['c1: [31,32,33] → c2: [41,42,43]', '_copy_draft_token_ids_to_cpu（copy stream 异步 D2H）'], C_W)
# t1: 站2-3 过河①
msg(xs0, 1, 'w', 'take_draft_token_ids → DraftTokenIds', '过河①：post_step 收草稿（core.py:L616-L623）', C_W, '站 2-3')
# t2: 站4 挂账（EngineCore 自动作）
selfbox(PX[0], xs0[1], TICKS[2], '站 4', 'update_draft_token_ids 挂账',
        ['spec_token_ids 挂 Request（跳过 finish/prefill chunk）', '语法请求先 validate_tokens 过滤'], C_E)
# t3: 站5-6 回流（EngineCore → worker）
msg(xs0, 3, 'e', '下一拍 SchedulerOutput 回流', '过河②：num_new_tokens=4（1 补喂+3 草稿）', C_E, '站 5-6')
# t4: 站7-14 worker 验证（自动作）
selfbox(PX[0], xs0[0], TICKS[4], '站 7-14', 'worker：摊平 → target 一次前向 → RejectionSampler',
        ['c1 输出行 [31,77,-1,-1]：接受 1、拒 2 → 产 2 token', 'c2 输出行 [41,42,43,99]：全收 3+bonus → 产 4 token'], C_W, w=430)
# t5: 站15 输出过河
msg(xs0, 5, 'w', 'parse_output → 变长 output_token_ids', 'c1 产 2 token / c2 产 4 token', C_W, '站 15')
# t6: 站16 记账回扣（EngineCore 自动作）
selfbox(PX[0], xs0[1], TICKS[6], '站 16', 'update_from_output 记账回扣',
        ['c1 拒 2：多记的 token 账退回', '（被拒两位退回『未计算』）'], C_E)
# t7: 环回（EngineCore → worker 虚线）
msg(xs0, 7, 'e', '环回下一轮：本步输出即下一轮 drafter 的输入', '', C_W, dash=True)

# ---------------- 右轨：async 异步调度 ----------------
panel(PX[1], 'async 异步调度：草稿不出门', '跨进程往返 0 次——草稿全程留在 worker 进程（站 3 注释原话：we update draft token ids in the worker process）')
xs1 = lifelines(PX[1])
selfbox(PX[1], xs1[0], TICKS[0], '站 1', 'drafter.propose → 产 K=3 草稿',
        ['采样尾部就起草：EAGLE 系吃 GPU 张量免等 bookkeeping'], C_W)
# t1: worker 自留（不画过河线——画一条被禁的手势？改为骑线注记）
selfbox(PX[1], xs1[0], TICKS[1] + 6, '站 2', '草稿留在 worker 进程（不过河）',
        ['只在语法/惩罚/bad_words 需要时才拷回（否则 0 拷贝）', '下一拍 _prepare_input_ids 直接 GPU scatter 进输入流'], C_W)
lc.text((xs1[0] + xs1[1]) / 2, TICKS[2] - 24, '跨进程往返 0 次', 9.2, lc.C_ABORT, 'middle', True,
        maxw=PANW - 40, tag='zero1')
lc.seg(xs1[0] + 60, TICKS[2] - 16, xs1[1] - 60, TICKS[2] - 16, lc.C_ABORT, 1.4, dash=True)
lc.text((xs1[0] + xs1[1]) / 2, TICKS[2] - 2, '（worker 与 EngineCore 之间无草稿消息）', 8.2, lc.C_MUTE,
        'middle', maxw=PANW - 40, tag='zero2')
# t3: 站7-14 worker 验证
selfbox(PX[1], xs1[0], TICKS[3], '站 7-14', 'worker：摊平 → target 前向 → 验证',
        ['async+语法走 deferred：先验草稿再算 bitmask'], C_W, w=400)
# t4: 站16 EngineCore 记账（异步只有账本消息、无草稿消息）
msg(xs1, 4, 'w', '输出记账（token 留 GPU，ch12）', '', lc.C_MUTE, dash=True)
selfbox(PX[1], xs1[1], TICKS[5], '站 16', '两本账同步共变',
        ['计算账 + 占位账一起回扣', '回扣守恒：不破坏每拍 ≥1 token'], C_E)
# t6: 环回（async 输出本就在 worker——内部闭环，不画跨进程消息）
selfbox(PX[1], xs1[0], TICKS[6], '站 1′', '环回下一轮：输出即 drafter 输入',
        ['worker 内部闭环——草稿与输出都没离开 worker'], C_W, w=430)

# ---------------- 底部对账条 ----------------
BY = PTY + PLH + 18
lc.rect(MX, BY, BXR - MX, 56, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, BY + 22, '对账：一个投机周期 = 站 1→16 的闭环——本步的输出就是下一轮草稿的原料；c1 一拍产 2 token（拒 2）、c2 一拍产 4 token（全收+bonus，一次前向的产出上限 k+1=4）。',
        9.6, lc.C_TXT, 'start', maxw=BXR - MX - 32, tag='ba1')
lc.text(MX + 16, BY + 42, 'sync 的两次过河换来了调度器对草稿的账面控制（挂账→排批→回扣全在 EngineCore）；async 把草稿留在 worker、省掉两次跨进程往返——代价是 EngineCore 只能靠占位账对齐。',
        8.6, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='ba2')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 26, 'vllm/v1/engine/core.py:L616-L623（post_step·sync 收草稿）· vllm/v1/worker/gpu_model_runner.py:L4889-L4941（take_draft_token_ids / 按需拷回）· '
                    'vllm/v1/core/sched/scheduler.py:L2146-L2167（挂账）· L516-L520（排批差账）· L1766-L1790（回扣）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 11, 'K=3 / 输出行 / 产出计数 ＝ 本章驱动脚本真跑实测（greedy kernel 零 RNG）· 站号 1-16 ＝ 本章 L2 站号账本 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m1_spec_cycle.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
