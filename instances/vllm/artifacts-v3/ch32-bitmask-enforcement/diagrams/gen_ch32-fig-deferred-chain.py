#!/usr/bin/env python3
"""ch32 机制图 · async 心跳下的 deferred 兑现链（figure_spec ch32-fig-deferred-chain，模板 swimlane）

放大自 L0 循环框③④两拍在异步调度（v0.27 默认心跳）下的展开——批队列两批重叠、掩码与
采样被 defer 一拍（L2 站 15 的时序放大），架构归属回指 L2 章图，不另立第二种架构画法
（FIGURE-SYSTEM §3）。三拍按事件账纵排（时间自上而下=拍序、拍内自左而右=代码序），
CPU/调度列恒 C_ENG_S 橙、GPU/worker 恒 C_GPU_S 绿、结构化输出动作恒 C_SAM_S 品红。

claim：pending 置位当拍不采样：掩码要吃上拍真实 token——deferred 批挂起，先收上批输出、
草稿 D2H 回调度器过滤补齐（take_draft→update_draft→bitmask→sample_tokens），米到了才开火。

数字全部取自 figure_spec.numbers（占位信号两拍：进门 0 → False、加账 +1+2=3；进门 3 →
True；拍 2 事件账 schedule → dispatch(non_block) → update_from_output(拍1批) →
take_draft_token_ids → update_draft_in_output → bitmask(deferred) → sample_tokens；
拍 1 立即路径对照；草稿过滤物证 [a,b,999,77777,a] → [64,65,-1,-1]）。
布局对齐图注（figure-integration 评审阻断项，2026-09-18 修）：草稿过滤物证挂在拍 2
兑现链中段（update_draft_in_output chip 正下方，即图注「兑现链中段」的落点）；
底部恰好两注 = 图注点名的「占位信号两拍账」与「deficit 回填公式」（队列不变式
并入拍 1 立即路径脚注）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1580, 930   # 内容收口到脚注下方（旧 1210 高有 ~320px 尾部空白，2026-09-18 顺手修）
MX, BXR = 60, 1540

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>'
    f'<marker id="gpu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '异步心跳下掩码的因果时序：pending 当拍不采样——草稿先 D2H 回来过滤补齐，米到了才开火', 16.5,
        lc.C_TXT, 'start', True, maxw=1230, tag='title')
lc.text(MX, 58, 'batch_queue_size=2：调度器提前一拍组批（GPU 不等 CPU），但结构化输出的表要吃『上拍真实 token』——'
               '上拍输出没回来，pending_structured_output_tokens 置位、采样整拍挂起；稳态下每拍都 pending',
        10.5, lc.C_MUTE, 'start', maxw=1370, tag='subtitle')
_ch = '放大自 L0 循环框③④两拍·异步调度 · L2 站 15 的时序放大'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道头 ----------------
HDR_Y, HDR_H = 88, 34
LANES = [
    (480, 'CPU · EngineCore 调度列', '事件账（拍内自左而右 = 代码序）', lc.C_ENG_F, lc.C_ENG_S),
    (1240, 'GPU · worker', '前向活动条（批 A / 批 B 错位重叠）', lc.C_GPU_F, lc.C_GPU_S),
]
for cx, nm, sub, f_, s_ in LANES:
    lc.rect(cx - 160, HDR_Y, 320, HDR_H, f_, s_, rx=7, sw=1.8)
    lc.text(cx, HDR_Y + 15, nm, 10.5, s_, 'middle', True, maxw=310, tag='lane:' + nm[:6])
    lc.text(cx, HDR_Y + 29, sub, 7.8, lc.C_MUTE, 'middle', maxw=310, tag='lane:s' + nm[:6])

# CPU 侧事件 chip 区自 x=210 起；GPU 前向条区 x=1120..1450
CPU_X0, CPU_X1 = 210, 1080
GPU_X0, GPU_X1 = 1130, 1450


def chip(x, y, text, stroke, deferred=False):
    w = lc.tw(text, 8.2, True) + 22
    lc.rect(x, y - 14, w, 28, '#ffffff', stroke, rx=13, sw=1.4)
    lc.text(x + w / 2, y + 3.5, text, 8.2, lc.C_TXT, 'middle', True, maxw=w - 8, tag='ch:' + text[:10])
    if deferred:
        dw = 18 + 7.6 * 5
        lc.rect(x + w - 4, y - 26, dw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
        lc.text(x + w - 4 + dw / 2, y - 14, 'defer', 7.4, lc.C_ENG_S, 'middle', True, maxw=dw - 4,
                tag='dfb:' + text[:8])
    return w


def flow_row(y, events):
    """一行事件链：events = [(text, color, deferred)]；返回 {text: (cx, right)}。"""
    pos = {}
    x = CPU_X0
    for i, (t_, c_, d_) in enumerate(events):
        w = chip(x, y, t_, c_, d_)
        pos[t_] = (x + w / 2, x + w)
        if i < len(events) - 1:
            lc.seg(x + w + 2, y, x + w + 14, y, lc.C_MUTE, 1.4, 'std')
        x += w + 16
    return pos


# ================= 拍 1（立即路径对照） =================
B1_Y, B1_H = 150, 148
lc.seg(MX, B1_Y, BXR, B1_Y, '#cbd5e1', 1.0, dash=True)
lc.rect(MX, B1_Y + 4, 120, 46, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.4)
lc.text(MX + 60, B1_Y + 24, '拍 1', 11, lc.C_GPU_S, 'middle', True, maxw=60, tag='b1')
lc.text(MX + 60, B1_Y + 40, 'pending=False', 7.6, lc.C_GPU_S, 'middle', True, maxw=100, tag='b1p')
lc.text(MX + 8, B1_Y + 68, '进门 placeholders=0 → 不置位', 7.8, lc.C_MUTE, 'start', maxw=130,
        tag='b1:s1')
lc.text(MX + 8, B1_Y + 84, '加账 +1+2=3（AR bonus 1', 7.8, lc.C_MUTE, 'start', maxw=130, tag='b1:s2')
lc.text(MX + 8, B1_Y + 98, '+ spec 2）', 7.8, lc.C_MUTE, 'start', maxw=130, tag='b1:s3')
RY1 = B1_Y + 56
p1 = flow_row(RY1, [('schedule', lc.C_ENG_S, False), ('dispatch(non_block)', lc.C_ENG_S, False),
                    ('bitmask(immediate)', lc.C_SAM_S, False),
                    ('sample_tokens(non_block=True)', lc.C_SAM_S, False),
                    ('入队早退（队 1<2 未满）', lc.C_MUTE, False)])
lc.text(CPU_X0, RY1 + 34, '立即路径：本拍历史完整 → 当拍就算表、当拍采样；队列不变式：填管道优先于收输出'
                        '（队未满即早退）——首个 deferred 拍必有上一批可 pop',
        8, lc.C_MUTE, 'start', maxw=870, tag='b1:foot')
# GPU 前向条：批 A（拍 1 起跑）
G1Y = B1_Y + 26
lc.rect(GPU_X0, G1Y, GPU_X1 - GPU_X0, 92, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text((GPU_X0 + GPU_X1) / 2, G1Y + 24, '批 A · 前向', 9.4, lc.C_GPU_S, 'middle', True,
        maxw=200, tag='ga:t')
lc.text((GPU_X0 + GPU_X1) / 2, G1Y + 42, '拍 1 组批、拍 1 发车', 7.8, lc.C_MUTE, 'middle',
        maxw=240, tag='ga:s')
# dispatch chip 顶边绕行 → GPU 条左缘
lc.parrow([(p1['dispatch(non_block)'][0], RY1 - 14), (p1['dispatch(non_block)'][0], RY1 - 26),
           (GPU_X0 - 4, RY1 - 26)], lc.C_ENG_S, 1.6, 'gpu')

# ================= 拍 2（deferred 兑现链） =================
B2_Y = B1_Y + B1_H + 8
B2_H = 262   # 两行事件链 + 草稿过滤物证条 + 链脚注（物证自底部注上移进带内）
lc.seg(MX, B2_Y, BXR, B2_Y, '#cbd5e1', 1.0, dash=True)
lc.rect(MX, B2_Y + 4, 120, 46, lc.C_BADGE_F, lc.C_ENG_S, rx=7, sw=1.4)
lc.text(MX + 60, B2_Y + 24, '拍 2', 11, lc.C_ENG_S, 'middle', True, maxw=60, tag='b2')
lc.text(MX + 60, B2_Y + 40, 'pending=True', 7.6, lc.C_ENG_S, 'middle', True, maxw=100, tag='b2p')
lc.text(MX + 8, B2_Y + 68, '进门 3>0（上拍 token', 7.8, lc.C_ABORT, 'start', maxw=130, tag='b2:s1')
lc.text(MX + 8, B2_Y + 82, '没回来）→ 本拍挂起', 7.8, lc.C_ABORT, 'start', maxw=130, tag='b2:s2')
lc.text(MX + 8, B2_Y + 98, '加账 +1+2 → 6', 7.8, lc.C_MUTE, 'start', maxw=130, tag='b2:s3')
RY2 = B2_Y + 46
p2 = flow_row(RY2, [('schedule', lc.C_ENG_S, False), ('dispatch(non_block)', lc.C_ENG_S, False),
                    ('update_from_output(拍1批)', lc.C_ENG_S, False)])
RY3 = RY2 + 46
p3 = flow_row(RY3, [('take_draft_token_ids', lc.C_GPU_S, False),
                    ('update_draft_in_output', lc.C_ENG_S, False),
                    ('bitmask(deferred)', lc.C_SAM_S, True),
                    ('sample_tokens(non_block=True)', lc.C_SAM_S, True),
                    ('重新入队', lc.C_MUTE, False)])
# 草稿过滤物证：兑现链中段证据条——挂在 update_draft_in_output chip 正下方
# （图注「兑现链中段」的落点；自底部注上移进拍 2 带内，底部只留图注点名的两注）
EV_X, EV_W = CPU_X0, 800
EV_Y, EV_H = RY3 + 30, 104
lc.rect(EV_X, EV_Y, EV_W, EV_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.2)
lc.text(EV_X + 16, EV_Y + 18, '草稿过滤物证（兑现链中段，真 validate）', 9.5, lc.C_TXT,
        'start', True, maxw=EV_W - 32, tag='ev:t')
for j, ln in enumerate(['回传草稿 [a, b, 999, 77777, a] 先截断到已排 4 → 真 validate 前缀过滤：'
                        'a,b 合法保留、999 起断链（77777 连带）',
                        '→ sched_spec = [64, 65, -1, -1]（-1 补齐保长度）· num_invalid={r1:2} 记账',
                        '掩码窗口把 -1 当『停约束停推进』处理——非法草稿没有通路进窗口']):
    lc.text(EV_X + 16, EV_Y + 38 + j * 17, ln, 8.2, '#334155', 'start', maxw=EV_W - 30,
            tag='ev:l' + str(j))
lc.parrow([(p3['update_draft_in_output'][0], RY3 + 14),
           (p3['update_draft_in_output'][0], RY3 + 30)], lc.C_ENG_S, 1.4, 'std')
lc.text(CPU_X0, EV_Y + EV_H + 22, '兑现链（因果序不可换）：先收上批输出 → 草稿 D2H 回调度器'
                               ' → validate 过滤 + (-1) 补齐 → 才算表 → 补采样重新入队',
        8, lc.C_ENG_S, 'start', maxw=880, tag='b2:foot')
# GPU 前向条：批 B（与批 A 错位重叠）
G2Y = B2_Y + 22
lc.rect(GPU_X0, G2Y, GPU_X1 - GPU_X0, 92, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text((GPU_X0 + GPU_X1) / 2, G2Y + 24, '批 B · 前向', 9.4, lc.C_GPU_S, 'middle', True,
        maxw=200, tag='gb:t')
lc.text((GPU_X0 + GPU_X1) / 2, G2Y + 42, '拍 2 组批发车——批 A 还在队里等采样兑现', 7.6,
        lc.C_MUTE, 'middle', maxw=300, tag='gb:s')
lc.parrow([(p2['dispatch(non_block)'][0], RY2 - 14), (p2['dispatch(non_block)'][0], RY2 - 26),
           (GPU_X0 - 4, RY2 - 26)], lc.C_ENG_S, 1.6, 'gpu')
# D2H 草稿回程：GPU 条左缘 →（行间走廊）→ take_draft chip 顶边
d2h_y = (RY2 + RY3) / 2
lc.parrow([(GPU_X0 - 4, d2h_y), (p3['take_draft_token_ids'][0], d2h_y),
           (p3['take_draft_token_ids'][0], RY3 - 14)], lc.C_GPU_S, 1.8, 'sam', dash=True)
lc.text(1020, d2h_y - 6, '草稿 D2H 回程', 8, lc.C_GPU_S, 'middle', True, maxw=120, tag='d2h')

# ================= 拍 3（同一兑现链再走一轮） =================
B3_Y = B2_Y + B2_H + 8
B3_H = 140
lc.seg(MX, B3_Y, BXR, B3_Y, '#cbd5e1', 1.0, dash=True)
lc.rect(MX, B3_Y + 4, 120, 46, lc.C_BADGE_F, lc.C_ENG_S, rx=7, sw=1.4)
lc.text(MX + 60, B3_Y + 24, '拍 3', 11, lc.C_ENG_S, 'middle', True, maxw=60, tag='b3')
lc.text(MX + 60, B3_Y + 40, 'pending=True', 7.6, lc.C_ENG_S, 'middle', True, maxw=100, tag='b3p')
lc.text(MX + 8, B3_Y + 68, '进门 6 → 仍挂起', 7.8, lc.C_MUTE, 'start', maxw=130, tag='b3:s1')
lc.text(MX + 8, B3_Y + 84, '（稳态常态）', 7.8, lc.C_MUTE, 'start', maxw=130, tag='b3:s2')
RY4 = B3_Y + 46
p4 = flow_row(RY4, [('schedule', lc.C_ENG_S, False), ('dispatch(non_block)', lc.C_ENG_S, False),
                    ('update_from_output(拍2批)', lc.C_ENG_S, False),
                    ('take_draft_token_ids', lc.C_GPU_S, False),
                    ('update_draft_in_output', lc.C_ENG_S, False)])
lc.text(CPU_X0, RY4 + 34, '同一兑现链再走一轮（pop 拍 2 批收输出 → …）——稳态流水线里 deferred 是常态而非例外',
        8, lc.C_MUTE, 'start', maxw=880, tag='b3:foot')
G3Y = B3_Y + 22
lc.rect(GPU_X0, G3Y, GPU_X1 - GPU_X0, 64, '#ffffff', lc.C_GPU_S, rx=8, sw=1.4, dash=True)
lc.text((GPU_X0 + GPU_X1) / 2, G3Y + 26, '下一批 · 前向（同构）', 8.4, lc.C_GPU_S, 'middle', True,
        maxw=260, tag='gc:t')
lc.text((GPU_X0 + GPU_X1) / 2, G3Y + 44, '拍 3 发车进 GPU', 7.6, lc.C_MUTE, 'middle', maxw=240,
        tag='gc:s')
lc.parrow([(p4['dispatch(non_block)'][0], RY4 - 14), (p4['dispatch(non_block)'][0], RY4 - 26),
           (GPU_X0 - 4, RY4 - 26)], lc.C_ENG_S, 1.6, 'gpu')

# ================= 底部两注（与图注逐字对齐：「占位信号两拍账」与「deficit 回填公式」） =================
# 队列不变式已并入拍 1 立即路径脚注；草稿过滤物证上移进拍 2 兑现链中段——底部只留
# 图注点名的两注（figure-integration 评审阻断项，2026-09-18 修）。
NY, NH = B3_Y + B3_H + 14, 104
lc.rect(MX, NY, 720, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 16, NY + 20, '占位信号两拍账（真 _update_after_schedule）', 9.5, lc.C_TXT,
        'start', True, maxw=680, tag='n1:t')
for j, ln in enumerate(['· 判定在加账之前：拍 1 进门 0 → 不置位；加账 +1（AR bonus）+2（spec）=3',
                        '· 拍 2 进门 3>0 → 置位挂起；拍 3 进门 6 仍挂起——稳态每拍都 pending']):
    lc.text(MX + 16, NY + 42 + j * 18, ln, 8.2, '#334155', 'start', maxw=690, tag='n1:l' + str(j))
lc.rect(800, NY, BXR - 800, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(816, NY + 20, 'deficit 回填公式（真实调度公式）', 9.5, lc.C_TXT, 'start', True,
        maxw=660, tag='n2:t')
for j, ln in enumerate(['· 拍 2 置位挂起仍排 2 个 token = deficit 回填',
                        '· num_tokens_with_spec + placeholders - computed',
                        '· 排不够会把请求误判回 prefill chunk']):
    lc.text(816, NY + 42 + j * 18, ln, 8.2, '#334155', 'start', maxw=690, tag='n2:l' + str(j))

# ================= 图例 + 页脚 =================
LY = NY + NH + 28
lx0 = MX
for c_, name in [(lc.C_ENG_S, 'CPU/调度事件（橙）'), (lc.C_GPU_S, 'GPU/回程事件（绿）'),
                 (lc.C_SAM_S, '结构化输出动作（品红）')]:
    lc.rect(lx0, LY - 9, 16, 11, '#ffffff', c_, rx=13, sw=1.3)
    lc.text(lx0 + 21, LY + 1, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='lg:' + name[:5])
    lx0 += 21 + lc.tw(name, 8.8) + 14
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_ENG_S, 1.8, 'gpu')
lc.text(lx0 + 40, LY + 1, '发车（CPU→GPU）', 8.8, lc.C_TXT, 'start', maxw=160, tag='lg4')
lx0 += 40 + lc.tw('发车（CPU→GPU）', 8.8) + 14
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_GPU_S, 1.8, 'sam', dash=True)
lc.text(lx0 + 40, LY + 1, '草稿 D2H 回程（GPU→CPU）', 8.8, lc.C_TXT, 'start', maxw=230, tag='lg5')
lx0 += 40 + lc.tw('草稿 D2H 回程（GPU→CPU）', 8.8) + 14
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, '#cbd5e1', 1.0, dash=True)
lc.text(lx0 + 40, LY + 1, '拍界栅格（时间自上而下）', 8.8, lc.C_TXT, 'start', maxw=210, tag='lg6')

lc.text(MX, LY + 28, 'vllm/v1/engine/core.py:L719-L737（step_with_batch_queue：deferred 兑现链）· core.py:L665-L677（分流）'
                     '· async_scheduler.py:L31-L44（pending 置位）· scheduler.py:L2168-L2202（update_draft_token_ids_in_output）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LY + 46, '三拍事件账 / 占位信号 0→3→6 / 草稿过滤 [a,b,999,77777,a]→[64,65,-1,-1] ＝ 本章驱动脚本实测'
                     '（spy executor/scheduler 承载 EngineCore 消费面，被驱动代码为实现树真代码）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-deferred-chain.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
