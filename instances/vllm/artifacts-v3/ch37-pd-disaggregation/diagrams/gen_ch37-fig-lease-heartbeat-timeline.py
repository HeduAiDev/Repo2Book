#!/usr/bin/env python3
"""ch37 机制图 m8 · lease-heartbeat-timeline（figure_spec ch37-fig-lease-heartbeat-timeline，模板 state-table·时间轴）

放大自 L0「双实例+KV 边界」中排⑧ 租约心跳与过期（D 等待期与 P 钉住期的对偶）、L2 章图站 10。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：三个数撑起自愈：租约 30s（P 钉住块等多久）、心跳间隔 5s（D 用什么证明还活着）、
续租 20s（每声心跳把到期推多远）——到期时刻单调不减、D 失联最多 30s 后 P 自动放块。

画法：同一时间轴两个场景面板（A=D 活着：游标只右移；B=D 失联：游标停在最高位 → 到期收割）。
到期游标 = 到期时刻在时间轴上的位置；每声心跳后到期 = max(旧, now+20)（只增不减）。

数字全部取自 figure_spec.numbers（traces 实测 + pin 源码锚点，逐字核对）：
  · 租约 30s / 心跳间隔 5s（30//6）/ 续租 20s（30×2//3）：每租约 6 次心跳、续期速率 4 倍消耗
  · 跨进程重基准：remaining = now_local + (deadline − scheduler_clock)；示例 99000/100000 → 1000s（实测 1000.0s，区间 (990,1010)）
  · 真 HB notif 续租：旧租约剩 1.0s → 到期后移 19.0s（新到期=now+20）；第二条心跳位移 +0.01s、max() 只增不减
  · 节流：间隔内第二份 meta 不带心跳；传输完成即停跳
  · 到期收割：过期即强制 done_sending 放块（D 失联自愈）；账本按到期非降序插入支撑早退扫描
  · #50326 反面教材：不重基准时跨机 perf_counter 纪元差大于 TTL → READ 到达瞬间判过期
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1600
MX = 54
BXR = 1546

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'P 的显存靠一只单调不减的到期游标自愈：租约 30s · 心跳 5s · 续租 20s', 16.5,
        lc.C_TXT, 'start', True, maxw=1280, tag='title')
lc.text(MX, 60, '押金单（租约）+ 续押（心跳）+ 没收（到期收割）：D 活着游标只右移，D 失联游标停在最高位、最多 30s 后自动放块',
        10.5, lc.C_MUTE, 'start', maxw=1200, tag='subtitle')
_ch = '放大自 L2 站 10（⑧ 租约心跳与过期）· L0：双实例+KV 边界中排⑧'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# 常数条
CN_Y = 84
lc.rect(MX, CN_Y, BXR - MX, 44, lc.C_ENG_F, lc.C_ENG_S, rx=8, sw=1.2)
lc.text((MX + BXR) / 2, CN_Y + 18, '三个数：租约 30s（P 钉住块等多久）· 心跳间隔 5s = 30//6（D 用什么证明还活着）· '
        '续租 20s = 30×2//3（每声把到期推多远）',
        10, lc.C_TXT, 'middle', True, maxw=BXR - MX - 30, tag='cn:1')
lc.text((MX + BXR) / 2, CN_Y + 34, '每租约 6 次心跳、续期速率 4 倍消耗——单条心跳丢失不致命',
        8.5, lc.C_MUTE, 'middle', maxw=BXR - MX - 30, tag='cn:2')

# ---------------- 时间轴两面板 ----------------
X0, K = 230, 24.0            # t(s) → x
T_END = 52


def tx(t):
    return X0 + t * K


PA_Y, PB_Y, PH = 152, 350, 170
AX_H = 0                     # 轴相对面板顶的深度（在 draw_panel 里定）


def draw_panel(py, label, label_color, hbs, cursor_marks, notes, exit_ev):
    """一个场景面板：左侧标签 + 时间轴 + 交接起桩 + HB 刻度 + 到期游标 + 出口。"""
    ax_y = py + 96                       # 时间轴 y
    lc.rect(MX, py, BXR - MX, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.1)
    lc.text(MX + 14, py + 20, label, 10.5, label_color, 'start', True, maxw=160,
            tag='pn:' + label[:6])
    # 时间轴 + 刻度
    lc.seg(tx(0), ax_y, tx(T_END), ax_y, lc.C_MUTE, 1.4)
    for t in range(0, T_END + 1, 10):
        lc.seg(tx(t), ax_y - 4, tx(t), ax_y + 4, lc.C_MUTE, 1.2)
        lc.text(tx(t), ax_y + 16, f'{t}s', 8, lc.C_MUTE, 'middle', tag='tick%d' % t)
    # 交接起桩（t=0，钉住 4 块）
    lc.seg(tx(0), ax_y, tx(0), ax_y - 26, lc.C_ENG_S, 2.0)
    lc.circle(tx(0), ax_y - 32, 4.5, lc.C_ENG_S, 1.4, dash=False)
    lc.text(tx(0) + 8, ax_y - 40, '交接：钉住 4 块 × 512 B · 起始到期 = t0+30s', 8.5,
            lc.C_ENG_S, 'start', maxw=330, tag='handoff')
    # 心跳刻度（向上的小竖线 + 顶点小圆）
    for t in hbs:
        lc.seg(tx(t), ax_y, tx(t), ax_y - 14, lc.C_ZMQ_S, 1.6)
        lc.circle(tx(t), ax_y - 18, 2.5, lc.C_ZMQ_S, 1.2, dash=False)
    if hbs:
        lc.text(tx(hbs[0]), ax_y + 30, 'HB:', 7.5, lc.C_ZMQ_S, 'middle', tag='hb:tag')
    # 到期游标（轴上的菱形标记 + 逐步右移的轨迹箭头）
    for i, (t, lab, solid) in enumerate(cursor_marks):
        d = 9
        s = (f'<path d="M{tx(t):.1f},{ax_y - d} L{tx(t) + d:.1f},{ax_y} '
             f'L{tx(t):.1f},{ax_y + d} L{tx(t) - d:.1f},{ax_y} Z" '
             f'fill="{lc.C_ENG_S if solid else "#ffffff"}" stroke="{lc.C_ENG_S}" stroke-width="1.6"/>')
        lc.ELEMS.append(((tx(t) - d - 2, ax_y - d - 2, tx(t) + d + 2, ax_y + d + 2), s))
        if i:
            tp = cursor_marks[i - 1][0]
            lc.seg(tx(tp) + 10, ax_y, tx(t) - 11, ax_y, lc.C_ENG_S, 1.2, 'std')
        if lab:
            lc.text(tx(t), ax_y - 26, lab, 8, lc.C_ENG_S, 'middle', maxw=150,
                    tag='cur:' + lab[:8])
    # 场内注记
    for (t, dy, s, color, anchor) in notes:
        if anchor == 'start':
            lc.text(tx(t), ax_y + dy, s, 8.5, color, 'start', maxw=600, tag='nt:' + s[:8])
        else:
            lc.text(tx(t), ax_y + dy, s, 8.5, color, 'middle', maxw=600, tag='nt:' + s[:8])
    # 出口事件（从轴向下/向上的箭头 + 标注）
    for (t, up, s1, s2, color) in exit_ev:
        y0, y1 = (ax_y - 6, ax_y - 46) if up else (ax_y + 6, ax_y + 46)
        lc.seg(tx(t), y0, tx(t), y1, color, 1.6, 'std', dash=True)
        for k, ln in enumerate((s1, s2)):
            lc.text(tx(t) + (10 if not up else 10), (y1 if up else y1) + 12 + k * 13 - (0 if up else 0),
                    ln, 8, color, 'start', maxw=330, tag='ex:' + ln[:8])


# 面板 A：D 活着
draw_panel(
    PA_Y, 'A · D 活着', lc.C_GPU_S,
    hbs=list(range(5, 50, 5)),
    cursor_marks=[(30, '到期=+30s', True), (35, None, False), (40, None, False),
                  (45, None, False), (50, '游标只右移', True)],
    notes=[
        (2, 52, 'HB 每声：新到期 = max(旧, now+20)——首两声（5s/10s）候选 25/30 不及旧值 30，max() 保旧：只增不减',
         lc.C_MUTE, 'start'),
        (31, -62, '此后每声净推 +5s：到期恒领先当前 ≥ 15s（丢一条心跳只损失 5s 预算，仍剩 ≥15s 等下一条）',
         lc.C_GPU_S, 'start'),
    ],
    exit_ev=[(40, False, '出口A · D 读完：READ-done notif 计数齐', '→ done_sending 放块（块回家）', lc.C_GPU_S)],
)

# 面板 B：D 失联
draw_panel(
    PB_Y, 'B · D 失联', lc.C_ABORT,
    hbs=[5, 10],
    cursor_marks=[(30, '游标停在最高位（30s）', True)],
    notes=[
        (13, -58, '心跳停发（D 崩溃）→ 无人续押', lc.C_ABORT, 'start'),
        (36, 40, '租约内无人来读 → 到期收割', lc.C_ABORT, 'start'),
    ],
    exit_ev=[(30, False, '出口B · 到期收割：get_finished 强制 done_sending={req-dead}',
              '账本删键——D 失联最多 30s 后 P 显存自愈（不用任何人来收）', lc.C_ABORT)],
)
# 面板 B 的「停发 ✕」标记
_ax_b = PB_Y + 96
lc.text(tx(15), _ax_b - 22, '✕', 11, lc.C_ABORT, 'middle', True, tag='stop:x')
lc.text(tx(15), _ax_b - 36, '停发', 8, lc.C_ABORT, 'middle', tag='stop:t')

# ---------------- 底部：重基准 / 续租实测 / 节流停跳 ----------------
BB_Y = PB_Y + PH + 18
BB_W = (BXR - MX - 2 * 16) / 3
boxes = [
    ('跨进程重基准（修 #50326）', lc.C_KV_S, [
        '· remaining = now_local',
        '  + (deadline − scheduler_clock)',
        '· 算例：scheduler_clock=99000 ·',
        '  截止=100000 → 1000s',
        '  （实测 1000.0s · 区间 (990,1010)）',
        '· perf_counter 跨进程不可比；',
        '  广播延迟只会拉长租约=安全方向',
        '· 反面教材：不重基准 → 纪元差',
        '  大于 TTL → READ 到达瞬间判过期',
    ], 'nixl/pull_worker.py:L94-L110'),
    ('续租实测（真 HB notif 过线）', lc.C_ENG_S, [
        '· D 发 ‘HB:req-1’、P 收到续租',
        '· 旧租约剩 1.0s → 到期后移 19.0s',
        '  （新到期 = now + 20）',
        '· 紧接着第二条心跳：位移仅 +0.01s',
        '  且 max() 保旧值——',
        '  乱序/重复心跳都不会把租约改短',
        '· P 只认 notif 清单里的请求续租',
    ], 'nixl/base_worker.py:L2189-L2208'),
    ('节流与停跳', lc.C_ZMQ_S, [
        '· build_meta 打包心跳：间隔内',
        '  第二份 meta 不带心跳（空表）',
        '· 心跳税按引擎分组、每 5s 至多一次',
        '· 传输完成（finished_recving）',
        '  即停跳——块已到手不再续押',
        '· 账本按到期非降序插入，',
        '  支撑 get_finished 按序早退扫描',
    ], 'base_scheduler.py:L459-L468 · L2145-L2164'),
]
for i, (t, color, lines, foot) in enumerate(boxes):
    x = MX + i * (BB_W + 16)
    h = 34 + len(lines) * 15.5 + 20
    lc.rect(x, BB_Y, BB_W, h, '#ffffff', color, rx=8, sw=1.4)
    lc.text(x + 14, BB_Y + 20, t, 10, color, 'start', True, maxw=BB_W - 28, tag='bb:t%d' % i)
    for j, ln in enumerate(lines):
        lc.text(x + 14, BB_Y + 42 + j * 15.5, ln, 8.5, '#334155', 'start', maxw=BB_W - 26,
                tag='bb:l%d_%d' % (i, j))
    lc.text(x + 14, BB_Y + h - 9, foot, 8, lc.C_FAINT, 'start', maxw=BB_W - 26, tag='bb:f%d' % i)

box_end = BB_Y + 34 + 9 * 15.5 + 20

# ---------------- 页脚 ----------------
FY = box_end + 20
lc.text(MX, FY, '逐字锚 nixl/base_scheduler.py:L70-L76（租约 30s · 间隔 =30//6）· base_worker.py:L277（续租 =30×2//3）· '
                'L2189-L2208（max(old, now+20) 只增不减）· L2145-L2164（到期收割）· pull_worker.py:L94-L110（重基准 · #50326 注释原文在源码）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 15, '行号基线 vLLM v0.27.1 · 游标=到期时刻在时间轴上的位置；面板 A/B 同一时间比例尺（1s = 24px）· '
                     '时钟偏移由握手 RTT 中点估计（同机实测 0.00028s）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 34

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch37-fig-lease-heartbeat-timeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
