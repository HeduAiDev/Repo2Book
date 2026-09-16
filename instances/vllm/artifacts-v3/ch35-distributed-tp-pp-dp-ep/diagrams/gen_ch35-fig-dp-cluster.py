#!/usr/bin/env python3
"""ch34 机制图 · m18 DP 集群全景：数据面与控制面分家（figure_spec ch34-fig-dp-cluster）

放大自 L0 多实例视角全景（DPCoordinator·三 socket L2 站 7 + 北条前端进出/
按 client_index 回发）——本章头图候选。

claim：DP 集群另立 DPCoordinator 控制面进程持三 socket（back XPUB→引擎控制、
output PULL←引擎统计、front XPUB→前端聚合看板），100ms 变化刷/5s 心跳发布——
请求与输出的数据面（前端 ROUTER/DEALER 定向进、引擎按 client_index 选 PUSH 回）
完全不经过它。

数字全部取自 explainer figure_spec.numbers（traces/m18 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common（数据面 ZMQ 紫实线、
控制面 MUTE 灰虚线、coordinator 橙边虚框——进程边界外的第三类角色）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1560, 1004
MX = 64
BXR = 1496

DEFS = lc.DEFS.replace('</defs>',
                       '<marker id="pu" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="6.5" markerHeight="4.6" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
                       '</defs>')


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


def sock_chips(x, y, labels, stroke):
    """socket 徽标排（左对齐）。"""
    for lab in labels:
        w = lc.tw(lab, 8.5, True) + 14
        lc.rect(x, y, w, 21, '#ffffff', stroke, rx=8, sw=1.1)
        lc.text(x + w / 2, y + 15, lab, 8.5, stroke, 'middle', True, maxw=w - 2,
                tag='sc:' + lab[:8])
        x += w + 10


def num_tag(cx, cy, n, color):
    """骑线小圆号。"""
    lc.ELEMS.append(((cx - 12, cy - 12, cx + 12, cy + 12),
                     f'<circle cx="{cx}" cy="{cy}" r="11" fill="#ffffff" '
                     f'stroke="{color}" stroke-width="1.6"/>'))
    lc.text(cx, cy + 4, n, 10, color, 'middle', True, tag='nt' + n)


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'DP 集群全景：数据面与控制面彻底分家——请求/输出一个字节都不经过 DPCoordinator', 16.5,
        lc.C_TXT, 'start', True, maxw=1180, tag='title')
lc.text(MX, 62, '上排前端 ×N、下排引擎 ×M 用 ZMQ 紫实线直达（定向进 / 定向回）；右侧独立的 '
        'DPCoordinator 只收负载小票、播汇总看板与 wave 信令（灰虚线）', 10.5, lc.C_MUTE,
        'start', maxw=1280, tag='subtitle')
chip(BXR, 12, '放大自 L2 站 7 + 北条 · L0：多实例视角', lc.C_ENG_S)

# ---------------- 几何常量 ----------------
FR_Y, FR_H = 110, 140                    # 前端行
EN_Y, EN_H = 470, 160                    # 引擎行
BOX_W = 340
FB0X, FB1X = 130, 500
EB0X, EB1X = 130, 500
FE_X, FE_Y, FE_W, FE_H = 870, 170, 80, 60          # 前端省略框（×N）
EE_X, EE_Y, EE_W, EE_H = 860, 510, 90, 130         # 引擎省略框（×M，高到 640 给 ③⑤ 落边）
CO_X, CO_Y, CO_W, CO_H = 1150, 300, 346, 510       # coordinator
DN0, UP0, DN1, UP1 = 240, 330, 610, 700            # 数据面竖直箭头 x

# ---------------- 上排：前端 ----------------
for i, fx in enumerate((FB0X, FB1X)):
    lc.rect(fx, FR_Y, BOX_W, FR_H, lc.C_API_F, lc.C_API_S, rx=10, sw=2.0)
    lc.text(fx + 14, FR_Y + 26, '前端 %d · AsyncMPClient 家族' % i, 12.5, lc.C_API_S,
            'start', True, maxw=BOX_W - 28, tag='fe%d:t' % i)
    lc.text(fx + 14, FR_Y + 46, '内部 LB=DPLB（替全部引擎选路）· 外部 LB=DPAsync（绑定引擎）',
            9, '#334155', 'start', maxw=BOX_W - 26, tag='fe%d:l' % i)
    sock_chips(fx + 14, FR_Y + 62, ['ROUTER(bind)', 'PULL(bind)', 'XSUB(订看板)'],
               lc.C_API_S)
    lc.text(fx + 14, FR_Y + 118, '请求从此出、输出回到此（零 GPU）', 8.5, lc.C_MUTE,
            'start', maxw=BOX_W - 26, tag='fe%d:n' % i)
lc.rect(FE_X, FE_Y, FE_W, FE_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(FE_X + FE_W / 2, FE_Y + 26, '×N', 13, lc.C_MUTE, 'middle', True)
lc.text(FE_X + FE_W / 2, FE_Y + 46, '前端', 9, lc.C_MUTE, 'middle')

# ---------------- 下排：引擎 ----------------
for i, ex in enumerate((EB0X, EB1X)):
    lc.rect(ex, EN_Y, BOX_W, EN_H, lc.C_ENG_F, lc.C_ENG_S, rx=10, sw=2.0)
    lc.text(ex + 14, EN_Y + 26, '引擎 %d · EngineCore 进程（DP>1 形态）' % i, 12.5,
            lc.C_ENG_S, 'start', True, maxw=BOX_W - 40, tag='en%d:t' % i)
    lc.text(ex + 14, EN_Y + 46, 'MoE→DPEngineCoreProc（wave 锁步）· 非 MoE→完全独立',
            9, '#334155', 'start', maxw=BOX_W - 26, tag='en%d:l' % i)
    lc.rect(ex + 14, EN_Y + 60, BOX_W - 28, 28, lc.C_GPU_F, lc.C_GPU_S, rx=5, sw=1.3)
    lc.text(ex + BOX_W / 2, EN_Y + 79, 'TP × PP × PCP workers（GPU 执行臂）', 9.5,
            lc.C_GPU_S, 'middle', True, maxw=BOX_W - 40, tag='en%d:w' % i)
    sock_chips(ex + 14, EN_Y + 98, ['DEALER(connect)', 'PUSH(×每前端)', 'XSUB(订 wave)'],
               lc.C_ENG_S)
    lc.text(ex + 14, EN_Y + 146, 'identity = engine_index（2 字节小端）', 8.5, lc.C_MUTE,
            'start', maxw=BOX_W - 26, tag='en%d:n' % i)
lc.rect(EE_X, EE_Y, EE_W, EE_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(EE_X + EE_W / 2, EE_Y + 60, '×M', 13, lc.C_MUTE, 'middle', True)
lc.text(EE_X + EE_W / 2, EE_Y + 80, '引擎', 9, lc.C_MUTE, 'middle')

# ---------------- 右：DPCoordinator（橙边虚框=进程边界外第三类角色） ----------------
lc.rect(CO_X, CO_Y, CO_W, CO_H, '#ffffff', lc.C_ENG_S, rx=12, sw=2.2, dash=True)
lc.text(CO_X + 14, CO_Y + 26, 'DPCoordinator（独立进程）', 13, lc.C_ENG_S, 'start', True,
        maxw=CO_W - 130, tag='co:t')
lc.text(CO_X + CO_W - 12, CO_Y + 26, 'coordinator.py:L208-L256', 8.5, lc.C_FAINT, 'end',
        tag='co:f')
lc.text(CO_X + 14, CO_Y + 46, '只做控制面——不碰任何请求/输出数据', 9, '#334155',
        'start', maxw=CO_W - 26, tag='co:l')
# 三 socket（front 在上 / output 中 / back 下）
co_socks = [
    ('publish_front · XPUB(bind)', ['前端聚合看板：变化 100ms 刷 / 5s 心跳',
                                    'LB 打分的快照来源（waiting·running·kv）'], 356),
    ('output_back · PULL(bind)', ['收引擎负载小票——client_index=-1 哨兵路由',
                                  '不走任何前端 PUSH（core.py:L1788-L1793）'], 540),
    ('publish_back · XPUB(bind)', ['引擎控制：START_DP_WAVE 广播 / stale 补广播',
                                   '引擎以订阅帧 b"\\x01" 认领（XSUB）'], 690),
]
for title, lines, y in co_socks:
    lc.rect(CO_X + 12, y, CO_W - 24, 84, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.4)
    lc.text(CO_X + 26, y + 22, title, 11, lc.C_TXT, 'start', True, maxw=CO_W - 50,
            tag='cs:' + title[:8])
    for k, ln in enumerate(lines):
        lc.text(CO_X + 26, y + 42 + k * 17, ln, 9, '#334155', 'start', maxw=CO_W - 50,
                tag='cs:%s:l%d' % (title[:6], k))
# 中部实测注（publish_front 框下缘 440 与 output_back 框上缘 540 之间的空白带）
# 7 行须整块留在带内、且与上下框线各留 ≥10px：行距压到 11.2、首行基线上移到 459
# （首行顶 450.3 / 末行底 529.8），否则末行会落进 output_back 框、压住其标题。
NOTE_Y0, NOTE_LH = 459, 11.2
notes = [
    ('看板载荷（实测）：', True),
    ('([[2,1,0.6],[0,0,0.0]], 0, False)', False),
    ('= (每引擎 counts, current_wave, engines_running)', False),
    ('发布时延实测：冷 0.06s · 暖 0.11s', False),
    ('节拍：变化 100ms · 心跳 5000ms · lockstep ≥50ms', False),
    ('信令实测：START_DP_WAVE(wave0·exclude 引擎1)', False),
    ('wave_complete→(None,1,False)·stale 补(wave3·exclude 0)', False),
]
for k, (ln, bold) in enumerate(notes):
    lc.text(CO_X + 26, NOTE_Y0 + k * NOTE_LH, ln, 8.5, lc.C_TXT if bold else '#334155', 'start',
            True if bold else False, maxw=CO_W - 50, tag='cn%d' % k)

# ---------------- 数据面（紫实线）：定向进 / 定向回 ----------------
lc.seg(DN0, FR_Y + FR_H, DN0, EN_Y, lc.C_ZMQ_S, 2.6, 'pu')
lc.seg(UP0, EN_Y, UP0, FR_Y + FR_H, lc.C_ZMQ_S, 2.6, 'pu')
lc.seg(DN1, FR_Y + FR_H, DN1, EN_Y, lc.C_ZMQ_S, 2.6, 'pu')
lc.seg(UP1, EN_Y, UP1, FR_Y + FR_H, lc.C_ZMQ_S, 2.6, 'pu')
num_tag(DN0, 360, '①', lc.C_ZMQ_S)
num_tag(UP0, 360, '②', lc.C_ZMQ_S)

# ---------------- 控制面（灰虚线） ----------------
# ③ 引擎小票 → output_back（引擎省略框右缘 → PULL 左缘）
lc.seg(EE_X + EE_W, 560, CO_X, 560, lc.C_MUTE, 2.0, 'std', dash=True)
num_tag((EE_X + EE_W + CO_X) / 2, 560, '③', lc.C_MUTE)
# ④ publish_front → 前端 XSUB（前端省略框右缘）
lc.parrow([(CO_X, 398), (1090, 398), (1090, 200), (FE_X + FE_W, 200)], lc.C_MUTE,
          2.0, 'std', dash=True)
num_tag(1090, 300, '④', lc.C_MUTE)
# ⑤ publish_back → 引擎 XSUB（START_DP_WAVE 广播）
lc.parrow([(CO_X, 732), (1090, 732), (1090, 625), (EE_X + EE_W, 625)], lc.C_MUTE,
          2.0, 'std', dash=True)
num_tag(1105, 732, '⑤', lc.C_MUTE)

# ---------------- 引擎行下方：编号图例带 ----------------
LG_Y = 668
legend = [
    ('①', '数据面·请求定向进：前端 ROUTER→引擎 DEALER，首帧 = 引擎 identity 信封（ch5 埋的定向路在此兑现）',
     lc.C_ZMQ_S),
    ('②', '数据面·输出回发起前端：引擎按 client_index 选 PUSH，每前端一条——①② 同样存在于前端 1 与引擎 1',
     lc.C_ZMQ_S),
    ('③', '控制面·负载小票：引擎 counts 经 client_index=-1 哨兵拐进 coordinator——不走任何前端 PUSH',
     lc.C_MUTE),
    ('④', '控制面·聚合看板：100ms 变化刷 / 5s 心跳 → 前端 XSUB——LB 打分的快照来源', lc.C_MUTE),
    ('⑤', '控制面·START_DP_WAVE：back XPUB 广播唤醒全体（MoE wave 锁步；stale 上报触发补广播）',
     lc.C_MUTE),
]
for k, (n, s, color) in enumerate(legend):
    y = LG_Y + k * 26
    num_tag(MX + 14, y, n, color)
    lc.text(MX + 34, y + 4, s, 9.5, '#334155', 'start', maxw=1060, tag='lg%d' % k)

# ---------------- 底部结论条 ----------------
BY = 826
lc.rect(MX, BY, BXR - MX, 86, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, BY + 24, '图注点题：请求与输出一个字节都不经过控制面——前端与引擎紫实线直达；'
        'coordinator 只收小票、播看板、发 wave 信令', 10.5, lc.C_TXT, 'start', True,
        maxw=BXR - MX - 32, tag='bot:l1')
lc.text(MX + 16, BY + 46, '三 socket 三个喇叭口：publish_back(XPUB)→引擎控制 · output_back(PULL)←引擎统计 · '
        'publish_front(XPUB)→前端看板；三节拍 100ms / 5000ms / ≥50ms（lockstep 先等收齐同拍）',
        9.5, '#334155', 'start', maxw=BXR - MX - 32, tag='bot:l2')
lc.text(MX + 16, BY + 68, '数据面寻址双兑现：下行信封定向（ch5 伏笔回收）＋上行按 client_index 分桶回发'
        '——两条都只在 DP 集群里真正用起来', 9.5, lc.C_MUTE, 'start',
        maxw=BXR - MX - 32, tag='bot:l3')

# ---------------- 图例 + 页脚 ----------------
LY = BY + 86 + 26
lx0 = MX
lc.seg(lx0 + 2, LY - 3, lx0 + 32, LY - 3, lc.C_ZMQ_S, 2.6, 'pu')
lc.text(lx0 + 40, LY + 1, '数据面（请求/输出直达）', 9.5, lc.C_TXT, 'start', tag='leg:d')
lx0 += 40 + lc.tw('数据面（请求/输出直达）', 9.5) + 22
lc.seg(lx0 + 2, LY - 3, lx0 + 32, LY - 3, lc.C_MUTE, 2.0, 'std', dash=True)
lc.text(lx0 + 40, LY + 1, '控制面（小票/看板/wave）', 9.5, lc.C_TXT, 'start', tag='leg:c')
lx0 += 40 + lc.tw('控制面（小票/看板/wave）', 9.5) + 22
lc.rect(lx0 + 4, LY - 10, 16, 13, '#ffffff', lc.C_ENG_S, rx=3, sw=1.6)
lc.text(lx0 + 26, LY + 1, '独立进程（虚框）', 9.5, lc.C_TXT, 'start', tag='leg:p')
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· ZMQ 控制面为真 XPUB/XSUB/PULL 实测（XSUB 订阅走 '
        'b"\\x01" 帧）；发布时延为本机 ZMQ 实测、绝对值受调度影响', 9, lc.C_MUTE, 'start',
        maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch34-fig-dp-cluster.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
