#!/usr/bin/env python3
"""ch34 机制图 · m19 wave 共识时序（figure_spec ch34-fig-wave-consensus）

放大自 L0 多实例视角 ⑥ DP 协调·拍间 wave（L2 站 16）的时序展开。

claim：MoE DP 引擎锁步：无活引擎跑 dummy batch 维持步进对齐（40 拍预算中前
32 拍全在空转）、每 32 步一次 2 元素 SUM all-reduce 同时裁决『任一有活全体继续』
（SUM[0]>0 ≡ OR）与『全体同意暂停』（SUM[1]==dp_size），全体无活时仅 dp_rank0
发 wave_complete（-1 哨兵走 coordinator）、current_wave+1。

时序图严格按 UML 文法（FIGURE-SYSTEM §0.2 / WRITING-CONTRACT §8）：
参与者=竖直生命线（顶部名牌）、共享时间轴（gridlines 贯穿、活动条高=时长×
统一比例尺）、消息=水平直线（跨中间生命线直穿，禁一切折线/肘形）、瞬时动作=
骑线时刻标记。数字全部取自 explainer figure_spec.numbers（traces/m19 实测 + pin 锚点）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1440, 1030
MX = 64
BXR = 1376

DEFS = lc.DEFS.replace('</defs>',
                       '<marker id="pu" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="6.5" markerHeight="4.6" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
                       '<marker id="puL" viewBox="0 0 10 6" refX="1" refY="3" '
                       'markerWidth="6.5" markerHeight="4.6" orient="auto">'
                       f'<path d="M10,0 L0,3 L10,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
                       '<marker id="mu" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="6.5" markerHeight="4.6" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_MUTE}"/></marker>'
                       '</defs>')


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


# ---------------- 生命线几何（统一比例尺） ----------------
E0_X, E1_X, CO_X = 300, 720, 1140
NP_Y, NP_H = 92, 40
LF_TOP, LF_BOT = NP_Y + NP_H, 812
T0, PPU = 190, 13          # 拍 0 的 y、每拍像素（时间轴统一比例尺）


def y_step(s):
    return T0 + s * PPU


BAR_W = 18


def xmark(cx, cy, color):
    """丢弃标记 ×。"""
    lc.ELEMS.append(((cx - 7, cy - 7, cx + 7, cy + 7),
                     f'<path d="M{cx - 6},{cy - 6} L{cx + 6},{cy + 6} '
                     f'M{cx + 6},{cy - 6} L{cx - 6},{cy + 6}" stroke="{color}" '
                     f'stroke-width="2.2"/>'))


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'wave 共识时序：每 32 拍一次 2 元素 SUM all-reduce——两个问题，一次通信全答', 16.5,
        lc.C_TXT, 'start', True, maxw=1100, tag='title')
lc.text(MX, 62, '没活儿的也要在原地踏步（dummy batch）：下一个动作（EP 全网重排）要全员到齐才能做；'
        '全体无活 → 共识暂停、wave+1', 10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
chip(BXR, 12, '放大自 L2 站 16 · L0：多实例视角', lc.C_ENG_S)

# ---------------- 顶部名牌 + 生命线 ----------------
for cx, name, sub in [(E0_X, '引擎 0 · dp_rank0', '无活（收尾后）'),
                      (E1_X, '引擎 1 · dp_rank1', '无活（收尾后）'),
                      (CO_X, 'DPCoordinator', '控制面')]:
    lc.rect(cx - 110, NP_Y, 220, NP_H, '#ffffff', lc.C_ENG_S, rx=8, sw=2.0)
    lc.text(cx, NP_Y + 17, name, 12, lc.C_ENG_S, 'middle', True, tag='np:' + name[:6])
    lc.text(cx, NP_Y + 33, sub, 8.5, lc.C_MUTE, 'middle', tag='nps:' + name[:6])
    lc.ELEMS.append(((cx - 1, LF_TOP, cx + 1, LF_BOT),
                     f'<line x1="{cx}" y1="{LF_TOP}" x2="{cx}" y2="{LF_BOT}" '
                     f'stroke="{lc.C_FAINT}" stroke-width="1.2" stroke-dasharray="4,5"/>'))

# ---------------- 共享时间轴：gridlines 贯穿 + 拍刻度 ----------------
GL_X0, GL_X1 = 130, 1400
for s in (0, 8, 16, 24, 32):
    y = y_step(s)
    lc.ELEMS.append(((GL_X0, y - 1, GL_X1, y + 1),
                     f'<line x1="{GL_X0}" y1="{y}" x2="{GL_X1}" y2="{y}" '
                     f'stroke="#e2e8f0" stroke-width="1"/>'))
    lc.text(118, y + 4, '拍 %d' % s, 9, lc.C_MUTE, 'end', tag='tick%d' % s)
lc.text(118, T0 - 16, 'step_counter', 8.5, lc.C_FAINT, 'end', tag='axis:name')
# 比例尺声明（暂停段压缩）
lc.text(GL_X0, y_step(32) + 18, '↓ 第 32 拍后进入暂停段——时长不定，压缩显示（不再按拍刻度）', 8.5,
        lc.C_FAINT, 'start', maxw=420, tag='axis:note', halo=True)

# ---------------- 活动条：dummy batch ×32（两引擎同高=同比例尺） ----------------
for cx in (E0_X, E1_X):
    lc.rect(cx - BAR_W / 2, y_step(1), BAR_W, y_step(32) - y_step(1), lc.C_ENG_F,
            lc.C_ENG_S, rx=3, sw=1.6)
lc.text((E0_X + E1_X) / 2, 300, '两引擎连续 dummy batch：第 1 拍起、第 32 拍止（实测各 32 次）',
        10, lc.C_ENG_S, 'middle', True, maxw=380, tag='bar:l1')
lc.text((E0_X + E1_X) / 2, 322, '空转维持锁步——EP all2all 缺席者会挂死全员', 9, '#334155',
        'middle', maxw=380, tag='bar:l2')
lc.text((E0_X + E1_X) / 2, 240, '拍 8 / 16 / 24：step_counter % 32 != 0 → 短路返回 True'
        '（不做通信）', 9, lc.C_MUTE, 'middle', maxw=380, tag='gate:l')
lc.text(288, 566, '预算 40 拍（max_iters）', 9, lc.C_MUTE, 'end',
        maxw=160, tag='budget')
lc.text(288, 582, '本例第 32 拍即收工', 9, lc.C_MUTE, 'end', maxw=160, tag='budget2')

# ---------------- 第 32 拍：SUM all-reduce（水平消息，双端箭头=集合通信） ----------------
Y_AR = y_step(32)
AR_X0, AR_X1 = E0_X + BAR_W / 2 + 3, E1_X - BAR_W / 2 - 3
lc.seg(AR_X0, Y_AR, AR_X1, Y_AR, lc.C_ZMQ_S, 2.6, 'pu')       # 右端外向箭头
lc.seg(AR_X1, Y_AR, AR_X0, Y_AR, lc.C_ZMQ_S, 2.6, 'puL')      # 左端外向箭头（双向=集合）
lc.text((E0_X + E1_X) / 2, Y_AR - 22, 'sync_dp_state：2 元素 SUM all-reduce（[int(has), int(pause)]）',
        10.5, lc.C_ZMQ_S, 'middle', True, maxw=400, tag='ar:l')
lc.text((E0_X + E1_X) / 2, Y_AR - 6, '本例全体无活：has=[F,F] · pause=[T,T] → SUM [0,2] → 全体暂停',
        9, '#334155', 'middle', maxw=400, tag='ar:v')

# ---------------- 右侧：真值表小框（挂在消息旁的插图） ----------------
TT_X, TT_Y, TT_W = 1188, 176, 236
lc.rect(TT_X, TT_Y, TT_W, 296, '#ffffff', lc.C_ZMQ_S, rx=8, sw=1.5)
lc.text(TT_X + 12, TT_Y + 20, 'SUM 真值表（2 rank · 6 组实测）', 10, lc.C_TXT, 'start', True,
        maxw=TT_W - 24, tag='tt:t')
lc.text(TT_X + 12, TT_Y + 37, 'has 两位 + pause 两位 → SUM → 判定', 8, lc.C_MUTE, 'start',
        maxw=TT_W - 24, tag='tt:s')
tt_rows = [
    ('[T,F]+[F,F] → [1,0]', '有活·全体继续'),
    ('[F,F]+[F,F] → [0,0]', '无活·不暂停'),
    ('[F,F]+[T,T] → [0,2]', '无活·全体暂停'),
    ('[F,F]+[T,F] → [0,1]', '仍有活·未到齐'),
]
for i, (a, b) in enumerate(tt_rows):
    ry = TT_Y + 60 + i * 44
    lc.text(TT_X + 12, ry, a, 9, '#334155', 'start', True, maxw=TT_W - 24,
            tag='tt%d:a' % i)
    lc.text(TT_X + 24, ry + 16, '→ %s' % b, 8.5, lc.C_MUTE, 'start', maxw=TT_W - 36,
            tag='tt%d:b' % i)
    if i == 2:
        lc.text(TT_X + 24, ry + 32, '（SUM[1]==dp_size：本例）', 8, lc.C_ZMQ_S, 'start',
                maxw=TT_W - 36, tag='tt:hit')
    if i == 3:
        lc.text(TT_X + 24, ry + 32, '（pause_count%dp_size!=0）', 8, lc.C_MUTE, 'start',
                maxw=TT_W - 36, tag='tt:edge')
lc.text(TT_X + 12, TT_Y + 250, '判据：SUM[0]>0 ≡ OR（任一有活全体继续）', 8.5, lc.C_ZMQ_S,
        'start', maxw=TT_W - 24, tag='tt:f1')
lc.text(TT_X + 12, TT_Y + 266, 'SUM[1]==dp_size ≡ 全体同意暂停', 8.5, lc.C_ZMQ_S, 'start',
        maxw=TT_W - 24, tag='tt:f2')
lc.text(TT_X + 12, TT_Y + 286, '另有单元素 MAX≡OR 共识（resume 路径）',
        8, lc.C_MUTE, 'start', maxw=TT_W - 24, tag='tt:f3')

# ---------------- wave_complete（仅 rank0 → coordinator，跨 E1 直穿） ----------------
Y_WC = 650
lc.seg(E0_X + 2, Y_WC, CO_X - 2, Y_WC, lc.C_MUTE, 2.2, 'mu')
lc.text((E0_X + CO_X) / 2, Y_WC - 20, 'wave_complete（wave 0 · client_index = -1 哨兵）',
        10, lc.C_MUTE, 'middle', True, maxw=460, tag='wc:l', halo=True)
lc.text((E0_X + CO_X) / 2, Y_WC - 4, '仅 dp_rank0 发（引擎 1 events 空）；经 output 队列拐进 coordinator',
        8.5, lc.C_MUTE, 'middle', maxw=460, tag='wc:l2', halo=True)
# 骑线时刻标记（coordinator 侧收到）
lc.ELEMS.append(((CO_X - 5, Y_WC - 5, CO_X + 5, Y_WC + 5),
                 f'<circle cx="{CO_X}" cy="{Y_WC}" r="4.5" fill="{lc.C_MUTE}"/>'))
lc.text(CO_X, Y_WC + 18, '收到 → 看板/新 wave', 8, lc.C_MUTE, 'middle', maxw=140,
        tag='wc:recv', halo=True)

# ---------------- 状态迁移：running → paused · wave 0→1 ----------------
Y_ST = 700
for cx in (E0_X, E1_X):
    lc.rect(cx - 33, Y_ST - 11, 66, 22, '#f1f5f9', lc.C_MUTE, rx=10, sw=1.2)
    lc.text(cx, Y_ST + 4, 'paused', 9, lc.C_MUTE, 'middle', True, tag='st%d' % cx)
lc.text((E0_X + E1_X) / 2, Y_ST + 4, 'running → paused · current_wave 0 → 1 · step_counter 归零',
        9.5, lc.C_TXT, 'middle', True, maxw=360, tag='st:l', halo=True)

# ---------------- 暂停段：在途 START_DP_WAVE 被丢弃（虚线打叉） ----------------
Y_SW1, Y_SW0 = 748, 772
lc.seg(CO_X - 2, Y_SW1, E1_X + BAR_W / 2 + 3, Y_SW1, lc.C_MUTE, 1.8, None, dash=True)
xmark((CO_X + E1_X) / 2, Y_SW1, lc.C_ABORT)
lc.seg(CO_X - 2, Y_SW0, E0_X + BAR_W / 2 + 3, Y_SW0, lc.C_MUTE, 1.8, None, dash=True)
xmark((CO_X + E0_X) / 2, Y_SW0, lc.C_ABORT)
xmark((E1_X + E0_X) / 2, Y_SW0, lc.C_ABORT)
lc.text((E0_X + CO_X) / 2, Y_SW0 + 22, '暂停期在途 START_DP_WAVE 送达 → ignore_start_dp_wave 丢弃'
        '（共识后旧 wave 唤醒一律作废）', 9, lc.C_MUTE, 'middle', maxw=520, tag='sw:l',
        halo=True)

# ---------------- 底部：代价与必要性 ----------------
BY = 848
lc.rect(MX, BY, BXR - MX, 100, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, BY + 22, '最坏滞留感知：共识门每 32 拍一开（step_counter%32!=0 短路）——无活状态'
        '最多再等 31 拍才被全组感知', 10.5, lc.C_TXT, 'start', True, maxw=BXR - MX - 32,
        tag='bot:l1')
lc.text(MX + 16, BY + 44, '锁步的必要性：EP 的 all2all 要全员到齐、缺席者会挂死全组——所以无活引擎'
        '宁可 dummy batch 空转也不离场（40 拍预算 max_iters 兜底）', 9.5, '#334155',
        'start', maxw=BXR - MX - 32, tag='bot:l2')
lc.text(MX + 16, BY + 64, '两个问题一次通信全答：OR（任一有活？）与全体一致（都同意暂停吗？）拼进同一个 '
        '2 元素张量，一次 SUM all-reduce 同时裁决', 9.5, '#334155', 'start',
        maxw=BXR - MX - 32, tag='bot:l3')
lc.text(MX + 16, BY + 86, '实测：2 进程 stateless gloo，busy 循环预算 40 拍——dummy 恰好 32 次'
        '（首@1 · 末@32）；final_wave 0→1，rank0 events [[-1,0]]、rank1 空', 9, lc.C_MUTE,
        'start', maxw=BXR - MX - 32, tag='bot:l4')

# ---------------- 图例 + 页脚 ----------------
LY = BY + 100 + 26
lx0 = MX
lc.seg(lx0 + 2, LY - 3, lx0 + 32, LY - 3, lc.C_ZMQ_S, 2.6, 'puL')
lc.text(lx0 + 40, LY + 1, 'gloo 集合通信（dp_group）', 9.5, lc.C_TXT, 'start', tag='leg:ar')
lx0 += 40 + lc.tw('gloo 集合通信（dp_group）', 9.5) + 22
lc.seg(lx0 + 2, LY - 3, lx0 + 32, LY - 3, lc.C_MUTE, 2.2, 'mu')
lc.text(lx0 + 40, LY + 1, '控制面消息（水平直线·跨线直穿）', 9.5, lc.C_TXT, 'start',
        tag='leg:wc')
lx0 += 40 + lc.tw('控制面消息（水平直线·跨线直穿）', 9.5) + 22
lc.seg(lx0 + 2, LY - 3, lx0 + 32, LY - 3, lc.C_MUTE, 1.8, None, dash=True)
xmark(lx0 + 17, LY - 3, lc.C_ABORT)
lc.text(lx0 + 40, LY + 1, '被丢弃的唤醒', 9.5, lc.C_TXT, 'start', tag='leg:x')
lx0 += 40 + lc.tw('被丢弃的唤醒', 9.5) + 22
lc.rect(lx0 + 4, LY - 10, 14, 15, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.4)
lc.text(lx0 + 26, LY + 1, '活动条=running（高=时长×13px/拍）', 9.5, lc.C_TXT, 'start',
        tag='leg:bar')
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· UML 时序文法：生命线竖直 + 共享时间轴 + 消息水平直线'
        '（无折线/肘形）· 2 进程 gloo 实测（真 pin 多卡 NCCL）', 9, lc.C_MUTE, 'start',
        maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch34-fig-wave-consensus.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
