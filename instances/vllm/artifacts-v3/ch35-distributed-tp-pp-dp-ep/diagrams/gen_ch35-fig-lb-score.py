#!/usr/bin/env python3
"""ch34 机制图 · m15 DPLB 打分决策表（figure_spec ch34-fig-lb-score）

放大自 L0 多实例视角北条『入·前端多实例』DPLB 打分（L2 站 9）的决策展开。

claim：score=max(client_count×inflight, waiting+running)+waiting×6×max(0, kv−0.5)：
突发期本地精确 inflight 地板把 5 个请求轮询散开 [0,1,2,3,0]（快照全 0 也不挤堆）、
稳态期快照抬分避重载、KV>50% 起罚（同队列 4 分→满载 16 分）。

数字全部取自 explainer figure_spec.numbers（traces/m15 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common（前端/API 蓝）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1440, 906
MX = 64
BXR = 1376


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


def check(x, y, s, color):
    """勾选标记：直接绘 SVG path，不依赖字体字形——文字版 '✓'(U+2713) 在渲染
    字体里缺字成豆腐空心方框（首版盲审 FAIL），path 与字体无关。
    x=左端 y=视觉中线，s=宽。"""
    d = (f'M{x:.1f},{y + s * 0.30:.1f} L{x + s * 0.36:.1f},{y + s * 0.58:.1f} '
         f'L{x + s:.1f},{y - s * 0.25:.1f}')
    lc.ELEMS.append(((x - 2, y - s * 0.25 - 2, x + s + 2, y + s * 0.58 + 2),
                     f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.6" '
                     'stroke-linecap="round" stroke-linejoin="round"/>'))


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'DPLB 打分决策表：快照全 0 的旧照片上，突发 5 请求照样轮询散开', 16.5,
        lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 62, 'score = max(client_count × inflight, waiting + running) + waiting × 6.0 × '
        'max(0, kv − 0.5)　（core_client.py:L1494-L1501）', 11, lc.C_API_S, 'start', True,
        maxw=1100, tag='subtitle')
lc.text(MX, 84, '行 = 同一份全 0 旧快照（100ms 刷新窗内）上连发的 r1-r5；client_count=2 → '
        '地板 = 2 × inflight——每选一家，地板立即抬起', 10, lc.C_MUTE, 'start',
        maxw=1100, tag='subtitle2')
chip(BXR, 12, '放大自 L2 站 9 · L0：多实例视角', lc.C_API_S)

# ---------------- 左：主矩阵（5 行 × 4 引擎） ----------------
TX, TY = MX, 130
LBL_W, COL_W, ROW_H = 92, 160, 88
HDR_H = 36
# 引擎表头
for i in range(4):
    lc.text(TX + LBL_W + i * COL_W + COL_W / 2, TY + 24, '引擎 %d' % i, 11.5, lc.C_TXT,
            'middle', True, tag='eh%d' % i)
lc.text(TX + LBL_W - 8, TY + 24, '请求/起点', 9.5, lc.C_MUTE, 'end', tag='rhdr')
# 突发 5 轮（traces/m15 scenario_a_burst）：snap / floor / score / chosen / start_before
# snap = coordinator 旧快照的 waiting+running——trace 记 lb_engines_snapshot_all_zero=true，
# 整个 100ms 刷新窗内恒 0；被选引擎的抬分全部来自 floor = client_count×inflight（本地精确
# 地板）。注意 trace 的 lb_engines[e][0] 槽在选中后会 +=client_count（L1503 本地预增），其
# 数值恰等于地板——那是地板一侧的本地记账，不并入快照项（与标题/图例『快照全 0』同口径）。
burst = [
    ('r1', 0, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 0, ''),
    ('r2', 1, [0, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], 1, ''),
    ('r3', 2, [0, 0, 0, 0], [2, 2, 0, 0], [2, 2, 0, 0], 2, ''),
    ('r4', 3, [0, 0, 0, 0], [2, 2, 2, 0], [2, 2, 2, 0], 3, ''),
    ('r5', 0, [0, 0, 0, 0], [2, 2, 2, 2], [2, 2, 2, 2], 0, '全平局→轮转选 0'),
]
for r, (rid, start, snap, floor, score, chosen, note) in enumerate(burst):
    ry = TY + HDR_H + r * (ROW_H + 6)
    lc.text(TX + LBL_W - 8, ry + 34, rid, 12, lc.C_TXT, 'end', True, tag='rl%d' % r)
    lc.text(TX + LBL_W - 8, ry + 56, 'start=%d' % start, 8.5, lc.C_MUTE, 'end',
            tag='rs%d' % r)
    for c in range(4):
        cx = TX + LBL_W + c * COL_W
        ccx = cx + COL_W / 2
        hit = (c == chosen)
        lc.rect(cx + 3, ry, COL_W - 6, ROW_H, lc.C_GPU_F if hit else '#ffffff',
                lc.C_GPU_S if hit else lc.C_FAINT, rx=6, sw=2.0 if hit else 1.0)
        if hit:
            # 勾选（path 绘制）+ 粗体 score 右移，整体仍以格心为中心
            lc.text(ccx + 11, ry + 34, str(score[c]), 15, lc.C_GPU_S, 'middle', True,
                    tag='c%d%d' % (r, c))
            check(ccx - 19, ry + 30, 12, lc.C_GPU_S)
        else:
            lc.text(ccx, ry + 34, str(score[c]), 13, lc.C_TXT, 'middle', True,
                    tag='c%d%d' % (r, c))
        lc.text(ccx, ry + 62, '快照 %d · 地板 %d' % (snap[c], floor[c]), 8.5,
                lc.C_MUTE, 'middle', tag='cs%d%d' % (r, c))
    if note:
        lc.text(TX + LBL_W + 4 * COL_W + 10, ry + 34, note, 9, lc.C_API_S, 'start', True,
                maxw=BXR - (TX + LBL_W + 4 * COL_W) - 14, tag='rnote%d' % r)
MTX_B = TY + HDR_H + 5 * (ROW_H + 6)
lc.text(TX, MTX_B + 20, '选中轨迹 = 对角线 [0,1,2,3,0]——视觉证明『旧快照全 0 时，本地精确地板把突发摊开』',
        10, lc.C_GPU_S, 'start', True, maxw=720, tag='mtx:foot')

# ---------------- 右上：KV 斜坡对照 ----------------
PX = 940
PW = BXR - PX
PY, PH_ = 130, 268
lc.rect(PX, PY, PW, PH_, '#ffffff', lc.C_API_S, rx=8, sw=1.5)
lc.text(PX + 14, PY + 22, 'KV 斜坡：同队列、差 4 倍', 11.5, lc.C_TXT, 'start', True,
        maxw=PW - 28, tag='kv:t')
lc.text(PX + 14, PY + 40, 'waiting×6.0×(kv−0.5)：≤50% 不罚，100% 满载 = 3×waiting',
        8.5, lc.C_MUTE, 'start', maxw=PW - 28, tag='kv:s')
kv_rows = [
    ('e0 · 4 waiting · kv 0.4', 4, lc.C_GPU_S, '4 分（不罚，选中）', True),
    ('e1 · 4 waiting · kv 1.0', 16, lc.C_ENG_S, '4 + 4×6.0×0.5 = 16 分', False),
    ('e2 · 100 waiting · 重载', 100, lc.C_MUTE, '100 分垫底', False),
    ('e3 · 100 waiting · 重载', 100, lc.C_MUTE, '100 分垫底', False),
]
BAR_X, BAR_MAX = PX + 194, 160
for i, (lab, score_, color, note, chosen_) in enumerate(kv_rows):
    ky = PY + 66 + i * 50
    lc.text(PX + 14, ky + 4, lab, 9, '#334155', 'start', maxw=192, tag='kvr%d' % i)
    bw = max(10, int(score_ / 100 * BAR_MAX))
    lc.rect(BAR_X, ky - 8, bw, 18, '#ffffff' if not chosen_ else lc.C_GPU_F, color,
            rx=3, sw=1.4)
    lc.text(BAR_X + bw + 8, ky + 5, note, 8.5, lc.C_MUTE if not chosen_ else lc.C_GPU_S,
            'start', maxw=PW - (BAR_X - PX) - bw - 22, tag='kvn%d' % i)

# ---------------- 右下：平局轮转 ----------------
QY = PY + PH_ + 18
QH = MTX_B - QY
lc.rect(PX, QY, PW, QH, '#ffffff', lc.C_API_S, rx=8, sw=1.5)
lc.text(PX + 14, QY + 22, '平局轮转：起点每次 +1 消偏置', 11.5, lc.C_TXT, 'start', True,
        maxw=PW - 28, tag='tie:t')
lc.text(PX + 14, QY + 40, 'eng_start_index = (eng_start_index + 1) % 4（L1508-L1513）',
        8.5, lc.C_MUTE, 'start', maxw=PW - 28, tag='tie:s')
tie = [('t1', 'scores [0,0,0,0] 全平局', 'start=0 → 选引擎 0'),
       ('t2', 'scores [1,0,0,0]', 'start=1 → 选引擎 1')]
for i, (tid, s1, s2) in enumerate(tie):
    ty = QY + 66 + i * 44
    lc.text(PX + 14, ty + 4, tid, 10.5, lc.C_TXT, 'start', True, tag='tie%d:r' % i)
    lc.text(PX + 46, ty + 4, s1, 9, '#334155', 'start', maxw=200, tag='tie%d:s' % i)
    lc.text(PX + 14, ty + 22, s2, 9, lc.C_API_S, 'start', True, maxw=PW - 28,
            tag='tie%d:o' % i)

# ---------------- 底部：记账与回收 ----------------
BY = MTX_B + 44
lc.rect(MX, BY, BXR - MX, 108, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, BY + 22, '选中即记账（热路径每请求 2 次字典操作）：reqs_in_flight[rid] = 引擎 '
        '＋ engine_inflight[引擎] += 1', 10.5, lc.C_TXT, 'start', True, maxw=BXR - MX - 32,
        tag='acct:l1')
lc.text(MX + 16, BY + 44, 'finished 回收成对 −1（实测 r5 完成：inflight [2,1,1,1] → [1,1,1,1]，'
        '登记表同步删 r5）；abort r3 查登记表 → 定向路由引擎 2 发 ABORT', 9.5, '#334155',
        'start', maxw=BXR - MX - 32, tag='acct:l2')
lc.text(MX + 16, BY + 64, 'coordinator 快照 100ms 一刷（变化才发、5s 心跳）——race 由本地地板兜底：'
        '『exact and can\x27t be erased by a snapshot rebind』（L1493 注释）', 9.5, '#334155',
        'start', maxw=BXR - MX - 32, tag='acct:l3')
lc.text(MX + 16, BY + 88, '外部 LB 模式（DPAsyncMPClient）不选路：绑定引擎原样返回（实测 bound 2 → '
        'returned 2）——同一能力的另一种用法', 9, lc.C_MUTE, 'start', maxw=BXR - MX - 32,
        tag='acct:l4')

# ---------------- 图例 + 页脚 ----------------
LY = BY + 108 + 26
lc.rect(MX, LY - 9, 16, 11, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=2.0)
lc.text(MX + 21, LY + 1, '选中格（score=地板项）', 9.5, lc.C_TXT, 'start', tag='leg:hit')
lx0 = MX + 21 + lc.tw('选中格（score=地板项）', 9.5) + 22
check(lx0, LY - 3, 11, lc.C_GPU_S)
lc.text(lx0 + 16, LY + 1, '勾 = 本请求选中的引擎', 9.5, lc.C_TXT, 'start', tag='leg:check')
lx0 += 16 + lc.tw('勾 = 本请求选中的引擎', 9.5) + 22
lc.text(lx0, LY + 1, '格内小字 = 快照项 / 地板项两项分列 · score = 两者取大', 9.5, lc.C_MUTE,
        'start', tag='leg:note')
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· chosen/inflight/registry/start_index 全部真调用产物，'
        'score 按源码公式从调用前真实状态复算 · 4 引擎场景实测', 9, lc.C_MUTE, 'start',
        maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch34-fig-lb-score.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
