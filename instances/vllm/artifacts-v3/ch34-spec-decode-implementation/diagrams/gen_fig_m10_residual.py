#!/usr/bin/env python3
"""ch34 机制图 · recovered 残差采样：残差×inv_q 免归一化 argmax（figure_spec fig_m10_residual，模板 before-after）

放大自 L0 采样列 spec 块验证期的残差补货——L2 章图 south『sample_recovered_tokens ·
残差预采』组件的机制小图（对应站 14；架构归属回指 L2，不另立第二种架构画法）。

claim：recovered 从残差分布补货：残差 = max(p_t−p_d, 0)（本例质量 0.2 / 0.35，恰为
两分布 TV 距离）、score = 残差×inv_q 免归一化取 argmax、每请求一份 q 全草稿位共享。

数字全部取自 figure_spec.numbers（traces/ch34_m10_recovered_residual.json part_a_exact /
part_b_marginal / part_c_no_draft）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 1050
MX, BXR = 56, 1444
DEFS = lc.DEFS
PT_C, PD_C, RES_C, SC_C = '#2563eb', '#ea580c', lc.C_GPU_S, '#be185d'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '退货补货从『差价』里挑：残差 = 两分布之差的正部，Gumbel-max 掷骰免归一化', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'score = 残差 × 1/q 取 argmax（q~Exp(1)，每词表位一份）· 每请求只采一份 q、全部草稿位共享（v0.27：[batch, V] 而非 [num_tokens, V]）· 拒绝之后补什么',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列+spec 验证期（L2 south 残差预采·站 14 步进图）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 双位置面板 =================
POS = [
    dict(tag='pos0', draft='草稿 token 1', pt=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05],
         pd=[0.2, 0.35, 0.1, 0.15, 0.1, 0.1],
         res=[0.1, 0.0, 0.1, 0.0, 0.0, 0.0], mass=0.2,
         score=[0.19814, 0.0, 0.266747, 0.0, 0.0, 0.0], argmax=2),
    dict(tag='pos1', draft='草稿 token 2', pt=[0.1, 0.1, 0.3, 0.25, 0.15, 0.1],
         pd=[0.1, 0.05, 0.1, 0.15, 0.3, 0.3],
         res=[0.0, 0.05, 0.2, 0.1, 0.0, 0.0], mass=0.35,
         score=[0.0, 0.041249, 0.533494, 0.330543, 0.0, 0.0], argmax=2),
]
INVQ = [1.981396, 0.824982, 2.667468, 3.305432, 0.431985, 0.879231]
VMAX = 0.4          # 概率条满刻度
SMAX = 0.6          # score 条满刻度
PANW = (BXR - MX - 24) / 2
for i, p in enumerate(POS):
    PX_, PY_ = MX + i * (PANW + 24), 100
    PH_ = 560
    lc.rect(PX_, PY_, PANW, PH_, '#ffffff', lc.C_MUTE, rx=10, sw=1.4)
    lc.text(PX_ + 16, PY_ + 24, p['tag'] + '（' + p['draft'] + '）', 11.5, lc.C_TXT, 'start', True,
            maxw=PANW - 32, tag='pt' + p['tag'])
    # ---- before：p_t / p_d 成对条 ----
    lx, ly = PX_ + 56, PY_ + 66
    SW, BARW, BH_ = PANW - 96, 44, 210
    lc.text(PX_ + 16, ly - 10, '① 残差 = max(p_t − p_d, 0)', 9.8, lc.C_TXT, 'start', True, maxw=280, tag='b1' + p['tag'])
    def bh_(v, vmax):
        return v / vmax * BH_
    for v in range(0, 5):                      # 0 / .1 / .2 / .3 / .4 网格
        gy_ = ly + BH_ - (VMAX * v / 4) / VMAX * BH_
        lc.seg(lx, gy_, lx + 6 * SW / 6, gy_, '#eef2f7', 1.0)
        lc.text(lx - 8, gy_ + 3, ('%.1f' % (VMAX * v / 4)).rstrip('0').rstrip('.') or '0',
                7.4, lc.C_FAINT, 'end', maxw=30, tag='gy' + p['tag'] + str(v))
    for j in range(6):
        x0 = lx + j * SW / 6
        hb = (SW / 6 - BARW / 2 - 8) / 2
        # p_t 条
        hh = bh_(p['pt'][j], VMAX)
        lc.rect(x0 + 2, ly + BH_ - hh, hb, hh, '#dbeafe', PT_C, rx=2, sw=0)
        # p_d 条
        hh2 = bh_(p['pd'][j], VMAX)
        lc.rect(x0 + 2 + hb + 3, ly + BH_ - hh2, hb, hh2, '#ffedd5', PD_C, rx=2, sw=0)
        # 残差标注（正部）：在 p_t 条顶端加粗段
        if p['res'][j] > 0:
            rh = bh_(p['res'][j], VMAX)
            lc.rect(x0 + 2, ly + BH_ - hh, hb, rh, PT_C, PT_C, rx=2, sw=0)
            lc.text(x0 + 2 + hb / 2, ly + BH_ - hh - 8, ('+' + ('%g' % p['res'][j])), 7.6, RES_C, 'middle',
                    True, maxw=44, tag='rr' + p['tag'] + str(j))
        lc.text(x0 + 2 + hb, ly + BH_ + 14, str(j), 8, lc.C_FAINT, 'middle', maxw=20, tag='vx' + p['tag'] + str(j))
    lc.seg(lx, ly + BH_, lx + 6 * SW / 6, ly + BH_, '#94a3b8', 1.2)
    lc.text(lx + 6 * SW / 6 + 10, ly + 40, 'p_t 蓝 / p_d 橙', 7.8, lc.C_MUTE, 'start', maxw=90, tag='lgp')
    lc.text(lx + 6 * SW / 6 + 10, ly + 56, '深蓝段=正部', 7.8, PT_C, 'start', maxw=90, tag='lgp2')
    lc.text(PX_ + 16, ly + BH_ + 34, '残差质量 ' + ('%g' % p['mass']) + '（= 两分布的 TV 距离）——归一化后即补货分布', 8.6,
            RES_C, 'start', True, maxw=PANW - 32, tag='tm' + p['tag'])
    # ---- after：score 条 ----
    sx, sy = PX_ + 56, ly + BH_ + 66
    SH_ = 150
    lc.text(PX_ + 16, sy - 10, '② score = 残差 × inv_q → argmax 免归一化', 9.8, lc.C_TXT, 'start', True,
            maxw=300, tag='b2' + p['tag'])
    for j in range(6):
        x0 = sx + j * SW / 6
        hh = p['score'][j] / SMAX * SH_
        wbar = SW / 6 - 26
        is_m = j == p['argmax']
        lc.rect(x0 + 2, sy + SH_ - hh, wbar, max(hh, 1.2), SC_C if is_m else '#fbcfe8', SC_C, rx=2, sw=0)
        lc.text(x0 + 2 + wbar / 2, sy + SH_ + 14, str(j), 8, lc.C_FAINT, 'middle', maxw=20, tag='sx' + p['tag'] + str(j))
        if p['score'][j] > 0:
            lc.text(x0 + 2 + wbar / 2, sy + SH_ - hh - 7, ('%g' % p['score'][j]), 7.2, SC_C if is_m else lc.C_MUTE,
                    'middle', True, maxw=54, tag='sv' + p['tag'] + str(j))
    lc.seg(sx, sy + SH_, sx + 6 * SW / 6, sy + SH_, '#94a3b8', 1.2)
    if i == 0:
        lc.text(sx + 2 * SW / 6 + SW / 12, sy - 26, 'argmax=2（0.19814 < 0.266747）→ recovered', 8.8, SC_C,
                'middle', True, maxw=320, tag='am0')
    else:
        lc.text(sx + 2 * SW / 6 + SW / 12, sy - 26, 'argmax=2（0.533494 最大）→ recovered', 8.8, SC_C,
                'middle', True, maxw=300, tag='am1')

# ================= 共享 q 行 =================
QY = 100 + 560 + 18
lc.rect(MX, QY, BXR - MX, 84, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, QY + 20, '一份 q、全草稿位共享：inv_q = [1.981396, 0.824982, 2.667468, 3.305432, 0.431985, 0.879231]（pos0 与 pos1 用的是同一行）', 9.4,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='q1')
lc.text(MX + 16, QY + 40, '为什么可以：只有首个拒绝位的 recovered 被消费——共享 q 只造成跨位相关，无人消费相关位；单看任一位，argmax(残差·inv_q) 的边缘分布 = 残差归一化分布（Gumbel-max）。',
        8.6, lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='q2')
lc.text(MX + 16, QY + 60, '噪声账：[batch=1, vocab=6] 一行 q 对照逐位一份的 [2,6]；真实批 B=256、num_tokens=1024 时省 4 倍。', 8.4,
        lc.C_FAINT, 'start', maxw=BXR - MX - 36, tag='q3')

# ================= 底部双卡：统计律 / ngram 变体 =================
DY0 = QY + 96
CARDW = (BXR - MX - 20) / 2
cards = [
    ('统计律（20000 样本边缘分布）', RES_C,
     ['理论 [0.5, 0, 0.5, 0, 0, 0] → 实测 [0.4961, 0, 0.5039, 0, 0, 0]', '最大偏差 0.0039——共享噪声的『为什么可以』被实测背书',
      '免归一化：argmax 不受正常数缩放影响，全词表求和都省了']),
    ('ngram 变体（无 draft 概率）', lc.C_ZMQ_S,
     ['残差 = p_t 屏蔽 draft token 位：[0, 0.3, 0.2, 0, 0, 0]', '（draft token 0 的 p_t=0.5 被 mask 掉）',
      'score 最大 0.447383 → argmax=2，与带概率路径同款免归一化']),
]
for i, (ttl, col, lines) in enumerate(cards):
    cx_ = MX + i * (CARDW + 20)
    lc.rect(cx_, DY0, CARDW, 100, '#ffffff', col, rx=9, sw=1.5)
    lc.text(cx_ + 14, DY0 + 22, ttl, 10, col, 'start', True, maxw=CARDW - 28, tag='c' + str(i))
    for j, ln in enumerate(lines):
        lc.text(cx_ + 14, DY0 + 42 + j * 17, '· ' + ln, 8.2, '#475569', 'start', maxw=CARDW - 26, tag='cl' + str(i) + str(j))

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/sample/rejection_sampler.py:L663-L710（sample_recovered_tokens·每请求一份 q）· L872-L953（残差 kernel）· L913-L918（ngram mask）· L931-L932（免归一化注释）· L944-L952（OOV mask+clamp）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, '残差/score/质量/边缘分布 ＝ seeded q 重构（与 kernel 输出逐位对拍一致）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m10_residual.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
