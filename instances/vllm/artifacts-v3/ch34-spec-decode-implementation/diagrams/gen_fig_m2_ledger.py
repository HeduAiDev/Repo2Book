#!/usr/bin/env python3
"""ch34 机制图 · 调度器 token 账本时间线（figure_spec fig_m2_ledger，模板 state-table）

放大自 L0 主循环 loop_box 调度器臂的账本——L2 章图第④拍（schedule 排批）与第⑨拍
（回扣）之间补一条账本时间线（架构归属回指 L2，不另立第二种架构画法）。

claim：调度器 token 账本的乐观推进与拒绝回扣：c1 拍按全收乐观记 18、验证拒 2 回扣到
16；c2 拍全收不回扣（20 对平）；async 下占位账 3→1 同步共变。

数字全部取自 figure_spec.numbers（traces/ch34_m02_scheduler_ledger.json ledger_walk /
async_rollback / production_account）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 980
MX, BXR = 56, 1444
DEFS = lc.DEFS

UP, DN, FLAT = lc.C_GPU_S, lc.C_ABORT, '#94a3b8'   # 乐观推进=绿 / 回扣=红 / 对平=灰

# ---------------- 标题区 ----------------
lc.text(MX, 34, '排批只认 token 数：草稿先当『要算的位』记进差账，验证后把被拒的位划回去', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'num_new_tokens = num_tokens_with_spec + num_output_placeholders − num_computed_tokens（差账式，scheduler.py:L516-L520）· '
               '拒绝数 = 数输出行里的 −1（K=3，eagle，sync）',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 主循环调度器臂（L2 ④排批 ↔ ⑨回扣 的账本时间线）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 左：账本阶梯线（num_computed_tokens 随事件演化） =================
CHX, CHY, CHW, CHH = MX, 100, 950, 560
lc.rect(CHX, CHY, CHW, CHH, '#ffffff', lc.C_ENG_S, rx=10, sw=1.8)
lc.text(CHX + 18, CHY + 26, 'num_computed_tokens 账本时间线（c1 拒 2 · c2 全收）', 12.5,
        lc.C_TXT, 'start', True, maxw=CHW - 36, tag='cht')

# y 轴（12..22）
AX0, AX1 = 14, 22                    # 值域
def vy(v):
    return CHY + CHH - 46 - (v - AX0) * (CHH - 96) / (AX1 - AX0)

PL, PR = CHX + 150, CHX + CHW - 120  # 数据区左右界
for v in range(AX0, AX1 + 1, 2):
    lc.seg(PL - 6, vy(v), PR, vy(v), '#eef2f7', 1.0)
    lc.text(PL - 14, vy(v) + 3.5, str(v), 9.5, lc.C_FAINT, 'end', maxw=40, tag='ax' + str(v))
lc.text(CHX + 26, CHY + 48, 'num_computed', 9.5, lc.C_MUTE, 'start', True, maxw=90, tag='axl')
lc.seg(PL - 6, vy(AX0), PL - 6, vy(AX1), '#cbd5e1', 1.2)

# 事件点：(值, 事件标签上, 事件标签下, 颜色)
EVS = [
    (14, 'c1 挂账', 'spec=[31,32,33]\nnum 14 / with_spec 18', lc.C_ENG_S),
    (18, 'c1 排批', 'num_new_tokens=4\n=1 补喂 + 3 草稿', UP),
    (16, 'c1 回扣', '输出 [31,77,-1,-1]\n接受 1、拒 2 → −2', DN),
    (20, 'c2 排批', '挂账 spec=[41,42,43]\n+4 = 1+3（16→20）', UP),
    (20, 'c2 对平', '输出 [41,42,43,99]\n全收 3+bonus → −0', FLAT),
]
n = len(EVS)
xs = [PL + 40 + i * (PR - PL - 80) / (n - 1) for i in range(n)]
for i in range(n - 1):
    v0, v1 = EVS[i][0], EVS[i + 1][0]
    col = UP if v1 > v0 else (DN if v1 < v0 else FLAT)
    dy = 14 if v1 >= v0 else -14
    lc.seg(xs[i], vy(v0), xs[i + 1], vy(v1), col, 2.6)
    mid = ((xs[i] + xs[i + 1]) / 2, (vy(v0) + vy(v1)) / 2 + dy)
    lab = '+4' if v1 - v0 == 4 else ('−2' if v1 - v0 == -2 else '−0')
    lc.text(mid[0], mid[1], lab, 10.5, col, 'middle', True, maxw=50, tag='dl' + str(i))
for i, (v, top, bot, col) in enumerate(EVS):
    lc.circle(xs[i], vy(v), 7, col, 2.2, dash=False)
    lc.text(xs[i], vy(v) - 16, str(v), 12, col, 'middle', True, maxw=40, tag='pv' + str(i))
    lines_top = top
    lc.text(xs[i], vy(v) + 30, lines_top, 10, lc.C_TXT, 'middle', True, maxw=160, tag='et' + str(i))
    for j, ln in enumerate(bot.split('\n')):
        lc.text(xs[i], vy(v) + 46 + j * 13, ln, 8.6, '#475569', 'middle', maxw=170, tag='eb' + str(i) + str(j))

# 拍分隔虚线（c1 | c2）
DXP = (xs[2] + xs[3]) / 2
lc.seg(DXP, CHY + 40, DXP, CHY + CHH - 14, '#e2e8f0', 1.2, dash=True)
lc.text(DXP, CHY + CHH - 4, '下一拍', 8.2, lc.C_FAINT, 'middle', maxw=60, tag='nb')

# 差账对平注记（c2 不回扣的机理）
lc.text((xs[3] + xs[4]) / 2, vy(20) + 76, 'num_rejected=0 → 不回扣：差账对平（20−0→20）', 8.8,
        lc.C_MUTE, 'middle', maxw=280, tag='flat')

# ================= 右：async 变体两本账 =================
RX, RY, RW, RH = CHX + CHW + 24, CHY, BXR - (CHX + CHW + 24), CHH
lc.rect(RX, RY, RW, RH, '#ffffff', lc.C_ZMQ_S, rx=10, sw=1.8)
lc.text(RX + 16, RY + 26, 'async 变体：同一 c1 场景拒 2', 12, lc.C_TXT, 'start', True, maxw=RW - 32, tag='at')
lc.text(RX + 16, RY + 44, '两本账同步共变（scheduler.py:L1779-L1784）', 8.8, lc.C_MUTE, 'start', maxw=RW - 32, tag='as')
LEDGERS = [
    ('num_computed_tokens', [('挂账后（乐观）', 18), ('回扣后', 16)], lc.C_ENG_S, '−2（拒2）', '被拒两位退回『未计算』'),
    ('num_output_placeholders', [('草稿占位', 3), ('回扣后', 1)], lc.C_ZMQ_S, '3→1', '两本账同步共变'),
]
ly = RY + 78
for name, vals, col, delta, note_ in LEDGERS:
    lh_ = 128
    lc.rect(RX + 14, ly, RW - 28, lh_, '#f8fafc', lc.C_MUTE, rx=7, sw=1.0)
    lc.text(RX + 26, ly + 20, name, 9.8, col, 'start', True, maxw=RW - 52, tag='ln' + name[:6])
    for k, (lbl, v) in enumerate(vals):
        bx = RX + 40 + k * 170
        bw = 96
        lc.rect(bx, ly + 34, bw, 44, '#ffffff', col, rx=6, sw=1.6)
        lc.text(bx + bw / 2, ly + 52, lbl, 8.4, lc.C_MUTE, 'middle', maxw=bw - 6, tag='bl' + name[:4] + str(k))
        lc.text(bx + bw / 2, ly + 70, str(v), 15, col, 'middle', True, maxw=bw - 6, tag='bv' + name[:4] + str(k))
        if k == 0:
            lc.seg(bx + bw + 4, ly + 56, bx + 166, ly + 56, col, 2.0, 'std')
            lc.text(bx + bw + 35, ly + 50, delta, 8.6, col, 'middle', True, maxw=64, tag='bd' + name[:4])
    lc.text(RX + 40, ly + 98, note_, 8.4, '#475569', 'start', maxw=RW - 70, tag='bn' + name[:4])
    ly += lh_ + 12
lc.text(RX + 16, ly + 6, 'async 下占位账与计算账一起回扣——', 8.6, '#475569', 'start', maxw=RW - 32, tag='an1')
lc.text(RX + 16, ly + 21, '不破坏『每拍 ≥1 token、账不超前』的守恒。', 8.6, '#475569', 'start', maxw=RW - 32, tag='an2')

# ================= 底部：产出账 =================
BY = CHY + CHH + 22
lc.rect(MX, BY, BXR - MX, 120, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 18, BY + 24, '产出账：11 token / 7 拍，对照无 spec 的 11 拍 → 1.571429 倍——加速完全由接受率决定', 11,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='ba')
# 拍条可视化：spec 7 拍（c1 产 2 / c2 产 4 / 余 5 拍各 1） vs 无 spec 11 拍（每拍 1）
bx0, by_, bw_, bh_ = MX + 220, BY + 42, 44, 30
lc.text(MX + 18, by_ + 19, 'spec 模式（7 拍）', 9.2, lc.C_GPU_S, 'start', True, maxw=160, tag='sm')
SPTOKS = [2, 4, 1, 1, 1, 1, 1]
cx = bx0
for i, t in enumerate(SPTOKS):
    hh_ = bh_ * (0.42 + 0.29 * t)          # 拍高 ∝ 产出 token 数
    lc.rect(cx, by_ + bh_ - hh_, bw_, hh_, lc.C_GPU_F if t > 1 else '#ffffff', lc.C_GPU_S, rx=3, sw=1.2)
    lc.text(cx + bw_ / 2, by_ + bh_ - hh_ / 2 + 3, str(t), 9.2 if t > 1 else 8.2,
            lc.C_GPU_S, 'middle', True, maxw=bw_ - 6, tag='sp' + str(i))
    cx += bw_ + 6
lc.text(cx + 10, by_ + 14, 'c1 产 2 · c2 产 4 · 余 5 拍各产 1（= 11 token / 7 拍）', 8.6,
        lc.C_MUTE, 'start', maxw=430, tag='sn')
by2 = by_ + 44
lc.text(MX + 18, by2 + 19, '无 spec（11 拍）', 9.2, lc.C_MUTE, 'start', True, maxw=160, tag='nm')
cx2 = bx0
for i in range(11):
    hh_ = bh_ * 0.71
    lc.rect(cx2, by2 + bh_ - hh_, bw_, hh_, '#ffffff', lc.C_MUTE, rx=3, sw=1.0)
    lc.text(cx2 + bw_ / 2, by2 + bh_ - hh_ / 2 + 3, '1', 8.2, lc.C_MUTE, 'middle', maxw=bw_ - 6, tag='ns' + str(i))
    cx2 += bw_ + 6
lc.text(cx2 + 10, by2 + 14, '每拍恰 1 token——同样 11 token 要 11 拍', 8.6, lc.C_MUTE, 'start', maxw=430, tag='nn')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/core/sched/scheduler.py:L516-L520（差账式）· L640-L656（排批裁 spec）· L2146-L2167（挂账）· L1766-L1790（回扣；async 占位账 L1779-L1784）· '
                    'vllm/v1/sample/rejection_sampler.py:L715-L769（验证真跑）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, '14→18→16→20 / num_new_tokens=4=1+3 / 输出行 / async 占位 3→1 / 11 token·7 拍·1.571429 倍 ＝ 本章驱动脚本真跑实测（greedy kernel 零 RNG）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 10, '差账式里 num_output_placeholders 在 sync 模式恒为 0（本例）——async 才有第二本账（右）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m2_ledger.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
