#!/usr/bin/env python3
"""ch34 机制图 · random kernel 接受判据 p_t/p_d ≥ u（figure_spec fig_m9_random_criterion，模板 state-table）

放大自 L0 采样列 spec 块验证期的 random 路径——L2 章图第⑧拍内 random kernel 分支
（算法心脏）的机制小图（架构归属回指 L2，不另立第二种架构画法）。

claim：random kernel 接受判据 p_t/p_d ≥ u：比值 0.5 的草稿在六枚骰子下 4 收 2 拒、
比值 4.0 恒通过——min(1,·) 由 u∈[0,1) 隐式实现，拒绝位写 recovered 并早停。

勘误已收口（盲审 FAIL 后按 suggested_fix 修 explainer 侧）：explainer m9 的 claim/
quantified/caption_draft 曾写「恰半数通过（3/6）/恰 3 收 3 拒」、numbers 漏列 seed 46 的
u=0.149175，现已按唯一 provenance（traces/ch34_m09_random_kernel.json part_a_exact_walk）
的实测改为 4 收 2 拒并补全 u 清单——图自始按 trace 画、无需改动内容，仅删去页脚对旧 spec
的勘误尾注；图与 spec 现一致。

数字全部取自 traces/ch34_m09_random_kernel.json（part_a_exact_walk / part_b_no_draft_probs /
part_c_statistical / part_d_float64）。坐标由常量/循环计算；文本全 esc()。

rev3（评审 figure-integration 空白带修复）：画布 1000→800——原内容止于 y≈630（三卡框底）、
页脚在 y≈953，中间约 323px 纯空白；收紧后内容→页脚间距 123px，对齐本章机制图族的
常规间距（97-169px）。纯版式收紧，内容/数字/措辞零改动。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 800
MX, BXR = 56, 1444
DEFS = lc.DEFS
ACC_C, REJ_C = lc.C_GPU_S, lc.C_ABORT

# ---------------- 标题区 ----------------
lc.text(MX, 34, '掷骰子验货：概率比值 ≥ 骰子值才收——min(1,·) 由 u∈[0,1) 隐式实现', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'random kernel 判据（rejection_sampler.py:L829）：draft_prob>0 且 target_prob/draft_prob ≥ uniform_prob · pos0 比值 0.5（p_t=0.3/p_d=0.6）、pos1 比值 4.0（0.8/0.2）',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列+spec 验证期（L2 ⑧ 内 random kernel 分支·算法心脏）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= A. 判据数轴（pos0 比值 0.5） =================
NLX, NLW, NLY = MX + 130, 1000, 130
def ux(v):
    return NLX + v * NLW

lc.text(MX, NLY - 12, 'pos0：比值 0.5', 10.5, lc.C_TXT, 'start', True, maxw=120, tag='nlt')
lc.text(MX, NLY + 4, 'p_t(1)=0.3 / p_d(1)=0.6', 8.2, lc.C_MUTE, 'start', maxw=124, tag='nlt2')
# 收/拒两段
lc.rect(ux(0), NLY, 0.5 * NLW, 26, lc.C_GPU_F, ACC_C, rx=0, sw=0)
lc.rect(ux(0.5), NLY, 0.5 * NLW, 26, '#fef2f2', REJ_C, rx=0, sw=0)
lc.text(ux(0.25), NLY + 17, '收（u < 0.5）', 9.2, ACC_C, 'middle', True, maxw=200, tag='z1')
lc.text(ux(0.75), NLY + 17, '拒（u ≥ 0.5）', 9.2, REJ_C, 'middle', True, maxw=200, tag='z2')
# 阈值线
lc.seg(ux(0.5), NLY - 10, ux(0.5), NLY + 56, lc.C_TXT, 1.8)
lc.text(ux(0.5), NLY + 68, '比值 0.5（=接受阈值）', 8.6, lc.C_TXT, 'middle', True, maxw=160, tag='thr')
# 六枚骰子（seed 42-47）
DICE = [(42, 0.468587, True), (43, 0.284905, True), (44, 0.086889, True),
        (45, 0.762339, False), (46, 0.149175, True), (47, 0.718814, False)]
for k, (seed, u, acc) in enumerate(DICE):
    x = ux(u)
    col = ACC_C if acc else REJ_C
    up_ = k in (1, 4)                            # 0.284905/0.149175 近邻——标签上下错位防撞
    lc.circle(x, NLY - 14, 7, col, 2.0, dash=False)
    lc.text(x, NLY - 44 if up_ else NLY - 26, 'seed ' + str(seed), 7.6, col, 'middle', True, maxw=70, tag='dl' + str(seed))
    lc.text(x, NLY + 40 + (12 if up_ else 0), 'u=' + str(u), 7.4, col, 'middle', maxw=80, tag='dv' + str(seed))
lc.text(NLX + NLW + 12, NLY + 17, '1', 9, lc.C_MUTE, 'start', maxw=20, tag='ax1')
lc.text(NLX - 12, NLY + 17, '0', 9, lc.C_MUTE, 'end', maxw=20, tag='ax0')
lc.text(ux(0.02), NLY + 68, 'u ~ U[0,1)（float64）', 8.0, lc.C_MUTE, 'start', maxw=150, tag='axu')

# ================= B. pos1：比值 4.0 恒收（第二条数轴） =================
NLY2 = NLY + 112
lc.text(MX, NLY2 - 12, 'pos1：比值 4.0', 10.5, lc.C_TXT, 'start', True, maxw=120, tag='n2t')
lc.text(MX, NLY2 + 4, 'p_t(1)=0.8 / p_d(1)=0.2', 8.2, lc.C_MUTE, 'start', maxw=124, tag='n2t2')
lc.rect(NLX, NLY2, NLW, 26, lc.C_GPU_F, ACC_C, rx=0, sw=0)
lc.text(NLX + NLW / 2, NLY2 + 17, '整段恒收：u∈[0,1) 永远小于 4.0 → min(1,·) 不用显式取', 9.2, ACC_C,
        'middle', True, maxw=600, tag='z3')
for u in [0.12625, 0.147737, 0.504142, 0.552894]:   # 只画真被消费的 4 枚（收了的 seed 42/43/44/46）
    x = ux(u)
    lc.circle(x, NLY2 - 14, 7, ACC_C, 2.0, dash=False)
lc.text(NLX + NLW + 12, NLY2 + 17, '1', 9, lc.C_MUTE, 'start', maxw=20, tag='ax1b')
lc.text(NLX - 12, NLY2 + 17, '0', 9, lc.C_MUTE, 'end', maxw=20, tag='ax0b')
lc.text(NLX + NLW / 2, NLY2 - 26, '（4 枚真被消费的 pos1 u 值全过；45/47 在 pos0 早停——pos1 的 u 不消费）', 7.6, lc.C_MUTE,
        'middle', maxw=430, tag='z3n')

# ================= C. 六种子走表 =================
TY0 = NLY2 + 64
TX, TY = MX, TY0
ROWS = [
    ('42', '0.5 ≥ 0.468587 → 收', '比值 4.0 恒收（u=0.12625 也过）', '[1,1,777]', '3 token'),
    ('43', '0.5 ≥ 0.284905 → 收', '恒收（u=0.147737）', '[1,1,777]', '3 token'),
    ('44', '0.5 ≥ 0.086889 → 收', '恒收（u=0.504142）', '[1,1,777]', '3 token'),
    ('45', '0.5 < 0.762339 → 拒', '不到（早停）', '[3,-1,-1]', '1 token'),
    ('46', '0.5 ≥ 0.149175 → 收', '恒收（u=0.552894）', '[1,1,777]', '3 token'),
    ('47', '0.5 < 0.718814 → 拒', '不到（早停）', '[3,-1,-1]', '1 token'),
]
TW = 60 + 240 + 210 + 150 + 100 + 60
lc.rect(TX, TY, TW, 34 + len(ROWS) * 26, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
hdr = ['seed', 'pos0 判据（0.5 ≥ u ?）', 'pos1（若到）', '输出行', '本拍产出']
cxs = [TX + 30, TX + 60 + 120, TX + 300 + 105, TX + 510 + 75, TX + 660 + 50]
for h_, cx_ in zip(hdr, cxs):
    lc.text(cx_, TY + 20, h_, 8.8, lc.C_MUTE, 'middle', True, maxw=220, tag='th' + h_[:6])
lc.seg(TX, TY + 30, TX + TW, TY + 30, lc.C_MUTE, 1.0)
for i, (sd, p0, p1, out, prod) in enumerate(ROWS):
    ry = TY + 34 + i * 26 + 14
    rej = '拒' in p0
    col = REJ_C if rej else ACC_C
    lc.text(cxs[0], ry, sd, 9, lc.C_TXT, 'middle', True, maxw=40, tag='s' + sd)
    lc.text(cxs[1], ry, p0, 8.8, col, 'middle', True, maxw=230, tag='p0' + sd)
    lc.text(cxs[2], ry, p1, 8.2, '#475569', 'middle', maxw=200, tag='p1' + sd)
    lc.text(cxs[3], ry, out, 8.8, col, 'middle', True, maxw=140, tag='o' + sd)
    lc.text(cxs[4], ry, prod, 8.4, '#475569', 'middle', maxw=90, tag='pr' + sd)
lc.text(TX + TW + 16, TY + 20, '六枚骰子 pos0：4 收 2 拒', 9.6,
        lc.C_TXT, 'start', True, maxw=250, tag='sum1')
lc.text(TX + TW + 16, TY + 36, '（seed 42/43/44/46 收、45/47 拒）', 8.0, lc.C_MUTE, 'start', maxw=250, tag='sum1b')
lc.text(TX + TW + 16, TY + 54, '「约半数」是统计性质：2 万样本', 8.2,
        lc.C_MUTE, 'start', maxw=250, tag='sum2')
lc.text(TX + TW + 16, TY + 69, '实测 0.4966 ≈ 理论 0.5——六枚的小样本', 8.2,
        lc.C_MUTE, 'start', maxw=250, tag='sum3')
lc.text(TX + TW + 16, TY + 84, '波动正常。拒绝位写 recovered（下一张图）、其后早停全作废。', 8.2,
        lc.C_MUTE, 'start', maxw=250, tag='sum4')

# ================= D. 底部三卡：ngram 退化 / 统计律 / float64 =================
DY0 = TY + 34 + len(ROWS) * 26 + 16
CARDW = (BXR - MX - 2 * 20) / 3
cards = [
    ('ngram 退化（NO_DRAFT_PROBS）', lc.C_ZMQ_S,
     ['draft_prob=1 → 判据退化为 p_t(x) ≥ u', '本例 p_t/1=0.3 vs u=0.468587 → 拒', '写 recovered=0，输出 [0,-1]，1 token']),
    ('统计律（20000 样本实测）', ACC_C,
     ['比值 0.5 → 实测 0.4966（理论 0.5）', '比值 4.0 → 实测 1.0（理论 1）', 'ngram p_t=0.3 → 0.30125', '接受率 = min(1, p_t/p_d)']),
    ('u 为什么用 float64', REJ_C,
     ['float32 有非平凡概率掷出精确 0.0', '判据是 ≥：u=0.0 时 p_t/p_d ≥ 0 恒真', '→ 坏草稿被无条件放行（pytorch#16706）', '例 u=0.468587（6 位精度）']),
]
for i, (ttl, col, lines) in enumerate(cards):
    cx_ = MX + i * (CARDW + 20)
    lc.rect(cx_, DY0, CARDW, 118, '#ffffff', col, rx=9, sw=1.5)
    lc.text(cx_ + 14, DY0 + 22, ttl, 10, col, 'start', True, maxw=CARDW - 28, tag='c' + str(i))
    for j, ln in enumerate(lines):
        lc.text(cx_ + 14, DY0 + 42 + j * 17, '· ' + ln, 8.2, '#475569', 'start', maxw=CARDW - 26, tag='cl' + str(i) + str(j))

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/sample/rejection_sampler.py:L772-L845（random kernel；判据 L829）· L811-L817（NO_DRAFT_PROBS 分支）· L608-L660（float64 uniform，注释引 pytorch#16706）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, '六种子 u/输出行/统计律/ngram ＝ seeded generator 重构（与 kernel 输出反向核验一致）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m9_random_criterion.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
