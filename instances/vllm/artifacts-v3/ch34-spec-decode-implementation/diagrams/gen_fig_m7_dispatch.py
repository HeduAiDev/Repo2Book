#!/usr/bin/env python3
"""ch34 机制图 · rejection_sample 的调度：预填/all_greedy 早退/双 kernel 同批（figure_spec fig_m7_dispatch，模板 flow）

放大自 L0 采样列 spec 块验证期的调度——L2 章图第⑧拍（rejection_sample 双 kernel）的
机制小图（架构归属回指 L2，不另立第二种架构画法）。

claim：rejection_sample 的调度：输出 buffer 预填 -1（第二维 max_spec_len+1，+1 即
bonus 槽）、all_greedy 一趟 greedy kernel 即返回、混批双 kernel 同批共存各自早退。

数字全部取自 figure_spec.numbers（traces/ch34_m07_dispatch.json：buffer_account /
case_b_mixed.params.is_greedy_mask / req1 u=0.958787 p_t=0.128571 /
case_a generator_state_unchanged；两例输出矩阵同 trace）。坐标由常量/循环计算。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 950
MX, BXR = 56, 1444
DEFS = lc.DEFS
PH_C = '#cbd5e1'          # -1 哨兵灰
VALID_C = lc.C_GPU_S

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'greedy 与随机同批共存：is_greedy mask 行级分流，批不用拆', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'rejection_sample 的调度：输出 buffer [batch, max_spec_len+1] 预填 PLACEHOLDER=-1（+1 即 bonus 槽）· all_greedy 一趟 greedy kernel 即返回 · 混批双 kernel 同批 launch、各自早退',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列+spec 验证期（L2 ⑧ rejection_sample 双 kernel 步进图）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 顶部：入口 + 预填 buffer =================
EX, EY, EW_, EH_ = MX, 96, 640, 190
lc.rect(EX, EY, EW_, EH_, '#ffffff', lc.C_MUTE, rx=10, sw=1.5)
lc.text(EX + 16, EY + 24, '入口：先预填，再分流', 12, lc.C_TXT, 'start', True, maxw=EW_ - 32, tag='et')
# buffer 2×3
GX, GY, GCW, GCH = EX + 230, EY + 44, 64, 40
lc.text(EX + 16, GY + 14, '输出 buffer', 9.6, lc.C_TXT, 'start', True, maxw=90, tag='bl')
lc.text(EX + 16, GY + 30, '[2, max_spec_len+1]', 8.4, lc.C_MUTE, 'start', maxw=110, tag='bl2')
lc.text(EX + 16, GY + 46, '=[2,3]', 8.4, lc.C_MUTE, 'start', maxw=90, tag='bl3')
for r in range(2):
    for c in range(3):
        x, y = GX + c * GCW, GY + 14 + r * GCH
        lc.rect(x + 1, y + 1, GCW - 2, GCH - 2, '#f1f5f9', PH_C, rx=4, sw=1.0)
        lc.text(x + GCW / 2, y + GCH / 2 + 4, '-1', 10.5, '#64748b', 'middle', True, maxw=GCW - 6, tag='g' + str(r) + str(c))
lc.text(GX + 1.5 * GCW, GY + 8, '← +1 即 bonus 槽', 8.2, lc.C_SAM_S, 'middle', True, maxw=140, tag='plus1')
lc.text(EX + 16, GY + 122, '未写位保持哨兵 -1：kernel 内首拒后零写入，', 8.4, '#475569', 'start', maxw=300, tag='pre1')
lc.text(EX + 16, GY + 137, '-1 截断与浪费共用一个记号（parse_output 一次过滤）', 8.4, '#475569', 'start', maxw=300, tag='pre2')

# 分流菱形（简化为判定框）
DX, DY = EX + EW_ + 60, EY + 55
lc.rect(DX, DY, 200, 60, '#ffffff', lc.C_ENG_S, rx=9, sw=1.8)
lc.text(DX + 100, DY + 24, 'all_greedy ？', 10.5, lc.C_ENG_S, 'middle', True, maxw=180, tag='dq')
lc.text(DX + 100, DY + 42, '（温度全 0）', 8.2, lc.C_MUTE, 'middle', maxw=180, tag='dq2')
lc.seg(EX + EW_, EY + 85, DX, DY + 30, lc.C_MUTE, 1.8, 'std')

# ================= 左：case A 全 greedy =================
AY0, AH_ = EY + EH_ + 46, 366
AW_ = 640
lc.rect(MX, AY0, AW_, AH_, '#ffffff', VALID_C, rx=10, sw=1.8)
lc.text(MX + 16, AY0 + 24, 'case A：全 greedy（2 请求 num_draft=[2,1]，all_greedy=True）', 11.5,
        VALID_C, 'start', True, maxw=AW_ - 32, tag='at')
lc.seg(DX + 40, DY + 60, DX + 40, AY0 + 44, VALID_C, 1.8, 'std')
lc.seg(DX + 40, AY0 + 44, MX + 210, AY0 + 44, VALID_C, 1.8)
lc.text(DX + 46, DY + 74, 'True', 8.6, VALID_C, 'start', True, maxw=40, tag='tlab')
# 流程行
FX, FY = MX + 40, AY0 + 62
flow_a = [('greedy kernel', '一趟出结果'), ('→ 直接 return', 'softmax / uniform / recovered 全跳过')]
cx_ = FX
for i, (hd, sb) in enumerate(flow_a):
    w_ = 190
    lc.rect(cx_, FY, w_, 46, '#ffffff', VALID_C, rx=7, sw=1.4)
    lc.text(cx_ + w_ / 2, FY + 20, hd, 9.8, VALID_C, 'middle', True, maxw=w_ - 10, tag='fa' + str(i))
    lc.text(cx_ + w_ / 2, FY + 36, sb, 7.9, '#475569', 'middle', maxw=w_ - 10, tag='fb' + str(i))
    if i < len(flow_a) - 1:
        lc.seg(cx_ + w_, FY + 23, cx_ + w_ + 24, FY + 23, VALID_C, 1.6, 'std')
    cx_ += w_ + 24
# 随机数账
lc.text(MX + 470, FY + 16, '随机数消耗：0 个', 9.6, lc.C_ABORT, 'start', True, maxw=150, tag='rng0')
lc.text(MX + 470, FY + 32, 'generator 状态不变（实证）', 8.2, lc.C_MUTE, 'start', maxw=150, tag='rng0b')
# 输出矩阵
OGX, OGY = MX + 40, FY + 66
OUTA = [[3, 5, 99], [1, 98, -1]]
for r in range(2):
    for c in range(3):
        x, y = OGX + c * 56, OGY + r * 36
        v = OUTA[r][c]
        ph = v == -1
        lc.rect(x + 1, y + 1, 54, 34, '#f1f5f9' if ph else lc.C_GPU_F, PH_C if ph else VALID_C, rx=4, sw=1.0)
        lc.text(x + 28, y + 22, str(v), 10, '#64748b' if ph else lc.C_TXT, 'middle', True, maxw=48, tag='oa' + str(r) + str(c))
lc.text(OGX + 3 * 56 + 14, OGY + 22, '有效 5 格 / 余 1 格 -1', 8.8, '#475569', 'start', maxw=180, tag='oan')
lc.text(OGX + 3 * 56 + 14, OGY + 40, '（第 2 请求只 1 草稿，bonus 后无位可写）', 7.9, lc.C_MUTE, 'start', maxw=210, tag='oan2')
lc.text(MX + 16, AY0 + AH_ - 14, '1 次 kernel launch——连 softmax 都不用算（greedy 只要 argmax）', 8.6, lc.C_MUTE, 'start', maxw=AW_ - 32, tag='af')

# ================= 右：case B 混批 =================
BXB = MX + AW_ + 60
BWB = BXR - BXB
lc.rect(BXB, AY0, BWB, AH_, '#ffffff', lc.C_ZMQ_S, rx=10, sw=1.8)
lc.text(BXB + 16, AY0 + 24, 'case B：混批（is_greedy=[True,False]，温度 [0.0,1.0]）', 11.5,
        lc.C_ZMQ_S, 'start', True, maxw=BWB - 32, tag='bt')
lc.seg(DX + 150, DY + 60, DX + 150, AY0 + 44, lc.C_ZMQ_S, 1.8, 'std')
lc.seg(DX + 150, AY0 + 44, BXB + 60, AY0 + 44, lc.C_ZMQ_S, 1.8)
lc.text(DX + 156, DY + 74, 'False', 8.6, lc.C_ZMQ_S, 'start', True, maxw=44, tag='flab')
# 预处理行
lc.text(BXB + 16, AY0 + 62, 'softmax 出 target_probs → 先采好全部草稿位的 recovered → 双 kernel 同批 launch：', 8.8,
        '#475569', 'start', maxw=BWB - 32, tag='bpre')
# 双 kernel 盒（左列堆叠）
K1X, KY, KW, KH = BXB + 24, AY0 + 84, 240, 118
K2Y = KY + KH + 14
lc.rect(K1X, KY, KW, KH, '#ffffff', VALID_C, rx=8, sw=1.5)
lc.text(K1X + 12, KY + 20, 'greedy kernel', 10, VALID_C, 'start', True, maxw=KW - 24, tag='k1')
for i, ln in enumerate(['req0（greedy）：正常判 draft==argmax', 'req1（随机）：is_greedy=False 早退', '产出第 0 行 [3,7,-1]']):
    lc.text(K1X + 12, KY + 40 + i * 16, '· ' + ln, 8.2, '#475569', 'start', maxw=KW - 22, tag='k1l' + str(i))
lc.rect(K1X, K2Y, KW, KH, '#ffffff', lc.C_ZMQ_S, rx=8, sw=1.5)
lc.text(K1X + 12, K2Y + 20, 'random kernel', 10, lc.C_ZMQ_S, 'start', True, maxw=KW - 24, tag='k2')
for i, ln in enumerate(['req0（greedy）：is_greedy=True 早退', 'req1（随机）：走 p_t/p_d ≥ u 判据', '产出第 1 行 [3,-1,-1]']):
    lc.text(K1X + 12, K2Y + 40 + i * 16, '· ' + ln, 8.2, '#475569', 'start', maxw=KW - 22, tag='k2l' + str(i))
# 判据盒 + 输出矩阵 B（右列）
JX = K1X + KW + 20
JW = BXR - 20 - JX
JH = K2Y + KH - KY
lc.rect(JX, KY, JW, JH, '#f8fafc', lc.C_MUTE, rx=8, sw=1.0)
lc.text(JX + 14, KY + 22, 'req1 判据（NO_DRAFT_PROBS）：', 9.2, lc.C_TXT, 'start', True, maxw=JW - 28, tag='j1')
lc.text(JX + 14, KY + 40, 'p_t(2)=0.128571 ≥ u=0.958787 ?', 9.4, lc.C_ZMQ_S, 'start', True, maxw=JW - 28, tag='j2')
lc.text(JX + 14, KY + 56, '不成立 → 拒 → 写 recovered=3、早停', 8.6, lc.C_ABORT, 'start', True, maxw=JW - 28, tag='j3')
lc.text(JX + 14, KY + 72, 'u 是 float64 uniform（一位一个）', 8.0, lc.C_MUTE, 'start', maxw=JW - 28, tag='j4')
OBX, OBY = JX + 14, KY + 128
OUTB = [[3, 7, -1], [3, -1, -1]]
for r in range(2):
    for c in range(3):
        x, y = OBX + c * 48, OBY + r * 30
        v = OUTB[r][c]
        ph = v == -1
        lc.rect(x + 1, y + 1, 46, 28, '#f1f5f9' if ph else lc.C_GPU_F, PH_C if ph else VALID_C, rx=4, sw=1.0)
        lc.text(x + 24, y + 19, str(v), 9.2, '#64748b' if ph else lc.C_TXT, 'middle', True, maxw=40, tag='ob' + str(r) + str(c))
NX_ = OBX + 3 * 48 + 12
lc.text(NX_, OBY + 14, '有效 3 格 / 余 3 格 -1', 8.4, '#475569', 'start', maxw=JW - 28 - (3 * 48 + 12), tag='obn')
lc.text(NX_, OBY + 30, '（两行都在首拒/全拒后截断）', 7.7, lc.C_MUTE, 'start', maxw=JW - 28 - (3 * 48 + 12), tag='obn2')
lc.text(NX_, OBY + 48, '对照拆批跑两遍：省一次往返，行级分流让批形状不动。', 7.7, lc.C_MUTE, 'start',
        maxw=JW - 28 - (3 * 48 + 12), tag='obn3')
lc.text(BXB + 16, AY0 + AH_ - 14, '2 次 kernel 同批 launch——随机数只花在 1 个随机请求上（每草稿位 1 个 float64 uniform + 每请求 1 行 q）', 8.6,
        lc.C_MUTE, 'start', maxw=BWB - 32, tag='bf2')

# ================= 底部 buffer 账 =================
BY0 = AY0 + AH_ + 20
lc.rect(MX, BY0, BXR - MX, 66, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 18, BY0 + 26, 'buffer 账：[2,3] 共 6 格——case A 有效 5 格、余 1 格 -1；case B 有效 3 格、余 3 格（-1 哨兵=截断+浪费）。',
        9.8, lc.C_TXT, 'start', maxw=BXR - MX - 36, tag='bufa')
lc.text(MX + 18, BY0 + 46, '第二维的 +1 就是 bonus 槽：全收时白嫖的那一格（case A 第 1 行的 99、第 2 行的 98 都写在第 k 位）。',
        8.8, lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='bufb')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/sample/rejection_sampler.py:L394-L507（rejection_sample 调度）· L413-L419（buffer 预填 -1）· L421-L449（all_greedy 早退）· '
                    'L424（is_greedy mask）· L730-L733 / L789-L792（两 kernel 各自早退）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, 'buffer 账 / is_greedy=[True,False] / u=0.958787、p_t=0.128571（float64 重构反向核验）/ 随机数 0 个 / 两例输出矩阵 ＝ 本章驱动脚本真跑实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m7_dispatch.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
