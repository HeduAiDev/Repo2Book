#!/usr/bin/env python3
"""ch34 机制图 · 变长草稿摊平 + 二次 gather 错位一格（figure_spec fig_m4_flatten，模板 tensor-flow）

放大自 L0 采样列+spec 块『target 一次前向』输入侧——L2 章图第⑤拍
（_calc_spec_decode_metadata）的机制小图（步进图层）。

claim：变长草稿摊平成一条输入流：3 请求草稿 [2,0,1] → 6 行扁平 logits，三组 index
（logits/target/bonus）定位每行，draft_token_ids 由 input_ids 经 logits_indices 再经
target_logits_indices+1 二次 gather 错位一格取出 [61,72,55]。

数字全部取自 figure_spec.numbers（traces/ch34_m04_index_flattening.json example_b_small
+ counts）。坐标由常量/循环计算；文本全 esc()。

rev3（评审 figure-integration 空白带修复）：画布 1020→560、右侧 padding 对照条
PH_ 380→262（原面板内容止于 y≈352、框底 488 内部 136px 空白）——纯版式收紧，
内容/数字/措辞零改动。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 560
MX, BXR = 56, 1444
DEFS = lc.DEFS

REQC = ['#2563eb', '#0891b2', '#16a34a']          # req0 蓝 / req1 青 / req2 绿（区分类，非角色色）
REQF = ['#eff6ff', '#ecfeff', '#f0fdf4']

# ---------------- 标题区 ----------------
lc.text(MX, 34, '摊平不 padding：变长草稿首尾相接成一条输入流，三组 index 定位每一行', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '例 B（3 请求，草稿 [2,0,1]，cu_num_scheduled=[3,5,8]）· 零草稿请求只占 1 个 bonus 位 · '
               '每请求 k_i+1 个采样位 → 6 行扁平 logits（= num_tokens 3 + batch 3）',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列+spec 输入侧（L2 ⑤ _calc_spec_decode_metadata 步进图）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= A. input_ids 输入流（8 位） =================
INPUT_IDS = [50, 61, 72, 10, 20, 30, 41, 55]
SEG = [(0, 3), (3, 5), (5, 8)]                    # 请求区间 [cu_{i-1}, cu_i)
LOGITS_POS = [0, 1, 2, 4, 6, 7]
AX0, AY0, CW_, CH_ = MX + 96, 108, 108, 54
lc.text(MX, AY0 + 16, 'input_ids', 10.5, lc.C_TXT, 'start', True, maxw=90, tag='ailab')
lc.text(MX, AY0 + 32, '（8 位）', 8.4, lc.C_MUTE, 'start', maxw=90, tag='ailab2')
# 请求括板（上方）
for i, (s, e) in enumerate(SEG):
    bx = AX0 + s * CW_
    lc.rect(bx + 2, AY0 - 30, (e - s) * CW_ - 4, 22, REQF[i], REQC[i], rx=5, sw=1.2)
    lbl = ['req0（草稿 2）', 'req1（零草稿）', 'req2（草稿 1）'][i]
    lc.text(bx + ((e - s) * CW_) / 2, AY0 - 15, lbl, 9.2, REQC[i], 'middle', True,
            maxw=(e - s) * CW_ - 10, tag='rq' + str(i))
# 单元格
for j, v in enumerate(INPUT_IDS):
    x = AX0 + j * CW_
    i = next(k for k, (s, e) in enumerate(SEG) if s <= j < e)
    has_logits = j in LOGITS_POS
    lc.rect(x + 2, AY0, CW_ - 4, CH_, REQF[i] if has_logits else '#ffffff', REQC[i],
            rx=5, sw=1.5 if has_logits else 1.0, dash=not has_logits)
    lc.text(x + CW_ / 2, AY0 + 26, str(v), 13, lc.C_TXT, 'middle', True, maxw=CW_ - 8, tag='c' + str(j))
    lc.text(x + CW_ / 2, AY0 + 46, '位 ' + str(j), 8.2, lc.C_FAINT, 'middle', maxw=CW_ - 8, tag='p' + str(j))
    # 角色：草稿位 / 补喂位
    role = ['补喂·采样', '草稿 d1', '草稿 d2', '补喂', '采样·bonus', '补喂', '补喂·采样', '草稿 d1'][j]
    lc.text(x + CW_ / 2, AY0 + CH_ + 16, role, 8.0, lc.C_MUTE, 'middle', maxw=CW_ - 4, tag='ro' + str(j))

# ================= B. 摊平：logits_indices 选位下行 =================
BY0 = AY0 + CH_ + 40
FLAT_X0, FLAT_CW, FLAT_CH = AX0 + 60, 96, 40
flat_vals = [INPUT_IDS[p] for p in LOGITS_POS]     # [50,61,72,20,41,55] = first_hop
ROW_OF_POS = {p: r for r, p in enumerate(LOGITS_POS)}
for p in LOGITS_POS:
    x_src = AX0 + p * CW_ + CW_ / 2
    r = ROW_OF_POS[p]
    x_dst = FLAT_X0 + r * FLAT_CW + FLAT_CW / 2
    lc.seg(x_src, AY0 + CH_ + 26, x_dst, BY0 - 4, '#94a3b8', 1.1)
lc.text(MX, BY0 - 22, 'logits_indices', 9.6, lc.C_TXT, 'start', True, maxw=150, tag='lig')
lc.text(MX, BY0 - 8, '= [0,1,2,4,6,7]', 9.6, lc.C_TXT, 'start', True, maxw=150, tag='lig1')
lc.text(MX, BY0 + 8, '（np.repeat(cu−num_sampled)', 7.6, lc.C_MUTE, 'start', maxw=150, tag='lig2')
lc.text(MX, BY0 + 20, ' + arange 段内偏移）', 7.6, lc.C_MUTE, 'start', maxw=150, tag='lig3')
for r, v in enumerate(flat_vals):
    x = FLAT_X0 + r * FLAT_CW
    i = next(k for k, (s, e) in enumerate(SEG) if s <= LOGITS_POS[r] < e)
    lc.rect(x + 2, BY0, FLAT_CW - 4, FLAT_CH, REQF[i], REQC[i], rx=4, sw=1.4)
    lc.text(x + FLAT_CW / 2, BY0 + 24, str(v), 11.5, lc.C_TXT, 'middle', True, maxw=FLAT_CW - 8, tag='f' + str(r))
    lc.text(x + FLAT_CW / 2, BY0 + FLAT_CH + 13, '行 ' + str(r), 8.0, lc.C_FAINT, 'middle', maxw=FLAT_CW - 6, tag='fr' + str(r))
lc.text(FLAT_X0 + 6 * FLAT_CW + 16, BY0 + 16, '6 行扁平 logits', 10, lc.C_TXT, 'start', True, maxw=120, tag='fl6')
lc.text(FLAT_X0 + 6 * FLAT_CW + 16, BY0 + 32, '[num_tokens+batch, vocab]', 8.0, lc.C_MUTE, 'start', maxw=140, tag='fl7')
lc.text(FLAT_X0 + 6 * FLAT_CW + 16, BY0 + 48, 'first_hop = input_ids[logits_indices]', 7.6, lc.C_FAINT,
        'start', maxw=250, tag='fh')
lc.text(FLAT_X0 + 6 * FLAT_CW + 16, BY0 + 64, '← 补喂位 3、5 不产 logits 行（虚线格）', 8.0, lc.C_MUTE,
        'start', maxw=250, tag='feedonly')

# 三组 index 底线标记
BONUS = [2, 3, 5]
TARGET = [0, 1, 4]
for r in BONUS:
    x = FLAT_X0 + r * FLAT_CW
    lc.rect(x + 6, BY0 - 12, FLAT_CW - 12, 6, lc.C_SAM_S, lc.C_SAM_S, rx=2, sw=0)
for r in TARGET:
    x = FLAT_X0 + r * FLAT_CW
    lc.rect(x + 6, BY0 - 12, FLAT_CW - 12, 6, lc.C_KV_S, lc.C_KV_S, rx=2, sw=0)
lc.text(MX, BY0 + FLAT_CH + 34, '■ bonus_logits_indices = [2,3,5]（每请求最后一行=bonus 位）   ■ target_logits_indices = [0,1,4]（验证草稿的行）',
        8.8, lc.C_MUTE, 'start', maxw=900, tag='idx')

# ================= C. 二次 gather：错位一格 =================
CY0 = BY0 + FLAT_CH + 62
lc.text(MX, CY0 - 12, '二次 gather（错位一格）：draft_token_ids = first_hop[target_logits_indices + 1]', 10.5,
        lc.C_TXT, 'start', True, maxw=700, tag='sg')
# target 行 r → +1 后是 first_hop 的行号 r+1（两跳：input_ids[logits_indices][target_logits_indices+1]，
# gpu_model_runner.py L2913-L2914；+1 索引的是 first_hop 行，不是 input_ids 位——位 5=30 是补喂位，恰不产 logits）
TGT_PLUS1 = [1, 2, 5]
DRAFTS = [61, 72, 55]
for k, (r, p1) in enumerate(zip(TARGET, TGT_PLUS1)):
    x_src = FLAT_X0 + r * FLAT_CW + FLAT_CW / 2
    x_pos = AX0 + LOGITS_POS[p1] * CW_ + CW_ / 2   # 该行对应的输入位（回指线索，不画穿层线）
    # 结果盒
    bx = MX + 430 + k * 120
    lc.seg(x_src, CY0 + 4, bx + 60 - 60, CY0 + 30, lc.C_KV_S, 1.6, 'std')
    lc.rect(bx, CY0 + 12, 104, 40, '#ffffff', lc.C_KV_S, rx=6, sw=1.6)
    lc.text(bx + 52, CY0 + 31, '+1 → 行 ' + str(p1) + ' → ' + str(flat_vals[p1]), 8.8, lc.C_KV_S,
            'middle', True, maxw=100, tag='dk' + str(k))
    # 回指虚线到 input_ids 对应格（穿层太多则省略——改为文字回指）
lc.text(MX + 430 + 3 * 120 + 20, CY0 + 32, '＝ draft_token_ids [61, 72, 55]', 10.5, lc.C_TXT, 'start', True,
        maxw=220, tag='dres')
lc.text(MX, CY0 + 74, '错位一格的含义：草稿填在输入流里、跟在采样位后面——验证 d_i 用的是『预测它之后那个词』的 logits 行，所以第二跳在 first_hop 里取行 r+1'
                     '（＝下一采样位的输入 token）恰好取回草稿本身。',
        8.8, lc.C_MUTE, 'start', maxw=1330, tag='off1')

# ================= 右侧：padding 对照 =================
PX0, PY0, PW_, PH_ = MX + 1010, 108, BXR - (MX + 1010), 262
lc.rect(PX0, PY0, PW_, PH_, '#ffffff', lc.C_MUTE, rx=10, sw=1.4)
lc.text(PX0 + 16, PY0 + 24, '为什么摊平：padding 浪费账', 11.5, lc.C_TXT, 'start', True, maxw=PW_ - 32, tag='pt')
ROWS = [
    ('例 B（本图，3 请求）', [('摊平', 6, lc.C_GPU_S), ('padding 3×(2+1)', 9, '#94a3b8')], '省 3 行'),
    ('例 A（源码 docstring，5 请求）', [('摊平', 11, lc.C_GPU_S), ('padding 5×4', 20, '#94a3b8')], '省 9 行'),
]
ry = PY0 + 44
for name, bars, note_ in ROWS:
    lc.text(PX0 + 16, ry + 12, name, 9.2, lc.C_TXT, 'start', True, maxw=PW_ - 32, tag='pn' + name[:6])
    bx, bw_unit = PX0 + 100, 7.4
    for lbl, n, col in bars:
        lc.rect(bx, ry + 22, n * bw_unit, 20, '#ffffff' if col == '#94a3b8' else lc.C_GPU_F, col, rx=3, sw=1.2)
        lc.text(bx + n * bw_unit / 2, ry + 35, str(n) + ' 行', 8.6, col, 'middle', True, maxw=n * bw_unit - 4, tag='pb' + str(n))
        lc.text(bx, ry + 56, lbl, 7.8, lc.C_MUTE, 'start', maxw=120, tag='pl' + lbl[:6])
        bx += n * bw_unit + 26
    lc.text(PX0 + PW_ - 16, ry + 35, note_, 9.6, lc.C_GPU_S, 'end', True, maxw=70, tag='psv' + note_[:4])
    ry += 78
lc.text(PX0 + 16, ry + 10, '零草稿请求只占 1 个 bonus 位——零浪费', 8.8, '#475569', 'start', maxw=PW_ - 32, tag='pz1')
lc.text(PX0 + 16, ry + 26, '是省算力的主力；代价是三组 index 的间接', 8.8, '#475569', 'start', maxw=PW_ - 32, tag='pz2')
lc.text(PX0 + 16, ry + 42, '定位 + 错位一格的二次 gather。', 8.8, '#475569', 'start', maxw=PW_ - 32, tag='pz3')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/worker/gpu_model_runner.py:L2851-L2924（_calc_spec_decode_metadata·docstring 例 A 逐项互核）· '
                    'L1743-L1767（_get_cumsum_and_arange）· vllm/v1/worker/gpu_input_batch.py:L503-L532（update_req_spec_token_ids）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, '草稿 [2,0,1] / cu=[3,5,8] / input_ids 8 位 / logits_indices=[0,1,2,4,6,7] / bonus=[2,3,5] / target=[0,1,4]（+1 后 [1,2,5]）/ [61,72,55] / 6 vs 9 行 / 11 vs 20 行 ＝ 本章驱动脚本逐字复刻算式实测',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 10, '行号基线 vLLM v0.27.1', 8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m4_flatten.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
