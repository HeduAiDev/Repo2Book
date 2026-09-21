#!/usr/bin/env python3
"""ch34 机制图 · bonus 缝：一份扁平 logits 切两刀（figure_spec fig_m5_bonus_seam，模板 flow）

放大自 L0 采样列 spec 块『验证期采样』——L2 章图第⑦拍（RejectionSampler.forward）的
机制小图（架构归属回指 L2，不另立第二种架构画法）。

claim：v0.27 的 RejectionSampler 组合持有普通 Sampler：同一份扁平 logits 切两刀——
bonus 位 logits[bonus_logits_indices] 外采（max_num_logprobs=-1、可 top_p/top_k），
target 位 logits[target_logits_indices] 转 fp32、非 processed 模式 clone 保 raw 后走
约束与拒绝采样；张量索引产生新存储，两条支线 in-place 互不污染。

数字全部取自 figure_spec.numbers（bonus=[2,3,5] / target=[0,1,4] ＝
traces/ch34_m04_index_flattening.json example_b_small；max_num_logprobs=-1 与 docstring
策略口径 ＝ vllm/v1/sample/rejection_sampler.py:L137 / L38-L59）。坐标由常量/循环计算。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 900
MX, BXR = 56, 1444
DEFS = lc.DEFS
REQC = ['#2563eb', '#0891b2', '#16a34a']
REQF = ['#eff6ff', '#ecfeff', '#f0fdf4']

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'bonus 缝：同一份扁平 logits 切两刀——一行外采、一行自采', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'RejectionSampler 组合持有普通 Sampler（v0.27 构造注入）· 例 B：6 行扁平 logits，bonus 行 [2,3,5] 外包隔壁 Sampler，target 行 [0,1,4] 自己消化',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列+spec 验证期（L2 ⑦ RejectionSampler.forward 步进图）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 顶部：扁平 logits =================
LX, LY, LRH = MX + 320, 96, 34
ROWSEG = [(0, 3, 0), (3, 4, 1), (4, 6, 2)]      # (行起, 行止, 请求)
LROWS = 6
LH_ = LROWS * LRH
lc.rect(LX, LY, 250, LH_, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX, LY + LH_ / 2 - 10, '扁平 logits', 11, lc.C_TXT, 'start', True, maxw=120, tag='ll')
lc.text(MX, LY + LH_ / 2 + 8, '[6, vocab]', 9, lc.C_MUTE, 'start', maxw=120, tag='ll2')
lc.text(MX, LY + LH_ / 2 + 24, '（例 B）', 8.4, lc.C_FAINT, 'start', maxw=120, tag='ll3')
BONUS = [2, 3, 5]
TARGET = [0, 1, 4]
for r in range(LROWS):
    i = next(k for a, b, k in ROWSEG if a <= r < b)
    y = LY + r * LRH
    is_bonus = r in BONUS
    is_target = r in TARGET
    lc.rect(LX + 3, y + 2, 244, LRH - 4, REQF[i], REQC[i], rx=3, sw=1.0)
    lc.text(LX + 16, y + 22, '行 ' + str(r), 9.2, lc.C_TXT, 'start', True, maxw=44, tag='lr' + str(r))
    tagtxt = 'bonus 位' if is_bonus else ('target 位' if is_target else '')
    col = lc.C_SAM_S if is_bonus else (lc.C_GPU_S if is_target else lc.C_MUTE)
    lc.text(LX + 236, y + 22, tagtxt, 8.8, col, 'end', True, maxw=90, tag='lt' + str(r))
    # 位条（示意 vocab 维度）
    lc.seg(LX + 64, y + 17, LX + 150, y + 17, REQC[i], 3.0)

# ================= 左下：target 支线 =================
TY0 = LY + LH_ + 78
lc.seg(LX + 70, LY + LH_, LX + 70, TY0, lc.C_GPU_S, 2.4, 'std')
TX, TW_ = MX, 560
lc.rect(TX, TY0, TW_, 200, '#ffffff', lc.C_GPU_S, rx=10, sw=1.8)
lc.text(TX + 16, TY0 + 24, 'target 支线：自己消化', 12, lc.C_GPU_S, 'start', True, maxw=TW_ - 32, tag='tt')
STEPS_T = [
    ('① target_logits = logits[target_logits_indices]', '行 [0,1,4]——验证草稿的行'),
    ('② .float() 转 fp32', 'kernel 算 p_t/p_d 要数值稳定'),
    ('③ 非 processed 模式先 clone 保 raw', 'in-place 改的是副本，raw 留给 logprobs'),
    ('④ apply_logits_processors → rejection_sample', '约束 spec 特化 + 逐位判（下一张图）'),
]
for i, (hd, sb) in enumerate(STEPS_T):
    y = TY0 + 48 + i * 38
    lc.circle(TX + 26, y - 4, 9, lc.C_GPU_S, 1.6, dash=False)
    lc.text(TX + 26, y - 1, str(i + 1), 9, lc.C_GPU_S, 'middle', True, maxw=18, tag='tn' + str(i))
    lc.text(TX + 44, y, hd, 9.8, lc.C_TXT, 'start', True, maxw=TW_ - 60, tag='th' + str(i))
    lc.text(TX + 44, y + 14, sb, 8.2, lc.C_MUTE, 'start', maxw=TW_ - 60, tag='ts' + str(i))
    if i < len(STEPS_T) - 1:
        lc.seg(TX + 26, y + 6, TX + 26, y + 28, lc.C_GPU_S, 1.2)

# ================= 右下：bonus 支线 =================
BX_ = MX + 640
lc.seg(LX + 190, LY + LH_, LX + 190, TY0, lc.C_SAM_S, 2.4, 'std')
lc.seg(LX + 190, TY0, BX_ + 60, TY0, lc.C_SAM_S, 2.4)
BW_ = BXR - BX_
lc.rect(BX_, TY0, BW_, 200, '#ffffff', lc.C_SAM_S, rx=10, sw=1.8)
lc.text(BX_ + 16, TY0 + 24, 'bonus 支线：外包给组合持有的普通 Sampler', 12, lc.C_SAM_S, 'start', True, maxw=BW_ - 32, tag='bt')
STEPS_B = [
    ('① bonus_logits = logits[bonus_logits_indices]', '行 [2,3,5]——每请求最后一行（+1 的那个采样位）'),
    ('② replace(max_num_logprobs=-1)', 'logprobs 不在这条支线上算，后面统一补'),
    ('③ top_p / top_k 照常用', 'spec 主路径不支持的策略，bonus 位还能用'),
    ('④ 产 bonus token（全收时白嫖的那一格）', '外采结果写进输出 buffer 的第 k 列'),
]
for i, (hd, sb) in enumerate(STEPS_B):
    y = TY0 + 48 + i * 38
    lc.circle(BX_ + 26, y - 4, 9, lc.C_SAM_S, 1.6, dash=False)
    lc.text(BX_ + 26, y - 1, str(i + 1), 9, lc.C_SAM_S, 'middle', True, maxw=18, tag='bn' + str(i))
    lc.text(BX_ + 44, y, hd, 9.8, lc.C_TXT, 'start', True, maxw=BW_ - 60, tag='bh' + str(i))
    lc.text(BX_ + 44, y + 14, sb, 8.2, lc.C_MUTE, 'start', maxw=BW_ - 60, tag='bs' + str(i))
    if i < len(STEPS_B) - 1:
        lc.seg(BX_ + 26, y + 6, BX_ + 26, y + 28, lc.C_SAM_S, 1.2)

# 中缝：两条支线的隔离注记
MY_ = TY0 + 100
lc.rect(MX + 570, MY_ - 14, 64, 28, '#ffffff', lc.C_MUTE, rx=13, sw=1.2)
lc.text(MX + 602, MY_ + 4, '隔离', 9.4, lc.C_MUTE, 'middle', True, maxw=52, tag='iso')

# ================= 底部：为什么互不污染 =================
BY0 = TY0 + 224
lc.rect(MX, BY0, BXR - MX, 108, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 18, BY0 + 24, '为什么互不污染：张量索引（logits[idx]）产生新存储——两条支线各自拿到的是自己的副本，',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='w1')
lc.text(MX + 18, BY0 + 44, '后续一边被 in-place 改（clone/约束/温度），另一边的 raw 值不动。这是『in-place 所有权』暗线在 spec 侧的实证：切两刀即分家。',
        9.6, lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='w2')
lc.text(MX + 18, BY0 + 68, '同一份 logits · 两种命运：target 行决定『草稿收几个』，bonus 行决定『白嫖哪一个』。',
        9.6, lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='w3')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/sample/rejection_sampler.py:L92-L201（RejectionSampler.forward 切片）· L133-L147（组合持有普通 Sampler）· L137（replace(max_num_logprobs=-1)）· '
                    'L152-L160（fp32+clone 保 raw）· L38-L59（类 docstring：bonus 可 top_p/top_k）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, 'bonus=[2,3,5] / target=[0,1,4] ＝ 例 B（与摊平图同一组 index）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 10, '切片下标/策略口径逐字对 pin 源码核（L137 replace 与 L38-L59 docstring 原话 we can use top_p, top_k sampling for bonus tokens）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m5_bonus_seam.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
