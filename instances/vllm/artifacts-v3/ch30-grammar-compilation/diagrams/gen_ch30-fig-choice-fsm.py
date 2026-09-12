#!/usr/bin/env python3
"""ch30 机制图 · m6 choice 编译出的 FSM（figure_spec ch30-fig-choice-fsm，模板 state-machine）

放大自 L0 采样列·结构化输出组『语法→FSM』编译产物的机制展开（L2 章图站 4 的内部
视图），架构归属回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。
不画 xgrammar 库内部 CFG/adaptive 细节（dossier m6 note 边界：正文按 API 契约讲）。

claim：choice ["yes","no"] 编译出的 FSM 只需两个可观测状态+终态：位置 0 允许 5 个
前缀 token（n/y/no/ye/yes——一个 token 可一口吃多个字符），位置 1 只允许 EOS；
50257 的词表里每步合法集是极小集，这就是掩码每步要传的全部信息。

数字全部取自 figure_spec.numbers（trace fsm_positions：位置 0 允许集 5 id、位置 1 仅
50256、4242 拒收计数不动、yes→EOS 后 is_terminated=True 计数 2）。坐标由常量/循环
计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 748
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'choice 的状态机：位置 0 亮 5 个键（前缀闭包），位置 1 只剩 EOS', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'root ::= "yes" | "no" 编译出的 GrammarMatcher 按 token 接受——词表里有 no/ye/yes 这种多字符 token；'
               '4242（####）词表内、语法外，accept 拒收且计数不动',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L0 采样列·结构化输出组 · L2 站 4『编译成 FSM』内部视图'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 状态机主体（三状态横排） =================
R = 46
S_Y = 330
S0X, S1X, S2X = 250, 640, 1010
STATES = [
    (S0X, '位置 0', '起始（串未开吃）'),
    (S1X, '位置 1', '串已完整（yes/no）'),
]
for cx, name, sub in STATES:
    lc.circle(cx, S_Y, R, lc.C_SAM_S, 2.0, dash=False)
    lc.text(cx, S_Y - 4, name, 12.5, lc.C_TXT, 'middle', True, maxw=80, tag='st:' + name)
    lc.text(cx, S_Y + 16, sub, 8, lc.C_MUTE, 'middle', maxw=86, tag='sts:' + name)
# 起点标记
lc.circle(S0X - R - 26, S_Y, 5, lc.C_MUTE, 1.4, dash=False)
lc.seg(S0X - R - 19, S_Y, S0X - R, S_Y, lc.C_MUTE, 1.6, 'std')
# 终态：双圈 ◆
lc.circle(S2X, S_Y, R, lc.C_SAM_S, 2.0, dash=False)
lc.circle(S2X, S_Y, R - 7, lc.C_SAM_S, 1.4, dash=False)
lc.text(S2X, S_Y - 4, '终态', 12.5, lc.C_SAM_S, 'middle', True, maxw=70, tag='st:term')
lc.text(S2X, S_Y + 16, 'is_terminated=True', 8, lc.C_MUTE, 'middle', maxw=86, tag='sts:term')

# 位置 0 允许集（键清单，挂在状态 0 上方）
KX, KY, KW, KH = 108, 138, 430, 120
lc.rect(KX, KY, KW, KH, lc.C_SAM_F, lc.C_SAM_S, rx=9, sw=1.6)
lc.text(KX + 14, KY + 20, '位置 0 允许集（fill_bitmask 实测）＝choice 的前缀闭包', 9.5,
        lc.C_SAM_S, 'start', True, maxw=KW - 28, tag='k:t')
KEYS = [('n', '77'), ('y', '88'), ('no', '3919'), ('ye', '5948'), ('yes', '8505')]
kx = KX + 16
for name, tid in KEYS:
    kw = 34 + 9.2 * len(name)
    lc.rect(kx, KY + 32, kw, 32, '#ffffff', lc.C_SAM_S, rx=6, sw=1.3)
    lc.text(kx + kw / 2, KY + 45, f"'{name}'", 9.5, lc.C_TXT, 'middle', True, maxw=kw - 4, tag='k' + name)
    lc.text(kx + kw / 2, KY + 58, tid, 7.8, lc.C_MUTE, 'middle', maxw=kw, tag='k' + name + ':id')
    kx += kw + 8
lc.text(KX + 14, KY + 82, '共 5 个 / 50257——半路键 n/y 吃下后串未完（停在中间位、需续字符），'
                          '整词键 no/yes 一口吃完',
        8, '#334155', 'start', maxw=KW - 24, tag='k:l1')
lc.text(KX + 14, KY + 98, '词表里有 ye/no/yes 多字符 token——FSM 按 token 接受，不按字符',
        8, '#334155', 'start', maxw=KW - 24, tag='k:l2')
lc.seg(KX + KW / 2, KY + KH, KX + KW / 2, S_Y - R - 4, lc.C_MUTE, 1.2, dash=True)

# 位置 1 允许集（挂在状态 1 上方）
K2X, K2Y, K2W, K2H = 560, 158, 210, 100
lc.rect(K2X, K2Y, K2W, K2H, lc.C_SAM_F, lc.C_SAM_S, rx=9, sw=1.6)
lc.text(K2X + 14, K2Y + 20, '位置 1 允许集', 9.5, lc.C_SAM_S, 'start', True, maxw=K2W - 28, tag='k2:t')
lc.rect(K2X + 14, K2Y + 32, 118, 30, '#ffffff', lc.C_SAM_S, rx=6, sw=1.3)
lc.text(K2X + 73, K2Y + 45, "'<|endoftext|>'", 8.8, lc.C_TXT, 'middle', True, maxw=112, tag='k2:eos')
lc.text(K2X + 73, K2Y + 57, '50256', 7.8, lc.C_MUTE, 'middle', maxw=80, tag='k2:eosid')
lc.text(K2X + 14, K2Y + 82, '仅 1 个：串已完整，只差停机', 8, '#334155', 'start', maxw=K2W - 24, tag='k2:l1')
lc.seg(S1X, S_Y - R - 2, S1X, K2Y + K2H + 2, lc.C_MUTE, 1.2, dash=True)

# ---- 转移边 1：位置 0 → 位置 1（整词键） ----
ey0 = S_Y
lc.seg(S0X + R, ey0, S1X - R, ey0, lc.C_SAM_S, 2.2, 'sam')
lc.text((S0X + R + S1X - R) / 2, ey0 - 34, "accept_tokens([8505 'yes']) → True · 计数 0→1", 9.2,
        lc.C_SAM_S, 'middle', True, maxw=300, tag='e1:t')
lc.text((S0X + R + S1X - R) / 2, ey0 - 20, "整词键一口吃完（'no' 3919 同理）；半路键停在中间位", 8,
        '#334155', 'middle', maxw=330, tag='e1:s')

# ---- 转移边 2：位置 1 → 终态 ----
lc.seg(S1X + R, ey0, S2X - R, ey0, lc.C_SAM_S, 2.2, 'sam')
lc.text((S1X + R + S2X - R) / 2, ey0 - 34, "accept_tokens([50256 EOS]) → True", 9.2,
        lc.C_SAM_S, 'middle', True, maxw=280, tag='e2:t')
lc.text((S1X + R + S2X - R) / 2, ey0 - 20, '计数 1→2 · is_terminated=True（接受链 yes→EOS）', 8,
        '#334155', 'middle', maxw=320, tag='e2:s')

# ---- 拒收边：4242 从位置 0 引出、打叉虚线（红） ----
rx_, ry_ = S0X - R * 0.707, S_Y + R * 0.707          # 状态圈左下 45° 点
rej_end = (rx_ - 88, ry_ + 74)
lc.seg(rx_, ry_, rej_end[0], rej_end[1], lc.C_ABORT, 1.8, dash=True)
lc.text(MX + 6, rej_end[1] + 26, "accept_token(4242 '####') → False", 8.6, lc.C_ABORT,
        'start', True, maxw=260, tag='rej:t')
lc.text(MX + 6, rej_end[1] + 42, '计数不动（仍 0）· 状态不动——词表内、语法外', 8,
        lc.C_ABORT, 'start', maxw=260, tag='rej:s')
# 打叉
mx_, my_ = (rx_ + rej_end[0]) / 2, (ry_ + rej_end[1]) / 2
lc.seg(mx_ - 11, my_ - 11, mx_ + 11, my_ + 11, lc.C_ABORT, 2.4)
lc.seg(mx_ - 11, my_ + 11, mx_ + 11, my_ - 11, lc.C_ABORT, 2.4)

# ================= 右侧读法面板 =================
PX, PW = 1120, 320
lc.rect(PX, 138, PW, 372, '#ffffff', lc.C_MUTE, rx=9, sw=1.3, dash=True)
lc.text(PX + 16, 162, '读法', 11, lc.C_TXT, 'start', True, maxw=PW - 32, tag='p:t')
for j, ln in enumerate([
        '· 状态数由语法决定（本例 3 个可观测状态：',
        '  位置 0 / 位置 1 / 终态），与词表大小无关',
        '· 词表大小决定每个状态上合法集的',
        '  「挑选」工作量——编译期一次性算好，',
        '  同语法复用走后端缓存（毫秒级 → 微秒级）',
        '· 每步采样只消费两样东西：',
        '  fill_bitmask 出的合法集 + accept 推进',
        '· 拒收 = 引擎 bug 路径：掩码之下不该采出',
        '  非法 token（采样后推进时兜底校验）',
        '· 这张极小集表就是掩码每步要传的全部信息',
        '  ——逐拍填表与 GPU 窗口归下一章（预告）']):
    lc.text(PX + 16, 186 + j * 24, ln, 8.6, '#334155', 'start', maxw=PW - 30, tag='p:l' + str(j))

# ================= 底部两位置允许集小表 =================
TY, TH = 548, 128
COLW = 660
lc.rect(MX + 8, TY, COLW, TH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 24, TY + 22, '两位置允许集对照（本章驱动脚本实测，gpt2 词表 50257）', 9.5,
        lc.C_TXT, 'start', True, maxw=COLW - 32, tag='t:t')
ROWS = [('位置 0', '77 n · 88 y · 3919 no · 5948 ye · 8505 yes', '5 个（前缀闭包）'),
        ('位置 1', '50256 <|endoftext|>', '1 个（仅 EOS）'),
        ('反例·位置 0', "4242 '####'：词表内但语法外", 'accept 拒收（False）')]
for i, (a, b, c) in enumerate(ROWS):
    yy = TY + 46 + i * 26
    lc.text(MX + 24, yy, a, 8.6, lc.C_TXT, 'start', True, maxw=110, tag='t:r' + str(i) + 'a')
    lc.text(MX + 150, yy, b, 8.6, '#334155', 'start', maxw=330, tag='t:r' + str(i) + 'b')
    lc.text(MX + COLW - 20, yy, c, 8.6, lc.C_SAM_S, 'end', True, maxw=160, tag='t:r' + str(i) + 'c')
    if i < 2:
        lc.seg(MX + 24, yy + 9, MX + COLW - 12, yy + 9, '#e2e8f0', 1.0)

# 接受链时间线小条（右下）
CT_X, CT_W = 748, BXR - 748
lc.rect(CT_X, TY, CT_W, TH, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.4)
lc.text(CT_X + 16, TY + 22, '接受链实测（同一 matcher 顺序执行）', 9.5, lc.C_SAM_S, 'start', True,
        maxw=CT_W - 32, tag='c:t')
CHAIN = ['① fill_bitmask → 5 键亮', "② accept(8505 'yes') → True · 计数 1",
         '③ fill_bitmask → 只剩 EOS', '④ accept(50256) → True · 计数 2 · 终态']
for i, s_ in enumerate(CHAIN):
    xx = CT_X + 24 + (i % 2) * ((CT_W - 48) / 2)
    yy = TY + 48 + (i // 2) * 34
    lc.rect(xx, yy - 14, (CT_W - 62) / 2, 26, '#ffffff', lc.C_SAM_S, rx=6, sw=1.1)
    lc.text(xx + (CT_W - 62) / 4, yy + 3, s_, 7.8, '#334155', 'middle', maxw=(CT_W - 62) / 2 - 10,
            tag='c:c' + str(i))
lc.text(CT_X + 16, TY + TH - 10, '每接受一个合法 token 计数恰 +1；拒收则状态与计数都不动', 8,
        lc.C_MUTE, 'start', maxw=CT_W - 32, tag='c:foot')

# ================= 图例 + 页脚 =================
LY = 700
lx0 = MX
lc.circle(lx0 + 8, LY - 4, 8, lc.C_SAM_S, 1.4, dash=False)
lc.text(lx0 + 24, LY + 1, '可观测状态（圆=非终态 · 双圈=终态）', 8.8, lc.C_TXT, 'start', maxw=280, tag='lg1')
lx0 += 24 + lc.tw('可观测状态（圆=非终态 · 双圈=终态）', 8.8) + 20
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_SAM_S, 2.0, 'sam')
lc.text(lx0 + 40, LY + 1, '接受转移（合法 token）', 8.8, lc.C_TXT, 'start', maxw=180, tag='lg2')
lx0 += 40 + lc.tw('接受转移（合法 token）', 8.8) + 20
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_ABORT, 1.8, dash=True)
lc.seg(lx0 + 15, LY - 12, lx0 + 23, LY + 6, lc.C_ABORT, 1.8)
lc.seg(lx0 + 15, LY + 6, lx0 + 23, LY - 12, lc.C_ABORT, 1.8)
lc.text(lx0 + 40, LY + 1, '拒收（词表内、语法外）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg3')

lc.text(MX, 728, 'vllm/v1/structured_output/backend_xgrammar.py:L78-L126（compile_grammar GRAMMAR 分支 → GrammarMatcher）· '
                 'L119-L123（max_rollback_tokens=num_speculative_tokens）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 742, '状态 / 允许集 / 计数 / 拒收 ＝ 本章驱动脚本实测（xgrammar 0.2.6，gpt2 词表 50257）· '
                 '不画 xgrammar 库内部 CFG/adaptive 细节（按 API 契约）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch30-fig-choice-fsm.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
