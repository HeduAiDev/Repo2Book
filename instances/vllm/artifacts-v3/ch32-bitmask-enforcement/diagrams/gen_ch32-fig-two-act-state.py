#!/usr/bin/env python3
"""ch32 机制图 · 两段式窗口的 worker 面（figure_spec ch32-fig-two-act-state，模板 state-machine）

放大自 L0 循环框②④两拍的 worker 侧展开——采样列『execute→sample』之间的一次状态寄存
（gpu_model_runner.py 两幕；L2 站 10 的机制放大），架构归属回指 L2 章图，
不另立第二种架构画法（FIGURE-SYSTEM §3）。worker/GPU 侧恒 C_GPU_S 绿（角色色铁律）。

claim：execute_model 返回 None 不是失败而是契约：第一幕前向完成即把十元组暂存进单槽、
谢幕；第二幕 sample_tokens 解包即清、先掩码后采样，配对错乱由双向 RuntimeError 防御。

数字全部取自 figure_spec.numbers（两幕契约 trace：返回 None / 暂存态 10 字段 /
State error 原文 / order=[apply, sample] / state_cleared / logits [1,50257] 4242→-inf、
8505 不变）与 pin 源码锚 worker_base.py:L142-L157。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 792
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="gpu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '两段式契约的 worker 面：第一幕不交付结果——打包十元组暂存、返回 None', 16.5,
        lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '读 execute_model 签名看不到 logits 去了哪：答案在这个单槽 NamedTuple 里（隐藏的隐式参数传递）；'
               '第二幕解包即清、先掩码后采样，配对错乱由双向 RuntimeError 防御',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L0 循环框②④两拍 worker 侧 · L2 站 10 的机制放大'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 两幕大框（worker/GPU 侧恒绿） =================
FY, FH = 118, 384
A1X, A1W = 92, 632
A2X, A2W = 776, 632

# ---- 第一幕 ----
lc.rect(A1X, FY, A1W, FH, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.2)
lc.text(A1X + 16, FY + 24, '第一幕 · execute_model(scheduler_output)', 12.5, lc.C_GPU_S,
        'start', True, maxw=A1W - 30, tag='a1:t')
lc.text(A1X + 16, FY + 44, '前向完成：logits [1, 50257]（CUDA）已算出——冻结在暂存态里的是张量引用、非拷贝',
        8.8, '#334155', 'start', maxw=A1W - 30, tag='a1:l0')

# 单槽十元组
SX, SY, SW, SH = A1X + 18, FY + 58, A1W - 36, 190
lc.rect(SX, SY, SW, SH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(SX + 14, SY + 20, 'execute_model_state（单槽 NamedTuple · 10 个字段）', 9.8, lc.C_TXT,
        'start', True, maxw=SW - 28, tag='slot:t')
FIELDS = ['scheduler_output', 'logits', 'spec_decode_metadata', 'spec_decode_common_attn_metadata',
          'hidden_states', 'sample_hidden_states', 'aux_hidden_states', 'ec_connector_output',
          'cudagraph_stats', 'slot_mappings']
CHW, CHH = (SW - 42) / 2, 24
for i, f_ in enumerate(FIELDS):
    cx = SX + 14 + (i % 2) * (CHW + 14)
    cy = SY + 30 + (i // 2) * (CHH + 6)
    hot = f_ == 'logits'
    lc.rect(cx, cy, CHW, CHH, lc.C_BADGE_F if hot else '#f8fafc', lc.C_GPU_S, rx=5, sw=1.0)
    lc.text(cx + CHW / 2, cy + 16, f_, 8.2, lc.C_ENG_S if hot else '#334155', 'middle',
            hot, maxw=CHW - 6, tag='fld:' + f_)
lc.text(SX + 14, SY + SH - 10, '第一幕打包一次、第二幕解包即清——单槽同时只装一步的账（spec×2 = 上两行 metadata）',
        8, lc.C_MUTE, 'start', maxw=SW - 28, tag='slot:foot')

# 出口：return None
lc.seg(A1X + A1W / 2, SY + SH, A1X + A1W / 2, SY + SH + 40, lc.C_GPU_S, 2.4, 'gpu')
lc.text(A1X + A1W / 2 + 10, SY + SH + 30, 'return None —— 不交付结果，谢幕', 9.2, lc.C_GPU_S,
        'start', True, maxw=280, tag='a1:exit')
lc.text(A1X + 16, FY + FH - 14, '前向已算出 logits（gpu_model_runner.py）· 打包进单槽后返回 None',
        8.2, lc.C_FAINT, 'start', maxw=A1W - 30, tag='a1:foot')

# ---- 第二幕 ----
lc.rect(A2X, FY, A2W, FH, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.2)
lc.text(A2X + 16, FY + 24, '第二幕 · sample_tokens(grammar_output)', 12.5, lc.C_GPU_S,
        'start', True, maxw=A2W - 30, tag='a2:t')
STEPS = [
    ('①', '解包即清', 'state_cleared=true——单槽自清，下一步重装', None),
    ('②', 'apply_grammar_bitmask', '原位改写冻结的 logits：4242 位 5.0 → -inf · 8505 位 2.0 不变', '先掩码'),
    ('③', '_sample', '采样器收到的 is state.logits=True——同一块张量（想留 raw 得自己 clone）', '后采样'),
]
sy = FY + 52
for sym, name, desc, badge in STEPS:
    sh = 78 if desc else 52
    lc.rect(A2X + 18, sy, A2W - 36, sh, '#ffffff', lc.C_GPU_S, rx=7, sw=1.3)
    lc.text(A2X + 32, sy + 20, sym + ' ' + name, 10, lc.C_TXT, 'start', True,
            maxw=A2W - 130, tag='st:' + name[:8])
    if badge:
        bw = 16 + 9.2 * len(badge)
        lc.rect(A2X + A2W - 18 - bw - 8, sy + 6, bw, 19, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.0)
        lc.text(A2X + A2W - 18 - bw / 2 - 8, sy + 19.5, badge, 8.6, lc.C_ENG_S, 'middle', True,
                maxw=bw - 4, tag='sb:' + badge)
    if desc:
        lc.text(A2X + 32, sy + 40, desc, 8.2, '#334155', 'start', maxw=A2W - 70, tag='sd:' + name[:8])
        if name == 'apply_grammar_bitmask':
            lc.text(A2X + 32, sy + 56, '掩码行 0 允许集 [77, 88, 3919, 5948, 8505]（choice yes|no 前缀闭包）',
                    8, lc.C_MUTE, 'start', maxw=A2W - 70, tag='sd2:apply')
    sy += sh + 12
lc.text(A2X + 18, FY + FH - 14, '落地次序实测 order = [apply, sample]——『先掩码后采样』钉死在这六行',
        8.2, lc.C_FAINT, 'start', maxw=A2W - 30, tag='a2:foot')

# ---- 幕间转移（合法） ----
EY = FY + 150
lc.seg(A1X + A1W, EY, A2X, EY, lc.C_GPU_S, 2.6, 'gpu')
lc.text((A1X + A1W + A2X) / 2, EY - 30, 'EngineCore 见 None', 9.4, lc.C_GPU_S, 'middle', True,
        maxw=120, tag='tr:t')
lc.text((A1X + A1W + A2X) / 2, EY - 16, '才调第二幕', 8, '#334155', 'middle', maxw=120, tag='tr:s')
lc.text((A1X + A1W + A2X) / 2, EY + 16, 'future.result() → None', 8, lc.C_MUTE, 'middle',
        maxw=130, tag='tr:l2')
lc.text((A1X + A1W + A2X) / 2, EY + 30, '（L602-L604）', 7.6, lc.C_FAINT, 'middle', maxw=120, tag='tr:l3')

# ---- 非法转移：第一幕自环（红虚线）+ RuntimeError 原文 ----
loop_x = A1X + A1W / 2 - 150
lc.parrow([(loop_x, FY), (loop_x, FY - 34), (loop_x + 300, FY - 34), (loop_x + 300, FY)],
          lc.C_ABORT, 1.8, 'ab', dash=True)
lc.text(loop_x + 150, FY - 42, '上一幕的 sample_tokens 没来就再发车（第一幕 → 第一幕）', 8.6,
        lc.C_ABORT, 'middle', True, maxw=420, tag='loop:t')

# ================= 底部：契约出处（docstring 逐字）+ RuntimeError 原文 =================
QY, QH = 534, 148
QW = 790
lc.rect(MX, QY, QW, QH, '#f8fafc', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 18, QY + 22, '契约出处 · worker_base.py:L142-L157 docstring 原话（技术债自注 L147-L149）',
        9.8, lc.C_TXT, 'start', True, maxw=QW - 36, tag='q:t')
Q1 = '"If this method returns None, sample_tokens should be called immediately after'
Q2 = ' to obtain the ModelRunnerOutput."'
Q3 = '"Note that this design may be changed in future if/when structured outputs parallelism'
Q4 = ' is re-architected."  ← 技术债自注：为结构化输出位掩码的并行而生、重构时可能改掉'
for j, ln in enumerate([Q1, Q2, Q3, Q4]):
    lc.text(MX + 18, QY + 44 + j * 18, ln, 8.4, '#334155', 'start', maxw=QW - 32,
            tag='q:l' + str(j))

# RuntimeError 原文框（红）
REX, REY, REW, REH = 880, QY, BXR - 880, QH
lc.rect(REX, REY, REW, REH, '#fef2f2', lc.C_ABORT, rx=9, sw=1.4)
lc.text(REX + 16, REY + 24, '配对错乱 → 双向 RuntimeError（真异常捕获）', 9.2, lc.C_ABORT,
        'start', True, maxw=REW - 32, tag='re:t')
lc.text(REX + 16, REY + 46, 'State error: sample_tokens() must be called', 8,
        lc.C_ABORT, 'start', maxw=REW - 32, tag='re:l1')
lc.text(REX + 16, REY + 60, 'after execute_model() returns None.', 8,
        lc.C_ABORT, 'start', maxw=REW - 32, tag='re:l2')
lc.text(REX + 16, REY + 84, 'worker 自己抛——读签名看不出这条防御，', 8, lc.C_MUTE,
        'start', maxw=REW - 32, tag='re:l3')
lc.text(REX + 16, REY + 100, '契约的另一半藏在 docstring 与异常里', 8, lc.C_MUTE,
        'start', maxw=REW - 32, tag='re:l4')

# ================= 图例 + 页脚 =================
LY = 712
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.5)
lc.text(lx0 + 21, LY + 1, 'worker / GPU 侧（两幕都在 worker 进程）', 8.8, lc.C_TXT, 'start',
        maxw=280, tag='lg1')
lx0 += 21 + lc.tw('worker / GPU 侧（两幕都在 worker 进程）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, lc.C_BADGE_F, lc.C_GPU_S, rx=3, sw=1.0)
lc.text(lx0 + 21, LY + 1, '十元组字段（logits 高亮）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg2')
lx0 += 21 + lc.tw('十元组字段（logits 高亮）', 8.8) + 18
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_ABORT, 1.8, 'ab', dash=True)
lc.text(lx0 + 40, LY + 1, '非法转移（配对错乱）', 8.8, lc.C_TXT, 'start', maxw=170, tag='lg3')
lx0 += 40 + lc.tw('非法转移（配对错乱）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, '#f8fafc', lc.C_MUTE, rx=3, sw=1.1)
lc.text(lx0 + 21, LY + 1, '源码逐字引文', 8.8, lc.C_TXT, 'start', maxw=140, tag='lg4')

lc.text(MX, 748, 'vllm/v1/worker/worker_base.py:L142-L157（契约 docstring）· vllm/v1/worker/gpu_model_runner.py:L4516-L4534（第一幕打包·返回 None）'
                 '· L4553-L4589（第二幕解包即清 → apply → _sample）· L4171-L4174（配对防御）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 766, '返回 None / 10 字段 / State error 原文 / order=[apply, sample] / state_cleared / 4242 位 5.0→-inf、8505 位 2.0 不变'
                 ' ＝ 本章驱动脚本实测（xgrammar 0.2.6，gpt2 词表 50257）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-two-act-state.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
