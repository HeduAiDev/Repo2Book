#!/usr/bin/env python3
"""ch30 机制图 · m1 概念链：『位掩码不是重试』（figure_spec ch30-fig-concept-chain，模板 flow）

放大自 L0 采样列·结构化输出组『语法→掩码』编译段的机制展开（L2 章图站 4），架构性
框图回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：一句『必须是 yes 或 no』变成 50257 维 logits 上的一次预过滤：choice→EBNF→
FSM 位置 0 合法集（词表中仅 5 个 token）→一行 1571 个 int32 的位掩码→bit=0 位写
-inf→argmax 从 4242 翻到 8505。

数字全部取自 figure_spec.numbers（允许集 5 token、argmax 翻转 4242→8505、掩码行
1571 int32、bit=1=允许/bit=0→-inf、-1 补码=全 1=全允许；配套 trace 字段
fsm_positions.sequence[0] / argmax_flip / bitmask_layout.int32_per_row / WC1 errata）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 872
MX, BXR = 60, 1440
BIT_ON = '#1e293b'      # bit=1（允许）暗格
BIT_OFF = '#f1f5f9'     # bit=0（→-inf）亮格

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>'
    f'<marker id="dn2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_API_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '位掩码不是重试：一句 choice 变成 50257 维 logits 上的一次预过滤',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, 'choice ["yes","no"] → EBNF → FSM 位置 0 合法集（5/50257≈0.01%）→ 一行 1571 个 int32 位掩码 → '
               'bit=0 写 -inf → argmax 从 4242 翻到 8505',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L0 采样列·结构化输出组 · L2 站 4『编译成 FSM』'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 上带：编译段（每个语法一次） =================
AY0, AY1 = 88, 330
lc.rect(MX + 8, AY0, BXR - MX - 16, AY1 - AY0, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 24, AY0 + 20, '编译段——每个语法只编一次（引擎侧后端内部缓存复用）', 10.5,
        lc.C_TXT, 'start', True, maxw=560, tag='a:sec')

# ① 用户给的约束（蓝=用户/前端侧）
A1X, A1W = 92, 296
lc.rect(A1X, AY0 + 38, A1W, 118, lc.C_API_F, lc.C_API_S, rx=8, sw=1.6)
lc.text(A1X + 14, AY0 + 60, '① 用户给的约束', 11.5, lc.C_API_S, 'start', True, maxw=A1W - 28, tag='a1:t')
for j, ln in enumerate(['choice=["yes", "no"]', '（StructuredOutputsParams 六形态之一：',
                        'json / json_object / regex /', 'choice / grammar / structural_tag）']):
    lc.text(A1X + 14, AY0 + 82 + j * 17, ln, 8.8, '#334155', 'start', maxw=A1W - 26, tag='a1:l' + str(j))

# ② 统一成 EBNF（蓝）
A2X, A2W = 452, 316
lc.rect(A2X, AY0 + 38, A2W, 118, lc.C_API_F, lc.C_API_S, rx=8, sw=1.6)
lc.text(A2X + 14, AY0 + 60, '② 校验期统一成 EBNF', 11.5, lc.C_API_S, 'start', True, maxw=A2W - 28, tag='a2:t')
for j, ln in enumerate(['root ::= "yes" | "no"', 'choice_as_grammar 原地改写：',
                        '生成 EBNF → choice=None → grammar=EBNF', '（引擎侧从此无 CHOICE 分支）']):
    lc.text(A2X + 14, AY0 + 82 + j * 17, ln, 8.8, '#334155', 'start', maxw=A2W - 26, tag='a2:l' + str(j))

# ③ 编译成 FSM（品红=采样列·结构化输出组）
A3X, A3W = 836, 590
lc.rect(A3X, AY0 + 38, A3W, 190, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.6)
lc.text(A3X + 14, AY0 + 60, '③ 编译成 FSM——逐位置算出词表上的合法集（按 token 接受）', 11.5,
        lc.C_SAM_S, 'start', True, maxw=A3W - 28, tag='a3:t')
# 位置 0 亮 5 个键（键盘排布）
KEYS = [('n', '77'), ('y', '88'), ('no', '3919'), ('ye', '5948'), ('yes', '8505')]
kx = A3X + 14
for name, tid in KEYS:
    kw = 30 + 9.5 * len(name)
    lc.rect(kx, AY0 + 74, kw, 30, '#ffffff', lc.C_SAM_S, rx=6, sw=1.4)
    lc.text(kx + kw / 2, AY0 + 87, f"'{name}'", 10.5, lc.C_TXT, 'middle', True, maxw=kw - 4, tag='k' + name)
    lc.text(kx + kw / 2, AY0 + 100, tid, 8.2, lc.C_MUTE, 'middle', maxw=kw, tag='k' + name + ':id')
    kx += kw + 10
lc.text(A3X + 14, AY0 + 124, '位置 0 亮 5 个键 = choice 的前缀闭包（半路键 n/y 与整词键 no/ye/yes 同亮——'
                             '一个 token 可一口吃多个字符）',
        8.8, '#334155', 'start', maxw=A3W - 28, tag='a3:l1')
lc.text(A3X + 14, AY0 + 142, '位置 1 只剩 1 个键：50256 EOS（串已完整，只差停机）', 8.8,
        '#334155', 'start', maxw=A3W - 28, tag='a3:l2')
lc.text(A3X + 14, AY0 + 160, '编译产物：GrammarMatcher 逐 token 状态机（accept / fill / rollback）', 8.8,
        '#334155', 'start', maxw=A3W - 28, tag='a3:l3')
lc.text(A3X + 14, AY0 + 182, 'vllm/v1/structured_output/backend_xgrammar.py:L78-L126', 8.2,
        lc.C_FAINT, 'start', maxw=A3W - 28, tag='a3:file')
# ①→②→③ 箭头
mid_y = AY0 + 97
lc.seg(A1X + A1W, mid_y, A2X, mid_y, lc.C_API_S, 1.8, 'dn2')
lc.seg(A2X + A2W, mid_y, A3X, mid_y, lc.C_API_S, 1.8, 'dn2')

# ================= 下带：应用段（每个采样位置一次） =================
BY0, BY1 = 356, 660
lc.rect(MX + 8, BY0, BXR - MX - 16, BY1 - BY0, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 24, BY0 + 20, '应用段——每个采样位置一次（fill 一行掩码 → 盖上 logits → 采样）', 10.5,
        lc.C_TXT, 'start', True, maxw=560, tag='b:sec')

# 交接箭头：③ → ④
lc.seg(1130, AY1, 1130, BY0, lc.C_SAM_S, 2.0, 'sam')
lc.text(1142, (AY1 + BY0) / 2 + 3, '每个位置合法集 → 打包成一行', 9, lc.C_SAM_S, 'start', maxw=280, tag='hand')

# ④ 合法集 → 位掩码行
B4X, B4W = 92, 560
lc.rect(B4X, BY0 + 38, B4W, 248, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.6)
lc.text(B4X + 14, BY0 + 60, '④ 合法集打包成位掩码行（fill_bitmask）', 11.5, lc.C_SAM_S,
        'start', True, maxw=B4W - 28, tag='b4:t')
# 50257 位窄条：5 位高亮（比例位置；77/88 挤在最左端，用错层引线标号）
SX0, SX1, SY = B4X + 20, B4X + 540, BY0 + 106
lc.rect(SX0, SY, SX1 - SX0, 22, '#ffffff', lc.C_MUTE, rx=3, sw=1.0)
CELLN = 66
cw = (SX1 - SX0) / CELLN
for i in range(CELLN):
    if i:
        lc.seg(SX0 + i * cw, SY, SX0 + i * cw, SY + 22, '#e2e8f0', 0.5)
POS = [77, 88, 3919, 5948, 8505]
xs = {t: SX0 + (SX1 - SX0) * (t / 50256) for t in POS}
# 高亮位：77/88 几乎重合（词表开头），各自画 3px 暗条并错层引线
for t in POS:
    lc.rect(xs[t] - 1.5, SY + 1, 3, 20, BIT_ON, BIT_ON, rx=1, sw=0)
lc.seg(xs[88] + 5, SY - 2, xs[88] + 20, SY - 26, lc.C_MUTE, 0.9)
lc.text(xs[88] + 24, SY - 28, '88', 8, lc.C_SAM_S, 'start', maxw=40, tag='bit88')
lc.text(xs[77], SY - 18, '77', 8, lc.C_SAM_S, 'middle', maxw=40, tag='bit77')
for t, dy in ((3919, -18), (5948, -30), (8505, -18)):
    lc.seg(xs[t], SY - 2, xs[t], SY + dy + 8, lc.C_MUTE, 0.9)
    lc.text(xs[t], SY + dy, str(t), 8, lc.C_SAM_S, 'middle', maxw=44, tag='bit' + str(t))
lc.text(SX0, SY + 40, '50257 位 · 每 token 一位 · 位置 0 仅 5 位=1（其余位=0）', 8.8,
        '#334155', 'start', maxw=520, tag='b4:l1')
# int32 块序列
BY2 = SY + 58
nx = SX0
for i in range(13):
    bw = 34 if i not in (3, 9) else 18
    if i in (3, 9):
        lc.rect(nx, BY2, bw, 18, '#ffffff', lc.C_MUTE, rx=2, sw=1.0)
        lc.text(nx + bw / 2, BY2 + 13, '…', 9, lc.C_MUTE, 'middle', maxw=bw, tag='dots' + str(i))
    else:
        lc.rect(nx, BY2, bw, 18, '#ffffff', lc.C_SAM_S, rx=2, sw=1.1)
    nx += bw + 5
lc.text(SX1, BY2 + 34, '一行 = ceil(50257/32) = 1571 个 int32 · 6284 B/行', 8.8, '#334155',
        'end', maxw=520, tag='b4:l2')
# 位展开小图：int32 #2（token 64-95 的 32 位）——bit 13=token 77 'n'、bit 24=token 88 'y' 置 1
EY = BY2 + 56
lc.text(SX0, EY - 6, '位展开：int32 #2 = token 64-95 的 32 位（bit 13=77 n、bit 24=88 y 置 1）', 8.4,
        '#334155', 'start', maxw=520, tag='b4:e0')
for i in range(32):
    on = i in (13, 24)      # 77-64=13、88-64=24：允许集内两位恰好同装一格（trace 允许集算术）
    lc.rect(SX0 + i * 15.5, EY, 13.5, 14, BIT_ON if on else BIT_OFF, lc.C_MUTE, rx=1.5, sw=0.7)
lc.text(SX1, EY - 6, '← 32 格', 8, lc.C_MUTE, 'end', maxw=60, tag='b4:e3')
lc.text(SX0, EY + 32, 'bit=1 = 允许（暗格）· bit=0 → 写 -inf · -1 补码 = 32 格全 1 = 全允许', 8.4,
        lc.C_TXT, 'start', maxw=520, tag='b4:e1')

# ⑤ 盖掩码
B5X, B5W = 700, 336
lc.rect(B5X, BY0 + 38, B5W, 248, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.6)
lc.text(B5X + 14, BY0 + 60, '⑤ 盖到 logits 上（worker 侧）', 11.5, lc.C_SAM_S, 'start', True,
        maxw=B5W - 28, tag='b5:t')
for j, ln in enumerate(['apply_token_bitmask_inplace：',
                        'bit=0 的位置原位写 -inf',
                        '',
                        'logit[4242] = 5.0  →  -inf',
                        '（非法 token 概率归零；',
                        'softmax 下 -inf 恰为 0，',
                        '骰子只能落在合法集上）']):
    if ln:
        lc.text(B5X + 14, BY0 + 84 + j * 19, ln, 9.2 if j < 2 else 9.6,
                '#334155' if j < 2 else lc.C_TXT, 'start', bold=(j == 3), maxw=B5W - 26, tag='b5:l' + str(j))
lc.seg(B4X + B4W, BY0 + 162, B5X, BY0 + 162, lc.C_SAM_S, 1.8, 'sam')

# ⑥ argmax 翻转（柱：掩码前 4242 最高 → 掩码后翻 8505）
B6X, B6W = 1084, 342
lc.rect(B6X, BY0 + 38, B6W, 248, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.6)
lc.text(B6X + 14, BY0 + 60, '⑥ 采样：argmax 翻转', 11.5, lc.C_SAM_S, 'start', True, maxw=B6W - 28, tag='b6:t')
base = BY0 + 210
bars = [(4242, "'####'", 5.0, True), (8505, "'yes'", 2.0, False), (3919, "'no'", 1.0, False)]
bx = B6X + 40
SCALE6 = 20.0     # 1.0 分 = 20px
bar_geo = {}
for tid, tok, v, banned in bars:
    top = base - v * SCALE6
    lc.rect(bx, top, 54, v * SCALE6, '#fee2e2' if banned else '#ffffff',
            lc.C_ABORT if banned else lc.C_MUTE, rx=2, sw=1.3)
    lc.text(bx + 27, top - 20, ('5.0 → -inf' if banned else f'{v:.1f}'), 9,
            lc.C_ABORT if banned else lc.C_TXT, 'middle', True, maxw=90, tag='b6:v' + str(tid))
    lc.text(bx + 27, top - 8, tok, 7.8, lc.C_MUTE, 'middle', maxw=70, tag='b6:t' + str(tid))
    lc.text(bx + 27, base + 14, str(tid), 8.4, lc.C_TXT, 'middle', True, maxw=60, tag='b6:i' + str(tid))
    bar_geo[tid] = (bx, top)
    bx += 96
# 打叉：骑在 4242 禁柱中部（词表内、语法外——被 -inf 掐掉）
x0, t0 = bar_geo[4242]
cx0 = x0 + 27
cy0 = t0 + 55
lc.seg(cx0 - 16, cy0 - 16, cx0 + 16, cy0 + 16, lc.C_ABORT, 2.4)
lc.seg(cx0 - 16, cy0 + 16, cx0 + 16, cy0 - 16, lc.C_ABORT, 2.4)
# argmax 翻转箭头：4242 柱右侧 → 8505 柱顶
x1, t1 = bar_geo[8505]
lc.seg(x0 + 58, cy0 - 4, x1 + 24, t1 - 4, lc.C_SAM_S, 2.0, 'sam')
lc.text((x0 + x1) / 2 + 24, (cy0 + t1) / 2 - 8, 'argmax 翻转', 8.6, lc.C_SAM_S, 'middle', True,
        maxw=90, tag='b6:am1')
lc.text(B6X + 14, base + 32, '掩码前 argmax=4242（5.0 最高）→ 掩码后翻到 8505；', 8.6,
        '#334155', 'start', maxw=B6W - 26, tag='b6:f1')
lc.text(B6X + 14, base + 48, '本例构造下自由采样踩非法 token 的概率 99.99%', 8.6,
        '#334155', 'start', maxw=B6W - 26, tag='b6:f2')
lc.seg(B5X + B5W, BY0 + 162, B6X, BY0 + 162, lc.C_SAM_S, 1.8, 'sam')

# ================= 底部对照：重试法 vs 掩码法 =================
CY0, CH_ = 684, 112
lc.rect(MX + 8, CY0, 676, CH_, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 24, CY0 + 20, '对照 · 重试法（采样 → 校验 → 重采样）', 10, lc.C_TXT, 'start', True,
        maxw=640, tag='c1:t')
for j, ln in enumerate(['· 落在非法集的概率非零，就永远可能再落错——不保证有限步终止',
                        '· 每轮重试多烧一次 GPU 前向；解的是「最近合法串」问题，无唯一解也不保证终止']):
    lc.text(MX + 24, CY0 + 42 + j * 18, ln, 8.6, '#334155', 'start', maxw=648, tag='c1:l' + str(j))
lc.rect(756, CY0, 684, CH_, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.4)
lc.text(772, CY0 + 20, '掩码法 = 对分布做合法集上的条件化——一步到位', 10, lc.C_SAM_S, 'start', True,
        maxw=640, tag='c2:t')
for j, ln in enumerate(['· ch9 已实测：同一行 logits 盖掩码前后 argmax 翻位（回指）；本章给「这张允许表从哪来」的前半',
                        '· 逐拍填表与 GPU 窗口归 ch31（预告）']):
    lc.text(772, CY0 + 42 + j * 18, ln, 8.6, '#334155', 'start', maxw=652, tag='c2:l' + str(j))

# ================= 图例 + 页脚 =================
LY = 824
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_API_F, lc.C_API_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '用户/前端侧（校验期）', 8.8, lc.C_TXT, 'start', maxw=180, tag='lg1')
lx0 += 21 + lc.tw('用户/前端侧（校验期）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '采样列·结构化输出组（编译与应用产物）', 8.8, lc.C_TXT, 'start', maxw=300, tag='lg2')
lx0 += 21 + lc.tw('采样列·结构化输出组（编译与应用产物）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, BIT_ON, BIT_ON, rx=2, sw=0.8)
lc.text(lx0 + 21, LY + 1, 'bit=1（允许）暗格', 8.8, lc.C_TXT, 'start', maxw=160, tag='lg3')

lc.text(MX, 850, 'vllm/v1/structured_output/backend_xgrammar.py:L78-L126（compile_grammar → GrammarMatcher）· '
                 'L272-L303（choice→EBNF 改写）· utils.py:L86-L175（apply_grammar_bitmask → xgr.apply_token_bitmask_inplace）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 866, '允许集 5 token / argmax 翻转 4242→8505 / 1571 int32 / 6284 B ＝ 本章驱动脚本实测（gpt2 词表 50257）· '
                 'bit=1=允许（WC1 三处互证）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch30-fig-concept-chain.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
