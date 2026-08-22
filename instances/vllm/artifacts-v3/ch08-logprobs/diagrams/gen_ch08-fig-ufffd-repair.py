#!/usr/bin/env python3
"""ch08 机制图 3 · U+FFFD 上下文字节重建（figure_spec ch08-fig-ufffd-repair，模板 state-table）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『上行泳道 logprobs 支路』）的装配工位——
即本章 L2 章图 center 拍片 ⑧ 『U+FFFD 修正』的机制展开（上游拍片 ⑦ 『sample 装配』送来
含 � 的 decoded_tokens、下游拍片 ⑨ 『落容器』收修正文本）。非新架构画法，架构归属回指 L0/L2。

claim：以替换字符结尾的候选经「纵向前文 ≤4 token 拼接重解码、剥干净前缀」修正：
中 = E4/B8/AD 三 byte token，位置 2/3 的碎片解码 ''（零件未齐）、位置 4 被采样 173 拿整字 '中'
（decode([228,184,173]) 成功、前缀长 0）、同位候选 228 独立修得 ''——横向候选各修各的、
纵向上下文共用一份。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 748
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'U+FFFD 修正：以替换字符结尾的候选，拿纵向前文 ≤ 4 token 重解码拼回整字',
        16.5, lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '_verify_tokens 只修以 � 结尾者（中置 � 是真不完整，不修）——横向候选各修各的、'
        '纵向上下文共用一份，两轴不能混', 10.5, lc.C_MUTE, 'start', maxw=940, tag='subtitle')
_ch = '放大自 L2 拍片 ⑧ U+FFFD 修正 · L0：上行泳道 logprobs 支路'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# 引子：中 的三袋零件 + max_context 注
iy = 104
lc.text(MX, iy, '中 的 UTF-8 = E4 B8 AD → byte-fallback 词表拆成三个 token：', 9.5, lc.C_TXT,
        'start', maxw=460, tag='intro')
bx = MX + lc.tw('中 的 UTF-8 = E4 B8 AD → byte-fallback 词表拆成三个 token：', 9.5) + 10
for lab in ('228 = E4', '184 = B8', '173 = AD'):
    bw = lc.tw(lab, 8.5) + 12
    lc.rect(bx, iy - 12, bw, 17, '#ffffff', lc.C_MUTE, rx=8, sw=1.0)
    lc.text(bx + bw / 2, iy, lab, 8.5, lc.C_MUTE, 'middle', maxw=bw - 4, tag='byte' + lab)
    bx += bw + 6
lc.text(bx, iy, '——每袋零件单独解码都是 �', 9.5, lc.C_TXT, 'start', maxw=280, tag='intro2')
_up = '← 上游 · L2 拍片 ⑦ sample 装配：非增量解码送来含 � 的 decoded_tokens'
_uw = lc.tw(_up, 9) + 14
lc.rect(MX, 128, _uw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(MX + _uw / 2, 141.5, _up, 9, lc.C_MUTE, 'middle', maxw=_uw - 4, tag='chip:up')
_note = 'max_context = min(len(上下文), 4)——UTF-8 多字节序列最长 4 字节'
lc.text(BXR, 141, _note, 9, lc.C_API_S, 'end', maxw=lc.tw(_note, 9) + 4, tag='note:maxctx')

# ---------------- 状态表 ----------------
C0, C1, C2, C3, C4, C5 = 60, 130, 410, 590, 1240, 1416      # 列边界
TY0, TY1 = 162, 180          # 表头带
ROWS_Y = [180, 252, 332, 424, 620]   # 各行 y 起点 / 表底
lc.rect(C0, TY0, C5 - C0, ROWS_Y[0] - TY0, '#eff6ff', lc.C_API_S, rx=0, sw=0)
for cx0, cx1, t in ((C0, C1, '位置'), (C1, C2, '候选（横向轴：同一位置的 top-k）'),
                    (C2, C3, '纵向上下文（已落定）'), (C3, C4, '探测轨迹（每档拼一次 decode）'),
                    (C4, C5, '修正后')):
    lc.text((cx0 + cx1) / 2, TY0 + 13, t, 9, lc.C_API_S, 'middle', True, maxw=cx1 - cx0 - 8,
            tag='th:' + t[:4])
for yy in ROWS_Y[1:]:
    lc.seg(C0, yy, C5, yy, '#e2e8f0', 1.1)


def cand_lines(y, sampled, top1):
    lc.text(C1 + 10, y + 18, f'被采样 {sampled}', 9, lc.C_SAM_S, 'start', True, maxw=120,
            tag='cd:s' + sampled)
    lc.text(C1 + 10, y + 36, f'top1 {top1}', 9, lc.C_MUTE, 'start', maxw=120, tag='cd:t' + top1)


def ctx_blocks(y, ids, caption=True):
    bx = (C2 + C3) / 2 - 26
    for i, tid in enumerate(ids):
        lc.rect(bx, y + 12 + i * 28, 52, 22, '#ffffff', lc.C_MUTE, rx=4, sw=1.1)
        lc.text(bx + 26, y + 27 + i * 28, str(tid), 9.5, lc.C_TXT, 'middle', True, maxw=40,
                tag='ctx' + str(tid))
    if caption:
        lc.text((C2 + C3) / 2, y + 12 + len(ids) * 28 + 14, '纵向共用一份', 7.5, lc.C_FAINT,
                'middle', maxw=100, tag='ctxcap')


def chip(x, y, text, verdict=None, dash=False):
    """探测 chip。verdict: 'fail' | 'ok' | 'sweep'。返回右缘 x。"""
    w_ = lc.tw(text, 8) + 16
    if verdict == 'ok':
        f, s, sw = lc.C_GPU_F, lc.C_GPU_S, 1.5
    else:
        f, s, sw = '#ffffff', lc.C_MUTE, 1.0
    lc.rect(x, y, w_, 20, f, s, rx=9, sw=sw, dash=dash)
    lc.text(x + w_ / 2, y + 13.5, text, 8, lc.C_TXT if verdict != 'ok' else lc.C_GPU_S,
            'middle', verdict == 'ok', maxw=w_ - 6, tag='pv:' + text[:10])
    return x + w_


def chain(x, y, chips_, gap=24):
    """chip 链，之间画 →。chips_ = [(text, verdict, dash)]"""
    for i, (t, v, d) in enumerate(chips_):
        x = chip(x, y, t, v, d)
        if i < len(chips_) - 1:
            lc.text(x + gap / 2, y + 13.5, '→', 9, lc.C_MUTE, 'middle', maxw=14,
                    tag='ar' + t[:6])
            x += gap
    return x


# ---- 位置 1：干净结尾不触发 ----
y = ROWS_Y[0]
lc.text((C0 + C1) / 2, y + 30, '位置 1', 9.5, lc.C_TXT, 'middle', True, maxw=60, tag='r1')
cand_lines(y, "256 → 'hello'", "257 → ' world'")
lc.text((C2 + C3) / 2, y + 30, '（空）', 9, lc.C_FAINT, 'middle', maxw=60, tag='r1ctx')
lc.text(C3 + 14, y + 25, '干净结尾（无 � 尾）——不触发修正', 8.5, lc.C_MUTE, 'start',
        maxw=300, tag='r1probe')
lc.text((C4 + C5) / 2, y + 25, "'hello' / ' world'", 9, lc.C_TXT, 'middle', maxw=160,
        tag='r1out')
lc.text((C4 + C5) / 2, y + 42, '（原样）', 7.5, lc.C_FAINT, 'middle', maxw=80, tag='r1out2')

# ---- 位置 2：零件未齐，放弃 ----
y = ROWS_Y[1]
lc.text((C0 + C1) / 2, y + 32, '位置 2', 9.5, lc.C_TXT, 'middle', True, maxw=60, tag='r2')
cand_lines(y, "228 → '�'", "256 → 'hello'")
ctx_blocks(y, [256])
chain(C3 + 14, y + 22, [("decode([256,228]) → 'hello�' ✗ 仍以 � 结尾 → 放弃", None, False)])
lc.text((C4 + C5) / 2, y + 30, "'' / 'hello'", 9, lc.C_TXT, 'middle', maxw=160, tag='r2out')
lc.text((C4 + C5) / 2, y + 47, '碎片记空串', 7.5, lc.C_FAINT, 'middle', maxw=80, tag='r2out2')

# ---- 位置 3：零件仍未齐，放弃 ----
y = ROWS_Y[2]
lc.text((C0 + C1) / 2, y + 36, '位置 3', 9.5, lc.C_TXT, 'middle', True, maxw=60, tag='r3')
cand_lines(y, "184 → '�'", "256 → 'hello'")
ctx_blocks(y, [256, 228])
chain(C3 + 14, y + 28, [("decode([228,184]) → '��' ✗", None, False),
                        ("decode([256,228,184]) → 'hello��' ✗ → 放弃", None, False)])
lc.text((C4 + C5) / 2, y + 34, "'' / 'hello'", 9, lc.C_TXT, 'middle', maxw=160, tag='r3out')
lc.text((C4 + C5) / 2, y + 51, '碎片记空串', 7.5, lc.C_FAINT, 'middle', maxw=80, tag='r3out2')

# ---- 位置 4：两轴同格——两条泳道各修各的 ----
y = ROWS_Y[3]
mid_y = (y + ROWS_Y[4]) / 2
lc.text((C0 + C1) / 2, mid_y, '位置 4', 9.5, lc.C_TXT, 'middle', True, maxw=60, tag='r4')
lc.text((C0 + C1) / 2, mid_y + 18, '（两轴）', 7.5, lc.C_FAINT, 'middle', maxw=60, tag='r4b')
cand_lines(y + 14, "173 → '�'", "228 → '�'")
lc.text(C1 + 10, y + 14 + 54, 'k=1：每位置 2 个横向候选，', 7.5, lc.C_MUTE, 'start', maxw=170,
        tag='r4n1')
lc.text(C1 + 10, y + 14 + 68, '各带自己的探测链（右）', 7.5, lc.C_MUTE, 'start', maxw=170,
        tag='r4n2')
ctx_blocks(y + 8, [256, 228, 184])
lc.seg(C3, y + 96, C5, y + 96, '#e2e8f0', 1.0, dash=True)   # 泳道分隔
# 泳道 A：被采样 173 → 整字 '中'
lab_w = lc.tw('被采样 173', 8.5, True) + 12
lc.rect(C3 + 10, y + 16, lab_w, 18, lc.C_SAM_F, lc.C_SAM_S, rx=9, sw=1.2)
lc.text(C3 + 10 + lab_w / 2, y + 28.5, '被采样 173', 8.5, lc.C_SAM_S, 'middle', True,
        maxw=lab_w - 4, tag='laneA')
chain(C3 + 20 + lab_w, y + 15, [
    ("decode([184,173]) → '��' ✗", None, False),
    ("decode([228,184,173]) → '中' ✓", 'ok', False),
    ("回扫：decode([184])='�' decode([228])='�' ⇒ 前缀长 0", None, True)])
# 泳道 B：候选 228 → ''
lab_w2 = lc.tw('候选 228', 8.5, True) + 12
lc.rect(C3 + 10, y + 112, lab_w2, 18, '#ffffff', lc.C_MUTE, rx=9, sw=1.1)
lc.text(C3 + 10 + lab_w2 / 2, y + 124.5, '候选 228', 8.5, lc.C_MUTE, 'middle', True,
        maxw=lab_w2 - 4, tag='laneB')
chain(C3 + 20 + lab_w2, y + 111, [
    ("decode([184,228]) → '��' ✗", None, False),
    ("decode([228,184,228]) → '���' ✗", None, False),
    ("decode([256,228,184,228]) → 'hello���' ✗ → ''", None, False)])
lc.text((C4 + C5) / 2, y + 40, '中', 15, lc.C_GPU_S, 'middle', True, maxw=40, tag='r4outA')
lc.text((C4 + C5) / 2, y + 58, '整字归完成者 173', 7.5, lc.C_FAINT, 'middle', maxw=110,
        tag='r4outA2')
lc.text((C4 + C5) / 2, y + 116, "''", 11, lc.C_TXT, 'middle', True, maxw=40, tag='r4outB')
lc.text((C4 + C5) / 2, y + 134, '没拼成，记空串', 7.5, lc.C_FAINT, 'middle', maxw=110,
        tag='r4outB2')
lc.seg(C0, ROWS_Y[4], C5, ROWS_Y[4], lc.C_API_S, 1.4)

# ---------------- 底部：终态序列 + 图例 + 页脚 ----------------
FB = (MX, 640, C5 - C0, 44)
lc.rect(*FB[:2], FB[2], FB[3], '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 14, 667, '修正后采样轴解码序列：', 9.5, lc.C_TXT, 'start', True, maxw=200,
        tag='fin:t')
fx = MX + 14 + lc.tw('修正后采样轴解码序列：', 9.5, True) + 8
for v in ("'hello'", "''", "''", "'中'"):
    fw = lc.tw(v, 9, True) + 14
    hot = (v == "'中'")
    lc.rect(fx, 653, fw, 20, lc.C_GPU_F if hot else '#ffffff',
            lc.C_GPU_S if hot else lc.C_MUTE, rx=9, sw=1.2 if hot else 1.0)
    lc.text(fx + fw / 2, 667, v, 9, lc.C_GPU_S if hot else lc.C_TXT, 'middle', hot,
            maxw=fw - 4, tag='fin' + v)
    fx += fw + 6
lc.text(fx + 14, 667, 'cumulative_logprob = -0.55（数值不受文本修正影响）', 9, lc.C_MUTE,
        'start', maxw=430, tag='fin:cum')

LEG_Y = 706
lx = MX
LEG = [('fail', '✗ 拼接试探失败（仍以 � 结尾）'), ('ok', '✓ 拼出整字'),
       ('sweep', '回扫剥干净前缀（决定整字归属）'), ('ctx', '小方块 = 纵向已落定 token')]
for kind, name in LEG:
    if kind == 'ok':
        lc.rect(lx, LEG_Y - 8, 22, 13, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.3)
    elif kind == 'fail':
        lc.rect(lx, LEG_Y - 8, 22, 13, '#ffffff', lc.C_MUTE, rx=3, sw=1.0)
    elif kind == 'sweep':
        lc.rect(lx, LEG_Y - 8, 22, 13, '#ffffff', lc.C_MUTE, rx=3, sw=1.0, dash=True)
    else:
        lc.rect(lx, LEG_Y - 8, 22, 15, '#ffffff', lc.C_MUTE, rx=3, sw=1.1)
    lc.text(lx + 28, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=300, tag='leg' + kind)
    lx += 28 + lc.tw(name, 9) + 20
_dn = '→ 下游 · L2 拍片 ⑨ 落容器：收修正后的文本'
_dw = lc.tw(_dn, 9) + 14
lc.rect(BXR - _dw, LEG_Y - 11, _dw, 19, '#ffffff', lc.C_MUTE, rx=9, sw=1.0, dash=True)
lc.text(BXR - _dw / 2, LEG_Y + 2, _dn, 9, lc.C_MUTE, 'middle', maxw=_dw - 4, tag='chip:dn')
lc.text(MX, 738, '_verify_tokens + _correct_decoded_token + _get_sampled_context_ids verbatim '
        'vllm/v1/engine/logprobs.py:L312-L346 / L249-L310 / L208-L247 · 逐次 decode 调用为'
        '真 tokenizers 0.22.2 byte-fallback 实测 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch08-fig-ufffd-repair.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
