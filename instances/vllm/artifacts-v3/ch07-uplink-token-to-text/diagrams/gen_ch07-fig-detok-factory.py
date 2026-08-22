#!/usr/bin/env python3
"""ch07 机制图 3 · detokenizer 三路工厂（figure_spec ch07-fig-detok-factory，模板 layout）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的去 token 化
工位入口——即本章 L2 章图 south『detokenizer 三路工厂』组件的机制展开。
架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：from_new_request 的三岔分派按两级判据走：tokenizer 为 None（或 detokenize=False
先行空化）→ 空壳只记账；USE_FAST_DETOKENIZER（tokenizers>=0.22.0）且 TokenizersBackend
→ Fast（Rust DecodeStream）；否则 → Slow（纯 Python 双 offset 窗口）——选择在 RequestState
诞生时做一次，实测四路分发全部按判据落位。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 706
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '三岔分派台：RequestState 诞生时按两级判据，选一条去 token 化路线',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '岔口只走一次——选择是请求级的，不是每 token 的：此后每个请求固定一条线',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 south『detokenizer 三路工厂』 · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 工厂入口 + 先行空化注 ----------------
ROOT_X, ROOT_Y, ROOT_W, ROOT_H = 60, 92, 420, 64
lc.rect(ROOT_X, ROOT_Y, ROOT_W, ROOT_H, lc.C_API_F, lc.C_API_S, rx=7, sw=1.6)
lc.text(ROOT_X + 14, ROOT_Y + 24, 'from_new_request(tokenizer, request)', 11, lc.C_TXT,
        'start', True, maxw=ROOT_W - 28, tag='root:t')
lc.text(ROOT_X + 14, ROOT_Y + 44, '工厂入口 · vllm/v1/engine/detokenizer.py:L49-L66', 8.5,
        lc.C_FAINT, 'start', maxw=ROOT_W - 28, tag='root:f')

NX, NY, NW, NH = 540, 84, 900, 76
lc.rect(NX, NY, NW, NH, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(NX + 14, NY + 22, '一层之上的先行空化（比判定 1 更早发生）', 9.5, lc.C_MUTE, 'start',
        True, maxw=400, tag='note:t')
lc.text(NX + 14, NY + 42, 'RequestState.from_new_request 先把 tokenizer 置 None——detokenize=False 的请求不用进工厂就知道归宿：空壳（output_processor.py:L223-L225）',
        8.8, '#334155', 'start', maxw=NW - 28, tag='note:l1')
lc.text(NX + 14, NY + 60, '实测：detokenize=False 的 RequestState.detokenizer = IncrementalDetokenizer（空壳）',
        8.8, '#334155', 'start', maxw=NW - 28, tag='note:l2')

# ---------------- 判定 1 ----------------
D1_Y, D1_H = 218, 58
lc.seg(ROOT_X + ROOT_W / 2, ROOT_Y + ROOT_H, ROOT_X + ROOT_W / 2, D1_Y, lc.C_API_S, 2.0, 'dn')
lc.text(ROOT_X + ROOT_W / 2 + 8, ROOT_Y + ROOT_H + 16, '每请求走一次', 8.5, lc.C_MUTE, 'start',
        maxw=150, tag='a:root')
lc.rect(MX, D1_Y, ROOT_W, D1_H, '#ffffff', lc.C_API_S, rx=7, sw=1.9)
lc.text(MX + 14, D1_Y + 24, '判定 1 · tokenizer is None？', 11, lc.C_API_S, 'start', True,
        maxw=ROOT_W - 28, tag='d1:t')
lc.text(MX + 14, D1_Y + 43, '（含先行空化来的 None——见右上注）', 8.5, lc.C_MUTE, 'start',
        maxw=ROOT_W - 28, tag='d1:s')

# 判定 1 → 空壳（是，向下）
NULL_Y = 332
lc.seg(ROOT_X + ROOT_W / 2, D1_Y + D1_H, ROOT_X + ROOT_W / 2, NULL_Y, lc.C_API_S, 2.0, 'dn')
lc.text(ROOT_X + ROOT_W / 2 + 8, D1_Y + D1_H + 18, '是', 9.5, lc.C_API_S, 'start', True,
        maxw=40, tag='a:y1')

# 判定 1 → 判定 2（否，向右）
D2_X, D2_W = 560, 420
D2_MIDY = D1_Y + D1_H / 2
lc.seg(MX + ROOT_W, D2_MIDY, D2_X, D2_MIDY, lc.C_API_S, 2.0, 'dn')
lc.text((MX + ROOT_W + D2_X) / 2, D2_MIDY - 8, '否', 9.5, lc.C_API_S, 'middle', True,
        maxw=40, tag='a:n1')

# ---------------- 判定 2 ----------------
D2_H = 74
lc.rect(D2_X, D1_Y, D2_W, D2_H, '#ffffff', lc.C_API_S, rx=7, sw=1.9)
lc.text(D2_X + 14, D1_Y + 22, '判定 2 · 快线判据（两级同时成立）', 11, lc.C_API_S, 'start', True,
        maxw=D2_W - 24, tag='d2:t')
lc.text(D2_X + 14, D1_Y + 40, 'USE_FAST_DETOKENIZER 且 isinstance(tokenizer, TokenizersBackend)？',
        8.8, lc.C_API_S, 'start', True, maxw=D2_W - 24, tag='d2:c')
lc.text(D2_X + 14, D1_Y + 58, 'USE_FAST_DETOKENIZER = tokenizers ≥ 0.22.0（host 0.22.2 实测 true）',
        8.5, lc.C_MUTE, 'start', maxw=D2_W - 24, tag='d2:s')

# 判定 2 → Fast（是，向下）
FAST_Y = 332
lc.seg(D2_X + D2_W / 2, D1_Y + D2_H, D2_X + D2_W / 2, FAST_Y, lc.C_API_S, 2.0, 'dn')
lc.text(D2_X + D2_W / 2 + 8, D1_Y + D2_H + 18, '是', 9.5, lc.C_API_S, 'start', True,
        maxw=40, tag='a:y2')

# 判定 2 → Slow（否，肘形向右下）
SLOW_X, SLOW_W = 1060, BXR - 1060
SLOW_INY = FAST_Y + 79          # Slow 盒左缘中点
lc.parrow([(D2_X + D2_W, D2_MIDY), (1016, D2_MIDY), (1016, SLOW_INY), (SLOW_X - 2, SLOW_INY)],
          lc.C_API_S, 2.0, 'dn')
lc.text((D2_X + D2_W + 1016) / 2, D2_MIDY - 8, '否', 9.5, lc.C_API_S, 'middle', True,
        maxw=40, tag='a:n2')

# ---------------- 三个终端盒 ----------------
def terminal(x, y, w, h, title, lines, file, probe):
    lc.rect(x, y, w, h, '#ffffff', lc.C_API_S, rx=7, sw=1.6)
    lc.text(x + 14, y + 24, title, 11, lc.C_TXT, 'start', True, maxw=w - 28, tag='t:' + title[:10])
    yy = y + 46
    for ln in lines:
        lc.text(x + 14, yy, ln, 8.8, '#334155', 'start', maxw=w - 26, tag='l:' + ln[:10])
        yy += 17
    lc.text(x + 14, y + h - 24, probe, 8.5, lc.C_API_S, 'start', True, maxw=w - 26,
            tag='p:' + title[:10])
    lc.text(x + 14, y + h - 8, file, 8.5, lc.C_FAINT, 'start', maxw=w - 26, tag='f:' + title[:10])


terminal(MX, NULL_Y, ROOT_W, 158, 'IncrementalDetokenizer（空壳）',
         ['· update 恒返 None · get_next_output_text 恒空串',
          '· 只累积 token_ids——id 账照数（实测 2 个）',
          '· 文本恒空而 id 仍计数：两本账只剩一本'],
         'vllm/v1/engine/detokenizer.py',
         '实测落位：tokenizer=None → 空壳')
terminal(D2_X, FAST_Y, D2_W, 158, 'FastIncrementalDetokenizer（Rust 快线）',
         ['· DecodeStream(ids=prompt) native prefill 预热——prompt 文本不泄漏',
          '· 每 token 一次 stream.step：只吐新完成的字符',
          '· _protected_step 容错：越界 id 吞掉 · 无效前缀重建流重放'],
         'vllm/v1/engine/detokenizer.py:L168-L248',
         '实测落位：TokenizersBackend → Fast')
terminal(SLOW_X, FAST_Y, SLOW_W, 158, 'SlowIncrementalDetokenizer（纯 Python 慢线）',
         ['· 双 offset 滑窗（prefix / read）',
          '· decode 窗内旧段与窗内旧段+新 token 相减得增量',
          '· 窗口只为给 cleanup 算法相邻上下文（见滑窗图）'],
         'vllm/v1/engine/detokenizer.py:L292-L307',
         '实测落位：其余 TokenizerLike → Slow')

# ---------------- 版本判据横幅 ----------------
VB_Y, VB_H = 540, 58
lc.rect(MX, VB_Y, BXR - MX, VB_H, lc.C_ENG_F, lc.C_ENG_S, rx=8, sw=1.3, dash=True)
lc.text(MX + 16, VB_Y + 24, '判据是 v0.27.1 的 TokenizersBackend（transformers 导入，detokenizer.py:L10）——旧资料里的 PreTrainedTokenizerFast 已不是分派依据',
        10, lc.C_TXT, 'start', True, maxw=1290, tag='vb:t')
lc.text(MX + 16, VB_Y + 44, 'USE_FAST_DETOKENIZER 版本闸：仅 tokenizers ≥ 0.22.0 支持 DecodeStream 原生 prefill（L23-L25）',
        9, '#334155', 'start', maxw=1290, tag='vb:s')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = VB_Y + VB_H + 32
lx = MX
items = [('box', '蓝框 = API 进程内工位'), ('dashbox', '虚线框 = 一层之上的先行空化注'),
         ('thick', '粗框 = 判定节点')]
for kind, name in items:
    if kind == 'box':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', lc.C_API_S, rx=4, sw=1.4)
    elif kind == 'dashbox':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', lc.C_MUTE, rx=4, sw=1.1, dash=True)
    else:
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', lc.C_API_S, rx=4, sw=1.9)
    lc.text(lx + 26, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=250, tag='leg' + name)
    lx += 26 + lc.tw(name, 9) + 22
lc.text(MX, LEG_Y + 28, '三路分派 verbatim vllm/v1/engine/detokenizer.py:L49-L66 · 四路落位 host 实测 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-detok-factory.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
