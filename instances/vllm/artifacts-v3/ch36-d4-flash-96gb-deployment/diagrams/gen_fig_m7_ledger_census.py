#!/usr/bin/env python3
"""ch36 机制图 · 五本账自报普查表（figure_spec fig_m7_ledger_census，模板 layout）

放大自 L2 章图拍片⑤（五类 spec 自报）的普查总表——L0 kv_column 的『账本收账』格。

claim：五本账自报 168 份 spec：窗口账 44 本（43 层+MTP，37,440B/块）、full_mla 62 本
（21 C4A 主 + 21 C4I indexer + 20 C128 主）、状态账 42+20 本（C4A 层是 twin 双生：
attention 压缩器 32,832B + indexer 压缩器 8,640B；C128 状态 32,832B）。

数字全部取自 figure_spec.numbers（普查 census / sparse_swa.py:L87-L102 /
attention.py:L655-L674 · L698-L710 · L781-L798 · L799-L809 / compressor.py:L196-L203）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1480, 636
MX, BXR = 56, 1424
DEFS = lc.DEFS

# ---------------- 标题区 ----------------
lc.text(MX, 34, '自报制普查：五本账、四种页规格、168 份 spec——账本管线不发明页大小，只收账', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '每本账的 get_kv_cache_spec 自己报页大小 · C4A 层其实背着 twin 状态账——Lightning Indexer 自带一台压缩器，运行实证配成 21 对双生',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L2 拍片⑤ · L0 显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 五张账本卡 ----------------
CARDS = [
    (lc.C_API_S, lc.C_API_F, '窗口账', '44 本',
     ['SlidingWindowMLASpec(64, 128)', 'uint8 · 对齐 576', '页 37,440B/块', '43 层 + MTP 人人有份',
      '（每层一本，无压缩）']),
    (lc.C_GPU_S, lc.C_GPU_F, 'C4A 压缩主账', '21 本',
     ['MLAAttentionSpec(ratio 4)', '256÷4 = 64 槽 × 584B', '页 37,440B/块', '带 Lightning Indexer',
      '（纯窗口层返回 None）']),
    (lc.C_KV_S, lc.C_KV_F, 'C128 压缩主账', '20 本',
     ['MLAAttentionSpec(ratio 128)', '256÷128 = 2 槽 × 584B', '页 1,728B/块', '（垫 47.95%——最小页）',
      '无 indexer']),
    (lc.C_ZMQ_S, lc.C_ZMQ_F, 'indexer 账 C4I', '21 本',
     ['MLAAttentionSpec(head_size=68)', 'fp4 档 · compress_ratio=4', '64 槽 × 68B', '页 4,608B/块',
      '（fp8 档则 132B/槽）']),
    (lc.C_ENG_S, lc.C_ENG_F, '状态账', '42+20 本',
     ['C4：block 4 / window 8', 'C128：block 8 / window 128', 'fp32 恒定（唯一不量化）', 'attn 态 32,832B/块',
      'C128 状态 32,832B/块']),
]
CY, CH = 100, 176
GAP = 14
CW = (BXR - MX - 4 * GAP) / 5          # 263.2
for i, (s, f, name, count, lines) in enumerate(CARDS):
    x = MX + i * (CW + GAP)
    lc.rect(x, CY, CW, CH, f, s, rx=8, sw=1.6)
    lc.text(x + 13, CY + 22, name, 10.8, s, 'start', True, maxw=CW - 76, tag=f'cd{i}n')
    bw = 15 + 8.4 * len(count)
    lc.rect(x + CW - bw - 8, CY + 8, bw, 20, lc.C_BADGE_F, s, rx=9, sw=1.1)
    lc.text(x + CW - bw / 2 - 8, CY + 22, count, 9, s, 'middle', True, maxw=bw - 4, tag=f'cd{i}c')
    for j, ln in enumerate(lines):
        lc.text(x + 13, CY + 44 + j * 16.5, ln, 8.6, '#334155', 'start', maxw=CW - 24, tag=f'cd{i}l{j}')

# ---------------- twin 双生 callout（indexer 卡 ↔ 状态账卡） ----------------
TW_Y = CY + CH + 14
tw_x0 = MX + 3 * (CW + GAP)            # C4I 卡左缘
tw_w = 2 * CW + GAP
lc.rect(tw_x0, TW_Y, tw_w, 58, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.seg(tw_x0 + CW / 2, CY + CH, tw_x0 + CW / 2, TW_Y - 2, lc.C_ZMQ_S, 1.6, 'std')
lc.seg(tw_x0 + CW + GAP + CW / 2, CY + CH, tw_x0 + CW + GAP + CW / 2, TW_Y - 2, lc.C_ENG_S, 1.6, 'std')
lc.text(tw_x0 + 14, TW_Y + 22, 'twin 双生：C4A 层的 indexer 自带一台压缩器（head 128 → 状态页 8,640B）', 9.6,
        lc.C_TXT, 'start', True, maxw=tw_w - 28, tag='tw1')
lc.text(tw_x0 + 14, TW_Y + 40, '与 attention 压缩器状态（32,832B）同层数 → zip 配成 21 对，进同一个 (4,8) 桶', 9.2,
        '#475569', 'start', maxw=tw_w - 28, tag='tw2')
lc.text(MX, TW_Y + 36, '（twin 是运行实证：档案理论段未记，', 9, lc.C_FAINT, 'start', maxw=300, tag='twn1')
lc.text(MX, TW_Y + 52, '普查口径按 168 份写）', 9, lc.C_FAINT, 'start', maxw=300, tag='twn2')

# ---------------- 普查合计条 ----------------
CB_Y = TW_Y + 78
lc.rect(MX, CB_Y, BXR - MX, 128, '#f8fafc', lc.C_MUTE, rx=10, sw=1.5)
lc.text(MX + 18, CB_Y + 24, '普查合计：44 + 62 + 42 + 20', 12, lc.C_TXT, 'start', True, maxw=420, tag='cb:t')
lc.text(BXR - 18, CB_Y + 24, '168 份 spec ＝ 后面 DSV4 专属分组与 packed 定账的全部输入', 9.5,
        lc.C_MUTE, 'end', maxw=520, tag='cb:r')

SEGS = [(lc.C_API_S, lc.C_API_F, '窗口 44', 44, '43 层 + MTP'),
        (lc.C_ZMQ_S, lc.C_ZMQ_F, 'full_mla 62', 62, '21 C4A 主 + 21 C4I + 20 C128 主'),
        (lc.C_ENG_S, lc.C_ENG_F, 'C4 状态 42', 42, '21 对双生（attn + indexer）'),
        (lc.C_SAM_S, lc.C_SAM_F, 'C128 状态 20', 20, 'block 8 / window 128')]
TOTAL = 168
BAR_X0, BAR_W, BAR_Y, BAR_H = MX + 18, BXR - MX - 36 - 96, CB_Y + 40, 44
bx = BAR_X0
for s, f, name, n, sub in SEGS:
    w_ = n / TOTAL * BAR_W
    lc.rect(bx, BAR_Y, w_, BAR_H, f, s, rx=3, sw=1.3)
    lc.text(bx + w_ / 2, BAR_Y + 19, name, 9.6, s, 'middle', True, maxw=w_ - 8, tag='cb:' + name)
    lc.text(bx + w_ / 2, BAR_Y + 35, sub, 7.4, s, 'middle', maxw=w_ - 8, tag='cb:s' + name[:4])
    bx += w_
lc.text(BAR_X0 + BAR_W + 10, BAR_Y + 27, '= 168 份', 12, lc.C_TXT, 'start', True, maxw=90, tag='cb:sum')
lc.text(MX + 18, CB_Y + 112, '账本管线不发明任何页大小，只收账分组——页大小全部由各 spec 自报（构造期对齐补齐已含在内）',
        9.2, '#475569', 'start', maxw=BXR - MX - 36, tag='cb:n')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 44, 'vllm/v1/attention/backends/mla/sparse_swa.py:L87-L102（窗口 spec · block 64）· vllm/models/deepseek_v4/attention.py:L655-L674（主账 spec · 纯窗口层 None）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 29, 'vllm/models/deepseek_v4/attention.py:L698-L710 + L781-L798（indexer spec · head_size=68）· L799-L809 + compressor.py:L295-L300（twin 压缩器）· compressor.py:L196-L203（状态 spec）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 14, '普查计数 / 页字节 ＝ 本章账本算术脚本实算（pin spec 类）· 行号基线 vLLM v0.27.1（6e448d0ea）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m7_ledger_census.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
