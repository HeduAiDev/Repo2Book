#!/usr/bin/env python3
"""ch36 机制图 · 层普查 compress_ratios（figure_spec fig_m4_layer_census，模板 layout）

放大自 L2 章图拍片②（层普查 compress_ratios）——L0 启动视角『模型形状进引擎』第一格。

claim：一列 44 项 compress_ratios 决定全部账本的层数：[0,0,(4,128)×20,4,0] 逐层取
max(1,r) → 2 纯窗口层 + 21 C4A（带 indexer）+ 20 C128 + MTP 固定 1（末位 0 占位、
layer_id≥43 走 else 分支永不查表）。

数字取自 figure_spec.numbers（config.json 44 项 / attention.py:L210-L213 / L277-L297 /
普查 44/62/42+20=168）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1460, 648
MX, BXR = 56, 1404
DEFS = lc.DEFS

C_WIN_S, C_WIN_F = lc.C_API_S, lc.C_API_F      # 纯窗口
C_C4A_S, C_C4A_F = lc.C_GPU_S, lc.C_GPU_F      # C4A（压缩 4:1 · 带 indexer）
C_C128_S, C_C128_F = lc.C_KV_S, lc.C_KV_F      # C128（压缩 128:1）
C_MTP_S, C_MTP_F = lc.C_ENG_S, lc.C_BADGE_F    # MTP 占位

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一列 44 项 compress_ratios 写死 43 层的身份——五本账各自的层数从这里数出来', 16.5,
        lc.C_TXT, 'start', True, maxw=1100, tag='title')
lc.text(MX, 58, '逐层 max(1, ratios[layer_id])；layer_id ≥ 43（MTP）固定 1、永不查表——末位 0 只是占位 · 仅 compress_ratio==4 的层装 Lightning Indexer',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L2 拍片② 层普查 · L0 启动视角'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 44 格纸带 ----------------
STRIPE_Y, CELL_H, CELL_W, CGAP = 104, 46, 26, 3.5
SX0 = 84
ratios = [0, 0]
for _ in range(20):
    ratios += [4, 128]
ratios += [4, 0]
assert len(ratios) == 44

lc.text(MX, 92, 'config.json · compress_ratios（44 项，逐层一项）', 10.5, lc.C_TXT, 'start', True,
        maxw=420, tag='cfg')
lc.text(BXR, 92, 'vllm/models/deepseek_v4/attention.py:L210-L213', 8.8, lc.C_FAINT, 'end', maxw=360, tag='cfg:r')

for i, v in enumerate(ratios):
    x = SX0 + i * (CELL_W + CGAP)
    if i == 43:                                     # MTP 占位：末位 0、虚线框
        lc.rect(x, STRIPE_Y, CELL_W, CELL_H, C_MTP_F, C_MTP_S, rx=4, sw=1.6, dash=True)
    elif v == 0:
        lc.rect(x, STRIPE_Y, CELL_W, CELL_H, C_WIN_F, C_WIN_S, rx=4, sw=1.2)
    elif v == 4:
        lc.rect(x, STRIPE_Y, CELL_W, CELL_H, C_C4A_F, C_C4A_S, rx=4, sw=1.2)
    else:
        lc.rect(x, STRIPE_Y, CELL_W, CELL_H, C_C128_F, C_C128_S, rx=4, sw=1.2)
    lc.text(x + CELL_W / 2, STRIPE_Y + 28, str(v), 9.5,
            C_MTP_S if i == 43 else (C_WIN_S if v == 0 else (C_C4A_S if v == 4 else C_C128_S)),
            'middle', True, maxw=CELL_W - 2, tag=f'cell{i}')
    if i % 4 == 0 or i == 43:                       # 每 4 层一枚层号
        lc.text(x + CELL_W / 2, STRIPE_Y + CELL_H + 14, str(i), 7.5, lc.C_FAINT, 'middle', maxw=24, tag=f'idx{i}')

lc.text(SX0 - 10, STRIPE_Y + CELL_H / 2 + 4, '层号', 8.2, lc.C_FAINT, 'end', maxw=40, tag='idxl')

# ---------------- 图例 ----------------
LGY = STRIPE_Y + CELL_H + 30
sw = [(C_WIN_S, C_WIN_F, '纯窗口（ratio 0 → max(1,·)=1）', False),
      (C_C4A_S, C_C4A_F, 'C4A：压缩 4:1 · 带 Lightning Indexer', False),
      (C_C128_S, C_C128_F, 'C128：压缩 128:1 · 无 indexer', False),
      (C_MTP_S, C_MTP_F, 'MTP 占位（固定 1）', True)]
lx0 = SX0
for s, f, name, dash in sw:
    lc.rect(lx0, LGY - 9, 16, 11, f, s, rx=3, sw=1.6, dash=dash)
    lc.text(lx0 + 21, LGY + 1, name, 9.2, lc.C_TXT, 'start', maxw=300, tag='lg:' + name[:6])
    lx0 += 21 + lc.tw(name, 9.2) + 26

# ---------------- 普查面板 ----------------
PY, PH = 196, 268
lc.rect(MX, PY, BXR - MX, PH, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(MX + 18, PY + 26, '普查：这一列数字数出五本账各自的层数', 12.5, lc.C_TXT, 'start', True,
        maxw=640, tag='census:t')
lc.text(BXR - 18, PY + 26, '168 份 spec ＝ 分组与 packed 定账的全部输入', 9.5, lc.C_MUTE, 'end',
        maxw=380, tag='census:r')

GRP = [
    (C_WIN_S, C_WIN_F, '窗口账', '44 本', '43 层 + MTP 人人有份 · SlidingWindowMLASpec(64,128)'),
    (lc.C_ZMQ_S, lc.C_ZMQ_F, 'full_mla 组', '62 本', '21 C4A 主 + 21 C4I（indexer）+ 20 C128 主 · 合成一个组'),
    (lc.C_ENG_S, lc.C_ENG_F, 'C4 状态账', '42 本', '21 对双生：attention 压缩器 + indexer 压缩器（twin）'),
    (lc.C_SAM_S, lc.C_SAM_F, 'C128 状态账', '20 本', '每 8 token 一份状态 · block 8 / window 128'),
]
GY0 = PY + 44
GH = 52
for i, (s, f, name, count, note) in enumerate(GRP):
    gy = GY0 + i * GH
    lc.rect(MX + 18, gy, 1180, GH - 8, f, s, rx=7, sw=1.2)
    lc.text(MX + 32, gy + 20, name, 10.5, s, 'start', True, maxw=140, tag='g:n' + name)
    lc.text(MX + 178, gy + 20, count, 13, s, 'start', True, maxw=90, tag='g:c' + name)
    lc.text(MX + 268, gy + 20, note, 9.2, '#334155', 'start', maxw=920, tag='g:o' + name[:4])

# 合计行
TY = GY0 + 4 * GH + 4
lc.text(MX + 32, TY + 14, '合计', 10.5, lc.C_TXT, 'start', True, maxw=60, tag='sum')
lc.text(MX + 178, TY + 14, '168 份', 13, lc.C_TXT, 'start', True, maxw=110, tag='sum:v')
lc.text(MX + 268, TY + 14, '44 + 62 + 42 + 20 = 168（运行实证：C4A 层的 indexer 自带一台压缩器，(4,8) 桶因此是 21 对双生）',
        9.2, '#334155', 'start', maxw=920, tag='sum:n')

# ---------------- 底注 ----------------
lc.text(MX, PY + PH + 26, '读法：第 0/1 层纯窗口（get_kv_cache_spec 返回 None，不进压缩主账）· 奇偶交替 C4A/C128 · 末位 0 是给 MTP 留的空位（layer_id≥43 走 else 分支）',
        9.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='note1')
lc.text(MX, PY + PH + 44, '仅 compress_ratio==4 的 21 层装 Lightning Indexer（attention.py:L277-L297）——indexer 账与 twin 状态账的存在全由这一条判定引出',
        9.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='note2')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 32, 'compress_ratios 44 项 ＝ deepseek-ai/DeepSeek-V4-Flash config.json（HF 官方）· vllm/models/deepseek_v4/attention.py:L210-L213（max(1,r) · MTP 固定 1）· '
                    'L277-L297（仅 ratio==4 装 indexer）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 17, '普查 44 / 62 / 42+20 = 168 份 spec ＝ 本章账本算术脚本实算（pin spec 类）· 行号基线 vLLM v0.27.1（6e448d0ea）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m4_layer_census.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
