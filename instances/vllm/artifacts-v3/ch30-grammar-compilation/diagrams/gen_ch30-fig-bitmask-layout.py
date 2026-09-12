#!/usr/bin/env python3
"""ch30 机制图 · m21 位掩码的物理形态（figure_spec ch30-fig-bitmask-layout，模板 layout）

放大自 L0 采样列·结构化输出组的『产物表示』层——allocate_token_bitmask 的物理形态
（L2 章图站 8 交棒件的内部视图；行的流转归 ch31），架构归属回指 L2 章图，不另立
第二种架构画法（FIGURE-SYSTEM §3）。

claim：位掩码行的物理形态：[行, ceil(V/32)] int32 的张量（本例 [16,1571]、每行 6284B；
128k 词表≈16KB/行）=逐 token fp32 logits 的 1/32——位打包是它能上每步热路径的定量
理由；-1 补码=全 1=全允许（非语法行的畅通兜底）。

数字全部取自 figure_spec.numbers（bitmask_layout：[16,1571]/6284 B/100544 B；
dossier theory[1]：129280→4040 int32=16160 B≈16KB vs ≈512KB 恰 1/32；预算公式
max_num_seqs*(1+num_spec)；-1 三处互证锚）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 782
MX, BXR = 60, 1440
BIT_ON = '#1e293b'      # bit=1（允许）暗格
BIT_OFF = '#f1f5f9'     # bit=0（→-inf）亮格

# ---------------- 标题区 ----------------
lc.text(MX, 34, '位掩码长什么样：一行 = 词表按 32 打包的 1571 个 int32，恰是 fp32 logits 的 1/32',
        16.5, lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'allocate_token_bitmask(16, 50257) → [16, 1571] int32 · 每行 6284 B · 16 行共 100544 B；'
               '生产 128k 词表 ≈16KB/行 vs 逐 token fp32 logits ≈512KB——32 倍位打包是它上每步热路径的定量理由',
        10.5, lc.C_MUTE, 'start', maxw=1340, tag='subtitle')
_ch = '放大自 L0 采样列·结构化输出组 · L2 站 8『交棒』件内部视图'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= ① 张量形态（左） =================
F1X, F1Y, F1W, F1H = MX, 92, 692, 470
lc.rect(F1X, F1Y, F1W, F1H, lc.C_SAM_F, lc.C_SAM_S, rx=10, sw=1.8)
lc.text(F1X + 16, F1Y + 22, '① 张量形态：[行, ceil(V/32)] int32', 11.5, lc.C_SAM_S, 'start', True,
        maxw=520, tag='f1:t')
lc.text(F1X + 16, F1Y + 40, 'allocate_token_bitmask(16, 50257) → shape [16, 1571]', 8.6,
        '#334155', 'start', maxw=520, tag='f1:l0')

# 16 行栅格（行=请求槽位）
GX, GY, GW, RH, RG = F1X + 20, F1Y + 58, 148, 18, 4
row_cy = []
for i in range(16):
    yy = GY + i * (RH + RG)
    row_cy.append(yy + RH / 2)
    if i == 0:                                   # 语法请求槽位（高亮）
        lc.rect(GX, yy, GW, RH, '#ffffff', lc.C_SAM_S, rx=3, sw=2.0)
    elif i == 3:                                 # -1 兜底行（全暗）
        lc.rect(GX, yy, GW, RH, BIT_ON, lc.C_SAM_S, rx=3, sw=1.2)
    else:
        lc.rect(GX, yy, GW, RH, '#ffffff', lc.C_MUTE, rx=3, sw=1.0)
    for k in range(1, 8):                        # int32 块刻度
        lc.seg(GX + k * GW / 8, yy + 2, GX + k * GW / 8, yy + RH - 2, '#e2e8f0', 0.5)
lc.text(GX, GY - 8, '16 行 = 请求槽位', 8.4, lc.C_TXT, 'start', True, maxw=150, tag='g:t')
lc.text(GX + GW / 2, GY + 16 * (RH + RG) + 14, '每行 1571 个 int32（示意压缩）', 7.6, lc.C_MUTE,
        'middle', maxw=200, tag='g:sub')
lc.text(F1X + 16, F1Y + F1H - 30, '分配预算：max_num_seqs × (1 + num_speculative_tokens) 行', 8.6,
        '#334155', 'start', maxw=640, tag='f1:b1')
lc.text(F1X + 16, F1Y + F1H - 14, '+1 = bonus / 非 spec 位 · spec 位给投机窗口预留（下一章展开·预告）', 8.2,
        lc.C_MUTE, 'start', maxw=640, tag='f1:b2')

# 三个放大行（右侧）
ZX, ZW = F1X + 200, F1W - 220
ZOOMS = [
    ('放大 · 语法请求行，第 1 步（位置 0）', '5 位=1（77 / 88 / 3919 / 5948 / 8505）· 其余位=0', 'few'),
    ('放大 · 同一语法请求行，第 2 步（位置 1）', '行内容逐拍重填：仅 1 位=1（50256 EOS）', 'eos'),
    ('放大 · 非语法请求行（-1 兜底）', 'torch.full(..., -1)：全 1 = 全允许——不设限、畅通无阻', 'all'),
]
zh = 116
for zi, (t_, cap, kind) in enumerate(ZOOMS):
    zy = F1Y + 58 + zi * (zh + 12)
    lc.rect(ZX, zy, ZW, zh, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(ZX + 12, zy + 18, t_, 9, lc.C_TXT, 'start', True, maxw=ZW - 24, tag='z' + str(zi) + 't')
    # 位条（带省略号断口）
    sx0, sx1, sy = ZX + 16, ZX + ZW - 16, zy + 30
    lc.rect(sx0, sy, sx1 - sx0, 18, '#ffffff', lc.C_MUTE, rx=3, sw=1.0)
    for i in range(1, 10):
        lc.seg(sx0 + i * (sx1 - sx0) / 10, sy + 2, sx0 + i * (sx1 - sx0) / 10, sy + 16, '#e2e8f0', 0.5)
    # 断口标记（左 1/3 与右 1/3 处的省略）
    for bx in (0.33, 0.66):
        lc.rect(sx0 + (sx1 - sx0) * bx - 10, sy - 1, 20, 20, '#ffffff', lc.C_MUTE, rx=2, sw=0.8)
        lc.text(sx0 + (sx1 - sx0) * bx, sy + 13, '…', 7.5, lc.C_MUTE, 'middle', maxw=18, tag='zbr' + str(zi))
    if kind == 'few':
        # 真实比例位（全部落在词表前 17%；77/88 挤在最左端）——id 清单见框内说明行
        for tid in (77, 88, 3919, 5948, 8505):
            frac = tid / 50256
            lc.rect(sx0 + (sx1 - sx0) * frac - 1.5, sy + 1, 3, 16, BIT_ON, BIT_ON, rx=1, sw=0)
    elif kind == 'eos':
        lc.rect(sx1 - 5, sy + 1, 3, 16, BIT_ON, BIT_ON, rx=1, sw=0)
    else:
        lc.rect(sx0 + 1, sy + 1, sx1 - sx0 - 2, 16, BIT_ON, BIT_ON, rx=2, sw=0)
    lc.text(ZX + 12, zy + zh - 10, cap, 8, '#334155', 'start', maxw=ZW - 24, tag='z' + str(zi) + 'c')
# 引线：栅格行 → 放大行
lc.seg(GX + GW, row_cy[0], ZX, F1Y + 58 + zh / 2, lc.C_MUTE, 1.0, dash=True)
lc.seg(GX + GW, row_cy[0] + 4, ZX, F1Y + 58 + zh + 12 + zh / 2, lc.C_MUTE, 1.0, dash=True)
lc.seg(GX + GW, row_cy[3], ZX, F1Y + 58 + 2 * (zh + 12) + zh / 2, lc.C_MUTE, 1.0, dash=True)

# ================= ② 位约定（右上） =================
F2X, F2Y, F2W, F2H = 778, 92, BXR - 778, 272
lc.rect(F2X, F2Y, F2W, F2H, '#ffffff', lc.C_SAM_S, rx=10, sw=1.8)
lc.text(F2X + 16, F2Y + 22, '② 位约定：一个 int32 = 32 个 token 的允许位', 11.5, lc.C_SAM_S,
        'start', True, maxw=560, tag='f2:t')
lc.text(F2X + 16, F2Y + 42, 'int32 #265（token 8480-8511 的 32 位）：', 8.4,
        '#334155', 'start', maxw=620, tag='f2:l0')
cx0, cy0, cp = F2X + 20, F2Y + 52, 19.2
for i in range(32):
    on = i == 25
    lc.rect(cx0 + i * cp, cy0, cp - 1.5, 16, BIT_ON if on else BIT_OFF, lc.C_MUTE, rx=2, sw=0.7)
lc.text(F2X + 16, F2Y + 84, 'bit 25 = 8505 "yes" 置 1，其余 31 位=0 → 除 yes 外全 -inf（bit 0 … bit 31）', 8,
        '#334155', 'start', maxw=640, tag='f2:b25')
lc.text(F2X + 16, F2Y + 108, 'bit=1 = 允许（暗格）· bit=0 → 该 token 的 logit 写 -inf（xgrammar 约定）', 8.6,
        lc.C_TXT, 'start', True, maxw=640, tag='f2:l1')
lc.text(F2X + 16, F2Y + 130, 'int32 = -1（补码全 1）= 32 格全允许：', 8.4, '#334155', 'start',
        maxw=620, tag='f2:l2')
for i in range(32):
    lc.rect(cx0 + i * cp, F2Y + 140, cp - 1.5, 16, BIT_ON, lc.C_MUTE, rx=2, sw=0.7)
lc.text(F2X + 16, F2Y + 178, '同一条约定三处落地：后端分配 torch.full(..., -1)（outlines / LMFE）·', 8.2,
        '#334155', 'start', maxw=640, tag='f2:l3')
lc.text(F2X + 16, F2Y + 194, '_full_mask=-1 非语法行兜底 · worker 侧重排基底 -1 预填', 8.2,
        '#334155', 'start', maxw=640, tag='f2:l4')
lc.text(F2X + 16, F2Y + 212, '（xgrammar 侧 bit=1=允许、0 才被 -inf——三处源码互证；非语法请求行 /', 8,
        lc.C_MUTE, 'start', maxw=640, tag='f2:l5')
lc.text(F2X + 16, F2Y + 228, '  思考段请求行以 -1 预填即「不设限」）', 8,
        lc.C_MUTE, 'start', maxw=640, tag='f2:l6')

# ================= ③ 1/32 对比条（右下） =================
F3X, F3Y, F3W, F3H = 778, 380, BXR - 778, 230
lc.rect(F3X, F3Y, F3W, F3H, '#ffffff', lc.C_MUTE, rx=10, sw=1.5)
lc.text(F3X + 16, F3Y + 22, '③ 1/32：位打包是它上每步热路径的定量理由（128k 词表 129280）', 11.5,
        lc.C_TXT, 'start', True, maxw=620, tag='f3:t')
bx0, bx1 = F3X + 180, F3X + F3W - 20
UNIT = (bx1 - bx0) / 34.0        # 16KB 的条宽 = 1 单位；512KB = 32 单位
bars = [('位掩码行', '4040 个 int32 = 16160 B ≈ 16KB', 1, lc.C_SAM_S, lc.C_SAM_F),
        ('fp32 logits 行', '129280 × 4 B ≈ 512KB', 32, lc.C_MUTE, '#e2e8f0')]
for i, (nm, val, units, s_, f_) in enumerate(bars):
    by = F3Y + 58 + i * 58
    lc.text(F3X + 16, by + 12, nm, 8.8, lc.C_TXT, 'start', True, maxw=140, tag='f3:n' + str(i))
    lc.text(bx0, by - 6, val, 8.2, '#334155', 'start', maxw=400, tag='f3:v' + str(i))
    lc.rect(bx0, by, UNIT * units, 24, f_, s_, rx=4, sw=1.4)
lc.text(bx0 + UNIT + 10, F3Y + 58 + 12, '← 恰为 1/32', 8.4, lc.C_SAM_S, 'start', True, maxw=120,
        tag='f3:ratio')
lc.text(F3X + 16, F3Y + 186, '每 token 4 字节 → 1 位——每步跨进程 + H2D 的行就这么大，掩码才跟得起', 8.4,
        '#334155', 'start', maxw=640, tag='f3:l1')
lc.text(F3X + 16, F3Y + 204, '逐拍采样；行的流转（跨进程 ndarray / pinned H2D / 盖 logits）归下一章（预告）', 8.4,
        '#334155', 'start', maxw=640, tag='f3:l2')

# ================= 图例 + 页脚 =================
LY = 712
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '采样列·结构化输出组（本章 L0 位置）', 8.8, lc.C_TXT, 'start', maxw=290, tag='lg1')
lx0 += 21 + lc.tw('采样列·结构化输出组（本章 L0 位置）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, BIT_ON, BIT_ON, rx=2, sw=0.8)
lc.text(lx0 + 21, LY + 1, 'bit=1（允许）暗格 · 全暗=全允许（-1）', 8.8, lc.C_TXT, 'start', maxw=290, tag='lg2')
lx0 += 21 + lc.tw('bit=1（允许）暗格 · 全暗=全允许（-1）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, BIT_OFF, lc.C_MUTE, rx=2, sw=0.8)
lc.text(lx0 + 21, LY + 1, 'bit=0（→写 -inf）亮格', 8.8, lc.C_TXT, 'start', maxw=190, tag='lg3')

lc.text(MX, 740, 'vllm/v1/structured_output/__init__.py:L225-L234（max_num_seqs×(1+num_spec) 预算公式）· '
                 'backend_outlines.py:L99-L105 / backend_lm_format_enforcer.py:L141-L147（torch.full(-1)）· '
                 '__init__.py:L58+L205（_full_mask=-1 兜底）· utils.py:L126-L131（重排基底 -1 预填）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 756, '[16,1571] / 6284 B / 100544 B / 位置 0-1 允许位 ＝ 本章驱动脚本实测（gpt2 词表 50257）· '
                 '129280→4040 int32=16160 B 与 1/32 ＝ 分配公式算术（≈512KB 为 129280×4B 取整）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch30-fig-bitmask-layout.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
