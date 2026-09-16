#!/usr/bin/env python3
"""ch29 机制图 · FP4 专家参数的字节布局(ch29-fig-fp4-byte-layout, 模板 layout)

放大自 L0『GPU 执行臂·模型层 forward + 编译』块里 MoE 专家参数的内存布局——
ch27 讲 FP4 格点数学, 本图讲它在 vLLM 参数张量里的字节落位(L2 拍片④/⑤ 的机制版下钻)。

claim: 一行 w13 权重的字节解剖——H 个 fp4 值打包成 H/2 个 uint8(两格一字节) +
H/32 个 UE8M0 尺度字节(每 32 值一个、只存 2 的幂指数); 参数以『已打包的 uint8
字节流』形态直接装载(数值 copy_ 会把 0.0078125 抹成 0), finalize 时才变换成
DeepGEMM 布局并丢弃原参数——52224 vs 196608 字节 = 3.7647× 压缩, 含 5.88% 尺度税。

数字 = 本章实跑字节账(与正文数值表同源) · 锚点 = vllm/models/deepseek_v4/nvidia/model.py:L202-L246/L310-L312/L324-L391/L1268-L1279。
坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 870
MX = 56
BXR = 1608
C_VAL_F, C_VAL_S = '#dbeafe', '#1d4ed8'
C_VAL2_F = '#eff6ff'
C_SC_F, C_SC_S = '#ffedd5', '#c2410c'

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'FP4 字节布局：两个 4 位值挤一字节，每 32 值配一位只会说 2 的幂的管理员', 16,
        lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, '权重以『已打包的 uint8 字节流』形态直接装载——装载纪律（按字节搬，不做数值换算）与 finalize 后的参数让位，都由这套布局决定',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
_ch = '放大自 L0『模型层 forward + 编译』块 · L2 拍片④/⑤'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- Panel A: 一行 w13 的字节解剖 ----------------
AX, AY_, AW, AH = MX, 90, 940, 320
lc.rect(AX, AY_, AW, AH, '#f8fafc', lc.C_GPU_S, rx=10, sw=1.8)
lc.text(AX + 16, AY_ + 24, '一行 w13 权重的字节解剖（H=128 个 fp4 值 → 64+4 字节）', 11.5,
        '#14532d', 'start', True, maxw=700, tag='pa:t')

BY0 = AY_ + 56
BW_, BH_ = 58, 44
x = AX + 24
lc.text(AX + 24, BY0 - 10, '值字节 ×64（H/2，两个 4 位值共享一字节）', 9.5, '#1e3a8a', 'start',
        True, maxw=500, tag='pa:vt')
for i in range(9):
    lc.rect(x, BY0, BW_, BH_, C_VAL_F, C_VAL_S, rx=3, sw=1.3)
    lc.rect(x, BY0, BW_ / 2, BH_, C_VAL2_F, C_VAL_S, rx=0, sw=0.9)
    lc.text(x + BW_ / 4, BY0 + 27, '4位', 7.5, '#1e3a8a', 'middle', tag='vb%d:a' % i)
    lc.text(x + 3 * BW_ / 4, BY0 + 27, '4位', 7.5, '#1e3a8a', 'middle', tag='vb%d:b' % i)
    if i == 0:
        lc.text(x + BW_ / 2, BY0 + BH_ + 14, '值0 | 值1', 7.5, lc.C_MUTE, 'middle', tag='vb:lab')
    x += BW_ + 4
lc.text(x, BY0 + 27, '……', 11, lc.C_MUTE, 'start', tag='vb:ell')
lc.text(x + 34, BY0 + 27, '(共 64 字节/行)', 8, lc.C_MUTE, 'start', tag='vb:cnt')

SY0 = BY0 + BH_ + 40
x = AX + 24
lc.text(AX + 24, SY0 - 10, '尺度字节 ×4（H/32，每 32 值一个 UE8M0，只存 2 的幂指数）', 9.5,
        C_SC_S, 'start', True, maxw=520, tag='pa:st')
for i in range(4):
    lc.rect(x, SY0, BW_, 34, C_SC_F, C_SC_S, rx=3, sw=1.3)
    lc.text(x + BW_ / 2, SY0 + 21, '2^k', 9, C_SC_S, 'middle', True, tag='sb%d' % i)
    x += BW_ + 4
lc.text(x + 6, SY0 + 21, '← 尺度解码 2^(b-127)，b=0 特判 0.0', 8.5, lc.C_MUTE, 'start',
        maxw=340, tag='sb:dec')

lc.text(AX + 24, SY0 + 68, '整张 w13_weight = [E_local, 2I, H//2] = [2, 256, 64] uint8（每专家 256 行 × 64 字节）',
        9, '#334155', 'start', maxw=AW - 48, tag='pa:shape')
lc.text(AX + 24, SY0 + 88, 'w13_weight_scale = [2, 256, 4] uint8（quant_method="block"）· w2 同构：[2,128,64] + [2,128,4]',
        9, '#334155', 'start', maxw=AW - 48, tag='pa:shape2')
lc.text(AX + 24, AY_ + AH - 12, 'vllm/models/deepseek_v4/nvidia/model.py:L202-L246', 8, lc.C_FAINT,
        'start', maxw=AW - 40, tag='pa:f')

# ---------------- Panel B: 字节账 ----------------
BX_, BYB, BWB, BHB = 1020, 90, BXR - 1020, 320
lc.rect(BX_, BYB, BWB, BHB, '#ffffff', lc.C_GPU_S, rx=10, sw=1.6)
lc.text(BX_ + 16, BYB + 24, '字节账（E_local=2 测试构造）', 11.5, '#14532d', 'start', True,
        maxw=400, tag='pb:t')
rows = [('w13_weight [2,256,64] uint8', '32768 B'), ('w13_weight_scale [2,256,4] uint8', '2048 B'),
        ('w2_weight [2,128,64] uint8', '16384 B'), ('w2_weight_scale [2,128,4] uint8', '1024 B')]
# 盲审修复 2026-09-14（第三轮）：原序「每行先写字、后铺本行斑马底色」——i%2==0 的
# 两行（w13_weight 32768 / w2_weight 16384）被 #f0fdf4 色块盖掉，账面上只剩两行
# scale、合计 52224 对不上。改为先铺全部斑马底、再写全部文字（与本章 four-kv 图同序）。
for i in range(len(rows)):
    if i % 2 == 0:
        yy = BYB + 52 + i * 24
        lc.rect(BX_ + 10, yy - 13, BWB - 20, 20, '#f0fdf4', 'none', rx=3, sw=0.5)
for i, (k, v) in enumerate(rows):
    yy = BYB + 52 + i * 24
    lc.text(BX_ + 16, yy, k, 9, '#334155', 'start', maxw=330, tag='pb:r%d' % i)
    lc.text(BX_ + BWB - 100, yy, v, 9.5, '#14532d', 'end', True, tag='pb:v%d' % i)
lc.seg(BX_ + 16, BYB + 152, BX_ + BWB - 16, BYB + 152, lc.C_MUTE, 1.2)
lc.text(BX_ + 16, BYB + 176, '合计 52224 B', 11, '#14532d', 'start', True, tag='pb:tot')
lc.text(BX_ + BWB - 16, BYB + 176, 'vs bf16 同逻辑 196608 B', 9.5, '#334155', 'end', tag='pb:bf16')
lc.text(BX_ + 16, BYB + 200, '压缩 3.7647×', 12, '#b91c1c', 'start', True, tag='pb:ratio')
lc.text(BX_ + BWB - 16, BYB + 200, '尺度税 5.88%', 9.5, C_SC_S, 'end', True, tag='pb:tax')
lc.text(BX_ + 16, BYB + 228, '每值成本恒等式：1/2 + 1/32 = 0.53125 B/值', 9.5, '#334155', 'start',
        maxw=BWB - 32, tag='pb:per')
lc.text(BX_ + 16, BYB + 248, '（与 E/I/H 无关，只由两条打包规则决定；', 8.5, lc.C_MUTE, 'start',
        maxw=BWB - 32, tag='pb:per2')
lc.text(BX_ + 16, BYB + 264, '  对 bf16 的 2 B/值恒为 3.7647× 压缩）', 8.5, lc.C_MUTE, 'start',
        maxw=BWB - 32, tag='pb:per3')
lc.text(BX_ + 16, BYB + 292, '每专家：w13 = 16384+1024 B · w2 = 8192+512 B', 8.5, '#334155',
        'start', maxw=BWB - 32, tag='pb:pe')
lc.text(BX_ + 16, BYB + BHB - 12, '本章实跑字节账（与正文数值表同源）', 8, lc.C_FAINT, 'start',
        maxw=BWB - 32, tag='pb:f')

# ---------------- Panel C: UE8M0 解码 + copy_ 死亡现场 ----------------
CX, CY, CW, CH = MX, 434, 780, 330
lc.rect(CX, CY, CW, CH, '#ffffff', lc.C_MUTE, rx=10, sw=1.5)
lc.text(CX + 16, CY + 24, 'UE8M0 解码与 copy_ 死亡现场', 11.5, lc.C_TXT, 'start', True, maxw=400,
        tag='pc:t')
lc.text(CX + 16, CY + 44, '解码公式：(sf.to(int32) << 23).view(float32)（model.py:L310-L312 逐字）',
        9, '#334155', 'start', maxw=CW - 32, tag='pc:formula')
# 解码表
tx0, ty0 = CX + 16, CY + 66
hdr = ['byte b', '0', '120', '126', '127', '128']
for j, hcell in enumerate(hdr):
    lc.text(tx0 + 90 + j * 96, ty0, hcell, 9, lc.C_MUTE, 'middle', True, tag='pc:h%d' % j)
vals = ['解码值', '0.0', '0.0078125', '0.5', '1.0', '2.0']
for j, vcell in enumerate(vals):
    col = C_SC_S if j in (1, 2) else '#334155'
    lc.text(tx0 + 90 + j * 96, ty0 + 20, vcell, 9, col, 'middle', bold=(j == 0), tag='pc:v%d' % j)
lc.text(tx0, ty0 + 44, 'b>0 → 2^(b-127)；b=0 特判 0.0 · 纯指数位模式 = 只能表示 2 的幂',
        8.5, lc.C_MUTE, 'start', maxw=CW - 32, tag='pc:sem')
lc.text(tx0, ty0 + 62, '树内测试 [0,126,127,128]→[0.0,0.5,1.0,2.0] 对拍通过（tests/models/test_deepseek_v4_mega_moe.py:L43-L54）',
        8.5, lc.C_MUTE, 'start', maxw=CW - 32, tag='pc:test')
# copy_ 现场
DY0 = CY + 158
lc.text(CX + 16, DY0 - 12, '装载纪律：数值 copy_ 会杀死小尺度，.view(torch.uint8) 按原始字节搬（model.py:L1268-L1279 注释实证）', 9.5,
        C_SC_S, 'start', True, maxw=CW - 32, tag='pc:copy')
cases = [('0.0078125（byte 120）', '→ 0', '→ 120', True),
         ('1.0（byte 127）', '→ 1', '→ 127', False),
         ('8.0（byte 130）', '→ 8', '→ 130', False)]
for i, (val, cp, vw, dead) in enumerate(cases):
    yy = DY0 + 16 + i * 30
    lc.text(CX + 26, yy, val, 9, '#334155', 'start', maxw=200, tag='cc:r%d' % i)
    lc.text(CX + 250, yy, '数值 copy_ ' + cp, 9, lc.C_ABORT if dead else '#334155', 'start',
            bold=dead, tag='cc:c%d' % i)
    if dead:
        # 盲审修复 2026-09-14：✗/✓（U+2717/U+2713）在本机字体栈渲染成空心豆腐，
        # 换 GB2312 内的 ×/√（任何 CJK 回退字体都有这两个字形）。
        lc.text(CX + 345, yy, '× 死', 9, lc.C_ABORT, 'start', True, tag='cc:x%d' % i)
    lc.text(CX + 430, yy, '.view(uint8) ' + vw, 9, '#166534', 'start', tag='cc:v%d' % i)
    if dead:
        lc.text(CX + 555, yy, '√ 活', 9, '#166534', 'start', True, tag='cc:o%d' % i)
lc.text(CX + 16, CY + CH - 34, '只有小于 uint8 量化步长的尺度会死（0.0078125=2^-7 被抹成 0；1.0/8.0 数值无损）——',
        8.5, lc.C_MUTE, 'start', maxw=CW - 32, tag='pc:note')
lc.text(CX + 16, CY + CH - 18, '装载纪律一句话：e8m0fnu 尺度必须 .view(torch.uint8) 按字节原样装', 8.5,
        C_SC_S, 'start', True, maxw=CW - 32, tag='pc:hot')

# ---------------- Panel D: 装载→finalize 流程 ----------------
DX, DY_, DW, DH = 860, 434, BXR - 860, 330
lc.rect(DX, DY_, DW, DH, '#ffffff', lc.C_MUTE, rx=10, sw=1.5)
lc.text(DX + 16, DY_ + 24, '装载 → finalize：原参数让位给 DeepGEMM 布局', 11.5, lc.C_TXT, 'start',
        True, maxw=560, tag='pd:t')
fy = DY_ + 56
lc.rect(DX + 16, fy, 200, 64, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(DX + 116, fy + 26, 'checkpoint', 9.5, lc.C_TXT, 'middle', True, tag='pd:c1')
lc.text(DX + 116, fy + 45, '尺度 dtype=float8_e8m0fnu', 8, lc.C_MUTE, 'middle', maxw=190,
        tag='pd:c1s')
lc.seg(DX + 216, fy + 32, DX + 300, fy + 32, lc.C_GPU_S, 1.8, 'std')
lc.text(DX + 258, fy + 24, '.view(uint8)', 8, '#166534', 'middle', True, maxw=84, tag='pd:a1')
lc.rect(DX + 300, fy, 200, 64, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.4)
lc.text(DX + 400, fy + 26, 'uint8 参数张量', 9.5, '#14532d', 'middle', True, tag='pd:c2')
lc.text(DX + 400, fy + 45, '（字节流直达，零换算）', 8, lc.C_MUTE, 'middle', maxw=190, tag='pd:c2s')
lc.seg(DX + 500, fy + 32, DX + 584, fy + 32, lc.C_GPU_S, 1.8, 'std')
lc.text(DX + 542, fy + 24, 'finalize_weights', 7.5, '#166534', 'middle', True, maxw=84, tag='pd:a2')
lc.rect(DX + 584, fy, DW - 600, 64, '#ffffff', lc.C_GPU_S, rx=7, sw=1.4)
lc.text(DX + 584 + (DW - 600) / 2, fy + 26, 'DeepGEMM 布局', 9.5, '#14532d', 'middle', True,
        maxw=170, tag='pd:c3')
lc.text(DX + 584 + (DW - 600) / 2, fy + 45, '（transform_sf + transform_weights）', 7.5, lc.C_MUTE,
        'middle', maxw=190, tag='pd:c3s')
for i, s in enumerate([
        '· finalize 后 w13_weight / w13_weight_scale / w2_weight /',
        '   w2_weight_scale 四个原参数全部置 None（model.py:L324-L362）',
        '· finalize 需 SM100 + deep_gemm（代码引用，不在 host 复现范围）',
        '· 权重热更新 / EPLB 重排走 get_expert_weights 专用视图',
        '   （model.py:L407-L434，不再依赖原参数）',
        '· 对称缓冲按 7 元组键 (group, device, E, max_tokens, topk, H, I)',
        '   跨层复用（model.py:L364-L391）']):
    lc.text(DX + 16, DY_ + 150 + i * 20, s, 8.5, '#334155', 'start', maxw=DW - 32, tag='pd:l%d' % i)
# 原参数置 None 的叉标记
lc.text(DX + 300, fy + 82, '↘ 四个原参数置 None', 8, lc.C_ABORT, 'start', True, maxw=180,
        tag='pd:none')

# ---------------- 页脚 ----------------
lc.text(MX, 800, '图例：蓝格 = fp4 值字节（半格=4 位）· 橙格 = UE8M0 尺度字节 · 绿 = 装载/finalize 流程（GPU 执行臂角色）· 红 = 数值 copy_ 死亡',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 820, '字节账/解码/死亡现场 = 本章实跑（E_local=2、H=I=128 测试构造，与正文数值表同源）· 锚点 = vllm/models/deepseek_v4/nvidia/model.py:L202-L246/L310-L312/L324-L391/L1268-L1279',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')
# 盲审修复 2026-09-14（第三轮）：图例原写『4B UE8M0 管 64 值』与本图自身规则
# 『每 32 值一个尺度字节』冲突——68B/条 = 64 打包字节 + 4 UE8M0，head_dim=128 个值
# 打包进 64 字节、4 个尺度各管 32 值 = 128 值（64 是打包字节数、不是值数）。
lc.text(MX, 840, '对照 KV 侧同族布局：主 MLA 584B/token（8B scale 管 576 值）· indexer 68B/条（4B UE8M0 管 128 值）——组尺度哲学贯穿权重与缓存',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-fp4-byte-layout.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
