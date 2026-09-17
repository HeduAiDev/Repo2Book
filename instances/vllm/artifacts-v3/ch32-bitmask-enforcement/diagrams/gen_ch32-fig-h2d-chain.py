#!/usr/bin/env python3
"""ch32 机制图 · payoff 函数的搬运链（figure_spec ch32-fig-h2d-chain，模板 flow）

放大自 L0 采样列·结构化输出组 worker 侧『H2D 落地』段展开（L2 站 13 的机制放大——
本章 payoff 函数的搬运链），架构归属回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。
CPU 侧恒 C_ENG_S 橙、GPU 侧恒 C_GPU_S 绿（角色色铁律）。

claim：sorted_bitmask 先在 CPU 上以 pinned 内存成型，再 non_blocking 异步 DMA 上卡，
xgr.apply_token_bitmask_inplace 原地把 bit=0 的 logits 写 -inf；indices 需要时自己经
async_tensor_h2d 搬（免 xgrammar 内 cpu sync）。

数字全部取自 figure_spec.numbers（生产形 [256,1571] int32=1.534MiB pinned 78.1µs vs
pageable 132.2µs 1.69 倍；skip 快路径 async_tensor_h2d 未被调用；部分覆盖
out_indices=[5,0,1,2,3] int32 上卡；utils.py:L143-L161 + torch_utils.py:L573-L588）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 780
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="gpu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
    f'<marker id="eng" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ENG_S}"/></marker>'
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'payoff 函数的搬运链：pinned CPU 成型 → 异步 DMA 上卡 → xgr 原地写 -inf', 16.5,
        lc.C_TXT, 'start', True, maxw=1050, tag='title')
lc.text(MX, 58, '同一载荷 [256, 1571] int32 = 1.534MiB：pinned 78.1µs vs pageable 132.2µs（1.69 倍）；'
               'indices 的巧劲——xgrammar 名义要 python list、实际吃 tensor：自己搬就能 non_blocking 且 xgrammar 内无 cpu sync',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列·worker 侧 H2D 落地 · L2 站 13 的机制放大（本章 payoff）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 进程分界底带 =================
CY0, CY1 = 110, 450
lc.rect(MX, CY0, 780, CY1 - CY0, lc.C_ENG_F, lc.C_ENG_S, rx=10, sw=1.6, dash=True)
lc.text(MX + 14, CY0 + 20, 'CPU · worker 进程（pinned 内存区）', 10, lc.C_ENG_S, 'start', True,
        maxw=400, tag='lane:cpu')
lc.rect(900, CY0, BXR - 900, CY1 - CY0, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.6, dash=True)
lc.text(914, CY0 + 20, 'GPU · logits 所在地', 10, lc.C_GPU_S, 'start', True, maxw=300,
        tag='lane:gpu')

# ================= 四站横排 =================
def station(x, y, w, h, title, lines, stroke, fill='#ffffff'):
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.6)
    lc.text(x + 12, y + 20, title, 9.6, lc.C_TXT, 'start', True, maxw=w - 24, tag='st:' + title[:8])
    for j, ln in enumerate(lines):
        lc.text(x + 12, y + 40 + j * 15.5, ln, 7.8, '#334155', 'start', maxw=w - 22,
                tag='sl:' + title[:6] + str(j))

S1X, S1Y, S1W, S1H = 84, 158, 240, 150
station(S1X, S1Y, S1W, S1H, 'sorted_bitmask（CPU）',
        ['重排产物，生来 pinned：', 'torch.full(..., pin_memory=', '  PIN_MEMORY)——pinned 页',
         '锁页物理内存，DMA 直取', '形状 [256, 1571] int32', '= 1.534MiB'], lc.C_ENG_S)
S2X, S2Y, S2W, S2H = 930, 158, 200, 110
station(S2X, S2Y, S2W, S2H, 'GPU bitmask 张量',
        ['device 端 [256, 1571]', 'int32 · 与 logits 同卡', '（bit=1 允许 · bit=0 禁）'], lc.C_GPU_S)
S3X, S3Y, S3W, S3H = 1240, 158, 180, 110
station(S3X, S3Y, S3W, S3H, 'logits [B, V]',
        ['fp32 · bit=0 位概率归零', '（非法 token 的分数）'], lc.C_SAM_S)

# DMA 主箭头（跨进程带）
lc.parrow([(S1X + S1W + 6, S1Y + S1H / 2), (S2X - 6, S2Y + S2H / 2)], lc.C_ENG_S, 2.6, 'gpu')
lc.text((S1X + S1W + S2X) / 2, S1Y + S1H / 2 - 34, '.to(logits.device,', 8.6, lc.C_ENG_S, 'middle',
        True, maxw=260, tag='dma:t1')
lc.text((S1X + S1W + S2X) / 2, S1Y + S1H / 2 - 20, '  non_blocking=True)', 8.6, lc.C_ENG_S,
        'middle', True, maxw=260, tag='dma:t2')
lc.text((S1X + S1W + S2X) / 2, S1Y + S1H / 2 + 18, '异步 DMA · PCIe', 8, lc.C_MUTE, 'middle',
        maxw=200, tag='dma:s1')
lc.text((S1X + S1W + S2X) / 2, S1Y + S1H / 2 + 34, '78.1µs / 1.534MiB（pinned 中位）', 8,
        lc.C_ENG_S, 'middle', True, maxw=280, tag='dma:s2')

# xgr kernel 应用框（GPU 带内下方，跨 S2/S3 之下）
KX, KY, KW, KH = 930, 310, 490, 120
lc.rect(KX, KY, KW, KH, lc.C_SAM_F, lc.C_SAM_S, rx=9, sw=1.8)
lc.text(KX + 14, KY + 20, 'xgr.apply_token_bitmask_inplace(logits, bitmask, indices)', 9.4,
        lc.C_SAM_S, 'start', True, maxw=KW - 28, tag='kr:t')
for j, ln in enumerate(['逐行逐位：bit=1 → 允许不动 · bit=0 → 写 -inf（概率精确归零）',
                        '原地改写——采样器随后读到的就是改后分数（ch30 契约的第一棒）']):
    lc.text(KX + 14, KY + 44 + j * 17, ln, 8, '#334155', 'start', maxw=KW - 26, tag='kr:l' + str(j))
lc.text(KX + 14, KY + KH - 10, 'indices 需要时才传：部分覆盖场景 [5, 0, 1, 2, 3]', 7.8, lc.C_MUTE,
        'start', maxw=KW - 26, tag='kr:f')

# GPU 带内箭头：bitmask 张量 ↓ kernel；logits ↓ kernel
lc.seg(S2X + S2W / 2, S2Y + S2H, S2X + S2W / 2, KY, lc.C_GPU_S, 2.0, 'gpu')
lc.text(S2X + S2W / 2 + 8, S2Y + S2H + 20, 'bitmask', 8, lc.C_GPU_S, 'start', maxw=80, tag='a1')
lc.parrow([(S3X + S3W / 2, S3Y + S3H), (S3X + S3W / 2, KY)], lc.C_SAM_S, 2.0, 'sam')
lc.text(S3X + S3W / 2 + 8, S3Y + S3H + 20, 'logits 原位', 8, lc.C_SAM_S, 'start', maxw=100, tag='a2')

# ================= 支线：out_indices 的搬运（CPU 带内下 + 跨带到 kernel） =================
UX, UY, UW, UH = 84, 344, 240, 36
lc.rect(UX, UY, UW, UH, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(UX + 12, UY + 23, 'out_indices（python list）', 8.4, lc.C_TXT, 'start', True,
        maxw=UW - 20, tag='ux:t')
HX, HY, HW, HH = 430, 306, 330, 80
lc.rect(HX, HY, HW, HH, '#ffffff', lc.C_ENG_S, rx=8, sw=1.6)
lc.text(HX + 14, HY + 20, 'async_tensor_h2d —— 自己搬 indices', 9.4, lc.C_ENG_S, 'start', True,
        maxw=HW - 28, tag='h:t')
lc.text(HX + 14, HY + 38, 'xgrammar 名义要 list、实际吃 tensor：', 7.8, '#334155', 'start',
        maxw=HW - 26, tag='h:l1')
lc.text(HX + 14, HY + 54, '自己搬 → non_blocking 且 xgrammar 内无 cpu sync', 7.8, '#334155',
        'start', maxw=HW - 26, tag='h:l2')
lc.text(HX + 14, HY + 70, 'torch_utils.py:L573-L588 搬运工', 7.4, lc.C_FAINT, 'start',
        maxw=HW - 26, tag='h:f')
lc.seg(UX + UW, UY + UH / 2, HX, HY + HH / 2, lc.C_MUTE, 1.8, 'std')
lc.parrow([(HX + HW, HY + HH / 2), (KX + 60, HY + HH / 2), (KX + 60, KY)],
          lc.C_ENG_S, 2.0, 'gpu')
lc.text(HX + HW + 12, HY + HH / 2 - 12, 'int32 张量上卡', 8, lc.C_ENG_S, 'start',
        maxw=120, tag='h2d:s')

# skip 路径注（CPU 带内底部）
lc.text(MX + 16, CY1 - 46, 'skip 快路径：行数对满（5==5）→ indices=None 直进 xgr，', 8.2,
        lc.C_GPU_S, 'start', True, maxw=700, tag='skip:t')
lc.text(MX + 16, CY1 - 30, 'async_tensor_h2d 实测未被调用——免传的对照就是它', 8.2,
        lc.C_GPU_S, 'start', maxw=700, tag='skip:s')

# ================= 底部：pinned vs pageable 对照条 =================
BY, BH = 470, 128
lc.rect(MX, BY, 700, BH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(MX + 16, BY + 22, 'pinned vs pageable（host 实测中位，[256, 1571] int32 = 1.534MiB）',
        9.6, lc.C_TXT, 'start', True, maxw=660, tag='bp:t')
BAR_X0, BAR_MAXW = MX + 150, 480
for j, (name, us, color) in enumerate([('pinned', 78.1, lc.C_ENG_S), ('pageable', 132.2, lc.C_MUTE)]):
    y = BY + 46 + j * 34
    lc.text(BAR_X0 - 12, y + 12, name, 8.6, lc.C_TXT, 'end', maxw=90, tag='bp:n' + str(j))
    wpx = us / 132.2 * BAR_MAXW
    lc.rect(BAR_X0, y, wpx, 20, '#ffffff', color, rx=4, sw=1.2)
    lc.ELEMS.append(((BAR_X0 + 2, y + 2, BAR_X0 + wpx - 2, y + 18),
                     f'<rect x="{BAR_X0 + 2}" y="{y + 2}" width="{wpx - 4:.1f}" height="16" '
                     f'fill="{color}" opacity="0.25" rx="3"/>'))
    lc.text(BAR_X0 + wpx + 10, y + 13, f'{us}µs', 9, color, 'start', True, maxw=80,
            tag='bp:v' + str(j))
lc.text(MX + 16, BY + BH - 10, '1.69 倍：pinned 锁页让 DMA 直取物理页，免 pageable 的中转拷贝', 8,
        lc.C_MUTE, 'start', maxw=660, tag='bp:f')

# 底右：为什么这条链是 payoff
lc.rect(790, BY, BXR - 790, BH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(806, BY + 22, '为什么说这是本章 payoff', 9.6, lc.C_TXT, 'start', True, maxw=600, tag='py:t')
for j, ln in enumerate(['· 语法 → 位表 → -inf 的最后一跳：CPU 上算好的整张约束表，',
                        '  到这里才真正压到 GPU 的 logits 上',
                        '· 全链异步（non_blocking + async_tensor_h2d）——不引入 cpu sync，',
                        '  不打断 GPU 流水',
                        '· 数值为 host 实测中位、数量级证据（不引为生产毫秒数）']):
    lc.text(806, BY + 44 + j * 16.5, ln, 8.2, '#334155', 'start', maxw=BXR - 806 - 14,
            tag='py:l' + str(j))

# ================= 图例 + 页脚 =================
LY = 622
lx0 = MX
for f_, s_, name in [(lc.C_ENG_F, lc.C_ENG_S, 'CPU · worker 进程（橙）'),
                     (lc.C_GPU_F, lc.C_GPU_S, 'GPU（绿）'),
                     (lc.C_SAM_F, lc.C_SAM_S, 'xgr 应用（品红=采样列落点）')]:
    lc.rect(lx0, LY - 9, 16, 11, f_, s_, rx=3, sw=1.4)
    lc.text(lx0 + 21, LY + 1, name, 8.8, lc.C_TXT, 'start', maxw=220, tag='lg:' + name[:6])
    lx0 += 21 + lc.tw(name, 8.8) + 16
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_ENG_S, 2.2, 'gpu')
lc.text(lx0 + 40, LY + 1, '数据搬运（H2D / 原地写）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg4')

lc.text(MX, 636, 'vllm/v1/structured_output/utils.py:L143-L161（sorted_bitmask .to(non_blocking) → async_tensor_h2d → '
                 'apply_token_bitmask_inplace）· vllm/torch_utils.py:L573-L588（搬运工）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 654, '78.1µs / 132.2µs / 1.534MiB / skip 场景 async_tensor_h2d 未被调用 / out_indices=[5,0,1,2,3] int32 上卡'
                 ' ＝ 本章驱动脚本实测（host 中位，数量级证据）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-h2d-chain.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
