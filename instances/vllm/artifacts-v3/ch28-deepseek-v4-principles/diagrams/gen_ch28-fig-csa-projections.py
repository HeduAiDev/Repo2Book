#!/usr/bin/env python3
"""ch28 机制图 · CSA 压缩入口：一条 hidden 变出四组投影（ch28-fig-csa-projections）

claim：一条 hidden 变出四组投影：两组是**要被混合**的向量（C^a / C^b）、两组是**决定怎么
混合**的分数原料（Z^a / Z^b）——a 半区取当前窗、b 半区取前一个窗，四组缺一不可。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出 / pin 源码）：
  H 形状 (16, 3) → C^a/C^b/Z^a/Z^b 各 (16, 2)（四组 W 的形状都是 d×c = 3×2）
  token 4 的四个投影值：C^a = [0.5, 1.5]、C^b = [1.0, 0.5]、Z^a = [0.5, 1.0]、Z^b = [1.5, 0.5]
  第 i = 1 条条目取行：C^a/Z^a 取当前窗 [4, 5, 6, 7]、C^b/Z^b 取前一个窗 [0, 1, 2, 3]
  实现侧四组 W 压成一条 GEMM：coff · head_dim × 2 组（coff = 1 + overlap = 2）

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩缓存侧）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 760
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'

EXTRA_DEFS = ('<defs>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '压缩的入口：一条 hidden 变出四组投影', 17, C_TXT, 'start', True, maxw=560,
        tag='title')
lc.text(MX, 56, '两组是「要被混合」的向量（C 组）、两组是「决定怎么混合」的分数原料'
                '（Z 组）——a 半区对齐当前窗、b 半区对齐前一个窗，四组缺一不可',
        10, C_MUTE, 'start', maxw=1120, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』压缩段的入口一格', 9, C_MUTE,
        'end', maxw=430, tag='l0:a')
lc.text(BXR, 64, '——上游是这一层的隐状态，下游是那张讲 2m 怎么归一的图', 9, C_MUTE, 'end',
        maxw=460, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, dash, lab in ((C_KV_F, C_KV_S, False, '要被混合的向量 C^a / C^b'),
                        ('#ffffff', C_KV_S, True, '分数原料 Z^a / Z^b（还不是权重）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1, dash=dash)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 左：hidden 竖条 =================
HBX, HBY, HBW, HBH = 70, 140, 130, 374
lc.rect(HBX, HBY, HBW, HBH, C_GPU_F, C_GPU_S, rx=8, sw=1.6)
HBCX = HBX + HBW / 2
lc.text(HBCX, 272, 'H', 22, C_GPU_S, 'middle', True, tag='h:1')
lc.text(HBCX, 300, '形状 (16, 3)', 10.5, C_TXT, 'middle', maxw=120, tag='h:2')
lc.text(HBCX, 322, 'n 个 token', 9, C_MUTE, 'middle', maxw=120, tag='h:3')
lc.text(HBCX, 340, '× d 维', 9, C_MUTE, 'middle', maxw=120, tag='h:4')
lc.text(HBCX, 380, '一条 entry', 9, C_MUTE, 'middle', maxw=120, tag='h:5')
lc.text(HBCX, 398, '（第 4 行是 token 4）', 8.4, C_MUTE, 'middle', maxw=120, tag='h:6')

# ================= 四组投影块 =================
PBX, PBW, PBH = 290, 330, 82
ROWS = [('C^a = H · W^aKV', '形状 (16, 2)：d × c = 3 × 2', '要被混合的向量（当前窗）', C_KV_S, C_KV_F, False),
        ('Z^a = H · W^aZ', '形状 (16, 2)：d × c = 3 × 2', '分数原料——还要过 softmax', C_KV_S, '#ffffff', True),
        ('C^b = H · W^bKV', '形状 (16, 2)：d × c = 3 × 2', '要被混合的向量（前一个窗）', C_KV_S, C_KV_F, False),
        ('Z^b = H · W^bZ', '形状 (16, 2)：d × c = 3 × 2', '分数原料——还要过 softmax', C_KV_S, '#ffffff', True)]
YS = [140, 230, 342, 432]
for (t1, t2, t3, stroke, fill, dash), py in zip(ROWS, YS):
    lc.rect(PBX, py, PBW, PBH, fill, stroke, rx=8, sw=1.5, dash=dash)
    lc.text(PBX + 16, py + 26, t1, 12, C_KV_DEEP, 'start', True, maxw=300, tag='p:t%s' % t1[:4])
    lc.text(PBX + 16, py + 50, t2, 9.2, '#334155', 'start', maxw=300, tag='p:s%s' % t1[:4])
    lc.text(PBX + 16, py + 70, t3, 8.8, C_MUTE, 'start', maxw=300, tag='p:n%s' % t1[:4])
    yc = py + PBH / 2
    lc.parrow([(HBX + HBW, yc), (PBX - 2, yc)], C_KV_S, 1.6, marker='kv')

# ================= 两半区括线 =================
BRX, BRL = 660, 670
for (top, bot, lab, sub) in ((140, 312, 'a 半区', '当前窗'), (342, 514, 'b 半区', '前一个窗')):
    lc.parrow([(BRX, top), (BRL, top), (BRL, (top + bot) / 2), (BRL, bot), (BRX, bot)],
              C_KV_S, 1.8, marker=None)
    lc.parrow([(BRL, (top + bot) / 2), (706, (top + bot) / 2)], C_KV_S, 2.0, marker='kv')
    lc.text(BRL + 10, top + 14, lab, 10.5, C_KV_DEEP, 'start', True, tag='br:%s' % lab)

# ================= 右：两个半区的说明卡 =================
for (py, title, rows) in (
        (140, 'a 半区 = 当前窗',
         ['第 i = 1 条条目取行 [4, 5, 6, 7]',
          'C^a、Z^a 都从这 4 行取——两组投影',
          '对齐的是同一个窗',
          'C^a = H[4:8] · W^aKV、Z^a = H[4:8] · W^aZ']),
        (342, 'b 半区 = 前一个窗',
         ['第 i = 1 条条目取行 [0, 1, 2, 3]',
          '正是第 i = 0 条用的那 4 行——两个窗',
          '在这 4 行上叠了起来',
          'C^b = H[0:4] · W^bKV、Z^b = H[0:4] · W^bZ'])):
    lc.rect(714, py, BXR - 714, 172, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
    lc.text(730, py + 30, title, 12.5, C_KV_DEEP, 'start', True, maxw=200, tag='cd:t')
    for i, s in enumerate(rows):
        lc.text(730, py + 58 + i * 28, s, 9.6 if i < 3 else 9.0,
                '#334155' if i < 3 else C_MUTE, 'start', maxw=700, tag='cd:%s%d' % (title[:2], i))

# ================= 底：token 4 的四个投影值 =================
BY0 = 534
lc.rect(MX, BY0, BXR - MX, 106, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 26, 'token 4（第 2 个窗的第 1 个格子）的四个投影值', 11.5, C_KV_DEEP,
        'start', True, maxw=420, tag='tv:t')
VALS = [('C^a', '[0.5, 1.5]'), ('C^b', '[1.0, 0.5]'), ('Z^a', '[0.5, 1.0]'), ('Z^b', '[1.5, 0.5]')]
CW = 340
for i, (k, v) in enumerate(VALS):
    cx = MX + 16 + i * (CW + 6)
    lc.rect(cx, BY0 + 38, CW, 52, C_KV_F if k[0] == 'C' else '#ffffff', C_KV_S, rx=6, sw=1.2,
            dash=k[0] == 'Z')
    lc.text(cx + 14, BY0 + 60, k, 11, C_KV_DEEP, 'start', True, tag='tv:k%d' % i)
    lc.text(cx + 70, BY0 + 60, v, 11, '#334155', 'start', True, maxw=250, tag='tv:v%d' % i)
    lc.text(cx + 14, BY0 + 80, '被混合' if k[0] == 'C' else '当分数原料', 8.6, C_MUTE, 'start',
            maxw=250, tag='tv:n%d' % i)

# ---------------- 底部注 ----------------
lc.text(MX, 672, '四组缺一不可：少一组就没有『窗』可叠——C 组决定拿什么去混、Z 组决定怎么混。',
        9.2, '#155e75', 'start', True, maxw=BXR - MX, tag='n:1')
lc.text(MX, 692, '实现侧把四组权重压成一条 GEMM 一枪出（2 倍头维 × 2 组：coff = 1 + overlap = 2）；'
                 '那只是打包，读式子时按四组看。', 9.0, C_MUTE, 'start', maxw=BXR - MX, tag='n:2')

# ---------------- 页脚 ----------------
lc.text(MX, 722, '数字口径：玩具 d = 3、c = 2、n = 16（含第二拍的 4 个 token）；'
                 '全部数值由本章算例实跑（第一、二拍合计与一次性 16 个 token 同结果）。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 740, '依据 DeepSeek-V4 技术报告 §2.3.1 的 Eq.(9)(10)（arXiv:2606.19348）；'
                 '『一条 GEMM 一枪出』照 vLLM v0.27.1 的压缩实现'
                 '（vllm/models/deepseek_v4/compressor.py:L284-L300）。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-csa-projections.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
