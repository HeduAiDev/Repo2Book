#!/usr/bin/env python3
"""ch28 机制图 · 主干尾部的分叉：先拷贝、再压扁（ch28-fig-mtp-fork）

claim：MTP 长在主干里，是因为它吃的是出口压回单流**之前**的多流残差：主干先把 4 条流的
展平态拷一份交给草稿方，再做出口的门控加权求和把 4 条流压回单流去算 logits——所以这一档
看到的是没被压扁的残差。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出 / pin 源码）：
  顺序：多流残差（4 条流展平）先拷给草稿器，之后才过出口压回单流 → 归一化 → 算 logits
  config 口径：残差 4 条流、多预测一步只开 1 档（与论文 D = 1 一致）
  这一档的入口：上一档隐状态与目标 token 的 embedding 各自归一化后拼接、再过投影

按 spec 的 illustrator_notes：**图上不出现任何代码标识符**（方法名/缓冲名/字段名/源码路径
一律不写），出口那一步只画「门控加权求和」；数值以散文写（4 条流 / 1 档）。
配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂、采样出口品红）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 730
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_SAM_S, C_SAM_F = lc.C_SAM_S, lc.C_SAM_F
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="mg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SAM_S}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '主干尾部的分叉：先拷贝、再压扁', 17, C_TXT, 'start', True, maxw=440,
        tag='title')
lc.text(MX, 56, '多预测一步那一档拿的是出口压回单流「之前」的残差——顺序不能反，'
                '所以它看到的是 4 条流合起来的完整信息', 10, C_MUTE, 'start', maxw=980,
        tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』主干尾部收尾那一格', 9, C_MUTE,
        'end', maxw=450, tag='l0:a')
lc.text(BXR, 64, '——它同时是出口动线的特写：收尾前的那个分叉点', 9, C_MUTE, 'end', maxw=420,
        tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '主干 / 四条残差流'),
                  (C_SAM_F, C_SAM_S, '拷给草稿器的那一支'),
                  (C_SOFT_F, C_SOFT_S, '出口：压回单流并算 logits')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=280, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 主干尾部：多流残差 =================
AX0, AX1, AY0, AY1 = 60, 400, 190, 420
lc.rect(AX0, AY0, AX1 - AX0, AY1 - AY0, C_GPU_F, C_GPU_S, rx=10, sw=1.6)
lc.text(AX0 + 16, AY0 + 28, '主干尾部：多流残差', 12, '#166534', 'start', True, maxw=300,
        tag='a:t')
lc.text(AX0 + 16, AY0 + 50, '4 条流（展平态），每条都带', 9.0, '#334155', 'start', maxw=300,
        tag='a:s1')
lc.text(AX0 + 16, AY0 + 68, '自己的一份信息', 9.0, '#334155', 'start', maxw=300, tag='a:s2')
for i in range(4):
    ly = AY0 + 96 + i * 26
    lc.seg(AX0 + 20, ly, AX1 - 20, ly, C_GPU_S, 1.4)
    lc.text(AX0 + 24, ly - 4, '流 %d' % (i + 1), 8.0, '#166534', 'start', tag='a:l%d' % i)

# ================= 分叉点 =================
FX, FY = 460, 305
lc.parrow([(AX1, FY), (FX - 14, FY)], C_GPU_S, 2.4, marker=None)
for i in range(4):
    ly = AY0 + 96 + i * 26
    lc.parrow([(AX1, ly), (AX1 + 26, ly), (AX1 + 26, FY), (FX - 14, FY)], C_GPU_S, 1.3,
              marker=None)
lc.rect(FX - 14, FY - 22, 28, 44, '#fff7ed', C_ACC_S, rx=6, sw=1.6)
lc.text(FX, FY + 5, '分叉', 9.6, C_ACC_S, 'middle', True, maxw=28, tag='f:t')
lc.text(FX, FY + 46, '只有一个分叉点', 8.6, C_MUTE, 'middle', maxw=140, tag='f:n')

# ================= 上支：拷给草稿器 =================
BX0, BX1, BY0, BY1 = 560, 1010, 156, 250
lc.parrow([(FX, FY - 22), (FX, BY0 + 42), (BX0 - 2, BY0 + 42)], C_SAM_S, 2.2, marker='mg')
lc.rect(BX0, BY0, BX1 - BX0, BY1 - BY0, C_SAM_F, C_SAM_S, rx=10, sw=1.6)
lc.text(BX0 + 16, BY0 + 30, '① 先拷贝：把展平残差交给草稿器', 12, C_SAM_S, 'start', True,
        maxw=430, tag='b:t')
lc.text(BX0 + 16, BY0 + 54, '拷贝发生在压扁之前 —— 多预测一步那一档', 9.2, '#334155', 'start',
        maxw=430, tag='b:s1')
lc.text(BX0 + 16, BY0 + 72, '看到的是 4 条流合起来的完整残差', 9.2, '#334155', 'start',
        maxw=430, tag='b:s2')

DX0 = 1050
lc.parrow([(BX1, BY0 + 42), (DX0 - 2, BY0 + 42)], C_SAM_S, 2.2, marker='mg')
lc.rect(DX0, BY0, BXR - DX0, BY1 - BY0, '#ffffff', C_SAM_S, rx=10, sw=1.5, dash=True)
lc.text(DX0 + 16, BY0 + 30, '一档多预测一步', 12, C_SAM_S, 'start', True, maxw=340, tag='d:t')
lc.text(DX0 + 16, BY0 + 54, '只开 1 档（与论文的 D = 1 一致）；', 9.2, '#334155', 'start',
        maxw=340, tag='d:s1')
lc.text(DX0 + 16, BY0 + 72, '推理期可以整档丢掉，也可以留下来起草', 9.2, '#334155', 'start',
        maxw=340, tag='d:s2')

# ================= 下支：出口 =================
CY0, CY1 = 276, 370
lc.parrow([(FX, FY + 22), (FX, CY0 + 47), (BX0 - 2, CY0 + 47)], C_GPU_S, 2.4, marker='gn')
lc.rect(BX0, CY0, BX1 - BX0, CY1 - CY0, '#ffffff', C_GPU_S, rx=10, sw=1.6)
lc.text(BX0 + 16, CY0 + 30, '② 再过出口：门控加权求和', 12, '#166534', 'start', True,
        maxw=430, tag='c:t')
lc.text(BX0 + 16, CY0 + 54, '把 4 条流压回 1 条（这一件是 V4 自造件，', 9.2, '#334155',
        'start', maxw=430, tag='c:s1')
lc.text(BX0 + 16, CY0 + 72, 'mHC 原论文里没有「怎么收回单流」这一步）', 9.2, '#334155',
        'start', maxw=430, tag='c:s2')

lc.parrow([(BX1, CY0 + 47), (DX0 - 2, CY0 + 47)], C_GPU_S, 3.4, marker='gn')
lc.rect(DX0, CY0, BXR - DX0, CY1 - CY0, C_SOFT_F, C_SOFT_S, rx=10, sw=1.5)
lc.text(DX0 + 16, CY0 + 30, '最后归一化 → 算 logits', 12, C_TXT, 'start', True, maxw=340,
        tag='e:t')
lc.text(DX0 + 16, CY0 + 54, '到这里已经是一条单流了 —— 下游的一切', 9.2, '#334155', 'start',
        maxw=340, tag='e:s1')
lc.text(DX0 + 16, CY0 + 72, '都只看到这一条', 9.2, '#334155', 'start', maxw=340, tag='e:s2')

# 线宽语言
lc.text(FX + 40, 400, '细线 = 四条流；粗线 = 压回后的单流', 8.8, C_MUTE, 'start', maxw=340,
        tag='lw:1')
lc.text(FX + 40, 418, '（用线宽读这张图：越往右下越「扁」）', 8.6, C_MUTE, 'start', maxw=340,
        tag='lw:2')

# ================= 底：三句话 =================
BY2 = 450
lc.rect(MX, BY2, BXR - MX, 138, '#fff7ed', C_ACC_S, rx=10, sw=1.4)
lc.text(MX + 16, BY2 + 28, '读这张图要抓住的三件事', 11.5, C_ACC_S, 'start', True, maxw=340,
        tag='b2:t')
B2 = [('① 顺序', '先拷贝、再压扁（不是先压扁再拷贝）',
       '反过来说，多预测一步那一档就只拿到一条平均过的残差——4 条流的差异全丢了'),
      ('② 为什么长在主干里', '它吃的是主干出口的残差，不是另起一份输入',
       '所以它是主干的一个分支，而不是一个外挂的独立小模型'),
      ('③ 这一档的入口长什么样', '上一档隐状态与目标 token 的 embedding 各自归一化后拼接、再过投影',
       '两个 RMSNorm 各管一半，拼接后再进那一档自己的 Transformer 块')]
for i, (a, b, c) in enumerate(B2):
    ty = BY2 + 60 + i * 30
    lc.text(MX + 16, ty, a, 9.6, C_ACC_S, 'start', True, maxw=180, tag='b2:a%d' % i)
    lc.text(MX + 200, ty, b, 9.6, '#334155', 'start', maxw=560, tag='b2:b%d' % i)
    lc.text(MX + 780, ty, c, 8.8, C_MUTE, 'start', maxw=640, tag='b2:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 624, '数字口径：残差 4 条流（config 口径）、多预测一步只开 1 档（与论文的 D = 1 '
                 '一致）；玩具参数见多预测一步的对齐表那张图。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 642, '依据 DeepSeek-V4 技术报告 §2.1（原样沿用）与 DeepSeek-V3 报告 §2.2'
                 '（arXiv:2606.19348 / arXiv:2412.19437）；收尾顺序照官方参考实现'
                 '（DeepSeek-V4-Pro inference/model.py:L1100-L1110）。', 8.5, C_MUTE,
        'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-mtp-fork.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
