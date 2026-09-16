#!/usr/bin/env python3
"""ch28 机制图 · 滑窗支路：n_win 条未压缩 KV 与压缩条目的合成（ch28-fig-sliding-window）

claim：滑窗把 n_win 条**未压缩** KV 与压缩条目并进同一次注意力：窗口长度是常数
（min(pos+1, n_win)），所以它是加账项——长上下文里被摊薄、短上下文里最贵；
而且未满一块的新 token 只有它看得见。

段序口径（rev4，2026-09-16 评审修复）：KV 轴 = [最近 n_win 条未压缩 KV] + [压缩条目]
——滑窗段在左、压缩条目段在右。四处同序：本图四行轴条与合成卡五格、trace
run_m07_m09_attention.json B 段的 kv_axis 行序（前 3 行滑窗、后 2 行压缩条目）、同章
整机图 ch28-fig-machine-map 左下的 KV 轴小条、正文「同一次注意力 KV 轴左侧那段恒长
的部分」与伪码 compose_kv 的 concat([swa, …])。权重数组按轴序读，不重排。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  n_win=3：pos=2 → 滑窗 3 条 + 候选 0 条 = KV 轴 3 条；pos=8 → 3 + 2 = 5 条
  n_win=128：pos=127 → 滑窗 128 条 + 候选 32 条 = KV 轴 160 条；pos=1000 → 128 + 250 = 378 条
  1M：CSA 层 128 + 512 = 640 条；HCA 层 128 + 7812 = 7940 条
  合成权 [0.230606, 0.179596, 0.179596, 0.230606, 0.179596] 和 = [1.0]；
  o = [0.55101, 0.589798, 0.564293, 0.179596]
  滑窗是加账：43 × 128 × 576 = 3170304 B 常数窗（1M 摊薄 3.1703 B/token）

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩条目，浅青 = 滑窗）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_SWA_F, C_SWA_S = '#cffafe', '#0e7490'

EXTRA_DEFS = ('<defs>'
              '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '滑窗：恒长 128 条原样 KV，排在压缩条目段前面', 17, C_TXT, 'start', True,
        maxw=680, tag='title')
lc.text(MX, 56, '窗口长度 = min(pos + 1, n_win)：与压缩率无关、与上下文长度无关——'
                '长上下文里被摊薄，短上下文里它反过来最贵', 10, C_MUTE, 'start', maxw=1050,
        tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』KV 轴的组成', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——左边恒是最近 128 条未压缩 KV，右边是压缩条目段', 9, C_MUTE, 'end', maxw=430,
        tag='l0:b')

# ---------------- 图例（顺序即轴序：滑窗段在左、压缩条目段在右） ----------------
lgx = MX
for f, s, lab in ((C_SWA_F, C_SWA_S, '滑窗段：最近 n_win 条未压缩 KV'),
                  (C_KV_F, C_KV_S, '压缩条目（软池化产物）'),
                  ('#ffffff', '#cbd5e1', '未使用 / 候选墙')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ================= ① 四个时间点的 KV 轴构成（行内比例真实） =================
BX0, BXRW = 300, 940                 # 条形区
TOT = [('pos = 127', 128, 32, '滑窗 128 + 候选 32', '起手不足一窗就从头开始'),
       ('pos = 1000', 128, 250, '滑窗 128 + 候选 250', 'm = 4：候选墙还只有 250 条'),
       ('1M · CSA 层', 128, 512, '滑窗 128 + 压缩 512', 'top-k 把压缩条目封在 512'),
       ('1M · HCA 层', 128, 7812, '滑窗 128 + 压缩 7812', '一条不挑：滑窗只剩 1.6% 的宽度')]
lc.text(BX0, 128, '一次注意力的 KV 轴 = 滑窗段 + 压缩条目段（每一行按该行真实比例画）', 10.5,
        C_TXT, 'start', True, maxw=900, tag='ax:t')
for i, (lab, nswa, ncomp, mid, note) in enumerate(TOT):
    cy = 150 + i * 46
    tot = nswa + ncomp
    wswa = (nswa / tot) * BXRW          # 滑窗段在左
    wcomp = BXRW - wswa                 # 压缩条目段在右
    lc.text(BX0 - 24, cy + 22, lab, 10, C_TXT, 'end', True, tag='ax:l%d' % i)
    lc.rect(BX0, cy, wswa, 34, C_SWA_F, C_SWA_S, rx=4, sw=1.4)
    lc.rect(BX0 + wswa, cy, wcomp, 34, C_KV_F, C_KV_S, rx=4, sw=1.4)
    if wswa >= 70:
        lc.text(BX0 + wswa / 2, cy + 22, '滑窗 %d' % nswa, 10, '#0e7490', 'middle', True,
                maxw=wswa - 12, tag='ax:s%d' % i)
    else:
        # 滑窗段压到十几像素（1M·HCA 层）：标签贴到它右侧，箭头指回那一段
        lc.text(BX0 + wswa + 8, cy + 22, '← 滑窗 %d' % nswa, 9.5, C_KV_DEEP, 'start', maxw=140,
                tag='ax:s%d' % i)
    if wcomp > 90:
        lc.text(BX0 + wswa + wcomp / 2, cy + 22, '压缩 %d' % ncomp, 10, C_KV_DEEP, 'middle', True,
                maxw=max(wcomp - 12, 60), tag='ax:c%d' % i)
    else:
        lc.text(BX0 + wswa + wcomp - 6, cy + 22, '压缩 %d' % ncomp, 9.5, C_KV_DEEP, 'end',
                maxw=140, tag='ax:c%d' % i)
    lc.text(BX0 + BXRW + 12, cy + 12, mid, 9.2, '#334155', 'start', True, maxw=200,
            tag='ax:m%d' % i)
    lc.text(BX0 + BXRW + 12, cy + 28, note, 8.5, C_MUTE, 'start', maxw=200, tag='ax:n%d' % i)

# ================= ② 玩具口径 + 合成真跑 =================
PY0 = 352
lc.rect(MX, PY0, 726 - MX, 172, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, PY0 + 24, '玩具口径（n_win = 3、m = 4）：滑窗长度与 m 无关', 11, C_TXT, 'start',
        True, maxw=460, tag='toy:t')
TOY = [('pos = 2', '滑窗 [0, 2] 共 3 条', '候选 (2+1)//4 = 0 条', 'KV 轴 = 3 + 0 = 3 条'),
       ('pos = 8', '滑窗 [6, 8] 共 3 条', '候选 (8+1)//4 = 2 条', 'KV 轴 = 3 + 2 = 5 条')]
for i, (a, b, c, d) in enumerate(TOY):
    ty = PY0 + 56 + i * 24
    lc.text(MX + 16, ty, a, 9.2, C_TXT, 'start', True, tag='toy:a%d' % i)
    lc.text(MX + 100, ty, b, 9.2, '#0e7490', 'start', tag='toy:b%d' % i)
    lc.text(MX + 290, ty, c, 9.2, C_MUTE, 'start', tag='toy:c%d' % i)
    lc.text(MX + 480, ty, d, 9.2, C_KV_DEEP, 'start', True, maxw=200, tag='toy:d%d' % i)
lc.text(MX + 16, PY0 + 116, '要点：pos = 2 时压缩条目是 0 条——还没攒满一块，'
                            '这一拍只有滑窗看得见那 3 个 token。', 9, '#155e75', 'start', False,
        maxw=640, tag='toy:n0')
lc.text(MX + 16, PY0 + 136, '『攒批时间差』：最新的一些 token 在压缩侧还不存在，'
                            '必须由原样窗兜住。', 9, C_MUTE, 'start', maxw=640, tag='toy:n1')
lc.text(MX + 16, PY0 + 158, '滑窗起点 = max(pos − n_win + 1, 0)；条数 = pos − 起点 + 1。', 8.8,
        lc.C_FAINT, 'start', maxw=640, tag='toy:n2')

# 合成真跑
QX0 = 774
lc.rect(QX0, PY0, BXR - QX0, 172, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(QX0 + 16, PY0 + 24, '合成后真跑一次：3 条滑窗 + 2 条压缩条目', 11, C_TXT, 'start', True,
        maxw=460, tag='cmp:t')
lc.text(QX0 + 16, PY0 + 48, '（pos = 8、m = 4、n_win = 3 → KV 轴 5 条：两路在同一份权重里竞争，'
                            '不分区；五格自左至右就是 KV 轴行序）',
        8.8, C_MUTE, 'start', maxw=620, tag='cmp:s')
WK = ['0.230606', '0.179596', '0.179596', '0.230606', '0.179596']
# 每格的来源：五格自左至右 = KV 轴行序（滑窗段在前、压缩条目段在后，与同图四行轴条、
# trace 的 kv_axis 行序同一口径）——前 3 格滑窗、后 2 格压缩；权重数组即轴序权重，不重排。
# 3 条滑窗 + 2 条压缩 = 5 条——与卡标题、左卡「3 + 2 = 5」、卡尾「多出来的 3 行」三处一致。
for i in range(5):
    cxx = QX0 + 30 + i * 106
    is_comp = i >= 3
    f, s = (C_KV_F, C_KV_S) if is_comp else (C_SWA_F, C_SWA_S)
    lc.rect(cxx, PY0 + 66, 96, 46, f, s, rx=4, sw=1.3)
    lc.text(cxx + 48, PY0 + 84, '压缩' if is_comp else '滑窗', 8.6, '#334155', 'middle',
            tag='cmp:k%d' % i)
    lc.text(cxx + 48, PY0 + 102, WK[i], 9.2, C_KV_DEEP if is_comp else '#0e7490', 'middle', True,
            tag='cmp:w%d' % i)
lc.text(QX0 + 16, PY0 + 132, '权重和 = [1.0]（两路进同一次 softmax）', 9.2, C_KV_DEEP, 'start',
        True, maxw=300, tag='cmp:l0')
lc.text(QX0 + 330, PY0 + 132, 'o = [0.55101, 0.589798, 0.564293, 0.179596]', 9.2, '#334155',
        'start', maxw=340, tag='cmp:l1')
lc.text(QX0 + 16, PY0 + 154, '谁分到的多由 q · kv 决定——滑窗不是外挂的第二套注意力，'
                             '而是 KV 轴上多出来的 3 行。', 8.8, C_MUTE, 'start', maxw=620,
        tag='cmp:l2')

# ================= ③ 加账带 =================
BY0, BY1 = 548, 700
lc.rect(MX, BY0, BXR - MX, BY1 - BY0, C_SWA_F, C_SWA_S, rx=10, sw=1.6)
lc.text(MX + 18, BY0 + 26, '滑窗是七件套里唯一只花不省的一件：43 层 × 128 条 × 576 B = 3170304 B 的常数窗',
        13, '#0e7490', 'start', True, maxw=760, tag='add:t')
ADD = [('1M 上下文', '3170304 / 1000000 = 3.1703 B/token', '只占压缩账的 0.035406'),
       ('1K 上下文', '3170304 / 1000 = 3170.304 B/token', '同一常数窗按 1/context 放大 1000 倍'),
       ('与 m 的关系', '条数 = min(pos + 1, 128)', '与压缩率 m 完全无关；只有层数能让它线性增长')]
for i, (a, b, c) in enumerate(ADD):
    ty = BY0 + 62 + i * 28
    lc.text(MX + 18, ty, a, 9.8, C_TXT, 'start', True, tag='add:a%d' % i)
    lc.text(MX + 120, ty, b, 9.8, '#0e7490', 'start', True, tag='add:b%d' % i)
    lc.text(MX + 500, ty, c, 8.8, '#334155', 'start', maxw=440, tag='add:c%d' % i)
lc.text(MX + 18, BY1 - 16, '注意上表第四条：1M · HCA 层里压缩条目占 7812/7940 = 0.983879，'
                           '滑窗 128 条看似微不足道——但短上下文里同样的 128 条会吃掉绝大部分。',
        8.8, '#155e75', 'start', maxw=BXR - MX - 40, tag='add:n')

# ================= ④ 底部三条读数 =================
CY0 = 720
lc.rect(MX, CY0, BXR - MX, 96, '#ffffff', C_KV_S, rx=10, sw=1.4)
CY = [('三条读数', 'pos = 127 → 128 + 32 = 160｜pos = 1000 → 128 + 250 = 378',
       '起手不足一窗就从头开始；常数窗从 pos = 127 起就一直贴着 128'),
      ('1M 的两层', 'CSA 层：128 + 512 = 640 条｜HCA 层：128 + 7812 = 7940 条',
       '压缩条目占 0.800000 / 0.983879——滑窗段被摊到最薄'),
      ('它兜住的那部分', '还没攒满一块的最新 token 没有压缩条目，只有原样窗看得见',
       '这就是『一核两源』在公式层的对应物')]
for i, (a, b, c) in enumerate(CY):
    ty = CY0 + 24 + i * 26
    lc.text(MX + 16, ty, a, 9.5, C_TXT, 'start', True, tag='cy:a%d' % i)
    lc.text(MX + 150, ty, b, 9.5, '#334155', 'start', maxw=560, tag='cy:b%d' % i)
    lc.text(MX + 760, ty, c, 8.8, C_MUTE, 'start', maxw=620, tag='cy:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 838, '数字口径：滑窗条数 = min(pos + 1, n_win)；压缩条目数 = min(top-k 512, 候选数)，'
                 '候选 = 上下文长度 // 压缩率。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 856, '依据 DeepSeek-V4 技术报告 §2.3.3 的 n_win 句与 §2.3.1（arXiv:2606.19348）；'
                 '两路合成后的权重与输出由本章驱动脚本实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-sliding-window.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
