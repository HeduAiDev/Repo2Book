#!/usr/bin/env python3
"""ch29 原理图 · 同一段 1M 历史的三档分辨率（ch29-fig-resolution-ladder）

原理示意图（论文级概念图）：不解释代码、不出现任何 vLLM 类名/文件名/行号/章号/
内部产物名。只回答「同一段历史被三种分辨率看待之后，两笔账各是多少」。

claim（自 figure-request 逐字）：
同一段 1M 历史被三种分辨率看待：滑窗层只留最近 128 条原样条目，其余看不见；CSA 层
每 4 个 token 压成 1 条（1M → 25 万条），每个 query 只看最近窗口加上挑中的 k 个块；
HCA 层每 128 个 token 压成 1 条（1M → 7,812 条），压完全看不再挑。三行并排以后，
两笔账一眼分开——「缓存里有多少条」是显存与带宽账，「每个 query 实看多少条」是算力账；
CSA 与 HCA 是分工（保分辨率 vs 保覆盖），不是省钱版与完整版。

numbers（逐字取自 figure-request，禁即兴新增）：
  1) 压缩比 4 与 128；挑 k：Flash 512、Pro 1024；最近窗口 128 条
  2) 1M 上下文下缓存条数：不压缩 1,000,000；CSA 层 250,000；HCA 层 7,812
     （加 128 个窗口槽为 7,940，约为全量的 0.79%）；滑窗层 128
  3) 每个 query 实看条数：滑窗层 128；CSA 层 128 + k；HCA 层 128 + 它前面的全部
     压缩块（最多 7,812）
provenance：官方 config 一手（compress_ratios / index_topk / sliding_window，
HF deepseek-ai/DeepSeek-V4-Flash 与 DeepSeek-V4-Pro，2026-09-16 抓取）；1M→25 万 /
7,812 为 1,000,000 除以 4 与 128 的算术；论文 §2.3.1/§2.3.2/§2.3.3。

配色走 book/cartography/l0_common.py 的角色常量；三档青沿用本章（与 ch26）三类层的
分档（滑窗最浅 / CSA 中 / HCA 深）。坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 640
MX, BXR = 56, 1604

# ---------------- 语义配色（三档青同本章三类层；窗口 = 原样条目另给条纹画法） ----------------
TIER = {'swa': ('#ecfeff', '#67e8f9', '#0e7490'),
        'csa': ('#a5f3fc', '#0891b2', '#0e7490'),
        'hca': ('#67e8f9', '#155e75', '#155e75')}
C_HI_F, C_HI_S = '#155e75', '#0f172a'      # 挑中的块（压实心）
C_WIN_F, C_WIN_S = '#ecfeff', '#0891b2'    # 最近窗口：原样条目
C_ACC_F, C_ACC_S = '#fff7ed', '#ea580c'    # 两笔账的两列（强调色）

EXTRA_DEFS = ('<defs>'
              '<marker id="ax" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_MUTE}"/></marker>'
              '</defs>')

# ---------------- 布局常量 ----------------
LBX, LBW = MX, 174                 # 行标签列
BAND_X0, BAND_W = 250, 740         # 压缩块带（三行同宽 = 同一段历史）
WIN_X0, WIN_W = 1002, 120          # 最近窗口段（三行都有，位置相同）
AX1 = WIN_X0 + WIN_W               # 时间轴右端 = 1122
CH_X0, CH_W = 1140, 200            # 「缓存里有多少条」列
QY_X0, QY_W = 1350, 254            # 「每个 query 实看多少条」列
N_FINE, N_COARSE = 40, 5           # 两条带的格数（密度示意，条数为实值）
HI = (4, 13, 22, 31)               # CSA 带里高亮的「挑中的块」下标（散布）
LANE_H, LANE_GAP = 96, 8
LANE_Y0 = 156


def laney(i):
    return LANE_Y0 + i * (LANE_H + LANE_GAP)


# ---------------- 标题区 ----------------
lc.text(MX, 34, '同一段 1M 历史，三种分辨率：缓存条数管显存与带宽，实看条数管算力', 16, lc.C_TXT,
        'start', True, maxw=1100, tag='title')
lc.text(MX, 58, '三行并排看：谁把历史压了几倍、谁每个 query 要实看多少条、谁需要挑——两笔账分开算',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')

# ---------------- 共享时间轴（三行共用同一段历史） ----------------
lc.text(MX, 134, '同一段 1M 历史（1,000,000 条原始 KV）：左旧 → 右新', 9.5, '#334155',
        'start', maxw=560, tag='ax:l')
lc.seg(BAND_X0, 116, BAND_X0, 130, lc.C_MUTE, 1.6)
lc.seg(AX1, 116, AX1, 130, lc.C_MUTE, 1.6)
lc.parrow([(BAND_X0, 116), (AX1, 116)], lc.C_MUTE, 1.8, marker='ax')
lc.text(AX1 + 12, 120, '现在', 9.5, lc.C_MUTE, 'start', True, maxw=60, tag='ax:r')

# ---------------- 两列账的表头（上槽 = 显存与带宽账 / 下槽 = 算力账） ----------------
for x0, w, t, sub in [(CH_X0, CH_W, '缓存里有多少条', '＝ 显存与带宽账'),
                      (QY_X0, QY_W, '每个 query 实看多少条', '＝ 算力账')]:
    lc.text(x0 + w / 2, 128, t, 10.5, lc.C_TXT, 'middle', True, maxw=w - 8, tag='hd:' + t[:6])
    lc.text(x0 + w / 2, 146, sub, 9, C_ACC_S, 'middle', True, maxw=w - 8, tag='hd:' + sub[:4])


def chip(x, y, w, h, big, sub, accent=False):
    """数字槽：上槽/下槽同构（accent=强调色描边，用于两笔账的表头呼应）。"""
    lc.rect(x, y, w, h, '#ffffff' if not accent else '#fffbf5', lc.C_MUTE if not accent else C_ACC_S,
            rx=7, sw=1.3)
    lc.text(x + w / 2, y + 24, big, 13, lc.C_TXT, 'middle', True, maxw=w - 16, tag='ch:' + big[:12])
    lc.text(x + w / 2, y + 42, sub, 8.5, lc.C_MUTE, 'middle', maxw=w - 14, tag='ch:' + sub[:12])


def lane_bg(i):
    lc.rect(MX, laney(i), BXR - MX, LANE_H, '#f8fafc' if i % 2 == 0 else '#ffffff',
            '#e2e8f0', rx=8, sw=1.1)


def row_label(i, name, l2, l3, col):
    y = laney(i)
    lc.text(MX + 16, y + 40, name, 12.5, col, 'start', True, maxw=LBW - 16, tag='rl%d:t' % i)
    lc.text(MX + 16, y + 58, l2, 8, '#334155', 'start', maxw=LBW - 16, tag='rl%d:2' % i)
    lc.text(MX + 16, y + 72, l3, 8, lc.C_FAINT, 'start', maxw=LBW - 16, tag='rl%d:3' % i)


def window_block(i, raw_n=8):
    """最近窗口段：原样条目（条纹画法，三行同款同位置）。"""
    y = laney(i) + 22
    lc.rect(WIN_X0, y, WIN_W, 34, C_WIN_F, C_WIN_S, rx=3, sw=1.4)
    step = WIN_W / raw_n
    for k in range(1, raw_n):
        x = WIN_X0 + k * step
        lc.seg(x, y + 4, x, y + 30, C_WIN_S, 1.0)
    lc.text(WIN_X0 + WIN_W / 2, y + 52, '最近 128 条原样', 8.5, C_WIN_S, 'middle', True,
            maxw=WIN_W + 30, tag='wb%d:l' % i)


# ================= 第 1 行：滑窗层 =================
i = 0
lane_bg(i)
row_label(i, '滑窗层', '不压缩：只留最近 128 条', '（compress_ratios 0 / 1）', TIER['swa'][2])
BY = laney(i) + 22
lc.rect(BAND_X0, BY, BAND_W, 34, '#ffffff', lc.C_FAINT, rx=3, sw=1.3, dash=True)
lc.text(BAND_X0 + BAND_W / 2, BY + 22, '看不见：这里没有压缩条目可用', 10, lc.C_FAINT, 'middle',
        maxw=BAND_W - 40, tag='swa:empty')
window_block(i)
chip(CH_X0, laney(i) + 20, CH_W, 58, '128', '窗口槽就这么多')
chip(QY_X0, laney(i) + 20, QY_W, 58, '128', '只看这 128 条原样条目', accent=True)

# ================= 第 2 行：CSA 层（每 4 个 token 一条） =================
i = 1
lane_bg(i)
row_label(i, 'CSA 层', '每 4 个 token 压成 1 条，再挑 k 个块', '（compress_ratios = 4）', TIER['csa'][2])
BY = laney(i) + 22
PITCH_F = BAND_W / N_FINE
for k in range(N_FINE):
    hi = k in HI
    lc.rect(BAND_X0 + k * PITCH_F + 0.8, BY, PITCH_F - 1.6, 34,
            C_HI_F if hi else TIER['csa'][0], C_HI_S if hi else TIER['csa'][1], rx=2, sw=1.1)
lc.text(BAND_X0, laney(i) + 16, '每一小格 = 一个压缩块（4 个 token 合成的 1 条）',
        8.5, '#334155', 'start', maxw=340, tag='csa:den')
LX = BAND_X0 + (HI[2] + 0.5) * PITCH_F
lc.seg(LX, laney(i) + 20, LX, BY, C_HI_S, 1.4, dash=True)
lc.text(BAND_X0 + BAND_W, laney(i) + 16, '深色 = 这个 query 挑中的 k 个块（Flash 512 / Pro 1024）',
        8.5, C_HI_S, 'end', True, maxw=400, tag='csa:hi')
window_block(i)
chip(CH_X0, laney(i) + 20, CH_W, 58, '250,000', '1M ÷ 4：压完还剩这么多条')
chip(QY_X0, laney(i) + 20, QY_W, 58, '128 + k', 'k = Flash 512 / Pro 1024', accent=True)

# ================= 第 3 行：HCA 层（每 128 个 token 一条） =================
i = 2
lane_bg(i)
row_label(i, 'HCA 层', '每 128 个 token 压成 1 条，压完全看', '（compress_ratios = 128）', TIER['hca'][2])
BY = laney(i) + 22
PITCH_C = BAND_W / N_COARSE
for k in range(N_COARSE):
    lc.rect(BAND_X0 + k * PITCH_C + 3, BY, PITCH_C - 6, 34, TIER['hca'][0], TIER['hca'][1],
            rx=3, sw=1.3)
lc.text(BAND_X0 + BAND_W / 2, laney(i) + 16, '带里每一大格 = 一个压缩块（128 个 token 合成的 1 条）· 整条全看，不挑',
        8.5, '#334155', 'middle', maxw=BAND_W - 20, tag='hca:den')
window_block(i)
chip(CH_X0, laney(i) + 20, CH_W, 58, '7,812', '1M ÷ 128；加 128 窗口槽 = 7,940')
chip(QY_X0, laney(i) + 20, QY_W, 58, '128 + 全部压缩块', '最多 128 + 7,812 = 7,940', accent=True)

# ---------------- 点睛：两档是分工不是降级 + 两笔账 ----------------
CY0, CH0 = 484, 74
lc.rect(MX, CY0, BXR - MX, CH0, C_ACC_F, C_ACC_S, rx=9, sw=1.6)
lc.text(MX + 20, CY0 + 30, '两档是分工，不是降级：CSA 保 4 倍分辨率、但要挑着看；HCA 用更粗的分辨率换「全看不挑、不会漏」。',
        11.5, C_ACC_S, 'start', True, maxw=1180, tag='cl:1')
lc.text(MX + 20, CY0 + 54, '两笔账也分开算：缓存里有多少条 = 显存与带宽账；每个 query 实看多少条 = 算力账——同一层里两列各走各的。',
        11, lc.C_TXT, 'start', maxw=1180, tag='cl:2')
lc.text(BXR - 16, CY0 + 30, '压得越多', 9, lc.C_MUTE, 'end', maxw=140, tag='cl:r1')
lc.text(BXR - 16, CY0 + 48, '缓存越省；', 9, lc.C_MUTE, 'end', maxw=140, tag='cl:r2')
lc.text(BXR - 16, CY0 + 66, '但不等于看全', 9, lc.C_MUTE, 'end', maxw=140, tag='cl:r3')

# ---------------- 页脚 ----------------
lc.text(MX, 588, '1M 按 1,000,000 取整算账；块带密度、高亮块数与窗格数均为示意，条数为实值——不压缩对照 1,000,000 条，滑窗层 128 条，HCA 的 7,940 槽约为全量的 0.79%。',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 606, '压缩比 4 / 128 与最近窗口 128 出自官方 config 的逐层表；k 取 Flash 512、Pro 1024。依据 DeepSeek-V4 技术报告 §2.3.1–§2.3.3（arXiv:2606.19348）。',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-resolution-ladder.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
