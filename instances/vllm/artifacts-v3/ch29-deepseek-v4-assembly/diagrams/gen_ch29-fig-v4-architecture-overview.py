#!/usr/bin/env python3
"""ch28 理论图 · DeepSeek-V4 整机全貌（ch28-fig-v4-architecture-overview，模板 layout）

本章开篇「开考：Llama 五件套的更换单」节首的模型级全貌图：**只回答「这台机器长
什么样、部件怎么配合怎么分布」，不解释代码**。纵向层栈 = 一个 token 的路径；右侧
一栏给层型判定、挂件挂法、调和核、出口与 MTP 的顺序，以及「代码分读」小面板。

claim（自 figure-request 逐字）：
DeepSeek-V4 整机架构（模型级全貌）：一个 token 从 embed 进、穿过 L 层主干、到 logits
出——每层 = 一个注意力半层 + 一个 MoE 半层；注意力半层按 compress_ratios 三型排布
（滑窗层只见最近窗口、CSA 层先压 4 倍再稀疏选、HCA 层压 128 倍不选；滑窗副本每层都挂、
打分用的索引器只挂 CSA 层、压缩器挂两类压缩层；压缩层实算是「压缩选块 ∪ 最近窗口」
一次合算）；残差是 hc_mult 条并行流贯穿全栈、每半层一个调和核、走出主干前由 hc_head
压回单流；最末层之后另挂 MTP 草稿头。

numbers 全部取自 figure-request 的 numbers（9 条，逐条带 provenance），图面数字与之
逐字一致，禁即兴新增：
  1) 三型层判定 = compress_ratios 三个合法取值 ≤1 滑窗 / ==4 CSA(C4A) / ==128 HCA(C128A)
     （pin sparse_swa.py:L34-L52 三常量 + 三分支；attention.py:L207-L213 逐层取表）
  2) V4-Flash 表形态 [0, 0, 4, 128, 4, 128, …, 4, 0]——中段省略号照录，**不补全**
     （HF 官方 config 一手 2026-09-12；pin 树内只有取表逻辑、不含该表）
  3) V4-Flash 43 层（表 44 行 = 43 层 + 1 行 MTP）；V4-Pro 61 层；图面按 L 层示意
  4) hc_mult=4（残差并行流条数）；hc_sinkhorn_iters=20 · hc_eps=1e-06（调和核内 Sinkhorn）
  5) 挂法：滑窗副本每层都挂；索引器只挂 compress_ratio==4 的层；压缩器挂 compress_ratio>1
  6) 压缩层两路输入「压缩选块 ∪ 最近窗口」一次核内合算；滑窗层只有窗口一路
  7) 每层 FFN 都是 MoE；最前 num_hash_layers=3 层按 token id 查表派单
  8) 出口顺序：末层先暂存 pre-hc_head 残差、再由 hc_head 压回单流；MTP 草稿头接在
     最末层之后、以这份残差为输入
  9) MTP 层自身不进逐层表：层号 ≥ 层数时恒 compress_ratio=1

配色走 book/cartography/l0_common.py 的角色常量；三档青沿用 ch26 三类层图（滑窗最浅 /
CSA 中 / HCA 深）。坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 1215
MX, BXR = 56, 1604

# ---------------- 语义配色（三档青同 ch26 三类层图；其余取 l0_common 角色色） ----------------
SWA_F, SWA_S = '#ecfeff', '#67e8f9'      # 滑窗层（SWA-only）
CSA_F, CSA_S = '#a5f3fc', '#0891b2'      # CSA 层（=C4A，compress_ratios 4）
HCA_F, HCA_S = '#67e8f9', '#155e75'      # HCA 层（=C128A，compress_ratios 128）
MOE_F, MOE_S = lc.C_API_F, lc.C_API_S     # MoE 前馈半层
HCR_S, HCR_F = lc.C_GPU_S, lc.C_GPU_F     # 残差流带 / 调和核（hc 角色 = GPU 执行臂绿）
MTP_S, MTP_F = lc.C_ENG_S, lc.C_ENG_F     # MTP 草稿头（橙虚线，同 mHC 图旁路款）
OUT_S, OUT_F = lc.C_SAM_S, lc.C_SAM_F     # 出口后段（norm → logits，采样出口色）
KIND_STYLE = {'swa': (SWA_F, SWA_S), 'csa': (CSA_F, CSA_S), 'hca': (HCA_F, HCA_S)}

EXTRA_DEFS = ('<defs>'
              '<marker id="gd" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{HCR_S}"/></marker>'
              '</defs>')

# ---------------- 版面几常（全部由这几个常量派生，无手写魔数） ----------------
CX0, CX1 = 76, 990            # 主干层栈容器左右缘
RX0, RX1 = 1030, 1604         # 右栏左右缘
LBL_X = 90                    # 层号标签 x
BXC = 152                     # 4 条残差流带的中心 x
BDY, BGAP = 8.0, 1.5          # 带内线间距 / (j-1.5) 系数
SQ_X0, SQ_W = 134, 36         # 调和核方块
RET_X = 198                   # 回填通道 x（方块与格位框之间）
BX0, BW = 226, 430            # 半层格位框
CHIP_X, CHIP_W, CHIP_GAP = 674, 90, 10
LANE_X0, LANE_X1 = 84, 982
LANE_H, LANE_GAP = 116, 8
LANES_Y0 = 202
BOX_H = 46
K1_CY, K2_CY = 28, 84         # 两个调和核相对 lane 顶的 y
E0 = 962                      # 出口行顶
BAND_T, BAND_CY, BAND_H = 1058, 1068, 42

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'DeepSeek-V4 整机全貌：token 从 embed 进、穿 L 层主干、到 logits 出——每层 = 注意力半层 + MoE 半层',
        16, lc.C_TXT, 'start', True, maxw=1140, tag='title')
lc.text(MX, 58, '注意力半层按 compress_ratios 逐层定型（滑窗 / CSA / HCA 三型）；残差是 hc_mult 条并行流贯穿全栈、'
                '每半层一个调和核、走出主干前由 hc_head 压回单流；最末层之后另挂 MTP 草稿头',
        10.5, lc.C_MUTE, 'start', maxw=1160, tag='subtitle')
_ch = '开篇理论图 · 模型级全貌（不解释代码）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 入口：token id → embed ----------------
ENT_Y, ENT_H = 88, 46
lc.rect(LANE_X0, ENT_Y, LANE_X1 - LANE_X0, ENT_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.5)
lc.text(LANE_X0 + 16, ENT_Y + 20, '入口：token id → embed', 11.5, lc.C_TXT, 'start', True,
        maxw=430, tag='ent:t')
lc.text(LANE_X0 + 16, ENT_Y + 38, '出单流 2D —— 首层核内展开成 4 条并行残差流（不显式复制）',
        9, '#334155', 'start', maxw=620, tag='ent:l')
lc.text(LANE_X1 - 16, ENT_Y + 20, '主干前：单流', 9, lc.C_MUTE, 'end', tag='ent:r')

# ---------------- 主干层栈容器 ----------------
lanes = [
    dict(lab='第 0 层', kind='swa', tail='首两层不压缩'),
    dict(lab='第 1 层', kind='swa', tail='首两层不压缩'),
    dict(lab='第 2 层', kind='csa', tail='此后 4/128 交替'),
    dict(lab='第 3 层', kind='hca', tail='此后 4/128 交替'),
]
ELL_Y = LANES_Y0 + 4 * (LANE_H + LANE_GAP)               # 第四层之后的中段省略行
ELL_H = 36
LAST_Y = ELL_Y + ELL_H + LANE_GAP
K_TAIL_CY = LAST_Y + 140
CONT_Y0, CONT_Y1 = 168, K_TAIL_CY + 32

lc.rect(CX0, CONT_Y0, CX1 - CX0, CONT_Y1 - CONT_Y0, '#ffffff', lc.C_MUTE, rx=12, sw=1.6, dash=True)
CONT_T = '主干：L 层重复 —— 每层 = 一个注意力半层 + 一个 MoE 半层'
CONT_R = '本图按 L 层示意（不逐格画）：V4-Flash 43 层 · V4-Pro 61 层'
lc.text(CX0 + 14, CONT_Y0 + 20, CONT_T, 10.5, lc.C_TXT, 'start', True, maxw=420, tag='cont:t')
lc.text(CX1 - 14, CONT_Y0 + 20, CONT_R, 8.5, lc.C_MUTE, 'end', maxw=460, tag='cont:r')


def lane_y(i):
    return LANES_Y0 + i * (LANE_H + LANE_GAP)


def attn_lines(kind):
    """注意力格位三行文案（按层型）。"""
    if kind == 'swa':
        return ('注意力半层 · 滑窗（SWA-only，compress_ratios 0/1）',
                '输入一路：最近窗口（没有压缩池，也就没有两路合算）',
                '表首两位都是 0 —— 前两层不压缩')
    if kind == 'csa':
        return ('注意力半层 · CSA（=C4A，compress_ratios 4）',
                '先压 4 倍、再稀疏选 ＋ 最近窗口 → 两路并集一次合算',
                '打分用的索引器挂在本层（只有这一类层挂）')
    return ('注意力半层 · HCA（=C128A，compress_ratios 128）',
            '压 128 倍、不挑选 ＋ 最近窗口 → 两路并集一次合算',
            '本层没有索引器（索引器只挂 CSA 层）')


MOE_L1 = 'MoE 前馈半层（每层都是 MoE）'
MOE_L2 = 'gate 打分选 top-k 专家，另有共享专家常驻'


def draw_lane(y, lab, kind, hash_moe, chips, chip_header=False):
    """一层 = 一条横向 lane：4 条残差流带｜调和核 → 注意力格位 → 调和核 → MoE 格位。"""
    f_, s_ = KIND_STYLE[kind]
    lc.rect(LANE_X0, y, LANE_X1 - LANE_X0, LANE_H, '#f8fafc', '#e2e8f0', rx=8, sw=1.0)
    lc.text(LBL_X, y + LANE_H / 2 + 3, lab, 9, lc.C_MUTE, 'start', bold=True, tag='lab:' + lab)
    # 两个调和核（每半层一个）——白底方块坐在残差流带上
    for cy in (y + K1_CY, y + K2_CY):
        lc.rect(SQ_X0, cy - SQ_W / 2, SQ_W, SQ_W, HCR_F, HCR_S, rx=6, sw=1.6)
    # 注意力格位
    lc.rect(BX0, y + 4, BW, BOX_H, f_, s_, rx=7, sw=1.6)
    l1, l2, l3 = attn_lines(kind)
    lc.text(BX0 + 14, y + 20, l1, 9.5, s_, 'start', True, maxw=BW - 28, tag='a1:' + kind)
    lc.text(BX0 + 14, y + 34, l2, 8.5, '#334155', 'start', maxw=BW - 26, tag='a2:' + kind)
    lc.text(BX0 + 14, y + 46, l3, 8.5, lc.C_MUTE, 'start', maxw=BW - 26, tag='a3:' + kind)
    # MoE 格位
    lc.rect(BX0, y + 62, BW, BOX_H, MOE_F, MOE_S, rx=7, sw=1.6)
    lc.text(BX0 + 14, y + 78, MOE_L1, 9.5, MOE_S, 'start', True, maxw=BW - 28, tag='m1')
    lc.text(BX0 + 14, y + 94, MOE_L2, 8.5, '#334155', 'start', maxw=BW - 26, tag='m2')
    # 挂件小方块（挂法随层型变）
    for j, name in enumerate(chips):
        cx = CHIP_X + j * (CHIP_W + CHIP_GAP)
        lc.rect(cx, y + 18, CHIP_W, 22, '#ffffff', lc.C_FAINT, rx=5, sw=1.3)
        lc.text(cx + CHIP_W / 2, y + 33, name, 8.5, lc.C_MUTE, 'middle', maxw=CHIP_W - 6,
                tag='chip:' + name)
    if chip_header:
        lc.text(LANE_X1 - 2, y + 13, '挂件（挂了什么随层型变）', 8, lc.C_FAINT, 'end',
                maxw=300, tag='chip:hdr')
    # 查表派单小标（最前三层）
    if hash_moe:
        cx = CHIP_X
        lc.rect(cx, y + 74, 200, 22, MTP_F, MTP_S, rx=5, sw=1.3, dash=True)
        lc.text(cx + 100, y + 89, '查表派单（按 token id · 最前三层）', 8.5, '#9a3412', 'middle',
                maxw=192, tag='hash')


# 残差流带（4 条贯穿全栈的平行细带）：先画，后画的方块/框把它压在下面
for j in range(4):
    x = BXC + (j - BGAP) * BDY
    lc.seg(x, LANES_Y0 + K1_CY, x, K_TAIL_CY, HCR_S, 2.0)

for i, d in enumerate(lanes):
    draw_lane(lane_y(i), d['lab'], d['kind'], i < 3,
              ['滑窗缓存'] if d['kind'] == 'swa' else
              (['滑窗缓存', '索引器', '压缩器'] if d['kind'] == 'csa' else ['滑窗缓存', '压缩器']),
              chip_header=(i == 0))

# 中段省略行（层序随 checkpoint 发布，不逐格画）
lc.rect(LANE_X0, ELL_Y, LANE_X1 - LANE_X0, ELL_H, '#f8fafc', '#e2e8f0', rx=8, sw=1.0, dash=True)
lc.text((LANE_X0 + LANE_X1) / 2, ELL_Y + 24,
        '…… 中段省略：4 / 128 交替（示意；逐层排布随 checkpoint 发布，不逐格画）',
        9, lc.C_MUTE, 'middle', maxw=LANE_X1 - LANE_X0 - 60, tag='ell')

# 末层（末位表值 4 → CSA）
draw_lane(LAST_Y, '末层', 'csa', False, ['滑窗缓存', '索引器', '压缩器'])

# 末层收尾核 + 出口
lc.text(186, K_TAIL_CY + 4, '末层收尾：4 条流塌回多流残差', 8.5, lc.C_MUTE, 'start',
        maxw=300, tag='ktail')

# ---------------- 格位内连线（调和核 → 半层；半层产出 → 下一个调和核） ----------------
LANE_YS = [lane_y(i) for i in range(len(lanes))] + [LAST_Y]
# 每半层产出的去处：本层的下一个调和核（注意力→MoE），以及下一层/中段/末层收尾的调和核
NEXT_K = {0: lane_y(1) + K1_CY, 1: lane_y(2) + K1_CY, 2: lane_y(3) + K1_CY,
          3: ELL_Y, 4: K_TAIL_CY}
for i, y in enumerate(LANE_YS):
    for cy in (y + K1_CY, y + K2_CY):
        lc.seg(SQ_X0 + SQ_W, cy, BX0, cy, lc.C_MUTE, 1.8, 'std')          # 调和核喂入半层
    for src_y, tgt in ((y + 41, y + K2_CY), (y + 99, NEXT_K[i])):         # 半层产出回填流带
        if tgt is ELL_Y:                    # 第三层之后回填进「中段省略」行
            lc.parrow([(BX0, src_y), (RET_X, src_y), (RET_X, ELL_Y)], HCR_S, 1.5, 'gd')
        else:
            lc.parrow([(BX0, src_y), (RET_X, src_y), (RET_X, tgt - 12),
                       (SQ_X0 + SQ_W + 2, tgt - 12)], HCR_S, 1.5, 'gd')
# 中段省略行 → 末层首个调和核（层序在中间继续）
lc.parrow([(RET_X, ELL_Y + ELL_H), (RET_X, LAST_Y + K1_CY - 12),
           (SQ_X0 + SQ_W + 2, LAST_Y + K1_CY - 12)], HCR_S, 1.5, 'gd')

# ---------------- 入口 → 首层 / 末层 → 出口 的穿越箭头 ----------------
# 入口竖箭头的列 IN_X 要避开同一 y 带里的三处文字盒：容器标题（start 锚 → 右缘 a）、
# 其右侧的层数旁注（end 锚 → 左缘 c）与下方展开注（start 锚 → 右缘 b）。三处宽度全由
# tw() 实测算出、箭杆落在 max(a, b) 与 c 之间的空档中线，故不压任何字形
# （2026-09-15 盲审 FAIL 项 D：旧箭杆纵贯容器标题「…一个 MoE 半层」的「个」字）。
EXP_T = '首层核内展开 hc_mult=4 条并行流'
EXP_X = BX0 + 114
IN_X = (max(CX0 + 14 + lc.tw(CONT_T, 10.5, True), EXP_X + lc.tw(EXP_T, 9))
        + CX1 - 14 - lc.tw(CONT_R, 8.5)) / 2
lc.seg(IN_X, ENT_Y + ENT_H, IN_X, LANES_Y0 + 4, lc.C_MUTE, 2.0, 'std')
lc.text(EXP_X, ENT_Y + ENT_H + 18, EXP_T, 9, lc.C_MUTE, 'start', maxw=320, tag='exp')
lc.seg(BX0 + 104, CONT_Y1, BX0 + 104, E0, lc.C_MUTE, 2.0, 'std')

# ---------------- 出口行：hc_head 压回单流 → norm/logits；MTP 草稿头挂在最末层之后 ----------------
EX_H = 58
lc.rect(LANE_X0, E0, 316, EX_H, HCR_F, HCR_S, rx=8, sw=1.8)
lc.text(LANE_X0 + 14, E0 + 22, '出口：hc_head 压回单流', 10.5, '#14532d', 'start', True,
        maxw=290, tag='ex:t')
lc.text(LANE_X0 + 14, E0 + 40, '4 条流加权求和 → 单流 (T, H)', 8.5, '#166534', 'start',
        maxw=290, tag='ex:l')
lc.text(LANE_X0 + 14, E0 + 53, '（唯一把多流收成单流的算子）', 8, lc.C_MUTE, 'start',
        maxw=290, tag='ex:l2')

NB_X, NB_W = LANE_X0 + 352, 264
lc.rect(NB_X, E0, NB_W, EX_H, OUT_F, OUT_S, rx=8, sw=1.6)
lc.text(NB_X + 14, E0 + 22, '主干之后：norm → logits', 10.5, OUT_S, 'start', True,
        maxw=NB_W - 28, tag='nb:t')
lc.text(NB_X + 14, E0 + 40, '采样位取 hidden，出词表分布', 8.5, '#9d174d', 'start',
        maxw=NB_W - 26, tag='nb:l')

MT_X, MT_W = NB_X + NB_W + 36, CX1 - (NB_X + NB_W + 36)
lc.rect(MT_X, E0, MT_W, EX_H, MTP_F, MTP_S, rx=8, sw=1.6, dash=True)
lc.text(MT_X + 12, E0 + 20, 'MTP 草稿头（第 L+1 行）', 10, '#9a3412', 'start', True,
        maxw=MT_W - 24, tag='mt:t')
lc.text(MT_X + 12, E0 + 37, '不在逐层表内 · 恒 compress_ratio=1', 8.5, '#9a3412', 'start',
        maxw=MT_W - 22, tag='mt:l')
lc.text(MT_X + 12, E0 + 53, '输入：pre-hc_head 残差', 8.5, '#334155', 'start',
        maxw=MT_W - 22, tag='mt:l2')
lc.seg(MT_X + MT_W - 60, CONT_Y1, MT_X + MT_W - 60, E0, MTP_S, 1.6, 'up', dash=True)
lc.text(MT_X + MT_W - 70, CONT_Y1 + 18, '旁路：先暂存 pre-hc_head 残差', 8.5, MTP_S, 'end',
        maxw=300, tag='mt:lead')

# ---------------- compress_ratios 实例带 ----------------
BAND = [('0', 'swa'), ('0', 'swa'), ('4', 'csa'), ('128', 'hca'), ('4', 'csa'),
        ('128', 'hca'), ('…', 'ell'), ('4', 'csa'), ('0', 'mtp')]
CELL_W, CELL_GAP = 76, 6
BX = LANE_X0
lc.text(LANE_X0, BAND_T, 'compress_ratios 逐层表（V4-Flash 官方 config 一手）：首两位 0、此后 4/128 交替、末位 0 属 MTP 行',
        9.5, lc.C_TXT, 'start', bold=True, maxw=820, tag='band:t')
lc.text(CX1, BAND_T, '中段省略号照录，不补全成具体排布', 9, lc.C_MUTE, 'end', maxw=300,
        tag='band:r')
BAND_LAB = {0: '第 0 层', 1: '第 1 层', 2: '第 2 层', 3: '第 3 层', 6: '中段（4/128 交替）',
            7: '末层', 8: 'MTP 行'}
for k, (val, kind) in enumerate(BAND):
    x = BX + k * (CELL_W + CELL_GAP)
    if kind == 'ell':
        lc.rect(x, BAND_CY, CELL_W, BAND_H, '#f8fafc', lc.C_FAINT, rx=6, sw=1.4, dash=True)
        lc.text(x + CELL_W / 2, BAND_CY + 28, '…', 14, lc.C_MUTE, 'middle', tag='band:ell')
    elif kind == 'mtp':
        lc.rect(x, BAND_CY, CELL_W, BAND_H, SWA_F, MTP_S, rx=6, sw=1.6, dash=True)
        lc.text(x + CELL_W / 2, BAND_CY + 28, val, 13, '#9a3412', 'middle', True, tag='band:v%d' % k)
    else:
        f_, s_ = KIND_STYLE[kind]
        lc.rect(x, BAND_CY, CELL_W, BAND_H, f_, s_, rx=6, sw=1.6)
        lc.text(x + CELL_W / 2, BAND_CY + 28, val, 13, s_, 'middle', True, tag='band:v%d' % k)
    if k in BAND_LAB:
        lc.text(x + CELL_W / 2, BAND_CY + BAND_H + 16, BAND_LAB[k], 8.5, lc.C_MUTE, 'middle',
                maxw=138, tag='band:lab%d' % k)

# ---------------- 右栏 ----------------
ry = 88


def note_box(y, w, title, lines, tags, tcol=None):
    """右栏注框：标题 + 若干行，返回框高。"""
    h = 30 + len(lines) * 18 + 10
    lc.rect(RX0, y, w, h, '#ffffff', lc.C_FAINT, rx=9, sw=1.2, dash=True)
    lc.text(RX0 + 14, y + 22, title, 10.5, tcol or lc.C_TXT, 'start', True, maxw=w - 28,
            tag='nt:' + title[:10])
    for i, s in enumerate(lines):
        lc.text(RX0 + 14, y + 42 + i * 18, s, 8.5, '#334155', 'start', maxw=w - 28,
                tag='nl:%s%d' % (tags, i))
    return h


RW = RX1 - RX0
# 图例（8 行）
LEG_ROWS = [
    (SWA_F, SWA_S, False, '滑窗层（SWA-only） —— compress_ratios ≤1（表中 0）'),
    (CSA_F, CSA_S, False, 'CSA 层（vLLM 记法 C4A） —— compress_ratios 4'),
    (HCA_F, HCA_S, False, 'HCA 层（vLLM 记法 C128A） —— compress_ratios 128'),
    (MOE_F, MOE_S, False, 'MoE 前馈半层 —— 每层一个'),
    (MTP_F, MTP_S, True, '查表派单 —— 最前三层 MoE 选人不看 gate 打分（gate 分数仍用于加权），按 token id 查表选人'),
    (HCR_F, HCR_S, False, 'hc_mult=4 条残差流 —— 贯穿全栈'),
    (HCR_F, HCR_S, False, '调和核 —— 每半层一个（白底方块坐在流带上）'),
    ('#ffffff', lc.C_FAINT, False, '挂件 —— 滑窗缓存 / 索引器 / 压缩器'),
    (MTP_F, MTP_S, True, 'MTP 草稿头 —— 橙虚线，不在逐层表内'),
]
LEG_H = 30 + len(LEG_ROWS) * 22 + 8
lc.rect(RX0, ry, RW, LEG_H, '#ffffff', lc.C_FAINT, rx=9, sw=1.2, dash=True)
lc.text(RX0 + 14, ry + 22, '图例', 10.5, lc.C_TXT, 'start', True, maxw=200, tag='leg:t')
for i, (f_, s_, dash, s) in enumerate(LEG_ROWS):
    yy = ry + 34 + i * 22
    lc.rect(RX0 + 14, yy + 3, 18, 14, f_, s_, rx=3, sw=1.4, dash=dash)
    lc.text(RX0 + 40, yy + 14, s, 8.5, lc.C_TXT, 'start', maxw=RW - 56, tag='leg:r%d' % i)
ry += LEG_H + 14

ry += note_box(ry, RW, '层型与层数', [
    '层型 = compress_ratios 的三个合法取值：≤1 滑窗 · 4 CSA · 128 HCA',
    '其它取值直接报错（expected 1, 4, or 128）；逐层表随 checkpoint 发布、不设默认值',
    'V4-Flash 43 层（表 44 行 = 43 层 + 1 行 MTP）；V4-Pro 61 层',
], 'kind') + 14

ry += note_box(ry, RW, '每个注意力半层挂什么', [
    '滑窗副本（滑窗缓存）—— 每层都挂',
    '打分用的索引器 —— 只挂 CSA 层',
    '压缩器 —— 挂两类压缩层（CSA 与 HCA）',
], 'chip') + 14

ry += note_box(ry, RW, '调和核（每个半层一个）', [
    '收编上一半层的产出（post）＋ 配出下一半层的输入（pre）',
    '流间搬运矩阵经 Sinkhorn 归一到双随机形态（行列和都是 1）',
    'hc_sinkhorn_iters=20 · hc_eps=1e-06',
], 'hc') + 14

ry += note_box(ry, RW, '出口与 MTP 的顺序', [
    '末层先暂存 pre-hc_head 残差、再由 hc_head 压回单流，顺序刻意',
    'MTP 草稿头接在最末层之后，以这份残差为输入',
    'MTP 层自身不进逐层表：层号 ≥ 层数时恒 compress_ratio=1',
], 'out') + 14

# 代码分读小面板（本图各部分 ↔ 本章各幕）
PANEL = [
    '三色注意力格位：层型判定与挂件挂法 → 装配幕三 · 第 6 站',
    '残差带与每半层一个调和核 → 运行幕一 · 第 10 站',
    '注意力半层的实算（压缩选块 ∪ 最近窗口）→ 运行幕二 · 第 11 站',
    'MoE 半层与查表派单 → 装配幕四 · 第 7-8 站、运行幕三 · 第 12 站',
    '出口与 MTP 草稿头 → 运行幕四 · 第 13-14 站',
    '（图外补充）实现分栋：registry 查表与平台三岔门 → 装配幕一 · 第 1-2 站',
]
PH = 30 + len(PANEL) * 19 + 12
lc.rect(RX0, ry, RW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.3, dash=True)
lc.text(RX0 + 14, ry + 22, '代码分读：本图各部分对应本章哪一幕', 10.5, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='pn:t')
for i, s in enumerate(PANEL):
    lc.text(RX0 + 14, ry + 44 + i * 19, s, 8.5, '#334155', 'start', maxw=RW - 28,
            tag='pn:r%d' % i)

# ---------------- 页脚 ----------------
FY = BAND_CY + BAND_H + 46
lc.text(MX, FY, '读法：自上而下 —— token 进 embed、穿 L 层主干（每层注意力半层 + MoE 半层），'
                '末层由 hc_head 压回单流后出 logits；4 条残差流全程并行、每个半层被一个调和核调和；MTP 草稿头挂在最末层之后。',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:read')
lc.text(MX, FY + 20, '层型三分支与报错串 = pin 源码 vLLM v0.27.1（vllm/v1/attention/backends/mla/sparse_swa.py:L44-L53）· '
                     '逐层取表与挂件挂法 = vllm/models/deepseek_v4/attention.py · 逐层表形态与层数 = 官方 config 一手 · '
                     '模型级理论图：算子与实现细节见本章各幕代码', 8.5, lc.C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-v4-architecture-overview.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
