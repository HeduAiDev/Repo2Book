#!/usr/bin/env python3
"""ch28 理论图 · 一台 V4 的方框图（ch28-fig-machine-map，模板 layout + tensor-flow 混合）

本章开篇「一台机器的形状：七件套各站在哪」节首的整机理论图（公式口径）：**只回答
「这台机器长什么样、七件套各站在哪」，不讲代码**。中央一条主干竖排（三段代表 43 层：
前两层牌子 0、中段 4 与 128 交替）、主干左侧挂残差带（4 条流）、左侧三块挂件说明
（残差带 / 压缩缓存与选择器 / 滑窗带）、右侧挂专家池；左下角是「一次注意力看到的
KV 轴」小条（滑窗段 + 压缩段，同一个比例尺）；右侧是主干尾部的分叉（先拷一份给
多预测一步、再由 hc_head 压回单流）。

claim（自 figure-request 逐字）：
一台 V4 的整机理论图（公式口径）：一个 token 从残差带的读入进、过注意力半层与 MoE
半层再写回，主干尾部把多流残差先分叉一份给多预测一步、再压回单流算 logits——七件套
在这台机器上各站一个位置。

numbers（逐字取自 figure-request 的 numbers，全部带 provenance；图面数字与之逐字一致，
禁即兴新增任何数字）：
  1) 43 层主干，压缩率逐层交替：21 层为 4、20 层为 128、前两层为 0（纯滑窗）
     ← traces/run_m01_m02_ledger.json（『[层型分类]』swaonly 2 / c4a 21 / c128a 20）
  2) 滑窗带恒长 128 条未压缩 KV；选择器挑 512 条（top-k 封顶）
     ← traces/run_m01_m02_ledger.json（『[FLOPs 账·三档并置]』）
  3) 一次注意力的 KV 轴：CSA 层 128 滑窗加 512 压缩条目 = 640 条；HCA 层 128 加 7812
     = 7940 条 ← traces/run_m07_m09_attention.json（B 段『③④』两行）
  4) 一条压缩条目 584 B；索引器小头一条 132 B（只有压缩率为 4 的层建它）
     ← traces/run_m01_m02_ledger.json（『KV 账·逐层摊销』两行）
  5) 残差流 4 条（hc_mult=4）、每半层一次读入加一次写回；MoE 每 token 挑 6 个路由专家
     加 1 个共享专家；最前 3 层改按 token id 查表；MTP 一档
     （num_nextn_predict_layers=1）← book/papers/ch28-deepseek-v4-principles/paper.md §八 config 表

配色走 book/cartography/l0_common.py 的角色常量（同源强制）：模型层/主干 = GPU 执行臂绿、
KV 缓存账 = 青、采样与出口 = 品红、多预测一步旁路 = 橙虚线（与第 29 章同一台机器的代码
口径图同款）。坐标全部由常量与循环计算；文本全 esc()；数字全部经 assert 卡住。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 1020
MX, BXR = 40, 1460
C_TXT, C_MUTE, C_FAINT = lc.C_TXT, lc.C_MUTE, lc.C_FAINT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F          # 模型层 / 主干 / 出口主路
C_KV_S, C_KV_F = lc.C_KV_S, lc.C_KV_F              # KV 缓存账（压缩条目 / 滑窗 / KV 轴）
C_SAM_S, C_SAM_F = lc.C_SAM_S, lc.C_SAM_F          # 采样与出口（logits）
C_MTP_S, C_MTP_F = lc.C_ENG_S, lc.C_ENG_F          # 多预测一步旁路（橙，同第 29 章代码口径图）

# ---------------- 数字（全部来自 figure-request numbers；图面逐字引用，禁改） ----------------
N_MAIN = 43                 # 主干层数（43 层）
N_SWA, N_C4A, N_C128A = 2, 21, 20     # 三类层：牌子 0 / 4 / 128
assert N_SWA + N_C4A + N_C128A == N_MAIN, '三类层计数必须加回 43'
N_WIN = 128                 # 滑窗带恒长 128 条未压缩 KV
TOPK = 512                  # 选择器挑 512 条（top-k 封顶）
N_HCA_FULL = 7812           # 牌子 128 的层一条不挑：全看 7812 条（1M // 128）
AXIS_CSA = N_WIN + TOPK                 # 640 条
AXIS_HCA = N_WIN + N_HCA_FULL           # 7940 条
assert AXIS_CSA == 640 and AXIS_HCA == 7940, '两条 KV 轴读数必须与 spec 逐字一致'
B_ENTRY, B_INDEX = 584, 132   # 一条压缩条目 584 B；索引器小头一条 132 B
N_HC = 4                      # 残差流 4 条（hc_mult）
N_EXPERT, N_TOK_EXPERT, N_SHARED = 256, 6, 1   # 256 挑 6 + 1 个共享专家
N_HASH, N_NEXTN = 3, 1        # 最前 3 层查表；MTP 一档

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
              '<marker id="mg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SAM_S}"/></marker>'
              '</defs>')

# ================= 标题区 =================
lc.text(MX, 34, '一台 V4 的方框图：七件套各站在哪（公式口径）', 17.5, C_TXT, 'start', True,
        maxw=880, tag='title')
lc.text(MX, 58, '一个 token 从残差带读进去、过注意力半层与 MoE 半层再写回；主干尾部先分叉一份给多预测一步、'
                '再压回单流算 logits',
        10, C_MUTE, 'start', maxw=1180, tag='sub')
lc.text(BXR, 32, '本图的位置', 9.5, C_MUTE, 'end', True, tag='pos:t')
lc.text(BXR, 50, '全景架构图（L0）里『GPU 执行臂 · 模型层 forward』这一块', 9, C_MUTE, 'end',
        maxw=520, tag='pos:a')
lc.text(BXR, 66, '——本章其余每一张机制图，都是本图上某一件的内部放大', 9, C_MUTE, 'end',
        maxw=520, tag='pos:b')

# ================= 图例（颜色即角色，角色色即身份） =================
lgx = MX
LEGEND = [('模型层（GPU 执行臂）', C_GPU_S, False),
          ('KV 缓存账', C_KV_S, False),
          ('采样与出口', C_SAM_S, False),
          ('多预测一步（旁路）', C_MTP_S, True)]
for role, color, dash in LEGEND:
    lc.rect(lgx, 80, 16, 11, '#ffffff', color, rx=2.5, sw=1.5, dash=dash)
    lc.text(lgx + 22, 90, role, 8.8, '#334155', 'start', tag='lg:%s' % role[:4])
    lgx += 22 + 16 + lc.tw(role, 8.8) + 24

# ================= 入口条（主干顶） =================
TRK_X0, TRK_X1 = 500, 1000        # 主干容器左右缘
IN_Y0, IN_H = 116, 44
lc.rect(TRK_X0, IN_Y0, TRK_X1 - TRK_X0, IN_H, '#ffffff', C_GPU_S, rx=8, sw=1.5)
lc.text(TRK_X0 + 16, IN_Y0 + 19, '入口：一个 token 的隐状态', 10, C_TXT, 'start', True,
        maxw=460, tag='in:t')
lc.text(TRK_X0 + 16, IN_Y0 + 36, '复制成 4 条流（hc_mult = 4）：主干里每一拍都在 4 条流上进、4 条流上出',
        8.6, C_MUTE, 'start', maxw=468, tag='in:s')

# ================= 主干容器 =================
S1_Y0, S1_H = 200, 130          # 段①：第 1–2 层 · 牌子 0
S2_Y0, S2_H = 342, 300          # 段②：第 3–43 层 · 牌子 4 与 128 交替
TK_Y0, TK_Y1 = 190, S2_Y0 + S2_H + 16
ARROW_X = TRK_X0 + 380          # 入口 → 主干 的连线（避开容器标题）
lc.seg(ARROW_X, IN_Y0 + IN_H, ARROW_X, TK_Y0, C_GPU_S, 2.0, 'gn')
lc.rect(TRK_X0, TK_Y0, TRK_X1 - TRK_X0, TK_Y1 - TK_Y0, '#ffffff', C_GPU_S, rx=10, sw=1.8)
lc.text(TRK_X0 + 8, 182, '主干：%d 层，每层 = 一个注意力半层 + 一个 MoE 半层' % N_MAIN,
        10.5, C_TXT, 'start', True, maxw=560, tag='tk:t')

# ---- 残差带（主干左侧那条 4 线带） ----
BD_X0, BD_W = TRK_X0 + 6, 34
BD_Y0, BD_Y1 = TK_Y0 + 16, TK_Y1 - 16
lc.rect(BD_X0, BD_Y0, BD_W, BD_Y1 - BD_Y0, C_GPU_F, '#86efac', rx=6, sw=1.0)
for i in range(N_HC):
    x = BD_X0 + 6 + i * 7
    lc.seg(x, BD_Y0 + 8, x, BD_Y1 - 8, '#22c55e', 2.2)
lc.text(BD_X0 + BD_W / 2, BD_Y0 - 6, '残差带', 8.2, '#15803d', 'middle', maxw=BD_W + 12, tag='bd:t')
lc.text(BD_X0 + BD_W / 2, BD_Y1 + 8, '%d 条流' % N_HC, 8.2, '#15803d', 'middle',
        maxw=BD_W + 12, tag='bd:s')

# ---- 主干两段（43 层的三段代表：前两层 / 中段 / 末层之后的出口见右侧面板） ----
SEG_X0, SEG_X1 = TRK_X0 + 52, TRK_X1 - 12
BOX_A_X0, BOX_A_W = SEG_X0 + 14, 214      # 注意力半层框
BOX_M_X0, BOX_M_W = BOX_A_X0 + 226, 188   # MoE 半层框


def half_box(y0, h, title, lines, tag):
    """画一个半层框：标题 + 若干行（行距 18）。"""
    lc.rect(BOX_A_X0, y0, BOX_A_W, h, '#ffffff', C_GPU_S, rx=6, sw=1.2)
    lc.text(BOX_A_X0 + 12, y0 + 18, title, 9.5, C_TXT, 'start', True, maxw=BOX_A_W - 24,
            tag=tag + ':t')
    for i, (s, fs, col) in enumerate(lines):
        lc.text(BOX_A_X0 + 12, y0 + 18 + (i + 1) * 18, s, fs, col, 'start',
                maxw=BOX_A_W - 24, tag=tag + ':%d' % i)


def moe_box(y0, h, lines, tag):
    lc.rect(BOX_M_X0, y0, BOX_M_W, h, '#ffffff', C_GPU_S, rx=6, sw=1.2)
    lc.text(BOX_M_X0 + 12, y0 + 18, 'MoE 半层', 9.5, C_TXT, 'start', True, maxw=BOX_M_W - 24,
            tag=tag + ':t')
    for i, (s, fs, col) in enumerate(lines):
        lc.text(BOX_M_X0 + 12, y0 + 18 + (i + 1) * 18, s, fs, col, 'start',
                maxw=BOX_M_W - 24, tag=tag + ':%d' % i)


# 段①：第 1–2 层 · 牌子 0
lc.rect(SEG_X0, S1_Y0, SEG_X1 - SEG_X0, S1_H, C_GPU_F, '#86efac', rx=8, sw=1.4)
lc.text(SEG_X0 + 14, S1_Y0 + 22, '① 第 1–%d 层 · 牌子 0（纯滑窗：这一层没有压缩机）' % N_SWA,
        10, '#14532d', 'start', True, maxw=SEG_X1 - SEG_X0 - 28, tag='s1:t')
half_box(S1_Y0 + 36, 80, '注意力半层',
         [('只读滑窗带：最近 %d 条原样 KV' % N_WIN, 8.6, '#334155'),
          ('（没有压缩条目，也没有选择器）', 8.6, C_MUTE)], 's1a')
moe_box(S1_Y0 + 36, 80,
        [('%d 挑 %d + %d 个共享专家' % (N_EXPERT, N_TOK_EXPERT, N_SHARED), 8.6, '#334155'),
         ('按 token id 查表（最前 %d 层）' % N_HASH, 8.6, '#334155')], 's1m')

# 段②：第 3–43 层 · 牌子 4 与 128 交替
lc.rect(SEG_X0, S2_Y0, SEG_X1 - SEG_X0, S2_H, C_GPU_F, '#86efac', rx=8, sw=1.4)
lc.text(SEG_X0 + 14, S2_Y0 + 22,
        '② 第 %d–%d 层 · 牌子 4 与 128 交替（%d 层 c4a + %d 层 c128a）'
        % (N_SWA + 1, N_MAIN, N_C4A, N_C128A),
        10, '#14532d', 'start', True, maxw=SEG_X1 - SEG_X0 - 28, tag='s2:t')
half_box(S2_Y0 + 36, 150, '注意力半层 · 三样 KV 来源',
         [('① 滑窗带：恒长 %d 条原样 KV' % N_WIN, 8.6, '#334155'),
          ('② 压缩条目：每 4 或 128 个 token 一条', 8.6, '#334155'),
          ('③ 选择器：只在牌子 4 挑 %d 条' % TOPK, 8.6, '#334155'),
          ('牌子 128 的层不建选择器：', 8.4, C_MUTE),
          ('条目本来就少，全看不比挑选贵', 8.4, C_MUTE)], 's2a')
moe_box(S2_Y0 + 36, 150,
        [('%d 挑 %d 个路由专家' % (N_EXPERT, N_TOK_EXPERT), 8.6, '#334155'),
         ('另有 %d 个共享专家无条件算' % N_SHARED, 8.6, '#334155'),
         ('按分数挑（第 %d 层起）' % (N_HASH + 1), 8.6, '#334155'),
         ('挑中的才花算力；', 8.4, C_MUTE),
         ('共享那个每 token 都花', 8.4, C_MUTE)], 's2m')
lc.text(SEG_X0 + 14, S2_Y0 + 216, '牌子 4 与牌子 128 交替：同一台压缩机的两个档位——',
        9, '#14532d', 'start', maxw=SEG_X1 - SEG_X0 - 28, tag='s2:n1')
lc.text(SEG_X0 + 14, S2_Y0 + 236, '压得少的挑着看（挑 %d 条），压得狠的全看（%d 条）'
        % (TOPK, N_HCA_FULL), 9, '#14532d', 'start', maxw=SEG_X1 - SEG_X0 - 28, tag='s2:n2')
lc.text(SEG_X0 + 14, S2_Y0 + 266,
        '两类层共用同一条滑窗带与同一次核心注意力；论文没解释它们为什么交替排布。',
        8.6, C_MUTE, 'start', maxw=SEG_X1 - SEG_X0 - 28, tag='s2:n3')

# ================= 左栏三块挂件 =================
LP_X0, LP_W = MX, 430
LEFT = [
    (190, 140, '残差带：%d 条流（hc_mult = 4）· 第五件' % N_HC,
     [('· 每个半层一次读入：%d 条流聚成 1 路，交给子层去算' % N_HC, 9, '#334155'),
      ('· 每个半层一次写回：结果散回 %d 条流，与旧残差相加' % N_HC, 9, '#334155'),
      ('· %d 条流之间还会互相混合（B_l）' % N_HC, 9, '#334155')],
     '→ 就是主干左侧那条 4 线带：进任何半层都先聚后散', 260),
    (346, 154, '压缩缓存与选择器（第一、二、三件）',
     [('· 一条压缩条目 %d B：每 4 或每 128 个 token 一条' % B_ENTRY, 9, '#334155'),
      ('· 牌子 4 的层还建索引器小头：一条 %d B' % B_INDEX, 9, '#334155'),
      ('· 选择器只站在牌子 4 的层上：给候选条目打分、挑 %d 条' % TOPK, 9, '#334155'),
      ('· 牌子 128 的层不建选择器：条目本来就少，全看不比挑选贵', 9, '#334155')],
     '→ 压得越狠、条数越少；挑不挑因此分了两档', 430),
    (516, 164, '滑窗带（第四件：唯一只花不省的一件）',
     [('· 恒长 %d 条未压缩 KV：留最近 %d 个 token 的原样缓存' % (N_WIN, N_WIN), 9, '#334155'),
      ('· 每层都挂：%d 层各自多背这一份' % N_MAIN, 9, '#334155'),
      ('· 与压缩条目并进同一次注意力：谁分多由 query 决定', 9, '#334155')],
     '→ 上下文越长它越被摊薄；短上下文里它最贵', 598),
]
for y0, h, title, lines, note, tie_y in LEFT:
    lc.rect(LP_X0, y0, LP_W, h, '#ffffff', C_MUTE, rx=8, sw=1.3)
    lc.text(LP_X0 + 14, y0 + 26, title, 10.5, C_TXT, 'start', True, maxw=LP_W - 28, tag='lp:t')
    for i, (s, fs, col) in enumerate(lines):
        lc.text(LP_X0 + 14, y0 + 50 + i * 18, s, fs, col, 'start', maxw=LP_W - 28,
                tag='lp:l%d' % i)
    lc.text(LP_X0 + 14, y0 + 50 + len(lines) * 18 + 8, note, 8.6, '#155e75', 'start',
            maxw=LP_W - 28, tag='lp:n')
    lc.seg(LP_X0 + LP_W, tie_y, TRK_X0, tie_y, '#94a3b8', 1.2, 'gy', dash=True)

# ================= 左下的 KV 轴小条（同一个比例尺） =================
KV_X0, KV_Y0, KV_W, KV_H = MX, 730, 430, 200
lc.rect(KV_X0, KV_Y0, KV_W, KV_H, C_KV_F, C_KV_S, rx=8, sw=1.4)
lc.text(KV_X0 + 14, KV_Y0 + 24, '一次注意力看到的 KV 轴（两行同一个比例尺）', 10.5, C_TXT,
        'start', True, maxw=KV_W - 28, tag='kv:t')
lgx = KV_X0 + 14
for fill, lab in ((('#a5f3fc', '滑窗段：原样 KV'), (C_KV_F, '压缩条目段'))):
    lc.rect(lgx, KV_Y0 + 34, 14, 10, fill, C_KV_S, rx=1.5, sw=1.0)
    lc.text(lgx + 20, KV_Y0 + 43, lab, 8.6, '#334155', 'start', tag='kv:lg')
    lgx += 20 + 14 + lc.tw(lab, 8.6) + 26
AX_L, AX_R = KV_X0 + 14, KV_X0 + KV_W - 14
SCALE = (AX_R - AX_L) / AXIS_HCA            # px / 条：两行共用一个比例尺
BARS = [('CSA 层（牌子 4）', AXIS_CSA, N_WIN, TOPK, KV_Y0 + 70),
        ('HCA 层（牌子 128）', AXIS_HCA, N_WIN, N_HCA_FULL, KV_Y0 + 126)]
for name, total, n_swa, n_comp, y in BARS:
    bw = total * SCALE
    lc.text(AX_L, y - 8, name, 9, '#155e75', 'start', True, maxw=180, tag='kv:n%s' % name[:3])
    lc.text(AX_R, y - 8, '%d + %d = %d 条' % (n_swa, n_comp, total), 8.8, '#155e75', 'end',
            maxw=200, tag='kv:r%s' % name[:3])
    sw = n_swa * SCALE
    lc.rect(AX_L, y, max(sw, 3.0), 18, '#a5f3fc', C_KV_S, rx=1.5, sw=1.0)
    lc.rect(AX_L + sw, y, max(bw - sw, 2.0), 18, C_KV_F, C_KV_S, rx=1.5, sw=1.0)
lc.text(AX_L, KV_Y0 + 172, '两行同一比例尺：条长正比于条数；两条都从同样 %d 条的滑窗段开头。' % N_WIN,
        8.4, C_MUTE, 'start', maxw=KV_W - 28, tag='kv:f1')
lc.text(AX_L, KV_Y0 + 188, '按 1M 上下文计：牌子 4 的压缩段被 %d 条封顶，牌子 128 的一条不挑、全看。'
        % TOPK, 8.4, C_FAINT, 'start', maxw=KV_W - 28, tag='kv:f2')

# ================= 右栏：专家池 =================
RP_X0, RP_W = 1014, BXR - 1014
EX_Y0, EX_H = 342, 232
lc.rect(RP_X0, EX_Y0, RP_W, EX_H, '#ffffff', C_MUTE, rx=8, sw=1.3)
lc.text(RP_X0 + 14, EX_Y0 + 26, '专家池（MoE 半层挑谁算）· 第六件', 10.5, C_TXT, 'start', True,
        maxw=RP_W - 28, tag='ex:t')
for i, (s, fs, col) in enumerate([
        ('· 池子里是 %d 个路由专家；每 token 挑 %d 个算' % (N_EXPERT, N_TOK_EXPERT), 9, '#334155'),
        ('· 另有 %d 个共享专家：每 token 必算，不参与挑选' % N_SHARED, 9, '#334155'),
        ('· 最前 %d 层的挑选方式不同：不按分数挑，按 token id 查表' % N_HASH, 9, '#334155'),
        ('（num_hash_layers = %d；本图主干里第 1–%d 层是查表层）' % (N_HASH, N_HASH), 8.6, C_MUTE),
        ('· 第 %d 层起回到按分数挑' % (N_HASH + 1), 9, '#334155')]):
    lc.text(RP_X0 + 14, EX_Y0 + 52 + i * 20, s, fs, col, 'start', maxw=RP_W - 28,
            tag='ex:l%d' % i)
lc.text(RP_X0 + 14, EX_Y0 + 168, '→ 挑几个、怎么挑，决定这个 token 的算力花在哪些专家上；',
        8.6, '#155e75', 'start', maxw=RP_W - 28, tag='ex:n1')
lc.text(RP_X0 + 14, EX_Y0 + 186, '   共享的那一份不挑也关不掉，是每 token 的固定开销。', 8.6, '#155e75',
        'start', maxw=RP_W - 28, tag='ex:n2')
lc.seg(BOX_M_X0 + BOX_M_W, EX_Y0 + 58, RP_X0, EX_Y0 + 58, '#94a3b8', 1.2, 'gy', dash=True)

# ================= 右下：主干尾部的分叉（出口） =================
PT_X0, PT_Y0, PT_W, PT_H = RP_X0, 620, RP_W, 310
lc.rect(PT_X0, PT_Y0, PT_W, PT_H, '#ffffff', C_MUTE, rx=8, sw=1.3)
lc.text(PT_X0 + 14, PT_Y0 + 26, '出口（主干尾部）：先拷一份，再压回单流', 10.5, C_TXT, 'start',
        True, maxw=PT_W - 28, tag='pt:t')
FLAT_X, FLAT_Y, FLAT_W, FLAT_H = PT_X0 + 14, PT_Y0 + 70, 140, 48
# 主干底 → 出口面板（折线：下→右，两端都贴框边；入口对准「4 条流展平」框的中线）
ENTRY_Y = FLAT_Y + FLAT_H / 2
lc.parrow([(750, TK_Y1), (750, ENTRY_Y), (PT_X0, ENTRY_Y)], C_GPU_S, 2.0, 'gn')
lc.text(762, ENTRY_Y - 8, '主干输出的残差：%d 条流' % N_HC, 8.8, '#14532d', 'start', maxw=260,
        tag='pt:ar')
lc.rect(FLAT_X, FLAT_Y, FLAT_W, FLAT_H, '#ffffff', C_GPU_S, rx=6, sw=1.4)
lc.text(FLAT_X + FLAT_W / 2, FLAT_Y + 20, '%d 条流展平' % N_HC, 9, C_TXT, 'middle', True,
        maxw=FLAT_W - 12, tag='pt:fa')
lc.text(FLAT_X + FLAT_W / 2, FLAT_Y + 37, '（主干侧未压回）', 8.4, C_MUTE, 'middle',
        maxw=FLAT_W - 12, tag='pt:fb')
BR_X, BR_W, BR_H = PT_X0 + 206, 226, 60
BRANCH = [
    (PT_Y0 + 52, C_MTP_F, C_MTP_S, True, '#9a3412', '① 拷一份给多预测一步（第七件）',
     '（num_nextn_predict_layers = %d）' % N_NEXTN, 702),
    (PT_Y0 + 144, C_GPU_F, C_GPU_S, False, '#14532d', '② hc_head：%d 条流压回 1 条' % N_HC,
     '（sigmoid 门控加权求和）', 794),
]
FLAT_CY = FLAT_Y + FLAT_H / 2
FORK_X = (FLAT_X + FLAT_W + BR_X) / 2
for y0, fill, stroke, dash, tcol, t1, t2, fy in BRANCH:
    lc.rect(BR_X, y0, BR_W, BR_H, fill, stroke, rx=6, sw=1.4, dash=dash)
    lc.text(BR_X + 14, y0 + 24, t1, 9, tcol, 'start', True, maxw=BR_W - 28, tag='pt:b1')
    lc.text(BR_X + 14, y0 + 43, t2, 8.4, tcol, 'start', maxw=BR_W - 28, tag='pt:b2')
    lc.parrow([(FLAT_X + FLAT_W, FLAT_CY), (FORK_X, FLAT_CY), (FORK_X, fy), (BR_X, fy)],
           '#94a3b8', 1.6, 'gy')
LOG_Y = PT_Y0 + 236
lc.rect(BR_X, LOG_Y, BR_W, 44, C_SAM_F, C_SAM_S, rx=6, sw=1.4)
lc.text(BR_X + BR_W / 2, LOG_Y + 27, '归一化 → 算 logits', 9, '#831843', 'middle', True,
        maxw=BR_W - 12, tag='pt:lg')
lc.seg(BR_X + BR_W / 2, PT_Y0 + 204, BR_X + BR_W / 2, LOG_Y, C_SAM_S, 1.8, 'mg')
lc.text(PT_X0 + 14, PT_Y0 + 296, '顺序不能反：多预测一步拿的是压回之前的残差（先拷贝、后压回）。',
        8.4, C_MUTE, 'start', maxw=PT_W - 28, tag='pt:n')

# ================= 中下：读图 =================
RD_X0, RD_Y0, RD_W, RD_H = 500, 756, 480, 174
lc.rect(RD_X0, RD_Y0, RD_W, RD_H, '#f8fafc', '#cbd5e1', rx=8, sw=1.3, dash=True)
lc.text(RD_X0 + 14, RD_Y0 + 24, '读图', 10, C_TXT, 'start', True, maxw=120, tag='rd:t')
for i, (s, col) in enumerate([
        ('① 竖着读：一个 token 的路——入口复制成 %d 条流、穿过 %d 个半层对、出口压回 1 条'
         % (N_HC, N_MAIN), '#334155'),
        ('② 横着读：七件套各站一个位置——注意力半层站三件（压缩、选择、保住近处），', '#334155'),
        ('    MoE 半层站一件（路由），残差带与出口各站一件', '#334155'),
        ('③ 账只算在「条数」与「条宽」上：压缩与选择降条数，滑窗是加账。', '#155e75')]):
    lc.text(RD_X0 + 14, RD_Y0 + 50 + i * 20, s, 9, col, 'start', maxw=RD_W - 28,
            tag='rd:%d' % i)
lc.text(RD_X0 + 14, RD_Y0 + 146,
        '本图是方框图（公式口径）；代码口径（哪个类、哪个字段）预告在第 29 章。', 8.6,
        C_FAINT, 'start', maxw=RD_W - 28, tag='rd:f')

# ================= 页脚：口径 =================
lc.text(MX, 956,
        '数字口径：4 / 128 / 512 / 256 / 6 / 3 / 1 均为官方 config 字段值；%d 层与三类层计数'
        '（%d / %d / %d）由本章驱动脚本按 compress_ratios 纯算术分类。'
        % (N_MAIN, N_SWA, N_C4A, N_C128A),
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 976,
        '条宽口径：一条压缩条目 %d B、索引器小头 %d B；本图不展开字节账——FLOPs 账与 KV 账'
        '在「开账」与「对账」两节逐行结算。' % (B_ENTRY, B_INDEX),
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-machine-map.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
for a, b in ((AXIS_CSA, 640), (AXIS_HCA, 7940)):
    print(f'  KV 轴 {a} 条 = {a * SCALE:.1f}px')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
