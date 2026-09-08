#!/usr/bin/env python3
"""ch15 机制图 · 为什么分组：看户口（KVCacheSpec），不看算法（figure_spec ch15-fig-why-groups，模板 layout）

放大自 L0 显存账本列 KV 半区——组化层与缓存面的交接（与 ch14 组化图呼应：full=蓝、SWA=青）。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：分组的判据是 KVCacheSpec 异同而非注意力算法：gpt-oss/Jamba/Gemma 的滑窗层/状态层
真的少存历史（spec 异型 → 分组各管各账）；DeepSeek V3.2 的 DSA 只稀疏计算不稀疏存储
（全层 MLA 同型 → 单组）。

结构/块表/键尾数字取自配套精简版 host 实跑（hybrid：full(16)+swa(16, 窗 48)）；模型案例
为 pin v0.27.1 行号锚 + arXiv 结构数汇编。DeepSeek V4 已入 v0.27.1（registry.py:L95）：
MLAAttentionSpec 与 SlidingWindowMLASpec 异型 → 分组（案例末卡，深讲归拆读整模型一章）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印符号免疫

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440

FUL_S, FUL_F = '#2563eb', '#eff6ff'      # FullAttentionSpec 家族 = 蓝（沿 ch14 组化图）
SWA_S, SWA_F = '#0891b2', '#ecfeff'      # SlidingWindowSpec = 青（沿 ch14 组化图）
MAM_S, MAM_F = '#7c3aed', '#f5f3ff'      # MambaSpec 状态型 = 紫
GRAY = '#94a3b8'
RED = '#dc2626'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '为什么分组：看户口（KVCacheSpec），不看算法', 17, lc.C_TXT, 'start', True,
        maxw=860, tag='title')
lc.text(MX, 58, '一个共享 BlockPool、每 group 一个管家各持自己的块表——spec 异型才分家；同一请求两组块表合法分叉'
                '（SWA 窗外 null 换位回收、full 纹丝不动）；DSA 反例：计算稀疏 ≠ 存储稀疏',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = 'L0 放大 · 显存账本列 KV 半区 · 组化层 × 缓存面'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

PY, PH = 96, 470

# ================= 左面板：结构（一个池子，多张账） =================
LX, LW = MX, 660
lc.rect(LX, PY, LW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, PY + 22, '结构：一个池子，多张账（hybrid 实拍）', 11.5, lc.C_TXT, 'start', True,
        maxw=LW - 32, tag='lp:t')

# 协调者容器 + 两个管家
CY0, CH0 = PY + 36, 210
lc.rect(LX + 16, CY0, LW - 32, CH0, '#f8fafc', lc.C_MUTE, rx=7, sw=1.2)
lc.text(LX + 30, CY0 + 20, 'HybridKVCacheCoordinator', 11, lc.C_TXT, 'start', True,
        maxw=340, tag='coord:t')
lc.text(LX + 30, CY0 + 36, 'attention_groups 按 spec 归并 · full 排首（给不动点最紧上界）', 8.4,
        '#334155', 'start', maxw=420, tag='coord:s')
lc.text(LX + LW - 30, CY0 + 20, 'vllm/v1/core/kv_cache_coordinator.py', 7.8, lc.C_FAINT, 'end',
        maxw=240, tag='coord:f')

MG_Y, MG_H, MG_W = CY0 + 46, 128, 292
for i, (name, badge, bcol, bfill, lines) in enumerate([
        ('FullAttentionManager', '组 0', FUL_S, FUL_F,
         ['FullAttentionSpec · block_size=16',
          'req_to_blocks：自己的一张账',
          '（每请求每组的块表账本）']),
        ('SlidingWindowManager', '组 1', SWA_S, SWA_F,
         ['SlidingWindowSpec · 窗 48（=3 块）',
          'block_size=16 · req_to_blocks',
          '（窗外块下一拍即回收）'])]):
    bx = LX + 30 + i * (MG_W + 16)
    lc.rect(bx, MG_Y, MG_W, MG_H, bfill, bcol, rx=7, sw=1.4)
    lc.text(bx + 14, MG_Y + 20, name, 10.5, lc.C_TXT, 'start', True, maxw=MG_W - 74, tag='mg:t')
    bw = lc.tw(badge, 9, True) + 14
    lc.rect(bx + MG_W - bw - 8, MG_Y + 6, bw, 18, '#ffffff', bcol, rx=9, sw=1.1)
    lc.text(bx + MG_W - bw / 2 - 8, MG_Y + 18.5, badge, 9, bcol, 'middle', True, maxw=bw - 4,
            tag='mg:b')
    for j, ln in enumerate(lines):
        lc.text(bx + 14, MG_Y + 42 + j * 16, ln, 8.6, '#334155', 'start', maxw=MG_W - 26,
                tag='mg:l%d' % j)

# 管家 → 池 的两条箭头 + 共享说明
POOL_Y = CY0 + CH0 + 30
for i in range(2):
    ax = LX + 30 + i * (MG_W + 16) + MG_W / 2
    lc.seg(ax, MG_Y + MG_H, ax, POOL_Y, lc.C_MUTE, 1.6, 'std')
lc.text(LX + LW / 2, POOL_Y - 8, 'touch · get_new_blocks · free——两组管家全落同一个池', 8.6,
        lc.C_MUTE, 'middle', maxw=430, tag='pool:arr')

# 共享 BlockPool + 10 块实拍
POOL_H = 118
lc.rect(LX + 16, POOL_Y, LW - 32, POOL_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(LX + 30, POOL_Y + 18, 'BlockPool（唯一一个——整套房子 · 等大房间）', 10, lc.C_TXT, 'start',
        True, maxw=380, tag='pool:t')
CELL_W, CELL_H, CELL_G = 56, 34, 4
CELLS = [(1, FUL_S, FUL_F), (2, FUL_S, FUL_F), (3, FUL_S, FUL_F), (4, FUL_S, FUL_F),
         (5, GRAY, '#ffffff'), (6, SWA_S, SWA_F), (7, SWA_S, SWA_F), (8, SWA_S, SWA_F),
         (9, FUL_S, FUL_F), (10, SWA_S, SWA_F)]
cells_w = len(CELLS) * CELL_W + (len(CELLS) - 1) * CELL_G
cx0 = LX + 16 + (LW - 32 - cells_w) / 2
CY = POOL_Y + 28
for i, (num, stroke, fill) in enumerate(CELLS):
    cx = cx0 + i * (CELL_W + CELL_G)
    lc.rect(cx, CY, CELL_W, CELL_H, fill, stroke, rx=4, sw=1.2, dash=(num == 5))
    lc.text(cx + CELL_W / 2, CY + 21, str(num), 11, stroke if num != 5 else GRAY, 'middle', True,
            maxw=CELL_W - 6, tag='cell%d' % num)
lc.text(LX + 30, POOL_Y + 84, '蓝 = full 组持有（1-4、9）· 青 = SWA 组持有（6-8、10）· 5 已 null 换位回收回公共区（灰虚）',
        8, '#334155', 'start', maxw=LW - 60, tag='pool:c1')
lc.text(LX + 30, POOL_Y + 100, 'decode 拍两组各取 1 个新块：9（full 组）/ 10（SWA 组）——同一池里各自演化',
        8, '#334155', 'start', maxw=LW - 60, tag='pool:c2')

# 构造侧注
NQ_Y = POOL_Y + POOL_H + 12
lc.rect(LX + 16, NQ_Y, LW - 32, 44, '#f8fafc', lc.C_FAINT, rx=6, sw=1.1, dash=True)
lc.text(LX + 30, NQ_Y + 18, '构造侧：基类 __init__ 建唯一 BlockPool，每个 group 建一个管家、', 8.4,
        lc.C_TXT, 'start', maxw=LW - 60, tag='nq:1')
lc.text(LX + 30, NQ_Y + 34, '全部注入同一 block_pool 引用——结构上不存在第二个池', 8.4, lc.C_TXT,
        'start', maxw=LW - 60, tag='nq:2')

# ================= 右面板：同一请求两组块表分叉 =================
RX, RW = LX + LW + 28, BXR - (LX + LW + 28)
lc.rect(RX, PY, RW, PH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, PY + 22, '同一请求的两组块表（64 token + 1 decode token）——分叉实拍', 11.5,
        lc.C_TXT, 'start', True, maxw=RW - 32, tag='rp:t')

# Request 芯片 → prefill 注
rq_w = 350
lc.rect(RX + (RW - rq_w) / 2, PY + 34, rq_w, 26, '#ffffff', lc.C_MUTE, rx=13, sw=1.2)
lc.text(RX + RW / 2, PY + 51, 'Request：prompt 4 满块 + 1 个 decode token', 9.5, lc.C_TXT,
        'middle', True, maxw=rq_w - 10, tag='rq:t')
lc.seg(RX + RW / 2, PY + 60, RX + RW / 2, PY + 74, lc.C_MUTE, 1.4, 'std')
lc.text(RX + RW / 2, PY + 88, 'prefill 后：full [1,2,3,4] · SWA [5,6,7,8]——两组各 4 块（页统一 · 等大块）',
        8.6, lc.C_MUTE, 'middle', maxw=RW - 30, tag='rp:pre')

TC_X, TC_W, TC_H, TC_G = RX + 172, 62, 40, 6


def table_row(ty, label, sub, cells, note, ncol, dashed_first=False):
    """cells = [(text, stroke, fill, dash)] ×5；label 在左、note 在右。"""
    lc.text(RX + 16, ty + 18, label, 9.5, lc.C_TXT, 'start', True, maxw=156, tag='tr:l')
    lc.text(RX + 16, ty + 33, sub, 8, lc.C_MUTE, 'start', maxw=156, tag='tr:s')
    for i, (txt, stroke, fill, dash) in enumerate(cells):
        cx = TC_X + i * (TC_W + TC_G)
        lc.rect(cx, ty, TC_W, TC_H, fill, stroke, rx=4, sw=1.3, dash=dash)
        lc.text(cx + TC_W / 2, ty + 25, txt, 10.5 if len(txt) <= 2 else 8.5,
                GRAY if txt == 'NULL' else lc.C_TXT, 'middle', True, maxw=TC_W - 6,
                tag='tc:%s%d' % (txt, i))
    nx = TC_X + 5 * (TC_W + TC_G) + 8
    for j, ln in enumerate(note):
        lc.text(nx, ty + 16 + j * 14, ln, 8.2, ncol, 'start', maxw=RX + RW - 12 - nx, tag='tr:n%d' % j)


T1Y = PY + 104
table_row(T1Y, 'full 组（组 0）', '全历史 · 纹丝不动',
          [('1', FUL_S, FUL_F, False), ('2', FUL_S, FUL_F, False), ('3', FUL_S, FUL_F, False),
           ('4', FUL_S, FUL_F, False), ('9', FUL_S, FUL_F, False)],
          ['prefill 4 块原封不动 + decode 1 块', '实持 5 块——full 要全历史'], lc.C_TXT)
T2Y = T1Y + 66
table_row(T2Y, 'SWA 组（组 1）', '窗 48（=3 块）',
          [('NULL', GRAY, '#f1f5f9', True), ('6', SWA_S, SWA_F, False), ('7', SWA_S, SWA_F, False),
           ('8', SWA_S, SWA_F, False), ('10', SWA_S, SWA_F, False)],
          ['头块窗外：64−48=16 token=1 块', '→ null 换位回收；实持 4 块', '= 窗 48（3 块）+ decode 1 块'],
          RED)
# 跨面板虚线：SWA 表 NULL 格 → 池中块 5（回收去向）
_null_c = (TC_X + TC_W / 2, T2Y + TC_H)
_p5_c = (cx0 + 4 * (CELL_W + CELL_G) + CELL_W, CY + CELL_H / 2)
lc.seg(_null_c[0] - 8, _null_c[1] - 6, _p5_c[0] + 4, _p5_c[1] - 8, GRAY, 1.4, dash=True)
lc.text((_null_c[0] + _p5_c[0]) / 2 - 62, (_null_c[1] + _p5_c[1]) / 2 - 10, '块 5 回公共区', 8,
        GRAY, 'middle', maxw=120, tag='rp:swap')

# 槽位不变量 + uniform 对照
IV_Y = T2Y + 66
lc.rect(RX + 16, IV_Y, RW - 32, 32, '#f8fafc', lc.C_FAINT, rx=6, sw=1.1, dash=True)
lc.text(RX + 28, IV_Y + 13, '槽位不变量：块表第 i 项 ↔ 第 i×block_size 个 token——null 只换条目、不缩表（ch14 已立）',
        8.4, lc.C_TXT, 'start', maxw=RW - 56, tag='rp:inv')
lc.text(RX + 28, IV_Y + 26, 'uniform 对照（全层同型 · 单组）：1 管家 1 张表——同一请求块表 [1,2,3,4,9]、全程无回收',
        8.4, lc.C_MUTE, 'start', maxw=RW - 56, tag='rp:uni')

# 键构成条（两组各一把钥匙）
KS_Y = IV_Y + 44
lc.text(RX + 16, KS_Y + 12, '两组各一把钥匙——组号进键才不串门', 9.5, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='ks:t')
KH_W, KG_W, KH2 = 196, 168, 30
for i, (tail, tcol, tfill, who) in enumerate([('00 00 00 00', FUL_S, FUL_F, 'full 组（组号 0）'),
                                              ('00 00 00 01', SWA_S, SWA_F, 'SWA 组（组号 1）')]):
    ky = KS_Y + 22 + i * 38
    lc.rect(RX + 24, ky, KH_W, 28, '#f1f5f9', lc.C_MUTE, rx=4, sw=1.0)
    lc.text(RX + 24 + KH_W / 2, ky + 18, 'hash₀ · 32 字节', 9, lc.C_TXT, 'middle', True,
            maxw=KH_W - 8, tag='ks:h%d' % i)
    lc.text(RX + 24 + KH_W, ky + 18, '∥', 10, lc.C_MUTE, 'middle', tag='ks:par%d' % i)
    lc.rect(RX + 24 + KH_W + 16, ky, KG_W, 28, tfill, tcol, rx=4, sw=1.3)
    lc.text(RX + 24 + KH_W + 16 + KG_W / 2, ky + 18, tail, 10, tcol, 'middle', True,
            maxw=KG_W - 8, tag='ks:g%d' % i)
    lc.seg(RX + 24 + KH_W + 16 + KG_W + 6, ky + 14, RX + 24 + KH_W + 16 + KG_W + 26, ky + 14,
           tcol, 1.4, 'std')
    lc.text(RX + 24 + KH_W + 16 + KG_W + 32, ky + 18, who, 8.6, lc.C_TXT, 'start', maxw=150,
            tag='ks:w%d' % i)
lc.text(RX + 24, KS_Y + 22 + 2 * 38 + 4, '键 = 32 字节哈希 ∥ 4 字节组号（big-endian）——同一前缀在每组各查各的物理块',
        8.2, lc.C_MUTE, 'start', maxw=RW - 56, tag='ks:note')

# ================= 判据带（全宽） =================
BY = PY + PH + 18
BH = 152
lc.rect(MX, BY, BXR - MX, BH, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.4)
lc.text(MX + 18, BY + 24, '判据：看户口（KVCacheSpec），不看算法——同型必同组、异型必分组', 12,
        lc.C_KV_S, 'start', True, maxw=700, tag='bd:t')
COLS3 = [
    (MX + 18, 430, 'spec 全同 ⇒ 单组（Unitary）', [
        '· 一张块表一个管家，命中一次链上查完',
        '· 纯 full、DeepSeek V3.2 走这条',
        '· 单组直通：无不动点、无会签']),
    (MX + 470, 430, 'spec 异型 ⇒ 分组（Hybrid）', [
        '· 每 spec 一管家，组号进哈希键',
        '· 多组命中要全部组会签（不动点调和）',
        '· gpt-oss / Jamba / Gemma 3 走这条']),
    (MX + 922, 440, '为什么单一账本不够', [
        '· 同一块 id 在不同 spec 下「覆盖多少 token /',
        '  留多少历史」不同——SWA 窗外可回收、full 不可；',
        '· Mamba 状态不随 block_size 缩放',
        '· 组内同型 = 页统一 · 等量化组的硬约束（ch14）']),
]
for cx0_, cw_, head, lines in COLS3:
    lc.text(cx0_, BY + 52, head, 10, lc.C_TXT, 'start', True, maxw=cw_, tag='bd:h')
    for j, ln in enumerate(lines):
        lc.text(cx0_, BY + 72 + j * 15, ln, 8.6, '#334155', 'start', maxw=cw_, tag='bd:l%d' % j)
lc.text(MX + 18, BY + BH - 12, '一句话：不是算法要分组，是「KV 要不要留、留多少、什么形状」出了分歧才分组'
        '——算法再稀疏、每层存的 KV 同型，就依然是一个组', 9.2, lc.C_KV_S, 'start', True,
        maxw=BXR - MX - 36, tag='bd:one')

# ================= 案例条（5 卡） =================
KY = BY + BH + 16
lc.text(MX, KY + 16, '模型案例：spec 异同定分组（层混排 → 各层自报 KVCacheSpec → 分不分）', 11.5,
        lc.C_TXT, 'start', True, maxw=900, tag='cs:t')
CARD_Y, CARD_H, GAP = KY + 30, 158, 12
CARDS = [
    ('纯 full（大多数）', 186,
     [('· 全部层同型：FullAttentionSpec', FUL_S)],
     '单组（Unitary 三态分派）', 'kv_cache_coordinator.py:L851-L903', []),
    ('gpt-oss', 230,
     [('· dense 与 sliding-128 每两层交替', None),
      ('· +EAGLE 草稿层后：12 滑窗 + 13 dense', None)],
     '分组 · 等量组 13/13（padding 1 层）', 'ch14 实跑表（组大小 13）', []),
    ('Jamba', 172,
     [('· attention : Mamba = 1:7 掺层', None)],
     '分组', 'arXiv:2403.19887', []),
    ('Gemma 3', 208,
     [('· 局部（窗 1024）: 全局 = 5:1', None),
      ('· layers[i::5] 交错入组', None)],
     '分组 · 6 组（组大小 2）', 'arXiv:2503.19786', []),
    ('DeepSeek V3.2（DSA）· 反例', 330,
     [('· 全层同一 lambda 构造（make_layers 单 lambda 造所有层）', None),
      ('· is_sparse=is_v32 是模型级标志、非层间混排', None),
      ('· MLAAttentionSpec ⊂ FullAttentionSpec——lightning indexer', None),
      ('  只挑 attention 计算的 token，KV 照存全量 MLA 压缩形式', None)],
     '单组——计算稀疏 ≠ 存储稀疏',
     'deepseek_v2.py:L1400-L1406 · L1159（全锚见页脚）', [RED]),
    ('DeepSeek V4', 190,
     [('· MLAAttentionSpec ⊂ Full 家族', FUL_S),
      ('· SlidingWindowMLASpec ⊂ 滑窗', SWA_S),
      ('· 全局层与 SWA/压缩层各报各的', None)],
     '分组（两类异型 spec）', 'registry.py:L95 · 拆读整模型章', []),
]
cxx = MX
for name, cwid, lines, verdict, prov, vcol in CARDS:
    lc.rect(cxx, CARD_Y, cwid, CARD_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
    lc.text(cxx + 12, CARD_Y + 19, name, 10, lc.C_TXT, 'start', True, maxw=cwid - 24,
            tag='cd:' + name[:8])
    ly = CARD_Y + 40
    for txt, col in lines:
        lc.text(cxx + 12, ly, txt, 8.4, col or lc.C_TXT, 'start', maxw=cwid - 22, tag='cd:l')
        ly += 15
    vw = lc.tw(verdict, 8.8, True) + 16
    lc.rect(cxx + 12, CARD_Y + CARD_H - 42, vw, 20, '#f8fafc', vcol[0] if vcol else lc.C_MUTE,
            rx=9, sw=1.2)
    lc.text(cxx + 12 + vw / 2, CARD_Y + CARD_H - 28.5, verdict, 8.8,
            vcol[0] if vcol else lc.C_TXT, 'middle', True, maxw=vw - 4, tag='cd:v')
    lc.text(cxx + 12, CARD_Y + CARD_H - 8, prov, 7.2, lc.C_FAINT, 'start', maxw=cwid - 20,
            tag='cd:p')
    cxx += cwid + GAP

# ================= 图例 + 页脚 =================
LY = CARD_Y + CARD_H + 24
lx = MX
for stroke, fill, name in [(FUL_S, FUL_F, 'FullAttentionSpec 家族（全历史 KV）'),
                           (SWA_S, SWA_F, 'SlidingWindowSpec（滑窗）'),
                           (MAM_S, MAM_F, 'MambaSpec（状态型）')]:
    lc.rect(lx, LY - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LY + 1, name, 8.8, lc.C_TXT, 'start', maxw=210, tag='lg:' + name[:6])
    lx += 26 + lc.tw(name, 8.8) + 20
lc.rect(lx, LY - 9, 20, 13, '#f1f5f9', GRAY, rx=3, sw=1.2, dash=True)
lc.text(lx + 26, LY + 1, 'NULL / 回收回公共区', 8.8, lc.C_TXT, 'start', maxw=170, tag='lg:null')
lx += 26 + lc.tw('NULL / 回收回公共区', 8.8) + 20
lc.text(lx, LY + 1, '红字/红框 = 分叉差异与反例的强调', 8.8, lc.C_MUTE, 'start', maxw=BXR - lx,
        tag='lg:note')

FY = LY + 26
lc.text(MX, FY, '逐字锚 vllm/v1/core/kv_cache_coordinator.py:L90-L120（基类建唯一 BlockPool · 每 group 建管家注入同一引用）'
        ' · L601-L625（Hybrid 归并 attention_groups · full 排首） · kv_cache_utils.py:L57-L66（组号 4 字节 big-endian 进键）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, 'vllm/model_executor/models/deepseek_v2.py:L1400-L1406（make_layers 单 lambda）· L1159（is_sparse=is_v32）'
        ' · vllm/v1/kv_cache_interface.py:L389（MLAAttentionSpec）· remove_skipped_blocks（null 换位，single_type_kv_cache_manager.py）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, FY + 32, '结构 / 块表 / 键尾数字取自配套精简版 host 实跑（full(16)+swa(16, 窗 48) · hash 粒度 16 · 64+1 token）；'
        '模型案例 = pin v0.27.1 行号锚 + arXiv 结构数汇编（DeepSeek V4 按 spec 异型走组化 · registry.py:L95；深讲归拆读整模型一章）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ---------------- 装配输出 ----------------
H = FY + 52
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-why-groups.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
