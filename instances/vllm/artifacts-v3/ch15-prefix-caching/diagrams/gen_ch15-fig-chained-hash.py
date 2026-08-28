#!/usr/bin/env python3
"""ch15 机制图 1 · 链式哈希指纹（figure_spec ch15-fig-chained-hash，模板 flow）

放大自 L0 KV 账本列（kv_column）缓存区·请求侧入口——「哈希在请求上增量算」一格的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：每满 16 token 把（父哈希、本块 token、extra_keys）喂进 sha256 得到本块哈希——
第 i 块哈希递归依赖第 i−1 块，因而是前 i+1 块全部内容的指纹，断一处后面全失效。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑，PYTHONHASHSEED=0）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
GREEN = '#16a34a'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '链式指纹：每满 16 token，把「父哈希 + 本块原文 + 批注」塞进 sha256 打一页新指纹',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'hash_i = H(hash_i-1, 本块 16 token, extra_keys)——第 i 块哈希递归盖住前 i+1 块全部内容；改任何一个字、后面所有指纹全部作废',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 请求侧入口'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96

# ---------------- 左：请求 A 的指纹链（50 token · B=16 → 3 满块） ----------------
LX, LW = MX, 844
ROW_H, ROW_GAP = 50, 64
TX, TW = LX + 20, 168          # token 块框
HX, HW = LX + 228, 230         # sha256 框
BX, BW = LX + 500, 292         # 结果哈希框
ROWS_Y0 = LY + 108
TAIL_Y = ROWS_Y0 + 3 * (ROW_H + ROW_GAP)
LH = (TAIL_Y + 44) - LY + 20
lc.rect(LX, LY, LW, LH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, LY + 22, '请求 A：50 token · hash_block_size=16 → 3 个满块哈希（尾 2 token 不打指纹）',
        11.5, lc.C_TXT, 'start', True, maxw=LW - 32, tag='lp:t')

# 种子框（居中于 sha256 列上方）
seed_cx = HX + HW / 2
SW, SH = 470, 44
sx, sy = seed_cx - SW / 2, LY + 40
lc.rect(sx, sy, SW, SH, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.3)
lc.text(seed_cx, sy + 19, 'NONE_HASH · 32 字节种子（首块的「父」）', 10, lc.C_KV_S, 'middle', True,
        maxw=SW - 10, tag='seed:t')
lc.text(seed_cx, sy + 36, 'PYTHONHASHSEED=0 → sha256("0")；未播种 → 32 随机字节', 8.2, '#334155',
        'middle', maxw=SW - 10, tag='seed:s')

BLOCKS = [
    ('块 0', 'token 0-15', 'h0 = 6358e31578a6…', '@16 · 指纹盖住 token 0-15'),
    ('块 1', 'token 16-31', 'h1 = 3c3c50f324ae…', '@32 · 指纹盖住 token 0-31'),
    ('块 2', 'token 32-47', 'h2 = 0728d91353ef…', '@48 · 指纹盖住 token 0-47'),
]
for i, (bt, tok, hname, cover) in enumerate(BLOCKS):
    ry = ROWS_Y0 + i * (ROW_H + ROW_GAP)
    # token 块
    lc.rect(TX, ry, TW, ROW_H, '#ffffff', lc.C_MUTE, rx=5, sw=1.2)
    lc.text(TX + TW / 2, ry + 21, bt, 10.5, lc.C_TXT, 'middle', True, maxw=TW - 8, tag='b%d:t' % i)
    lc.text(TX + TW / 2, ry + 39, tok + ' · 16 原文', 8.2, '#64748b', 'middle', maxw=TW - 8,
            tag='b%d:s' % i)
    # sha256 框
    lc.rect(HX, ry, HW, ROW_H, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.2)
    lc.text(HX + HW / 2, ry + 21, 'sha256', 10.5, lc.C_KV_S, 'middle', True, maxw=HW - 8,
            tag='h%d:f' % i)
    lc.text(HX + HW / 2, ry + 39, '输入：父哈希 + 16 token + extra_keys', 8.2, '#334155', 'middle',
            maxw=HW - 8, tag='h%d:i' % i)
    # 结果哈希
    lc.rect(BX, ry, BW, ROW_H, '#ffffff', GREEN, rx=5, sw=1.2)
    lc.text(BX + 14, ry + 21, hname, 10, GREEN, 'start', True, maxw=BW - 20, tag='r%d:t' % i)
    lc.text(BX + 14, ry + 39, cover, 8.2, '#64748b', 'start', maxw=BW - 20, tag='r%d:s' % i)
    # token → sha256
    lc.seg(TX + TW, ry + 25, HX, ry + 25, lc.C_MUTE, 1.5, 'std')
    lc.text((TX + TW + HX) / 2, ry + 17, '本块原文', 8, lc.C_MUTE, 'middle', maxw=56, tag='a%d:1' % i)
    # sha256 → 结果
    lc.seg(HX + HW, ry + 25, BX, ry + 25, lc.C_MUTE, 1.5, 'std')
    lc.text((HX + HW + BX) / 2, ry + 17, '指纹', 8, lc.C_MUTE, 'middle', maxw=42, tag='a%d:2' % i)
    # 父哈希链：h_i → 下一块的 sha256
    if i < 2:
        mid = ry + ROW_H + 30
        lc.parrow([(BX + BW / 2, ry + ROW_H), (BX + BW / 2, mid), (HX + HW / 2, mid),
                   (HX + HW / 2, ry + ROW_H + ROW_GAP)], lc.C_KV_S, 1.5, 'std')
        lc.text((BX + BW / 2 + HX + HW / 2) / 2, mid - 7, '父哈希 parent——链就在这一步接上', 8.4,
                lc.C_KV_S, 'middle', maxw=320, tag='chain%d' % i)
# 种子 → 块 0 的 sha256
lc.seg(seed_cx, sy + SH, seed_cx, ROWS_Y0, lc.C_KV_S, 1.6, 'std')
# 尾段（不满块）
lc.rect(TX, TAIL_Y, TW, 44, '#ffffff', GRAY, rx=5, sw=1.2, dash=True)
lc.text(TX + TW / 2, TAIL_Y + 19, '尾段 · token 48-49', 9.8, GRAY, 'middle', True, maxw=TW - 8,
        tag='tail:t')
lc.text(TX + TW / 2, TAIL_Y + 36, '2 token 不满 16', 8.2, GRAY, 'middle', maxw=TW - 8, tag='tail:s')
lc.text(HX, TAIL_Y + 19, '未跨满块边界 → 哈希账本不动（append 至 43、至 51 都是 0 次新哈希）——增量零成本',
        8.6, '#475569', 'start', maxw=BX + BW - HX, tag='tail:n1')
lc.text(HX, TAIL_Y + 36, 'block_hashes 账本累计 3 条：[h0, h1, h2]', 8.6, '#475569', 'start',
        maxw=BX + BW - HX, tag='tail:n2')

# ---------------- 右：两请求共享前 32 token ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
RH = LH
lc.rect(RX, LY, RW, RH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, LY + 22, '两请求共享前 32 token ⇒ 沿链逐块相等到分叉', 11.5, lc.C_TXT, 'start',
        True, maxw=RW - 32, tag='rp:t')
PAIRS = [
    ('块 0', 'h0 = 6358e31578a6', 'h0 逐字节相等', '✓'),
    ('块 1', 'h1 = 3c3c50f324ae', 'h1 逐字节相等', '✓'),
    ('块 2', 'h2 = 0728d91353ef', 'h2 分叉', '✗'),
]
py = LY + 42
PH, PGAP = 56, 14
for i, (bt, hv, verdict, mark) in enumerate(PAIRS):
    yy = py + i * (PH + PGAP)
    col = GREEN if mark == '✓' else lc.C_ABORT
    lc.rect(RX + 16, yy, RW - 32, PH, '#ffffff' if mark == '✗' else '#f0fdf4', col, rx=6, sw=1.3)
    lc.text(RX + 30, yy + 23, bt, 10, lc.C_TXT, 'start', True, maxw=52, tag='p%d:t' % i)
    lc.text(RX + 88, yy + 23, hv, 9.4, '#475569', 'start', maxw=170, tag='p%d:h' % i)
    lc.text(RX + 88, yy + 42, 'B 的第 %d 块同 token' % i, 8, '#94a3b8', 'start', maxw=120,
            tag='p%d:b' % i)
    lc.text(RX + 280, yy + 27, mark, 15, col, 'middle', True, maxw=24, tag='p%d:m' % i)
    lc.text(RX + 306, yy + 27, verdict + ('（token 32-47 起不同）' if mark == '✗' else ''),
            9.4, col, 'start', True, maxw=RW - 306 - 28, tag='p%d:v' % i)
vy = py + 3 * (PH + PGAP) + 4
lc.rect(RX + 16, vy, RW - 32, 52, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.2)
lc.text(RX + 30, vy + 21, '前缀一致 ⇔ 沿链哈希逐一相等', 9.8, lc.C_KV_S, 'start', True,
        maxw=RW - 60, tag='rp:v')
lc.text(RX + 30, vy + 40, '这正是下一站命中查找敢在第一个 miss 处停下的全部底气', 8.6, '#334155',
        'start', maxw=RW - 60, tag='rp:v2')
by = vy + 66
lc.rect(RX + 16, by, RW - 32, 84, '#f8fafc', GRAY, rx=6, sw=1.1, dash=True)
lc.text(RX + 30, by + 19, '断链实验：块 0 只改 1 个 token', 9.6, lc.C_TXT, 'start', True,
        maxw=RW - 60, tag='rp:bt')
lc.text(RX + 30, by + 38, '块 1 的 token 原封不动，但 h1 从 3c3c50f324ae', 8.6, '#334155',
        'start', maxw=RW - 60, tag='rp:bl1')
lc.text(RX + 30, by + 55, '变成 c6bd6310bdcd——parent 一变、后面全作废', 8.6, '#334155',
        'start', maxw=RW - 60, tag='rp:bl2')
lc.text(RX + 30, by + 72, '（链式传播：无需检查后块，结构上必然全变）', 8, lc.C_MUTE, 'start',
        maxw=RW - 60, tag='rp:bl3')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + LH + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不变量：hash_i 递归含 hash_{i-1}（雪崩性把父的全部信息拌进子）⇒ 第 i 块哈希是指纹前 i+1 块全部内容的章',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '⇒ 任一 hash_i miss ⇒ 对一切 j>i 的 hash_j 必 miss——断一处即停、无需回溯，是结构保证不是启发式',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for stroke, dash, tcol, name in [
        (lc.C_KV_S, False, lc.C_KV_S, '指纹链（哈希计算）'),
        (GREEN, False, GREEN, '相等 / 命中'),
        (lc.C_ABORT, False, lc.C_ABORT, '分叉 / 失效'),
        (GRAY, True, GRAY, '不满尾段（不入账）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=170, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, 'h* 十六进制为实跑值前 12 位（sha256 摘要 32 字节）；@N = 该指纹盖住的 token 上界',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/request.py:L249-L265（构造尾与 append 后 update_block_hashes 增量算）· '
        'vllm/v1/core/kv_cache_utils.py:L596-L628（hash_block_tokens：parent+tokens+extra_keys 进 sha256）· '
        'L99-L114（init_none_hash 种子，PYTHONHASHSEED 可播种）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（PYTHONHASHSEED=0）· 行号基线 vLLM v0.27.1', 8.2,
        lc.C_FAINT, 'start', maxw=500, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-chained-hash.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
