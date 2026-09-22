#!/usr/bin/env python3
"""ch14 机制图 · 混合缓存理论模型 v2（figure_spec ch14-fig-hybrid-theory-model-v2，模板 layout）

整体替换 v1 ch14-fig-hybrid-theory-model（v1=三种形状关系三格阶梯；v2 扩为
两公理 → 四推论罩全领地）。一等约束：零 vLLM 标识符（「落地于站 N」的站号是
章节位置号、非代码符号；玩具数字体系 2B×16 / 6B×16 / 96%32 / 80%32 沿用 v1）。

上带两张公理卡：A1 层形状 = 每 token 字节 × 每块 token 数 = 页宽（字节格逐格画，
2B×16=32、6B×16=96）；A2 块号是唯一货币（三条货币规则）。
下带四推论 2×2：①一表一语义（A2）②页宽对齐必然（A1）③pack 必然（A1+A2）
④适配程度（A1+A2），每格标来源公理 + 玩具演算 + 右下角「落地于站 N」角标。

数字全部取自 figure_spec.numbers（theory-toy 玩具示教算术；落地站号见 spec
caption_draft 的【落地：站…】四条）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印乱码防

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与语义色 ----------------
W, MX, BXR = 1500, 60, 1440
CA, CA_F = lc.C_API_S, lc.C_API_F            # A 层形状 = 蓝
CB, CB_F = lc.C_KV_S, lc.C_KV_F              # B / C 层形状 = 青
C_DEAD = lc.C_ABORT                           # 死路 / 反例 = 红
AMBER = '#b45309'                            # 唯一解 / 落地角标 = 琥珀
AXIOM_S = '#7c2d12'                          # 公理卡描边 = 深棕（区别于推论卡的灰）

BPT_A, BPT_B, TOKS = 2, 6, 16                # 玩具：A 层 2 B/token、B 层 6 B/token、16 token/块
PAGE_A, PAGE_B, PAGE_C = 32, 96, 80          # 页宽：A 32、B 96、跨型例 C 80

# ---------------- 标题区 ----------------
lc.text(MX, 34, '混合缓存的公理化：两条公理推出四条推论', 16.5, lc.C_TXT,
        'start', True, maxw=1050, tag='title')
lc.text(MX, 58, 'A1 层形状 = 每 token 字节 × 每块 token 数 = 页宽（玩具：A 层 2 B×16 = 32、B 层 6 B×16 = 96）；'
                'A2 块号是池子的唯一货币——四条推论罩住混合缓存的全部领地',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='sub1')
lc.text(MX, 76, '读图：上排两公理；下排四推论，各标「由哪条公理推出」与右下角落地站号——①②③沿页宽代数走，'
                '④把镜头拉到模型接入侧',
        9.0, lc.C_MUTE, 'start', maxw=1330, tag='sub2')
_ch = '理论铺垫 · 纯形状代数（零代码）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 公理带：两卡并排 =================
PY0, PH = 94, 212
CARD_W = 678
A1X, A2X = MX, MX + CARD_W + 24

# ---- 公理一 A1：页宽（含字节格） ----
lc.rect(A1X, PY0, CARD_W, PH, '#ffffff', AXIOM_S, rx=9, sw=1.6)
lc.rect(A1X + 8, PY0 + 30, 30, 24, '#ffffff', AXIOM_S, rx=4, sw=1.2)
lc.text(A1X + 23, PY0 + 46.5, 'A1', 11, AXIOM_S, 'middle', True, tag='a1:badge')
lc.text(A1X + 56, PY0 + 24, '公理一 · 层形状 → 页宽：页宽 = 每 token 字节 × 每块 token 数', 11.5,
        lc.C_TXT, 'start', True, maxw=CARD_W - 70, tag='a1:t')

CS = 4.4                                      # 每字节像素宽
BAR_X, BAR_H = A1X + 60, 18
rows = [('A 层', BPT_A, CA, CA_F, PAGE_A),
        ('B 层', BPT_B, CB, CB_F, PAGE_B)]
for i, (name, bpt, cs_, cf_, page) in enumerate(rows):
    ry = PY0 + 52 + i * 58
    lc.text(A1X + 56, ry + 13, name, 9.6, cs_, 'start', True, maxw=40, tag='a1:' + name)
    for b in range(page):                      # 字节格：每格 1 B
        lc.rect(BAR_X + b * CS, ry, CS - 0.7, BAR_H, cf_, cs_, rx=0.8, sw=0.5)
    for t in range(TOKS + 1):                  # token 刻度
        tx = BAR_X + t * bpt * CS
        lc.seg(tx, ry + BAR_H + 1, tx, ry + BAR_H + 5, cs_, 0.9)
    lx = BAR_X + page * CS + 12
    lw_ = A1X + CARD_W - 14 - lx
    lc.text(lx, ry + 8, f'页宽 {page} B', 9.6, cs_, 'start', True, maxw=lw_, tag='a1:pw' + name)
    lc.text(lx, ry + 22, f'（{TOKS} token × {bpt} B/token）', 8.4, lc.C_MUTE, 'start', maxw=lw_,
            tag='a1:calc' + name)
lc.text(A1X + 56, PY0 + PH - 16, '同是 16 token/块，每 token 字节不同 → 页宽不同——这就是「混合」的根源',
        8.8, '#334155', 'start', maxw=CARD_W - 70, tag='a1:n')

# ---- 公理二 A2：块号是唯一货币（三条规则） ----
lc.rect(A2X, PY0, CARD_W, PH, '#ffffff', AXIOM_S, rx=9, sw=1.6)
lc.rect(A2X + 8, PY0 + 30, 30, 24, '#ffffff', AXIOM_S, rx=4, sw=1.2)
lc.text(A2X + 23, PY0 + 46.5, 'A2', 11, AXIOM_S, 'middle', True, tag='a2:badge')
lc.text(A2X + 56, PY0 + 24, '公理二 · 块号是池子的唯一货币', 11.5, lc.C_TXT, 'start', True,
        maxw=CARD_W - 70, tag='a2:t')
# 块槽一排（等宽块号）
SL_W, SL_H, SL_GAP = 86, 36, 10
sx0 = A2X + 56
for j in range(5):
    lc.rect(sx0 + j * (SL_W + SL_GAP), PY0 + 46, SL_W, SL_H, '#ffffff', lc.C_MUTE, rx=5, sw=1.2)
    lc.text(sx0 + j * (SL_W + SL_GAP) + SL_W / 2, PY0 + 68, f'块 {j}', 9.4, lc.C_TXT, 'middle', True,
            maxw=SL_W - 6, tag='a2:s%d' % j)
lc.text(sx0 + 5 * (SL_W + SL_GAP) + 6, PY0 + 68, '… 共 N 块（每块等宽）', 9.0, lc.C_MUTE, 'start',
        maxw=A2X + CARD_W - 14 - (sx0 + 5 * (SL_W + SL_GAP) + 6), tag='a2:ell')
RULES = ['① 一个块号同一时刻只归一组',
         '② 块 = 分配 / 驱逐 / 释放的决策单位——块元数据不认识层',
         '③ 块表按组发、组内层共用一张表——一个块号在组内映射到每层各一页']
for j, note in enumerate(RULES):
    lc.text(A2X + 56, PY0 + 110 + j * 32, note, 9.2, '#334155' if j < 2 else lc.C_TXT, 'start',
            maxw=CARD_W - 70, tag='a2:r%d' % j)
lc.text(A2X + 56, PY0 + PH - 16, '池子只会数块号、要求每块等宽——「宽」的归一问题全部由此而来',
        8.8, lc.C_MUTE, 'start', maxw=CARD_W - 70, tag='a2:n')

# ================= 过渡带：两公理 ⇒ 四推论 =================
TR_Y = PY0 + PH + 22
lc.seg(MX, TR_Y, 700, TR_Y, lc.C_FAINT, 1.1, dash=True)
lc.seg(800, TR_Y, BXR, TR_Y, lc.C_FAINT, 1.1, dash=True)
lc.text(750, TR_Y + 3.5, '两条公理 ⇒ 四条推论', 10.5, AXIOM_S, 'middle', True, tag='tr')
INFX = [MX + CARD_W / 2, MX + CARD_W + 24 + CARD_W / 2]
for ix in INFX:
    lc.seg(ix, TR_Y + 8, ix, TR_Y + 22, lc.C_MUTE, 1.6, 'std')

# ================= 推论带：2×2 四格 =================
LY0 = TR_Y + 30
CELL_H, ROW_GAP = 176, 14
CCX = [MX, MX + CARD_W + 24]                   # 两列左缘

INFERENCE = [('推论① · 一表一语义', '由 A2'),
             ('推论② · 页宽对齐必然', '由 A1'),
             ('推论③ · pack 必然', '由 A1 + A2'),
             ('推论④ · 适配程度', '由 A1 + A2')]
for k, (t, src) in enumerate(INFERENCE):
    cx = CCX[k % 2]
    cy = LY0 + (k // 2) * (CELL_H + ROW_GAP)
    lc.rect(cx, cy, CARD_W, CELL_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
    lc.text(cx + 14, cy + 22, t, 11.2, lc.C_TXT, 'start', True, maxw=CARD_W - 150, tag='inf:%d' % k)
    vw = lc.tw(src, 8.4, True) + 12
    lc.rect(cx + CARD_W - vw - 10, cy + 7, vw, 18, '#fff7ed', AXIOM_S, rx=9, sw=1.0)
    lc.text(cx + CARD_W - vw / 2 - 10, cy + 20, src, 8.4, AXIOM_S, 'middle', True, maxw=vw - 4,
            tag='inf:src%d' % k)

# 落地角标（右下，四条取自 spec caption_draft【落地：…】）
LAND = ['落地于站 11/12 · 块表与回收', '落地于站 5 · 页统一',
        '落地于站 5 · packed 布局', '落地于站 4/5/8/12 · 适配协议']
for k, lab in enumerate(LAND):
    cx = CCX[k % 2]
    cy = LY0 + (k // 2) * (CELL_H + ROW_GAP)
    lc.text(cx + CARD_W - 12, cy + CELL_H - 10, lab, 8.4, AMBER, 'end', True,
            maxw=CARD_W - 24, tag='land:%d' % k)

# ---- ① 一表一语义：左文右示意 ----
c0x, c0y = CCX[0], LY0
L1 = ['块整借整还 ⇒ 共用一张表的层，保留语义必须同一',
      '滑窗层混进全历史组：窗外整块回收时，全历史层的账一起被砍 ✗',
      '故同组形状 + 语义全等（含滑窗）——构造上焊死，语义混不进来']
for j, s in enumerate(L1):
    lc.text(c0x + 14, c0y + 44 + j * 20, s, 8.7, '#334155', 'start', maxw=396, tag='inf1:l%d' % j)
DIA_X, DIA_W = c0x + 420, CARD_W - 420 - 14
lc.rect(DIA_X, c0y + 38, DIA_W, 96, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
lc.text(DIA_X + DIA_W / 2, c0y + 54, '一个组 · 一张块表', 8.8, lc.C_TXT, 'middle', True,
        maxw=DIA_W - 10, tag='inf1:dia')
BK_W = (DIA_W - 24 - 3 * 6) / 4
for j in range(4):
    lc.rect(DIA_X + 12 + j * (BK_W + 6), c0y + 62, BK_W, 20, '#eff6ff', lc.C_API_S, rx=3, sw=1.0)
    lc.text(DIA_X + 12 + j * (BK_W + 6) + BK_W / 2, c0y + 76, f'块 {j}', 8.2, lc.C_API_S, 'middle',
            True, maxw=BK_W - 4, tag='inf1:bk%d' % j)
lc.text(DIA_X + DIA_W / 2, c0y + 102, '表只认块、不认层——语义必须先谈拢', 8.2, lc.C_MUTE, 'middle',
        maxw=DIA_W - 10, tag='inf1:dian')
lc.text(DIA_X + DIA_W / 2, c0y + 120, '全历史层 ✓ 同组 ｜ 滑窗层 ✗ 另立一组', 8.4, lc.C_TXT, 'middle',
        True, maxw=DIA_W - 10, tag='inf1:dial')

# ---- ② 页宽对齐必然：before → after 演算 ----
c1x, c1y = CCX[1], LY0
SC2 = 3.0                                     # ②③ 演算区每字节像素宽
b_h = 16
pairs = [('A 层', PAGE_A, CA, CA_F), ('B 层', PAGE_B, CB, CB_F)]
b1x = c1x + 64
for j, (name, page, cs_, cf_) in enumerate(pairs):
    ry = c1y + 46 + j * 26
    lc.text(b1x - 8, ry + 12, name, 8.4, cs_, 'end', maxw=52, tag='inf2:%s' % name)
    lc.rect(b1x, ry, page * SC2, b_h, cf_, cs_, rx=2, sw=1.1)
af_x = c1x + 400
for j, (name, page, cs_, cf_) in enumerate(pairs):
    ry = c1y + 46 + j * 26
    lc.rect(af_x, ry, PAGE_B * SC2, b_h, cf_, cs_, rx=2, sw=1.1)
    lc.text(af_x + PAGE_B * SC2 / 2, ry + 12, 'A′ 96 B' if j == 0 else 'B 96 B', 8.0, cs_,
            'middle', True, maxw=PAGE_B * SC2 - 4, tag='inf2:af%d' % j)
lc.text(c1x + 14, c1y + 102, '等宽的块装不下两种宽', 8.2, C_DEAD, 'start', True, maxw=160, tag='inf2:mis')
lc.parrow([(c1x + 240, c1y + 100), (c1x + 560, c1y + 100)], lc.C_MUTE, 1.4, 'std')
lc.text(c1x + 400, c1y + 114, '小层块长 16→48（×3）→ 页宽 32→96', 8.2, AMBER, 'middle', True,
        maxw=290, tag='inf2:xfm')
lc.text(c1x + 14, c1y + 134, '96 % 32 = 0（整除）→ 调大块长即可全池同宽', 9.0, lc.C_TXT, 'start', True,
        maxw=CARD_W - 28, tag='inf2:l1')
lc.text(c1x + 14, c1y + 152, '块在组间自由流动 ⇒ 租一块付的面积必须全池统一；每 token 成本不变——无损', 8.6,
        AXIOM_S, 'start', True, maxw=CARD_W - 28, tag='inf2:l2')

# ---- ③ pack 必然：三死路 + 重叠时间复用 ----
c2x, c2y = CCX[0], LY0 + CELL_H + ROW_GAP
b3x = c2x + 64
pairs3 = [('A 层', PAGE_A, CA, CA_F), ('C 层', PAGE_C, CB, CB_F)]
for j, (name, page, cs_, cf_) in enumerate(pairs3):
    ry = c2y + 46 + j * 26
    lc.text(b3x - 8, ry + 12, name, 8.4, cs_, 'end', maxw=52, tag='inf3:%s' % name)
    lc.rect(b3x, ry, page * SC2, b_h, cf_, cs_, rx=2, sw=1.1)
ov_x = b3x + PAGE_C * SC2 + 12
lc.rect(b3x - 3, c2y + 43, PAGE_A * SC2 + 6, 54, 'none', C_DEAD, rx=3, sw=1.3, dash=True)
lc.text(ov_x, c2y + 56, '同一块号内 A / C 条带物理重叠', 8.2, C_DEAD, 'start', True,
        maxw=c2x + CARD_W - 14 - ov_x, tag='inf3:ov')
lc.text(ov_x, c2y + 72, 't₁ 归 A 组 ｜ t₂ 归 C 组（时间互斥）', 8.0, lc.C_MUTE, 'start',
        maxw=c2x + CARD_W - 14 - ov_x, tag='inf3:t12')
dy = c2y + 104
dchip = ['调块长 ✗ 不整除', '垫 pad ✗ 浪费', '拆池 ✗ 浪费']
dx = c2x + 14
for dc in dchip:
    dw = lc.tw(dc, 8.2, True) + 12
    lc.rect(dx, dy, dw, 18, '#fef2f2', C_DEAD, rx=9, sw=1.0)
    lc.text(dx + dw / 2, dy + 12.5, dc, 8.2, C_DEAD, 'middle', True, maxw=dw - 4, tag='dd:' + dc[:6])
    dx += dw + 8
lc.text(c2x + CARD_W - 14, dy + 12.5, '80 % 32 = 16 ≠ 0', 8.6, C_DEAD, 'end', True, maxw=160,
        tag='inf3:mod')
lc.text(c2x + 14, dy + 40, '唯一无损解 = 重叠时间复用：块号同一时刻只借一组', 8.8, AMBER, 'start', True,
        maxw=CARD_W - 28, tag='inf3:l2')
lc.text(c2x + 14, dy + 57, '等宽约束换成「组内偏移互不冲突」——块号是租期、偏移是各自的房间', 8.6,
        AMBER, 'start', maxw=CARD_W - 28, tag='inf3:l3')

# ---- ④ 适配程度：四档递进色条 ----
c3x, c3y = CCX[1], LY0 + CELL_H + ROW_GAP
lc.text(c3x + 14, c3y + 44, '模型只申报「形状 + 语义」——申报离既有抽象越远，要过的档越多', 8.8, lc.C_TXT,
        'start', True, maxw=CARD_W - 28, tag='inf4:l0')
LADDER = [('零适配', '#f0fdf4', '#86efac', '#166534'),
          ('申报', '#dcfce7', '#4ade80', '#14532d'),
          ('管理器', '#86efac', '#16a34a', '#14532d'),
          ('后端', '#16a34a', '#14532d', '#ffffff')]
LD_Y, LD_H = c3y + 58, 34
gap4, aw4 = 26, (CARD_W - 28 - 3 * 26) / 4
for j, (nm, fl, st, tx) in enumerate(LADDER):
    lx4 = c3x + 14 + j * (aw4 + gap4)
    lc.rect(lx4, LD_Y, aw4, LD_H, fl, st, rx=5, sw=1.2)
    lc.text(lx4 + aw4 / 2, LD_Y + LD_H / 2 + 3.5, nm, 9.2, tx, 'middle', True, maxw=aw4 - 8,
            tag='inf4:ld%d' % j)
    if j < 3:
        lc.parrow([(lx4 + aw4 + 3, LD_Y + LD_H / 2), (lx4 + aw4 + gap4 - 3, LD_Y + LD_H / 2)],
                  lc.C_MUTE, 1.4, 'std')
lc.text(c3x + 14, c3y + 112, '零适配 = 只交配置文件 ｜ 申报 = 改账本 ｜ 管理器 = 改回收规则 ｜ 后端 = 改物理形状', 8.6,
        lc.C_TXT, 'start', True, maxw=CARD_W - 28, tag='inf4:l1')
lc.text(c3x + 14, c3y + 129, '每档解锁的能力恰好是下一档的默认前提——不存在跳档白做；档间边界 =', 8.6,
        '#334155', 'start', maxw=CARD_W - 28, tag='inf4:l2')
lc.text(c3x + 14, c3y + 146, '「既有抽象表达不了」：申报改不了管理行为，管理器改不了物理形状', 8.6,
        '#334155', 'start', maxw=CARD_W - 28, tag='inf4:l3')

# ================= 底带：图例 + 指针 =================
GY = LY0 + 2 * CELL_H + ROW_GAP + 24
lx = MX
for cs_, name in [(CA, 'A 层（2 B/token）'), (CB, 'B / C 层（B 层 6 B/token；C 层页宽 80 B）')]:
    lc.rect(lx, GY - 9, 18, 12, '#ffffff', cs_, rx=3, sw=1.5)
    lc.text(lx + 24, GY, name, 8.8, lc.C_TXT, 'start', maxw=340, tag='gl' + name[:6])
    lx += 24 + lc.tw(name, 8.8) + 20
lc.text(lx, GY, '红虚线框 = 同一块号内的物理重叠区', 8.8, lc.C_TXT, 'start', maxw=BXR - lx,
        tag='gl:ov')
lc.text(MX, GY + 19, '玩具例为 2 层示教算术（页宽 32 / 96 / 80，块长 16→48）——数字与代码侧页宽换算同源；'
                     '本图零代码标识符，代码形态见后文 packed 切片图；角标 = 站号（正文章节位置号）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ================= 装配输出 =================
H = int(GY + 40)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-hybrid-theory-model-v2.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
