#!/usr/bin/env python3
"""ch14 机制图 · 适配地图：生命周期八站 × 模型侧/框架侧（figure_spec ch14-fig-adaptation-map，模板 layout）

新图（社区空白，research/community-narrative-figures.json P7 手法：生命周期流水线为骨架、
每站上下两行——上行「模型侧要实现什么」/下行「框架侧提供什么」），回答用户点名的问题
「一个新模型要适配什么」。四档适配深度（零/申报/管理器/后端）用绿色四档色深标出，
各钉一个 pin 实例（通用 / RSWA / SWA / V4）；与 ch14-fig-hybrid-theory-model-v2 推论④互指。

横轴 8 站：spec 申报 → 收集 → 分发四路 → 分组 → 布局 → 分配 → 绑定 → 寻址。
数字全部取自 figure_spec.numbers（attention.py / rswa_attention.py / kv_cache_utils.py /
gpu_model_runner.py / sparse_swa.py 等 pin 锚）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印乱码防

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与四档色深（绿系，干预越深色越深） ----------------
W, MX, BXR = 1500, 44, 1456
L0_F, L0_S, L0_T = '#f0fdf4', '#86efac', '#166534'      # 零档
L1_F, L1_S, L1_T = '#dcfce7', '#4ade80', '#14532d'      # 申报档
L2_F, L2_S, L2_T = '#86efac', '#16a34a', '#14532d'      # 管理器档
L3_F, L3_S, L3_T = '#16a34a', '#14532d', '#ffffff'      # 后端档
AMBER = '#b45309'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '接入新模型要动哪里：生命周期八站，上行模型侧、下行框架侧', 16.5, lc.C_TXT,
        'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '上行 = 模型侧要实现什么（绿越深、要写的越多：多数模型全程零代码）；下行 = 框架侧提供什么'
                '（申报一次、全链托管）——四档深度：通用 / RSWA / SWA / V4 各钉一位',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='sub1')
lc.text(MX, 76, '读图：先沿横轴走一遍八站（初始化执行序），再看上行哪里离开「零代码」底色——离开的格子就是新模型要写的东西',
        9.0, lc.C_MUTE, 'start', maxw=1330, tag='sub2')
_ch = 'L2 回指 · 站 5（组化 + 布局）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 站牌带（横轴骨架） ----------------
LBL_W = 46                                     # 左侧行标签列宽
SX0 = MX + LBL_W + 8                           # 站区左缘
N_ST = 8
ST_GAP = 16
ST_W = (BXR - SX0 - (N_ST - 1) * ST_GAP) / N_ST
STN_Y, STN_H = 92, 34
STATIONS = ['① spec 申报', '② 收集', '③ 分发四路', '④ 分组',
            '⑤ 布局', '⑥ 分配', '⑦ 绑定', '⑧ 寻址']
st_cx = []
for i, nm in enumerate(STATIONS):
    sx = SX0 + i * (ST_W + ST_GAP)
    st_cx.append(sx + ST_W / 2)
    lc.rect(sx, STN_Y, ST_W, STN_H, lc.C_BADGE_F, lc.C_ENG_S, rx=6, sw=1.2)
    lc.text(sx + ST_W / 2, STN_Y + 21.5, nm, 9.4, lc.C_BEAT_T, 'middle', True, maxw=ST_W - 8,
            tag='stn:%d' % i)
    if i < N_ST - 1:                            # 站间小箭头（执行序）
        lc.seg(sx + ST_W + 2, STN_Y + STN_H / 2, sx + ST_W + ST_GAP - 2, STN_Y + STN_H / 2,
               lc.C_ENG_S, 1.4, 'std')

# ---------------- 两行几何 ----------------
M_Y, M_H = 138, 176                             # 模型侧行
F_Y, F_H = 326, 170                             # 框架侧行


def vlabel(x, y0, h, chars, color):
    """左侧行标签：竖排逐字。"""
    n = len(chars)
    lh = 17
    y = y0 + (h - (n - 1) * lh) / 2
    for ch_ in chars:
        lc.text(x, y, ch_, 9.5, color, 'middle', True, tag='vl:' + ch_)
        y += lh


vlabel(MX + LBL_W / 2, M_Y, M_H, '模型侧要实现', lc.C_GPU_S)
vlabel(MX + LBL_W / 2, F_Y, F_H, '框架侧提供', lc.C_KV_S)

# ---------------- 模型侧行：8 格（零代码底色 + 干预点加深） ----------------
mrows = [
    [('零档 · 通用 Attention', ['默认申报 · 模型零代码'], L0_F, L0_S, L0_T, 50),
     ('申报档 · RSWA', ['重写 get_kv_cache_spec', '整段 13 行', 'RSWASpec 只加 rswa_window',
                        '+ 注册表注册'], L1_F, L1_S, L1_T, None)],
    None, None, None, None, None,
    [('管理器档 · SWA', ['SlidingWindowManager', '窗外整块回收 + null 占位'], L2_F, L2_S, L2_T, 92),
     ('后端档 · V4', ['自研后端 preferred 256', '行布局 (nb,bs,584)', '+ 压缩槽位映射'],
      L3_F, L3_S, L3_T, None)],
]
for i, cell in enumerate(mrows):
    sx = SX0 + i * (ST_W + ST_GAP)
    if cell is None:                            # 零代码默认格（整行底色）
        lc.rect(sx, M_Y, ST_W, M_H, L0_F, L0_S, rx=6, sw=1.1)
        lc.text(sx + ST_W / 2, M_Y + M_H / 2 - 6, '零代码', 10.5, L0_T, 'middle', True,
                maxw=ST_W - 10, tag='mz:%d' % i)
        lc.text(sx + ST_W / 2, M_Y + M_H / 2 + 14, '框架全托管', 8.2, L0_T, 'middle',
                maxw=ST_W - 10, tag='mzs:%d' % i)
        continue
    subs, ycur = cell, M_Y
    for k, (head, lines, fl, st, tx, hgt) in enumerate(subs):
        hgt = hgt if hgt else (M_Y + M_H - ycur - 6)
        lc.rect(sx, ycur, ST_W, hgt, fl, st, rx=6, sw=1.3)
        pad = 18 if k == 0 and len(subs) > 1 else 10
        lc.text(sx + ST_W / 2, ycur + pad, head, 8.8, tx, 'middle', True, maxw=ST_W - 8,
                tag='mh:%d:%d' % (i, k))
        ly = ycur + pad + 16
        for ln in lines:
            lc.text(sx + ST_W / 2, ly, ln, 7.9, tx, 'middle', maxw=ST_W - 8,
                    tag='ml:%d:%d:%s' % (i, k, ln[:8]))
            ly += 14.5
        ycur += hgt + 6

# ---------------- 框架侧行：8 格（白底） ----------------
frows = [
    ('默认申报三分支', ['encoder-only → None', '配滑窗 → SlidingWindowSpec', '其余 → FullAttentionSpec']),
    ('收集三分支', ['EC 专职整体返 {}', 'cross-layer 共享层跳过', '逐层自报 get_kv_cache_spec']),
    ('分发四路', ['uniform 一组', 'uniform-type 一组', 'DSV4 → group_and_unify', '一般均衡拆分']),
    ('分组', ['同类型装桶', '+ 均衡拆分']),
    ('布局', ['offset 折叠', '+ 池声明']),
    ('分配', ['全模型一个块池', 'packed Allocate once', '202 条声明全部别名', '同一 slab']),
    ('绑定', ['视图切片四步链', 'view(-1,stride)', '[:,off:off+page]', '.view(dtype).view(shape)',
              '+ bind_kv_cache 双写']),
    ('寻址', ['每组一张块表', 'kernel 块细分', '（核内 64 对齐）']),
]
for i, (head, lines) in enumerate(frows):
    sx = SX0 + i * (ST_W + ST_GAP)
    lc.rect(sx, F_Y, ST_W, F_H, '#ffffff', lc.C_KV_S, rx=6, sw=1.2)
    lc.text(sx + ST_W / 2, F_Y + 20, head, 8.8, lc.C_KV_S, 'middle', True, maxw=ST_W - 8,
            tag='fh:%d' % i)
    ly = F_Y + 40
    for ln in lines:
        lc.text(sx + ST_W / 2, ly, ln, 7.9, '#334155', 'middle', maxw=ST_W - 8,
                tag='fl:%d:%s' % (i, ln[:8]))
        ly += 15
# 上下格连接虚线（同站两侧对位）
for i in range(N_ST):
    lc.seg(st_cx[i], M_Y + M_H + 2, st_cx[i], F_Y - 2, lc.C_FAINT, 1.1, dash=True)

# ---------------- 底部：四档图例 + 判据 + 注脚 ----------------
GY = F_Y + F_H + 26
lx = MX
LEG = [('零档 · 通用', L0_F, L0_S, L0_T), ('申报档 · RSWA', L1_F, L1_S, L1_T),
       ('管理器档 · SWA', L2_F, L2_S, L2_T), ('后端档 · V4', L3_F, L3_S, L3_T)]
for nm, fl, st, tx in LEG:
    wlab = lc.tw(nm, 8.8) + 26
    lc.rect(lx, GY - 10, 20, 14, fl, st, rx=3, sw=1.3)
    lc.text(lx + 26, GY, nm, 8.8, lc.C_TXT, 'start', maxw=140, tag='lg:' + nm[:6])
    lx += wlab + 14
lc.text(lx, GY, '⇒ 加深判据：既有抽象表达不了（申报改不了管理行为、管理器改不了物理形状）', 8.8,
        AMBER, 'start', maxw=BXR - lx, tag='lg:rule')
lc.text(MX, GY + 20, 'DSV4 四档全占：四个申报面 + 复用滑窗管理器 + 三个自研后端——接入成本的极限样本；'
                     '「13 行」为 pin 实数行，行号与各站代码走读见正文',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ================= 装配输出 =================
H = int(GY + 42)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-adaptation-map.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
