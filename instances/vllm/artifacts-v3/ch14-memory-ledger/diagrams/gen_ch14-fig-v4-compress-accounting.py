#!/usr/bin/env python3
"""ch14 机制图 · V4 压缩记账双视角(figure_spec ch14-fig-v4-compress-accounting,模板 layout)

站 5(混合组化)的放大:压缩块的记账口径——页宽 census(上一图)与写入路径的接缝。
管理面按 256-token 块发块号(块表/分配/前缀哈希全按 256 走),物理面每块只有
256//r 行(c4=64、c128=2);槽位映射 is_valid=(pos+1)%r==0、压缩位=pos//r、
不满置 -1 缝合两套坐标系;融合 kernel 双保险早退。

数字全部取自 figure_spec.numbers(provenance = kv_cache_interface.py:L403-L405
storage_block_size + 官方回归测试 test_indexer_deepseek_v4_slot_mapping.py
L63-L65/L94-L96/L100-L118 + fused_compress_quant_cache.py:L157-L165 +
platforms/interface.py:L628-L633)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
MGMT_S = lc.C_API_S            # 管理面 = 蓝(调度/账本侧口径)
PHY_S = lc.C_GPU_S             # 物理面 = 绿(GPU 侧货架)
AMBER = '#b45309'              # 关键现象(跨块界)高亮
BAD = '#dc2626'                # -1 / 早退

# ---------------- 标题区 ----------------
lc.text(MX, 34, '压缩记账双视角:管理面按 256-token 块发块号,物理面每块只有 256//r 行',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '块表 / 分配 / 前缀哈希全按 256 走;压缩货架 c4=64 行、c128=2 行——槽位映射把两套坐标系缝合,'
                '不满 r 的位置开 -1 空单,kernel 双保险保证 -1 永不落盘',
        10.5, lc.C_MUTE, 'start', maxw=1100, tag='subtitle')
_ch = 'L0 放大 · KV 账本列 · 站 5 写入路径'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左主列:三段(管理面 → 槽位映射 → 物理面) ----------------
LX, LW = MX, 1010
PY0 = 92
H1 = 150          # 管理面
H2 = 190          # 槽位映射
H3 = 240          # 物理面
GAP = 26

# ---- 上:管理面 ----
lc.rect(LX, PY0, LW, H1, '#ffffff', MGMT_S, rx=9, sw=1.2)
lc.text(LX + 16, PY0 + 20, '① 管理面 · 块表第 k 项管 256 token——分配、前缀哈希、调度全按这个口径', 11.5,
        lc.C_TXT, 'start', True, maxw=LW - 32, tag='p1:t')
lc.text(LX + 16, PY0 + 37, '官方回归测试场景:block_size=256 · r=4 · num_computed=240 · query_len=40(块表 [5, 7])',
        8.8, lc.C_MUTE, 'start', maxw=LW - 32, tag='p1:s')
# 两个管理块条(40-token chunk 骑在两块上:240-255 在块 0,256-279 在块 1)
MB_Y, MB_H = PY0 + 82, 34
MB0_X, MB0_W = LX + 40, 400
MB1_X, MB1_W = MB0_X + MB0_W + 60, 400
lc.rect(MB0_X, MB_Y, MB0_W, MB_H, lc.C_API_F, MGMT_S, rx=5, sw=1.3)
lc.text(MB0_X + MB0_W / 2, MB_Y + 15, '管理块 0(token 0-255)', 9.2, MGMT_S, 'middle', True,
        maxw=MB0_W - 10, tag='mb0')
lc.text(MB0_X + MB0_W / 2, MB_Y + 28, '块表[0] → 物理块 5', 8.2, lc.C_MUTE, 'middle',
        maxw=MB0_W - 10, tag='mb0s')
lc.rect(MB1_X, MB_Y, MB1_W, MB_H, lc.C_API_F, MGMT_S, rx=5, sw=1.3)
lc.text(MB1_X + MB1_W / 2, MB_Y + 15, '管理块 1(token 256-511)', 9.2, MGMT_S, 'middle', True,
        maxw=MB1_W - 10, tag='mb1')
lc.text(MB1_X + MB1_W / 2, MB_Y + 28, '块表[1] → 物理块 7', 8.2, lc.C_MUTE, 'middle',
        maxw=MB1_W - 10, tag='mb1s')
# chunk 覆盖段:块 0 的尾 16 token + 块 1 的头 24 token(按 256:16 = 25:10 比例)
CH_W0 = MB0_W * 16 / 256
CH_W1 = MB1_W * 24 / 256
CH_Y, CH_H = MB_Y - 20, 12
lc.rect(MB0_X + MB0_W - CH_W0, CH_Y, CH_W0, CH_H, MGMT_S, MGMT_S, rx=2, sw=0.8)
lc.rect(MB1_X, CH_Y, CH_W1, CH_H, MGMT_S, MGMT_S, rx=2, sw=0.8)
lc.text(MB0_X + MB0_W - CH_W0 / 2, CH_Y - 8, '本拍 40 token(pos 240-279)', 8.4, MGMT_S, 'middle', True,
        maxw=220, tag='chklbl')
lc.seg(MB0_X + MB0_W - CH_W0 / 2, CH_Y + CH_H, MB0_X + MB0_W - CH_W0 / 2, MB_Y - 1, MGMT_S, 1.2,
       marker='dn')
lc.seg(MB1_X + CH_W1 / 2, CH_Y + CH_H, MB1_X + CH_W1 / 2, MB_Y - 1, MGMT_S, 1.2, marker='dn')

# ---- 中:槽位映射(40 格) ----
P2 = PY0 + H1 + GAP
lc.rect(LX, P2, LW, H2, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, P2 + 22, '② 槽位映射 · is_valid=(pos+1)%r==0 · 压缩位=pos//r · 不满置 -1', 11.5,
        lc.C_TXT, 'start', True, maxw=LW - 32, tag='p2:t')
lc.text(LX + 16, P2 + 40, '40 个位置逐一过筛:每凑满 4 个 token 恰一位有效(压缩位 60..69 连续 10 位),其余 30 位 -1',
        8.8, lc.C_MUTE, 'start', maxw=LW - 32, tag='p2:s')
# 40 格:8 组×5 格(4 token 格 + 1 有效格)?直接 40 格一行太密——排两行各 20 格
SL_X0, SL_Y0, SL_N, SL_W, SL_H, SL_GAP = LX + 60, P2 + 62, 20, 41, 34, 3
for half in range(2):
    y0 = SL_Y0 + half * (SL_H + 22)
    for k in range(SL_N):
        idx = half * SL_N + k                        # pos = 240 + idx
        pos = 240 + idx
        valid = (pos + 1) % 4 == 0
        x = SL_X0 + k * (SL_W + SL_GAP)
        if valid:
            cpos = pos // 4                          # 压缩位 60..69
            lc.rect(x, y0, SL_W, SL_H, lc.C_API_F, MGMT_S, rx=3, sw=1.3)
            lc.text(x + SL_W / 2, y0 + 14, str(pos), 8.0, MGMT_S, 'middle', True, maxw=SL_W - 2,
                    tag='sl:%d' % pos)
            lc.text(x + SL_W / 2, y0 + 26, str(cpos), 9.5, MGMT_S, 'middle', True, maxw=SL_W - 2,
                    tag='slc:%d' % pos)
        else:
            lc.rect(x, y0, SL_W, SL_H, '#ffffff', lc.C_MUTE, rx=3, sw=0.7)
            lc.text(x + SL_W / 2, y0 + 14, str(pos), 7.4, lc.C_FAINT, 'middle', maxw=SL_W - 2,
                    tag='sl:%d' % pos)
            lc.text(x + SL_W / 2, y0 + 26, '-1', 9.5, BAD, 'middle', True, maxw=SL_W - 2,
                    tag='slm:%d' % pos)
    if half == 0:
        lc.text(SL_X0 + SL_N * (SL_W + SL_GAP) + 4, y0 + 20, '→ 接下行', 8.0, lc.C_FAINT,
                'start', tag='sl:cont')
# 图例与合计
LEG_Y = SL_Y0 + 2 * SL_H + 22 + 22
lc.rect(SL_X0, LEG_Y - 11, 16, 12, lc.C_API_F, MGMT_S, rx=2, sw=1.1)
lc.text(SL_X0 + 22, LEG_Y, '上=token 位 pos · 下=压缩位(pos//4)', 8.2, lc.C_MUTE, 'start', maxw=300,
        tag='lg1')
lc.rect(SL_X0 + 320, LEG_Y - 11, 16, 12, '#ffffff', lc.C_MUTE, rx=2, sw=0.7)
lc.text(SL_X0 + 342, LEG_Y, '下=-1(不满 r 不写)', 8.2, lc.C_MUTE, 'start', maxw=240, tag='lg2')
lc.text(LX + LW - 16, LEG_Y, '10 有效 / 30 个 -1', 10, MGMT_S, 'end', True, maxw=200, tag='lg3')

# ---- 下:物理面(压缩货架) ----
P3 = P2 + H2 + GAP
lc.rect(LX, P3, LW, H3, '#ffffff', PHY_S, rx=9, sw=1.2)
lc.text(LX + 16, P3 + 22, '③ 物理面 · 压缩货架:c4 每块 64 行(256//4)、c128 每块 2 行(256//128)', 11.5,
        lc.C_TXT, 'start', True, maxw=LW - 32, tag='p3:t')
lc.text(LX + 16, P3 + 40, '压缩位 60..69 恰跨存储块界 64——前 4 位落块 5(380-383)、后 6 位落块 7(448-453)',
        8.8, AMBER, 'start', True, maxw=LW - 32, tag='p3:s')
# 两个物理块:各画 64 行的「行带」(细横条示意,亮出有效行)
PB_Y, PB_H = P3 + 58, 120
PB0_X, PB0_W = LX + 40, 440
PB1_X, PB1_W = PB0_X + PB0_W + 60, 440
for bx, bw, blk, rows_lbl in [(PB0_X, PB0_W, 5, '块 5 · 64 行'), (PB1_X, PB1_W, 7, '块 7 · 64 行')]:
    lc.rect(bx, PB_Y, bw, PB_H, lc.C_GPU_F, PHY_S, rx=5, sw=1.3)
    lc.text(bx + 14, PB_Y + 18, rows_lbl, 9.6, PHY_S, 'start', True, maxw=bw - 28,
            tag='pb:%d' % blk)
# 行刻度:画 16 条细横线示意 64 行(每 4 行一条),有效行段亮出
ROW_N = 64
def row_rect(bx, bw, r0, r1, fill, stroke, tag):
    """行段 [r0, r1) 占物理块内行带"""
    RB_Y, RB_H = PB_Y + 30, 74
    rh = RB_H / ROW_N
    lc.rect(bx + 40 + (bw - 60) * r0 / ROW_N, RB_Y + rh * r0, (bw - 60) * (r1 - r0) / ROW_N,
            rh * (r1 - r0) + 0.5, fill, stroke, rx=1, sw=0.8)
# 块 5:行 60-63 有效(slot 380-383);块 7:行 0-5 有效(slot 448-453)
row_rect(PB0_X, PB0_W, 60, 64, MGMT_S, MGMT_S, 'r5')
row_rect(PB1_X, PB1_W, 0, 6, MGMT_S, MGMT_S, 'r7')
# 行带底图(浅色 64 行刻度)
for bx, bw in [(PB0_X, PB0_W), (PB1_X, PB1_W)]:
    RB_Y, RB_H = PB_Y + 30, 74
    for r in range(0, ROW_N, 4):
        y = RB_Y + RB_H * r / ROW_N
        lc.seg(bx + 40, y, bx + bw - 20, y, '#d1fae5', 0.7)
    lc.rect(bx + 40, RB_Y, bw - 60, RB_H, 'none', PHY_S, rx=2, sw=0.8)
# slot 标注
lc.text(PB0_X + PB0_W - 20, PB_Y + PB_H - 8, 'slot 380-383(行 60-63)', 8.6, MGMT_S, 'end', True,
        maxw=240, tag='sl5')
lc.text(PB1_X + PB1_W - 20, PB_Y + PB_H - 8, 'slot 448-453(行 0-5)', 8.6, MGMT_S, 'end', True,
        maxw=240, tag='sl7')
# 跨块界标注(两块之间)
MID_X = (PB0_X + PB0_W + PB1_X) / 2
lc.text(MID_X, PB_Y + 60, '跨存储', 8.6, AMBER, 'middle', True, maxw=52, tag='x1')
lc.text(MID_X, PB_Y + 74, '块界 64', 8.6, AMBER, 'middle', True, maxw=52, tag='x2')
# 底行小结
lc.text(LX + 16, PB_Y + PB_H + 22, '『压缩发生在 block_size 维』——管理面块数不变、物理行数被 r 除掉:'
        '本例 2 个管理块 × 256 token,c4 口径下每块 64 行(c128 则每块 2 行)', 8.8, '#334155', 'start',
        maxw=LW - 32, tag='p3:n')

# 段间连接箭头(管理面 → 槽位映射 → 物理面,走左外侧)
CONN_X = LX - 18
lc.parrow([(CONN_X, PY0 + H1 / 2), (CONN_X, P2 + H2 / 2)], MGMT_S, 1.6, marker='std')
lc.parrow([(CONN_X, P2 + H2 / 2 + 40), (CONN_X, P3 + H3 / 2)], PHY_S, 1.6, marker='std')

# ---------------- 右侧栏:双保险 + 换算 + 来源 ----------------
RX = LX + LW + 24
RW = BXR - RX
RB0 = PY0
RB_H1 = 196
lc.rect(RX, RB0, RW, RB_H1, '#ffffff', BAD, rx=9, sw=1.3)
lc.text(RX + 14, RB0 + 22, '④ 写入 kernel 双保险', 11.5, BAD, 'start', True, maxw=RW - 28, tag='r1:t')
lc.text(RX + 14, RB0 + 40, 'fused_compress_quant_cache.py:L157-L165(Triton,逐字)', 8.2, lc.C_MUTE,
        'start', maxw=RW - 28, tag='r1:s')
for j, (code, note) in enumerate([
    ('if slot_id < 0: return', '槽位映射的 -1 到这里歇手'),
    ('if (position + 1) % r != 0: return', '非凑整位到这里歇手'),
]):
    by = RB0 + 58 + j * 62
    lc.rect(RX + 14, by, RW - 28, 26, '#fef2f2', BAD, rx=4, sw=1.0)
    lc.text(RX + 22, by + 17, code, 8.6, BAD, 'middle', True, maxw=RW - 44, tag='r1:c%d' % j)
    lc.text(RX + 22, by + 42, note, 8.2, lc.C_MUTE, 'start', maxw=RW - 44, tag='r1:n%d' % j)
lc.text(RX + 14, RB0 + RB_H1 - 10, '两道独立早退互为冗余保险:-1 永不落盘', 8.4, '#334155', 'start',
        maxw=RW - 28, tag='r1:f')

RB1 = RB0 + RB_H1 + 16
RB_H2 = 150
lc.rect(RX, RB1, RW, RB_H2, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 14, RB1 + 22, 'storage_block_size 换算', 11.5, lc.C_TXT, 'start', True, maxw=RW - 28,
        tag='r2:t')
lc.text(RX + 14, RB1 + 40, 'kv_cache_interface.py:L403-L405(逐字属性)', 8.2, lc.C_MUTE, 'start',
        maxw=RW - 28, tag='r2:s')
for j, (a, b) in enumerate([('c4', '256 // 4 = 64 行 / 块'), ('c128', '256 // 128 = 2 行 / 块')]):
    by = RB1 + 58 + j * 40
    lc.rect(RX + 14, by, 52, 26, lc.C_API_F if j == 0 else '#ffffff', MGMT_S, rx=4, sw=1.1)
    lc.text(RX + 40, by + 17, a, 9.6, MGMT_S, 'middle', True, tag='r2:a%d' % j)
    lc.text(RX + 76, by + 17, b, 9.6, lc.C_TXT, 'start', True, maxw=RW - 90, tag='r2:b%d' % j)

RB2 = RB1 + RB_H2 + 16
RB_H3 = 108
lc.rect(RX, RB2, RW, RB_H3, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(RX + 14, RB2 + 22, '块大小 256 来自后端偏好而非写死', 10.2, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='r3:t')
lc.text(RX + 14, RB2 + 42, 'platforms/interface.py:L628-L633:非用户指定时', 8.2, lc.C_MUTE, 'start',
        maxw=RW - 28, tag='r3:s1')
lc.text(RX + 14, RB2 + 58, '从 backend.get_preferred_block_size 取——', 8.2, lc.C_MUTE, 'start',
        maxw=RW - 28, tag='r3:s2')
lc.text(RX + 14, RB2 + 74, 'DeepseekSparseSWABackend 返回 256', 8.2, lc.C_MUTE, 'start',
        maxw=RW - 28, tag='r3:s3')
lc.text(RX + 14, RB2 + 92, '(sparse_swa.py:L119-L121)', 8.0, lc.C_FAINT, 'start', maxw=RW - 28,
        tag='r3:s4')

# ---------------- 页脚 ----------------
FY = P3 + H3 + 14
lc.text(MX, FY + 12, '期望槽位出自官方回归测试 tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py'
        '(L63-L65 场景 · L94-L96 断言 numel()==10 · L100-L118 期望张量;host 无 CUDA 未实跑,期望值已按公式逐位手推复核)'
        ' · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
H = int(FY + 12 + 18)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-v4-compress-accounting.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
