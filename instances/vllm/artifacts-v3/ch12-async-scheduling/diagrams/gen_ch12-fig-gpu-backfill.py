#!/usr/bin/env python3
"""ch12 机制图 6 · 下一拍输入的三岔口（figure_spec ch12-fig-gpu-backfill，模板 tensor-flow）

放大自 L0 执行臂列（gpu_column）内部的输入准备段——即本章 L2 章图 south
『worker · 下一拍 GPU 回填』框的三岔口展开：_prepare_input_ids 对 input_ids.gpu
的三条写入路径。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：下一拍输入的三岔口——正常拍整段 H2D、批次未变单 slice 直拷、变过按 index
scatter；重排例 CPU 侧 [100,101,102] + GPU 侧 prev[7,8,9] → GPU 结果 [9,101,7]，
证明采样 token 从 GPU 直达 GPU。

版式：CPU 源在左、prev（GPU）源在右；三条泳道全宽收窄，右侧留走廊——CPU 从左进、
GPU 回填从右进（结果条两头收货，一眼分清来源）。

数字全部取自 figure_spec.numbers（路径1 CPU [1,2]→copy_to_gpu→GPU [1,2] prev=None；
路径2 prev_positions=[0,1]、CPU 故意清零 [0,0]→GPU [7,8] 单 slice；路径3
prev_positions=[2,-1,0]、CPU [100,101,102]→GPU [9,101,7]，位置0 从 prev 行2、
位置2 从 prev 行0、位置1 新请求保留 CPU；判定 cu_num_tokens=[1,2]/[1,2,3]、
common_indices_match 且 max_flattened==N−1 才走单 slice）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 912
MX, BXR = 60, 1440
LANE_W = 1100          # 泳道右缘 1160，右侧走廊 1180..1440 给 GPU 来源连线
C_GPU_TXT = lc.C_GPU_S

# ---------------- 标题区 ----------------
lc.text(MX, 34, '下一拍输入的三岔口：上一拍采出的 token 从 GPU 直达 GPU',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '重排例 CPU 侧 [100,101,102] + GPU 侧 prev[7,8,9] → 结果 [9,101,7]——CPU 侧清零/放占位值的'
        '证伪设计说明回填只能来自 GPU 的 prev', 10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 下一拍 GPU 回填框 · L0：执行臂列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 两个源（顶部）：CPU 左 / prev（GPU）右 ----------------
SRC_Y, SRC_H = 92, 64
CPU_SRC_X, CPU_SRC_W = MX, 640
PREV_SRC_X, PREV_SRC_W = 800, 640
lc.rect(CPU_SRC_X, SRC_Y, CPU_SRC_W, SRC_H, '#ffffff', lc.C_API_S, rx=7, sw=1.6)
lc.text(CPU_SRC_X + 14, SRC_Y + 22, 'CPU 侧 input_ids 账本（左进）', 10, lc.C_API_S, 'start',
        True, maxw=CPU_SRC_W - 30, tag='cpusrc:t')
lc.text(CPU_SRC_X + 14, SRC_Y + 42, '新请求 / prefill 窗口的 token 从这里走 H2D —— decode 回填不走它',
        8.6, '#334155', 'start', maxw=CPU_SRC_W - 28, tag='cpusrc:sub')
lc.rect(PREV_SRC_X, SRC_Y, PREV_SRC_W, SRC_H, lc.C_GPU_F, C_GPU_TXT, rx=7, sw=1.6)
lc.text(PREV_SRC_X + 14, SRC_Y + 22, 'prev_sampled_token_ids（GPU 张量 · 右进）', 10, C_GPU_TXT,
        'start', True, maxw=PREV_SRC_W - 30, tag='prev:t')
lc.text(PREV_SRC_X + 14, SRC_Y + 42, '行0 = 7　行1 = 8　行2 = 9 —— 上拍缓存的采样结果（上一图快路）',
        8.6, '#334155', 'start', maxw=PREV_SRC_W - 28, tag='prev:sub')

# ---------------- 三条泳道几何 ----------------
LY = [186, 354, 536]
LH = [150, 164, 230]
BADGE_W = 176
CX0 = MX + BADGE_W + 24          # 泳道内容起点 x=260
CELL_W, CELL_H, CELL_GAP = 62, 30, 6


def cells(x, y, vals, kinds, hot_idx=None, fs=10):
    for k, (v, kind) in enumerate(zip(vals, kinds)):
        cx = x + k * (CELL_W + CELL_GAP)
        if kind == 'gpu':
            lc.rect(cx, y, CELL_W, CELL_H, lc.C_GPU_F, C_GPU_TXT, rx=4, sw=1.3)
            lc.text(cx + CELL_W / 2, y + 20, v, fs, C_GPU_TXT, 'middle', True, tag='cell%d' % k)
        elif kind == 'cpu':
            hot = (hot_idx == k)
            lc.rect(cx, y, CELL_W, CELL_H, '#eff6ff' if hot else '#ffffff', lc.C_API_S, rx=4,
                    sw=1.5 if hot else 1.2)
            lc.text(cx + CELL_W / 2, y + 20, v, fs, lc.C_API_S, 'middle', True, tag='cell%d' % k)
        else:
            lc.rect(cx, y, CELL_W, CELL_H, '#ffffff', '#cbd5e1', rx=4, sw=1.0, dash=True)
            lc.text(cx + CELL_W / 2, y + 20, v, fs, '#94a3b8', 'middle', tag='cell%d' % k)


def lane_badge(ly, lh, no, name, sub, stroke, fill):
    lc.rect(MX, ly, LANE_W, lh, '#ffffff', stroke, rx=7, sw=1.4)
    lc.rect(MX, ly, BADGE_W, lh, fill, stroke, rx=7, sw=1.4)
    lc.rect(MX, ly + 14, BADGE_W, lh - 28, fill, 'none', rx=0, sw=0)
    lc.text(MX + BADGE_W / 2, ly + 30, '路径 ' + no, 11, stroke, 'middle', True, tag='bd' + no)
    lc.text(MX + BADGE_W / 2, ly + 50, name, 9.5, stroke, 'middle', True, tag='bdn' + no)
    for j, s in enumerate(sub):
        lc.text(MX + BADGE_W / 2, ly + 70 + j * 15, s, 8.4, lc.C_MUTE, 'middle', maxw=BADGE_W - 12,
                tag='bds%s%d' % (no, j))


# ---- 路径 1：正常拍 ----
lane_badge(LY[0], LH[0], '1', '正常拍', ['prev=None', '整段 H2D'], lc.C_MUTE, '#f8fafc')
lc.text(CX0, LY[0] + 24, 'GPU 上没有上一拍可回填 → token 全来自 CPU 侧账本（copy_to_gpu，L1801-L1807）',
        9, lc.C_TXT, 'start', True, maxw=860, tag='p1:d')
lc.text(CX0, LY[0] + 62, 'CPU 侧', 8.6, lc.C_API_S, 'start', maxw=60, tag='p1:cl')
cells(CX0 + 62, LY[0] + 46, ['1', '2'], ['cpu', 'cpu'])
_x = CX0 + 62 + 2 * (CELL_W + CELL_GAP)
lc.seg(_x + 8, LY[0] + 61, _x + 66, LY[0] + 61, lc.C_API_S, 2.0, 'std')
lc.text(_x + 74, LY[0] + 64, 'copy_to_gpu', 8.6, lc.C_API_S, 'start', maxw=90, tag='p1:op')
cells(_x + 168, LY[0] + 46, ['1', '2'], ['gpu', 'gpu'])
lc.text(_x + 168 + 2 * (CELL_W + CELL_GAP) + 14, LY[0] + 64, 'GPU 结果 = CPU 账本的镜像上桥',
        8.4, lc.C_MUTE, 'start', maxw=300, tag='p1:note')

# ---- 路径 2：common-case 单 slice ----
lane_badge(LY[1], LH[1], '2', '批次未变', ['common-case', 'decode 稳态', '单 slice 直拷'],
          C_GPU_TXT, lc.C_GPU_F)
lc.text(CX0, LY[1] + 24, '判定：prev_positions=[0,1] · cu_num_tokens=[1,2] · common_indices_match 且 max_flattened==N−1',
        9, lc.C_TXT, 'start', True, maxw=860, tag='p2:d')
lc.text(CX0, LY[1] + 62, 'CPU 侧（故意清零）', 8.6, lc.C_API_S, 'start', maxw=130, tag='p2:cl')
cells(CX0 + 140, LY[1] + 46, ['0', '0'], ['cpu', 'cpu'])
_x = CX0 + 140 + 2 * (CELL_W + CELL_GAP)
lc.seg(_x + 8, LY[1] + 61, _x + 66, LY[1] + 61, C_GPU_TXT, 2.0, 'std')
lc.text(_x + 74, LY[1] + 64, '单 slice 直拷', 8.6, C_GPU_TXT, 'start', maxw=90, tag='p2:op')
cells(_x + 168, LY[1] + 46, ['7', '8'], ['gpu', 'gpu'])
lc.text(_x + 168 + 2 * (CELL_W + CELL_GAP) + 14, LY[1] + 58,
        'CPU 全 0 而结果 [7,8]——token 只能来自 prev（L1868-L1877）', 8.4, lc.C_MUTE,
        'start', maxw=330, tag='p2:note')
lc.text(_x + 168 + 2 * (CELL_W + CELL_GAP) + 14, LY[1] + 76,
        '一条 copy 指令完成全部回填，无索引张量上传', 8.4, lc.C_MUTE, 'start', maxw=330,
        tag='p2:note2')
lc.text(CX0, LY[1] + 108, '稳态每拍都走这条：input_ids.gpu[:N] ← prev_sampled_token_ids[:N,0]',
        8.4, lc.C_MUTE, 'start', maxw=700, tag='p2:steady')

# ---- 路径 3：scatter ----
lane_badge(LY[2], LH[2], '3', '批次重排', ['scatter 兜底', '进出批 / 重排'], lc.C_ENG_S, lc.C_ENG_F)
lc.text(CX0, LY[2] + 24, '判定：prev_positions=[2, −1, 0] · cu_num_tokens=[1,2,3]'
        '（req-2 提前 + 新请求 req-x 插中间 + req-1 离场）', 9, lc.C_TXT, 'start', True,
        maxw=880, tag='p3:d')
lc.text(CX0, LY[2] + 60, 'CPU 侧（故意放占位值）', 8.6, lc.C_API_S, 'start', maxw=160, tag='p3:cl')
cells(CX0 + 168, LY[2] + 44, ['100', '101', '102'], ['cpu', 'cpu', 'cpu'], hot_idx=1, fs=9.5)
_x = CX0 + 168 + 3 * (CELL_W + CELL_GAP)
lc.seg(_x + 8, LY[2] + 59, _x + 66, LY[2] + 59, lc.C_ENG_S, 2.0, 'std')
lc.text(_x + 74, LY[2] + 62, 'scatter', 8.6, lc.C_ENG_S, 'start', maxw=70, tag='p3:op')
cells(_x + 152, LY[2] + 44, ['9', '101', '7'], ['gpu', 'cpu', 'gpu'], hot_idx=1, fs=9.5)
lc.text(_x + 152 - 4, LY[2] + 30, 'GPU 结果', 8.6, C_GPU_TXT, 'end', maxw=80, tag='p3:rl')
MAP = [
    ('位置0 = 9', '← prev 行2 回填（GPU→GPU）'),
    ('位置1 = 101', '← CPU 保留（新请求 req-x：prev_positions=−1 免采）'),
    ('位置2 = 7', '← prev 行0 回填（GPU→GPU）'),
]
for i, (a, b) in enumerate(MAP):
    yy = LY[2] + 96 + i * 16
    lc.text(CX0, yy, a, 8.6, C_GPU_TXT if i != 1 else lc.C_API_S, 'start', True, maxw=110,
            tag='mp%d' % i)
    lc.text(CX0 + 118, yy, b, 8.5, '#334155', 'start', maxw=640, tag='mpn%d' % i)
lc.text(CX0, LY[2] + 156, '槽位查表 _compute_prev_positions（L1769-L1782）：当前批每个槽位查 prev_req_id_to_index——'
        '上拍在批的请求得旧槽号、新请求得 −1（scatter 循环里 −1 直接 continue）',
        8.4, lc.C_MUTE, 'start', maxw=880, tag='p3:mapping')
lc.text(CX0, LY[2] + 180, '证伪设计：位置1 的 101 只可能来自 CPU——其余位置的 9/7 只可能来自 prev（真 token 全程没经过 CPU）',
        8.8, lc.C_ENG_S, 'start', True, maxw=880, tag='p3:proof')
lc.text(CX0, LY[2] + 202, '代价：2 个索引张量上传 + 1 次 scatter kernel——仍是 O(1) 次 kernel launch',
        8.4, lc.C_MUTE, 'start', maxw=700, tag='p3:cost')

# ---------------- 走廊：prev（GPU）→ 路径2/路径3 右缘 ----------------
PREV_BOT = SRC_Y + SRC_H
CONN2_X, CONN3_X = 1270, 1345
lc.parrow([(CONN2_X, PREV_BOT + 2), (CONN2_X, LY[1] + 82), (MX + LANE_W + 6, LY[1] + 82)],
          C_GPU_TXT, 2.0, 'std')
lc.parrow([(CONN3_X, PREV_BOT + 2), (CONN3_X, LY[2] + 82), (MX + LANE_W + 6, LY[2] + 82)],
          C_GPU_TXT, 2.0, 'std')
lc.text((CONN2_X + MX + LANE_W) / 2 + 6, LY[1] + 74, 'prev[:N,0] 直拷', 8.2, C_GPU_TXT,
        'middle', maxw=100, tag='conn2')
lc.text((CONN3_X + MX + LANE_W) / 2 + 6, LY[2] + 74, 'src=prev[prev_indices,0]', 8.2, C_GPU_TXT,
        'middle', maxw=170, tag='conn3')
lc.text((CONN2_X + CONN3_X) / 2, PREV_BOT + 16, 'GPU 侧来源（右进）', 8.2, C_GPU_TXT, 'middle',
        maxw=120, tag='connlbl')
# CPU 源 → 路径1（左进）
lc.seg(CPU_SRC_X + 380, PREV_BOT + 2, CPU_SRC_X + 380, LY[0] - 4, lc.C_API_S, 2.0, 'std')
lc.text(CPU_SRC_X + 388, PREV_BOT + 14, '整段 H2D', 8.2, lc.C_API_S, 'start', maxw=90, tag='conn1')

# ---------------- 结论横幅 ----------------
BN_Y = LY[2] + LH[2] + 16
lc.rect(MX, BN_Y, 1380, 38, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.4)
lc.text(MX + 690, BN_Y + 16.5, 'kernel launch 次数 O(1) 于批内请求规模：common-case 1 条 slice；重排 2 个索引张量上传 + 1 次 scatter kernel 处理 3 行',
        9.6, '#166534', 'middle', True, maxw=1360, tag='banner1')
lc.text(MX + 690, BN_Y + 30.5, '没有 per-token 的 CPU 循环——索引张量异步上传、回填全部在 GPU 端完成（真 token 全程没经过 CPU）',
        8.8, '#166534', 'middle', maxw=1360, tag='banner2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BN_Y + 64
lx = MX
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_GPU_F, C_GPU_TXT, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, 'GPU 侧值（来自 prev / 结果）', 8.5, lc.C_TXT, 'start', maxw=240, tag='leg:gpu')
lx += 28 + lc.tw('GPU 侧值（来自 prev / 结果）', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 22, 13, '#ffffff', lc.C_API_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, 'CPU 侧值（账本 / 保留）', 8.5, lc.C_TXT, 'start', maxw=220, tag='leg:cpu')
lx += 28 + lc.tw('CPU 侧值（账本 / 保留）', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 22, 13, '#eff6ff', lc.C_API_S, rx=3, sw=1.4)
lc.text(lx + 28, LEG_Y + 1, '新请求位（高亮）', 8.5, lc.C_TXT, 'start', maxw=160, tag='leg:hot')
lx += 28 + lc.tw('新请求位（高亮）', 8.5) + 16
lc.seg(lx + 4, LEG_Y - 3, lx + 34, LEG_Y - 3, C_GPU_TXT, 2.0, 'std')
lc.text(lx + 42, LEG_Y + 1, 'GPU→GPU 回填（右进）', 8.5, lc.C_TXT, 'start', maxw=170, tag='leg:arrow')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/worker/gpu_model_runner.py:L1769-L1782（_compute_prev_positions）· '
        'L1801-L1807（copy_to_gpu）· L1868-L1877（common-case 单 slice 判定与直拷）· L1878-L1891（scatter）· '
        '数字取自配套精简版 host 实跑（三路径对照 + 证伪设计）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-gpu-backfill.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
