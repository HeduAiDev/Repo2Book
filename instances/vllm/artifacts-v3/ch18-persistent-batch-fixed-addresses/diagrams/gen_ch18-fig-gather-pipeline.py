#!/usr/bin/env python3
"""ch18 机制图 6 · [2,5,3] 扁平收集四步向量算术（figure_spec ch18-fig-gather-pipeline，模板 tensor-flow）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）『执行臂中层』GPUModelRunner 框 center ③
拍片『_prepare_inputs · 收集装配』——即本章 L2 章图 center ③ 拍片（站 7『扁平收集』）
的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：三个请求的 [2,5,3] 份 token 账经过 np.repeat 展开→cumsum 折偏移→二维坐标编一维
（token_indices=pos+req_index·16）→一次 index_select，变成一列连续的 10 个 input_ids
——收集是 O(total) 的向量算子链，不是逐请求循环。

数字全部取自 figure_spec.numbers（traces/ch18_m06_gather.json observed/derived +
源码注释自带同一算例 gpu_model_runner.py:L1981-L1989）。坐标由常量/循环计算；文本全
esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 830
MX = 60
BXR = 1440

RC = {0: lc.C_KV_S, 1: lc.C_API_S, 2: lc.C_ZMQ_S}     # r1/r2/r3 身份色
RNAME = ['r1', 'r2', 'r3']

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'np.repeat 铺开、cumsum 折边、编成一维流水号、一次 index_select——收集是向量算子链',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '源码注释自带的 [2,5,3] 算例真跑（gpu_model_runner.py:L1981-L1989）：前缀命中的 r1 从 4 起、全新 prefill 的 r2 从 0 起、续块的 r3 从 7 起——同一公式，无 prefill/decode 相位分叉',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 拍片 ③ _prepare_inputs 收集装配 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

NSCH = [2, 5, 3]
REQ_IDX = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
QPOS = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
CU = [2, 7, 10]
NCOMP = [4, 4, 0, 0, 0, 0, 0, 7, 7, 7]
POS = [4, 5, 0, 1, 2, 3, 4, 7, 8, 9]
TOK_IDX = [4, 5, 16, 17, 18, 19, 20, 39, 40, 41]
GATHERED = [15, 16, 21, 22, 23, 24, 25, 24, 23, 22]

def lane_header(y, num, title, sub):
    lc.rect(MX, y, 30, 30, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.2)
    lc.text(MX + 15, y + 20, num, 11, lc.C_ENG_S, 'middle', True, tag='lh' + num)
    lc.text(MX + 40, y + 13, title, 10.5, lc.C_TXT, 'start', True, maxw=900, tag='lt' + num)
    lc.text(MX + 40, y + 27, sub, 8.2, lc.C_MUTE, 'start', maxw=900, tag='ls' + num)

# ---------------- 泳道 ①：num_scheduled [2,5,3] ----------------
L1_Y, L1_CY = 96, 134
lane_header(L1_Y, '①', 'num_scheduled_tokens = [2,5,3]',
            '调度器只发份数（ch10『批是 token 账单』在 worker 侧的镜像）——r1 续算尾 2 · r2 全新 prefill 5 · r3 续块 3')
tk_x, tk_h = 340, 34
for i, n in enumerate(NSCH):
    if i > 0:
        tk_x += NSCH[i - 1] * 46 + 14
    w = n * 46
    lc.rect(tk_x, L1_CY, w, tk_h, '#ffffff', RC[i], rx=6, sw=1.8)
    lc.text(tk_x + w / 2, L1_CY + 21, f'{RNAME[i]}:{n}', 11, RC[i], 'middle', True, maxw=w - 8,
            tag=f'tk1{i}')
t1_end = 340 + (NSCH[0] * 46 + 14) + (NSCH[1] * 46 + 14) + NSCH[2] * 46
lc.text(t1_end + 22, L1_CY + 21, 'total = 10（本拍收集量）', 9, lc.C_MUTE, 'start', maxw=240,
        tag='tot1')

# ① → ② 箭头
a1x = 400
lc.seg(a1x, L1_CY + 38, a1x, L1_CY + 56, lc.C_ENG_S, 1.8, 'std')
lc.text(a1x + 8, L1_CY + 52, 'np.repeat(arange(3), [2,5,3])', 8, lc.C_ENG_S, 'start', maxw=240,
        tag='a12')

# ---------------- 泳道 ②：展开条 + cu 刻度 ----------------
L2_Y, L2_CY = 196, 240
lane_header(L2_Y, '②', 'req_indices = [0,0,1,1,1,1,1,2,2,2]（排号） · cu = cumsum → [2,7,10]',
            '每请求份数展开成排号序列；cumsum 折出区间边界——展开序与区间序逐元素对齐')
bx0, bw_, bh_ = 340, 46, 34
for k in range(10):
    x = bx0 + k * (bw_ + 4)
    lc.rect(x, L2_CY, bw_, bh_, '#ffffff', RC[REQ_IDX[k]], rx=5, sw=1.6)
    lc.text(x + bw_ / 2, L2_CY + 21, str(REQ_IDX[k]), 11, RC[REQ_IDX[k]], 'middle', True,
            maxw=bw_ - 6, tag=f'ri{k}')
for v in CU:
    x = bx0 + v * (bw_ + 4) - 2
    lc.seg(x, L2_CY - 6, x, L2_CY + bh_ + 6, '#0f172a', 1.4, dash=True)
bar_end = bx0 + 10 * (bw_ + 4)
lc.text(bar_end + 16, L2_CY + 21, '← 10 格三色段 = 三个请求的份', 8.6, lc.C_MUTE, 'start',
        maxw=230, tag='bar:note')
lc.text(bx0, L2_CY + bh_ + 16, 'cu 边界：2 / 7 / 10（竖虚线）——query_pos = arange(10) − 区间头 = [0,1,0,1,2,3,4,0,1,2]',
        8.2, lc.C_MUTE, 'start', maxw=820, tag='cu:note')

# ② → ③ 箭头
lc.seg(a1x, L2_CY + 56, a1x, L2_CY + 74, lc.C_ENG_S, 1.8, 'std')
lc.text(a1x + 8, L2_CY + 70, 'query_pos ∈ [0, n_r)', 8, lc.C_ENG_S, 'start', maxw=200, tag='a23')

# ---------------- 泳道 ③：positions 竖式 ----------------
L3_Y, L3_CY = 324, 362
lane_header(L3_Y, '③', 'positions = num_computed[req] + query_pos',
            '前缀命中的 r1 从 4 起、全新的 r2 从 0 起、续块的 r3 从 7 起——一条公式吃下所有相位')
vx0, vw_, vh_ = 340, 46, 26
rows3 = [('query_pos', QPOS, '#334155', False),
         ('+ num_computed', NCOMP, '#334155', False),
         ('= positions', POS, lc.C_TXT, True)]
for ri, (name, vals, colr, bold) in enumerate(rows3):
    ry = L3_CY + ri * (vh_ + 8)
    lc.text(vx0 - 10, ry + vh_ / 2 + 3, name, 9, colr, 'end', True, maxw=180, tag='r3n' + str(ri))
    for k in range(10):
        x = vx0 + k * (vw_ + 4)
        lc.rect(x, ry, vw_, vh_, '#ffffff', RC[REQ_IDX[k]], rx=4, sw=1.5 if bold else 1.0)
        lc.text(x + vw_ / 2, ry + vh_ / 2 + 3.5, str(vals[k]), 10 if bold else 9.5,
                RC[REQ_IDX[k]] if bold else colr, 'middle', bold, maxw=vw_ - 6, tag=f'p{ri}{k}')
ann_x = vx0 + 10 * (vw_ + 4) + 24
for i, txt in enumerate(['r1 起 4（前缀命中 4，续算尾 chunk）', 'r2 起 0（全新首拍全量 prefill）',
                         'r3 起 7（chunked prefill 续块）']):
    lc.text(ann_x, L3_CY + 12 + i * 26, txt, 8.6, RC[i], 'start', True, maxw=BXR - ann_x,
            tag=f'ann{i}')

# ③ → ④ 箭头
lc.seg(a1x, L3_CY + 3 * (vh_ + 8) - 2, a1x, L3_CY + 3 * (vh_ + 8) + 16, lc.C_ENG_S, 1.8, 'std')
lc.text(a1x + 8, L3_CY + 3 * (vh_ + 8) + 12, 'token_indices = pos + 排号×16', 8, lc.C_ENG_S,
        'start', maxw=240, tag='a34')

# ---------------- 泳道 ④：token_ids_cpu 3×16 网格 + 一维编址 + 收集 ----------------
L4_Y, GRID_Y = 498, 536
lane_header(L4_Y, '④', 'token_ids_cpu 扁平视图（3 行×16 列）→ 二维坐标编一维 → 一次 index_select',
            'token_indices[i] = 行×16 + 列——(行,列) 折成连续流水号，从扁平大数组一次 gather 出全部 input_ids')
GRID_X = 340
GW, GH = 32, 30
ROW_KEYS = [('r1@0', 0, [11, 12, 13, 14, 15, 16]), ('r2@1', 1, [21, 22, 23, 24, 25]),
            ('r3@2', 2, [31, 30, 29, 28, 27, 26, 25, 24, 23, 22])]
HL_COLS = {0: [4, 5], 1: [0, 1, 2, 3, 4], 2: [7, 8, 9]}     # 收集高亮列（= token_indices 所指）
# 列标尺
lc.text(GRID_X + 38, GRID_Y - 6, '列→', 7.6, '#94a3b8', 'start', maxw=44, tag='gcol')
for c in range(0, 16, 4):
    lc.text(GRID_X + 44 + c * (GW + 2) + GW / 2, GRID_Y - 6, str(c), 7.4, '#94a3b8',
            'middle', maxw=30, tag=f'gc{c}')
for r, (key, ri, toks) in enumerate(ROW_KEYS):
    ry = GRID_Y + 4 + r * (GH + 8)
    lc.text(GRID_X, ry + GH / 2 + 3, f'{key}·{r * 16}', 7.8, RC[ri], 'end', True, maxw=88,
            tag='gk' + key)
    for c in range(16):
        x = GRID_X + 44 + c * (GW + 2)
        if c in HL_COLS[ri]:
            lc.rect(x, ry, GW, GH, RC[ri], RC[ri], rx=3.5, sw=0)
            lc.text(x + GW / 2, ry + GH / 2 + 3, str(toks[HL_COLS[ri].index(c)]), 8.4,
                    '#ffffff', 'middle', True, maxw=GW - 3, tag=f'g{r}{c}')
        elif c < len(toks):
            lc.rect(x, ry, GW, GH, '#ffffff', RC[ri], rx=3.5, sw=1.0)
            lc.text(x + GW / 2, ry + GH / 2 + 3, str(toks[c]), 8.4, RC[ri], 'middle',
                    maxw=GW - 3, tag=f'g{r}{c}')
        else:
            lc.rect(x, ry, GW, GH, '#f8fafc', '#e2e8f0', rx=3.5, sw=0.5)
GRID_BOT = GRID_Y + 4 + 3 * (GH + 8)
GRID_R = GRID_X + 44 + 16 * (GW + 2)
lc.text(GRID_X + 44, GRID_BOT + 14,
        '高亮格 = token_indices 所指（r1 列 4-5 · r2 列 0-4 · r3 列 7-9）；行首「·0/·16/·32」= 该行扁平起点',
        8.2, lc.C_MUTE, 'start', maxw=660, tag='grid:note')

# 右侧：token_indices 条 + 逐对下引 + gathered 条
TI_X, TI_Y = 1050, GRID_Y
GA_Y = TI_Y + 76
lc.text(TI_X - 12, TI_Y + 22, 'token_indices', 9, lc.C_TXT, 'end', True, maxw=104, tag='ti:t')
lc.text(TI_X - 12, GA_Y + 22, 'index_select', 9, lc.C_ENG_S, 'end', True, maxw=104, tag='ga:t')
for k in range(10):
    x = TI_X + k * 38
    lc.rect(x, TI_Y + 6, 34, 26, '#ffffff', RC[REQ_IDX[k]], rx=4, sw=1.4)
    lc.text(x + 17, TI_Y + 23, str(TOK_IDX[k]), 9.3, RC[REQ_IDX[k]], 'middle', True, maxw=32,
            tag=f'ti{k}')
    lc.seg(x + 17, TI_Y + 34, x + 17, GA_Y + 4, '#cbd5e1', 1.0, 'std')
    lc.rect(x, GA_Y + 8, 34, 26, RC[REQ_IDX[k]], RC[REQ_IDX[k]], rx=4, sw=0)
    lc.text(x + 17, GA_Y + 25, str(GATHERED[k]), 9.3, '#ffffff', 'middle', True, maxw=32,
            tag=f'ga{k}')
lc.text(TI_X, GA_Y + 50, 'gathered input_ids[0:10] → 前缀上载（固定地址缓冲）', 8.4,
        lc.C_GPU_S, 'start', True, maxw=390, tag='ga:note')
lc.text(TI_X, GA_Y + 66, '值可手验：flat[4]=15 · flat[16]=21 · flat[39]=24 …', 8, lc.C_MUTE,
        'start', maxw=390, tag='ga:chk')
gy = GRID_Y + 4 + 1.5 * (GH + 8)
lc.seg(GRID_R + 4, gy, TI_X - 14, gy, lc.C_GPU_S, 2.2, 'std')
lc.text((GRID_R + TI_X) / 2, gy - 8, 'flatten() 后一次收齐', 8.2, lc.C_GPU_S, 'middle', True,
        maxw=140, tag='big:a')

# ---------------- 底部：qsl + logits_indices + NOTE 引文 + 页脚 ----------------
BT_Y = GRID_BOT + 34
qx = MX + 100
lc.text(MX, BT_Y, '装配副产物：', 9.5, lc.C_TXT, 'start', True, maxw=90, tag='bt:t')
lc.text(qx, BT_Y, 'query_start_loc=[0,2,7,10,10]（CU 偏移 + 尾部 pad 非递减——FlashAttention 类 kernel 的要求，L2073-L2078）',
        8.6, lc.C_MUTE, 'start', maxw=680, tag='bt:qsl')
lc.text(qx, BT_Y + 17, 'logits_indices = qsl[1:]−1 = [1,6,9]——每请求末 token 即采样位（接 ch17 compute_logits 切片）',
        8.6, lc.C_MUTE, 'start', maxw=680, tag='bt:li')

QT_Y = BT_Y + 34
lc.rect(MX, QT_Y, 700, 52, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, QT_Y + 17, '算子选型自述 · NOTE(woosuk)（gpu_model_runner.py:L2016-L2018）', 9,
        lc.C_TXT, 'start', True, maxw=660, tag='qt:t')
lc.text(MX + 14, QT_Y + 35, '「We use torch.index_select instead of np.take here because torch.index_select is much faster than np.take for large tensors.」',
        8.2, '#334155', 'start', maxw=672, tag='qt:q')
SX = MX + 716
lc.text(SX, QT_Y + 14, '全程 = np.repeat + cumsum + arange + 一次 index_select：O(total) 的常数次向量算子',
        9, lc.C_GPU_S, 'start', True, maxw=BXR - SX, tag='sum:1')
lc.text(SX, QT_Y + 32, '——没有逐请求 Python 循环（对 v0『逐请求组批』的结构性替代）。', 9,
        lc.C_GPU_S, 'start', True, maxw=BXR - SX, tag='sum:2')

FT_Y = QT_Y + 66
lc.text(MX, FT_Y, '逐字锚 vllm/v1/worker/gpu_model_runner.py:L1743-L1767（_get_cumsum_and_arange）· L1977-L2024（收集主段）· L1981-L1989（注释算例）· L2016-L2018（index_select NOTE）· L2073-L2078（qsl pad）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 16, '全部读数取自精简版 companion host 实测的 3 请求单拍实录（num_scheduled [2,5,3]）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch18-fig-gather-pipeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
