#!/usr/bin/env python3
"""ch18 机制图 4 · InputBatch 行式大网格 + 列式镜像双视图（figure_spec ch18-fig-inputbatch-layout，模板 layout）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）『执行臂中层』GPUModelRunner 框内的
InputBatch 容器——即本章 L2 章图 south『InputBatch · 固定全长内存布局』块本体的内存
视图展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：InputBatch 是『行式 R×L 大网格 + 列式 CPU 镜像』的双视图容器：一行=一个请求的
全长 token 缓冲（prompt 前缀+output 紧随，写回只前移游标），旁列各记
num_prompt_tokens/num_tokens_no_spec/num_computed_tokens_cpu——一切搬移只动活跃前缀、
一切布局按 max 预留。

数字全部取自 figure_spec.numbers（traces/ch18_m02_reconcile.json beats[4].after_sample
（拍 5 行内容与三列读数）+ traces/ch18_m03_slots.json（陈旧尾巴铁证）+
gpu_input_batch.py L129-L141/L366-L388 + arg_utils.py:L2545-L2562）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 596
MX = 60
BXR = 1440

RC = {'r1': lc.C_KV_S, 'r3': lc.C_ZMQ_S}

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一行一请求的全长储物格 + 一列一个小账本——InputBatch 的两张视图按 max 预留',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'token_ids_cpu 是 [max_num_reqs, max_model_len] 的 int32 大网格（本玩具刻度 4×32）：add_request 写 prompt 前缀、写回只前移 num_tokens_no_spec 游标；旁列三组横条是列式 CPU 镜像',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 南行 InputBatch 固定全长内存布局 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：行式大网格（4 行 × 32 列，拍 5 采样后真实行内容） ----------------
GR_X, GR_Y = MX, 116
N_ROW, N_COL = 4, 32
CELL_W, CELL_H, GAP = 21, 32, 1.2
ROW_LBL_W = 46
GRID_W = N_COL * (CELL_W + GAP)                 # 710
CELL_X0 = GR_X + ROW_LBL_W + 6

# 拍 5 采样后（beats[4].after_sample.token_rows_active_prefix_plus3）：
# r1 活跃 7=[101,102,11,12,13,14,15] 之后全 0（从未写过）；r3 活跃 4=[301,302,31,30] 之后 [22]（r2 让位残留）
ROWS = [
    ('r1@0', 'r1', [101, 102, 11, 12, 13, 14, 15], 7, [], False),
    ('r3@1', 'r3', [301, 302, 31, 30], 4, [22], True),
    (None, None, [], 0, [], False),
    (None, None, [], 0, [], False),
]

lc.text(GR_X, GR_Y - 8, '行式大网格 token_ids_cpu[4, 32] · int32 · pin_memory=False',
        10, lc.C_TXT, 'start', True, maxw=GRID_W, tag='gr:t')

for r in range(N_ROW):
    ry = GR_Y + 10 + r * (CELL_H + 12)
    key, rid, toks, n_active, stale, has_stale = ROWS[r]
    if key:
        lc.text(GR_X, ry + CELL_H / 2 + 3, key, 9, RC[rid], 'end', True, maxw=ROW_LBL_W - 6,
                tag='row:' + key)
    else:
        lc.text(GR_X, ry + CELL_H / 2 + 3, '空行', 8.2, '#cbd5e1', 'end', maxw=ROW_LBL_W - 6,
                tag=f'row:empty{r}')
    for c in range(N_COL):
        x = CELL_X0 + c * (CELL_W + GAP)
        if rid and c < n_active:
            is_prompt = c < 2                     # 两请求 prompt 都是 2 token
            solid = is_prompt
            lc.rect(x, ry, CELL_W, CELL_H, RC[rid] if solid else '#ffffff', RC[rid],
                    rx=2.5, sw=1.1)
            lc.text(x + CELL_W / 2, ry + CELL_H / 2 + 3, str(toks[c]), 7.4,
                    '#ffffff' if solid else RC[rid], 'middle', maxw=CELL_W - 1, tag=f'c{r}{c}')
        elif rid and has_stale and c == n_active:
            lc.rect(x, ry, CELL_W, CELL_H, '#f1f5f9', '#94a3b8', rx=2.5, sw=1.0)
            lc.text(x + CELL_W / 2, ry + CELL_H / 2 + 3, str(stale[0]), 7.4, '#94a3b8',
                    'middle', maxw=CELL_W - 1, tag=f'st{r}')
        else:
            lc.rect(x, ry, CELL_W, CELL_H, '#f8fafc', '#e2e8f0', rx=2.5, sw=0.5)
    # 块界白线（16 列一处，block_size=16 → 每请求至多 2 块）
    bx = CELL_X0 + 16 * (CELL_W + GAP)
    lc.seg(bx, ry - 3, bx, ry + CELL_H + 3, '#ffffff', 2.4)

GRID_BOT = GR_Y + 10 + N_ROW * (CELL_H + 12)
# 游标刻线（r1 行活跃边界，col 6/7 之间）：写回只前移 num_tokens_no_spec
cur_x = CELL_X0 + 7 * (CELL_W + GAP)
r1_ry = GR_Y + 10
lc.seg(cur_x, r1_ry - 4, cur_x, r1_ry + CELL_H + 4, lc.C_GPU_S, 2.0, dash=True)
lc.text(GR_X, GRID_BOT + 14,
        '深色格 = prompt 前缀（写行起点）· 白底描边 = output 紧随其后 · 灰 = 陈旧尾巴/未写（不清理也不读）',
        8.2, lc.C_MUTE, 'start', maxw=GRID_W + 30, tag='gr:leg')
lc.text(GR_X, GRID_BOT + 30,
        '绿刻线 = num_tokens_no_spec 游标（r1=7：prompt 2 + output 5）——写回只前移游标，格子不搬；',
        8.2, lc.C_GPU_S, 'start', maxw=GRID_W + 30, tag='gr:cur')
lc.text(GR_X, GRID_BOT + 46,
        'r3 行活跃前缀外的 22 是 r2 让位时留下的陈旧尾巴；白线 = 块界（block_size=16）。',
        8.2, lc.C_MUTE, 'start', maxw=GRID_W + 30, tag='gr:stale')

# ---------------- 右：列式 CPU 镜像（三组横条 + 索引，拍 5 读数） ----------------
CL_X, CL_Y = 850, 116
CL_W = 560
BAR_X0, UNIT, BAR_H = CL_X + 152, 11.5, 12      # 量程 = max_model_len 32 → 368px
lc.text(CL_X, CL_Y - 8, '列式 CPU 镜像（同一批请求的第二张视图，拍 5 读数）', 10, lc.C_TXT,
        'start', True, maxw=CL_W, tag='cl:t')
# 量程刻度
for v in [0, 8, 16, 24, 32]:
    x = BAR_X0 + v * UNIT
    lc.seg(x, CL_Y + 4, x, CL_Y + 8 + 3 * 74 - 6, '#e2e8f0', 0.9)
    lc.text(x, CL_Y + 8 + 3 * 74 + 6, str(v), 7, '#cbd5e1', 'middle', maxw=20, tag=f'ax{v}')

COLS = [
    ('num_prompt_tokens', [2, 2], 'prompt 长度（add_request 写）'),
    ('num_tokens_no_spec', [7, 4], '已写 token 数（写回前移）'),
    ('num_computed_tokens_cpu', [5, 0], '已算进度（差量覆盖）'),
]
for ci, (name, vals, sub) in enumerate(COLS):
    gy = CL_Y + 8 + ci * 74
    lc.text(CL_X, gy + 8, name, 9.2, lc.C_TXT, 'start', True, maxw=148, tag='cl:n' + str(ci))
    lc.text(CL_X, gy + 22, sub, 7.5, lc.C_MUTE, 'start', maxw=148, tag='cl:s' + str(ci))
    for ri in range(2):
        rid = ['r1', 'r3'][ri]
        v = vals[ri]
        by = gy + 32 + ri * 17
        lc.text(BAR_X0 - 26, by + 9, rid, 8, RC[rid], 'end', True, maxw=22, tag=f'cl:{ci}:{rid}')
        if v > 0:
            lc.rect(BAR_X0, by, v * UNIT, BAR_H, RC[rid], RC[rid], rx=2, sw=0)
        lc.text(BAR_X0 + v * UNIT + 6, by + 9, str(v), 8.6, RC[rid], 'start', True, maxw=24,
                tag=f'cl:{ci}:{ri}:v')

# req_id_to_index 索引注
ix_y = CL_Y + 8 + 3 * 74 + 18
lc.rect(CL_X, ix_y, CL_W - 70, 52, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(CL_X + 12, ix_y + 17, 'req_id_to_index 索引：{r1: 0, r3: 1}——打洞/搬移后重建；',
        8.6, lc.C_TXT, 'start', maxw=CL_W - 94, tag='ix:1')
lc.text(CL_X + 12, ix_y + 35, '另有全套采样参数列（每请求一行快照）随行镜像。', 8.6, lc.C_TXT,
        'start', maxw=CL_W - 94, tag='ix:2')

# ---------------- 底部：真实刻度换算 + 陈旧尾巴铁证 ----------------
BT_Y = GRID_BOT + 128
lc.rect(MX, BT_Y, 700, 104, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 14, BT_Y + 18, '真实刻度：内存按 max 预留', 10, lc.C_TXT, 'start', True, maxw=400, tag='bt:t')
lc.text(MX + 14, BT_Y + 37, 'token_ids_cpu = max_num_seqs × max_model_len × 4B（int32）——服务端默认 max_num_seqs=256（小卡）/1024（≥70GiB 非 A100）',
        8.4, '#334155', 'start', maxw=672, tag='bt:l1')
lc.text(MX + 14, BT_Y + 54, '按 max_model_len=8192：256×8192×4B ≈ 8MiB ～ 1024×8192×4B ≈ 32MiB 的 CPU 常驻',
        8.4, '#334155', 'start', maxw=672, tag='bt:l2')
lc.text(MX + 14, BT_Y + 72, '「TODO(woosuk): This buffer could be too large if max_model_len is big.」——gpu_input_batch.py:L129-L131 自注',
        8.2, '#475569', 'start', maxw=672, tag='bt:q')
lc.text(MX + 14, BT_Y + 90, '两因子锚：vllm/engine/arg_utils.py:L2545-L2562（max_num_seqs 默认值）',
        8, lc.C_MUTE, 'start', maxw=672, tag='bt:qa')

SB_X = MX + 716
lc.rect(SB_X, BT_Y, BXR - SB_X, 104, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(SB_X + 14, BT_Y + 18, '陈旧尾巴无害的铁证（8×16 直驱实录）', 10, lc.C_TXT, 'start', True,
        maxw=430, tag='sb:t')
lc.text(SB_X + 14, BT_Y + 37, 'd 滑入洞 1 后 row1=[40,21,22]——b 的尾巴 [21,22] 原地残留；', 8.4,
        '#334155', 'start', maxw=BXR - SB_X - 28, tag='sb:l1')
lc.text(SB_X + 14, BT_Y + 54, 'e 复用洞 0 后 row0=[50,51,12,13]——a 的尾巴 [12,13] 同理；', 8.4,
        '#334155', 'start', maxw=BXR - SB_X - 28, tag='sb:l2')
lc.text(SB_X + 14, BT_Y + 72, '读边界由活跃前缀（active_prefix_len 1 / 2）界定——', 8.4, '#334155',
        'start', maxw=BXR - SB_X - 28, tag='sb:l3')
lc.text(SB_X + 14, BT_Y + 90, '格子里的旧数据不清理也不读，下次写入自然覆盖。', 8.4, '#334155',
        'start', maxw=BXR - SB_X - 28, tag='sb:l4')

# 页脚
FT_Y = BT_Y + 124
lc.text(MX, FT_Y, '逐字锚 vllm/v1/worker/gpu_input_batch.py:L92-L172（InputBatch 布局）· L129-L141（token_ids_cpu + TODO）· L366-L388（add_request 写行）· vllm/engine/arg_utils.py:L2545-L2562（max_num_seqs 默认）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 16, '网格行内容/三列读数取自精简版 companion host 实测的拍 5 采样后记录（玩具刻度 4×32）· 陈旧尾巴读数取自 8×16 直驱实录 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch18-fig-inputbatch-layout.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
