#!/usr/bin/env python3
"""ch18 机制图 3 · slot 复用与压实九步演化（figure_spec ch18-fig-slot-lifecycle，模板 tiling）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）『执行臂中层』GPUModelRunner 框内的
InputBatch 结构——即本章 L2 章图 south『InputBatch · 固定全长内存布局』块的机制展开
（该块方法行已列 pop_removed/打洞/condense，本图把三段算法的 slot 级过程摊开）。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：slot 复用与压实是三段式算法：remove 打洞（数据不搬）→ add 弹最小洞优先复用
（洞尽才追加）→ condense 双指针把尾部活请求滑进前部空洞（只拷活跃前缀、尾部洞直接
截断），结束时 [0, num_reqs) 恒连续。

数字全部取自 figure_spec.numbers（traces/ch18_m03_slots.json 九步 _req_ids / removed /
rows + gpu_input_batch.py:L731 骨架）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 742
MX = 60
BXR = 1440

# 请求身份色（非架构色——图例注明）
RC = {'a': lc.C_KV_S, 'b': lc.C_API_S, 'c': lc.C_ZMQ_S,
      'd': lc.C_SAM_S, 'e': lc.C_ENG_S, 'f': lc.C_GPU_S}

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'remove 打洞不搬数据，add 弹最小洞，condense 把尾部滑进前部——slot 板三段式维护',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'InputBatch 直驱九步实录（max_num_reqs=8）：四请求占满 0-3 → 洞@1/@2 → d 从 3 滑入 1（只拷 1 个活跃 token）→ 洞@0 → e 复用 0 → f 洞尽追加 → condense 零成本早退',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 南行 InputBatch 块（打洞/复用/压实）· L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 九步数据（traces/ch18_m03_slots.json） ----------------
# 每列：(步号, 动作标签, 动作副行, _req_ids 快照, moved)
STEPS = [
    (2, 'add a', 'pop=None→slot0', ['a']),
    (5, 'add b/c/d', '批满 0-3', ['a', 'b', 'c', 'd']),
    (6, 'remove b', '打洞@1', ['a', None, 'c', 'd']),
    (7, 'remove c', '打洞@2', ['a', None, None, 'd']),
    (8, 'condense', 'd: 3→1', ['a', 'd']),
    (10, 'remove a', '打洞@0', [None, 'd']),
    (11, 'add e', '弹洞 0', ['e', 'd']),
    (12, 'add f', '追加@2', ['e', 'd', 'f']),
    (13, 'condense', '早退不变', ['e', 'd', 'f']),
]
TOKN = {'a': 10, 'b': 3, 'c': 1, 'd': 1, 'e': 2, 'f': 1}

N_SLOT = 8
CELL_W, CELL_H, GAP_X, GAP_Y = 64, 42, 10, 8
GRID_X, GRID_Y = 300, 152
ROW_HDR_X = GRID_X - 14

# 列头（步号 + 动作）
for ci, (sn, act, sub, _ids) in enumerate(STEPS):
    cx = GRID_X + ci * (CELL_W + GAP_X) + CELL_W / 2
    lc.text(cx, GRID_Y - 44, f'步 {sn}', 10, lc.C_ENG_S, 'middle', True, maxw=CELL_W + GAP_X,
            tag=f'ch{sn}')
    lc.text(cx, GRID_Y - 30, act, 8.6, lc.C_TXT, 'middle', True, maxw=CELL_W + GAP_X, tag=f'ca{sn}')
    lc.text(cx, GRID_Y - 18, sub, 7.6, lc.C_MUTE, 'middle', maxw=CELL_W + GAP_X, tag=f'cs{sn}')

# 行头（slot 0-7）
for r in range(N_SLOT):
    ry = GRID_Y + r * (CELL_H + GAP_Y)
    lc.text(ROW_HDR_X, ry + CELL_H / 2 + 3, f'slot{r}', 8.6, lc.C_MUTE, 'end', maxw=44,
            tag=f'rh{r}')

# 网格主体：三态
def draw_cell(ci, r, rid):
    x = GRID_X + ci * (CELL_W + GAP_X)
    y = GRID_Y + r * (CELL_H + GAP_Y)
    ids = STEPS[ci][3]
    if r < len(ids):
        if ids[r] is None:                     # 洞：数据不搬
            lc.rect(x, y, CELL_W, CELL_H, '#ffffff', '#94a3b8', rx=6, sw=1.2, dash=True)
            lc.text(x + CELL_W / 2, y + CELL_H / 2 + 3, '洞', 9, '#94a3b8', 'middle', True,
                    maxw=CELL_W - 8, tag=f'hole{ci}{r}')
        else:                                  # 活请求
            rid = ids[r]
            lc.rect(x, y, CELL_W, CELL_H, '#ffffff', RC[rid], rx=6, sw=1.8)
            lc.text(x + CELL_W / 2, y + 19, rid, 12, RC[rid], 'middle', True, maxw=CELL_W - 8,
                    tag=f'act{ci}{r}')
            lc.text(x + CELL_W / 2, y + 35, f'{TOKN[rid]} tok', 7.6, '#334155', 'middle',
                    maxw=CELL_W - 8, tag=f'tn{ci}{r}')
    else:                                      # 截断不存在（del 之后）
        lc.rect(x, y, CELL_W, CELL_H, '#f1f5f9', '#f1f5f9', rx=6, sw=0)

for ci in range(len(STEPS)):
    for r in range(N_SLOT):
        draw_cell(ci, r, None)

# d 滑入箭头：步7 列 slot3 → 步8 列 slot1
d_from = (GRID_X + 3 * (CELL_W + GAP_X) + CELL_W, GRID_Y + 3 * (CELL_H + GAP_Y) + CELL_H / 2)
d_to = (GRID_X + 4 * (CELL_W + GAP_X), GRID_Y + 1 * (CELL_H + GAP_Y) + CELL_H / 2)
lc.parrow([(d_from[0] + 1, d_from[1]), (d_from[0] + GAP_X / 2, d_from[1]),
           (d_to[0] - GAP_X / 2, d_to[1]), (d_to[0] - 1, d_to[1])], RC['d'], 2.2)
lc.text(GRID_X + 3 * (CELL_W + GAP_X) + CELL_W + GAP_X / 2 + 6, d_from[1] + 30, 'moved=(3,1)',
        7.8, RC['d'], 'middle', True, maxw=90, tag='moved')

# ---------------- 右上：token 级放大镜（只拷活跃前缀） ----------------
MG_X, MG_Y, MG_W, MG_H = 1000, 118, 440, 208
lc.rect(MG_X, MG_Y, MG_W, MG_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(MG_X + 14, MG_Y + 20, '只拷活跃前缀——token 级铁证', 10.5, lc.C_TXT, 'start', True,
        maxw=MG_W - 28, tag='mg:t')
lc.text(MG_X + 14, MG_Y + 36, '（搬移只拷 [0, 活跃前缀)，陈旧尾巴原地无害）', 8.2, lc.C_MUTE,
        'start', maxw=MG_W - 28, tag='mg:s')

def token_row(y, title, cells, owner):
    """cells = [(val, kind)]，kind ∈ act/stale/zero。"""
    lc.text(MG_X + 14, y + 13, title, 8.4, lc.C_TXT, 'start', True, maxw=330, tag='mg:r:' + title[:8])
    x = MG_X + 14
    for val, kind in cells:
        if kind == 'act':
            lc.rect(x, y + 20, 46, 24, '#ffffff', RC[owner], rx=4, sw=1.6)
            lc.text(x + 23, y + 36, str(val), 9.3, RC[owner], 'middle', True, maxw=42, tag=f'mg:{val}')
        elif kind == 'stale':
            lc.rect(x, y + 20, 46, 24, '#e2e8f0', '#cbd5e1', rx=4, sw=1.0)
            lc.text(x + 23, y + 36, str(val), 9.3, '#94a3b8', 'middle', maxw=42, tag=f'mg:{val}')
        else:
            lc.rect(x, y + 20, 46, 24, '#f8fafc', '#e2e8f0', rx=4, sw=0.8)
            lc.text(x + 23, y + 36, str(val), 9.3, '#cbd5e1', 'middle', maxw=42, tag=f'mg:{val}')
        x += 52
    return x

xr = token_row(MG_Y + 46, '步 8 后 row1（d 滑入 1）', [(40, 'act'), (21, 'stale'), (22, 'stale'), (0, 'zero')], 'd')
lc.text(xr + 6, MG_Y + 79, 'b 的陈旧尾巴', 7.8, '#94a3b8', 'start', maxw=110, tag='mg:tail1')
xr = token_row(MG_Y + 96, '步 11 后 row0（e 复用洞 0）', [(50, 'act'), (51, 'act'), (12, 'stale'), (13, 'stale')], 'e')
lc.text(xr + 6, MG_Y + 129, 'a 的陈旧尾巴', 7.8, '#94a3b8', 'start', maxw=110, tag='mg:tail2')
lc.text(MG_X + 14, MG_Y + 164, '下次写入自然覆盖——收集/搬移都以活跃前缀为界，尾巴不清理也不读。',
        8, lc.C_MUTE, 'start', maxw=MG_W - 28, tag='mg:f1')
lc.text(MG_X + 14, MG_Y + 182, 'moved=(3,1) 记入 BatchUpdate，供 logitsprocs 增量重建状态。',
        8, lc.C_MUTE, 'start', maxw=MG_W - 28, tag='mg:f2')

# ---------------- 右下：builder.removed 降序表演化 ----------------
RM_X, RM_Y, RM_W, RM_H = 1000, 344, 440, 250
lc.rect(RM_X, RM_Y, RM_W, RM_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(RM_X + 14, RM_Y + 20, 'builder.removed（恒降序）+ condense 双指针', 10.5, lc.C_TXT,
        'start', True, maxw=RM_W - 28, tag='rm:t')
RM_ROWS = [
    ('步 2-5', '[]（无 remove）', ''),
    ('步 6', '[1]', 'remove b 打洞'),
    ('步 7', '[2,1]', '公开属性合法时点读=降序'),
    ('步 8', '[2]', 'condense 后残留'),
    ('步 9', '[]（refresh 清空·解封）', ''),
    ('步 10', '[0]', 'remove a 打洞'),
    ('步 11', '[]（add e 弹出 0）', ''),
    ('步 12', '[]（洞尽→slot=2 追加）', 'pop 返回 None'),
]
for i, (s, v, note) in enumerate(RM_ROWS):
    y = RM_Y + 40 + i * 21
    lc.text(RM_X + 14, y, s, 8.6, lc.C_TXT, 'start', True, maxw=64, tag='rm:s' + str(i))
    lc.text(RM_X + 84, y, v, 8.6, lc.C_KV_S, 'start', maxw=180, tag='rm:v' + str(i))
    if note:
        lc.text(RM_X + 268, y, note, 7.8, lc.C_MUTE, 'start', maxw=RM_W - 282, tag='rm:n' + str(i))
lc.text(RM_X + 14, RM_Y + 40 + 8 * 21 + 14,
        'condense 初值：last = num_reqs + len(removed) − 1 = 2+2−1 = 3；peek=1 < 3 → 滑 3→1。',
        8, '#334155', 'start', maxw=RM_W - 28, tag='rm:ptr')

# ---------------- 网格下方：步8 / 步13 两段 condense 注 ----------------
G_BOT = GRID_Y + N_SLOT * (CELL_H + GAP_Y)          # 568
c8_x = GRID_X + 4 * (CELL_W + GAP_X) + CELL_W / 2
lc.rect(c8_x - 150, G_BOT + 10, 300, 74, '#ffffff', RC['d'], rx=7, sw=1.2)
lc.text(c8_x, G_BOT + 27, '步 8 condense 双指针', 9, RC['d'], 'middle', True, maxw=280, tag='c8:t')
lc.text(c8_x, G_BOT + 43, 'last=3 降到 2∈removed 再降 1；peek=2 ≥ 1 → break', 7.8, '#334155',
        'middle', maxw=286, tag='c8:l1')
lc.text(c8_x, G_BOT + 58, '洞@2 位于尾部 → 直接截断（分文未花）；只拷 d 的 1 列活跃 token', 7.8,
        '#334155', 'middle', maxw=286, tag='c8:l2')
lc.text(c8_x, G_BOT + 73, '截断 del 至 num_reqs=2', 7.8, lc.C_MUTE, 'middle', maxw=286, tag='c8:l3')

c13_x = GRID_X + 8 * (CELL_W + GAP_X) + CELL_W / 2
lc.rect(c13_x - 140, G_BOT + 10, 280, 74, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(c13_x, G_BOT + 27, '步 13 condense 早退', 9, lc.C_MUTE, 'middle', True, maxw=260, tag='c13:t')
lc.text(c13_x, G_BOT + 43, 'removed 空（洞已全被 add 填平）', 7.8, '#334155', 'middle', maxw=266,
        tag='c13:l1')
lc.text(c13_x, G_BOT + 58, '首分支零成本早退——批不变', 7.8, '#334155', 'middle', maxw=266,
        tag='c13:l2')
lc.text(c13_x, G_BOT + 73, '跨机制对照：拍 3 r3 填洞即此路径', 7.8, lc.C_MUTE, 'middle', maxw=266,
        tag='c13:l3')

# ---------------- 底部：不变量 + 图例 + 页脚 ----------------
BOT_Y = G_BOT + 104
lc.text(MX, BOT_Y, '不变量：任一 add_request 都不覆盖活请求——要么弹 removed 中的洞（打洞与登记在 remove 内原子成对），要么洞尽后追加到 num_reqs；拍末 [0, num_reqs) 恒连续。',
        9.3, lc.C_GPU_S, 'start', True, maxw=BXR - MX, tag='inv')

LEG_Y = BOT_Y + 26
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', lc.C_KV_S, rx=3, sw=1.5)
lc.text(lx + 25, LEG_Y + 1, '活请求（标 token 数）', 8.5, lc.C_TXT, 'start', maxw=150, tag='lg1')
lx += 25 + lc.tw('活请求（标 token 数）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', '#94a3b8', rx=3, sw=1.2, dash=True)
lc.text(lx + 25, LEG_Y + 1, '洞（_req_ids=None，数据不搬）', 8.5, lc.C_TXT, 'start', maxw=200, tag='lg2')
lx += 25 + lc.tw('洞（_req_ids=None，数据不搬）', 8.5) + 18
lc.rect(lx, LEG_Y - 9, 20, 12, '#f1f5f9', '#f1f5f9', rx=3, sw=0)
lc.text(lx + 25, LEG_Y + 1, '已截断（del，列表之外）', 8.5, lc.C_TXT, 'start', maxw=170, tag='lg3')
lx += 25 + lc.tw('已截断（del，列表之外）', 8.5) + 18
lc.text(lx, LEG_Y + 1, 'a-f 六色 = 请求身份色（非架构色）', 8.5, lc.C_MUTE, 'start', maxw=240,
        tag='lg4')

lc.text(MX, LEG_Y + 22, '逐字锚 vllm/v1/worker/gpu_input_batch.py:L324-L348（_register_add_request 弹洞/追加）· L530-L548（remove 打洞）· L708-L838（condense 双指针 + 截断）· vllm/v1/sample/logits_processor/state.py:L18-L145（BatchUpdateBuilder 降序账）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 38, '九步读数取自精简版 companion host 实测（InputBatch 直驱：_req_ids 快照 / removed 探针 / 行 token 前 4 格）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch18-fig-slot-lifecycle.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
