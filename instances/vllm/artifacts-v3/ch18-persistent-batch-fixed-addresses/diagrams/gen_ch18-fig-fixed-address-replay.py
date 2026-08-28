#!/usr/bin/env python3
"""ch18 机制图 7 · 捕获时刻 vs 第 4 拍回放（figure_spec ch18-fig-fixed-address-replay，模板 before-after）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）『执行臂中层』GPUModelRunner 框 center ④
拍片『前向 · 固定地址喂图』与 south『why · 固定地址 → CUDA graph』注——即本章 L2 章图
该拍片/该注的机制展开（完整捕获-回放链归 ch19，本图只画『地址不变』这半边证据）。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：回放命中 = BatchDescriptor 全等 AND 输入张量地址不变——后半条件由本章全部设计
供给：五拍里批次从 2 请求变 1 再变 2、token 账从 5 变 1 再变 4，但六个喂图缓冲的
data_ptr 一个都没动。

数字全部取自 figure_spec.numbers（traces/ch18_m02_reconcile.json beats[*].data_ptrs +
wire（形状变化）+ cuda_graph.py:L346-L355 + forward_context.py:L29-L58 +
gpu_model_runner.py:L763-L764/L3864-L3877）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 612
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '批在变形，地址纹丝不动——回放断言逐个 data_ptr 比对捕获时刻，六项全中',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '回放命中 = BatchDescriptor 全等 AND 输入张量地址不变（后半条件由本章供给）：五拍批形状 2→2→2→1→2 请求、token 账 5→2→3→1→4，六个喂图缓冲的 data_ptr 全程不变',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 拍片 ④ 前向·固定地址喂图 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 六个喂图缓冲 ----------------
# (name, 固定全长格数, 捕获拍活跃格数, 回放拍活跃格数, 左计数签, 右计数签)
#   input/positions 活跃前缀 = 本拍 total（拍1=5、拍4=1）；qsl 写入项 = num_reqs+1（3→2）；
#   seq_lens 有效项 = num_reqs（2→1）；token_ids_cpu_tensor 活跃行 = 在批请求数（2 行→1 行，每行 32 格）
BUFERS = [
    ('input_ids（cpu）', 64, 5, 1, 'n=5', 'n=1'),
    ('input_ids（gpu）', 64, 5, 1, 'n=5', 'n=1'),
    ('positions', 64, 5, 1, 'n=5', 'n=1'),
    ('query_start_loc（cpu）', 5, 3, 2, '写入 3 项', '写入 2 项'),
    ('seq_lens', 4, 2, 1, '2 项', '1 项'),
    ('token_ids_cpu_tensor', 128, 64, 32, '活跃 2 行', '活跃 1 行'),
]

LP_X, LP_W = MX, 560
RP_X, RP_W = BXR - 560, 560
PANEL_Y, PANEL_H = 100, 312
BAR_H, BAR_GAP = 26, 38
BAR_X_OFF, BAR_X_W = 168, 330

def panel(x, y, w, h, title, sub, stroke, shape_txt):
    lc.rect(x, y, w, h, '#ffffff', stroke, rx=10, sw=2.0)
    lc.text(x + 16, y + 24, title, 12, stroke, 'start', True, maxw=w - 32, tag='p:t' + title[:6])
    lc.text(x + 16, y + 42, sub, 8.4, lc.C_MUTE, 'start', maxw=w - 32, tag='p:s' + title[:6])
    # 形状签（右上）
    lc.rect(x + w - 150, y + 8, 138, 34, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.2)
    lc.text(x + w - 81, y + 22, shape_txt, 9, lc.C_BEAT_T, 'middle', True, maxw=130,
            tag='p:sh' + title[:6])
    lc.text(x + w - 81, y + 35, 'token', 7.4, lc.C_BEAT_T, 'middle', maxw=130,
            tag='p:sh2' + title[:6])

# ---- 左联：捕获时刻（拍 1） ----
panel(LP_X, PANEL_Y, LP_W, PANEL_H, '捕获时刻（拍 1）', 'CUDA graph 把 kernel 指针与 grid 烤死在图里——记下每个输入的 data_ptr',
      lc.C_ENG_S, '2 请求 · 5')
for i, (name, ncell, nact, _ra, cap, _cr) in enumerate(BUFERS):
    y = PANEL_Y + 66 + i * BAR_GAP
    lc.text(LP_X + 158, y + BAR_H / 2 + 3, name, 8.6, lc.C_TXT, 'end', maxw=150,
            tag='l:n' + str(i))
    pitch = BAR_X_W / ncell
    for c in range(ncell):
        lc.rect(LP_X + BAR_X_OFF + c * pitch, y, pitch - 0.8, BAR_H, '#f8fafc', '#e2e8f0',
                rx=1.2, sw=0.5)
    for c in range(nact):
        lc.rect(LP_X + BAR_X_OFF + c * pitch, y, pitch - 0.8, BAR_H, lc.C_ENG_S, lc.C_ENG_S,
                rx=1.2, sw=0)
    # 地址锚点（左端锁记）+ 计数签
    lc.text(LP_X + BAR_X_OFF + 4, y + BAR_H / 2 + 4, '🔒', 11, lc.C_GPU_S, 'start', tag='l:lck' + str(i))
    lc.text(LP_X + BAR_X_OFF + BAR_X_W + 8, y + BAR_H / 2 - 5, cap, 8.2, lc.C_ENG_S,
            'start', True, maxw=90, tag='l:cap' + str(i))
    lc.text(LP_X + BAR_X_OFF + BAR_X_W + 8, y + BAR_H / 2 + 9, 'data_ptr ✓', 8.6, lc.C_GPU_S,
            'start', True, maxw=80, tag='l:ok' + str(i))

# ---- 右联：第 4 拍回放 ----
panel(RP_X, PANEL_Y, RP_W, PANEL_H, '第 4 拍回放（批 2→1 请求）', '同一批缓冲、同一组地址——前缀变短，条全长与锚点不动',
      lc.C_GPU_S, '1 请求 · 1')
for i, (name, ncell, _la, nact, _cl, cap) in enumerate(BUFERS):
    y = PANEL_Y + 66 + i * BAR_GAP
    lc.text(RP_X + 158, y + BAR_H / 2 + 3, name, 8.6, lc.C_TXT, 'end', maxw=150,
            tag='r:n' + str(i))
    pitch = BAR_X_W / ncell
    for c in range(ncell):
        lc.rect(RP_X + BAR_X_OFF + c * pitch, y, pitch - 0.8, BAR_H, '#f8fafc', '#e2e8f0',
                rx=1.2, sw=0.5)
    for c in range(nact):
        lc.rect(RP_X + BAR_X_OFF + c * pitch, y, pitch - 0.8, BAR_H, lc.C_GPU_S, lc.C_GPU_S,
                rx=1.2, sw=0)
    lc.text(RP_X + BAR_X_OFF + 4, y + BAR_H / 2 + 4, '🔒', 11, lc.C_GPU_S, 'start', tag='r:lck' + str(i))
    lc.text(RP_X + BAR_X_OFF + BAR_X_W + 8, y + BAR_H / 2 - 5, cap, 8.2, lc.C_GPU_S,
            'start', True, maxw=90, tag='r:cap' + str(i))
    lc.text(RP_X + BAR_X_OFF + BAR_X_W + 8, y + BAR_H / 2 + 9, '== 捕获 ✓', 8.6, lc.C_GPU_S,
            'start', True, maxw=90, tag='r:ok' + str(i))

# ---- 中缝：断言框 ----
AS_X = LP_X + LP_W + 10
AS_W = RP_X - AS_X - 10
lc.rect(AS_X, PANEL_Y + 36, AS_W, 258, '#ffffff', lc.C_ABORT, rx=8, sw=1.5, dash=True)
_as = AS_X + AS_W / 2
lc.text(_as, PANEL_Y + 58, '回放前断言（DEBUG）', 10.5, lc.C_ABORT, 'middle', True,
        maxw=AS_W - 16, tag='as:t')
lc.text(_as, PANEL_Y + 76, 'vllm/compilation/cuda_graph.py:L346-L355', 8, lc.C_MUTE,
        'middle', maxw=AS_W - 16, tag='as:file')
lc.text(_as, PANEL_Y + 104, 'assert new_input_addresses', 8.4, '#334155', 'middle',
        maxw=AS_W - 12, tag='as:q1')
lc.text(_as, PANEL_Y + 120, '  == entry.input_addresses', 8.4, '#334155', 'middle',
        maxw=AS_W - 12, tag='as:q2')
lc.text(_as, PANEL_Y + 144, '「Input addresses for', 8.2, '#475569', 'middle',
        maxw=AS_W - 12, tag='as:q3')
lc.text(_as, PANEL_Y + 160, 'cudagraphs are different', 8.2, '#475569', 'middle',
        maxw=AS_W - 12, tag='as:q4')
lc.text(_as, PANEL_Y + 176, 'during replay. Expected', 8.2, '#475569', 'middle',
        maxw=AS_W - 12, tag='as:q5')
lc.text(_as, PANEL_Y + 192, '…, got …」', 8.2, '#475569', 'middle', maxw=AS_W - 12,
        tag='as:q6')
lc.text(_as, PANEL_Y + 218, '六缓冲 × 五拍', 9.5, lc.C_TXT, 'middle', True,
        maxw=AS_W - 16, tag='as:c1')
lc.text(_as, PANEL_Y + 236, 'data_ptr 全等', 9.5, lc.C_TXT, 'middle', True,
        maxw=AS_W - 16, tag='as:c2')
lc.text(_as, PANEL_Y + 260, '形状四变：请求 2→2→2→1→2', 8.4, lc.C_MUTE, 'middle',
        maxw=AS_W - 12, tag='as:c3')
lc.text(_as, PANEL_Y + 276, 'token 5→2→3→1→4', 8.4, lc.C_MUTE, 'middle',
        maxw=AS_W - 16, tag='as:c4')
# 两联 → 断言 的对照箭头
for yy in (PANEL_Y + 140, PANEL_Y + 230):
    lc.seg(LP_X + LP_W - 3, yy, AS_X - 3, yy, lc.C_MUTE, 1.4, 'std')
    lc.seg(AS_X + AS_W + 3, yy, RP_X + 3, yy, lc.C_MUTE, 1.4, 'std')

# ---------------- 底部：另一半条件 + 先等后录 + 图例 + 页脚 ----------------
BT_Y = PANEL_Y + PANEL_H + 22
lc.rect(MX, BT_Y, 660, 92, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 14, BT_Y + 18, '回放命中的另一半（预告 → ch19）', 9.6, lc.C_TXT, 'start', True,
        maxw=400, tag='bt:t')
lc.text(MX + 14, BT_Y + 37, 'BatchDescriptor(num_tokens, num_reqs, uniform, has_lora, num_active_loras) 全等——',
        8.4, '#334155', 'start', maxw=630, tag='bt:l1')
lc.text(MX + 14, BT_Y + 54, '形状侧由 padding 到捕获形状满足（frozen dataclass，forward_context.py:L29-L58）；',
        8.4, '#334155', 'start', maxw=630, tag='bt:l2')
lc.text(MX + 14, BT_Y + 71, '「Persistent buffers for CUDA graphs」——地址不变的 why 直指回放（L763-L764）。',
        8.4, '#334155', 'start', maxw=630, tag='bt:l3')

SX = MX + 676
lc.rect(SX, BT_Y, BXR - SX, 92, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(SX + 14, BT_Y + 18, '地址稳定的跨拍前提：先等后录', 9.6, lc.C_TXT, 'start', True,
        maxw=400, tag='sy:t')
lc.text(SX + 14, BT_Y + 37, '复用同一块 CPU buffer 必须自管同步——synchronize_input_prep 每拍先',
        8.4, '#334155', 'start', maxw=BXR - SX - 28, tag='sy:l1')
lc.text(SX + 14, BT_Y + 54, '等上一拍的 prepare_inputs_event 再写（L3864-L3877）；异步心跳下',
        8.4, '#334155', 'start', maxw=BXR - SX - 28, tag='sy:l2')
lc.text(SX + 14, BT_Y + 71, '写回与下拍调和本就在重叠窗口（ch12 已立）。', 8.4, '#334155',
        'start', maxw=BXR - SX - 28, tag='sy:l3')

LEG_Y = BT_Y + 112
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_ENG_S, lc.C_ENG_S, rx=3, sw=0)
lc.text(lx + 25, LEG_Y + 1, '活跃前缀（捕获拍）', 8.5, lc.C_TXT, 'start', maxw=140, tag='lg1')
lx += 25 + lc.tw('活跃前缀（捕获拍）', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_GPU_S, lc.C_GPU_S, rx=3, sw=0)
lc.text(lx + 25, LEG_Y + 1, '活跃前缀（回放拍）', 8.5, lc.C_TXT, 'start', maxw=140, tag='lg2')
lx += 25 + lc.tw('活跃前缀（回放拍）', 8.5) + 16
lc.text(lx, LEG_Y + 1, '🔒 = data_ptr 锚点（左右联逐一相同）', 8.5, lc.C_GPU_S, 'start', True,
        maxw=260, tag='lg3')
lx += lc.tw('🔒 = data_ptr 锚点（左右联逐一相同）', 8.5, True) + 16
lc.text(lx, LEG_Y + 1, '浅灰 = 固定全长（按 max 预留）', 8.5, lc.C_MUTE, 'start', maxw=200,
        tag='lg4')

lc.text(MX, LEG_Y + 22, '逐字锚 vllm/compilation/cuda_graph.py:L346-L355（DEBUG data_ptr 断言）· vllm/forward_context.py:L29-L58（BatchDescriptor）· vllm/v1/worker/gpu_model_runner.py:L763-L764（Persistent buffers 注）· L3864-L3877（synchronize_input_prep）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 38, '五拍形状与 data_ptr 读数取自精简版 companion host 实测（六缓冲×五拍 data_ptr 全等、批形状逐拍记录）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch18-fig-fixed-address-replay.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
