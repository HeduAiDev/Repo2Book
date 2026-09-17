#!/usr/bin/env python3
"""ch38 机制图 m6 · dma-ordering（figure_spec ch38-fig-dma-ordering，模板 swimlane·UML 时序）

放大自 L2 站 7（⑤ DMA 引擎）· L0：GPU 池与外部池之间的搬运通道。
时序图严格 UML 文法（FIGURE-SYSTEM §0 硬规则 2 + WRITING-CONTRACT-v3 §8）：
竖直生命线 + 共享时间轴（gridlines 贯穿）+ 水平消息直线，禁一切折线/肘形；
瞬时动作退化为骑线时刻标记 + 标注。

claim：DMA 三戒的时序：GPU→CPU 的 store 先 wait_stream(计算流)（等模型写完再搬）、
同方向 transfer 逐个 wait_event(前一 transfer 的 end_event) 保序串行；CPU→GPU 的 load
才开 CU_MEMCPY_SRC_ACCESS_ORDER_ANY 让驱动流水线源读——安全性差异在源：
活 GPU KV 被计算流持续写 vs pinned host 无并发 GPU 写。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · 三戒代码原文锚：wait_stream(计算流)/同向保序/SRC_ACCESS_ORDER_ANY（gpu_worker.py:L362-L400 注释原文）
  · 真实 store job 描述符：18 条 × 512B = 9216B（9 块 × 2 层引用）；CPU 槽和 [93696, 98304, 102912]
  · 半块跳越：块 [1,2] skip=1 → 偏移 [300,400,500]
  · 完成=事件轮询非回调：get_finished 按 query() 出队、wait()=synchronize；三池复用
  · kernel 选择：GPU→CPU 恒用专用拷贝引擎 swap_blocks_batch
坐标由常量/循环计算；文本全 esc()。时间轴只表顺序、不标时长（host 不可观察，不杜撰间隔）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1560
MX = 60
BXR = W - MX

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'DMA 三戒：store 等工头、同向排队；load 才许抄近道', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 60, '安全性押在『源有没有人并发写』上——活 GPU KV 被计算流持续写（store 必须等+保序）；pinned host 内存无并发 GPU 写（load 源读可流水线）',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L2 站 7（⑤ DMA 引擎）· L0：GPU 池↔外部池搬运通道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 时序面板 ----------------
PX0, PY0, PX1, PY1 = MX, 116, 1020, 596
lc.rect(PX0, PY0, PX1 - PX0, PY1 - PY0, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
A_X, B_X = 340, 760
LF_T, LF_B = PY0 + 62, PY1 - 58
# 名牌
for x, name, color in [(A_X, '计算流（模型 forward）', lc.C_GPU_S),
                       (B_X, '传输专用流（DMA）', lc.C_KV_S)]:
    lc.rect(x - 105, PY0 + 14, 210, 28, '#ffffff', color, rx=7, sw=1.4)
    lc.text(x, PY0 + 32, name, 9.5, color, 'middle', True, maxw=200, tag='life:' + name[:6])
# 生命线（竖直虚线）+ 共享时间轴 gridlines
for x in (A_X, B_X):
    lc.seg(x, PY0 + 42, x, LF_B, lc.C_MUTE, 1.2, dash=True)
for i in range(7):
    gy = PY0 + 62 + i * 64
    lc.seg(A_X - 150, gy, B_X + 150, gy, '#e2e8f0', 1.0)


def ev_dot(x, y, color):
    s = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#ffffff" '
         f'stroke-width="1.2"/>')
    lc.ELEMS.append(((x - 5, y - 5, x + 5, y + 5), s))


# 计算流活动条：写本步 KV
CB_Y0, CB_Y1 = PY0 + 74, PY0 + 134
lc.rect(A_X - 16, CB_Y0, 32, CB_Y1 - CB_Y0, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.4)
lc.text(A_X - 24, (CB_Y0 + CB_Y1) / 2, '写本步 KV（forward）', 9, lc.C_GPU_S, 'end',
        maxw=180, tag='ev:compute')
# 消息①：wait_stream（水平虚线消息，A → B）
M1_Y = PY0 + 154
lc.seg(A_X, M1_Y, B_X, M1_Y, lc.C_ENG_S, 1.8, 'std', dash=True)
lc.text((A_X + B_X) / 2, M1_Y - 7, 'wait_stream(计算流) ｜ 戒一：等模型写完再搬', 9,
        lc.C_ENG_S, 'middle', maxw=B_X - A_X - 10, tag='msg1')
# store₁ 活动条（青）
S1_Y0, S1_Y1 = M1_Y + 16, M1_Y + 66
lc.rect(B_X - 16, S1_Y0, 32, S1_Y1 - S1_Y0, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.4)
lc.text(B_X + 26, S1_Y0 + 12, 'store₁：job 0 的 18 条描述符（一次内核调用整批）', 8.8,
        lc.C_KV_S, 'start', maxw=250, tag='ev:store1')
ev_dot(B_X, S1_Y1, lc.C_KV_S)
lc.text(B_X + 26, S1_Y1 - 2, '● end_event', 8, lc.C_MUTE, 'start', maxw=120, tag='ev:s1e')
# 骑线时刻标记：wait_event（瞬时动作，戒二）
WE_Y = S1_Y1 + 20
d = 8
s = (f'<path d="M{B_X:.1f},{WE_Y - d} L{B_X + d:.1f},{WE_Y} L{B_X:.1f},{WE_Y + d} '
     f'L{B_X - d:.1f},{WE_Y} Z" fill="#ffffff" stroke="{lc.C_ENG_S}" stroke-width="1.6"/>')
lc.ELEMS.append(((B_X - d - 2, WE_Y - d - 2, B_X + d + 2, WE_Y + d + 2), s))
lc.text(B_X + 26, WE_Y - 3, 'wait_event(前一 transfer 的 end_event)', 8.8,
        lc.C_ENG_S, 'start', maxw=250, tag='ev:wait1')
lc.text(B_X + 26, WE_Y + 12, '戒二：同向保序串行', 8.8, lc.C_ENG_S, 'start', maxw=250,
        tag='ev:wait2')
# store₂ 活动条
S2_Y0, S2_Y1 = WE_Y + 18, WE_Y + 58
lc.rect(B_X - 16, S2_Y0, 32, S2_Y1 - S2_Y0, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.4)
lc.text(B_X + 26, S2_Y0 + 12, 'store₂：下一批 transfer——同向只能排队', 8.8,
        lc.C_KV_S, 'start', maxw=250, tag='ev:store2')
ev_dot(B_X, S2_Y1, lc.C_KV_S)
# load 活动条（三条平行细条 = 源读流水线；绿=回流 GPU）
LD_Y0 = S2_Y1 + 34
for k in range(3):
    ly0 = LD_Y0 + k * 16
    lc.rect(B_X - 16, ly0, 32, 10, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.2)
lc.text(B_X + 26, LD_Y0 + 6, 'load（CPU→GPU）｜ 戒三：SRC_ACCESS_ORDER_ANY', 8.8,
        lc.C_GPU_S, 'start', maxw=250, tag='ev:load1')
lc.text(B_X + 26, LD_Y0 + 22, '驱动可流水线源读（pinned host 无人并发写）', 8.3,
        lc.C_MUTE, 'start', maxw=250, tag='ev:load2')
ev_dot(B_X, LD_Y0 + 3 * 16, lc.C_GPU_S)
# 完成方式注记 + 图例（面板底，生命线终点之下）
lc.text(PX0 + 16, LF_B + 18, '图例：虚线消息 = 等待/依赖 · ● = end_event · ◆ = 瞬时等待点 · 青活动条 = store transfer · 绿活动条 = 计算 / load',
        8, lc.C_MUTE, 'start', maxw=PX1 - PX0 - 32, tag='lg:line')
lc.text(PX0 + 16, PY1 - 16, '完成 = 事件轮询非回调：get_finished 按 query() 出队 · wait()=synchronize · 流 / 事件 / 描述符缓冲三池复用（get_finished 出队时归还三池）',
        8.5, lc.C_MUTE, 'start', maxw=PX1 - PX0 - 32, tag='seq:done')

# ---------------- 右侧：戒律的为什么 ----------------
RX0, RW = 1060, BXR - 1060
RY0, RH = PY0, PY1 - PY0
lc.rect(RX0, RY0, RW, RH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(RX0 + 14, RY0 + 22, '为什么差在源：并发写者存在与否', 10.5, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='why:t')
WHY_Y = RY0 + 40
lc.rect(RX0 + 14, WHY_Y, RW - 28, 150, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.3)
lc.text(RX0 + 26, WHY_Y + 20, 'store 的源 = 活 GPU KV', 9.5, lc.C_KV_S, 'start', True,
        maxw=RW - 52, tag='why:s:t')
for j, ln in enumerate([
        '· 计算流还在持续写这块显存',
        '· 不等就搬 → 可能读到半新半旧',
        '· 戒一 wait_stream：等工头撒手',
        '· 戒二 wait_event：同方向排队串行，',
        '  前一趟 end_event 之后才发车']):
    lc.text(RX0 + 26, WHY_Y + 40 + j * 16, ln, 8.6, '#334155', 'start',
            maxw=RW - 48, tag='why:s:l%d' % j)
WHY2_Y = WHY_Y + 166
lc.rect(RX0 + 14, WHY2_Y, RW - 28, 150, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.3)
lc.text(RX0 + 26, WHY2_Y + 20, 'load 的源 = pinned host 内存', 9.5, lc.C_GPU_S, 'start', True,
        maxw=RW - 52, tag='why:l:t')
for j, ln in enumerate([
        '· 没有任何 GPU 流并发写它',
        '· 戒三 SRC_ACCESS_ORDER_ANY：让驱动',
        '  流水线化多源读（抄近道并行）',
        '· kernel 选择：GPU→CPU 恒用专用拷贝',
        '  引擎 swap_blocks_batch（胜过 Triton）']):
    lc.text(RX0 + 26, WHY2_Y + 40 + j * 16, ln, 8.6, '#334155', 'start',
            maxw=RW - 48, tag='why:l:l%d' % j)
lc.text(RX0 + 14, RY0 + RH - 12, '时序锚：cpu/gpu_worker.py:L362-L400 注释原文', 8,
        lc.C_FAINT, 'start', maxw=RW - 28, tag='why:foot')

# ---------------- 下半左：描述符三列小抄 ----------------
DS_Y = PY1 + 22
DS_X, DS_W = MX, 760
DS_H = 268
lc.rect(DS_X, DS_Y, DS_W, DS_H, '#ffffff', lc.C_KV_S, rx=8, sw=1.3)
lc.text(DS_X + 14, DS_Y + 20, '一趟活的清单 = 描述符三列小抄（src 指针 / dst 指针 / size）', 10,
        lc.C_KV_S, 'start', True, maxw=DS_W - 28, tag='ds:t')
# 三列表头 + 18 条行条（结构条，不填杜撰指针值）
TB_X, TB_Y = DS_X + 20, DS_Y + 34
COL_W, ROW_H = 118, 6.5
cols = ['src 指针', 'dst 指针', 'size']
for j, c in enumerate(cols):
    lc.rect(TB_X + j * (COL_W + 12), TB_Y, COL_W, 18, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.1)
    lc.text(TB_X + j * (COL_W + 12) + COL_W / 2, TB_Y + 13, c, 8.5, lc.C_KV_S, 'middle',
            True, maxw=COL_W - 6, tag='ds:c%d' % j)
for r in range(18):
    ry = TB_Y + 24 + r * (ROW_H + 1.5)
    for j in range(3):
        lc.rect(TB_X + j * (COL_W + 12), ry, COL_W, ROW_H, '#e0f2fe', lc.C_KV_S, rx=2, sw=0.6)
lc.text(TB_X + 3 * (COL_W + 12) + 12, TB_Y + 24 + 9 * (ROW_H + 1.5) + 5,
        '18 条（int64 ×3 张量）', 8.5, lc.C_MUTE, 'start',
        maxw=DS_W - (TB_X - DS_X) - 3 * (COL_W + 12) - 20, tag='ds:n')
ds_notes = [
    '· 真实 store job：18 条 × 512B = 9216B（计数公式 = Σ group_size × 层引用数 = 9 × 2）',
    '· CPU 槽校验和 [93696, 98304, 102912] = 每 3 块拼接之和（逐槽相等，字节守恒实测）',
    '· 2 条 × 32B 描述符三缓冲实验：交换搬运后 dst 两半互换、字节和 2016 守恒',
]
NT_Y = TB_Y + 24 + 18 * (ROW_H + 1.5) + 16
for j, ln in enumerate(ds_notes):
    lc.text(TB_X, NT_Y + j * 14.5, ln, 8.4, '#334155', 'start', maxw=DS_W - 40,
            tag='ds:l%d' % j)

# ---------------- 下半右：半块跳越 ----------------
SK_X = DS_X + DS_W + 24
SK_W = BXR - SK_X
lc.rect(SK_X, DS_Y, SK_W, DS_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.3)
lc.text(SK_X + 14, DS_Y + 20, '半块跳越：block_indices 记逻辑起点', 10, lc.C_GPU_S,
        'start', True, maxw=SK_W - 28, tag='sk:t')
# 3 个 CPU 块 × 每块 2 子块（100B），偏移标尺 0..500（组内紧邻、组间 8px）
SKB_Y, SKB_H = DS_Y + 64, 34
cell = (SK_W - 28 - 2 * 8) / 6
for idx in range(6):
    blk, sub = divmod(idx, 2)
    gap = 8 if sub == 0 and idx > 0 else 0
    x = SK_X + 14 + idx * cell + ((idx + 1) // 2) * 8
    fill = lc.C_GPU_F if not (blk == 1 and sub == 0) else '#bbf7d0'
    stroke = lc.C_GPU_S
    lc.rect(x, SKB_Y, cell, SKB_H, fill, stroke, rx=3, sw=1.1)
    lc.text(x + cell / 2, SKB_Y + SKB_H / 2 + 3.5, '%d' % (blk * 200 + sub * 100), 8,
            '#334155', 'middle', maxw=cell - 4, tag='sk:c%d' % idx)
# skip 箭头：从 300 起读
ar_x = SK_X + 14 + 2 * cell + 8 + cell  # 子块(1,0) 的右缘 = 偏移 300 处
lc.seg(ar_x, SKB_Y - 6, ar_x, SKB_Y - 22, lc.C_ENG_S, 1.6, 'std')
lc.text(ar_x, SKB_Y - 30, 'skip=1 → 从偏移 300 起读（块 1 的第 2 子块）', 8.3, lc.C_ENG_S,
        'middle', maxw=SK_W - 40, tag='sk:arrow')
sk_notes = [
    '· 3 个 CPU 块 · 行距 200B · 每块 2 子块 × 100B（指针 = base + b×行距 + j×子块）',
    '· 搬块 [1,2]：skip=1 → 偏移 [300, 400, 500]；skip=0 → [200, 300]',
    '· 用途：GPU 首块不对齐 offload 块边界时，从 CPU 块中部起读（GPULoadStoreSpec.block_indices）',
]
for j, ln in enumerate(sk_notes):
    lc.text(SK_X + 14, SKB_Y + SKB_H + 26 + j * 15.5, ln, 8.4, '#334155', 'start',
            maxw=SK_W - 26, tag='sk:l%d' % j)

# ---------------- 结论 + 页脚 ----------------
CONC_Y = DS_Y + DS_H + 24
lc.text(MX, CONC_Y,
        '图注结论：store 等工头、同向排队；load 才许抄近道——一趟活的全部信息装进三列描述符（从哪读 / 写到哪 / 搬多少），一次内核调用整批执行。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 vllm/v1/kv_offload/cpu/gpu_worker.py:L362-L400（三戒注释原文）· L40-L42（swap_blocks_batch 选择）· L419-L447（query() 轮询 + 三池复用）· L166-L172（双向各一 handler）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 时间轴只表顺序不表时长（同一竖直网格线 = 同一时刻）；「戒」序号对应正文三戒',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-dma-ordering.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
