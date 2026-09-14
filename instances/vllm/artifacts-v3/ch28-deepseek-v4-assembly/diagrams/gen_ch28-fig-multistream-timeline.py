#!/usr/bin/env python3
"""ch28 机制图 · 一拍注意力的多流 overlap 时间线(ch28-fig-multistream-timeline, 模板 swimlane)

放大自 L0『GPU 执行臂·模型层 forward + 编译』块内一拍注意力的执行细节——
L2 拍片⑧ 的机制版下钻(ch19 的 cudagraph 机制在此处被 @eager_break_during_capture 打断)。

时序图按 UML 文法(FIGURE-SYSTEM §0 时序图种规约): 参与者=竖直生命线(顶部名牌)、
共享时间轴纵向下行、活动条高=时长×统一比例尺(本图=定性: 重 GEMM>轻 GEMM)、
消息/事件=水平直线(跨中间生命线直穿是标准画法, 无折线绕行)。

claim: Attention 一拍内的 3 路 overlap: 默认流 fused_wqa_wkv(最重) + aux[0..2] 三个轻 GEMM
并行, join 后 split+双 rmsnorm 回默认流; eager break 内默认流跑 wq_b+qnorm+RoPE+kv_insert,
aux[0] 跑整个 indexer、aux[1] 跑 compressor, 最后 forward_mqa+o_proj 收尾——两道闸门
(token≤1024 才开多流; breakable cudagraph 捕获期强制顺序)守住切换。

锚点 = attention.py:L350-L540 · nvidia/model.py:L1020-L1024 · utils/multi_stream_utils.py:L20-L66 ·
eager_scratch.py:L12-L60。坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 1030
MX = 56
BXR = 1608
C_AUX_F, C_AUX_D = '#dcfce7', '#86efac'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一拍之内，四条泳道：重 GEMM 留默认流，三个轻 GEMM 各占一条 aux 道', 16, lc.C_TXT,
        'start', True, maxw=1040, tag='title')
lc.text(MX, 58, 'eager break 段整体跳出图捕获——默认流 wq_b+kv_insert ‖ indexer ‖ compressor 三路并行，收尾回捕获图；两道闸门决定泳道何时坍缩成单道顺序',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
_ch = '放大自 L0『模型层 forward + 编译』块 · L2 拍片⑧'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道 ----------------
LANES = [('默认流（重活）', 470, lc.C_GPU_F, lc.C_GPU_S, False),
         ('aux[0]', 800, C_AUX_F, '#16a34a', False),
         ('aux[1]', 1090, C_AUX_F, '#16a34a', False),
         ('aux[2]（预留）', 1380, '#ffffff', '#16a34a', True)]
LANE_TOP, LANE_BOT = 84, 816
for name, x, fill, stroke, dash in LANES:
    lc.rect(x - 75, LANE_TOP, 150, 30, fill, stroke, rx=15, sw=1.6, dash=dash)
    lc.text(x, LANE_TOP + 20, name, 10, '#14532d' if not dash else lc.C_MUTE, 'middle', True,
            maxw=140, tag='plate:' + name)
    lc.seg(x, LANE_TOP + 30, x, LANE_BOT, lc.C_GPU_S if x == 470 else lc.C_FAINT,
           2.0 if x == 470 else 1.2, dash=(x != 470))

# 时间轴
lc.seg(396, 150, 396, 800, lc.C_MUTE, 1.6, 'std')
lc.text(396, 138, '时间', 9.5, lc.C_MUTE, 'middle', True, tag='time')


def bar(x, y, h, label, sub=None, default=True, reserved=False):
    bw = 36
    lc.rect(x - bw / 2, y, bw, h, lc.C_GPU_S if default else C_AUX_D,
            '#14532d' if default else '#16a34a', rx=4, sw=1.3,
            dash=reserved)
    tx = x + bw / 2 + 10
    lc.text(tx, y + h / 2 - (5 if sub else 0), label, 9, '#14532d' if default else '#166534',
            'start', True, maxw=300, tag='bar:' + label[:10])
    if sub:
        lc.text(tx, y + h / 2 + 10, sub, 8, lc.C_MUTE, 'start', maxw=300, tag='bar:s' + label[:8])


def event_line(y, x1, x2, label, above=True):
    lc.seg(x1, y, x2, y, '#0369a1', 1.2, dash=True)
    for xx in (x1, x2):
        lc.circle(xx, y, 3.5, '#0369a1', 1.0, dash=False)
    lc.text(x2 - 6, y + (12 if above else -6), label, 8, '#0369a1', 'end', maxw=520,
            tag='ev:' + label[:10])


def record(x, y, label):
    lc.circle(x, y, 3.5, '#0369a1', 1.0, dash=False)
    lc.text(x + 10, y + 3, label, 7.5, '#0369a1', 'start', maxw=200, tag='rec:' + label[:10])


# ---------------- 段① 输入 GEMM（捕获图内） ----------------
lc.text(470, 160, '① 输入 GEMM 段（留在捕获图内）· 四道并行', 10.5, '#14532d', 'start', True,
        maxw=520, tag='ph:a')
event_line(176, 470, 1380, 'ln_events[0].record() → 三条 aux 道 wait（fan-out）')
bar(470, 188, 140, 'fused_wqa_wkv（最重）', 'qr_kv = 一枪出 q_lora|kv')
bar(800, 188, 50, 'compressor kv_score（轻）')
bar(1090, 188, 35, 'indexer.weights_proj（轻）')
bar(1380, 188, 60, 'indexer.compressor', 'kv_score（轻）', default=False)
record(800, 238, 'ln_events[1]')
record(1090, 223, 'ln_events[2]')
record(1380, 248, 'ln_events[3]')
event_line(344, 470, 1380, '默认流 join：wait(ln_events[1..3]) —— GEMM 段先全 join', above=False)

# ---------------- 段② split + 双归一（默认流） ----------------
lc.text(470, 372, '② split + fused_q_kv_rmsnorm（post-GEMM，回默认流）', 10.5, '#14532d',
        'start', True, maxw=520, tag='ph:b')
bar(470, 384, 34, 'split(q_lora | kv) + 双 RMSNorm 一核')

# ---------------- 段③ eager break ----------------
EB_Y, EB_H = 434, 286
lc.rect(386, EB_Y, 1210, EB_H, '#fffbeb', '#d97706', rx=8, sw=1.6, dash=True)
lc.text(470, EB_Y + 20, '③ eager break 段：@eager_break_during_capture——捕获图在此断开，注意力段 eager 跑', 10.5,
        '#92400e', 'start', True, maxw=900, tag='ph:c')
event_line(EB_Y + 42, 470, 1090, 'ln_events[0]（复用安全：GEMM 段已全 join）→ aux[0]/aux[1] wait')
bar(470, EB_Y + 54, 70, 'wq_b + qnorm + RoPE + kv_insert', '（q 留在消费它的默认流）')
bar(800, EB_Y + 54, 160, '整个 indexer', '（打分 + 选块，最长的支）', default=False)
bar(1090, EB_Y + 54, 100, 'compressor', '（压缩 KV 写池）', default=False)
bar(1380, EB_Y + 54, 60, 'indexer 内部 overlap', '（slot[2]：wq_b+q rope 量化‖compressor）',
    default=False, reserved=True)
record(800, EB_Y + 214, 'ln_events[1]')
record(1090, EB_Y + 154, 'ln_events[2]')
event_line(EB_Y + 228, 470, 1090, '默认流 join：wait(ln_events[1],[2]) → forward_mqa', above=False)
bar(470, EB_Y + 240, 36, 'forward_mqa（平台核，写 o_padded）')

# ---------------- 段④ 收尾（回捕获图） ----------------
lc.text(470, 738, '④ 收尾（回到捕获图）· 默认流单道', 10.5, '#14532d', 'start', True, maxw=520,
        tag='ph:d')
bar(470, 750, 56, 'o 切片真头数 + _o_proj', 'fused_inv_rope_fp8_quant → fp8_einsum(wo_a) → wo_b')

# ---------------- 左侧两道闸门 ----------------
def gate(y, title, lines, file):
    lc.rect(MX, y, 300, 116, '#ffffff', lc.C_ABORT, rx=9, sw=1.5, dash=True)
    lc.text(MX + 14, y + 20, title, 10.5, lc.C_ABORT, 'start', True, maxw=270, tag='g:' + title[:6])
    for i, s in enumerate(lines):
        lc.text(MX + 14, y + 40 + i * 16, s, 8.5, '#334155', 'start', maxw=272, tag='gl:%s%d' % (title[:4], i))
    lc.text(MX + 14, y + 106, file, 8, lc.C_FAINT, 'start', maxw=272, tag='gf:' + title[:6])


gate(150, '闸门① 阈值门（大批关多流）', [
    'enable = hidden_states.shape[0] <=', 'VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD',
    '（默认 1024）——切换开销赚不回就坍缩单道'], 'attention.py:L454-L456 · envs.py:L277')
gate(286, '闸门② 捕获门（图捕获期禁多流）', [
    'BreakableCUDAGraphCapture.is_active()', '时 aux_stream=None 顺序回退',
    '（maybe/execute_in_parallel 共用判定）'], 'multi_stream_utils.py:L52-L55')
lc.rect(MX, 422, 300, 116, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 14, 442, '一对互相退让的优化', 10.5, lc.C_TXT, 'start', True, maxw=270, tag='rel:t')
for i, s in enumerate([
        '多流 overlap 与 breakable cudagraph', '互相退让：捕获期强制顺序，eager 段',
        '放开并行；ROCm aux_streams=None 恒顺序']):
    lc.text(MX + 14, 462 + i * 16, s, 8.5, '#334155', 'start', maxw=272, tag='rel:l%d' % i)
lc.text(MX + 14, 528, 'attention.py:L493-L497', 8, lc.C_FAINT, 'start', maxw=272, tag='rel:f')

# ---------------- 底部三注 ----------------
NY, NH = 848, 132
lc.rect(MX, NY, 620, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 14, NY + 20, '泳道谱系（3 路 overlap 的出处）', 10.5, lc.C_TXT, 'start', True,
        maxw=400, tag='n1:t')
for i, s in enumerate([
        '· 注释原话 \'matches TRT-LLM PR #14142 Level 1\'',
        '· 3 条 aux 流一枪建在全模型级，注释点名归属：compressor',
        '   kv_score / indexer.weights_proj / indexer.compressor kv_score',
        '   三个轻 GEMM；fused_wqa_wkv（最重）留默认流']):
    lc.text(MX + 14, NY + 40 + i * 17, s, 8.5, '#334155', 'start', maxw=590, tag='n1:l%d' % i)
lc.text(MX + 14, NY + NH - 10, 'nvidia/model.py:L1020-L1024 · attention.py:L403-L459', 8,
        lc.C_FAINT, 'start', maxw=590, tag='n1:f')

lc.rect(700, NY, 430, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(714, NY + 20, 'scratch 池：eager 段不裸分配', 10.5, lc.C_TXT, 'start', True, maxw=380,
        tag='n2:t')
for i, s in enumerate([
        '· eager break 内的分配不经图缓存（捕获图',
        '   假设固定地址），故预分配草稿区',
        '· DeepseekV4EagerScratchPool 四组 256 对齐：',
        '   q / FP4 indexer / global / compressor']):
    lc.text(714, NY + 40 + i * 17, s, 8.5, '#334155', 'start', maxw=400, tag='n2:l%d' % i)
lc.text(714, NY + NH - 10, 'eager_scratch.py:L12-L60 · nvidia/model.py:L1028-L1040', 8,
        lc.C_FAINT, 'start', maxw=400, tag='n2:f')

lc.rect(1154, NY, BXR - 1154, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(1168, NY + 20, 'ln_events：4 个 CUDA 事件', 10.5, lc.C_TXT, 'start', True, maxw=380,
        tag='n3:t')
for i, s in enumerate([
        '· [0] fan-out 起点事件，兼 post-GEMM event0',
        '· [1..3] 各 aux 完成事件；[1] 兼 post-GEMM',
        '   event1——复用安全：GEMM 段先全 join，',
        '   post-GEMM 才起步（attention.py:L301-L304）']):
    lc.text(1168, NY + 40 + i * 17, s, 8.5, '#334155', 'start', maxw=430, tag='n3:l%d' % i)
lc.text(1168, NY + NH - 10, 'attention.py:L317-L321', 8, lc.C_FAINT, 'start', maxw=400, tag='n3:f')

# ---------------- 页脚 ----------------
lc.text(MX, 1006, '图例：深绿条 = 默认流活动 · 浅绿条 = aux 流活动 · 虚框条 = 预留槽位 · 蓝虚横线+圆点 = CUDA event 同步（消息=水平直线）· 琥珀虚框 = eager break 边界',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 1022, '活动条高 = 定性时长（重 GEMM > 轻 GEMM > 归一，非实测刻度）· 锚点 = vllm/models/deepseek_v4/attention.py:L350-L540 · vllm/utils/multi_stream_utils.py:L20-L66',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-multistream-timeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
