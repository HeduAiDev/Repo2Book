#!/usr/bin/env python3
"""ch16 机制图 5 · worker 一拍生命周期（figure_spec ch16-fig-worker-tick，模板 flow）

放大自 L0「GPU 列·执行格」（本章 l0_zoom）、L2 站 8（worker 一拍·逐层收发）。

claim：worker 一拍的生命周期 = 上下文管理器包住前向：bind 计划 → start_load_kv
异步发起 → 前向内逐层 wait_for_layer_load/save_kv_layer → finally wait_for_save
强制同步 + get_finished + 清计划；无 token 步（no_forward）也走收发，但跳过
wait_for_save。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：2 层事件序 7 条；
get_finished 收 finished_req_ids={dead}；no_forward 变体仅 2 条事件、wait_for_save 缺席）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 54
BXR = 1446
MAIN_X, MAIN_W = MX, 900          # 主流程列
SIDE_X = MAIN_X + MAIN_W + 26     # 右栏
SIDE_W = BXR - SIDE_X

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'worker 的一拍：上下文管理器把前向包在中间', 16.5, lc.C_TXT, 'start', True,
        maxw=900, tag='title')
lc.text(MX, 60, 'execute_model 入口先 handle_preemptions（覆写前抢救），随后 _get_kv_connector_output 上下文展开：'
                'bind → start_load_kv 异步发起 → 前向在 yield 处跑 → finally 强制同步收尾', 10.5, lc.C_MUTE,
        'start', maxw=1060, tag='subtitle')
_ch = '放大自 L2 站 8 worker 一拍·逐层收发 · L0：GPU 列·执行格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 步 0：入口 ----------------
S0_Y, S0_H = 100, 56
lc.rect(MAIN_X + 60, S0_Y, MAIN_W - 120, S0_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.5)
lc.text(MAIN_X + 78, S0_Y + 23, 'execute_model 入口 · handle_preemptions', 11.5, lc.C_TXT, 'start', True,
        maxw=MAIN_W - 160, tag='s0:t')
lc.text(MAIN_X + 78, S0_Y + 42, '被抢占请求的块覆写前抢救（本例无抢占，未触发）· gpu_model_runner.py:L4197-L4200', 9,
        lc.C_MUTE, 'start', maxw=MAIN_W - 160, tag='s0:s')

# ---------------- 上下文管理器信封 ----------------
ENV_Y = S0_Y + S0_H + 30
ENV_H = 566
ENV_X, ENV_W = MAIN_X, MAIN_W
lc.rect(ENV_X, ENV_Y, ENV_W, ENV_H, 'none', lc.C_GPU_S, rx=12, sw=1.8, dash=True)
lc.text(ENV_X + 20, ENV_Y - 8, 'maybe_get_kv_connector_output = _get_kv_connector_output(scheduler_output) '
        '@contextmanager · kv_connector_model_runner_mixin.py:L76-L110', 9.5, lc.C_GPU_S, 'start', True,
        maxw=ENV_W - 40, tag='env:t')

STEP_X, STEP_W = ENV_X + 56, ENV_W - 112


def step_box(y, h, tagline, title, sub, fill, stroke, sw=1.5, tcol=None):
    lc.rect(STEP_X, y, STEP_W, h, fill, stroke, rx=8, sw=sw)
    lc.text(STEP_X + 16, y + 21, tagline, 8.5, stroke, 'start', True, maxw=STEP_W - 32, tag=f'{title[:6]}:tag')
    lc.text(STEP_X + 16, y + 40, title, 11.5, tcol or lc.C_TXT, 'start', True, maxw=STEP_W - 32,
            tag=f'{title[:6]}:t')
    if sub:
        lc.text(STEP_X + 16, y + 59, sub, 8.5, lc.C_MUTE, 'start', maxw=STEP_W - 32, tag=f'{title[:6]}:s')


S1_Y = ENV_Y + 22
step_box(S1_Y, 68, '__enter__ ①', 'bind_connector_metadata',
         '把调度器侧的不透明搬运计划交给 worker 侧 connector', '#ffffff', lc.C_GPU_S)
S2_Y = S1_Y + 68 + 22
step_box(S2_Y, 68, '__enter__ ②', 'start_load_kv',
         '第一层注意力之前：全部层的加载异步发起（后台传输与本拍 running 可不相交）', '#ffffff', lc.C_GPU_S)
S3_Y = S2_Y + 68 + 22
FWD_H = 118
lc.rect(STEP_X, S3_Y, STEP_W, FWD_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(STEP_X + 16, S3_Y + 21, 'yield —— 前向跑在这里', 8.5, lc.C_GPU_S, 'start', True, maxw=STEP_W - 32, tag='fwd:tag')
lc.text(STEP_X + 16, S3_Y + 40, '_model_forward（逐层，每层被 maybe_transfer_kv_layer 装饰）', 11.5, lc.C_TXT,
        'start', True, maxw=STEP_W - 32, tag='fwd:t')
lc.text(STEP_X + 16, S3_Y + 60, '层前 wait_for_layer_load(层名) → 注意力本层 → 层后 save_kv_layer(层名, kv_cache)',
        8.5, '#334155', 'start', maxw=STEP_W - 32, tag='fwd:s1')
lc.text(STEP_X + 16, S3_Y + 78, '（kv_transfer_utils.py:L15-L43 装饰器：进函数等本层、出函数存本层）', 8.5, lc.C_MUTE,
        'start', maxw=STEP_W - 32, tag='fwd:s2')
lc.text(STEP_X + 16, S3_Y + 100, '2 层实测：L0 [wait → attn → save] · L1 [wait → attn → save]', 9, lc.C_GPU_S,
        'start', True, maxw=STEP_W - 32, tag='fwd:s3')
S4_Y = S3_Y + FWD_H + 22
step_box(S4_Y, 68, 'finally ③（强制同步点）', 'wait_for_save',
         '不出这个栅栏，paged buffer 可能被下一步覆写——正确性优先于重叠极限', '#ffffff', lc.C_GPU_S)
S5_Y = S4_Y + 68 + 22
step_box(S5_Y, 68, 'finally ④', 'get_finished(finished_req_ids)',
         '带着本拍 finished_req_ids 询问收发完成——实测收到 {dead} 并上报', '#ffffff', lc.C_GPU_S)
S6_Y = S5_Y + 68 + 22
step_box(S6_Y, 68, 'finally ⑤', 'get_block_ids_with_load_errors + clear',
         '失败块上报（本例空集）；get_kv_connector_stats / events / worker_meta；清计划', '#ffffff', lc.C_GPU_S)

# 主线箭头（贴框边：起点=上一框底边中点，终点=下一框顶边中点）
cx = STEP_X + STEP_W / 2
ys = [S0_Y + S0_H, S1_Y, S2_Y, S3_Y, S4_Y, S5_Y, S6_Y]
hs = [S0_H, 68, 68, FWD_H, 68, 68, 68]
seq_ys = [(ys[i] + hs[i], ys[i + 1]) for i in range(6)]
# 入口 → 信封首步：先垂直下到信封顶边再进
lc.seg(cx, ys[0] + hs[0], cx, ys[1], lc.C_GPU_S, 1.8, 'std')
for a, b in seq_ys[1:]:
    lc.seg(cx, a, cx, b, lc.C_GPU_S, 1.8, 'std')

# ---------------- 右栏：no_forward 变体 + 实测事件序 ----------------
NF_Y = S0_Y
lc.rect(SIDE_X, NF_Y, SIDE_W, 210, '#ffffff', lc.C_MUTE, rx=9, sw=1.5, dash=True)
lc.text(SIDE_X + 16, NF_Y + 26, '无 token 步（no_forward 变体）', 12, lc.C_TXT, 'start', True,
        maxw=SIDE_W - 32, tag='nf:t')
lc.text(SIDE_X + 16, NF_Y + 48, '事件只剩 2 条：start_load_kv → get_finished', 9.5, '#334155', 'start',
        maxw=SIDE_W - 32, tag='nf:l1')
lc.text(SIDE_X + 16, NF_Y + 66, 'wait_for_save 缺席——无前向 = 无覆写风险', 9.5, '#334155', 'start',
        maxw=SIDE_W - 32, tag='nf:l2')
lc.text(SIDE_X + 16, NF_Y + 84, '（wait_for_save=False，mixin:L36-L48）', 8.5, lc.C_MUTE, 'start',
        maxw=SIDE_W - 32, tag='nf:l3')
lc.text(SIDE_X + 16, NF_Y + 108, '空拍也走收发：异步传输的请求可以与本拍', 9, lc.C_GPU_S, 'start', True,
        maxw=SIDE_W - 32, tag='nf:l4')
lc.text(SIDE_X + 16, NF_Y + 126, 'running 完全不相交——这一拍只为它们存在', 9, lc.C_GPU_S, 'start', True,
        maxw=SIDE_W - 32, tag='nf:l5')
lc.text(SIDE_X + 16, NF_Y + 150, 'kv_connector_output 直接作为本拍唯一产出', 8.5, lc.C_MUTE, 'start',
        maxw=SIDE_W - 32, tag='nf:l6')
lc.text(SIDE_X + 16, NF_Y + 172, '（ModelRunnerOutput.with_kv_conn_output_only）', 8, lc.C_FAINT, 'start',
        maxw=SIDE_W - 32, tag='nf:l7')

EV_Y = NF_Y + 234
lc.rect(SIDE_X, EV_Y, SIDE_W, 332, '#ffffff', lc.C_GPU_S, rx=9, sw=1.5)
lc.text(SIDE_X + 16, EV_Y + 26, '实测事件序（2 层 · 共 7 条）', 12, lc.C_GPU_S, 'start', True,
        maxw=SIDE_W - 32, tag='ev:t')
EVENTS = [
    ('1', 'start_load_kv', ''),
    ('2', 'wait_for_layer_load', 'l0'),
    ('3', 'save_kv_layer', 'l0'),
    ('4', 'wait_for_layer_load', 'l1'),
    ('5', 'save_kv_layer', 'l1'),
    ('6', 'wait_for_save', ''),
    ('7', 'get_finished', '→ {dead}'),
]
for i, (n, ev, layer) in enumerate(EVENTS):
    ey = EV_Y + 54 + i * 36
    bw = 22
    lc.rect(SIDE_X + 16, ey - 14, bw, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.0)
    lc.text(SIDE_X + 16 + bw / 2, ey, n, 9, lc.C_ENG_S, 'middle', True, maxw=bw - 4, tag=f'ev{i}n')
    lc.text(SIDE_X + 16 + bw + 10, ey, ev, 10, lc.C_TXT, 'start', True, maxw=230, tag=f'ev{i}e')
    if layer:
        lc.text(SIDE_X + SIDE_W - 16, ey, layer, 9, lc.C_MUTE, 'end', maxw=150, tag=f'ev{i}l')
    if i < len(EVENTS) - 1:
        lc.seg(SIDE_X + 16 + bw / 2, ey + 8, SIDE_X + 16 + bw / 2, ey + 20, lc.C_MUTE, 1.2)

# ---------------- 页脚 ----------------
FY = ENV_Y + ENV_H + 34
lc.text(MX, FY, '逐字锚 vllm/v1/worker/gpu_model_runner.py:L4197-L4200（入口 handle_preemptions）· '
                'vllm/v1/worker/kv_connector_model_runner_mixin.py:L36-L48（no_forward）· L76-L110（@contextmanager：bind / start_load_kv / yield / finally）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, 'vllm/model_executor/layers/attention/kv_transfer_utils.py:L15-L43（maybe_transfer_kv_layer 装饰器：层前 wait / 层后 save）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, FY + 32, '事件序 7 条与 no_forward 2 条、get_finished 收 {dead} 取自精简版 companion host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

H = FY + 52

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-worker-tick.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
