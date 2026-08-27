#!/usr/bin/env python3
"""ch12 机制图 1 · 启动装配的三级间接链（figure_spec ch12-fig-step-binding-chain，模板 flow）

放大自 L0 循环框（loop_box）的入口装配段——即本章 L2 章图 north『启动装配 · 默认即重叠』框
的机制展开：从配置位 async_scheduling=None 到心跳函数 step_fn 绑定的一条链。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：从 async_scheduling=None 到 step_fn=step_with_batch_queue 要走三级间接——
标志仲裁成 True → 深度算出 2 → 建 deque(maxlen=2) → 绑定只看队列建没建
（读绑定代码看不到 async 字样）。

数字全部取自 figure_spec.numbers（深度矩阵五案例：async+V1+pp1→2 · V2+pp4→5 ·
V1+async+pp4→4 · 纯PP pp4→4 · 无async无PP→1；默认链 scheduler_cls=AsyncScheduler ·
queue_built=True · maxlen=2 · step_fn=step_with_batch_queue；pooling 降级链
queue_built=False · step_fn=step；五类降级 pooling/medusa/eagle/
disable_padded_drafter_batch/ray，显式 True+ray 硬失败 ValueError）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 748
MX, BXR = 60, 1440
C_KV_DEEP = '#0e7490'   # 深度数字（青系深）
C_DOWN = '#b45309'      # 仲裁降为 False（amber 深）

# ---------------- 标题区 ----------------
lc.text(MX, 34, '启动装配的三级间接：在 step_fn 的绑定代码里搜 async_scheduling 必然扑空',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'EngineCore 不看标志、看队列建没建（core.py:L231-L233）——『标志 True → 深度 2 → '
        'deque 建立 → 绑重叠版』，第一环被故意藏起来；深度仲裁（vllm.py:L539-L550）才是唯一出图纸的地方',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 启动装配 · L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 四级链（顶部） ----------------
CHAIN_Y, CHAIN_H = 96, 104
N_BOX, BOX_W, BOX_GAP = 4, 322, 26
BX0 = MX
STAGES = [
    ('① 标志仲裁', lc.C_KV_S, lc.C_KV_F,
     ['async_scheduling = None（缺省）',
      '默认仲裁（vllm.py:L1095-L1143）',
      '生成服务 → True；get_scheduler_cls',
      '→ AsyncScheduler（config/scheduler.py:L170-L178）']),
    ('② 深度算出', lc.C_ENG_S, lc.C_ENG_F,
     ['max_concurrent_batches property',
      '（vllm.py:L539-L550，三输入纯函数）',
      '默认链 async+V1+单 PP → 2',
      '（一个工位在算、一个在备料）']),
    ('③ 建队列', lc.C_ENG_S, lc.C_ENG_F,
     ['EngineCore.__init__（core.py:L206-L212）',
      'batch_queue_size > 1 才建',
      'batch_queue = deque(maxlen=2)',
      '建后连心跳函数都换掉']),
    ('④ 绑心跳', lc.C_ENG_S, lc.C_ENG_F,
     ['step_fn 静态绑定（core.py:L231-L233）',
      'batch_queue is not None',
      '→ step_with_batch_queue',
      '绑定代码里没有 async 字样']),
]
box_cx = []
for i, (title, stroke, fill, lines) in enumerate(STAGES):
    x = BX0 + i * (BOX_W + BOX_GAP)
    box_cx.append(x + BOX_W / 2)
    lc.rect(x, CHAIN_Y, BOX_W, CHAIN_H, fill, stroke, rx=7, sw=1.5)
    lc.text(x + 14, CHAIN_Y + 22, title, 11, stroke, 'start', True, maxw=BOX_W - 28, tag='st' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x + 14, CHAIN_Y + 42 + j * 16, ln, 8.8, '#334155', 'start',
                maxw=BOX_W - 26, tag='st' + str(i) + 'l' + str(j))
LINKS = ['仲裁成 True', '算出深度 2', '队列建没建？']
for i in range(3):
    x0 = BX0 + (i + 1) * BOX_W + i * BOX_GAP
    lc.seg(x0 + 2, CHAIN_Y + CHAIN_H / 2, x0 + BOX_GAP - 2, CHAIN_Y + CHAIN_H / 2,
           lc.C_ENG_S, 2.0, 'std')
    lc.text(x0 + BOX_GAP / 2 + 1, CHAIN_Y + CHAIN_H / 2 - 10, LINKS[i], 8.2, lc.C_MUTE,
            'middle', maxw=BOX_GAP + 60, tag='lk' + str(i))

# ---------------- 绑定代码逐字条 ----------------
CODE_Y = 222
lc.rect(MX, CODE_Y, 1380, 62, '#ffffff', lc.C_ENG_S, rx=7, sw=1.4)
lc.text(MX + 16, CODE_Y + 21, '绑定代码全文（core.py:L231-L233 逐字）——条件是 batch_queue 是否存在，不是 async 标志：',
        9.5, lc.C_TXT, 'start', True, maxw=900, tag='code:head')
lc.text(MX + 16, CODE_Y + 41, "self.step_fn = ( self.step if self.batch_queue is None else self.step_with_batch_queue )",
        10, lc.C_ENG_S, 'start', True, maxw=940, tag='code:body', )
lc.text(MX + 16, CODE_Y + 56, 'v2 读者按旧代码找 executor.max_concurrent_batches 也必扑空——深度出处已从 executor 各自声明上移 VllmConfig（vllm.py:L540），EngineCore 直读 core.py:L206',
        8.5, lc.C_MUTE, 'start', maxw=1330, tag='code:mig')

# ---------------- 左panel：深度矩阵 ----------------
PM_Y, PM_H = 312, 262
PW_L = 660
lc.rect(MX, PM_Y, PW_L, PM_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(MX + 16, PM_Y + 22, '深度矩阵——max_concurrent_batches = f(async, runner 代际, pp_size)，五案例实测',
        10, lc.C_TXT, 'start', True, maxw=PW_L - 30, tag='pmL:t')
COLS = [42, 52, 50, 40]
CX0 = MX + 22
HDRS = ['async', 'pp_size', 'runner', '深度', '含义']
HR_Y = PM_Y + 42
for j, (cw, htxt) in enumerate(zip(COLS, HDRS)):
    lc.text(CX0 + sum(COLS[:j]) + cw / 2, HR_Y, htxt, 8.8, lc.C_MUTE, 'middle', True,
            maxw=cw - 4, tag='hd' + htxt)
lc.seg(MX + 10, HR_Y + 7, MX + PW_L - 10, HR_Y + 7, '#e2e8f0', 1.0)
MATRIX = [
    ('True', '1', 'V1', '2', '双缓冲：一拍在 GPU 算、一拍在 CPU 调度（默认链）', True),
    ('True', '4', 'V2', '5', 'pp_size+1：消末段气泡', False),
    ('True', '4', 'V1', '4', '落 pp_size（V1 不完全支持 async+PP）', False),
    ('False', '4', 'V1', '4', '纯 PP：填流水线', False),
    ('False', '1', 'V1', '1', '不建队列 → step_fn=step（同步版）', False),
]
ROW_H = 36
for i, (a, pp, run, depth, note, hot) in enumerate(MATRIX):
    ry = HR_Y + 18 + i * ROW_H
    mid = ry + ROW_H / 2
    if hot:
        lc.rect(MX + 10, ry, PW_L - 20, ROW_H, lc.C_BEAT_F, 'none', rx=5, sw=0)
    vals = [a, pp, run, depth]
    for j, (cw, v) in enumerate(zip(COLS, vals)):
        col = lc.C_ENG_S if (j == 3 and hot) else (lc.C_TXT if j < 3 else C_KV_DEEP)
        lc.text(CX0 + sum(COLS[:j]) + cw / 2, mid + 3.5, v, 9.5, col, 'middle', True,
                maxw=cw - 4, tag='mx' + str(i) + str(j))
    nx = CX0 + sum(COLS[:4]) + 10
    lc.text(nx, mid + 3.5, note, 8.6, '#334155', 'start', maxw=PW_L - (nx - MX) - 30,
            tag='mxn' + str(i))

# ---------------- 右panel：默认仲裁与五类降级 ----------------
PW_R = 1380 - PW_L - 24
PX_R = MX + PW_L + 24
lc.rect(PX_R, PM_Y, PW_R, PM_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(PX_R + 16, PM_Y + 22, '默认仲裁与五类降级（None → True；显式 True 不做静默降级）',
        10, lc.C_TXT, 'start', True, maxw=PW_R - 30, tag='pmR:t')
DOWNGRADES = [
    ('pooling 模型', 'False', '异步反而拖慢 → 降级链 queue_built=False · step_fn=step'),
    ('spec=medusa（非 EAGLE 系）', 'False', '非 EAGLE 系 spec 方法不支持'),
    ('spec=eagle（EAGLE 系）', 'True', 'EAGLE 系仍默认开（AsyncScheduler）'),
    ('disable_padded_drafter_batch=True', 'False', 'padded drafter 批与占位调度互斥'),
    ('executor=ray（supports_async_scheduling 未覆写）', 'False', '后端不支持 → Scheduler'),
]
for i, (cond, res, note) in enumerate(DOWNGRADES):
    ry = HR_Y + 18 + i * 26
    col = lc.C_GPU_S if res == 'True' else C_DOWN
    lc.text(PX_R + 16, ry + 4, cond, 8.6, '#334155', 'start', maxw=330, tag='dg' + str(i))
    lc.rect(PX_R + 356, ry - 7, 36, 15, '#ffffff', col, rx=7, sw=1.2)
    lc.text(PX_R + 374, ry + 4, res, 8.5, col, 'middle', True, maxw=32, tag='dgv' + str(i))
    lc.text(PX_R + 404, ry + 4, note, 8.6, lc.C_MUTE, 'start', maxw=PW_R - 404 - 16,
            tag='dgn' + str(i))
VF_Y = HR_Y + 18 + 5 * 26 + 6
lc.rect(PX_R + 16, VF_Y, PW_R - 32, 34, '#fef2f2', lc.C_ABORT, rx=6, sw=1.2)
lc.text(PX_R + 28, VF_Y + 14, '显式 async=True + ray → ValueError：`ray` does not support async scheduling yet.',
        8.6, lc.C_ABORT, 'start', True, maxw=PW_R - 56, tag='vf1')
lc.text(PX_R + 28, VF_Y + 27, '（vllm.py:L1091-L1094——显式要求不做静默降级，直接硬失败）',
        8.2, lc.C_MUTE, 'start', maxw=PW_R - 56, tag='vf2')

# ---------------- 底部结论横幅 ----------------
BN_Y = 598
lc.rect(MX, BN_Y, 1380, 36, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(MX + 690, BN_Y + 22.5,
        '数字大于 1 才造缓冲架、架上架子以后连心跳函数都换掉——深度是三输入的纯函数，装配后终身不变（队列 maxlen 建后固定）',
        10.5, lc.C_BEAT_T, 'middle', True, maxw=1360, tag='banner')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 662
lx = MX
for kind, name in [('kv', '标志仲裁（配置期）'), ('eng', '深度/装配/绑定（EngineCore 启动期）'),
                   ('hot', '默认链（本图高亮）'), ('down', '仲裁降为 False'), ('fail', '硬失败 ValueError')]:
    if kind == 'kv':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
    elif kind == 'eng':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.4)
    elif kind == 'hot':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.4)
    elif kind == 'down':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', C_DOWN, rx=3, sw=1.2)
        lx -= 0
    else:
        lc.rect(lx, LEG_Y - 9, 20, 12, '#fef2f2', lc.C_ABORT, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=260, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 18

lc.text(MX, 700, '逐字锚 vllm/config/vllm.py:L539-L550（深度）/ L1095-L1143（默认仲裁）/ L1091-L1094（硬失败）· '
        'vllm/config/scheduler.py:L170-L178（get_scheduler_cls）· vllm/v1/engine/core.py:L206-L212, L231-L233 · '
        '深度矩阵与降级链取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')
lc.text(MX, 718, 'v0.21 → v0.27.1 迁移：深度从 executor 各自 @cached_property 声明上移 VllmConfig 唯一出处——'
        '旧代码入口已换，按 v0.21 行号找 executor 属性会扑空（git 证据链）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-step-binding-chain.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
