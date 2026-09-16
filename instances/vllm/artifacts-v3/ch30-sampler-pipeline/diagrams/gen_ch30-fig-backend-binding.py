#!/usr/bin/env python3
"""ch29 机制图 4 · TopKTopPSampler 构造期后端绑定（figure_spec ch29-fig-backend-binding，模板 flow）

放大自 L0 采样出口列（L2 章图 south『TopKTopPSampler·构造期绑后端』支撑块）——
它是拍片 ⑦d『Apply top_k and/or top_p』在 CUDA 上的执行体。
不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：TopKTopPSampler 在 __init__ 一次性把 self.forward 绑定到具体后端——CUDA 且
FlashInfer 可用（env+算力）且 logprobs_mode 非 processed 两态 → forward_cuda，否则
forward_native；运行期零平台分支。forward_cuda 内部再按 (k,p)==(None,None) /
逐请求 generator / use_fp64_gumbel 三条件递回 forward_native。

数字全部取自本章 explainer 素材（m13 容器实测：constructor_binding /
forward_cuda_fallbacks）与 pin 源码锚点；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 792
MX = 60
BXR = 1440
GRID = '#e2e8f0'

DEFS = lc.DEFS + (
    f'<marker id="gpu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
    f'<marker id="mut" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_MUTE}"/></marker>')

lc.text(MX, 34, '构造期一次绑定、运行期零平台分支——__init__ 把 self.forward 定死',
        16.5, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, 'CUDA 且 FlashInfer 可用（env+算力）且 logprobs_mode 非 processed 两态 → forward_cuda（拒绝采样核）；'
               '否则 forward_native。绑定发生在进程启动期，decode 热路径上没有一次平台判断',
        10, lc.C_MUTE, 'start', maxw=1300, tag='subtitle')
_ch = '放大自 L0 采样出口列 · L2 章图 south「TopKTopPSampler·构造期绑后端」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 构造期泳道（采样列角色色底） =================
LK_X, LK_Y, LK_W, LK_H = MX, 92, 920, 480
lc.rect(LK_X, LK_Y, LK_W, LK_H, lc.C_SAM_F, lc.C_SAM_S, rx=10, sw=2.0)
lc.text(LK_X + 16, LK_Y + 22, '构造期（进程启动 · __init__）', 11.5, lc.C_SAM_S, 'start', True,
        maxw=420, tag='lk:t')
lc.text(LK_X + LK_W - 14, LK_Y + 22, 'topk_topp_sampler.py:L85-L129', 8.5, lc.C_FAINT, 'end',
        tag='lk:w')

# __init__ 顶节点
IT_X, IT_Y, IT_W, IT_H = 110, 132, 420, 56
lc.rect(IT_X, IT_Y, IT_W, IT_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.5)
lc.text(IT_X + 14, IT_Y + 22, 'TopKTopPSampler.__init__(logprobs_mode, use_fp64_gumbel)', 9.4,
        lc.C_TXT, 'start', True, maxw=IT_W - 28, tag='it:t')
lc.text(IT_X + 14, IT_Y + 41, '一次性绑定 self.forward = 具体方法（非每次调用现判）', 8.4,
        '#334155', 'start', maxw=IT_W - 28, tag='it:l')

# 三层判定（是/否分支；meas=实测值行，绿色粗体）


def decision(y, h, q, sublines, meas=None):
    x, w = 170, 300
    lc.rect(x, y, w, h, '#ffffff', lc.C_SAM_S, rx=7, sw=1.5)
    lc.text(x + 14, y + 20, q, 9.2, lc.C_TXT, 'start', True, maxw=w - 28, tag='d:' + q[:8])
    for i, s in enumerate(sublines):
        lc.text(x + 14, y + 37 + i * 14.5, s, 7.5, '#334155', 'start', maxw=w - 28,
                tag='ds:' + q[:6] + str(i))
    if meas:
        lc.text(x + 14, y + h - 9, meas, 7.6, lc.C_GPU_S, 'start', True, maxw=w - 28,
                tag='dm:' + q[:6])


D1_Y, D1_H = 216, 66
D2_Y, D2_H = 298, 84
D3_Y, D3_H = 398, 64
decision(D1_Y, D1_H, '① current_platform.is_cuda()？',
         ['平台判定——非 CUDA 走各自平台位'], '实测 is_cuda=True')
decision(D2_Y, D2_H, '② FlashInfer 可用？',
         ['VLLM_USE_FLASHINFER_SAMPLER + 算力门槛 SM80-SM121',
          '（flashinfer_sampler_supported L28-L74）'], '实测 capability=sm_120 · supported=True')
decision(D3_Y, D3_H, '③ logprobs_mode ∈ PROCESSED 两态？',
         ['processed_logits / processed_logprobs',
          '——FlashInfer 拿不到截断后中间量'])

# 平台变体角标（灰、点名不展开）
PV_X, PV_Y, PV_W, PV_H = 620, 216, 340, 50
lc.rect(PV_X, PV_Y, PV_W, PV_H, '#f8fafc', GRID, rx=7, sw=1.0, dash=True)
lc.text(PV_X + 14, PV_Y + 20, '非 CUDA 平台变体（本图不展开）', 8.4, lc.C_MUTE, 'start', True,
        maxw=PV_W - 28, tag='pv:t')
lc.text(PV_X + 14, PV_Y + 37, 'CPU / XPU / ROCm aiter 各有绑定；RISCV/POWERPC 等落 forward_native',
        7.4, lc.C_MUTE, 'start', maxw=PV_W - 28, tag='pv:l')

# 两个绑定终态
FC_X, FC_Y, FC_W, FC_H = 620, 306, 340, 64
lc.rect(FC_X, FC_Y, FC_W, FC_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=2.0)
lc.text(FC_X + 14, FC_Y + 22, '绑定 self.forward = forward_cuda', 9.6, lc.C_GPU_S, 'start', True,
        maxw=FC_W - 28, tag='fc:t')
lc.text(FC_X + 14, FC_Y + 41, '默认 raw 模式 · FlashInfer 拒绝采样核', 8.2, '#334155', 'start',
        maxw=FC_W - 28, tag='fc:l')
FN_X, FN_Y, FN_W, FN_H = 620, 404, 340, 64
lc.rect(FN_X, FN_Y, FN_W, FN_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.8)
lc.text(FN_X + 14, FN_Y + 22, '绑定 self.forward = forward_native', 9.6, lc.C_TXT, 'start', True,
        maxw=FN_W - 28, tag='fn:t')
lc.text(FN_X + 14, FN_Y + 41, 'PyTorch 原生：sort 截断 + Gumbel 掷骰', 8.2, '#334155', 'start',
        maxw=FN_W - 28, tag='fn:l')

# 判定边
lc.seg(320, IT_Y + IT_H, 320, D1_Y, lc.C_MUTE, 1.8, 'mut')
lc.seg(320, D1_Y + D1_H, 320, D2_Y, lc.C_MUTE, 1.8, 'mut')
lc.text(328, D1_Y + D1_H + 13, '是', 8.4, lc.C_MUTE, 'start', True, tag='e:y1')
lc.seg(320, D2_Y + D2_H, 320, D3_Y, lc.C_MUTE, 1.8, 'mut')
lc.text(328, D2_Y + D2_H + 13, '是', 8.4, lc.C_MUTE, 'start', True, tag='e:y2')
# ①否 → 平台变体
lc.seg(470, D1_Y + 25, PV_X, D1_Y + 25, lc.C_MUTE, 1.8, 'mut')
lc.text(476, D1_Y + 19, '否', 8.4, lc.C_MUTE, 'start', True, tag='e:n1')
# ②否 → forward_native
lc.parrow([(470, D2_Y + 40), (560, D2_Y + 40), (560, FN_Y + 18), (FN_X, FN_Y + 18)],
          lc.C_MUTE, 1.8, 'mut')
lc.text(476, D2_Y + 34, '否', 8.4, lc.C_MUTE, 'start', True, tag='e:n2')
# ③否（raw 模式）→ forward_cuda
lc.parrow([(470, D3_Y + 20), (540, D3_Y + 20), (540, FC_Y + 44), (FC_X, FC_Y + 44)],
          lc.C_GPU_S, 1.8, 'gpu')
lc.text(476, D3_Y + 14, '否（raw 模式）', 8.4, lc.C_GPU_S, 'start', True, maxw=120, tag='e:n3')
# ③是（processed 两态）→ forward_native
lc.parrow([(470, D3_Y + 46), (575, D3_Y + 46), (575, FN_Y + 50), (FN_X, FN_Y + 50)],
          lc.C_MUTE, 1.8, 'mut')
lc.text(476, D3_Y + 60, '是（processed 两态）', 8.4, lc.C_MUTE, 'start', True, maxw=140,
        tag='e:y3')
# ② 的裁决语义（泳道底注）
lc.text(LK_X + 16, LK_Y + LK_H - 44, '② 的裁决语义：默认态不可用 → 静默回退 native 并告警；用户显式设 '
        'VLLM_USE_FLASHINFER_SAMPLER=1 但算力不支持 → 直接 RuntimeError', 8.2, lc.C_MUTE,
        'start', maxw=880, tag='lk:n1')
lc.text(LK_X + 16, LK_Y + LK_H - 26, '（「显式开但不可用」按字面抛错——构造期就暴露问题，不留到 decode 热路径）',
        8.2, lc.C_MUTE, 'start', maxw=880, tag='lk:n2')

# ================= 运行期泳道（decode 热路径） =================
RT_X, RT_Y, RT_W, RT_H = 1000, 92, 440, 480
lc.rect(RT_X, RT_Y, RT_W, RT_H, '#ffffff', GRID, rx=10, sw=1.6)
lc.text(RT_X + 14, RT_Y + 22, '运行期（decode 每步调用 self.forward）', 11.5, lc.C_TXT,
        'start', True, maxw=400, tag='rt:t')

# forward_cuda 三道回退守卫
GV_X, GV_Y, GV_W, GV_H = 1014, 132, 412, 150
lc.rect(GV_X, GV_Y, GV_W, GV_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(GV_X + 12, GV_Y + 20, 'forward_cuda 三道回退守卫（L159-L168）', 9.6, lc.C_GPU_S,
        'start', True, maxw=GV_W - 24, tag='gv:t')
GUARDS = [
    'k 与 p 全 None → 无过滤可做 → native',
    'generators 非空 → FlashInfer 0.2.3+ 不支持 → native',
    'use_fp64_gumbel=True → native（L167-L168）',
]
for j, g in enumerate(GUARDS):
    lc.text(GV_X + 12, GV_Y + 42 + j * 17, '· ' + g, 7.8, '#334155', 'start', maxw=GV_W - 24,
            tag='gvl' + str(j))
lc.text(GV_X + 12, GV_Y + GV_H - 10, '实测：前两例命中回退；正常 k 调用不经 native', 7.6,
        lc.C_GPU_S, 'start', True, maxw=GV_W - 24, tag='gv:m')

# flashinfer_sample
FI_X, FI_Y, FI_W, FI_H = 1014, 318, 412, 62
lc.rect(FI_X, FI_Y, FI_W, FI_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.6)
lc.text(FI_X + 12, FI_Y + 21, 'flashinfer_sample(logits, k, p)（L475-L512）', 9.2, lc.C_GPU_S,
        'start', True, maxw=FI_W - 24, tag='fi:t')
lc.text(FI_X + 12, FI_Y + 40, '拒绝采样：免整词表排序 · deterministic=True · 与 Gumbel 统计等价', 7.8,
        '#334155', 'start', maxw=FI_W - 24, tag='fi:l')

# native 运行期体
NT_X, NT_Y, NT_W, NT_H = 1014, 416, 412, 62
lc.rect(NT_X, NT_Y, NT_W, NT_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.5)
lc.text(NT_X + 12, NT_Y + 21, 'forward_native 运行期体', 9.2, lc.C_TXT, 'start', True,
        maxw=NT_W - 24, tag='nt:t')
lc.text(NT_X + 12, NT_Y + 40, 'apply_top_k_top_p（sort/pivot 分流）+ random_sample Gumbel 掷骰', 7.8,
        '#334155', 'start', maxw=NT_W - 24, tag='nt:l')

# 构造期 → 运行期 的两条延续
lc.seg(FC_X + FC_W, FC_Y + 32, GV_X, FC_Y + 32, lc.C_GPU_S, 2.0, 'gpu')
lc.text(984, 310, 'decode 调用', 7.8, lc.C_GPU_S, 'start', maxw=110, tag='e:call')
lc.seg(FN_X + FN_W, NT_Y + 30, NT_X, NT_Y + 30, lc.C_MUTE, 1.5, 'mut')

# 运行期内部箭头
lc.parrow([(GV_X + 90, GV_Y + GV_H), (GV_X + 90, FI_Y)], lc.C_GPU_S, 1.8, 'gpu')
lc.text(GV_X + 96, GV_Y + GV_H + 13, '正常 k/p 调用直达', 7.8, lc.C_GPU_S, 'start', maxw=160,
        tag='e:fi')
lc.parrow([(GV_X + 300, GV_Y + GV_H), (GV_X + 300, 298), (1433, 298), (1433, NT_Y + 31),
           (NT_X + NT_W, NT_Y + 31)], lc.C_MUTE, 1.8, 'mut')
lc.text(GV_X + 294, 294, '三道守卫回退', 7.8, lc.C_MUTE, 'end', maxw=140, tag='e:nt')
lc.text(RT_X + 14, RT_Y + RT_H - 14, '运行期零平台分支——平台判断全在左栏构造期完成', 8.2,
        lc.C_MUTE, 'start', maxw=400, tag='rt:f')

# ================= 底部实证条 =================
EY, EH, EW = 596, 118, 452
EVID = [
    ('构造期绑定实测（本章 GPU 容器）',
     ['is_cuda=True · capability=sm_120 · supported=True',
      '→ 默认 raw 模式绑 forward_cuda；processed 两态',
      '（processed_logits / processed_logprobs）强制 forward_native']),
    ('运行期回退实测（本章 GPU 容器）',
     ['k/p 全 None → forward_native（FlashInfer 无事可做）',
      'generators 非空 → forward_native（0.2.3+ 不支持）',
      '正常 k 调用直达 flashinfer_sample、不经 native']),
    ('能力裁决语义（flashinfer.py:L461-L470）',
     ['算力门槛 SM80-SM121 逐字镜像（本机实测 sm_120）',
      '默认态不可用 → 静默回退；显式开但不可用',
      '→ RuntimeError（构造期抛错，不留到热路径）']),
]
for i, (h, lines) in enumerate(EVID):
    x = MX + i * (EW + 12)
    lc.rect(x, EY, EW, EH, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
    lc.text(x + 14, EY + 20, h, 9.5, lc.C_TXT, 'start', True, maxw=EW - 28, tag='ev' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x + 14, EY + 41 + j * 17, ln, 8.2, '#334155', 'start', maxw=EW - 26,
                tag='evl' + str(i) + str(j))

# ================= 页脚锚点 =================
lc.text(MX, 748, 'vllm/v1/sample/ops/topk_topp_sampler.py:L85-L129（__init__ 绑定）· L155-L182（forward_cuda 回退守卫）'
        '· L28-L74（flashinfer_sampler_supported）· L475-L512（flashinfer_sample）· v1/attention/backends/flashinfer.py:L461-L470（算力门槛）',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft1')
lc.text(MX, 768, 'sm_120 / 绑定名 / 回退命中＝本章 GPU 容器实测（RTX PRO 6000 Blackwell）· 行号基线 vLLM v0.27.1 · '
        '机制图：构造期泳道用采样列角色色、CUDA 核终态用 GPU 角色绿', 8.5, lc.C_FAINT, 'start',
        maxw=1380, tag='ft2')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-backend-binding.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
