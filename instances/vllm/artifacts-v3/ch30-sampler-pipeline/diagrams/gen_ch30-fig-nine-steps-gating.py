#!/usr/bin/env python3
"""ch29 机制图 1 · 9 步骨架与批级门控（figure_spec ch29-fig-nine-steps-gating，模板 swimlane）

放大自 L0 采样出口列（L2 章图 cartography/l2-specs/ch29.json center 拍片 ①-⑨）的
「执行形态」层：上泳道 = 批级 python 门控（站号与 L2 章图一致，站 3-12），下泳道 =
按需下沉的 kernel（GPU 角色绿）与两块整段留在 python 的块（约束源在 CPU）。
不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：采样列的 9 步不是一个 kernel——每步 = 批级 python 门控 + 按需 kernel：
只有计算密集子步骤（top-k/top-p 截断、FlashInfer 拒绝采样、Exp 噪声掷骰）下沉核；
bad_words 前缀匹配、惩罚计数（约束源在 CPU 的 python list）等门控整段留在 python。

数字全部取自本章 explainer 素材（m12 分流/m13 绑定/m05 封禁计数/m03 早退）与
pin 源码 file:L 锚点；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX = 60
BXR = 1440
GRID = '#e2e8f0'

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>'
    f'<marker id="gpu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>')

lc.text(MX, 34, '9 步不是一个 kernel：每步 = 批级 python 门控 + 按需下沉 kernel',
        16.5, lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, '门控留 python 的三层原因：批内请求异构（greedy/随机、有无惩罚/bad_words 混批，一行式无法按行分派）· '
               '约束源在 CPU（惩罚吃 python list 历史、bad_words 是 dict 前缀匹配）· 计算密集子步骤才下沉核',
        10, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L0 采样出口列 · L2 章图拍片 ①-⑨ 的「执行形态」层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 上泳道：批级 python 门控 =================
UY, UH = 88, 180
lc.rect(MX, UY, 1380, UH, lc.C_SAM_F, lc.C_SAM_S, rx=10, sw=2.0)
lc.text(MX + 16, UY + 18, '上泳道 · 批级 python 门控 —— sampler.py 编排，每步先判「本批有没有请求需要」，有才动手',
        9.5, lc.C_SAM_S, 'start', True, maxw=760, tag='lane:up')

GATE_Y, GATE_H, GATE_W, PITCH = 122, 118, 140, 152
GATES = [
    ('①', 'raw 留底', '站3', ['有 logprobs 请求才', 'log_softmax / clone', '留底（一切变换之前）']),
    ('②', 'fp32', '站4', ['logits.to(float32)', '无条件；此后全程', '原地改写（契约）']),
    ('③', 'allowed', '站5', ['allowed_token_ids_mask', '非 None 才动手', 'masked_fill_(-inf)']),
    ('④', 'bad_words', '站6', ['bad_words 非空才循环', '逐请求纯 python：', '会补全禁语才封末位']),
    ('⑤', '非不变列', '站7', ['两列分类的非不变列', '列表非空才 apply', 'MinTokens / LogitBias']),
    ('⑥', '惩罚', '站8', ['no_penalties=False 才算', 'repetition/freq/', 'presence + thinking budget']),
    ('⑦', 'sample', '站9', ['all_greedy / all_random', '/ 混合三分：温度', '→min_p→top-k/p→掷骰']),
    ('⑧', 'gather', '站11', ['有 logprobs 请求才', 'topk + 被采样 logprob', '+ rank 计数']),
    ('⑨', '出件', '站12', ['int64 统一', '→ int32 [B,1]', '+ logprobs_tensors']),
]
gate_cx = []
for i, (sym, name, badge, lines) in enumerate(GATES):
    x = 72 + i * PITCH
    cx = x + GATE_W / 2
    gate_cx.append(cx)
    lc.rect(x, GATE_Y, GATE_W, GATE_H, '#ffffff', lc.C_SAM_S, rx=7, sw=1.3)
    lc.text(x + 10, GATE_Y + 19, sym + ' ' + name, 9.8, lc.C_TXT, 'start', True,
            maxw=GATE_W - 54, tag='g' + sym)
    bw = 15 + 9 * len(badge)
    lc.rect(x + GATE_W - bw - 7, GATE_Y + 6, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
    lc.text(x + GATE_W - bw / 2 - 7, GATE_Y + 18.5, badge, 8.3, lc.C_ENG_S, 'middle', True,
            maxw=bw - 4, tag='gb' + sym)
    for j, ln in enumerate(lines):
        lc.text(x + 10, GATE_Y + 40 + j * 16, ln, 7.6, '#334155', 'start',
                maxw=GATE_W - 18, tag='gl' + sym + str(j))
    if i < 8:
        lc.seg(x + GATE_W, GATE_Y + 59, x + PITCH, GATE_Y + 59, lc.C_SAM_S, 1.6, 'sam')

# all_greedy 早退旁路：⑦ 顶部 → ⑧ 顶部（7a argmax 即返回，7b-7f 跳过）
lc.parrow([(gate_cx[6], GATE_Y), (gate_cx[6], GATE_Y - 14), (gate_cx[7], GATE_Y - 14),
           (gate_cx[7], GATE_Y)], lc.C_ENG_S, 1.6, 'up')
lc.text((gate_cx[6] + gate_cx[7]) / 2, GATE_Y - 20, 'all_greedy 早退：7a 即返回', 7.8,
        lc.C_ENG_S, 'middle', True, maxw=300, tag='bypass')

# ================= 下泳道 A：下沉 GPU kernel =================
KY, KH = 296, 336
lc.rect(MX, KY, 940, KH, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(MX + 16, KY + 20, '下泳道 A · 下沉 GPU kernel —— 只有计算密集子步骤（实线下探 = 按需调用）',
        10.5, lc.C_GPU_S, 'start', True, maxw=880, tag='lane:k')


def kernel_chip(x, y, w, h, title, badge, lines, stroke=None, fill='#ffffff'):
    stroke = stroke or lc.C_GPU_S
    lc.rect(x, y, w, h, fill, stroke, rx=6, sw=1.3)
    lc.text(x + 10, y + 18, title, 9.3, lc.C_TXT, 'start', True, maxw=w - 58, tag='k:' + title[:8])
    bw = 15 + 8.6 * len(badge)
    lc.rect(x + w - bw - 7, y + 5, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
    lc.text(x + w - bw / 2 - 7, y + 17.5, badge, 8, lc.C_ENG_S, 'middle', True,
            maxw=bw - 4, tag='kb:' + title[:6])
    for j, ln in enumerate(lines):
        lc.text(x + 10, y + 37 + j * 15.5, ln, 7.6, '#334155', 'start', maxw=w - 18,
                tag='kl' + title[:6] + str(j))


K1Y, K1H, K1W, K1GAP = KY + 34, 112, 220, 8
kernel_chip(76, K1Y, K1W, K1H, 'masked_fill_(-inf)', '③',
            ['allowed 白名单整行一次 O(V)', 'mask 极性：True=禁位', '（gpu_input_batch L282-283）'])
kernel_chip(76 + (K1W + K1GAP), K1Y, K1W, K1H, 'index_put_ / 稀疏 +=', '⑤',
            ['MinTokens 封 stop/EOS 写 -inf', 'LogitBias 稀疏坐标 +=', 'O(1) 个坐标、不扫全表'])
kernel_chip(76 + 2 * (K1W + K1GAP), K1Y, K1W, K1H, 'scatter_add_ 计数', '⑥',
            ['output=[2,2,3]', '→ bin_counts=[0,0,2,1,0]', '一遍数出每 token 出现次数'])
kernel_chip(76 + 3 * (K1W + K1GAP), K1Y, K1W, K1H, 'topk + rank 计数', '⑧',
            ['rank=(x>=v).sum(-1)', '不排序数名次', 'O(V) 计数替代 O(V log V)'])
K2Y, K2H, K2W = K1Y + K1H + 12, 122, 296
kernel_chip(76, K2Y, K2W, K2H, 'Triton pivot 截断核', '⑦d',
            ['分位查表 + 三分搜索逼近 pivot，免排序全词表（Qrita）',
             '分流谓词 HAS_TRITON and shape[0]>=8',
             '实测：批 6 → pytorch sort · 批 16 → 核'])
kernel_chip(76 + K2W + K1GAP, K2Y, K2W, K2H, 'FlashInfer 拒绝采样', '⑦d+⑦e·融合',
            ['构造期绑 forward_cuda（默认 raw 模式）',
             '拒绝采样：免整词表排序、统计等价',
             'processed 两态强制 forward_native'])
kernel_chip(76 + 2 * (K2W + K1GAP), K2Y, K2W, K2H, 'exponential_ + argmax', '⑦e',
            ['q.exponential_() 后 probs/q 取 argmax',
             'Gumbel-max 掷骰',
             '替代会强制 CPU-GPU 同步的 multinomial'])
lc.text(MX + 16, KY + KH - 22, '⑦ 的截断/掷骰 kernel 即第 10 站（TopKTopPSampler，⑦ 内部深处）；批<8 或无 Triton 时 ⑦d 走 apply_top_k_top_p_pytorch（pytorch sort 教学主实现）',
        8.3, lc.C_MUTE, 'start', maxw=900, tag='k:foot')

# ================= 下泳道 B：整段留在 python =================
PY = KY
lc.rect(1020, PY, 420, KH, '#f8fafc', lc.C_MUTE, rx=10, sw=1.8)
lc.text(1036, PY + 20, '下泳道 B · 整段留在 python —— 约束源在 CPU（无下探箭头）',
        10.5, lc.C_MUTE, 'start', True, maxw=390, tag='lane:p')


def plain_chip(x, y, w, h, title, lines):
    lc.rect(x, y, w, h, '#ffffff', lc.C_MUTE, rx=6, sw=1.2)
    lc.text(x + 10, y + 18, title, 9.3, lc.C_TXT, 'start', True, maxw=w - 20, tag='p:' + title[:8])
    for j, ln in enumerate(lines):
        lc.text(x + 10, y + 37 + j * 15.5, ln, 7.6, '#334155', 'start', maxw=w - 18,
                tag='pl' + title[:6] + str(j))


plain_chip(1036, PY + 34, 388, 132, '④ bad_words 逐请求循环',
           ['每请求 × 每禁短语：前缀匹配',
            '会补全成被禁短语才封末 token=-inf',
            '实测：4 禁短语 4 分支判定，',
            '2 封（51/53）2 不封（52/54）',
            'slice 赋值避免 cpu→gpu sync'])
plain_chip(1036, PY + 182, 388, 132, '⑥ 前置 · 惩罚张量化（H2D）',
           ['惩罚吃 python list 历史（prompt+output）',
            'make_tensor_with_pad → 按需 H2D 变张量',
            'output=[2,2,3] 先成张量再进核计数',
            '（快照只在批组成变化时重造）',
            '注释自注 quite inefficient'])
lc.text(1036, PY + KH - 12, '语法 FSM 活在调度器进程（→ ch30/31）同理不出 CPU',
        8.3, lc.C_MUTE, 'start', maxw=390, tag='p:foot')

# ================= 门控步 → kernel 下探箭头 =================
lc.seg(gate_cx[2], GATE_Y + GATE_H, gate_cx[2], KY, lc.C_GPU_S, 1.8, 'gpu')
lc.seg(gate_cx[4], GATE_Y + GATE_H, gate_cx[4], KY, lc.C_GPU_S, 1.8, 'gpu')
lc.seg(gate_cx[5], GATE_Y + GATE_H, gate_cx[5], KY, lc.C_GPU_S, 1.8, 'gpu')
lc.parrow([(gate_cx[6], GATE_Y + GATE_H), (gate_cx[6], 276), (962, 276), (962, KY)],
          lc.C_GPU_S, 1.8, 'gpu')
lc.parrow([(gate_cx[7], GATE_Y + GATE_H), (gate_cx[7], 286), (866, 286), (866, KY)],
          lc.C_GPU_S, 1.8, 'gpu')

# ================= 图例行 =================
LY = 648
lx0 = MX
for fill, stroke, name in [(lc.C_SAM_F, lc.C_SAM_S, '上泳道门控步（站号与 L2 章图一致）'),
                           (lc.C_GPU_F, lc.C_GPU_S, '下沉 GPU kernel'),
                           ('#f8fafc', lc.C_MUTE, '整段留在 python')]:
    lc.rect(lx0, LY - 9, 16, 11, fill, stroke, rx=3, sw=1.5)
    lc.text(lx0 + 21, LY + 1, name, 9, lc.C_TXT, 'start')
    lx0 += 21 + lc.tw(name, 9) + 20
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_GPU_S, 2.0, 'gpu')
lc.text(lx0 + 40, LY + 1, '实线下探 = 按需调用', 9, lc.C_TXT, 'start')
lx0 += 40 + lc.tw('实线下探 = 按需调用', 9) + 20
lc.text(lx0, LY + 1, '无下探箭头 = 无专门下沉的计算 kernel（轻量原地操作 / 纯 python 循环在步内完成）',
        9, lc.C_MUTE, 'start', maxw=BXR - lx0, tag='legend:tail')

# ================= 底部实证条 =================
EY, EH, EW = 676, 128, 452
EVID = [
    ('分流实证（本章 GPU 容器实测）',
     ['同一族 logits：批 6 走 pytorch sort、批 16 走 Triton pivot 核；',
      '分流只看批大小与 Triton 可用性（HAS_TRITON and shape[0]>=8），',
      '与 k/p 无关——两路幸存集逐行一致、有限值零差']),
    ('greedy 快路径（本章驱动实测）',
     ['all_greedy 早退时温度没跑（temperature_ran=False）、',
      '调用方 logits 未被改写——贪心整批跳过',
      '温度 / min_p / top-k / top-p']),
    ('后端下沉（本章 GPU 容器实测）',
     ['默认 raw 模式构造期绑 forward_cuda（FlashInfer 拒绝采样）；',
      'processed 两态强制 forward_native',
      '——FlashInfer 拿不到截断后中间量']),
]
for i, (h, lines) in enumerate(EVID):
    x = MX + i * (EW + 12)
    lc.rect(x, EY, EW, EH, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
    lc.text(x + 14, EY + 20, h, 9.5, lc.C_TXT, 'start', True, maxw=EW - 28, tag='ev' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x + 14, EY + 41 + j * 17, ln, 8.3, '#334155', 'start', maxw=EW - 26,
                tag='evl' + str(i) + str(j))

# ================= 页脚锚点 =================
lc.text(MX, 836, 'vllm/v1/sample/sampler.py:L371-L417（③-⑥ 门控）· L243-L302（⑦ sample；docstring L20-L59 即 9 步目录）· L309-L356（⑧）· L139-L149（⑨）',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft1')
lc.text(MX, 856, 'vllm/v1/sample/ops/topk_topp_sampler.py（⑦ 分流与掷骰）· ops/bad_words.py:L9-L36 · ops/penalties.py:L10-L56 · 分流/绑定/封禁计数数值＝本章驱动脚本与 GPU 容器实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft2')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-nine-steps-gating.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
