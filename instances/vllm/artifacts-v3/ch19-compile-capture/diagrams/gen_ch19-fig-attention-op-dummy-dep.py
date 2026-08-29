#!/usr/bin/env python3
"""ch19 机制图 3 · 副作用与保序：空回执（figure_spec ch19-fig-attention-op-dummy-dep，模板 tensor-flow）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ⑧ 拍片
『喂图·算子与回放』（站 14，一拍⑤算子化前向）内 Attention.forward 框的机制展开。
架构归属回指 L0/L2：右上角指北小签。

claim：attention 前向拆成两个自定义算子加一张空回执：KV 写独立成算子、其返回的
空张量作 dummy 数据依赖传入注意力算子——副作用能进图，且顺序不被编译器重排。

数字/引语全部取自 figure_spec.numbers（attention.py:L829-L832 保序注释原话 ·
L531-L541 NOTE(woosuk) op 外预分配 · L817-L846 op 内从 context 取环境 ·
回执空张量实测 numel=0/shape=[0] · 每层每拍 2 次 dispatch · fake_impl）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 668
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '两个算子加一张空回执：副作用进图、先后交给编译器保管',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'KV 写拆成独立算子 unified_kv_cache_update、返回 key.new_empty(0) 作 dummy 数据依赖传入注意力算子；本体做成 out-variant（attention.py:L488-L846）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑧ 喂图·算子与回放 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主张量流 ----------------
TOPY = 96
# q/k/v 三条张量线
CHIPS = ['query', 'key', 'value']
CX, CW_, CH = MX, 104, 34
for i, nm in enumerate(CHIPS):
    cy = 130 + i * 44
    lc.rect(CX, cy, CW_, CH, '#ffffff', lc.C_MUTE, rx=6, sw=1.2)
    lc.text(CX + CW_ / 2, cy + 21, nm, 9, lc.C_TXT, 'middle', True, maxw=CW_ - 8, tag='tk' + nm)

# op 外框（Python 侧 eager）
OX, OY, OW, OH = 204, 118, 216, 164
lc.rect(OX, OY, OW, OH, '#ffffff', lc.C_BEAT_S, rx=8, sw=1.6)
lc.text(OX + 12, OY + 20, 'op 外（Python 侧）', 9.5, lc.C_BEAT_T, 'start', True,
        maxw=OW - 24, tag='opx:t')
OPL = ['output = torch.empty(output_shape)', 'q = q.view(-1, heads, head_dim) …',
       'NOTE(woosuk)：We do this outside', 'the custom op to minimize the CPU',
       'overheads from the non-CUDA-graph', 'regions.（L531-L541 原话）']
for i, ln in enumerate(OPL):
    lc.text(OX + 12, OY + 40 + i * 17, ln, 7.8, '#334155', 'start', maxw=OW - 24,
            tag='opx:l' + str(i))
lc.text(OX + 12, OY + OH - 10, '预分配 + reshape 都刻意留在 op 外', 7.6, lc.C_BEAT_T,
        'start', maxw=OW - 24, tag='opx:f')

# 三条张量箭头 → op 外（端点落在框左缘）
for i in range(3):
    cy = 130 + i * 44
    lc.parrow([(CX + CW_, cy + 17), (OX, 148 + i * 34)], lc.C_MUTE, 1.4)

# KV 写算子框
KX, KY, KW, KH = 476, TOPY, 316, 150
lc.rect(KX, KY, KW, KH, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.8)
lc.text(KX + 14, KY + 22, 'torch.ops.vllm.unified_kv_cache_update', 9.3, lc.C_BEAT_T, 'start',
        True, maxw=KW - 28, tag='kv:t')
lc.text(KX + 14, KY + 38, '(key, value, layer_name) → Tensor', 8, '#334155', 'start',
        maxw=KW - 28, tag='kv:sig')
KVL = ['· op 内 get_attention_context 取 kv_cache + slot_mapping',
       '· attn_layer.impl.do_kv_cache_update(…) 写入 KV cache',
       '· 返回 key.new_empty(0) —— 一张空回执',
       '（attention.py:L775-L798）']
for i, ln in enumerate(KVL):
    lc.text(KX + 14, KY + 58 + i * 17, ln, 7.9, '#334155', 'start', maxw=KW - 28,
            tag='kvl' + str(i))

# attention 算子框
AX, AY, AW, AH = 836, TOPY, 344, 166
lc.rect(AX, AY, AW, AH, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(AX + 14, AY + 20, '@eager_break_during_capture @maybe_transfer_kv_layer', 7.2,
        lc.C_MUTE, 'start', maxw=AW - 28, tag='at:dec')
lc.text(AX + 14, AY + 38, 'torch.ops.vllm.unified_attention_with_output', 9.3, lc.C_GPU_S,
        'start', True, maxw=AW - 28, tag='at:t')
lc.text(AX + 14, AY + 54, '(query, key, value, output, layer_name, kv_cache_dummy_dep)',
        7.6, '#334155', 'start', maxw=AW - 28, tag='at:sig')
ATL = ['· 签名里只有一个层名，其余执行环境全部从 context 来',
       '· attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)',
       '· self.impl.forward(self, q, k, v, kv_cache, attn_metadata,',
       '  output=output, …)——out-variant：结果写进预分配 output',
       '（attention.py:L817-L846）']
for i, ln in enumerate(ATL):
    lc.text(AX + 14, AY + 74 + i * 17, ln, 7.6, '#334155', 'start', maxw=AW - 28,
            tag='atl' + str(i))

# 主横向箭头：op 外 → kv 写（key, value）
lc.parrow([(OX + OW, 160), (KX, 160)], lc.C_BEAT_S, 1.6)
lc.text((OX + OW + KX) / 2, 152, 'key, value', 7.8, lc.C_BEAT_T, 'middle', True, maxw=60,
        tag='a1')

# 绕行箭头：op 外 → attention（q,k,v + output）
lc.parrow([(OX + OW, 226), (OX + OW + 14, 226), (OX + OW + 14, 300), (796, 300), (796, 226),
           (AX, 226)], lc.C_GPU_S, 1.5)
lc.text(608, 292, 'q, k, v（view 后）+ 预分配 output', 7.8, lc.C_GPU_S, 'middle', maxw=200,
        tag='a2')

# KV cache 副作用框
VX, VY, VW, VH = 476, 330, 190, 56
lc.rect(VX, VY, VW, VH, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(VX + VW / 2, VY + 24, 'KV cache', 9.5, lc.C_KV_S, 'middle', True, maxw=VW - 12,
        tag='kvb:t')
lc.text(VX + VW / 2, VY + 42, '写副作用（块页物理槽位）', 7.4, lc.C_MUTE, 'middle',
        maxw=VW - 12, tag='kvb:l')
lc.parrow([(VX + 40, KY + KH), (VX + 40, VY)], lc.C_KV_S, 1.6)
lc.text(VX + 48, (KY + KH + VY) / 2 + 3, '写', 8, lc.C_KV_S, 'start', True, maxw=30, tag='a3')

# 空回执票
TX, TY, TW, TH = 706, 330, 200, 56
lc.rect(TX, TY, TW, TH, '#ffffff', lc.C_BEAT_S, rx=6, sw=1.4, dash=True)
lc.text(TX + TW / 2, TY + 20, '空回执 key.new_empty(0)', 8.4, lc.C_BEAT_T, 'middle', True,
        maxw=TW - 10, tag='tkt:t')
lc.text(TX + TW / 2, TY + 38, 'numel=0 · shape=[0] · dtype 同 key', 7.4, lc.C_MUTE, 'middle',
        maxw=TW - 10, tag='tkt:l')
lc.parrow([(TX + 50, KY + KH), (TX + 50, TY)], lc.C_BEAT_S, 1.6, dash=True)
lc.parrow([(TX + TW, TY + 20), (AX + 130, AY + AH)], lc.C_BEAT_S, 1.6, dash=True)
lc.text(TX + TW + 10, TY + 44, 'kv_cache_dummy_dep：只搬先后、不搬数据', 7.6, lc.C_BEAT_T,
        'start', maxw=200, tag='a4')

# output 出线
lc.rect(1220, 140, 116, 44, '#ffffff', lc.C_GPU_S, rx=7, sw=1.5)
lc.text(1220 + 58, 158, 'output', 9.5, lc.C_GPU_S, 'middle', True, maxw=100, tag='ob')
lc.text(1220 + 58, 174, '（预分配、被填）', 7.2, lc.C_MUTE, 'middle', maxw=100, tag='obl')
lc.parrow([(AX + AW, 162), (1220, 162)], lc.C_GPU_S, 1.8)

# ---------------- 保序注释原话条 ----------------
QY = 420
lc.rect(MX, QY, BXR - MX, 62, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 16, QY + 22, '保序注释原话（attention.py:L829-L832）：kv_cache_dummy_dep is not used but accepting it creates a data dependency that ensures torch.compile preserves',
        8.6, '#334155', 'start', maxw=BXR - MX - 32, tag='q1')
lc.text(MX + 16, QY + 42, 'ordering between KV cache update and attention forward.——回执不搬数据，只向编译器声明『写 KV 必须先于算 attention』。',
        8.6, '#334155', 'start', maxw=BXR - MX - 32, tag='q2')

# ---------------- 底部三卡 ----------------
CY, CH2 = 506, 108
cards = [
    (MX, 424, '每层每拍 2 次 op dispatch', lc.C_BEAT_S, lc.C_BEAT_F,
     ['KV 写与 attention 各一次——forward_includes_kv_cache_update',
      '语义分叉：基类默认 True（v1/attention/backend.py:L67）、',
      'FlashAttentionBackend 改 False（flash_attn.py:L86）。']),
    (MX + 448, 424, 'fake_impl 让 Dynamo 可 trace', lc.C_GPU_S, lc.C_GPU_F,
     ['unified_attention_with_output_fake / unified_kv_cache_update_fake',
      '（attention.py:L801-L806 · L849-L859）：trace 期走 fake 实现，',
      '真副作用只在 eager 执行时发生。']),
    (MX + 896, BXR - MX - 896, '空回执实测（精简版 companion host 实跑）', lc.C_MUTE, '#ffffff',
     ['numel=0 · shape=[0] · dtype 与 key 一致（真实返回 key.new_empty(0)）；',
      '它一个字都不写——图里这道边的权重是『先后』而非数据。',
      'GPU 专属量（捕获/回放路径）不实测，引用源码注释锚点。']),
]
for cx, cwd, t, st, fl, lines in cards:
    lc.rect(cx, CY, cwd, CH2, fl, st, rx=8, sw=1.4)
    lc.text(cx + 14, CY + 22, t, 9.5, st, 'start', True, maxw=cwd - 28, tag='c:' + t[:10])
    for i, ln in enumerate(lines):
        lc.text(cx + 14, CY + 44 + i * 18, ln, 8.0, '#334155', 'start', maxw=cwd - 28,
                tag='cl:' + t[:8] + str(i))

# ---------------- 页脚 ----------------
lc.text(MX, 646, '逐字锚 vllm/model_executor/layers/attention/attention.py:L531-L541（op 外 NOTE）· L775-L798（KV 写算子+空回执）· L817-L846（注意力算子+保序注释）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-attention-op-dummy-dep.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
