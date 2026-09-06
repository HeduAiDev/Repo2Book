#!/usr/bin/env python3
"""ch21 机制图 ④ · 逐层归组与混布三笔账(figure_spec ch21-fig-m08-groups,模板 flow)

放大自 L0 GPU 执行臂(绿)·initialize_kv_cache 一侧的归组(站 6)——把『模型层一摞
Attention 层』按后端拆班组的装配期动作画出来。架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:层按 (后端类全名, KVCacheSpec, num_heads_q) 三元组聚成 AttentionGroup——同一模型
逐层混布不同后端是合法态,代价是每组一套 metadata builder、全模型 cudagraph 档被最弱组
拖累、kernel 块大小要组内协商。

数字全部取自 figure_spec.numbers:3 层→2 组(仅后端类全名不同就分家)、
min_cg_support=UNIFORM_BATCH(2)(刻度 ALWAYS=3>UNIFORM_BATCH=2>UNIFORM_SINGLE_TOKEN_
DECODE=1>NEVER=0)、管理块 256=4×64 协商——本章精简版 host 实测 + pin 源码逐字。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 694
MX = 60
BXR = 1440
C_TRITON_S = '#64748b'          # Triton 桩组:中性灰(不冒领系统角色色)

# ---------------- 标题区 ----------------
lc.text(MX, 34, '拆班组:(后端类全名, KVCacheSpec, num_heads_q) 三元组把层聚成 AttentionGroup',
        16.5, lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '同一模型逐层混布不同后端是合法态——代价三笔:每组一套 metadata builder、全模型 cudagraph 档被最弱组拖累、kernel 块要组内协商;全在装配期一次结清',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(绿)·initialize_kv_cache 归组(站 6)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上:三个层卡 + 三元组核对 ----------------
LAYER_Y, LAYER_H, LAYER_W, LAYER_GAP = 96, 122, 436, 16
layers = [
    ('model.layers.0.self_attn', 'FlashAttentionBackend', lc.C_GPU_S,
     [('后端类全名 ✓', lc.C_GPU_S), ('KVCacheSpec ✓', lc.C_GPU_S), ('num_heads_q=4 ✓', lc.C_GPU_S)]),
    ('model.layers.1.self_attn', 'FlashAttentionBackend', lc.C_GPU_S,
     [('后端类全名 ✓', lc.C_GPU_S), ('KVCacheSpec ✓', lc.C_GPU_S), ('num_heads_q=4 ✓', lc.C_GPU_S)]),
    ('model.layers.2.self_attn', 'TritonStubBackend(TRITON_ATTN 的 host 桩)', C_TRITON_S,
     [('后端类全名 ✗', lc.C_BEAT_S), ('KVCacheSpec ✓ 相同', C_TRITON_S), ('num_heads_q=4 ✓ 相同', C_TRITON_S)]),
]
layer_x = [MX + i * (LAYER_W + LAYER_GAP) for i in range(3)]
for i, (name, backend, bc, tags) in enumerate(layers):
    x = layer_x[i]
    lc.rect(x, LAYER_Y, LAYER_W, LAYER_H, '#ffffff', bc, rx=7, sw=1.5)
    lc.text(x + 14, LAYER_Y + 21, name, 10, lc.C_TXT, 'start', True, maxw=LAYER_W - 28,
            tag=f'ly{i}:n')
    lc.text(x + 14, LAYER_Y + 40, f'后端:{backend}', 8.6, bc, 'start', True,
            maxw=LAYER_W - 28, tag=f'ly{i}:b')
    lc.text(x + 14, LAYER_Y + 58, 'spec:FullAttentionSpec(block_size=256, num_kv_heads=2, head_size=64)',
            7.8, '#334155', 'start', maxw=LAYER_W - 26, tag=f'ly{i}:s')
    tx = x + 14
    for lab, tc in tags:
        tw_ = lc.tw(lab, 7.5) + 12
        lc.rect(tx, LAYER_Y + 74, tw_, 17, '#ffffff', tc, rx=8, sw=1.0)
        lc.text(tx + tw_ / 2, LAYER_Y + 86, lab, 7.5, tc, 'middle', maxw=tw_ - 4, tag=f'ly{i}:t{lab[:4]}')
        tx += tw_ + 8
    if i == 2:
        lc.text(x + 14, LAYER_Y + 110, 'spec 与头数全同也没用——三元组里只要有一项不同,就是新 key → 另开组 B',
                7.8, lc.C_BEAT_S, 'start', maxw=LAYER_W - 26, tag='ly2:note')
    elif i == 0:
        lc.text(x + 14, LAYER_Y + 110, '首见此 key → 开组 A', 7.8, lc.C_MUTE,
                'start', maxw=LAYER_W - 26, tag='ly0:note')
    else:
        lc.text(x + 14, LAYER_Y + 110, '同 key → 并入组 A', 7.8, lc.C_MUTE,
                'start', maxw=LAYER_W - 26, tag='ly1:note')

# ---------------- 中:两个组卡 + 规则注 ----------------
GRP_Y, GRP_H = 262, 96
ga_x, ga_w = MX, 620
gb_x, gb_w = 740, 400
lc.parrow([(layer_x[0] + LAYER_W / 2, LAYER_Y + LAYER_H), (layer_x[0] + LAYER_W / 2, 236),
           (ga_x + 180, 236), (ga_x + 180, GRP_Y)], lc.C_GPU_S, 1.8, 'std')
lc.parrow([(layer_x[1] + LAYER_W / 2, LAYER_Y + LAYER_H), (layer_x[1] + LAYER_W / 2, 244),
           (ga_x + 420, 244), (ga_x + 420, GRP_Y)], lc.C_GPU_S, 1.8, 'std')
lc.parrow([(layer_x[2] + LAYER_W / 2, LAYER_Y + LAYER_H), (layer_x[2] + LAYER_W / 2, GRP_Y)],
          C_TRITON_S, 1.8, 'std')

lc.rect(ga_x, GRP_Y, ga_w, GRP_H, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.8)
lc.text(ga_x + 14, GRP_Y + 21, '组 A = [layers.0, layers.1]  (FlashAttentionBackend)', 10,
        lc.C_GPU_S, 'start', True, maxw=ga_w - 28, tag='ga:t')
lc.text(ga_x + 14, GRP_Y + 40, '一套 metadata builder:FlashAttentionMetadataBuilder', 8.4,
        '#334155', 'start', maxw=ga_w - 28, tag='ga:l1')
lc.text(ga_x + 14, GRP_Y + 57, '组内共享同一份后端专属 metadata(attn_metadata[layer_name] = 同一对象)', 8.4,
        '#334155', 'start', maxw=ga_w - 28, tag='ga:l2')
lc.text(ga_x + 14, GRP_Y + 80, 'builder 档位:UNIFORM_BATCH(2)  ·  声明 kernel 块 [MultipleOf(16)](16 的倍数)', 8,
        lc.C_MUTE, 'start', maxw=ga_w - 28, tag='ga:l3')

lc.rect(gb_x, GRP_Y, gb_w, GRP_H, '#f8fafc', C_TRITON_S, rx=7, sw=1.5)
lc.text(gb_x + 14, GRP_Y + 21, '组 B = [layers.2]  (TritonStubBackend)', 10, C_TRITON_S,
        'start', True, maxw=gb_w - 28, tag='gb:t')
lc.text(gb_x + 14, GRP_Y + 40, '又一套 builder(每组各配一套,不共享)', 8.4, '#334155',
        'start', maxw=gb_w - 28, tag='gb:l1')
lc.text(gb_x + 14, GRP_Y + 57, '同 spec 同头数——但后端类全名不同,只能另立', 8.4, '#334155',
        'start', maxw=gb_w - 28, tag='gb:l2')
lc.text(gb_x + 14, GRP_Y + 80, 'builder 档位:UNIFORM_BATCH(2)  ·  声明 kernel 块 [24, 64]', 8,
        lc.C_MUTE, 'start', maxw=gb_w - 28, tag='gb:l3')

rn_x = gb_x + gb_w + 20
lc.rect(rn_x, GRP_Y, BXR - rn_x, GRP_H, '#ffffff', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(rn_x + 12, GRP_Y + 19, '归组规则(等价类划分)', 8.8, lc.C_TXT, 'start', True,
        maxw=BXR - rn_x - 24, tag='rn:t')
lc.text(rn_x + 12, GRP_Y + 37, '逐层生成 key、append 进同 key 桶:', 8, lc.C_MUTE, 'start',
        maxw=BXR - rn_x - 24, tag='rn:l1')
lc.text(rn_x + 12, GRP_Y + 52, '每层恰属一组,不丢、不重、不串组;', 8, lc.C_MUTE, 'start',
        maxw=BXR - rn_x - 24, tag='rn:l2')
lc.text(rn_x + 12, GRP_Y + 67, '组数 = 不同 key 的个数(本例 3 层 → 2 组)', 8, lc.C_MUTE, 'start',
        maxw=BXR - rn_x - 24, tag='rn:l3')
lc.text(rn_x + 12, GRP_Y + 84, '同一 KV cache 组内照样可拆多组', 8, lc.C_MUTE, 'start',
        maxw=BXR - rn_x - 24, tag='rn:l4')

# ---------------- 下:混布三笔账 ----------------
lc.text(MX, 396, '混布的三笔账(都在装配期一次结清)', 10.5, lc.C_TXT, 'start', True,
        maxw=500, tag='cost:t')
COST_Y, COST_H = 410, 186
c_w, c_gap = 436, 16
cost_x = [MX + i * (c_w + c_gap) for i in range(3)]

# 账 1:每组一套 builder
x = cost_x[0]
lc.rect(x, COST_Y, c_w, COST_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(x + 14, COST_Y + 21, '账一:每组一套 metadata builder', 9.5, lc.C_TXT, 'start', True,
        maxw=c_w - 28, tag='c1:t')
for k, ln in enumerate([
        '2 个组 = 2 套 builder,各译各的后端专属单;',
        '组内多层共享同一份(两个层名指向同一对象),',
        '只有跨组才需要第二份;',
        '不混布则全模型一套——这是混布的直接开销。']):
    lc.text(x + 14, COST_Y + 44 + k * 17, ln, 8.4, '#334155', 'start', maxw=c_w - 26,
            tag=f'c1:l{k}')
lc.text(x + 14, COST_Y + 122, '3 层 → 2 个 attn group(1 个 KV cache 组内)', 8.4, lc.C_GPU_S,
        'start', True, maxw=c_w - 26, tag='c1:n')

# 账 2:最弱链 CG
x = cost_x[1]
lc.rect(x, COST_Y, c_w, COST_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.5)
lc.text(x + 14, COST_Y + 21, '账二:最弱链——cudagraph 档取最弱组', 9.5, lc.C_BEAT_T,
        'start', True, maxw=c_w - 28, tag='c2:t')
ladder = [('NEVER = 0', False), ('UNIFORM_SINGLE_TOKEN_DECODE = 1', False),
          ('UNIFORM_BATCH = 2 ← 本例两组皆此', True), ('ALWAYS = 3', False)]
for k, (lab, hot) in enumerate(ladder):
    y_ = COST_Y + 36 + k * 17
    lw_ = 60 + k * 34
    lc.rect(x + 14, y_, lw_, 15, (lc.C_BEAT_S if hot else '#ffffff'), lc.C_BEAT_S,
            rx=3, sw=1.0)
    lc.text(x + 14 + lw_ + 8, y_ + 11.5, lab, 8, (lc.C_BEAT_T if hot else lc.C_MUTE),
            'start', maxw=c_w - 40 - lw_, tag=f'c2:s{k}')
lc.text(x + 14, COST_Y + 112, 'min(UNIFORM_BATCH, UNIFORM_BATCH) = UNIFORM_BATCH(2)', 8.4,
        lc.C_BEAT_T, 'start', True, maxw=c_w - 26, tag='c2:m')
lc.text(x + 14, COST_Y + 130, '一个只支持「等长批」的后端,就把全模型 FULL 档', 8.2, '#334155',
        'start', maxw=c_w - 26, tag='c2:n1')
lc.text(x + 14, COST_Y + 146, 'capture 压到均匀批——档位刻度:3>2>1>0。', 8.2, '#334155',
        'start', maxw=c_w - 26, tag='c2:n2')
lc.text(x + 14, COST_Y + 166, '注:观测值 2 受 host 恒 FA2 影响(真机 SM90+FA3 的 FA builder 应为 ALWAYS)',
        7.4, lc.C_MUTE, 'start', maxw=c_w - 26, tag='c2:h')

# 账 3:kernel 块协商
x = cost_x[2]
lc.rect(x, COST_Y, c_w, COST_H, '#ffffff', lc.C_KV_S, rx=7, sw=1.4)
lc.text(x + 14, COST_Y + 21, '账三:kernel 块大小组内协商', 9.5, lc.C_KV_S, 'start', True,
        maxw=c_w - 28, tag='c3:t')
bar_y, bar_h = COST_Y + 36, 24
bar_w = c_w - 28
for k in range(4):
    bx = x + 14 + k * (bar_w / 4)
    lc.rect(bx, bar_y, bar_w / 4, bar_h, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
    lc.text(bx + bar_w / 8, bar_y + 16, '64', 8.5, lc.C_KV_S, 'middle', True,
            maxw=bar_w / 4 - 4, tag=f'c3:b{k}')
lc.text(x + 14, bar_y + bar_h + 14, '管理块 256 token = 4 × 64(拆成 4 个 kernel 块)', 8.4,
        lc.C_KV_S, 'start', True, maxw=c_w - 26, tag='c3:bar')
for k, ln in enumerate([
        '候选:FA [MultipleOf(16)](16 的倍数) ∩ Triton 桩 [24, 64];',
        '24 不能整除 256,64 能整除、最大、又是 16 的倍数(FA ✓) → 64;',
        'KV 逻辑形随之定为 (256, 2, 64, 128):',
        '256 个 kernel 块 = 64 管理块 × 4 细分(形状详见上一图)。']):
    lc.text(x + 14, COST_Y + 92 + k * 17, ln, 8.4, '#334155', 'start', maxw=c_w - 26,
            tag=f'c3:l{k}')

# ---------------- 页脚 ----------------
FY = COST_Y + COST_H + 24
lc.text(MX, FY, '图例:绿 = FlashAttentionBackend 组 · 灰 = Triton 桩组(host 无 vllm 包,桩经 register_backend 注册——真实第三方注册机制) · 橙 = 最弱链代价 · 青 = KV 张量协商结果',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/v1/attention/backend.py:L606-L620(CG 档刻度 ALWAYS/UNIFORM_BATCH/UNIFORM_SINGLE_TOKEN_DECODE/NEVER)· vllm/v1/worker/utils.py:L266-L376(kernel 块协商)· 分组/档位/协商数值 = 本章精简版 host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch21-fig-m08-groups.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
