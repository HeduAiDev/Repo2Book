#!/usr/bin/env python3
"""ch21 机制图 ⑥ · builder.build() 翻译(figure_spec ch21-fig-m11-translate,模板 before-after)

放大自 L0 GPU 执行臂(绿)·每拍心跳「builder 翻译」一格(站 9)——Common 与后端专属
metadata 的对照摊开。架构归属回指 L0(FIGURE-SYSTEM §3.3),不另立第二种架构画法。

claim:builder.build() 的「翻译」= Common 字段改名搬入(block_table_tensor→block_table、
query_start_loc/seq_lens/slot_mapping 直搬)+ 后端特有字段补算(FA:scheduler_metadata/
use_cascade/sliding_window)——同 (spec, builder) 的混合组走 update_block_table 浅拷只换表。

数字全部取自 figure_spec.numbers:改名值 [[0,1],[2,3],[0,0],[0,0]](尾 2 行 [9,9]→NULL_
BLOCK_ID=0)、FA 特有字段三值、fanout m[L0] is m[L1]、组 1 换表值——本章精简版 host 实测
+ pin 源码逐字。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 706
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '报关单改封面:Builder 把通用 Common 单,翻译成后端自己的单子',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '共享字段改名搬入、后端特有字段补算——翻译不改字段值,只换容器与名字;组内两个层名共享同一对象,第二个 KV 组同格式不重填整单,浅拷换表即用',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(绿)·每拍心跳「builder 翻译」(站 9)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

CARD_Y, CARD_H = 96, 336
ROW_Y = [196, 226, 256, 286]          # 四行对齐字段(qsl/seq/slot/block_table)

# ---------------- 左:before = CommonAttentionMetadata ----------------
LX, LW = MX, 380
lc.rect(LX, CARD_Y, LW, CARD_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(LX + 16, CARD_Y + 24, 'CommonAttentionMetadata', 10.5, lc.C_GPU_S, 'start', True,
        maxw=250, tag='l:t')
lc.text(LX + LW - 14, CARD_Y + 24, 'backend.py:L412-L444', 8.5, lc.C_FAINT, 'end',
        maxw=140, tag='l:file')
lc.text(LX + 16, CARD_Y + 42, '国际标准报关单——所有后端都认这份', 8.2, lc.C_MUTE,
        'start', maxw=LW - 30, tag='l:sub')
left_rows = [('query_start_loc', '[0, 4, 6, 6, 6]'),
             ('seq_lens', '[16, 20, 16, 20]'),
             ('slot_mapping', '[12,13,14,15,50,51,-1,-1]'),
             ('block_table_tensor', '[[0,1],[2,3],[0,0],[0,0]]')]
for k, (name, val) in enumerate(left_rows):
    y = ROW_Y[k]
    lc.text(LX + 16, y, name, 8.6, lc.C_TXT, 'start', True, maxw=150, tag=f'l:k{k}')
    lc.text(LX + 148, y, val, 8.2, '#334155', 'start', maxw=222, tag=f'l:v{k}')
    if k == 3:
        lc.rect(LX + 8, y - 16, LW - 16, 26, 'none', lc.C_BEAT_S, rx=4, sw=1.2, dash=True)
lc.text(LX + 16, 330, 'causal=True 等其余共享字段同搬', 8, lc.C_MUTE, 'start',
        maxw=LW - 30, tag='l:n1')
lc.text(LX + 16, 348, '(共十字段,全貌见上一图;值 = 本章精简版 host 实测一拍)', 8, lc.C_MUTE,
        'start', maxw=LW - 26, tag='l:n2')

# ---------------- 中:翻译官 + 三种搬法 ----------------
BD_X, BD_W = 480, 260
lc.rect(BD_X, CARD_Y, BD_W, 84, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.6)
lc.text(BD_X + 12, CARD_Y + 20, 'FlashAttentionMetadataBuilder', 9.5, lc.C_GPU_S, 'start',
        True, maxw=BD_W - 24, tag='bd:t')
lc.text(BD_X + 12, CARD_Y + 37, '.build(common_prefix_len, common_attn_metadata)', 7.6,
        '#334155', 'start', maxw=BD_W - 22, tag='bd:sig')
lc.text(BD_X + 12, CARD_Y + 53, '翻译官:不改字段值,只换容器与名字', 8, lc.C_MUTE, 'start',
        maxw=BD_W - 24, tag='bd:n1')
lc.text(BD_X + 12, CARD_Y + 72, 'vllm/v1/attention/backends/flash_attn.py:L672-L696', 7.2,
        lc.C_FAINT, 'start', maxw=BD_W - 22, tag='bd:file')

R_X = 760
for k in range(3):                        # 直搬 ×3
    lc.seg(LX + LW, ROW_Y[k] - 3, R_X, ROW_Y[k] - 3, lc.C_GPU_S, 1.5, 'std')
lc.text((LX + LW + R_X) / 2, ROW_Y[0] - 12, '直搬 ×3(同张量别名,零拷贝)', 8, lc.C_GPU_S,
        'middle', True, maxw=320, tag='m:d')
lc.seg(LX + LW, ROW_Y[3] - 3, R_X, ROW_Y[3] - 3, lc.C_BEAT_S, 2.2, 'std')
lc.text((LX + LW + R_X) / 2, ROW_Y[3] - 12, '改名:block_table_tensor → block_table', 8,
        lc.C_BEAT_S, 'middle', True, maxw=320, tag='m:r')
nf_x, nf_y, nf_w, nf_h = BD_X - 10, 316, BD_W + 20, 46
lc.rect(nf_x, nf_y, nf_w, nf_h, '#ffffff', lc.C_BEAT_S, rx=6, sw=1.1, dash=True)
lc.text(nf_x + nf_w / 2, nf_y + 19, '尾 2 行由 [9,9] 填成 NULL_BLOCK_ID=0', 7.8,
        lc.C_BEAT_T, 'middle', True, maxw=nf_w - 12, tag='m:nf1')
lc.text(nf_x + nf_w / 2, nf_y + 35, '(块 0 保留给 padding;进 builder 之前已由 runner 填好)', 7.6,
        lc.C_MUTE, 'middle', maxw=nf_w - 12, tag='m:nf2')

# ---------------- 右:after = FlashAttentionMetadata ----------------
RW = BXR - R_X
lc.rect(R_X, CARD_Y, RW, CARD_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(R_X + 16, CARD_Y + 24, 'FlashAttentionMetadata', 10.5, lc.C_GPU_S, 'start', True,
        maxw=250, tag='r:t')
lc.text(R_X + RW - 14, CARD_Y + 24, 'flash_attn.py:L672-L696', 8.5, lc.C_FAINT, 'end',
        maxw=180, tag='r:file')
lc.text(R_X + 16, CARD_Y + 42, 'FA 自己的单子——容器与字段名换了,值没换', 8.2, lc.C_MUTE,
        'start', maxw=RW - 30, tag='r:sub')
right_rows = [('query_start_loc', '[0, 4, 6, 6, 6]', '与 Common 同一张量(别名)'),
              ('seq_lens', '[16, 20, 16, 20]', '同张量'),
              ('slot_mapping', '[12,13,14,15,50,51,-1,-1]', '同张量'),
              ('block_table', '[[0,1],[2,3],[0,0],[0,0]]', '改名搬入')]
for k, (name, val, note) in enumerate(right_rows):
    y = ROW_Y[k]
    lc.text(R_X + 16, y, name, 8.6, (lc.C_BEAT_T if k == 3 else lc.C_TXT), 'start', True,
            maxw=140, tag=f'r:k{k}')
    lc.text(R_X + 152, y, val, 8.2, (lc.C_BEAT_T if k == 3 else '#334155'), 'start', True,
            maxw=230, tag=f'r:v{k}')
    lc.text(R_X + RW - 16, y, note, 7.8, lc.C_MUTE, 'end', maxw=250, tag=f'r:n{k}')
FB_Y, FB_H = 318, 100
lc.rect(R_X + 12, FB_Y, RW - 24, FB_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=6, sw=1.2, dash=True)
lc.text(R_X + 24, FB_Y + 18, 'FA 特有(Builder 补算,Common 单里没有)', 8.6, lc.C_BEAT_T,
        'start', True, maxw=RW - 48, tag='fb:t')
for k, ln in enumerate([
        'scheduler_metadata = None(host 恒 FA2;真机 FA3 才有 AOT 调度)',
        'use_cascade = False   ·   sliding_window = (-1, -1)',
        'num_decode_reqs = num_prefill_reqs = 0']):
    lc.text(R_X + 24, FB_Y + 38 + k * 17, ln, 8.2, '#334155', 'start', maxw=RW - 48,
            tag=f'fb:l{k}')

# ---------------- 底部:铺设 fanout + 组 1 换表复用 ----------------
BB_Y, BB_H = CARD_Y + CARD_H + 22, 178
# (a) fanout
lc.rect(MX, BB_Y, 660, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 14, BB_Y + 20, '铺设:组内两个层名 → 同一对象', 10, lc.C_TXT, 'start', True,
        maxw=400, tag='a:t')
tag_w, tag_h = 216, 24
for k, nm in enumerate(['model.layers.0.self_attn', 'model.layers.1.self_attn']):
    ty = BB_Y + 34 + k * 34
    lc.rect(MX + 14, ty, tag_w, tag_h, '#ffffff', lc.C_GPU_S, rx=5, sw=1.2)
    lc.text(MX + 14 + tag_w / 2, ty + 16, nm, 8, lc.C_GPU_S, 'middle', maxw=tag_w - 8,
            tag=f'a:tag{k}')
    lc.parrow([(MX + 14 + tag_w, ty + tag_h / 2), (MX + 300, BB_Y + 62), (MX + 328, BB_Y + 62)],
              lc.C_GPU_S, 1.5, 'std')
ob_x, ob_w = MX + 332, 250
lc.rect(ob_x, BB_Y + 38, ob_w, 48, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.5)
lc.text(ob_x + ob_w / 2, BB_Y + 57, 'FlashAttentionMetadata', 9, lc.C_GPU_S, 'middle', True,
        maxw=ob_w - 12, tag='a:ob')
lc.text(ob_x + ob_w / 2, BB_Y + 74, '(组 A 内唯一实例)', 8, lc.C_MUTE, 'middle',
        maxw=ob_w - 12, tag='a:ob2')
lc.text(MX + 14, BB_Y + 112, 'fanout:attn_metadata[layer_name] = 同一对象——组内全部层读到的就是同一份',
        8.4, '#334155', 'start', maxw=630, tag='a:n1')
lc.text(MX + 14, BB_Y + 132, '实测:两个层名的取值 is 同一对象 = true(共享字段零重复)', 8.4,
        lc.C_GPU_S, 'start', True, maxw=630, tag='a:n2')
lc.text(MX + 14, BB_Y + 156, '第二个层名不触发第二次 build——铺设只是往 dict 里再放一个键', 8,
        lc.C_MUTE, 'start', maxw=630, tag='a:n3')

# (b) 组 1 换表复用
b_x = MX + 680
b_w = BXR - b_x
lc.rect(b_x, BB_Y, b_w, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.4, dash=True)
lc.text(b_x + 14, BB_Y + 20, '第二个 KV 组(组 1):不重填整单,浅拷换表', 10, lc.C_TXT,
        'start', True, maxw=520, tag='b:t')
for k, ln in enumerate([
        '前提:同 (KVCacheSpec, builder 类型) 已建过,且 builder.supports_update_block_table',
        'update_block_table:浅拷已建单,只换两张表——',
        '  block_table = [[4,5],[6,7],[0,0],[0,0]]   slot_mapping = [76,77,78,79, 114,115, -1,-1]',
        'layers.2 → m′:非 m,但共享 query_start_loc 同一张量——共享字段零重算',
        '对比:不走缓存,组 1 要再来一次完整 build(共享字段全部重算一遍)']):
    c = lc.C_BEAT_S if k == 2 else ('#334155' if k < 4 else lc.C_MUTE)
    lc.text(b_x + 14, BB_Y + 42 + k * 18, ln, 8.4, c, 'start', True if k == 2 else False,
            maxw=b_w - 26, tag=f'b:l{k}')
lc.text(b_x + 14, BB_Y + 152, '本例每拍账单:1 次完整 build 翻译 + 1 次浅拷换表', 8,
        lc.C_MUTE, 'start', maxw=b_w - 26, tag='b:n')

# ---------------- 页脚 ----------------
FY = BB_Y + BB_H + 22
lc.text(MX, FY, '图例:绿 = Common→FA 的共享字段搬运(直搬 ×3) · 橙 = 改名 / FA 特有补算 · 灰虚线框 = 浅拷复用路径(host 实测口径)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/v1/attention/backends/flash_attn.py:L672-L696(build 翻译与 FA 字段)· vllm/v1/worker/gpu_model_runner.py:L2530-L2574(缓存命中/换表/铺设)· 改名值与复用实证 = 本章精简版 host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch21-fig-m11-translate.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
