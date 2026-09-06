#!/usr/bin/env python3
"""ch21 机制图 ⑦ · 读腿喂料(figure_spec ch21-fig-m12-readleg,模板 tensor-flow)

放大自 L0 GPU 执行臂(绿)·模型层 Attention() 插座通电后的读腿(站 12)——把 ch20 的
kernel 调用面与 ch22 的页表在这里接上。架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:读腿一行 flash_attn_varlen_func 同时吃下 varlen 打平(cu_seqlens_q 前缀和切序列)
与 paged KV(block_table 页表指路)——query 按因果只看自己位置之前的键,输出写进预分配 output。

数字全部取自 figure_spec.numbers:调用面(flash_attn.py:L1041-L1066)、实测 cu_seqlens_q=[0,4]/
seqused_k=[16]/block_table=[[0,1]]、q0..q3 看 13..16 键、输出 (4,256)、对拍 ≤5e-4、K/V 拆包
L904-L905——本章精简版 host 实测 + pin 源码逐字。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 762
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '读腿喂料:varlen 打平与页表取数,发生在同一次 kernel 调用里',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '一行 flash_attn_varlen_func 同时吃下 cu_seqlens_q(前缀和切序列)与 block_table(沿表取分散的 K/V 块);query 按因果只看自己位置之前(含自己)的键——ch20 讲过的 tiling 数学,在真实源码里就发生在这一行',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(绿)·Attention() 通电后的读腿(站 12)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 调用面 ----------------
CALL_Y, CALL_H = 92, 44
lc.rect(MX, CALL_Y, BXR - MX, CALL_H, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.5)
lc.text(MX + 16, CALL_Y + 27,
        'flash_attn_varlen_func(q, k=key_cache, v=value_cache, cu_seqlens_q=query_start_loc, seqused_k=seq_lens, block_table=block_table, causal=True)',
        10, lc.C_GPU_S, 'start', True, maxw=1180, tag='call')
lc.text(BXR - 14, CALL_Y + 27, 'flash_attn.py:L1041-L1066', 8.5, lc.C_FAINT, 'end',
        maxw=190, tag='call:file')

# ---------------- K/V 拆包 shape 流 ----------------
SP_Y, SP_H = 152, 60
lc.text(MX, SP_Y + 14, 'K/V 拆包(进 kernel 前的两次零拷视图变换):', 9, lc.C_TXT, 'start',
        True, maxw=320, tag='sp:t')
boxes = [('kv_cache', '(64, 2, 16, 128)', 'B,H,N,2D'),
         ('transpose(1, 2)', '(64, 16, 2, 128)', 'B,N,H,2D'),
         ('split(head_size=64, dim=-1)', 'key_cache (64,16,2,64)', '+ value_cache 同形')]
bx = MX + 330
for k, (t, v, d) in enumerate(boxes):
    w = 250
    lc.rect(bx, SP_Y - 6, w, 48, '#ffffff', lc.C_KV_S, rx=6, sw=1.3)
    lc.text(bx + w / 2, SP_Y + 12, t, 8.4, lc.C_KV_S, 'middle', True, maxw=w - 10,
            tag=f'sp:k{k}')
    lc.text(bx + w / 2, SP_Y + 30, v, 8, '#334155', 'middle', maxw=w - 10, tag=f'sp:v{k}')
    if k < 2:
        lc.seg(bx + w, SP_Y + 18, bx + w + 34, SP_Y + 18, lc.C_KV_S, 1.6, 'std')
    bx += w + 34

# ---------------- 因果网格 ----------------
G_TITLE_Y = 244
lc.text(MX, G_TITLE_Y, 'causal=True 的因果阶梯:本切片 = req0(4 个新 query,位置 12..15)读 16 个键(历史 12 + 新 4)',
        10, lc.C_TXT, 'start', True, maxw=900, tag='g:t')

BT_Y = 268                      # block_table 条
GX0, CELL = 200, 34
NKEYS = 16
b0_w = NKEYS * CELL
lc.rect(GX0, BT_Y, b0_w, 26, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.4)
lc.text(GX0 + b0_w / 2, BT_Y + 17, '块 0(block_table[0][0])', 8.5, lc.C_KV_S, 'middle', True,
        maxw=b0_w - 10, tag='bt:b0')
b1_x, b1_w = GX0 + b0_w + 16, 120
lc.rect(b1_x, BT_Y, b1_w, 26, '#f1f5f9', lc.C_FAINT, rx=4, sw=1.2, dash=True)
lc.text(b1_x + b1_w / 2, BT_Y + 17, '块 1(表内第 2 项)', 8, lc.C_FAINT, 'middle',
        maxw=b1_w - 8, tag='bt:b1')
lc.text(GX0 - 12, BT_Y + 17, 'block_table = [[0, 1]]', 8.5, lc.C_KV_S, 'end', maxw=130,
        tag='bt:l')
for cx in (GX0 + b0_w / 2, b1_x + b1_w / 2):
    lc.seg(cx, BT_Y + 26, cx, BT_Y + 44, lc.C_KV_S, 1.2, None, dash=True)
lc.text(GX0 + b0_w / 2, BT_Y + 56, '沿表取块:16 个键 = ceil(16/16) = 1 块', 8, lc.C_KV_S,
        'middle', maxw=b0_w, tag='bt:n')

KY_Y = BT_Y + 92                # 键位置号行
for j in range(NKEYS):
    lc.text(GX0 + j * CELL + CELL / 2, KY_Y, str(j), 7, lc.C_MUTE, 'middle', tag=f'kx{j}')

ROW_Y0 = KY_Y + 10
ROW_H = 34
qinfo = [('q0', 12, 13), ('q1', 13, 14), ('q2', 14, 15), ('q3', 15, 16)]
for i, (qn, pos, seen) in enumerate(qinfo):
    ry = ROW_Y0 + i * ROW_H
    lc.text(GX0 - 12, ry + ROW_H / 2 + 3, f'{qn}(位置 {pos})', 8.2, lc.C_TXT, 'end', True,
            maxw=120, tag=f'q{i}')
    for j in range(NKEYS):
        rx = GX0 + j * CELL
        visible = j <= pos
        if visible:
            new = j >= 12
            fill = '#bbf7d0' if new else lc.C_GPU_F
            lc.rect(rx + 1, ry + 1, CELL - 2, ROW_H - 2, fill, lc.C_GPU_S, rx=3, sw=1.0)
        else:
            lc.rect(rx + 1, ry + 1, CELL - 2, ROW_H - 2, '#f8fafc', '#e2e8f0', rx=3, sw=0.8)
    ax = GX0 + NKEYS * CELL + 16
    lc.text(ax, ry + ROW_H / 2 + 3, f'看 {seen} 键', 8.4, lc.C_GPU_S, 'start', True,
            maxw=90, tag=f'a{i}')

GRID_BOT = ROW_Y0 + 4 * ROW_H
split_x = GX0 + 12 * CELL
lc.seg(split_x, KY_Y - 6, split_x, GRID_BOT + 4, lc.C_ABORT, 1.3, None, dash=True)
lc.text(GX0 + 6 * CELL, GRID_BOT + 18, '历史 12(块 0 的 0..11 行)', 8, lc.C_MUTE, 'middle',
        maxw=2 * 11 * CELL, tag='h:old')
lc.text(GX0 + 14 * CELL, GRID_BOT + 18, '本拍新 4(写腿刚落位)', 8, lc.C_GPU_S, 'middle', True,
        maxw=2 * 4 * CELL, tag='h:new')
lc.text(b1_x + b1_w / 2, GRID_BOT + 18, '本拍未被盖到', 7.5, lc.C_FAINT, 'middle',
        maxw=130, tag='h:b1')
lc.text(MX, BT_Y + 44, 'cu_seqlens_q = [0, 4]', 8.4, lc.C_KV_S, 'start', True, maxw=130,
        tag='csl')
lc.text(MX, BT_Y + 60, '前缀和切序列:0..4', 8, lc.C_MUTE, 'start', maxw=130, tag='csl2')
lc.text(MX, BT_Y + 76, '是同一家人(req0)', 8, lc.C_MUTE, 'start', maxw=130, tag='csl3')
lc.text(MX, BT_Y + 96, 'seqused_k = [16]', 8.4, lc.C_KV_S, 'start', True, maxw=130,
        tag='sk')
lc.text(MX, BT_Y + 112, '该序列读 16 个键', 8, lc.C_MUTE, 'start', maxw=130, tag='sk2')

# ---------------- 底部两块 ----------------
BB_Y = GRID_BOT + 36
BB_H = 118
lc.rect(MX, BB_Y, 680, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 14, BB_Y + 20, '输出与对拍', 10, lc.C_TXT, 'start', True, maxw=300, tag='o:t')
for k, ln in enumerate([
        '输出 (4, 256) = 4 token × 4 头 × 64 维——写进预分配 output,不新建张量;',
        '与手工 attention 逐行对拍(fp64 softmax 重算):max_abs_diff 6.4e-5 / 5.0e-4',
        '  ——fp16 精度之内,数值与 ch20 的 tiling 推导一致(绿格 = 本拍新写的 4 键)。']):
    lc.text(MX + 14, BB_Y + 42 + k * 17, ln, 8.4, '#334155', 'start', maxw=650,
            tag=f'o:l{k}')

bx2 = MX + 700
lc.rect(bx2, BB_Y, BXR - bx2, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(bx2 + 14, BB_Y + 20, '接缝:一章回指,一章预告', 10, lc.C_TXT, 'start', True,
        maxw=300, tag='j:t')
lc.text(bx2 + 14, BB_Y + 44, '回指 ch20:tiling / online softmax 的数学在前章已推导,本行只是它的真实调用面;',
        8.4, '#334155', 'start', maxw=BXR - bx2 - 28, tag='j:l1')
lc.text(bx2 + 14, BB_Y + 64, '预告 ch22:block_table 的块号怎么分、slot_mapping 的槽位怎么算,',
        8.4, lc.C_MUTE, 'start', maxw=BXR - bx2 - 28, tag='j:l2')
lc.text(bx2 + 14, BB_Y + 82, '  在下一章(slot_mapping 与 block_table)——本图的表是它的消费端。',
        8.4, lc.C_MUTE, 'start', maxw=BXR - bx2 - 28, tag='j:l3')
_jw = 64
lc.rect(bx2 + 14, BB_Y + 92, 44, 15, '#ffffff', lc.C_GPU_S, rx=4, sw=1.1)
lc.text(bx2 + 36, BB_Y + 103, '回指', 8, lc.C_GPU_S, 'middle', True, maxw=36, tag='j:c1')
lc.rect(bx2 + 66, BB_Y + 92, 44, 15, '#ffffff', lc.C_MUTE, rx=4, sw=1.1, dash=True)
lc.text(bx2 + 88, BB_Y + 103, '预告', 8, lc.C_MUTE, 'middle', True, maxw=36, tag='j:c2')
lc.text(bx2 + 120, BB_Y + 103, '= 跨章引用方向(本章 21)', 7.8, lc.C_MUTE, 'start',
        maxw=200, tag='j:c3')

# ---------------- 页脚 ----------------
FY = BB_Y + BB_H + 24
lc.text(MX, FY, '图例:绿格 = 该 query 因果可见的键(深绿 = 本拍新写、浅绿 = 历史) · 灰格 = 被因果挡住 · 青 = KV 块与序列切分 · 灰虚块 = 表内本拍未用项',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/v1/attention/backends/flash_attn.py:L1041-L1066(调用面)· L904-L905(transpose+split 拆包)· 键数 13/14/15/16、输出 (4,256)、对拍差 = 本章精简版 host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch21-fig-m12-readleg.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
