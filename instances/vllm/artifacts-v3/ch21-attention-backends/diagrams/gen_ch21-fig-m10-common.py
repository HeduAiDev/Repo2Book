#!/usr/bin/env python3
"""ch21 机制图 ⑤ · CommonAttentionMetadata 共用水表(figure_spec ch21-fig-m10-common,模板 tensor-flow)

放大自 L0 GPU 执行臂(绿)·每拍心跳「组装 metadata」一格(站 8)——向下接 builder 翻译、
向上接块表/槽位产出。架构归属回指 L0(FIGURE-SYSTEM §3.3),不另立第二种架构画法。

claim:每拍 runner 组装一份 CommonAttentionMetadata(核心 10 字段)供所有后端所有层共享——
块表/槽位先填组 0 的,组 0 之外 shallow copy 只换这两样,组内全部层铺同一份后端专属 metadata。

数字全部取自 figure_spec.numbers:核心 10 字段名(backend.py:L412-L444)、实测一拍
query_start_loc=[0,4,6,6,6]/seq_lens=[16,20,16,20]、组外只换 block_table/slot_mapping
两字段——本章精简版 host 实测 + pin 源码逐字。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 712
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '共用水表:一拍的 per-batch 读数,整栋楼只抄一次',
        16.5, lc.C_TXT, 'start', True, maxw=900, tag='title')
lc.text(MX, 58, 'CommonAttentionMetadata 核心十字段由 runner 每拍组装一份,供所有后端、所有层共享——几十个后端能塞进同一个 runner 每拍心跳,靠的就是把后端差异压缩到「每个 KV 组两张表」',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(绿)·每拍心跳「组装 metadata」(站 8)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左:这一拍的输入 ----------------
IN_X, IN_W = MX, 300
a_y, a_h = 116, 128
lc.rect(IN_X, a_y, IN_W, a_h, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(IN_X + 14, a_y + 21, '这一拍的批', 10, lc.C_TXT, 'start', True, maxw=IN_W - 28, tag='a:t')
for k, ln in enumerate([
        '2 个 prefill 请求:4 + 2 个新 token',
        '(req0 位置 12..15 / req1 位置 18..19)',
        'cudagraph padding:2 请求 → 4 行,',
        '6 token → 8 token(块表尾行填 0、',
        '槽位尾部填 -1——max 形状的代价)']):
    lc.text(IN_X + 14, a_y + 42 + k * 16, ln, 8.4, '#334155', 'start', maxw=IN_W - 26,
            tag=f'a:l{k}')

b_y, b_h = a_y + a_h + 14, 92
lc.rect(IN_X, b_y, IN_W, b_h, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(IN_X + 14, b_y + 21, '怎么组装', 10, lc.C_TXT, 'start', True, maxw=IN_W - 28, tag='b:t')
lc.text(IN_X + 14, b_y + 42, '持久缓冲切片,不新建张量:', 8.4, '#334155', 'start',
        maxw=IN_W - 26, tag='b:l1')
lc.text(IN_X + 14, b_y + 60, '_build_attention_metadata 从常驻', 8.4, '#334155', 'start',
        maxw=IN_W - 26, tag='b:l2')
lc.text(IN_X + 14, b_y + 78, 'CpuGpuBuffer 切出本拍的字段', 8.4, '#334155', 'start',
        maxw=IN_W - 26, tag='b:l3')

# ---------------- 中:CommonAttentionMetadata 卡 ----------------
CM_X, CM_W = 460, 620
CM_Y, CM_H = 116, 398
lc.rect(CM_X, CM_Y, CM_W, CM_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(CM_X + 16, CM_Y + 22, 'CommonAttentionMetadata —— 核心十字段(每拍一份)', 11,
        lc.C_GPU_S, 'start', True, maxw=440, tag='cm:t')
lc.text(CM_X + CM_W - 14, CM_Y + 22, 'vllm/v1/attention/backend.py:L412-L444', 8.5,
        lc.C_FAINT, 'end', maxw=200, tag='cm:file')
lc.text(CM_X + 16, CM_Y + 40, 'query_start_loc 计 gpu + cpu 两份(行内合并记);值 = 本章精简版 host 实测一拍',
        8, lc.C_MUTE, 'start', maxw=CM_W - 30, tag='cm:sub')

fields = [
    ('query_start_loc', '[0, 4, 6, 6, 6]', '前缀和切序列;尾部非递减', False),
    ('seq_lens', '[16, 20, 16, 20]', '尾 2 行 = 上拍残留', False),
    ('num_reqs', '4', 'padding 后的行数', False),
    ('num_actual_tokens', '8', 'padding 后的 token 数', False),
    ('max_query_len', '4', '', False),
    ('max_seq_len', '20', '', False),
    ('block_table_tensor', '[[0,1],[2,3],[0,0],[0,0]]', '组 0 的块表(尾 2 行填 NULL_BLOCK_ID=0)', True),
    ('slot_mapping', '[12,13,14,15, 50,51, -1,-1]', '组 0 的槽位(-1 = PAD_SLOT_ID)', True),
    ('causal', 'True', '', False),
]
FR_Y = CM_Y + 56
ROW_H = 30
for k, (name, val, note, swappable) in enumerate(fields):
    ry = FR_Y + k * ROW_H
    if swappable:
        lc.rect(CM_X + 8, ry - 17, CM_W - 16, ROW_H - 3, lc.C_BEAT_F, 'none', rx=4, sw=1)
        lc.text(CM_X + CM_W - 30, ry, '◆', 11, lc.C_BEAT_S, 'middle', True, tag=f'f{k}:d')
    lc.text(CM_X + 18, ry, name, 8.8, lc.C_TXT, 'start', True, maxw=150, tag=f'f{k}:n')
    lc.text(CM_X + 172, ry, '=', 8.8, lc.C_MUTE, 'start', tag=f'f{k}:eq')
    lc.text(CM_X + 186, ry, val, 8.8, (lc.C_BEAT_T if swappable else lc.C_GPU_S), 'start',
            True, maxw=240, tag=f'f{k}:v')
    if note:
        lc.text(CM_X + CM_W - 44, ry, note, 7.8, lc.C_MUTE, 'end', maxw=220, tag=f'f{k}:nt')
lc.text(CM_X + 16, CM_Y + CM_H - 34, '十字段里逐组会变的只有标 ◆ 的两样——其余八样整楼同一份,',
        8.4, lc.C_MUTE, 'start', maxw=CM_W - 30, tag='cm:n1')
lc.text(CM_X + 16, CM_Y + CM_H - 17, '这就是「几十个后端 × 几十层」不必各抄各表的原因', 8.4,
        lc.C_MUTE, 'start', maxw=CM_W - 30, tag='cm:n2')

# ---------------- 右:消费者 ----------------
OU_X, OU_W = 1140, BXR - 1140
c_y, c_h = 116, 150
lc.rect(OU_X, c_y, OU_W, c_h, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.6)
lc.text(OU_X + 14, c_y + 21, '全部后端 · 全部层', 10, lc.C_GPU_S, 'start', True,
        maxw=OU_W - 28, tag='c:t')
for k, ln in enumerate([
        '共享字段每拍只算一次;',
        'builder 拿它翻译成各自的后端专属单',
        '(翻译 = 下一图);',
        '组内多层铺同一份,层间零重复。']):
    lc.text(OU_X + 14, c_y + 42 + k * 16, ln, 8.4, '#334155', 'start', maxw=OU_W - 26,
            tag=f'c:l{k}')
lc.text(OU_X + 14, c_y + 112, '消费发生在下一站(站 9)', 8, lc.C_MUTE, 'start',
        maxw=OU_W - 26, tag='c:n')

d_y, d_h = c_y + c_h + 14, 234
lc.rect(OU_X, d_y, OU_W, d_h, '#ffffff', lc.C_MUTE, rx=7, sw=1.4, dash=True)
lc.text(OU_X + 14, d_y + 21, '组 1 = copy(cm_base) 浅拷', 10, lc.C_TXT, 'start', True,
        maxw=OU_W - 28, tag='d:t')
for k, ln in enumerate([
        '组 0 之外的组,只换 ◆ 两样:',
        'block_table_tensor =',
        '  [[4,5],[6,7],[0,0],[0,0]]',
        'slot_mapping =',
        '  [76,77,78,79, 114,115, -1,-1]',
        '其余字段原样继承(浅拷不动)——',
        '每个 KV cache 组与组 0 的差异,',
        '全部装在这两行里。']):
    lc.text(OU_X + 14, d_y + 42 + k * 16, ln, 8.4, '#334155', 'start', maxw=OU_W - 26,
            tag=f'd:l{k}')

# ---------------- 流箭头 ----------------
mid_l = (a_y + a_h / 2 + b_y + b_h / 2) / 2
lc.parrow([(IN_X + IN_W, a_y + a_h / 2), (CM_X, a_y + a_h / 2)], lc.C_MUTE, 1.6, 'std')
lc.parrow([(IN_X + IN_W, b_y + b_h / 2), (CM_X, b_y + b_h / 2)], lc.C_MUTE, 1.6, 'std')
lc.parrow([(CM_X + CM_W, c_y + c_h / 2), (OU_X, c_y + c_h / 2)], lc.C_GPU_S, 1.8, 'std')
lc.parrow([(CM_X + CM_W, d_y + d_h / 2), (OU_X, d_y + d_h / 2)], lc.C_MUTE, 1.6, 'std',
          dash=True)

# ---------------- 底条:padding 读法 ----------------
PB_Y = CM_Y + CM_H + 22
lc.rect(MX, PB_Y, BXR - MX, 56, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
lc.text(MX + 14, PB_Y + 20, 'padding 读法:query_start_loc 尾部 [.., 6, 6, 6] = padding 行的空段(非递减,不是新请求);seq_lens 尾 2 行 [16, 20] = 上拍残留——cudagraph 捕获的是 max 形状,尾部每拍都要重填',
        8.6, lc.C_TXT, 'start', maxw=BXR - MX - 28, tag='pb:1')
lc.text(MX + 14, PB_Y + 40, '块表/槽位的产出(块表谁维护、槽位怎么算)是前文两章的账;本图只管「这一拍它们长什么样、被谁共用」',
        8.4, lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='pb:2')

# ---------------- 页脚 ----------------
FY = PB_Y + 76
lc.text(MX, FY, '图例:绿 = runner 每拍产物(执行臂) · 橙底/◆ = 逐组要换的仅有的两字段 · 灰虚线 = 组 0 之外的 KV 组(shallow copy)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/v1/attention/backend.py:L412-L444(核心字段)· vllm/v1/worker/gpu_model_runner.py:L2430-L2449(持久缓冲切片组装)· L2561-L2574(组外浅拷换两表)· 十字段实测值 = 本章精简版 host 实测一拍 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch21-fig-m10-common.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
