#!/usr/bin/env python3
"""ch23 机制图 3 · Attention 插座（figure_spec ch23-fig-attention-socket，模板 flow）

放大自本章 L2 章图拍片 ⑦ Attention 插座（站 9/10）· L0：GPU 执行臂 × 模型层框。

claim: Attention 是插座不是实现——LlamaAttention.forward 五行签名里没有
attn_metadata/kv_cache/slot_mapping，三件上下文按 layer_name 从 ForwardContext 取
（get_attention_context 返回四元组）；KV 写算子返回空张量 dummy 给 torch.compile
造数据依赖、保「先写后读」顺序。

数字全部取自 figure_spec.numbers（5 行 forward / 3 件上下文不在签名 / 四元组 /
2 个自定义算子 / new_empty(0) / spec decode 取 [0]）。坐标由常量/循环计算；文本全 esc()。
行号基线 vLLM v0.27.1（llama.py / attention.py）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1560, 890
MX = 56
BXR = 1504

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'Attention 是插座不是实现：签名里没有的三件上下文，从 ForwardContext 竖井自取',
        15.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, 'LlamaAttention.forward 五行（llama.py:L221-L231）→ 插座内部两个自定义算子（attention.py:L488-L846）——插头形状里没有的，由大楼配电（runner）提前送进墙内线路',
        10, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ Attention 插座（站 9/10）· L0：GPU 执行臂 × 模型层框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上层：模型侧（灰带） ----------------
MBX, MBY, MBW, MBH = 60, 96, 1444, 162
lc.rect(MBX, MBY, MBW, MBH, '#f8fafc', lc.C_MUTE, rx=9, sw=1.6)
lc.text(MBX + 16, MBY + 22, '模型侧（灰）：LlamaAttention.forward —— 五行，签名只有 (positions, hidden_states)',
        11, lc.C_TXT, 'start', True, maxw=760, tag='mb:t')
CHIPS = [('① qkv_proj(hidden)', 200), ('② split([q,k,v])', 186),
         ('③ rotary_emb(positions,q,k)', 246), ('④ attn(q,k,v) · 插头', 226),
         ('⑤ o_proj', 120)]
cx = 90
chip_rects = []
for i, (label, cw_) in enumerate(CHIPS):
    hot = (i == 3)
    lc.rect(cx, MBY + 40, cw_, 56, lc.C_GPU_F if hot else '#ffffff',
            lc.C_GPU_S if hot else '#94a3b8', rx=7, sw=1.8 if hot else 1.2)
    lc.text(cx + cw_ / 2, MBY + 72, label, 9.3 if not hot else 9.8,
            lc.C_GPU_S if hot else '#334155', 'middle', hot, maxw=cw_ - 12, tag='chip' + str(i))
    chip_rects.append((cx, cw_))
    if i < 4:
        lc.seg(cx + cw_ + 2, MBY + 68, cx + cw_ + 24, MBY + 68, lc.C_MUTE, 1.5, 'std')
    cx += cw_ + 26
# 签名注（band 右侧）：三件不在签名
lc.text(1210, MBY + 52, 'forward 签名里没有的三件上下文：', 8.8, lc.C_TXT, 'start', True,
        maxw=280, tag='mb:n1')
lc.text(1210, MBY + 70, 'attn_metadata · kv_cache · slot_mapping', 8.5, '#334155', 'start',
        maxw=280, tag='mb:n2')
lc.text(1210, MBY + 88, '（虚线插孔，不进模型代码）', 8.2, lc.C_MUTE, 'start', maxw=280, tag='mb:n3')
lc.text(1210, MBY + 108, '④ 的插头只有 q / k / v 三根线', 8.5, lc.C_GPU_S, 'start', True,
        maxw=280, tag='mb:n4')

# 插脚：④ 芯片底部 → 插座面板顶边
PLUG_X0, PLUG_X1 = chip_rects[3][0], chip_rects[3][0] + chip_rects[3][1]
pin_xs = [PLUG_X0 + 56, PLUG_X0 + 113, PLUG_X0 + 170]
PLATE_Y = 470
for i, px_ in enumerate(pin_xs):
    lc.seg(px_, MBY + 96, px_, PLATE_Y - 4, lc.C_GPU_S, 2.4, 'std')
    lc.text(px_ + 7, 396, 'qkv'[i], 9, lc.C_GPU_S, 'start', True, tag='pin' + str(i))
lc.text(PLUG_X0 + 113, 330, 'q / k / v 三根插脚', 8.5, lc.C_GPU_S, 'middle', tag='pins:t')
lc.text(PLUG_X0 + 113, 348, '（forward 的全部注意力入参）', 8, lc.C_MUTE, 'middle', tag='pins:s')

# 回程：attn_output 回 ⑤（回程线在算子盒定义之后统一画，见下文 ret 块）

# ---------------- 左侧：ForwardContext 竖井 ----------------
SHX, SHY, SHW, SHH = 60, 300, 240, 530
lc.rect(SHX, SHY, SHW, SHH, '#ffffff', lc.C_GPU_S, rx=9, sw=2.0)
lc.text(SHX + SHW / 2, SHY + 26, 'ForwardContext', 12, lc.C_GPU_S, 'middle', True, tag='sh:t')
lc.text(SHX + SHW / 2, SHY + 44, '（模块级全局竖井）', 9, lc.C_MUTE, 'middle', tag='sh:st')
lc.text(SHX + 16, SHY + 70, '进模型前由 runner 挂好：', 8.5, '#334155', 'start', tag='sh:l0')
lc.text(SHX + 16, SHY + 86, 'set_forward_context（站 6）', 8.5, '#334155', 'start', tag='sh:l1')
lc.text(SHX + 16, SHY + 102, 'ch17 已立的 GPUModelRunner', 8, lc.C_MUTE, 'start', tag='sh:l2')
lc.seg(SHX + 14, SHY + 116, SHX + SHW - 14, SHY + 116, '#cbd5e1', 1.2)
_shc = [('· attn_metadata', True),
        ('　spec decode 时是 list[dict]', False),
        ('　取 [0] 为 base 模型元数据', False),
        ('· no_compile_layers[layer_name]', True),
        ('　→ attn_layer（Attention 实例）', False),
        ('· slot_mapping[layer_name]', True)]
sy = SHY + 138
for ln, bold in _shc:
    lc.text(SHX + 16, sy, ln, 8.5 if not bold else 8.8, lc.C_TXT if bold else '#334155',
            'start', bold, maxw=SHW - 30, tag='sh:c' + ln[:8])
    sy += 17
lc.text(SHX + 16, SHY + SHH - 60, 'attn_layer 持有 kv_cache：', 8.2, '#334155', 'start',
        maxw=SHW - 30, tag='sh:k1')
lc.text(SHX + 16, SHY + SHH - 44, 'no_compile_layers 即花名册快照', 8.2, '#334155', 'start',
        maxw=SHW - 30, tag='sh:k2')
lc.text(SHX + 16, SHY + SHH - 20, 'vllm/forward_context.py', 8, lc.C_FAINT, 'start',
        maxw=SHW - 30, tag='sh:ft')

# 竖井 → 上下文取线盒
CTBX, CTBY, CTBW, CTBH = 356, 580, 284, 180
lc.seg(SHX + SHW, 670, CTBX, 670, lc.C_GPU_S, 1.8, 'std')
lc.rect(CTBX, CTBY, CTBW, CTBH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6, dash=True)
lc.text(CTBX + 14, CTBY + 22, 'get_attention_context', 9.8, lc.C_GPU_S, 'start', True,
        maxw=CTBW - 28, tag='ct:t1')
lc.text(CTBX + 14, CTBY + 38, '(layer_name)', 9.2, lc.C_GPU_S, 'start', tag='ct:t2')
lc.text(CTBX + 14, CTBY + 58, '返回四元组：', 8.5, lc.C_TXT, 'start', True, tag='ct:h')
_c4 = ['· attn_metadata', '· attn_layer', '· kv_cache', '· layer_slot_mapping']
sy = CTBY + 76
for ln in _c4:
    lc.text(CTBX + 22, sy, ln, 8.5, '#334155', 'start', tag='ct:' + ln[:8])
    sy += 16.5
lc.text(CTBX + 14, CTBY + CTBH - 10, 'attention.py:L732-L772', 8, lc.C_FAINT, 'start',
        maxw=CTBW - 28, tag='ct:ft')

# ---------------- 下层：插座面板（绿） ----------------
PTX, PTY, PTW, PTH = 340, 470, 1164, 360
lc.rect(PTX, PTY, PTW, PTH, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.4)
lc.text(PTX + 16, PTY + 24, '插座内部：Attention.forward——q/k/v view 重排后，先写后读两个自定义算子（attention.py:L488-L582）',
        11, lc.C_GPU_S, 'start', True, maxw=780, tag='pt:t')

# view 重排条（插脚进入插座的第一站）
VWX, VWY, VWW, VWH = 760, 506, 300, 40
for px_ in pin_xs:
    lc.seg(px_, PLATE_Y + 2, px_, VWY - 2, lc.C_GPU_S, 2.0, 'std')
lc.rect(VWX, VWY, VWW, VWH, '#ffffff', lc.C_GPU_S, rx=6, sw=1.4)
lc.text(VWX + VWW / 2, VWY + 25, 'q / k / v view 重排', 9.3, lc.C_GPU_S, 'middle', True,
        maxw=VWW - 12, tag='vw:t')

# 墙上虚线插孔（不在签名）
SKX, SKY, SKW, SKH = 1100, 506, 388, 78
lc.rect(SKX, SKY, SKW, SKH, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(SKX + 14, SKY + 20, '墙上另三个虚线插孔（不在签名）：', 8.8, lc.C_TXT, 'start', True,
        maxw=SKW - 28, tag='sk:t')
lc.text(SKX + 14, SKY + 38, 'attn_metadata · kv_cache · slot_mapping', 8.5, '#334155', 'start',
        maxw=SKW - 28, tag='sk:l1')
lc.text(SKX + 14, SKY + 56, '→ 由两个算子经 get_attention_context 自取（左）', 8.2, lc.C_MUTE,
        'start', maxw=SKW - 28, tag='sk:l2')

# 两个自定义算子（上下堆叠）
OPX, OPW = 680, 380
OP1Y, OPH = 560, 92
OP2Y = 690
lc.rect(OPX, OP1Y, OPW, OPH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(OPX + 14, OP1Y + 22, 'unified_kv_cache_update', 10, lc.C_GPU_S, 'start', True,
        maxw=OPW - 28, tag='op1:t1')
lc.text(OPX + 14, OP1Y + 38, '(key, value, layer_name)', 8.8, '#334155', 'start', tag='op1:t2')
lc.text(OPX + 14, OP1Y + 58, '写算：把 key/value 写进 KV cache', 8.5, '#334155', 'start',
        tag='op1:l1')
lc.text(OPX + 14, OP1Y + 74, 'return key.new_empty(0) ← 空张量 dummy', 8.5, lc.C_GPU_S,
        'start', True, maxw=OPW - 28, tag='op1:l2')
lc.text(OPX + OPW - 12, OP1Y + 74, 'L775-L798', 7.8, lc.C_FAINT, 'end', tag='op1:ft')
lc.rect(OPX, OP2Y, OPW, OPH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(OPX + 14, OP2Y + 22, 'unified_attention_with_output', 10, lc.C_GPU_S, 'start', True,
        maxw=OPW - 28, tag='op2:t1')
lc.text(OPX + 14, OP2Y + 38, '(query, key, value, output, layer_name)', 8.5, '#334155',
        'start', maxw=OPW - 28, tag='op2:t2')
lc.text(OPX + 14, OP2Y + 58, '读算：真正算注意力（调 impl.forward）', 8.5, '#334155', 'start',
        tag='op2:l1')
lc.text(OPX + OPW - 12, OP2Y + 74, 'L819-L846', 7.8, lc.C_FAINT, 'end', tag='op2:ft')

# 回程：attn_output 回 ⑤（自读算子右缘出发，走算子列与后端盒之间的竖走廊）
xr1, xr2 = 1085, 1112
ret_pts = [(OPX + OPW, 718), (xr1, 718), (xr1, 430), (xr2, 430), (xr2, MBY + 96)]
lc.parrow(ret_pts, lc.C_GPU_S, 1.8, 'std')
lc.text(xr1 + 12, 448, 'attn_output 回 ⑤ o_proj', 8.5, lc.C_GPU_S, 'start', tag='ret')

# view → op1；op1 → op2 两条（实线数据 + 虚线 dummy）
lc.seg(820, VWY + VWH, 820, OP1Y, lc.C_GPU_S, 1.8, 'std')
lc.alabel(828, VWY + VWH + 14, 'q,k,v', 8.5, lc.C_GPU_S)
lc.seg(790, OP1Y + OPH, 790, OP2Y, lc.C_GPU_S, 1.8, 'std')
lc.alabel(700, OP1Y + OPH + 13, 'q,k,v,output', 8.5, lc.C_GPU_S)
lc.seg(940, OP1Y + OPH, 940, OP2Y, lc.C_GPU_S, 1.8, 'std', dash=True)
lc.alabel(948, OP1Y + OPH + 13, 'dummy：钉住先写后读', 8.2, lc.C_GPU_S)

# 上下文取线盒 → 两个算子
lc.seg(CTBX + CTBW, 610, OPX, 610, lc.C_GPU_S, 1.8, 'std')
lc.text(CTBX + CTBW + 20, 604, '四元组', 7.5, lc.C_GPU_S, 'middle', tag='ctxa1')
lc.seg(CTBX + CTBW, 730, OPX, 730, lc.C_GPU_S, 1.8, 'std')
lc.text(CTBX + CTBW + 20, 724, '四元组', 7.5, lc.C_GPU_S, 'middle', tag='ctxa2')

# impl 后端盒（右下）
IMX, IMY, IMW, IMH = 1100, 690, 388, 128
lc.rect(IMX, IMY, IMW, IMH, '#ffffff', lc.C_MUTE, rx=8, sw=1.5, dash=True)
lc.text(IMX + 14, IMY + 22, 'impl.forward —— 后端实现', 10, lc.C_TXT, 'start', True,
        maxw=IMW - 28, tag='im:t')
_im = ['· 由 ch21 已立的优先级表选定', '　（FlashAttention / FlashInfer / …）',
       '· 模型层不参与选择——同一模型', '　换一栋楼（换后端）照常工作']
sy = IMY + 44
for ln in _im:
    lc.text(IMX + 14, sy, ln, 8.5, '#334155', 'start', maxw=IMW - 28, tag='im:l' + ln[:8])
    sy += 17.5
lc.seg(OPX + OPW, 745, IMX, 745, lc.C_GPU_S, 1.8, 'std')

# ---------------- 页脚 ----------------
lc.text(MX, 856, '读图：上层模型侧五行里没有三件上下文（右注）→ 插脚只有 q/k/v；墙内线路（ForwardContext 竖井）由 runner 提前挂好，两个算子按 layer_name 自取四元组——先写后读由 dummy 空张量钉住。',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 874, '行号基线 vLLM v0.27.1（llama.py / vllm/model_executor/layers/attention/attention.py）· KV 写算子返回 new_empty(0) 的语义引源码注释（docstring）',
        8, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch23-fig-attention-socket.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
