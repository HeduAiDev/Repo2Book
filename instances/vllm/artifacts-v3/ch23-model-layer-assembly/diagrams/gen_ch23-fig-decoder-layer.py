#!/usr/bin/env python3
"""ch23 机制图 1 · DecoderLayer 单层剖面（figure_spec ch23-fig-decoder-layer，模板 layout）

放大自本章 L2 章图拍片 ⑥ DecoderLayer 穿针（站 8）· L0：GPU 执行臂 × 模型层框。

claim: 一个 DecoderLayer 只装配四个积木（双 RMSNorm + LlamaAttention + LlamaMLP），
residual 是穿出整层的一等公民——RMSNorm(hidden, residual) 一次 kernel 同时完成
加残差+归一化并返回 (hidden, residual) 二元组，首层 residual=None 特判起步。

数字全部取自 figure_spec.numbers（4 积木 / 2 RMSNorm / 5 行 forward / 2 元组 /
hidden=64 / 2 层，源码锚点 llama.py:L282-L327 + 本章示例模型参数）。
坐标由常量/循环计算；文本全 esc()。行号基线 vLLM v0.27.1。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1560, 808
MX = 56
BXR = 1504

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'DecoderLayer 剖面：四个积木 + 一根穿出整层的残差总线',
        17, lc.C_TXT, 'start', True, maxw=960, tag='title')
lc.text(MX, 58, '双 RMSNorm 垫片 + LlamaAttention + LlamaMLP 按固定图样串接——每半层 RMSNorm(hidden, residual) 一次 kernel 同时完成「加残差+归一化」，吐回二元组（vllm/model_executor/models/llama.py:L282-L327）',
        10, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L2 拍片 ⑥ DecoderLayer 穿针（站 8）· L0：GPU 执行臂 × 模型层框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主流程几何：hidden 上行 / residual 总线下行 ----------------
# 行高：RMSNorm 框跨双行(110..360)；attn/mlp 只占 hidden 行；总线 y=320 穿框而过
N_Y, N_H = 110, 250
T_Y, T_H = 110, 186
HID_Y = 210                          # hidden 主线 y
BUS_Y = 320                          # residual 总线 y
IN_X, IN_W = MX, 172
NORM_W, ATTN_W, MLP_W = 178, 252, 218
GAP = 46
x_in = IN_X                          # 56
x_n1 = x_in + IN_W + GAP + 14        # 288
x_at = x_n1 + NORM_W + GAP           # 512
x_n2 = x_at + ATTN_W + GAP           # 810
x_ml = x_n2 + NORM_W + GAP           # 1034
x_out = x_ml + MLP_W + GAP + 14      # 1312
OUT_W = BXR - x_out                  # 192

# ---- 进线二元组（跨双行，高对齐 RMSNorm 框）----
for xx, ww, tt in ((x_in, IN_W, '进线'), (x_out, OUT_W, '出线')):
    lc.rect(xx, N_Y, ww, N_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
    lc.text(xx + ww / 2, N_Y + 24, tt, 11.5, lc.C_GPU_S, 'middle', True, tag=tt)
    lc.text(xx + ww / 2, N_Y + 46, '(hidden_states,', 9.5, '#334155', 'middle')
    lc.text(xx + ww / 2, N_Y + 62, 'residual)', 9.5, '#334155', 'middle')
lc.text(x_in + IN_W / 2, HID_Y - 14, 'hidden 主线', 8.5, lc.C_MUTE, 'middle')
lc.rect(x_in + IN_W / 2 - 16, HID_Y - 6, 32, 12, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.2)
lc.text(x_in + IN_W / 2, BUS_Y - 14, 'residual 总线', 8.5, lc.C_MUTE, 'middle')
lc.rect(x_in + IN_W / 2 - 16, BUS_Y - 6, 32, 12, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.2)
lc.text(x_out + OUT_W / 2, HID_Y - 14, 'hidden 主线', 8.5, lc.C_MUTE, 'middle')
lc.rect(x_out + OUT_W / 2 - 16, HID_Y - 6, 32, 12, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.2)
lc.text(x_out + OUT_W / 2, BUS_Y - 14, 'residual 总线', 8.5, lc.C_MUTE, 'middle')
lc.rect(x_out + OUT_W / 2 - 16, BUS_Y - 6, 32, 12, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.2)
lc.text(x_out + OUT_W / 2, N_Y + 96, 'forward 返回二元组', 8.5, lc.C_MUTE, 'middle')
lc.text(x_out + OUT_W / 2, N_Y + 112, '(llama.py:L327)', 7.8, lc.C_FAINT, 'middle')

# 首层特判（虚线注，挂进线框下方）
FY = N_Y + N_H + 24                  # 384
lc.rect(x_in - 6, FY, IN_W + 12, 64, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(x_in + IN_W / 2, FY + 16, '首层特判（虚线）', 8.5, lc.C_MUTE, 'middle', True, tag='first:t')
lc.text(x_in + IN_W / 2, FY + 32, 'residual=None →', 8.2, '#334155', 'middle')
lc.text(x_in + IN_W / 2, FY + 45, 'residual=hidden_states', 8.2, '#334155', 'middle')
lc.text(x_in + IN_W / 2, FY + 58, '(llama.py:L317-L319)', 7.8, lc.C_FAINT, 'middle')
lc.seg(x_in + IN_W / 2, FY, x_in + IN_W / 2, N_Y + N_H, lc.C_MUTE, 1.2, dash=True)

# ---- 两个 RMSNorm 垫片（跨双行：总线在框内被消费/再产出）----
def norm_box(x, title_lines):
    lc.rect(x, N_Y, NORM_W, N_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
    ty = N_Y + 22
    for t in title_lines:
        lc.text(x + NORM_W / 2, ty, t, 11.5, lc.C_GPU_S, 'middle', True,
                maxw=NORM_W - 16, tag='nrm:' + t)
        ty += 16
    lc.text(x + NORM_W / 2, N_Y + 76, 'RMSNorm 垫片 ×1', 8.5, lc.C_MUTE, 'middle')
    lc.text(x + NORM_W / 2, N_Y + 98, 'RMSNorm(hidden, residual)', 8.8, '#334155', 'middle', True)
    lc.text(x + NORM_W / 2, N_Y + 114, '一次 kernel = 加残差+归一化', 8.2, '#334155', 'middle')
    lc.text(x + NORM_W / 2, N_Y + 128, 'fused_add_rms_norm', 8.2, '#334155', 'middle')
    lc.text(x + NORM_W / 2, N_Y + 142, '(layernorm.py:L74-L95)', 7.8, lc.C_FAINT, 'middle')
    lc.text(x + NORM_W / 2, N_Y + 170, '收二元组 → 回二元组', 8.5, lc.C_GPU_S, 'middle', True)
    lc.text(x + NORM_W / 2, N_Y + 186, '(hidden, residual)', 8.2, '#334155', 'middle')

norm_box(x_n1, ['input_layernorm'])
norm_box(x_n2, ['post_attention_', 'layernorm'])

# ---- self_attn（只占 hidden 行，总线从框下 24px 直穿）----
lc.rect(x_at, T_Y, ATTN_W, T_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(x_at + ATTN_W / 2, T_Y + 20, 'self_attn · LlamaAttention', 11, lc.C_GPU_S, 'middle',
        True, maxw=ATTN_W - 16, tag='at:t')
lc.text(x_at + ATTN_W / 2, T_Y + 36, 'forward 五行（L221-L231）', 8.5, lc.C_MUTE, 'middle')
_attn_steps = ['① qkv_proj(hidden)', '② split([q,k,v])', '③ rotary_emb(positions,q,k)',
               '④ attn(q,k,v) ← 插座', '⑤ o_proj']
sy = T_Y + 52
for i, st in enumerate(_attn_steps):
    hot = (i == 3)
    lc.rect(x_at + 18, sy + i * 24, ATTN_W - 36, 19,
            lc.C_GPU_F if hot else '#f8fafc', lc.C_GPU_S if hot else '#cbd5e1', rx=4, sw=1.0)
    lc.text(x_at + ATTN_W / 2, sy + i * 24 + 13.5, st, 8.3,
            lc.C_GPU_S if hot else '#334155', 'middle', hot, maxw=ATTN_W - 44, tag='ats' + str(i))

# ---- mlp ----
lc.rect(x_ml, T_Y, MLP_W, T_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(x_ml + MLP_W / 2, T_Y + 20, 'mlp · LlamaMLP', 11, lc.C_GPU_S, 'middle', True,
        maxw=MLP_W - 16, tag='ml:t')
lc.text(x_ml + MLP_W / 2, T_Y + 36, 'gate_up → SiluAndMul → down', 8.5, lc.C_MUTE, 'middle')
_mlp_steps = ['gate_up_proj(x)', 'act_fn = SiluAndMul()', 'down_proj(x)']
sy = T_Y + 56
for i, st in enumerate(_mlp_steps):
    lc.rect(x_ml + 18, sy + i * 30, MLP_W - 36, 22, '#f8fafc', '#cbd5e1', rx=4, sw=1.0)
    lc.text(x_ml + MLP_W / 2, sy + i * 30 + 15, st, 8.3, '#334155', 'middle', maxw=MLP_W - 44,
            tag='mls' + str(i))

# ---- 连线：hidden 主线（细箭头）与 residual 总线（粗线）----
for x0, x1 in ((x_in + IN_W, x_n1), (x_n1 + NORM_W, x_at), (x_at + ATTN_W, x_n2),
               (x_n2 + NORM_W, x_ml), (x_ml + MLP_W, x_out)):
    lc.seg(x0, HID_Y, x1, HID_Y, lc.C_GPU_S, 2.0, 'std')
lc.alabel(x_n1 + NORM_W + 4, HID_Y - 8, 'hidden', 8.5, lc.C_GPU_S)
lc.alabel(x_at + ATTN_W + 4, HID_Y - 8, 'hidden', 8.5, lc.C_GPU_S)
# residual 总线三段：入线→norm1 / norm1→norm2 / norm2→出线（attn/mlp 段从框下直穿）
lc.seg(x_in + IN_W, BUS_Y, x_n1, BUS_Y, lc.C_GPU_S, 3.2, 'std')
lc.seg(x_n1 + NORM_W, BUS_Y, x_n2, BUS_Y, lc.C_GPU_S, 3.2, 'std')
lc.seg(x_n2 + NORM_W, BUS_Y, x_out, BUS_Y, lc.C_GPU_S, 3.2, 'std')
lc.text((x_n1 + NORM_W + x_n2) / 2, BUS_Y + 26, 'residual 总线：一等公民接线，attn/mlp 阶段原样通过',
        8.5, lc.C_GPU_S, 'middle', tag='buslbl')

# 图内小图例：主线/总线（右移避开左下首层特判注框 x 50..234）
LGY = N_Y + N_H + 52                  # 412
LGX = 262
lc.seg(LGX, LGY, LGX + 26, LGY, lc.C_GPU_S, 2.0, 'std')
lc.text(LGX + 32, LGY + 3.5, 'hidden_states 主线（进积木加工）', 8.5, '#334155', 'start', tag='leg1')
lc.seg(LGX + 254, LGY, LGX + 280, LGY, lc.C_GPU_S, 3.2)
lc.text(LGX + 286, LGY + 3.5, 'residual 总线（粗线：穿出整层的一等公民）', 8.5, '#334155',
        'start', tag='leg2')

# ---------------- 底部：HF 对照 + 去向注 ----------------
CY = LGY + 26                         # 438
C_H = 172
C_W = (BXR - MX - 24) / 2
lc.rect(MX, CY, C_W, C_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 16, CY + 24, '对照：HF transformers 的 modeling_llama', 11, lc.C_TXT, 'start',
        True, maxw=C_W - 32, tag='hf:t')
_hf = ['· residual 是局部变量：hidden_states = hidden_states + attn_out，加完再单独 RMSNorm(hidden)',
       '· add 与 norm 是两次独立 kernel——一层四次单独操作（两个半层 × add+norm）',
       '· 尾部 return hidden_states——residual 留在局部变量里，不穿层']
hy = CY + 50
for ln in _hf:
    lc.text(MX + 16, hy, ln, 8.8, '#475569', 'start', maxw=C_W - 30, tag='hf:l')
    hy += 30
cx2 = MX + C_W + 24
lc.rect(cx2, CY, C_W, C_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.6)
lc.text(cx2 + 16, CY + 24, 'vLLM：残差提升为一等公民总线', 11, lc.C_GPU_S, 'start', True,
        maxw=C_W - 32, tag='vl:t')
_vl = ['· RMSNorm(x, residual) 收二元组、回二元组——加料+归一化合并成一次 kernel',
       '· 残差是显式接线：forward(positions, hidden_states, residual) 签名里就有一等公民',
       '· 末层由 LlamaModel.forward 的 norm(hidden, residual) 收尾；PP 分段时装进',
       '   IntermediateTensors 双字段（hidden_states + residual）交下一 stage']
hy = CY + 50
for ln in _vl:
    lc.text(cx2 + 16, hy, ln, 8.8, '#1f5132', 'start', maxw=C_W - 30, tag='vl:l')
    hy += 30

# ---------------- 数值徽标条 + 页脚 ----------------
NY = CY + C_H + 26                    # 636
lc.rect(MX, NY, BXR - MX, 42, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
_badge = [('四件套 = 4', 'self_attn + mlp + 双 RMSNorm（L282-L308）'),
          ('RMSNorm 垫片 × 2', 'input + post_attention（L305-L308）'),
          ('forward 五行', 'LlamaAttention.forward（L221-L231）'),
          ('返回二元组', '(hidden_states, residual)（L327）'),
          ('示例模型', 'hidden=64 · 2 层')]
bx = MX + 18
for t, s in _badge:
    lc.text(bx, NY + 17, t, 8.8, lc.C_GPU_S, 'start', True, tag='bdg:' + t)
    lc.text(bx, NY + 33, s, 7.8, lc.C_MUTE, 'start', maxw=254, tag='bdg:s:' + t[:8])
    bx += 292

lc.text(MX, NY + 72, '读图：左侧进线二元组 → 每半层「先在总线上加料、再融合归一化」→ 四件套按序穿针 → 右侧仍是二元组；残差总线（粗线）首层由 residual=None 特判搭起、此后原样穿过 attn/mlp。',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, NY + 90, '行号基线 vLLM v0.27.1（llama.py / layernorm.py）· 示例模型参数（hidden=64、2 层）取自本章精简版实跑核验（host CPU）',
        8, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch23-fig-decoder-layer.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
