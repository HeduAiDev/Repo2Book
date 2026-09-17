#!/usr/bin/env python3
"""ch32 机制图 · V1/V2 双路径选择器决策表（figure_spec ch32-fig-v1v2-selector，模板 state-table）

放大自 L0 采样列的双路径分岔——worker 侧落地实现（V1 utils.py→xgr / V2
StructuredOutputsWorker+Triton）由 use_v2_model_runner 选择（L2 站 16 的决策放大），
架构归属回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：v0.27.1 里 Llama 这类稠密生成模型默认走 V2（非 MoE 即 V2），与 v0.21『V2
opt-in』完全反转；env/PCP/dspark/diffusion 强制 V2，pooling/attention-free/无
Triton/ngram 回退 V1——同一掩码契约的两种落地。

数字全部取自 figure_spec.numbers（13 路判定实录；回退警告原文；默认 V2 架构名单
7 项；选择器源码锚 config/vllm.py:L578-L616）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'V1/V2 不是用户开关而是模型清单：Llama 默认走 V2——与 v0.21『V2 opt-in』完全反转', 16.5,
        lc.C_TXT, 'start', True, maxw=1200, tag='title')
lc.text(MX, 58, '同一掩码契约（先掩码后采样、bit=0 → -inf）的两种落地，差别只在搬运与 kernel：V1 重排整表上卡 · V2 kernel 内间接寻址'
               '——选择器是 property，13 路配置逐路实测',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列双路径选择器 · L2 站 16 的决策放大'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 决策阶梯（顶部小流程） =================
LY0, LH = 92, 86
LADDER = [
    ('① env 显式设置', '最优先、直接短路', lc.C_ENG_S),
    ('② 只在 V2 实现的特性', 'PCP / dspark / DFlash 混合 KV / diffusion → 强制 V2', lc.C_GPU_S),
    ('③ 模型清单', '默认 V2 名单（MoE 七家）或非 MoE 生成模型 → V2', lc.C_GPU_S),
    ('④ 回退项', '缺 Triton / 不支持的 spec 方法（ngram）→ 警告回退 V1', lc.C_MUTE),
]
LW = (BXR - MX - 3 * 14) / 4
for k, (t_, s_, c_) in enumerate(LADDER):
    x = MX + k * (LW + 14)
    lc.rect(x, LY0, LW, LH, '#ffffff', c_, rx=8, sw=1.5)
    lc.text(x + 12, LY0 + 20, t_, 9.4, c_, 'start', True, maxw=LW - 24, tag='ld' + str(k))
    lc.text(x + 12, LY0 + 40, s_, 7.8, '#334155', 'start', maxw=LW - 22, tag='lds' + str(k))
    if k < 3:
        lc.seg(x + LW + 1, LY0 + LH / 2, x + LW + 13, LY0 + LH / 2, lc.C_MUTE, 1.6, 'std')
lc.text(MX, LY0 + LH + 16, '命中即停：从左到右第一处命中即定 V1/V2（源码即这条阶梯，config/vllm.py:L578-L616）',
        8, lc.C_MUTE, 'start', maxw=1000, tag='ladder:f')

# ================= 13 路判定表（左） =================
TX, TY = MX, 208
COLS = [('配置（13 路逐路实测）', 330), ('判定', 70), ('理由', 460)]
HDR_H, ROW_H = 34, 35
ROWS = [
    ('env=1 强制 V2', True, 'env 显式设置优先，直接短路'),
    ('env=0 强制 V1', False, '即便稠密默认 V2 也被压回 V1'),
    ('PCP=2', True, 'PCP 运行时只在 V2 实现（无 MLA 也 force V2）'),
    ('spec=dspark', True, 'DSpark 只在 V2 GPU runner 实现 → 强制 V2'),
    ('diffusion', True, 'diffusion 模型强制 V2'),
    ('Llama 稠密生成（默认）', True, '非 MoE 生成模型 → V2——v0.21『V2 opt-in』在 v0.27 反转为默认'),
    ('Mixtral MoE（不在默认名单）', False, 'MoE 且架构不在默认 V2 名单 → V1'),
    ('Qwen2Moe（在默认名单）', True, '名单内 MoE → V2'),
    ('pooling 模型', False, 'runner_type != generate → V1'),
    ('attention-free', False, 'is_attention_free → V1'),
    ('稠密但无 Triton', False, 'V2 requires Triton → 警告并回退 V1'),
    ('spec=ngram（稠密）', False, 'ngram/ngram_gpu 未被 V2 支持 → 警告回退 V1'),
    ('spec=eagle（稠密）', True, 'eagle 在 V2 支持名单 → V2'),
]
TW = sum(w_ for _, w_ in COLS)
lc.rect(TX, TY, TW, HDR_H + len(ROWS) * ROW_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.5)
cx = TX
for name, w_ in COLS:
    lc.text(cx + w_ / 2, TY + 21, name, 9, lc.C_TXT, 'middle', True, maxw=w_ - 8,
            tag='th:' + name[:5])
    cx += w_
lc.seg(TX, TY + HDR_H - 4, TX + TW, TY + HDR_H - 4, lc.C_MUTE, 1.1)
for i, (cfg, v2, why) in enumerate(ROWS):
    ry = TY + HDR_H + i * ROW_H
    if i:
        lc.seg(TX, ry, TX + TW, ry, '#e2e8f0', 1.0)
    cy = ry + ROW_H / 2
    star = cfg.startswith('Llama')
    lc.text(TX + 12, cy + 3.5, cfg, 8.6, lc.C_TXT if not star else lc.C_GPU_S, 'start', True,
            maxw=COLS[0][1] - 20, tag='r' + str(i))
    bw = 40
    bx = TX + COLS[0][1] + (COLS[1][1] - bw) / 2
    lc.rect(bx, cy - 12, bw, 24, lc.C_GPU_F if v2 else '#f1f5f9',
            lc.C_GPU_S if v2 else lc.C_MUTE, rx=11, sw=1.3)
    lc.text(bx + bw / 2, cy + 4, 'V2' if v2 else 'V1', 9.2,
            lc.C_GPU_S if v2 else lc.C_MUTE, 'middle', True, maxw=36, tag='v' + str(i))
    lc.text(TX + COLS[0][1] + COLS[1][1] + 12, cy + 3.5, why, 8.2,
            '#334155', 'start', maxw=COLS[2][1] - 20, tag='w' + str(i))
    if star:
        lc.rect(TX + 2, ry + 2, TW - 4, ROW_H - 4, 'none', lc.C_GPU_S, rx=7, sw=1.2, dash=True)
        lc.text(TX + TW - 10, cy + 3.5, '★ 与 v0.21 反转', 7.6, lc.C_GPU_S, 'end', True,
                maxw=110, tag='star')

# ================= 右栏：名单 + 警告原文 =================
RX = TX + TW + 20
RW = BXR - RX
# 默认 V2 名单
AMY, AMH = 208, 226
lc.rect(RX, AMY, RW, AMH, '#ffffff', lc.C_GPU_S, rx=9, sw=1.5)
lc.text(RX + 14, AMY + 20, '默认 V2 架构名单（7 项，MoE）', 9.8, lc.C_GPU_S, 'start', True,
        maxw=RW - 28, tag='am:t')
ARCHS = ['DeepseekV2ForCausalLM', 'GraniteMoeForCausalLM', 'InklingForCausalLM',
         'InklingForConditionalGeneration', 'KimiK3ForConditionalGeneration',
         'LongcatFlashNgramForCausalLM', 'Qwen2MoeForCausalLM']
for j, a_ in enumerate(ARCHS):
    yy = AMY + 42 + j * 22
    lc.rect(RX + 12, yy - 12, RW - 24, 19, '#ffffff', lc.C_GPU_S, rx=4, sw=1.0)
    lc.text(RX + 20, yy + 1.5, a_, 7.8, '#334155', 'start', maxw=RW - 40, tag='am:l' + str(j))
lc.text(RX + 14, AMY + AMH - 10, '名单内 MoE 直接 V2；名单外 MoE（如 Mixtral）回 V1', 7.8,
        lc.C_MUTE, 'start', maxw=RW - 28, tag='am:f')

# 回退警告原文
WRY = AMY + AMH + 14
WRH = 118
lc.rect(RX, WRY, RW, WRH, '#fef2f2', lc.C_ABORT, rx=8, sw=1.3)
lc.text(RX + 14, WRY + 20, '回退时的真警告（warning_once 捕获）', 9.2, lc.C_ABORT, 'start', True,
        maxw=RW - 28, tag='wr:t')
lc.text(RX + 14, WRY + 40, '"Model Runner V2 requires Triton;', 7.8, lc.C_ABORT, 'start',
        maxw=RW - 26, tag='wr:l1')
lc.text(RX + 14, WRY + 54, ' using the V1 model runner instead."', 7.8, lc.C_ABORT, 'start',
        maxw=RW - 26, tag='wr:l2')
lc.text(RX + 14, WRY + 74, '"... does not yet support ngram/ngram_gpu', 7.8, lc.C_ABORT,
        'start', maxw=RW - 26, tag='wr:l3')
lc.text(RX + 14, WRY + 88, ' speculative decoding"', 7.8, lc.C_ABORT, 'start', maxw=RW - 26,
        tag='wr:l4')
lc.text(RX + 14, WRY + 106, '回退不失败服务，只降路径', 7.6, lc.C_MUTE, 'start', maxw=RW - 26,
        tag='wr:f')

# 两路径契约对照
CTY = WRY + WRH + 14
CTH = 118
lc.rect(RX, CTY, RW, CTH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(RX + 14, CTY + 20, '同一契约，两种落地', 9.4, lc.C_TXT, 'start', True, maxw=RW - 28,
        tag='ct:t')
for j, ln in enumerate(['· V1：重排整表进 sorted_bitmask → xgr',
                        '  apply_token_bitmask_inplace（本章主路径）',
                        '· V2：GPU 常驻掩码缓冲 + copy_stream 双异步 H2D，',
                        '  cu_num_logits 前缀和展开 indices、Triton kernel',
                        '  间接寻址（重排搬进 kernel）——契约同为 bit=0 → -inf']):
    lc.text(RX + 14, CTY + 40 + j * 16, ln, 7.8, '#334155', 'start', maxw=RW - 26,
            tag='ct:l' + str(j))

# ================= 图例 + 页脚 =================
LG_Y = TY + HDR_H + len(ROWS) * ROW_H + 26
lx0 = MX
lc.rect(lx0, LG_Y - 9, 16, 11, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.3)
lc.text(lx0 + 21, LG_Y + 1, '判定 V2（绿）', 8.8, lc.C_TXT, 'start', maxw=120, tag='lg1')
lx0 += 21 + lc.tw('判定 V2（绿）', 8.8) + 14
lc.rect(lx0, LG_Y - 9, 16, 11, '#f1f5f9', lc.C_MUTE, rx=3, sw=1.1)
lc.text(lx0 + 21, LG_Y + 1, '判定 V1（灰）', 8.8, lc.C_TXT, 'start', maxw=120, tag='lg2')
lx0 += 21 + lc.tw('判定 V1（灰）', 8.8) + 14
lc.rect(lx0, LG_Y - 10, 22, 13, 'none', lc.C_GPU_S, rx=5, sw=1.2)
lc.text(lx0 + 27, LG_Y + 1, '默认反转行（Llama）', 8.8, lc.C_TXT, 'start', maxw=180, tag='lg3')

lc.text(MX, LG_Y + 30, 'vllm/config/vllm.py:L578-L616（use_v2_model_runner property：env 优先 → 强制项 → 名单或非 MoE → Triton/特性回退）'
                 '· V2 落地 vllm/v1/worker/gpu/structured_outputs.py:L12-L115',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LG_Y + 48, '13 路判定 / 警告原文 / 默认 V2 名单 7 项 ＝ 真 use_v2_model_runner property 逐路实测（本章驱动脚本）'
                 '· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-v1v2-selector.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
