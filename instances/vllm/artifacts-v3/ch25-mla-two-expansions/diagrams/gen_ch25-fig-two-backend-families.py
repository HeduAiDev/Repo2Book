#!/usr/bin/env python3
"""ch25 机制图 · 两套后端家族:一个 MLA 层同时是两套家族的客户(figure ch25-fig-two-backend-families,模板 layout)

放大自 L0『模型层 MLA 框』与后端世界的接口面。中轴=MLAAttention 插座;左翼=decode 家族
(特形 cache 身份证);右翼=prefill 家族五名牌+CUSTOM 槽;两翼各自独立的选择箭头(各选各的,
优先级表回指 ch21)。本章只立『为什么 MLA 要两套』,不重表选择机制。

claim:特形 KV cache(num_kv_heads=1、head_size=576、形状不拆 K/V 维)与所有通用后端不兼容,
MLA 因此有两套独立后端家族——decode 家族以 is_mla()=True 为身份证(forward_mha/forward_mqa
双抽象),prefill 家族 MLAPrefillBackendEnum 五选一,两轴独立选择。

数字全部取自 figure spec 的 numbers(get_kv_cache_shape 返回 (num_blocks, block_size,
head_size) · get_supported_head_sizes=[320,576] · is_mla()=True · 五成员+CUSTOM ·
装配位 L423-L436/L520-L550)。坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 760
MX = 52
BXR = 1448
C_DEC = lc.C_GPU_S         # decode 家族 = 绿(GPU kernel 侧)
C_DEC_T = '#166534'
C_PRE = lc.C_KV_S          # prefill 家族 = 青(标准 MHA 计算侧,图例声明)
C_PRE_T = '#155e75'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '为什么 MLA 有两套后端:cache 形状连 K/V 都不拆,通用后端对不上号',
        16, lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, 'decode 家族读特形潜 cache(单『头』576 维),prefill 家族吃上投影后的标准 MHA 形状——两轴各选各的,互不代管',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』与后端世界的接口面'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 中轴:MLAAttention 插座 ----------------
SOB = (560, 190, 380, 300)
lc.rect(*SOB, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.2)
lc.text(SOB[0] + SOB[2] / 2, 216, 'MLAAttention —— 插座', 13, C_DEC_T, 'middle', True, tag='so:t')
lc.text(SOB[0] + SOB[2] / 2, 236, '一个注意力层,两套家族的客户', 9, lc.C_MUTE, 'middle', tag='so:s')
# 上接口:decode 抽象
lc.rect(SOB[0] + 24, 258, SOB[2] - 48, 88, '#ffffff', C_DEC, rx=7, sw=1.5)
lc.text(SOB[0] + SOB[2] / 2, 280, 'decode 轴抽象(特形 kernel)', 9.5, C_DEC_T, 'middle', True, tag='so:d1')
lc.text(SOB[0] + SOB[2] / 2, 300, 'forward_mha(q, k, v) —— 标准形状', 8.5, '#334155', 'middle',
        maxw=SOB[2] - 60, tag='so:d2')
lc.text(SOB[0] + SOB[2] / 2, 318, 'forward_mqa(q, cache) —— 单头读潜 cache', 8.5, '#334155',
        'middle', maxw=SOB[2] - 60, tag='so:d3')
lc.text(SOB[0] + SOB[2] / 2, 336, '双抽象并存,按批切刀各走各的(站 11)', 7.5, lc.C_MUTE, 'middle',
        maxw=SOB[2] - 60, tag='so:d4')
# 下接口:prefill 装配
lc.rect(SOB[0] + 24, 360, SOB[2] - 48, 88, '#ffffff', C_PRE, rx=7, sw=1.5)
lc.text(SOB[0] + SOB[2] / 2, 382, 'prefill 轴抽象(标准 MHA)', 9.5, C_PRE_T, 'middle', True, tag='so:p1')
lc.text(SOB[0] + SOB[2] / 2, 402, 'run_prefill_new_tokens(q, k, v)', 8.5, '#334155', 'middle',
        maxw=SOB[2] - 60, tag='so:p2')
lc.text(SOB[0] + SOB[2] / 2, 420, '上投影完成后形状回归标准', 8.5, '#334155', 'middle', tag='so:p3')
lc.text(SOB[0] + SOB[2] / 2, 438, '__init__ 断言后端 is_mla(L423-L436)', 7.5, lc.C_MUTE, 'middle',
        maxw=SOB[2] - 60, tag='so:p4')
lc.text(SOB[0] + SOB[2] / 2, 466, '__init__ 同时装配第二套 prefill 后端(L520-L550)', 8, C_DEC_T,
        'middle', maxw=SOB[2] - 40, tag='so:n')
lc.text(SOB[0] + SOB[2] / 2, 482, 'vllm/model_executor/layers/attention/mla_attention.py', 7.5,
        lc.C_FAINT, 'middle', tag='so:f')

# ---------------- 左翼:decode 家族 ----------------
LF = (60, 150, 420, 400)
lc.rect(*LF, '#f0fdf4', C_DEC, rx=10, sw=1.8)
lc.text(LF[0] + LF[2] / 2, 174, 'decode 家族(is_mla 俱乐部)', 12, C_DEC_T, 'middle', True, tag='lf:t')
# 身份证三件套
lc.rect(LF[0] + 24, 192, LF[2] - 48, 118, '#ffffff', C_DEC, rx=7, sw=1.3)
lc.text(LF[0] + LF[2] / 2, 212, '身份证三件套', 9.5, C_DEC_T, 'middle', True, tag='lf:id')
lc.text(LF[0] + LF[2] / 2, 232, 'get_kv_cache_shape → (num_blocks, block_size, head_size)', 8,
        '#334155', 'middle', maxw=LF[2] - 60, tag='lf:id1')
lc.text(LF[0] + LF[2] / 2, 250, '——没有 K/V 拆维,通用后端连形状都对不上', 8, lc.C_MUTE, 'middle',
        maxw=LF[2] - 60, tag='lf:id2')
lc.text(LF[0] + LF[2] / 2, 270, 'get_supported_head_sizes = [320, 576]', 8.5, C_DEC_T, 'middle',
        True, tag='lf:id3')
lc.text(LF[0] + LF[2] / 2, 290, 'is_mla() = True', 9.5, C_DEC_T, 'middle', True, tag='lf:id4')
lc.text(LF[0] + LF[2] / 2, 304, '(mla_attention.py:L1360-L1397)', 7.5, lc.C_FAINT, 'middle', tag='lf:id5')
# 成员注
lc.rect(LF[0] + 24, 322, LF[2] - 48, 92, '#ffffff', C_DEC, rx=7, sw=1.3)
lc.text(LF[0] + LF[2] / 2, 342, '成员:FlashMLA / FlashInfer-MLA / Triton-MLA …', 8.5, '#334155',
        'middle', maxw=LF[2] - 60, tag='lf:m1')
lc.text(LF[0] + LF[2] / 2, 362, '全族特形 kernel:单 KV『头』直读分页潜 cache', 8, '#334155', 'middle',
        maxw=LF[2] - 60, tag='lf:m2')
lc.text(LF[0] + LF[2] / 2, 386, '自报 reorder 阈值:FlashMLA 128 / FA-MLA 512', 8, C_DEC_T, 'middle',
        True, maxw=LF[2] - 60, tag='lf:m3')
lc.text(LF[0] + LF[2] / 2, 404, '(runner 每拍取各组最小值,站 7)', 7.5, lc.C_MUTE, 'middle', tag='lf:m4')
lc.text(LF[0] + LF[2] / 2, 436, '选择优先级表:ch21 已立(回指),本章不重表', 8.5, lc.C_MUTE, 'middle',
        maxw=LF[2] - 40, tag='lf:ref')

# ---------------- 右翼:prefill 家族 ----------------
RF = (1020, 150, 420, 400)
lc.rect(*RF, '#ecfeff', C_PRE, rx=10, sw=1.8)
lc.text(RF[0] + RF[2] / 2, 174, 'prefill 家族(MLAPrefillBackendEnum 五选一)', 11.5, C_PRE_T, 'middle',
        True, maxw=RF[2] - 20, tag='rf:t')
members = ['FLASH_ATTN', 'FLASHINFER', 'TRTLLM_RAGGED', 'TOKENSPEED_MLA', 'ROCM_AITER_FA']
for i, m in enumerate(members):
    y = 196 + i * 40
    lc.rect(RF[0] + 24, y, RF[2] - 48, 32, '#ffffff', C_PRE, rx=6, sw=1.3)
    lc.text(RF[0] + RF[2] / 2, y + 20, m, 9.5, C_PRE_T, 'middle', True, maxw=RF[2] - 60,
            tag='rf:' + m)
lc.rect(RF[0] + 24, 196 + 5 * 40, RF[2] - 48, 32, '#f8fafc', lc.C_MUTE, rx=6, sw=1.1, dash=True)
lc.text(RF[0] + RF[2] / 2, 196 + 5 * 40 + 20, 'CUSTOM(第三方注册槽,用前先注册)', 8.5, lc.C_MUTE,
        'middle', maxw=RF[2] - 60, tag='rf:custom')
lc.text(RF[0] + RF[2] / 2, 196 + 6 * 40 + 14, '成员自 vllm/v1/attention/backends/mla/prefill/registry.py:L34-L57', 7.5,
        lc.C_FAINT, 'middle', maxw=RF[2] - 40, tag='rf:src')
lc.text(RF[0] + RF[2] / 2, 196 + 6 * 40 + 36, '吃的形状:上投影后的标准 MHA——', 8.5, '#334155',
        'middle', maxw=RF[2] - 40, tag='rf:n1')
lc.text(RF[0] + RF[2] / 2, 196 + 6 * 40 + 54, '所以任何通用 prefill 后端都能加入', 8.5, '#334155',
        'middle', maxw=RF[2] - 40, tag='rf:n2')

# ---------------- 两翼 → 插座 的独立选择箭头 ----------------
lc.parrow([(LF[0] + LF[2], 302), (SOB[0], 302)], C_DEC, 2.2, 'std')
lc.text((LF[0] + LF[2] + SOB[0]) / 2, 292, '各选各的 ①', 9, C_DEC_T, 'middle', True, tag='ar:dec')
lc.parrow([(RF[0], 404), (SOB[0] + SOB[2], 404)], C_PRE, 2.2, 'std')
lc.text((RF[0] + SOB[0] + SOB[2]) / 2, 394, '各选各的 ②', 9, C_PRE_T, 'middle', True, tag='ar:pre')
lc.text((LF[0] + LF[2] + SOB[0]) / 2, 316, 'decode 轴', 8, lc.C_MUTE, 'middle', tag='ar:decn')
lc.text((RF[0] + SOB[0] + SOB[2]) / 2, 418, 'prefill 轴', 8, lc.C_MUTE, 'middle', tag='ar:pren')

# ---------------- 底部:cache 形状对照注 ----------------
BB = (60, 586, 1384, 66)
lc.rect(*BB, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(BB[0] + BB[2] / 2, 608, '特形 cache:(num_blocks, block_size, 576)——num_kv_heads=1、不拆 K/V;MLAAttentionSpec 自报 head_size = kv_lora_rank + qk_rope_head_dim = 576',
        9.5, lc.C_TXT, 'middle', True, maxw=1360, tag='bb:t')
lc.text(BB[0] + BB[2] / 2, 630, '两轴独立:decode 后端管 kernel 读潜 cache,prefill 后端管标准 MHA 计算——不存在一个后端同时伺候两轴',
        8.5, lc.C_MUTE, 'middle', maxw=1360, tag='bb:l1')

# ---------------- 页脚 ----------------
lc.text(MX, 700, '图例:绿 = decode 家族/GPU kernel 侧 · 青 = prefill 家族/标准 MHA 计算侧 · 两支箭头 = 两轴独立选择',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 720, '身份证/成员/装配位 = vLLM v0.27.1(6e448d0ea) 源码原文 · 选择优先级表见 ch21(回指)· 阈值 128/512 见 flashmla.py:L121 / flashattn_mla.py:L118',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-two-backend-families.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
