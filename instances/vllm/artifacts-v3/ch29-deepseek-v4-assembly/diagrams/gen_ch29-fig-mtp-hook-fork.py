#!/usr/bin/env python3
"""ch28 机制图 · MTP 钩子三段接力(ch28-fig-mtp-hook-fork, 模板 flow)

放大自 L0『GPU 执行臂·模型层 forward + 编译』块尾部与采样列交界的 drafter 重绑点
(L2 拍片⑨ 的机制版下钻; 驱动机制 propose/verify 归 ch32/33, 本图止步钩子两端)。

claim: MTP 钩子三段接力——模型层在层尾 mhc_post 之后、hc_head 之前把多流残差 copy_
进 _mtp_hidden_buffer(get_mtp_target_hidden_states 暴露); runner 在采样后 getattr
探测该钩子重绑 drafter 输入; draft 侧 e_proj/h_proj 融合 embed+prev residual 后
复用整层 DecoderLayer, hc_head 推迟到自己的 compute_logits——pre-hc_head 形态全程保持。

锚点 = nvidia/model.py:L1095-L1106/L1200-L1212 · v1/worker/gpu_model_runner.py:L5198-L5206 ·
nvidia/mtp.py:L144-L189/L253-L279。坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 920
MX = 56
BXR = 1608

# ---------------- 标题区 ----------------
lc.text(MX, 34, '主干尾部的样品分流口：定型前的多流残差，先舀一勺给 draft 车间', 16, lc.C_TXT,
        'start', True, maxw=1040, tag='title')
lc.text(MX, 58, 'mhc_post 塌回 3D 后先 copy_ 进 _mtp_hidden_buffer 再 hc_head 定型；runner 采样后 getattr 探测重绑，draft 车间全程拿着 pre-hc_head 残差干活，hc_head 推迟到自己的 compute_logits',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
_ch = '放大自 L0『模型层 forward + 编译』块尾部 × 采样列交界 · L2 拍片⑨'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')


def box(x, y, w, h, title, lines, file='', stroke=lc.C_GPU_S, fill='#ffffff', dash=False,
        tfs=10, lfs=8.5, tc=lc.C_TXT):
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.5, dash=dash)
    lc.text(x + 12, y + 19, title, tfs, tc, 'start', True, maxw=w - 24, tag='t:' + title)
    yy = y + 37
    for s in lines:
        lc.text(x + 12, yy, s, lfs, '#334155', 'start', maxw=w - 22, tag='l:' + s[:12])
        yy += 15.5
    if file:
        lc.text(x + 12, y + h - 9, file, 7.5, lc.C_FAINT, 'start', maxw=w - 20, tag='f:' + title)


# ---------------- ① 模型层尾部 ----------------
lc.rect(MX, 96, 500, 556, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.8)
lc.text(MX + 16, 118, '① 模型层尾部（DeepseekV4Model.forward 收尾）', 11.5, '#14532d', 'start',
        True, maxw=460, tag='p1:t')
box(MX + 20, 138, 210, 62, '层尾 mhc_post', ['塌回多流残差', '(T, hc_mult, H)'])
lc.seg(MX + 125, 200, MX + 125, 228, lc.C_GPU_S, 1.8, 'std')
box(MX + 20, 228, 210, 62, '多流残差 chip', ['pre-hc_head 残差', '定型前的「草稿」'])
# 主路（左列下行）
lc.seg(MX + 125, 290, MX + 125, 322, lc.C_GPU_S, 2.0, 'std')
lc.text(MX + 133, 312, '主路（定型）', 8.5, lc.C_GPU_S, 'start', tag='main:lab')
box(MX + 20, 322, 210, 62, 'hc_head', ['加权压回单流 (T,H)'], '', fill=lc.C_GPU_F)
lc.seg(MX + 125, 384, MX + 125, 414, lc.C_GPU_S, 1.8, 'std')
box(MX + 20, 414, 210, 56, 'norm', ['→ hidden_states (T,H)'])
lc.seg(MX + 125, 470, MX + 125, 500, lc.C_GPU_S, 1.8, 'std')
box(MX + 20, 500, 210, 70, 'compute_logits', ['→ LogitsProcessor → 采样位', '（与 Llama 同款两行，',
    'ch23 契约原样成立）'], 'model.py:L1515-L1538')
# 旁路（右列）
box(MX + 264, 236, 216, 122, '_mtp_hidden_buffer', [
    '.copy_(flatten(1))', 'shape (max_num_batched_tokens,', '  hc_mult × hidden_size)',
    'spec_config 用 eagle /', 'draft_model 才建'], 'model.py:L1095-L1106', dash=True)
lc.seg(MX + 230, 267, MX + 264, 267, lc.C_GPU_S, 1.6, 'std', dash=True)
lc.text(MX + 234, 256, '旁路 copy_', 7.5, lc.C_GPU_S, 'start', tag='bypass:lab')
box(MX + 264, 378, 216, 100, '暴露口', ['get_mtp_target_hidden_states()', '返回 buffer 切片',
    '（先暂存后定型：copy_ 先于 hc_head）'], 'model.py:L1200-L1212')
lc.seg(MX + 372, 358, MX + 372, 378, lc.C_GPU_S, 1.6, 'std')
lc.seg(MX + 372, 466, MX + 372, 470, lc.C_GPU_S, 1.0, None)
lc.text(MX + 20, 596, '层尾顺序：mhc_post → buffer.copy_(flatten(1)) → hc_head → norm',
        9, '#166534', 'start', True, maxw=460, tag='p1:order')
lc.text(MX + 20, 616, '（定型前的形态从这里离开主干）', 8.5, lc.C_MUTE, 'start', maxw=460,
        tag='p1:note')
lc.text(MX + 20, 636, 'vllm/models/deepseek_v4/nvidia/model.py', 8, lc.C_FAINT, 'start',
        maxw=460, tag='p1:f')

# ---------------- ② runner 采样后 ----------------
MPX = 596
lc.rect(MPX, 96, 400, 556, lc.C_SAM_F, lc.C_SAM_S, rx=10, sw=1.8)
lc.text(MPX + 16, 118, '② GPUModelRunner 采样后', 11.5, '#9d174d', 'start', True, maxw=340,
        tag='p2:t')
lc.text(MPX + 16, 140, '（采样列角色）', 8.5, lc.C_MUTE, 'start', tag='p2:s')
box(MPX + 20, 158, 360, 56, '采样完成', ['sampled_token_ids 已出；', 'hidden_states 已完成本轮消费'],
    '', stroke=lc.C_SAM_S, fill='#ffffff')
lc.seg(MPX + 200, 214, MPX + 200, 242, lc.C_SAM_S, 1.8, 'std')
box(MPX + 20, 242, 360, 136, 'getattr 探测钩子', [
    "alt = getattr(model,", "  'get_mtp_target_hidden_states',",
    '  lambda: None)()', 'if alt is not None:', '  hidden_states = alt  ← 重绑 drafter 输入'],
    'gpu_model_runner.py:L5198-L5206', stroke=lc.C_SAM_S, lfs=8.5)
lc.text(MPX + 20, 388, '注释原话：\'Let the target override the hidden state fed to the drafter\'',
        8.5, '#9d174d', 'start', maxw=370, tag='p2:quote')
lc.text(MPX + 20, 404, '(e.g. DeepSeek V4 MTP needs the pre-hc_head residual)', 8, lc.C_MUTE,
        'start', maxw=370, tag='p2:quote2')
box(MPX + 20, 428, 360, 96, '双向解耦（谁也不 import 谁）', [
    '· 模型侧不知道 runner 存在', '· runner 侧不 import 模型类型',
    '· 缺钩子 → alt=None → 原样用采样位 hidden', '（Llama 类走这条退化路）'], '',
    stroke=lc.C_SAM_S, lfs=8.5)
# buffer → runner 取数
lc.seg(MX + 480, 302, MPX + 20, 302, lc.C_ENG_S, 2.0, 'std')
lc.text((MX + 480 + MPX + 20) / 2, 292, '采样后来取', 8.5, lc.C_ENG_S, 'middle', True,
        maxw=110, tag='fetch:lab')

# ---------------- ③ draft 侧车间 ----------------
RPX = 1036
lc.rect(RPX, 96, BXR - RPX, 556, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(RPX + 16, 118, '③ draft 侧车间（DeepseekV4MTP · nvidia/mtp.py）', 11.5, lc.C_TXT,
        'start', True, maxw=520, tag='p3:t')
chain = [
    (138, 58, 'reshape 回 3D (T, hc_mult, D)', ['flat 残差还原训练期布局'], ''),
    (212, 62, 'fused_mtp_input_rmsnorm', ['mask 首位 + enorm + hnorm 一核（mtp.py:L159-L168）'], ''),
    (288, 62, 'h_proj(prev) + e_proj(embed)', ['双投影融合：草稿残差 + 本步词嵌入（L177-L179）'], ''),
    (364, 62, 'mtp_block = 复用整层 DecoderLayer', ['同一套 hc 门控骨架跑一遍（L180-L182）'], ''),
    (440, 74, 'mhc_post → 返回 flat 残差 (T, hc_mult×D)', [
        'num_speculative_tokens>1：可作下一 spec step', '的 previous_hidden_states 续喂（L186-L189）'], ''),
    (528, 74, '延迟的 hc_head（画在最后一步）', ['compute_logits 里才跑 hc_head_fused_',
        'kernel_tilelang + shared_head（L253-L279）'], 'mtp.py:L253-L279'),
]
prev_y = None
for (yy, hh, title, lines, f) in chain:
    dash = (yy == 528)
    box(RPX + 20, yy, 500, hh, title, lines, f, stroke=lc.C_MUTE, dash=dash, lfs=8)
    if prev_y is not None:
        lc.seg(RPX + 270, prev_y, RPX + 270, yy, lc.C_MUTE, 1.8, 'std')
    prev_y = yy + hh
# 续喂回环（右缘）
lc.parrow([(RPX + 520, 477), (RPX + 544, 477), (RPX + 544, 167), (RPX + 520, 167)],
          lc.C_MUTE, 1.4, 'std', dash=True)
lc.text(RPX + 538, 320, '下一 spec step 续喂', 8, lc.C_MUTE, 'middle', maxw=90, tag='p3:loop')
# runner → draft 重绑
lc.parrow([(MPX + 380, 303), (MPX + 396, 303), (MPX + 396, 167), (RPX + 20, 167)],
          lc.C_SAM_S, 2.0, 'std')
lc.text(996, 131, '重绑 drafter 输入', 9, '#9d174d', 'middle', True, maxw=200, tag='rebind:lab')

# ---------------- 底部三注 ----------------
NY, NH = 676, 160
lc.rect(MX, NY, 500, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(MX + 14, NY + 20, '形态红利：pre-hc_head', 10.5, lc.C_TXT, 'start', True, maxw=400,
        tag='n1:t')
for i, s in enumerate([
        '· 「还没被输出头定型的多流草稿」这个形态，',
        '   在单流残差的模型里根本不存在',
        '· mHC 把多流残差留在层尾，MTP 的原料免费',
        '· draft 侧拿到的是完整 hc_mult 条流，',
        '   而不是压扁后的单流 hidden']):
    lc.text(MX + 14, NY + 42 + i * 17, s, 8.5, '#334155', 'start', maxw=470, tag='n1:l%d' % i)
lc.text(MX + 14, NY + NH - 10, '（hc_head 延迟 = 形态红利的技术代价）', 8, lc.C_FAINT, 'start',
        maxw=470, tag='n1:f')

lc.rect(620, NY, 390, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(634, NY + 20, '部署真值与完整性', 10.5, lc.C_TXT, 'start', True, maxw=340, tag='n2:t')
for i, s in enumerate([
        '· speculative_config.method=mtp、',
        '   num_speculative_tokens=2',
        '   （DeepSeek-V4-Flash eval 配置）',
        '· MTP 层缺权重即报错——完整性检查',
        '   （mtp.py:L493-L504）']):
    lc.text(634, NY + 42 + i * 17, s, 8.5, '#334155', 'start', maxw=360, tag='n2:l%d' % i)
lc.text(634, NY + NH - 10, 'tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-…yaml', 7.5,
        lc.C_FAINT, 'start', maxw=360, tag='n2:f')

lc.rect(1030, NY, BXR - 1030, NH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(1044, NY + 20, '边界（本图画到哪为止）', 10.5, lc.C_TXT, 'start', True, maxw=400, tag='n3:t')
for i, s in enumerate([
        '· 驱动机制（propose / verify / 拒绝采样）',
        '   归 ch32/33——本图止步钩子两端',
        '· draft 侧形状账：3D ↔ flat 全程 pre-hc_head，',
        '   hc_head 只在算 logits 的那一刻出现',
        '· 采样位切片策略仍归 runner（ch23 契约）']):
    lc.text(1044, NY + 42 + i * 17, s, 8.5, '#334155', 'start', maxw=540, tag='n3:l%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 872, '图例：绿 = 模型层（GPU 执行臂角色）· 品红 = runner 采样段（采样列角色）· 灰 = draft 侧 · 绿虚线框 = 旁路暂存 · 灰虚框 = 延迟步骤',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 892, '锚点 = vllm/models/deepseek_v4/nvidia/model.py:L1095-L1106/L1200-L1212/L1515-L1538 · v1/worker/gpu_model_runner.py:L5198-L5206 · nvidia/mtp.py:L144-L189/L253-L279/L493-L504',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-mtp-hook-fork.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
