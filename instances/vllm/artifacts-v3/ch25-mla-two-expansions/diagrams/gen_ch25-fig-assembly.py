#!/usr/bin/env python3
"""ch25 机制图 · MLA 装配期类型展开(figure ch25-fig-assembly,模板 layout)

放大自 L0『模型层 MLA 框』的装配期静态结构——L2 站 1-3(选型三岔→投影积木→Wrapper/插座)
的类型展开。三段:上=DecoderLayer 三岔选型;中=六块投影积木(DSV3 实尺);下=MLAModules
打包进 Wrapper(PluggableLayer 注册点)内嵌 MLAAttention 插座,kv_b_proj 引用直连插座。

claim:MLA 不是运行时开关而是装配期定死的层类型:DecoderLayer 三岔(老 Deepseek MHA /
use_mla / 普通)选出 DeepseekV2MLAAttention,它装六块投影积木打包成 MLAModules 注入
Wrapper,内层 MLAAttention 插座接过 kv_b_proj 引用(后面的权重吸收重排要靠它拿权重)。

数字全部取自 figure spec 的 numbers(三岔条件 deepseek_v2.py:L1226-L1238 / use_mla 定义
model.py:L1791-L1792 / 六积木 DSV3 形状 host 实测例化 / 注册点 mla.py:L34-L36 /
kv_b_proj 引用交接 mla.py:L110-L127)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 940
MX = 56
BXR = 1444

C_REF = lc.C_API_S          # 蓝 = 权重引用线(图例声明,区别于角色色)

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'MLA 的装配现场:不是运行时开关,是 DecoderLayer 装配时一次定死的层类型',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '三岔选型 → 六块标准投影积木 → MLAModules 打包注入 Wrapper → 内层插座接过 kv_b_proj 引用(吸收重排的权重来源)',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 1-3 的类型展开'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- ① 三岔选型 ----------------
B1Y = 88
lc.rect(MX, B1Y, 26, 26, lc.C_BEAT_F, lc.C_BEAT_S, rx=13, sw=1.2)
lc.text(MX + 13, B1Y + 18, '①', 12, lc.C_BEAT_T, 'middle', True)

DLB = (596, B1Y, 308, 52)      # DecoderLayer 框
lc.rect(*DLB, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(DLB[0] + DLB[2] / 2, B1Y + 22, 'DecoderLayer 装配:按 config 三岔', 11.5, lc.C_TXT,
        'middle', True, tag='dl:t')
lc.text(DLB[0] + DLB[2] / 2, B1Y + 41, 'vllm/model_executor/models/deepseek_v2.py:L1226-L1238',
        8, lc.C_FAINT, 'middle', tag='dl:f')

# 三个类框(y 同排)
CLS_Y, CLS_H = 216, 74
cls = [
    (120, 320, 'DeepseekAttention', '老 Deepseek:MHA', 'K/V 直存,无潜向量压缩', lc.C_MUTE,
     '#ffffff', ''),
    (590, 320, 'DeepseekV2MLAAttention', '本章:只做拼装,不写新算子', '六块标准积木的组合器', lc.C_GPU_S,
     lc.C_GPU_F, '本章'),
    (1104, 320, 'DeepseekV2Attention', '普通注意力', '不带 MLA 压缩的常规路径', lc.C_MUTE,
     '#ffffff', ''),
]
cbox = {}
for x, w, t, s1, s2, stk, fl, badge in cls:
    lc.rect(x, CLS_Y, w, CLS_H, fl, stk, rx=8, sw=1.8 if badge else 1.4)
    lc.text(x + w / 2, CLS_Y + 24, t, 11.5, lc.C_TXT, 'middle', True, maxw=w - 60, tag='cls:' + t)
    lc.text(x + w / 2, CLS_Y + 45, s1, 8.5, lc.C_MUTE if not badge else '#166534', 'middle', tag='cls:s1' + t)
    lc.text(x + w / 2, CLS_Y + 61, s2, 8.5, lc.C_MUTE if not badge else '#166534', 'middle', tag='cls:s2' + t)
    if badge:
        bw = 16 + 11 * len(badge)
        lc.rect(x + w - bw - 8, CLS_Y + 6, bw, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
        lc.text(x + w - bw / 2 - 8, CLS_Y + 20, badge, 9.5, lc.C_ENG_S, 'middle', True)
    cbox[t] = (x, CLS_Y, w, CLS_H)

# 三岔箭头(DecoderLayer 底边 → 各类框顶边),条件标注在线侧
dlb_bx, dlb_by = DLB[0] + DLB[2] / 2, B1Y + DLB[3]
conds = [
    ('DeepseekAttention', 'use_mha = model_type=="deepseek"', '或全部 head dim == 0'),
    ('DeepseekV2MLAAttention', 'elif model_config.use_mla', ''),
    ('DeepseekV2Attention', 'else', ''),
]
for name, c1, c2 in conds:
    x, y, w_, h_ = cbox[name]
    lc.seg(dlb_bx, dlb_by, x + w_ / 2, CLS_Y,
           lc.C_GPU_S if name == 'DeepseekV2MLAAttention' else lc.C_MUTE, 1.8, 'std')
mid_x = (dlb_bx + 280) / 2
lc.text(mid_x - 30, 158, conds[0][1], 8.5, lc.C_MUTE, 'end', maxw=260, tag='c0a')
lc.text(mid_x - 30, 172, conds[0][2], 8.5, lc.C_MUTE, 'end', maxw=260, tag='c0b')
lc.text(dlb_bx + 10, 168, conds[1][1], 8.5, lc.C_GPU_S, 'start', True, tag='c1')
lc.text((dlb_bx + 1264) / 2 + 34, 158, conds[2][1], 8.5, lc.C_MUTE, 'start', tag='c2')

# use_mla 判定注(虚线框,整排下沿)
NB = (330, 318, 840, 44)
lc.rect(*NB, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(NB[0] + NB[2] / 2, 334, 'use_mla = is_deepseek_mla 且未设 VLLM_MLA_DISABLE(vllm/config/model.py:L1791-L1792)',
        9, lc.C_MUTE, 'middle', tag='nb:1')
lc.text(NB[0] + NB[2] / 2, 351, '装配期判定,一生一次——运行时不再切换层类型', 9, lc.C_MUTE, 'middle', tag='nb:2')

# ① → ② 箭头(选中的类框底 → 积木条带顶,kv_a_layernorm 块顶心)
_, _, _, _ = cbox['DeepseekV2MLAAttention']
lc.seg(750, CLS_Y + CLS_H, 868, 424, lc.C_GPU_S, 2.0, 'std')

# ---------------- ② 六积木条带 ----------------
B2Y = 388
lc.rect(MX, B2Y, 26, 26, lc.C_BEAT_F, lc.C_BEAT_S, rx=13, sw=1.2)
lc.text(MX + 13, B2Y + 18, '②', 12, lc.C_BEAT_T, 'middle', True)
lc.text(96, B2Y + 18, 'DeepseekV2MLAAttention 装的六块投影积木(DSV3 实尺真实例化)',
        11.5, lc.C_TXT, 'start', True, maxw=700, tag='strip:t')

BR_Y, BR_H = 424, 108
blocks = [
    ('fused_qkv_a_proj', '[2112, 7168]', '一个 MergedColumnParallelLinear', 'q 下投影 + KV 压缩一次乘完'),
    ('q_a_layernorm', '[1536]', 'q 低秩链中点归一化', ''),
    ('q_b_proj', '[24576, 1536]', 'q 潜段升回 128 头 × 192 维', ''),
    ('kv_a_layernorm', '[512]', '只打 kv_c(k_pe 不归一)', ''),
    ('kv_b_proj', '[32768, 512]', '潜向量 → 每头 K/V 上投影', '★ 吸收重排的主角'),
    ('o_proj', '[7168, 16384]', '注意力输出收尾', ''),
]
bw, gap = 212, 22
bx0 = MX + 4
fused_kv_x0 = None
for i, (nm, shp, d1, d2) in enumerate(blocks):
    x = bx0 + i * (bw + gap)
    hot = nm == 'kv_b_proj'
    lc.rect(x, BR_Y, bw, BR_H, lc.C_GPU_F if not hot else '#eff6ff',
            lc.C_GPU_S if not hot else C_REF, rx=7, sw=1.4 if not hot else 2.0)
    lc.text(x + bw / 2, BR_Y + 22, nm, 10, lc.C_TXT, 'middle', True, maxw=bw - 12, tag='b:' + nm)
    lc.text(x + bw / 2, BR_Y + 42, shp, 9, lc.C_GPU_S if not hot else C_REF, 'middle', True, tag='bs:' + nm)
    lc.text(x + bw / 2, BR_Y + 63, d1, 8, '#334155', 'middle', maxw=bw - 12, tag='bd1:' + nm)
    if d2:
        lc.text(x + bw / 2, BR_Y + 81, d2, 8, '#334155', 'middle', maxw=bw - 12, tag='bd2:' + nm)
    if nm == 'fused_qkv_a_proj':
        # 双拼块:W_DQ | W_DKV‖W_KR 融合
        sy = BR_Y + 90
        half = (bw - 18) / 2
        lc.rect(x + 9, sy, half - 3, 12, '#dcfce7', lc.C_GPU_S, rx=2, sw=1.0)
        lc.rect(x + 9 + half + 3, sy, half - 3, 12, '#cffafe', lc.C_KV_S, rx=2, sw=1.0)
        lc.text(x + 9 + (half - 3) / 2, sy + 9.5, 'W_DQ → 1536', 6.8, '#166534', 'middle', tag='fused:a')
        lc.text(x + 9 + half + 3 + (half - 3) / 2, sy + 9.5, 'W_DKV‖W_KR → 576', 6.8, '#155e75', 'middle', tag='fused:b')

# 链路注(条带下方)
lc.text(bx0 + 3 * bw + 2 * gap + bw / 2, BR_Y + BR_H + 18,
        'q 侧低秩链:fused 的 W_DQ 段 → q_a_layernorm → q_b_proj', 8.5, '#166534', 'middle', tag='chain:q')
lc.text(bx0 + 4 * bw + 3 * gap + bw / 2, BR_Y + BR_H + 34,
        'KV 侧压缩链:fused 的 W_DKV‖W_KR 段 → kv_a_layernorm → kv_b_proj', 8.5, '#155e75', 'middle', tag='chain:kv')
lc.text(bx0 + 5 * bw + 4 * gap + bw / 2, BR_Y + BR_H + 18, '注意力输出 → o_proj 收尾', 8.5, '#334155',
        'middle', tag='chain:o')

# ---------------- ③ MLAModules → Wrapper → 插座 ----------------
B3Y = 600
lc.rect(MX, B3Y, 26, 26, lc.C_BEAT_F, lc.C_BEAT_S, rx=13, sw=1.2)
lc.text(MX + 13, B3Y + 18, '③', 12, lc.C_BEAT_T, 'middle', True)

PKB = (96, 648, 250, 108)     # MLAModules 打包箱
lc.rect(*PKB, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(PKB[0] + PKB[2] / 2, 672, 'MLAModules', 11.5, lc.C_TXT, 'middle', True, tag='pk:t')
lc.text(PKB[0] + PKB[2] / 2, 692, '六积木整体打包(mla.py:L15)', 8,
        lc.C_MUTE, 'middle', maxw=PKB[2] - 14, tag='pk:s')
for j, nm in enumerate(['fused_qkv_a_proj + 归一化', 'q_b_proj / kv_b_proj', 'o_proj + RoPE 模块']):
    lc.text(PKB[0] + PKB[2] / 2, 710 + j * 14, '· ' + nm, 8, '#334155', 'middle', maxw=PKB[2] - 14,
            tag='pk:l%d' % j)

# 打包箭头:积木条带(kv_a_layernorm 块底) → MLAModules 顶右角
lc.parrow([(868, BR_Y + BR_H), (868, 596), (PKB[0] + PKB[2], 596), (PKB[0] + PKB[2], 648)],
          lc.C_GPU_S, 1.8, 'std')
lc.text(760, 588, '整体打包', 8.5, lc.C_GPU_S, 'start', tag='pk:lbl')

# Wrapper 大框
WRB = (430, 636, 1014, 180)
lc.rect(*WRB, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(WRB[0] + 18, 660, 'MultiHeadLatentAttentionWrapper —— 外层前向(q 低秩链 + 解耦 RoPE)',
        11.5, '#166534', 'start', True, maxw=620, tag='wr:t')
REG = (WRB[0] + 18, 672, 470, 40)
lc.rect(*REG, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(REG[0] + 8, 687, "@PluggableLayer.register('multi_head_latent_attention')", 8.5, lc.C_MUTE,
        'start', tag='reg:1')
lc.text(REG[0] + 8, 703, 'Wrapper 本身是注册点——OOT 平台可整体替换的后门(mla.py:L34-L36)', 8, lc.C_MUTE,
        'start', maxw=REG[2] - 16, tag='reg:2')

# 内嵌插座小框
SOB = (990, 690, 430, 106)
lc.rect(*SOB, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(SOB[0] + SOB[2] / 2, 712, 'MLAAttention —— 内层「插座」', 11, lc.C_TXT, 'middle', True, tag='so:t')
lc.text(SOB[0] + SOB[2] / 2, 731, '__init__ 断言后端 is_mla,注册 static_forward_context', 8.5,
        '#334155', 'middle', maxw=SOB[2] - 16, tag='so:s1')
lc.text(SOB[0] + SOB[2] / 2, 749, '构造时接过 kv_b_proj 引用(kv_b_proj=self.kv_b_proj)', 8.5,
        '#334155', 'middle', maxw=SOB[2] - 16, tag='so:s2')
lc.text(SOB[0] + SOB[2] / 2, 771, '吸收重排 process_weights_after_loading', 9, C_REF, 'middle', True,
        tag='so:s3')
lc.text(SOB[0] + SOB[2] / 2, 786, '正是从这里拿的权重(站 4)', 9, C_REF, 'middle', tag='so:s4')
# MLAModules → Wrapper 箭头
lc.seg(PKB[0] + PKB[2], 702, WRB[0], 702, lc.C_GPU_S, 2.0, 'std')
lc.text((PKB[0] + PKB[2] + WRB[0]) / 2, 694, '注入', 8.5, lc.C_GPU_S, 'middle', tag='wr:lbl')

# kv_b_proj 引用线(积木区 → 插座,蓝虚线):右缘出,经右空白绕进插座右边
kv_x = bx0 + 4 * (bw + gap) + bw          # kv_b_proj 积木右缘 = 1208
kv_y = BR_Y + BR_H / 2
lc.parrow([(kv_x, kv_y), (1458, kv_y), (1458, 743), (SOB[0] + SOB[2], 743)],
          C_REF, 1.8, 'std', dash=True)
lc.text(1452, 590, 'kv_b_proj 引用直连插座', 8.5, C_REF, 'end', True, tag='ref:lbl')

# Wrapper 底行:下游提示 + 源码路径
lc.text(WRB[0] + 18, 810, '装配收尾 = 站 4 权重吸收重排 · 每拍前向由此框 forward 承担(下一张图)', 8.5,
        lc.C_GPU_S, 'start', maxw=560, tag='wr:next')
lc.text(WRB[0] + WRB[2] - 16, 810, 'vllm/model_executor/layers/mla.py:L110-L127', 8, lc.C_FAINT, 'end')

# ---------------- 页脚 ----------------
lc.text(MX, 880, '图例:绿 = 模型层 / GPU 执行臂角色(L0 中列) · 蓝框 = kv_b_proj(吸收重排主角) · 蓝虚线 = 权重引用交接 · 灰 = 未选中的岔路',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 900, '形状账 = DSV3 实尺真实例化(host 实测) · 三岔/use_mla 条件 = vLLM v0.27.1(6e448d0ea) 源码原文',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-assembly.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
