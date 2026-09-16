#!/usr/bin/env python3
"""ch29 机制图 · MoE 装配双后端分岔(ch29-fig-moe-dual-backend, 模板 flow)

放大自 L0『GPU 执行臂·模型层 forward + 编译』块里 DecoderLayer 的 FFN 半层
(L2 拍片④/⑨ 的机制版下钻; ch23 同位置是 dense SwiGLU MLP)。

claim: 同一份 gate/路由代码后面是两个完全不同的专家后端——mega 路(EP 单算子大厦,
FP4+对称缓冲+shared 外部相加) vs fused 路(通用 FusedMoE 工厂, TP 切+shared 内部聚合);
分岔权在 kernel_config.moe_backend 装配期一次定死, 三条 NotImplementedError 是这道门的守卫。

数字 = 本章实跑场景表(E=4 top-2 / num_hash_layers=1 / EP world=2 / H=I=128, 与正文同源)·
锚点 = vllm/models/deepseek_v4/nvidia/model.py:L519-L764 · vllm/config/kernel.py:L121-L131。
坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 1000
MX = 56
BXR = 1608

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'MoE 装配：一个旋钮（moe_backend）装配期定终身，两种楼各走各的', 16, lc.C_TXT,
        'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '同一份 gate 打分代码后面，可接 MegaMoE 专属大厦（只收 FP4+EP+sqrtsoftplus 租客，一步直达单算子）或通用 FusedMoE 商场（什么精度都收、走 TP）',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
_ch = '放大自 L0『模型层 forward + 编译』块 FFN 半层 · L2 拍片④/⑨'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')


def box(x, y, w, h, title, lines, file='', stroke=lc.C_GPU_S, fill='#ffffff', dash=False,
        tfs=10.5, lfs=8.5, tc=lc.C_TXT, lc_='#334155'):
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=dash)
    lc.text(x + 12, y + 20, title, tfs, tc, 'start', True, maxw=w - 24, tag='t:' + title)
    yy = y + 39
    for s in lines:
        lc.text(x + 12, yy, s, lfs, lc_, 'start', maxw=w - 22, tag='l:' + s[:12])
        yy += 16
    if file:
        lc.text(x + 12, y + h - 9, file, 8, lc.C_FAINT, 'start', maxw=w - 20, tag='f:' + title)


# ---------------- 公共段: gate → fused_topk_bias ----------------
lc.text(MX, 122, '公共段（两路共用同一份路由代码）', 11, lc.C_TXT, 'start', True, maxw=400, tag='pub:t')
PY, PH = 136, 118
box(MX, PY, 100, 92, 'hidden', ['(T, H)'], '', fill='#ffffff')
box(176, PY, 250, PH, 'gate = GateLinear', [
    '打分恒 fp32（out_dtype=float32）', '出 [T, E]（本例 128×4）', '六级分派 · 六种出分方式'],
    'router/gate_linear.py:L18-L48')
box(452, PY, 300, PH, 'fused_topk_bias', [
    'scores = sqrt(softplus(g))（恒非负）', '选择/加权三分离：bias 只影响选择、',
    '不影响权重；hash 表只覆写 ids', '（三分离详解见本章路由节）'],
    'router/fused_topk_bias_router.py:L75-L119')
AY = PY + 52
lc.seg(MX + 100, AY, 176, AY, lc.C_GPU_S, 1.8, 'std')
lc.seg(176 + 250, AY, 452, AY, lc.C_GPU_S, 1.8, 'std')
# hash 旁路（虚线, 从 gate 上方接入 topk）
lc.parrow([(452 + 120, PY - 4), (452 + 120, PY - 20), (301, PY - 20), (301, PY)],
          lc.C_MUTE, 1.4, 'std', dash=True)
lc.text(376, PY - 27, 'hash 层旁路：前 num_hash_layers 层 tid2eid(16,2) 查表免打分——ids 定死，分数只算权重',
        8.5, lc.C_MUTE, 'middle', maxw=520, tag='hash:lab')

# 分岔旋钮
KN_X, KN_W = 800, 330
box(KN_X, PY, KN_W, PH, '分岔旋钮：kernel_config.moe_backend', [
    "use_mega_moe =（moe_backend == 'deep_gemm_mega_moe'）",
    '布尔二值，装配期一次定死（构造期 if 二岔）', '默认 auto → fused 路'],
    'model.py:L533-L535 · vllm/config/kernel.py:L121-L131', lfs=8)
lc.seg(452 + 300, AY, KN_X, AY, lc.C_GPU_S, 1.8, 'std')

# ---------------- 三条守卫 ----------------
GY, GH = 310, 124
lc.text(MX, GY - 24, 'mega 三前置守卫：任一不满足，建专家之前当场 NotImplementedError', 10,
        lc.C_ABORT, 'start', True, maxw=760, tag='guard:t')
lc.text(MX, GY - 10, 'fail-fast 不静默降级——失败时零参数落地（raise 位于任何专家参数分配之前）', 8.5,
        lc.C_MUTE, 'start', maxw=760, tag='guard:t2')
guards = [
    ('守卫① EP 强制', ['mega=on 但 EP=off →', '', '"...requires expert parallel.',
                        'Enable it with --enable-expert-parallel,', 'or pick a different moe backend."'], 'model.py:L536-L541'),
    ('守卫② sqrtsoftplus only', ['mega=on 但 scoring=softmax →', '', '"...currently supports',
                                  'sqrtsoftplus routing only."'], 'model.py:L552-L555'),
    ('守卫③ fp4 only', ['mega=on 但 expert_dtype=fp8 →', '', '"...only supports fp4 experts;',
                         "got expert_dtype='fp8'. Drop --kernel-config", 'moe_backend=deep_gemm_mega_moe ..."'],
     'model.py:L556-L561'),
]
gw = 500
for i, (t, lines, f) in enumerate(guards):
    gx = MX + i * (gw + 24)
    box(gx, GY, gw, GH, t, lines, f, stroke=lc.C_ABORT, dash=True, tc=lc.C_ABORT, lfs=8)
lc.parrow([(KN_X + KN_W / 2, PY + PH), (KN_X + KN_W / 2, 268), (306, 268), (306, GY)],
          lc.C_ABORT, 1.4, None, dash=True)

# ---------------- 两个专家后端 panel ----------------
BY, BH_ = 486, 380
# ---- mega 路（左） ----
lc.rect(MX, BY, 760, BH_, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(MX + 16, BY + 24, 'mega 路：MegaMoE 专属大厦（EP 切物理专家）', 11.5, '#14532d', 'start',
        True, maxw=560, tag='mega:t')
lc.text(MX + 744, BY + 24, 'use_mega_moe=true', 9, lc.C_GPU_S, 'end', True, tag='mega:b')
box(MX + 20, BY + 40, 340, 120, 'experts = DeepseekV4MegaMoEExperts', [
    'FP4 字节参数（本例 E_local=2）：', 'w13 (2,256,64) uint8 + scale (2,256,4)', 'w2 (2,128,64) uint8 + scale (2,128,4)',
    'tid2eid dtype=int64'], 'model.py:L202-L246', lfs=8)
box(MX + 384, BY + 40, 356, 120, '单算子前向', [
    '① prepare_megamoe_inputs：量化+重排', '   进对称缓冲 symm_buffer',
    '② deep_gemm.fp8_fp4_mega_moe(', '   y, l1, l2, symm_buffer,', '   activation_clamp, fast_math)'],
    'model.py:L439-L512', lfs=8)
lc.text(MX + 20, BY + 176, '对称缓冲按 7 元组键 (group, device, E, max_tokens, topk, H, I) 跨层复用（model.py:L364-L391）',
        8.5, '#166534', 'start', maxw=720, tag='mega:symm')
box(MX + 20, BY + 192, 340, 96, 'shared_experts（外部相加）', [
    'DeepseekV4MLP · reduce_results=use_mega_moe', '（EP 下输出已 all_reduce，免二次）',
    'final_hidden_states += shared_output'], 'model.py:L739-L741', lfs=8)
lc.parrow([(MX + 360, BY + 240), (MX + 384, BY + 240)], lc.C_GPU_S, 2.0, 'std')
lc.text(MX + 372, BY + 228, '+=', 10, '#14532d', 'middle', True, tag='mega:add')
box(MX + 384, BY + 192, 356, 104, 'experts 输出 y (T,H) bf16', [
    'activation_clamp=swiglu_limit', '（clamp 进单算子，不另开核）',
    'EPLB：logical→physical 重排在进缓冲前'], 'model.py:L467-L486', lfs=8)
# hash mega note
lc.text(MX + 20, BY + 306, 'hash 层：ids 查表（int64）免打分，分数仍算（定工资）——本章路由节案例 C', 8.5,
        lc.C_MUTE, 'start', maxw=720, tag='mega:hash')
lc.text(MX + 20, BY + 330, '部署真值（DeepSeek-V4-Flash eval）：--moe-backend deep_gemm_mega_moe + --enable-expert-parallel + TP=2',
        9, '#9a3412', 'start', True, maxw=730, tag='mega:deploy')
lc.text(MX + 20, BY + 348, '——三重限定全部满足的活体（GSM8K · MTP 2 token）', 8.5, lc.C_MUTE, 'start',
        maxw=730, tag='mega:deploy2')

# ---- fused 路（右） ----
FX = 856
lc.rect(FX, BY, BXR - FX, BH_, '#ffffff', lc.C_MUTE, rx=10, sw=1.8)
lc.text(FX + 16, BY + 24, 'fused 路：通用 FusedMoE 商场（TP 切）', 11.5, lc.C_TXT, 'start', True,
        maxw=480, tag='fused:t')
lc.text(FX + (BXR - FX) - 16, BY + 24, 'use_mega_moe=false（auto 默认）', 9, lc.C_MUTE, 'end',
        True, tag='fused:b')
box(FX + 20, BY + 40, 330, 108, 'experts = FusedMoE 工厂', [
    'FusedMoEFactory(shared_experts=…, gate=…)', '什么精度都收（fp8/bf16/…）· TP 切',
    'tid2eid dtype=int32', 'renormalize/scaling/bias 全由工厂收口'], 'model.py:L657-L702', lfs=8)
box(FX + 20, BY + 192, 700, 96, 'is_internal_router 分岔', [
    'true：router 在 MoERunner 内部——hidden 直喂 experts（gate 不在外面先算）',
    'false：外部 gate 先算 router_logits，再进 FusedMoE（本图公共段画的就是这支）',
    '与 mega 路相反：shared_experts 聚合在 FusedMoE 内部（reduce_results=false）'],
    'model.py:L745-L764', lfs=8.5)
lc.parrow([(FX + 185, BY + 148), (FX + 185, BY + 192)], lc.C_MUTE, 1.6, 'std')
lc.text(FX + 360, BY + 166, '两种进法都合法——分岔在工厂装配期自报', 8.5, lc.C_MUTE, 'start',
        maxw=320, tag='fused:d')
# noaux_tc note
lc.text(FX + 20, BY + 306, 'noaux_tc 层：e_score_correction_bias(4,) fp32 挂在 gate 上（与 tid2eid if/elif 互斥）',
        8.5, lc.C_MUTE, 'start', maxw=700, tag='fused:bias')
lc.text(FX + 20, BY + 330, 'SP 门（四条件合取）：pp=1 ∧ EP ∧ tp>1 ∧ (mega ∨ dp>1)', 8.5, '#334155',
        'start', maxw=700, tag='sp:t')
lc.text(FX + 20, BY + 348, 'tp=1→false · pp=2→false · dp=2+auto→true（model.py:L805-L813；原理归分布式章）',
        8.5, lc.C_MUTE, 'start', maxw=700, tag='sp:l')

# 分岔箭头: 旋钮 → 两 panel（盲审修复 2026-09-14 重布线：原两条纵线 x=860/1040 从旋钮底
# 直落、整段纵穿守卫②红虚线框(x 580-1080)并切断其上下边框；现改走框外净空走廊——
# fork:a 经守卫①/②之间缝(x=568)下行、fork:b 经守卫②/③之间缝(x=1092)下行，
# 两纵线距两侧框边各留 ~9px 净空，横段走在守卫带上方(y=296)/下方(y=458)的空带。）
lc.parrow([(KN_X + 20, PY + PH), (KN_X + 20, 296), (568, 296), (568, 458), (436, 458), (436, BY)],
          lc.C_GPU_S, 2.2, 'std')
lc.text(500, 474, 'true → mega（EP）', 9, lc.C_GPU_S, 'middle', True, tag='fork:a')
lc.parrow([(1092, PY + PH), (1092, 458), (1232, 458), (1232, BY)],
          lc.C_MUTE, 2.2, 'std')
lc.text(1160, 474, 'false → fused（TP）', 9, lc.C_MUTE, 'middle', True, tag='fork:b')

# ---------------- 页脚 ----------------
lc.text(MX, 892, '图例：绿 = mega 路计算（GPU 执行臂角色）· 灰白 = fused 路（通用工厂）· 红虚 = 三条守卫（fail-fast）· 灰虚线 = hash 旁路',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 912, '形状/判定/报错消息 = 本章实跑场景表（E=4 top-2 · num_hash_layers=1 · EP world=2 · H=I=128，与正文同源，消息原文逐字）· 锚点 = vllm/models/deepseek_v4/nvidia/model.py:L519-L764',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')
lc.text(MX, 932, '参数形状取自树内测试构造 tests/models/test_deepseek_v4_mega_moe.py:L58-L67 · tid2eid 本例 randint 示形（真实模型从 checkpoint 装载训练好的表）',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-moe-dual-backend.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
