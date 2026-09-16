#!/usr/bin/env python3
"""ch29 机制图 · 第三代 MLA 装配投影链(ch29-fig-mla-gen3-projections, 模板 flow)

放大自 L0『GPU 执行臂·模型层 forward + 编译』块里 DecoderLayer 的注意力半层投影链
(L2 拍片③/⑧ 的机制版下钻——③=MLA 第三代装配(站 6)、⑧=Attention 一拍(站 12);
ch25 讲核内两种展开, 本章讲装配侧数据流)。

claim: 第三代 MLA 的模型侧装配是一条低秩进低秩出的数据流——fused_wqa_wkv
(disable_tp=True 复制不切)一枪出 q_lora_rank+head_dim → split 后双 rmsnorm →
wq_b 上投进 eager break(平台核 forward_mqa 写入 padded_heads 预分配缓冲) →
输出切片真头数 → 逆 RoPE+FP8 量化 → fp8_einsum('bhr,hdr->bhd') 按组低秩 wo_a →
wo_b 回 hidden。

锚点 = vllm/models/deepseek_v4/attention.py:L179-L348/L461-L540/L655-L674 ·
nvidia/flashmla.py:L62-L67 · nvidia/ops/o_proj.py:L13-L73 ·
v1/kv_cache_interface.py:L409-L413。坐标由常量/循环计算; 文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 905
MX = 56
BXR = 1608

# ---------------- 标题区 ----------------
lc.text(MX, 34, '注意力插座的旗舰版接线：两头低秩、中间平台核', 16, lc.C_TXT, 'start', True,
        maxw=1000, tag='title')
lc.text(MX, 58, '输入瓶颈复制不切 TP，输出还要逆 RoPE + FP8 einsum 走 wo_a/wo_b 两段低秩——「平台核要求什么、装配层就长什么样」的实证',
        10.5, lc.C_MUTE, 'start', maxw=1180, tag='subtitle')
_ch = '放大自 L0『模型层 forward + 编译』块 · L2 拍片③/⑧'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')


def box(x, y, w, h, title, lines, file='', stroke=lc.C_GPU_S, fill='#ffffff',
        dash=False, tfs=10.5, lfs=8.5, tc=lc.C_TXT):
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=dash)
    lc.text(x + 12, y + 20, title, tfs, tc, 'start', True, maxw=w - 24, tag='t:' + title)
    yy = y + 40
    for s in lines:
        lc.text(x + 12, yy, s, lfs, '#334155', 'start', maxw=w - 22, tag='l:' + s[:12])
        yy += 16.5
    if file:
        lc.text(x + 12, y + h - 10, file, 8, lc.C_FAINT, 'start', maxw=w - 20, tag='f:' + title)


# ---------------- 输入侧(上排): 低秩进 ----------------
RY, RH = 150, 150
lc.text(MX, 128, '① 输入侧：一枪融合的低秩瓶颈（复制不切 TP）', 11, lc.C_TXT, 'start', True,
        maxw=500, tag='rowA:t')

lc.rect(MX, RY, 96, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(MX + 48, RY + 42, 'hidden', 10.5, lc.C_TXT, 'middle', True, tag='hid')
lc.text(MX + 48, RY + 62, '(T, H)', 9, lc.C_MUTE, 'middle', tag='hid:s')

box(176, RY, 214, RH, 'fused_wqa_wkv', [
    'MergedColumnParallelLinear(', '[q_lora_rank, head_dim],', 'disable_tp=True',
    '——『fused ReplicatedLinear』', '（注释原文）低秩瓶颈每卡全量复制'],
    'attention.py:L229-L239')
box(410, RY, 150, RH, 'split', ['qr | kv 潜向量', 'q_lora_rank | head_dim', '一刀两段，各自归一'],
    '', fill='#ffffff')
box(580, RY, 200, RH, 'fused_q_kv_rmsnorm', ['q_norm + kv_norm 一核双归一', '（q_lora_rank 与 head_dim 段）'],
    'attention.py:L234-L244')
box(800, RY, 160, RH, 'wq_b', ['ColumnParallelLinear', '列切上投进 eager break', 'q (n_local_heads, head_dim)'],
    'attention.py:L235-L242')

AY = RY + 48
for a, b in [(MX + 96, 176), (176 + 214, 410), (410 + 150, 580), (580 + 200, 800), (800 + 160, 1010)]:
    lc.seg(a, AY, b, AY, lc.C_GPU_S, 1.8, 'std')

# ---------------- 中部: eager break 平台核区 ----------------
EB_X, EB_Y, EB_W, EB_H = 1010, 120, 330, 560
lc.rect(EB_X, EB_Y, EB_W, EB_H, '#f8fafc', lc.C_GPU_S, rx=10, sw=1.8, dash=True)
lc.text(EB_X + 14, EB_Y + 22, 'eager break：捕获图在此断开', 10,
        lc.C_GPU_S, 'start', True, maxw=EB_W - 28, tag='eb:t')
lc.text(EB_X + EB_W - 14, EB_Y + 22, 'attention.py:L461-L470', 8, lc.C_FAINT, 'end',
        maxw=140, tag='eb:file')

box(EB_X + 20, EB_Y + 46, 290, 92, 'kv_insert（kv 潜向量写 KV 池）', [
    '主 KV 账 584B/token', '= 448B NoPE + 128B RoPE + 8B fp8 scale', 'fp8_ds_mla uint8 · alignment 576'],
    '', stroke=lc.C_KV_S, fill=lc.C_KV_F, tc='#155e75')
box(EB_X + 20, EB_Y + 158, 290, 130, '平台核 forward_mqa', [
    '平台子类实现（FlashMLA/FlashInfer）', '写入 padded_heads 预分配缓冲', '（o_padded，本拍 forward 开头分配）',
    '核内两种展开回指 ch25'], 'attention.py:L350-L396')
box(EB_X + 20, EB_Y + 310, 138, 118, 'indexer 支', ['（打分+选块）', '淡化：', '归 ch26'], '',
    stroke=lc.C_FAINT, dash=True, tc=lc.C_MUTE)
box(EB_X + 172, EB_Y + 310, 138, 118, 'compressor 支', ['（压缩 KV）', '淡化：', '归 ch26'], '',
    stroke=lc.C_FAINT, dash=True, tc=lc.C_MUTE)
# 注区五行的 y 基线整体上移（盲审修复 2026-09-14）：原 f1 行(baseline 669)的右半句
# 被右下注框（白底，x>=1080、y>=660，绘制序在后）盖住只剩「被 @eager_b」；
# 上移后末行字形带底 ~644，距注框顶边净空 ~16px。
lc.text(EB_X + 20, EB_Y + 446, '两支与主路并行（多流时间线见本章 overlap 图）', 8, lc.C_MUTE, 'start',
        maxw=EB_W - 36, tag='eb:aux')
lc.text(EB_X + 20, EB_Y + 468, 'wq_b + qnorm + RoPE + kv_insert 留默认流，', 8.5, '#334155', 'start',
        maxw=EB_W - 36, tag='eb:d0')
lc.text(EB_X + 20, EB_Y + 483, 'indexer/compressor 走 aux 流（attention.py:L461-L540）', 8.5,
        '#334155', 'start', maxw=EB_W - 36, tag='eb:d1')
lc.text(EB_X + 20, EB_Y + 506, 'attention_impl 整段 = metadata 相关段，', 8, lc.C_MUTE, 'start',
        maxw=EB_W - 36, tag='eb:f0')
lc.text(EB_X + 20, EB_Y + 521, '被 @eager_break_during_capture 包住（L461-L470）', 8, lc.C_MUTE,
        'start', maxw=EB_W - 36, tag='eb:f1')

# o 芯片(右侧)
box(1370, RY, 238, RH, 'o（padded_heads 输出）', [
    "Q 头数垫到 64/128：'FP8 decode", "kernel only supports h_q = 64 or 128'", 'attn_sink 按 padded_heads 填 -inf',
    '（装载只填前 n_local_heads 槽）'], 'flashmla.py:L62-L67 · attention.py:L222-L228', lfs=8)
lc.seg(EB_X + EB_W, AY, 1370, AY, lc.C_GPU_S, 1.8, 'std')

# ---------------- 输出侧(下排): 低秩出 ----------------
lc.text(MX, 690, '② 输出侧：逆 RoPE + FP8 einsum 的两段低秩（先按组压到 o_lora_rank 再回 hidden）', 11,
        lc.C_TXT, 'start', True, maxw=760, tag='rowB:t')
BY, BH = 706, 148
box(MX, BY, 250, BH, 'fused_inv_rope_fp8_quant', [
    '逆 RoPE（还原 o 上被旋转的段）', '+ FP8 量化一步融合', 'o_fp8, o_scale'], 'ops/o_proj.py:L48-L57')
box(326, BY, 300, BH, "fp8_einsum('bhr,hdr->bhd')", [
    'wo_a：n_heads·head_dim/n_groups', '   → n_groups·o_lora_rank（按组低秩）', 'bmm 视图（n_local_groups 批）',
    'head_dim=512 · n_groups=o_groups'], 'ops/o_proj.py:L63-L72')
box(646, BY, 220, BH, 'wo_b', ['RowParallelLinear', 'n_groups·o_lora_rank → hidden', '回全宽'],
    'attention.py:L255-L262')
lc.rect(886, BY, 140, BH, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(956, BY + 60, 'hidden', 10.5, lc.C_TXT, 'middle', True, tag='hout')
lc.text(956, BY + 80, '(T, H) 出', 9, lc.C_MUTE, 'middle', tag='hout:s')
BYA = BY + 74
for a, b in [(MX + 250, 326), (326 + 300, 646), (646 + 220, 886)]:
    lc.seg(a, BYA, b, BYA, lc.C_GPU_S, 1.8, 'std')

# o 芯片 → 输出侧 回绕总线（右缘下行、左行、下插）
lc.parrow([(1489, RY + RH), (1489, 660), (MX + 125, 660), (MX + 125, BY)],
          lc.C_GPU_S, 2.0, 'std')
# 盲审修复 2026-09-14（第三轮）：单行标签（左端 ≈x1321）被 eager break 绿色虚线右缘
# （x≈1340、y120-680 纵贯）穿过「片/真」一带字形呈划掉观感——单行右移不可行
# （右端 1481 已贴回绕总线竖线 x=1489），拆两行右锚 x=1481：行首左端 ≈x1363，
# 距虚线右缘 ≥22px；两行字形带 y627-653 与 y=660 横段/注框顶均留 ≥7px。
lc.text(1489 - 8, 636, '切片真头数 n_local_heads', 9, lc.C_GPU_S, 'end', tag='wrap:lab1')
lc.text(1489 - 8, 650, '后进输出链', 9, lc.C_GPU_S, 'end', tag='wrap:lab2')

# ---------------- 右下注: 两头低秩总结 + scale 布局分派 ----------------
lc.rect(1080, 660, 528, 194, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(1094, 682, '为什么叫「两头低秩、中间平台核」', 10.5, lc.C_TXT, 'start', True, maxw=480, tag='sum:t')
for i, s in enumerate([
        '· 输入侧：fused_wqa_wkv 一枪出 q_lora_rank+head_dim 潜向量，',
        '   瓶颈瘦到切不动——disable_tp=True 每卡全量复制',
        '· 输出侧：比 V3.2 多收一段低秩税（wo_a 按组压到 o_lora_rank，',
        '   wo_b 再回 hidden），换来 attn_sink 等平台核自由度',
        '· SM90/SM100 scale 布局由 compute_fp8_einsum_recipe 分派：',
        '   SM90 FP32 块尺度 [g,r/128,d/128] / SM100 INT32 打包 [g,r,…]']):
    lc.text(1094, 704 + i * 18, s, 8.5, '#334155', 'start', maxw=500, tag='sum:l%d' % i)
# 盲审修复 2026-09-14：连接符 ↔（U+2194）在本机字体栈不渲染成豆腐，换成文字「对应」。
lc.text(1094, 838, 'ops/o_proj.py:L13-L25 · 与 ch25 V3.2 外层低秩链同构（fused_qkv_a_proj 对应 fused_wqa_wkv）',
        8, lc.C_FAINT, 'start', maxw=500, tag='sum:f')

# ---------------- 页脚 ----------------
lc.text(MX, 876, '图例：绿 = 模型层投影链（GPU 执行臂角色）· 青 = KV 池 · 灰虚 = 平台核辅助支（归 ch26）· 绿虚线大框 = eager break 边界（cudagraph 捕获在此断开）',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 894, '锚点 = vllm/models/deepseek_v4/attention.py:L179-L348/L461-L540/L655-L674 · nvidia/flashmla.py:L62-L67 · nvidia/ops/o_proj.py:L13-L73 · v1/kv_cache_interface.py:L409-L413',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch29-fig-mla-gen3-projections.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
