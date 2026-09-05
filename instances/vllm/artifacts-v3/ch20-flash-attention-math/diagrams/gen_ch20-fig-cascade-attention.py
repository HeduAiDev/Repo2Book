#!/usr/bin/env python3
"""ch20 机制图 ⑦ · cascade attention 落地(figure_spec ch20-fig-cascade-attention,模板 tiling)

放大自 L0 中列『GPU 执行臂』(绿色列)『模型层 forward + 编译』块内 attention 的稀有路径分支
(cascade 分流在 FlashAttentionImpl.forward L1069-L1095)——与 A 列『BlockPool + 前缀缓存』
(青,C_KV_S)隔列呼应:前缀共享的账本来源在 A 列。primer 推导链第 ⑧ 环:m07 的合并数学
在 vLLM 的真实调用现场。架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:cascade 把『共享前缀+私有后缀』拆两段各带 LSE 再 merge_attn_states 合并——前缀段
causal=False + block_table[:1] 一次算完全批复用,后缀段 causal=True + 页表切片各算各的;
拆分是否值得由启发式定(共享前缀 < 256 token 直接不拆),拆分正确性由 ⊕ 结合律保证。

数字全部取自 figure_spec.numbers(门槛 256/≥8 条;两段调用形状与 merge 签名——pin 源码
逐字;扫描账 13 vs 9、省 4、比例 0.3077、一般式 P·(R−1)——host 实算)。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 60
BXR = 1440
EXTRA_DEFS = ('<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
              '<marker id="cyn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0891b2"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'cascade attention:共享前缀只算一遍,私有后缀各算各的,LSE 缝回一个精确结果',
        16.5, lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, '『值不值得拆』是工程启发式(阈值 + 粗性能模型),『拆了精确』是 ⊕ 结合律——两条线分开看(vllm/v1/attention/backends/flash_attn.py:L1521-L1690)',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '推导链 ⑧ · 放大自 L0 GPU 执行臂 attention 稀有分支 · 与 A 列『BlockPool+前缀缓存』呼应'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左上:共享前缀色带 ----------------
lc.text(MX, 96, '共享前缀 4 token(全批同一段,如长 system prompt)', 10.5, lc.C_TXT, 'start',
        True, maxw=400, tag='pb:t')
PB_Y, PCW, PCH, PGAP = 110, 64, 44, 6
for i in range(4):
    lc.rect(MX + i * (PCW + PGAP), PB_Y, PCW, PCH, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.5)
    lc.text(MX + i * (PCW + PGAP) + PCW / 2, PB_Y + 27, 'K/V', 9, lc.C_KV_S, 'middle', True,
            maxw=PCW - 8, tag=f'pb:c{i}')
lc.text(MX + 4 * (PCW + PGAP) + 6, PB_Y + 20, 'block_table[:1]', 8.5, lc.C_KV_S, 'start', True,
        maxw=150, tag='pb:bt')
lc.text(MX + 4 * (PCW + PGAP) + 6, PB_Y + 36, 'causal=False', 8.5, lc.C_KV_S, 'start',
        maxw=150, tag='pb:cs')

# 中上:前缀段一次调用
PC_X, PC_W = 500, 500
lc.rect(PC_X, 90, PC_W, 84, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(PC_X + 14, 110, '前缀段:一次调用算完全批(flash_attn_varlen_func · return_softmax_lse=True)',
        9.5, lc.C_GPU_S, 'start', True, maxw=PC_W - 28, tag='pc:t')
lc.text(PC_X + 14, 130, '产出 (O_前缀, lse_前缀)——全批共用一份,不随请求数重复算', 9.5,
        '#334155', 'start', maxw=PC_W - 28, tag='pc:l1')
lc.text(PC_X + 14, 152, '为什么能缝回去:lse = 该段 softmax 分母的对数——上一环的六步合并', 8.7,
        lc.C_MUTE, 'start', maxw=PC_W - 28, tag='pc:l2')
lc.seg(MX + 4 * (PCW + PGAP) - PGAP, PB_Y + PCH / 2, PC_X - 2, PB_Y + PCH / 2, lc.C_KV_S, 2.0,
       'cyn')

# 左:两条请求泳道
LANE_X, LANE_W = 60, 400
LANES = [
    (230, '请求 A:私有后缀 3 token + query 2 个(最后 2 个 token)', 3,
     'block_table[:, num_common_kv_blocks:] · causal=True · query_offset=1'),
    (370, '请求 B:私有后缀 2 token + query 2 个', 2,
     'block_table[:, num_common_kv_blocks:] · causal=True · query_offset=0'),
]
for ly, title, nsuf, ann in LANES:
    lc.rect(LANE_X, ly, LANE_W, 110, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
    lc.text(LANE_X + 14, ly + 20, title, 9.5, lc.C_TXT, 'start', True, maxw=LANE_W - 28,
            tag='ln:t' + str(ly))
    cy = ly + 34
    cx = LANE_X + 14
    for i in range(nsuf):
        lc.rect(cx, cy, 40, 30, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.2)
        lc.text(cx + 20, cy + 19, 'K/V', 8, lc.C_KV_S, 'middle', True, maxw=34, tag=f'ln:s{ly}{i}')
        cx += 46
    cx += 14
    for i in range(2):
        lc.rect(cx, cy, 40, 30, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.2)
        lc.text(cx + 20, cy + 19, 'q', 8.5, lc.C_GPU_S, 'middle', True, maxw=34,
                tag=f'ln:q{ly}{i}')
        cx += 46
    lc.text(LANE_X + 14, cy + 46, '后缀 K/V(页表切片)', 8, lc.C_KV_S, 'start', maxw=170,
            tag='ln:kl' + str(ly))
    lc.text(LANE_X + 14 + nsuf * 46 + 14, cy + 46, 'query', 8, lc.C_GPU_S, 'start', maxw=90,
            tag='ln:ql' + str(ly))
    lc.text(LANE_X + 14, ly + 98, ann, 8, lc.C_MUTE, 'start', maxw=LANE_W - 28,
            tag='ln:a' + str(ly))

# 中:merge_attn_states
MG_X, MG_W = 500, 500
lc.rect(MG_X, 230, MG_W, 250, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(MG_X + 14, 252, 'merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)',
        8.8, lc.C_GPU_S, 'start', True, maxw=MG_W - 28, tag='mg:sig')
lc.text(MG_X + 14, 274, '逐请求行执行:前缀摘要(共用一份)⊕ 本请求后缀摘要', 9.5, lc.C_TXT,
        'start', True, maxw=MG_W - 28, tag='mg:t')
lc.text(MG_X + 14, 298, '六步:max_lse 稳定化 → e^(lse−max) → out_se → 权重=占比', 9,
        '#334155', 'start', maxw=MG_W - 28, tag='mg:s1')
lc.text(MG_X + 14, 316, '→ 加权 O(先算 scale 再乘 output)→ 合并 lse 可继续并段', 9,
        '#334155', 'start', maxw=MG_W - 28, tag='mg:s2')
lc.text(MG_X + 14, 344, '正确性:⊕ 满足结合律——前缀段与后缀段任意先后算、再合并,', 9,
        lc.C_GPU_S, 'start', True, maxw=MG_W - 28, tag='mg:c1')
lc.text(MG_X + 14, 362, '仍与一口气算逐位相等(实跑 4 行 O 与 lse 差 0.0)', 9, lc.C_GPU_S,
        'start', True, maxw=MG_W - 28, tag='mg:c2')
lc.text(MG_X + 14, 392, '反超例(请求 A 行 1):前缀 lse 2.0064 < 后缀 lse 2.4076', 8.5,
        lc.C_MUTE, 'start', maxw=MG_W - 28, tag='mg:r1')
lc.text(MG_X + 14, 408, '→ 后缀权重 0.599 更大——权重跟着归一化质量走,不认『前缀』名分', 8.5,
        lc.C_MUTE, 'start', maxw=MG_W - 28, tag='mg:r2')
lc.text(MG_X + 14, 436, 'split-KV 合并也是这同一个函数——cascade 只是它的一次应用', 8.5,
        lc.C_MUTE, 'start', maxw=MG_W - 28, tag='mg:r3')
lc.text(MG_X + 14, 462, 'vllm/v1/attention/ops/merge_attn_states.py', 8, lc.C_FAINT, 'start',
        maxw=MG_W - 28, tag='mg:f')
# 前缀 → merge(复用两箭头)
lc.seg(650, 174, 650, 228, lc.C_GPU_S, 2.0, 'grn')
lc.seg(850, 174, 850, 228, lc.C_GPU_S, 2.0, 'grn')
lc.text(750, 205, '(O_前缀, lse_前缀) 复用给全批每条请求', 8.5, lc.C_GPU_S, 'middle', True,
        maxw=240, tag='mg:reuse')
# 泳道 → merge
lc.seg(LANE_X + LANE_W + 1, 285, MG_X - 2, 285, lc.C_GPU_S, 2.0, 'grn')
lc.text(MG_X - 96, 278, '(O_A后缀, lse_A)', 8, lc.C_GPU_S, 'end', maxw=140, tag='mg:ina')
lc.seg(LANE_X + LANE_W + 1, 425, MG_X - 2, 425, lc.C_GPU_S, 2.0, 'grn')
lc.text(MG_X - 96, 418, '(O_B后缀, lse_B)', 8, lc.C_GPU_S, 'end', maxw=140, tag='mg:inb')

# 右:出口
OX, OW = 1060, 380
lc.rect(OX, 300, OW, 110, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(OX + 14, 324, 'output(就地写回)', 10.5, lc.C_GPU_S, 'start', True, maxw=OW - 28,
        tag='ox:t')
lc.text(OX + 14, 346, '= 对全部 KV(前缀+后缀)一次性', 9.5, lc.C_TXT, 'start', maxw=OW - 28,
        tag='ox:l1')
lc.text(OX + 14, 364, '  softmax(QK^T)V,逐位相等(差 0.0)', 9.5, lc.C_TXT, 'start',
        maxw=OW - 28, tag='ox:l2')
lc.text(OX + 14, 390, '长 system prompt 的重复扫描被消掉,数学一点没变', 8.7, lc.C_MUTE,
        'start', maxw=OW - 28, tag='ox:l3')
lc.seg(MG_X + MG_W + 1, 355, OX - 2, 355, lc.C_GPU_S, 2.2, 'grn')

# ---------------- 左下:决策门槛(工程线,橙框) ----------------
DY = 520
lc.rect(MX, DY, 900, 112, '#ffffff', lc.C_ENG_S, rx=8, sw=1.5)
lc.text(MX + 14, DY + 20, '值不值得拆?先过门槛(与上面『怎么拆、拆了精确』是两件事)', 10,
        lc.C_ENG_S, 'start', True, maxw=872, tag='dy:t')
lc.text(MX + 14, DY + 40, 'common_prefix_len < 256 → 直接不拆(NOTE(woosuk):This is the common case——尽早返回)', 8.7,
        '#334155', 'start', maxw=872, tag='dy:l1')
lc.text(MX + 14, DY + 58, '≥ 256 token 且 ≥ 8 条请求才值得拆;另有与 FlashDecoding 的 CTA 波数粗模型比较', 8.7,
        '#334155', 'start', maxw=872, tag='dy:l2')
lc.text(MX + 14, DY + 76, 'use_cascade 开关:common_prefix_len > 0(由 _compute_cascade_attn_prefix_lens 先行算出传 build())', 8.7,
        '#334155', 'start', maxw=872, tag='dy:l3')
lc.text(MX + 14, DY + 94, 'cascade 分流在 FlashAttentionImpl.forward 的稀有路径分支(L1069-L1095)', 8.7,
        '#334155', 'start', maxw=872, tag='dy:l4')

# ---------------- 右下:扫描账 ----------------
AY = DY
lc.rect(980, AY, 460, 112, '#ffffff', lc.C_KV_S, rx=8, sw=1.4)
lc.text(994, AY + 20, '示例账:2 请求 4+3 / 4+2 键元素', 10, lc.C_KV_S, 'start', True,
        maxw=432, tag='ay:t')
lc.text(994, AY + 40, '全扫 13 个键元素(7+6)vs cascade 9 个(4+3+2)', 8.7, '#334155',
        'start', True, maxw=432, tag='ay:l1')
lc.text(994, AY + 58, '——省 4,比例 0.3077', 8.7, lc.C_KV_S, 'start', maxw=432, tag='ay:l2')
lc.text(994, AY + 80, '一般式 R 条请求共享前缀 P、后缀 S_i:全扫 Σ(P+S_i),', 8.3, lc.C_MUTE,
        'start', maxw=432, tag='ay:l3')
lc.text(994, AY + 96, 'cascade 只扫 P+ΣS_i——省 P·(R−1)', 8.3, lc.C_MUTE, 'start',
        maxw=432, tag='ay:l4')

# ---------------- 页脚:图例 + 出处 ----------------
LY = DY + 134
lc.text(MX, LY, '图例:青 = KV(共享前缀 / 私有后缀,页表切片) · 绿 = query 与 kernel 计算 · 橙框 = 工程决策线(与数学正确性分开讲)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, LY + 18, '出处 vllm/v1/attention/backends/flash_attn.py:L1531-L1551(门槛)· L1638-L1690(两段调用 + merge,签名逐字)· L1069-L1095(分流)· L543(use_cascade)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, LY + 36, '合并方法自引 arXiv:2501.01005 §2.2 · 数值取自 NumPy 参考实现实跑(host,float64)· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-cascade-attention.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
