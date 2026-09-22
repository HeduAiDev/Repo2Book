#!/usr/bin/env python3
"""ch36 机制图 · 三层档位交集树（figure_spec fig_m1_dtype_tree，模板 flow）

放大自 L2 章图拍片①③④（checkpoint 量化档位 / 硬件×backend / KV 档解析）——
L0 启动视角显存账的『谁定哪档』全景树（架构归属回指 L2，不另立第二种架构画法）。

claim：一台卡上 DSV4-Flash 的最终字节形状 = 三层档位的交集：checkpoint 层出生即定、
硬件层是命令、操作员层只剩自由度。

数字全部取自 figure_spec.numbers（quant_config.py:L181-L194 / sparse_mla.py:L92-L93 /
flashinfer_sparse.py:L139-L141 / model.py:L314-L317 / indexer 68|132B / cache.py:L68）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1460, 916
MX, BXR = 56, 1404
DEFS = lc.DEFS

C_LOCK_S, C_LOCK_F = lc.C_ENG_S, lc.C_ENG_F          # ① checkpoint 层 = 出厂即定
C_GATE_S = lc.C_ABORT                                # ② 硬件层 = 命令/门禁
C_FREE_S, C_FREE_F = lc.C_GPU_S, lc.C_GPU_F          # ③ 操作员层 = 自由度
C_OUT_S, C_OUT_F = lc.C_KV_S, lc.C_KV_F              # 交集结果 = 字节形状

# ---------------- 标题区 ----------------
lc.text(MX, 34, '这台卡上 DSV4-Flash 的最终字节形状 = 三层档位的交集（根到叶唯一一条活路）', 16.5,
        lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '① checkpoint 层出生即定（选模型即选档）· ② 硬件层不是建议（不合规不让进）· ③ 操作员层只剩自由度'
               ' —— 三张纸的交集才是这台机器上跑起来的那个模型',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L2 拍片①③④ · L0 启动视角'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 三层档位带 ----------------
BAND_X0, BAND_W = MX, BXR - MX
CARD_GAP = 14
CARD_W = (BAND_W - 32 - 2 * CARD_GAP) / 3            # 429.3


def band(y, h, title, right_note, color, fill):
    lc.rect(BAND_X0, y, BAND_W, h, fill, color, rx=10, sw=1.8)
    lc.text(BAND_X0 + 16, y + 24, title, 12.5, color, 'start', True, maxw=880, tag='bt:' + title[:10])
    lc.text(BXR - 16, y + 24, right_note, 8.8, lc.C_FAINT, 'end', maxw=490, tag='bn:' + title[:8])


def cards(y, h, color, items):
    """items = [[line, ...], ...] 3 张卡。"""
    cy = y + 38
    ch = h - 52
    for i, lines in enumerate(items):
        cx = BAND_X0 + 16 + i * (CARD_W + CARD_GAP)
        lc.rect(cx, cy, CARD_W, ch, '#ffffff', color, rx=7, sw=1.4)
        for j, ln in enumerate(lines):
            lc.text(cx + 13, cy + 22 + j * 17.5, ln, 9.6, lc.C_TXT if j == 0 else '#334155',
                    'start', bold=(j == 0), maxw=CARD_W - 26, tag=f'c{i}.{j}')


B1Y, B1H = 96, 174
band(B1Y, B1H, '① checkpoint 层 —— 出生即定，改不了一个字节',
     'vllm/models/deepseek_v4/quant_config.py:L181-L194（get_quant_method 分档）', C_LOCK_S, C_LOCK_F)
cards(B1Y, B1H, C_LOCK_S, [
    ['线性 / 注意力层', '恒 FP8 块量化', '无档可选——quant_config 读表即定'],
    ['专家层 expert_dtype', 'fp4 → Mxfp4MoEMethod（MXFP4 · Flash）', 'fp8 → Fp8MoEMethod（块专家 · Flash-Base）'],
    ['『再量化』不在档位树里', 'ch27 已读的量化谱系在 DSV4 上不适用', '选模型即选档，出生定格'],
])

B2Y, B2H = 304, 194
band(B2Y, B2H, '② 硬件层 —— 不是建议，是命令（门禁）',
     'vllm/models/deepseek_v4/sparse_mla.py:L92-L93 · nvidia/flashinfer_sparse.py:L139-L141', C_GATE_S, '#fef2f2')
cards(B2Y, B2H, C_GATE_S, [
    ['SM90 / SM100（H100 · H20-96G · B200）', '→ FlashMLA（major∈[9,10]）', 'KV 必须 fp8 系（写回 fp8_ds_mla 布局）'],
    ['SM120（RTX PRO 6000 · 96GB 档）', '→ FlashInfer SM120', 'KV 只放行 fp8 / fp8_e4m3 / fp8_ds_mla', 'bf16 / auto 直接拒（不合规不让进）'],
    ['MegaMoE 仅 SM100', '其它架构 NotImplementedError 门', 'nvidia/model.py:L314-L317'],
])

B3Y, B3H = 534, 152
band(B3Y, B3H, '③ 操作员层 —— 真正能拨的旋钮',
     'vllm/config/cache.py:L68（util 默认 0.92）', C_FREE_S, C_FREE_F)
cards(B3Y, B3H, C_FREE_S, [
    ['--attention_config.use_fp4_indexer_cache', 'indexer 档：68B（fp4）| 132B（fp8）/ 压缩槽', '官方配方拨 fp4'],
    ['TP / EP 并行数', '决定权重怎么切、池里剩多少', '（整本权重账单卡装不下）'],
    ['gpu_memory_utilization（默认 0.92）', 'num_gpu_blocks_override / max_model_len', 'kv_cache_memory_bytes 手动档'],
])

# ---------------- 带间漏斗箭头（①→②→③→交集） ----------------
RY = B3Y + B3H + 34          # 交集结果框顶（漏斗箭头终点，先定再画）
RH = 96
FUN_X = W / 2
for y0, y1 in ((B1Y + B1H, B2Y), (B2Y + B2H, B3Y), (B3Y + B3H, RY)):
    lc.seg(FUN_X, y0 + 2, FUN_X, y1 - 2, lc.C_MUTE, 2.2, 'std')

# ---------------- 交集结果 ----------------
lc.rect(BAND_X0, RY, BAND_W, RH, C_OUT_F, C_OUT_S, rx=10, sw=2.0)
lc.text(BAND_X0 + 16, RY + 28, '三层交集 → 这台机器上跑起来的那个 DSV4-Flash', 13, C_OUT_S, 'start', True,
        maxw=900, tag='out:t')
lc.text(BXR - 16, RY + 28, '档位树从根到叶只有一条活路', 10, lc.C_MUTE, 'end', maxw=330, tag='out:r')
lc.text(BAND_X0 + 16, RY + 54, '这台卡上：KV 必是 fp8 系（硬件命令）· 专家必是 MXFP4（checkpoint 印刷）· indexer 档与池旋钮由操作员收尾 —— 每本账的字节形状唯一确定',
        10, '#334155', 'start', maxw=BAND_W - 36, tag='out:l1')

# ---------------- 图例 ----------------
LY = RY + RH + 22
sw = [(C_LOCK_S, C_LOCK_F, '出厂即定（不可拨）'), (C_GATE_S, '#fef2f2', '硬件命令（门禁）'),
      (C_FREE_S, C_FREE_F, '操作员自由度'), (C_OUT_S, C_OUT_F, '交集结果（字节形状）')]
lx0 = MX
for s, f, name in sw:
    lc.rect(lx0, LY - 9, 16, 11, f, s, rx=3, sw=1.6)
    lc.text(lx0 + 21, LY + 1, name, 9.5, lc.C_TXT, 'start', maxw=170, tag='lg:' + name[:6])
    lx0 += 21 + lc.tw(name, 9.5) + 26

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 58, 'vllm/models/deepseek_v4/quant_config.py:L181-L194（expert_dtype 分档）· vllm/models/deepseek_v4/sparse_mla.py:L92-L93（major∈[9,10]）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 43, 'vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:L139-L141（SM120 放行表）· vllm/models/deepseek_v4/nvidia/model.py:L314-L317（MegaMoE 仅 SM100）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 28, 'vllm/config/cache.py:L68（gpu_memory_utilization 默认 0.92）· indexer 档 68|132B/槽 ＝ 本章账本算术脚本实算（pin spec 类）· 『fp8 → 写回 fp8_ds_mla』见 attention.py:L90-L120',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')
lc.text(MX, H - 13, '行号基线 vLLM v0.27.1（6e448d0ea）', 8.5, lc.C_FAINT, 'start', maxw=400, tag='foot4')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m1_dtype_tree.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
