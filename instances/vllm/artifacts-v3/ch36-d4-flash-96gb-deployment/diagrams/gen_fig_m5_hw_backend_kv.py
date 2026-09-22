#!/usr/bin/env python3
"""ch36 机制图 · 硬件×backend×KV 档三元耦合（figure_spec fig_m5_hw_backend_kv，模板 state-machine）

放大自 L2 章图拍片③④（硬件×backend / KV 档解析）——L0 启动视角『这台卡』分岔的地方。

claim：无显式 backend 旋钮时 _select_dsv4_attn_cls 按 compute capability major 分岔——
major 12（RTX PRO 6000）→FlashInfer SM120（KV 只放行 fp8 系），其余 CUDA（major 9/10）
→FlashMLA（fp8_ds_mla 布局，assert KV 必须 fp8 系）；显式 FlashInfer DSV4 时 major 10
走普通行（明确拒 fp8_ds_mla）、major 12 仍走 SM120。

数字全部取自 figure_spec.numbers（model.py:L790-L802 / sparse_mla.py:L92-L93 /
flashinfer_sparse.py:L130-L150 · L576-L582 · L177/L540 / attention.py:L90-L120 写回）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1460, 826
MX, BXR = 56, 1404
DEFS = lc.DEFS

C_FMLA_S, C_FMLA_F = lc.C_API_S, lc.C_API_F    # FlashMLA 路径 = 蓝
C_GATE_S = lc.C_ABORT                           # SM120 门禁 = 红
C_PLAIN_S, C_PLAIN_F = lc.C_GPU_S, lc.C_GPU_F   # FlashInfer 普通行 = 绿

# ---------------- 标题区 ----------------
lc.text(MX, 34, '同一句 --kv-cache-dtype fp8，三种硬件三种命运——机器替你做完了选择题', 16.5,
        lc.C_TXT, 'start', True, maxw=1090, tag='title')
lc.text(MX, 58, '无显式 backend 旋钮时按 compute capability major 分岔（显式旋钮优先）· 显式 FlashInfer DSV4 时 SM10x 走普通行、SM12x 仍走 SM120',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L2 拍片③④ · L0 启动视角'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶节点 ----------------
TN_W, TN_H = 620, 50
TN_X, TN_Y = (W - TN_W) / 2, 92
lc.rect(TN_X, TN_Y, TN_W, TN_H, '#ffffff', lc.C_TXT, rx=9, sw=1.8)
lc.text(W / 2, TN_Y + 21, '无显式 backend 旋钮（默认路径）', 12, lc.C_TXT, 'middle', True, maxw=TN_W - 20, tag='tn:t')
lc.text(W / 2, TN_Y + 39, '看这台卡的 compute capability（vllm/models/deepseek_v4/nvidia/model.py:L790-L802）', 8.8,
        lc.C_MUTE, 'middle', maxw=TN_W - 16, tag='tn:s')

# ---------------- 两列分岔 ----------------
LX, LW = 76, 616           # 左列（FlashMLA）
RX, RW = 768, 616          # 右列（FlashInfer SM120）
LCX, RCX = LX + LW / 2, RX + RW / 2

# 分岔箭头：顶节点底边 → 条件芯片顶边
CHIP_Y, CHIP_H = 176, 34
lc.seg(TN_X + 120, TN_Y + TN_H, LCX, CHIP_Y - 2, C_FMLA_S, 2.0, 'std')
lc.seg(TN_X + TN_W - 120, TN_Y + TN_H, RCX, CHIP_Y - 2, C_GATE_S, 2.0, 'std')

lchip = 'major ∈ [9, 10]（H100 · H20-96G · B200）'
lw_ = lc.tw(lchip, 10.5, True) + 26
lc.rect(LCX - lw_ / 2, CHIP_Y, lw_, CHIP_H, C_FMLA_F, C_FMLA_S, rx=16, sw=1.6)
lc.text(LCX, CHIP_Y + 22, lchip, 10.5, C_FMLA_S, 'middle', True, maxw=lw_ - 10, tag='chipL')

rchip = 'major == 12（RTX PRO 6000 Blackwell · 96GB 档）'
rw_ = lc.tw(rchip, 10.5, True) + 26
lc.rect(RCX - rw_ / 2, CHIP_Y, rw_, CHIP_H, '#fef2f2', C_GATE_S, rx=16, sw=1.6)
lc.text(RCX, CHIP_Y + 22, rchip, 10.5, C_GATE_S, 'middle', True, maxw=rw_ - 10, tag='chipR')

# 类框
CLS_Y, CLS_H = 232, 60
lc.seg(LCX, CHIP_Y + CHIP_H, LCX, CLS_Y - 2, C_FMLA_S, 2.0, 'std')
lc.seg(RCX, CHIP_Y + CHIP_H, RCX, CLS_Y - 2, C_GATE_S, 2.0, 'std')


def cls_box(x, w, y, h, name, sub, color, fill):
    lc.rect(x, y, w, h, fill, color, rx=8, sw=1.6)
    lc.text(x + w / 2, y + 24, name, 11.5, color, 'middle', True, maxw=w - 20, tag='cls:' + name[:10])
    lc.text(x + w / 2, y + 44, sub, 8.8, lc.C_FAINT, 'middle', maxw=w - 16, tag='cls:s' + name[:8])


cls_box(LX, LW, CLS_Y, CLS_H, 'DeepseekV4FlashMLAAttention',
        'FlashMLA sparse · vllm/models/deepseek_v4/sparse_mla.py:L92-L93', C_FMLA_S, '#ffffff')
cls_box(RX, RW, CLS_Y, CLS_H, 'DeepseekV4FlashInferSM120Attention',
        'vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:L536-L540（use_fp8_ds_mla_layout=True）', C_GATE_S, '#ffffff')

# KV 档框
KV_Y, KV_H = 316, 122
lc.seg(LCX, CLS_Y + CLS_H, LCX, KV_Y - 2, C_FMLA_S, 2.0, 'std')
lc.seg(RCX, CLS_Y + CLS_H, RCX, KV_Y - 2, C_GATE_S, 2.0, 'std')

lc.rect(LX, KV_Y, LW, KV_H, C_FMLA_F, C_FMLA_S, rx=8, sw=1.6)
lc.text(LX + 16, KV_Y + 22, 'KV 档解析：写回自家暗号（attention.py:L90-L120）', 10.5, C_FMLA_S, 'start', True,
        maxw=LW - 32, tag='kvl:t')
for i, ln in enumerate(['· --kv-cache-dtype fp8 → 写回 fp8_ds_mla（UE8M0 块缩放 uint8 · 584B/槽=448+128+8 · 页按 576 对齐）',
                        '· assert：fp8_ds_mla 布局 only supports fp8（bf16 直接拒）',
                        '· 操作员只拨一个词，字节格式由 backend 布局决定']):
    lc.text(LX + 16, KV_Y + 44 + i * 20, ln, 9.4, '#334155', 'start', maxw=LW - 30, tag=f'kvl:{i}')

lc.rect(RX, KV_Y, RW, KV_H, '#fef2f2', C_GATE_S, rx=8, sw=1.6)
lc.text(RX + 16, KV_Y + 22, 'KV 门禁：supports_combination 只放行 fp8 系（flashinfer_sparse.py:L130-L150）', 10.5,
        C_GATE_S, 'start', True, maxw=RW - 32, tag='kvr:t')
for i, ln in enumerate(['· 放行 fp8 / fp8_e4m3 / fp8_ds_mla；bf16 / auto → not supported',
                        '· 运行时还需 has_flashinfer_sparse_mla_sm120()（L576-L582）',
                        '· FlashInfer sparse MLA decode API 不可用 → RuntimeError']):
    lc.text(RX + 16, KV_Y + 44 + i * 20, ln, 9.4, '#334155', 'start', maxw=RW - 30, tag=f'kvr:{i}')

# ---------------- 显式旋钮侧门（虚线） ----------------
EK_Y, EK_H = 458, 100
lc.rect(MX, EK_Y, BXR - MX, EK_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.3, dash=True)
lc.text(MX + 16, EK_Y + 22, '显式 backend 旋钮（优先于默认分岔）：拨 FLASHINFER_MLA_SPARSE_DSV4 时', 10.5,
        lc.C_TXT, 'start', True, maxw=900, tag='ek:t')
lc.text(BXR - 16, EK_Y + 22, 'vllm/models/deepseek_v4/nvidia/model.py:L785-L793', 8.8, lc.C_FAINT, 'end',
        maxw=340, tag='ek:r')
for i, ln in enumerate(['· major 10 → DeepseekV4FlashInferMLAAttention：明确拒 fp8_ds_mla（SM10x uses the plain per-tensor FP8 KV layout）· use_fp8_ds_mla_layout=False（L177）',
                        '· major 12 → 仍走 SM120 类（use_fp8_ds_mla_layout=True · L540）——显式旋钮不豁免硬件门禁']):
    lc.text(MX + 16, EK_Y + 46 + i * 20, ln, 9.4, '#334155', 'start', maxw=BXR - MX - 32, tag=f'ek:{i}')

# ---------------- 三种命运带 ----------------
FB_Y, FB_H = 582, 162
lc.rect(MX, FB_Y, BXR - MX, FB_H, '#f8fafc', lc.C_MUTE, rx=10, sw=1.5)
lc.text(MX + 18, FB_Y + 24, '同一句 --kv-cache-dtype fp8 的三种命运', 12.5, lc.C_TXT, 'start', True,
        maxw=600, tag='fb:t')
FATE_GAP = 14
FATE_W = (BXR - MX - 36 - 2 * FATE_GAP) / 3
fates = [
    (C_FMLA_S, C_FMLA_F, '① FlashMLA · SM90/100',
     ['写回自家暗号 fp8_ds_mla', '（UE8M0 块缩放 uint8 · 584B/槽，页按 576 对齐）', '进 FlashMLA 的 fp8 变成它的格式']),
    (C_PLAIN_S, C_PLAIN_F, '② FlashInfer · SM100',
     ['明确拒收暗号、走明码', '普通逐张量 fp8 行', '（use_fp8_ds_mla_layout=False）']),
    (C_GATE_S, '#fef2f2', '③ FlashInfer · SM120（RTX PRO 6000）',
     ['只收 fp8 系——bf16 / auto 直接拒', '『KV 用不用 fp8』不是操作员的自由', '是硬件的命令（96GB 档位的现实）']),
]
for i, (s, f, name, lines) in enumerate(fates):
    fx = MX + 18 + i * (FATE_W + FATE_GAP)
    lc.rect(fx, FB_Y + 36, FATE_W, FB_H - 52, f, s, rx=7, sw=1.5)
    lc.text(fx + 13, FB_Y + 58, name, 10.2, s, 'start', True, maxw=FATE_W - 26, tag=f'ft{i}')
    for j, ln in enumerate(lines):
        lc.text(fx + 13, FB_Y + 80 + j * 17, ln, 9.2, '#334155', 'start', maxw=FATE_W - 26, tag=f'fl{i}.{j}')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 44, 'vllm/models/deepseek_v4/nvidia/model.py:L790-L802（major 分岔 · 显式 backend 优先 L785-L793）· vllm/models/deepseek_v4/sparse_mla.py:L92-L93（major∈[9,10]）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 29, 'vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:L130-L150（SM10x 拒 fp8_ds_mla / SM12x 只放行 fp8 系）· L576-L582（SM120 运行时 API 门）· L177 / L540（use_fp8_ds_mla_layout 两类）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 14, '『fp8 → 写回 fp8_ds_mla』＝ vllm/models/deepseek_v4/attention.py:L90-L120（_resolve_dsv4_kv_cache_dtype）· 行号基线 vLLM v0.27.1（6e448d0ea）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m5_hw_backend_kv.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
