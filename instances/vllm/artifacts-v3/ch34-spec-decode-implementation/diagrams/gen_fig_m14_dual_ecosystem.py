#!/usr/bin/env python3
"""ch34 机制图 · drafter 双生态一账两制（figure_spec fig_m14_dual_ecosystem，模板 layout）

放大自 L0 采样列+spec 块里 drafter 一侧的双生态——L2 章图 south『drafter 谱系 · V1 ↔ V2
双生态』组件的展开（对应站 1 的谱系面；架构归属回指 L2，不另立第二种架构画法）。

claim：同一套调度器 token 账本喂两代 spec 生态：V1 = *Proposer 谱系 +
rejection_sampler.py（gpu_model_runner.py:L583-L658 装配，本章走读主线）；V2 =
*Speculator 树（init_speculator 分发 dflash/dspark/gemma4/mtp/eagle 五支，dspark 与
混合 KV 的 DFlash 强制 V2 runner）。

数字/类名全部取自 figure_spec.numbers 与 pin 源码（spec_decode/__init__.py:L8-L40
分发表、gpu_model_runner.py:L583-L658 装配表、config/vllm.py:L583-L600 / L1074-L1090）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 960
MX, BXR = 56, 1444
DEFS = lc.DEFS

# ---------------- 标题区 ----------------
lc.text(MX, 34, '双生态并存，一本账通用：调度器只认 token 差账——V1 Proposer 老楼与 V2 Speculator 新楼', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '一账两制：挂账/排批/回扣/占位/动态 K 对两代 spec 一视同仁——『调度只认 token 数』的通用性红利',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列+spec drafter 侧（L2 south 谱系双生态·站 1 展开）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 顶：调度器账本（一账两制的『账』） =================
SW_ = 640
SX = MX + (BXR - MX - SW_) / 2
lc.rect(SX, 92, SW_, 88, '#ffffff', lc.C_ENG_S, rx=10, sw=1.8)
lc.text(SX + SW_ / 2, 114, '调度器 token 账本（EngineCore 侧）', 11.5, lc.C_ENG_S, 'middle', True,
        maxw=SW_ - 30, tag='st')
lc.text(SX + SW_ / 2, 134, 'num_new_tokens 差账 · 拒绝回扣 · async 占位账 · dynamic K 查表', 9.2,
        lc.C_TXT, 'middle', True, maxw=SW_ - 30, tag='st2')
lc.text(SX + SW_ / 2, 154, '—— 从不关心草稿是哪一代 drafter 产的', 8.4, lc.C_MUTE, 'middle', maxw=SW_ - 30, tag='st3')
# 喂线（账本 → 两楼）
FEED_Y = 196
for tx_ in (MX + 330, BXR - 330):
    lc.seg(SX + SW_ / 2 + (tx_ - SX - SW_ / 2) * 0.32, 180, tx_, 206, lc.C_ENG_S, 1.6, dash=True)

# ================= 左楼：V1 =================
V1X, V1Y, V1W, V1H = MX, 208, 640, 470
lc.rect(V1X, V1Y, V1W, V1H, '#ffffff', lc.C_GPU_S, rx=10, sw=1.8)
lc.rect(V1X, V1Y, V1W, 36, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.8)
lc.text(V1X + 14, V1Y + 24, 'V1：*Proposer 谱系（本章走读主线）', 11.5, lc.C_GPU_S, 'start', True,
        maxw=V1W - 150, tag='v1t')
bw_ = 46
lc.rect(V1X + V1W - bw_ - 10, V1Y + 7, bw_, 20, lc.C_BADGE_F, lc.C_GPU_S, rx=9, sw=1.1)
lc.text(V1X + V1W - bw_ / 2 - 10, V1Y + 21, '站 1', 9, lc.C_GPU_S, 'middle', True, maxw=40, tag='v1b')
lc.text(V1X + 14, V1Y + 52, '装配点：gpu_model_runner.__init__（L583-L658）——self.drafter = 按 method 分发', 8.6,
        '#475569', 'start', maxw=V1W - 28, tag='v1a')
# 谱系 chips（两列）
CHIPS1 = ['NgramProposer', 'NgramProposerGPU', 'SuffixDecodingProposer', 'EagleProposer',
          'DraftModelProposer', 'MedusaProposer', 'ExtractHiddenStatesProposer', 'Gemma4Proposer',
          'Step3p5MTPProposer', 'DFlashProposer']
CX0, CY0, CGAPX, CGAPY = V1X + 16, V1Y + 68, 0, 0
cwid = (V1W - 32 - 12) / 2
for k, name in enumerate(CHIPS1):
    col_ = k % 2
    row_ = k // 2
    x = CX0 + col_ * (cwid + 12)
    y = CY0 + 6 + row_ * 30
    lc.rect(x, y, cwid, 24, '#ffffff', lc.C_GPU_S, rx=6, sw=1.1)
    lc.text(x + cwid / 2, y + 16, name, 8.6, lc.C_TXT, 'middle', True, maxw=cwid - 8, tag='ch' + name[:8])
# 配套 rejection_sampler
RY_ = CY0 + 6 + 5 * 30 + 10
lc.rect(V1X + 16, RY_, V1W - 32, 74, '#ffffff', lc.C_SAM_S, rx=8, sw=1.4)
lc.text(V1X + 30, RY_ + 20, '配套验证：RejectionSampler（rejection_sampler.py）', 9.6, lc.C_SAM_S,
        'start', True, maxw=V1W - 60, tag='rs')
lc.text(V1X + 30, RY_ + 38, '组合持有普通 Sampler（bonus 缝）· 双 kernel 逐位判 · 残差预采', 8.2,
        '#475569', 'start', maxw=V1W - 60, tag='rs2')
lc.text(V1X + 30, RY_ + 56, '本章 ⑦-⑨ 拍走读的就是这一套', 8.2, lc.C_MUTE, 'start', maxw=V1W - 60, tag='rs3')
lc.text(V1X + 16, V1Y + V1H - 14, 'ngram / EAGLE / draft_model / medusa / suffix … 都住在这栋老楼里', 8.2,
        lc.C_MUTE, 'start', maxw=V1W - 32, tag='v1f')

# ================= 右楼：V2 =================
V2X, V2Y, V2W, V2H = BXR - 640, 208, 640, 470
lc.rect(V2X, V2Y, V2W, V2H, '#ffffff', lc.C_GPU_S, rx=10, sw=1.8)
lc.rect(V2X, V2Y, V2W, 36, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.8)
lc.text(V2X + 14, V2Y + 24, 'V2：*Speculator 树（worker/gpu/spec_decode/ 新楼）', 11.5, lc.C_GPU_S,
        'start', True, maxw=V2W - 40, tag='v2t')
lc.text(V2X + 14, V2Y + 52, '装配点：init_speculator（__init__.py:L8-L40）——按 method 分发 5 支：', 8.6,
        '#475569', 'start', maxw=V2W - 28, tag='v2a')
CHIPS2 = [('DFlashSpeculator', 'dflash'), ('DSparkSpeculator', 'dspark'),
          ('Gemma4Speculator', 'gemma4_mtp'), ('MTPSpeculator', 'mtp'), ('EagleSpeculator', 'eagle')]
for k, (name, method) in enumerate(CHIPS2):
    y = V2Y + 68 + k * 34
    lc.rect(V2X + 16, y, 250, 26, '#ffffff', lc.C_GPU_S, rx=6, sw=1.1)
    lc.text(V2X + 146, y + 17, name, 8.6, lc.C_TXT, 'middle', True, maxw=238, tag='c2' + name[:8])
    lc.seg(V2X + 266, y + 13, V2X + 286, y + 13, lc.C_GPU_S, 1.2, 'std')
    lc.text(V2X + 292, y + 17, 'method=' + method, 8.2, lc.C_MUTE, 'start', maxw=150, tag='m2' + method)
# 强制 V2 注记
FY_ = V2Y + 68 + 5 * 34 + 8
lc.rect(V2X + 16, FY_, V2W - 32, 92, '#fff7ed', lc.C_ENG_S, rx=8, sw=1.2)
lc.text(V2X + 30, FY_ + 20, '强制换 V2 model runner 的两支：', 9.2, lc.C_ENG_S, 'start', True,
        maxw=V2W - 60, tag='f2a')
lc.text(V2X + 30, FY_ + 38, '· dspark——注释原话 "V1 can\'t run dspark"，不支持即 raise 不静默回退', 8.2,
        '#475569', 'start', maxw=V2W - 60, tag='f2b')
lc.text(V2X + 30, FY_ + 54, '· 混合 KV 的 DFlash（需多 KV 组，V1 只有单组）', 8.2, '#475569', 'start',
        maxw=V2W - 60, tag='f2c')
lc.text(V2X + 30, FY_ + 72, '（config/vllm.py:L583-L600 use_v2_model_runner 判定）', 7.8, lc.C_MUTE,
        'start', maxw=V2W - 60, tag='f2d')
lc.text(V2X + 16, V2Y + V2H - 14, 'Speculator 内部结构归上一章（ch33 谱系走读）', 8.2, lc.C_MUTE, 'start',
        maxw=V2W - 32, tag='v2f')

# ================= 底部：async 兼容子集 + 对账 =================
BY0 = V1Y + V1H + 20
lc.rect(MX, BY0, BXR - MX, 118, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 18, BY0 + 24, 'async 调度的兼容子集（硬校验，config/vllm.py:L1074-L1090）：EAGLE / MTP / draft_model / ngram_gpu / dspark——其余 method 开 async 直接 ValueError。',
        9.8, lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='ba1')
lc.text(MX + 18, BY0 + 48, '为什么两代能共用一本账：调度器只见 Request 上的 spec_token_ids / num_tokens_with_spec / num_output_placeholders——', 8.8,
        lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='ba2')
lc.text(MX + 18, BY0 + 66, '草稿谁产的、哪一代产的，账本不问；回扣/占位/动态 K 的算式两代一字不差。', 8.8,
        lc.C_MUTE, 'start', maxw=BXR - MX - 36, tag='ba3')
lc.text(MX + 18, BY0 + 90, '（V1 装配表类名逐一对 pin 源码核过：10 个 *Proposer；V2 分发 5 个 *Speculator——无杜撰）', 8.0,
        lc.C_FAINT, 'start', maxw=BXR - MX - 36, tag='ba4')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 40, 'vllm/v1/worker/gpu/spec_decode/__init__.py:L8-L40（init_speculator 分发 5 支）· vllm/v1/worker/gpu_model_runner.py:L583-L658（V1 装配 + RejectionSampler 构造）· '
                    'vllm/config/vllm.py:L583-L600（dspark 强制 V2）· L1074-L1090（async 兼容子集）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 25, '类名清单 ＝ pin 源码 v0.27.1 逐字核（L586-L595 类型注解即 V1 全谱）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m14_dual_ecosystem.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
